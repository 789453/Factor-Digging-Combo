
# alpha-factor-mvp 第一阶段工程优化文档

## 目标

在 `outputs/real_200_test` 已经跑通真实 200 股票池小数据的基础上，升级项目，使其支持：

1. 10000+ 表达式稳定运行；
2. 断点续跑；
3. 分段写入；
4. 每完成 2% / 5% 输出 top50 checkpoint；
5. train/test 拆分：20240101-20250831 为 train，20250901-20260430 为 test；
6. 进度条低频刷新；
7. SQLite 结果库；
8. 后续为 Numba / 并行 / DAG cache 打基础。

## 当前真实运行结果

`summary.json` 显示：

- n_raw_rows: 111223
- n_dates: 557
- n_codes: 200
- n_features: 30
- n_exprs: 100
- n_results: 100
- out_dir: outputs\real_200_test

`factor_results.csv` 显示：

- 100 个表达式全部 status=OK；
- coverage 均值约 0.945；
- mean_rank_ic 均值约 -0.032；
- top 表达式集中在 `ret_1d` 的 EMA / WMA / Mean 类反转结构；
- 这说明代码链路已经有效跑通，但表达式空间目前还很窄。

## 当前主要瓶颈

### 1. pipeline.py

当前逻辑：

```python
records = []
for expr in exprs:
    arr, status = ev.eval_expr(expr)
    if arr is None:
        records.append({"expr": expr, "status": status})
        continue
    m = summarize_factor(arr, fwd, dates, ...)
    records.append({...})
```

问题：

- 一次性把所有结果放在内存；
- 只有最后写 CSV；
- 中途失败会丢掉全部进度；
- 没有 resume；
- 没有 checkpoint top50；
- 没有 train/test 分开算；
- 没有进度显示；
- 没有 SQLite 持久化。

### 2. evaluator.py

当前 evaluator 有 `_cache`，但只是运行期内 AST 节点缓存：

```python
self._cache: dict[str, np.ndarray] = {}
```

优点：

- 同一个表达式批次中公共子表达式会被缓存。

问题：

- 没有 cache 统计；
- 没有 LRU / 内存上限；
- 10000 表达式后可能内存增长；
- 没有按 operator/window/field 预计算；
- 没有将 `Rank(TsMean($x,20))` 这类节点做分层缓存管理。

### 3. ops.py

当前 rolling 基本依赖 Pandas：

```python
pd.DataFrame(x).rolling(...)
```

问题：

- 200 股票 x 557 日期不大，能跑；
- 但 10000+ 表达式时，重复 rolling 会成为主瓶颈；
- `TsRank` 用 `rolling.apply(last_rank, raw=False)` 很慢；
- `TsCorr/TsCov` 每列循环 Pandas Series，也较慢。

### 4. metrics.py

当前每天循环算 corr / rank corr：

```python
for i in range(x.shape[0]):
    ...
    spearmanr(a[m], b[m])
```

问题：

- spearmanr 每日调用开销高；
- 10000 表达式时比单表达式计算更慢；
- 没有 train/test 拆分；
- 没有 horizon 多周期；
- quantile spread 每个表达式重新 rank 一次。

## 修改任务拆分

---

# Task 1：增加运行配置

文件：`src/alpha_mvp/config.py`

新增字段：

```python
@dataclass(frozen=True)
class RunConfig:
    ...
    resume: bool = True
    checkpoint_pct: float = 0.05
    first_checkpoint_pct: float = 0.02
    topk_checkpoint: int = 50
    write_every: int = 200
    sqlite_path: str | None = None
    train_end: str = "20250831"
    test_start: str = "20250901"
    progress_min_interval_sec: float = 5.0
```

含义：

- `resume`: 是否跳过已完成表达式；
- `checkpoint_pct`: 每完成 5% 写一次 top50；
- `first_checkpoint_pct`: 第一次 2% 写一次；
- `write_every`: 每 200 条结果落盘；
- `sqlite_path`: SQLite 结果库路径；
- `train_end/test_start`: train/test 分割；
- `progress_min_interval_sec`: 控制进度输出频率。

---

# Task 2：新增结果存储模块

新增文件：`src/alpha_mvp/store.py`

```python
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

SCHEMA = """
create table if not exists factor_results (
    run_id text not null,
    expr text not null,
    canonical text,
    status text,
    coverage real,
    usable_days integer,
    mean_ic real,
    icir real,
    mean_rank_ic real,
    rank_icir real,
    positive_rank_ic_ratio real,
    turnover_proxy real,
    quantile_spread real,
    train_mean_rank_ic real,
    train_rank_icir real,
    test_mean_rank_ic real,
    test_rank_icir real,
    score real,
    error text,
    created_at text default current_timestamp,
    primary key (run_id, canonical)
);

create index if not exists idx_factor_results_run_score
on factor_results(run_id, score desc);
"""

class ResultStore:
    def __init__(self, sqlite_path: str):
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.execute("pragma journal_mode=WAL")
        self.conn.execute("pragma synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def completed_keys(self, run_id: str) -> set[str]:
        rows = self.conn.execute(
            "select canonical from factor_results where run_id=?",
            (run_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def upsert_many(self, run_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["run_id"] = run_id
        cols = [
            "run_id", "expr", "canonical", "status", "coverage", "usable_days",
            "mean_ic", "icir", "mean_rank_ic", "rank_icir",
            "positive_rank_ic_ratio", "turnover_proxy", "quantile_spread",
            "train_mean_rank_ic", "train_rank_icir",
            "test_mean_rank_ic", "test_rank_icir",
            "score", "error",
        ]
        for c in cols:
            if c not in df:
                df[c] = None
        sql = f"""
        insert or replace into factor_results ({",".join(cols)})
        values ({",".join(["?"] * len(cols))})
        """
        self.conn.executemany(sql, df[cols].itertuples(index=False, name=None))
        self.conn.commit()

    def load_all(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "select * from factor_results where run_id=? order by score desc",
            self.conn,
            params=(run_id,),
        )

    def close(self):
        self.conn.close()
```

---

# Task 3：新增 train/test 指标计算

文件：`src/alpha_mvp/metrics.py`

新增函数：

```python
def summarize_factor_split(
    factor: np.ndarray,
    fwd: np.ndarray,
    dates: list[str],
    train_end: str = "20250831",
    test_start: str = "20250901",
    min_daily_valid: int = 30,
) -> dict:
    base = summarize_factor(factor, fwd, dates, min_daily_valid=min_daily_valid)
    d = np.array(dates).astype(str)
    train_mask = d <= train_end
    test_mask = d >= test_start

    train = summarize_factor(
        factor[train_mask],
        fwd[train_mask],
        d[train_mask].tolist(),
        min_daily_valid=min_daily_valid,
    )
    test = summarize_factor(
        factor[test_mask],
        fwd[test_mask],
        d[test_mask].tolist(),
        min_daily_valid=min_daily_valid,
    )

    base.update({
        "train_mean_rank_ic": train.get("mean_rank_ic"),
        "train_rank_icir": train.get("rank_icir"),
        "test_mean_rank_ic": test.get("mean_rank_ic"),
        "test_rank_icir": test.get("rank_icir"),
    })

    # 推荐 score：train/test 同向优先，惩罚 test 崩塌
    tr = base.get("train_mean_rank_ic")
    te = base.get("test_mean_rank_ic")
    if tr is not None and te is not None and np.isfinite(tr) and np.isfinite(te):
        same_sign = 1.0 if tr * te > 0 else 0.25
        base["score"] = abs(te) * abs(base.get("test_rank_icir", 0) or 0) * same_sign
    else:
        base["score"] = np.nan
    return base
```

---

# Task 4：改造 pipeline.py，支持 resume、分批写入、checkpoint top50、进度条

文件：`src/alpha_mvp/pipeline.py`

建议替换主循环为：

```python
import time
from .metrics import summarize_factor_split
from .store import ResultStore

def _write_checkpoint(out: Path, all_results: pd.DataFrame, pct: float, topk: int):
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if all_results.empty:
        return
    fn = ckpt_dir / f"top{topk}_pct_{pct:.0%}.csv"
    all_results.sort_values("score", ascending=False, na_position="last").head(topk).to_csv(
        fn, index=False, encoding="utf-8-sig"
    )

def run_pipeline(cfg: RunConfig) -> dict:
    ...
    run_id = f"{cfg.start}_{cfg.end}_{len(codes)}_{cfg.max_exprs}"
    sqlite_path = cfg.sqlite_path or str(out / "factor_results.sqlite3")
    store = ResultStore(sqlite_path)

    done = store.completed_keys(run_id) if cfg.resume else set()
    exprs_to_run = []
    for e in exprs:
        key = canonical(parse_expr(e))
        if key not in done:
            exprs_to_run.append(e)

    total = len(exprs)
    buffer = []
    last_progress = time.time()
    next_checkpoint_ratio = cfg.first_checkpoint_pct
    checkpoint_ratios_done = set()

    for idx, expr in enumerate(exprs_to_run, start=len(done) + 1):
        try:
            key = canonical(parse_expr(expr))
            arr, status = ev.eval_expr(expr)
            if arr is None:
                rec = {"expr": expr, "canonical": key, "status": status, "error": status}
            else:
                m = summarize_factor_split(
                    arr, fwd, dates,
                    train_end=cfg.train_end,
                    test_start=cfg.test_start,
                    min_daily_valid=cfg.eval.min_daily_valid_names,
                )
                rec = {"expr": expr, "canonical": key, "status": status, **m, "error": None}
        except Exception as e:
            rec = {"expr": expr, "canonical": expr, "status": "ERROR", "error": repr(e)}

        buffer.append(rec)

        if len(buffer) >= cfg.write_every:
            store.upsert_many(run_id, buffer)
            buffer.clear()

        ratio = idx / total
        now = time.time()

        if ratio >= next_checkpoint_ratio:
            store.upsert_many(run_id, buffer)
            buffer.clear()
            all_df = store.load_all(run_id)
            _write_checkpoint(out, all_df, next_checkpoint_ratio, cfg.topk_checkpoint)
            checkpoint_ratios_done.add(next_checkpoint_ratio)
            next_checkpoint_ratio = max(
                cfg.checkpoint_pct,
                (int(ratio / cfg.checkpoint_pct) + 1) * cfg.checkpoint_pct,
            )

        if now - last_progress >= cfg.progress_min_interval_sec:
            print(f"[progress] {idx}/{total} ({ratio:.1%}), cache={len(ev._cache)}")
            last_progress = now

    store.upsert_many(run_id, buffer)
    res = store.load_all(run_id)
    res.to_csv(out / "factor_results.csv", index=False, encoding="utf-8-sig")
    res.head(100).to_csv(out / "top100.csv", index=False, encoding="utf-8-sig")
```

---

# Task 5：改造 CLI

文件：`src/alpha_mvp/cli.py`

新增参数：

```python
p.add_argument("--resume", action="store_true")
p.add_argument("--no-resume", dest="resume", action="store_false")
p.set_defaults(resume=True)

p.add_argument("--write-every", type=int, default=200)
p.add_argument("--checkpoint-pct", type=float, default=0.05)
p.add_argument("--first-checkpoint-pct", type=float, default=0.02)
p.add_argument("--topk-checkpoint", type=int, default=50)
p.add_argument("--sqlite-path", default=None)
p.add_argument("--train-end", default="20250831")
p.add_argument("--test-start", default="20250901")
p.add_argument("--progress-min-interval-sec", type=float, default=5.0)
```

构造 `RunConfig` 时传入这些参数。

---

# Task 6：优化数据读取

文件：`src/alpha_mvp/data.py`

当前 SQL 已经只 select 必要列，这是对的。下一步建议：

```python
conn.execute("set threads=16")
conn.execute("set memory_limit='24GB'")
```

同时把 pool 改成临时表，避免巨大 `IN (...)` 字符串：

```python
if pool:
    pool_df = pd.DataFrame({"ts_code": pool})
    conn.register("pool_df", pool_df)
    pool_join = "join pool_df p on d.ts_code=p.ts_code"
else:
    pool_join = ""
```

SQL 改为：

```sql
from {daily} d
{pool_join}
left join ...
where d.trade_date >= ? and d.trade_date <= ?
```

第一阶段 200/800 股票池差异不大，但后续更稳。

---

# Task 7：减少无效表达式计算

文件：`src/alpha_mvp/grammar.py`

当前 `generate_templates` 有 `max_exprs` 早停，但表达式顺序会导致前 100 基本集中在前几个字段。建议加入分层 round-robin：

```python
def interleave_groups(groups: list[list[str]], max_exprs: int):
    out = []
    max_len = max(len(g) for g in groups)
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                out.append(g[i])
                if len(out) >= max_exprs:
                    return out
    return out
```

把模板分成：

```python
single_field_group
binary_group
pair_corr_group
directional_group
```

这样前 100 就不会全部是 `ret_1d`。

---

# Task 8：Numba 加速优先级

短期不要重写全部算子。先改最划算的 3 个：

1. cross-sectional rank；
2. daily Pearson/Spearman IC；
3. rolling corr。

建议新增：

```text
src/alpha_mvp/fastops.py
```

第一版只做 `fast_rank_cs` 和 `fast_daily_corr`。`TsRank` / `TsCorr` 后续再做。

---

# Task 9：并行策略

目前 557 x 200 的面板很小，主要瓶颈是 Python 调度和 Pandas rolling，不建议上 GPU。

推荐：

1. 单进程 + DAG cache：第一优先；
2. 轻量 ThreadPool 不明显；
3. 多进程最多 4 个 worker；
4. 每个 worker 独立 evaluator；
5. 只由主进程写 SQLite，worker 返回 records；
6. 不要多个进程同时写 SQLite。

如果做多进程：

```text
主进程：
  - 加载数据
  - 分 expression chunks
  - 收集 records
  - 写 SQLite

worker：
  - 接收 expression chunk
  - 只读 shared panels
  - 计算结果
  - 返回 list[dict]
```

但第一版不建议立刻并行，因为 `_cache` 在多进程之间不能共享，容易重复计算。

---

# Task 10：验证测试命令

小测试：

```powershell
python -m src.alpha_mvp.cli `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --pool-json "D:\Trading\data_ever_26_3_14\static_pool_200.json" `
  --start 20240101 `
  --end 20260430 `
  --out outputs\real_200_test_v2 `
  --max-exprs 300 `
  --write-every 50 `
  --first-checkpoint-pct 0.02 `
  --checkpoint-pct 0.05
```

预期检查：

```text
outputs\real_200_test_v2\factor_results.sqlite3
outputs\real_200_test_v2\checkpoints\top50_pct_2%.csv
outputs\real_200_test_v2\checkpoints\top50_pct_5%.csv
outputs\real_200_test_v2\factor_results.csv
outputs\real_200_test_v2\top100.csv
```

---

# Task 11：潜在研究风险修正

## 1. 当前 top 表达式全是负 RankIC

这说明第一轮 top 是短期反转结构。后续 score 应该保留方向，不要只看 abs：

```python
score_long = mean_rank_ic * rank_icir
score_abs = abs(mean_rank_ic) * abs(rank_icir)
```

记录两种。入库时标记 `direction = sign(mean_rank_ic)`。

## 2. 前 100 表达式偏置

当前前 100 主要来自单字段 ret_1d。需要 grammar round-robin。

## 3. 没有 train/test

必须加，否则 test 期表现无法评估。

## 4. ICIR 定义过于简单

可以先保留，但后续建议用：

```python
icir_annual = mean_ic / std_ic * sqrt(252)
```

当前不年化也可以，但要命名清楚。

## 5. 换手 proxy 不是实际组合换手

当前是 rank 差分 proxy。后续 TopK 策略需要实际换手：

```text
top_quantile_membership turnover
```

## 6. 没有行业/市值中性

后续应增加：

```text
raw_rank_ic
industry_neutral_rank_ic
size_bucket_rank_ic
industry_size_neutral_rank_ic
```

---

# 推荐修改顺序

1. `config.py` 加运行参数；
2. 新增 `store.py`；
3. `metrics.py` 加 train/test split；
4. `pipeline.py` 加 resume + checkpoint + sqlite；
5. `cli.py` 暴露参数；
6. `grammar.py` 加 round-robin；
7. `data.py` 优化 DuckDB pool 临时表；
8. 再做 fastops.py；
9. 最后考虑多进程。

不要一开始就并行或 GPU。先把断点、分批写、checkpoint、train/test 跑通。
