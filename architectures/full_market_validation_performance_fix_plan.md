# feature-init-factor-mining-427 性能与验证层修复工程文档

适用仓库：`789453/feature-init-factor-mining-427`  
目标：在 `outputs\real_all\top100.csv` 已经由 9000+ 因子筛出后，构建更快、更稳健、更可复用的全市场验证与可视化系统。  
建议数据起点：`20180101`  
数据源：`D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb`

---

## 1. 当前项目最新状态判断

仓库 README 仍描述为第一阶段 MVP：从 DuckDB 或模拟数据读取 `daily / moneyflow / cyq_perf / daily_basic`，构建约 30 个基础字段，使用 typed grammar + 模板枚举生成表达式，在线计算，输出 IC/RankIC/ICIR 等结果。

当前最新 `pipeline.py` 已经比最初版本有明显改进：

```text
已具备：
- ResultStore / SQLite
- resume
- checkpoint topK
- expr_file
- start_expr / end_expr
- train/test summarize_factor_split
- progress_min_interval_sec
```

但它仍然是“挖掘初筛主流程”，不适合继续承担全市场 Top100 验证、VectorBT 回测、Alphalens tear sheet、精美可视化和长期结果缓存。

---

## 2. 当前主要实现问题

### 2.1 数据读取与保存问题

当前 `pipeline.py` 每次运行都会：

```text
load_from_duckdb
→ add_basic_features
→ make_panels
→ 重新计算表达式
```

问题：

1. 全市场从 2018 年开始后数据量显著增加；
2. 30 个基础字段每次重新构建，浪费；
3. Top100 因子面板没有持久化；
4. `raw_loaded_sample.parquet` 命名不准确，全市场时会变成大文件；
5. 后续 Alphalens / VectorBT / 自研 analytics 都会重复需要 factor panel；
6. 没有 feature cache version，字段定义变化后难以追踪。

建议：

```text
把基础字段面板和 Top100 factor panel 单独保存成 Parquet / DuckDB 表
```

---

### 2.2 表达式计算方式问题

当前 evaluator 有运行时 `_cache`，但缓存粒度和生命周期不足：

```python
self._cache: dict[str, np.ndarray] = {}
```

优点：

- 单次运行中相同 AST 节点可复用。

问题：

- 运行结束后丢失；
- 无 cache 命中统计；
- 无内存上限；
- 无按字段/窗口/算子预计算；
- 全市场 Top100 验证时不能复用前一轮 9000+ 初筛产生的中间计算；
- 多进程时 cache 不能共享。

建议：

```text
新增持久化 factor panel cache
新增 hot operator precompute cache
新增表达式 DAG cache key
```

---

### 2.3 速度瓶颈

当前 `ops.py` 主要使用 Pandas rolling：

```text
TsMean / TsStd / TsRank / TsCorr / TsCov
```

真正慢的通常是：

```text
TsRank
TsCorr
TsCov
rolling.apply
逐日 Spearman
逐表达式循环
```

全市场约：

```text
2018-2026: 约 2000+ 交易日
股票数: 5000+
单 factor panel: 2000 × 5000 × float32 ≈ 40MB
Top100: ≈ 4GB float32
```

这已经不是“随手跑”的小规模。必须做：

```text
float32 化
分批计算
Parquet 落盘
避免重复 rolling
Numba 化关键算子
```

---

### 2.4 指标计算问题

当前 `metrics.py` 是逐表达式、逐日循环计算 IC/RankIC。问题：

1. 全市场 Top100 时，重复 rank/corr 开销大；
2. 没有统一计算 factor batch × date 的 IC 矩阵；
3. rolling IC/IR 没有按 factor 批量计算；
4. size bucket / industry group 没有统一面板化；
5. TopK 回测和 IC 诊断没有共用中间结果。

建议：

```text
新增 validation/analytics_fast.py
支持批量 factor panels 输入：
  factors: K × T × N 或 dict[factor_id] -> T × N
  forward_returns: T × N
输出：
  K × metric table
```

---

### 2.5 结果展示问题

当前输出主要是 CSV：

```text
factor_results.csv
top100.csv
checkpoint CSV
summary.json
```

问题：

1. 没有统一 HTML 报告；
2. 没有 rolling IC 图；
3. 没有 Top5/Top10 净值图；
4. 没有分行业热力图；
5. 没有市值桶表现图；
6. 没有 factor_id 页面；
7. 没有图表资产路径管理；
8. 没有 top100 因子验证 dashboard。

建议：

```text
新增 reports/
  validation_report.html
  factor_cards/
  plots/
```

---

## 3. 总体修复方案

新增三层缓存和验证系统：

```text
Layer 1: Market Feature Cache
  全市场 raw -> 30 个基础字段 -> T × N panels -> Parquet

Layer 2: Factor Panel Cache
  outputs\real_all\top100.csv -> Top100 factor panels -> Parquet

Layer 3: Validation Analytics
  factor panels -> IC/RankIC/rolling/group/backtest/Alphalens/report
```

推荐目录：

```text
src/alpha_mvp/
  cache/
    __init__.py
    feature_cache.py
    factor_cache.py
    manifest.py

  validation/
    __init__.py
    config.py
    top_factors.py
    market_data.py
    factor_compute.py
    analytics_fast.py
    group_metrics.py
    vectorbot.py
    alphalens_runner.py
    report_builder.py
    plots.py
    cli_validate.py
```

输出目录：

```text
outputs/real_all_validation_2018/
  cache/
    market_features/
      manifest.json
      dates.parquet
      codes.parquet
      close.parquet
      ret_1d.parquet
      main_net_ratio.parquet
      ...
    factor_panels/
      manifest.json
      F0001_xxxxxxxx.parquet
      F0002_xxxxxxxx.parquet

  metrics/
    summary.csv
    train_test.csv
    rolling_ic.parquet
    group_size.csv
    group_industry.csv

  vectorbot/
    portfolio_summary.csv
    equity_top5.parquet
    equity_top10.parquet
    layer_equity.parquet

  alphalens/
    F0001_xxxxxxxx/
      clean_factor_data.parquet
      summary_tear_sheet.png
      returns_tear_sheet.png
      information_tear_sheet.png

  reports/
    validation_report.html
    plots/
```

---

## 4. 数据读取优化

### 4.1 DuckDB 查询优化

文件：`src/alpha_mvp/validation/market_data.py`

建议读取全市场时：

```python
conn = duckdb.connect(cfg.duckdb_path, read_only=True)
conn.execute(f"set threads={cfg.duckdb_threads}")
conn.execute(f"set memory_limit='{cfg.duckdb_memory_limit}'")
conn.execute("set temp_directory='D:/Trading/tmp_duckdb'")
```

SQL 只取需要列：

```sql
select
  d.ts_code, d.trade_date,
  d.open, d.high, d.low, d.close, d.pre_close, d.pct_chg, d.vol, d.amount,
  m.buy_sm_amount, m.sell_sm_amount,
  m.buy_md_amount, m.sell_md_amount,
  m.buy_lg_amount, m.sell_lg_amount,
  m.net_mf_amount,
  c.his_low, c.his_high,
  c.cost_5pct, c.cost_15pct, c.cost_50pct,
  c.cost_85pct, c.cost_95pct,
  c.weight_avg, c.winner_rate,
  b.turnover_rate, b.turnover_rate_f,
  b.volume_ratio, b.total_mv, b.circ_mv,
  s.industry, s.name, s.market, s.list_date
from daily d
left join moneyflow m
  on d.ts_code=m.ts_code and d.trade_date=m.trade_date
left join cyq c
  on d.ts_code=c.ts_code and d.trade_date=c.trade_date
left join daily_basic b
  on d.ts_code=b.ts_code and d.trade_date=b.trade_date
left join stock_basic_snapshot s
  on d.ts_code=s.ts_code
where d.trade_date >= '20180101'
  and d.trade_date <= cfg.end
  and d.close is not null
```

### 4.2 不要每个因子重复读 DuckDB

错误方式：

```text
for factor in top100:
    load DuckDB
    build fields
    eval factor
```

正确方式：

```text
load DuckDB once
build feature panels once
eval Top100 from shared panels
```

---

## 5. Market Feature Cache 设计

新增文件：`src/alpha_mvp/cache/feature_cache.py`

### 5.1 保存方式

建议保存为宽矩阵 Parquet：

```text
index = trade_date
columns = ts_code
values = feature value
```

每个 feature 一个文件：

```text
cache/market_features/ret_1d.parquet
cache/market_features/main_net_ratio.parquet
cache/market_features/chip_width_90.parquet
cache/market_features/close.parquet
cache/market_features/circ_mv.parquet
cache/market_features/industry.parquet
```

相比一个巨大 long parquet，这样更适合：

```text
读取某几个特征
表达式 evaluator
按列股票矩阵运算
```

### 5.2 manifest.json

```json
{
  "start": "20180101",
  "end": "20260430",
  "n_dates": 2020,
  "n_codes": 5400,
  "features": ["ret_1d", "main_net_ratio"],
  "dtype": "float32",
  "field_version": "v1",
  "created_at": "2026-04-27"
}
```

### 5.3 接口

```python
def build_market_feature_cache(raw_df, out_dir, dtype="float32"):
    df = add_basic_features(raw_df)
    panels, dates, codes = make_panels(df, DEFAULT_FEATURES, value_col="close")

    for name, arr in panels.items():
        arr = arr.astype(dtype)
        save_panel_parquet(arr, dates, codes, out_dir / f"{name}.parquet")

    save_manifest(...)
    return panels, dates, codes
```

---

## 6. Factor Panel Cache 设计

新增文件：`src/alpha_mvp/cache/factor_cache.py`

目标：`outputs\real_all\top100.csv` 在全市场数据上计算一次后保存，后续 VectorBT / Alphalens / 报告全部复用。

### 6.1 factor_id

```python
def make_factor_id(expr: str, rank: int) -> str:
    h = hashlib.md5(expr.encode("utf-8")).hexdigest()[:8]
    return f"F{rank:04d}_{h}"
```

### 6.2 保存格式

每个因子一个 Parquet：

```text
factor_panels/F0001_abcdef12.parquet
```

宽矩阵格式：

```text
trade_date | 000001.SZ | 000002.SZ | ...
```

优点：

1. 单因子读取快；
2. 断点友好；
3. Alphalens 单因子直接读取；
4. 不需要一次加载 100 个；
5. 崩溃后已完成因子不用重算。

### 6.3 manifest

```json
{
  "factor_id": "F0001_abcdef12",
  "expr": "Rank(TsEMA($ret_1d,30))",
  "source_top100": "outputs/real_all/top100.csv",
  "start": "20180101",
  "end": "20260430",
  "shape": [2020, 5400],
  "dtype": "float32",
  "status": "OK",
  "coverage": 0.93
}
```

### 6.4 接口

```python
def compute_and_cache_top_factors(top_factors, evaluator, dates, codes, out_dir, overwrite=False):
    done = load_existing_factor_manifest(out_dir)
    for row in top_factors:
        fid = row.factor_id
        if fid in done and not overwrite:
            continue

        arr, status = evaluator.eval_expr(row.factor_expr)
        if status == "OK":
            save_panel_parquet(arr.astype("float32"), dates, codes, out_dir / f"{fid}.parquet")
        update_manifest(fid, expr, status, coverage)
```

---

## 7. 计算加速方案

### 7.1 优先级排序

不要一开始重写所有算子。优先优化：

```text
1. Rank 截面排序
2. IC / RankIC 批量计算
3. TsRank
4. TsCorr
5. TsCov
6. rolling std / var
```

### 7.2 fastops.py

新增文件：`src/alpha_mvp/fastops.py`

建议接口：

```python
def rank_cs_fast(x: np.ndarray) -> np.ndarray:
    ...

def daily_corr_fast(factor: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    ...

def daily_rank_corr_fast(factor: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    ...

def rolling_mean_fast(x: np.ndarray, w: int) -> np.ndarray:
    ...

def rolling_corr_fast(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    ...
```

### 7.3 rank_cs_fast 代码骨架

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def rank_cs_numba(x):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float32)
    out[:] = np.nan

    for t in prange(T):
        idx = []
        vals = []
        for j in range(N):
            v = x[t, j]
            if not np.isnan(v):
                idx.append(j)
                vals.append(v)

        m = len(vals)
        if m == 0:
            continue

        order = np.argsort(np.array(vals))
        ranks = np.empty(m, dtype=np.float32)
        for r in range(m):
            ranks[order[r]] = (r + 1) / m

        for k in range(m):
            out[t, idx[k]] = ranks[k]

    return out
```

注意：Numba list 在 njit 中不是最优，最终可以改成预分配数组版本。第一版可以先实现 NumPy argsort 版：

```python
def rank_cs_numpy(x):
    order = np.argsort(x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row = np.arange(x.shape[0])[:, None]
    ranks[row, order] = np.arange(1, x.shape[1] + 1, dtype=np.float32)
    ranks = ranks / x.shape[1]
    ranks[~np.isfinite(x)] = np.nan
    return ranks
```

但这个版本对 NaN 处理要小心，因为 NaN 会被排到最后。

---

## 8. 批量 analytics 设计

新增文件：`src/alpha_mvp/validation/analytics_fast.py`

输入：

```python
factor_panels: dict[str, np.ndarray]  # factor_id -> T x N
close: np.ndarray                     # T x N
dates: list[str]
codes: list[str]
circ_mv: np.ndarray
industry: np.ndarray or DataFrame
```

输出：

```text
metrics/summary.csv
metrics/rolling_ic.parquet
metrics/group_size.csv
metrics/group_industry.csv
```

### 8.1 forward return

```python
def forward_returns(close, horizon=5):
    out = np.full_like(close, np.nan, dtype=np.float32)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return out
```

### 8.2 批量 IC

```python
def calc_factor_metrics(fid, factor, fwd, dates, cfg):
    ic = daily_corr_fast(factor, fwd)
    ric = daily_rank_corr_fast(factor, fwd)

    return {
        "factor_id": fid,
        "mean_ic": np.nanmean(ic),
        "icir": np.nanmean(ic) / np.nanstd(ic),
        "mean_rank_ic": np.nanmean(ric),
        "rank_icir": np.nanmean(ric) / np.nanstd(ric),
        "positive_rank_ic_ratio": np.nanmean(ric > 0),
        ...
    }
```

### 8.3 rolling IC

```python
def rolling_mean_std(x, w):
    s = pd.Series(x)
    m = s.rolling(w, min_periods=max(5, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(5, w // 3)).std()
    return m, m / sd
```

---

## 9. VectorBot / VectorBT 验证设计

新增文件：`src/alpha_mvp/validation/vectorbot.py`

第一版不建议直接依赖 VectorBT 的复杂订单层，而是用矩阵化权重回测，然后可选把组合收益传给 VectorBT 做 stats / 图。

### 9.1 权重矩阵

```python
def make_top_weights(factor, dates, top_pct=0.05, rebalance_n=5, direction=1):
    x = factor * direction
    T, N = x.shape
    weights = np.zeros((T, N), dtype=np.float32)
    last_w = np.zeros(N, dtype=np.float32)

    for t in range(T):
        if t % rebalance_n == 0:
            row = x[t]
            valid = np.isfinite(row)
            k = max(1, int(valid.sum() * top_pct))
            valid_idx = np.where(valid)[0]
            selected = valid_idx[np.argsort(row[valid])[-k:]]
            w = np.zeros(N, dtype=np.float32)
            w[selected] = 1.0 / k
            last_w = w
        weights[t] = last_w
    return weights
```

### 9.2 避免未来函数

组合收益必须：

```text
weights[t] × close[t+1] / close[t] - 1
```

不能用：

```text
weights[t] × return[t]
```

### 9.3 输出

```text
vectorbot/portfolio_summary.csv
vectorbot/equity_top5.parquet
vectorbot/equity_top10.parquet
vectorbot/turnover.parquet
```

---

## 10. Alphalens-reloaded 诊断

新增文件：`src/alpha_mvp/validation/alphalens_runner.py`

只对精选因子运行：

```text
--alphalens-top-n 10
```

### 10.1 数据对齐

Alphalens 需要：

```text
factor: MultiIndex(date, asset)
prices: DataFrame(date × asset)
groupby: asset -> industry
```

转换：

```python
factor_s.index.names = ["date", "asset"]
prices.index = pd.to_datetime(dates, format="%Y%m%d")
prices.columns = codes
```

### 10.2 输出

```text
alphalens/F0001_xxxxxxxx/
  clean_factor_data.parquet
  summary_tear_sheet.png
  returns_tear_sheet.png
  information_tear_sheet.png
```

---

## 11. 可视化升级设计

新增文件：

```text
src/alpha_mvp/validation/plots.py
src/alpha_mvp/validation/report_builder.py
```

### 11.1 推荐图表

每个 Top 因子：

```text
1. cumulative top5/top10 equity curve
2. rolling 60D RankIC
3. IC histogram
4. size bucket RankIC bar
5. industry RankIC heatmap
6. quantile layer cumulative return
7. train/test comparison bar
```

### 11.2 HTML 报告结构

```text
reports/validation_report.html

Sections:
1. Run Summary
2. Top Factors Table
3. IC / RankIC Summary
4. VectorBot Portfolio Summary
5. Rolling IC Charts
6. Size Bucket Diagnostics
7. Industry Diagnostics
8. Alphalens Links
9. Warnings
```

### 11.3 精美展示建议

使用 Plotly 生成交互图：

```python
import plotly.express as px
import plotly.graph_objects as go
```

保存：

```python
fig.write_html(out / "reports/plots/F0001_rolling_ic.html")
```

优点：

1. 鼠标悬浮查看数值；
2. 多因子对比方便；
3. 不依赖 Notebook；
4. 可以嵌入最终 HTML。

---

## 12. CLI 设计

新增：

```text
src/alpha_mvp/validation/cli_validate.py
```

命令：

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\real_all\top100.csv" `
  --out "outputs\real_all_validation_2018" `
  --start 20180101 `
  --end 20260430 `
  --horizon 5 `
  --rebalance-n 5 `
  --top-n 100 `
  --alphalens-top-n 10 `
  --cache-features `
  --cache-factors
```

参数：

```text
--duckdb
--top100
--out
--start
--end
--horizon
--rebalance-n
--top-n
--alphalens-top-n
--cache-features
--cache-factors
--overwrite-cache
--duckdb-threads
--duckdb-memory-limit
--skip-alphalens
--skip-vectorbot
```

---

## 13. 任务拆分

### Task 1：新增 cache 模块

文件：

```text
src/alpha_mvp/cache/__init__.py
src/alpha_mvp/cache/panel_io.py
src/alpha_mvp/cache/feature_cache.py
src/alpha_mvp/cache/factor_cache.py
```

目标：

```text
保存/读取 T × N panel parquet
保存 manifest.json
支持 float32
```

测试：

```powershell
python -m src.alpha_mvp.cache.feature_cache_test
```

---

### Task 2：全市场 2018 数据读取

文件：

```text
src/alpha_mvp/validation/market_data.py
```

目标：

```text
从 20180101 读取全市场 daily/moneyflow/cyq/daily_basic/snapshot
只选必要字段
设置 DuckDB threads/memory/temp_directory
```

测试：

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --start 20180101 `
  --top-n 1 `
  --skip-alphalens `
  --skip-vectorbot
```

---

### Task 3：构建基础字段缓存

文件：

```text
src/alpha_mvp/cache/feature_cache.py
```

目标：

```text
raw_df -> add_basic_features -> panels -> parquet
```

输出：

```text
cache/market_features/*.parquet
cache/market_features/manifest.json
```

---

### Task 4：Top100 factor panel 计算与保存

文件：

```text
src/alpha_mvp/cache/factor_cache.py
```

目标：

```text
读取 outputs\real_all\top100.csv
生成 factor_id
计算 factor panel
保存 factor panel parquet
支持断点跳过
```

输出：

```text
cache/factor_panels/F0001_xxxxxxxx.parquet
cache/factor_panels/manifest.json
```

---

### Task 5：批量 analytics

文件：

```text
src/alpha_mvp/validation/analytics_fast.py
```

目标：

```text
IC/IR/RankIC/RankIR
rolling IC/IR
train/test
size bucket
industry
```

输出：

```text
metrics/summary.csv
metrics/rolling_ic.parquet
metrics/group_size.csv
metrics/group_industry.csv
```

---

### Task 6：VectorBot 回测

文件：

```text
src/alpha_mvp/validation/vectorbot.py
```

目标：

```text
top5/top10
5日调仓
纯多头
手续费/滑点
净值/Sharpe/MaxDD/换手
```

---

### Task 7：Alphalens 精选因子

文件：

```text
src/alpha_mvp/validation/alphalens_runner.py
```

目标：

```text
只跑 selected top N
保存 tear sheet png
保存 clean_factor_data.parquet
```

---

### Task 8：Plotly HTML 报告

文件：

```text
src/alpha_mvp/validation/report_builder.py
src/alpha_mvp/validation/plots.py
```

目标：

```text
validation_report.html
每个 factor card
交互式 rolling IC / equity / size / industry 图
```

---

## 14. 最小验证路径

第一步只跑 Top3：

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\real_all\top100.csv" `
  --out "outputs\real_all_validation_smoke" `
  --start 20180101 `
  --end 20260430 `
  --top-n 3 `
  --alphalens-top-n 1 `
  --cache-features `
  --cache-factors
```

第二步跑 Top20：

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --out "outputs\real_all_validation_top20" `
  --top-n 20 `
  --alphalens-top-n 5
```

第三步跑 Top100：

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --out "outputs\real_all_validation_2018" `
  --top-n 100 `
  --alphalens-top-n 10
```

---

## 15. 不建议事项

### 不建议 1：一开始 GPU

当前瓶颈不是神经网络矩阵乘，而是 rolling/rank/corr/groupby/io。GPU 引入 CuPy/cuDF 成本高，且 VectorBT/Alphalens 生态主要是 CPU/Pandas/NumPy。

### 不建议 2：Alphalens 跑全部 Top100

Alphalens 适合精细诊断，不适合作为批量验证主引擎。Top100 全部跑会慢且输出难消化。

### 不建议 3：factor panel 不落盘

如果不保存，后续每次改可视化或回测参数都要重算表达式，非常浪费。

### 不建议 4：继续扩大 pipeline.py

`pipeline.py` 已经足够复杂，验证层应独立，不要继续堆功能。

---

## 16. 最终推荐路线

```text
第一阶段：
  保存 2018 至今全市场基础字段 panel cache

第二阶段：
  读取 outputs\real_all\top100.csv
  计算 Top100 factor panel
  保存为 parquet

第三阶段：
  基于 factor parquet 做批量 IC / RankIC / rolling / size / industry

第四阶段：
  基于 factor parquet 做 top5/top10 纯多头 vectorbot 回测

第五阶段：
  精选 10 个因子跑 Alphalens-reloaded

第六阶段：
  Plotly + HTML 生成总报告
```

完成后，你的系统就从“初筛挖掘脚本”升级为：

```text
可复用的全市场因子验证平台
```
