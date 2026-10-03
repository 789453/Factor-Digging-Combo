# 全市场 Top100 因子验证工程设计文档

适用仓库：`789453/feature-init-factor-mining-427`  
目标输入：`outputs\\real_all\\top100.csv`  
目标数据库：`D:\\Trading\\data_ever_26_3_14\\data\\meta\\warehouse.duckdb`  
目标阶段：在 9000+ 表达式挖掘完成后，对 Top100 因子做全市场真实数据验证、批量分层回测、单因子专业诊断与可视化报告。

---

## 1. 当前项目状态判断

当前项目主流程仍偏“因子挖掘和初筛”：

```text
src/alpha_mvp/
  config.py
  data.py
  evaluator.py
  fields.py
  grammar.py
  metrics.py
  ops.py
  parser.py
  pipeline.py
  validator.py
```

当前链路是：

```text
DuckDB 数据读取
→ 构建 30 个基础字段
→ 生成表达式
→ BatchEvaluator 在线计算
→ summarize_factor 初筛
→ factor_results.csv / top100.csv
```

现在你已经在 `outputs\\real_all` 完成 9000+ 表达式测试，并筛出了 `top100.csv`。下一阶段不应该继续塞进 `pipeline.py`，而应该新建独立验证层：

```text
src/alpha_mvp/validation/
```

原因：

1. 挖掘阶段强调快速筛选；
2. 验证阶段强调全市场、全指标、分组、可视化、报告；
3. VectorBT 更适合组合/信号回测；
4. Alphalens-reloaded 更适合单因子专业诊断；
5. 两者的数据结构、输出和性能瓶颈不同，应该分开。

---

## 2. VectorBT 与 Alphalens-reloaded 分工

### 2.1 VectorBT / 自研 VectorBot 层

用于：

```text
Top5% / Top10% 纯多头
每 n 天调仓
因子方向自动识别
多因子批量策略曲线
组合收益、回撤、夏普、换手
分层组合收益
```

但 IC、RankIC、行业分组、size bucket、rolling IC 这些更适合自研 analytics 模块。

### 2.2 Alphalens-reloaded 层

用于少量精选单因子：

```text
get_clean_factor_and_forward_returns
create_summary_tear_sheet
create_returns_tear_sheet
create_information_tear_sheet
IC 图
分位数组合图
因子收益图
换手图
Rank autocorrelation
```

不建议一次性对 100 个因子全跑 tear sheet。建议只对 Top 5~20 个，或不同模板族代表因子做标准诊断。

---

## 3. 推荐新增文件架构

```text
src/alpha_mvp/validation/
  __init__.py
  config.py
  top_factors.py
  market_data.py
  factor_compute.py
  panel_store.py
  analytics.py
  size_industry.py
  vectorbt_runner.py
  alphalens_runner.py
  reports.py
  runner.py
  cli_validate.py
```

输出目录：

```text
outputs/validation_real_all/
  factor_panels/
    F0001_xxxxxxxx.parquet
    F0002_xxxxxxxx.parquet
  metrics/
    factor_metrics_summary.csv
    group_metrics_by_size.csv
    group_metrics_by_industry.csv
    rolling_ic.csv
  vectorbt/
    portfolio_summary.csv
    top5_equity_curves.parquet
    top10_equity_curves.parquet
    layer_returns.parquet
  alphalens/
    F0001_xxxxxxxx/
      clean_factor_data.parquet
      summary_tear_sheet.png
      returns_tear_sheet.png
      information_tear_sheet.png
  reports/
    validation_report.html
  validation.sqlite3
```

---

## 4. config.py

文件：`src/alpha_mvp/validation/config.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationConfig:
    duckdb_path: str
    top100_path: str = "outputs/real_all/top100.csv"
    out_dir: str = "outputs/validation_real_all"
    start: str = "20240101"
    end: str = "20260430"

    horizon: int = 5
    rebalance_n: int = 5

    min_price: float = 1.0
    min_amount: float = 0.0
    exclude_st: bool = False

    top_quantiles: tuple[float, ...] = (0.05, 0.10)
    layer_quantiles: int = 5
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    train_end: str = "20250831"
    test_start: str = "20250901"

    alphalens_top_n: int = 10
    alphalens_periods: tuple[int, ...] = (1, 5, 10)
    alphalens_quantiles: int = 5
    alphalens_max_loss: float = 0.45

    factor_batch_size: int = 10
    write_factor_panels: bool = True
    use_gpu: bool = False
    duckdb_threads: int = 24
    duckdb_memory_limit: str = "24GB"
```

---

## 5. top_factors.py：读取 top100 并生成 factor_id

文件：`src/alpha_mvp/validation/top_factors.py`

```python
import hashlib
import pandas as pd

def make_factor_id(expr: str, i: int) -> str:
    h = hashlib.md5(expr.encode("utf-8")).hexdigest()[:8]
    return f"F{i:04d}_{h}"

def load_top_factors(path: str, top_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    expr_col = "canonical" if "canonical" in df.columns else "expr"
    df = df.dropna(subset=[expr_col]).copy()
    df = df.drop_duplicates(subset=[expr_col])
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    df["factor_expr"] = df[expr_col].astype(str)
    df["factor_id"] = [
        make_factor_id(expr, i + 1)
        for i, expr in enumerate(df["factor_expr"])
    ]
    return df
```

---

## 6. market_data.py：全市场数据读取

文件：`src/alpha_mvp/validation/market_data.py`

```python
import duckdb
import pandas as pd

TABLE_CANDIDATES = {
    "daily": ["silver.fact_stock_daily", "stock_daily", "silver.stock_daily"],
    "moneyflow": ["silver.fact_stock_moneyflow", "stock_moneyflow", "silver.stock_moneyflow"],
    "cyq": ["silver.fact_stock_cyq_perf", "stock_cyq_perf", "silver.stock_cyq_perf"],
    "basic": ["silver.fact_stock_daily_basic", "stock_daily_basic", "silver.stock_daily_basic"],
    "snapshot": ["silver.fact_stock_basic_snapshot", "stock_basic_snapshot", "silver.stock_basic_snapshot"],
}

def first_existing_table(conn, candidates):
    rows = conn.execute(
        "select table_schema || '.' || table_name as name from information_schema.tables"
    ).fetchall()
    names = {r[0] for r in rows}
    rows2 = conn.execute("select table_name from information_schema.tables").fetchall()
    names |= {r[0] for r in rows2}
    for t in candidates:
        if t in names:
            return t
    return candidates[0]

def load_market_data(cfg):
    conn = duckdb.connect(cfg.duckdb_path, read_only=True)
    conn.execute(f"set threads={cfg.duckdb_threads}")
    conn.execute(f"set memory_limit='{cfg.duckdb_memory_limit}'")

    daily = first_existing_table(conn, TABLE_CANDIDATES["daily"])
    money = first_existing_table(conn, TABLE_CANDIDATES["moneyflow"])
    cyq = first_existing_table(conn, TABLE_CANDIDATES["cyq"])
    basic = first_existing_table(conn, TABLE_CANDIDATES["basic"])
    snap = first_existing_table(conn, TABLE_CANDIDATES["snapshot"])

    q = f'''
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
    from {daily} d
    left join {money} m
      on d.ts_code=m.ts_code and d.trade_date=m.trade_date
    left join {cyq} c
      on d.ts_code=c.ts_code and d.trade_date=c.trade_date
    left join {basic} b
      on d.ts_code=b.ts_code and d.trade_date=b.trade_date
    left join {snap} s
      on d.ts_code=s.ts_code
    where d.trade_date >= '{cfg.start}'
      and d.trade_date <= '{cfg.end}'
      and d.close is not null
      and d.close > {cfg.min_price}
    order by d.trade_date, d.ts_code
    '''
    df = conn.execute(q).fetchdf()
    conn.close()
    return df
```

全市场约 `557 × 5000 ≈ 278.5 万行`。32GB 内存足够第一版一次性读取，但不要每个因子重复读取。应一次读取、一次构建 panels，然后 Top100 复用。

---

## 7. factor_compute.py：计算 Top100 factor panels

文件：`src/alpha_mvp/validation/factor_compute.py`

```python
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_mvp.fields import add_basic_features, DEFAULT_FEATURES
from alpha_mvp.evaluator import BatchEvaluator, make_panels

def panel_to_long(arr, dates, codes, value_name="value"):
    df = pd.DataFrame(arr, index=pd.Index(dates, name="trade_date"), columns=codes)
    s = df.stack(dropna=False).rename(value_name).reset_index()
    s.columns = ["trade_date", "ts_code", value_name]
    return s

def build_feature_panels(raw_df):
    df = add_basic_features(raw_df)
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    panels, dates, codes = make_panels(df, feature_cols, value_col="close")
    meta = (
        df[["trade_date", "ts_code", "industry", "circ_mv"]]
        .drop_duplicates(["trade_date", "ts_code"])
    )
    return df, panels, dates, codes, meta

def compute_factor_panels(top_factors, panels, dates, codes, cfg):
    evaluator = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k != "close"},
        dates=dates,
        codes=codes,
        windows=(10, 20, 30, 40, 50),
        max_depth=6,
        max_nodes=16,
    )

    out_dir = Path(cfg.out_dir) / "factor_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    status_rows = []

    for _, row in top_factors.iterrows():
        fid = row["factor_id"]
        expr = row["factor_expr"]
        arr, status = evaluator.eval_expr(expr)

        status_rows.append({
            "factor_id": fid,
            "expr": expr,
            "status": status,
            "coverage": float(np.isfinite(arr).mean()) if arr is not None else np.nan,
        })

        if arr is None:
            continue

        results[fid] = arr

        if cfg.write_factor_panels:
            long_df = panel_to_long(arr, dates, codes, value_name="factor")
            long_df["factor_id"] = fid
            long_df.to_parquet(out_dir / f"{fid}.parquet", index=False)

    return results, pd.DataFrame(status_rows)
```

---

## 8. analytics.py：全市场 IC / RankIC / 分组 / rolling

文件：`src/alpha_mvp/validation/analytics.py`

需要输出：

```text
metrics/factor_metrics_summary.csv
metrics/group_metrics_by_size.csv
metrics/group_metrics_by_industry.csv
metrics/rolling_ic.csv
```

核心指标：

```text
coverage
mean_ic
icir
mean_rank_ic
rank_icir
positive_rank_ic_ratio
train_mean_rank_ic
train_rank_icir
test_mean_rank_ic
test_rank_icir
rolling_20_rank_ic
rolling_60_rank_ic
rolling_120_rank_ic
```

分组：

```text
size_bucket:
  micro, small, mid, large, mega

industry:
  stock_basic_snapshot.industry
```

市值桶建议每日动态分位数，不建议固定阈值。

伪代码：

```python
def make_size_bucket(meta, dates, codes):
    circ = meta.pivot(index="trade_date", columns="ts_code", values="circ_mv").reindex(index=dates, columns=codes)
    log_mv = np.log1p(circ.to_numpy(dtype=float))
    bucket = np.full(log_mv.shape, -1, dtype=int)
    for t in range(log_mv.shape[0]):
        row = log_mv[t]
        m = np.isfinite(row)
        if m.sum() < 50:
            continue
        ranks = pd.Series(row[m]).rank(pct=True).to_numpy()
        b = np.floor(ranks * 5).astype(int)
        b[b == 5] = 4
        bucket[t, np.where(m)[0]] = b
    return bucket
```

---

## 9. vectorbt_runner.py：Top5/Top10 纯多头回测

文件：`src/alpha_mvp/validation/vectorbt_runner.py`

第一版建议用自研矩阵回测作为主实现，再可选调用 VectorBT 生成 stats。

### 9.1 因子方向

方向只用 train 期判断，避免泄露：

```python
direction = 1 if train_mean_rank_ic >= 0 else -1
effective_factor = factor * direction
```

### 9.2 每 5 天调仓 Top5/Top10

```python
import numpy as np

def make_rebalance_mask(dates, n=5):
    mask = np.zeros(len(dates), dtype=bool)
    mask[::n] = True
    return mask

def make_top_weights(factor, dates, top_pct=0.05, rebalance_n=5, direction=1):
    x = factor * direction
    T, N = x.shape
    weights = np.zeros((T, N), dtype=float)
    rebalance = make_rebalance_mask(dates, rebalance_n)
    last_w = np.zeros(N, dtype=float)

    for t in range(T):
        if rebalance[t]:
            row = x[t]
            valid = np.isfinite(row)
            n_valid = valid.sum()
            if n_valid > 0:
                k = max(1, int(n_valid * top_pct))
                valid_idx = np.where(valid)[0]
                selected = valid_idx[np.argsort(row[valid])[-k:]]
                w = np.zeros(N, dtype=float)
                w[selected] = 1.0 / k
                last_w = w
        weights[t] = last_w
    return weights

def portfolio_returns(weights, close, fee_bps=10.0, slippage_bps=5.0):
    ret = close[1:] / close[:-1] - 1
    w = weights[:-1]
    gross = np.nansum(w * ret, axis=1)

    turnover = np.nansum(np.abs(weights[1:] - weights[:-1]), axis=1)
    cost = turnover * (fee_bps + slippage_bps) / 10000.0

    net = gross - cost
    return net, turnover
```

### 9.3 输出指标

```text
factor_id
top_pct
direction
ann_return
ann_vol
sharpe
max_drawdown
calmar
win_rate
avg_turnover
total_return
train_ann_return
test_ann_return
test_sharpe
```

---

## 10. alphalens_runner.py：单因子专业诊断

文件：`src/alpha_mvp/validation/alphalens_runner.py`

### 10.1 数据格式

Alphalens 需要：

```text
factor:
  pd.Series
  MultiIndex(date, asset)

prices:
  pd.DataFrame
  index=date
  columns=asset

groupby:
  asset -> industry
```

你的 `trade_date` 必须转成 datetime：

```python
pd.to_datetime(trade_date, format="%Y%m%d")
```

asset 必须完全等于 `ts_code`。

### 10.2 代码骨架

```python
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import alphalens as al

def panel_to_alphalens_series(arr, dates, codes):
    df = pd.DataFrame(
        arr,
        index=pd.to_datetime(dates, format="%Y%m%d"),
        columns=codes,
    )
    s = df.stack(dropna=False)
    s.index.names = ["date", "asset"]
    return s.sort_index()

def run_alphalens_for_factor(fid, factor_arr, dates, codes, close, industry_map, cfg):
    out = Path(cfg.out_dir) / "alphalens" / fid
    out.mkdir(parents=True, exist_ok=True)

    factor = panel_to_alphalens_series(factor_arr, dates, codes)
    prices = pd.DataFrame(
        close,
        index=pd.to_datetime(dates, format="%Y%m%d"),
        columns=codes,
    )

    clean = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        groupby=industry_map,
        quantiles=cfg.alphalens_quantiles,
        periods=cfg.alphalens_periods,
        max_loss=cfg.alphalens_max_loss,
    )

    clean.to_parquet(out / "clean_factor_data.parquet")

    al.tears.create_summary_tear_sheet(clean)
    plt.savefig(out / "summary_tear_sheet.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    al.tears.create_returns_tear_sheet(clean)
    plt.savefig(out / "returns_tear_sheet.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    al.tears.create_information_tear_sheet(clean)
    plt.savefig(out / "information_tear_sheet.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    return clean
```

---

## 11. runner.py：总控流程

文件：`src/alpha_mvp/validation/runner.py`

```python
from pathlib import Path

def run_validation(cfg):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    top_factors = load_top_factors(cfg.top100_path)

    raw = load_market_data(cfg)

    feature_df, panels, dates, codes, meta = build_feature_panels(raw)

    factor_panels, status = compute_factor_panels(
        top_factors, panels, dates, codes, cfg
    )
    status.to_csv(out / "factor_compute_status.csv", index=False, encoding="utf-8-sig")

    metrics = run_factor_analytics(
        factor_panels=factor_panels,
        close=panels["close"],
        dates=dates,
        codes=codes,
        meta=meta,
        cfg=cfg,
    )

    bt = run_vectorbt_like_backtest(
        factor_panels=factor_panels,
        close=panels["close"],
        dates=dates,
        codes=codes,
        factor_metrics=metrics["summary"],
        cfg=cfg,
    )

    selected = select_alphalens_factors(metrics["summary"], top_n=cfg.alphalens_top_n)
    run_alphalens_batch(selected, factor_panels, panels["close"], dates, codes, meta, cfg)

    build_validation_report(out, metrics, bt, selected)
```

---

## 12. cli_validate.py

文件：`src/alpha_mvp/validation/cli_validate.py`

```python
import argparse
from .config import ValidationConfig
from .runner import run_validation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", required=True)
    p.add_argument("--top100", default="outputs/real_all/top100.csv")
    p.add_argument("--out", default="outputs/validation_real_all")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20260430")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rebalance-n", type=int, default=5)
    p.add_argument("--alphalens-top-n", type=int, default=10)
    p.add_argument("--duckdb-threads", type=int, default=24)
    p.add_argument("--duckdb-memory-limit", default="24GB")
    args = p.parse_args()

    cfg = ValidationConfig(
        duckdb_path=args.duckdb,
        top100_path=args.top100,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        rebalance_n=args.rebalance_n,
        alphalens_top_n=args.alphalens_top_n,
        duckdb_threads=args.duckdb_threads,
        duckdb_memory_limit=args.duckdb_memory_limit,
    )
    run_validation(cfg)

if __name__ == "__main__":
    main()
```

---

## 13. 最小测试命令

### 13.1 先跑 3 个因子

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\real_all\top100.csv" `
  --out "outputs\validation_real_all_smoke" `
  --start 20180101 `
  --end 20260430 `
  --horizon 5 `
  --rebalance-n 5 `
  --alphalens-top-n 1
```

### 13.2 再跑 20 个因子

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\real_all\top100.csv" `
  --out "outputs\validation_real_all_top20" `
  --horizon 5 `
  --rebalance-n 5 `
  --alphalens-top-n 5
```

### 13.3 最后跑 100 个，Alphalens 只跑 10 个

```powershell
python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\real_all\top100.csv" `
  --out "outputs\validation_real_all" `
  --horizon 5 `
  --rebalance-n 5 `
  --alphalens-top-n 10
```

---

## 14. 性能策略

### 14.1 不建议一开始用 GPU

当前主要计算是：

```text
rolling
rank
corr
groupby
因子分层
组合权重生成
Alphalens Pandas 分析
```

这不是 GPU 最擅长的深度学习矩阵运算。第一版更应依赖：

```text
DuckDB threads=24
NumPy / Pandas / Numba
一次读取全市场数据
复用 panels
factor panel 落 Parquet
Alphalens 只跑精选因子
```

### 14.2 内存估算

单个 factor panel：

```text
557 × 5000 × float64 ≈ 22 MB
```

Top100 理论上约 2.2GB，32GB 内存可接受。但图表、Alphalens clean data、临时 DataFrame 会额外占内存，所以建议：

```text
因子逐个保存 parquet
analytics 分批处理
Alphalens 分批跑并及时 plt.close("all")
```

---

## 15. 结果判断标准

不要只看一个 score。推荐排序：

```text
1. test_rank_ic 与 train_rank_ic 同向
2. test_rank_icir 绝对值较高
3. rolling_60_rank_ic 稳定
4. top5/top10 组合收益曲线不崩
5. size bucket 不是只在微盘有效
6. industry 分组不是只靠单一行业
7. turnover 不过高
8. Alphalens 分位数组合有单调性
```

示例筛选条件：

```text
abs(test_rank_ic) > 0.015
abs(test_rank_icir) > 0.15
train/test same sign
top10 test sharpe > 0.5
industry positive ratio > 50%
非单一 size bucket 贡献
```

---

## 16. 关键工程坑

### 16.1 Alphalens 日期和 asset

必须保证：

```text
factor index: MultiIndex(date, asset)
prices index: date
prices columns: asset
asset 名称完全等于 ts_code
date 是 pandas datetime，不是 "YYYYMMDD" 字符串
```

### 16.2 forward return 对齐

如果因子在 t 日收盘后计算，应该预测 t+1 到 t+5。  
纯多头回测不能让 `factor[t]` 乘 `return[t]`，而应使用：

```text
weights[t] × return[t+1]
```

### 16.3 行业是静态快照

`stock_basic_snapshot.industry` 是静态行业，短周期验证可以先用；后续更严谨应接历史行业分类。

### 16.4 市值分组必须每日动态

按每天 `log(circ_mv)` 分位数分组，不要固定市值阈值。

### 16.5 ST / 停牌 / 涨跌停

第一版可先不做，正式验证必须接：

```text
stock_suspend_d
stock_stk_limit
ST 标记
涨停不可买
跌停不可卖
```

---

## 17. 最终建议

第一版验证系统不要做成“大而全回测平台”，而应做成：

```text
Top100 因子全市场再验证器
```

主线：

```text
top100.csv
→ factor_id
→ 全市场数据
→ 基础字段 panels
→ Top100 factor panels
→ IC / RankIC / rolling / group analytics
→ top5 / top10 纯多头组合回测
→ 精选因子 Alphalens tear sheet
→ validation_report.html
```

VectorBT 和 Alphalens 分工：

```text
VectorBT / 自研 vectorbot：
  批量 top quantile 组合回测、净值、Sharpe、回撤、换手。

Alphalens-reloaded：
  少量精选单因子的专业 tear sheet 和可视化诊断。
```

不要让 Alphalens 承担 Top100 全市场批量搜索；不要让 VectorBT 承担所有 IC/分组/rolling 诊断。两者应通过统一的 factor panel 和 factor_id 连接。
