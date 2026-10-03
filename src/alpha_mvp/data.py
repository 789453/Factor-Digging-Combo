from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

TABLE_CANDIDATES = {
    "daily": ["silver.fact_stock_daily", "stock_daily", "silver.stock_daily"],
    "moneyflow": ["silver.fact_stock_moneyflow", "stock_moneyflow", "silver.stock_moneyflow"],
    "cyq": ["silver.fact_stock_cyq_perf", "stock_cyq_perf", "silver.stock_cyq_perf"],
    "basic": ["silver.fact_stock_daily_basic", "stock_daily_basic", "silver.stock_daily_basic"],
    "snapshot": ["silver.fact_stock_basic_snapshot", "stock_basic_snapshot", "silver.stock_basic_snapshot"],
}

def load_pool(path: str | None) -> list[str] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _first_existing_table(conn, candidates):
    existing = set()
    try:
        rows = conn.execute("select table_schema || '.' || table_name as name from information_schema.tables").fetchall()
        existing |= {r[0] for r in rows}
        rows2 = conn.execute("select table_name as name from information_schema.tables").fetchall()
        existing |= {r[0] for r in rows2}
    except Exception:
        pass
    for t in candidates:
        if t in existing:
            return t
    return candidates[0]

def load_from_duckdb(duckdb_path: str, pool_json: str | None, start: str, end: str, columns: list[str] | None = None) -> pd.DataFrame:
    import duckdb
    pool = load_pool(pool_json)

    conn = duckdb.connect(duckdb_path, read_only=True)
    conn.execute("SET threads=16")
    temp_dir = Path(tempfile.gettempdir()) / "alpha_mvp_duckdb"
    # 进一步优化并发设置
    conn.execute("SET threads=16")
    conn.execute("SET memory_limit='32GB'")
    conn.execute("SET preserve_insertion_order=false") # 提高查询并行度
    temp_dir = Path(tempfile.gettempdir()) / "alpha_mvp_duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    escaped_temp_dir = str(temp_dir).replace("'", "''")
    conn.execute(f"SET temp_directory='{escaped_temp_dir}'")

    if pool:
        pool_list = [{"ts_code": c} for c in pool]
        conn.execute("CREATE TEMP TABLE pool_tmp AS SELECT * FROM (VALUES " + ",".join([f"('{c}')" for c in pool]) + ") AS t(ts_code)")
        pool_join = "JOIN pool_tmp p ON d.ts_code = p.ts_code"
    else:
        pool_join = ""

    daily = _first_existing_table(conn, TABLE_CANDIDATES["daily"])
    money = _first_existing_table(conn, TABLE_CANDIDATES["moneyflow"])
    cyq = _first_existing_table(conn, TABLE_CANDIDATES["cyq"])
    basic = _first_existing_table(conn, TABLE_CANDIDATES["basic"])
    snap = _first_existing_table(conn, TABLE_CANDIDATES["snapshot"])

    # 默认选择列，如果传入了 columns 则按需选择（减少 IO）
    # 注意：trade_date 和 ts_code 是必须的
    all_possible_cols = {
        "open": "d.open", "high": "d.high", "low": "d.low", "close": "d.close", 
        "pre_close": "d.pre_close", "pct_chg": "d.pct_chg", "vol": "d.vol", "amount": "d.amount",
        "buy_sm_amount": "m.buy_sm_amount", "sell_sm_amount": "m.sell_sm_amount", 
        "buy_md_amount": "m.buy_md_amount", "sell_md_amount": "m.sell_md_amount",
        "buy_lg_amount": "m.buy_lg_amount", "sell_lg_amount": "m.sell_lg_amount", 
        "net_mf_amount": "m.net_mf_amount",
        "his_low": "c.his_low", "his_high": "c.his_high", "cost_5pct": "c.cost_5pct", 
        "cost_15pct": "c.cost_15pct", "cost_50pct": "c.cost_50pct",
        "cost_85pct": "c.cost_85pct", "cost_95pct": "c.cost_95pct", 
        "weight_avg": "c.weight_avg", "winner_rate": "c.winner_rate",
        "turnover_rate": "b.turnover_rate", "turnover_rate_f": "b.turnover_rate_f", 
        "volume_ratio": "b.volume_ratio", "total_mv": "b.total_mv", "circ_mv": "b.circ_mv",
        "industry": "s.industry"
    }
    
    selected_cols_str = ", ".join([all_possible_cols[c] for c in (columns if columns else all_possible_cols.keys())])

    q = f"""
    select
      d.ts_code, d.trade_date,
      {selected_cols_str}
    from {daily} d
    {pool_join}
    left join {money} m on d.ts_code=m.ts_code and d.trade_date=m.trade_date
    left join {cyq} c on d.ts_code=c.ts_code and d.trade_date=c.trade_date
    left join {basic} b on d.ts_code=b.ts_code and d.trade_date=b.trade_date
    left join {snap} s on d.ts_code=s.ts_code
    where d.trade_date >= '{start}' and d.trade_date <= '{end}'
    """
    # 移除 order by 以减少 DuckDB 端的计算开销，我们后面在 make_panels 中通过 mapping 处理顺序
    df = conn.execute(q).fetchdf()
    conn.close()
    return df


def _numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse vendor numeric strings without treating malformed values as zero."""
    for column in columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_ratio(numerator, denominator):
    denominator = np.asarray(denominator, dtype=float)
    return np.asarray(numerator, dtype=float) / np.where(
        np.abs(denominator) < 1e-12, np.nan, denominator
    )


def load_futures_parquet(
    parquet_path: str,
    futures_universe: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load one continuous futures contract per product and trading day.

    ``commodity_main`` prefers the vendor's main-contract flag and then its
    continuous-contract flag.  ``index_continuous`` contains only the vendor's
    continuous series.  Both outputs conform to the research panel contract:
    ``trade_date``, ``ts_code``, OHLC, volume, and futures-specific OI fields.
    Settlement is used as close because it is the only fully covered price in
    both vendor extracts.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Futures parquet does not exist: {path}")
    if futures_universe == "commodity_main":
        required = [
            "product_code", "contract_name", "end_date", "main_flag",
            "continuous_flag", "open", "high", "low", "close",
            "settlement", "volume", "open_interest", "open_interest_change",
        ]
        raw = pd.read_parquet(path, columns=required)
        raw = raw.loc[
            raw["main_flag"].eq("1") | raw["continuous_flag"].eq("1")
        ].copy()
        raw["_priority"] = np.where(raw["main_flag"].eq("1"), 0, 1)
        raw = raw.sort_values(["end_date", "product_code", "_priority", "contract_name"])
        raw = raw.drop_duplicates(["end_date", "product_code"], keep="first")
        raw["ts_code"] = "FUT_COM_" + raw["product_code"].astype(str)
    elif futures_universe == "index_continuous":
        required = [
            "underlying_id", "contract_name", "end_date", "continuous_flag",
            "open", "high", "low", "close", "settlement", "volume",
            "open_interest", "open_interest_change",
        ]
        raw = pd.read_parquet(path, columns=required)
        raw = raw.loc[raw["continuous_flag"].eq("1")].copy()
        raw = raw.sort_values(["end_date", "underlying_id", "contract_name"])
        raw = raw.drop_duplicates(["end_date", "underlying_id"], keep="first")
        raw["ts_code"] = "FUT_IDX_" + raw["underlying_id"].astype(str)
    else:
        raise ValueError(
            "Unsupported futures universe: "
            f"{futures_universe}; expected commodity_main or index_continuous"
        )

    raw["trade_date"] = pd.to_datetime(raw["end_date"], errors="coerce").dt.strftime("%Y%m%d")
    raw = raw.loc[raw["trade_date"].notna() & raw["trade_date"].between(start, end)].copy()
    raw = _numeric_columns(
        raw,
        ["open", "high", "low", "close", "settlement", "volume", "open_interest", "open_interest_change"],
    )
    raw = raw.sort_values(["ts_code", "trade_date"]).copy()
    # Settlement is complete in the supplied files; it prevents sparse close
    # values from turning a cross-sectional signal into a missing-data proxy.
    raw["close"] = raw["settlement"]
    raw["pre_close"] = raw.groupby("ts_code", sort=False)["close"].shift(1)
    raw["open"] = raw["open"].fillna(raw["pre_close"])
    raw["high"] = raw["high"].fillna(raw[["open", "close"]].max(axis=1))
    raw["low"] = raw["low"].fillna(raw[["open", "close"]].min(axis=1))
    raw["vol"] = raw["volume"].where(raw["volume"] >= 0)
    raw["open_interest"] = raw["open_interest"].where(raw["open_interest"] > 0)
    # Contract multipliers are not supplied.  This is an activity proxy only;
    # all configured futures fields normalize it before cross-sectional use.
    raw["amount"] = raw["close"].abs() * raw["vol"]
    raw["turnover_rate"] = raw["vol"] / raw["open_interest"]
    raw["volume_ratio"] = raw["vol"] / raw.groupby("ts_code", sort=False)["vol"].transform(
        lambda series: series.rolling(20, min_periods=10).mean()
    )
    output = raw[
        [
            "trade_date", "ts_code", "open", "high", "low", "close",
            "pre_close", "vol", "amount", "turnover_rate", "volume_ratio",
            "open_interest", "open_interest_change",
        ]
    ].copy()
    if output.empty:
        raise ValueError(
            "No futures rows remained after universe/date filtering; check "
            "parquet_path, futures_universe, start, and end"
        )
    if output.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Futures loader produced duplicate trade_date/ts_code rows")
    return output


def load_crypto_parquet(
    parquet_root: str,
    start: str,
    end: str,
    primary_timeframe: str = "1h",
    micro_timeframe: str = "15m",
    fast_timeframe: str | None = None,
) -> pd.DataFrame:
    """Build an hourly panel enriched by vectorized intrahour aggregates."""
    if primary_timeframe == "5m":
        return load_crypto_native_5m(parquet_root, start, end, micro_timeframe)
    root = Path(parquet_root)
    if not root.exists():
        raise FileNotFoundError(f"Crypto parquet root does not exist: {root}")
    symbols = sorted(
        path.name for path in root.iterdir()
        if path.is_dir()
        and (path / f"{primary_timeframe}.parquet").exists()
        and (path / f"{micro_timeframe}.parquet").exists()
        and (fast_timeframe is None or (path / f"{fast_timeframe}.parquet").exists())
    )
    if not symbols:
        raise ValueError(
            "No symbols have all requested primary/micro/fast timeframe parquet files"
        )
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    hourly_frames = []
    micro_frames: dict[str, list[pd.DataFrame]] = {micro_timeframe: []}
    if fast_timeframe and fast_timeframe != micro_timeframe:
        micro_frames[fast_timeframe] = []
    hourly_columns = [
        "date", "open", "high", "low", "close", "volume", "quote_volume",
        "trade_count", "taker_buy_volume", "taker_sell_volume",
        "taker_buy_ratio", "vwap",
    ]
    micro_columns = [
        "date", "open", "high", "low", "close", "volume", "quote_volume",
        "trade_count", "taker_buy_volume", "taker_buy_ratio", "log_return",
        "range_pct", "vwap",
    ]
    for symbol in symbols:
        hourly = pd.read_parquet(
            root / symbol / f"{primary_timeframe}.parquet",
            columns=hourly_columns,
        )
        hourly["date"] = pd.to_datetime(hourly["date"], utc=True)
        hourly = hourly.loc[hourly["date"].between(start_ts, end_ts)].copy()
        hourly["ts_code"] = symbol
        hourly_frames.append(hourly)

        for timeframe in micro_frames:
            micro = pd.read_parquet(
                root / symbol / f"{timeframe}.parquet", columns=micro_columns,
            )
            micro["date"] = pd.to_datetime(micro["date"], utc=True)
            micro = micro.loc[micro["date"].between(start_ts, end_ts)].copy()
            micro["ts_code"] = symbol
            micro["hour"] = micro["date"].dt.floor("h")
            micro_frames[timeframe].append(micro)
    hourly = pd.concat(hourly_frames, ignore_index=True)
    def aggregate_micro(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        frame = frame.sort_values(["ts_code", "hour", "date"]).copy()
        keys = [frame["ts_code"], frame["hour"]]
        grouped = frame.groupby(["ts_code", "hour"], sort=False)
        frame["signed_volume"] = frame["volume"] * (2.0 * frame["taker_buy_ratio"] - 1.0)
        frame["abs_return"] = frame["log_return"].abs()
        frame["return_sq"] = frame["log_return"].pow(2)
        frame["up_sq"] = frame["return_sq"].where(frame["log_return"] > 0, 0.0)
        frame["down_sq"] = frame["return_sq"].where(frame["log_return"] < 0, 0.0)
        frame["position"] = grouped.cumcount()
        frame["count"] = grouped["date"].transform("size")
        frame["late"] = frame["position"] >= (frame["count"] / 2.0)
        frame["late_volume"] = frame["volume"].where(frame["late"], 0.0)
        frame["early_volume"] = frame["volume"].where(~frame["late"], 0.0)
        frame["late_signed"] = frame["signed_volume"].where(frame["late"], 0.0)
        frame["early_signed"] = frame["signed_volume"].where(~frame["late"], 0.0)
        totals = grouped["volume"].transform("sum")
        trade_totals = grouped["trade_count"].transform("sum")
        frame["volume_share_sq"] = np.square(_safe_ratio(frame["volume"], totals))
        frame["trade_share_sq"] = np.square(_safe_ratio(frame["trade_count"], trade_totals))
        result = grouped.agg(
            bar_count=("date", "size"), first_return=("log_return", "first"),
            last_return=("log_return", "last"), path=("abs_return", "sum"),
            rv_sq=("return_sq", "sum"), up_rv_sq=("up_sq", "sum"),
            down_rv_sq=("down_sq", "sum"), max_abs_return=("abs_return", "max"),
            volume_sum=("volume", "sum"), volume_max=("volume", "max"),
            signed_volume=("signed_volume", "sum"), range_mean=("range_pct", "mean"),
            range_last=("range_pct", "last"), trade_sum=("trade_count", "sum"),
            trade_max=("trade_count", "max"), vwap_std=("vwap", "std"),
            late_volume=("late_volume", "sum"), early_volume=("early_volume", "sum"),
            late_signed=("late_signed", "sum"), early_signed=("early_signed", "sum"),
            volume_hhi=("volume_share_sq", "sum"), trade_hhi=("trade_share_sq", "sum"),
        ).reset_index()
        result = result.rename(columns={
            column: f"{prefix}_{column}" for column in result.columns
            if column not in {"ts_code", "hour"}
        })
        return result

    for timeframe, frames in micro_frames.items():
        prefix = "micro" if timeframe == micro_timeframe else "micro5"
        aggregates = aggregate_micro(pd.concat(frames, ignore_index=True), prefix)
        hourly = hourly.merge(
            aggregates, left_on=["ts_code", "date"], right_on=["ts_code", "hour"],
            how="left", validate="one_to_one",
        ).drop(columns=["hour"])
    hourly = hourly.sort_values(["ts_code", "date"]).copy()
    hourly["trade_date"] = hourly["date"].dt.strftime("%Y%m%d%H%M")
    hourly["pre_close"] = hourly.groupby("ts_code", sort=False)["close"].shift(1)
    hourly["vol"] = hourly["volume"]
    hourly["amount"] = hourly["quote_volume"]
    hourly["micro_realized_vol"] = np.sqrt(hourly["micro_rv_sq"])
    hourly["micro_volume_concentration"] = _safe_ratio(
        hourly["micro_volume_max"], hourly["micro_volume_sum"]
    )
    hourly["micro_taker_imbalance"] = _safe_ratio(
        hourly["micro_signed_volume"], hourly["micro_volume_sum"]
    )
    hourly["micro_trade_concentration"] = _safe_ratio(
        hourly["micro_trade_max"], hourly["micro_trade_sum"]
    )
    hourly["micro_range_expansion"] = _safe_ratio(
        hourly["micro_range_last"], hourly["micro_range_mean"]
    )
    hourly["micro_vwap_dispersion"] = _safe_ratio(
        hourly["micro_vwap_std"], hourly["close"].abs()
    )
    hourly["micro_path_efficiency"] = _safe_ratio(
        np.abs(np.log(hourly["close"] / hourly["open"])), hourly["micro_path"]
    )
    hourly["micro_return_reversal"] = (
        hourly["micro_last_return"] - hourly["micro_first_return"]
    )
    hourly["micro_upside_share"] = _safe_ratio(
        hourly["micro_up_rv_sq"], hourly["micro_rv_sq"]
    )
    hourly["micro_jump_share"] = _safe_ratio(
        hourly["micro_max_abs_return"], hourly["micro_path"]
    )
    hourly["micro_volume_trend"] = _safe_ratio(
        hourly["micro_late_volume"] - hourly["micro_early_volume"],
        hourly["micro_volume_sum"],
    )
    hourly["micro_imbalance_trend"] = _safe_ratio(
        hourly["micro_late_signed"] - hourly["micro_early_signed"],
        hourly["micro_volume_sum"],
    )
    extra_columns = ["micro_upside_share", "micro_jump_share", "micro_volume_hhi", "micro_trade_hhi"]
    if fast_timeframe and fast_timeframe != micro_timeframe:
        hourly["micro5_realized_vol"] = np.sqrt(hourly["micro5_rv_sq"])
        hourly["micro5_upside_share"] = _safe_ratio(hourly["micro5_up_rv_sq"], hourly["micro5_rv_sq"])
        hourly["micro5_jump_share"] = _safe_ratio(hourly["micro5_max_abs_return"], hourly["micro5_path"])
        hourly["micro5_path_efficiency"] = _safe_ratio(np.abs(np.log(hourly["close"] / hourly["open"])), hourly["micro5_path"])
        hourly["micro5_taker_imbalance"] = _safe_ratio(hourly["micro5_signed_volume"], hourly["micro5_volume_sum"])
        hourly["micro5_volume_trend"] = _safe_ratio(hourly["micro5_late_volume"] - hourly["micro5_early_volume"], hourly["micro5_volume_sum"])
        hourly["micro5_imbalance_trend"] = _safe_ratio(hourly["micro5_late_signed"] - hourly["micro5_early_signed"], hourly["micro5_volume_sum"])
        extra_columns.extend([
            "micro5_bar_count", "micro5_first_return", "micro5_last_return",
            "micro5_realized_vol", "micro5_upside_share", "micro5_jump_share",
            "micro5_path_efficiency", "micro5_taker_imbalance", "micro5_volume_trend",
            "micro5_imbalance_trend", "micro5_volume_hhi", "micro5_trade_hhi",
        ])
    output_columns = [
        "trade_date", "ts_code", "open", "high", "low", "close",
        "pre_close", "vol", "amount", "quote_volume", "trade_count",
        "taker_buy_volume", "taker_sell_volume", "taker_buy_ratio", "vwap",
        "micro_bar_count", "micro_first_return", "micro_last_return",
        "micro_realized_vol", "micro_volume_concentration",
        "micro_taker_imbalance", "micro_trade_concentration",
        "micro_volume_trend", "micro_imbalance_trend",
        "micro_range_expansion", "micro_vwap_dispersion",
        "micro_path_efficiency", "micro_return_reversal",
        *extra_columns,
    ]
    output = hourly[output_columns].copy()
    numeric = [column for column in output_columns if column not in {"trade_date", "ts_code"}]
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan)
    if output.empty:
        raise ValueError("No crypto rows remained after date filtering")
    if output.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Crypto loader produced duplicate trade_date/ts_code rows")
    return output


def load_crypto_native_5m(
    parquet_root: str, start: str, end: str, micro_timeframe: str = "15m",
) -> pd.DataFrame:
    """Completed 5m bars plus the most recently completed 15m bar, per asset."""
    root = Path(parquet_root)
    if micro_timeframe != "15m":
        raise ValueError("native 5m currently requires 15m context")
    if not root.exists():
        raise FileNotFoundError(f"Crypto parquet root does not exist: {root}")
    symbols = sorted(p.name for p in root.iterdir() if p.is_dir()
                     and (p / "5m.parquet").exists() and (p / "15m.parquet").exists())
    if not symbols:
        raise ValueError("No symbols have both 5m and 15m parquet files")
    start_ts, end_ts = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    frames = []
    columns = ["date", "open", "high", "low", "close", "volume", "quote_volume",
               "trade_count", "taker_buy_ratio", "vwap"]
    for symbol in symbols:
        fast = pd.read_parquet(root / symbol / "5m.parquet", columns=columns)
        fast["date"] = pd.to_datetime(fast["date"], utc=True)
        fast = fast.loc[fast["date"].between(start_ts, end_ts)].copy()
        if fast.empty:
            continue
        fast["available_at"] = fast["date"] + pd.Timedelta(minutes=5)
        context = pd.read_parquet(
            root / symbol / "15m.parquet",
            columns=["date", "open", "close", "taker_buy_ratio"],
        )
        context["date"] = pd.to_datetime(context["date"], utc=True)
        context = context.loc[
            context["date"].between(start_ts - pd.Timedelta(minutes=15), end_ts)
        ].copy()
        context["context_available_at"] = context["date"] + pd.Timedelta(minutes=15)
        context["ret_15m_completed"] = np.log(context["close"] / context["open"])
        valid_ratio = context["taker_buy_ratio"].where(
            context["taker_buy_ratio"].between(0, 1)
        )
        context["flow_15m_completed"] = 2.0 * valid_ratio - 1.0
        joined = pd.merge_asof(
            fast.sort_values("available_at"),
            context[["context_available_at", "ret_15m_completed",
                     "flow_15m_completed"]].sort_values("context_available_at"),
            left_on="available_at", right_on="context_available_at",
            direction="backward",
        )
        joined["ts_code"] = symbol
        frames.append(joined)
    if not frames:
        raise ValueError("No native 5m rows remained after date filtering")
    output = pd.concat(frames, ignore_index=True).sort_values(
        ["ts_code", "available_at"]
    )
    output["trade_date"] = output["available_at"].dt.strftime("%Y%m%d%H%M")
    output["pre_close"] = output.groupby("ts_code", sort=False)["close"].shift(1)
    output["vol"] = output["volume"]
    output["amount"] = output["quote_volume"]
    if output.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Native 5m loader produced duplicate time/asset rows")
    return output.drop(columns=["date", "available_at", "context_available_at"])

def make_simulated_data(n_days=90, n_stocks=60, start="2024-01-01", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates.strftime("%Y%m%d"), codes], names=["trade_date", "ts_code"])
    df = pd.DataFrame(index=idx).reset_index()
    n = len(df)

    base = rng.lognormal(mean=3.3, sigma=0.4, size=n_stocks)
    ret = rng.normal(0, 0.018, size=(n_days, n_stocks))
    flow_signal = rng.normal(0, 1, size=(n_days, n_stocks))
    ret[1:] += 0.0025 * np.tanh(flow_signal[:-1])
    close = base[None, :] * np.exp(np.cumsum(ret, axis=0))
    open_ = close * (1 + rng.normal(0, 0.006, size=close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, size=close.shape)))
    pre_close = np.vstack([close[0], close[:-1]])
    vol = rng.lognormal(12, 0.5, size=close.shape)
    amount = vol * close * 0.01
    flat = lambda x: x.reshape(-1)
    df["open"], df["high"], df["low"], df["close"] = flat(open_), flat(high), flat(low), flat(close)
    df["pre_close"], df["pct_chg"], df["vol"], df["amount"] = flat(pre_close), flat((close / pre_close - 1) * 100), flat(vol), flat(amount)

    for side in ["sm", "md", "lg"]:
        scale = {"sm": 0.7, "md": 0.5, "lg": 0.35}[side]
        buy = amount * np.maximum(0.01, 0.5 + scale * flow_signal + rng.normal(0, 0.2, close.shape))
        sell = amount * np.maximum(0.01, 0.5 - scale * flow_signal + rng.normal(0, 0.2, close.shape))
        df[f"buy_{side}_amount"] = flat(buy)
        df[f"sell_{side}_amount"] = flat(sell)
    df["net_mf_amount"] = df["buy_lg_amount"] + df["buy_md_amount"] - df["sell_lg_amount"] - df["sell_md_amount"]

    df["cost_50pct"] = df["close"] * (1 + rng.normal(0, 0.04, n))
    df["cost_5pct"] = df["cost_50pct"] * (1 - rng.uniform(0.05, 0.15, n))
    df["cost_15pct"] = df["cost_50pct"] * (1 - rng.uniform(0.03, 0.10, n))
    df["cost_85pct"] = df["cost_50pct"] * (1 + rng.uniform(0.03, 0.10, n))
    df["cost_95pct"] = df["cost_50pct"] * (1 + rng.uniform(0.05, 0.15, n))
    df["weight_avg"] = df["cost_50pct"] * (1 + rng.normal(0, 0.02, n))
    df["winner_rate"] = np.clip(50 + 300 * (df["close"] / df["weight_avg"] - 1) + rng.normal(0, 10, n), 0, 100)
    df["his_low"] = df["close"] * rng.uniform(0.55, 0.9, n)
    df["his_high"] = df["close"] * rng.uniform(1.1, 1.8, n)

    df["turnover_rate"] = rng.uniform(0.5, 8, n)
    df["turnover_rate_f"] = df["turnover_rate"] * rng.uniform(1.0, 2.0, n)
    df["volume_ratio"] = rng.lognormal(0, 0.5, n)
    df["circ_mv"] = rng.lognormal(6.5, 0.8, n)
    df["total_mv"] = df["circ_mv"] * rng.uniform(1.0, 1.8, n)
    industries = np.array(["电子", "医药", "机械", "化工", "计算机", "银行", "汽车", "电力"])
    code_ind = {c: industries[i % len(industries)] for i, c in enumerate(codes)}
    df["industry"] = df["ts_code"].map(code_ind)
    return df
