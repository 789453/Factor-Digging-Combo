from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

EPS = 1e-9
FIELD_FORMULA_VERSION = "2026-09-05-v3"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: str
    dependencies: tuple[str, ...]
    description: str
    role: str = "search"
    enabled: bool = True

def safe_div(a, b):
    return a / np.where(np.abs(b) < EPS, np.nan, b)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)

    df["ret_1d"] = g["close"].pct_change()
    df["hl_range"] = safe_div(df["high"] - df["low"], df["pre_close"])
    df["oc_ret"] = safe_div(df["close"] - df["open"], df["open"])
    df["upper_shadow"] = safe_div(df["high"] - np.maximum(df["open"], df["close"]), df["pre_close"])
    df["lower_shadow"] = safe_div(np.minimum(df["open"], df["close"]) - df["low"], df["pre_close"])
    df["close_pos"] = safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["gap_ret"] = safe_div(df["open"] - df["pre_close"], df["pre_close"])
    df["intraday_reversal"] = -safe_div(df["close"] - df["open"], df["high"] - df["low"])

    # A few futures settlements can be negative (for example, the 2020 oil
    # dislocation).  Liquidity magnitude remains meaningful in that case.
    df["amount_log"] = np.log1p(np.abs(df.get("amount", np.nan)))
    df["vol_log"] = np.log1p(df.get("vol", np.nan))
    df["turnover_log"] = np.log1p(df["turnover_rate"]) if "turnover_rate" in df else np.nan
    df["amount_per_vol"] = safe_div(df.get("amount", np.nan), df.get("vol", np.nan))
    df["price_volume_pressure"] = df["ret_1d"] * df["vol_log"]
    df["amplitude_turnover"] = df["hl_range"] * df["turnover_log"]
    df["volume_ratio_x_ret"] = df.get("volume_ratio", np.nan) * df["ret_1d"]

    sm_buy = df.get("buy_sm_amount", np.nan)
    sm_sell = df.get("sell_sm_amount", np.nan)
    md_buy = df.get("buy_md_amount", np.nan)
    md_sell = df.get("sell_md_amount", np.nan)
    lg_buy = df.get("buy_lg_amount", np.nan)
    lg_sell = df.get("sell_lg_amount", np.nan)

    df["sm_net_ratio"] = safe_div(sm_buy - sm_sell, sm_buy + sm_sell)
    df["md_net_ratio"] = safe_div(md_buy - md_sell, md_buy + md_sell)
    df["lg_net_ratio"] = safe_div(lg_buy - lg_sell, lg_buy + lg_sell)
    df["main_net_ratio"] = safe_div((md_buy + lg_buy) - (md_sell + lg_sell), md_buy + lg_buy + md_sell + lg_sell)
    df["retail_pressure"] = -df["sm_net_ratio"]
    df["big_vs_small_flow"] = df["lg_net_ratio"] - df["sm_net_ratio"]
    df["flow_imbalance"] = safe_div(df.get("net_mf_amount", np.nan), df.get("amount", np.nan))
    df["large_order_intensity"] = safe_div(lg_buy + lg_sell, sm_buy + sm_sell + md_buy + md_sell + lg_buy + lg_sell)
    df["active_big_buy_pressure"] = safe_div(lg_buy, lg_buy + lg_sell) - 0.5

    df["chip_width_90"] = safe_div(df.get("cost_95pct", np.nan) - df.get("cost_5pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_width_70"] = safe_div(df.get("cost_85pct", np.nan) - df.get("cost_15pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_cost_bias"] = safe_div(df["close"] - df.get("weight_avg", np.nan), df.get("weight_avg", np.nan))
    df["chip_median_bias"] = safe_div(df["close"] - df.get("cost_50pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_upper_pressure"] = safe_div(df.get("cost_95pct", np.nan) - df["close"], df["close"])
    df["chip_lower_support"] = safe_div(df["close"] - df.get("cost_5pct", np.nan), df["close"])
    df["winner_rate_norm"] = df.get("winner_rate", np.nan) / 100.0
    df["hist_price_position"] = safe_div(df["close"] - df.get("his_low", np.nan), df.get("his_high", np.nan) - df.get("his_low", np.nan))
    df["winner_cost_divergence"] = df["winner_rate_norm"] - df["chip_cost_bias"]

    df["size_log"] = np.log1p(df.get("circ_mv", np.nan))
    df["free_turnover_gap"] = df.get("turnover_rate_f", np.nan) - df.get("turnover_rate", np.nan)
    df["liquidity_crowding"] = df["turnover_log"] * df.get("volume_ratio", np.nan)

    # Versioned research vocabulary extensions. These use only information
    # available at or before the current row.
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_20d"] = g["close"].pct_change(20)
    market_ret = df.groupby("trade_date", sort=False)["ret_1d"].transform("mean")
    df["residual_ret_1d"] = df["ret_1d"] - market_ret
    df["realized_vol_10"] = g["ret_1d"].transform(
        lambda s: s.rolling(10, min_periods=6).std(ddof=0)
    )
    df["realized_vol_20"] = g["ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=11).std(ddof=0)
    )
    downside_sq = np.minimum(df["ret_1d"].fillna(0.0), 0.0) ** 2
    df["downside_vol_20"] = downside_sq.groupby(df["ts_code"]).transform(
        lambda s: np.sqrt(s.rolling(20, min_periods=11).mean())
    )
    close_delta_20 = g["close"].diff(20).abs()
    path_length_20 = g["close"].diff().abs().groupby(df["ts_code"]).transform(
        lambda s: s.rolling(20, min_periods=11).sum()
    )
    df["trend_efficiency_20"] = safe_div(close_delta_20, path_length_20)
    df["vol_change_5d"] = g["vol_log"].diff(5)
    df["amount_change_5d"] = g["amount_log"].diff(5)
    df["turnover_change_5d"] = g["turnover_log"].diff(5)
    amihud_raw = safe_div(df["ret_1d"].abs(), df.get("amount", np.nan)) * 1e8
    df["amihud_20"] = amihud_raw.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(20, min_periods=11).mean()
    )
    df["main_flow_persistence_5"] = g["main_net_ratio"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["flow_agreement"] = df["main_net_ratio"] * df["lg_net_ratio"]
    df["main_return_divergence"] = df["main_net_ratio"] - df["ret_1d"]
    df["chip_pressure_balance"] = (
        df["chip_lower_support"] - df["chip_upper_pressure"]
    )
    df["chip_concentration"] = safe_div(1.0, 1.0 + df["chip_width_90"].abs())
    df["winner_position_interaction"] = (
        df["winner_rate_norm"] * df["hist_price_position"]
    )
    turnover_mean_20 = g["turnover_log"].transform(
        lambda s: s.rolling(20, min_periods=11).mean()
    )
    turnover_std_20 = g["turnover_log"].transform(
        lambda s: s.rolling(20, min_periods=11).std(ddof=0)
    )
    df["liquidity_shock_20"] = safe_div(
        df["turnover_log"] - turnover_mean_20,
        turnover_std_20,
    )
    df["overnight_intraday_spread"] = df["gap_ret"] - df["oc_ret"]
    df["range_vol_ratio"] = safe_div(df["hl_range"], df["realized_vol_10"])

    # Futures-specific, scale-free open-interest vocabulary.  These fields are
    # intentionally harmless for equities: absent OI inputs propagate NaN.
    oi = (
        pd.to_numeric(df["open_interest"], errors="coerce")
        if "open_interest" in df
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    oi_change = (
        g["open_interest"].pct_change()
        if "open_interest" in df
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    df["oi_change_1d"] = oi_change
    df["oi_change_5d"] = (
        g["open_interest"].pct_change(5)
        if "open_interest" in df
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    df["oi_volume_ratio"] = safe_div(df.get("vol", np.nan), oi)
    df["oi_turnover_pressure"] = df["oi_volume_ratio"] * df["oi_change_1d"]
    df["oi_price_divergence"] = df["ret_1d"] - df["oi_change_1d"]
    df["oi_return_agreement"] = df["ret_1d"] * df["oi_change_1d"]
    oi_change_mean_5 = df["oi_change_1d"].groupby(df["ts_code"]).transform(
        lambda series: series.rolling(5, min_periods=3).mean()
    )
    df["oi_change_acceleration_5"] = df["oi_change_1d"] - oi_change_mean_5
    oi_log = np.log1p(oi.where(oi >= 0))
    oi_log_mean_20 = oi_log.groupby(df["ts_code"]).transform(
        lambda series: series.rolling(20, min_periods=10).mean()
    )
    oi_log_std_20 = oi_log.groupby(df["ts_code"]).transform(
        lambda series: series.rolling(20, min_periods=10).std(ddof=0)
    )
    df["oi_relative_20"] = safe_div(oi_log - oi_log_mean_20, oi_log_std_20)

    for c in DEFAULT_FEATURES:
        if c in df:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


FULL_MARKET_OPTIMIZED_FIELDS = {
    "amihud_20",
    "big_vs_small_flow",
    "chip_median_bias",
    "chip_pressure_balance",
    "flow_imbalance",
    "hist_price_position",
    "liquidity_shock_20",
    "main_flow_persistence_5",
    "main_return_divergence",
    "range_vol_ratio",
    "realized_vol_20",
    "residual_ret_1d",
    "ret_1d",
    "retail_pressure",
    "trend_efficiency_20",
    "vol_change_5d",
    "volume_ratio_x_ret",
}


def add_selected_features(
    df: pd.DataFrame,
    requested_fields: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Memory-conscious feature path for the frozen full-market Top50 audit."""
    requested = tuple(dict.fromkeys(requested_fields))
    if not set(requested).issubset(FULL_MARKET_OPTIMIZED_FIELDS):
        featured = add_basic_features(df)
        return featured[["ts_code", "trade_date", "close", *requested]].copy()

    df = df.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)
    df["ret_1d"] = g["close"].pct_change()
    df["hl_range"] = safe_div(df["high"] - df["low"], df["pre_close"])
    df["vol_log"] = np.log1p(df["vol"])
    df["turnover_log"] = np.log1p(df["turnover_rate"])

    sm_buy, sm_sell = df["buy_sm_amount"], df["sell_sm_amount"]
    md_buy, md_sell = df["buy_md_amount"], df["sell_md_amount"]
    lg_buy, lg_sell = df["buy_lg_amount"], df["sell_lg_amount"]
    df["sm_net_ratio"] = safe_div(sm_buy - sm_sell, sm_buy + sm_sell)
    df["lg_net_ratio"] = safe_div(lg_buy - lg_sell, lg_buy + lg_sell)
    df["main_net_ratio"] = safe_div(
        (md_buy + lg_buy) - (md_sell + lg_sell),
        md_buy + lg_buy + md_sell + lg_sell,
    )
    df["retail_pressure"] = -df["sm_net_ratio"]
    df["big_vs_small_flow"] = df["lg_net_ratio"] - df["sm_net_ratio"]
    df["flow_imbalance"] = safe_div(df["net_mf_amount"], df["amount"])
    df["volume_ratio_x_ret"] = df["volume_ratio"] * df["ret_1d"]

    df["chip_median_bias"] = safe_div(
        df["close"] - df["cost_50pct"],
        df["cost_50pct"],
    )
    chip_upper = safe_div(df["cost_95pct"] - df["close"], df["close"])
    chip_lower = safe_div(df["close"] - df["cost_5pct"], df["close"])
    df["chip_pressure_balance"] = chip_lower - chip_upper
    df["hist_price_position"] = safe_div(
        df["close"] - df["his_low"],
        df["his_high"] - df["his_low"],
    )

    market_ret = df.groupby("trade_date", sort=False)["ret_1d"].transform("mean")
    df["residual_ret_1d"] = df["ret_1d"] - market_ret
    df["realized_vol_10"] = g["ret_1d"].transform(
        lambda s: s.rolling(10, min_periods=6).std(ddof=0)
    )
    df["realized_vol_20"] = g["ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=11).std(ddof=0)
    )
    close_delta = g["close"].diff(20).abs()
    path_length = g["close"].diff().abs().groupby(df["ts_code"]).transform(
        lambda s: s.rolling(20, min_periods=11).sum()
    )
    df["trend_efficiency_20"] = safe_div(close_delta, path_length)
    df["vol_change_5d"] = g["vol_log"].diff(5)
    amihud_raw = safe_div(df["ret_1d"].abs(), df["amount"]) * 1e8
    df["amihud_20"] = amihud_raw.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(20, min_periods=11).mean()
    )
    df["main_flow_persistence_5"] = g["main_net_ratio"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["main_return_divergence"] = df["main_net_ratio"] - df["ret_1d"]
    turnover_mean = g["turnover_log"].transform(
        lambda s: s.rolling(20, min_periods=11).mean()
    )
    turnover_std = g["turnover_log"].transform(
        lambda s: s.rolling(20, min_periods=11).std(ddof=0)
    )
    df["liquidity_shock_20"] = safe_div(
        df["turnover_log"] - turnover_mean,
        turnover_std,
    )
    df["range_vol_ratio"] = safe_div(df["hl_range"], df["realized_vol_10"])

    output = df[["ts_code", "trade_date", "close", *requested]].copy()
    for field in requested:
        output[field] = output[field].replace([np.inf, -np.inf], np.nan)
    return output

FIELD_SPECS = {
    spec.name: spec for spec in [
        FieldSpec("ret_1d", "price", ("close",), "One-day close return"),
        FieldSpec("hl_range", "price", ("high", "low", "pre_close"), "High-low range scaled by prior close"),
        FieldSpec("oc_ret", "price", ("open", "close"), "Intraday open-to-close return"),
        FieldSpec("upper_shadow", "price", ("high", "open", "close", "pre_close"), "Upper candlestick shadow", enabled=False),
        FieldSpec("lower_shadow", "price", ("low", "open", "close", "pre_close"), "Lower candlestick shadow", enabled=False),
        FieldSpec("close_pos", "price", ("high", "low", "close"), "Close position within daily range"),
        FieldSpec("gap_ret", "price", ("open", "pre_close"), "Opening gap return"),
        FieldSpec("intraday_reversal", "price", ("open", "close", "high", "low"), "Negative intraday return scaled by range"),
        FieldSpec("amount_log", "liquidity", ("amount",), "Log traded amount"),
        FieldSpec("vol_log", "liquidity", ("vol",), "Log traded volume"),
        FieldSpec("turnover_log", "liquidity", ("turnover_rate",), "Log turnover rate"),
        FieldSpec("amount_per_vol", "liquidity", ("amount", "vol"), "Average traded price proxy", enabled=False),
        FieldSpec("price_volume_pressure", "interaction", ("ret_1d", "vol_log"), "Return-volume pressure"),
        FieldSpec("amplitude_turnover", "interaction", ("hl_range", "turnover_log"), "Amplitude-turnover interaction"),
        FieldSpec("volume_ratio_x_ret", "interaction", ("volume_ratio", "ret_1d"), "Volume-ratio return interaction"),
        FieldSpec("sm_net_ratio", "flow", ("buy_sm_amount", "sell_sm_amount"), "Small-order net ratio"),
        FieldSpec("md_net_ratio", "flow", ("buy_md_amount", "sell_md_amount"), "Medium-order net ratio"),
        FieldSpec("lg_net_ratio", "flow", ("buy_lg_amount", "sell_lg_amount"), "Large-order net ratio"),
        FieldSpec("main_net_ratio", "flow", ("buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount"), "Medium-plus-large net ratio"),
        FieldSpec("retail_pressure", "flow", ("sm_net_ratio",), "Negative small-order net pressure"),
        FieldSpec("big_vs_small_flow", "flow", ("lg_net_ratio", "sm_net_ratio"), "Large-versus-small flow divergence"),
        FieldSpec("flow_imbalance", "flow", ("net_mf_amount", "amount"), "Net money flow scaled by amount"),
        FieldSpec("large_order_intensity", "flow", ("buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount"), "Large-order share of observed flow"),
        FieldSpec("active_big_buy_pressure", "flow", ("buy_lg_amount", "sell_lg_amount"), "Large-order buy-side pressure"),
        FieldSpec("chip_width_90", "chip", ("cost_95pct", "cost_5pct", "cost_50pct"), "90-percent chip-cost width"),
        FieldSpec("chip_width_70", "chip", ("cost_85pct", "cost_15pct", "cost_50pct"), "70-percent chip-cost width"),
        FieldSpec("chip_cost_bias", "chip", ("close", "weight_avg"), "Price bias from weighted chip cost"),
        FieldSpec("chip_median_bias", "chip", ("close", "cost_50pct"), "Price bias from median chip cost"),
        FieldSpec("chip_upper_pressure", "chip", ("cost_95pct", "close"), "Upper chip-cost pressure"),
        FieldSpec("chip_lower_support", "chip", ("cost_5pct", "close"), "Lower chip-cost support"),
        FieldSpec("winner_rate_norm", "chip", ("winner_rate",), "Normalized profitable-holder rate"),
        FieldSpec("hist_price_position", "chip", ("close", "his_low", "his_high"), "Position in historical price range"),
        FieldSpec("winner_cost_divergence", "chip", ("winner_rate_norm", "chip_cost_bias"), "Winner-rate and cost-bias divergence"),
        FieldSpec("ret_5d", "price", ("close",), "Five-day close return"),
        FieldSpec("ret_20d", "price", ("close",), "Twenty-day close return"),
        FieldSpec("residual_ret_1d", "price", ("ret_1d",), "One-day return minus same-day universe mean"),
        FieldSpec("realized_vol_10", "risk", ("ret_1d",), "Ten-day realized volatility"),
        FieldSpec("realized_vol_20", "risk", ("ret_1d",), "Twenty-day realized volatility"),
        FieldSpec("downside_vol_20", "risk", ("ret_1d",), "Twenty-day downside volatility"),
        FieldSpec("trend_efficiency_20", "price", ("close",), "Twenty-day net displacement divided by path length"),
        FieldSpec("vol_change_5d", "liquidity", ("vol_log",), "Five-day log-volume change"),
        FieldSpec("amount_change_5d", "liquidity", ("amount_log",), "Five-day log-amount change"),
        FieldSpec("turnover_change_5d", "liquidity", ("turnover_log",), "Five-day log-turnover change"),
        FieldSpec("amihud_20", "liquidity", ("ret_1d", "amount"), "Twenty-day Amihud-style illiquidity"),
        FieldSpec("main_flow_persistence_5", "flow", ("main_net_ratio",), "Five-day persistent main-order flow"),
        FieldSpec("flow_agreement", "flow", ("main_net_ratio", "lg_net_ratio"), "Agreement between main and large-order flow"),
        FieldSpec("main_return_divergence", "interaction", ("main_net_ratio", "ret_1d"), "Main-flow and return divergence"),
        FieldSpec("chip_pressure_balance", "chip", ("chip_lower_support", "chip_upper_pressure"), "Lower support minus upper pressure"),
        FieldSpec("chip_concentration", "chip", ("chip_width_90",), "Inverse absolute chip width"),
        FieldSpec("winner_position_interaction", "chip", ("winner_rate_norm", "hist_price_position"), "Winner-rate and price-position interaction"),
        FieldSpec("liquidity_shock_20", "liquidity", ("turnover_log",), "Twenty-day turnover z-score"),
        FieldSpec("overnight_intraday_spread", "interaction", ("gap_ret", "oc_ret"), "Overnight gap minus intraday return"),
        FieldSpec("range_vol_ratio", "risk", ("hl_range", "realized_vol_10"), "Daily range relative to recent realized volatility"),
        FieldSpec("oi_change_1d", "futures_oi", ("open_interest",), "One-day open-interest change rate"),
        FieldSpec("oi_change_5d", "futures_oi", ("open_interest",), "Five-day open-interest change rate"),
        FieldSpec("oi_volume_ratio", "futures_oi", ("vol", "open_interest"), "Daily volume divided by open interest"),
        FieldSpec("oi_turnover_pressure", "futures_oi", ("oi_volume_ratio", "oi_change_1d"), "OI turnover ratio times OI change rate"),
        FieldSpec("oi_price_divergence", "futures_oi", ("ret_1d", "oi_change_1d"), "Price-return minus OI-change divergence"),
        FieldSpec("oi_return_agreement", "futures_oi", ("ret_1d", "oi_change_1d"), "Price-return and OI-change agreement"),
        FieldSpec("oi_change_acceleration_5", "futures_oi", ("oi_change_1d",), "One-day OI change minus its five-day mean"),
        FieldSpec("oi_relative_20", "futures_oi", ("open_interest",), "Twenty-day z-score of log open interest"),
        FieldSpec("size_log", "control", ("circ_mv",), "Log free-float market value", role="control", enabled=False),
        FieldSpec("free_turnover_gap", "control", ("turnover_rate_f", "turnover_rate"), "Free-float turnover gap", role="control", enabled=False),
        FieldSpec("liquidity_crowding", "control", ("turnover_log", "volume_ratio"), "Turnover-volume crowding", role="control", enabled=False),
    ]
}

DEFAULT_FEATURES = [
    name
    for name, spec in FIELD_SPECS.items()
    if spec.role == "search" and spec.enabled
]
