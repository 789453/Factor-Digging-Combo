from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import FieldSpec, safe_div


CRYPTO_FIELD_FORMULA_VERSION = "2026-09-26-crypto-multiscale-session-v2"


def _rolling_mean(grouped, column: str, window: int, minimum: int):
    return grouped[column].transform(
        lambda series: series.rolling(window, min_periods=minimum).mean()
    )


def _rolling_std(grouped, column: str, window: int, minimum: int):
    return grouped[column].transform(
        lambda series: series.rolling(window, min_periods=minimum).std(ddof=0)
    )


def _rolling_z(grouped, column: str, window: int, minimum: int):
    mean = _rolling_mean(grouped, column, window, minimum)
    std = _rolling_std(grouped, column, window, minimum)
    return safe_div(grouped.obj[column] - mean, std)


def add_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal, scale-free hourly crypto and multiscale/session fields."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)

    df["ret_1h"] = g["close"].pct_change()
    df["ret_4h"] = g["close"].pct_change(4)
    df["ret_24h"] = g["close"].pct_change(24)
    df["ret_168h"] = g["close"].pct_change(168)
    df["residual_ret_1h"] = df["ret_1h"] - df.groupby(
        "trade_date", sort=False
    )["ret_1h"].transform("mean")
    df["residual_ret_24h"] = df["ret_24h"] - df.groupby(
        "trade_date", sort=False
    )["ret_24h"].transform("mean")
    df["hl_range"] = safe_div(df["high"] - df["low"], df["open"])
    df["oc_ret"] = safe_div(df["close"] - df["open"], df["open"])
    df["close_pos"] = safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["vwap_bias"] = safe_div(df["close"] - df["vwap"], df["vwap"])
    df["range_body_ratio"] = safe_div(df["oc_ret"].abs(), df["hl_range"])

    for window, minimum in ((4, 3), (24, 12), (168, 84)):
        df[f"realized_vol_{window}h"] = g["ret_1h"].transform(
            lambda series, w=window, m=minimum: series.rolling(
                w, min_periods=m
            ).std(ddof=0)
        )
    downside_sq = np.minimum(df["ret_1h"].fillna(0.0), 0.0) ** 2
    df["downside_vol_24h"] = downside_sq.groupby(df["ts_code"]).transform(
        lambda series: np.sqrt(series.rolling(24, min_periods=12).mean())
    )
    upside_sq = np.maximum(df["ret_1h"].fillna(0.0), 0.0) ** 2
    df["upside_vol_24h"] = upside_sq.groupby(df["ts_code"]).transform(
        lambda series: np.sqrt(series.rolling(24, min_periods=12).mean())
    )
    df["volatility_direction_24h"] = safe_div(
        df["upside_vol_24h"] - df["downside_vol_24h"],
        df["upside_vol_24h"] + df["downside_vol_24h"],
    )
    df["return_skew_24h"] = g["ret_1h"].transform(
        lambda series: series.rolling(24, min_periods=12).skew()
    )
    path = g["close"].pct_change().abs()
    for window, minimum in ((24, 12), (168, 84)):
        displacement = g["close"].pct_change(window).abs()
        path_sum = path.groupby(df["ts_code"]).transform(
            lambda series, w=window, m=minimum: series.rolling(
                w, min_periods=m
            ).sum()
        )
        df[f"trend_efficiency_{window}h"] = safe_div(displacement, path_sum)
    df["short_long_momentum_gap"] = df["ret_4h"] - df["ret_24h"]
    df["volatility_term_slope"] = safe_div(
        df["realized_vol_4h"], df["realized_vol_24h"]
    ) - 1.0

    df["quote_volume_log"] = np.log1p(df["quote_volume"].clip(lower=0))
    df["trade_count_log"] = np.log1p(df["trade_count"].clip(lower=0))
    df["avg_trade_size_log"] = np.log1p(
        safe_div(df["quote_volume"], df["trade_count"]).clip(lower=0)
    )
    df["volume_ratio_24h"] = safe_div(
        df["quote_volume"], _rolling_mean(g, "quote_volume", 24, 12)
    )
    df["volume_ratio_168h"] = safe_div(
        df["quote_volume"], _rolling_mean(g, "quote_volume", 168, 84)
    )
    df["volume_shock_24h"] = _rolling_z(g, "quote_volume_log", 24, 12)
    df["trade_count_shock_24h"] = _rolling_z(g, "trade_count_log", 24, 12)
    df["avg_trade_size_shock_24h"] = _rolling_z(
        g, "avg_trade_size_log", 24, 12
    )
    df["amihud_raw"] = safe_div(df["ret_1h"].abs(), df["quote_volume"])
    df["amihud_log"] = np.log1p(df["amihud_raw"] * 1e9)
    df["amihud_shock_24h"] = _rolling_z(g, "amihud_log", 24, 12)
    df["liquidity_efficiency"] = safe_div(
        df["ret_1h"].abs(), np.log1p(df["trade_count"].clip(lower=0))
    )

    df["taker_imbalance"] = 2.0 * df["taker_buy_ratio"] - 1.0
    df["taker_imbalance_change_4h"] = g["taker_imbalance"].diff(4)
    df["taker_imbalance_mean_4h"] = _rolling_mean(
        g, "taker_imbalance", 4, 3
    )
    df["taker_imbalance_mean_24h"] = _rolling_mean(
        g, "taker_imbalance", 24, 12
    )
    df["taker_price_agreement"] = df["taker_imbalance"] * df["ret_1h"]
    df["taker_return_divergence"] = df["taker_imbalance"] - df["ret_1h"]
    df["taker_micro_divergence"] = (
        df["taker_imbalance"] - df["micro_taker_imbalance"]
    )

    # The raw micro fields are already dimensionless within-hour aggregates.
    df["micro_last_return"] = pd.to_numeric(
        df["micro_last_return"], errors="coerce"
    )
    df["micro_first_return"] = pd.to_numeric(
        df["micro_first_return"], errors="coerce"
    )
    df["micro_volatility_ratio"] = safe_div(
        df["micro_realized_vol"], df["realized_vol_4h"]
    )
    df["micro_flow_price_agreement"] = (
        df["micro_taker_imbalance"] * df["micro_last_return"]
    )
    for column in (
        "micro_upside_share", "micro_jump_share", "micro_volume_hhi",
        "micro_trade_hhi",
    ):
        if column not in df:
            df[column] = np.nan

    # Optional 5m aggregates. Missing columns remain explicit NaN rather than
    # silently falling back to 15m inputs.
    for column in (
        "micro5_realized_vol", "micro5_upside_share", "micro5_jump_share",
        "micro5_path_efficiency", "micro5_taker_imbalance",
        "micro5_volume_trend", "micro5_imbalance_trend",
        "micro5_volume_hhi", "micro5_trade_hhi",
    ):
        if column not in df:
            df[column] = np.nan
    df["micro5_volatility_ratio"] = safe_div(
        df["micro5_realized_vol"], df["micro_realized_vol"]
    )
    df["micro5_15_efficiency_gap"] = (
        df["micro5_path_efficiency"] - df["micro_path_efficiency"]
    )
    df["micro5_15_flow_gap"] = (
        df["micro5_taker_imbalance"] - df["micro_taker_imbalance"]
    )
    df["micro5_15_jump_gap"] = df["micro5_jump_share"] - df["micro_jump_share"]

    # New York local time is DST-aware. The three mutually exclusive session
    # flags and cyclical encodings are deterministic ex-ante calendar fields.
    utc_time = pd.to_datetime(df["trade_date"], format="%Y%m%d%H%M", utc=True)
    ny_time = utc_time.dt.tz_convert("America/New_York")
    ny_hour = ny_time.dt.hour
    df["us_day_flag"] = ((ny_hour >= 8) & (ny_hour < 16)).astype(float)
    df["us_evening_flag"] = ((ny_hour >= 16) & (ny_hour < 24)).astype(float)
    df["us_overnight_flag"] = (ny_hour < 8).astype(float)
    df["ny_hour_sin"] = np.sin(2.0 * np.pi * ny_hour / 24.0)
    df["ny_hour_cos"] = np.cos(2.0 * np.pi * ny_hour / 24.0)
    df["weekday_sin"] = np.sin(2.0 * np.pi * ny_time.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2.0 * np.pi * ny_time.dt.dayofweek / 7.0)
    session_code = np.select(
        [ny_hour < 8, ny_hour < 16], ["overnight", "day"], default="evening"
    )
    session_key = pd.Series(
        df["ts_code"].astype(str) + "|" + ny_time.dt.strftime("%Y%m%d") + "|" + session_code,
        index=df.index,
    )
    session_group = df.groupby(session_key, sort=False)
    session_open = session_group["open"].transform("first")
    df["session_cumulative_return"] = safe_div(df["close"], session_open) - 1.0
    df["session_progress"] = session_group.cumcount().astype(float) / 7.0
    session_volume = session_group["quote_volume"].cumsum()
    expected_volume = _rolling_mean(g, "quote_volume", 24, 12) * (
        session_group.cumcount().astype(float) + 1.0
    )
    df["session_cumulative_volume_ratio"] = safe_div(session_volume, expected_volume)

    # Same-local-hour baselines remove the strong 24h seasonality using only
    # prior occurrences of that local hour (roughly the previous 10-60 days).
    same_hour = df.groupby([df["ts_code"], ny_hour], sort=False)
    for source, target in (
        ("quote_volume_log", "session_volume_surprise"),
        ("hl_range", "session_volatility_surprise"),
        ("amihud_log", "session_illiquidity_surprise"),
        ("taker_imbalance", "session_flow_surprise"),
    ):
        mean = same_hour[source].transform(
            lambda series: series.shift(1).rolling(60, min_periods=10).mean()
        )
        std = same_hour[source].transform(
            lambda series: series.shift(1).rolling(60, min_periods=10).std(ddof=0)
        )
        df[target] = safe_div(df[source] - mean, std)

    for name in CRYPTO_DEFAULT_FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return df


CRYPTO_FIELD_SPECS = {
    spec.name: spec for spec in [
        FieldSpec("ret_1h", "crypto_price", ("close",), "One-hour close return"),
        FieldSpec("ret_4h", "crypto_price", ("close",), "Four-hour close return"),
        FieldSpec("ret_24h", "crypto_price", ("close",), "Twenty-four-hour close return"),
        FieldSpec("ret_168h", "crypto_price", ("close",), "Seven-day close return"),
        FieldSpec("residual_ret_1h", "crypto_price", ("ret_1h",), "One-hour return minus universe mean"),
        FieldSpec("residual_ret_24h", "crypto_price", ("ret_24h",), "Twenty-four-hour return minus universe mean"),
        FieldSpec("hl_range", "crypto_price", ("high", "low", "open"), "Hourly range scaled by open"),
        FieldSpec("oc_ret", "crypto_price", ("open", "close"), "Hourly open-to-close return"),
        FieldSpec("close_pos", "crypto_price", ("high", "low", "close"), "Close position within hourly range"),
        FieldSpec("vwap_bias", "crypto_price", ("close", "vwap"), "Close deviation from hourly VWAP"),
        FieldSpec("range_body_ratio", "crypto_price", ("oc_ret", "hl_range"), "Absolute candle body divided by range"),
        FieldSpec("realized_vol_4h", "crypto_risk", ("ret_1h",), "Four-hour realized volatility"),
        FieldSpec("realized_vol_24h", "crypto_risk", ("ret_1h",), "Twenty-four-hour realized volatility"),
        FieldSpec("realized_vol_168h", "crypto_risk", ("ret_1h",), "Seven-day realized volatility"),
        FieldSpec("downside_vol_24h", "crypto_risk", ("ret_1h",), "Twenty-four-hour downside volatility"),
        FieldSpec("upside_vol_24h", "crypto_risk", ("ret_1h",), "Twenty-four-hour upside volatility"),
        FieldSpec("volatility_direction_24h", "crypto_risk_direction", ("upside_vol_24h", "downside_vol_24h"), "Normalized upside-minus-downside semivolatility"),
        FieldSpec("return_skew_24h", "crypto_risk_direction", ("ret_1h",), "Rolling 24h return skewness"),
        FieldSpec("trend_efficiency_24h", "crypto_trend", ("close",), "24h displacement divided by path length"),
        FieldSpec("trend_efficiency_168h", "crypto_trend", ("close",), "7d displacement divided by path length"),
        FieldSpec("short_long_momentum_gap", "crypto_trend", ("ret_4h", "ret_24h"), "Four-hour minus twenty-four-hour momentum"),
        FieldSpec("volatility_term_slope", "crypto_risk", ("realized_vol_4h", "realized_vol_24h"), "Short/long volatility slope"),
        FieldSpec("volume_ratio_24h", "crypto_liquidity", ("quote_volume",), "Hourly quote volume relative to 24h mean"),
        FieldSpec("volume_ratio_168h", "crypto_liquidity", ("quote_volume",), "Hourly quote volume relative to 7d mean"),
        FieldSpec("volume_shock_24h", "crypto_liquidity", ("quote_volume",), "24h z-score of log quote volume"),
        FieldSpec("trade_count_shock_24h", "crypto_liquidity", ("trade_count",), "24h z-score of log trade count"),
        FieldSpec("avg_trade_size_shock_24h", "crypto_liquidity", ("quote_volume", "trade_count"), "24h z-score of average trade size"),
        FieldSpec("amihud_shock_24h", "crypto_liquidity", ("ret_1h", "quote_volume"), "24h normalized price impact"),
        FieldSpec("liquidity_efficiency", "crypto_efficiency", ("ret_1h", "trade_count"), "Absolute return per log trade count"),
        FieldSpec("taker_imbalance", "crypto_flow", ("taker_buy_ratio",), "Taker buy minus sell share"),
        FieldSpec("taker_imbalance_change_4h", "crypto_flow", ("taker_imbalance",), "Four-hour change in taker imbalance"),
        FieldSpec("taker_imbalance_mean_4h", "crypto_flow", ("taker_imbalance",), "Four-hour mean taker imbalance"),
        FieldSpec("taker_imbalance_mean_24h", "crypto_flow", ("taker_imbalance",), "Twenty-four-hour mean taker imbalance"),
        FieldSpec("taker_price_agreement", "crypto_flow", ("taker_imbalance", "ret_1h"), "Taker-flow and return agreement"),
        FieldSpec("taker_return_divergence", "crypto_flow", ("taker_imbalance", "ret_1h"), "Taker-flow minus return divergence"),
        FieldSpec("taker_micro_divergence", "crypto_flow", ("taker_imbalance", "micro_taker_imbalance"), "Hourly versus 15m flow divergence"),
        FieldSpec("micro_first_return", "crypto_micro", ("15m_close",), "First 15m return in the hour"),
        FieldSpec("micro_last_return", "crypto_micro", ("15m_close",), "Last 15m return in the hour"),
        FieldSpec("micro_return_reversal", "crypto_micro", ("micro_first_return", "micro_last_return"), "Last-minus-first 15m return"),
        FieldSpec("micro_realized_vol", "crypto_micro", ("15m_return",), "Within-hour realized volatility"),
        FieldSpec("micro_volatility_ratio", "crypto_micro", ("micro_realized_vol", "realized_vol_4h"), "Within-hour versus four-hour volatility"),
        FieldSpec("micro_path_efficiency", "crypto_micro", ("15m_return",), "Hourly displacement divided by 15m path"),
        FieldSpec("micro_volume_concentration", "crypto_micro", ("15m_volume",), "Largest 15m volume share"),
        FieldSpec("micro_volume_trend", "crypto_micro", ("15m_volume",), "Second-half versus first-half volume balance"),
        FieldSpec("micro_taker_imbalance", "crypto_micro", ("15m_taker_buy_ratio",), "Within-hour signed taker volume ratio"),
        FieldSpec("micro_imbalance_trend", "crypto_micro", ("15m_taker_buy_ratio",), "Second-half versus first-half flow balance"),
        FieldSpec("micro_flow_price_agreement", "crypto_micro", ("micro_taker_imbalance", "micro_last_return"), "15m flow/price agreement"),
        FieldSpec("micro_trade_concentration", "crypto_micro", ("15m_trade_count",), "Largest 15m trade-count share"),
        FieldSpec("micro_range_expansion", "crypto_micro", ("15m_range",), "Last 15m range divided by hourly mean"),
        FieldSpec("micro_vwap_dispersion", "crypto_micro", ("15m_vwap", "close"), "Within-hour VWAP dispersion scaled by price"),
        FieldSpec("micro_upside_share", "crypto_micro", ("15m_return",), "Share of 15m realized variance from positive bars"),
        FieldSpec("micro_jump_share", "crypto_micro", ("15m_return",), "Largest 15m absolute return divided by path"),
        FieldSpec("micro_volume_hhi", "crypto_micro", ("15m_volume",), "15m volume Herfindahl concentration"),
        FieldSpec("micro_trade_hhi", "crypto_micro", ("15m_trade_count",), "15m trade-count Herfindahl concentration"),
        FieldSpec("micro5_realized_vol", "crypto_micro5", ("5m_return",), "Within-hour 5m realized volatility"),
        FieldSpec("micro5_upside_share", "crypto_micro5", ("5m_return",), "Share of 5m variance from positive bars"),
        FieldSpec("micro5_jump_share", "crypto_micro5", ("5m_return",), "Largest 5m absolute return divided by path"),
        FieldSpec("micro5_path_efficiency", "crypto_micro5", ("5m_return",), "Hourly displacement divided by 5m path"),
        FieldSpec("micro5_taker_imbalance", "crypto_micro5", ("5m_taker_buy_ratio",), "Within-hour 5m signed taker volume ratio"),
        FieldSpec("micro5_volume_trend", "crypto_micro5", ("5m_volume",), "Late-minus-early 5m volume balance"),
        FieldSpec("micro5_imbalance_trend", "crypto_micro5", ("5m_taker_buy_ratio",), "Late-minus-early 5m signed-flow balance"),
        FieldSpec("micro5_volume_hhi", "crypto_micro5", ("5m_volume",), "5m volume Herfindahl concentration"),
        FieldSpec("micro5_trade_hhi", "crypto_micro5", ("5m_trade_count",), "5m trade-count Herfindahl concentration"),
        FieldSpec("micro5_volatility_ratio", "crypto_multiscale", ("micro5_realized_vol", "micro_realized_vol"), "5m-to-15m realized volatility ratio"),
        FieldSpec("micro5_15_efficiency_gap", "crypto_multiscale", ("micro5_path_efficiency", "micro_path_efficiency"), "5m minus 15m path efficiency"),
        FieldSpec("micro5_15_flow_gap", "crypto_multiscale", ("micro5_taker_imbalance", "micro_taker_imbalance"), "5m minus 15m taker imbalance"),
        FieldSpec("micro5_15_jump_gap", "crypto_multiscale", ("micro5_jump_share", "micro_jump_share"), "5m minus 15m jump share"),
        FieldSpec("us_day_flag", "crypto_session", ("trade_date",), "NY local 08:00-15:59 indicator, DST-aware"),
        FieldSpec("us_evening_flag", "crypto_session", ("trade_date",), "NY local 16:00-23:59 indicator, DST-aware"),
        FieldSpec("us_overnight_flag", "crypto_session", ("trade_date",), "NY local 00:00-07:59 indicator, DST-aware"),
        FieldSpec("ny_hour_sin", "crypto_session", ("trade_date",), "Cyclical sine of NY local hour"),
        FieldSpec("ny_hour_cos", "crypto_session", ("trade_date",), "Cyclical cosine of NY local hour"),
        FieldSpec("weekday_sin", "crypto_session", ("trade_date",), "Cyclical sine of NY weekday"),
        FieldSpec("weekday_cos", "crypto_session", ("trade_date",), "Cyclical cosine of NY weekday"),
        FieldSpec("session_cumulative_return", "crypto_session_state", ("close", "open", "trade_date"), "Return since current 8h NY session open"),
        FieldSpec("session_progress", "crypto_session_state", ("trade_date",), "Progress through current 8h NY session"),
        FieldSpec("session_cumulative_volume_ratio", "crypto_session_state", ("quote_volume", "trade_date"), "Session cumulative volume versus causal 24h expectation"),
        FieldSpec("session_volume_surprise", "crypto_session_state", ("quote_volume", "trade_date"), "Log volume z-score versus prior same-NY-hour observations"),
        FieldSpec("session_volatility_surprise", "crypto_session_state", ("hl_range", "trade_date"), "Range z-score versus prior same-NY-hour observations"),
        FieldSpec("session_illiquidity_surprise", "crypto_session_state", ("amihud_log", "trade_date"), "Illiquidity z-score versus prior same-NY-hour observations"),
        FieldSpec("session_flow_surprise", "crypto_session_state", ("taker_imbalance", "trade_date"), "Taker-flow z-score versus prior same-NY-hour observations"),
    ]
}


CRYPTO_DEFAULT_FEATURES = list(CRYPTO_FIELD_SPECS)
