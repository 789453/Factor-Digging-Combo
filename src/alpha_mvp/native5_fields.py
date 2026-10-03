"""Causal fields for the native, completed-bar 5m crypto panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import FieldSpec, safe_div


NATIVE5_FIELD_FORMULA_VERSION = "2026-09-28-native5-fields-v1"


def _spec(name: str, dependencies: tuple[str, ...], description: str) -> FieldSpec:
    return FieldSpec(name, "crypto_native_5m", dependencies, description)


NATIVE5_FIELD_SPECS = {
    item.name: item for item in (
        _spec("ret_5m", ("close", "ts_code"), "Completed-bar log return"),
        _spec("ret_15m", ("close", "ts_code"), "Three completed 5m-bar log return"),
        _spec("ret_1h", ("close", "ts_code"), "Twelve completed 5m-bar log return"),
        _spec("range_5m", ("high", "low", "open"), "Current range divided by open"),
        _spec("vwap_bias", ("close", "vwap"), "Current close relative to bar VWAP"),
        _spec("taker_imbalance", ("taker_buy_ratio",), "Twice taker buy share minus one"),
        _spec("volume_shock_12", ("quote_volume", "ts_code"), "Causal 12-bar log volume z-score"),
        _spec("trade_shock_12", ("trade_count", "ts_code"), "Causal 12-bar log trades z-score"),
        _spec("amihud_shock_12", ("ret_5m", "quote_volume", "ts_code"), "Causal log impact z-score"),
        _spec("rv_12", ("ret_5m", "ts_code"), "Square root of trailing 12 squared returns"),
        _spec("upside_share_12", ("ret_5m", "ts_code"), "Positive squared-return share of trailing 12 bars"),
        _spec("jump_share_12", ("ret_5m", "ts_code"), "Max absolute return over trailing absolute path"),
        _spec("path_efficiency_12", ("close", "ret_5m", "ts_code"), "Trailing net displacement over absolute path"),
        _spec("flow_trend_12", ("taker_imbalance", "ts_code"), "Current flow minus trailing flow mean"),
        _spec("flow_price_agreement", ("taker_imbalance", "ret_5m"), "Flow direction times return"),
        _spec("ret_15m_completed", ("date", "open", "close"), "Most recently completed 15m-bar return"),
        _spec("flow_15m_completed", ("date", "taker_buy_ratio"), "Most recently completed 15m-bar flow"),
        _spec("us_day_flag", ("trade_date",), "DST-aware New York 08:00–16:00 flag"),
        _spec("us_evening_flag", ("trade_date",), "DST-aware New York 16:00–24:00 flag"),
        _spec("us_overnight_flag", ("trade_date",), "DST-aware New York 00:00–08:00 flag"),
    )
}


def _rolling_z(grouped, field: str, window: int) -> pd.Series:
    mean = grouped[field].transform(lambda s: s.rolling(window, min_periods=window).mean())
    std = grouped[field].transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    return safe_div(grouped.obj[field] - mean, std)


def add_crypto_native_5m_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["ts_code", "trade_date"]).copy()
    grouped = df.groupby("ts_code", sort=False)
    positive_close = df["close"].where(df["close"] > 0)
    log_close = np.log(positive_close)
    for n, name in ((1, "ret_5m"), (3, "ret_15m"), (12, "ret_1h")):
        df[name] = log_close - log_close.groupby(df["ts_code"]).shift(n)
    df["range_5m"] = safe_div(df["high"] - df["low"], df["open"])
    df["vwap_bias"] = safe_div(df["close"] - df["vwap"], df["vwap"])
    df["taker_imbalance"] = 2.0 * df["taker_buy_ratio"].where(
        df["taker_buy_ratio"].between(0, 1)
    ) - 1.0
    df["quote_volume_log"] = np.log1p(df["quote_volume"].where(df["quote_volume"] >= 0))
    df["trade_count_log"] = np.log1p(df["trade_count"].where(df["trade_count"] >= 0))
    df["volume_shock_12"] = _rolling_z(grouped, "quote_volume_log", 12)
    df["trade_shock_12"] = _rolling_z(grouped, "trade_count_log", 12)
    df["amihud_log"] = np.log1p(
        safe_div(df["ret_5m"].abs(), df["quote_volume"]) * 1e9
    )
    df["amihud_shock_12"] = _rolling_z(grouped, "amihud_log", 12)
    squared = df["ret_5m"].pow(2)
    rolling_sq = squared.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(12, min_periods=12).sum()
    )
    df["rv_12"] = np.sqrt(rolling_sq)
    positive_sq = squared.where(df["ret_5m"] > 0, 0.0)
    up = positive_sq.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(12, min_periods=12).sum()
    )
    df["upside_share_12"] = safe_div(up, rolling_sq)
    abs_return = df["ret_5m"].abs()
    path = abs_return.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(12, min_periods=12).sum()
    )
    jump = abs_return.groupby(df["ts_code"]).transform(
        lambda s: s.rolling(12, min_periods=12).max()
    )
    df["jump_share_12"] = safe_div(jump, path)
    df["path_efficiency_12"] = safe_div(df["ret_1h"].abs(), path)
    flow_mean = grouped["taker_imbalance"].transform(
        lambda s: s.rolling(12, min_periods=12).mean()
    )
    df["flow_trend_12"] = df["taker_imbalance"] - flow_mean
    df["flow_price_agreement"] = df["taker_imbalance"] * df["ret_5m"]
    ny_hour = pd.to_datetime(df["trade_date"], format="%Y%m%d%H%M", utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.hour
    df["us_day_flag"] = ((ny_hour >= 8) & (ny_hour < 16)).astype(float)
    df["us_evening_flag"] = (ny_hour >= 16).astype(float)
    df["us_overnight_flag"] = (ny_hour < 8).astype(float)
    numeric = df.select_dtypes(include=["number"]).columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df
