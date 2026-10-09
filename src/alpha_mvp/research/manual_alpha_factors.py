"""Twenty semantic crypto-alpha hypotheses, not an operator template grammar.

Each function is a fixed economic mechanism. YAML restricts only its time scale,
event strength and holding clock; every rolling statistic is causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MECHANISM_VERSION = "manual-crypto-alpha-20261003-v3-quarter-phase"


def _divide(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = a / np.where(np.abs(b) >= floor, b, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


class FeatureContext:
    def __init__(self, panels: dict[str, np.ndarray]):
        self.panels = panels
        self.cache: dict[tuple, np.ndarray] = {}
        shapes = {x.shape for x in panels.values()}
        if len(shapes) != 1:
            raise ValueError("manual factor fields must share one panel shape")

    def field(self, name: str) -> np.ndarray:
        if name not in self.panels:
            raise ValueError(f"undeclared manual-alpha field: {name}")
        return self.panels[name]

    def roll(self, name: str, window: int, operation: str = "mean") -> np.ndarray:
        key = (name, window, operation)
        if key not in self.cache:
            if window < 2 or window > 168:
                raise ValueError("manual-alpha rolling window outside complexity budget")
            data = pd.DataFrame(self.field(name))
            rolling = data.rolling(window, min_periods=max(2, window // 2))
            if operation == "mean":
                result = rolling.mean()
            elif operation == "std":
                result = rolling.std(ddof=0)
            elif operation == "sum":
                result = rolling.sum()
            else:
                raise ValueError(f"invalid manual-alpha rolling operation: {operation}")
            self.cache[key] = result.to_numpy(dtype=np.float32)
        return self.cache[key]

    def z(self, name: str, window: int) -> np.ndarray:
        return np.clip(_divide(self.field(name) - self.roll(name, window),
                               self.roll(name, window, "std")), -8, 8)

    def lag(self, name: str, hours: int) -> np.ndarray:
        key = (name, hours, "lag")
        if key not in self.cache:
            x = self.field(name)
            self.cache[key] = np.concatenate((np.full((hours, x.shape[1]), np.nan), x[:-hours]))
        return self.cache[key]


def _base(c: FeatureContext, w: int, k: float):
    f = c.field("taker_imbalance")
    r = c.field("ret_1h")
    v = c.field("realized_vol_24h")
    return f, r, v


def sell_absorption(c, w, k):
    f, r, v = _base(c, w, k)
    pressure = np.maximum(-c.z("taker_imbalance", w) - k, 0)
    resilience = np.maximum(_divide(r, v) + .25, 0)
    return pressure * resilience * (1 + np.maximum(c.field("micro5_path_efficiency"), 0))


def buy_failure(c, w, k):
    pressure = np.maximum(c.z("taker_imbalance", w) - k, 0)
    rejection = np.maximum(.5 - c.field("close_pos"), 0)
    return -pressure * rejection * (1 + np.maximum(c.field("micro5_jump_share"), 0))


def fast_flow_rejection(c, w, k):
    fast = c.field("micro5_taker_imbalance")
    slow = c.roll("taker_imbalance", w)
    price = c.field("micro_last_return")
    return np.where(np.abs(fast - slow) > k * .05,
                    (fast - slow) * np.maximum(-np.sign(fast) * price, 0), 0)


def latent_flow_acceleration(c, w, k):
    f = c.field("taker_imbalance")
    impulse = f - c.roll("taker_imbalance", w)
    price = c.field("ret_1h")
    return np.where(np.abs(impulse) > k * .04,
                    impulse * np.exp(-np.abs(_divide(price, c.field("realized_vol_24h")))), 0)


def fragmented_execution(c, w, k):
    fragmentation = c.z("trade_count_log", w) - c.z("quote_volume_log", w)
    flow = c.field("taker_imbalance")
    return np.maximum(fragmentation - k, 0) * flow * (1 - c.field("micro5_trade_hhi"))


def block_pressure_exhaustion(c, w, k):
    block = c.z("avg_trade_size_log", w)
    f = c.field("taker_imbalance")
    impact = np.maximum(c.field("micro5_jump_share"), 0)
    return -np.sign(f) * np.maximum(block - k, 0) * impact * np.abs(f)


def jump_flow_exhaustion(c, w, k):
    jump = c.field("micro5_jump_share")
    f = c.field("micro5_taker_imbalance")
    excess = jump - c.roll("micro5_jump_share", w)
    return -np.sign(c.field("micro_last_return")) * np.maximum(excess - k * .15, 0) * np.abs(f)


def liquidity_vacuum_repair(c, w, k):
    illiquid = c.z("amihud_log", w)
    weak_path = 1 - np.clip(c.field("micro5_path_efficiency"), 0, 1)
    return -np.sign(c.field("ret_1h")) * np.maximum(illiquid - k, 0) * weak_path


def downside_wick_reclaim(c, w, k):
    wick = c.field("close_pos")
    range_z = c.z("hl_range", w)
    return np.maximum(wick - .5, 0) * np.maximum(range_z - k, 0) * np.maximum(-c.field("oc_ret"), 0)


def upside_wick_rejection(c, w, k):
    wick = c.field("close_pos")
    range_z = c.z("hl_range", w)
    return -np.maximum(.5 - wick, 0) * np.maximum(range_z - k, 0) * np.maximum(c.field("oc_ret"), 0)


def vwap_flow_dislocation(c, w, k):
    bias = c.field("vwap_bias") - c.roll("vwap_bias", w)
    f = c.field("taker_imbalance")
    return -np.sign(bias) * np.maximum(np.abs(_divide(bias, c.field("realized_vol_24h"))) - k * .2, 0) * np.maximum(np.sign(bias) * f, 0)


def us_open_absorption(c, w, k):
    surprise = c.field("session_flow_surprise") - c.roll("session_flow_surprise", w)
    return c.field("us_day_flag") * -np.sign(surprise) * np.maximum(np.abs(surprise) - k, 0) * np.maximum(-np.sign(surprise) * c.field("ret_1h"), 0)


def us_evening_inventory(c, w, k):
    stretch = c.field("session_cumulative_return")
    excess = c.field("session_cumulative_volume_ratio") - c.roll("session_cumulative_volume_ratio", w)
    return -c.field("us_evening_flag") * stretch * np.maximum(excess - k * .1, 0)


def us_evening_continuation(c, w, k):
    # Explicit R2 inverse-hypothesis revision: high-volume evening inventory
    # may persist into the next session rather than mean-revert.
    return -us_evening_inventory(c, w, k)


def compression_release(c, w, k):
    compression = np.maximum(k - c.field("volatility_term_slope"), 0)
    trend = c.field("micro5_imbalance_trend") - c.roll("micro5_imbalance_trend", w)
    return compression * trend * np.maximum(c.field("micro5_path_efficiency"), 0)


def downside_vol_flow_confirmation(c, w, k):
    downside = c.field("downside_vol_24h")
    rv = c.field("realized_vol_24h")
    volume = c.field("micro5_volume_trend") - c.roll("micro5_volume_trend", w)
    return np.maximum(_divide(downside, rv) - k * .5, 0) * c.field("taker_imbalance_mean_4h") * np.maximum(volume, 0)


def btc_lead_catchup(c, w, k):
    btc = c.field("btc_ret_4h")
    own = c.field("ret_4h")
    beta = c.field("discovery_beta")
    delay = beta * btc - own
    volatility = c.roll("realized_vol_24h", w)
    return np.where(np.abs(delay) > k * volatility, delay, 0)


def shock_resilience(c, w, k):
    market = c.field("market_ret_1h")
    own = c.field("ret_1h")
    beta = c.field("discovery_beta")
    shock = np.maximum(_divide(np.abs(market), c.roll("market_vol_24h", w)) - k, 0)
    return (own - beta * market) * shock * np.sign(-market)


def basis_absorption(c, w, k):
    basis = c.z("derivative_basis", w)
    flow = c.field("taker_imbalance")
    # Crowded premium with opposing contemporaneous flow may mean absorption.
    return -np.sign(basis) * np.maximum(np.abs(basis) - k, 0) * np.maximum(-np.sign(basis) * flow, 0)


def trade_mark_repair(c, w, k):
    gap = c.z("derivative_trade_mark_gap", w)
    price = c.field("ret_1h")
    return -np.sign(gap) * np.maximum(np.abs(gap) - k, 0) * np.maximum(np.sign(gap) * price, 0)


def funding_crowd_disagreement(c, w, k):
    funding = c.field("derivative_funding_state")
    f = c.roll("taker_imbalance_mean_4h", w)
    # Funding sign is a crowd proxy, not a predictor by itself.
    return -np.sign(funding) * np.maximum(np.abs(funding) * 1e4 - k, 0) * np.maximum(np.sign(funding) * f, 0)


def funding_basis_carry(c, w, k):
    funding = c.field("derivative_funding_state")
    basis_z = c.z("derivative_basis", w)
    crowded_basis = np.maximum(np.sign(funding) * basis_z - k, 0)
    # Positive funding makes shorts receive; negative makes longs receive.
    return -funding * 1e4 * crowded_basis


def quarter_open_pressure(c, w, k):
    opening = c.field("quarter_open_imbalance")
    other = c.field("quarter_other_imbalance")
    share = c.field("quarter_open_volume_share")
    excess = share - c.roll("quarter_open_volume_share", w)
    muted_price = np.exp(-np.abs(_divide(c.field("ret_1h"), c.field("realized_vol_24h"))))
    return (opening - other) * np.maximum(excess - k * .02, 0) * muted_price


def quarter_open_absorption(c, w, k):
    opening = c.field("quarter_open_imbalance") - c.roll("quarter_open_imbalance", w)
    later = c.field("quarter_other_imbalance")
    price = c.field("ret_1h")
    opposed = np.maximum(-np.sign(opening) * later, 0)
    return -np.sign(opening) * np.maximum(np.abs(opening) - k * .05, 0) * opposed * np.maximum(np.sign(opening) * price, 0)


MECHANISMS = {name: obj for name, obj in list(globals().items())
              if name in {
                  "sell_absorption", "buy_failure", "fast_flow_rejection", "latent_flow_acceleration",
                  "fragmented_execution", "block_pressure_exhaustion", "jump_flow_exhaustion",
                  "liquidity_vacuum_repair", "downside_wick_reclaim", "upside_wick_rejection",
                  "vwap_flow_dislocation", "us_open_absorption", "us_evening_inventory",
                  "compression_release", "downside_vol_flow_confirmation", "btc_lead_catchup",
                  "shock_resilience", "basis_absorption", "trade_mark_repair", "funding_crowd_disagreement",
                  "us_evening_continuation", "funding_basis_carry",
                  "quarter_open_pressure", "quarter_open_absorption",
              }}


def evaluate_mechanism(name: str, context: FeatureContext, window: int,
                       threshold: float) -> np.ndarray:
    if name not in MECHANISMS:
        raise ValueError(f"unknown manual-alpha mechanism: {name}")
    result = np.asarray(MECHANISMS[name](context, window, threshold), dtype=np.float32)
    if result.shape != next(iter(context.panels.values())).shape:
        raise ValueError(f"manual-alpha shape mismatch: {name}")
    return np.where(np.isfinite(result), result, np.nan)
