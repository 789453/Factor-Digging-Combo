from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

EPS = 1e-9
OPERATOR_SEMANTICS_VERSION = "2026-09-26-v3"

def abs_(x): return np.abs(x)
def slog1p(x): return np.sign(x) * np.log1p(np.abs(x))
def inv(x): return 1.0 / np.where(np.abs(x) < EPS, np.nan, x)
def sign(x): return np.sign(x)
def log(x): return np.log(np.where(x > 0, x, np.nan))
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / np.where(np.abs(y) < EPS, np.nan, y)
def pow_(x, y):
    return np.power(np.where((x < 0) & (np.abs(y - np.round(y)) > 1e-12), np.nan, x), y)
def greater(x, y): return np.maximum(x, y)
def less(x, y): return np.minimum(x, y)
def gate_pos(state, signal): return np.where(state > 0.0, signal, 0.0)
def gate_neg(state, signal): return np.where(state < 0.0, signal, 0.0)

def rank_cs(x):
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(dtype=float)

def rolling_apply(x, window, func, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    return func(pd.DataFrame(x).rolling(window=window, min_periods=min_periods)).to_numpy(dtype=float)

def ref(x, window):
    return pd.DataFrame(x).shift(window).to_numpy(dtype=float)

from . import fastops

def ts_mean(x, w): return fastops.fast_rolling_mean(x, w)
def ts_sum(x, w): return fastops.fast_rolling_sum(x, w)
def ts_std(x, w): return fastops.fast_rolling_std(x, w)
def ts_var(x, w): return fastops.fast_rolling_std(x, w)**2
def ts_max(x, w): return fastops.fast_rolling_max(x, w)
def ts_min(x, w): return fastops.fast_rolling_min(x, w)
def ts_med(x, w):
    # Median is hard to optimize with Numba without complex logic, keep as is for now or use bottleneck
    return rolling_apply(x, w, lambda r: r.median())
def ts_delta(x, w): return x - ref(x, w)
def ts_pct_change(x, w):
    lag = ref(x, w)
    return div(x - lag, lag)
def ts_div(x, w): return div(x, ts_mean(x, w))
def ts_ir(x, w): return div(ts_mean(x, w), ts_std(x, w))
def ts_minmaxdiff(x, w): return ts_max(x, w) - ts_min(x, w)
def ts_maxdiff(x, w): return x - ts_max(x, w)
def ts_mindiff(x, w): return x - ts_min(x, w)
def ts_mad(x, w):
    df = pd.DataFrame(x)
    def mad(a):
        return np.nanmean(np.abs(a - np.nanmean(a)))
    return df.rolling(window=w, min_periods=w).apply(mad, raw=True).to_numpy(dtype=float)
def ts_skew(x, w):
    return pd.DataFrame(x).rolling(window=w, min_periods=w).skew().to_numpy(dtype=float)
def ts_kurt(x, w):
    return pd.DataFrame(x).rolling(window=w, min_periods=w).kurt().to_numpy(dtype=float)
def ts_rank(x, w): return fastops.fast_rolling_rank(x, w)
def ts_wma(x, w): return fastops.fast_rolling_wma(x, w)
def ts_ema(x, w):
    # EMA is already relatively fast with Pandas ewm, but can be optimized if needed
    return pd.DataFrame(x).ewm(span=w, min_periods=w, adjust=False).mean().to_numpy(dtype=float)
def ts_zscore(x, w): return fastops.fast_rolling_zscore(x, w)
def ts_cov(x, y, w):
    # Vectorized covariance: E[XY] - E[X]E[Y]
    mean_x = fastops.fast_rolling_mean(x, w)
    mean_y = fastops.fast_rolling_mean(y, w)
    mean_xy = fastops.fast_rolling_mean(x * y, w)
    return mean_xy - mean_x * mean_y

def ts_corr(x, y, w):
    return fastops.fast_rolling_corr(x, y, w)

UNARY = {"Abs": abs_, "SLog1p": slog1p, "Inv": inv, "Sign": sign, "Log": log, "Rank": rank_cs}
BINARY = {
    "Add": add, "Sub": sub, "Mul": mul, "Div": div, "Pow": pow_,
    "Greater": greater, "Less": less, "GatePos": gate_pos, "GateNeg": gate_neg,
}
ROLLING = {
    "Ref": ref, "TsMean": ts_mean, "TsSum": ts_sum, "TsStd": ts_std, "TsIr": ts_ir,
    "TsMinMaxDiff": ts_minmaxdiff, "TsMaxDiff": ts_maxdiff, "TsMinDiff": ts_mindiff,
    "TsVar": ts_var, "TsSkew": ts_skew, "TsKurt": ts_kurt, "TsMax": ts_max, "TsMin": ts_min,
    "TsMed": ts_med, "TsMad": ts_mad, "TsRank": ts_rank, "TsDelta": ts_delta,
    "TsDiv": ts_div, "TsPctChange": ts_pct_change, "TsWMA": ts_wma, "TsEMA": ts_ema,
    "TsZScore": ts_zscore,
}
PAIR_ROLLING = {"TsCov": ts_cov, "TsCorr": ts_corr}
ALL_OPERATORS = set(UNARY) | set(BINARY) | set(ROLLING) | set(PAIR_ROLLING)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    arity: int
    commutative: bool = False
    scale_invariant: bool = False
    risk: str = "normal"


OPERATOR_SPECS = {
    name: OperatorSpec(
        name=name,
        category=category,
        arity=arity,
        commutative=name in {"Add", "Mul", "TsCorr", "TsCov"},
        scale_invariant=name in {"Rank", "Sign", "TsRank", "TsCorr"},
        risk=(
            "restricted"
            if name in {"Log", "Inv", "Pow", "Div", "TsDiv", "TsPctChange", "TsCov"}
            else "normal"
        ),
    )
    for mapping, category, arity in [
        (UNARY, "unary", 1),
        (BINARY, "binary", 2),
        (ROLLING, "rolling", 2),
        (PAIR_ROLLING, "pair_rolling", 3),
    ]
    for name in mapping
}
