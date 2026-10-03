"""Explicit prediction labels for crypto research (never search inputs)."""

from __future__ import annotations

import numpy as np


TARGET_FORMULA_VERSION = "2026-09-28-micro5-forward-semivariance-v1"


def forward_micro5_variance(
    realized_vol: np.ndarray,
    upside_share: np.ndarray,
    bar_count: np.ndarray,
    horizon: int,
    entry_lag: int,
    mode: str,
    scope: str = "absolute",
) -> np.ndarray:
    """Sum completed 5m squared log returns over future hourly decision bars.

    Signal at t uses hours t+entry_lag+1 through t+entry_lag+horizon,
    matching the close-to-close return label's execution interval.
    """
    rv = np.asarray(realized_vol, dtype=float)
    share = np.asarray(upside_share, dtype=float)
    counts = np.asarray(bar_count, dtype=float)
    if rv.ndim != 2 or rv.shape != share.shape or rv.shape != counts.shape:
        raise ValueError("variance inputs must be equal two-dimensional panels")
    if horizon <= 0 or entry_lag < 0:
        raise ValueError("horizon must be positive and entry_lag nonnegative")
    if mode not in {"total_variance", "upside_semivariance", "downside_semivariance"}:
        raise ValueError("invalid variance target mode")
    if scope not in {"absolute", "market_relative"}:
        raise ValueError("invalid variance target scope")
    variance = np.square(rv)
    valid = (counts == 12) & np.isfinite(variance) & (variance >= 0)
    valid_share = np.isfinite(share) & (share >= 0) & (share <= 1)
    valid &= valid_share | (variance == 0)
    share = np.where(variance == 0, 0.0, share)
    if mode == "upside_semivariance":
        values = variance * share
    elif mode == "downside_semivariance":
        values = variance * (1.0 - share)
    else:
        values = variance
    values = np.where(valid, values, 0.0)
    sums = np.vstack([np.zeros((1, rv.shape[1])), np.cumsum(values, axis=0)])
    counts_cumulative = np.vstack([
        np.zeros((1, rv.shape[1]), dtype=int),
        np.cumsum(valid.astype(int), axis=0),
    ])
    output = np.full_like(rv, np.nan)
    n = len(rv) - entry_lag - horizon
    if n > 0:
        start = np.arange(n) + entry_lag + 1
        end = start + horizon
        complete = counts_cumulative[end] - counts_cumulative[start] == horizon
        output[:n] = np.where(complete, sums[end] - sums[start], np.nan)
    if scope == "market_relative":
        finite = np.isfinite(output)
        n_assets = finite.sum(axis=1, keepdims=True)
        common = np.divide(
            np.nansum(output, axis=1, keepdims=True), n_assets,
            out=np.full((len(output), 1), np.nan), where=n_assets >= 3,
        )
        output = output - common
    return output
