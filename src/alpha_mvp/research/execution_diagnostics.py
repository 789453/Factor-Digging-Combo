"""Bounded, holdout-free execution surface for time-series crypto factors.

All periods are decision-panel bars. This module does not claim native 5m/15m
execution when the decision panel is hourly.
"""

from __future__ import annotations

import numpy as np

from .. import fastops
from .config import EvaluationConfig, SplitConfig
from .crypto_performance import _period_strategy_metrics
from .evaluation import forward_returns
from .time_series_evaluation import asset_time_ic


def _positions_and_returns(
    factor: np.ndarray,
    close: np.ndarray,
    direction: int,
    config: EvaluationConfig,
    holding_bars: int,
    rebalance_bars: int,
    no_trade_band: float,
) -> tuple[np.ndarray, np.ndarray]:
    if factor.shape != close.shape or factor.ndim != 2:
        raise ValueError("factor and close must have the same two-dimensional shape")
    n_rows, n_assets = factor.shape
    z = fastops.fast_rolling_zscore(
        np.asarray(factor, dtype=float), config.time_series_position_window
    )
    target = np.clip(z * direction, -config.time_series_position_clip,
                     config.time_series_position_clip)
    target = target / (config.time_series_position_clip * n_assets)
    if holding_bars > 1:
        target = fastops.fast_rolling_mean(target, holding_bars)
    shift = config.entry_lag + 1
    positions = np.zeros_like(target)
    previous = np.zeros(n_assets, dtype=float)
    available = np.zeros(n_assets, dtype=bool)
    ever_active = False
    turnover = np.zeros(n_rows, dtype=float)
    for t in range(n_rows):
        if t >= shift and t % rebalance_bars == 0:
            desired = target[t - shift]
            valid = np.isfinite(desired)
            ever_active = ever_active or bool(valid.any())
            # A missing signal closes the position at the next scheduled rebalance.
            next_position = np.where(valid, desired, 0.0)
            change = np.abs(next_position - previous)
            trade = (~valid & available) | (
                valid & (~available | (change >= no_trade_band / n_assets))
            )
            updated = np.where(trade, next_position, previous)
            turnover[t] = np.abs(updated - previous).sum()
            previous = updated
            available = valid
        positions[t] = previous
        if not ever_active:
            positions[t] = np.nan
    returns = np.full_like(close, np.nan, dtype=float)
    returns[1:] = close[1:] / close[:-1] - 1.0
    gross = np.nansum(positions * returns, axis=1)
    gross[(~np.isfinite(returns).any(axis=1)) | (~np.isfinite(positions).any(axis=1))] = np.nan
    return gross, turnover


def execution_diagnostic_rows(
    factor: np.ndarray,
    close: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    direction: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate a predeclared grid using only discovery and validation outcomes."""
    if direction not in (-1, 1):
        raise ValueError("direction must be inferred from discovery and be +/-1")
    labels = np.asarray(dates).astype(str)
    masks = {
        "discovery": labels <= split.discovery_end,
        "validation": (labels >= split.validation_start) & (labels <= split.validation_end),
    }
    rows: list[dict[str, float | int | str]] = []
    for horizon in config.execution_horizons:
        forward = forward_returns(close, horizon, config.entry_lag)
        # A forward label must finish within its own period; no boundary crossing.
        decay = {}
        for period, mask in masks.items():
            valid = mask.copy()
            indices = np.flatnonzero(mask)
            tail = config.entry_lag + horizon
            if tail:
                valid[indices[-tail:]] = False
            values = asset_time_ic(factor, forward, valid,
                                   min_observations=min(24, max(3, int(valid.sum() // 2))))
            values = values[np.isfinite(values)] * direction
            decay[period] = float(values.mean()) if len(values) else np.nan
        for rebalance in config.execution_rebalance_bars:
            for band in config.execution_no_trade_bands:
                gross, turnover = _positions_and_returns(
                    factor, close, direction, config, horizon, rebalance, band
                )
                for cost in config.execution_cost_bps:
                    net = gross - turnover * (cost / 10000.0)
                    for period, mask in masks.items():
                        metrics = _period_strategy_metrics(net, mask, config.periods_per_year)
                        gross_metrics = _period_strategy_metrics(
                            gross, mask, config.periods_per_year
                        )
                        active = mask & np.isfinite(gross)
                        mean_turnover = float(turnover[active].mean()) if active.any() else np.nan
                        breakeven = (
                            gross_metrics["mean"] / mean_turnover * 10000.0
                            if np.isfinite(mean_turnover) and mean_turnover > 0 else np.nan
                        )
                        rows.append({
                            "period": period,
                            "holding_bars": horizon,
                            "rebalance_bars": rebalance,
                            "no_trade_band": band,
                            "cost_bps": cost,
                            "direction": direction,
                            "ic": decay[period],
                            "gross_total_return": gross_metrics["total_return"],
                            "net_total_return": metrics["total_return"],
                            "net_sharpe": metrics["sharpe"],
                            "mean_turnover": mean_turnover,
                            "breakeven_cost_bps": breakeven,
                            "n_bars": int(active.sum()),
                        })
    return rows
