from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.config import EvaluationConfig, SplitConfig
from src.alpha_mvp.research.execution_diagnostics import (
    _positions_and_returns,
    execution_diagnostic_rows,
)
from src.alpha_mvp.research.evaluation import forward_returns, purge_cross_split_labels


def _fixture():
    n = 240
    t = np.arange(n, dtype=float)
    factor = np.column_stack([
        np.sin(t / 7), np.cos(t / 9), np.sin(t / 11 + 0.7)
    ])
    close = 100 * np.exp(np.cumsum(np.column_stack([
        0.001 + 0.002 * np.sin(t / 8),
        0.0005 + 0.002 * np.cos(t / 7),
        0.001 + 0.002 * np.sin(t / 6),
    ]), axis=0))
    dates = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y%m%d%H%M").tolist()
    split = SplitConfig(
        discovery_end=dates[119], validation_start=dates[120],
        validation_end=dates[199], holdout_start=dates[200],
    )
    config = EvaluationConfig(
        factor_mode="time_series", time_series_position_window=24,
        execution_horizons=(1, 4), execution_rebalance_bars=(1, 4),
        execution_no_trade_bands=(0.0, 0.25),
        execution_cost_bps=(0.0, 4.0),
    )
    return factor, close, dates, split, config


def test_execution_surface_ignores_holdout_changes():
    factor, close, dates, split, config = _fixture()
    first = pd.DataFrame(execution_diagnostic_rows(
        factor, close, dates, split, config, 1
    ))
    changed_factor = factor.copy()
    changed_close = close.copy()
    changed_factor[200:] = 1e9
    changed_close[200:] = 1e-3
    second = pd.DataFrame(execution_diagnostic_rows(
        changed_factor, changed_close, dates, split, config, 1
    ))
    pd.testing.assert_frame_equal(first, second)
    assert set(first.period) == {"discovery", "validation"}
    assert len(first) == 32


def test_rebalance_and_band_reduce_turnover_and_cost_is_linear():
    factor, close, dates, split, config = _fixture()
    _, fast = _positions_and_returns(factor, close, 1, config, 1, 1, 0.0)
    _, slow = _positions_and_returns(factor, close, 1, config, 4, 4, 0.25)
    assert slow.sum() < fast.sum()
    rows = pd.DataFrame(execution_diagnostic_rows(
        factor, close, dates, split, config, 1
    ))
    zero = rows[(rows.period == "validation") & (rows.cost_bps == 0)]
    assert np.allclose(zero.net_total_return, zero.gross_total_return)


def test_execution_grid_is_bounded_and_mode_is_explicit():
    _, _, _, _, config = _fixture()
    with pytest.raises(ValueError, match="96 scenarios"):
        replace(config, execution_horizons=(1, 2, 3, 4, 5, 6, 7, 8),
                execution_rebalance_bars=(1, 2, 3, 4),
                execution_no_trade_bands=(0.0, 0.1),
                execution_cost_bps=(0.0, 2.0)).validate()


def test_forward_labels_cannot_cross_research_split():
    close = np.arange(1.0, 13.0)[:, None]
    dates = [f"202401{day:02d}" for day in range(1, 13)]
    split = SplitConfig(
        discovery_end=dates[3], validation_start=dates[4],
        validation_end=dates[7], holdout_start=dates[8],
    )
    raw = forward_returns(close, horizon=2, entry_lag=1)
    safe = purge_cross_split_labels(raw, dates, split, 2, 1)
    assert np.isfinite(safe[0, 0])
    assert np.isnan(safe[1:4]).all()
    assert np.isfinite(safe[4, 0])
    assert np.isnan(safe[5:8]).all()
    assert np.array_equal(safe[8:], raw[8:], equal_nan=True)
