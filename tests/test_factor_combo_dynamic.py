import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.factor_combo_dynamic import (
    DynamicSpec, _alpha_target, _execute_target, _matured_payoffs,
    _strategy_event_band,
    online_factor_weights,
)
from src.alpha_mvp.research.factor_combo_reporting import _equal_market_buyhold_log_returns


def _spec():
    return DynamicSpec.from_dict({
        "lookback_bars": 48, "update_bars": 12, "min_history_bars": 24,
        "max_factors_per_family": 1, "max_active_families": 2,
        "entry_z": .25, "exit_z": -.25, "anchor_weight": .35,
        "weight_speed": .25, "max_factor_weight": .4,
        "alpha_budget": .8, "beta_budget": .2, "risk_floor": .5,
        "target_half_life_bars": 12, "event_band": .25,
        "strategy_cost_bps": 4.0, "plot_stride_bars": 12,
    })


def test_online_weights_never_read_unmatured_or_later_labels():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(600, 4, 2)).astype(np.float32)
    y = (raw[:, :, 0] * .002 + rng.normal(0, .001, (600, 4))).astype(np.float32)
    dates = pd.date_range("2023-01-01", periods=600, freq="5min", tz="UTC").strftime("%Y%m%d%H%M").to_numpy()
    registry = pd.DataFrame({"expr_hash": ["a", "b"], "family": ["fast", "slow"],
                             "clock": ["native5", "hourly"], "role": ["alpha", "mixed"]})
    first, _, events = online_factor_weights(raw, y, dates, [400, 200],
                                              dates[299], registry, _spec())
    changed = y.copy()
    changed[300:] *= -100
    second, _, _ = online_factor_weights(raw, changed, dates, [400, 200],
                                          dates[299], registry, _spec())
    np.testing.assert_array_equal(first[:301], second[:301])
    assert np.any(first[350:] != second[350:])
    assert np.all(first >= 0)
    assert np.max(first) <= .400001
    assert set(events.event) <= {"enter", "exit"}


def test_hourly_utility_waits_for_lag_and_full_four_hour_label():
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="UTC").strftime("%Y%m%d%H%M").to_numpy()
    raw = np.ones((100, 2, 2), dtype=np.float32)
    y = np.zeros((100, 2), dtype=np.float32)
    y[12:60] = 1.0
    registry = pd.DataFrame({"clock": ["native5", "hourly"],
                             "lag_bars": [0, 12], "horizon_bars": [1, 48],
                             "stride_bars": [1, 12]})
    payoff, stride, overlap = _matured_payoffs(
        raw, y, dates, [100], registry, "source_horizon")
    assert np.isnan(payoff[59, 1])
    assert payoff[60, 1] == pytest.approx(1.0)
    assert payoff[1, 0] == 0.0
    np.testing.assert_array_equal(stride, [1, 12])
    np.testing.assert_array_equal(overlap, [1, 4])
    later = y.copy()
    later[59] = 10
    changed, _, _ = _matured_payoffs(raw, later, dates, [100], registry,
                                     "source_horizon")
    np.testing.assert_array_equal(payoff[:60], changed[:60])
    assert changed[60, 1] > payoff[60, 1]


def test_event_execution_charges_exact_change_and_next_bar_return():
    target = np.full((100, 2), .8)
    future = np.full((100, 2), .01)
    p, change, net = _execute_target(target, future, [100], _spec())
    assert np.allclose(net, p * future - np.abs(change) * 4 / 1e4)
    assert (np.abs(change) > 0).sum() < 100
    assert np.all(p <= .8)


def test_beta_sleeve_budget_can_trigger_trades():
    target = np.full((100, 2), .2)
    future = np.zeros((100, 2))
    position, change, _ = _execute_target(
        target, future, [100], _spec(),
        event_band=_strategy_event_band("beta_only", _spec()))
    assert np.count_nonzero(change) > 0
    assert np.max(position) <= .2
    assert _strategy_event_band("mix_risk_dynamic", _spec()) == .25


def test_equal_market_benchmark_is_initial_weight_buy_and_hold():
    panel = np.log(np.array([[1.1, .9], [1.1, .9], [1.2, .8]]))
    log_return = _equal_market_buyhold_log_returns(panel, [2, 1])
    np.testing.assert_allclose(np.exp(log_return[:2].sum()), 1.01)
    np.testing.assert_allclose(np.exp(log_return[2]), 1.0)


def test_missing_future_return_cannot_prevent_current_trade():
    target = np.full((100, 2), .8)
    future = np.full((100, 2), .01)
    _, baseline_change, _ = _execute_target(target, future, [100], _spec())
    first_trade = np.flatnonzero(baseline_change[:, 0])[0]
    future[first_trade] = np.nan
    p, change, net = _execute_target(target, future, [100], _spec())
    assert np.abs(change[first_trade]).sum() > 0
    np.testing.assert_allclose(change[first_trade], baseline_change[first_trade])
    np.testing.assert_allclose(net[first_trade], -np.abs(change[first_trade]) * 4 / 1e4)


def test_dynamic_config_rejects_unknown_key():
    values = dict(_spec().__dict__)
    values["silent_fallback"] = True
    with pytest.raises(ValueError, match="unknown"):
        DynamicSpec.from_dict(values)


def test_alpha_target_stays_market_neutral_after_position_cap():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=(200, 6)) + 3
    signal[100, 0] = 1000
    neutral, _ = _alpha_target(signal, np.ones_like(signal, dtype=bool))
    assert np.max(np.abs(neutral.mean(axis=1))) < 1e-6
    assert np.max(np.abs(neutral)) <= 1
