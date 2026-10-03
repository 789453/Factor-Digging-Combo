import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.factor_combo_signal_engine import (
    ProfileSpec, SignalEngineSpec, _net_from_positions, _simulate_profile,
    run_signal_engine,
)


def _profile(name="fast", **override):
    raw = dict(name=name, native5_share=.7, hourly_share=.2, beta_share=.1,
               allocation=.4, half_life_bars=4, min_hold_bars=24,
               max_hold_bars=72, cooldown_bars=6, rebalance_bars=12,
               target_trades_per_asset_week=4., entry_candidates=[.3, .6, 1.],
               exit_ratio=.45, size_band=.1)
    raw.update(override)
    return ProfileSpec.from_dict(raw)


def test_profile_configuration_rejects_invalid_or_unknown_values():
    with pytest.raises(ValueError):
        _profile(extra=1)
    with pytest.raises(ValueError):
        _profile(native5_share=.8)
    with pytest.raises(ValueError):
        _profile(min_hold_bars=100)


def test_profile_state_machine_holds_cools_and_closes_at_segment_end():
    p = _profile(min_hold_bars=24, max_hold_bars=36, cooldown_bars=6,
                 entry_candidates=[.5])
    strength = np.ones((95, 1), dtype=np.float32)
    position, events, holds, count = _simulate_profile(
        strength, np.ones(95), [95], p, .5)
    assert count == sum(not h["right_censored"] for h in holds) >= 2
    assert all(24 <= h["end_t"] - h["start_t"] <= 36
               for h in holds if not h["right_censored"])
    assert holds[-1]["exit_reason"] == "segment_end"
    opens = [e["t"] for e in events if e["event"].startswith("open")]
    closes = [e["t"] for e in events if e["event"] == "close"]
    assert all(b - a >= 6 for a, b in zip(closes, opens[1:]))
    assert position[-1, 0] == 0


def test_exact_fee_uses_netted_position_change_and_segment_reset():
    position = np.array([[.2], [.4], [0], [-.3]], dtype=np.float32)
    raw = np.full_like(position, .01)
    change, net = _net_from_positions(position, raw, [2, 2], 4.)
    np.testing.assert_allclose(change[:, 0], [.2, .2, 0, -.3])
    np.testing.assert_allclose(net, position * raw - abs(change) * .0004)


def test_signal_calibration_and_positions_do_not_read_returns_or_later_signals():
    n, assets = 2600, 2
    t = np.arange(n)
    native = np.column_stack([np.sin(t / 27), np.sin(t / 31)]).astype(np.float32)
    hourly = np.column_stack([np.sin(t / 100), -np.sin(t / 110)]).astype(np.float32)
    beta = np.sin(t / 230).astype(np.float32)
    risk = np.ones(n)
    raw = np.zeros((n, assets), dtype=np.float32)
    dates = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC").strftime("%Y%m%d%H%M").to_numpy()
    train = np.zeros((n, assets), dtype=bool)
    train[:1500] = True
    spec = SignalEngineSpec(scale_lookback_bars=96, scale_floor_fraction=.5,
                            max_abs_position=1.5, leverage_diagnostic=1.5,
                            profiles=(_profile(), _profile("medium", allocation=.3)))
    first = run_signal_engine(native, hourly, beta, risk, raw, dates, train,
                              [1800, 800], spec, 4.)
    changed_raw = np.full_like(raw, 100.)
    changed_native = native.copy()
    changed_native[2000:] *= -1
    second = run_signal_engine(changed_native, hourly, beta, risk,
                               changed_raw, dates, train, [1800, 800], spec, 4.)
    pd.testing.assert_frame_equal(first["calibration"], second["calibration"])
    np.testing.assert_array_equal(first["positions"]["multiscale_trigger"][:2000],
                                  second["positions"]["multiscale_trigger"][:2000])
    assert np.any(first["positions"]["multiscale_trigger"][2100:] !=
                  second["positions"]["multiscale_trigger"][2100:])
    assert first["calibration"].groupby("profile").selected.sum().eq(1).all()
