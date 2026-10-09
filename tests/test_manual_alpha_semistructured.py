from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.alpha_mvp.research.manual_alpha_derivatives import align_derivative_frame
from src.alpha_mvp.research.manual_alpha_factors import FeatureContext, MECHANISMS, _divide, evaluate_mechanism
from src.alpha_mvp.research.manual_alpha_workflow import (
    _direction_path, _five_minute_ledger, _ledger, _normalize_signal,
    _positions, validate_manual_config,
)


def test_derivative_event_clock_and_missing_extremes():
    date = pd.to_datetime(["2023-01-01 00:00:00Z", "2023-01-01 01:00:00Z",
                           "2023-01-01 08:00:00Z", "2023-01-01 08:00:00.008Z",
                           "2023-01-01 09:00:00Z"], format="mixed", utc=True)
    frame = pd.DataFrame({
        "date": date,
        "mark_close": [100., 0., 101., 999999., 103.],
        "mark_close_time": pd.to_datetime(["2023-01-01 00:59:59.999Z",
            "2023-01-01 01:59:59.999Z", "2023-01-01 09:00:01Z", None,
            "2023-01-01 09:59:59.999Z"], format="mixed", utc=True),
        "index_close": [100., 100., 100., 100., 100.],
        "index_close_time": pd.to_datetime(["2023-01-01 00:59:59.999Z",
            "2023-01-01 01:59:59.999Z", "2023-01-01 08:59:59.999Z", None,
            "2023-01-01 09:59:59.999Z"], format="mixed", utc=True),
        "funding_rate": [1e-4, np.nan, np.nan, -2e-4, np.nan],
        "funding_interval_hours": [8., np.nan, np.nan, 8., np.nan],
    })
    raw = pd.date_range("2023-01-01", periods=11, freq="h", tz="UTC").strftime("%Y%m%d%H%M").to_numpy()
    result = align_derivative_frame(frame, raw, np.full(len(raw), 102.))
    assert result["derivative_basis"][0] == pytest.approx(0)
    assert np.isnan(result["derivative_basis"][1])  # invalid zero mark
    assert np.isnan(result["derivative_basis"][8])  # mark completes after decision
    assert result["derivative_funding_state"][7] == pytest.approx(1e-4)
    assert result["derivative_funding_state"][8] == pytest.approx(-2e-4)
    assert result["derivative_funding_age"][8] == pytest.approx(0)
    assert result["derivative_funding_event"][7] == pytest.approx(-2e-4)
    assert np.isnan(result["derivative_funding_event"][8])
    frame.loc[3, "funding_interval_hours"] = np.nan
    retained = align_derivative_frame(frame, raw, np.full(len(raw), 102.))
    assert retained["derivative_funding_event"][7] == pytest.approx(-2e-4)
    assert retained["derivative_funding_interval_missing"][7] == 1
    frame.loc[3, "funding_interval_hours"] = 4.
    with pytest.raises(ValueError, match="non-8h"):
        align_derivative_frame(frame, raw, np.full(len(raw), 102.))


def test_derivative_requires_source_columns():
    with pytest.raises(ValueError, match="missing"):
        align_derivative_frame(pd.DataFrame({"date": []}), np.array(["202301010000"]), np.array([100.]))


def test_all_manual_mechanisms_are_causal_and_finite_when_defined():
    cfg = yaml.safe_load(open("configs/research/manual_alpha_semistructured_20261003_round1.yaml", encoding="utf-8"))
    assert len(MECHANISMS) >= len(cfg["manual_alpha"]["mechanisms"]) == 20
    rng = np.random.default_rng(9)
    names = {x for card in cfg["manual_alpha"]["mechanisms"] for x in card["dependencies"]}
    n, a = 120, 12
    panels = {name: rng.normal(.1, .03, (n, a)).astype(np.float32) for name in names}
    for name in ("close_pos", "micro5_jump_share", "micro5_path_efficiency", "micro5_trade_hhi"):
        if name in panels:
            panels[name] = np.clip(panels[name], 0, 1)
    panels["derivative_basis"] = rng.normal(0, .001, (n, a)).astype(np.float32)
    panels["derivative_trade_mark_gap"] = rng.normal(0, .001, (n, a)).astype(np.float32)
    panels["derivative_funding_state"] = rng.normal(0, .0001, (n, a)).astype(np.float32)
    panels["realized_vol_24h"] = np.full((n, a), .02, dtype=np.float32)
    panels["market_vol_24h"] = np.full((n, a), .02, dtype=np.float32)
    before = FeatureContext({k: v.copy() for k, v in panels.items()})
    after_panels = {k: v.copy() for k, v in panels.items()}
    for array in after_panels.values():
        array[90:] = 1e9
    after = FeatureContext(after_panels)
    for card in cfg["manual_alpha"]["mechanisms"]:
        name = card["id"]
        a1 = evaluate_mechanism(name, before, card["windows"][0], card["thresholds"][0])
        a2 = evaluate_mechanism(name, after, card["windows"][0], card["thresholds"][0])
        np.testing.assert_allclose(a1[:90], a2[:90], equal_nan=True)
        assert not np.isinf(a1).any()
        assert set(card["dependencies"]).issubset(panels)


def test_training_scale_and_beta_projection_ignore_future_extremes():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 12)).astype(np.float32)
    train = np.arange(100) < 70
    beta = np.linspace(.5, 1.5, 12)
    p1, s1 = _normalize_signal(x, train, beta, .5)
    x[70:] = 1e9
    p2, s2 = _normalize_signal(x, train, beta, .5)
    np.testing.assert_allclose(p1[:70], p2[:70])
    assert s1 == s2
    assert np.max(np.abs(np.sum(p1, axis=1))) < 1e-5
    assert np.max(np.abs(p1 @ beta)) < 1e-5


def test_invalid_declaration_fails_loudly():
    cfg = yaml.safe_load(open("configs/research/manual_alpha_semistructured_20261003_round1.yaml", encoding="utf-8"))
    validate_manual_config(cfg)
    cfg["manual_alpha"]["mechanisms"][0]["id"] = "unknown"
    with pytest.raises(ValueError, match="invalid manual-alpha mechanism"):
        validate_manual_config(cfg)


def test_expanding_direction_only_uses_previous_discovery_labels():
    proposed = np.ones((12, 2), dtype=np.float32)
    residual = np.ones((12, 2), dtype=np.float32)
    residual[3:6] = -4
    calibration = np.arange(12) < 3
    folds = [(np.arange(12) >= 3) & (np.arange(12) < 6),
             (np.arange(12) >= 6) & (np.arange(12) < 9)]
    path, final, directions, _ = _direction_path(proposed, residual, calibration, folds, "expanding_fold")
    assert directions[0] == 1
    assert directions[1] == -1
    assert final == -1
    residual[9:] = 1e9  # validation/holdout labels are invisible to direction.
    path2, final2, directions2, _ = _direction_path(proposed, residual, calibration, folds, "expanding_fold")
    np.testing.assert_array_equal(path, path2)
    assert (final2, directions2) == (final, directions)


def test_48h_anchor_uses_epoch_hours_not_hour_of_day():
    dates = pd.date_range("2023-01-01", periods=120, freq="h", tz="UTC").strftime("%Y%m%d%H%M").to_numpy()
    proposed = np.arange(120, dtype=np.float32)[:, None]
    position = _positions(proposed, dates, 48, 1)
    changes = np.flatnonzero(np.diff(position[:, 0]) != 0)
    assert len(changes) in (2, 3)
    assert np.all(np.diff(changes) == 48)


def test_funding_cashflow_charges_prior_position_not_new_signal():
    position = np.array([[1.], [-1.], [0.]], dtype=np.float32)
    event = np.array([[np.nan], [1e-4], [np.nan]])
    book = _ledger(position, np.zeros_like(position), 4., np.zeros(3, dtype=bool), event)
    assert book["funding"][1, 0] == pytest.approx(-1e-4)
    assert book["funding"][0, 0] == 0
    np.testing.assert_allclose(book["net"], book["gross"] + book["funding"] - book["fee"])


def test_native_5m_ledger_books_settlement_before_new_hourly_position():
    clock = pd.date_range("2023-01-01 00:05", "2023-01-01 02:05", freq="5min", tz="UTC")
    fast_dates = clock.strftime("%Y%m%d%H%M").to_numpy()
    dates = np.array(["202301010100", "202301010200"])
    holdings = np.array([[1.], [-1.]], dtype=np.float32)
    event = np.array([[np.nan], [1e-4]])
    frame, totals = _five_minute_ledger(holdings, dates, fast_dates,
        np.full((len(fast_dates), 1), 100.), np.zeros(len(fast_dates), dtype=int), 4., event)
    row = frame[frame.completed_5m.eq("202301010200")].iloc[0]
    assert row.funding == pytest.approx(-1e-4)
    assert row.fee == pytest.approx(8e-4)  # +1 to -1 costs 2 units.
    assert totals["bridge_max_error"] < 1e-12


def test_internal_rolling_primitives_match_simple_reference_with_nan():
    x = np.array([[1., 2.], [2., np.nan], [3., 4.], [4., 6.], [5., 8.]], dtype=np.float32)
    context = FeatureContext({"field": x})
    for operation in ("mean", "std", "sum"):
        reference = getattr(pd.DataFrame(x).rolling(4, min_periods=2), operation)
        if operation == "std":
            expected = reference(ddof=0).to_numpy()
        else:
            expected = reference().to_numpy()
        np.testing.assert_allclose(context.roll("field", 4, operation), expected, equal_nan=True)
    expected_z = (x - context.roll("field", 4)) / context.roll("field", 4, "std")
    np.testing.assert_allclose(context.z("field", 4), np.clip(expected_z, -8, 8), equal_nan=True)
    np.testing.assert_allclose(context.lag("field", 2)[2:], x[:-2], equal_nan=True)
    assert np.isnan(context.lag("field", 2)[:2]).all()
    assert np.isnan(_divide(np.array([1., np.nan]), np.array([0., 2.]))).all()
    with pytest.raises(ValueError, match="outside complexity"):
        context.roll("field", 169)
