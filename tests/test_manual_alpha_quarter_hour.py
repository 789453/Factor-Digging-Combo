from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.manual_alpha_quarter_hour import align_quarter_hour_frame
from src.alpha_mvp.research.manual_alpha_factors import FeatureContext, evaluate_mechanism


def _two_hours() -> pd.DataFrame:
    date = pd.date_range("2023-01-01", periods=24, freq="5min", tz="UTC")
    open_phase = date.minute % 15 == 0
    ratio = np.where(open_phase, .75, .25)
    return pd.DataFrame({"date": date, "quote_volume": np.full(24, 10.),
                         "taker_buy_ratio": ratio})


def test_quarter_hour_complete_hour_weighting_and_future_firewall():
    frame = _two_hours()
    raw = np.array(["202301010000", "202301010100"])
    original = align_quarter_hour_frame(frame, raw)
    assert original["quarter_open_imbalance"][0] == pytest.approx(.5)
    assert original["quarter_other_imbalance"][0] == pytest.approx(-.5)
    assert original["quarter_open_volume_share"][0] == pytest.approx(1 / 3)
    frame.loc[12:, "taker_buy_ratio"] = 1.
    changed = align_quarter_hour_frame(frame, raw)
    for name in original:
        assert changed[name][0] == pytest.approx(original[name][0])


@pytest.mark.parametrize("corruption", ["missing", "bad_ratio", "zero_volume", "infinite_volume"])
def test_quarter_hour_missing_and_extreme_inputs_invalidate_only_affected_hour(corruption):
    frame = _two_hours()
    if corruption == "missing":
        frame = frame.drop(index=0)
    elif corruption == "bad_ratio":
        frame.loc[0, "taker_buy_ratio"] = 2.
    elif corruption == "zero_volume":
        frame.loc[0, "quote_volume"] = 0.
    else:
        frame.loc[0, "quote_volume"] = np.inf
    values = align_quarter_hour_frame(frame, np.array(["202301010000", "202301010100"]))
    assert all(np.isnan(x[0]) for x in values.values())
    assert all(np.isfinite(x[1]) for x in values.values())


def test_quarter_hour_rejects_duplicate_and_off_grid_clock():
    frame = _two_hours()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        align_quarter_hour_frame(duplicate, np.array(["202301010000"]))
    frame.loc[0, "date"] = pd.Timestamp("2023-01-01 00:00:00.001", tz="UTC")
    with pytest.raises(ValueError, match="off-grid"):
        align_quarter_hour_frame(frame, np.array(["202301010000"]))


def test_quarter_mechanisms_ignore_future_rows_and_extremes():
    rng = np.random.default_rng(9)
    panels = {
        "quarter_open_imbalance": rng.normal(0, .2, (120, 12)).astype(np.float32),
        "quarter_other_imbalance": rng.normal(0, .2, (120, 12)).astype(np.float32),
        "quarter_open_volume_share": rng.uniform(.1, .5, (120, 12)).astype(np.float32),
        "ret_1h": rng.normal(0, .01, (120, 12)).astype(np.float32),
        "realized_vol_24h": np.full((120, 12), .02, dtype=np.float32),
    }
    prior = FeatureContext({k: v.copy() for k, v in panels.items()})
    changed = {k: v.copy() for k, v in panels.items()}
    for values in changed.values():
        values[90:] = 1e9
    future = FeatureContext(changed)
    for name in ("quarter_open_pressure", "quarter_open_absorption"):
        x = evaluate_mechanism(name, prior, 12, .5)
        y = evaluate_mechanism(name, future, 12, .5)
        np.testing.assert_allclose(x[:90], y[:90], equal_nan=True)
        assert not np.isinf(x).any()
