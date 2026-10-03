"""Accounting and period semantics for the frozen evidence dashboard."""
import numpy as np
import pandas as pd

from src.alpha_mvp.research.aligned_visualization import (
    _asset_period_summary, _daily, _period,
)


def test_period_boundaries_and_daily_net_aggregation():
    split = {"discovery_end": "202506302355", "validation_end": "202601312355"}
    dates = ["202506302355", "202507010000", "202601312355", "202602010000"]
    assert _period(dates, split).tolist() == [
        "discovery", "validation", "validation", "holdout"]
    bars = pd.DataFrame({"date": dates, "strategy": ["main"] * 4,
                         "gross": [.001, .002, .003, .004],
                         "fee": [.0002] * 4,
                         "net": [.0008, .0018, .0028, .0038],
                         "abs_position": [.1, .2, .3, .4]})
    daily = _daily(bars, split)
    assert np.isclose(daily.net.sum(), bars.net.sum())
    assert daily.groupby("period").net.sum().to_dict() == {
        "discovery": .0008, "validation": .0046, "holdout": .0038}


def test_asset_contribution_is_equal_weight_not_standalone_return():
    ledger = pd.DataFrame({"date": ["202602010000"] * 2,
                           "period": ["holdout"] * 2,
                           "asset": ["A", "B"], "position": [1., -1.],
                           "gross": [.03, -.01], "fee": [.004, .002],
                           "net": [.026, -.012]})
    frame = _asset_period_summary(ledger)
    assert np.isclose(frame.net_contribution.sum(), ledger.net.mean())
    assert np.isclose(frame.net_contribution_pct.sum(), .7)
