import numpy as np
import pandas as pd

from src.alpha_mvp.research.config import SplitConfig
from src.alpha_mvp.research.incremental_information import (
    incremental_variance_information, variance_baseline_panels,
)


def test_discovery_fitted_increment_detects_new_signal_and_ignores_holdout():
    rng = np.random.default_rng(7)
    n, assets = 420, 4
    x = rng.normal(size=(n, assets))
    candidate = rng.normal(size=(n, assets))
    y = 2.0 * x + 1.5 * candidate + rng.normal(scale=0.2, size=(n, assets))
    baseline = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=2)
    dates = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y%m%d%H%M").tolist()
    split = SplitConfig(
        discovery_end=dates[239], validation_start=dates[240],
        validation_end=dates[359], holdout_start=dates[360],
    )
    useful = incremental_variance_information(candidate, y, baseline, dates, split)
    duplicate = incremental_variance_information(x, y, baseline, dates, split)
    assert useful["incremental_mean_delta_r2"] > 0.5
    assert useful["incremental_positive_asset_share"] == 1.0
    assert abs(duplicate["incremental_mean_delta_r2"]) < 0.02
    changed = candidate.copy()
    changed[360:] = 1e8
    y_changed = y.copy()
    y_changed[360:] = -1e8
    assert incremental_variance_information(
        changed, y_changed, baseline, dates, split
    ) == useful


def test_baseline_relative_target_and_missing_values():
    rv = np.sqrt(np.tile([1.0, 2.0, 3.0], (8, 1)))
    share = np.full_like(rv, 0.25)
    day = np.ones_like(rv)
    evening = np.zeros_like(rv)
    rv[2, 0] = np.nan
    controls = variance_baseline_panels(
        rv, share, day, evening, "downside_semivariance", "market_relative"
    )
    assert controls.shape == (8, 3, 3)
    assert np.isnan(controls[2, :, 0]).all()  # fewer than three valid assets
    np.testing.assert_allclose(controls[0, :, 0].sum(), 0.0)
