from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.cluster_state import build_cluster_stress_affinity
from src.alpha_mvp.research.config import ClusterStateConfig, SplitConfig


def _data():
    rng = np.random.default_rng(19)
    t, n = 500, 4
    stressed = rng.random((t, n)) < 0.16
    flow = rng.normal(0.2, 0.2, (t, n)) - stressed * 1.4
    jump = rng.normal(0.1, 0.15, (t, n)) + stressed * 1.2
    illiquid = rng.normal(0.1, 0.2, (t, n)) + stressed * 1.3
    dates = pd.date_range("2024-01-01", periods=t, freq="h").strftime("%Y%m%d%H%M").tolist()
    split = SplitConfig(
        discovery_end=dates[299], validation_start=dates[300],
        validation_end=dates[399], holdout_start=dates[400],
    )
    config = ClusterStateConfig(
        enabled=True, history_window=24, fit_stride=2,
        max_fit_samples=1000, min_cluster_share=0.02,
    )
    panels = {
        "micro5_taker_imbalance": flow,
        "micro5_jump_share": jump,
        "session_illiquidity_surprise": illiquid,
    }
    return panels, stressed, dates, split, config


def test_cluster_field_is_causal_bounded_and_semantic():
    panels, stressed, dates, split, config = _data()
    affinity, metadata = build_cluster_stress_affinity(panels, dates, split, config)
    assert affinity.shape == stressed.shape
    assert np.nanmin(affinity) >= 0 and np.nanmax(affinity) <= 1
    assert np.nanmean(affinity[300:400][stressed[300:400]]) > np.nanmean(
        affinity[300:400][~stressed[300:400]]
    )
    assert metadata["fit_end"] == split.discovery_end
    assert metadata["fit_samples"] <= config.max_fit_samples
    changed = {name: value.copy() for name, value in panels.items()}
    for value in changed.values():
        value[400:] = 1e6
    other, other_metadata = build_cluster_stress_affinity(changed, dates, split, config)
    np.testing.assert_allclose(affinity[:400], other[:400], equal_nan=True)
    assert metadata == other_metadata


def test_cluster_field_missing_and_extreme_inputs():
    panels, _, dates, split, config = _data()
    panels["micro5_jump_share"][350, 0] = np.nan
    panels["session_illiquidity_surprise"][351, 1] = 1e6
    affinity, _ = build_cluster_stress_affinity(panels, dates, split, config)
    assert np.isnan(affinity[350, 0])
    assert np.isfinite(affinity[351, 1])
    with pytest.raises(ValueError, match="missing dependencies"):
        build_cluster_stress_affinity(
            {"micro5_jump_share": panels["micro5_jump_share"]},
            dates, split, config,
        )
    with pytest.raises(ValueError, match="n_clusters"):
        replace(config, n_clusters=12).validate()
