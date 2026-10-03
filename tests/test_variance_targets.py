from dataclasses import replace

import numpy as np
import pytest

from src.alpha_mvp.research.targets import forward_micro5_variance


def test_semivariance_label_alignment_and_decomposition():
    rv = np.sqrt(np.arange(1.0, 11.0))[:, None]
    share = np.full_like(rv, 0.25)
    count = np.full_like(rv, 12.0)
    up = forward_micro5_variance(rv, share, count, 2, 1, "upside_semivariance")
    down = forward_micro5_variance(rv, share, count, 2, 1, "downside_semivariance")
    total = forward_micro5_variance(rv, share, count, 2, 1, "total_variance")
    # Signal at t=0, one-bar lag: realized 5m variance from hours 2 and 3.
    assert up[0, 0] == pytest.approx((3 + 4) * 0.25)
    assert down[0, 0] == pytest.approx((3 + 4) * 0.75)
    np.testing.assert_allclose(up + down, total, equal_nan=True)
    assert np.isnan(total[-3:]).all()


def test_missing_or_partial_hour_invalidates_only_affected_labels():
    rv = np.ones((12, 3))
    share = np.full_like(rv, 0.5)
    count = np.full_like(rv, 12.0)
    count[3, 0] = 11
    share[4, 1] = np.nan
    label = forward_micro5_variance(rv, share, count, 2, 1, "downside_semivariance")
    assert np.isnan(label[0, 0])
    assert np.isnan(label[1, 0])
    assert np.isnan(label[1, 1])
    assert label[0, 2] == pytest.approx(1.0)


def test_market_relative_label_removes_common_component():
    rv = np.sqrt(np.tile([1.0, 2.0, 3.0], (10, 1)))
    share = np.full_like(rv, 0.5)
    count = np.full_like(rv, 12.0)
    label = forward_micro5_variance(
        rv, share, count, 1, 1, "upside_semivariance", "market_relative"
    )
    np.testing.assert_allclose(np.nansum(label[:8], axis=1), 0.0, atol=1e-12)
    assert label[0, 0] < 0 < label[0, 2]


def test_variance_config_requires_explicit_5m_crypto_source():
    from src.alpha_mvp.research.config import load_research_config
    cfg = load_research_config("configs/research/crypto_multiscale_sessions_smoke.yaml")
    variance_weights = {
        "validation_edge_rank": 0.6, "discovery_edge_rank": 0.2,
        "validation_ir_rank": 0.1, "coverage_rank": 0.1,
    }
    valid = replace(cfg, evaluation=replace(
        cfg.evaluation, target_mode="downside_semivariance",
        score_weights=variance_weights, robustness_horizons=(),
    ))
    valid.validate()
    invalid = replace(valid, data=replace(cfg.data, crypto_fast_timeframe=None))
    with pytest.raises(ValueError, match="5m bars"):
        invalid.validate()
    with pytest.raises(ValueError, match="return strategy score"):
        replace(cfg.evaluation, target_mode="downside_semivariance").validate()
