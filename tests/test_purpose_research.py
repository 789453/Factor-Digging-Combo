from dataclasses import replace

import numpy as np
import pandas as pd

from src.alpha_mvp.native5_fields import NATIVE5_FIELD_SPECS, add_crypto_native_5m_features
from src.alpha_mvp.data import load_crypto_native_5m
from src.alpha_mvp.research.config import EvaluationConfig, SplitConfig, load_research_config
from src.alpha_mvp.research.purpose import (
    build_purpose_targets, common_factor, evaluate_purpose,
    incremental_purpose_information, liquidity_style_baseline_panels,
)
from src.alpha_mvp.research.selection import select_signal_distinct_factors
from src.alpha_mvp.research.templates import generate_expressions, load_template_families
from src.alpha_mvp.research.attribution import compute_attribution


def _synthetic_panel():
    rng = np.random.default_rng(29)
    n, assets = 700, 5
    common = rng.normal(0, 0.002, n)
    innovations = rng.normal(0, 0.001, (n, assets))
    returns = common[:, None] * np.array([0.7, 0.9, 1.1, 1.3, 1.5]) + innovations
    prices = np.exp(np.cumsum(returns, axis=0)) * 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y%m%d%H%M").tolist()
    split = SplitConfig(dates[399], dates[400], dates[599], dates[600])
    return prices, dates, split


def test_residual_target_fits_discovery_only_and_common_has_one_sample_per_time():
    prices, dates, split = _synthetic_panel()
    first = build_purpose_targets(prices, dates, split, horizon=2, lag=0)
    changed = prices.copy()
    changed[600:] *= np.linspace(1, 2, len(changed) - 600)[:, None]
    second = build_purpose_targets(changed, dates, split, horizon=2, lag=0)
    np.testing.assert_allclose(first.betas, second.betas, equal_nan=True)
    np.testing.assert_allclose(first.residual[:598], second.residual[:598], equal_nan=True)
    assert first.common.shape == (len(dates), 1)
    assert np.isnan(first.residual[399]).all()  # split-crossing label purged
    assert common_factor(np.ones_like(prices)).shape == (len(dates), 1)


def test_hybrid_requires_both_axes_and_incremental_ignores_holdout():
    prices, dates, split = _synthetic_panel()
    targets = build_purpose_targets(prices, dates, split, horizon=1, lag=0)
    factor = prices / prices.mean(axis=0) - 1
    config = EvaluationConfig(research_track="hybrid", factor_mode="time_series", horizon=1, entry_lag=0)
    result = evaluate_purpose(factor, targets, dates, split, config, "coarse")
    assert "alpha_validation_mean_rank_ic" in result
    assert "common_validation_mean_rank_ic" in result
    assert result["validation_mean_rank_ic"] <= result["alpha_validation_mean_rank_ic"]
    baseline = np.ones((*factor.shape, 1))
    baseline[:, :, 0] = factor
    metrics = incremental_purpose_information(factor, targets.residual, baseline, dates, split)
    changed = replace(split, holdout_start=dates[620])
    assert metrics == incremental_purpose_information(factor, targets.residual, baseline, dates, changed)


def test_native5_fields_missing_extreme_and_completed_15m_input():
    times = pd.date_range("2024-03-10 06:00", periods=60, freq="5min", tz="UTC")
    base = pd.DataFrame({
        "trade_date": times.strftime("%Y%m%d%H%M"), "ts_code": "BTCUSDT",
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100 + np.arange(60) * 0.02,
        "vwap": 100.0, "quote_volume": 1000.0 + np.arange(60),
        "trade_count": 100.0 + np.arange(60),
        "taker_buy_ratio": 0.55,
        "ret_15m_completed": 0.001, "flow_15m_completed": 0.1,
    })
    base.loc[30, "close"] = np.nan
    base.loc[40, "quote_volume"] = 1e12
    featured = add_crypto_native_5m_features(base)
    assert set(["ret_5m", "rv_12", "upside_share_12", "flow_15m_completed"]) <= set(NATIVE5_FIELD_SPECS)
    assert np.isnan(featured.loc[30, "ret_5m"])
    assert np.isfinite(featured.loc[40, "volume_shock_12"])
    assert featured.loc[20, ["us_day_flag", "us_evening_flag", "us_overnight_flag"]].sum() == 1


def test_native5_loader_never_joins_unfinished_15m_bar(tmp_path):
    root = tmp_path / "BTCUSDT"
    root.mkdir()
    five_times = pd.date_range("2024-01-01", periods=7, freq="5min", tz="UTC")
    fast = pd.DataFrame({
        "date": five_times, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "volume": 2.0,
        "quote_volume": 200.0, "trade_count": 10,
        "taker_buy_ratio": 0.5, "vwap": 100.0,
    })
    context = pd.DataFrame({
        "date": [five_times[0], five_times[3], five_times[6]],
        "open": [100.0] * 3, "close": [101.0, 102.0, 103.0],
        "taker_buy_ratio": [0.6, 0.7, 0.8],
    })
    fast.to_parquet(root / "5m.parquet")
    context.to_parquet(root / "15m.parquet")
    loaded = load_crypto_native_5m(str(tmp_path), "2024-01-01", "2024-01-01 01:00")
    assert np.isnan(loaded.iloc[0]["ret_15m_completed"])
    assert np.isclose(loaded.iloc[2]["flow_15m_completed"], 0.2)
    assert np.isclose(loaded.iloc[3]["flow_15m_completed"], 0.2)
    assert np.isclose(loaded.iloc[5]["flow_15m_completed"], 0.4)


def test_signal_distinct_selection_drops_alias_and_never_reads_holdout():
    dates, assets = 500, 4
    base = np.arange(dates * assets, dtype=float).reshape(dates, assets)
    panels = {"a": base, "alias": base + 1e-9, "other": np.sin(base)}
    frame = pd.DataFrame({
        "expr_hash": ["a", "alias", "other"],
        "expr": ["a", "alias", "other"],
        "template_name": ["one", "two", "three"],
        "fields": ["x", "y", "z"],
        "eligible": [True, True, True],
        "research_score": [0.9, 0.8, 0.7],
        "holdout_mean_rank_ic": [-1e9, 1e9, 1e9],
    })
    selected, audit = select_signal_distinct_factors(
        frame, panels.__getitem__, np.arange(dates) < 400,
        2, 2, 2, 0.98,
    )
    assert selected.expr_hash.tolist() == ["a", "other"]
    assert audit.set_index("expr_hash").loc["alias", "decision"] == "near_duplicate"
    frame["holdout_mean_rank_ic"] *= -1
    again, _ = select_signal_distinct_factors(
        frame, panels.__getitem__, np.arange(dates) < 400,
        2, 2, 2, 0.98,
    )
    assert again.expr_hash.tolist() == selected.expr_hash.tolist()


def test_liquidity_concentration_target_is_future_complete_and_purged():
    prices, dates, split = _synthetic_panel()
    hhi = np.full(prices.shape, 0.12)
    hhi[500, 0] = np.nan
    hhi[510, 1] = 1e6
    targets = build_purpose_targets(prices, dates, split, 2, 1, style_hhi=hhi)
    assert np.isnan(targets.style[497, 0])
    assert np.isnan(targets.style[399]).all()
    assert np.isclose(targets.style[506, 1], 0.12)
    assert targets.style[508, 1] > 1e5
    changed = hhi.copy()
    changed[600:] = 0.8
    second = build_purpose_targets(prices, dates, split, 2, 1, style_hhi=changed)
    np.testing.assert_allclose(targets.style[:597], second.style[:597], equal_nan=True)
    config = EvaluationConfig(research_track="style_proxy", target_mode="liquidity_concentration",
                              factor_mode="time_series", horizon=2, entry_lag=1)
    result = evaluate_purpose(hhi, targets, dates, split, config, "coarse")
    assert "validation_mean_rank_ic" in result


def test_liquidity_baseline_requires_complete_hour_and_valid_configuration():
    names = ("micro5_volume_hhi", "micro5_trade_hhi", "volume_shock_24h",
             "trade_count_shock_24h", "us_day_flag", "us_evening_flag")
    panels = {name: np.ones((3, 4)) for name in names}
    panels["micro5_bar_count"] = np.full((3, 4), 12)
    panels["micro5_bar_count"][1, 0] = 11
    baseline = liquidity_style_baseline_panels(panels)
    assert baseline.shape == (3, 4, len(names))
    assert np.isnan(baseline[1, 0]).all()
    assert np.isfinite(baseline[0, 0]).all()
    EvaluationConfig(research_track="style_proxy", target_mode="liquidity_concentration").validate()
    import pytest
    with pytest.raises(ValueError, match="requires"):
        EvaluationConfig(research_track="style_proxy", target_mode="return").validate()
    with pytest.raises(ValueError, match="target_scope=absolute"):
        EvaluationConfig(research_track="style_proxy", target_mode="liquidity_concentration",
                         target_scope="market_relative").validate()


def test_style_template_is_bounded_deterministic_and_attributed():
    config = load_research_config("configs/research/crypto_style_hhi_10000_2023.yaml")
    families, _ = load_template_families(config.search.template_config)
    assert len(families) == 6
    assert all(f.target_modes == ("liquidity_concentration",) for f in families)
    assert all(f.hypothesis_card and f.max_count and f.max_depth and f.max_nodes for f in families)
    args = dict(fields=config.search.fields, windows=config.search.windows,
                families=families, max_expressions=120, max_per_family=30,
                seed=config.search.seed)
    first = generate_expressions(**args)
    second = generate_expressions(**args)
    assert len(first) == 120
    assert [r.expr_hash for r in first] == [r.expr_hash for r in second]
    assert len({r.template_name for r in first}) == 6
    frame = pd.DataFrame({
        "template_name": [r.template_name for r in first],
        "template_family": [r.template_family for r in first],
        "fields": ["|".join(r.fields) for r in first],
        "operators": ["|".join(r.operators) for r in first],
        "windows": ["|".join(map(str, r.windows)) for r in first],
        "eligible": [True] * len(first),
        "research_score": np.linspace(0, 1, len(first)),
        "discovery_mean_rank_ic": [0.01] * len(first),
        "validation_mean_rank_ic": [0.01] * len(first),
        "coverage": [1.0] * len(first),
        "turnover_proxy": [0.0] * len(first),
    })
    attribution = compute_attribution(frame)["template"]
    assert set(attribution.template_name) == {f.name for f in families}
