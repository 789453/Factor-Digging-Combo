from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.alpha_mvp.research.config import (
    EvaluationConfig,
    SplitConfig,
    load_research_config,
)
from src.alpha_mvp.research.attribution import compute_attribution
from src.alpha_mvp.research.evaluation import (
    daily_rank_ic,
    evaluate_factor,
    forward_returns,
    rank_research_results,
)
from src.alpha_mvp.research.selection import (
    build_factor_pairs,
    select_diverse_factors,
)
from src.alpha_mvp.research.templates import (
    generate_expressions,
    load_template_families,
)
from src.alpha_mvp.research.workflow import run_research
from src.alpha_mvp.research.store import ExperimentStore
from src.alpha_mvp.research.long_only import _curve_metrics, _portfolio_returns
from src.alpha_mvp.research.futures_performance import (
    _curve_metrics as futures_curve_metrics,
    _long_short_daily_returns,
)
from src.alpha_mvp.research.crypto_performance import crypto_long_short_returns
from src.alpha_mvp.research.crypto_return_patch import select_return_patch_candidates
from src.alpha_mvp.research.crypto_time_series_performance import (
    crypto_time_series_returns,
)
from src.alpha_mvp.research.time_series_evaluation import (
    asset_time_ic,
    evaluate_time_series_factor_medium,
)


def test_medium_time_series_screen_does_not_read_holdout():
    dates = pd.date_range("2023-01-01", periods=24 * 900, freq="h", tz="UTC")
    labels = dates.strftime("%Y%m%d%H%M").tolist()
    rng = np.random.default_rng(7)
    factor = rng.normal(size=(len(dates), 4))
    forward = 0.02 * factor + rng.normal(scale=0.5, size=factor.shape)
    split = SplitConfig(
        discovery_end="202312312300", validation_start="202401010000",
        validation_end="202412312300", holdout_start="202501010000",
    )
    config = EvaluationConfig(
        factor_mode="time_series", min_daily_names=3, time_series_block=168,
        time_series_position_window=168,
    )
    changed = factor.copy()
    changed[np.asarray(labels) >= split.holdout_start] = np.nan
    left = evaluate_time_series_factor_medium(factor, forward, labels, split, config)
    right = evaluate_time_series_factor_medium(changed, forward, labels, split, config)
    assert left == right


def test_template_generation_is_deterministic_and_interleaved():
    config = load_research_config("configs/research/smoke.yaml")
    families, _ = load_template_families(config.search.template_config)
    kwargs = dict(
        fields=config.search.fields,
        windows=config.search.windows,
        families=families,
        max_expressions=60,
        max_per_family=20,
        seed=17,
    )
    left = generate_expressions(**kwargs)
    right = generate_expressions(**kwargs)
    assert [record.canonical for record in left] == [
        record.canonical for record in right
    ]
    assert len({record.rank_equivalence_hash for record in left}) == len(left)
    counts = Counter(record.template_family for record in left)
    assert {
        "single", "binary_same", "binary_mixed", "pair_rolling",
        "multi_window", "cross_window_binary", "normalized_binary",
        "pair_change", "triple",
    }.issubset(counts)
    assert min(counts.values()) >= 3
    assert any("TsCorr" in record.expr for record in left)
    assert any(",5)" in record.expr and ",30)" in record.expr for record in left)
    assert any(len(set(record.fields)) == 3 for record in left)


def test_forward_returns_respects_entry_lag():
    close = np.arange(1.0, 9.0)[:, None]
    actual = forward_returns(close, horizon=2, entry_lag=1)
    assert np.isclose(actual[0, 0], close[3, 0] / close[1, 0] - 1)
    assert np.isnan(actual[-3:]).all()


def test_daily_rank_ic_uses_average_ties():
    factor = np.array([[1.0, 1.0, 3.0, 4.0]])
    returns = np.array([[4.0, 3.0, 2.0, 1.0]])
    actual = daily_rank_ic(factor, returns, min_names=3)[0]
    expected = pd.Series(factor[0]).rank(method="average").corr(
        pd.Series(returns[0]).rank(method="average")
    )
    assert np.isclose(actual, expected)


def test_direction_comes_only_from_discovery():
    dates = [
        "20230101", "20230102", "20230103",
        "20240101", "20240102", "20240103",
        "20250101", "20250102", "20250103",
    ]
    base = np.tile(np.arange(40, dtype=float), (len(dates), 1))
    forward = base.copy()
    forward[3:6] *= -1
    forward[6:] *= -1
    config = EvaluationConfig(
        min_daily_names=30,
        bootstrap_samples=10,
        bootstrap_block=2,
    )
    split = SplitConfig(
        discovery_end="20231231",
        validation_start="20240101",
        validation_end="20241231",
        holdout_start="20250101",
    )
    result = evaluate_factor(base, forward, dates, split, config)
    assert result.direction == 1
    assert result.discovery.mean_rank_ic > 0
    assert result.validation.mean_rank_ic < 0
    assert result.holdout.mean_rank_ic < 0
    assert not result.sign_consistent


def test_holdout_factor_values_do_not_change_search_diagnostics():
    dates = [
        "20230101", "20230102", "20230103",
        "20240101", "20240102", "20240103",
        "20250101", "20250102", "20250103",
    ]
    factor = np.tile(np.arange(40, dtype=float), (len(dates), 1))
    changed = factor.copy()
    changed[6:] = np.nan
    forward = factor.copy()
    config = EvaluationConfig(
        min_daily_names=30,
        bootstrap_samples=10,
        bootstrap_block=2,
    )
    split = SplitConfig(
        discovery_end="20231231",
        validation_start="20240101",
        validation_end="20241231",
        holdout_start="20250101",
    )
    left = evaluate_factor(
        factor, forward, dates, split, config, include_holdout=False
    )
    right = evaluate_factor(
        changed, forward, dates, split, config, include_holdout=False
    )
    assert left.coverage == right.coverage
    assert left.usable_days == right.usable_days
    assert left.turnover_proxy == right.turnover_proxy
    assert np.isnan(left.holdout.mean_rank_ic)


def test_holdout_metrics_do_not_change_research_score():
    config = EvaluationConfig()
    frame = pd.DataFrame({
        "status": ["OK", "OK"],
        "sign_consistent": [True, True],
        "validation_mean_rank_ic": [0.04, 0.02],
        "validation_rank_icir": [0.5, 0.3],
        "validation_positive_ratio": [0.60, 0.55],
        "discovery_mean_rank_ic": [0.03, 0.02],
        "coverage": [0.9, 0.8],
        "turnover_proxy": [0.1, 0.2],
        "nodes": [5, 8],
        "holdout_mean_rank_ic": [-1.0, 1.0],
    })
    first = rank_research_results(frame, config)
    changed = frame.copy()
    changed["holdout_mean_rank_ic"] *= -1000
    second = rank_research_results(changed, config)
    assert np.allclose(first["research_score"], second["research_score"])


def test_attribution_uses_search_evidence():
    frame = pd.DataFrame({
        "template_family": ["single", "binary"],
        "fields": ["x", "x|y"],
        "operators": ["Rank|TsMean", "Rank|Sub|TsMean|TsMean"],
        "windows": ["10", "10|10"],
        "eligible": [True, False],
        "research_score": [0.8, np.nan],
        "discovery_mean_rank_ic": [0.03, 0.01],
        "validation_mean_rank_ic": [0.02, -0.01],
        "coverage": [0.9, 0.8],
        "turnover_proxy": [0.1, 0.2],
        "holdout_mean_rank_ic": [-9.0, 9.0],
    })
    tables = compute_attribution(frame)
    assert set(tables) == {"template", "field", "operator", "window"}
    field_x = tables["field"].set_index("field").loc["x"]
    assert field_x["n_evaluated"] == 2
    assert "holdout_mean_rank_ic" not in tables["field"].columns


def test_diversity_selection_and_pairing():
    results = pd.DataFrame({
        "expr_hash": ["a", "b", "c", "d"],
        "expr": ["A", "B", "C", "D"],
        "template_family": ["single", "single", "binary", "binary"],
        "fields": ["x", "x", "y", "z"],
        "eligible": [True] * 4,
        "research_score": [0.9, 0.8, 0.7, 0.6],
    })
    selected = select_diverse_factors(
        results, top_n=3, max_per_family=2, max_per_primary_field=1
    )
    assert selected["expr_hash"].tolist() == ["a", "c", "d"]

    rng = np.random.default_rng(1)
    panels = {
        "a": rng.normal(size=(20, 40)),
        "c": rng.normal(size=(20, 40)),
        "d": rng.normal(size=(20, 40)),
    }
    pairs = build_factor_pairs(
        selected,
        panels,
        pair_top_n=2,
        candidate_n=3,
        max_similarity=0.8,
        research_mask=np.array([True] * 15 + [False] * 5),
    )
    assert 0 < len(pairs) <= 2
    assert (pairs["similarity"] <= 0.8).all()


def test_time_series_formula_diversity_uses_concrete_template_name():
    results = pd.DataFrame({
        "expr_hash": ["a", "b", "c"],
        "expr": ["A", "B", "C"],
        "template_family": ["time_series_formula"] * 3,
        "template_name": ["price_flow", "price_flow", "risk_gate"],
        "fields": ["x", "y", "z"],
        "eligible": [True] * 3,
        "research_score": [0.9, 0.8, 0.7],
    })
    selected = select_diverse_factors(
        results, top_n=3, max_per_family=1, max_per_primary_field=1
    )
    assert selected["expr_hash"].tolist() == ["a", "c"]


def test_unified_workflow_smoke(tmp_path):
    config = load_research_config("configs/research/smoke.yaml")
    small_search = replace(
        config.search,
        max_expressions=12,
        max_per_family=4,
    )
    small_selection = replace(
        config.selection,
        top_n=6,
        pair_candidate_n=6,
        pair_top_n=3,
    )
    small_evaluation = replace(
        config.evaluation,
        bootstrap_samples=10,
        bootstrap_block=5,
    )
    config = replace(
        config,
        search=small_search,
        selection=small_selection,
        evaluation=small_evaluation,
        output=replace(
            config.output,
            out_dir=str(tmp_path / "experiment"),
            store_path=str(tmp_path / "index.duckdb"),
        ),
    )
    manifest = run_research(config)
    assert manifest["status"] == "COMPLETED"
    assert manifest["research_summary"]["expressions"] == 12
    output = tmp_path / "experiment"
    search = pd.read_csv(output / "search_results.csv")
    holdout = pd.read_csv(output / "holdout_audit.csv")
    assert not any(column.startswith("holdout_") for column in search)
    assert any(column.startswith("holdout_") for column in holdout)
    assert len(holdout) == manifest["research_summary"]["selected"]
    assert (output / "overview.html").exists()
    assert (output / "top50.html").exists()
    assert "holdout_" not in (output / "top50.html").read_text(encoding="utf-8")

    second = replace(
        config,
        output=replace(config.output, out_dir=str(tmp_path / "experiment_2")),
    )
    reused = run_research(second)
    assert reused["research_summary"]["reused_results"] == 12
    assert reused["research_summary"]["newly_evaluated"] == 0


def test_completed_experiment_results_are_reused_by_evaluation_signature(tmp_path):
    store = ExperimentStore(str(tmp_path / "reuse.duckdb"))
    record = SimpleNamespace(
        expr_hash="hash-a", rank_equivalence_hash="rank-a", expr="Rank(x)",
        canonical="Rank(x)", template_name="one", template_family="single",
        template_order=0, fields=("x",), operators=("Rank",), windows=(),
        depth=2, nodes=2, priority_score=1.0,
    )
    store.initialize("experiment-a", "config-a")
    store.enqueue("experiment-a", [record])
    store.write_result("experiment-a", "hash-a", {"status": "OK", "value": 7})
    store.complete("experiment-a", {"evaluation_signature": "same-inputs"})
    reused = store.compatible_experiment_results("same-inputs", {"hash-a"})
    assert reused["hash-a"]["value"] == 7
    assert store.compatible_experiment_results("different", {"hash-a"}) == {}
    store.close()


def test_long_only_portfolio_uses_delayed_equal_weight_positions():
    factor = np.array([
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    stock_returns = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, -0.1],
        [0.2, 0.0, -0.2],
        [0.3, 0.0, -0.3],
        [0.4, 0.0, -0.4],
        [0.5, 0.0, -0.5],
    ])
    close = np.ones_like(factor)
    result = _portfolio_returns(
        factor,
        stock_returns,
        close,
        rebalance_days=2,
        top_pcts=[1 / 3],
        entry_lag=1,
    )[1 / 3]
    assert result[0] == result[1] == 0.0
    assert result[2] == 0.2
    assert result[3] == 0.3
    curve = _curve_metrics(result, rolling_window=3)
    assert np.isclose(curve["nav"][-1], 1.2 * 1.3 * 0.6 * 0.5)


def test_futures_long_short_curve_respects_direction_lag_and_report_mask():
    factor = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (5, 1))
    close = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.1, 1.0, 0.9, 0.8],
        [1.2, 1.0, 0.8, 0.6],
        [1.3, 1.0, 0.7, 0.5],
    ])
    report_mask = np.array([True, True, True, True, False])
    actual = _long_short_daily_returns(
        factor, close, direction=1, top_pct=0.25, entry_lag=1,
        report_mask=report_mask,
    )
    assert np.isnan(actual[:2]).all()
    assert np.isclose(actual[2], 0.1 - (-0.2))
    assert np.isclose(actual[3], 1.2 / 1.1 - 0.6 / 0.8)
    assert np.isnan(actual[4])


def test_futures_curve_metrics_handles_nonpositive_terminal_nav():
    metrics = futures_curve_metrics(np.array([-1.5, 0.0]))
    assert metrics["total_return"] < -1.0
    assert np.isnan(metrics["annualized_return"])


def test_crypto_long_short_returns_respects_lag_direction_and_cost():
    factor = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), (6, 1))
    close = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.1, 1.0, 1.0, 0.9],
        [1.2, 1.0, 1.0, 0.8],
        [1.3, 1.0, 1.0, 0.7],
        [1.4, 1.0, 1.0, 0.6],
    ])
    returns, turnover = crypto_long_short_returns(
        factor, close, direction=1, top_pct=0.25, entry_lag=1,
        cost_bps=10.0,
    )
    assert np.isnan(returns[:2]).all()
    assert np.isclose(turnover[2], 2.0)
    assert np.isclose(returns[2], 0.1 - (-0.1) - 0.002)
    assert np.isclose(returns[3], 1.2 / 1.1 - 0.8 / 0.9)


def test_crypto_return_patch_selects_top30_from_existing_metric():
    frame = pd.DataFrame({
        "expr_hash": [f"h{i}" for i in range(35)],
        "expr": [f"Rank(x{i})" for i in range(35)],
        "direction": [1] * 35,
        "validation_net_total_return": np.arange(35, dtype=float),
        "research_score": np.arange(35, dtype=float),
    })
    selected = select_return_patch_candidates(
        frame, "validation_net_total_return", 30
    )
    assert selected["expr_hash"].iloc[0] == "h34"
    assert selected["expr_hash"].iloc[-1] == "h5"
    assert selected["return_rank"].tolist() == list(range(1, 31))


def test_time_series_templates_are_role_constrained_and_not_cross_section_ranked():
    families, _ = load_template_families(
        "configs/research/templates_crypto_timeseries.yaml"
    )
    fields = (
        "ret_24h", "volume_shock_24h", "taker_imbalance_mean_24h",
        "realized_vol_24h", "micro_volume_trend", "micro_imbalance_trend",
    )
    records = generate_expressions(
        fields=fields,
        windows=(4, 12, 72, 168),
        families=families,
        max_expressions=80,
        max_per_family=20,
        seed=11,
    )
    assert records
    assert all(not record.expr.startswith("Rank(") for record in records)
    assert any("GatePos" in record.expr or "GateNeg" in record.expr for record in records)
    assert any("TsCorr" in record.expr for record in records)


def test_asset_time_ic_and_time_series_strategy_preserve_asset_identity():
    base = np.arange(60, dtype=float)
    factor = np.column_stack([base, -base])
    forward = np.column_stack([base, -base])
    ic = asset_time_ic(factor, forward, np.ones(60, dtype=bool), 24)
    assert np.allclose(ic, [1.0, 1.0])
    close = np.exp(np.column_stack([base, -base]) * 0.001)
    returns, turnover = crypto_time_series_returns(
        factor, close, direction=1, entry_lag=1, cost_bps=0.0,
        normalization_window=24, position_clip=2.0, holding_period=4,
    )
    assert np.isnan(returns[:24]).all()
    assert np.nanmean(returns[30:]) > 0
    assert np.nanmean(turnover[30:]) >= 0
