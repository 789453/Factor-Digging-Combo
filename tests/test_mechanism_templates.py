import pytest
import pandas as pd

from src.alpha_mvp.research.attribution import compute_attribution

from src.alpha_mvp.research.templates import (
    generate_expressions,
    load_template_families,
)


TEMPLATE_PATH = "configs/research/templates_crypto_mechanisms_v1.yaml"


def test_mechanism_templates_are_bounded_deterministic_and_attributed():
    families, raw = load_template_families(TEMPLATE_PATH)
    assert raw["version"] == "crypto_mechanism_templates_v1"
    assert all(family.hypothesis_card for family in families)
    assert all(family.max_count and family.max_depth and family.max_nodes for family in families)
    fields = sorted(set().union(*(
        set(family.fields_a) | set(family.fields_b) | set(family.fields_c)
        for family in families
    )))
    downside = [
        family for family in families
        if "downside_semivariance" in family.target_modes
    ]
    arguments = dict(
        fields=fields, windows=[3, 6, 12, 48, 96, 168],
        families=downside, max_expressions=100, max_per_family=35,
        seed=20260928,
    )
    first = generate_expressions(**arguments)
    second = generate_expressions(**arguments)
    assert [row.expr_hash for row in first] == [row.expr_hash for row in second]
    assert len(first) == 100
    assert {row.template_name for row in first} >= {
        "downside_flow_absorption", "ny_session_transfer",
        "continuous_liquidity_stress", "variance_persistence_baseline",
    }
    assert all(row.depth <= 10 and row.nodes <= 24 for row in first)


def test_missing_hypothesis_card_fails_clearly(tmp_path):
    config = tmp_path / "templates.yaml"
    config.write_text(
        "version: test\nfamilies:\n"
        "  - name: bad\n    kind: time_series_formula\n    order: 1\n"
        "    hypothesis_card: docs/cards/does_not_exist.md\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hypothesis_card not found"):
        load_template_families(str(config))


def test_template_attribution_uses_declared_name():
    frame = pd.DataFrame({
        "template_name": ["joint_pressure_state", "downside_flow_absorption"],
        "template_family": ["time_series_formula"] * 2,
        "fields": ["cluster_stress_affinity", "micro5_jump_share"],
        "operators": ["TsZScore", "GateNeg"],
        "windows": ["48", "48"],
        "eligible": [True, True],
        "research_score": [0.8, 0.6],
        "discovery_mean_rank_ic": [0.02, 0.01],
        "validation_mean_rank_ic": [0.01, 0.02],
        "coverage": [0.9, 0.9],
        "turnover_proxy": [0.1, 0.2],
    })
    templates = compute_attribution(frame)["template"]
    assert set(templates["template_name"]) == {
        "joint_pressure_state", "downside_flow_absorption",
    }
