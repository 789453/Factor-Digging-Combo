import pandas as pd
import pytest

from src.alpha_mvp.research.proposal_prior import (
    load_proposal_prior, proposal_bonus,
)
from src.alpha_mvp.research.templates import TemplateFamily, generate_expressions


def test_smoothed_prior_mildly_prefers_supported_discovery_validation_evidence(tmp_path):
    rows = []
    for field, edge in (("flow", 0.03), ("noise", -0.03)):
        for _ in range(24):
            rows.append({
                "status": "OK", "fields": field, "operators": "TsMean",
                "template_name": "single", "windows": "6",
                "discovery_mean_rank_ic": edge,
                "validation_mean_rank_ic": edge,
            })
    path = tmp_path / "search_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    prior = load_proposal_prior(str(path))
    good = proposal_bonus(prior, "single", ("flow",), ("TsMean",), (6,), 1.0)
    bad = proposal_bonus(prior, "single", ("noise",), ("TsMean",), (6,), 1.0)
    assert good > bad
    assert abs(good) <= 0.25 and abs(bad) <= 0.25
    family = TemplateFamily(
        name="single", kind="single", order=1, ts_ops=("TsMean",),
        outer_transforms=("Id",), max_count=10,
    )
    proposals = generate_expressions(
        ["flow", "noise"], [6], [family], 2, 2,
        evidence_prior=prior, evidence_strength=1.0,
    )
    scores = {row.fields[0]: row.priority_score for row in proposals}
    assert scores["flow"] > scores["noise"]


def test_prior_refuses_holdout_columns(tmp_path):
    path = tmp_path / "contaminated.csv"
    pd.DataFrame({
        "status": ["OK"], "fields": ["flow"], "operators": ["TsMean"],
        "template_name": ["single"], "windows": ["6"],
        "discovery_mean_rank_ic": [0.1], "validation_mean_rank_ic": [0.1],
        "holdout_mean_rank_ic": [0.8],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="holdout"):
        load_proposal_prior(str(path))
