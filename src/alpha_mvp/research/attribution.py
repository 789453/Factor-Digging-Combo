from __future__ import annotations

import numpy as np
import pandas as pd


def _summary(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    global_median = frame["research_score"].median(skipna=True)
    for value, group in frame.groupby(key, dropna=False):
        scores = group["research_score"].dropna()
        rows.append({
            key: value,
            "n_evaluated": len(group),
            "n_eligible": int(group["eligible"].fillna(False).sum()),
            "eligible_rate": float(group["eligible"].fillna(False).mean()),
            "mean_score": float(scores.mean()) if len(scores) else np.nan,
            "median_score": float(scores.median()) if len(scores) else np.nan,
            "p90_score": float(scores.quantile(0.9)) if len(scores) else np.nan,
            "score_uplift_vs_global_median": (
                float(scores.median() - global_median)
                if len(scores) and np.isfinite(global_median)
                else np.nan
            ),
            "mean_discovery_rank_ic": group[
                "discovery_mean_rank_ic"
            ].mean(skipna=True),
            "mean_validation_rank_ic": group[
                "validation_mean_rank_ic"
            ].mean(skipna=True),
            "validation_positive_rate": float(
                (group["validation_mean_rank_ic"] > 0).mean()
            ),
            "mean_coverage": group["coverage"].mean(skipna=True),
            "mean_turnover": group["turnover_proxy"].mean(skipna=True),
        })
    return pd.DataFrame(rows).sort_values(
        ["median_score", "n_evaluated"],
        ascending=[False, False],
        na_position="last",
    )


def _explode(frame: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    expanded = frame.copy()
    expanded[target] = expanded[column].fillna("").map(
        lambda value: [part for part in str(value).split("|") if part]
    )
    return expanded.explode(target)


def compute_attribution(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute search-only attribution; holdout columns are never accessed."""
    if results.empty:
        return {}
    required = {
        "template_family", "fields", "operators", "windows", "eligible",
        "research_score", "discovery_mean_rank_ic", "validation_mean_rank_ic",
        "coverage", "turnover_proxy",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Attribution missing columns: {sorted(missing)}")
    template_key = "template_name" if "template_name" in results else "template_family"
    return {
        "template": _summary(results, template_key),
        "field": _summary(_explode(results, "fields", "field"), "field"),
        "operator": _summary(
            _explode(results, "operators", "operator"), "operator"
        ),
        "window": _summary(_explode(results, "windows", "window"), "window"),
    }
