from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from .. import fastops


def select_diverse_factors(
    results: pd.DataFrame,
    top_n: int,
    max_per_family: int,
    max_per_primary_field: int,
) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    ranked = results[
        results["eligible"].astype(bool) & results["research_score"].notna()
    ].sort_values("research_score", ascending=False)
    selected = []
    family_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        family = str(row.get("template_family", "unknown"))
        if family == "time_series_formula":
            family = str(row.get("template_name", family))
        fields = row.get("fields", ())
        if isinstance(fields, str):
            fields = tuple(x for x in fields.split("|") if x)
        primary = fields[0] if fields else "unknown"
        if family_counts.get(family, 0) >= max_per_family:
            continue
        if field_counts.get(primary, 0) >= max_per_primary_field:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        field_counts[primary] = field_counts.get(primary, 0) + 1
        if len(selected) >= top_n:
            break
    return pd.DataFrame(selected, columns=results.columns).reset_index(drop=True)


def select_signal_distinct_factors(
    results: pd.DataFrame,
    factor_provider: Callable[[str], np.ndarray],
    research_mask: np.ndarray,
    top_n: int,
    max_per_family: int,
    max_per_primary_field: int,
    max_similarity: float,
    max_candidates: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select D/V-distinct information while retaining template/field quotas."""
    if not 0 < max_similarity < 1:
        raise ValueError("max_similarity must be in (0, 1)")
    ranked = results.loc[
        results["eligible"].fillna(False).astype(bool)
        & results["research_score"].notna()
    ].sort_values("research_score", ascending=False).head(max_candidates)
    if ranked.empty:
        return ranked.copy(), pd.DataFrame(columns=["expr_hash", "decision", "similarity", "nearest_hash"])
    mask = np.asarray(research_mask, dtype=bool)
    sampled = np.flatnonzero(mask)
    sampled = sampled[::max(1, len(sampled) // 5000)]
    chosen, panels, audits = [], [], []
    family_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        family = str(row["template_name"])
        fields = str(row.get("fields", "")).split("|")
        primary = fields[0] if fields and fields[0] else "unknown"
        if family_counts.get(family, 0) >= max_per_family or field_counts.get(primary, 0) >= max_per_primary_field:
            audits.append({"expr_hash": row["expr_hash"], "decision": "quota", "similarity": np.nan, "nearest_hash": None})
            continue
        values = np.asarray(factor_provider(str(row["expr"])), dtype=float)
        if values.ndim != 2 or len(values) != len(mask):
            raise ValueError("factor_provider returned an unaligned panel")
        flattened = values[sampled].reshape(-1)
        nearest, nearest_hash = -1.0, None
        for previous, previous_hash in panels:
            valid = np.isfinite(flattened) & np.isfinite(previous)
            if valid.sum() < 300:
                continue
            a, b = flattened[valid], previous[valid]
            a, b = a - a.mean(), b - b.mean()
            denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
            similarity = abs(float(np.dot(a, b) / denominator)) if denominator > 1e-12 else np.nan
            if np.isfinite(similarity) and similarity > nearest:
                nearest, nearest_hash = similarity, previous_hash
        if nearest >= max_similarity:
            audits.append({"expr_hash": row["expr_hash"], "decision": "near_duplicate", "similarity": nearest, "nearest_hash": nearest_hash})
            continue
        chosen.append(row)
        panels.append((flattened, row["expr_hash"]))
        family_counts[family] = family_counts.get(family, 0) + 1
        field_counts[primary] = field_counts.get(primary, 0) + 1
        audits.append({"expr_hash": row["expr_hash"], "decision": "selected", "similarity": nearest if nearest >= 0 else np.nan, "nearest_hash": nearest_hash})
        if len(chosen) >= top_n:
            break
    selected = pd.DataFrame(chosen, columns=results.columns).reset_index(drop=True)
    return selected, pd.DataFrame(audits)


def _daily_similarity(left: np.ndarray, right: np.ndarray) -> float:
    correlations = fastops.daily_corr(
        left,
        right,
        rank=False,
        min_valid=10,
    )
    finite = correlations[np.isfinite(correlations)]
    return float(np.median(np.abs(finite))) if len(finite) else np.nan


def build_factor_pairs(
    selected: pd.DataFrame,
    factor_panels: dict[str, np.ndarray],
    pair_top_n: int,
    candidate_n: int,
    max_similarity: float,
    research_mask: np.ndarray | None = None,
    similarity_mode: str = "cross_sectional",
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    candidates = selected.head(candidate_n)
    ranked_panels = {}
    for expr_hash, panel in factor_panels.items():
        research_panel = panel if research_mask is None else panel[research_mask]
        ranked_panels[expr_hash] = (
            fastops.rank_cs(research_panel)
            if similarity_mode == "cross_sectional" else research_panel
        )
    rows = []
    for (_, left), (_, right) in combinations(candidates.iterrows(), 2):
        left_hash = str(left["expr_hash"])
        right_hash = str(right["expr_hash"])
        if left_hash not in factor_panels or right_hash not in factor_panels:
            continue
        if similarity_mode == "time_series":
            correlations = []
            for asset in range(ranked_panels[left_hash].shape[1]):
                x = ranked_panels[left_hash][:, asset]
                y = ranked_panels[right_hash][:, asset]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 168 and np.std(x[valid]) > 1e-12 and np.std(y[valid]) > 1e-12:
                    correlations.append(abs(float(np.corrcoef(x[valid], y[valid])[0, 1])))
            similarity = float(np.median(correlations)) if correlations else np.nan
        else:
            similarity = _daily_similarity(
                ranked_panels[left_hash], ranked_panels[right_hash]
            )
        if not np.isfinite(similarity) or similarity > max_similarity:
            continue
        left_score = float(left["research_score"])
        right_score = float(right["research_score"])
        pair_score = (left_score + right_score) / 2.0 + 0.15 * (1.0 - similarity)
        rows.append({
            "left_expr_hash": left_hash,
            "right_expr_hash": right_hash,
            "left_expr": left["expr"],
            "right_expr": right["expr"],
            "left_family": left["template_family"],
            "right_family": right["template_family"],
            "similarity": similarity,
            "pair_score": pair_score,
        })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("pair_score", ascending=False)
        .head(pair_top_n)
        .reset_index(drop=True)
    )
