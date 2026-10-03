from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _safe(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index)), errors="coerce")


def _short_expr(value: str, limit: int = 54) -> str:
    value = str(value)
    return value if len(value) <= limit else value[:limit - 1] + "…"


def build_reports(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    attribution: dict[str, pd.DataFrame],
    out_dir: Path,
    top_n: int = 50,
) -> dict[str, str]:
    """Create search-only portfolio-level reports; holdout metrics are excluded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = results[[c for c in results if not c.startswith("holdout_")]].copy()
    top = selected[[c for c in selected if not c.startswith("holdout_")]].head(top_n).copy()

    overview = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "模板评估量与入选量",
            "研究评分分布",
            "发现期与验证期 RankIC",
            "验证期 RankIC 与 ICIR",
            "字段归因（中位评分）",
            "算子归因（中位评分）",
        ),
    )
    family = clean.groupby("template_family", dropna=False).agg(
        evaluated=("expr_hash", "size"),
        eligible=("eligible", "sum"),
    ).reset_index()
    overview.add_trace(go.Bar(x=family["template_family"], y=family["evaluated"], name="评估"), 1, 1)
    overview.add_trace(go.Bar(x=family["template_family"], y=family["eligible"], name="合格"), 1, 1)
    overview.add_trace(go.Histogram(x=_safe(clean, "research_score"), nbinsx=30, name="评分"), 1, 2)
    overview.add_trace(go.Scatter(
        x=_safe(clean, "discovery_mean_rank_ic"),
        y=_safe(clean, "validation_mean_rank_ic"),
        mode="markers",
        marker={"size": 5, "opacity": 0.55},
        text=clean["template_family"],
        name="候选",
    ), 1, 3)
    overview.add_trace(go.Scatter(
        x=_safe(clean, "validation_mean_rank_ic"),
        y=_safe(clean, "validation_rank_icir"),
        mode="markers",
        marker={"size": 5, "opacity": 0.55, "color": _safe(clean, "coverage"), "colorscale": "Viridis"},
        text=clean["template_family"],
        name="验证",
    ), 2, 1)
    for column, position, label in (
        ("field", (2, 2), "字段"),
        ("operator", (2, 3), "算子"),
    ):
        frame = attribution.get(column, pd.DataFrame()).head(15)
        if not frame.empty:
            overview.add_trace(go.Bar(
                x=frame["median_score"],
                y=frame[column].astype(str),
                orientation="h",
                name=label,
            ), *position)
    overview.update_layout(
        title="因子研究总体概览（仅发现期与验证期）",
        height=900,
        barmode="group",
        template="plotly_white",
    )
    overview_path = out_dir / "overview.html"
    overview.write_html(overview_path, include_plotlyjs="cdn")

    metrics = [
        "research_score",
        "validation_mean_rank_ic",
        "validation_rank_icir",
        "validation_nw_tstat",
        "validation_positive_ratio",
        "coverage",
        "turnover_proxy",
    ]
    labels = [_short_expr(value) for value in top.get("expr", pd.Series(dtype=str))]
    normalized = top[metrics].apply(pd.to_numeric, errors="coerce")
    normalized = (normalized - normalized.mean()) / normalized.std(ddof=0).replace(0, np.nan)
    detail = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top50 指标标准化热力图",
            "验证期 RankIC（95% 区间）",
            "覆盖率与换手代理",
            "模板构成",
        ),
        specs=[[{"type": "heatmap"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]],
    )
    detail.add_trace(go.Heatmap(
        z=normalized.to_numpy(),
        x=metrics,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        colorbar={"title": "z"},
    ), 1, 1)
    detail.add_trace(go.Scatter(
        x=_safe(top, "validation_mean_rank_ic"),
        y=labels,
        mode="markers",
        error_x={
            "type": "data",
            "symmetric": False,
            "array": (_safe(top, "validation_ci_high") - _safe(top, "validation_mean_rank_ic")).clip(lower=0),
            "arrayminus": (_safe(top, "validation_mean_rank_ic") - _safe(top, "validation_ci_low")).clip(lower=0),
        },
        name="RankIC",
    ), 1, 2)
    detail.add_trace(go.Scatter(
        x=_safe(top, "turnover_proxy"),
        y=_safe(top, "coverage"),
        mode="markers",
        marker={"color": _safe(top, "research_score"), "colorscale": "Viridis", "size": 9},
        text=labels,
        name="Top50",
    ), 2, 1)
    composition = top["template_family"].value_counts()
    detail.add_trace(go.Pie(labels=composition.index, values=composition.values, hole=0.45), 2, 2)
    detail.update_layout(
        title=f"Top {len(top)} 因子技术指标（选择阶段，不含留出期）",
        height=max(1050, 430 + len(top) * 13),
        template="plotly_white",
    )
    top_path = out_dir / "top50.html"
    detail.write_html(top_path, include_plotlyjs="cdn")
    top.to_csv(out_dir / "top50_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "evaluated": int(len(clean)),
        "eligible": int(clean.get("eligible", pd.Series(dtype=bool)).fillna(False).sum()),
        "selected": int(len(selected)),
        "reported_top": int(len(top)),
        "best_template_by_median_score": (
            attribution["template"].iloc[0][
                "template_name" if "template_name" in attribution["template"]
                else "template_family"
            ]
            if not attribution.get("template", pd.DataFrame()).empty else None
        ),
        "best_field_by_median_score": (
            attribution["field"].iloc[0]["field"]
            if not attribution.get("field", pd.DataFrame()).empty else None
        ),
        "best_operator_by_median_score": (
            attribution["operator"].iloc[0]["operator"]
            if not attribution.get("operator", pd.DataFrame()).empty else None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "overview": "overview.html",
        "top50": "top50.html",
        "top50_metrics": "top50_metrics.csv",
        "summary": "summary.json",
    }
