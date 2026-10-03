from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from ..crypto_fields import add_crypto_features
from ..data import load_crypto_parquet
from ..evaluator import BatchEvaluator, make_panels
from .config import load_research_config
from .crypto_performance import (
    _curve_from_returns,
    _group_return_series,
    _period_strategy_metrics,
    crypto_long_short_returns,
)
from .templates import load_template_families


@dataclass(frozen=True)
class CryptoReturnPatchConfig:
    base_config: str
    source_csv: str
    output_html: str
    ranking_column: str = "validation_net_total_return"
    top_n: int = 30

    def validate(self) -> None:
        if self.top_n < 30:
            raise ValueError("top_n must be at least 30 for the 5/10/15 buckets")
        if not Path(self.source_csv).is_file():
            raise FileNotFoundError(f"source_csv does not exist: {self.source_csv}")
        if not Path(self.base_config).is_file():
            raise FileNotFoundError(f"base_config does not exist: {self.base_config}")


def load_crypto_return_patch_config(path: str) -> CryptoReturnPatchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = CryptoReturnPatchConfig(**raw)
    config.validate()
    return config


def select_return_patch_candidates(
    results: pd.DataFrame,
    ranking_column: str,
    top_n: int = 30,
) -> pd.DataFrame:
    required = {"expr_hash", "expr", "direction", ranking_column}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"search results missing columns: {sorted(missing)}")
    frame = results.copy()
    frame[ranking_column] = pd.to_numeric(frame[ranking_column], errors="coerce")
    frame["direction"] = pd.to_numeric(frame["direction"], errors="coerce")
    frame = frame[
        frame[ranking_column].notna()
        & frame["direction"].isin([-1, 1])
        & frame["expr"].notna()
    ]
    if len(frame) < top_n:
        raise ValueError(
            f"Only {len(frame)} rows have finite {ranking_column}; need {top_n}"
        )
    selected = frame.sort_values(
        [ranking_column, "research_score"],
        ascending=False,
        na_position="last",
    ).head(top_n).reset_index(drop=True)
    selected["return_rank"] = np.arange(1, len(selected) + 1)
    return selected


def run_crypto_return_patch(config: CryptoReturnPatchConfig) -> dict:
    config.validate()
    base = load_research_config(config.base_config)
    if base.data.source != "crypto_parquet":
        raise ValueError("base_config must use crypto_parquet data")
    output_html = Path(config.output_html)
    if output_html.exists():
        raise FileExistsError(
            f"Patch report already exists: {output_html}; choose a new path"
        )
    output_html.parent.mkdir(parents=True, exist_ok=True)

    search = pd.read_csv(config.source_csv)
    selected = select_return_patch_candidates(
        search, config.ranking_column, config.top_n
    )
    raw = load_crypto_parquet(
        base.data.crypto_parquet_root,
        base.data.start,
        base.data.end,
        base.data.crypto_primary_timeframe,
        base.data.crypto_micro_timeframe,
    )
    featured = add_crypto_features(raw)
    requested = [
        field for field in base.search.fields
        if field not in set(base.search.exclude_fields)
    ]
    panels, dates, codes = make_panels(featured, requested, value_col="close")
    families, _ = load_template_families(base.search.template_config)
    windows = set(base.search.windows)
    for family in families:
        windows.update(family.short_windows)
        windows.update(family.long_windows)
    evaluator = BatchEvaluator(
        panels={name: panel for name, panel in panels.items() if name != "close"},
        dates=dates,
        codes=codes,
        windows=tuple(sorted(windows)),
        max_depth=max(family.max_depth for family in families if family.enabled),
        max_nodes=max(family.max_nodes for family in families if family.enabled),
        max_ts_ops=max(family.max_ts_ops for family in families if family.enabled),
        max_pair_ops=max(
            family.max_pair_ops for family in families if family.enabled
        ),
        max_binary_ops=max(
            family.max_binary_ops for family in families if family.enabled
        ),
        max_cache_items=base.search.max_cache_items,
    )

    date_values = np.asarray(dates).astype(str)
    research_mask = date_values <= base.split.validation_end
    validation_mask = (
        (date_values >= base.split.validation_start)
        & (date_values <= base.split.validation_end)
    )
    chart_dates = pd.to_datetime(pd.Series(dates))
    sample_index = np.flatnonzero(research_mask)[::4]
    curves: dict[str, np.ndarray] = {}
    group_curves: dict[str, dict[str, np.ndarray]] = {}
    calculated_rows = []
    for _, row in selected.iterrows():
        values, status = evaluator.eval_expr(str(row["expr"]))
        if values is None or status != "OK":
            raise RuntimeError(f"Patch factor failed: {row['expr']} ({status})")
        direction = int(row["direction"])
        returns, turnover = crypto_long_short_returns(
            values,
            panels["close"],
            direction,
            base.evaluation.strategy_top_pct,
            base.evaluation.entry_lag,
            base.evaluation.strategy_cost_bps,
        )
        nav = _curve_from_returns(returns, research_mask)
        validation_metrics = _period_strategy_metrics(
            returns, validation_mask, base.evaluation.periods_per_year
        )
        expr_hash = str(row["expr_hash"])
        curves[expr_hash] = nav
        if int(row["return_rank"]) <= 5:
            group_curves[expr_hash] = _group_return_series(
                values * direction,
                panels["close"],
                base.evaluation.entry_lag,
            )
        calculated_rows.append({
            **row.to_dict(),
            "recomputed_research_total_return": float(nav[research_mask][-1] - 1.0),
            "recomputed_validation_total_return": validation_metrics["total_return"],
            "validation_return_difference": (
                validation_metrics["total_return"]
                - float(row[config.ranking_column])
            ),
            "recomputed_mean_turnover": float(np.nanmean(turnover[research_mask])),
        })
    calculated = pd.DataFrame(calculated_rows)
    calculated.to_csv(
        output_html.with_name("crypto_return_patch_top30.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    buckets = [calculated.iloc[:5], calculated.iloc[5:15], calculated.iloc[15:30]]
    cumulative = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "验证期累计收益排名 1–5",
            "验证期累计收益排名 6–15",
            "验证期累计收益排名 16–30",
        ),
    )
    for subplot_row, bucket in enumerate(buckets, start=1):
        for _, row in bucket.iterrows():
            label = f"R{int(row['return_rank']):02d}"
            cumulative.add_trace(go.Scatter(
                x=chart_dates.iloc[sample_index],
                y=(curves[str(row["expr_hash"])][sample_index] - 1.0) * 100.0,
                mode="lines",
                name=label,
                hovertemplate=(
                    f"{label}<br>{row['expr']}<br>"
                    "%{x|%Y-%m-%d %H:%M}<br>累计净收益=%{y:.2f}%<extra></extra>"
                ),
            ), row=subplot_row, col=1)
        cumulative.update_yaxes(
            title_text="累计净收益 (%)", row=subplot_row, col=1
        )
    cumulative.update_xaxes(title_text="UTC 时间", row=3, col=1)
    cumulative.update_layout(
        title="累计收益 Top30 因子：发现期 + 验证期净值曲线",
        template="plotly_white",
        hovermode="x unified",
        height=1500,
    )

    top5 = calculated.iloc[:5]
    grouped = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=tuple(
            f"R{int(row.return_rank):02d} 分组累计收益"
            for row in top5.itertuples(index=False)
        ),
    )
    colors = {"Q1": "#d62728", "Q2": "#ff9896", "Q3": "#9ecae1", "Q4": "#1f77b4"}
    for subplot_row, row in enumerate(top5.itertuples(index=False), start=1):
        expr_hash = str(row.expr_hash)
        for group_name, group_returns in group_curves[expr_hash].items():
            nav = _curve_from_returns(group_returns, research_mask)
            grouped.add_trace(go.Scatter(
                x=chart_dates.iloc[sample_index],
                y=(nav[sample_index] - 1.0) * 100.0,
                mode="lines",
                name=group_name,
                line={"color": colors[group_name]},
                legendgroup=group_name,
                showlegend=subplot_row == 1,
            ), row=subplot_row, col=1)
        grouped.update_yaxes(
            title_text="累计收益 (%)", row=subplot_row, col=1
        )
    grouped.update_xaxes(title_text="UTC 时间", row=5, col=1)
    grouped.update_layout(
        title="累计收益前5因子：Q1–Q4 分组收益曲线",
        template="plotly_white",
        hovermode="x unified",
        height=1900,
    )

    table_rows = "".join(
        "<tr>"
        f"<td>R{int(row.return_rank):02d}</td>"
        f"<td><code>{row.expr}</code></td>"
        f"<td>{getattr(row, config.ranking_column):.2%}</td>"
        f"<td>{row.recomputed_research_total_return:.2%}</td>"
        f"<td>{row.recomputed_mean_turnover:.3f}</td>"
        "</tr>"
        for row in calculated.itertuples(index=False)
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Crypto cumulative-return Top30 patch</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#222}}
.note{{padding:12px;background:#f4f7fb;border-left:4px solid #1f77b4;line-height:1.6}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
th:nth-child(2),td:nth-child(2){{text-align:left;word-break:break-all}}code{{white-space:normal}}
</style></head><body><h1>加密货币累计收益 Top30 报告补丁</h1>
<div class="note">排名直接读取 <code>{config.source_csv}</code> 的
<code>{config.ranking_column}</code>；只重新计算前30个表达式。累计曲线覆盖发现期+验证期，
计 {base.evaluation.strategy_cost_bps:.1f} bps/单位换手；前5分组曲线为Q1–Q4资产组原始收益，不扣组内交易成本。
未读取留出期指标，也未改写原实验产物。</div>
{cumulative.to_html(full_html=False, include_plotlyjs='cdn')}
{grouped.to_html(full_html=False, include_plotlyjs=False)}
<h2>Top30 表达式</h2><table><thead><tr><th>排名</th><th>完整表达式</th>
<th>CSV验证累计收益</th><th>重算研究期累计净收益</th><th>平均换手</th></tr></thead>
<tbody>{table_rows}</tbody></table></body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return {
        "status": "COMPLETED",
        "source_csv": config.source_csv,
        "ranking_column": config.ranking_column,
        "calculated_factors": len(calculated),
        "output_html": str(output_html),
    }
