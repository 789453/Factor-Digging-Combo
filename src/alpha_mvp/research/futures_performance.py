from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _long_short_daily_returns(
    factor: np.ndarray,
    close: np.ndarray,
    direction: int,
    top_pct: float,
    entry_lag: int,
    report_mask: np.ndarray,
) -> np.ndarray:
    """Daily-rebalanced equal-weight top-minus-bottom futures portfolio.

    A signal from day ``t`` is ranked cross-sectionally, executed at the close
    of ``t + entry_lag``, and earns the following close-to-close return.  The
    report mask is applied to return dates, preventing holdout observations
    from entering cumulative return, Sharpe, or display ranking.
    """
    if not 0 < top_pct < 0.5:
        raise ValueError("top_pct must be in (0, 0.5) for a long-short spread")
    if entry_lag < 0:
        raise ValueError("entry_lag cannot be negative")
    n_dates, n_names = factor.shape
    if close.shape != factor.shape:
        raise ValueError("factor and close must have the same panel shape")
    if len(report_mask) != n_dates:
        raise ValueError("report_mask length must match factor dates")
    returns = np.full(n_dates, np.nan, dtype=float)
    count = max(1, int(np.ceil(n_names * top_pct)))
    for signal_day in range(n_dates):
        execution_day = signal_day + entry_lag
        return_day = execution_day + 1
        if return_day >= n_dates or not report_mask[return_day]:
            continue
        valid = (
            np.isfinite(factor[signal_day])
            & np.isfinite(close[execution_day])
            & np.isfinite(close[return_day])
            & (close[execution_day] != 0.0)
        )
        indices = np.flatnonzero(valid)
        if len(indices) < 2 * count:
            continue
        scores = factor[signal_day, indices] * direction
        daily = close[return_day, indices] / close[execution_day, indices] - 1.0
        high = np.argpartition(-scores, count - 1)[:count]
        low = np.argpartition(scores, count - 1)[:count]
        returns[return_day] = float(daily[high].mean() - daily[low].mean())
    return returns


def _curve_metrics(returns: np.ndarray) -> dict:
    valid = np.isfinite(returns)
    clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    nav = np.cumprod(1.0 + clean)
    observed = clean[valid]
    std = float(observed.std(ddof=0)) if len(observed) else np.nan
    sharpe = (
        float(np.sqrt(252.0) * observed.mean() / std)
        if len(observed) and std > 1e-12
        else np.nan
    )
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.where(peak > 0, peak, np.nan) - 1.0
    annualized = (
        float(nav[-1] ** (252.0 / max(len(observed), 1)) - 1.0)
        if nav[-1] > 0.0
        else np.nan
    )
    max_drawdown = (
        float(np.nanmin(drawdown)) if np.isfinite(drawdown).any() else np.nan
    )
    return {
        "nav": nav,
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_return_days": int(len(observed)),
    }


def build_futures_factor_return_report(
    selected: pd.DataFrame,
    evaluator,
    close: np.ndarray,
    dates: list[str],
    research_end: str,
    entry_lag: int,
    out_dir: Path,
    top_pct: float = 0.30,
    top_each: int = 5,
) -> dict[str, str]:
    """Create a research-only long-short curve report for frozen candidates."""
    if selected.empty:
        return {}
    date_values = np.asarray(dates).astype(str)
    report_mask = date_values <= research_end
    curves: dict[str, dict] = {}
    rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        values, status = evaluator.eval_expr(row["expr"])
        if values is None or status != "OK":
            raise RuntimeError(f"Selected futures factor cannot be evaluated: {row['expr']} ({status})")
        returns = _long_short_daily_returns(
            values,
            close,
            int(row["direction"]),
            top_pct,
            entry_lag,
            report_mask,
        )
        metrics = _curve_metrics(returns)
        expr_hash = str(row["expr_hash"])
        curves[expr_hash] = metrics
        rows.append({
            "factor_rank": rank,
            "expr_hash": expr_hash,
            "expr": row["expr"],
            "direction": int(row["direction"]),
            "template_family": row["template_family"],
            "research_score": float(row["research_score"]),
            "top_pct": top_pct,
            **{key: value for key, value in metrics.items() if key != "nav"},
        })
    metrics_frame = pd.DataFrame(rows)
    metrics_frame["return_rank"] = metrics_frame["total_return"].rank(
        ascending=False, method="min"
    )
    metrics_frame["sharpe_rank"] = metrics_frame["sharpe"].rank(
        ascending=False, method="min"
    )
    metrics_path = out_dir / "futures_factor_returns.csv"
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    top_return = metrics_frame.nlargest(top_each, "total_return")
    top_sharpe = metrics_frame.nlargest(top_each, "sharpe")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"累计收益最高的 {len(top_return)} 个因子",
            f"全期 Sharpe 最高的 {len(top_sharpe)} 个因子",
        ),
    )
    date_axis = pd.to_datetime(pd.Series(dates))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]
    for plot_row, candidates in ((1, top_return), (2, top_sharpe)):
        for color, (_, row) in enumerate(candidates.iterrows()):
            curve = curves[str(row["expr_hash"])]
            figure.add_trace(go.Scatter(
                x=date_axis[report_mask],
                y=(curve["nav"][report_mask] - 1.0) * 100.0,
                mode="lines",
                name=f"F{int(row['factor_rank']):02d}",
                line={"color": palette[color % len(palette)], "width": 1.8},
                hovertemplate=(
                    f"F{int(row['factor_rank']):02d}<br>{row['expr']}<br>"
                    "日期=%{x|%Y-%m-%d}<br>累计收益=%{y:.2f}%<extra></extra>"
                ),
            ), plot_row, 1)
        figure.update_yaxes(title_text="累计收益 (%)", row=plot_row, col=1)
    figure.update_layout(
        title=(
            "期货截面因子：发现期 + 验证期的日调仓多空组合 "
            f"（Top {top_pct:.0%} − Bottom {top_pct:.0%}）"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=900,
        legend={"orientation": "h", "y": 1.03},
    )
    html_path = out_dir / "futures_factor_returns.html"
    table = pd.concat([top_return.assign(selection="累计收益 Top"), top_sharpe.assign(selection="Sharpe Top")])
    table = table.drop_duplicates("expr_hash").sort_values(["return_rank", "sharpe_rank"])
    rows_html = "".join(
        "<tr>"
        f"<td>{row.selection}</td><td>F{int(row.factor_rank):02d}</td>"
        f"<td><code>{row.expr}</code></td><td>{row.direction:+d}</td>"
        f"<td>{row.total_return:.2%}</td><td>{row.annualized_return:.2%}</td>"
        f"<td>{row.sharpe:.3f}</td><td>{row.max_drawdown:.2%}</td>"
        f"<td>{int(row.n_return_days)}</td>"
        "</tr>"
        for row in table.itertuples(index=False)
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>期货因子累计收益</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#222}}
.note{{background:#f4f7fb;padding:12px;border-left:4px solid #1f77b4}}
table{{border-collapse:collapse;width:100%;margin-top:18px;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px;text-align:right}}
th:nth-child(3),td:nth-child(3){{text-align:left;word-break:break-all}} code{{white-space:normal}}
</style></head><body><h1>冻结候选的累计收益与 Sharpe</h1>
<div class="note">仅使用发现期与验证期（截至 {research_end}）的收益；不读取留出期。
每日按发现期确定的方向做等权 Top {top_pct:.0%} − Bottom {top_pct:.0%} 多空，
收盘信号在 {entry_lag} 日后执行、下一日开始计收益；未计成本、保证金、展期和交易约束。</div>
{figure.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>曲线所含因子与完整表达式</h2><table><thead><tr><th>入选原因</th><th>因子</th><th>表达式</th><th>方向</th><th>累计收益</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>有效日</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return {
        "futures_factor_returns": html_path.name,
        "futures_factor_returns_csv": metrics_path.name,
    }
