from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import fastops
from .config import EvaluationConfig, SplitConfig
from .evaluation import daily_rank_ic, forward_returns


def crypto_long_short_returns(
    factor: np.ndarray,
    close: np.ndarray,
    direction: int,
    top_pct: float,
    entry_lag: int,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hourly top-minus-bottom returns with turnover costs."""
    ranked = fastops.rank_cs(np.asarray(factor, dtype=float) * direction)
    long_mask = ranked > (1.0 - top_pct)
    short_mask = ranked <= top_pct
    long_count = long_mask.sum(axis=1, keepdims=True)
    short_count = short_mask.sum(axis=1, keepdims=True)
    valid = (long_count[:, 0] > 0) & (short_count[:, 0] > 0)
    weights = np.zeros_like(ranked, dtype=float)
    np.divide(long_mask, long_count, out=weights, where=long_count > 0)
    short_weights = np.zeros_like(ranked, dtype=float)
    np.divide(short_mask, short_count, out=short_weights, where=short_count > 0)
    weights -= short_weights
    weights[~valid] = np.nan

    asset_returns = np.full_like(close, np.nan, dtype=float)
    asset_returns[1:] = close[1:] / close[:-1] - 1.0
    shift = entry_lag + 1
    positions = np.full_like(weights, np.nan, dtype=float)
    if shift < len(weights):
        positions[shift:] = weights[:-shift]
    portfolio = np.nansum(positions * asset_returns, axis=1)
    active = np.isfinite(positions).any(axis=1) & np.isfinite(asset_returns).any(axis=1)
    portfolio[~active] = np.nan
    filled = np.nan_to_num(positions, nan=0.0)
    turnover = np.full(len(positions), np.nan, dtype=float)
    turnover[1:] = np.abs(filled[1:] - filled[:-1]).sum(axis=1)
    portfolio = portfolio - turnover * (cost_bps / 10000.0)
    portfolio[~active] = np.nan
    return portfolio, turnover


def _period_strategy_metrics(
    returns: np.ndarray,
    mask: np.ndarray,
    periods_per_year: int,
) -> dict[str, float]:
    values = returns[mask]
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": np.nan, "sharpe": np.nan, "total_return": np.nan}
    std = float(values.std(ddof=0))
    total = float(np.prod(1.0 + values) - 1.0)
    return {
        "mean": float(values.mean()),
        "sharpe": (
            float(np.sqrt(periods_per_year) * values.mean() / std)
            if std > 1e-12 else np.nan
        ),
        "total_return": total,
    }


def crypto_search_metrics(
    factor: np.ndarray,
    close: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    direction: int,
    robustness_forward: dict[int, np.ndarray],
) -> dict[str, float]:
    date_values = np.asarray(dates).astype(str)
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    returns, turnover = crypto_long_short_returns(
        factor,
        close,
        direction,
        config.strategy_top_pct,
        config.entry_lag,
        config.strategy_cost_bps,
    )
    discovery = _period_strategy_metrics(
        returns, discovery_mask, config.periods_per_year
    )
    validation = _period_strategy_metrics(
        returns, validation_mask, config.periods_per_year
    )
    horizon_values = []
    for horizon in config.robustness_horizons:
        ic = daily_rank_ic(
            factor, robustness_forward[horizon], config.min_daily_names
        ) * direction
        value = ic[validation_mask]
        value = value[np.isfinite(value)]
        horizon_values.append(float(value.mean()) if len(value) else np.nan)
    finite_horizons = np.asarray(horizon_values, dtype=float)
    finite_horizons = finite_horizons[np.isfinite(finite_horizons)]
    research_turnover = turnover[date_values <= split.validation_end]
    return {
        "discovery_net_mean": discovery["mean"],
        "discovery_net_sharpe": discovery["sharpe"],
        "discovery_net_total_return": discovery["total_return"],
        "validation_net_mean": validation["mean"],
        "validation_net_sharpe": validation["sharpe"],
        "validation_net_total_return": validation["total_return"],
        "strategy_turnover": float(np.nanmean(research_turnover)),
        "horizon_worst_validation_ic": (
            float(finite_horizons.min()) if len(finite_horizons) else np.nan
        ),
        "horizon_mean_validation_ic": (
            float(finite_horizons.mean()) if len(finite_horizons) else np.nan
        ),
        "horizon_positive_share": (
            float(np.mean(finite_horizons > 0)) if len(finite_horizons) else 0.0
        ),
    }


def _curve_from_returns(returns: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clean = np.where(mask, np.nan_to_num(returns, nan=0.0), 0.0)
    return np.cumprod(1.0 + clean)


def _group_return_series(
    score: np.ndarray,
    close: np.ndarray,
    entry_lag: int,
) -> dict[str, np.ndarray]:
    ranked = fastops.rank_cs(score)
    asset_returns = np.full_like(close, np.nan, dtype=float)
    asset_returns[1:] = close[1:] / close[:-1] - 1.0
    shift = entry_lag + 1
    delayed = np.full_like(ranked, np.nan)
    if shift < len(ranked):
        delayed[shift:] = ranked[:-shift]
    groups = {
        "Q1": delayed <= 0.25,
        "Q2": (delayed > 0.25) & (delayed <= 0.50),
        "Q3": (delayed > 0.50) & (delayed <= 0.75),
        "Q4": delayed > 0.75,
    }
    output = {}
    for name, membership in groups.items():
        count = membership.sum(axis=1)
        values = np.nansum(np.where(membership, asset_returns, np.nan), axis=1)
        result = np.full(len(score), np.nan)
        np.divide(values, count, out=result, where=count > 0)
        output[name] = result
    return output


def build_crypto_reports(
    selected: pd.DataFrame,
    evaluator,
    close: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    out_dir: Path,
) -> dict[str, str]:
    if selected.empty:
        return {}
    date_values = np.asarray(dates).astype(str)
    research_mask = date_values <= split.validation_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    forward = forward_returns(close, config.horizon, config.entry_lag)
    rows = []
    curves = {}
    oriented_ranks = []
    group_heatmap = []
    for factor_rank, (_, row) in enumerate(selected.iterrows(), start=1):
        values, status = evaluator.eval_expr(row["expr"])
        if values is None or status != "OK":
            raise RuntimeError(f"Selected crypto factor failed: {row['expr']} ({status})")
        direction = int(row["direction"])
        returns, turnover = crypto_long_short_returns(
            values,
            close,
            direction,
            config.strategy_top_pct,
            config.entry_lag,
            config.strategy_cost_bps,
        )
        research = _period_strategy_metrics(
            returns, research_mask, config.periods_per_year
        )
        validation = _period_strategy_metrics(
            returns, validation_mask, config.periods_per_year
        )
        nav = _curve_from_returns(returns, research_mask)
        peak = np.maximum.accumulate(nav)
        drawdown = nav / np.where(peak > 0, peak, np.nan) - 1.0
        expr_hash = str(row["expr_hash"])
        curves[expr_hash] = nav
        rows.append({
            "factor_rank": factor_rank,
            "expr_hash": expr_hash,
            "expr": row["expr"],
            "template_family": row["template_family"],
            "research_score": row["research_score"],
            "research_total_return": research["total_return"],
            "research_net_sharpe": research["sharpe"],
            "validation_total_return": validation["total_return"],
            "validation_net_sharpe": validation["sharpe"],
            "mean_turnover": float(np.nanmean(turnover[research_mask])),
            "max_drawdown": float(np.nanmin(drawdown)),
        })
        oriented = fastops.rank_cs(values * direction)
        oriented_ranks.append(oriented)
        group_means = []
        for lower, upper in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
            membership = (oriented > lower) & (oriented <= upper)
            sample = forward[validation_mask]
            mask = membership[validation_mask] & np.isfinite(sample)
            group_means.append(float(np.nanmean(np.where(mask, sample, np.nan))) * 1e4)
        group_heatmap.append(group_means)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        out_dir / "crypto_selected_backtest_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chosen = metrics.sort_values(
        ["validation_net_sharpe", "research_score"], ascending=False
    ).head(10)
    chart = go.Figure()
    chart_dates = pd.to_datetime(pd.Series(dates))
    sample_index = np.flatnonzero(research_mask)[::4]
    for _, row in chosen.iterrows():
        chart.add_trace(go.Scatter(
            x=chart_dates.iloc[sample_index],
            y=(curves[row["expr_hash"]][sample_index] - 1.0) * 100.0,
            mode="lines",
            name=f"F{int(row['factor_rank']):02d}",
            hovertemplate=(
                f"F{int(row['factor_rank']):02d}<br>{row['expr']}<br>"
                "%{x|%Y-%m-%d %H:%M}<br>累计净收益=%{y:.2f}%<extra></extra>"
            ),
        ))
    chart.update_layout(
        title="加密货币小时截面因子：研究期净收益曲线（验证期 Sharpe 前10）",
        xaxis_title="UTC 时间",
        yaxis_title="累计净收益 (%)",
        template="plotly_white",
        hovermode="x unified",
        height=720,
    )
    table_rows = "".join(
        "<tr>"
        f"<td>F{int(row.factor_rank):02d}</td><td><code>{row.expr}</code></td>"
        f"<td>{row.validation_net_sharpe:.3f}</td>"
        f"<td>{row.research_total_return:.2%}</td>"
        f"<td>{row.max_drawdown:.2%}</td><td>{row.mean_turnover:.3f}</td>"
        "</tr>"
        for row in chosen.itertuples(index=False)
    )
    returns_html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Crypto factor returns</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#222}}
.note{{padding:12px;background:#f4f7fb;border-left:4px solid #1f77b4}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
th:nth-child(2),td:nth-child(2){{text-align:left;word-break:break-all}}code{{white-space:normal}}
</style></head><body><h1>20 个冻结候选：净收益曲线</h1>
<div class="note">仅按发现期与验证期选择和排序；成本={config.strategy_cost_bps:.1f} bps/单位换手，
Top {config.strategy_top_pct:.0%} − Bottom {config.strategy_top_pct:.0%}，信号延迟 {config.entry_lag} 小时执行。
不含资金费率、滑点和容量约束。</div>{chart.to_html(full_html=False, include_plotlyjs='cdn')}
<table><thead><tr><th>因子</th><th>完整表达式</th><th>验证净Sharpe</th><th>研究累计净收益</th><th>最大回撤</th><th>平均换手</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    (out_dir / "crypto_factor_returns.html").write_text(returns_html, encoding="utf-8")

    stacked_ranks = np.stack(oriented_ranks)
    rank_count = np.isfinite(stacked_ranks).sum(axis=0)
    ensemble = np.full(rank_count.shape, np.nan, dtype=float)
    np.divide(
        np.nansum(stacked_ranks, axis=0),
        rank_count,
        out=ensemble,
        where=rank_count > 0,
    )
    group_returns = _group_return_series(ensemble, close, config.entry_lag)
    group_figure = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.14,
        subplot_titles=(
            "验证期各因子四分组未来4小时收益（bps）",
            "20因子等权集成：研究期四分组累计收益",
        ),
        row_heights=[0.52, 0.48],
    )
    group_figure.add_trace(go.Heatmap(
        z=np.asarray(group_heatmap),
        x=["Q1 低", "Q2", "Q3", "Q4 高"],
        y=[f"F{i:02d}" for i in range(1, len(selected) + 1)],
        colorscale="RdBu",
        zmid=0,
        colorbar={"title": "bps"},
        hovertemplate="%{y} · %{x}<br>未来4小时=%{z:.3f} bps<extra></extra>",
    ), 1, 1)
    for group_name, returns in group_returns.items():
        nav = _curve_from_returns(returns, research_mask)
        group_figure.add_trace(go.Scatter(
            x=chart_dates.iloc[sample_index],
            y=(nav[sample_index] - 1.0) * 100.0,
            mode="lines",
            name=group_name,
        ), 2, 1)
    group_figure.update_yaxes(title_text="因子", row=1, col=1)
    group_figure.update_yaxes(title_text="累计收益 (%)", row=2, col=1)
    group_figure.update_xaxes(title_text="分组", row=1, col=1)
    group_figure.update_xaxes(title_text="UTC 时间", row=2, col=1)
    group_figure.update_layout(
        title="加密货币截面因子分组单调性",
        template="plotly_white",
        height=1050,
    )
    group_html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Crypto grouped returns</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#222}}.note{{padding:12px;background:#f4f7fb;border-left:4px solid #1f77b4}}</style></head>
<body><h1>分组收益与单调性</h1><div class="note">热力图仅使用验证期；集成曲线使用发现期+验证期。
Q4 是按发现期方向调整后的高分组。所有选择均在打开留出期前冻结。</div>
{group_figure.to_html(full_html=False, include_plotlyjs='cdn')}</body></html>"""
    (out_dir / "crypto_group_returns.html").write_text(group_html, encoding="utf-8")
    return {
        "crypto_factor_returns": "crypto_factor_returns.html",
        "crypto_group_returns": "crypto_group_returns.html",
        "crypto_backtest_metrics": "crypto_selected_backtest_metrics.csv",
    }
