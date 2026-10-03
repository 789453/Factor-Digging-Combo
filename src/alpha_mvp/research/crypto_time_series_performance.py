from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import fastops
from .config import EvaluationConfig, SplitConfig
from .crypto_performance import _curve_from_returns, _period_strategy_metrics
from .time_series_evaluation import asset_time_ic


def crypto_time_series_returns(
    factor: np.ndarray,
    close: np.ndarray,
    direction: int,
    entry_lag: int,
    cost_bps: float,
    normalization_window: int,
    position_clip: float,
    holding_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed positions formed from overlapping horizon-matched tranches."""
    zscore = fastops.fast_rolling_zscore(
        np.asarray(factor, dtype=float), normalization_window
    )
    target = np.clip(zscore * direction, -position_clip, position_clip)
    target = target / position_clip / target.shape[1]
    if holding_period > 1:
        target = fastops.fast_rolling_mean(target, holding_period)
    asset_returns = np.full_like(close, np.nan, dtype=float)
    asset_returns[1:] = close[1:] / close[:-1] - 1.0
    shift = entry_lag + 1
    positions = np.full_like(target, np.nan, dtype=float)
    if shift < len(target):
        positions[shift:] = target[:-shift]
    portfolio = np.nansum(positions * asset_returns, axis=1)
    active = np.isfinite(positions).any(axis=1) & np.isfinite(asset_returns).any(axis=1)
    portfolio[~active] = np.nan
    filled = np.nan_to_num(positions, nan=0.0)
    turnover = np.full(len(positions), np.nan, dtype=float)
    turnover[1:] = np.abs(filled[1:] - filled[:-1]).sum(axis=1)
    portfolio -= turnover * (cost_bps / 10000.0)
    portfolio[~active] = np.nan
    return portfolio, turnover


def crypto_time_series_search_metrics(
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
    returns, turnover = crypto_time_series_returns(
        factor,
        close,
        direction,
        config.entry_lag,
        config.strategy_cost_bps,
        config.time_series_position_window,
        config.time_series_position_clip,
        config.horizon,
    )
    discovery = _period_strategy_metrics(returns, discovery_mask, config.periods_per_year)
    validation = _period_strategy_metrics(returns, validation_mask, config.periods_per_year)
    horizon_values = []
    for horizon in config.robustness_horizons:
        per_asset = asset_time_ic(
            factor, robustness_forward[horizon], validation_mask
        ) * direction
        finite = per_asset[np.isfinite(per_asset)]
        horizon_values.append(float(finite.mean()) if len(finite) else np.nan)
    finite_horizons = np.asarray(horizon_values, dtype=float)
    finite_horizons = finite_horizons[np.isfinite(finite_horizons)]
    research_turnover = turnover[date_values <= split.validation_end]
    parsed = pd.to_datetime(pd.Series(dates), format="%Y%m%d%H%M", utc=True)
    ny_hour = parsed.dt.tz_convert("America/New_York").dt.hour.to_numpy()
    session_masks = {
        "us_overnight": validation_mask & (ny_hour < 8),
        "us_day": validation_mask & (ny_hour >= 8) & (ny_hour < 16),
        "us_evening": validation_mask & (ny_hour >= 16),
    }
    session_metrics = {
        name: _period_strategy_metrics(returns, mask, config.periods_per_year / 3.0)
        for name, mask in session_masks.items()
    }
    output = {
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
    for name, metrics in session_metrics.items():
        output[f"validation_{name}_mean"] = metrics["mean"]
        output[f"validation_{name}_sharpe"] = metrics["sharpe"]
        output[f"validation_{name}_total_return"] = metrics["total_return"]
    finite_session_sharpes = np.asarray(
        [metrics["sharpe"] for metrics in session_metrics.values()], dtype=float
    )
    finite_session_sharpes = finite_session_sharpes[np.isfinite(finite_session_sharpes)]
    output["validation_worst_session_sharpe"] = (
        float(finite_session_sharpes.min()) if len(finite_session_sharpes) else np.nan
    )
    return output


def _time_signal_groups(
    factor: np.ndarray,
    close: np.ndarray,
    direction: int,
    config: EvaluationConfig,
) -> dict[str, np.ndarray]:
    zscore = fastops.fast_rolling_zscore(
        factor * direction, config.time_series_position_window
    )
    shift = config.entry_lag + 1
    delayed = np.full_like(zscore, np.nan)
    if shift < len(zscore):
        delayed[shift:] = zscore[:-shift]
    asset_returns = np.full_like(close, np.nan)
    asset_returns[1:] = close[1:] / close[:-1] - 1.0
    groups = {
        "Q1 <= -0.67": delayed <= -0.674,
        "Q2 -0.67~0": (delayed > -0.674) & (delayed <= 0.0),
        "Q3 0~0.67": (delayed > 0.0) & (delayed <= 0.674),
        "Q4 > 0.67": delayed > 0.674,
    }
    output = {}
    for name, membership in groups.items():
        count = membership.sum(axis=1)
        values = np.nansum(np.where(membership, asset_returns, np.nan), axis=1)
        result = np.full(len(factor), np.nan)
        np.divide(values, count, out=result, where=count > 0)
        output[name] = result
    return output


def build_crypto_time_series_reports(
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
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    chart_dates = pd.to_datetime(pd.Series(dates))
    rows, validation_curves, discovery_curves, factor_values = [], {}, {}, {}
    for factor_rank, (_, row) in enumerate(selected.iterrows(), start=1):
        values, status = evaluator.eval_expr(row["expr"])
        if values is None or status != "OK":
            raise RuntimeError(f"Selected time-series factor failed: {status}")
        returns, turnover = crypto_time_series_returns(
            values, close, int(row["direction"]), config.entry_lag,
            config.strategy_cost_bps, config.time_series_position_window,
            config.time_series_position_clip, config.horizon,
        )
        discovery = _period_strategy_metrics(returns, discovery_mask, config.periods_per_year)
        validation = _period_strategy_metrics(returns, validation_mask, config.periods_per_year)
        expr_hash = str(row["expr_hash"])
        factor_values[expr_hash] = values
        validation_curves[expr_hash] = _curve_from_returns(returns, validation_mask)
        discovery_curves[expr_hash] = _curve_from_returns(returns, discovery_mask)
        rows.append({
            "factor_rank": factor_rank,
            "expr_hash": expr_hash,
            "expr": row["expr"],
            "template_name": row.get("template_name", ""),
            "research_score": row["research_score"],
            "discovery_total_return": discovery["total_return"],
            "discovery_net_sharpe": discovery["sharpe"],
            "validation_total_return": validation["total_return"],
            "validation_net_sharpe": validation["sharpe"],
            "mean_turnover": float(np.nanmean(turnover[date_values <= split.validation_end])),
            "us_overnight_sharpe": row.get("validation_us_overnight_sharpe", np.nan),
            "us_day_sharpe": row.get("validation_us_day_sharpe", np.nan),
            "us_evening_sharpe": row.get("validation_us_evening_sharpe", np.nan),
        })
    metrics = pd.DataFrame(rows).sort_values(
        ["validation_total_return", "research_score"], ascending=False
    ).reset_index(drop=True)
    metrics["return_rank"] = np.arange(1, len(metrics) + 1)
    metrics.to_csv(
        out_dir / "crypto_time_series_backtest_metrics.csv",
        index=False, encoding="utf-8-sig",
    )

    buckets = [metrics.iloc[:5], metrics.iloc[5:15], metrics.iloc[15:20]]
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("验证累计收益 1–5", "验证累计收益 6–15", "验证累计收益 16–20"),
    )
    sample = np.flatnonzero(validation_mask)[::4]
    for plot_row, bucket in enumerate(buckets, start=1):
        for item in bucket.itertuples(index=False):
            figure.add_trace(go.Scatter(
                x=chart_dates.iloc[sample],
                y=(validation_curves[item.expr_hash][sample] - 1.0) * 100.0,
                mode="lines", name=f"R{int(item.return_rank):02d}",
                hovertemplate=f"{item.expr}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
            ), row=plot_row, col=1)
        figure.update_yaxes(title_text="验证累计净收益 (%)", row=plot_row, col=1)
    figure.update_layout(
        title="加密货币时序因子：验证期累计净收益（排序与曲线同区间）",
        template="plotly_white", hovermode="x unified", height=1450,
    )

    top5 = metrics.iloc[:5]
    direction_by_hash = selected.set_index("expr_hash")["direction"].to_dict()
    group_curves = {
        str(item.expr_hash): _time_signal_groups(
            factor_values[str(item.expr_hash)],
            close,
            int(direction_by_hash[str(item.expr_hash)]),
            config,
        )
        for item in top5.itertuples(index=False)
    }
    groups = make_subplots(
        rows=len(top5), cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=tuple(f"R{int(x.return_rank):02d} 时序信号状态分组" for x in top5.itertuples()),
    )
    for plot_row, item in enumerate(top5.itertuples(index=False), start=1):
        for name, returns in group_curves[item.expr_hash].items():
            nav = _curve_from_returns(returns, validation_mask)
            groups.add_trace(go.Scatter(
                x=chart_dates.iloc[sample], y=(nav[sample] - 1.0) * 100.0,
                mode="lines", name=name, showlegend=plot_row == 1,
                legendgroup=name,
            ), row=plot_row, col=1)
    groups.update_layout(
        title="验证期：累计收益前5因子的时序状态分组",
        template="plotly_white", hovermode="x unified", height=1800,
    )

    session_heatmap = go.Figure(go.Heatmap(
        z=metrics[["us_overnight_sharpe", "us_day_sharpe", "us_evening_sharpe"]].to_numpy(),
        x=["纽约夜间 00–08", "美国白天 08–16", "纽约晚间 16–24"],
        y=[f"R{int(value):02d}" for value in metrics["return_rank"]],
        colorscale="RdBu", zmid=0.0, colorbar_title="Sharpe",
        hovertemplate="%{y}<br>%{x}<br>Sharpe=%{z:.3f}<extra></extra>",
    ))
    session_heatmap.update_layout(
        title="验证期分时段 Sharpe（纽约本地时间，DST-aware）",
        template="plotly_white", height=700,
    )

    table_rows = "".join(
        f"<tr><td>R{int(x.return_rank):02d}</td><td><code>{x.expr}</code></td>"
        f"<td>{x.validation_total_return:.2%}</td><td>{x.validation_net_sharpe:.3f}</td>"
        f"<td>{x.discovery_total_return:.2%}</td><td>{x.discovery_net_sharpe:.3f}</td>"
        f"<td>{x.mean_turnover:.3f}</td></tr>" for x in metrics.itertuples(index=False)
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Crypto time-series factors</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px}}.note{{padding:12px;background:#f4f7fb;border-left:4px solid #1f77b4}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left;word-break:break-all}}</style></head>
<body><h1>加密货币时序因子报告</h1><div class="note">每个币种按自身历史滚动标准化，不使用截面Rank。策略是 signed long-short：正信号做多、负信号做空，不是纯多头；允许组合随状态呈现净多或净空。验证收益排序与曲线均严格使用验证期；发现期指标单列。成本={config.strategy_cost_bps:.1f} bps/单位换手。Q1–Q4仅用于信号状态诊断，不代表实际组合。</div>
{figure.to_html(full_html=False, include_plotlyjs='cdn')}
{groups.to_html(full_html=False, include_plotlyjs=False)}
{session_heatmap.to_html(full_html=False, include_plotlyjs=False)}
<table><thead><tr><th>收益排名</th><th>表达式</th><th>验证累计收益</th><th>验证Sharpe</th><th>发现累计收益</th><th>发现Sharpe</th><th>换手</th></tr></thead><tbody>{table_rows}</tbody></table></body></html>"""
    (out_dir / "crypto_time_series_factor_returns.html").write_text(html, encoding="utf-8")
    return {
        "crypto_time_series_returns": "crypto_time_series_factor_returns.html",
        "crypto_time_series_metrics": "crypto_time_series_backtest_metrics.csv",
    }


def build_crypto_pair_report(
    pairs: pd.DataFrame,
    selected: pd.DataFrame,
    factor_panels: dict[str, np.ndarray],
    close: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    out_dir: Path,
) -> dict[str, str]:
    """Visualize selected low-similarity pairs as equal-signal signed portfolios."""
    if pairs.empty:
        return {}
    directions = selected.set_index("expr_hash")["direction"].astype(int).to_dict()
    date_values = np.asarray(dates).astype(str)
    validation = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    chart_dates = pd.to_datetime(pd.Series(dates))
    sample = np.flatnonzero(validation)[::4]
    figure = go.Figure()
    rows = []
    for rank, pair in enumerate(pairs.itertuples(index=False), start=1):
        left = factor_panels[str(pair.left_expr_hash)] * directions[str(pair.left_expr_hash)]
        right = factor_panels[str(pair.right_expr_hash)] * directions[str(pair.right_expr_hash)]
        composite = 0.5 * (left + right)
        returns, turnover = crypto_time_series_returns(
            composite, close, 1, config.entry_lag, config.strategy_cost_bps,
            config.time_series_position_window, config.time_series_position_clip,
            config.horizon,
        )
        metrics = _period_strategy_metrics(returns, validation, config.periods_per_year)
        nav = _curve_from_returns(returns, validation)
        figure.add_trace(go.Scatter(
            x=chart_dates.iloc[sample], y=(nav[sample] - 1.0) * 100.0,
            mode="lines", name=f"P{rank:02d}",
            hovertemplate=(
                f"P{rank:02d}<br>{pair.left_expr}<br>+ {pair.right_expr}"
                "<br>%{x}<br>%{y:.2f}%<extra></extra>"
            ),
        ))
        rows.append({
            "pair_rank": rank, "left_expr_hash": pair.left_expr_hash,
            "right_expr_hash": pair.right_expr_hash, "left_expr": pair.left_expr,
            "right_expr": pair.right_expr, "similarity": pair.similarity,
            "validation_total_return": metrics["total_return"],
            "validation_sharpe": metrics["sharpe"],
            "mean_turnover": float(np.nanmean(turnover[validation])),
        })
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        out_dir / "crypto_time_series_pair_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    figure.update_layout(
        title="低相似度因子配对：验证期 signed long-short 累计净收益",
        template="plotly_white", hovermode="x unified", height=760,
        yaxis_title="累计净收益 (%)",
    )
    table = metrics_frame.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}")
    html = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>Crypto factor pairs</title></head><body>"
        "<h1>加密货币时序因子配对</h1><p>配对相似度按每个币种的时间序列相关性中位数计算；"
        "组合先按发现期确定的方向对齐，再等权合成。收益为含成本的 signed long-short。</p>"
        + figure.to_html(full_html=False, include_plotlyjs="cdn") + table + "</body></html>"
    )
    (out_dir / "crypto_time_series_factor_pairs.html").write_text(html, encoding="utf-8")
    return {
        "crypto_time_series_pairs": "crypto_time_series_factor_pairs.html",
        "crypto_time_series_pair_metrics": "crypto_time_series_pair_metrics.csv",
    }
