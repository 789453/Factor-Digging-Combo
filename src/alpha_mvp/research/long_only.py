from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots

from ..data import load_from_duckdb
from ..evaluator import BatchEvaluator, make_panels
from ..fields import add_selected_features


FULL_MARKET_RAW_COLUMNS = [
    "open", "high", "low", "close", "pre_close", "vol", "amount",
    "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
    "buy_lg_amount", "sell_lg_amount", "net_mf_amount",
    "cost_5pct", "cost_50pct", "cost_95pct", "his_low", "his_high",
    "turnover_rate", "volume_ratio",
]


def load_long_only_config(path: str) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {"data", "selected_factors", "backtest", "output"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Long-only config missing sections: {sorted(missing)}")
    return raw


def _portfolio_returns(
    factor: np.ndarray,
    stock_returns: np.ndarray,
    close: np.ndarray,
    rebalance_days: int,
    top_pcts: list[float],
    entry_lag: int,
) -> dict[float, np.ndarray]:
    n_dates = factor.shape[0]
    output = {pct: np.zeros(n_dates, dtype=float) for pct in top_pcts}
    for signal_day in range(0, n_dates, rebalance_days):
        execution_day = signal_day + entry_lag
        if execution_day >= n_dates - 1:
            break
        next_execution = min(execution_day + rebalance_days, n_dates - 1)
        start = execution_day + 1
        end = next_execution + 1
        valid = np.isfinite(factor[signal_day]) & np.isfinite(close[execution_day])
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        scores = factor[signal_day, valid_indices]
        period_returns = np.nan_to_num(
            stock_returns[start:end],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for pct in top_pcts:
            count = max(1, int(math.ceil(len(valid_indices) * pct)))
            local = np.argpartition(-scores, count - 1)[:count]
            held = valid_indices[local]
            output[pct][start:end] = period_returns[:, held].mean(axis=1)
    return output


def _curve_metrics(returns: np.ndarray, rolling_window: int) -> dict:
    clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    nav = np.cumprod(1.0 + clean)
    series = pd.Series(clean)
    min_periods = min(rolling_window, max(2, rolling_window // 2))
    rolling_mean = series.rolling(
        rolling_window,
        min_periods=min_periods,
    ).mean()
    rolling_std = series.rolling(
        rolling_window,
        min_periods=min_periods,
    ).std(ddof=0)
    rolling_sharpe = (
        np.sqrt(252.0) * rolling_mean / rolling_std.replace(0.0, np.nan)
    ).to_numpy()
    std = float(np.std(clean, ddof=0))
    sharpe = float(np.sqrt(252.0) * np.mean(clean) / std) if std > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.where(peak > 0, peak, np.nan) - 1.0
    return {
        "returns": clean,
        "nav": nav,
        "rolling_sharpe": rolling_sharpe,
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": float(nav[-1] ** (252.0 / len(nav)) - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(np.nanmin(drawdown)),
    }


def _build_html(
    dates: list[str],
    metrics: pd.DataFrame,
    curves: dict[tuple[int, float, str], dict],
    rebalance_days: list[int],
    top_pcts: list[float],
    top_curves: int,
    output_path: Path,
) -> None:
    date_values = pd.to_datetime(pd.Series(dates))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf",
    ]
    blocks = []
    buttons = []
    include_js = True
    for condition_index, (days, pct) in enumerate(
        (item for d in rebalance_days for item in [(d, p) for p in top_pcts])
    ):
        condition = metrics[
            (metrics["rebalance_days"] == days)
            & np.isclose(metrics["top_pct"], pct)
        ].copy()
        condition["return_rank"] = condition["total_return"].rank(
            ascending=False,
            method="min",
        )
        condition["sharpe_rank"] = condition["sharpe"].rank(
            ascending=False,
            method="min",
        )
        condition["combined_rank"] = (
            condition["return_rank"] + condition["sharpe_rank"]
        )
        chosen = condition.sort_values(
            ["combined_rank", "return_rank", "sharpe_rank"]
        ).head(top_curves)
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(
                "累计收益",
                "252日滚动年化 Sharpe",
            ),
        )
        for color_index, (_, row) in enumerate(chosen.iterrows()):
            key = (days, float(pct), row["expr_hash"])
            curve = curves[key]
            label = f"F{int(row['factor_rank']):02d}"
            expression = str(row["expr"])
            hover = (
                f"{label}<br>{expression}<br>"
                "日期=%{x|%Y-%m-%d}<br>累计收益=%{y:.2f}%<extra></extra>"
            )
            figure.add_trace(go.Scatter(
                x=date_values,
                y=(curve["nav"] - 1.0) * 100.0,
                mode="lines",
                name=label,
                line={"color": palette[color_index % len(palette)], "width": 1.7},
                hovertemplate=hover,
                legendgroup=label,
            ), 1, 1)
            figure.add_trace(go.Scatter(
                x=date_values,
                y=curve["rolling_sharpe"],
                mode="lines",
                name=label,
                showlegend=False,
                line={"color": palette[color_index % len(palette)], "width": 1.5},
                hovertemplate=(
                    f"{label}<br>{expression}<br>"
                    "日期=%{x|%Y-%m-%d}<br>滚动Sharpe=%{y:.3f}<extra></extra>"
                ),
                legendgroup=label,
            ), 2, 1)
        figure.update_layout(
            title=f"持有/调仓 {days} 日 · Top {pct:.0%} · 综合排名前 {len(chosen)}",
            height=820,
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.04},
            margin={"l": 60, "r": 30, "t": 110, "b": 50},
        )
        figure.update_yaxes(title_text="累计收益 (%)", row=1, col=1)
        figure.update_yaxes(title_text="Rolling Sharpe", row=2, col=1)
        div_id = f"condition-{days}-{int(round(pct * 100))}"
        plot = pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=True if include_js else False,
            div_id=f"plot-{div_id}",
        )
        include_js = False
        table_rows = []
        for _, row in chosen.iterrows():
            table_rows.append(
                "<tr>"
                f"<td>F{int(row['factor_rank']):02d}</td>"
                f"<td><code>{row['expr']}</code></td>"
                f"<td>{row['direction']:+.0f}</td>"
                f"<td>{row['total_return']:.2%}</td>"
                f"<td>{row['annualized_return']:.2%}</td>"
                f"<td>{row['sharpe']:.3f}</td>"
                f"<td>{row['max_drawdown']:.2%}</td>"
                "</tr>"
            )
        display = "block" if condition_index == 0 else "none"
        blocks.append(
            f'<section id="{div_id}" class="condition" style="display:{display}">'
            f"{plot}<table><thead><tr><th>因子</th><th>完整表达式</th>"
            "<th>方向</th><th>累计收益</th><th>年化收益</th>"
            "<th>Sharpe</th><th>最大回撤</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table></section>"
        )
        buttons.append(
            f'<button onclick="showCondition(\'{div_id}\',this)"'
            f' class="{"active" if condition_index == 0 else ""}">'
            f"{days}日 / Top {pct:.0%}</button>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Top50 全市场纯多头审计</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:18px;color:#222}}
.toolbar{{position:sticky;top:0;background:white;padding:10px 0;z-index:20}}
button{{padding:8px 12px;margin:3px;border:1px solid #bbb;background:#f7f7f7;cursor:pointer}}
button.active{{background:#1f77b4;color:white;border-color:#1f77b4}}
table{{border-collapse:collapse;width:100%;margin:8px 0 30px;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px;text-align:right}}
th:nth-child(2),td:nth-child(2){{text-align:left;max-width:760px;word-break:break-all}}
code{{white-space:normal}}
.note{{background:#f4f7fb;padding:12px;border-left:4px solid #1f77b4}}
</style></head><body>
<h1>冻结 Top50：全市场 A 股纯多头审计</h1>
<div class="note">区间：2018-01-01 至 2026-04-23。信号收盘后形成，
下一交易日收盘执行，之后开始计收益；等权纯多头、无交易成本。
每个条件按累计收益排名与全期 Sharpe 排名之和选出前 {top_curves}。
滚动 Sharpe 使用 252 个交易日窗口。</div>
<div class="toolbar">{''.join(buttons)}</div>
{''.join(blocks)}
<script>
function showCondition(id,button){{
 document.querySelectorAll('.condition').forEach(x=>x.style.display='none');
 document.querySelectorAll('.toolbar button').forEach(x=>x.classList.remove('active'));
 document.getElementById(id).style.display='block'; button.classList.add('active');
 window.dispatchEvent(new Event('resize'));
}}
</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def run_long_only_report(config: dict) -> dict:
    data_cfg = config["data"]
    backtest_cfg = config["backtest"]
    output_cfg = config["output"]
    out_dir = Path(output_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / output_cfg.get(
        "html_name",
        "top50_full_market_long_only.html",
    )
    if html_path.exists():
        raise FileExistsError(f"Long-only report already exists: {html_path}")

    selected = pd.read_csv(config["selected_factors"]).head(
        int(config.get("top_n", 50))
    )
    selected = selected.reset_index(drop=True)
    selected["factor_rank"] = np.arange(1, len(selected) + 1)
    fields = sorted({
        field
        for value in selected["fields"].fillna("")
        for field in value.split("|")
        if field
    })
    started = time.perf_counter()
    print(
        f"[long-only] loading full market: {data_cfg['start']}..{data_cfg['end']} "
        f"| factors={len(selected)} fields={len(fields)}",
        flush=True,
    )
    raw = load_from_duckdb(
        data_cfg["duckdb_path"],
        None,
        str(data_cfg["start"]),
        str(data_cfg["end"]),
        columns=FULL_MARKET_RAW_COLUMNS,
    )
    print(
        f"[long-only] loaded rows={len(raw):,} stocks={raw.ts_code.nunique():,} "
        f"| {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    featured = add_selected_features(raw, fields)
    del raw
    panels, dates, codes = make_panels(featured, fields, value_col="close")
    del featured
    close = panels["close"]
    stock_returns = np.full_like(close, np.nan, dtype=float)
    stock_returns[1:] = close[1:] / close[:-1] - 1.0

    windows = sorted({
        int(window)
        for value in selected["windows"].fillna("")
        for window in str(value).split("|")
        if window
    })
    evaluator = BatchEvaluator(
        panels={name: panel for name, panel in panels.items() if name != "close"},
        dates=dates,
        codes=codes,
        windows=windows,
        max_depth=10,
        max_nodes=36,
        max_ts_ops=8,
        max_pair_ops=3,
        max_binary_ops=8,
        max_cache_items=int(backtest_cfg.get("max_cache_items", 6)),
    )
    rebalance_days = [int(x) for x in backtest_cfg["rebalance_days"]]
    top_pcts = [float(x) for x in backtest_cfg["top_pcts"]]
    entry_lag = int(backtest_cfg.get("entry_lag", 1))
    rolling_window = int(backtest_cfg.get("rolling_sharpe_window", 252))
    curves: dict[tuple[int, float, str], dict] = {}
    rows = []
    for index, row in selected.iterrows():
        values, status = evaluator.eval_expr(row["expr"])
        if values is None or status != "OK":
            raise RuntimeError(f"Factor evaluation failed: {row['expr']} ({status})")
        oriented = values * int(row["direction"])
        for days in rebalance_days:
            portfolios = _portfolio_returns(
                oriented,
                stock_returns,
                close,
                days,
                top_pcts,
                entry_lag,
            )
            for pct, returns in portfolios.items():
                curve = _curve_metrics(returns, rolling_window)
                key = (days, float(pct), row["expr_hash"])
                curves[key] = curve
                rows.append({
                    "factor_rank": int(row["factor_rank"]),
                    "expr_hash": row["expr_hash"],
                    "expr": row["expr"],
                    "direction": int(row["direction"]),
                    "rebalance_days": days,
                    "top_pct": pct,
                    "total_return": curve["total_return"],
                    "annualized_return": curve["annualized_return"],
                    "sharpe": curve["sharpe"],
                    "max_drawdown": curve["max_drawdown"],
                })
        del oriented, values
        print(
            f"[long-only] factor {index + 1}/{len(selected)} | "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / output_cfg.get(
        "metrics_name",
        "top50_full_market_long_only_metrics.csv",
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    _build_html(
        dates,
        metrics,
        curves,
        rebalance_days,
        top_pcts,
        int(backtest_cfg.get("top_curves", 8)),
        html_path,
    )
    manifest = {
        "status": "COMPLETED",
        "selected_factors": str(Path(config["selected_factors"]).resolve()),
        "rows": int(len(metrics)),
        "market_rows": int(len(dates) * len(codes)),
        "dates": len(dates),
        "stocks": len(codes),
        "date_start": dates[0],
        "date_end": dates[-1],
        "rebalance_days": rebalance_days,
        "top_pcts": top_pcts,
        "entry_lag": entry_lag,
        "rolling_sharpe_window": rolling_window,
        "html": html_path.name,
        "metrics": metrics_path.name,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
