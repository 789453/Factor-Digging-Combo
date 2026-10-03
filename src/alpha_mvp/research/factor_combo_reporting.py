"""Auditable combo trade ledger, performance metrics and separated HTML reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .factor_combo_dynamic import DynamicSpec, _mean_asset, _strategy_event_band
from .factor_combo_signal_engine import SignalEngineSpec, SIGNAL_ENGINE_VERSION


REPORT_VERSION = "2026-09-29-combo-ledger-v4"
PRIMARY = "multiscale_trigger"
LEGACY_ORDER = (
    "mix_risk_dynamic", "mix_risk_5m_utility", "mix_no_risk",
    "alpha_only", "beta_only", "family_equal", "single_frozen",
    "elastic_net", "explicit_gate", "style_B0",
)


def _strategy_order(dynamic: dict) -> list[str]:
    profiles = [f"profile_{name}" for name in dynamic["signal_engine"]["profile_positions"]]
    return [PRIMARY, "multiscale_leverage_diagnostic", *profiles, *LEGACY_ORDER]


def _daily_sharpe(log_returns: np.ndarray, dates: np.ndarray) -> float:
    day = np.asarray([str(value)[:8] for value in dates])
    totals = pd.Series(log_returns).groupby(day).sum().to_numpy()
    if len(totals) < 3 or np.std(totals, ddof=1) < 1e-12:
        return np.nan
    return float(np.mean(totals) / np.std(totals, ddof=1) * np.sqrt(365))


def _drawdown(log_returns: np.ndarray) -> float:
    equity = np.exp(np.cumsum(log_returns))
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1))


def _session_names(dates: np.ndarray) -> np.ndarray:
    hour = pd.to_datetime(dates, format="%Y%m%d%H%M", utc=True).tz_convert(
        "America/New_York").hour.to_numpy()
    return np.where((hour >= 8) & (hour < 16), "NY_day",
                    np.where(hour >= 16, "NY_evening", "NY_overnight"))


def _equal_market_buyhold_log_returns(raw_y: np.ndarray,
                                      segment_lengths: list[int]) -> np.ndarray:
    """Equal starting capital per asset, with no rebalance inside a segment."""
    if sum(segment_lengths) != len(raw_y):
        raise ValueError("market benchmark segment lengths do not align")
    result = np.empty(len(raw_y), dtype=float)
    offset = 0
    for length in segment_lengths:
        path = np.exp(np.cumsum(np.nan_to_num(raw_y[offset:offset + length], nan=0.0), axis=0))
        basket = path.mean(axis=1)
        result[offset:offset + length] = np.diff(np.log(np.r_[1.0, basket]))
        offset += length
    return result


def build_ledgers(dynamic: dict, close: np.ndarray, raw_y: np.ndarray,
                  dates: np.ndarray, periods: np.ndarray, codes: list[str],
                  segment_lengths: list[int], spec: DynamicSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    ntime, nasset = raw_y.shape
    if close.shape != raw_y.shape or len(dates) != ntime or len(codes) != nasset:
        raise ValueError("trade ledger dimensions do not align")
    common = {"date": dates, "period": periods,
              "session": _session_names(dates)}
    portfolio_frames = []
    for name in _strategy_order(dynamic):
        p = dynamic["positions"][name]
        move = dynamic["changes"][name]
        net_asset = dynamic["net_returns"][name]
        valid = np.isfinite(raw_y)
        gross_asset = np.where(valid, p * raw_y, 0.0)
        fee_asset = np.abs(move) * spec.strategy_cost_bps / 1e4
        frame = pd.DataFrame({**common, "strategy": name,
                              "gross_log_return": _mean_asset(gross_asset),
                              "fee_log_return": _mean_asset(fee_asset),
                              "net_log_return": _mean_asset(net_asset),
                              "turnover": _mean_asset(np.abs(move)),
                              "gross_exposure": _mean_asset(np.abs(p)),
                              "net_exposure": _mean_asset(p),
                              "long_share": _mean_asset((p > .1).astype(float)),
                              "short_share": _mean_asset((p < -.1).astype(float))})
        portfolio_frames.append(frame)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    # Market benchmarks are context, not same-risk alpha competitors.
    market = _equal_market_buyhold_log_returns(raw_y, segment_lengths)
    benchmark = pd.DataFrame({**common, "strategy": "equal_market_buyhold",
                              "gross_log_return": market, "fee_log_return": 0.0,
                              "net_log_return": market, "turnover": 0.0,
                              "gross_exposure": 1.0, "net_exposure": 1.0,
                              "long_share": 1.0, "short_share": 0.0})
    portfolio = pd.concat([portfolio, benchmark], ignore_index=True)
    btc_index = codes.index("BTCUSDT") if "BTCUSDT" in codes else None
    if btc_index is not None:
        btc = np.nan_to_num(raw_y[:, btc_index], nan=0.0)
        benchmark_btc = benchmark.assign(strategy="BTC_buyhold",
                                         gross_log_return=btc,
                                         net_log_return=btc)
        portfolio = pd.concat([portfolio, benchmark_btc], ignore_index=True)
    portfolio["equity_100"] = np.nan
    for (period, name), index in portfolio.groupby(["period", "strategy"], sort=False).groups.items():
        idx = np.asarray(list(index))
        portfolio.loc[idx, "equity_100"] = 100 * np.exp(np.cumsum(
            portfolio.loc[idx, "net_log_return"].to_numpy()))

    p = dynamic["positions"][PRIMARY]
    move = dynamic["changes"][PRIMARY]
    valid = np.isfinite(raw_y)
    gross = np.where(valid, p * raw_y, 0.0)
    fee = np.abs(move) * spec.strategy_cost_bps / 1e4
    net = gross - fee
    asset = pd.DataFrame({
        "date": np.repeat(dates, nasset), "period": np.repeat(periods, nasset),
        "session": np.repeat(common["session"], nasset),
        "asset": np.tile(codes, ntime), "price": close.ravel(),
        "next_raw_log_return": raw_y.ravel(),
        "dynamic_alpha_strength": dynamic["dynamic_signal"].ravel(),
        "alpha_unit_target": dynamic["dynamic_alpha"].ravel(),
        "native5_alpha_unit": dynamic["native_alpha"].ravel(),
        "hourly_alpha_unit": dynamic["hourly_alpha"].ravel(),
        "beta_direction": np.repeat(dynamic["beta_direction"], nasset),
        "beta_prediction": np.repeat(dynamic["beta_prediction"], nasset),
        "risk_budget": np.repeat(dynamic["risk_budget"], nasset),
        "common_risk_forecast_1h": np.repeat(dynamic["q_5m"], nasset),
        "alpha_sleeve": dynamic["alpha_sleeve"].ravel(),
        "beta_sleeve": dynamic["beta_sleeve"].ravel(),
        "target": dynamic["targets"][PRIMARY].ravel(),
        "position": p.ravel(), "position_change": move.ravel(),
        "gross_log_return": gross.ravel(), "fee_log_return": fee.ravel(),
        "net_log_return": net.ravel(),
    })
    calibration = dynamic["signal_engine"]["calibration"]
    for name, profile_position in dynamic["signal_engine"]["profile_positions"].items():
        asset[f"{name}_strength"] = dynamic["signal_engine"]["profile_strengths"][name].ravel()
        asset[f"{name}_scale"] = dynamic["signal_engine"]["profile_scales"][name].ravel()
        asset[f"{name}_position"] = profile_position.ravel()
        asset[f"{name}_change"] = dynamic["changes"][f"profile_{name}"].ravel()
        chosen = calibration[(calibration.profile == name) & calibration.selected]
        asset[f"{name}_entry_threshold"] = float(chosen.entry_threshold.iloc[0])
    asset["equity_100"] = np.nan
    for (_, _), index in asset.groupby(["period", "asset"], sort=False).groups.items():
        idx = np.asarray(list(index))
        asset.loc[idx, "equity_100"] = 100 * np.exp(np.cumsum(
            asset.loc[idx, "net_log_return"].to_numpy()))
    return portfolio, asset


def trade_events(asset_ledger: pd.DataFrame, raw: np.ndarray,
                 weights: np.ndarray, registry: pd.DataFrame,
                 dates: np.ndarray, segment_lengths: list[int],
                 codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Display events use 0.1 state bands; fees use exact position changes."""
    nasset = len(codes)
    p = asset_ledger.position.to_numpy().reshape(-1, nasset)
    move = asset_ledger.position_change.to_numpy().reshape(-1, nasset)
    price = asset_ledger.price.to_numpy().reshape(-1, nasset)
    net = asset_ledger.net_log_return.to_numpy().reshape(-1, nasset)
    fee = asset_ledger.fee_log_return.to_numpy().reshape(-1, nasset)
    period_label = asset_ledger.period.to_numpy().reshape(-1, nasset)
    events, holds = [], []
    offset = 0
    for length in segment_lengths:
        for j, code in enumerate(codes):
            previous = 0.0
            entered = None
            direction = 0
            trade_net = 0.0
            for local in range(length):
                t = offset + local
                current = float(p[t, j])
                old_bucket = int(np.sign(previous)) if abs(previous) >= .1 else 0
                new_bucket = int(np.sign(current)) if abs(current) >= .1 else 0
                event = None
                if old_bucket == 0 and new_bucket != 0:
                    event = "open_long" if new_bucket > 0 else "open_short"
                elif old_bucket != 0 and new_bucket == 0:
                    event = "close_long" if old_bucket > 0 else "close_short"
                elif old_bucket != 0 and new_bucket != 0 and old_bucket != new_bucket:
                    event = "flip_to_long" if new_bucket > 0 else "flip_to_short"
                elif abs(move[t, j]) > 1e-12:
                    event = "scale_up" if abs(current) > abs(previous) else "scale_down"
                if old_bucket != 0 and new_bucket != old_bucket and entered is not None:
                    holds.append({"asset": code, "start": str(dates[entered]),
                                  "end": str(dates[t]), "direction": direction,
                                  "period": str(period_label[t, j]),
                                  "holding_hours": (t - entered) * 5 / 60,
                                  "net_log_return": trade_net - (fee[t, j] if new_bucket == 0 else 0),
                                  "win": trade_net - (fee[t, j] if new_bucket == 0 else 0) > 0,
                                  "right_censored": False})
                    entered = None
                    trade_net = 0.0
                if new_bucket != 0 and new_bucket != old_bucket:
                    entered = t
                    direction = new_bucket
                if new_bucket != 0:
                    trade_net += net[t, j]
                if event is not None:
                    contributions = raw[t, j] * weights[t]
                    top = np.argsort(-np.abs(contributions))[:3]
                    top_text = " | ".join(
                        f"{str(registry.iloc[k].expr_hash)[:8]}:{contributions[k]:+.3f}"
                        for k in top if abs(contributions[k]) > 1e-8)
                    events.append({"date": str(dates[t]), "asset": code,
                                   "event": event, "price": float(price[t, j]),
                                   "previous_position": previous,
                                   "new_position": current,
                                   "position_change": float(move[t, j]),
                                   "fee_log_return": float(fee[t, j]),
                                   "top_factor_contributions": top_text})
                previous = current
            if entered is not None:
                holds.append({"asset": code, "start": str(dates[entered]),
                              "end": str(dates[offset + length - 1]),
                              "period": str(period_label[offset + length - 1, j]),
                              "direction": direction,
                              "holding_hours": (offset + length - entered) * 5 / 60,
                              "net_log_return": trade_net,
                              "win": trade_net > 0,
                              "right_censored": True})
        offset += length
    event_columns = ["date", "asset", "event", "price", "previous_position",
                     "new_position", "position_change", "fee_log_return",
                     "top_factor_contributions"]
    hold_columns = ["asset", "start", "end", "period", "direction",
                    "holding_hours", "net_log_return", "win", "right_censored"]
    return pd.DataFrame(events, columns=event_columns), pd.DataFrame(holds, columns=hold_columns)


def performance_tables(portfolio: pd.DataFrame, asset: pd.DataFrame,
                       holds: pd.DataFrame, weight_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (period, strategy), group in portfolio.groupby(["period", "strategy"], sort=False):
        r = group.net_log_return.to_numpy(dtype=float)
        gross = group.gross_log_return.to_numpy(dtype=float)
        turnover = group.turnover.sum()
        days = group.date.str.slice(0, 8).nunique()
        total_log = float(r.sum())
        dd = _drawdown(r)
        annual = np.exp(total_log * 365 / max(days, 1)) - 1
        active = group.gross_exposure.to_numpy() > .01
        daily = group.groupby(group.date.str.slice(0, 8)).net_log_return.sum()
        positive = r[r > 0].sum()
        negative = abs(r[r < 0].sum())
        rows.append({"period": period, "strategy": strategy,
                     "bars": len(group), "days": days,
                     "gross_return_pct": 100 * np.expm1(gross.sum()),
                     "net_return_pct": 100 * np.expm1(total_log),
                     "total_net_log_return": total_log,
                     "daily_sharpe_365": _daily_sharpe(r, group.date.to_numpy()),
                     "max_drawdown_pct": 100 * dd,
                     "annualized_return_pct": 100 * annual,
                     "calmar": annual / abs(dd) if dd < -1e-9 else np.nan,
                     "bar_win_rate_active": float(np.mean(r[active] > 0)) if active.any() else np.nan,
                     "day_win_rate": float(np.mean(daily.to_numpy() > 0)),
                     "profit_factor": positive / negative if negative > 0 else np.nan,
                     "turnover": float(turnover),
                     "break_even_cost_bps": float(gross.sum() / turnover * 1e4)
                     if turnover > 0 else np.nan,
                     "average_gross_exposure": float(group.gross_exposure.mean()),
                     "average_net_exposure": float(group.net_exposure.mean()),
                     "long_asset_fraction": float(group.long_share.mean()),
                     "short_asset_fraction": float(group.short_share.mean()),
                     "flat_asset_fraction": float(1 - group.long_share.mean() - group.short_share.mean()),
                     "max_gross_exposure": float(group.gross_exposure.max())})
    performance = pd.DataFrame(rows)
    assets = []
    for (period, code), group in asset.groupby(["period", "asset"], sort=False):
        r = group.net_log_return.to_numpy(dtype=float)
        closed = holds[(holds.asset == code) & (holds.period == period)
                       & (~holds.right_censored)]
        completed = closed.net_log_return.to_numpy(dtype=float)
        assets.append({"period": period, "asset": code,
                       "net_return_pct": 100 * np.expm1(r.sum()),
                       "net_log_contribution": float(r.sum()),
                       "daily_sharpe_365": _daily_sharpe(r, group.date.to_numpy()),
                       "bar_win_rate_active": float(np.mean(r[np.abs(group.position.to_numpy()) > .1] > 0))
                       if (np.abs(group.position.to_numpy()) > .1).any() else np.nan,
                       "long_fraction": float(np.mean(group.position.to_numpy() > .1)),
                       "short_fraction": float(np.mean(group.position.to_numpy() < -.1)),
                       "flat_fraction": float(np.mean(np.abs(group.position.to_numpy()) <= .1)),
                       "trade_count_closed": len(closed),
                       "trade_win_rate": float(np.mean(completed > 0)) if len(completed) else np.nan,
                       "median_holding_hours": float(closed.holding_hours.median())
                       if len(closed) else np.nan,
                       "turnover": float(np.abs(group.position_change).sum())})
    asset_metrics = pd.DataFrame(assets)
    sessions = asset.groupby(["period", "asset", "session"], sort=False).agg(
        net_log_return=("net_log_return", "sum"),
        bars=("date", "size"),
        active_fraction=("position", lambda s: float(np.mean(np.abs(s) > .1))),
        long_fraction=("position", lambda s: float(np.mean(s > .1))),
        short_fraction=("position", lambda s: float(np.mean(s < -.1))),
    ).reset_index()
    uniformity = []
    asset_count = asset_metrics.asset.nunique()
    asset_metrics["net_log_contribution"] /= asset_count
    for period, group in asset_metrics.groupby("period", sort=False):
        contribution = group.net_log_contribution.to_numpy()
        magnitude = np.abs(contribution)
        share = magnitude / magnitude.sum() if magnitude.sum() > 0 else np.zeros_like(magnitude)
        hhi = np.sum(share ** 2)
        session_group = sessions[sessions.period == period]
        session_breadth = session_group.groupby("session").net_log_return.apply(
            lambda x: float(np.mean(x > 0))).to_dict()
        uniformity.append({"period": period,
                           "positive_asset_share": float(np.mean(contribution > 0)),
                           "abs_contribution_hhi": float(hhi),
                           "effective_contributing_assets": float(1 / hhi) if hhi > 0 else 0,
                           "worst_session_positive_asset_share": min(session_breadth.values()),
                           "session_positive_asset_share_json": json.dumps(session_breadth)})
    return performance, asset_metrics, sessions, pd.DataFrame(uniformity)


def profile_performance_tables(asset: pd.DataFrame, holds: pd.DataFrame,
                               performance: pd.DataFrame,
                               signal_spec: SignalEngineSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Completed profile trades use exact state-machine opens/closes, not display bands."""
    rows, by_asset = [], []
    nasset = asset.asset.nunique()
    for period, group in asset.groupby("period", sort=False):
        weeks = group.date.nunique() / 288 / 7
        for profile in signal_spec.profiles:
            name = profile.name
            closed = holds[(holds.period == period) & (holds.profile == name) &
                           (~holds.right_censored)]
            position = group[f"{name}_position"].to_numpy()
            strategy = performance[(performance.period == period) &
                                   (performance.strategy == f"profile_{name}")].iloc[0]
            hours = closed.holding_hours.to_numpy(dtype=float)
            rows.append({
                "period": period, "profile": name, "closed_trades": len(closed),
                "trades_per_asset_week": len(closed) / nasset / weeks,
                "mean_holding_hours": float(np.mean(hours)) if len(hours) else np.nan,
                "median_holding_hours": float(np.median(hours)) if len(hours) else np.nan,
                "p10_holding_hours": float(np.quantile(hours, .1)) if len(hours) else np.nan,
                "p90_holding_hours": float(np.quantile(hours, .9)) if len(hours) else np.nan,
                "holding_under_2h_fraction": float(np.mean(hours < 2)) if len(hours) else np.nan,
                "holding_over_5d_fraction": float(np.mean(hours > 120)) if len(hours) else np.nan,
                "active_asset_fraction": float(np.mean(np.abs(position) > 1e-9)),
                "long_asset_fraction": float(np.mean(position > 1e-9)),
                "short_asset_fraction": float(np.mean(position < -1e-9)),
                "net_return_pct": float(strategy.net_return_pct),
                "daily_sharpe_365": float(strategy.daily_sharpe_365),
                "turnover": float(strategy.turnover),
                "break_even_cost_bps": float(strategy.break_even_cost_bps),
            })
            for code, local in group.groupby("asset", sort=False):
                trades = closed[closed.asset == code]
                p = local[f"{name}_position"].to_numpy()
                by_asset.append({"period": period, "profile": name, "asset": code,
                                 "closed_trades": len(trades),
                                 "trades_per_week": len(trades) / weeks,
                                 "median_holding_hours": float(trades.holding_hours.median())
                                 if len(trades) else np.nan,
                                 "active_fraction": float(np.mean(np.abs(p) > 1e-9)),
                                 "turnover": float(np.abs(local[f"{name}_change"]).sum())})
    return pd.DataFrame(rows), pd.DataFrame(by_asset)


_CSS = """
body{margin:0;background:#f5f7fa;color:#192638;font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif}
nav{background:#102b47;color:white;padding:13px 28px;position:sticky;top:0;z-index:10}
nav a{color:#d6eaff;margin-right:20px;text-decoration:none;font-weight:600}
main{max-width:1440px;margin:25px auto;padding:0 20px 50px}
h1{font-size:29px;margin:10px 0}h2{font-size:21px;margin:32px 0 12px}
.muted{color:#52677a}.note{background:#fff8e8;border-left:4px solid #dfac35;padding:12px 16px;margin:18px 0}
.card{background:white;border:1px solid #dce4ec;border-radius:10px;padding:18px;margin:18px 0;box-shadow:0 2px 8px #1433500b}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.stat{background:#edf4fb;border-radius:8px;padding:13px}.stat b{font-size:24px;display:block}
table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:8px;border-bottom:1px solid #e1e8ee;text-align:right}th:first-child,td:first-child{text-align:left}
th{position:sticky;top:0;background:#eaf0f5}tr:hover{background:#f4f9fd}
.scroll{overflow:auto;max-height:560px}.links a{display:inline-block;padding:6px 10px;margin:4px;background:#e6f0fb;border-radius:5px;color:#12385b;text-decoration:none}
footer{color:#657789;font-size:12px;margin-top:30px}
"""


def _plot(fig: go.Figure, title: str, height: int = 370) -> str:
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=55, r=35, t=55, b=45),
                      hovermode="x unified", legend=dict(orientation="h", y=-.2))
    return f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo":False,"responsive":True})}</div>'


def _table(frame: pd.DataFrame, columns: list[str] | None = None,
           rows: int = 100) -> str:
    selected = frame[columns] if columns else frame
    selected = selected.head(rows).copy()
    for col in selected.select_dtypes(include=["number"]).columns:
        selected[col] = selected[col].map(lambda value: "—" if not np.isfinite(value)
                                           else f"{value:.4f}")
    return f'<div class="card scroll">{selected.to_html(index=False, escape=True, border=0)}</div>'


def _page(out: Path, relative: str, title: str, body: str,
          subtitle: str = "") -> None:
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = "../" if relative.startswith("assets/") else ""
    links = [("总览", "index.html"), ("触发与交易", "index.html#execution"),
             ("预测与基线", "models.html"),
             ("动态权重", "weights.html"), ("Alpha/Beta/Risk", "sleeves.html"),
             ("时段与均匀性", "sessions.html")]
    nav = "".join(f'<a href="{prefix}{name}">{label}</a>' for label, name in links)
    doc = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
           f"<script src='{prefix}plotly.min.js'></script></head><body>"
           f"<nav>{nav}</nav><main><h1>{html.escape(title)}</h1>"
           f"<p class='muted'>{html.escape(subtitle)}</p>{body}"
           "<footer>5m 完成后决策；目标承担下一根完整 5m 收益；2023 发现方向；"
           "逐币等权组合；仓位变化扣费；累计净值 = 100×exp(累计近似对数收益)。"
           "2023/2024/2025 为已见年份历史复核。</footer></main></body></html>")
    target.write_text(doc, encoding="utf-8")


def _datetime(values: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(values, format="%Y%m%d%H%M", utc=True)


def _line_figure(frame: pd.DataFrame, names: list[str], stride: int,
                 y: str = "equity_100") -> go.Figure:
    fig = go.Figure()
    for name in names:
        sub = frame[frame.strategy == name].iloc[::stride]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(x=_datetime(sub.date), y=sub[y], mode="lines",
                                 name=name, line=dict(width=2)))
    fig.update_yaxes(title_text="净值（起点 100）" if y == "equity_100" else y)
    return fig


def _equity_pages(out: Path, portfolio: pd.DataFrame,
                  performance: pd.DataFrame, spec: DynamicSpec,
                  signal_spec: SignalEngineSpec) -> list[str]:
    pages = []
    for period in dict.fromkeys(portfolio.period):
        if str(period).endswith("_train"):
            continue
        data = portfolio[portfolio.period == period]
        title = f"{period}｜策略净值与基准"
        body = '<div class="note">策略线共享下一根 5m 原始收益和 4 bps 变仓费用。多时钟执行按 2023 训练段的信号频率校准开仓阈值；旧 0.25 绝对带和静态模型保留对照。买入持有图另列。历史复核，非盲测。</div>'
        body += _plot(_line_figure(data, [PRIMARY, "mix_risk_dynamic", "mix_risk_5m_utility"],
                                   spec.plot_stride_bars),
                      "新触发机制与旧绝对带、统一 5m 效用对照")
        body += _plot(_line_figure(data, [PRIMARY, *[f"profile_{p.name}" for p in signal_spec.profiles]],
                                   spec.plot_stride_bars),
                      "快/中/慢各通道与合成：费后净值")
        body += _plot(_line_figure(data, ["mix_risk_dynamic", "family_equal", "elastic_net", "style_B0"],
                                   spec.plot_stride_bars),
                      "旧动态 combo 与静态因子/风格基线")
        body += _plot(_line_figure(data, ["alpha_only", "beta_only", "mix_no_risk", "mix_risk_dynamic"],
                                   spec.plot_stride_bars),
                      "原持有期因子权重的 Alpha/Beta/Risk 消融")
        body += _plot(_line_figure(data, ["equal_market_buyhold", "BTC_buyhold"],
                                   spec.plot_stride_bars),
                      "行情背景：等权市场与 BTC 买入持有（未扣执行费用）")
        selected = data[data.strategy == PRIMARY].copy()
        selected["day"] = selected.date.str.slice(0, 8)
        daily = selected.groupby("day")["net_log_return"].sum().reset_index()
        fig = go.Figure(go.Bar(x=pd.to_datetime(daily.day, format="%Y%m%d", utc=True),
                               y=100 * np.expm1(daily.net_log_return),
                               marker_color=np.where(daily.net_log_return >= 0, "#167b5a", "#bd5050")))
        fig.update_yaxes(title_text="每日净收益 %")
        body += _plot(fig, "动态 combo 每日收益：正负日期分布")
        metrics = performance[performance.period == period]
        body += "<h2>收益、风险、交易与持仓指标</h2>" + _table(metrics, [
            "strategy", "net_return_pct", "gross_return_pct", "daily_sharpe_365",
            "max_drawdown_pct", "bar_win_rate_active", "day_win_rate", "profit_factor",
            "turnover", "break_even_cost_bps", "average_gross_exposure",
            "long_asset_fraction", "short_asset_fraction", "flat_asset_fraction"], rows=30)
        filename = f"equity_{period}.html"
        _page(out, filename, title, body, subtitle="单独 y 轴；收益与市场方向背景分别展示")
        pages.append(filename)
    return pages


def _models_page(out: Path, forecast: pd.DataFrame, risk: pd.DataFrame,
                 performance: pd.DataFrame) -> None:
    review = forecast[~forecast.period.str.endswith("_train")].copy()
    fig = go.Figure()
    for model, group in review.groupby("model", sort=False):
        fig.add_trace(go.Bar(name=model, x=group.period,
                             y=100 * group.delta_loss_vs_B0))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="相对 B0 的 MSE 改善 %")
    body = '<div class="note">预测损失与交易净收益是不同问题。下表所有模型对相同下一根 5m 残差标签逐行比较；这里不按已见年份重新挑模型。</div>'
    body += _plot(fig, "冻结模型预测误差：按年份分组，百分比刻度")
    body += "<h2>预测指标</h2>" + _table(review, [
        "period", "model", "rows", "delta_loss_vs_B0", "pooled_ic", "positive_asset_share"], rows=100)
    body += "<h2>共同风险预测</h2>" + _table(risk, rows=20)
    body += "<h2>同仓位规则的策略指标</h2>" + _table(performance[~performance.period.str.endswith("_train")], [
        "period", "strategy", "net_return_pct", "daily_sharpe_365", "max_drawdown_pct",
        "day_win_rate", "turnover", "break_even_cost_bps"], rows=100)
    _page(out, "models.html", "预测、基线与交易结果", body,
          subtitle="误差、IC、费后收益各有独立单位和 y 轴")


def _weights_page(out: Path, weights: pd.DataFrame, events: pd.DataFrame,
                  period_lookup: dict[str, str], registry: pd.DataFrame) -> None:
    frame = weights.copy()
    frame["period"] = frame.date.map(period_lookup)
    labels = {row.expr_hash: f"{row.family.split(':')[-1][:18]} · {row.expr_hash[:6]}"
              for row in registry.itertuples()}
    body = '<div class="note">每小时仅用截至上一根已成熟标签更新效用；因子方向来自 2023 发现期。色深代表权重，0 为退出。完整每 5m 权重在 parquet。</div>'
    for period in dict.fromkeys(frame.period):
        if str(period).endswith("_train"):
            continue
        group = frame[frame.period == period]
        pivot = group.pivot(index="expr_hash", columns="date", values="weight")
        pivot = pivot.reindex(registry.expr_hash.tolist())
        heat = go.Figure(go.Heatmap(x=_datetime(pivot.columns.to_numpy()),
                                    y=[labels.get(k, k[:8]) for k in pivot.index],
                                    z=pivot.to_numpy(), colorscale="Blues",
                                    zmin=0, zmax=.4, colorbar=dict(title="权重")))
        body += f"<h2>{html.escape(str(period))}：因子权重与出入</h2>"
        body += _plot(heat, "各表达式权重热图（逐小时）", height=510)
        active = group.groupby("date").active.sum().reset_index()
        fig = go.Figure(go.Scatter(x=_datetime(active.date), y=active.active,
                                   mode="lines", name="活跃因子数"))
        fig.update_yaxes(title_text="活跃因子数", rangemode="tozero")
        body += _plot(fig, "活跃因子数随时间变化")
    summary = frame.groupby(["expr_hash", "family", "clock", "role"], sort=False).agg(
        mean_weight=("weight", "mean"), active_hours=("active", "sum"),
        mean_utility_z=("utility_z", "mean")).reset_index()
    body += "<h2>因子参与度与权重</h2>" + _table(summary, rows=100)
    body += "<h2>入选与退出事件（前 200 条；完整 CSV 已保存）</h2>" + _table(events, rows=200)
    _page(out, "weights.html", "动态因子权重、入选与退出", body,
          subtitle="表达式层选择，家族限额，小时更新")


def _sleeves_page(out: Path, portfolio: pd.DataFrame, asset: pd.DataFrame,
                  spec: DynamicSpec) -> None:
    body = '<div class="note">Alpha：动态因子加权并去除逐时刻跨币共同均值。原生 5m 因子使用下一根 5m 标签，小时因子使用延迟 1 小时后完整 4 小时标签；效用在标签完全成熟后才更新。Beta：训练段拟合的共同市场方向。共同下行风险 q 只缩放预算，不决定多空方向。以下分开显示预测、预算、目标和最终持仓。</div>'
    for period in dict.fromkeys(asset.period):
        if str(period).endswith("_train"):
            continue
        group = asset[asset.period == period].groupby("date", sort=False).agg(
            mean_abs_alpha=("alpha_sleeve", lambda s: float(np.mean(np.abs(s)))),
            beta_sleeve=("beta_sleeve", "mean"),
            risk_budget=("risk_budget", "first"),
            common_risk=("common_risk_forecast_1h", "first"),
            gross_position=("position", lambda s: float(np.mean(np.abs(s)))),
            net_position=("position", "mean"),
        ).reset_index().iloc[::spec.plot_stride_bars]
        x = _datetime(group.date)
        body += f"<h2>{html.escape(str(period))}：三条协作链</h2>"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=group.mean_abs_alpha, name="Alpha 袖绝对目标"))
        fig.add_trace(go.Scatter(x=x, y=group.beta_sleeve, name="Beta 共同方向目标"))
        fig.update_yaxes(title_text="目标仓位单位")
        body += _plot(fig, "Alpha 与 Beta 原始目标：方向和强度")
        risk_fig = go.Figure(go.Scatter(x=x, y=group.common_risk, name="预测共同下行风险"))
        risk_fig.update_yaxes(title_text="未来 1h 下行平方和预测（对数轴）", type="log")
        body += _plot(risk_fig, "风险预测 q（单独量纲）")
        budget_fig = go.Figure(go.Scatter(x=x, y=group.risk_budget, name="风险预算系数"))
        budget_fig.update_yaxes(title_text="预算乘数", range=[0, 1.05])
        body += _plot(budget_fig, "风险预算缩放")
        pos_fig = go.Figure()
        pos_fig.add_trace(go.Scatter(x=x, y=group.gross_position, name="平均绝对持仓"))
        pos_fig.add_trace(go.Scatter(x=x, y=group.net_position, name="平均净方向"))
        pos_fig.update_yaxes(title_text="仓位单位")
        body += _plot(pos_fig, "执行后的持仓幅度和净方向")
        comparison = portfolio[portfolio.period == period]
        body += _plot(_line_figure(comparison,
                                   ["alpha_only", "beta_only", "mix_no_risk", "mix_risk_dynamic"],
                                   spec.plot_stride_bars),
                      "原持有期因子权重的 Alpha/Beta/Risk 四组消融")
    _page(out, "sleeves.html", "Alpha、Beta 与风险预算如何协作", body,
          subtitle="方向预测、风险预测与执行持仓采用独立尺度")


def _sessions_page(out: Path, asset_metrics: pd.DataFrame, sessions: pd.DataFrame,
                   uniformity: pd.DataFrame, holds: pd.DataFrame) -> None:
    body = '<div class="note">资产和纽约时段的表现按固定评价行分解；均匀性按绝对收益贡献计算，有效品种数越低表示结果越集中。</div>'
    body += "<h2>资产与时段均匀性</h2>" + _table(uniformity, rows=20)
    for period in dict.fromkeys(sessions.period):
        if str(period).endswith("_train"):
            continue
        group = sessions[sessions.period == period]
        matrix = group.pivot(index="asset", columns="session", values="net_log_return")
        heat = go.Figure(go.Heatmap(z=100 * matrix.to_numpy(), x=matrix.columns,
                                    y=matrix.index, colorscale="RdBu", zmid=0,
                                    colorbar=dict(title="净对数收益 %")))
        body += f"<h2>{html.escape(str(period))}：按品种与纽约时段</h2>"
        body += _plot(heat, "逐币 × 纽约时段费后贡献", height=440)
        am = asset_metrics[asset_metrics.period == period]
        bar = go.Figure()
        for key, title in (("long_fraction", "多头"), ("short_fraction", "空头"),
                           ("flat_fraction", "空仓")):
            bar.add_trace(go.Bar(name=title, x=am.asset, y=100 * am[key]))
        bar.update_layout(barmode="stack")
        bar.update_yaxes(title_text="时间比例 %", range=[0, 100])
        body += _plot(bar, "逐币多/空/空仓时间比例")
        body += _table(am, ["asset", "net_return_pct", "daily_sharpe_365",
                            "trade_count_closed", "trade_win_rate", "median_holding_hours",
                            "long_fraction", "short_fraction", "flat_fraction", "turnover"], rows=50)
        h = holds[(holds.period == period) & (~holds.right_censored)]
        if not h.empty:
            fig = go.Figure(go.Histogram(x=h.holding_hours, nbinsx=40,
                                         marker_color="#2b719b"))
            fig.update_xaxes(title_text="已平仓持仓小时")
            fig.update_yaxes(title_text="次数")
            body += _plot(fig, "已平仓交易持仓时间分布")
    _page(out, "sessions.html", "持仓时间、纽约时段和品种均匀性", body,
          subtitle="胜率分别给出 bar、日和已平仓交易定义")


def _asset_pages(out: Path, asset: pd.DataFrame, events: pd.DataFrame,
                 holds: pd.DataFrame, asset_metrics: pd.DataFrame,
                 profile_events: pd.DataFrame, profile_holds: pd.DataFrame,
                 spec: DynamicSpec, signal_spec: SignalEngineSpec) -> list[str]:
    pages = []
    for (period, code), group in asset.groupby(["period", "asset"], sort=False):
        if str(period).endswith("_train"):
            continue
        group = group.reset_index(drop=True)
        plot = group.iloc[::spec.plot_stride_bars]
        x = _datetime(plot.date)
        markers = events[(events.asset == code) &
                         (events.date >= group.date.iloc[0]) &
                         (events.date <= group.date.iloc[-1])]
        title = f"{code} · {period}｜信号、仓位与交易"
        body = ("<div class='note'>价格图的标记是完成 bar 后产生的持仓事件，收益从下一根 5m bar 开始。"
                "实际费用按完整 5m 仓位变化计算；图线固定每小时抽样，完整账本未抽样。"
                "跨 2023 训练/复核边界的持仓会沿用，页首可能已有仓位。</div>")
        price = go.Figure(go.Scatter(x=x, y=plot.price, mode="lines",
                                      name="完成 bar 收盘价", line=dict(color="#305c80", width=1.5)))
        for event_type, label, symbol, color in (
            ("open_long", "开多", "triangle-up", "#13845c"),
            ("open_short", "开空", "triangle-down", "#c04b52"),
            ("close_long", "平多", "x", "#ae7c22"),
            ("close_short", "平空", "x", "#ae7c22"),
            ("flip_to_long", "翻多", "diamond", "#13845c"),
            ("flip_to_short", "翻空", "diamond", "#c04b52"),
        ):
            subset = markers[markers.event == event_type]
            if subset.empty:
                continue
            price.add_trace(go.Scatter(x=_datetime(subset.date), y=subset.price,
                                       mode="markers", name=label,
                                       marker=dict(symbol=symbol, color=color, size=9),
                                       customdata=np.stack([subset.previous_position,
                                                            subset.new_position], axis=1),
                                       hovertemplate="%{x}<br>价格 %{y:.5f}<br>仓位 %{customdata[0]:.2f} → %{customdata[1]:.2f}<extra>%{fullData.name}</extra>"))
        price.update_yaxes(title_text="价格（原币报价）")
        body += _plot(price, "价格与开多/开空/平仓/翻向信号", height=460)

        strength = go.Figure()
        alpha_smooth = group.alpha_unit_target.ewm(span=12, adjust=False).mean().iloc[::spec.plot_stride_bars]
        beta_smooth = group.beta_direction.ewm(span=12, adjust=False).mean().iloc[::spec.plot_stride_bars]
        strength.add_trace(go.Scatter(x=x, y=plot.alpha_unit_target,
                                      name="Alpha 原始（淡色）",
                                      line=dict(color="rgba(52,102,201,.22)", width=1)))
        strength.add_trace(go.Scatter(x=x, y=plot.beta_direction,
                                      name="Beta 原始（淡色）",
                                      line=dict(color="rgba(211,92,72,.22)", width=1)))
        strength.add_trace(go.Scatter(x=x, y=alpha_smooth,
                                      name="Alpha 1h 因果 EMA",
                                      line=dict(color="#315dc0", width=2)))
        strength.add_trace(go.Scatter(x=x, y=beta_smooth,
                                      name="Beta 1h 因果 EMA",
                                      line=dict(color="#bd4d43", width=2)))
        strength.update_yaxes(title_text="无量纲方向强度", range=[-1.1, 1.1])
        body += _plot(strength, "方向信号强度：Alpha 与 Beta 分开")

        trigger_plot = group.iloc[::max(1, spec.plot_stride_bars // 2)]
        trigger_x = _datetime(trigger_plot.date)
        trigger = make_subplots(rows=len(signal_spec.profiles), cols=1,
                                shared_xaxes=True, vertical_spacing=.08,
                                subplot_titles=[f"{p.name}：标准化强度与入场带" for p in signal_spec.profiles])
        for row, profile in enumerate(signal_spec.profiles, start=1):
            threshold = float(group[f"{profile.name}_entry_threshold"].iloc[0])
            trigger.add_trace(go.Scatter(x=trigger_x,
                                         y=trigger_plot[f"{profile.name}_strength"],
                                         name=f"{profile.name} 强度",
                                         line=dict(width=1.3)), row=row, col=1)
            trigger.add_hline(y=threshold, line_dash="dash", line_color="#b56d21",
                              row=row, col=1)
            trigger.add_hline(y=-threshold, line_dash="dash", line_color="#b56d21",
                              row=row, col=1)
            trigger.update_yaxes(title_text="强度", row=row, col=1)
        body += _plot(trigger, "快/中/慢触发强度：虚线为发现期冻结的入场阈值",
                      height=260 * len(signal_spec.profiles))

        risk = go.Figure(go.Scatter(x=x, y=plot.common_risk_forecast_1h,
                                    name="未来 1h 共同下行风险预测"))
        risk.update_yaxes(title_text="下行平方和预测（对数轴）", type="log")
        body += _plot(risk, "共同风险预测 q（独立量纲）")
        budget = go.Figure(go.Scatter(x=x, y=plot.risk_budget,
                                      name="风险预算乘数"))
        budget.update_yaxes(title_text="风险预算", range=[0, 1.05])
        body += _plot(budget, "风险状态如何放缩目标仓位")

        position = go.Figure()
        for field, label in (("target", "触发前合成目标"),
                             ("position", "执行后净持仓")):
            position.add_trace(go.Scatter(x=x, y=plot[field], mode="lines", name=label))
        position.update_yaxes(title_text="仓位单位")
        body += _plot(position, "合成方向目标与实际净持仓")
        profile_position = go.Figure()
        for profile in signal_spec.profiles:
            profile_position.add_trace(go.Scatter(
                x=x, y=plot[f"{profile.name}_position"],
                mode="lines", name=f"{profile.name} 持仓"))
        profile_position.add_trace(go.Scatter(x=x, y=plot.position,
                                              mode="lines", name="净额后总仓位",
                                              line=dict(color="#172e55", width=2.5)))
        profile_position.update_yaxes(title_text="仓位单位")
        body += _plot(profile_position, "多时间尺度仓位贡献与净额")

        equity = go.Figure(go.Scatter(x=x, y=plot.equity_100,
                                      name="该币策略净值（含 4 bps）"))
        equity.update_yaxes(title_text="策略净值，起点 100")
        body += _plot(equity, "该币 signed long-short 费后净值")
        raw = np.nan_to_num(group.next_raw_log_return.to_numpy(dtype=float), nan=0)
        hold_equity = 100 * np.exp(np.cumsum(raw))
        market_fig = go.Figure(go.Scatter(x=x, y=hold_equity[::spec.plot_stride_bars],
                                          name="该币买入持有"))
        market_fig.update_yaxes(title_text="买入持有净值，起点 100")
        body += _plot(market_fig, "该币买入持有背景（单独 y 轴，不扣交易费用）")

        metric = asset_metrics[(asset_metrics.period == period) &
                               (asset_metrics.asset == code)]
        body += "<h2>品种级指标</h2>" + _table(metric, rows=1)
        local_holds = holds[(holds.period == period) & (holds.asset == code)]
        body += "<h2>持仓区间（末尾未结束的持仓标记右删失）</h2>" + _table(local_holds, rows=200)
        body += "<h2>成交和仓位变化（前 200 条；完整 CSV 已保存）</h2>" + _table(
            markers, ["date", "event", "price", "previous_position",
                      "new_position", "position_change", "fee_log_return",
                      "top_factor_contributions"], rows=200)
        local_profile_events = profile_events[(profile_events.asset == code) &
                                              (profile_events.period == period)]
        local_profile_holds = profile_holds[(profile_holds.asset == code) &
                                            (profile_holds.period == period)]
        body += "<h2>分通道开平仓和调仓原因（前 200 条）</h2>" + _table(
            local_profile_events, ["date", "profile", "event", "reason",
                                   "strength", "entry_threshold", "exit_threshold",
                                   "previous_position", "new_position"], rows=200)
        body += "<h2>分通道持仓区间（前 200 条）</h2>" + _table(
            local_profile_holds, ["profile", "start", "end", "direction",
                                  "holding_hours", "exit_reason"], rows=200)
        filename = f"assets/{code}_{period}.html"
        _page(out, filename, title, body,
              subtitle="逐品种独立页面；价格、风险、强度、仓位与收益分开坐标")
        pages.append(filename)
    return pages


def _trigger_pages(out: Path, asset: pd.DataFrame,
                   events: pd.DataFrame, holds: pd.DataFrame,
                   metrics: pd.DataFrame, asset_metrics: pd.DataFrame,
                   calibration: pd.DataFrame,
                   signal_spec: SignalEngineSpec) -> list[str]:
    pages = []
    selected = calibration[calibration.selected]
    for period in dict.fromkeys(asset.period):
        if str(period).endswith("_train"):
            continue
        sub = asset[asset.period == period]
        p_events = events[events.period == period]
        p_holds = holds[holds.period == period]
        p_metrics = metrics[metrics.period == period]
        p_asset = asset_metrics[asset_metrics.period == period]
        body = ("<div class='note'>开平仓次数由分通道状态机精确记录。候选阈值仅按 2023 训练段"
                "的每品种每周交易次数选择；2024/2025 不参与选择。收益全部按每次净仓位变化"
                "扣 4 bps，持仓时长以完整 5m bar 计。</div>")
        body += "<h2>训练期冻结阈值</h2>" + _table(selected, [
            "profile", "entry_threshold", "train_trades_per_asset_week",
            "target_trades_per_asset_week", "target_within_candidate_range"], rows=10)
        body += "<h2>通道交易统计</h2>" + _table(p_metrics, rows=10)
        fig = go.Figure()
        for profile in signal_spec.profiles:
            part = p_asset[p_asset.profile == profile.name]
            fig.add_trace(go.Bar(name=profile.name, x=part.asset,
                                 y=part.trades_per_week))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="每品种每周完成交易次数")
        body += _plot(fig, "各品种交易频率：快/中/慢通道")
        fig = go.Figure()
        for profile in signal_spec.profiles:
            part = p_holds[p_holds.profile == profile.name]
            fig.add_trace(go.Histogram(name=profile.name, x=part.holding_hours,
                                       opacity=.65, nbinsx=45))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="完成交易的持仓小时")
        fig.update_yaxes(title_text="交易次数")
        body += _plot(fig, "持仓时长分布：各通道独立刻度")
        sample = sub.groupby("date", sort=False).agg(
            active=("position", lambda s: float(np.mean(np.abs(s) > 1e-9))),
            gross=("position", lambda s: float(np.mean(np.abs(s)))),
            net=("position", "mean")).reset_index().iloc[::12]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("活跃品种比例", "平均总敞口及净方向"))
        fig.add_trace(go.Scatter(x=_datetime(sample.date), y=sample.active,
                                 name="活跃比例"), row=1, col=1)
        fig.add_trace(go.Scatter(x=_datetime(sample.date), y=sample.gross,
                                 name="总敞口"), row=2, col=1)
        fig.add_trace(go.Scatter(x=_datetime(sample.date), y=sample.net,
                                 name="净方向"), row=2, col=1)
        fig.update_yaxes(title_text="0–1", row=1, col=1)
        fig.update_yaxes(title_text="单位仓位", row=2, col=1)
        body += _plot(fig, "仓位覆盖、方向与放缩", height=550)
        reasons = p_events.groupby(["profile", "event", "reason"]).size().reset_index(name="count")
        if not reasons.empty:
            fig = go.Figure()
            for profile, local in reasons.groupby("profile", sort=False):
                fig.add_trace(go.Bar(name=profile,
                                     x=local.event + " / " + local.reason,
                                     y=local["count"]))
            fig.update_layout(barmode="group")
            fig.update_yaxes(title_text="事件次数")
            body += _plot(fig, "开平仓、时间止盈止损与再平衡原因")
        body += "<h2>品种与时长明细</h2>" + _table(p_asset, rows=100)
        body += "<h2>最近开平仓事件</h2>" + _table(
            p_events.tail(100), ["date", "asset", "profile", "event", "reason",
                                 "strength", "entry_threshold", "previous_position",
                                 "new_position"], rows=100)
        filename = f"trigger_{period}.html"
        _page(out, filename, f"{period}｜触发与交易统计", body,
              subtitle="精确状态机交易；分通道阈值、时长、品种均匀性与仓位覆盖")
        pages.append(filename)
    return pages


def _index_page(out: Path, performance: pd.DataFrame,
                uniformity: pd.DataFrame, equity_pages: list[str],
                asset_pages: list[str], factor_events: pd.DataFrame,
                trigger_pages: list[str], profile_metrics: pd.DataFrame,
                spec: DynamicSpec) -> None:
    primary = performance[(performance.strategy == PRIMARY) &
                          (~performance.period.str.endswith("_train"))]
    cards = "".join(
        f"<div class='stat'>{html.escape(str(row.period))}<b>{row.net_return_pct:+.2f}%</b>"
        f"日 Sharpe {row.daily_sharpe_365:.2f} · 最大回撤 {row.max_drawdown_pct:.2f}%</div>"
        for row in primary.itertuples())
    body = ("<div class='note'>这是一套历史复核与研究账本，不是把最佳曲线挑出来的回测。"
            "所有模型共享下一根 5m 收益与同一 4 bps 成本；多尺度信号按训练期交易频率校准。"
            "主动态组合按因子原挖掘持有期评分，另列统一下一根 5m 评分的动态对照。"
            "Alpha 逐时刻去市场共同均值；Beta 是共同市场方向；风险 q 只改变预算。"
            "2023 年 3 月和 2024/2025 一季度已用于研究，均非全新盲测。</div>")
    body += f"<div class='grid'>{cards}</div>"
    body += "<h2>年度净值页</h2><div class='links'>" + "".join(
        f'<a href="{name}">{name.replace("equity_", "").replace(".html", "")}</a>'
        for name in equity_pages) + "</div>"
    body += "<h2>分项研究页</h2><div class='links'>" + "".join(
        f'<a href="{name}">{label}</a>' for name, label in (
            ("models.html", "预测与基线"), ("weights.html", "动态因子权重/出入"),
            ("sleeves.html", "Alpha/Beta/Risk 协作"),
            ("sessions.html", "持仓/时段/均匀性"))) + "</div>"
    body += "<h2 id='execution'>触发、开平仓与持仓</h2><div class='links'>" + "".join(
        f'<a href="{name}">{name.replace("trigger_", "").replace(".html", "")}</a>'
        for name in trigger_pages) + "</div>"
    body += _table(profile_metrics[~profile_metrics.period.str.endswith("_train")], [
        "period", "profile", "closed_trades", "trades_per_asset_week",
        "median_holding_hours", "active_asset_fraction", "net_return_pct",
        "daily_sharpe_365", "turnover", "break_even_cost_bps"], rows=40)
    body += "<h2>动态组合核心指标</h2>" + _table(primary, [
        "period", "net_return_pct", "daily_sharpe_365",
        "max_drawdown_pct", "bar_win_rate_active", "day_win_rate",
        "turnover", "break_even_cost_bps"], rows=20)
    body += "<h2>因子变化与品种均匀性</h2>"
    body += f"<p>记录因子出入事件 {len(factor_events)} 次；动态参数：{html.escape(str(spec))}</p>"
    body += _table(uniformity, ["period", "positive_asset_share",
                                "effective_contributing_assets",
                                "worst_session_positive_asset_share"], rows=20)
    body += "<h2>逐品种独立页面</h2><div class='links'>" + "".join(
        f'<a href="{name}">{name.split("/")[-1].replace(".html", "")}</a>'
        for name in asset_pages) + "</div>"
    body += "<h2>完整证据文件</h2><p>portfolio_5m.parquet、asset_positions_5m.parquet、factor_weights_5m.parquet、factor_weights_hourly.parquet、profile_events_5m.csv、profile_holding_intervals.csv、trigger_calibration.csv、profile_metrics.csv、profile_asset_metrics.csv、strategy_metrics.csv。</p>"
    _page(out, "index.html", "Factor combo 动态权重与交易研究台", body,
          subtitle="12 币原生 5m 预测；独立年度、品种和机制页面；所有图表可追溯到账本")


def build_dynamic_reports(out: Path, dynamic: dict, raw: np.ndarray,
                          registry: pd.DataFrame, segments: list[dict],
                          dates: np.ndarray, periods: np.ndarray,
                          spec: DynamicSpec, signal_spec: SignalEngineSpec,
                          forecast_comparison: pd.DataFrame,
                          risk_comparison: pd.DataFrame) -> dict:
    codes = segments[0]["codes"]
    segment_lengths = [len(s["dates"]) for s in segments]
    close = np.concatenate([s["close"] for s in segments])
    raw_y = np.concatenate([s["raw_y"] for s in segments])
    portfolio, asset = build_ledgers(dynamic, close, raw_y, dates, periods,
                                     codes, segment_lengths, spec)
    fills, holds = trade_events(asset, raw, dynamic["weights"], registry,
                                dates, segment_lengths, codes)
    performance, asset_metrics, sessions, uniformity = performance_tables(
        portfolio, asset, holds, dynamic["weight_rows"])
    signal = dynamic["signal_engine"]
    profile_events = signal["profile_events"].copy()
    profile_holds = signal["profile_holds"].copy()
    if not profile_events.empty:
        profile_events["asset"] = [codes[j] for j in profile_events.asset_index]
        profile_events["period"] = periods[profile_events.t.to_numpy(dtype=int)]
    else:
        profile_events = pd.DataFrame(columns=["date", "asset", "period", "profile",
                                               "event", "reason", "strength", "entry_threshold",
                                               "exit_threshold", "previous_position", "new_position"])
    if not profile_holds.empty:
        profile_holds["asset"] = [codes[j] for j in profile_holds.asset_index]
        profile_holds["period"] = periods[profile_holds.end_t.to_numpy(dtype=int)]
    else:
        profile_holds = pd.DataFrame(columns=["asset", "period", "profile", "start",
                                              "end", "direction", "holding_hours", "exit_reason",
                                              "right_censored"])
    profile_metrics, profile_asset_metrics = profile_performance_tables(
        asset, profile_holds, performance, signal_spec)
    lookup = dict(zip(dates, periods))
    factor_events = dynamic["weight_events"].copy()
    factor_events["period"] = factor_events.date.map(lookup)
    portfolio.to_parquet(out / "portfolio_5m.parquet", index=False)
    asset.to_parquet(out / "asset_positions_5m.parquet", index=False)
    dynamic["weight_rows"].to_parquet(out / "factor_weights_hourly.parquet", index=False)
    pd.DataFrame({"date": dates, "period": periods,
                  **{str(registry.iloc[j].expr_hash): dynamic["weights"][:, j]
                     for j in range(len(registry))}}).to_parquet(
                         out / "factor_weights_5m.parquet", index=False)
    fills.to_csv(out / "fills_5m.csv", index=False)
    holds.to_csv(out / "holding_intervals.csv", index=False)
    factor_events.to_csv(out / "factor_entry_exit.csv", index=False)
    performance.to_csv(out / "strategy_metrics.csv", index=False)
    asset_metrics.to_csv(out / "asset_metrics.csv", index=False)
    sessions.to_csv(out / "session_metrics.csv", index=False)
    uniformity.to_csv(out / "uniformity.csv", index=False)
    profile_events.to_csv(out / "profile_events_5m.csv", index=False)
    profile_holds.to_csv(out / "profile_holding_intervals.csv", index=False)
    signal["calibration"].to_csv(out / "trigger_calibration.csv", index=False)
    profile_metrics.to_csv(out / "profile_metrics.csv", index=False)
    profile_asset_metrics.to_csv(out / "profile_asset_metrics.csv", index=False)
    (out / "dynamic_model_metadata.json").write_text(json.dumps({
        "version": REPORT_VERSION,
        "signal_engine_version": SIGNAL_ENGINE_VERSION,
        "signal_engine_config": {"scale_lookback_bars": signal_spec.scale_lookback_bars,
                                 "scale_floor_fraction": signal_spec.scale_floor_fraction,
                                 "max_abs_position": signal_spec.max_abs_position,
                                 "leverage_diagnostic": signal_spec.leverage_diagnostic,
                                 "profiles": [p.__dict__ for p in signal_spec.profiles]},
        "selected_entry_thresholds": signal["calibration"].loc[
            signal["calibration"].selected, ["profile", "entry_threshold",
                                               "train_trades_per_asset_week"]].to_dict("records"),
        "beta_model_coefficients_bps": dynamic["beta_model_coef_bps"],
        "q_training_median": dynamic["q50"], "q_training_p90": dynamic["q90"],
        "dynamic_alpha_training_scale": dynamic["dynamic_signal_scale"],
        "fast_dynamic_alpha_training_scale": dynamic["fast_dynamic_signal_scale"],
        "beta_training_scale": dynamic["beta_signal_scale"],
        "factor_utility": {"native5": "next completed 5m residual return, maturity 1 bar",
                           "hourly": "4h residual return beginning after 1h entry lag, maturity 60 bars; one observation per completed hour",
                           "control": "all factors use next completed 5m residual return"},
        "portfolio_return": "mean_i(position_i,t * next_5m_log_return_i,t - abs(delta_position_i,t) * cost_bps/1e4)",
        "equity": "100 * exp(cumulative approximately log portfolio return)",
        "equal_market_benchmark": "equal initial capital per asset, no rebalance within each declared segment; missing next-bar returns imputed zero",
        "legacy_execution_event_band_by_strategy": {name: _strategy_event_band(name, spec)
                                                     for name in LEGACY_ORDER},
        "trade_event_display_threshold": 0.1,
        "daily_sharpe_annualization": 365,
        "holding_interval_censoring": "open positions at segment end are right-censored",
    }, indent=2), encoding="utf-8")
    (out / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    equity_pages = _equity_pages(out, portfolio, performance, spec, signal_spec)
    _models_page(out, forecast_comparison, risk_comparison, performance)
    _weights_page(out, dynamic["weight_rows"], factor_events, lookup, registry)
    _sleeves_page(out, portfolio, asset, spec)
    _sessions_page(out, asset_metrics, sessions, uniformity, holds)
    asset_pages = _asset_pages(out, asset, fills, holds, asset_metrics,
                               profile_events, profile_holds, spec, signal_spec)
    trigger_pages = _trigger_pages(out, asset, profile_events, profile_holds,
                                   profile_metrics, profile_asset_metrics,
                                   signal["calibration"], signal_spec)
    _index_page(out, performance, uniformity, equity_pages, asset_pages,
                factor_events, trigger_pages, profile_metrics, spec)
    return {"portfolio_rows": len(portfolio), "asset_rows": len(asset),
            "fills": len(fills), "closed_trades": int((~holds.right_censored).sum()),
            "factor_entry_exit_events": len(factor_events),
            "profile_closed_trades": int((~profile_holds.right_censored).sum()),
            "html_pages": 1 + len(equity_pages) + len(asset_pages) + len(trigger_pages) + 4}
