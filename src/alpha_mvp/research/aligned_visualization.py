"""Multi-page, read-only visualization of a completed aligned crypto experiment.

The research source is immutable. This renderer creates a separate evidence
directory and reuses the combo report's Plotly/CSS presentation primitives.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from .factor_combo_reporting import _CSS, _plot, _table


PERIODS = ("discovery", "validation", "holdout")
STRATEGIES = ("main", "alpha", "beta", "native5")
def _period(dates: pd.Series | np.ndarray, split: dict) -> np.ndarray:
    values = np.asarray(dates).astype(str)
    return np.where(values <= split["discovery_end"], "discovery",
                    np.where(values <= split["validation_end"], "validation", "holdout"))


def _period_labels(cfg: dict, manifest: dict) -> dict[str, str]:
    fmt = lambda s: f"{str(s)[:4]}-{str(s)[4:6]}"
    return {"discovery": f"发现期 {fmt(str(cfg['data']['start']).replace('-', ''))}—{fmt(cfg['split']['discovery_end'])}",
            "validation": f"验证期 {fmt(cfg['split']['validation_start'])}—{fmt(cfg['split']['validation_end'])}",
            "holdout": f"测试期 {fmt(cfg['split']['holdout_start'])}—{fmt(manifest['actual_data_end'])}"}


def _dates(values) -> pd.DatetimeIndex:
    return pd.to_datetime(values, format="%Y%m%d%H%M", utc=True)


def _pct(log_return: float) -> str:
    return f"{100 * np.expm1(log_return):+.2f}%"


def _links(items: list[tuple[str, str]], prefix: str = "") -> str:
    return '<div class="links">' + ''.join(
        f'<a href="{prefix}{html.escape(target)}">{html.escape(label)}</a>'
        for label, target in items) + '</div>'


def _page(out: Path, name: str, title: str, body: str, subtitle: str = "") -> None:
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = "../" if name.startswith("assets/") else ""
    nav = _links([("总览", "index.html"), ("搜索漏斗", "search.html"),
                  ("入选因子", "factors.html"), ("Alpha / Beta", "sleeves.html"),
                  ("风险与 5m", "risk_native.html"), ("品种与时段", "breadth.html")], prefix)
    footer = ("研究时钟：完成 bar 决策，承担下一根完整 5m 对数收益；逐币等权，"
              "仓位每单位变动扣 4 bps。方向与幅度来自发现期，验证期选择，测试期只审计。"
              "本轮测试段曾用于早期诊断，不是研究全过程的全新盲样本。")
    doc = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
           f"<script src='{prefix}plotly.min.js'></script></head><body>"
           f"<nav>{nav}</nav><main><h1>{html.escape(title)}</h1>"
           f"<p class='muted'>{html.escape(subtitle)}</p>{body}"
           f"<footer>{html.escape(footer)}</footer></main></body></html>")
    target.write_text(doc, encoding="utf-8")


def _daily(portfolio: pd.DataFrame, split: dict) -> pd.DataFrame:
    frame = portfolio.copy()
    frame["period"] = _period(frame.date, split)
    frame["day"] = frame.date.str[:8]
    return frame.groupby(["period", "day", "strategy"], sort=False).agg(
        gross=("gross", "sum"), fee=("fee", "sum"), net=("net", "sum"),
        abs_position=("abs_position", "mean")).reset_index()


def _curve(frame: pd.DataFrame, names=STRATEGIES) -> go.Figure:
    fig = go.Figure()
    for name in names:
        part = frame[frame.strategy.eq(name)].sort_values("day")
        if part.empty:
            continue
        fig.add_trace(go.Scatter(x=pd.to_datetime(part.day, format="%Y%m%d", utc=True),
                                 y=100 * np.exp(part.net.cumsum()), name=name,
                                 mode="lines", line=dict(width=2)))
    fig.update_yaxes(title_text="净值，期初 = 100")
    return fig


def _metric_rows(summary: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = summary[summary.period.eq(period)].copy()
    frame["net_return_pct"] = 100 * frame.simple_return
    frame["gross_log_pct"] = 100 * frame.gross_log_return
    frame["fee_log_pct"] = 100 * frame.fee_log_return
    frame["max_drawdown_pct"] = 100 * frame.max_drawdown
    frame["break_even_bps"] = np.where(frame.fee_log_return > 0,
                                        4 * frame.gross_log_return / frame.fee_log_return,
                                        np.nan)
    return frame[["strategy", "net_return_pct", "gross_log_pct", "fee_log_pct",
                  "daily_sharpe", "max_drawdown_pct", "mean_abs_position",
                  "break_even_bps"]]


def _search_page(out: Path, source: Path, manifest: dict, selected: pd.DataFrame) -> None:
    coarse = pd.read_csv(source / "search_results.csv")
    medium = pd.read_csv(source / "medium_results.csv")
    fine = pd.read_csv(source / "fine_results.csv")
    native = pd.read_csv(source / "native5_fine.csv")
    fields = pd.read_csv(source / "field_baselines.csv")
    role_order = ["alpha", "conditional_alpha", "beta", "risk"]
    counts = pd.DataFrame({
        "role": role_order,
        "coarse": [int(coarse.role.eq(r).sum()) for r in role_order],
        "medium": [int(medium.role.eq(r).sum()) for r in role_order],
        "fine_generated": [int(fine.role.eq(r).sum()) for r in role_order],
        "selected": [int(selected.role.eq(r).sum()) for r in role_order],
    })
    fig = go.Figure()
    for stage in ["coarse", "medium", "fine_generated", "selected"]:
        fig.add_trace(go.Bar(name=stage, x=counts.role, y=counts[stage]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="候选数；中筛为表达式 × 期限")
    body = ("<div class='note'>粗筛和中筛只读取发现期；细筛用验证期扣费收益或风险预测增量。"
            "原始字段基线与复杂式同合同竞争。测试期指标没有进入搜索或排序。</div>")
    body += _plot(fig, "主赛道搜索漏斗：用途分层") + _table(counts)
    family = selected.groupby(["role", "template_name"], sort=False).size().reset_index(name="count")
    ffig = go.Figure(go.Bar(x=family.template_name, y=family["count"],
                            marker_color="#2b719b", text=family.role))
    ffig.update_xaxes(tickangle=-35)
    body += _plot(ffig, "冻结入选因子的模板归因", 460)
    histogram = go.Figure()
    for role in role_order:
        vals = pd.to_numeric(medium.loc[medium.role.eq(role), "signed_ic"], errors="coerce")
        histogram.add_trace(go.Histogram(x=vals[np.isfinite(vals)], name=role, opacity=.55))
    histogram.update_layout(barmode="overlay")
    histogram.update_xaxes(title_text="发现期有向 IC；仅同目标内比较")
    body += _plot(histogram, "中筛发现期信息分布", 430)
    bh = go.Figure()
    for role in ["alpha", "beta"]:
        sub = fields[fields.role.eq(role)]
        bh.add_trace(go.Box(x=sub.horizon_bars / 12, y=sub.validation_net_mean_bar_bps,
                            name=role, boxpoints=False))
    bh.update_xaxes(title_text="声明持有小时")
    bh.update_yaxes(title_text="验证期单因子净收益 bps / 小时")
    body += _plot(bh, "原始字段 × 用途 × 期限：成本后的分布")
    body += "<h2>原始字段验证期最佳二十项（仅验证期排序）</h2>"
    raw_direction = fields[fields.role.isin(["alpha", "beta"])].sort_values(
        "validation_net_mean_bar_bps", ascending=False)
    body += _table(raw_direction, ["field", "role", "horizon_bars", "discovery_ic",
                                    "validation_net_mean_bar_bps",
                                    "validation_break_even_cost_bps"], rows=20)
    body += (f"<h2>原生 5m 赛道</h2><p>粗／中／细分别为 "
             f"{manifest['counts']['native5']['coarse']}／{manifest['counts']['native5']['medium']}／"
             f"{len(native)}；在独立页面查看预测幅度和交易门槛。</p>")
    _page(out, "search.html", "万级挖掘｜搜索漏斗与筛选", body,
          "表达式、用途、条件和期限共同定义一个研究候选")


def _factor_page(out: Path, source: Path, selected: pd.DataFrame) -> None:
    audit = pd.read_csv(source / "holdout_factor_audit.csv")
    frame = selected.merge(audit[["candidate_id", "holdout_gross_hour_bps",
                                   "holdout_delta_r2"]], on="candidate_id", how="left")
    frame = frame.sort_values(["role", "validation_net_mean_bar_bps"],
                              ascending=[True, False], na_position="last")
    body = ("<div class='note'>下表按冻结时的用途和验证期指标展示。测试期单因子毛收益与验证期净收益"
            "口径不同，只作为冻结后的审计，不重新排名。风险因子用增量 R²，不与方向 bps 混排。</div>")
    counts = frame.groupby(["role", "template_name"], sort=False).size().reset_index(name="count")
    heat = counts.pivot(index="role", columns="template_name", values="count").fillna(0)
    fig = go.Figure(go.Heatmap(x=heat.columns, y=heat.index, z=heat.to_numpy(),
                               colorscale="Blues", colorbar=dict(title="入选数")))
    body += _plot(fig, "用途 × 模板：冻结因子结构", 420)
    for role in ["alpha", "conditional_alpha", "beta", "risk"]:
        sub = frame[frame.role.eq(role)].copy()
        body += f"<h2>{html.escape(role)}｜{len(sub)} 个</h2>"
        cols = ["expr", "template_name", "fields", "horizon_bars", "coverage",
                "signed_ic", "positive_asset_share", "validation_net_mean_bar_bps",
                "validation_delta_r2", "holdout_gross_hour_bps", "holdout_delta_r2"]
        body += _table(sub, cols, rows=50)
    _page(out, "factors.html", "冻结入选因子｜公式与样本外审计", body,
          "保留公式、字段、期限、发现期信息、验证期交易价值与测试期审计")


def _period_page(out: Path, period: str, daily: pd.DataFrame,
                 summary: pd.DataFrame, asset_period: pd.DataFrame,
                 labels: dict) -> None:
    sub = daily[daily.period.eq(period)]
    main = sub[sub.strategy.eq("main")].sort_values("day")
    body = ("<div class='note'>所有净值在本段起点重置为 100；Alpha、Beta、主组合分别展示。"
            "买入持有没有并入本图，避免仓位尺度误读。5m 逐笔原始记录见源实验账本。</div>")
    body += _plot(_curve(sub), "Alpha / Beta / 主组合：费后净值", 480)
    dfig = go.Figure(go.Bar(x=pd.to_datetime(main.day, format="%Y%m%d", utc=True),
                            y=100 * np.expm1(main.net),
                            marker_color=np.where(main.net >= 0, "#167b5a", "#bd5050")))
    dfig.update_yaxes(title_text="主组合每日净收益 %")
    body += _plot(dfig, "日收益：正负日期", 380)
    month = main.assign(month=main.day.str[:6]).groupby("month")[["gross", "fee", "net"]].sum()
    mfig = go.Figure()
    for col, title in [("gross", "毛收益"), ("fee", "费用"), ("net", "净收益")]:
        mfig.add_trace(go.Bar(x=month.index, y=100 * month[col], name=title))
    mfig.update_layout(barmode="group")
    mfig.update_yaxes(title_text="月对数收益 %；费用为正成本")
    body += _plot(mfig, "月度毛收益—费用—净收益", 390)
    draw = np.exp(main.net.cumsum().to_numpy())
    draw = draw / np.maximum.accumulate(np.r_[1., draw])[1:] - 1
    fig = go.Figure(go.Scatter(x=pd.to_datetime(main.day, format="%Y%m%d", utc=True),
                               y=100 * draw, fill="tozeroy", name="主组合回撤"))
    fig.update_yaxes(title_text="回撤 %")
    body += _plot(fig, "日频峰谷回撤")
    byasset = asset_period[asset_period.period.eq(period)].sort_values("net_contribution_pct")
    bar = go.Figure(go.Bar(x=byasset.asset, y=byasset.net_contribution_pct,
                           marker_color=np.where(byasset.net_contribution_pct >= 0,
                                                 "#167b5a", "#bd5050")))
    bar.update_yaxes(title_text="组合净对数收益贡献，百分点")
    body += _plot(bar, "12 币贡献：原始收益减变仓费")
    body += "<h2>费用、收益与仓位</h2>" + _table(_metric_rows(summary, period))
    body += _links([("执行与仓位页", f"execution_{period}.html")])
    _page(out, f"equity_{period}.html", f"{labels[period]}｜净值与成本", body,
          "年度/阶段独立页面；所有收益均来自同一 5m 精确账本")


def _sleeves_page(out: Path, daily: pd.DataFrame, decisions: pd.DataFrame,
                  summary: pd.DataFrame, freeze: dict, labels: dict) -> None:
    body = ("<div class='note'>Alpha 是逐时现金中性币篮子，Beta 是共同方向；风险预算只改变规模。"
            "名义预算 0.7 / 0.3 不等于实际平均仓位。风险和原生 5m 开关在测试前由验证期冻结。</div>")
    body += (f"<div class='grid'><div class='stat'>风险预算<b>{'启用' if freeze['risk_enabled'] else '关闭'}</b></div>"
             f"<div class='stat'>原生 5m 增量<b>{'启用' if freeze['native_enabled'] else '关闭'}</b></div>"
             f"<div class='stat'>冻结候选<b>{len(freeze['selected_candidates'])}</b>含风险预测因子</div></div>")
    for period in PERIODS:
        body += f"<h2>{labels[period]}</h2>"
        body += _plot(_curve(daily[daily.period.eq(period)], ("alpha", "beta", "main")),
                      "分通道费后净值", 430)
        dec = decisions[decisions.period.eq(period)].iloc[::24]
        fig = go.Figure()
        for col, label in [("alpha_abs", "Alpha 绝对仓位"),
                           ("beta_abs", "Beta 绝对仓位"), ("main_abs", "主组合绝对仓位")]:
            fig.add_trace(go.Scatter(x=_dates(dec.date), y=dec[col], name=label))
        fig.update_yaxes(title_text="平均绝对仓位")
        body += _plot(fig, "资本占用：名义预算与实际仓位的差别")
        monthly = daily[daily.period.eq(period)].copy()
        monthly["month"] = monthly.day.str[:6]
        monthly = monthly.groupby(["month", "strategy"]).net.sum().reset_index()
        bar = go.Figure()
        for name in ["alpha", "beta", "main"]:
            s = monthly[monthly.strategy.eq(name)]
            bar.add_trace(go.Bar(x=s.month, y=100*s.net, name=name))
        bar.update_layout(barmode="group")
        bar.update_yaxes(title_text="月净对数收益 %")
        body += _plot(bar, "月度 Alpha / Beta / 主组合")
        body += _table(_metric_rows(summary, period))
    _page(out, "sleeves.html", "Alpha、Beta 与预算协作", body,
          "归因、实际仓位和成本在发现/验证/测试三个阶段连续展示")


def _risk_native_page(out: Path, source: Path, selected: pd.DataFrame,
                      freeze: dict) -> None:
    risk = selected[selected.role.eq("risk")].copy()
    audit = pd.read_csv(source / "holdout_factor_audit.csv")
    risk = risk.merge(audit[["candidate_id", "holdout_delta_r2"]], on="candidate_id")
    native = pd.read_csv(source / "native5_selected.csv")
    body = ("<div class='note'>风险预测的增量 R² 与交易净收益分开。原生 5m 候选虽有方向 IC，"
            "预测累计收益幅度未覆盖 4 bps 单边变仓成本，所以本次没有形成可交易仓位；"
            "不能由此断言 5m 信息不存在。</div>")
    fig = go.Figure()
    short = risk.expr.str.slice(0, 38)
    fig.add_trace(go.Bar(x=short, y=100*risk.validation_delta_r2,
                         name="验证期增量 R²"))
    fig.add_trace(go.Bar(x=short, y=100*risk.holdout_delta_r2,
                         name="测试期增量 R²"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="增量 R²，百分点")
    body += _plot(fig, "下行风险预测：验证与测试", 440)
    body += _table(risk, ["expr", "horizon_bars", "validation_delta_r2",
                          "holdout_delta_r2"])
    scale = 1e4 * native.prediction_scale
    nfig = go.Figure(go.Bar(x=native.expr.str.slice(0, 28), y=scale,
                            name="发现期 90 分位预测幅度"))
    nfig.add_hline(y=8, line_dash="dash", line_color="#bd5050",
                   annotation_text="4 bps 单边的往返参考线 8 bps")
    nfig.update_yaxes(title_text="预测累计收益 bps")
    body += _plot(nfig, "原生 5m：预测幅度与交易门槛", 440)
    body += _table(native, ["expr", "horizon_bars", "signed_ic",
                            "positive_asset_share", "validation_net_mean_5m_bps",
                            "prediction_scale"])
    body += ("<h2>冻结的验证期预算比较</h2>" + _table(pd.DataFrame([{
        "base_net_mean_5m_bps": freeze["validation_native_base_net_mean_5m_bps"],
        "with_native_5m_bps": freeze["validation_native_augmented_net_mean_5m_bps"],
        "with_risk_5m_bps": freeze["validation_risk_net_mean_5m_bps"],
        "native_enabled": freeze["native_enabled"],
        "risk_enabled": freeze["risk_enabled"],
    }])))
    _page(out, "risk_native.html", "风险预测与原生 5m 信息", body,
          "预测质量、成本盈亏平衡和预算开关是三个不同判断")


def _asset_period_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.copy()
    frame["net_contribution"] = frame.net / frame.asset.nunique()
    group = frame.groupby(["period", "asset"], sort=False).agg(
        gross=("gross", "sum"), fee=("fee", "sum"), net=("net", "sum"),
        net_contribution=("net_contribution", "sum"),
        mean_abs_position=("position", lambda s: float(s.abs().mean())),
        long_fraction=("position", lambda s: float((s > .1).mean())),
        short_fraction=("position", lambda s: float((s < -.1).mean()))).reset_index()
    group["net_contribution_pct"] = 100 * group.net_contribution
    group["standalone_simple_pct"] = 100 * np.expm1(group.net)
    return group


def _breadth_page(out: Path, ledger: pd.DataFrame,
                  asset_period: pd.DataFrame, labels: dict) -> None:
    # Convert unique 5m timestamps once; map the resulting session labels to rows.
    unique = pd.Index(ledger.date.unique())
    ny_hour = _dates(unique).tz_convert("America/New_York").hour
    session = np.where((ny_hour >= 8) & (ny_hour < 16), "NY_day",
                       np.where(ny_hour >= 16, "NY_evening", "NY_overnight"))
    session_by_date = pd.Series(session, index=unique)
    frame = ledger[["date", "asset", "net"]].copy()
    frame["period"] = ledger.period
    frame["session"] = frame.date.map(session_by_date)
    frame["portfolio_contribution"] = frame.net / ledger.asset.nunique()
    grouped = frame.groupby(["period", "asset", "session"], sort=False).portfolio_contribution.sum().reset_index()
    body = ("<div class='note'>逐币贡献是原始逐币净对数收益除以 12，不是逐币单独满仓收益。"
            "纽约时段只做冻结后的归因；不改变候选选择。多空比例用 |仓位| > 0.1 的展示阈值。</div>")
    for period in PERIODS:
        body += f"<h2>{labels[period]}</h2>"
        part = grouped[grouped.period.eq(period)]
        pivot = part.pivot(index="asset", columns="session", values="portfolio_contribution")
        fig = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index,
                                   z=100 * pivot.to_numpy(), colorscale="RdBu", zmid=0,
                                   colorbar=dict(title="净贡献百分点")))
        body += _plot(fig, "逐币 × 纽约时段的组合贡献", 450)
        ap = asset_period[asset_period.period.eq(period)]
        body += _table(ap, ["asset", "net_contribution_pct", "standalone_simple_pct",
                            "mean_abs_position", "long_fraction", "short_fraction"], 20)
    _page(out, "breadth.html", "品种与纽约时段广度", body,
          "组合贡献、单币路径和持仓方向用不同单位标示")


def _execution_page(out: Path, period: str, ledger: pd.DataFrame,
                    decisions: pd.DataFrame, labels: dict) -> None:
    part = ledger[ledger.period.eq(period)].copy()
    dec = decisions[decisions.period.eq(period)].iloc[::12]
    fig = go.Figure()
    for col, name in [("alpha_abs", "Alpha"), ("beta_abs", "Beta"),
                      ("main_abs", "主组合")]:
        fig.add_trace(go.Scatter(x=_dates(dec.date), y=dec[col], name=name))
    fig.update_yaxes(title_text="每币平均绝对仓位")
    body = ("<div class='note'>仓位变化按主组合净仓位逐币扣费；可见开平仓标记采用 |仓位| > 0.1，"
            "是展示口径，不冒充独立状态机的自然完成交易笔数。原生 5m 与风险预算在本实验冻结关闭。</div>")
    body += _plot(fig, "完成小时决策后的实际资本占用", 440)
    part["abs_position"] = part.position.abs()
    part["long_indicator"] = (part.position > .1).astype(np.float32)
    part["short_indicator"] = (part.position < -.1).astype(np.float32)
    bydate = part.groupby("date", sort=False).agg(
        fee=("fee", "mean"), gross=("gross", "mean"), net=("net", "mean"),
        signed=("position", "mean"), gross_position=("abs_position", "mean"),
        long_fraction=("long_indicator", "mean"),
        short_fraction=("short_indicator", "mean")).reset_index()
    bydate["day"] = bydate.date.str[:8]
    daily = bydate.groupby("day", sort=False).agg(
        fee=("fee", "sum"), gross=("gross", "sum"), net=("net", "sum"),
        signed=("signed", "mean"), gross_position=("gross_position", "mean"),
        long_fraction=("long_fraction", "mean"),
        short_fraction=("short_fraction", "mean")).reset_index()
    bar = go.Figure()
    bar.add_trace(go.Bar(x=pd.to_datetime(daily.day, format="%Y%m%d", utc=True),
                         y=1e4*daily.gross, name="毛收益"))
    bar.add_trace(go.Bar(x=pd.to_datetime(daily.day, format="%Y%m%d", utc=True),
                         y=-1e4*daily.fee, name="费用扣减"))
    bar.update_layout(barmode="relative")
    bar.update_yaxes(title_text="日对数收益 bps")
    body += _plot(bar, "每日毛收益与费用桥", 430)
    lines = go.Figure()
    x = pd.to_datetime(daily.day, format="%Y%m%d", utc=True)
    for col, name in [("gross_position", "绝对仓位"), ("signed", "净方向")]:
        lines.add_trace(go.Scatter(x=x, y=daily[col], name=name))
    lines.update_yaxes(title_text="仓位单位")
    body += _plot(lines, "平均仓位强度与共同方向")
    frac = go.Figure()
    for col, name in [("long_fraction", "可见多头"), ("short_fraction", "可见空头")]:
        frac.add_trace(go.Scatter(x=x, y=100*daily[col], name=name))
    frac.update_yaxes(title_text="币时比例 %")
    body += _plot(frac, "多空参与覆盖率")
    sampled_dates = part.date.drop_duplicates().iloc[
        ::144 if period == "discovery" else 36]
    pivot = part[part.date.isin(sampled_dates)].pivot(
        index="asset", columns="date", values="position")
    heat = go.Figure(go.Heatmap(x=_dates(pivot.columns), y=pivot.index,
                                z=pivot.to_numpy(), colorscale="RdBu", zmid=0,
                                colorbar=dict(title="仓位")))
    body += _plot(heat, "逐币仓位热图：图形抽样，原账本仍为 5m 全量", 490)
    body += "<h2>逐日执行数据（前 60 行）</h2>" + _table(daily, rows=60)
    _page(out, f"execution_{period}.html", f"{labels[period]}｜持仓与执行", body,
          "费用、仓位、方向与品种覆盖均由主组合逐币 5m 账本复算")


def _asset_pages(out: Path, ledger: pd.DataFrame,
                 asset_period: pd.DataFrame, labels: dict) -> list[str]:
    pages = []
    for asset, frame in ledger.groupby("asset", sort=False):
        for period in PERIODS:
            part = frame[frame.period.eq(period)].copy()
            if part.empty:
                continue
            part["price_index"] = 100 * np.exp(part.next_log_return.fillna(0).cumsum())
            part["net_index"] = 100 * np.exp(part.net.cumsum())
            part["day"] = part.date.str[:8]
            stride = max(1, len(part) // 2000)
            view = part.iloc[::stride]
            entry = ((part.position.abs() > .1) &
                     (part.position.shift(fill_value=0).abs() <= .1))
            exit_ = ((part.position.abs() <= .1) &
                     (part.position.shift(fill_value=0).abs() > .1))
            body = ("<div class='note'>价格指数以本段第一根可用价格为 100，由原始 5m 对数收益重建；"
                    "单币净值为该币单独承担其记录仓位的路径，不等于其对等权组合的贡献。"
                    "开平标记只用 |仓位| > 0.1 的可见阈值。</div>")
            x = _dates(view.date)
            price = go.Figure(go.Scatter(x=x, y=view.price_index, name="价格指数"))
            price.update_yaxes(title_text="价格指数，期初 100")
            body += _plot(price, "价格路径（单独 y 轴）", 400)
            pnl = go.Figure(go.Scatter(x=x, y=view.net_index, name="单币费后净值"))
            pnl.update_yaxes(title_text="单币策略净值，期初 100")
            body += _plot(pnl, "该币仓位的费后净值（单独 y 轴）", 400)
            pos = go.Figure(go.Scatter(x=x, y=view.position, name="持仓"))
            for marker, name, color in [(entry, "可见开仓", "#167b5a"),
                                        (exit_, "可见平仓", "#bd5050")]:
                event = part.loc[marker].iloc[::max(1, int(marker.sum()) // 500)]
                pos.add_trace(go.Scatter(x=_dates(event.date), y=event.position,
                                         mode="markers", name=name,
                                         marker=dict(size=6, color=color)))
            pos.update_yaxes(title_text="有向仓位")
            body += _plot(pos, "实际仓位与展示阈值事件", 460)
            daily = part.groupby("day", sort=False)[["gross", "fee", "net"]].sum().reset_index()
            bar = go.Figure(go.Bar(x=pd.to_datetime(daily.day, format="%Y%m%d", utc=True),
                                   y=100*np.expm1(daily.net),
                                   marker_color=np.where(daily.net >= 0,
                                                         "#167b5a", "#bd5050")))
            bar.update_yaxes(title_text="单币策略日净收益 %")
            body += _plot(bar, "该币的每日费后收益")
            stats = asset_period[(asset_period.period.eq(period)) &
                                 (asset_period.asset.eq(asset))]
            body += "<h2>单币与组合贡献</h2>" + _table(stats)
            body += (f"<p>可见开仓标记 {int(entry.sum())} 次，平仓标记 {int(exit_.sum())} 次；"
                     "完整换仓与逐根费用见源实验 execution_ledger_5m.parquet。</p>")
            name = f"assets/{asset}_{period}.html"
            _page(out, name, f"{asset} · {labels[period]}", body,
                  "价格、仓位、执行与收益分轴展示")
            pages.append(name)
    return pages


def build_aligned_visualization(source: Path, destination: Path,
                                plotly_asset: Path) -> dict:
    """Render frozen research evidence without changing or rerunning the source."""
    source = Path(source).resolve(); destination = Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("visualization destination must be a new directory")
    building = destination.with_name(destination.name + "_building")
    if building.exists() and (building / "visualization_manifest.json").exists():
        raise ValueError("completed visualization build directory already exists")
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("completed source manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or manifest.get("mode") != "aligned_crypto":
        raise ValueError("source must be a completed aligned_crypto experiment")
    config_path = Path(manifest["config_path"])
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = cfg["split"]
    labels = _period_labels(cfg, manifest)
    required = ["research_freeze.json", "strategy_metrics.csv", "selected_factors.csv",
                "search_results.csv", "medium_results.csv", "fine_results.csv",
                "field_baselines.csv", "native5_fine.csv", "native5_selected.csv",
                "holdout_factor_audit.csv", "portfolio_5m.parquet",
                "hourly_decisions.parquet", "execution_ledger_5m.parquet"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise ValueError(f"source evidence missing: {missing}")
    freeze = json.loads((source / "research_freeze.json").read_text(encoding="utf-8"))
    if (freeze.get("status") != "FROZEN_BEFORE_HOLDOUT" or
            freeze.get("signature") != manifest.get("signature")):
        raise ValueError("research freeze does not match completed manifest")
    if not Path(plotly_asset).is_file():
        raise ValueError("local plotly.min.js asset is required")
    summary = pd.read_csv(source / "strategy_metrics.csv")
    selected = pd.read_csv(source / "selected_factors.csv")
    portfolio = pd.read_parquet(source / "portfolio_5m.parquet")
    decisions = pd.read_parquet(source / "hourly_decisions.parquet")
    ledger = pd.read_parquet(source / "execution_ledger_5m.parquet")
    if not np.allclose(ledger.gross - ledger.fee, ledger.net, equal_nan=True,
                       rtol=0, atol=1e-12):
        raise ValueError("source ledger gross - fee != net")
    bydate = ledger.groupby("date", sort=False)[["gross", "fee", "net"]].mean()
    main = portfolio[portfolio.strategy.eq("main")].set_index("date")
    if not np.allclose(bydate.loc[main.index], main[["gross", "fee", "net"]],
                       rtol=0, atol=1e-12):
        raise ValueError("asset ledger does not reconcile to main portfolio")
    daily = _daily(portfolio, split)
    decisions["period"] = _period(decisions.date, split)
    ledger["period"] = _period(ledger.date, split)
    asset_period = _asset_period_summary(ledger)
    building.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plotly_asset, building / "plotly.min.js")
    _search_page(building, source, manifest, selected)
    _factor_page(building, source, selected)
    _sleeves_page(building, daily, decisions, summary, freeze, labels)
    _risk_native_page(building, source, selected, freeze)
    _breadth_page(building, ledger, asset_period, labels)
    for period in PERIODS:
        _period_page(building, period, daily, summary, asset_period, labels)
        _execution_page(building, period, ledger, decisions, labels)
    asset_pages = _asset_pages(building, ledger, asset_period, labels)
    stage = manifest["counts"]
    cards = []
    for period in PERIODS:
        m = summary[(summary.strategy.eq("main")) & (summary.period.eq(period))].iloc[0]
        cards.append(f'<div class="stat">{labels[period]}<b>{_pct(m.net_log_return)}</b>'
                     f'日 Sharpe {m.daily_sharpe:.2f} · 最大回撤 {100*m.max_drawdown:.2f}%</div>')
    body = (f"<div class='note'>这份总览只呈现已完成的 {html.escape(source.name)} 实验："
            f"{stage['coarse']:,} 个小时表达式与 {stage['native5']['coarse']:,} 个原生 5m 表达式。"
            "验证期强 Beta 在测试期成本后失效；Alpha 有弱正收益但实际配置仓位偏低。"
            "图表为研究证据，不据测试期结果重新选择因子或阈值。</div>")
    body += '<div class="grid">' + ''.join(cards) + '</div>'
    body += _plot(_curve(daily, ("main", "alpha", "beta")),
                  "三个阶段连续日频展示；各阶段净值在下方独立页面重置", 470)
    body += (f"<div class='grid'><div class='stat'>主赛道粗筛<b>{stage['coarse']:,}</b></div>"
             f"<div class='stat'>中筛候选 × 期限<b>{stage['medium']:,}</b></div>"
             f"<div class='stat'>细筛含原始字段<b>{stage['fine']:,}</b></div>"
             f"<div class='stat'>冻结 Alpha/Beta/Risk<b>{stage['selected']}</b></div></div>")
    body += "<h2>分阶段净值和执行</h2>" + _links([
        (labels[p] + " · 净值", f"equity_{p}.html") for p in PERIODS] +
        [(labels[p] + " · 执行", f"execution_{p}.html") for p in PERIODS])
    body += "<h2>研究机制</h2>" + _links([
        ("搜索漏斗", "search.html"), ("入选因子公式", "factors.html"),
        ("Alpha/Beta 协作", "sleeves.html"), ("风险与原生 5m", "risk_native.html"),
        ("品种与纽约时段", "breadth.html")])
    body += "<h2>逐币页面</h2>" + _links([
        (f"{asset} · {period}", f"assets/{asset}_{period}.html")
        for asset in sorted(ledger.asset.unique()) for period in PERIODS])
    body += ("<h2>证据源</h2><p>原始完整实验目录："
             f"<code>{html.escape(str(source))}</code>。图表不修改该目录；"
             "研究细节见 docs/CRYPTO_ALIGNED_FACTOR_RESEARCH_20260930.md。</p>")
    _page(building, "index.html", "一体化万级因子挖掘与组合研究台", body,
          f"{len(manifest['codes'])} 币 · 1h 主搜索 + 原生 5m 辅助 · 逐期证据")
    pages = sorted(str(p.relative_to(building)).replace("\\", "/")
                   for p in building.rglob("*.html"))
    result = {"status": "COMPLETED", "source": str(source),
              "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_signature": manifest["signature"], "pages": pages,
              "page_count": len(pages), "plotly_asset": "plotly.min.js",
              "contract": "read-only post-freeze visualization; no selection or tuning"}
    (building / "visualization_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    building.rename(destination)
    return result
