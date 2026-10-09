"""Read-only HTML atlas for a completed semi-structured manual-alpha study."""
from __future__ import annotations

import ast
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import yaml

from .factor_combo_reporting import _CSS, _plot, _table


def _page(out: Path, name: str, title: str, body: str, subtitle: str = "") -> None:
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = "../" if name.startswith("mechanisms/") else ""
    nav = (f'<nav><a href="{prefix}index.html">总览</a> · '
           f'<a href="{prefix}search.html">搜参</a> · '
           f'<a href="{prefix}data.html">数据</a> · '
           f'<a href="{prefix}execution.html">5m 执行</a> · '
           f'<a href="{prefix}method.html">研究口径</a></nav>')
    footer = ("完成小时决策，下一根 5m 起收益；12 币等权，市场 beta 正交；"
              "仓位变化每单位 4 bps。2026 段曾在早期项目诊断中被看过，非全新盲测。")
    doc = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
           f"<script src='{prefix}plotly.min.js'></script></head><body>"
           f"{nav}<main><h1>{html.escape(title)}</h1>"
           f"<p class='muted'>{html.escape(subtitle)}</p>{body}"
           f"<footer>{html.escape(footer)}</footer></main></body></html>")
    target.write_text(doc, encoding="utf-8")


def _period_curve(hourly: pd.DataFrame, cfg: dict) -> go.Figure:
    split = cfg["split"]
    date = hourly.decision_time.astype(str)
    period = np.where(date <= split["discovery_end"], "发现期",
                      np.where(date <= split["validation_end"], "验证期", "历史测试期"))
    fig = go.Figure()
    for key in ("发现期", "验证期", "历史测试期"):
        sub = hourly[period == key]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(sub.decision_time, format="%Y%m%d%H%M", utc=True),
            y=100 * np.exp(sub.net.cumsum()), mode="lines", name=key,
            line=dict(width=2)))
    fig.update_yaxes(title_text="各段净值，段初 = 100")
    return fig


def _results_plot(frame: pd.DataFrame) -> go.Figure:
    frame = frame.copy()
    fig = go.Figure()
    for col, title in (("discovery_gross_bps_per_hour", "发现价格毛"),
                       ("discovery_funding_bps_per_hour", "发现 funding"),
                       ("discovery_fee_bps_per_hour", "发现费"),
                       ("discovery_net_bps_per_hour", "发现净"),
                       ("validation_net_bps_per_hour", "验证净")):
        if col in frame:
            fig.add_trace(go.Bar(x=frame.mechanism, y=frame[col], name=title))
    fig.update_layout(barmode="group", xaxis_tickangle=-45)
    fig.update_yaxes(title_text="每币等权，每小时 bps")
    return fig


def _trial_plot(trials: pd.DataFrame, mechanism: str) -> go.Figure:
    sub = trials[trials.mechanism.eq(mechanism)].copy()
    value = pd.to_numeric(sub.objective, errors="coerce")
    valid = value > -1e5
    fig = go.Figure(go.Scatter(x=sub.loc[valid, "trial"], y=value[valid],
                              mode="markers+lines", name="发现期内折目标",
                              text=sub.loc[valid, "horizon_hours"].astype(str) + "h"))
    fig.update_yaxes(title_text="稳健净收益目标，bps/h")
    return fig


def build_manual_alpha_visualization(source: Path, destination: Path) -> dict:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination.exists() or source == destination:
        raise FileExistsError("visualization destination must be new")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("completed manual-alpha source is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or manifest.get("lane") != "manual_alpha":
        raise ValueError("source is not a completed manual-alpha study")
    required = ["config_snapshot.yaml", "selection_freeze.json", "mechanism_results.csv",
                "search_trials.csv", "data_audit.json", "combo_hourly_ledger.parquet"]
    missing = [x for x in required if not (source / x).is_file()]
    if missing:
        raise ValueError(f"manual-alpha evidence missing: {missing}")
    cfg = yaml.safe_load((source / "config_snapshot.yaml").read_text(encoding="utf-8"))
    freeze = json.loads((source / "selection_freeze.json").read_text(encoding="utf-8"))
    if freeze["signature"] != manifest["signature"]:
        raise ValueError("selection freeze does not match source signature")
    frame = pd.read_csv(source / "mechanism_results.csv")
    trials = pd.read_csv(source / "search_trials.csv")
    audit = json.loads((source / "data_audit.json").read_text(encoding="utf-8"))
    hourly = pd.read_parquet(source / "combo_hourly_ledger.parquet")
    building = destination.with_name(destination.name + "_building")
    if building.exists():
        raise FileExistsError(building)
    building.mkdir(parents=True)
    (building / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    selected = set(freeze["selected"])
    n_ok = int(frame.status.eq("EVALUATED").sum())
    body = (f"<div class='note'>20 个语义机制，{len(trials)} 次有界 Optuna 试验，"
            f"{n_ok} 个机制有效评估；验证期与发现期均达标的冻结入选数为 "
            f"<b>{len(selected)}</b>。未达标机制完整保留，不把失败改写成 Alpha。</div>")
    body += _plot(_results_plot(frame), "机制逐一比较：毛收益、费用与净收益", 520)
    if selected:
        body += _plot(_period_curve(hourly, cfg), "冻结组合各段净值；按对数收益复利")
    else:
        body += "<div class='note'>本轮没有符合发现与验证规则的组合，净值曲线为现金基准 100。</div>"
    cards = frame[["mechanism", "status", "discovery_objective",
                   "validation_net_bps_per_hour", "validation_positive_asset_share"]].copy()
    cards["selected"] = cards.mechanism.isin(selected)
    cards["semantic"] = np.where(frame.direction.eq(1), "as_declared",
                                  np.where(frame.direction.eq(-1), "inverse_hypothesis", "unresolved"))
    body += _table(cards)
    if "funding" in hourly:
        split = cfg["split"]
        stamp = hourly.decision_time.astype(str)
        period = np.where(stamp <= split["discovery_end"], "发现期",
            np.where(stamp <= split["validation_end"], "验证期", "历史测试期"))
        bridge = hourly.assign(period=period).groupby("period", sort=False)[["gross", "funding", "fee", "net"]].sum() * 100
        fig = go.Figure()
        for field, label in (("gross", "价格毛"), ("funding", "资金费"), ("fee", "交易费"), ("net", "净")):
            fig.add_trace(go.Bar(x=bridge.index, y=bridge[field], name=label))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="累计对数收益，%")
        body += _plot(fig, "逐阶段经济收益桥：价格 + funding − 费用")
    body += "<h2>逐机制研究卡</h2><div class='links'>" + "".join(
        f'<a href="mechanisms/{html.escape(name)}.html">{html.escape(name)}</a>'
        for name in frame.mechanism) + "</div>"
    _page(building, "index.html", "半结构化人工 Alpha 研究台", body,
          "2023—2025H1 发现 · 2025H2—2026-01 验证 · 2026-02 起历史测试")
    top = trials.copy()
    top["valid"] = pd.to_numeric(top.objective, errors="coerce") > -1e5
    summary = top.groupby("mechanism", sort=False).agg(trials=("trial", "size"),
        valid_trials=("valid", "sum"), best_objective=("objective", lambda s: s[s > -1e5].max())).reset_index()
    sb = "<div class='note'>每条机制独立固定种子 TPE；SQLite 保存完整试验。排序只读发现内折与验证期，测试期不参与。</div>"
    sb += _table(summary)
    sb += _plot(_results_plot(frame), "20 条机制：费后与费用桥", 520)
    _page(building, "search.html", "搜索与淘汰", sb)
    coverage = pd.DataFrame(audit["coverage"]).T.reset_index(names="field")
    db = _table(coverage)
    if audit.get("funding_interval_metadata_missing_events"):
        missing_meta = pd.DataFrame([audit["funding_interval_metadata_missing_events"]])
        db += "<h2>费率有效但 interval 元数据缺失的结算事件数</h2>" + _table(missing_meta)
    db += "<p>OI、多空比与未知发布时间的链上字段不进入本轮训练。mark/index 完成时间和 funding 结算时间分别对齐。</p>"
    _page(building, "data.html", "数据覆盖和来源", db)
    native_path = source / "combo_5m_ledger.parquet"
    if native_path.is_file():
        native = pd.read_parquet(native_path)
        native["day"] = native.completed_5m.astype(str).str[:8]
        daily = native.groupby("day", sort=False)[["price_gross", "funding", "fee", "net", "turnover"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(daily.day, format="%Y%m%d", utc=True),
                                 y=100 * np.exp(daily.net.cumsum()), name="5m 精确净值", mode="lines"))
        fig.update_yaxes(title_text="净值，期初 = 100")
        execution = ("<div class='note'>小时信号在完成小时后才进入 5m 仓位；该仓位承担下一根完整 5m 价格收益。"
                     "资金费率在计划结算点由此前持仓支付/收取；费用按每次实际净仓位变化扣除。</div>")
        execution += _plot(fig, "原生 5m 组合净值")
        summary = native.groupby("period", sort=False)[["price_gross", "funding", "fee", "net", "turnover"]].sum() * 100
        execution += _table(summary.reset_index())
        execution += _table(daily.tail(30), rows=30)
    else:
        execution = "<div class='note'>这一轮尚未保存原生 5m 账本；小时净值见总览。</div>"
    _page(building, "execution.html", "5m 原生执行与费用", execution)
    direction_text = ("2023H1 固定方向" if cfg["manual_alpha"].get("direction_policy", "initial_fixed") == "initial_fixed"
                      else "发现期每半年只用此前成熟标签扩展估计方向，验证前最终冻结")
    funding_text = ("实际结算 funding 现金流计入账本" if cfg["manual_alpha"].get("funding_cashflow", False)
                    else "资金费率只作已知状态，主账本不含结算现金流")
    method = ("<div class='note'>一条机制只搜窗口、事件阈值、激活门限和预先声明持有期。"
              f"方向规则：{html.escape(direction_text)}；量纲尺度来自 2023H1；四个发现内折用于成本后目标，验证期检验生存，"
              "测试期只在选择文件冻结后计算。组合取入选等权，持仓按实际变仓扣费。"
              f"{html.escape(funding_text)}。</div>")
    method += _table(pd.DataFrame(cfg["manual_alpha"]["mechanisms"])[["id", "hypothesis", "failure", "dependencies"]])
    _page(building, "method.html", "机制与评估合同", method)
    for card in cfg["manual_alpha"]["mechanisms"]:
        name = card["id"]
        row = frame[frame.mechanism.eq(name)].iloc[0]
        b = (f"<div class='note'><b>假设：</b>{html.escape(card['hypothesis'])}<br>"
             f"<b>伪证据/失效：</b>{html.escape(card['failure'])}<br>"
             f"<b>字段：</b>{html.escape(', '.join(card['dependencies']))}<br>"
             f"<b>语义方向：</b>{'与原假设一致' if row.direction == 1 else '原假设反向或未校准'}<br>"
             f"<b>冻结入选：</b>{'是' if name in selected else '否'}</div>")
        columns = ["status", "window", "threshold", "horizon_hours", "direction",
                   "discovery_gross_bps_per_hour", "discovery_fee_bps_per_hour",
                   "discovery_net_bps_per_hour", "validation_gross_bps_per_hour",
                   "validation_fee_bps_per_hour", "validation_net_bps_per_hour",
                   "validation_positive_asset_share"]
        if "discovery_funding_bps_per_hour" in row:
            columns.insert(columns.index("discovery_fee_bps_per_hour"), "discovery_funding_bps_per_hour")
            columns.insert(columns.index("validation_fee_bps_per_hour"), "validation_funding_bps_per_hour")
        b += _table(pd.DataFrame([row])[columns])
        if "fold_directions" in row and pd.notna(row.fold_directions):
            b += f"<p>发现期各折实用方向：<code>{html.escape(str(row.fold_directions))}</code></p>"
        b += _plot(_trial_plot(trials, name), "全部有效 trial 的发现期目标")
        if row.status == "EVALUATED" and pd.notna(row.fold_net_bps_per_hour):
            values = ast.literal_eval(row.fold_net_bps_per_hour)
            fig = go.Figure(go.Bar(x=["2023H2", "2024H1", "2024H2", "2025H1"], y=values))
            fig.update_yaxes(title_text="费后 bps/h")
            b += _plot(fig, "发现期逐块生存")
        _page(building, f"mechanisms/{name}.html", f"机制 · {name}", b)
    pages = sorted(str(x.relative_to(building)).replace("\\", "/") for x in building.rglob("*.html"))
    result = {"status": "COMPLETED", "source": str(source),
              "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "page_count": len(pages), "pages": pages,
              "contract": "read-only, post-freeze visualization"}
    (building / "visualization_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    building.rename(destination)
    return result
