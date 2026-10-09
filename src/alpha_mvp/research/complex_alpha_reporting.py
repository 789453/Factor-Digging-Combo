"""Read-only atlas reusing the project's existing Plotly/CSS reporting framework."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .factor_combo_reporting import _CSS,_plot,_table


def render_complex_report(source: Path,out: Path):
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETED':raise ValueError('only completed frozen evidence may be visualized')
    if (out/'visualization_manifest.json').exists():raise ValueError('completed visualization immutable')
    out.mkdir(parents=True,exist_ok=True);(out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    results=pd.read_csv(source/'model_survival.csv');coarse=pd.read_csv(source/'coarse_results.csv');medium=pd.read_csv(source/'medium_results.csv')
    registry=json.loads((source/'field_registry.json').read_text(encoding='utf-8'))
    ledger=pd.read_parquet(source/'combo_5m_ledger.parquet');metrics=json.loads((source/'combo_metrics.json').read_text())
    freeze=json.loads((source/'selection_freeze.json').read_text());pages=[]
    note='发现 2023—2025H1；历史验证 2025H2—2026-01；2026-02 起为此前已见的历史测试。冻结后审计，不能称新盲测。15m 完成特征、下一完整 5m 起价格收益，按实际份额和现金记账，4bps/变仓。'
    def page(name,title,body):
        prefix='../' if '/' in name else ''
        nav=' · '.join(f'<a href="{prefix}{file}">{label}</a>' for file,label in [('index.html','总览'),('search.html','搜索'),('representations.html','结构表示'),('readouts.html','联合读出'),('execution.html','5m 账本'),('method.html','合同与限制')])
        doc=f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style><script src='{prefix}plotly.min.js'></script></head><body><nav>{nav}</nav><main><h1>{html.escape(title)}</h1><p class='muted'>{html.escape(note)}</p>{body}<footer>源实验：{html.escape(str(source.resolve()))}</footer></main></body></html>"
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(doc,encoding='utf-8');pages.append(name)
    def plot(fig,title):return '<div class="card">'+_plot(fig,title)+'</div>'
    def curve(frame,title):
        fig=go.Figure()
        for period in frame.period.unique():
            f=frame[frame.period.eq(period)]
            fig.add_trace(go.Scatter(x=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M'),y=100*np.cumprod(1+f.net.to_numpy()),name=period))
        fig.update_yaxes(title_text='分段初值100，实际财富简单收益复利');return plot(fig,title)
    stats=manifest['results']; cards=''.join(f'<div class="stat">{html.escape(label)}<b>{value}</b></div>' for label,value in [('候选定义',stats['candidate_definitions']),('候选×目标',stats['candidate_target_records']),('结构坐标',stats['coordinates']),('冻结入选',stats['selected'])])
    figure=go.Figure()
    for col,label in [('mean_loss_gain_fraction','发现 OOF 损失改善'),('validation_incremental_ic','验证基线外 IC')]:figure.add_trace(go.Bar(x=results.id,y=results[col],name=label))
    figure.update_layout(barmode='group',xaxis_tickangle=-45)
    links='<div class="links">'+''.join(f'<a href="models/{r.id}.html">{r.id}</a>' for r in results.itertuples())+'</div>'
    page('index.html','复杂 Alpha 表示研究台',f'<div class="grid">{cards}</div><div class="note">冻结集合：{html.escape(str(freeze["selected"]))}。结构定义与真实预测、可交易结果分开显示。未通过候选完整保留。</div>'+plot(figure,'新读出的发现增量与历史验证')+curve(ledger,'冻结组合各段财富')+links)
    counts=coarse.groupby(['family','horizon_hours']).agg(definitions=('id','count'),mean_score=('discovery_score','mean'),best_score=('discovery_score','max')).reset_index()
    fig=go.Figure()
    for fam,g in coarse.groupby('family'):fig.add_trace(go.Box(y=g.discovery_score,name=fam,boxpoints=False))
    page('search.html','搜索预算、多样性与幸存者',plot(fig,'全部候选发现期稳健分数分布')+_table(counts)+_table(medium.head(60)))
    coords=pd.DataFrame(registry['coordinates']);summary=coords.groupby('family').agg(coordinates=('name','count'),max_support=('support_minutes','max'),max_order=('order','max')).reset_index()
    page('representations.html','观测、结构坐标与数值依赖',_table(summary)+_table(coords)+f'<p>倾斜可靠性及逐币缺失/截尾审计：<code>{source.name}/field_registry.json</code>。目录中保留完整机器可读知识卡。</p>')
    for family,g in coords.groupby('family'):page(f'families/{family}.html',f'{family} 数学表示',_table(g))
    trials=pd.read_csv(source/'readout_trials.csv');fig=go.Figure(go.Scatter(x=trials.trial,y=trials.objective,mode='lines+markers',text=trials.kind))
    page('readouts.html','结构块共同读出与公平模型对照',plot(fig,'全部 Optuna 读出尝试；发现 OOF 目标')+_table(trials)+_table(results))
    fig=go.Figure()
    for k in ['price_gross','funding','fee','net']:fig.add_trace(go.Bar(x=list(metrics),y=[metrics[p].get({'price_gross':'price_gross_pct','funding':'funding_pct','fee':'fee_pct','net':'net_sum_pct'}[k],0) for p in metrics],name=k))
    page('execution.html','原生 5m 现金、份额与费用账本',curve(ledger,'组合真实财富')+plot(fig,'价格 / funding / 费用分解')+_table(pd.DataFrame(metrics).T.reset_index()))
    for row in results.itertuples():
        file=source/f'{row.id}_frozen_audit_ledger.parquet'
        if not file.exists():file=source/f'{row.id}_research_ledger.parquet'
        f=pd.read_parquet(file)
        body=curve(f,row.id+' 分段曲线')+_table(results[results.id.eq(row.id)])
        t=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M',utc=True)
        ny=t.dt.tz_convert('America/New_York');g=pd.DataFrame({'hour':ny.dt.hour,'year':ny.dt.year,'net':f.net}).groupby(['year','hour']).net.sum().unstack()
        heat=go.Figure(go.Heatmap(z=g.to_numpy()*100,x=g.columns,y=g.index,colorbar={'title':'净收益百分点'}))
        body+=plot(heat,'纽约本地小时贡献；包含 DST')
        page(f'models/{row.id}.html',row.id+' 模型与证据',body)
    page('method.html','研究合同与证据边界','<div class="note">本轮验证和测试均为历史迁移。表示优于旧统计、结构必要性、预测增量与成本后价值需要分别证明。模型选择不读取任何 holdout 指标。</div>'+_table(pd.DataFrame([{'key':k,'value':str(v)} for k,v in freeze.items()]))+'<p>具体原语、参数量、缺失规则、训练与冻结文件均在源实验保存。仅输出新数学坐标或低相关，不足以称残差 Alpha。</p>')
    (out/'visualization_manifest.json').write_text(json.dumps({'status':'COMPLETED','source':str(source.resolve()),'pages':pages,'page_count':len(pages)},ensure_ascii=False,indent=2),encoding='utf-8')
