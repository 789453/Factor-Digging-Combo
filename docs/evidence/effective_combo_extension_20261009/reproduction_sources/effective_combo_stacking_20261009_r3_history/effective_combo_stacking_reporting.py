"""Full local interactive report, preserving predeclared anchors and frozen choice."""
from __future__ import annotations
import html,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from .factor_combo_reporting import _CSS,_plot,_table
from .effective_combo_extension import source_metadata


def render_stacking_report(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('new visualization directory required')
    out.mkdir(parents=True);(out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    load=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    manifest=load('manifest.json');freeze=load('protocol_freeze.json');selection=load('selection_freeze.json');contract=load('feature_contract.json')
    phase=manifest['phase'];period='oos' if phase=='oos' else 'validation';showcase='pair_position__edge16'
    dates=np.load(source/'dates.npy');y=np.load(source/'target.npy');stats=pd.read_csv(source/'portfolio_metrics.csv')
    current=stats.loc[stats.period==period];pmetrics=pd.read_csv(source/'prediction_metrics.csv')
    breadth=pd.read_csv(source/'prediction_breadth.csv');updates=pd.read_csv(source/'update_records.csv');weights=pd.read_csv(source/'ensemble_weights.csv')
    nav=[('index.html','总览'),('comparison.html','公平模型比较'),('factors.html','弱信号扩池'),('weights.html','组合权重'),
        ('periods.html','月季广度'),('assets.html','逐币证据'),('costs.html','成本与暴露'),('method.html','冻结与复现')]
    note='4h完成决策与收益预测，之后原生5m实际份额账本；单边4bps/终点平仓/简单收益复利；日Sharpe sqrt(365.25)。旧候选发现期反馈OOF非独立，历史验证已见；2026为历史OOS复核，月更仅用过去成熟标签。'
    pages=[]
    table=lambda df:_table(df,rows=len(df))
    def plot(fig,title):return _plot(fig,title,height=fig.layout.height or 400)
    def link(file,label):return '<a href="'+html.escape(os.path.relpath(source/file,out).replace('\\','/'))+'">'+html.escape(label)+'</a>'
    def page(file,title,body):
        prefix='../' if '/' in file else '';n=' · '.join(f'<a href="{prefix}{f}">{t}</a>' for f,t in nav)
        text=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{_CSS}</style><script src="{prefix}plotly.min.js"></script></head><body><nav>{n}</nav><main><h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>{body}<footer>{html.escape(str(source.resolve()))}</footer></main></body></html>'
        p=out/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8');pages.append(file)
    frames={k:pd.read_parquet(source/(k+'_'+period+'_ledger.parquet')) for k in current.id}
    def curve(keys):
        fig=go.Figure()
        for key in keys:
            f=frames[key];ii=np.unique(np.r_[np.arange(0,len(f),12),len(f)-1]);dt=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M')
            fig.add_trace(go.Scatter(x=dt.iloc[ii],y=np.cumprod(1+f.net.to_numpy())[ii]*100,name=key))
        fig.update_yaxes(title='完整5m复利，初值100；小时抽点')
        return plot(fig,'预声明组合财富对照')
    anchors=['base_ensemble__edge16','base15__edge16','pair_prediction__edge16','pair_position__edge16','anchor_wide__edge16','stack_global__edge16','stack_asset__edge16']
    body='<p>当前固定展示事前声明的 pair_position_edge16。它先将成员各自映射成仓位再平均，图上的预测均值只是成员诊断，不是该仓位的单一输入；所有扩展产品完整报告，不按OOS收益重新选冠军。实际终点：'+manifest['results']['actual_end']+('；请求上限2026-09-30，不足整月的9月按部分月份报告。' if phase=='oos' else '；此阶段未打开2026年2月及之后。')+'</p>'
    body+=curve(anchors)+table(current)+table(pmetrics.loc[pmetrics.period==period])
    if phase=='oos':
        history=Path(freeze['protocol']['frozen_run']) if 'frozen_run' in freeze['protocol'] else Path(__import__('yaml').safe_load((source/'config.yaml').read_text())['complex_alpha']['frozen_run'])
        historical=pd.read_csv(history/'portfolio_metrics.csv');historical=historical.loc[historical.period=='validation']
        body+='<h2>历史验证与新增OOS分别评价</h2>'+table(historical.merge(current,on='id',suffixes=('_validation','_oos')))
    page('index.html','R3 有效基线、共享收缩权重与'+period+'完整研究台',body)
    comparisons=[]
    pairs=[('base_ridge','wide_ridge','同容量Ridge扩池'),('base15','wide15','同容量15叶扩池'),('base31','wide31','同容量31叶扩池'),
        ('wide15_static','wide15','同容量静态/月更'),('wide15','asset_shrunk','收缩币种适配'),('wide_equal','wide_cov','等权/误差协方差'),('base_ensemble','adaptive_equal','R2混合'),('base15','pair_prediction','在线/静态预测平均'),('pair_prediction','pair_position','预测平均/映射后仓位平均'),('base15','anchor_wide','有效共享基线75%加扩池25%'),('base_ensemble','stack_global','预测尺度收缩凸组合'),('stack_global','stack_asset','全局/共享单币收缩权重')]
    for a,b,name in pairs:
        aa=current.loc[current.id==a+'__edge16'].iloc[0];bb=current.loc[current.id==b+'__edge16'].iloc[0]
        comparisons.append({'comparison':name,'control':a,'variant':b,'net_change_pp':bb.net_return_pct-aa.net_return_pct,
            'sharpe_change':bb.sharpe-aa.sharpe,'drawdown_change_pp':bb.max_drawdown_pct-aa.max_drawdown_pct})
    pd.DataFrame(comparisons).to_csv(source/'fair_comparisons.csv',index=False)
    page('comparison.html','公平比较：因素、容量、更新与执行分开',table(pd.DataFrame(comparisons))+table(stats)+table(pmetrics)+
        ''.join(f'<p><a href="models/{key}.html">{key} 预测/交易/持仓/净值</a></p>' for key in pmetrics.model.unique()))
    pool_source,mining,_,_,_=source_metadata(freeze['protocol']['source_pool_run'])
    definitions=json.loads((mining/'candidate_registry.json').read_text(encoding='utf-8'))['definitions'];definitions={r['id']:r for r in definitions}
    original=json.loads((pool_source/'feature_pool_freeze.json').read_text(encoding='utf-8'))['features']
    expanded=pd.read_csv(source/'expansion_archive.csv');retained=expanded.loc[expanded.reason=='retained']
    factors=pd.read_csv(source/'individual_features.csv');registry=pd.concat([pd.DataFrame(original).assign(role='original'),retained.assign(role='added')],ignore_index=True)
    body='<p>原输入全部保留；新增按发现期全局相关去重与核心/条件各半配额。弱边际IC不等同于不能进入联合预测；分裂gain与单变量IC都不是净收益因果归因。</p>'+table(registry.drop(columns=['definition'],errors='ignore'))+table(factors)
    body+=''.join(f'<p><a href="families/{html.escape(k)}.html">{html.escape(k)} 公式定义、层级与效果</a></p>' for k in registry.family.dropna().unique())
    page('factors.html','扩池与弱信号：逐条公式证据',body)
    for family,g in registry.groupby('family'):
        cards=''.join('<details><summary>'+html.escape(r.id)+'</summary><pre>'+html.escape(json.dumps(definitions[r.id],ensure_ascii=False,indent=2))+'</pre></details>' for _,r in g.iterrows())
        page('families/'+family+'.html',family+'公式卡',table(g.drop(columns='definition',errors='ignore'))+table(factors.loc[factors.feature.isin(g.id)])+cards)
    fig=go.Figure()
    global_weights=weights.loc[weights.asset=='GLOBAL']
    for name in [k for k in weights if k.startswith('weight_')]:fig.add_trace(go.Scatter(x=global_weights['update'].astype(str),y=global_weights[name],name=name))
    page('weights.html','成熟误差相关、非负收缩权重与币种模型',plot(fig,'月度组合权重')+table(weights)+table(updates)+
        '<p>四成员为base15/base_ensemble/wide15_static/asset_shrunk，先验40%/35%/20%/5%。预测矩阵对角收缩25%，最优凸权重75%向先验收缩；单币权重再与全局各半。只用成熟过去预测/标签。pair_position在净组合仓位变化上一次收费。</p>')
    segments=[];assetrows=[];episode_rows=[]
    for key in current.id:
        f=frames[key];dt=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M');r=f.net.to_numpy()
        for freq in ['M','Q']:
            period_index=dt.dt.to_period(freq)
            for segment in period_index.unique():
                m=np.asarray(period_index==segment);v=r[m];sd=np.std(pd.Series(v,index=dt[m]).resample('D').sum().to_numpy())
                segments.append({'id':key,'frequency':freq,'segment':str(segment),'first':str(f.completed_5m[m].iloc[0]),'last':str(f.completed_5m[m].iloc[-1]),
                    'net_return_pct':float((np.prod(1+v)-1)*100),'daily_sharpe':float(pd.Series(v,index=dt[m]).resample('D').sum().mean()/sd*np.sqrt(365.25)) if sd>0 else 0})
        path=np.load(source/(key+'_'+period+'_asset_path.npz'));capital=np.r_[1,np.cumprod(1+r)[:-1]]
        for j,code in enumerate(contract['assets']):
            assetrows.append({'id':key,'asset':code,'wealth_contribution_pct':float(np.sum(capital*path['net'][:,j])/len(contract['assets'])*100),
                'asset_net_return_pct':float((np.prod(1+path['net'][:,j])-1)*100),'mean_abs_position':float(np.mean(abs(path['position'][:,j])))})
        trades=pd.read_csv(source/(key+'_'+period+'_trades.csv'));done=trades.loc[~trades.censored]
        episode_rows.append({'id':key,'natural_episodes':len(done),'right_censored':int(trades.censored.sum()),
            'median_hours':float(done.hours.median()),'p90_hours':float(done.hours.quantile(.9)),
            'active_5m_fraction':float(np.mean(np.any(abs(path['position'])>1e-5,axis=1)))})
    segments=pd.DataFrame(segments);assetrows=pd.DataFrame(assetrows);episodes=pd.DataFrame(episode_rows)
    for name,frame in [('period_breadth',segments),('asset_contribution',assetrows),('holding_episodes',episodes)]:frame.to_csv(source/(name+'.csv'),index=False)
    page('periods.html','全产品月度、季度及持有广度',table(segments)+table(episodes)+table(breadth.loc[breadth.axis=='month']))
    page('assets.html','逐币贡献、预测与实际交易',''.join(f'<p><a href="assets/{code}.html">{code} 价格/预测/持仓/净值独立坐标</a></p>' for code in contract['assets'])+table(assetrows)+table(breadth.loc[breadth.axis=='asset']))
    dm=(dates>=str(frames[showcase].completed_5m.iloc[0]))&(dates<=str(frames[showcase].completed_5m.iloc[-1]))
    chosen=['base15','pair_prediction','pair_position','stack_global','stack_asset'];predictions={k:np.load(source/(k+'_prediction.npy')) for k in chosen}
    paths={k:np.load(source/(k+'__edge16_'+period+'_asset_path.npz')) for k in chosen}
    for j,code in enumerate(contract['assets']):
        path=paths['base15'];fast=path['dates'];ii=np.unique(np.r_[np.arange(0,len(fast),12),len(fast)-1]);dt=pd.to_datetime(fast,format='%Y%m%d%H%M');dd=pd.to_datetime(dates[dm],format='%Y%m%d%H%M')
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=['完成价格','4h预测与实现标签（bps）','实际持仓','独立币账本与组合财富贡献'])
        fig.add_trace(go.Scatter(x=dt[ii],y=path['price'][ii,j],name='price'),row=1,col=1)
        fig.add_trace(go.Scatter(x=dd,y=y[dm,j]*1e4,name='realized label',opacity=.2),row=2,col=1)
        for k in chosen:
            p=paths[k];fig.add_trace(go.Scatter(x=dd,y=predictions[k][dm,j]*1e4,name=k+' prediction'),row=2,col=1)
            fig.add_trace(go.Scatter(x=dt[ii],y=p['position'][ii,j],name=k+' position'),row=3,col=1)
            fig.add_trace(go.Scatter(x=dt[ii],y=np.cumprod(1+p['net'][:,j])[ii]*100,name=k+' wealth'),row=4,col=1)
        fig.update_layout(height=1050)
        trades=pd.read_csv(source/(showcase+'_'+period+'_trades.csv'));trades=trades.loc[trades.asset==code]
        page('assets/'+code+'.html',code+'多因子预测与适配证据',plot(fig,code+'不同量纲独立坐标')+table(assetrows.loc[assetrows.asset==code])+table(breadth.loc[(breadth.axis=='asset')&(breadth.segment==code)])+
            '<p>持有为连续有符号暴露片段，非逐笔可平仓lot；末尾强制平仓右删失。预览100条，完整记录见输出CSV。</p>'+table(trades.head(100)))
    for model in pmetrics.model.unique():
        keys=[k for k in current.id if k.startswith(model+'__')];body=curve(keys)+table(current.loc[current.id.isin(keys)])+table(pmetrics.loc[pmetrics.model==model])+table(breadth.loc[breadth.model==model])+table(episodes.loc[episodes.id.isin(keys)])
        for key in keys:
            f=frames[key];ii=np.unique(np.r_[np.arange(0,len(f),12),len(f)-1]);dt=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M');fig=make_subplots(rows=3,cols=1,shared_xaxes=True)
            fig.add_trace(go.Scatter(x=dt.iloc[ii],y=f.mean_abs_position.iloc[ii],name='gross position'),row=1,col=1)
            for name in ['fee','funding']:fig.add_trace(go.Scatter(x=dt.iloc[ii],y=np.cumsum(f[name].to_numpy())[ii]*100,name=name+' arithmetic sum'),row=2,col=1)
            for name,r in [('price',f.price_gross),('price+funding',f.price_gross+f.funding),('net',f.net)]:fig.add_trace(go.Scatter(x=dt.iloc[ii],y=np.cumprod(1+r.to_numpy())[ii]*100,name=name+' wealth'),row=3,col=1)
            fig.update_layout(height=750);body+=plot(fig,key+'暴露、费用、分层净值')
        page('models/'+model+'.html',model+'完整预测与执行卡',body)
    page('costs.html','成本余量及共同方向/相对价差分解',table(pd.read_csv(source/'cost_sensitivity.csv'))+table(pd.read_csv(source/'exposure_decomposition.csv'))+
        '<p>共同方向=当行实际平均仓位×下一5m等权市场收益；相对部分为剩余价格收益，按当期组合资本加权为财富贡献。不是beta中性回测或纯alpha证明；资金费使用完成价格代理。</p>')
    page('method.html','事前冻结、来源、标签成熟与复现', '<p>自动发现推荐保持 '+html.escape(selection['selected']['id'])+'，不按OOS重新选。每次输出不可变；同一生产CLI入口。</p><pre>'+html.escape(json.dumps(freeze,ensure_ascii=False,indent=2))+'</pre>'+link('config.yaml','运行配置')+' · '+link('portfolio_metrics.csv','全账本指标')+' · '+link('update_records.csv','完整更新记录'))
    (out/'visualization_manifest.json').write_text(json.dumps({'pages':pages,'page_count':len(pages),'phase':phase,'automatic_selection':selection['selected']['id'],
        'showcase':showcase,'source':str(source.resolve()),'full_rows':True},ensure_ascii=False,indent=2),encoding='utf-8')
