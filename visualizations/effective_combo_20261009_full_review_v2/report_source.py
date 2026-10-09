"""Full interactive, read-only evidence for conditional effective-combo research."""
from __future__ import annotations
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from .factor_combo_reporting import _CSS, _plot, _table as _shared_table


def render_effective_report(source, out, review=None, signal_review=None, showcase=None):
    if out.exists():raise ValueError('new visualization directory required')
    out.mkdir(parents=True); (out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    shutil.copyfile(__file__,out/'report_source.py')
    def _table(frame,columns=None):return _shared_table(frame,columns,rows=len(frame))
    def evidence(file,label):return '<a href="'+html.escape(os.path.relpath(source/file,out).replace('\\','/'))+'">'+html.escape(label)+'</a>'
    load=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    manifest=load('manifest.json'); freeze=load('selection_freeze.json'); contract=load('feature_contract.json')
    stats=pd.read_csv(source/'portfolio_metrics.csv'); predmetrics=pd.read_csv(source/'prediction_metrics.csv')
    automatic=freeze['selected']['id']; best=showcase or automatic
    if best not in set(stats.id):raise ValueError('showcase must be a frozen, predeclared product')
    codes=contract['assets']; dates=np.load(source/'dates.npy')
    pages=[]
    note='源候选经过全发现搜索；反馈OOF非独立。历史验证已见。成熟标签训练／更新；未来收益不作输入。4h完成决策、之后完整5m收益，实际份额资金费价格代理，单位单向变仓4bps，终点平仓；净值逐行简单收益复利。'
    navitems=[('index.html','总览'),('models.html','预测对照'),('factors.html','因子池'),('execution.html','组合执行'),
              ('periods.html','月度季度'),('assets.html','十二币贡献'),('overfit.html','强拟合诊断'),('method.html','规范与迭代')]
    if review:navitems += [('robustness.html','稳健验收'),('ablation.html','固定移除'),('forecasts.html','预测分组')]
    if signal_review:navitems += [('costs.html','成本余量'),('exposures.html','长短与共同暴露'),('feature_evidence.html','逐因子证据')]
    def page(name,title,body):
        prefix='../' if '/' in name else ''
        nav=' · '.join(f'<a href="{prefix}{file}">{label}</a>' for file,label in navitems)
        doc=f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style><script src='{prefix}plotly.min.js'></script></head><body><nav>{nav}</nav><main><h1>{html.escape(title)}</h1><p class='muted'>{html.escape(note)}</p>{body}<footer>源：{html.escape(str(source.resolve()))}</footer></main></body></html>"
        path=out/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_text(doc,encoding='utf-8');pages.append(name)
    def plot(fig,title):return '<div class="card">'+_plot(fig,title,height=fig.layout.height or 370)+'</div>'
    def wealth(frame,title):
        fig=go.Figure(); dt=pd.to_datetime(frame.completed_5m,format='%Y%m%d%H%M')
        idx=np.arange(0,len(frame),12); idx=np.unique(np.r_[idx,len(frame)-1])
        for label,r in [('价格毛',frame.price_gross),('价格＋资金费',frame.price_gross+frame.funding),('费后',frame.net)]:
            fig.add_trace(go.Scatter(x=dt.iloc[idx],y=100*np.cumprod(1+r.to_numpy())[idx],name=label))
        fig.update_yaxes(title='初值100；全5m账本复利，图按小时取点')
        return plot(fig,title)
    def frame(key,period='validation'):return pd.read_parquet(source/(key+'_'+period+'_ledger.parquet'))
    validation=stats.loc[stats.period.eq('validation')].copy()
    selected=validation.loc[validation.id.eq(best)].iloc[0]
    diag=load('overfit_diagnostic_metrics.json');diagtrade=load('overfit_diagnostic_strategy.json')
    body=f'<div class="note">当前展示预声明产品：{html.escape(best)}，历史验证费后={selected.net_return_pct:.3f}%，日Sharpe={selected.sharpe:.3f}，最大回撤={selected.max_drawdown_pct:.3f}%。自动发现排名首位仍为{html.escape(automatic)}；展示有效产品不修改原选择冻结。</div>'
    if review:
        acceptance=pd.read_csv(review/'acceptance.csv')
        core=acceptance.loc[acceptance.id.isin(['ensemble__edge16','tree15_online__edge16','tree7_static__edge16'])]
        body+='<h2>有效因子信号与三套预声明历史基线</h2><p>以下产品在实验开始前已定义。展示所有预声明产品的验收，不用验证赢家重写发现选择。2025Q3与Q4为完整季度，2026Q1只含1月；已见历史证据，不称未来保证。</p>'+_table(core[['id','net_return_pct','sharpe','max_drawdown_pct','positive_discovery_folds','positive_validation_segments','natural_episodes','mean_abs_position','break_even_bps']])
    body+=wealth(frame(best),'预声明弱信号组合的历史验证财富')
    fig=go.Figure()
    for key in ([best,'tree15_online__edge16','tree7_static__edge16',automatic] if review else [best]):
        f=frame(key);dd=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M');ii=np.unique(np.r_[np.arange(0,len(f),12),len(f)-1])
        fig.add_trace(go.Scatter(x=dd.iloc[ii],y=100*np.cumprod(1+f.net.to_numpy())[ii],name=key))
    body+=plot(fig,'有效产品及自动排名失败项：全5m复利净值')
    body+=f'<h2>强拟合诊断单列</h2><p>真实发现标签同样本训练IC={diag["ic"]:.5f}，R²={diag["r2_vs_zero"]:.3%}。样本内反复复利财富倍率约{1+diagtrade["net_return_pct"]/100:.3e}，用于暴露过拟合容量，不能视为可实现资金增长。</p>'
    fig=go.Figure(go.Bar(x=validation.id,y=validation.net_return_pct));fig.update_layout(xaxis_tickangle=-60)
    body+=plot(fig,'所有预声明模型×执行：历史验证费后财富')+_table(validation)
    page('index.html','有效因子与条件组合：完整研究台',body)
    fig=go.Figure()
    for period,part in predmetrics.groupby('period'):fig.add_trace(go.Bar(x=part.model,y=part.ic,name=period))
    page('models.html','条件模型、成熟更新和预测迁移',plot(fig,'全部组合预测IC')+''.join(f'<p><a href="models/{html.escape(model)}.html">{html.escape(model)}预测与执行详情</a></p>' for model in predmetrics.model.unique())+_table(predmetrics)+_table(pd.read_csv(source/'update_records.csv')))
    pools=pd.concat([pd.read_csv(source/'old_feature_pool.csv').assign(stream='OLD'),pd.read_csv(source/'mixed_feature_pool.csv').assign(stream='Mixed')])
    importance=pd.read_csv(source/'feature_importance.csv')
    page('factors.html','弱信号定义、条件档案与组合用法',''.join(f'<p><a href="families/{html.escape(family)}.html">{html.escape(family)}冻结函数与逐条定义</a></p>' for family in pools.family.unique())+_table(pools)+_table(importance.sort_values('gain',ascending=False).head(100))+f'<p>全公式：{html.escape(str(source/"feature_pool_freeze.json"))}。训练分裂gain为使用度，不是因果增量或PnL归因。</p>')
    definitions=load('feature_pool_freeze.json')['features']
    for family,part in pools.groupby('family'):
        cards=''.join('<details><summary>'+html.escape(r['stream']+' / '+r['id'])+'</summary><pre>'+html.escape(json.dumps(r['definition'],ensure_ascii=False,indent=2))+'</pre></details>' for r in definitions if r['family']==family)
        page('families/'+family+'.html',family+' 函数卡',_table(part)+cards)
    ledger=frame(best); path=np.load(source/(best+'_validation_asset_path.npz'));asset=path['net'];position=path['position']; prices=path['price'];fast=path['dates']
    dt=pd.to_datetime(fast,format='%Y%m%d%H%M');idx=np.unique(np.r_[np.arange(0,len(fast),12),len(fast)-1])
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=['组合实际绝对／净暴露','逐行变仓与费用','累计费用与资金费'])
    fig.add_trace(go.Scatter(x=dt[idx],y=ledger.mean_abs_position.iloc[idx],name='gross'),row=1,col=1)
    fig.add_trace(go.Scatter(x=dt[idx],y=ledger.cash_exposure.iloc[idx],name='net cash'),row=1,col=1)
    fig.add_trace(go.Scatter(x=dt[idx],y=ledger.turnover.iloc[idx],name='turnover'),row=2,col=1)
    fig.add_trace(go.Scatter(x=dt[idx],y=np.cumsum(ledger.fee)[idx]*100,name='fee sum %'),row=3,col=1)
    fig.add_trace(go.Scatter(x=dt[idx],y=np.cumsum(ledger.funding)[idx]*100,name='funding sum %'),row=3,col=1)
    trades=pd.read_csv(source/(best+'_validation_trades.csv'));done=trades.loc[~trades.censored]
    summary={'natural_completed_episodes':len(done),'terminal_censored':int(trades.censored.sum()),
        'holding_hours_median':float(done.hours.median()),'holding_hours_p10':float(done.hours.quantile(.1)),
        'holding_hours_p90':float(done.hours.quantile(.9)), 'definition':'signed exposure episodes, epsilon 1e-5; not exact lot PnL attribution'}
    page('execution.html','真实份额、交易费用与持仓路径',wealth(ledger,'固定策略财富')+plot(fig,'执行与费用分轴')+_table(pd.DataFrame([summary]))+'<p>'+evidence(best+'_validation_trades.csv','全部持仓段CSV（下表只预览前100段）')+' · '+evidence(best+'_validation_ledger.parquet','完整5m组合账本')+'</p>'+_table(done.head(100)))
    month=dt.strftime('%Y-%m');quarter=dt.to_period('Q').astype(str)
    def period_table(labels):
        rows=[]
        for label in np.unique(labels):
            mask=labels==label;f=ledger.loc[mask];r=f.net.to_numpy()
            rows.append({'period':label,'net_wealth_pct':(np.prod(1+r)-1)*100,'price_sum_pct':f.price_gross.sum()*100,
                'funding_sum_pct':f.funding.sum()*100,'fee_pct':f.fee.sum()*100,'mean_abs_position':f.mean_abs_position.mean()})
        return pd.DataFrame(rows)
    monthly=period_table(month);quarterly=period_table(quarter)
    monthly.to_csv(out/'selected_monthly.csv',index=False);quarterly.to_csv(out/'selected_quarterly.csv',index=False)
    fig=go.Figure(go.Bar(x=monthly.period,y=monthly.net_wealth_pct))
    page('periods.html','月度、季度与持仓有效区间',plot(fig,'选定组合每月独立复利净收益')+_table(monthly)+_table(quarterly))
    # Contributions compound with total portfolio wealth, and add to final portfolio PnL.
    capital=np.r_[1,np.cumprod(1+ledger.net.to_numpy())[:-1]]
    contribution=(asset/len(codes)*capital[:,None]).sum(axis=0)*100
    aset=pd.DataFrame({'asset':codes,'portfolio_wealth_contribution_pct':contribution,'mean_abs_position':np.mean(np.abs(position),axis=0)})
    fig=go.Figure(go.Bar(x=codes,y=contribution))
    page('assets.html','逐币贡献与局部信号',plot(fig,'精确组合财富归因；全部币种贡献之和还原组合净财富')+_table(aset)+''.join(f'<p><a href="assets/{c}.html">{c}价格、预测、仓位与贡献</a></p>' for c in codes))
    pred=np.load(source/(best.split('__')[0]+'_prediction.npy'))
    mask=dates>=str(fast[0]);dd=pd.to_datetime(dates[mask],format='%Y%m%d%H%M')
    for j,code in enumerate(codes):
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=['完成价格','4h组合预测（bps）','实际持仓与长短方向','组合财富贡献（百分点）'])
        fig.add_trace(go.Scatter(x=dt[idx],y=prices[idx,j],name='price'),row=1,col=1)
        fig.add_trace(go.Scatter(x=dd,y=pred[mask,j]*1e4,name='forecast bps'),row=2,col=1)
        fig.add_trace(go.Scatter(x=dt[idx],y=position[idx,j],name='held exposure'),row=3,col=1)
        fig.add_trace(go.Scatter(x=dt[idx],y=np.cumsum(asset[:,j]/len(codes)*capital)[idx]*100,name='contribution %'),row=4,col=1)
        fig.update_layout(height=1000)
        page('assets/'+code+'.html',code+' 预测与交易',plot(fig,code+'独立尺度的信号和交易路径')+_table(trades.loc[trades.asset.eq(code)].head(200)))
    for model,part in predmetrics.groupby('model'):
        choices=[r for r in freeze['ranked_discovery'] if r['model']==model];pick=choices[0]['id']
        page('models/'+model+'.html',model+'条件读出',_table(part)+wealth(frame(pick),'该模型按发现选择执行的验证财富')+_table(validation.loc[validation.id.str.startswith(model+'__')]))
    page('overfit.html','高容量强拟合诊断：真实可观察但非前向策略',f'<div class="note">同一发现标签用于训练与交易回放；明确过拟合诊断。该曲线不能用作真实迁移成功。</div>'+_table(pd.DataFrame([diag]))+_table(pd.DataFrame([diagtrade]))+wealth(frame('overfit_diagnostic','in_sample'),'高容量模型训练内财富'))
    docs=['EFFECTIVE_FACTOR_AND_COMBO_RESEARCH_RESULTS_20261009.md','EFFECTIVE_FACTOR_AND_STRATEGY_MANDATE_20261009.md','EFFECTIVE_COMBO_ROUND1_DESIGN_20261009.md','EFFECTIVE_COMBO_ITERATION_LOG_20261009.md']
    page('method.html','第一性目标、数据合同与研究迭代','<p>当前成功标准：有效信号＋历史稳健基线。没有通过就继续研究，过拟合诊断不能冒充稳健。</p>'+''.join(f'<p><a href="../../docs/{doc}">{doc}</a></p>' for doc in docs)+'<p>记录发现选择、所有模型和执行，不读holdout。</p>'+_table(pd.DataFrame([manifest]))+_table(pd.DataFrame(freeze['ranked_discovery'])))
    if review:
        acceptance=pd.read_csv(review/'acceptance.csv');quarters=pd.read_csv(review/'quarterly.csv');months=pd.read_csv(review/'monthly.csv')
        part=quarters.loc[quarters.id.isin(['ensemble__edge16','tree15_online__edge16','tree7_static__edge16',automatic])]
        fig=go.Figure()
        for key,f in part.groupby('id'):fig.add_trace(go.Bar(x=f.period,y=f.net_return_pct,name=key))
        page('robustness.html','全部21项验收、完整季度与薄信号区别',plot(fig,'完整Q3、Q4与仅1月的Q1')+_table(part)+_table(acceptance)+_table(months.loc[months.id.isin(['ensemble__edge16','tree15_online__edge16'])]))
        removal=pd.read_csv(review/'fixed_removal.csv');fig=go.Figure(go.Bar(x=removal.diagnostic,y=removal.net_return_pct));fig.update_layout(xaxis_tickangle=-40)
        page('ablation.html','冻结模型移除：真实组合合作与依赖', '<p>成员移除置零、原除数3保留；输入移除仅将对应数值坐标置零。所有月度模型、阈值、幅度及资金预算固定，没有重训、重新排名或重新归一化。零输入可能离开真实联合分布；这些是固定模型依赖诊断，不是可加PnL归因或重训最优结论。</p>'+plot(fig,'固定组合移除后费后财富')+_table(removal)+_table(pd.read_csv(review/'fixed_removal_prediction.csv'))+'<pre>'+html.escape((review/'reproduction.json').read_text(encoding='utf-8'))+'</pre>')
        groups=pd.read_csv(review/'forecast_groups.csv');fig=go.Figure()
        for model,part in groups.groupby('model'):fig.add_trace(go.Scatter(x=part.predicted_mean_bps,y=part.realized_mean_bps,mode='lines+markers',name=model))
        fig.update_xaxes(title='预测均值 bps');fig.update_yaxes(title='实际4h目标均值 bps')
        page('forecasts.html','预测方向、幅度过估与固定成本边际', '<p>分组边界固定为±8、16、40、80bps与0。不用这些验证分组再调参数。IC正而R²负表示方向结构存在且幅度尚未标定；正策略收益不能替代校准合格。</p>'+plot(fig,'预测组均值与真实收益：显示幅度偏差')+_table(groups))
    if signal_review:
        costs=pd.read_csv(signal_review/'cost_sensitivity.csv');fig=go.Figure()
        for key,part in costs.groupby('id'):fig.add_trace(go.Scatter(x=part.cost_bps,y=part.net_return_pct,mode='lines+markers',name=key))
        fig.update_xaxes(title='单向单位变仓 bps');fig.update_yaxes(title='费后净财富 %')
        page('costs.html','同一冻结策略的真实份额成本敏感性',plot(fig,'0／2／4／6／8／12bps：每点重新记账，终点平仓')+_table(costs))
        exposure=pd.read_csv(signal_review/'exposure_decomposition.csv');fig=go.Figure()
        for col in ['common_price_wealth_contribution_pct','relative_price_wealth_contribution_pct','funding_wealth_contribution_pct','fee_wealth_contribution_pct']:fig.add_trace(go.Bar(x=exposure.id,y=exposure[col]*( -1 if col=='fee_wealth_contribution_pct' else 1),name=col))
        fig.update_layout(barmode='relative')
        page('exposures.html','长短贡献、共同价格方向与相对价格信息', '<p>按实际财富权重作精确价格贡献分解：共同部分＝现金净暴露×十二币等权价格变化；剩余为相对价格部分。这不是风险beta残差模型，也不是已扣对冲成本的中性策略。此产品含方向／择时信息，不标纯Alpha。</p>'+plot(fig,'价格共同与相对贡献，加资金费减费用还原净财富')+_table(exposure))
        individual=pd.read_csv(signal_review/'individual_features.csv');breadth=pd.read_csv(signal_review/'prediction_breadth.csv')
        page('feature_evidence.html','202个输入逐项信息与组合迁移广度', '<p>函数方向沿用发现元数据，只显示原始与冻结方向IC；该表不据验证重新筛因子，不要求单因子交易盈利。条件或不稳定特征原档案保留。坐标不是收益预测，不能把其平方损失当预测R²。</p>'+_table(individual)+_table(breadth))
    (out/'visualization_manifest.json').write_text(json.dumps({'status':'COMPLETED','source':str(source.resolve()),'showcase':best,'automatic_selection':automatic,'review':str(review),'signal_review':str(signal_review),'pages':pages,'page_count':len(pages)},ensure_ascii=False,indent=2),encoding='utf-8')
