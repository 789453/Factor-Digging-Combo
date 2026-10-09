"""Read-only comparison atlas over completed immutable research experiments."""
from __future__ import annotations
import html,json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .factor_combo_reporting import _CSS,_plot,_table


def render_comparison(sources,out):
    out=Path(out)
    if (out/'visualization_manifest.json').exists():raise ValueError('completed comparison atlas immutable')
    out.mkdir(parents=True,exist_ok=True);(out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    models=[];trials=[];summary=[];cost=[];null=[];matched=[];structure=[];pages=[];wealth=[];baselines=[]
    for source in map(Path,sources):
        manifest=json.loads((source/'manifest.json').read_text());freeze=json.loads((source/'selection_freeze.json').read_text())
        if manifest['status']!='COMPLETED':raise ValueError('comparison requires complete frozen evidence')
        label=source.name.rsplit('_',1)[-1];r=manifest['results']
        snapshot=source/'source_snapshot'/'complex_alpha_workflow.py'
        legacy_clock=snapshot.exists() and 'stamp.asi8//900_000_000_000' in snapshot.read_text(encoding='utf-8')
        summary.append({'round':label,'definitions':r['candidate_definitions'],'target_records':r['candidate_target_records'],
            'coordinates':r['coordinates'],'readout_trials':r['readout_trials'],'selected':r['selected'],
            'unique_readout_fits':r.get('unique_readout_fits',np.nan),
            'return_contract':freeze.get('return_contract','log'),'label_projection':freeze.get('label_projection','loo'),
            'test_opened_after_freeze':manifest['holdout_opened_after_freeze'],'frozen_set':str(freeze['selected']),
            'sampling_clock':'legacy implicit epoch unit; see clock corrigendum' if legacy_clock else freeze.get('time_contract','not recorded')})
        for period,metric in r['combo'].items():wealth.append({'round':label,'period':period,**metric})
        for horizon in [4,12]:
            file=source/f'old_baseline_{horizon}h_metrics.json'
            if file.exists():
                for period,metric in json.loads(file.read_text()).items():
                    baselines.append({'round':label,'horizon':horizon,'period':period,**metric})
        for collection,name in [(models,'model_survival.csv'),(trials,'readout_trials.csv'),(cost,'cost_sensitivity.csv'),
                                (null,'survivor_shift_null.csv'),(matched,'matched_observation_windows.csv'),(structure,'structural_counterfactuals.csv')]:
            if (source/name).exists():
                frame=pd.read_csv(source/name);frame.insert(0,'round',label);collection.append(frame)
                if name=='model_survival.csv' and legacy_clock:
                    frame['nonoverlap_incremental_ic']=np.nan
                    frame['nonoverlap_status']='invalid historical epoch-unit claim; source preserved'
    models=pd.concat(models,ignore_index=True);trials=pd.concat(trials,ignore_index=True)
    def chart(fig,title):return '<div class="card">'+_plot(fig,title)+'</div>'
    def page(name,title,body):
        nav=' · '.join(f'<a href="{file}">{label}</a>' for file,label in [('index.html','研究结论'),('readouts.html','读出与消融'),('uncertainty.html','不确定性'),('structure.html','结构辨识'),('economics.html','成本与执行'),('iterations.html','迭代档案')])
        text=f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><nav>{nav}</nav><main><h1>{html.escape(title)}</h1><div class='note'>历史发现与验证可用于研究；此前已见测试只有最终冻结审计。零入选表示没有通过全部门槛，空仓不是收益胜利。结构输出变化不等于预测或可交易 Alpha。</div>{body}</main></body></html>"
        (out/name).write_text(text,encoding='utf-8');pages.append(name)
    fig=go.Figure()
    for r,g in models.groupby('round'):
        fig.add_trace(go.Scatter(x=g.mean_loss_gain_fraction*100,y=g.validation_net_return_pct,mode='markers',name=r,text=g.id,
            hovertemplate='%{text}<br>损失改善 %{x:.4f}%<br>验证净财富 %{y:.3f}%<extra></extra>'))
    fig.update_xaxes(title='发现中筛OOF损失改善 %；目标权重与采样合同按轮次区分');fig.update_yaxes(title='历史验证净财富 %，4bps')
    fig.add_hline(y=0,line_dash='dot');fig.add_vline(x=0,line_dash='dot')
    page('index.html','复杂表示：信息、迁移与经济价值',_table(pd.DataFrame(summary))+chart(fig,'预测改善与净财富必须同时成立')+
        '<p>冻结集合为空时，组合财富为零仅表示未投入资金。各轮标签、风险权重和输入空间有明确差异；不能把横向比较当作单项改动的因果消融。</p>'+_table(pd.DataFrame(wealth)))
    f=go.Figure()
    for r,g in models.groupby('round'):f.add_trace(go.Bar(x=g.kind.astype(str)+'/'+g.horizon_hours.astype(str),y=g.mean_incremental_ic,name=r))
    f.update_layout(barmode='group',xaxis_tickangle=-60)
    page('readouts.html','结构块、低秩交互、稀疏种子与消融',chart(f,'发现期基线外 IC')+_table(models)+_table(trials))
    f=go.Figure()
    for r,g in models.groupby('round'):
        good=np.isfinite(g.discovery_gain_ci_low)&np.isfinite(g.discovery_gain_ci_high)
        g=g[good];center=(g.discovery_gain_ci_low+g.discovery_gain_ci_high)/2
        f.add_trace(go.Scatter(x=g.id,y=center*100,mode='markers',name=r,
            error_y={'type':'data','array':(g.discovery_gain_ci_high-center)*100,'arrayminus':(center-g.discovery_gain_ci_low)*100,'symmetric':False}))
    f.add_hline(y=0,line_dash='dot');f.update_layout(xaxis_tickangle=-60)
    body=chart(f,'30天整组币种联合块 bootstrap 的95%区间')+'<p>区间条件于已经选择的模型，未校正全部搜索。旧轮次隐式时间单位导致的非重叠IC声明已作无效处理，源文件保留；仅显式UTC纳秒修正版提供非重叠起点。标签循环错位只重评有限幸存者，不重做六万搜索，不能据此宣布搜索赢家显著。</p>'
    if null:body+=_table(pd.concat(null).groupby(['round','horizon_hours']).survivor_winner_ic.agg(['mean','max']).reset_index())
    page('uncertainty.html','时间依赖、不重叠起点与搜索不确定性',body+_table(models[['round','id','nonoverlap_incremental_ic','discovery_gain_ci_low','discovery_gain_ci_high']]))
    body=''
    if matched:
        m=pd.concat(matched);f=go.Figure()
        for r,g in m.groupby('round'):f.add_trace(go.Scatter(x=g.old_distance,y=g.path_distance,mode='markers',name=r))
        f.update_xaxes(title='旧汇总距离');f.update_yaxes(title='二三阶路径距离');body+=chart(f,'窗口仅按历史统计近邻选择，没有未来标签')+_table(m.head(60))
    if structure:
        s=pd.concat(structure);g=s.groupby(['mode','object']).agg(rms_change=('rms_change','median'),rms_original=('rms_original','median')).reset_index()
        f=go.Figure()
        for mode,v in g.groupby('mode'):f.add_trace(go.Bar(x=v.object,y=v.rms_change,name=mode))
        body+=chart(f,'保留边际 / 联合样本后破坏顺序或对齐')+_table(g)+'<p>此处证明结构辨识，不证明预测必要性。预测必要性由同折一阶/二阶/三阶、占据/转移、对角/全矩阵和主效应/交互读出对照衡量。</p>'
    page('structure.html','新表示到底保存了什么',body)
    f=go.Figure()
    for r,g in models.groupby('round'):f.add_trace(go.Scatter(x=g.validation_turnover,y=g.validation_break_even_bps,mode='markers',name=r,text=g.id))
    f.add_hline(y=4,line_dash='dot');f.update_xaxes(title='验证累计单位资本变仓');f.update_yaxes(title='盈亏平衡 bps / 单位变仓')
    page('economics.html','可预测信息怎样损失在费用与执行中',chart(f,'盈亏平衡成本与换手')+_table(models[['round','id','validation_price_gross_pct','validation_funding_pct','validation_fee_pct','validation_net_return_pct','validation_mean_abs_position']])+_table(pd.concat(cost))+
        '<h2>同合同旧表示完整预测基线</h2><p>下表是完整旧预测器；上面的 OLD 增量通道在跨度剔除后可为零，两者含义不同。净财富按复利，分项是简单收益之和。</p>'+_table(pd.DataFrame(baselines)))
    links=''.join(f'<p><a href="../{Path(s).name}/index.html">{html.escape(Path(s).name)}：完整家族和逐模型多页报告</a></p>' for s in sources)
    page('iterations.html','完整档案与复现边界',links+_table(pd.DataFrame(summary))+'<p>源输出保存配置、源代码快照、初始坐标尺度、候选依赖与系数、全部Optuna试验及SQLite、折内模型、验证账本和最终冻结。真实pilot被中断，模拟v4样本不足失败，均未冒称完成实验。</p>')
    (out/'visualization_manifest.json').write_text(json.dumps({'status':'COMPLETED','pages':pages,'sources':[str(Path(s).resolve()) for s in sources]},ensure_ascii=False,indent=2),encoding='utf-8')
    return models
