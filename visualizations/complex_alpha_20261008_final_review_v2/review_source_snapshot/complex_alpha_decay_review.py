"""Post-freeze forecast-decay reporting, with no tuning or test labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_evaluation import past_market_betas,targets_from_prices,project_basket_returns,neutral_positions,five_minute_ledger,ledger_metrics
from src.alpha_mvp.research.complex_alpha_search import correlation
from src.alpha_mvp.research.complex_alpha_time import nonoverlap_phase,TIME_VERSION
from src.alpha_mvp.research.complex_alpha_workflow import _funding
from src.alpha_mvp.research.factor_combo_reporting import _CSS,_plot,_table


def forecast_decay_metrics(prediction,target,mask,active,nonoverlap,min_count=200):
    valid=np.isfinite(prediction)&np.isfinite(target)&mask[:,None]
    def measure(use):
        return correlation(prediction[use],target[use]) if use.sum()>=min_count else np.nan
    per_asset=[correlation(prediction[valid[:,j],j],target[valid[:,j],j])
               if valid[:,j].sum()>=min_count else np.nan for j in range(target.shape[1])]
    finite=np.isfinite(per_asset)
    return {'forecast_to_projected_return_ic':measure(valid),
        'mean_asset_ic':float(np.mean(np.asarray(per_asset)[finite])) if np.any(finite) else np.nan,
        'nonoverlap_ic':measure(valid&nonoverlap[:,None]),'edge_condition_ic':measure(valid&active),
        'asset_ic':json.dumps(per_asset),'finite_asset_observations':int(valid.sum()),
        'edge_condition_observations':int((valid&active).sum())}


def volatility_activation(edge,risk,masks,threshold):
    discovery=masks['discovery'];edges=[];groups=np.full(risk.shape,-1,int)
    for j in range(risk.shape[1]):
        good=discovery&np.isfinite(risk[:,j])&(risk[:,j]>0)
        if not np.any(good):edges.append(None);continue
        cut=np.quantile(risk[good,j],[.2,.4,.6,.8]);edges.append(cut.tolist())
        valid=np.isfinite(risk[:,j])&(risk[:,j]>0)
        groups[valid,j]=np.searchsorted(cut,risk[valid,j],side='right')
    rows=[]
    for period,mask in masks.items():
        for k in range(5):
            use=mask[:,None]&(groups==k)&np.isfinite(edge);n=int(use.sum())
            rows.append({'period':period,'past_risk_quintile':k+1,'observations':n,
                'mean_abs_forecast_bps':float(np.mean(np.abs(edge[use]))*1e4) if n else np.nan,
                'mean_abs_forecast_in_risk_units':float(np.mean(np.abs(edge[use])/risk[use])) if n else np.nan,
                'forecast_edge_active_share':float(np.mean(np.abs(edge[use])>=threshold)) if n else np.nan})
    return rows,edges


def render_decay(source,out,audit_all=False,additive=False):
    source=Path(source);out=Path(out)
    manifest=json.loads((source/'manifest.json').read_text());freeze=json.loads((source/'selection_freeze.json').read_text())
    if manifest['status']!='COMPLETED' or freeze['status']!='FROZEN':raise ValueError('decay requires completed frozen forecast evidence')
    cfg=yaml.safe_load((source/'config.yaml').read_text());spec=cfg['complex_alpha'];split=Split(**cfg['split'])
    if spec.get('return_contract')!='native_simple' or spec.get('label_projection')!='cash_common_beta':
        raise ValueError('declared decay diagnostic requires final native cash/common-beta contract')
    bank=Path(cfg['data']['feature_cache'])/manifest['source_signature'][:20]
    bank_manifest=json.loads((bank/'bank_manifest.json').read_text())
    if bank_manifest['signature']!=manifest['source_signature']:raise ValueError('decay bank signature mismatch')
    dates=np.load(bank/'dates.npy');fast_dates=np.load(bank/'fast_dates.npy');price=np.load(bank/'fast_close.npy',mmap_mode='r')
    beta=past_market_betas(price[np.searchsorted(fast_dates,dates)],width=2880 if cfg['data']['source']=='crypto_parquet' else 192,return_contract='native_simple')
    funding,_,derivatives=_funding(cfg,dates,fast_dates,price)
    raw,_=targets_from_prices(dates,fast_dates,price,[1,4,12,24],split,beta,funding,include_holdout=False,return_contract='native_simple')
    targets={h:project_basket_returns(value,beta) for h,value in raw.items()}
    if any(np.isfinite(value[dates>=split.holdout_start]).any() for value in targets.values()):raise ValueError('test labels reached reporting diagnostic')
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    masks={'discovery':(dates>=spec['folds'][1][0])&(dates<=split.discovery_end),
           'validation':(dates>=split.validation_start)&(dates<=split.validation_end)}
    records=[];audits=[];asset_audits=[];monthly=[];activation=[];activation_edges={}
    for model in pd.read_csv(source/'model_survival.csv').itertuples():
        prediction=np.load(source/f'{model.id}_prediction.npy',mmap_mode='r')
        execution=spec.get('execution_by_horizon',{}).get(model.horizon_hours,spec['execution'])
        edge=pd.DataFrame(np.nan_to_num(prediction,nan=0.)).ewm(span=execution['ema_bars'],adjust=False).mean().to_numpy()
        active=np.abs(edge)>=spec['edge_floor_bps']/1e4
        risk=np.load(source/f'target_price_scale_{model.horizon_hours}h.npy',mmap_mode='r')
        rows,cuts=volatility_activation(edge,risk,masks,spec['edge_floor_bps']/1e4)
        activation.extend({'id':model.id,**row} for row in rows);activation_edges[model.id]=cuts
        if audit_all:
            position=neutral_positions(prediction,beta,model.signal_scale,spec['gross_budget'],execution['ema_bars'],execution['deadband'],spec['edge_floor_bps'],spec['risk_mapping'],spec['exit_on_inactive'])
            ledger,asset_net,_=five_minute_ledger(position,dates,fast_dates,price,funding,spec['cost_bps'],split)
            for period in ledger.period.unique():
                selected=ledger.period.eq(period).to_numpy()
                audits.append({'id':model.id,'period':period,**ledger_metrics(ledger,asset_net,selected)})
                for j,code in enumerate(cfg['data']['assets']):asset_audits.append({'id':model.id,'period':period,'asset':code,'net_contribution_sum_pct':float(asset_net[selected,j].sum()*100/len(cfg['data']['assets']))})
            months=pd.to_datetime(ledger.completed_5m,format='%Y%m%d%H%M').dt.strftime('%Y-%m')
            for month,group in ledger.groupby(months):monthly.append({'id':model.id,'month':month,
                'net_return_pct':float((np.prod(1+group.net.to_numpy())-1)*100),'net_sum_pct':float(group.net.sum()*100)})
        for h,target in targets.items():
            phase=nonoverlap_phase(stamp,h)
            for period,mask in masks.items():records.append({'id':model.id,'forecast_horizon':model.horizon_hours,
                'return_horizon':h,'period':period,**forecast_decay_metrics(prediction,target,mask,active,phase)})
    frame=pd.DataFrame(records);frame.to_csv(out/'forecast_decay.csv',index=False)
    pd.DataFrame(activation).to_csv(out/'past_risk_activation.csv',index=False)
    (out/'past_risk_quintile_edges.json').write_text(json.dumps(activation_edges,ensure_ascii=False),encoding='utf-8')
    if audit_all:
        pd.DataFrame(audits).to_csv(out/'all_frozen_channel_audit.csv',index=False)
        pd.DataFrame(asset_audits).to_csv(out/'all_frozen_asset_contributions.csv',index=False)
        pd.DataFrame(monthly).to_csv(out/'all_frozen_monthly_contributions.csv',index=False)
    body=''
    for period in masks:
        figure=go.Figure()
        for identifier,g in frame[frame.period.eq(period)].groupby('id'):
            if g.mean_asset_ic.notna().any():figure.add_trace(go.Scatter(x=g.return_horizon.astype(str),y=g.mean_asset_ic,mode='lines+markers',name=identifier))
        figure.add_hline(y=0,line_dash='dot');figure.update_xaxes(title='Diagnostic return horizon (hours)');figure.update_yaxes(title='Mean signed per-asset IC')
        body+=_plot(figure,period+'：已冻结预测的期限曲线')
    body+='<p>原4h/12h预测未重训、未重新选择方向。曲线是预测与可交易投影收益的相关，不是基线外损失改善、显著性或净财富。1h/24h不参与任何晋级或执行调参；仅发现2023H2—2025H1与历史验证，没有测试标签。条件样本由冻结EMA后的8bps幅度选取，空信号的IC未定义。</p>'+_table(frame)
    body+='<h2>过去风险尺度与预测门限激活</h2><p>每币发现期分位冻结；并列归较高层，验证不重估。下表是EMA预测越过8bps的概率，尚未经过篮子对冲和无交易带，不是实际成交概率。风险单位内幅度与价格单位幅度须同时看。</p>'+_table(pd.DataFrame(activation))
    doc=f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>冻结预测的期限衰减</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><nav><a href='index.html'>四轮研究总览</a></nav><main><h1>结构信息是否主要存在于更短期限</h1>{body}</main></body></html>"
    (out/'decay.html').write_text(doc,encoding='utf-8')
    if audit_all:
        a=pd.DataFrame(audits);fig=go.Figure()
        for period,g in a.groupby('period',sort=False):fig.add_trace(go.Bar(x=g.id,y=g.net_return_pct,name=period))
        fig.update_layout(barmode='group',xaxis_tickangle=-60)
        note='<p>全部最佳发现配置，按源冻结执行合同重建；未因测试表现增删产品。原冻结组合不变。单腿贡献是资金损益归因，包含风险对冲腿，不是独立Alpha收益。</p>'
        content=_plot(fig,'已声明固定预测的逐段净财富')+_table(a)+_table(pd.DataFrame(asset_audits))+_table(pd.DataFrame(monthly))
        (out/'audit.html').write_text(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全部冻结通道迁移审计</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><nav><a href='index.html'>四轮总览</a> · <a href='decay.html'>期限衰减</a></nav><main><h1>固定产品的完整历史迁移</h1>{note}{content}</main></body></html>",encoding='utf-8')
    if additive:
        from scripts.complex_alpha_additive_review import additive_review
        comparison=additive_review(source,out,cfg,bank,bank_manifest,dates,fast_dates,price,beta,funding,derivatives,split)
        figure=go.Figure()
        for period,g in comparison.groupby('period',sort=False):figure.add_trace(go.Bar(x=g.id,y=g.additive_net_wealth_difference_pp,name=period))
        figure.update_layout(barmode='group',xaxis_tickangle=-60)
        note='<p>恢复保存的OLD参考预测，再加既有冻结增量；两策略共用发现期OLD幅度尺度和相同预算/执行/费用。没有重训和新选择。增量财富差不是独立Alpha收益；平均资本占用也须一起比较。连续审计路径在验证边界不额外清仓，源研究边界清仓另已精确对账。</p>'
        body=_plot(figure,'OLD + 修正，对比相同合同OLD的财富差（百分点）')+_table(comparison)
        (out/'additive.html').write_text(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>基线修正的真实经济贡献</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><nav><a href='index.html'>四轮总览</a> · <a href='audit.html'>独立通道</a></nav><main><h1>新增信息应该怎样加入基线</h1>{note}{body}</main></body></html>",encoding='utf-8')
    (out/'decay_manifest.json').write_text(json.dumps({'status':'COMPLETED','source':str(source.resolve()),
        'horizons':[1,4,12,24],'periods':list(masks),'decay_labels_use_holdout':False,'selection_feedback':False,
        'contract':'fixed-share funded simple returns projected to frozen-origin cash/common-beta space',
        'protocol':'docs/COMPLEX_ALPHA_DECAY_DIAGNOSTIC_PROTOCOL_20261008.md','rows':len(frame),
        'discovery_start':spec['folds'][1][0],'synthetic':cfg['data']['source']=='simulated',
        'supplemental_fixed_channel_audit':audit_all,'supplemental_audit_opens_historical_holdout_after_freeze':audit_all,
        'additive_baseline_comparison':additive,
        'audit_changes_frozen_selection':False,'time_contract':TIME_VERSION},ensure_ascii=False,indent=2),encoding='utf-8')
    return frame
