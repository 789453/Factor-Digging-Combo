"""Frozen predictor interventions on historical validation; no fit or selection.

This is an explanatory audit, not another production research entry point.
Remove centered POOL coordinates while retaining OLD nuisance, all coefficients,
OOF amplitude, signal scale and execution. No holdout labels/metrics are read.
"""
from __future__ import annotations
import argparse,ast,hashlib,html,json,shutil
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import yaml
from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_evaluation import past_market_betas,targets_from_prices,neutral_positions,five_minute_ledger,ledger_metrics
from src.alpha_mvp.research.complex_alpha_search import Candidate,evaluate_candidate,correlation
from src.alpha_mvp.research.complex_alpha_workflow import _funding
from src.alpha_mvp.research.factor_combo_reporting import _CSS,_plot,_table


def coordinate_contribution(values,state,index):
    center=np.asarray(state['center']);scale=np.asarray(state['scale']);coef=np.asarray(state['coef'])
    if not 0<=index<len(center) or not np.isfinite(scale[index]) or scale[index]<=0:
        raise ValueError('invalid frozen coordinate or scale')
    if not np.isfinite(values).all():raise ValueError('reconstructed coordinate contains missing/nonfinite values')
    z=np.clip((np.asarray(values,dtype=float)-center[index])/scale[index],-8,8)
    return z*coef[index]*state['target_scale']


def same_function(source,name,file):
    def body(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    if body(source/'source_snapshot'/file)!=body(Path('src/alpha_mvp/research')/file):
        raise ValueError('replay function differs from original: '+name)


def run(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('new diagnostic directory required')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    frozen=json.loads((source/'selection_freeze.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETED' or frozen['selected']!=['pool_4h_trial61']:
        raise ValueError('this predeclared audit requires the original R3 singleton')
    for file,names in [('complex_alpha_search.py',['evaluate_candidate']),
        ('complex_alpha_workflow.py',['_funding']),('complex_alpha_evaluation.py',['five_minute_ledger','_self_financing_path'])]:
        for name in names:same_function(source,name,file)
    for item in manifest['source_files']:
        p=Path(item['path']);st=p.stat()
        if st.st_size!=item['size'] or st.st_mtime_ns!=item['mtime_ns']:raise ValueError('input fingerprint changed')
    out.mkdir();(out/'manifest.json').write_text(json.dumps({'status':'RUNNING','holdout_used':False}))
    try:
        cfg=yaml.safe_load((source/'config.yaml').read_text());s=cfg['complex_alpha'];split=Split(**cfg['split'])
        bank=Path(cfg['data']['feature_cache'])/manifest['source_signature'][:20]
        bank_meta=json.loads((bank/'bank_manifest.json').read_text());codes=bank_meta['codes'];raw_count=len(bank_meta['coordinates'])
        all_dates=np.load(bank/'dates.npy');stop=np.searchsorted(all_dates,split.holdout_start);dates=all_dates[:stop]
        all_fast=np.load(bank/'fast_dates.npy');fast_stop=np.searchsorted(all_fast,split.holdout_start);fast=all_fast[:fast_stop]
        price=np.load(bank/'fast_close.npy',mmap_mode='r')[:fast_stop]
        validation=(dates>=split.validation_start)&(dates<=split.validation_end)
        scalers=json.loads((source/'initial_coordinate_scaler.json').read_text())
        models=json.loads((source/'frozen_models.json').read_text())['models']
        model=next(m for m in models if m['id']=='pool_4h_trial61')
        cards=json.loads((source/'readout_coordinate_registry.json').read_text())
        registry={c['id']:c for c in json.loads((source/'candidate_registry.json').read_text())['definitions']}
        pool=[(j,cards[index]) for j,index in enumerate(model['design']['coordinates']) if cards[index]['family']=='POOL']
        candidates=[Candidate(**registry[card['name']]) for _,card in pool]
        needed=sorted({i for c in candidates for i in tuple(c.left)+tuple(c.right)})
        mapping={k:i for i,k in enumerate(needed)}
        x=np.empty((validation.sum(),len(codes),len(needed)),np.float32)
        raw_indices=[k for k in needed if k<raw_count];dest=[mapping[k] for k in raw_indices]
        for j,code in enumerate(codes):
            parameters=scalers[j]
            if parameters['asset']!=code:raise ValueError('scaler asset order differs')
            raw=np.load(bank/f'{code}.npy',mmap_mode='r')
            mu=np.asarray(parameters['mean'],np.float32)[raw_indices];sd=np.asarray(parameters['scale'],np.float32)[raw_indices]
            admitted=np.asarray(parameters['valid'],bool)[raw_indices]
            values=np.clip(np.nan_to_num((raw[:stop][validation][:,raw_indices]-mu)/sd,nan=0.),-8,8)
            values[:,~admitted]=0
            asset_values=x[:,j,:];asset_values[:,dest]=values
        funding,_,controls=_funding(cfg,dates,fast,price)
        stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
        for name,value in {'clock_sin':np.sin(2*np.pi*(stamp.hour+stamp.minute/60)/24),
            'clock_cos':np.cos(2*np.pi*(stamp.hour+stamp.minute/60)/24),
            'weekday_sin':np.sin(2*np.pi*stamp.dayofweek/7),'weekday_cos':np.cos(2*np.pi*stamp.dayofweek/7)}.items():
            controls[name]=np.repeat(np.asarray(value)[:,None],len(codes),axis=1)
        names=list(controls)
        for k in needed:
            if k<raw_count:continue
            name=names[k-raw_count]
            if cards[k]['name']!=name:raise ValueError('control order differs')
            v=controls[name];sample=v[dates<=s['feature_calibration_end']]
            mu=np.nanmean(sample,axis=0);sd=np.nanstd(sample,axis=0);valid=np.isfinite(sd)&(sd>1e-12)
            mu[~valid]=0;sd[~valid]=1
            values=np.clip(np.nan_to_num((v[validation]-mu)/sd,nan=0.),-8,8);values[:,~valid]=0
            x[:,:,mapping[k]]=values.astype(np.float32)
        contributions={family:np.zeros(x.shape[:2]) for family in ['OLD','F1','F2','F3','F4','F5','F6','F7']}
        for (j,card),c in zip(pool,candidates,strict=True):
            c=replace(c,left=tuple(mapping[k] for k in c.left),right=tuple(mapping[k] for k in c.right))
            # Production POOL values were explicitly stored float32 before ridge.
            value=evaluate_candidate(x,c).astype(np.float32)
            contributions[card['source_family']]+=coordinate_contribution(value,model['state'],j)
        del x,models,registry
        risk=np.load(source/'target_price_scale_4h.npy',mmap_mode='r')[:stop]
        amplitude=model['oof_gamma']*model['increment']['gamma']
        contributions={k:v*risk[validation]*amplitude for k,v in contributions.items()}
        interventions={'original':np.zeros_like(contributions['OLD']),
            **{'remove_'+k:v for k,v in contributions.items()},
            'remove_all_structural_pool':sum(v for k,v in contributions.items() if k!='OLD'),
            'remove_all_pool':sum(contributions.values())}
        prediction=np.load(source/'pool_4h_trial61_prediction.npy',mmap_mode='r')[:stop]
        beta=past_market_betas(price[np.searchsorted(fast,dates)],return_contract='log')
        _,targets=targets_from_prices(dates,fast,price,[4],split,beta,funding,include_holdout=False,return_contract='log')
        original=pd.read_csv(source/'model_survival.csv').set_index('id').loc[model['id']]
        records=[];curves=[]
        for name,removed in interventions.items():
            signal=np.array(prediction,dtype=float);signal[validation]-=removed
            e=model['execution']
            position=neutral_positions(signal,beta,model['scale'],s['gross_budget'],e['ema_bars'],e['deadband'],s['edge_floor_bps'],'legacy_loo',False)
            ledger,asset,_=five_minute_ledger(position,dates,fast,price,funding,4.,split)
            mask=ledger.period.eq('validation').to_numpy();metric=ledger_metrics(ledger,asset,mask)
            if name=='original':
                np.testing.assert_allclose(metric['net_return_pct'],original.validation_net_return_pct,atol=1e-6,rtol=0)
                replay_error=metric['net_return_pct']-original.validation_net_return_pct
            record={'intervention':name,**metric,'validation_residual_ic':correlation(signal[validation]/risk[validation],targets[4][validation]/risk[validation]),
                'removed_forecast_rms_bps':float(np.sqrt(np.nanmean(removed**2))*1e4)}
            records.append(record)
            frame=ledger.loc[mask].copy();clock=pd.to_datetime(frame.completed_5m,format='%Y%m%d%H%M',utc=True)
            series=pd.Series(frame.net.to_numpy(),index=clock).resample('1D').apply(lambda v:np.prod(1+v)-1)
            curves.append((name,series))
        frame=pd.DataFrame(records);frame['wealth_difference_from_original_pp']=frame.net_return_pct-frame.iloc[0].net_return_pct
        frame.to_csv(out/'fixed_readout_ablation.csv',index=False)
        figure=go.Figure()
        for name,series in curves:figure.add_trace(go.Scatter(x=series.index,y=(np.cumprod(1+series)-1)*100,mode='lines',name=name))
        figure.update_yaxes(title='验证期复利净财富%，4bps；原幅度和执行不改')
        text='<p>固定R3模型：只在验证开始后把指定POOL家族坐标替换为其ridge训练中心，保留所有其他系数、OLD剔除投影、OOF幅度和原仓位尺度。不是删除原料后重训，不证明因果必要性，也不是可晋级的新策略。全部预声明移除结果展示，不读测试，不以诊断重选产品。预测保存为float32，原版复原容差事先固定为1e-6百分点。</p>'
        (out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
        (out/'index.html').write_text(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>冻结联合读出的家族移除诊断</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><main><h1>冻结联合读出的家族移除诊断</h1>{text}{_plot(figure,'全部固定移除：历史验证')}{_table(frame)}</main></body></html>",encoding='utf-8')
        shutil.copyfile(__file__,out/'diagnostic_source.py')
        meta={'status':'COMPLETED','source':str(source.resolve()),'product':model['id'],'holdout_used':False,'new_fits':0,'selection_feedback':False,
            'scope':'fixed readout coordinate interventions on historical validation only; not refitted conditional feature absence',
            'original_wealth_replay_error_pp':float(replay_error),'original_replay_tolerance_pp':1e-6,'coordinate_count':len(needed),
            'diagnostic_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(frame.to_string(index=False))
    except Exception as exc:
        (out/'manifest.json').write_text(json.dumps({'status':'FAILED','error':str(exc),'holdout_used':False},ensure_ascii=False,indent=2),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
