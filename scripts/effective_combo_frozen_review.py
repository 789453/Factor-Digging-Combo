"""Read-only audit of completed effective-combo evidence; no fitting or selection.

Production experiments remain exclusively research_cli. This script reproduces
frozen model predictions, zero-input diagnostics and fixed ensemble removals.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,shutil
import numpy as np
import pandas as pd
import yaml
from lightgbm import Booster
from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_evaluation import RidgeState,five_minute_ledger,ledger_metrics
from src.alpha_mvp.research.complex_alpha_prediction import prediction_metrics
from src.alpha_mvp.research.complex_alpha_workflow import _funding
from src.alpha_mvp.research.effective_combo import asset_conditional_design,positions_from_prediction


def fixed_average(predictions, removed=None):
    """Removed member becomes zero; original divisor and all other units stay fixed."""
    return sum(p if name!=removed else np.zeros_like(p) for name,p in predictions.items())/len(predictions)


def review(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('immutable review: new output directory required')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));c=cfg['complex_alpha']
    if load(source/'manifest.json')['status']!='COMPLETED':raise ValueError('completed evidence required')
    pool=Path(c['source_pool_run']);pcfg=yaml.safe_load((pool/'config.yaml').read_text(encoding='utf-8'))
    mining=Path(pcfg['complex_alpha']['source_run']);prior=yaml.safe_load((mining/'config.yaml').read_text(encoding='utf-8'))
    split=Split(**prior['split']);bank=Path(pcfg['complex_alpha']['bank_manifest']).parent
    contract=load(source/'feature_contract.json');codes=contract['assets']
    dates=np.load(source/'dates.npy');target=np.load(source/'target.npy');val=dates>=split.validation_start
    if dates[-1]>=split.holdout_start:raise ValueError('holdout firewall')
    inputs=[p for p in source.rglob('*') if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'review_source.py')
    write=lambda name,value:(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    write('review_contract.json',{'role':'read-only diagnostics, no refit/ranking/tuning',
       'preset_showcase':'ensemble__edge16','additional_preset':'tree15_online__edge16',
       'fixed_member_removal':'replace by zero, original divisor 3',
       'fixed_input_removal':'replace specified numeric coordinates by zero; no model update',
       'reproduction_tolerance_return_pct':1e-5,'holdout_opened':False})
    stats=pd.read_csv(source/'portfolio_metrics.csv');summaries=[];monthly=[];quarterly=[];assets=[]
    for key in stats.id.unique():
        f=pd.read_parquet(source/(key+'_validation_ledger.parquet'));d=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M')
        a=np.load(source/(key+'_validation_asset_path.npz'));tr=pd.read_csv(source/(key+'_validation_trades.csv'))
        wealth=np.cumprod(1+f.net.to_numpy());capital=np.r_[1,wealth[:-1]]
        con=(a['net']*capital[:,None]/len(codes)).sum(axis=0)*100
        for freq,store in [('M',monthly),('Q',quarterly)]:
            for period in d.dt.to_period(freq).unique():
                mask=d.dt.to_period(freq)==period;part=f.loc[mask]
                store.append({'id':key,'period':str(period),'net_return_pct':(np.prod(1+part.net)-1)*100,
                    'price_sum_pct':part.price_gross.sum()*100,'funding_sum_pct':part.funding.sum()*100,
                    'fee_sum_pct':part.fee.sum()*100,'mean_abs_position':part.mean_abs_position.mean()})
        v=stats.loc[(stats.id==key)&(stats.period=='validation')].iloc[0].to_dict()
        folds=stats.loc[(stats.id==key)&stats.period.str.startswith('fold')]
        q=[r['net_return_pct'] for r in quarterly if r['id']==key]
        done=tr.loc[~tr.censored]
        v.update(positive_discovery_folds=int((folds.net_return_pct>0).sum()),positive_validation_segments=sum(t>0 for t in q),
           natural_episodes=len(done),censored_episodes=int(tr.censored.sum()),holding_median_hours=float(done.hours.median()),
           holding_p90_hours=float(done.hours.quantile(.9)),positive_contribution_assets=int((con>0).sum()),
           max_positive_asset_share=float(max(con)/max(con[con>0].sum(),1e-16)),
           active_time_share=float((f.mean_abs_position>1e-5).mean()),
           mean_net_exposure=float(f.cash_exposure.mean()),mean_abs_net_exposure=float(f.cash_exposure.abs().mean()),
           wealth_attribution_error_pct=float(abs(con.sum()-(wealth[-1]-1)*100)))
        v['historical_rule_pass']=bool(v['positive_discovery_folds']>=2 and v['net_return_pct']>0 and sum(t>0 for t in q)>=2)
        # Thin Ridge is separately flagged; activity and concentration remain observable judgments.
        v['activity_flag']='thin' if v['mean_abs_position']<.01 else 'substantive'
        summaries.append(v)
        assets.extend({'id':key,'asset':code,'contribution_pct':float(con[j]),'mean_abs_position':float(np.mean(abs(a['position'][:,j])))} for j,code in enumerate(codes))
    for name,rows in [('acceptance',summaries),('monthly',monthly),('quarterly',quarterly),('asset_contribution',assets)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    x=asset_conditional_design(np.concatenate([np.load(pool/(name+'_features.npy')) for name in ['old_coordinates','old_mined','mixed_mined']],axis=-1))[val]
    dv=dates[val];tv=target[val]
    fast=np.load(bank/'fast_dates.npy');n=np.searchsorted(fast,split.validation_end,side='right');fast=fast[:n]
    close=np.load(bank/'fast_close.npy',mmap_mode='r')[:n].copy()
    full=np.load(bank/'dates.npy');full=full[:np.searchsorted(full,split.validation_end,side='right')]
    fund,fingerprints,_=_funding(prior,full,fast,close)
    if fingerprints!=load(mining/'input_signature.json')['derivative_files']:raise ValueError('funding changed')
    fm=fast>=split.validation_start;fast=fast[fm];close=close[fm];fund=fund[fm]
    members={k:np.load(source/(k+'_prediction.npy'))[val].astype(float) for k in c['ensemble']}
    execution=next(e for e in c['executions'] if e['id']=='edge16')
    rows=[];pred_stats=[]
    def replay(key,p):
        q=positions_from_prediction(p,1.,execution,c['gross_budget'])
        f,a,actual=five_minute_ledger(q,dv,fast,close,fund,c['cost_bps'],split)
        f.to_parquet(out/(key+'_ledger.parquet'),index=False);np.save(out/(key+'_prediction.npy'),p.astype(np.float32))
        metric={'diagnostic':key,**ledger_metrics(f,a,np.ones(len(f),bool))};rows.append(metric)
        pred_stats.append({'diagnostic':key,**prediction_metrics(p,tv)})
        return metric
    fullprediction=fixed_average(members);reproduced=replay('full_fixed',fullprediction)
    reference=stats.loc[(stats.id=='ensemble__edge16')&(stats.period=='validation')].iloc[0]
    error=abs(reproduced['net_return_pct']-reference.net_return_pct)
    if error>1e-5:raise ValueError('frozen full replay does not reproduce source')
    for removed in members:replay('remove_member_'+removed,fixed_average(members,removed))
    # Restore each monthly frozen readout. Zero probes never retrain or choose a winner.
    def predict_frozen(z):
        pp={}
        for name in c['ensemble']:
            result=np.full(tv.shape,np.nan)
            files=sorted((source/'models'/name).glob('*'))
            for i,file in enumerate(files):
                start=file.stem
                if start<split.validation_start:continue
                stop=files[i+1].stem if i+1<len(files) else split.holdout_start
                mask=(dv>=start)&(dv<stop)
                if not mask.any():continue
                if file.suffix=='.json':
                    record=load(file)
                    model=RidgeState(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in record.items()})
                    result[mask]=model.predict(z[mask])/1e4
                else:
                    model=Booster(model_file=str(file));result[mask]=model.predict(z[mask].reshape(-1,z.shape[-1]),num_threads=6).reshape(-1,len(codes))/1e4
            if not np.isfinite(result).all():raise ValueError('incomplete frozen prediction clock')
            pp[name]=result
        return fixed_average(pp)
    restored=predict_frozen(x)
    write('reproduction.json',{'saved_float32_replay_error_return_pct':error,'restored_model_prediction_max_error':float(np.max(abs(restored-fullprediction)))})
    metadata={r['id']:r for r in load(source/'feature_pool_freeze.json')['features']}
    conditional=[i for i,name in enumerate(contract['features']) if metadata.get(name,{}).get('research_tier')=='conditional_or_unstable']
    stable=[i for i,name in enumerate(contract['features']) if metadata.get(name,{}).get('research_tier')=='predictive_candidate']
    for group,indices in [('old_coordinates',range(106)),('old_mined',range(106,154)),('mixed_mined',range(154,202)),
                         ('asset_identity',range(202,x.shape[-1])),('conditional_functions',conditional),('stable_functions',stable)]:
        z=x.copy();z[...,list(indices)]=0;replay('zero_input_'+group,predict_frozen(z))
    pd.DataFrame(rows).to_csv(out/'fixed_removal.csv',index=False);pd.DataFrame(pred_stats).to_csv(out/'fixed_removal_prediction.csv',index=False)
    # Forecast group bins fixed before reading outcomes; no calibrator or retuned execution.
    bins=[-np.inf,-80,-40,-16,-8,0,8,16,40,80,np.inf];groups=[]
    for model in ['ridge_asset_online','tree15_online','tree31_online','ensemble']:
        p=np.load(source/(model+'_prediction.npy'))[val];good=np.isfinite(tv)&np.isfinite(p)
        for lo,hi in zip(bins[:-1],bins[1:]):
            mask=good&(p*1e4>=lo)&(p*1e4<hi)
            groups.append({'model':model,'bin_low_bps':lo,'bin_high_bps':hi,'labels':int(mask.sum()),
                'predicted_mean_bps':float(np.mean(p[mask])*1e4) if mask.any() else None,
                'realized_mean_bps':float(np.mean(tv[mask])*1e4) if mask.any() else None})
    pd.DataFrame(groups).to_csv(out/'forecast_groups.csv',index=False)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('source changed')
    write('source_hashes.json',hashes)
    write('manifest.json',{'status':'COMPLETED','stage':'read_only_effective_review','source':str(source.resolve()),'holdout_opened':False,
       'portfolios':len(summaries),'fixed_diagnostics':len(rows),'fitting_performed':False,
       'preset_products':['ensemble__edge16','tree15_online__edge16','tree7_static__edge16'],
       'quarter_caveat':'2026Q1 is January only; Q3/Q4 are full quarters'})
    print(pd.DataFrame(rows).to_string(index=False));print('Review completed: '+str(out))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();review(args.source,args.output)
