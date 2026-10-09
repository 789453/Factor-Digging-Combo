"""Read-only feature, forecast, cost and exposure diagnostics on frozen products."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,shutil,hashlib
import numpy as np
import pandas as pd
import yaml
from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_evaluation import five_minute_ledger,ledger_metrics
from src.alpha_mvp.research.complex_alpha_prediction import prediction_metrics
from src.alpha_mvp.research.effective_combo import positions_from_prediction
from src.alpha_mvp.research.complex_alpha_workflow import _funding


def review(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('new immutable output required')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if load(source/'manifest.json')['status']!='COMPLETED':raise ValueError('completed input required')
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));c=cfg['complex_alpha'];pool=Path(c['source_pool_run'])
    pcfg=yaml.safe_load((pool/'config.yaml').read_text(encoding='utf-8'))
    mining=Path(pcfg['complex_alpha']['source_run']);prior=yaml.safe_load((mining/'config.yaml').read_text(encoding='utf-8'));split=Split(**prior['split'])
    contract=load(source/'feature_contract.json');codes=contract['assets'];dates=np.load(source/'dates.npy');y=np.load(source/'target.npy')
    if dates[-1]>=split.holdout_start:raise ValueError('holdout firewall')
    val=dates>=split.validation_start;disc=(dates>='202401010000')&(dates<=split.discovery_end)
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'review_source.py')
    write=lambda name,value:(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    feature=np.concatenate([np.load(pool/(name+'_features.npy')) for name in ['old_coordinates','old_mined','mixed_mined']],axis=-1)
    registry=load(source/'feature_pool_freeze.json')['features'];meta={r['id']:r for r in registry}
    rows=[]
    for j,name in enumerate(contract['features'][:202]):
        r=meta.get(name,{})
        d=prediction_metrics(feature[disc,:,j],y[disc]);v=prediction_metrics(feature[val,:,j],y[val])
        # IC is meaningful; raw feature units are not return predictions or valid R2 baselines.
        rows.append({'feature':name,'stream':r.get('stream','old_coordinates'),'family':r.get('family','OLD'),
           'research_tier':r.get('research_tier','known_coordinate'),'frozen_direction':r.get('direction',None),
           'discovery_raw_ic':d['ic'],'validation_raw_ic':v['ic'],
           'discovery_signed_ic':d['ic']*r.get('direction',1),'validation_signed_ic':v['ic']*r.get('direction',1),
           'validation_observations':v['observations']})
    pd.DataFrame(rows).to_csv(out/'individual_features.csv',index=False)
    stamps=pd.to_datetime(dates[val],format='%Y%m%d%H%M');models=[]
    for name in ['ridge_asset_online','tree7_static','tree15_online','tree31_online','tree63_online','fit255_static','ensemble']:
        p=np.load(source/(name+'_prediction.npy'))[val]
        for j,code in enumerate(codes):models.append({'model':name,'axis':'asset','segment':code,**prediction_metrics(p[:,j],y[val,j])})
        for month in stamps.to_period('M').unique():
            mask=stamps.to_period('M')==month
            models.append({'model':name,'axis':'month','segment':str(month),**prediction_metrics(p[mask],y[val][mask])})
    pd.DataFrame(models).to_csv(out/'prediction_breadth.csv',index=False)
    bank=Path(pcfg['complex_alpha']['bank_manifest']).parent
    fast=np.load(bank/'fast_dates.npy');n=np.searchsorted(fast,split.validation_end,side='right');fast=fast[:n]
    close=np.load(bank/'fast_close.npy',mmap_mode='r')[:n].copy();full=np.load(bank/'dates.npy');full=full[:np.searchsorted(full,split.validation_end,side='right')]
    fund,fp,_=_funding(prior,full,fast,close)
    if fp!=load(mining/'input_signature.json')['derivative_files']:raise ValueError('funding changed')
    mask=fast>=split.validation_start;fast=fast[mask];close=close[mask];fund=fund[mask]
    execution=next(e for e in c['executions'] if e['id']=='edge16');costs=[];exposure=[]
    forward=np.zeros_like(close);forward[:-1]=close[1:]/close[:-1]-1
    for model in ['ensemble','tree15_online','tree7_static']:
        p=np.load(source/(model+'_prediction.npy'))[val].astype(float);q=positions_from_prediction(p,1.,execution,c['gross_budget'])
        for cost in [0,2,4,6,8,12]:
            f,a,actual=five_minute_ledger(q,dates[val],fast,close,fund,cost,split)
            row={'id':model+'__edge16','cost_bps':cost,**ledger_metrics(f,a,np.ones(len(f),bool))};costs.append(row)
            f.to_parquet(out/(model+'_cost'+str(cost)+'_ledger.parquet'),index=False)
        f=pd.read_parquet(source/(model+'__edge16_validation_ledger.parquet'));a=np.load(source/(model+'__edge16_validation_asset_path.npz'))
        common=a['position'].mean(axis=1)*forward.mean(axis=1)
        relative=f.price_gross.to_numpy()-common;capital=np.r_[1,np.cumprod(1+f.net.to_numpy())[:-1]]
        check=np.max(abs(np.mean(a['position']*forward,axis=1)-f.price_gross))
        if check>1e-12:raise ValueError('exposure price reconstruction')
        signed_long=np.mean(np.maximum(a['position'],0)*forward,axis=1)
        signed_short=np.mean(np.minimum(a['position'],0)*forward,axis=1)
        exposure.append({'id':model+'__edge16','net_exposure_mean':f.cash_exposure.mean(),
           'abs_net_exposure_mean':f.cash_exposure.abs().mean(),'mean_gross':f.mean_abs_position.mean(),
           'common_price_wealth_contribution_pct':float(np.sum(capital*common)*100),
           'relative_price_wealth_contribution_pct':float(np.sum(capital*relative)*100),
           'long_price_wealth_contribution_pct':float(np.sum(capital*signed_long)*100),
           'short_price_wealth_contribution_pct':float(np.sum(capital*signed_short)*100),
           'funding_wealth_contribution_pct':float(np.sum(capital*f.funding)*100),
           'fee_wealth_contribution_pct':float(np.sum(capital*f.fee)*100),
           'price_reconstruction_error':float(check)})
    pd.DataFrame(costs).to_csv(out/'cost_sensitivity.csv',index=False);pd.DataFrame(exposure).to_csv(out/'exposure_decomposition.csv',index=False)
    write('manifest.json',{'status':'COMPLETED','role':'read-only frozen signal and exposure review',
        'source':str(source.resolve()),'holdout_opened':False,'feature_count':202,'fitting_performed':False,
        'exposure_contract':'common equal-asset price basket decomposition; not beta-neutral return or causal alpha',
        'cost_contract':'same forecast/execution, replay actual shares and terminal close at every cost',
        'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()})
    print(pd.DataFrame(costs).to_string(index=False));print(pd.DataFrame(exposure).to_string(index=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();review(a.source,a.output)
