"""Frozen calibration probes: no refitting, selection or threshold retuning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,shutil
import numpy as np
import pandas as pd
import yaml
from src.alpha_mvp.research.effective_combo_extension import source_metadata
from src.alpha_mvp.research.effective_combo import positions_from_prediction
from src.alpha_mvp.research.complex_alpha_evaluation import five_minute_ledger,ledger_metrics
from src.alpha_mvp.research.complex_alpha_workflow import _funding


def review(history,oos,out):
    out=Path(out)
    if out.exists():raise ValueError('new frozen probe directory required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'review_source.py');rows=[];hashes={}
    for source,period in [(Path(history),'validation'),(Path(oos),'oos')]:
        load=lambda p:json.loads(p.read_text(encoding='utf-8'))
        if load(source/'manifest.json')['status']!='COMPLETED':raise ValueError('completed source required')
        cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))['complex_alpha']
        history_cfg=yaml.safe_load((Path(cfg['source_history'])/'config.yaml').read_text(encoding='utf-8'))['complex_alpha']
        _,mining,prior,_,pcfg=source_metadata(history_cfg['source_pool_run']);bank=Path(pcfg['complex_alpha']['bank_manifest']).parent
        from src.alpha_mvp.research.aligned_contract import Split
        split=Split(**prior['split']);end=load(source/'manifest.json')['results']['actual_end']
        full=np.load(bank/'dates.npy');full=full[:np.searchsorted(full,end,side='right')]
        fast=np.load(bank/'fast_dates.npy');fast=fast[:np.searchsorted(fast,end,side='right')]
        close=np.load(bank/'fast_close.npy',mmap_mode='r')[:len(fast)].copy();fund,files,_=_funding(prior,full,fast,close)
        if files!=load(mining/'input_signature.json')['derivative_files']:raise ValueError('source funding changed')
        dates=np.load(source/'dates.npy');pred=np.load(source/'state_cal1_prediction.npy').astype(float)
        states=pd.read_csv(source/'state_coefficients.csv');states=states.loc[states.window_months==1].copy()
        stamp=pd.to_datetime(dates,format='%Y%m%d%H%M');warm=np.zeros(len(dates),bool);negative=warm.copy();cap=np.ones(len(dates))
        detail=[]
        for row in states.itertuples():
            start=pd.to_datetime(str(row.update),format='%Y%m%d%H%M');mask=(stamp>=start)&(stamp<start+pd.DateOffset(months=1))
            warm[mask]=row.mature_labels<cfg['state_calibration']['minimum_labels'];negative[mask]=row.gamma<0
            if abs(row.gamma)>.5:cap[mask]=.5/abs(row.gamma)
            detail.append({'period':period,'update':str(row.update),'labels':row.mature_labels,'gamma':row.gamma,
                'warmup':row.mature_labels<cfg['state_calibration']['minimum_labels'],'bound_active':abs(row.gamma)>=1})
        probes={'full_replay':pred,'warmup_zero':np.where(warm[:,None],0,pred),
            'negative_gamma_zero':np.where(negative[:,None],0,pred),'gamma_abs_cap_half':pred*cap[:,None]}
        start=split.validation_start if period=='validation' else history_cfg['oos_start'];dm=(dates>=start)&(dates<=end);fm=(fast>=start)&(fast<=end)
        stats=pd.read_csv(source/'portfolio_metrics.csv')
        for name,value in probes.items():
            for execution in cfg['executions']:
                identifier='state_cal1__'+execution['id'];q=positions_from_prediction(value[dm],1.,execution,cfg['gross_budget'])
                frame,asset,_=five_minute_ledger(q,dates[dm],fast[fm],close[fm],fund[fm],cfg['cost_bps'],split)
                result=ledger_metrics(frame,asset,np.ones(len(frame),bool));rows.append({'id':identifier,'period':period,'probe':name,**result})
                if name=='full_replay':
                    ref=stats.loc[(stats.id==identifier)&stats.period.eq(period)].iloc[0]
                    error=max(abs(result[k]-ref[k]) for k in ['net_return_pct','sharpe','max_drawdown_pct'])
                    if error>1e-5:raise ValueError('frozen full replay mismatch')
        pd.DataFrame(detail).to_csv(out/(period+'_state_boundary_flags.csv'),index=False)
        for name in ['state_cal1_prediction.npy','state_coefficients.csv','portfolio_metrics.csv','config.yaml','manifest.json']:
            p=source/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(out/'frozen_calibration_probes.csv',index=False)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('probe source changed')
    (out/'manifest.json').write_text(json.dumps({'status':'COMPLETED','role':'fixed coefficient/input probes, not new selected strategies',
        'no_refit':True,'no_rescale':True,'sources_unchanged':hashes,'probes':['warmup_zero','negative_gamma_zero','gamma_abs_cap_half'],
        'interpretation':'all coefficients outside each intervention remain frozen; not additive causal PnL attribution'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--oos',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();review(a.history,a.oos,a.output)
