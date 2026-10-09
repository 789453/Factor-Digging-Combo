"""Reconstruct saved reference forecasts, not new training or selection."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from src.alpha_mvp.research.complex_alpha_evaluation import RidgeState,causal_risk_scales,neutral_positions,five_minute_ledger,ledger_metrics


def rebuild_old_design(bank,bank_manifest,scalers,dates,derivatives,calibration_end):
    raw_indices=[i for i,c in enumerate(bank_manifest['coordinates']) if c['family']=='OLD']
    codes=bank_manifest['codes'];design=np.empty((len(dates),len(codes),len(raw_indices)+len(derivatives)),np.float32)
    for j,code in enumerate(codes):
        parameters=scalers[j]
        if parameters['asset']!=code:raise ValueError('saved initial scaler asset order mismatch')
        mean=np.asarray(parameters['mean'],np.float32)[raw_indices];scale=np.asarray(parameters['scale'],np.float32)[raw_indices]
        valid=np.asarray(parameters['valid'],bool)[raw_indices];raw=np.load(bank/f'{code}.npy',mmap_mode='r')
        for start in range(0,len(dates),16384):
            values=np.clip(np.nan_to_num((raw[start:start+16384,raw_indices]-mean)/scale,nan=0.),-8,8)
            values[:,~valid]=0;design[start:start+len(values),j,:len(raw_indices)]=values
    for k,values in enumerate(derivatives.values()):
        sample=values[dates<=calibration_end];mean=np.nanmean(sample,axis=0);scale=np.nanstd(sample,axis=0)
        valid=np.isfinite(scale)&(scale>1e-12);mean[~valid]=0;scale[~valid]=1
        normalized=np.clip(np.nan_to_num((values-mean)/scale,nan=0.),-8,8);normalized[:,~valid]=0
        design[:,:,len(raw_indices)+k]=normalized.astype(np.float32)
    return design


def _state(record):
    return RidgeState(np.asarray(record['center']),np.asarray(record['scale']),np.asarray(record['coef']),record['intercept'],record['target_scale'])


def reference_forecasts(cfg,models,design,dates):
    forecasts={}
    for horizon in [4,12]:
        past=next(m for m in models if m['horizon']==horizon and m['kind']!='old')
        final=next(m for m in models if m['horizon']==horizon and m['kind']=='joint_interaction_residual')
        prediction=np.full(design.shape[:2],np.nan)
        for fold,record in zip(cfg['complex_alpha']['folds'],past['discovery_oof_states'],strict=True):
            start,end=fold
            if record['start']!=start:raise ValueError('reference fold alignment mismatch')
            mask=(dates>=start)&(dates<=end)
            prediction[mask]=_state(record['increment']['base_state']).predict(design[mask])
        forward=dates>cfg['split']['discovery_end']
        prediction[forward]=_state(final['design']['interaction_nuisance']).predict(design[forward])
        forecasts[horizon]=prediction
    return forecasts


def additive_review(source,out,cfg,bank,bank_manifest,dates,fast_dates,price,beta,funding,derivatives,split):
    spec=cfg['complex_alpha'];scalers=json.loads((source/'initial_coordinate_scaler.json').read_text())
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True);n=price.shape[1]
    for name,value in {'clock_sin':np.sin(2*np.pi*(stamp.hour+stamp.minute/60)/24),
                       'clock_cos':np.cos(2*np.pi*(stamp.hour+stamp.minute/60)/24),
                       'weekday_sin':np.sin(2*np.pi*stamp.dayofweek/7),'weekday_cos':np.cos(2*np.pi*stamp.dayofweek/7)}.items():
        derivatives[name]=np.repeat(np.asarray(value)[:,None],n,axis=1)
    design=rebuild_old_design(bank,bank_manifest,scalers,dates,derivatives,spec['feature_calibration_end'])
    models=json.loads((source/'frozen_models.json').read_text())['models']
    reference=reference_forecasts(cfg,models,design,dates);del design
    risk=causal_risk_scales(price[np.searchsorted(fast_dates,dates)],beta,[4,12],return_contract=spec['return_contract'],label_projection=spec['label_projection'])
    old_metrics={};scale={};checks=[];train=dates<=split.discovery_end
    for h,units in reference.items():
        reference[h]=units*risk[h];scale[h]=float(np.sqrt(np.nanmean(reference[h][train]**2)))
        execution=spec['execution_by_horizon'][h]
        position=neutral_positions(reference[h],beta,scale[h],spec['gross_budget'],execution['ema_bars'],execution['deadband'],spec['edge_floor_bps'],spec['risk_mapping'],spec['exit_on_inactive'])
        ledger,asset,_=five_minute_ledger(position,dates,fast_dates,price,funding,spec['cost_bps'],split)
        old_metrics[h]={p:ledger_metrics(ledger,asset,ledger.period.eq(p).to_numpy()) for p in ledger.period.unique()}
        # Source research ledger liquidates at validation end. Verify identical
        # endpoint before opening the longer audit path's reference comparison.
        stop=np.searchsorted(fast_dates,split.holdout_start)
        research,a,_=five_minute_ledger(position,dates,fast_dates[:stop],price[:stop],funding[:stop],spec['cost_bps'],split)
        expected=json.loads((source/f'old_baseline_{h}h_metrics.json').read_text())
        for period in ['discovery','validation']:
            metric=ledger_metrics(research,a,research.period.eq(period).to_numpy())
            np.testing.assert_allclose(metric['net_return_pct'],expected[period]['net_return_pct'],atol=1e-8,rtol=0)
            checks.append({'horizon':h,'period':period,'wealth_reconstruction_error_pct':metric['net_return_pct']-expected[period]['net_return_pct']})
    records=[]
    for model in pd.read_csv(source/'model_survival.csv').itertuples():
        h=model.horizon_hours;prediction=reference[h]+np.load(source/f'{model.id}_prediction.npy',mmap_mode='r')
        execution=spec['execution_by_horizon'][h]
        position=neutral_positions(prediction,beta,scale[h],spec['gross_budget'],execution['ema_bars'],execution['deadband'],spec['edge_floor_bps'],spec['risk_mapping'],spec['exit_on_inactive'])
        ledger,asset,_=five_minute_ledger(position,dates,fast_dates,price,funding,spec['cost_bps'],split)
        for period in ledger.period.unique():
            metric=ledger_metrics(ledger,asset,ledger.period.eq(period).to_numpy());base=old_metrics[h][period]
            records.append({'id':model.id,'horizon':h,'period':period,**metric,
                'old_net_return_pct':base['net_return_pct'],'additive_net_wealth_difference_pp':metric['net_return_pct']-base['net_return_pct'],
                'old_mean_abs_position':base['mean_abs_position'],'old_turnover':base['turnover']})
    frame=pd.DataFrame(records);frame.to_csv(out/'additive_frozen_channel_audit.csv',index=False)
    (out/'additive_reconstruction_checks.json').write_text(json.dumps({'status':'PASS','checks':checks,
        'reference_scale':'same discovery OLD RMS for both strategies','new_training':False,'new_selection':False,
        'note':'historical test audit; terminal close at each source research endpoint checked separately'},ensure_ascii=False,indent=2),encoding='utf-8')
    return frame
