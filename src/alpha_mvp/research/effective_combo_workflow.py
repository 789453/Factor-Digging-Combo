"""Effective combo research inside the existing aligned/complex production entry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .aligned_contract import Split
from .complex_alpha_evaluation import five_minute_ledger, ledger_metrics
from .complex_alpha_prediction import prediction_metrics
from .effective_combo import (asset_conditional_design, mature_window, fit_model,
    causal_predictions, positions_from_prediction, completed_trades)
from .manual_alpha_workflow import _write_json, _write_csv


def validate_effective_config(cfg):
    if set(cfg) != {'mode','version','complex_alpha','output'} or cfg['mode']!='aligned_crypto':
        raise ValueError('explicit aligned crypto declaration required')
    s=cfg['complex_alpha']
    if set(s) != {'stage','source_pool_run','models','ensemble','executions','cost_bps','gross_budget',
                  'seed','threads','label_clip_bps','diagnostic_model'} or s['stage']!='effective_combo':
        raise ValueError('invalid effective combo configuration')
    if not 1<=len(s['models'])<=8 or not 1<=s['threads']<=12 or not 0<s['gross_budget']<=1 or s['cost_bps']<0:
        raise ValueError('invalid bounded models/resources/execution')
    if not 100<=s['label_clip_bps']<=5000: raise ValueError('invalid declared training label clip')
    keys={'id','backend','trees','leaves','min_leaf','learning_rate','penalty','window_months','update_months'}
    ids=[]
    for spec in s['models']:
        if set(spec)!=keys or spec['backend'] not in {'ridge','lightgbm'}:
            raise ValueError('invalid explicit model declaration')
        if not 1<=spec['trees']<=1000 or not 2<=spec['leaves']<=511 or not 4<=spec['min_leaf']<=2048:
            raise ValueError('bounded nonlinear capacity required')
        if not 0<spec['learning_rate']<=.2 or spec['penalty']<=0 or spec['window_months'] not in {0,12} or spec['update_months'] not in {0,1,3}:
            raise ValueError('invalid training protocol')
        if not spec['id'].replace('_','').isalnum(): raise ValueError('unsafe model id')
        ids.append(spec['id'])
    if len(set(ids))!=len(ids) or s['diagnostic_model'] not in ids or not set(s['ensemble']).issubset(ids) or not s['ensemble']:
        raise ValueError('model ids/diagnostic/ensemble mismatch')
    if not 1<=len(s['executions'])<=4: raise ValueError('bounded executions required')
    names=[]
    for e in s['executions']:
        if set(e)!={'id','name','floor_bps','amplitude_bps'} or e['name'] not in {'rms','net_edge'} or e['floor_bps']<0 or e['amplitude_bps']<=0:
            raise ValueError('invalid explicit execution contract')
        names.append(e['id'])
    if len(set(names))!=len(names): raise ValueError('duplicate execution ids')
    if set(cfg['output'])!={'out_dir','visualization_dir'}: raise ValueError('explicit outputs required')


def run_effective_combo(cfg,config_path):
    validate_effective_config(cfg)
    from .complex_alpha_workflow import _funding
    s=cfg['complex_alpha']; source=Path(s['source_pool_run'])
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=load(source/'manifest.json')
    if manifest['status']!='COMPLETED' or manifest['stage']!='predictive_pool':
        raise ValueError('completed numeric prediction source required')
    contract=load(source/'numeric_pool_contract.json'); codes=contract['assets']; horizon=contract['horizon_hours']
    src_cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    mining=Path(src_cfg['complex_alpha']['source_run']); bankroot=Path(src_cfg['complex_alpha']['bank_manifest']).parent
    prior=yaml.safe_load((mining/'config.yaml').read_text(encoding='utf-8')); split=Split(**prior['split'])
    dates=np.load(source/'dates.npy'); y=np.load(source/'raw_return_target.npy')
    if np.any(dates[1:]<=dates[:-1]) or dates[-1]>=split.holdout_start: raise ValueError('source clock/firewall mismatch')
    paths=[source/(name+'_features.npy') for name in ['old_coordinates','old_mined','mixed_mined']]
    matrices=[np.load(p) for p in paths]
    if any(a.shape[:2]!=y.shape for a in matrices): raise ValueError('numeric pool axis mismatch')
    x=asset_conditional_design(np.concatenate(matrices,axis=-1))
    names=sum([contract['feature_order'][name] for name in ['old_coordinates','old_mined','mixed_mined']],[])+['asset_'+c for c in codes]
    if len(names)!=x.shape[-1]: raise ValueError('numeric feature order mismatch')
    full_dates=np.load(bankroot/'dates.npy'); full_dates=full_dates[:np.searchsorted(full_dates,split.validation_end,side='right')]
    fast=np.load(bankroot/'fast_dates.npy'); n=np.searchsorted(fast,split.validation_end,side='right'); fast=fast[:n]
    close=np.load(bankroot/'fast_close.npy',mmap_mode='r')[:n].copy()
    funding,fingerprints,_=_funding(prior,full_dates,fast,close)
    if fingerprints!=load(mining/'input_signature.json')['derivative_files']: raise ValueError('funding source changed')
    out=Path(cfg['output']['out_dir']); viz=Path(cfg['output']['visualization_dir'])
    if out.exists() or viz.exists(): raise ValueError('completed/new effective outputs must not be overwritten')
    out.mkdir(parents=True)
    _write_json(out/'manifest.json',{'status':'RUNNING','stage':'effective_combo'})
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    tracked=paths+[source/p for p in ['config.yaml','manifest.json','numeric_pool_contract.json','raw_return_target.npy','dates.npy','feature_pool_freeze.json']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    _write_json(out/'source_provenance.json',hashes)
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for p in list(Path(__file__).parent.glob('effective_combo*.py'))+[Path(__file__).with_name('complex_alpha_evaluation.py')]:shutil.copyfile(p,snapshot/p.name)
    np.save(out/'dates.npy',dates); np.save(out/'target.npy',y)
    _write_json(out/'feature_contract.json',{'features':names,'assets':codes,'source':str(source),
        'asset_identity':'known static one-hot control; not market data', 'horizon_hours':horizon,
        'holdout_opened':False,'training_label_clip_bps':s['label_clip_bps']})
    for name in ['old_feature_pool.csv','mixed_feature_pool.csv','feature_pool_freeze.json']:shutil.copyfile(source/name,out/name)
    folds=src_cfg['complex_alpha']['folds']; periods=folds+[[split.validation_start,split.validation_end]]
    predictions={}; metrics=[]; update_records=[]; importances=[]
    for spec in s['models']:
        folder=out/'models'/spec['id']; folder.mkdir(parents=True)
        def save(model,key,record):
            if spec['backend']=='ridge':_write_json(folder/(key+'.json'),model.record())
            else:
                model.booster_.save_model(str(folder/(key+'.txt')))
                if key==split.validation_start:
                    importances.extend({'model':spec['id'],'feature':f,'gain':float(v)} for f,v in zip(names,model.feature_importances_))
        prediction,records=causal_predictions(x,y,dates,periods,spec,horizon,s['seed'],s['threads'],s['label_clip_bps'],save)
        predictions[spec['id']]=prediction; np.save(out/(spec['id']+'_prediction.npy'),prediction.astype(np.float32))
        update_records.extend({'model':spec['id'],**r} for r in records)
        for period,mask in [('discovery_feedback_oof',dates<=split.discovery_end),('historical_validation',dates>=split.validation_start)]:
            metrics.append({'model':spec['id'],'period':period,**prediction_metrics(prediction[mask],y[mask])})
        print('[effective model] '+spec['id']+' complete; updates='+str(len(records)),flush=True)
    predictions['ensemble']=np.mean([predictions[k] for k in s['ensemble']],axis=0)
    np.save(out/'ensemble_prediction.npy',predictions['ensemble'].astype(np.float32))
    for period,mask in [('discovery_feedback_oof',dates<=split.discovery_end),('historical_validation',dates>=split.validation_start)]:
        metrics.append({'model':'ensemble','period':period,**prediction_metrics(predictions['ensemble'][mask],y[mask])})
    _write_csv(out/'prediction_metrics.csv',pd.DataFrame(metrics)); _write_csv(out/'update_records.csv',pd.DataFrame(update_records))
    _write_csv(out/'feature_importance.csv',pd.DataFrame(importances))
    # Every declared model and execution is evaluated; the task is not reduced to zero products.
    evaluations=[]; scales={}; candidates={}
    def evaluate(pred,key,execution,period,start,end,scale,save_arrays=False):
        dm=(dates>=start)&(dates<=end); fm=(fast>=start)&(fast<=end)
        position=positions_from_prediction(pred[dm],scale,execution,s['gross_budget'])
        frame,asset,actual=five_minute_ledger(position,dates[dm],fast[fm],close[fm],funding[fm],s['cost_bps'],split)
        if abs((frame.price_gross+frame.funding-frame.fee-frame.net)).max()>1e-12:raise ValueError('ledger accounting failed')
        if save_arrays:
            frame.to_parquet(out/(key+'_'+period+'_ledger.parquet'),index=False)
            np.savez_compressed(out/(key+'_'+period+'_asset_path.npz'),net=asset,position=actual,price=close[fm],dates=fast[fm])
            trades=completed_trades(fast[fm],actual,asset,codes); _write_csv(out/(key+'_'+period+'_trades.csv'),trades)
        stat={'id':key,'period':period,**ledger_metrics(frame,asset,np.ones(len(frame),bool))}
        return stat
    for name,pred in predictions.items():
        eligible=(dates<=split.discovery_end)&np.isfinite(pred).all(axis=1)
        scale=float(np.sqrt(np.nanmean(pred[eligible]**2))); scales[name]=scale
        for execution in s['executions']:
            key=name+'__'+execution['id']; fold_stats=[]
            for i,(start,end) in enumerate(folds):
                # RMS magnitude is frozen from earlier predictions only for each fold.
                past=(dates<start)&np.isfinite(pred).all(axis=1)
                fold_scale=float(np.sqrt(np.nanmean(pred[past]**2))) if past.any() else float(np.nanstd(y[mature_window(dates,start,horizon,12)]))*.05
                row=evaluate(pred,key,execution,'fold'+str(i),start,end,max(fold_scale,1e-8))
                fold_stats.append(row); evaluations.append(row)
            sharpes=np.array([r['sharpe'] for r in fold_stats]); positive=sum(r['net_return_pct']>0 for r in fold_stats)
            candidates[key]={'id':key,'model':name,'execution':execution,'scale':scale,
                'discovery_score':float(sharpes.mean()-.25*sharpes.std()),'positive_discovery_folds':positive}
            evaluations.append(evaluate(pred,key,execution,'validation',split.validation_start,split.validation_end,scale,True))
    # Freeze research choice using discovery only, before considering validation results.
    ranked=sorted(candidates.values(),key=lambda r:(r['positive_discovery_folds']>=2,r['discovery_score'],r['id']),reverse=True)
    _write_json(out/'selection_freeze.json',{'selected':ranked[0], 'ranked_discovery':ranked,
        'selection_data':'discovery chronological portfolio metrics only','holdout_opened':False,
        'independent_oof':False,'historical_validation_previously_seen':True})
    _write_json(out/'execution_scales.json',scales); _write_csv(out/'portfolio_metrics.csv',pd.DataFrame(evaluations))
    # Separate strong fit diagnostic; same-sample replay is never a causal strategy.
    diagnostic=next(m for m in s['models'] if m['id']==s['diagnostic_model'])
    train=mature_window(dates,split.validation_start,horizon,0)
    model,predict=fit_model(x[train],y[train],diagnostic,s['seed'],s['threads'],s['label_clip_bps'])
    fitted=predict(x); np.save(out/'overfit_diagnostic_prediction.npy',fitted.astype(np.float32))
    _write_json(out/'overfit_diagnostic_metrics.json',{'role':'same-sample expressivity and strategy replay; not chronological',
        **prediction_metrics(fitted[train],y[train])})
    if diagnostic['backend']=='lightgbm':model.booster_.save_model(str(out/'overfit_diagnostic_model.txt'))
    else:_write_json(out/'overfit_diagnostic_model.json',model.record())
    execution=next(e for e in s['executions'] if e['name']=='net_edge')
    scale=float(np.sqrt(np.nanmean(fitted[train]**2)))
    diagstat=evaluate(fitted,'overfit_diagnostic',execution,'in_sample',str(dates[train][0]),split.discovery_end,scale,True)
    _write_json(out/'overfit_diagnostic_strategy.json',diagstat)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('source evidence changed')
    _write_json(out/'manifest.json',{'status':'COMPLETED','stage':'effective_combo','version':cfg['version'],
        'source':str(source),'models':len(predictions),'selected':ranked[0]['id'],'holdout_opened':False,
        'feature_columns':x.shape[-1],'diagnostic_product':'overfit_diagnostic','config':str(config_path),
        'results':{'models':len(predictions),'portfolio_variants':len(candidates),'selected':ranked[0]['id'],
                   'holdout_opened':False,'diagnostic_product':'overfit_diagnostic'}})
    from .effective_combo_reporting import render_effective_report
    render_effective_report(out,viz)
    return load(out/'manifest.json')
