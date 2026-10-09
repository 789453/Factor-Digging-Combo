"""Bounded R2 extension in the existing production research entry."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml
from .effective_combo_extension import (source_metadata,reconstruct,expand_pool,causal_weighted_average,asset_shrunk_predictions)
from .effective_combo import asset_conditional_design,causal_predictions,positions_from_prediction,completed_trades
from .complex_alpha_prediction import prediction_metrics
from .complex_alpha_evaluation import five_minute_ledger,ledger_metrics
from .manual_alpha_workflow import _write_json,_write_csv


def validate_extension_config(cfg):
    if set(cfg)!={'mode','version','complex_alpha','output'} or cfg['mode']!='aligned_crypto':raise ValueError('explicit extension config required')
    s=cfg['complex_alpha']
    keys={'stage','phase','source_pool_run','baseline_run','frozen_run','pool','models','adaptation','weighting','executions',
        'cost_bps','gross_budget','seed','threads','label_clip_bps','oos_start','oos_end'}
    if set(s)!=keys or s['stage']!='effective_combo_extension' or s['phase'] not in {'historical','oos'}:raise ValueError('invalid extension stage/phase')
    if (s['phase']=='oos')!=bool(s['frozen_run']):raise ValueError('OOS requires frozen historical run')
    if set(s['pool'])!={'shortlist_count','additional_per_stream','correlation_limit'}:raise ValueError('invalid pool keys')
    p=s['pool']
    if not 96<=p['shortlist_count']<=256 or p['additional_per_stream'] not in {8,16,32,48,64} or not .8<=p['correlation_limit']<1:raise ValueError('bounded pool required')
    if not 1<=s['threads']<=12 or not 0<s['gross_budget']<=1 or s['cost_bps']<0 or not 100<=s['label_clip_bps']<=5000:raise ValueError('invalid resource/financial contract')
    from .effective_combo_workflow import validate_effective_config
    models=[{k:v for k,v in m.items() if k!='pool'} for m in s['models']]
    shell=dict(mode='aligned_crypto',version=cfg['version'],complex_alpha=dict(stage='effective_combo',source_pool_run=s['source_pool_run'],
        models=models,ensemble=[m['id'] for m in models[:3]],executions=s['executions'],cost_bps=s['cost_bps'],gross_budget=s['gross_budget'],
        seed=s['seed'],threads=s['threads'],label_clip_bps=s['label_clip_bps'],diagnostic_model=models[0]['id']),output=cfg['output'])
    validate_effective_config(shell)
    if any(set(m)!=set(models[0])|{'pool'} or m['pool'] not in {'base','wide'} for m in s['models']):raise ValueError('explicit model pool required')
    if set(s['weighting'])!={'members','diagonal_shrink','equal_shrink','minimum_labels'} or len(s['weighting']['members'])!=3:raise ValueError('explicit three-member weighting required')
    if not set(s['weighting']['members']).issubset({m['id'] for m in models}) or not 0<=s['weighting']['diagonal_shrink']<=1 or not 0<=s['weighting']['equal_shrink']<=1 or s['weighting']['minimum_labels']<100:raise ValueError('invalid weighting')
    a=s['adaptation']
    if set(a)!={'shared_model','asset_share','model'} or a['shared_model'] not in {m['id'] for m in models} or not 0<=a['asset_share']<=.5:raise ValueError('invalid asset adaptation')
    if a['model']['backend']!='lightgbm' or a['model']['window_months']!=12 or a['model']['update_months']!=1:raise ValueError('asset adaptation requires monthly trees')
    shell['complex_alpha']['models']=models+[a['model']];validate_effective_config(shell)
    if len(s['models'])!=7 or sum(m['pool']=='base' for m in s['models'])!=3:raise ValueError('declared seven fair baselines required')
    if s['oos_start']!='202603010000' or s['oos_end']!='202609302355':raise ValueError('explicit March-September OOS required')


def protocol(s):return {k:v for k,v in s.items() if k not in {'phase','frozen_run'}}
def hash_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_signature():
    root=Path(__file__).parent
    files=set(root.glob('effective_combo_extension*.py'))|{root/k for k in ['effective_combo.py','complex_alpha_prediction.py',
        'complex_alpha_search.py','complex_alpha_evaluation.py','complex_alpha_workflow.py','manual_alpha_derivatives.py','aligned_contract.py']}
    return {p.name:hash_file(p) for p in sorted(files)}


def run_extension(cfg,config_path):
    validate_extension_config(cfg);s=cfg['complex_alpha'];load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    out=Path(cfg['output']['out_dir']);viz=Path(cfg['output']['visualization_dir'])
    if out.exists() or viz.exists():raise ValueError('extension outputs must be new')
    source,mining,prior,contract,source_cfg=source_metadata(s['source_pool_run']);baseline=Path(s['baseline_run'])
    if load(baseline/'manifest.json')['status']!='COMPLETED':raise ValueError('completed baseline required')
    bcfg=yaml.safe_load((baseline/'config.yaml').read_text(encoding='utf-8'))['complex_alpha']
    names={'base_ridge':'ridge_asset_online','base15':'tree15_online','base31':'tree31_online'}
    for m in s['models']:
        if m['pool']!='base':continue
        oldspec=next(v for v in bcfg['models'] if v['id']==names[m['id']])
        keys=['backend','penalty','window_months','update_months']+(['trees','leaves','min_leaf','learning_rate'] if m['backend']=='lightgbm' else [])
        if any(m[k]!=oldspec[k] for k in keys):raise ValueError('base model protocol differs from frozen baseline')
    if any(s[k]!=bcfg[k] for k in ['source_pool_run','seed','label_clip_bps','cost_bps','gross_budget']):raise ValueError('baseline financial/training contract changed')
    out.mkdir(parents=True);_write_json(out/'manifest.json',{'status':'RUNNING','stage':s['stage'],'phase':s['phase']})
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    source_paths=[source/n for n in ['config.yaml','numeric_pool_contract.json','feature_pool_freeze.json','manifest.json']]
    source_paths += [mining/n for n in ['coarse_results.csv','candidate_registry.json','initial_coordinate_scaler.json','input_signature.json']]
    source_paths += [baseline/n for n in ['config.yaml','manifest.json','portfolio_metrics.csv']+[k+'_prediction.npy' for k in names.values()]]
    source_hashes={str(p):hash_file(p) for p in source_paths}
    signature=code_signature();snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('effective_combo*.py'):shutil.copyfile(p,snapshot/p.name)
    _write_json(out/'source_provenance.json',source_hashes)
    if s['phase']=='historical':
        data=reconstruct(source,prior['split']['validation_end'],shortlist_count=s['pool']['shortlist_count'])
        extra,archive,corr=expand_pool(data['formulas'],data['formula_ids'],data['rows'],data['dates']<=data['split'].discovery_end,
            s['pool']['additional_per_stream'],s['pool']['correlation_limit'],data['original_count'])
        _write_csv(out/'expansion_archive.csv',archive);np.save(out/'formula_discovery_correlation.npy',corr)
        freeze={'protocol':protocol(s),'extra_ids':extra,'source_hashes':source_hashes,'code':signature,
            'selection_data':'source discovery allowlist and numerical discovery correlation only',
            'predeclared_anchors':['base_ensemble__edge16','base15__edge16'],'holdout_opened':False,
            'historical_validation_previously_seen':True,'independent_oof':False}
        _write_json(out/'protocol_freeze.json',freeze)
        periods=data['folds']+[[data['split'].validation_start,data['split'].validation_end]]
    else:
        frozen=Path(s['frozen_run']);freeze=load(frozen/'protocol_freeze.json')
        if load(frozen/'manifest.json')['status']!='COMPLETED' or freeze['protocol']!=protocol(s) or freeze['source_hashes']!=source_hashes or freeze['code']!=signature:
            raise ValueError('OOS freeze/config/source/code mismatch')
        # Freeze validated BEFORE any rows beyond the historical validation period.
        shutil.copyfile(frozen/'protocol_freeze.json',out/'protocol_freeze.json')
        shutil.copyfile(frozen/'selection_freeze.json',out/'selection_freeze.json')
        shutil.copyfile(frozen/'expansion_archive.csv',out/'expansion_archive.csv')
        extra=freeze['extra_ids'];data=reconstruct(source,s['oos_end'],extra_ids=extra)
        periods=[[data['split'].holdout_start,str(data['dates'][-1])]]
    idx={k:i for i,k in enumerate(data['formula_ids'])}
    wide=np.concatenate([data['base'],data['formulas'][...,[idx[k] for k in extra]]],axis=-1)
    designs={'base':asset_conditional_design(data['base']),'wide':asset_conditional_design(wide)}
    dates=data['dates'];y=data['y'];codes=contract['assets'];horizon=data['horizon'];split=data['split']
    base_names=sum([contract['feature_order'][k] for k in ['old_coordinates','old_mined','mixed_mined']],[])
    _write_json(out/'feature_contract.json',{'assets':codes,'horizon_hours':horizon,'features':base_names+extra+['asset_'+k for k in codes],
        'base_columns':designs['base'].shape[-1],'wide_columns':designs['wide'].shape[-1],'holdout_opened':s['phase']=='oos',
        'original_pool_replay':'exact','formula_order':contract['feature_order']['old_mined']+contract['feature_order']['mixed_mined']+extra})
    np.save(out/'dates.npy',dates);np.save(out/'target.npy',y)
    predictions={};updates=[];gains=[]
    mapping={'base_ridge':'ridge_asset_online','base15':'tree15_online','base31':'tree31_online'}
    for spec in s['models']:
        key=spec['id'];folder=out/'models'/key;folder.mkdir(parents=True)
        if s['phase']=='historical' and key in mapping:
            pred=np.load(baseline/(mapping[key]+'_prediction.npy')).astype(float)
            r=pd.read_csv(baseline/'update_records.csv');r=r.loc[r.model==mapping[key]].drop(columns='model').to_dict('records')
            updates.extend({'model':key,'reuse':True,**v} for v in r)
        else:
            def save(model,stamp,record):
                if spec['backend']=='ridge':_write_json(folder/(stamp+'.json'),model.record())
                else:
                    model.booster_.save_model(str(folder/(stamp+'.txt')))
                    gains.extend({'model':key,'update':stamp,'feature':name,'gain':float(value)} for name,value in zip(
                        (base_names if spec['pool']=='base' else base_names+extra)+['asset_'+k for k in codes],model.feature_importances_))
            if s['phase']=='oos' and spec['update_months']==0:
                import lightgbm as lgb
                model=lgb.Booster(model_file=str(frozen/'models'/key/(split.validation_start+'.txt')))
                pred=np.full_like(y,np.nan);mask=dates>=split.holdout_start
                z=designs[spec['pool']][mask];pred[mask]=model.predict(z.reshape(-1,z.shape[-1]),num_threads=s['threads']).reshape(z.shape[:-1])/1e4
                updates.append({'model':key,'update':split.holdout_start,'reuse_static_origin':split.validation_start,'prediction_rows':int(mask.sum())})
                shutil.copyfile(frozen/'models'/key/(split.validation_start+'.txt'),folder/(split.validation_start+'.txt'))
            else:
                pred,r=causal_predictions(designs[spec['pool']],y,dates,periods,spec,horizon,s['seed'],s['threads'],s['label_clip_bps'],save)
                updates.extend({'model':key,**v} for v in r)
            if s['phase']=='oos':
                old=np.load(frozen/(key+'_prediction.npy'));pred[:len(old)]=old
        predictions[key]=pred
        print('[extension model] '+key+' complete',flush=True)
    base_members=['base_ridge','base15','base31'];wide_members=s['weighting']['members']
    predictions['base_ensemble']=np.mean([predictions[k] for k in base_members],axis=0)
    predictions['wide_equal']=np.mean([predictions[k] for k in wide_members],axis=0)
    weighted,weights=causal_weighted_average([predictions[k] for k in wide_members],y,dates,periods,horizon,s['weighting'])
    if s['phase']=='oos':weighted[:len(old)]=np.load(frozen/'wide_cov_prediction.npy')
    predictions['wide_cov']=weighted;_write_csv(out/'ensemble_weights.csv',pd.DataFrame(weights))
    a=s['adaptation'];folder=out/'models'/'asset_shrunk';folder.mkdir(parents=True)
    def save_asset(model,stamp,j,record):model.booster_.save_model(str(folder/(codes[j]+'_'+stamp+'.txt')))
    adapted,r=asset_shrunk_predictions(wide,y,dates,periods,predictions[a['shared_model']],a['model'],horizon,s['seed'],s['threads'],s['label_clip_bps'],a['asset_share'],save_asset)
    if s['phase']=='oos':adapted[:len(old)]=np.load(frozen/'asset_shrunk_prediction.npy')
    predictions['asset_shrunk']=adapted;updates.extend({'model':'asset_shrunk','asset':codes[v['asset_index']],**v} for v in r)
    # A fixed conservative hybrid tests complementary weighting without tuning.
    predictions['adaptive_equal']=(predictions['wide_cov']+predictions['asset_shrunk']+predictions['base_ensemble'])/3
    _write_csv(out/'update_records.csv',pd.DataFrame(updates));_write_csv(out/'feature_importance.csv',pd.DataFrame(gains))
    for k,p in predictions.items():np.save(out/(k+'_prediction.npy'),p.astype(np.float32))
    metrics=[];portfolio=[];ranks=[];scales={};costs=[];exposures=[];breadth=[];feature_evidence=[]
    eval_periods=[('fold'+str(i),a,b) for i,(a,b) in enumerate(data['folds'])]+[('validation',split.validation_start,split.validation_end)]
    if s['phase']=='oos':eval_periods=[('oos',s['oos_start'],str(data['fast'][-1]))]
    period,start,end=eval_periods[-1];mask=(dates>=start)&(dates<=end);discovery=dates<=split.discovery_end
    from .complex_alpha_search import correlation
    for i,name in enumerate(base_names+extra):
        d=correlation(wide[discovery,:,i],y[discovery]);v=correlation(wide[mask,:,i],y[mask]);sign=1 if d>=0 else -1
        feature_evidence.append({'feature':name,'discovery_raw_ic':d,'period':period,'evaluation_raw_ic':v,
            'discovery_direction':sign,'evaluation_discovery_signed_ic':sign*v,'direction_source':'same-source discovery only; feedback evidence'})
    _write_csv(out/'individual_features.csv',pd.DataFrame(feature_evidence))
    for key,pred in predictions.items():
        train=(dates<=split.discovery_end)&np.isfinite(pred).all(axis=1);scale=float(np.sqrt(np.nanmean(pred[train]**2)));scales[key]=scale
        for period,start,end in eval_periods:
            dm=(dates>=start)&(dates<=end);fm=(data['fast']>=start)&(data['fast']<=end)
            metrics.append({'model':key,'period':period,**prediction_metrics(pred[dm],y[dm])})
            if period in {'validation','oos'}:
                for j,code in enumerate(codes):breadth.append({'model':key,'axis':'asset','segment':code,**prediction_metrics(pred[dm,j],y[dm,j])})
                months=pd.to_datetime(dates[dm],format='%Y%m%d%H%M').to_period('M')
                for month in months.unique():
                    mm=months==month
                    breadth.append({'model':key,'axis':'month','segment':str(month),**prediction_metrics(pred[dm][mm],y[dm][mm])})
            for execution in s['executions']:
                identifier=key+'__'+execution['id'];position=positions_from_prediction(pred[dm],max(scale,1e-8),execution,s['gross_budget'])
                frame,asset,actual=five_minute_ledger(position,dates[dm],data['fast'][fm],data['close'][fm],data['funding'][fm],s['cost_bps'],split)
                stat={'id':identifier,'period':period,**ledger_metrics(frame,asset,np.ones(len(frame),bool))};portfolio.append(stat)
                if period in {'validation','oos'}:
                    frame.to_parquet(out/(identifier+'_'+period+'_ledger.parquet'),index=False)
                    np.savez_compressed(out/(identifier+'_'+period+'_asset_path.npz'),net=asset,position=actual,price=data['close'][fm],dates=data['fast'][fm])
                    _write_csv(out/(identifier+'_'+period+'_trades.csv'),completed_trades(data['fast'][fm],actual,asset,codes))
                    for cost in [0.,4.,8.]:
                        f,a,_=five_minute_ledger(position,dates[dm],data['fast'][fm],data['close'][fm],data['funding'][fm],cost,split)
                        costs.append({'id':identifier,'period':period,'cost_bps':cost,**ledger_metrics(f,a,np.ones(len(f),bool))})
                    forward=np.zeros_like(data['close'][fm]);forward[:-1]=data['close'][fm][1:]/data['close'][fm][:-1]-1
                    common=actual.mean(axis=1)*forward.mean(axis=1);relative=frame.price_gross.to_numpy()-common
                    capital=np.r_[1,np.cumprod(1+frame.net.to_numpy())[:-1]]
                    check=float(np.max(abs(np.mean(actual*forward,axis=1)-frame.price_gross)))
                    if check>1e-12:raise ValueError('exposure price accounting failed')
                    exposures.append({'id':identifier,'period':period,'common_price_contribution_pct':float(np.sum(capital*common)*100),
                        'relative_price_contribution_pct':float(np.sum(capital*relative)*100),
                        'long_price_contribution_pct':float(np.sum(capital*np.mean(np.maximum(actual,0)*forward,axis=1))*100),
                        'short_price_contribution_pct':float(np.sum(capital*np.mean(np.minimum(actual,0)*forward,axis=1))*100),
                        'funding_contribution_pct':float(np.sum(capital*frame.funding)*100),
                        'fee_contribution_pct':float(np.sum(capital*frame.fee)*100),'price_check':check})
        print('[extension ledger] '+key+' complete',flush=True)
    _write_csv(out/'prediction_metrics.csv',pd.DataFrame(metrics));table=pd.DataFrame(portfolio);_write_csv(out/'portfolio_metrics.csv',table)
    _write_csv(out/'prediction_breadth.csv',pd.DataFrame(breadth));_write_csv(out/'cost_sensitivity.csv',pd.DataFrame(costs))
    _write_csv(out/'exposure_decomposition.csv',pd.DataFrame(exposures))
    _write_json(out/'execution_scales.json',scales)
    if s['phase']=='historical':
        for key,g in table.loc[table.period.str.startswith('fold')].groupby('id'):
            ranks.append({'id':key,'positive_discovery_folds':int(g.net_return_pct.gt(0).sum()),'discovery_score':float(g.sharpe.mean()-.25*g.sharpe.std(ddof=0))})
        ranks.sort(key=lambda r:(r['positive_discovery_folds']>=2,r['discovery_score'],r['id']),reverse=True)
        _write_json(out/'selection_freeze.json',{'selected':ranks[0],'ranked_discovery':ranks,'selection_data':'discovery only',
            'anchors':freeze['predeclared_anchors'],'holdout_opened':False,'independent_oof':False})
        # Anchor reproduction checks include saved prediction float32 and full ledger.
        original=pd.read_csv(baseline/'portfolio_metrics.csv');checks=[]
        for k,oldkey in [('base15__edge16','tree15_online__edge16'),('base_ensemble__edge16','ensemble__edge16')]:
            current=table.loc[(table.id==k)&table.period.eq('validation')].iloc[0]
            reference=original.loc[(original.id==oldkey)&original.period.eq('validation')].iloc[0]
            error=max(abs(current[c]-reference[c]) for c in ['net_return_pct','sharpe','max_drawdown_pct'])
            checks.append({'id':k,'max_metric_error':float(error)})
            if error>1e-5:raise ValueError('baseline ledger replay failed')
        _write_json(out/'baseline_replay_check.json',{'status':'PASS','checks':checks})
    if any(hash_file(p)!=h for p,h in source_hashes.items()):raise ValueError('source evidence changed')
    summary={'models':len(predictions),'portfolio_variants':len(predictions)*len(s['executions']),
        'wide_columns':designs['wide'].shape[-1],'extra_formulas':len(extra),'actual_end':str(data['fast'][-1]),
        'requested_oos_end':s['oos_end'],'holdout_opened':s['phase']=='oos','frozen_before_oos':True}
    result={'status':'RUNNING','stage':s['stage'],'phase':s['phase'],'results':summary,
        'historical_oos_reaudit':True,'independent_oof':False,'source':str(source),'baseline':str(baseline),'config':str(config_path)}
    _write_json(out/'manifest.json',result)
    from .effective_combo_extension_reporting import render_extension_report
    render_extension_report(out,viz)
    result['status']='COMPLETED';_write_json(out/'manifest.json',result)
    return load(out/'manifest.json')
