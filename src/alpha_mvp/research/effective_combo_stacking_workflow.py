"""A bounded prediction/position stacking experiment in the production CLI."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .effective_combo_stacking import causal_stacking
from .effective_combo import positions_from_prediction,completed_trades
from .effective_combo_extension import source_metadata
from .complex_alpha_evaluation import five_minute_ledger,ledger_metrics
from .complex_alpha_prediction import prediction_metrics
from .manual_alpha_workflow import _write_json,_write_csv


def validate_stacking_config(cfg):
    if set(cfg)!={'mode','version','complex_alpha','output'} or cfg['mode']!='aligned_crypto':raise ValueError('explicit stacking config required')
    s=cfg['complex_alpha'];keys={'stage','phase','source_history','source_oos','frozen_run','stacking','cost_bps','gross_budget','executions'}
    if set(s) not in (keys,keys|{'state_calibration'}) or s['stage']!='effective_combo_stacking' or s['phase'] not in {'historical','oos'} or (s['phase']=='oos')!=bool(s['frozen_run']):raise ValueError('invalid stacking phase/freeze')
    p=s['stacking']
    if set(p)!={'members','prior','diagonal_shrink','prior_shrink','minimum_labels','asset_share'} or p['members']!=['base15','base_ensemble','wide15_static','asset_shrunk']:raise ValueError('explicit four frozen members required')
    if len(p['prior'])!=4 or min(p['prior'])<0 or not np.isclose(sum(p['prior']),1) or not 0<=p['diagonal_shrink']<=1 or not .5<=p['prior_shrink']<=1 or not 0<=p['asset_share']<=.5 or p['minimum_labels']<100:raise ValueError('invalid stacking shrinkage')
    if s['cost_bps']<0 or not 0<s['gross_budget']<=1 or set(cfg['output'])!={'out_dir','visualization_dir'}:raise ValueError('invalid stacking financial outputs')
    if [e['id'] for e in s['executions']]!=['edge8','edge16'] or any(set(e)!={'id','name','floor_bps','amplitude_bps'} or e['name']!='net_edge' or e['floor_bps']<0 or e['amplitude_bps']<=0 for e in s['executions']):raise ValueError('explicit shared executions required')
    if 'state_calibration' in s:
        p=s['state_calibration']
        if set(p)!={'source_model','windows_months','training_floor_bps','minimum_labels','skill_deadzone','smooth_span','deadband'} or p['source_model']!='pair_prediction' or p['windows_months']!=[1,3,6] or p['minimum_labels']<100 or p['training_floor_bps']<0 or not 0<=p['skill_deadzone']<=.2 or not 1<=p['smooth_span']<=24 or not 0<=p['deadband']<=.1:raise ValueError('invalid bounded skill-state protocol')


def run_stacking(cfg,config_path):
    validate_stacking_config(cfg);s=cfg['complex_alpha'];out=Path(cfg['output']['out_dir']);viz=Path(cfg['output']['visualization_dir'])
    if out.exists() or viz.exists():raise ValueError('stacking outputs must be new')
    load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'));history=Path(s['source_history'])
    histcfg=yaml.safe_load((history/'config.yaml').read_text())['complex_alpha']
    if any(s[k]!=histcfg[k] for k in ['cost_bps','gross_budget','executions']):raise ValueError('stacking changed shared execution contract')
    paths=[history/n for n in ['manifest.json','protocol_freeze.json','selection_freeze.json','dates.npy','target.npy']+[k+'_prediction.npy' for k in ['base15','base_ensemble','wide15','wide15_static','asset_shrunk']]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path(__file__).parent.glob('effective_combo_stacking*.py'))+[Path(__file__).with_name('effective_combo_maturity_calibration.py')]}
    protocol={k:v for k,v in s.items() if k not in {'phase','frozen_run'}}
    if s['phase']=='oos':
        frozen=Path(s['frozen_run']);freeze=load(frozen/'stacking_freeze.json')
        if freeze['protocol']!=protocol or freeze['source_hashes']!=hashes or freeze['code']!=code or load(frozen/'manifest.json')['status']!='COMPLETED':raise ValueError('stacking OOS freeze mismatch')
    source=history if s['phase']=='historical' else Path(s['source_oos'])
    if load(source/'manifest.json')['status']!='COMPLETED':raise ValueError('completed stacking source required')
    if (history/'protocol_freeze.json').read_bytes()!=(source/'protocol_freeze.json').read_bytes():raise ValueError('source extension freeze differs')
    dates=np.load(source/'dates.npy');y=np.load(source/'target.npy');contract=load(source/'feature_contract.json');codes=contract['assets']
    _,_,prior,_,scfg=source_metadata(histcfg['source_pool_run'])
    from .aligned_contract import Split
    split=Split(**prior['split']);folds=scfg['complex_alpha']['folds'];horizon=contract['horizon_hours']
    periods=folds+[[split.validation_start,split.validation_end]] if s['phase']=='historical' else [[split.holdout_start,str(dates[-1])]]
    predictions={k:np.load(source/(k+'_prediction.npy')).astype(float) for k in pd.read_csv(source/'prediction_metrics.csv').model.unique()}
    global_pred,asset_pred,records=causal_stacking([predictions[k] for k in s['stacking']['members']],y,dates,periods,horizon,s['stacking'])
    if s['phase']=='oos':
        old=np.load(frozen/'stack_global_prediction.npy');global_pred[:len(old)]=old;asset_pred[:len(old)]=np.load(frozen/'stack_asset_prediction.npy')
    predictions.update(pair_prediction=(predictions['base15']+predictions['wide15_static'])/2,
        pair_position=(predictions['base15']+predictions['wide15_static'])/2,
        anchor_wide=.75*predictions['base15']+.25*predictions['wide15'],stack_global=global_pred,stack_asset=asset_pred)
    states=[]
    if 'state_calibration' in s:
        from .effective_combo_maturity_calibration import causal_skill_states
        cal,score,states=causal_skill_states(predictions['pair_prediction'],y,dates,periods,horizon,s['state_calibration'])
        for months in [1,3,6]:
            for prefix,values in [('state_cal',cal[months]),('state_score',score[months])]:
                name=prefix+str(months)
                if s['phase']=='oos':values[:len(old)]=np.load(frozen/(name+'_prediction.npy'))
                predictions[name]=values
        predictions['state_score3_smooth']=predictions['state_score3'].copy()
    out.mkdir(parents=True);(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    _write_json(out/'manifest.json',{'status':'RUNNING','phase':s['phase']})
    _write_json(out/'stacking_freeze.json',{'protocol':protocol,'source_hashes':hashes,'code':code,'holdout_opened':False,'showcase':('state_score3_smooth__edge16' if states else 'pair_position__edge16')})
    if states:_write_csv(out/'state_coefficients.csv',pd.DataFrame(states))
    _write_json(out/'source_provenance.json',hashes);shutil.copyfile(history/'protocol_freeze.json',out/'protocol_freeze.json')
    if s['phase']=='oos':shutil.copyfile(frozen/'selection_freeze.json',out/'selection_freeze.json')
    for name in ['expansion_archive.csv','feature_contract.json','individual_features.csv','feature_importance.csv']:shutil.copyfile(source/name,out/name)
    np.save(out/'dates.npy',dates);np.save(out/'target.npy',y)
    for k,p in predictions.items():np.save(out/(k+'_prediction.npy'),p.astype(np.float32))
    for row in records:
        if row['asset']!='GLOBAL':row['asset']=codes[row['asset']]
    _write_csv(out/'ensemble_weights.csv',pd.DataFrame(records));_write_csv(out/'update_records.csv',pd.DataFrame(records).assign(model='stacking'))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in list(Path(__file__).parent.glob('effective_combo_stacking*.py'))+[Path(__file__).with_name('effective_combo_maturity_calibration.py')]:shutil.copyfile(p,snapshot/p.name)
    bank=Path(scfg['complex_alpha']['bank_manifest']).parent;full=np.load(bank/'dates.npy');full=full[:np.searchsorted(full,dates[-1],side='right')]
    fast=np.load(bank/'fast_dates.npy');end=split.validation_end if s['phase']=='historical' else load(source/'manifest.json')['results']['actual_end'];fast=fast[:np.searchsorted(fast,end,side='right')]
    close=np.load(bank/'fast_close.npy',mmap_mode='r')[:len(fast)].copy()
    from .complex_alpha_workflow import _funding
    funding,files,_=_funding(prior,full,fast,close)
    if files!=load(Path(scfg['complex_alpha']['source_run'])/'input_signature.json')['derivative_files']:raise ValueError('stack funding changed')
    evaluate=[('fold'+str(i),a,b) for i,(a,b) in enumerate(folds)]+[('validation',split.validation_start,split.validation_end)] if s['phase']=='historical' else [('oos',histcfg['oos_start'],str(fast[-1]))]
    portfolio=[];metrics=[];breadth=[];costs=[];exposures=[]
    for key,pred in predictions.items():
        for period,start,end in evaluate:
            dm=(dates>=start)&(dates<=end);fm=(fast>=start)&(fast<=end);d=dates[dm];ff=fast[fm];c=close[fm];fund=funding[fm]
            metrics.append({'model':key,'period':period,**prediction_metrics(pred[dm],y[dm])})
            if period in {'validation','oos'}:
                for j,asset in enumerate(codes):breadth.append({'model':key,'axis':'asset','segment':asset,**prediction_metrics(pred[dm,j],y[dm,j])})
                months=pd.to_datetime(d,format='%Y%m%d%H%M').to_period('M')
                for month in months.unique():breadth.append({'model':key,'axis':'month','segment':str(month),**prediction_metrics(pred[dm][months==month],y[dm][months==month])})
            for execution in s['executions']:
                identifier=key+'__'+execution['id'];q=positions_from_prediction(pred[dm],1.,execution,s['gross_budget'])
                if key=='pair_position':q=(positions_from_prediction(predictions['base15'][dm],1.,execution,s['gross_budget'])+positions_from_prediction(predictions['wide15_static'][dm],1.,execution,s['gross_budget']))/2
                if key=='state_score3_smooth':
                    from .effective_combo_maturity_calibration import smooth_targets
                    q=smooth_targets(q,s['state_calibration']['smooth_span'],s['state_calibration']['deadband'])
                frame,asset,actual=five_minute_ledger(q,d,ff,c,fund,s['cost_bps'],split)
                portfolio.append({'id':identifier,'period':period,**ledger_metrics(frame,asset,np.ones(len(frame),bool))})
                if period in {'validation','oos'}:
                    frame.to_parquet(out/(identifier+'_'+period+'_ledger.parquet'),index=False)
                    np.savez_compressed(out/(identifier+'_'+period+'_asset_path.npz'),net=asset,position=actual,price=c,dates=ff)
                    _write_csv(out/(identifier+'_'+period+'_trades.csv'),completed_trades(ff,actual,asset,codes))
                    for cost in [0.,4.,8.]:
                        f,a,_=five_minute_ledger(q,d,ff,c,fund,cost,split);costs.append({'id':identifier,'period':period,'cost_bps':cost,**ledger_metrics(f,a,np.ones(len(f),bool))})
                    forward=np.zeros_like(c);forward[:-1]=c[1:]/c[:-1]-1;common=actual.mean(axis=1)*forward.mean(axis=1);capital=np.r_[1,np.cumprod(1+frame.net.to_numpy())[:-1]]
                    exposures.append({'id':identifier,'period':period,'common_price_contribution_pct':float(np.sum(capital*common)*100),
                        'relative_price_contribution_pct':float(np.sum(capital*(frame.price_gross-common))*100),'funding_contribution_pct':float(np.sum(capital*frame.funding)*100),'fee_contribution_pct':float(np.sum(capital*frame.fee)*100)})
        print('[stacking ledger] '+key+' complete',flush=True)
    for name,rows in [('portfolio_metrics',portfolio),('prediction_metrics',metrics),('prediction_breadth',breadth),('cost_sensitivity',costs),('exposure_decomposition',exposures)]:_write_csv(out/(name+'.csv'),pd.DataFrame(rows))
    if s['phase']=='historical':
        table=pd.DataFrame(portfolio);ranks=[]
        for key,g in table.loc[table.period.str.startswith('fold')].groupby('id'):ranks.append({'id':key,'positive_discovery_folds':int(g.net_return_pct.gt(0).sum()),'discovery_score':float(g.sharpe.mean()-.25*g.sharpe.std(ddof=0))})
        ranks.sort(key=lambda r:(r['positive_discovery_folds']>=2,r['discovery_score'],r['id']),reverse=True)
        _write_json(out/'selection_freeze.json',{'selected':ranks[0],'ranked_discovery':ranks,'holdout_opened':False,'selection_data':'source discovery only'})
    # R3 carries all R2 products for complete fair comparisons; source metrics must replay.
    original=pd.read_csv(source/'portfolio_metrics.csv');current=pd.DataFrame(portfolio);merged=current.merge(original,on=['id','period'],suffixes=('','_source'))
    err=max(float(abs(merged[k]-merged[k+'_source']).max()) for k in ['net_return_pct','sharpe','max_drawdown_pct'])
    if err>1e-5:raise ValueError('source strategy replay failed')
    _write_json(out/'baseline_replay_check.json',{'status':'PASS','rows':len(merged),'max_metric_error':err})
    result={'status':'RUNNING','stage':s['stage'],'phase':s['phase'],'results':{'models':len(predictions),'portfolio_variants':len(predictions)*2,
        'wide_columns':contract['wide_columns'],'extra_formulas':len(load(history/'protocol_freeze.json')['extra_ids']),'actual_end':str(fast[-1]),'holdout_opened':s['phase']=='oos'},'config':str(config_path)}
    _write_json(out/'manifest.json',result)
    from .effective_combo_stacking_reporting import render_stacking_report
    render_stacking_report(out,viz)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('stacking source changed')
    result['status']='COMPLETED';_write_json(out/'manifest.json',result);return result
