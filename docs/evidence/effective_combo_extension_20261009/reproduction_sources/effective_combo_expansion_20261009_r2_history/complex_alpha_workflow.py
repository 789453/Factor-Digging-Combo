"""Complex representations within the existing aligned research production mode."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import optuna
import yaml
import sys

from .aligned_contract import Split
from .complex_alpha_time import TIME_VERSION,rotating_screening_masks,nonoverlap_phase
from .complex_alpha_data import (prepare_minute_bank,representation_bank,minute_observations,
                                DATA_VERSION,CHANNELS)
from .complex_alpha_search import (generate_candidates,screen_candidates,diversity_survivors,
                                   evaluate_candidate,correlation,numeric_archive)
from .complex_alpha_evaluation import (past_market_betas,targets_from_prices,crossfit_readout,
    parameter_design,fit_ridge,neutral_positions,five_minute_ledger,ledger_metrics,fit_increment_projector,causal_risk_scales,common_market_loadings,project_basket_returns)
from .manual_alpha_workflow import _write_json,_write_csv
from .manual_alpha_derivatives import load_derivative_panels
from .complex_alpha_diagnostics import block_gain_interval,limited_shift_null,matched_structure_windows

WORKFLOW_VERSION='complex-alpha-research-20261008-v2'


def validate_complex_config(cfg):
    if set(cfg)!={'mode','version','data','split','complex_alpha','output'} or cfg['mode']!='aligned_crypto':
        raise ValueError('complex alpha requires existing aligned_crypto mode and exact declaration')
    data=cfg['data']; spec=cfg['complex_alpha']
    if set(data)!={'source','root','derivative_root','assets','start','end','feature_cache'}:
        raise ValueError('invalid complex alpha data declaration')
    if data['source'] not in {'crypto_parquet','simulated'} or len(data['assets'])<4 or len(set(data['assets']))!=len(data['assets']):
        raise ValueError('explicit source and >=4 unique assets required')
    keys={'seed','device','representation','budgets','old_control_budget','modes','coarse_batch',
          'medium_count','fine_count','model_trials','model_kinds','folds','feature_calibration_end',
          'cost_bps','gross_budget','execution','max_selected','bootstrap_trials','audit_holdout'}
    if not keys.issubset(spec) or set(spec)-keys-{'extension','readout_policy','execution_by_horizon','oof_calibration','edge_floor_bps','target_policy','initial_penalty','initial_penalties','structure_probe_examples','return_contract','risk_mapping','label_projection','exit_on_inactive','kernel_reference'}: raise ValueError(f'invalid complex research keys {set(spec)^keys}')
    if spec.get('kernel_reference','none') not in {'none','gaussian_moments'}:raise ValueError('invalid kernel reference')
    if not isinstance(spec.get('exit_on_inactive',False),bool):raise ValueError('explicit inactive-signal exit flag must be boolean')
    if spec.get('return_contract','log') not in {'log','native_simple'}:raise ValueError('invalid label return contract')
    if spec.get('risk_mapping','legacy_loo') not in {'legacy_loo','exact_loo_to_common','common_label'}:raise ValueError('invalid basket risk mapping')
    if spec.get('label_projection','loo') not in {'loo','cash_common_beta'}:raise ValueError('invalid tradable target subspace')
    if (spec.get('label_projection','loo')=='cash_common_beta')!=(spec.get('risk_mapping','legacy_loo')=='common_label'):raise ValueError('label and executable basket subspaces disagree')
    if spec.get('initial_penalty',.1) not in [.01,.1,1.,10.]:raise ValueError('invalid initial ridge penalty')
    initial_penalties=spec.get('initial_penalties',[spec.get('initial_penalty',.1)])
    if not initial_penalties or len(set(initial_penalties))!=len(initial_penalties) or set(initial_penalties)-{.01,.1,1.,10.}:raise ValueError('invalid common penalty coverage')
    if spec['model_trials']<2*len(spec['model_kinds'])*len(initial_penalties):raise ValueError('model trial budget cannot cover declared fair grid')
    if not 0<=spec.get('structure_probe_examples',0)<=64:raise ValueError('structural probe budget out of bounds')
    if spec.get('structure_probe_examples',0) and data['source']!='crypto_parquet':raise ValueError('real structural probe requires real minute observations')
    if spec.get('target_policy','raw') not in {'raw','causal_volatility'}:raise ValueError('invalid residual target units')
    if not isinstance(spec.get('oof_calibration',False),bool) or spec.get('edge_floor_bps',0)<0:raise ValueError('invalid OOF calibration or economic edge floor')
    if spec.get('readout_policy','joint') not in {'joint','orthogonal_increment'}:raise ValueError('invalid readout policy')
    if 'execution_by_horizon' in spec:
        if set(spec['execution_by_horizon'])!={4,12}:raise ValueError('execution channels must match fixed target horizons')
        for policy in spec['execution_by_horizon'].values():
            if set(policy)!={'ema_bars','deadband'} or policy['ema_bars']<1 or policy['deadband']<0:raise ValueError('invalid horizon execution channel')
    if 'extension' in spec:
        ext=spec['extension']
        if set(ext)!={'parent_bank_manifest','parent_config','age_blocks'} or not ext['age_blocks']:raise ValueError('invalid additive representation extension')
        prior=0
        for a,b in ext['age_blocks']:
            if a!=prior or not a<b<=1440:raise ValueError('log-age blocks must be disjoint and contiguous')
            prior=b
    if spec['device'] not in {'cpu','cuda'}: raise ValueError('explicit cpu or cuda required')
    if set(spec['representation'])!={'windows_minutes','normalization_minutes','normalization_min_count','seed'}:
        raise ValueError('invalid representation specification')
    rep=spec['representation'];w=rep['windows_minutes']
    if len(w)!=3 or w!=sorted(set(w)) or not all(30<=v<=1440 for v in w): raise ValueError('three bounded increasing representation windows required')
    if not 2<=rep['normalization_min_count']<=rep['normalization_minutes']<=20160: raise ValueError('invalid causal normalizer')
    if set(spec['budgets'])!={'F1','F2','F3','F4','F5','F6','F7','OLD'} or any(not 1<=v<=30000 for v in spec['budgets'].values()):
        raise ValueError('bounded family budgets required')
    if not 1<=spec['old_control_budget']<=30000 or not 1<=spec['model_trials']<=100: raise ValueError('search budget invalid')
    if not 1<=spec['fine_count']<=spec['medium_count']<=1000 or not 1<=spec['max_selected']<=12: raise ValueError('screening/selection budget invalid')
    allowed={'old','old_nonlinear','joint','joint_interaction','joint_interaction_residual','pool','F1','F2','F3','F4','F5','F6','F7','F8',
          'F3_order1','F3_order2','F3_natural','F3_activity','F1_composition','F2_energy','F4_occupancy','F5_diagonal','F7_raw','F7_reference',
        'seed_F1','seed_F2','seed_F3','seed_F4','seed_F5','seed_F6','seed_F7','seed_OLD'}
    if set(spec['model_kinds'])-allowed or not set(['old','old_nonlinear','joint']).issubset(spec['model_kinds']): raise ValueError('fair baseline/model kinds required')
    if 'F7_reference' in spec['model_kinds'] and spec.get('kernel_reference','none')!='gaussian_moments':raise ValueError('F7 reference readout requires a declared Gaussian moment reference')
    split=Split(**cfg['split']); prior=spec['feature_calibration_end']
    if len(spec['folds'])!=5: raise ValueError('calibration OOF plus four discovery folds required')
    for a,b in spec['folds']:
        if not prior<a<=b<=split.discovery_end: raise ValueError('overlapping or firewall-crossing discovery folds')
        prior=b
    if spec['cost_bps']<0 or not 0<spec['gross_budget']<=1:raise ValueError('invalid trading budget')
    if set(spec['execution'])!={'ema_bars','deadband'} or spec['execution']['ema_bars']<1 or spec['execution']['deadband']<0:raise ValueError('frozen shared execution required')
    if not isinstance(spec['audit_holdout'],bool):raise ValueError('explicit audit switch required')
    if set(cfg['output'])!={'out_dir','visualization_dir'}:raise ValueError('output and visualization directories required')


def _simulated_bank(cfg,cache_root):
    """Known synthetic relationships exercise clocks and fitting, never evidence of alpha."""
    spec=cfg['complex_alpha']['representation']; data=cfg['data'];seed=cfg['complex_alpha']['seed']
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+WORKFLOW_VERSION.encode()).hexdigest()
    dest=cache_root/('sim_'+signature[:16]);dest.mkdir(parents=True,exist_ok=True)
    clock=pd.date_range(data['start'],data['end'],freq='min',tz='UTC');rng=np.random.default_rng(seed)
    n=len(clock);assets=len(data['assets']);flow=rng.normal(size=(n,assets))
    for t in range(1,n):flow[t]=.92*flow[t-1]+.2*flow[t]
    ret=rng.normal(0,.0004,(n,assets))+.0001*np.roll(np.tanh(flow),15,axis=0)
    close=100*np.exp(np.cumsum(ret,axis=0));market=ret.mean(axis=1)
    available=clock+pd.Timedelta(minutes=1);ends=np.flatnonzero(available.minute%15==0)+1
    fast=np.flatnonzero(available.minute%5==0)
    np.save(dest/'dates.npy',available[ends-1].strftime('%Y%m%d%H%M').to_numpy().astype('U12'))
    np.save(dest/'fast_dates.npy',available[fast].strftime('%Y%m%d%H%M').to_numpy().astype('U12'))
    np.save(dest/'fast_close.npy',close[fast])
    quality=[]
    for j,code in enumerate(data['assets']):
        q=np.exp(12+rng.normal(0,.7,n)); count=np.maximum((q/1000).astype(int),1)
        frame=pd.DataFrame({'date':clock,'close':close[:,j],'high':close[:,j]*1.0005,'low':close[:,j]*.9995,
            'volume':q/close[:,j],'quote_volume':q,'trade_count':count,
            'taker_buy_quote_volume':q*(1+np.tanh(flow[:,j]))/2})
        z,qa=minute_observations(frame,market,spec['normalization_minutes'],spec['normalization_min_count'])
        value,cards,rel=representation_bank(z,ends,spec['windows_minutes'],spec['seed'],q)
        if cfg['complex_alpha'].get('kernel_reference','none')=='gaussian_moments':
            from .complex_alpha_kernel_reference import joint_covariance_controls
            controls,control_cards=joint_covariance_controls(z,ends,spec['windows_minutes'])
            value=np.column_stack((value,controls));cards+=control_cards
        np.save(dest/f'{code}.npy',value);quality.append({'asset':code,'channels':qa,'synthetic':True})
    return {'status':'COMPLETED','signature':signature,'cache_dir':str(dest),'codes':data['assets'],
        'coordinates':cards,'quality':quality,'reliability':[],'source_files':[{'simulated':True}],
        'minute_rows':n,'decision_rows':len(ends),'version':DATA_VERSION}


def _load_bank(bank,calibration_end):
    root=Path(bank['cache_dir']);dates=np.load(root/'dates.npy')
    calibration=dates<=calibration_end
    values=np.empty((len(dates),len(bank['codes']),len(bank['coordinates'])),np.float32);parameters=[]
    for asset,code in enumerate(bank['codes']):
        x=np.load(root/f'{code}.npy',mmap_mode='r');sample=x[calibration]
        count=np.isfinite(sample).sum(axis=0)
        mean=np.nanmean(sample,axis=0);scale=np.nanstd(sample,axis=0)
        valid=(count>=max(96,int(calibration.sum()*.90)))&np.isfinite(scale)&(scale>1e-7)
        mean[~valid]=0;scale[~valid]=1
        for start in range(0,len(x),16384):
            normalized=np.clip(np.nan_to_num((x[start:start+16384]-mean)/scale,nan=0.),-8,8).astype(np.float32)
            normalized[:,~valid]=0;values[start:start+len(normalized),asset]=normalized
        parameters.append({'asset':code,'mean':mean.tolist(),'scale':scale.tolist(),'valid':valid.tolist()})
    return values,dates,np.load(root/'fast_dates.npy'),np.load(root/'fast_close.npy'),parameters


def _funding(cfg,dates,fast_dates,fast_close):
    if cfg['data']['source']=='simulated':return np.zeros_like(fast_close),[],{}
    hour=np.flatnonzero(np.array([v[-2:]=='00' for v in fast_dates]))
    decision=fast_dates[hour]
    raw=(pd.to_datetime(decision,format='%Y%m%d%H%M',utc=True)-pd.Timedelta(hours=1)).strftime('%Y%m%d%H%M').to_numpy()
    panels,files=load_derivative_panels(cfg['data']['derivative_root'],cfg['data']['assets'],raw,fast_close[hour])
    funding=np.zeros_like(fast_close)
    funding[hour]=np.nan_to_num(panels['derivative_funding_event'],nan=0.)
    at=np.searchsorted(decision,dates,side='right')-1
    states={}
    for name in ['derivative_basis','derivative_trade_mark_gap','derivative_funding_state']:
        v=np.full((len(dates),fast_close.shape[1]),np.nan);good=at>=0
        v[good]=panels[name][at[good]];states[name]=v
    return funding,files,states


def _oof_masks(dates,folds):
    return {('calibration' if i==0 else f'fold{i-1}'):(dates>=a)&(dates<=b) for i,(a,b) in enumerate(folds)}


def _model_stats(pred,base,target,masks):
    rows=[]
    for name,mask in masks.items():
        y=target[mask];b=base[mask];p=pred[mask]; e=y-b;delta=p-b
        loss=float(np.nanmean(e*e));gain=float(np.nanmean(e*e-(y-p)**2))
        rows.append({'fold':name,'incremental_ic':correlation(delta,e),
            'loss_gain_fraction':gain/loss if loss>0 else np.nan,'prediction_ic':correlation(p,y)})
    main=[r for r in rows if r['fold']!='calibration']
    gains=np.array([r['loss_gain_fraction'] for r in main]);ics=np.array([r['incremental_ic'] for r in main])
    return {'objective':float(np.mean(gains)-.25*np.std(gains)),
        'mean_incremental_ic':float(np.nanmean(ics)) if np.isfinite(ics).any() else 0.,'positive_gain_folds':int(np.sum(gains>0)),
        'mean_loss_gain_fraction':float(np.mean(gains)),'folds':rows}


def _calibrate_oof(pred,base,target,dates,folds,horizon):
    """Fit only on previous fully matured OOF outputs, no in-sample scaling."""
    result=pred.copy();stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True);records=[]
    for start,end in folds[1:]:
        before=(stamp+pd.Timedelta(hours=horizon)<pd.to_datetime(start,format='%Y%m%d%H%M',utc=True))
        a=(pred-base)[before].ravel();b=(target-base)[before].ravel();valid=np.isfinite(a)&np.isfinite(b)
        gamma=float(np.clip(np.dot(a[valid],b[valid])/max(np.dot(a[valid],a[valid]),1e-16),0,1)) if valid.sum()>100 else 0.
        test=(dates>=start)&(dates<=end);result[test]=base[test]+gamma*(pred[test]-base[test])
        records.append({'start':start,'end':end,'gamma':gamma,'matured_samples':int(valid.sum())})
    initial=(dates>=folds[0][0])&(dates<=folds[0][1]);result[initial]=base[initial]
    return result,records


def _final_oof_gamma(pred,base,target,mask):
    a=(pred-base)[mask].ravel();b=(target-base)[mask].ravel();valid=np.isfinite(a)&np.isfinite(b)
    return float(np.clip(np.dot(a[valid],b[valid])/max(np.dot(a[valid],a[valid]),1e-16),0,1)) if valid.sum()>100 else 0.


def run_complex_alpha(cfg,config_path):
    if cfg.get('complex_alpha', {}).get('stage') == 'effective_combo_extension':
        from .effective_combo_extension_workflow import run_extension
        return run_extension(cfg, config_path)
    if cfg.get('complex_alpha', {}).get('stage') == 'effective_combo':
        from .effective_combo_workflow import run_effective_combo
        return run_effective_combo(cfg, config_path)
    if cfg.get('complex_alpha', {}).get('stage') == 'predictive_pool':
        from .complex_alpha_prediction_workflow import run_prediction_stage
        return run_prediction_stage(cfg, config_path)
    validate_complex_config(cfg);start=time.monotonic();s=cfg['complex_alpha'];out=Path(cfg['output']['out_dir']);out.mkdir(parents=True,exist_ok=True)
    marker=out/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())['status']=='COMPLETED':raise ValueError('completed complex-alpha experiment is immutable')
    code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('complex_alpha_*.py')))).hexdigest()
    run_signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+code.encode()).hexdigest()
    lock=out/'run_signature.json'
    if lock.exists() and json.loads(lock.read_text())['signature']!=run_signature:raise ValueError('resume config/code signature changed; use a new output directory')
    _write_json(lock,{'signature':run_signature,'code_sha256':code});(out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    import shutil
    for p in list(Path(__file__).parent.glob('complex_alpha_*.py'))+[Path(__file__).with_name('manual_alpha_derivatives.py'),Path(__file__).with_name('aligned_contract.py')]:
        shutil.copyfile(p,snapshot/p.name)
    _write_json(marker,{'status':'RUNNING','lane':'complex_alpha','signature':run_signature})
    _write_json(out/'runtime.json',{'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,
        'time_contract':TIME_VERSION,'epoch_unit':'explicit_nanoseconds'})
    if 'extension' in s:
        from .complex_alpha_extension import extend_verified_bank
        bank=extend_verified_bank(cfg)
    else:
        bank=(prepare_minute_bank(cfg,Path(cfg['data']['feature_cache'])) if cfg['data']['source']=='crypto_parquet'
              else _simulated_bank(cfg,Path(cfg['data']['feature_cache'])))
    if s.get('kernel_reference','none')=='gaussian_moments':
        from .complex_alpha_kernel_reference import augment_kernel_reference
        bank=augment_kernel_reference(bank,Path(cfg['data']['feature_cache']),s['representation']['windows_minutes'],s['representation']['seed'])
    _write_json(out/'field_registry.json',{'coordinates':bank['coordinates'],'quality':bank['quality'],'reliability':bank['reliability']})
    x,dates,fast_dates,fast_close,scaler=_load_bank(bank,s['feature_calibration_end'])
    _write_json(out/'initial_coordinate_scaler.json',scaler)
    codes=bank['codes'];cards=bank['coordinates'];split=Split(**cfg['split']);idx=np.searchsorted(fast_dates,dates)
    beta=past_market_betas(fast_close[idx],width=2880 if cfg['data']['source']=='crypto_parquet' else 192,return_contract=s.get('return_contract','log'))
    funding,funding_files,derivatives=_funding(cfg,dates,fast_dates,fast_close)
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    calendar={'clock_sin':np.sin(2*np.pi*(stamp.hour+stamp.minute/60)/24),
              'clock_cos':np.cos(2*np.pi*(stamp.hour+stamp.minute/60)/24),
              'weekday_sin':np.sin(2*np.pi*stamp.dayofweek/7),'weekday_cos':np.cos(2*np.pi*stamp.dayofweek/7)}
    for name,v in calendar.items():derivatives[name]=np.repeat(np.asarray(v)[:,None],len(codes),axis=1)
    for name,values in derivatives.items():
        sample=values[dates<=s['feature_calibration_end']];mu=np.nanmean(sample,axis=0);sd=np.nanstd(sample,axis=0)
        valid=np.isfinite(sd)&(sd>1e-12);mu[~valid]=0;sd[~valid]=1
        normalized=np.clip(np.nan_to_num((values-mu)/sd,nan=0.),-8,8);normalized[:,~valid]=0
        x=np.concatenate((x,normalized[...,None].astype(np.float32)),axis=-1)
        cards.append({'name':name,'family':'OLD','support_minutes':480,'order':1,'fitted':False,
            'dependencies':['hourly_derivatives'] if name.startswith('derivative') else ['completed_timestamp'],
            'unit':'dimensionless','clock':'completed_15m','missing':'initial mean imputation; scale frozen before OOF'})
    initial=x[dates<=s['feature_calibration_end']]
    asset_variation=np.std(initial,axis=0)
    for k,card in enumerate(cards):
        card['admitted_assets']=int(np.sum(asset_variation[:,k]>1e-7))
        card['admitted']=card['admitted_assets']>=max(3,len(codes)//2)
        card['type']='dimensionless_scalar_at_completed_15m'
    input_signature=hashlib.sha256(json.dumps({'bank':bank['signature'],'derivatives':funding_files},sort_keys=True).encode()).hexdigest()
    input_lock=out/'input_signature.json'
    if input_lock.exists() and json.loads(input_lock.read_text())['signature']!=input_signature:raise ValueError('resume input data changed')
    _write_json(input_lock,{'signature':input_signature,'minute_bank':bank['signature'],'derivative_files':funding_files})
    _write_json(out/'field_registry.json',{'coordinates':cards,'quality':bank['quality'],'reliability':bank['reliability']})
    raw,y=targets_from_prices(dates,fast_dates,fast_close,[4,12],split,beta,funding,return_contract=s.get('return_contract','log'))
    if s.get('label_projection','loo')=='cash_common_beta':y={h:project_basket_returns(v,beta) for h,v in raw.items()}
    risk_scales={h:np.ones_like(v) for h,v in y.items()}
    if s.get('target_policy','raw')=='causal_volatility':
        risk_scales=causal_risk_scales(fast_close[idx],beta,[4,12],return_contract=s.get('return_contract','log'),label_projection=s.get('label_projection','loo'))
        y={h:v/risk_scales[h] for h,v in y.items()}
    for h,v in risk_scales.items():np.save(out/f'target_price_scale_{h}h.npy',v.astype(np.float32))
    # Exactly four-hour sampled, rotating day/minute phases. Full medium uses 1h.
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    coarse,medium=rotating_screening_masks(stamp)
    coarse=coarse&(dates<=split.discovery_end)
    medium=medium&(dates<=split.validation_end)
    if cfg['data']['source']=='simulated':coarse=(np.arange(len(dates))%4==0)&(dates<=split.discovery_end)
    old_predictions={}; old_summaries={}; residual={};coarse_masks=_oof_masks(dates[coarse],s['folds'])
    for h in [4,12]:
        base,summary,_=crossfit_readout(x[coarse],dates[coarse],y[h][coarse],cards,'old',.1,s['folds'],h,s['seed'],s['device'])
        old_predictions[h]=base;old_summaries[h]=summary
        scale=np.nanstd(y[h][coarse][coarse_masks['calibration']],axis=0)
        scale[~np.isfinite(scale)|(scale<1e-6)]=1
        residual[h]=(y[h][coarse]-base)/scale
    candidates=generate_candidates(cards,s['budgets'],s['seed'],s['modes'])
    controls=generate_candidates(cards,{'OLD':s['old_control_budget']},s['seed']+1,s['modes'],'control')
    all_candidates=candidates+controls;by_id={c.id:c for c in all_candidates}
    _write_json(out/'candidate_registry.json',{'version':WORKFLOW_VERSION,'definitions':[c.record() for c in all_candidates],
        'coordinate_count':len(cards),'candidate_count':len(all_candidates),'target_count':2*len(all_candidates),
        'baseline_oof':old_summaries,'source_signature':bank['signature']})
    coarse_path=out/'coarse_results.csv';resume=pd.read_csv(coarse_path) if coarse_path.exists() else None
    coarse_frame=screen_candidates(x[coarse],residual,coarse_masks,all_candidates,s['device'],s['coarse_batch'],
        lambda frame:_write_csv(coarse_path,frame),resume)
    survivors=diversity_survivors(coarse_frame,s['medium_count'],max(20,s['medium_count']//5))
    survivors,aliases=numeric_archive(x[coarse],[by_id[v] for v in survivors.id],survivors)
    _write_csv(out/'numeric_equivalence_archive.csv',aliases)
    # Medium survival uses all rotating-hour discovery samples, baseline residuals
    # from expanding fit. Candidate directions remain from initial discovery OOF.
    xm=x[medium];dm=dates[medium];mm=_oof_masks(dm,s['folds']);bm={};em={}
    for h in [4,12]:
        bm[h],_,_=crossfit_readout(xm,dm,y[h][medium],cards,'old',.1,s['folds'],h,s['seed'],s['device'])
        em[h]=y[h][medium]-bm[h]
    rows=[]
    for number,row in enumerate(survivors.itertuples()):
        c=by_id[row.id];v=evaluate_candidate(xm,c)*row.direction
        fold=[correlation(v[m],em[row.horizon_hours][m]) for key,m in mm.items() if key!='calibration']
        rows.append({**row._asdict(),'medium_mean_ic':float(np.nanmean(fold)),
            'medium_positive_folds':int(np.sum(np.asarray(fold)>0)),'medium_min_ic':float(np.nanmin(fold)),
            'medium_score':float(np.nanmean(fold)-.35*np.nanstd(fold))})
        if number%50==0:print(f'[complex medium] {number}/{len(survivors)} discovery survival',flush=True)
    medium_frame=pd.DataFrame(rows).sort_values(['medium_score','id'],ascending=[False,True]);_write_csv(out/'medium_results.csv',medium_frame)
    # Reserve one discovery-ranked seed per represented family for bounded
    # exploration. This never forces final admission or reads validation.
    family_seeds=medium_frame.groupby('family',sort=True).head(1)
    fine=pd.concat((family_seeds,medium_frame)).drop_duplicates('id').head(s['fine_count'])
    fine=fine.sort_values(['medium_score','id'],ascending=[False,True]);_write_csv(out/'fine_candidates.csv',fine)
    # A prediction feature pool survives independently of later trading admission.
    from .complex_alpha_prediction import discovery_archive
    information_archive=discovery_archive(coarse_frame)
    _write_csv(out/'candidate_information_archive.csv',information_archive)
    feature_pool=fine.merge(information_archive[['id','horizon_hours','research_tier','sign_changes']],
                           on=['id','horizon_hours'],how='left',validate='one_to_one')
    _write_csv(out/'prediction_feature_pool.csv',feature_pool)
    matched=matched_structure_windows(x,dates,cards,dates<=split.discovery_end)
    _write_csv(out/'matched_observation_windows.csv',matched)
    null=limited_shift_null(x[coarse],residual,coarse_masks,[by_id[v] for v in fine.id],s['seed'],
        repeats=min(40,s['bootstrap_trials']))
    _write_csv(out/'survivor_shift_null.csv',null)
    # A searched collection enters a separate readout channel. Its outer folds
    # are research feedback, not claimed independent after candidate selection.
    pool=[]
    for row in fine.itertuples():
        c=by_id[row.id];pool.append(evaluate_candidate(x,c).astype(np.float32))
        cards.append({'name':row.id,'family':'POOL','source_family':c.family,'support_minutes':c.support_minutes,'order':c.nonlinear_order,
            'dependencies':[bank['coordinates'][i]['name'] for i in c.left+c.right],
            'fitted':False,'selection':'discovery searched collection'})
    x=np.concatenate((x,np.stack(pool,axis=-1)),axis=-1);xm=x[medium]
    _write_json(out/'readout_coordinate_registry.json',cards)
    # Independent fixed-family structure blocks are not filtered by individual IC.
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=s['seed']),
        study_name=cfg['version'],storage=f'sqlite:///{(out/"readout_study.sqlite3").resolve().as_posix()}',load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results=[]; objective_cache={}
    for t in study.trials:
        if t.state==optuna.trial.TrialState.COMPLETE:
            objective_cache[(t.params['kind'],t.params['horizon'],t.params['penalty'])]=t.number
    def trial_objective(trial):
        kind=trial.suggest_categorical('kind',s['model_kinds']);h=trial.suggest_categorical('horizon',[4,12]);penalty=trial.suggest_categorical('penalty',[.01,.1,1.,10.])
        key=(kind,h,penalty)
        if key in objective_cache:
            previous=objective_cache[key]
            evidence=json.loads((out/f'readout_trial_{previous:03d}.json').read_text())
            record={**evidence['result'],'trial':trial.number,'reused_from_trial':previous}
            trial.set_user_attr('metrics',record);results.append(record)
            _write_json(out/f'readout_trial_{trial.number:03d}.json',{**evidence,'result':record})
            _write_csv(out/'readout_trials.csv',pd.DataFrame([{k:v for k,v in row.items() if k!='folds'} for row in results]))
            print(f'[complex readout] trial {trial.number}: reused deterministic {previous}',flush=True)
            return record['objective']
        pred,_,states=crossfit_readout(xm,dm,y[h][medium],cards,kind,penalty,s['folds'],h,s['seed'],s['device'],s.get('readout_policy','joint'))
        calibration=[]
        if s.get('oof_calibration',False):pred,calibration=_calibrate_oof(pred,bm[h],y[h][medium],dm,s['folds'],h)
        metrics=_model_stats(pred,bm[h],y[h][medium],mm)
        record={'trial':trial.number,'kind':kind,'horizon_hours':h,'penalty':penalty,**metrics}
        results.append(record);objective_cache[key]=trial.number
        trial.set_user_attr('metrics',record)
        _write_json(out/f'readout_trial_{trial.number:03d}.json',{'result':record,'states':states,'oof_calibration':calibration})
        _write_csv(out/'readout_trials.csv',pd.DataFrame([{k:v for k,v in row.items() if k!='folds'} for row in results]))
        print(f'[complex readout] trial {trial.number}: {kind}/{h}h gain={metrics["mean_loss_gain_fraction"]:+.6f}',flush=True)
        return metrics['objective']
    # Cover every family and fair readout at both targets before adaptive search.
    if not study.trials:
        for kind in s['model_kinds']:
            for h in [4,12]:
                for penalty in s.get('initial_penalties',[s.get('initial_penalty',.1)]):
                    study.enqueue_trial({'kind':kind,'horizon':h,'penalty':penalty})
    remaining=max(0,s['model_trials']-len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]))
    if remaining:study.optimize(trial_objective,n_trials=remaining,n_jobs=1)
    records=[t.user_attrs['metrics'] for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    _write_csv(out/'readout_trials.csv',pd.DataFrame([{k:v for k,v in row.items() if k!='folds'} for row in records]))
    # Best per architecture/horizon selected on discovery; validation only survival.
    ranked=pd.DataFrame([{k:v for k,v in row.items() if k!='folds'} for row in records]).sort_values('objective',ascending=False)
    best=ranked.drop_duplicates(['kind','horizon_hours']);models=[];validation=[];oof_predictions={}
    train=dates<=split.discovery_end
    # Shared base fit plus full-resolution base OOF.
    base_full={};base_final={}
    for h in [4,12]:
        base_full[h],_,_=crossfit_readout(x,dates,y[h],cards,'old',.1,s['folds'],h,s['seed'],s['device'])
        design,meta=parameter_design(x,cards,'old',train,y[h],s['seed'],s['device'])
        base_state=fit_ridge(design[train].reshape(-1,design.shape[-1]),y[h][train].reshape(-1),.1,s['device'])
        base_final[h]=base_state.predict(design.reshape(-1,design.shape[-1])).reshape(y[h].shape)
        base_signal=base_full[h].copy();base_signal[~train]=base_final[h][~train];base_signal*=risk_scales[h]
        base_scale=float(np.sqrt(np.nanmean(base_signal[train]**2)))
        execution=s.get('execution_by_horizon',{}).get(h,s['execution'])
        baseline_position=neutral_positions(base_signal,beta,max(base_scale,1e-8),s['gross_budget'],execution['ema_bars'],execution['deadband'],s.get('edge_floor_bps',0.),s.get('risk_mapping','legacy_loo'),s.get('exit_on_inactive',False))
        end_fast=np.searchsorted(fast_dates,split.holdout_start)
        baseline_ledger,baseline_asset,_=five_minute_ledger(baseline_position,dates,fast_dates[:end_fast],fast_close[:end_fast],funding[:end_fast],s['cost_bps'],split)
        baseline_ledger.to_parquet(out/f'old_baseline_{h}h_research_ledger.parquet',index=False)
        _write_json(out/f'old_baseline_{h}h_metrics.json',{p:ledger_metrics(baseline_ledger,baseline_asset,baseline_ledger.period.eq(p).to_numpy()) for p in ['discovery','validation']})
    for row in best.itertuples():
        h=row.horizon_hours;kind=row.kind;identifier=f'{kind}_{h}h_trial{row.trial}'
        pred,_,states=crossfit_readout(x,dates,y[h],cards,kind,row.penalty,s['folds'],h,s['seed'],s['device'],s.get('readout_policy','joint'))
        final_gamma=1.;calibration=[]
        if s.get('oof_calibration',False):
            final_gamma=_final_oof_gamma(pred,base_full[h],y[h],train)
            pred,calibration=_calibrate_oof(pred,base_full[h],y[h],dates,s['folds'],h)
        delta_units=pred-base_full[h];delta=delta_units*risk_scales[h];oof_predictions[identifier]=delta
        use_oof=np.isfinite(delta)&train[:,None];scale=float(np.sqrt(np.nanmean(delta[use_oof]**2)))
        # Zero calibrated increments remain explicit no-trade research results.
        if not np.isfinite(scale):scale=1e-8
        scale=max(scale,1e-8)
        design,meta=parameter_design(x,cards,kind,train,y[h],s['seed'],s['device'])
        state=fit_ridge(design[train].reshape(-1,design.shape[-1]),y[h][train].reshape(-1),row.penalty,s['device'])
        final=state.predict(design.reshape(-1,design.shape[-1])).reshape(y[h].shape)-base_final[h]
        increment_metadata={}
        if s.get('readout_policy','joint')=='orthogonal_increment' and kind!='old':
            old_design,_=parameter_design(x,cards,'old',train,y[h],s['seed'],s['device'])
            base_state=fit_ridge(old_design[train].reshape(-1,old_design.shape[-1]),y[h][train].reshape(-1),.1,s['device'])
            projector,gamma=fit_increment_projector(design,old_design,state,base_state,train,y[h],s['device'])
            final=gamma*(final-projector.predict(old_design.reshape(-1,old_design.shape[-1])).reshape(y[h].shape))
            increment_metadata={'projector':projector.record(),'gamma':gamma}
            meta['fit_parameters']+=old_design.shape[-1]+2
        elif s.get('readout_policy','joint')=='orthogonal_increment' and kind=='old':
            final[:]=0;delta[:]=0
        final*=final_gamma
        final_units=final.copy();final*=risk_scales[h]
        delta[~train]=final[~train]
        execution=s.get('execution_by_horizon',{}).get(h,s['execution'])
        position=neutral_positions(delta,beta,scale,s['gross_budget'],execution['ema_bars'],execution['deadband'],s.get('edge_floor_bps',0.),s.get('risk_mapping','legacy_loo'),s.get('exit_on_inactive',False))
        # Holdout prices/cashflows are not evaluated at this stage.
        end_fast=np.searchsorted(fast_dates,split.holdout_start)
        ledger,asset_net,holdings=five_minute_ledger(position,dates,fast_dates[:end_fast],fast_close[:end_fast],funding[:end_fast],s['cost_bps'],split)
        vmask=ledger.period.eq('validation').to_numpy();dmask=ledger.period.eq('discovery').to_numpy()
        metric=ledger_metrics(ledger,asset_net,vmask);discover_metric=ledger_metrics(ledger,asset_net,dmask)
        valmask=(dates>=split.validation_start)&(dates<=split.validation_end)
        vic=correlation(final_units[valmask],y[h][valmask]-base_final[h][valmask])
        record={**row._asdict(),'id':identifier,'signal_scale':scale,'validation_incremental_ic':vic,
            **{f'validation_{k}':v for k,v in metric.items()},**{f'discovery_{k}':v for k,v in discover_metric.items()},
            'fit_parameters':meta['fit_parameters']}
        interval=block_gain_interval(pred-base_full[h],y[h],base_full[h],dates,
            (dates>=s['folds'][1][0])&train,s['bootstrap_trials'],s['seed'])
        record['discovery_gain_ci_low']=interval['low'];record['discovery_gain_ci_high']=interval['high']
        _write_json(out/f'{identifier}_gain_bootstrap.json',interval)
        maturity=nonoverlap_phase(stamp,h)&train&np.isfinite(delta_units).any(axis=1)
        record['nonoverlap_incremental_ic']=correlation(delta_units[maturity],(y[h]-base_full[h])[maturity])
        record['research_role']=('provisional_incremental_alpha' if row.mean_loss_gain_fraction>0 and metric['net_return_pct']>0 and vic>0
            else ('information_not_tradable' if row.mean_incremental_ic>0 and vic>0 else 'unsupported_or_unstable'))
        validation.append(record);models.append({'id':identifier,'horizon':h,'kind':kind,'penalty':row.penalty,
            'scale':scale,'state':state.record(),'design':meta,'discovery_oof_states':states,'increment':increment_metadata,'execution':execution,
            'oof_gamma':final_gamma,'oof_calibration':calibration})
        ledger.to_parquet(out/f'{identifier}_research_ledger.parquet',index=False)
        np.save(out/f'{identifier}_prediction.npy',delta.astype(np.float32))
        print(f'[complex validation] {identifier}: net={metric["net_return_pct"]:+.3f}% IC={vic:+.4f}',flush=True)
    validation_frame=pd.DataFrame(validation);_write_csv(out/'model_survival.csv',validation_frame)
    eligible=validation_frame[(validation_frame.mean_loss_gain_fraction>0)&(validation_frame.positive_gain_folds>=2)&
        (validation_frame.validation_incremental_ic>0)&(validation_frame.validation_net_return_pct>0)&
        (validation_frame.validation_positive_asset_share>=.5)]
    selected=eligible.sort_values(['objective','id'],ascending=[False,True]).head(s['max_selected'])
    _write_csv(out/'selected_factors.csv',selected)
    # Legacy selected_factors is a trading-product alias, never a feature-pool count.
    _write_csv(out/'selected_trade_products.csv',selected)
    _write_json(out/'frozen_models.json',{'models':models,'selected':selected.id.tolist(),'scaler':str(out/'initial_coordinate_scaler.json')})
    freeze={'status':'FROZEN','selection_uses':['discovery_oof_loss','discovery_fold_breadth','historical_validation_survival'],
        'selected':selected.id.tolist(),'cost_bps':s['cost_bps'],'execution':s['execution'],'source_signature':bank['signature'],
        'run_signature':run_signature,'holdout_opened':False,'historical_periods_previously_seen':True,
        'return_contract':s.get('return_contract','log'),'label_projection':s.get('label_projection','loo'),
        'risk_mapping':s.get('risk_mapping','legacy_loo'),'exit_on_inactive':s.get('exit_on_inactive',False),
        'execution_by_horizon':s.get('execution_by_horizon',{}),'edge_floor_bps':s.get('edge_floor_bps',0.),
        'target_policy':s.get('target_policy','raw'),'oof_calibration':s.get('oof_calibration',False),
        'time_contract':TIME_VERSION,'epoch_unit':'explicit_nanoseconds'}
    _write_json(out/'selection_freeze.json',freeze)
    if s.get('structure_probe_examples',0):
        from .complex_alpha_structure_probe import run_structure_probe
        _write_csv(out/'structural_counterfactuals.csv',run_structure_probe(cfg,s['structure_probe_examples']))
    combo=np.zeros((len(dates),len(codes)));position=np.zeros_like(combo)
    for row in selected.itertuples():
        prediction=np.load(out/f'{row.id}_prediction.npy')
        combo+=np.nan_to_num(prediction/row.signal_scale,nan=0.)/len(selected)
        execution=s.get('execution_by_horizon',{}).get(row.horizon_hours,s['execution'])
        position+=neutral_positions(prediction,beta,row.signal_scale,s['gross_budget'],execution['ema_bars'],execution['deadband'],s.get('edge_floor_bps',0.),s.get('risk_mapping','legacy_loo'),s.get('exit_on_inactive',False))/len(selected)
    end_fast=len(fast_dates) if s['audit_holdout'] else np.searchsorted(fast_dates,split.holdout_start)
    ledger,asset_net,holdings=five_minute_ledger(position,dates,fast_dates[:end_fast],fast_close[:end_fast],funding[:end_fast],s['cost_bps'],split)
    ledger.to_parquet(out/'combo_5m_ledger.parquet',index=False)
    pd.DataFrame(asset_net,columns=codes).to_parquet(out/'combo_asset_pnl.parquet',index=False)
    np.save(out/'combo_decision_position.npy',position.astype(np.float32))
    metrics={p:ledger_metrics(ledger,asset_net,ledger.period.eq(p).to_numpy()) for p in ledger.period.unique()}
    _write_json(out/'combo_metrics.json',metrics)
    sensitivity=[]
    for cost in [0.,2.,4.,6.,8.]:
        f,a,_=five_minute_ledger(position,dates,fast_dates[:end_fast],fast_close[:end_fast],funding[:end_fast],cost,split)
        for period in f.period.unique():sensitivity.append({'cost_bps':cost,'period':period,**ledger_metrics(f,a,f.period.eq(period).to_numpy())})
    _write_csv(out/'cost_sensitivity.csv',pd.DataFrame(sensitivity))
    bindex=np.searchsorted(dates,fast_dates[:end_fast],side='right')-1;safe=np.maximum(bindex,0)
    risk_beta=common_market_loadings(beta) if s.get('risk_mapping','legacy_loo') in {'exact_loo_to_common','common_label'} else beta
    exposure=np.nan_to_num(risk_beta[safe],nan=0.)*holdings
    _write_json(out/'exposure_diagnostics.json',{'mean_abs_cash':float(np.mean(np.abs(holdings.mean(axis=1)))),
        'mean_abs_past_beta':float(np.mean(np.abs(exposure.mean(axis=1)))),
        'max_abs_past_beta':float(np.max(np.abs(exposure.mean(axis=1)))),'note':'weights drift between 15m orders; omitted/nonlinear styles remain possible'})
    # Correlation, leave-one-out contribution and no-trade full reporting are post-freeze.
    if s['audit_holdout']:
        audit=[]
        for row in selected.itertuples():
            prediction=np.load(out/f'{row.id}_prediction.npy');execution=s.get('execution_by_horizon',{}).get(row.horizon_hours,s['execution']);p=neutral_positions(prediction,beta,row.signal_scale,s['gross_budget'],execution['ema_bars'],execution['deadband'],s.get('edge_floor_bps',0.),s.get('risk_mapping','legacy_loo'),s.get('exit_on_inactive',False))
            l,a,_=five_minute_ledger(p,dates,fast_dates,fast_close,funding,s['cost_bps'],split)
            audit.append({'id':row.id,**ledger_metrics(l,a,l.period.eq('holdout').to_numpy())})
            l.to_parquet(out/f'{row.id}_frozen_audit_ledger.parquet',index=False)
        _write_csv(out/'holdout_audit.csv',pd.DataFrame(audit))
    summary={'candidate_definitions':len(all_candidates),'candidate_target_records':len(coarse_frame),
        'main_definitions':len(candidates),'old_control_definitions':len(controls),'coordinates':len(cards),
        'medium_survivors':len(survivors),'fine_definitions':len(fine),'readout_trials':len(records),
        'unique_readout_fits':len({(r['kind'],r['horizon_hours'],r['penalty']) for r in records}),'selected':len(selected),
        'combo':metrics,'elapsed_seconds':time.monotonic()-start}
    manifest={'status':'COMPLETED','mode':'aligned_crypto','lane':'complex_alpha','version':cfg['version'],
        'source_signature':bank['signature'],'source_files':bank['source_files']+funding_files,'code_sha256':code,
        'results':summary,'config':str(Path(config_path).resolve()),'wealth_convention':'native5 simple-return basket',
        'holdout_opened_after_freeze':s['audit_holdout']}
    _write_json(marker,manifest)
    from .complex_alpha_reporting import render_complex_report
    render_complex_report(out,Path(cfg['output']['visualization_dir']))
    return manifest
