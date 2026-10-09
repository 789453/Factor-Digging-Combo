"""One-entry aligned crypto research: discover, validate, freeze, then audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from ..crypto_fields import add_crypto_features, CRYPTO_FIELD_FORMULA_VERSION
from ..native5_fields import NATIVE5_FIELD_FORMULA_VERSION
from ..data import load_crypto_parquet
from ..evaluator import BatchEvaluator, make_panels
from ..ops import OPERATOR_SEMANTICS_VERSION
from .aligned_contract import (CONTRACT_VERSION, Split, completed_hour, align_hourly_to_5m,
                               return_targets, discovery_betas, labels, exact_pnl)
from .aligned_screen import (conditional_expression, screen_horizons, score_one,
                             diversity_pick, phase_rotating_coarse_mask)
from .aligned_combo import (fit_candidate, forecast, hourly_positions, utility_metrics,
                            hourly_pnl, scheduled_risk_budget)
from .templates import generate_expressions, load_template_families

RAW_BASELINE_TEMPLATE=Path('configs/research/templates_crypto_raw_field_baselines_v1.yaml')


def _fingerprint(path: Path) -> dict:
    s=path.stat()
    return {"path":str(path.resolve()),"size":s.st_size,"mtime_ns":s.st_mtime_ns}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def _validate(cfg: dict) -> None:
    need={"mode","version","data","split","search","selection","combo","output"}
    if 'native5' in cfg:need.add('native5')
    if set(cfg)!=need or cfg['mode']!='aligned_crypto':
        raise ValueError(f"aligned_crypto requires exactly {sorted(need)}")
    for part,keys in {
        'data':{'source','root','start','end','feature_cache'},
        'split':{'discovery_end','validation_start','validation_end','holdout_start'},
        'search':{'sources','fields','windows','horizons_5m','coarse_keep','medium_keep','checkpoint_batch','seed'},
        'selection':{'max_per_family','max_per_field','max_per_horizon','alpha_n','beta_n','risk_n'},
        'combo':{'cost_bps','alpha_budget','beta_budget','max_abs_position','risk_floor'},
        'output':{'out_dir'},
    }.items():
        if set(cfg[part])!=keys:
            raise ValueError(f"{part} needs exactly {sorted(keys)}; got {sorted(cfg[part])}")
    split=Split(**cfg['split']); split.masks(np.array([
        cfg['data']['start'].replace('-','').replace(' ','').replace(':','')[:12],
        split.validation_start,split.holdout_start]))
    if cfg['data']['source'] not in {'crypto_parquet','simulated'}:
        raise ValueError('aligned data source must be crypto_parquet or simulated')
    horizons=cfg['search']['horizons_5m']
    if not horizons or any(not isinstance(h,int) or h<12 or h%12 for h in horizons):
        raise ValueError('aligned horizons must be positive whole hours in 5m bars')
    sources=cfg['search']['sources']
    if not sources or any(set(s)!={'role','path','limit'} or s['role'] not in
        {'alpha','conditional_alpha','beta','risk'} or s['limit']<1 for s in sources):
        raise ValueError('invalid aligned search sources')
    if sum(s['limit'] for s in sources)<10000 and cfg['data']['source']=='crypto_parquet':
        raise ValueError('real aligned research requires >=10000 generated candidate budget')
    if not (0<cfg['search']['medium_keep']<=cfg['search']['coarse_keep']
            and cfg['search']['checkpoint_batch']>=20):
        raise ValueError('invalid coarse/medium/checkpoint budget')
    if any(cfg['selection'][k]<1 for k in cfg['selection']):
        raise ValueError('selection quotas must be positive')
    c=cfg['combo']
    if (c['cost_bps']<0 or not 0<c['alpha_budget']<=1 or not 0<=c['beta_budget']<=1
            or c['alpha_budget']+c['beta_budget']>c['max_abs_position']
            or not 0<c['risk_floor']<=1):
        raise ValueError('invalid combo costs or budgets')
    if not cfg['search']['fields'] or not cfg['search']['windows']:
        raise ValueError('aligned search fields and windows must be explicit')
    if 'native5' in cfg:
        n=cfg['native5'];nk={'enabled','template','fields','windows','limit',
             'coarse_keep','medium_keep','horizons_5m','selected_n','alpha_budget','feature_cache'}
        if (set(n)!=nk or n['enabled'] is not True or not n['fields'] or not n['windows']
                or not Path(n['template']).is_file() or n['limit']<1
                or not 0<n['medium_keep']<=n['coarse_keep']<=n['limit']
                or n['selected_n']<1 or not 0<n['alpha_budget']<c['alpha_budget']
                or any(not isinstance(h,int) or h<1 for h in n['horizons_5m'])):
            raise ValueError('invalid bounded native5 declaration')


def _records(cfg: dict):
    records=[]; roles={}; seen=set(); template_hashes=[]
    for source in cfg['search']['sources']:
        path=Path(source['path'])
        families,_=load_template_families(str(path))
        template_hashes.append(_fingerprint(path))
        limit=source['limit']
        sample=generate_expressions(
            cfg['search']['fields'],cfg['search']['windows'],families,
            max_expressions=limit*(3 if source['role']=='conditional_alpha' else 2),
            max_per_family=max(250,math.ceil(limit/max(1,len(families)))+200),seed=cfg['search']['seed'],
            diversity_share=.8)
        accepted=0
        for record in sample:
            cond=conditional_expression(record.expr)
            if source['role']=='conditional_alpha' and not cond:continue
            if record.expr_hash in seen:continue
            records.append(record); roles[record.expr_hash]=source['role']
            seen.add(record.expr_hash);accepted+=1
            if accepted>=limit:break
        if accepted!=limit:
            raise ValueError(f"{source['role']} generated {accepted}, requested {limit}; revise declared YAML budget")
    return records,roles,template_hashes


def _real_data(cfg:dict,fields:list[str]):
    root=Path(cfg['data']['root'])
    if not root.is_dir():raise FileNotFoundError(root)
    files=sorted(root.glob('*/*.parquet'))
    fingerprints=[_fingerprint(p) for p in files if p.name in {'1h.parquet','15m.parquet','5m.parquet'}]
    signature=_hash({'data':cfg['data'],'fingerprints':fingerprints,
                     'field_version':CRYPTO_FIELD_FORMULA_VERSION})[:20]
    cache=Path(cfg['data']['feature_cache'])/f'aligned_features_{signature}.parquet'
    if cache.exists():
        featured=pd.read_parquet(cache)
        print(f'[data] feature cache hit {cache}',flush=True)
    else:
        raw=load_crypto_parquet(str(root),cfg['data']['start'],cfg['data']['end'],'1h','15m','5m')
        featured=add_crypto_features(raw)
        cache.parent.mkdir(parents=True,exist_ok=True)
        temporary=cache.with_suffix('.tmp.parquet')
        featured.to_parquet(temporary,index=False,compression='zstd')
        temporary.replace(cache)
        print(f'[data] feature cache saved {cache}',flush=True)
    missing=sorted(set(fields)-set(featured))
    if missing:raise ValueError(f'configured fields unavailable: {missing}')
    panels,dates,codes=make_panels(featured,fields)
    fast_date=None;fast_close=[]
    for code in codes:
        path=root/code/'5m.parquet'
        if not path.exists():raise FileNotFoundError(path)
        frame=pq.read_table(path,columns=['date','close']).to_pandas()
        t=pd.to_datetime(frame.date,utc=True)
        available=(t+pd.Timedelta(minutes=5)).dt.strftime('%Y%m%d%H%M').to_numpy()
        good=(t>=pd.Timestamp(cfg['data']['start'],tz='UTC')) & (t<=pd.Timestamp(cfg['data']['end'],tz='UTC'))
        available=available[good]; close=frame.close.to_numpy(dtype=np.float32)[good]
        if fast_date is None:fast_date=available
        elif not np.array_equal(available,fast_date):raise ValueError(f'5m clock differs for {code}')
        fast_close.append(close)
    return panels,np.asarray(dates),codes,fast_date,np.column_stack(fast_close),fingerprints


def _simulated(cfg:dict,fields:list[str]):
    start=pd.Timestamp(cfg['data']['start'],tz='UTC')
    end=pd.Timestamp(cfg['data']['end'],tz='UTC')
    fast_dt=pd.date_range(start,end,freq='5min',tz='UTC')
    fast_date=(fast_dt+pd.Timedelta(minutes=5)).strftime('%Y%m%d%H%M').to_numpy()
    dates=pd.date_range(start,end,freq='1h',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    rng=np.random.default_rng(cfg['search']['seed']);nasset=12
    r=rng.normal(0,.001,size=(len(fast_date),nasset))
    close=100*np.exp(np.cumsum(r,axis=0)).astype(np.float32)
    hourly_idx=align_hourly_to_5m(completed_hour(dates),fast_date)
    panels={name:rng.normal(size=(len(dates),nasset)).astype(np.float32) for name in fields}
    panels['close']=close[hourly_idx]
    return panels,dates,[f'SIM{j}' for j in range(nasset)],fast_date,close,[{'simulated':True}]


def _risk_targets(fast_close:np.ndarray, start:np.ndarray, horizons:tuple[int,...],
                  periods:np.ndarray) -> dict[int,np.ndarray]:
    p=np.log(np.where(fast_close>0,fast_close,np.nan))
    r=np.diff(p,axis=0,prepend=np.nan)
    market=np.nanmean(r,axis=1)
    down=np.minimum(market,0)**2
    sums=np.r_[0,np.cumsum(np.nan_to_num(down))]
    counts=np.r_[0,np.cumsum(np.isfinite(down))]
    result={}
    for horizon in horizons:
        end=start+horizon
        valid=end<len(fast_close)
        valid &= periods[start]==periods[np.minimum(end,len(periods)-1)]
        valid &= (counts[np.minimum(end+1,len(counts)-1)]-counts[start+1])==horizon
        y=np.full(len(start),np.nan,dtype=np.float32)
        ii=np.flatnonzero(valid)
        y[ii]=(sums[end[ii]+1]-sums[start[ii]+1])*1e6
        result[horizon]=y
    return result


def _row(record,role:str,metric:dict,stage:str) -> dict:
    return {'expr_hash':record.expr_hash,'expr':record.expr,'role':role,
            'candidate_id':f"{record.expr_hash}:{metric.get('horizon_bars','invalid')}",
            'conditional':conditional_expression(record.expr),
            'template_name':record.template_name,'template_family':record.template_family,
            'fields':'|'.join(record.fields),'windows':'|'.join(map(str,record.windows)),
            'operators':'|'.join(record.operators),'nodes':record.nodes,
            'evaluation_stage':stage,**metric}


def _read_checkpoint(path:Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _write_checkpoint(path:Path,frame:pd.DataFrame):
    tmp=path.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp,index=False);tmp.replace(path)


def _role_diversity_pick(frame:pd.DataFrame,n:int,cfg:dict,multiplier:int=1,
                         stratify_horizon:bool=False)->pd.DataFrame:
    """Reserve survivor capacity by research purpose before comparing unlike ICs."""
    sources=cfg['search']['sources']; total=sum(x['limit'] for x in sources)
    budgets=[n*x['limit']//total for x in sources]
    for i in range(n-sum(budgets)):budgets[i%len(budgets)]+=1
    c=cfg['selection']; picks=[]
    for source,quota in zip(sources,budgets):
        if quota<1:continue
        subset=frame[frame.role.eq(source['role'])]
        if stratify_horizon:
            hvalues=cfg['search']['horizons_5m']
            hbudgets=[quota//len(hvalues)+(j<quota%len(hvalues)) for j in range(len(hvalues))]
            parts=[]
            for h,hquota in zip(hvalues,hbudgets):
                portion=diversity_pick(subset[subset.horizon_bars.eq(h)],hquota,
                    c['max_per_family']*multiplier,c['max_per_field']*multiplier,
                    c['max_per_horizon']*multiplier)
                if not portion.empty:parts.append(portion)
            part=pd.concat(parts,ignore_index=True) if parts else subset.iloc[:0].copy()
            if len(part)<quota:
                leftover=subset[~subset.candidate_id.isin(part.candidate_id)]
                fill=diversity_pick(leftover,quota-len(part),
                    c['max_per_family']*multiplier,c['max_per_field']*multiplier,
                    c['max_per_horizon']*multiplier)
                part=pd.concat([part,fill],ignore_index=True)
        else:
            part=diversity_pick(subset,quota,c['max_per_family']*multiplier,
                  c['max_per_field']*multiplier,c['max_per_horizon']*multiplier)
        if part.empty:raise ValueError(f"no valid {source['role']} candidate for reserved screen quota")
        picks.append(part)
    chosen=pd.concat(picks,ignore_index=True)
    if len(chosen)<n:
        spare=frame[~frame.candidate_id.isin(chosen.candidate_id)]
        more=diversity_pick(spare,n-len(chosen),c['max_per_family']*multiplier,
              c['max_per_field']*multiplier,c['max_per_horizon']*multiplier)
        chosen=pd.concat([chosen,more],ignore_index=True)
    return chosen


def _candidate_metrics(values, record,role, horizon, direction, target, raw_hour,
                       dates,masks,cost_bps, boundary,budget):
    conditional=conditional_expression(record.expr)
    model=fit_candidate(values,target,masks['discovery'],role,conditional,int(direction))
    pred=forecast(values,model)
    if role=='risk':
        valid=masks['validation'] & np.isfinite(target) & np.isfinite(pred)
        if valid.sum()<100:raise ValueError('too few risk validation observations')
        # Simple discovery-only constant benchmark; augmented signal must beat it.
        baseline=float(np.nanmean(target[masks['discovery']]))
        base_mse=float(np.mean((target[valid]-baseline)**2))
        full_mse=float(np.mean((target[valid]-(baseline+pred[valid]))**2))
        metrics={'validation_delta_r2':1-full_mse/base_mse,'validation_net_mean_bar_bps':np.nan}
        return metrics,model,pred
    if role=='beta':
        pred=np.broadcast_to(pred[:,None],raw_hour.shape)
    discovery_pred=pred[masks['discovery']]
    scale=max(float(np.quantile(np.abs(discovery_pred[np.isfinite(discovery_pred)]),.90)),1e-9)
    position=hourly_positions(pred,dates,horizon,scale,budget,role!='beta',cost_bps)
    # Same horizon, same positions, same fee as the future final portfolio.
    valid=masks['validation'] & np.isfinite(raw_hour).all(axis=1)
    metrics=utility_metrics(position,raw_hour,cost_bps,valid,boundary)
    metrics={'validation_'+k:v for k,v in metrics.items()}
    train_valid=masks['discovery'] & np.isfinite(raw_hour).all(axis=1)
    train=utility_metrics(position,raw_hour,cost_bps,train_valid,boundary)
    metrics['discovery_net_mean_bar_bps']=train['net_mean_bar_bps']
    if conditional:
        active=np.isfinite(values) & (values!=0)
        metrics['discovery_active_fraction']=float(active[masks['discovery']].mean())
        metrics['validation_active_fraction']=float(active[masks['validation']].mean())
    metrics['prediction_scale']=scale
    return metrics,{**model,'prediction_scale':scale},pred


def run_aligned_crypto(cfg:dict,config_path:str) -> dict:
    if 'complex_alpha' in cfg:
        from .complex_alpha_workflow import run_complex_alpha
        return run_complex_alpha(cfg, config_path)
    if 'manual_alpha' in cfg:
        from .manual_alpha_workflow import run_manual_alpha
        return run_manual_alpha(cfg, config_path)
    _validate(cfg)
    out=Path(cfg['output']['out_dir']);out.mkdir(parents=True,exist_ok=True)
    complete=out/'manifest.json'
    if complete.exists() and json.loads(complete.read_text()).get('status')=='COMPLETED':
        raise FileExistsError(f'completed research evidence exists: {out}')
    records,roles,templates=_records(cfg)
    record_map={r.expr_hash:r for r in records}
    fields=cfg['search']['fields']
    if cfg['data']['source']=='crypto_parquet':
        panels,raw_dates,codes,fast_dates,fast_close,datafiles=_real_data(cfg,fields)
    else:
        panels,raw_dates,codes,fast_dates,fast_close,datafiles=_simulated(cfg,fields)
    dates=completed_hour(raw_dates)
    split=Split(**cfg['split']);masks=split.masks(dates)
    fast_idx=align_hourly_to_5m(dates,fast_dates)
    fast_period=np.where(fast_dates<=split.discovery_end,0,
                 np.where(fast_dates<split.validation_start,-1,
                 np.where(fast_dates<=split.validation_end,1,
                 np.where(fast_dates<split.holdout_start,-1,2))))
    hour_period=fast_period[fast_idx]
    if np.any(hour_period<0):raise ValueError('configured data creates unlabeled split gap')
    log_fast=np.log(np.where(fast_close>0,fast_close,np.nan))
    horizons=tuple(cfg['search']['horizons_5m'])
    future_raw=return_targets(log_fast,fast_idx,horizons,fast_period)
    betas=discovery_betas(np.log(panels['close']),masks['discovery'])
    response={h:labels(y,betas) for h,y in future_raw.items()}
    risk=_risk_targets(fast_close,fast_idx,horizons,fast_period)
    # Native 5m accounting at the hour boundary: t+1 through t+12.
    one_hour=future_raw[12]
    no_future=np.isfinite(one_hour).all(axis=1)
    boundary=np.zeros(len(dates),dtype=bool)
    for period in ('discovery','validation','holdout'):
        idx=np.flatnonzero(masks[period])
        if len(idx):boundary[idx[0]]=True
    family_fields=set(fields);all_windows=set(cfg['search']['windows'])
    for record in records:
        family_fields.update(record.fields);all_windows.update(record.windows)
    evaluator=BatchEvaluator({name:panels[name] for name in family_fields},raw_dates,codes,
              windows=tuple(sorted(all_windows)),max_depth=12,max_nodes=32,
              max_ts_ops=8,max_pair_ops=3,max_binary_ops=8,max_cache_items=64)
    signature=_hash({'config':cfg,'templates':templates,'data':datafiles,
                     'contract':CONTRACT_VERSION,'operators':OPERATOR_SEMANTICS_VERSION,
                     'fields':CRYPTO_FIELD_FORMULA_VERSION,
                     'native_fields':NATIVE5_FIELD_FORMULA_VERSION if 'native5' in cfg else None,
                     'native_template':hashlib.sha256(Path(cfg['native5']['template']).read_bytes()).hexdigest()
                         if 'native5' in cfg else None,
                     'raw_baseline_template':hashlib.sha256(RAW_BASELINE_TEMPLATE.read_bytes()).hexdigest(),
                     'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        Path(__file__).parent.glob('aligned_*.py')},
                     'native_code':hashlib.sha256((Path(__file__).parent/'native5_aligned.py').read_bytes()).hexdigest()
                         if 'native5' in cfg else None})
    state=out/'checkpoint_signature.json'
    if state.exists() and json.loads(state.read_text())['signature']!=signature:
        raise ValueError('checkpoint signature differs from current config, data or code contract')
    if not state.exists():state.write_text(json.dumps({'signature':signature},indent=2))
    coarse_path=out/'coarse_checkpoint.parquet'
    existing=_read_checkpoint(coarse_path)
    rows=existing.to_dict('records');done=set(existing.expr_hash) if not existing.empty else set()
    coarse_mask=phase_rotating_coarse_mask(masks['discovery'],24,6)
    by_role={'alpha':{h:response[h]['relative'] for h in horizons},
             'conditional_alpha':{h:response[h]['relative'] for h in horizons},
             'beta':{h:response[h]['market'] for h in horizons},'risk':risk}
    start=time.perf_counter();batch=cfg['search']['checkpoint_batch']
    for i,record in enumerate(records,1):
        if record.expr_hash in done:continue
        value,status=evaluator.eval_expr(record.expr)
        if value is None:metric={'status':status,'coarse_score':np.nan}
        else:metric=screen_horizons(value,by_role[roles[record.expr_hash]],coarse_mask,
                          'alpha' if roles[record.expr_hash]=='conditional_alpha' else roles[record.expr_hash],
                          conditional_expression(record.expr))
        rows.append(_row(record,roles[record.expr_hash],metric,'coarse'))
        if len(rows)%batch==0:
            _write_checkpoint(coarse_path,pd.DataFrame(rows))
        if i%250==0 or i==len(records):
            print(f'[aligned coarse] {i}/{len(records)} done; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    _write_checkpoint(coarse_path,pd.DataFrame(rows))
    coarse=pd.DataFrame(rows)
    c=cfg['selection']; s=cfg['search']
    coarse_pick=_role_diversity_pick(coarse,s['coarse_keep'],cfg,5)
    medium_rows=[]
    for i,row in enumerate(coarse_pick.itertuples(),1):
        record=record_map[row.expr_hash];value,status=evaluator.eval_expr(record.expr)
        if value is None:continue
        for h in horizons:
            metric=score_one(value,by_role[row.role][h],masks['discovery'],
                  'alpha' if row.role=='conditional_alpha' else row.role,row.conditional)
            medium_rows.append(_row(record,row.role,{**metric,'horizon_bars':h},'medium'))
        if i%100==0:print(f'[aligned medium] {i}/{len(coarse_pick)}',flush=True)
    medium=pd.DataFrame(medium_rows)
    medium_pick=_role_diversity_pick(medium,s['medium_keep'],cfg,stratify_horizon=True)
    fine=[];models={}
    for i,row in enumerate(medium_pick.itertuples(),1):
        record=record_map[row.expr_hash]
        value,status=evaluator.eval_expr(record.expr)
        role='alpha' if row.role=='conditional_alpha' else row.role
        target=(risk[row.horizon_bars] if role=='risk' else
                response[row.horizon_bars]['relative' if role=='alpha' else 'market'])
        try:
            metric,model,_=_candidate_metrics(value,record,role,int(row.horizon_bars),
                           int(row.direction),target,one_hour,dates,masks,
                           cfg['combo']['cost_bps'],boundary,
                           cfg['combo']['beta_budget' if role=='beta' else 'alpha_budget'])
            metric['status']='OK';models[row.candidate_id]=model
        except (ValueError,FloatingPointError) as exc:
            metric={'status':'CALIBRATION_FAILED','error':str(exc)}
        fine.append({**row._asdict(),**metric,'evaluation_stage':'fine'})
        if i%50==0:print(f'[aligned fine] {i}/{len(medium_pick)}',flush=True)
    # Raw fields are both controls and eligible candidates. Excluding a field
    # that beats the complex search would violate the shared research target.
    raw_families,_=load_template_families(str(RAW_BASELINE_TEMPLATE))
    if len(raw_families)!=1 or set(raw_families[0].fields_a)!=set(fields):
        raise ValueError('raw field baseline YAML must declare exactly the configured fields')
    raw_records=generate_expressions(fields,s['windows'],raw_families,
        max_expressions=len(fields),max_per_family=len(fields),
        seed=s['seed'],diversity_share=.8)
    if len(raw_records)!=len(fields):
        raise ValueError('raw field baseline YAML did not generate every configured field')
    baseline_rows=[]
    for base_record in raw_records:
        field=base_record.fields[0]
        value=panels[field]
        for role in ('alpha','beta','risk'):
            record=SimpleNamespace(**{**base_record.__dict__,
                'expr_hash':f'raw:{role}:{base_record.expr_hash}'})
            record_map[record.expr_hash]=record
            for h in horizons:
                metric=score_one(value,by_role[role][h],masks['discovery'],role,False)
                if metric['status']!='OK':continue
                target=risk[h] if role=='risk' else response[h]['relative' if role=='alpha' else 'market']
                try:
                    valid,model,_=_candidate_metrics(value,record,role,h,int(metric['direction']),
                        target,one_hour,dates,masks,cfg['combo']['cost_bps'],boundary,
                        cfg['combo']['beta_budget' if role=='beta' else 'alpha_budget'])
                    baseline_rows.append({'field':field,'role':role,'horizon_bars':h,
                                          'discovery_ic':metric['raw_ic'],**valid})
                    row_metric={**metric,'horizon_bars':h,**valid,'status':'OK'}
                    row=_row(record,role,row_metric,'raw_baseline')
                    fine.append(row)
                    models[row['candidate_id']]=model
                except (ValueError,FloatingPointError):continue
    fine=pd.DataFrame(fine)
    pd.DataFrame(baseline_rows).to_csv(out/'field_baselines.csv',index=False)
    viable=fine[fine.status.eq('OK')].copy()
    selected=[]
    for role,n in [('alpha',c['alpha_n']),('beta',c['beta_n']),('risk',c['risk_n'])]:
        family_role=viable[viable.role.isin(['alpha','conditional_alpha'])] if role=='alpha' else viable[viable.role.eq(role)]
        key='validation_delta_r2' if role=='risk' else 'validation_net_mean_bar_bps'
        ranked=family_role.sort_values([key,'candidate_id'],ascending=[False,True]).drop_duplicates('expr_hash')
        profitable=ranked[ranked[key]>0]
        trade_enabled=not profitable.empty
        if trade_enabled:ranked=profitable
        else:ranked=ranked.head(1)  # retain an explicitly inactive research diagnostic
        family_count={};kept=[];family_cap=max(2,n//3)
        for candidate in ranked.itertuples():
            family=candidate.template_name
            if family_count.get(family,0)>=family_cap:continue
            kept.append(candidate.Index);family_count[family]=family_count.get(family,0)+1
            if len(kept)==n:break
        if len(kept)<n:
            for candidate in ranked.itertuples():
                if candidate.Index not in kept:kept.append(candidate.Index)
                if len(kept)==n:break
        chosen=ranked.loc[kept]
        if len(chosen)==0:raise ValueError(f'no viable {role} candidates after fine screen')
        chosen=chosen.copy()
        chosen['validation_trade_enabled']=trade_enabled
        selected.append(chosen)
    selected=pd.concat(selected,ignore_index=True)
    coarse.to_csv(out/'search_results.csv',index=False)
    medium.to_csv(out/'medium_results.csv',index=False)
    fine.to_csv(out/'fine_results.csv',index=False)
    selected.to_csv(out/'selected_factors.csv',index=False)
    chosen_models={h:models[h] for h in selected.candidate_id}
    # Validation judges whether the risk forecast improves the actual net strategy.
    sleeves={'alpha':[],'beta':[]}; risk_predictions=[]
    for row in selected.itertuples():
        value,status=evaluator.eval_expr(record_map[row.expr_hash].expr)
        if status!='OK' or value is None:raise ValueError(f'frozen reevaluation failed: {row.expr_hash}')
        pred=forecast(value,chosen_models[row.candidate_id])
        if row.role=='risk':
            if float(row.validation_delta_r2)>0:
                baseline=float(np.nanmean(risk[int(row.horizon_bars)][masks['discovery']]))
                risk_predictions.append(baseline+pred)
            continue
        if not row.validation_trade_enabled:continue
        if row.role=='beta':pred=np.broadcast_to(pred[:,None],one_hour.shape)
        budget=cfg['combo']['beta_budget' if row.role=='beta' else 'alpha_budget']
        sleeves['beta' if row.role=='beta' else 'alpha'].append(
            hourly_positions(pred,dates,int(row.horizon_bars),
                             float(chosen_models[row.candidate_id]['prediction_scale']),
                             budget,row.role!='beta',cfg['combo']['cost_bps']))
    alpha_hour=np.mean(np.stack(sleeves['alpha']),axis=0) if sleeves['alpha'] else np.zeros_like(one_hour)
    beta_hour=np.mean(np.stack(sleeves['beta']),axis=0) if sleeves['beta'] else np.zeros_like(one_hour)
    base_pos=alpha_hour+beta_hour
    validation_mask=masks['validation'] & no_future
    base_val=utility_metrics(base_pos,one_hour,cfg['combo']['cost_bps'],
                             validation_mask,boundary)['net_mean_bar_bps']
    hour_at_bar=np.searchsorted(dates,fast_dates,side='right')-1
    active_bar=hour_at_bar>=0;hour_at_bar=np.clip(hour_at_bar,0,len(dates)-1)
    next_fast=np.vstack([np.diff(log_fast,axis=0),np.full((1,fast_close.shape[1]),np.nan)])
    fast_validation=(fast_period==1) & np.r_[fast_period[1:]==1,False] & np.isfinite(next_fast).all(axis=1)
    fast_boundary=np.zeros(len(fast_dates),dtype=bool)
    for period_code in (0,1,2):
        where=np.flatnonzero(fast_period==period_code)
        if len(where):fast_boundary[where[0]]=True
    def fast_net_mean(position):
        ledger=exact_pnl(position,next_fast,cfg['combo']['cost_bps'],fast_boundary)
        return float(np.mean(ledger['net'][fast_validation])*1e4)
    base_fast=base_pos[hour_at_bar].copy();base_fast[~active_bar]=0
    base_fast_val=fast_net_mean(base_fast)
    native_result=None;native_position=None;native_enabled=False;native_val=np.nan
    current_fast=base_fast
    if 'native5' in cfg:
        from .native5_aligned import run_native5_stage
        native_result=run_native5_stage(cfg,out,fast_dates,fast_close,codes)
        native_position=native_result['position']
        keep=(cfg['combo']['alpha_budget']-cfg['native5']['alpha_budget'])/cfg['combo']['alpha_budget']
        candidate_fast=(alpha_hour*keep+beta_hour)[hour_at_bar].copy()
        candidate_fast[~active_bar]=0
        candidate_fast+=native_position
        native_val=fast_net_mean(candidate_fast)
        native_enabled=bool(native_val>base_fast_val)
        if native_enabled:current_fast=candidate_fast
    risk_val=np.nan; risk_enabled=False
    if risk_predictions:
        risk_budget=scheduled_risk_budget(np.mean(np.stack(risk_predictions),axis=0),
                                          dates,masks['discovery'],cfg['combo']['risk_floor'])
        risk_mult=risk_budget[hour_at_bar].copy();risk_mult[~active_bar]=1
        risk_val=fast_net_mean(current_fast*risk_mult[:,None])
        risk_enabled=bool(risk_val>fast_net_mean(current_fast))
    freeze={'status':'FROZEN_BEFORE_HOLDOUT','signature':signature,
            'selected_candidates':selected.candidate_id.tolist(), 'models':chosen_models,
            'native_enabled':native_enabled,
            'native_selected_candidates':native_result['selected'].candidate_id.tolist()
                 if native_result is not None else [],
            'native_models':native_result['models'] if native_result is not None else {},
            'validation_native_base_net_mean_5m_bps':base_fast_val,
            'validation_native_augmented_net_mean_5m_bps':native_val,
            'risk_enabled':risk_enabled,
            'validation_base_net_mean_hour_bps':base_val,
            'validation_risk_net_mean_5m_bps':risk_val,
            'test_start':split.holdout_start,'direction_source':'discovery only',
            'selection_source':'discovery and validation only'}
    (out/'research_freeze.json').write_text(json.dumps(freeze,indent=2,default=float))
    print('[aligned] selection frozen before holdout audit',flush=True)
    # All following accesses to holdout labels are downstream of the freeze file.
    from .aligned_reporting import finish_aligned_report
    report=finish_aligned_report(out,cfg,selected,chosen_models,record_map,evaluator,
            dates,fast_dates,fast_idx,fast_close,one_hour,risk,masks,boundary,
            native_position=native_position)
    manifest={'status':'COMPLETED','mode':'aligned_crypto','config_path':config_path,
              'signature':signature,'contract_version':CONTRACT_VERSION,
              'data_files':datafiles,'template_files':templates,
              'actual_data_end':str(fast_dates[-1]),'codes':codes,
              'counts':{'coarse':len(coarse),'medium':len(medium),'fine':len(fine),
                        'selected':len(selected),'native5':native_result['counts']
                            if native_result is not None else None},'results':report}
    complete.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    return manifest
