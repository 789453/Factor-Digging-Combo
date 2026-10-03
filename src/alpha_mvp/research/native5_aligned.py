"""Bounded native-5m lane using the same target, fee and freeze contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..data import load_crypto_native_5m
from ..native5_fields import add_crypto_native_5m_features, NATIVE5_FIELD_FORMULA_VERSION
from ..evaluator import BatchEvaluator, make_panels
from .aligned_combo import fit_candidate, forecast
from .aligned_contract import Split, discovery_betas, exact_pnl, labels, return_targets, positions_from_score
from .aligned_screen import (conditional_expression, diversity_pick, score_one,
                             screen_horizons, phase_rotating_coarse_mask)
from .templates import generate_expressions, load_template_families


def fixed_cadence_positions(prediction:np.ndarray,horizon:int,scale:float,
                            budget:float,cost_bps:float)->np.ndarray:
    """Portfolio-level cost gate on completed 5m bars, preserving neutrality."""
    if horizon<1 or scale<=0 or budget<0 or cost_bps<0:
        raise ValueError('invalid native 5m cadence or position inputs')
    proposal=positions_from_score(prediction,scale,budget,True)
    out=np.empty_like(proposal)
    fee=cost_bps/1e4
    for start in range(0,len(prediction),horizon):
        new=proposal[start]
        gain=float(np.mean(new*prediction[start]))
        round_trip=2*fee*float(np.mean(np.abs(new)))
        out[start:start+horizon]=new if gain>=round_trip else 0
    return out


def _native_panels(cfg:dict,fast_dates:np.ndarray,fast_close:np.ndarray,codes:list[str]):
    ncfg=cfg['native5']; fields=ncfg['fields']
    if cfg['data']['source']=='simulated':
        rng=np.random.default_rng(cfg['search']['seed']+71)
        ret=np.diff(np.log(fast_close),axis=0,prepend=np.nan)
        panels={field:rng.normal(size=fast_close.shape).astype(np.float32) for field in fields}
        for key in ('ret_5m','ret_15m','ret_1h'):
            if key in panels:
                h={'ret_5m':1,'ret_15m':3,'ret_1h':12}[key]
                panels[key]=np.log(fast_close/np.roll(fast_close,h,axis=0)).astype(np.float32)
                panels[key][:h]=np.nan
        panels['close']=fast_close
        return panels,[{'simulated_native5':True}]
    root=Path(cfg['data']['root']); cache_root=Path(ncfg['feature_cache'])
    files=[root/code/f'{freq}.parquet' for code in codes for freq in ('5m','15m')]
    inputs=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    key=hashlib.sha256(json.dumps([cfg['data']['start'],cfg['data']['end'],fields,
          NATIVE5_FIELD_FORMULA_VERSION,inputs],default=str).encode()).hexdigest()[:20]
    cache=cache_root/f'native5_aligned_{key}.parquet'
    if cache.exists():
        frame=pd.read_parquet(cache)
        print(f'[native5] feature cache hit {cache}',flush=True)
    else:
        raw=load_crypto_native_5m(str(root),cfg['data']['start'],cfg['data']['end'])
        frame=add_crypto_native_5m_features(raw)
        missing=sorted(set(fields)-set(frame))
        if missing:raise ValueError(f'native5 fields unavailable: {missing}')
        frame=frame[['trade_date','ts_code','close',*fields]]
        cache.parent.mkdir(parents=True,exist_ok=True)
        temp=cache.with_suffix('.tmp.parquet')
        frame.to_parquet(temp,index=False,compression='zstd');temp.replace(cache)
        print(f'[native5] feature cache saved {cache}',flush=True)
    panels,dates,found=make_panels(frame,fields)
    if list(found)!=list(codes) or not np.array_equal(dates,fast_dates):
        raise ValueError('native5 features must exactly match the 5m execution clock and assets')
    return panels,[{'cache':str(cache),'inputs':inputs}]


def _horizon_pick(frame:pd.DataFrame,n:int,horizons:list[int],family:int,field:int)->pd.DataFrame:
    pieces=[]
    for i,h in enumerate(horizons):
        quota=n//len(horizons)+(i<n%len(horizons))
        part=diversity_pick(frame[frame.horizon_bars.eq(h)],quota,family,field,n)
        if not part.empty:pieces.append(part)
    chosen=pd.concat(pieces,ignore_index=True) if pieces else frame.iloc[:0].copy()
    if len(chosen)<n:
        leftover=frame[~frame.candidate_id.isin(chosen.candidate_id)]
        fill=diversity_pick(leftover,n-len(chosen),family,field,n)
        chosen=pd.concat([chosen,fill],ignore_index=True)
    return chosen


def run_native5_stage(cfg:dict,out:Path,fast_dates:np.ndarray,
                      fast_close:np.ndarray,codes:list[str]) -> dict:
    """All ranking and calibration use discovery/validation only. No holdout metric read."""
    ncfg=cfg['native5'];split=Split(**cfg['split'])
    panels,datafiles=_native_panels(cfg,fast_dates,fast_close,codes)
    masks=split.masks(fast_dates)
    period=np.where(masks['discovery'],0,np.where(masks['validation'],1,2))
    log_close=np.log(np.where(fast_close>0,fast_close,np.nan))
    horizons=tuple(ncfg['horizons_5m'])
    raw=return_targets(log_close,np.arange(len(fast_dates)),horizons,period)
    betas=discovery_betas(log_close,masks['discovery'])
    targets={h:labels(raw[h],betas)['relative'] for h in horizons}
    families,_=load_template_families(ncfg['template'])
    records=generate_expressions(ncfg['fields'],ncfg['windows'],families,
          max_expressions=ncfg['limit'],max_per_family=max(100,ncfg['limit']//len(families)+100),
          seed=cfg['search']['seed']+71,diversity_share=.8)
    if len(records)!=ncfg['limit']:
        raise ValueError(f'native5 generated {len(records)}, requested {ncfg["limit"]}')
    record_map={r.expr_hash:r for r in records}
    end_discovery=np.flatnonzero(masks['discovery'])[-1]+1
    def evaluator(limit):
        return BatchEvaluator({name:panel[:limit] for name,panel in panels.items()},
            fast_dates[:limit],codes,windows=tuple(ncfg['windows']),
            max_depth=12,max_nodes=32,max_ts_ops=8,max_pair_ops=3,
            max_binary_ops=8,max_cache_items=24)
    research=evaluator(end_discovery)
    full=evaluator(len(fast_dates))
    coarse_path=out/'native5_coarse_checkpoint.parquet'
    if coarse_path.exists():
        prev=pd.read_parquet(coarse_path);rows=prev.to_dict('records');done=set(prev.expr_hash)
    else:rows=[];done=set()
    coarse_mask=phase_rotating_coarse_mask(masks['discovery'][:end_discovery],288,12)
    start=time.perf_counter()
    for i,record in enumerate(records,1):
        if record.expr_hash in done:continue
        value,status=research.eval_expr(record.expr)
        if value is None:
            metric={'status':status,'coarse_score':np.nan}
        else:
            metric=screen_horizons(value,{h:y[:end_discovery] for h,y in targets.items()},
                  coarse_mask,'alpha',conditional_expression(record.expr))
        rows.append({'candidate_id':f"{record.expr_hash}:{metric.get('horizon_bars','invalid')}",
                     'expr_hash':record.expr_hash,'expr':record.expr,
                     'template_name':record.template_name,'template_family':record.template_family,
                     'fields':'|'.join(record.fields),'windows':'|'.join(map(str,record.windows)),
                     **metric})
        if len(rows)%50==0:
            temp=coarse_path.with_suffix('.tmp.parquet')
            pd.DataFrame(rows).to_parquet(temp,index=False);temp.replace(coarse_path)
        if i%250==0:print(f'[native5 coarse] {i}/{len(records)} {time.perf_counter()-start:.0f}s',flush=True)
    coarse=pd.DataFrame(rows)
    temp=coarse_path.with_suffix('.tmp.parquet');coarse.to_parquet(temp,index=False);temp.replace(coarse_path)
    picked=diversity_pick(coarse,ncfg['coarse_keep'],ncfg['coarse_keep']//3,
                           ncfg['coarse_keep']//4,ncfg['coarse_keep'])
    medium=[]
    for i,row in enumerate(picked.itertuples(),1):
        value,status=research.eval_expr(record_map[row.expr_hash].expr)
        if value is None:continue
        for h in horizons:
            metric=score_one(value,targets[h][:end_discovery],masks['discovery'][:end_discovery],
                      'alpha',conditional_expression(row.expr))
            medium.append({**row._asdict(),'candidate_id':f'{row.expr_hash}:{h}',
                           'horizon_bars':h,**metric})
        if i%50==0:print(f'[native5 medium] {i}/{len(picked)}',flush=True)
    medium=pd.DataFrame(medium)
    fine_pick=_horizon_pick(medium,ncfg['medium_keep'],ncfg['horizons_5m'],
                            max(5,ncfg['medium_keep']//3),max(5,ncfg['medium_keep']//4))
    next_bar=np.vstack([np.diff(log_close,axis=0),np.full((1,log_close.shape[1]),np.nan)])
    boundary=np.zeros(len(fast_dates),dtype=bool)
    for name in ('discovery','validation','holdout'):
        idx=np.flatnonzero(masks[name]);boundary[idx[0]]=True
    fine=[];models={}
    for i,row in enumerate(fine_pick.itertuples(),1):
        value,status=full.eval_expr(record_map[row.expr_hash].expr)
        try:
            model=fit_candidate(value,targets[int(row.horizon_bars)],masks['discovery'],
                                'alpha',conditional_expression(row.expr),int(row.direction))
            pred=forecast(value,model)
            sample=pred[masks['discovery']]
            scale=max(float(np.quantile(np.abs(sample[np.isfinite(sample)]),.90)),1e-9)
            position=fixed_cadence_positions(pred,int(row.horizon_bars),scale,
                                ncfg['alpha_budget'],cfg['combo']['cost_bps'])
            ledger=exact_pnl(position,next_bar,cfg['combo']['cost_bps'],boundary)
            valid=masks['validation'] & np.r_[masks['validation'][1:],False] & np.isfinite(next_bar).all(axis=1)
            gross=float(np.mean(ledger['gross'][valid]));fee=float(np.mean(ledger['fee'][valid]))
            metric={'status':'OK','validation_gross_mean_5m_bps':gross*1e4,
                    'validation_fee_mean_5m_bps':fee*1e4,
                    'validation_net_mean_5m_bps':(gross-fee)*1e4,
                    'prediction_scale':scale}
            models[row.candidate_id]={**model,'prediction_scale':scale}
        except (ValueError,FloatingPointError,TypeError) as exc:
            metric={'status':'CALIBRATION_FAILED','error':str(exc)}
        fine.append({**row._asdict(),**metric})
        if i%10==0:print(f'[native5 fine] {i}/{len(fine_pick)}',flush=True)
    fine=pd.DataFrame(fine)
    viable=fine[fine.status.eq('OK')].sort_values(
        ['validation_net_mean_5m_bps','candidate_id'],ascending=[False,True]).drop_duplicates('expr_hash')
    selected=viable.head(ncfg['selected_n'])
    if selected.empty:raise ValueError('no native5 candidate survived fine evaluation')
    positions=[]
    for row in selected.itertuples():
        value,status=full.eval_expr(record_map[row.expr_hash].expr)
        pred=forecast(value,models[row.candidate_id])
        positions.append(fixed_cadence_positions(pred,int(row.horizon_bars),
            models[row.candidate_id]['prediction_scale'],ncfg['alpha_budget'],
            cfg['combo']['cost_bps']))
    native_pos=np.mean(np.stack(positions),axis=0)
    coarse.to_csv(out/'native5_search.csv',index=False)
    medium.to_csv(out/'native5_medium.csv',index=False)
    fine.to_csv(out/'native5_fine.csv',index=False)
    selected.to_csv(out/'native5_selected.csv',index=False)
    return {'position':native_pos,'selected':selected,'models':{x:models[x] for x in selected.candidate_id},
            'counts':{'coarse':len(coarse),'medium':len(medium),'fine':len(fine),
                      'selected':len(selected)},'data_files':datafiles}
