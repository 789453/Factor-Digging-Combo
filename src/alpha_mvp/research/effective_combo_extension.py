"""Discovery-only pool expansion and matured prediction combination (R2)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .aligned_contract import Split
from .complex_alpha_prediction import DISCOVERY_COLUMNS, discovery_archive, shortlist
from .complex_alpha_search import Candidate, evaluate_candidate
from .complex_alpha_evaluation import targets_from_prices
from .effective_combo import mature_window, fit_model


def source_metadata(source):
    source=Path(source)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    mining=Path(cfg['complex_alpha']['source_run'])
    prior=yaml.safe_load((mining/'config.yaml').read_text(encoding='utf-8'))
    return source,mining,prior,load(source/'numeric_pool_contract.json'),cfg


def reconstruct(source, end, extra_ids=None, shortlist_count=256):
    """Slice the requested period physically before feature/price reconstruction.

    Formula and initial-coordinate scalers come from immutable R5 evidence.
    Candidate ranking consumes the discovery column allowlist only.
    """
    from .complex_alpha_workflow import _funding
    source,mining,prior,contract,cfg=source_metadata(source)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    s=cfg['complex_alpha']; root=Path(s['bank_manifest']).parent; split=Split(**prior['split'])
    cards=load(mining/'field_registry.json')['coordinates']
    definitions={d['id']:Candidate(**d) for d in load(mining/'candidate_registry.json')['definitions']}
    ids=sum([contract['feature_order'][name] for name in ['old_mined','mixed_mined']],[])
    original_count=len(ids)
    if extra_ids is None:
        archive=discovery_archive(pd.read_csv(mining/'coarse_results.csv',usecols=DISCOVERY_COLUMNS))
        frames={k:shortlist(archive,s['horizon_hours'],k,shortlist_count) for k in ['old','mixed']}
        rows=pd.concat([v.assign(stream=k) for k,v in frames.items()],ignore_index=True).drop_duplicates('id')
        ids+= [v for v in rows.id if v not in ids]
    else:
        ids+=list(extra_ids); rows=None
    old_indices=[i for i,c in enumerate(cards) if c['family']=='OLD' and c.get('admitted',True)]
    needed=sorted(set(old_indices).union(*(set(definitions[k].left+definitions[k].right) for k in ids)))
    remap={v:k for k,v in enumerate(needed)}
    all_dates=np.load(root/'dates.npy'); full=all_dates[:np.searchsorted(all_dates,end,side='right')]
    stamp=pd.to_datetime(full,format='%Y%m%d%H%M',utc=True)
    sampled=np.flatnonzero((stamp.hour*60+stamp.minute)%s['sample_minutes']==0); dates=full[sampled]
    all_fast=np.load(root/'fast_dates.npy'); n=np.searchsorted(all_fast,end,side='right'); fast=all_fast[:n]
    close=np.load(root/'fast_close.npy',mmap_mode='r')[:n].copy()
    scaler=load(mining/'initial_coordinate_scaler.json'); base_count=len(scaler[0]['mean'])
    original=[i for i in needed if i<base_count]
    x=np.empty((len(dates),len(contract['assets']),len(needed)),np.float32)
    for j,code in enumerate(contract['assets']):
        cached=np.load(root/f'{code}.npy',mmap_mode='r')
        if cached.shape!=(len(all_dates),base_count) or scaler[j]['asset']!=code:raise ValueError('source axes changed')
        center=np.asarray(scaler[j]['mean'])[original]; scale=np.asarray(scaler[j]['scale'])[original]
        values=np.clip(np.nan_to_num((cached[np.ix_(sampled,original)]-center)/scale,nan=0.),-8,8)
        values[:,~np.asarray(scaler[j]['valid'])[original]]=0
        x[:,j,[remap[i] for i in original]]=values
    funding,files,extra=_funding(prior,full,fast,close)
    if files!=load(mining/'input_signature.json')['derivative_files']:raise ValueError('funding source changed')
    for name,value in {'clock_sin':np.sin(2*np.pi*(stamp.hour+stamp.minute/60)/24),
        'clock_cos':np.cos(2*np.pi*(stamp.hour+stamp.minute/60)/24),
        'weekday_sin':np.sin(2*np.pi*stamp.dayofweek/7),'weekday_cos':np.cos(2*np.pi*stamp.dayofweek/7)}.items():
        extra[name]=np.repeat(np.asarray(value)[:,None],len(contract['assets']),axis=1)
    calibration=full<=prior['complex_alpha']['feature_calibration_end']
    for i in needed:
        if i<base_count:continue
        values=extra[cards[i]['name']]; center=np.nanmean(values[calibration],axis=0); scale=np.nanstd(values[calibration],axis=0)
        valid=np.isfinite(scale)&(scale>1e-12); center[~valid]=0; scale[~valid]=1
        v=np.clip(np.nan_to_num((values[sampled]-center)/scale,nan=0.),-8,8);v[:,~valid]=0;x[...,remap[i]]=v
    market=x[...,[remap[i] for i in old_indices]]
    features=[]
    for identifier in ids:
        c=definitions[identifier]
        mapped=Candidate(c.id,c.family,c.mode,tuple(remap[i] for i in c.left),c.left_weights,
            tuple(remap[i] for i in c.right),c.right_weights,c.support_minutes,c.nonlinear_order,c.nodes)
        features.append(evaluate_candidate(x,mapped).astype(np.float32))
    formula=np.stack(features,axis=-1)
    base=np.concatenate([market,formula[...,:original_count]],axis=-1)
    # Each old source period is byte-for-byte reconstructed before new OOS is admitted.
    source_dates=np.load(source/'dates.npy'); source_y=np.load(source/'raw_return_target.npy')
    count=min(len(dates),len(source_dates))
    reference=np.concatenate([np.load(source/(k+'_features.npy')) for k in ['old_coordinates','old_mined','mixed_mined']],axis=-1)
    if not np.array_equal(dates[:count],source_dates[:count]) or not np.array_equal(base[:count],reference[:count]):
        raise ValueError('original numerical pool replay failed')
    beta=np.zeros_like(base[...,0])
    raw,_=targets_from_prices(dates,fast,close,[s['horizon_hours']],split,beta,funding,
        include_holdout=end>split.validation_end,return_contract='native_simple')
    y=raw[s['horizon_hours']]
    if not np.allclose(y[:count],source_y[:count],atol=1e-12,rtol=0,equal_nan=True):raise ValueError('original target replay failed')
    return dict(base=base,formulas=formula,formula_ids=ids,rows=rows,dates=dates,y=y,fast=fast,close=close,
        funding=funding,split=split,contract=contract,folds=s['folds'],prior=prior,horizon=s['horizon_hours'],original_count=original_count)


def expand_pool(formulas, ids, rows, train, additions, correlation_limit, original_count=96):
    """Keep the original prefix; global discovery aliases cannot consume new slots.

    New core/conditional quotas are equal in each stream. High correlation among
    existing columns never removes frozen inputs. No outcome/fee filter is used.
    """
    if additions%2 or not 0<correlation_limit<=1 or len(ids)!=formulas.shape[-1]:raise ValueError('invalid expansion contract')
    sample=formulas[train].reshape(-1,len(ids)).astype(float);sample-=sample.mean(axis=0)
    norms=np.sqrt(np.sum(sample*sample,axis=0));unit=sample/np.maximum(norms,1e-12);similarity=unit.T@unit
    kept=list(range(original_count));lookup={v:k for k,v in enumerate(ids)};records=[]
    counters={(stream,tier):0 for stream in ['old','mixed'] for tier in ['predictive_candidate','conditional_or_unstable']}
    # Alternate tiers while retaining shortlist family order.
    ordered=[]
    groups=[g for _,g in rows.groupby(['stream','research_tier'],sort=True)]
    for rank in range(max(map(len,groups))):
        ordered.extend(g.iloc[rank] for g in groups if rank<len(g))
    for row in ordered:
        k=lookup[row.id];bucket=(row.stream,row.research_tier)
        if k<original_count:continue
        alias=next((j for j in kept if abs(similarity[k,j])>=correlation_limit),None)
        reason='constant' if norms[k]<1e-8 else 'alias' if alias is not None else 'quota_archive' if counters[bucket]>=additions//2 else 'retained'
        record=row.to_dict();record.update(reason=reason,representative=ids[alias] if alias is not None else '',
            max_abs_correlation=float(np.max(np.abs(similarity[k,kept]))))
        records.append(record)
        if reason=='retained':kept.append(k);counters[bucket]+=1
    if any(v!=additions//2 for v in counters.values()):raise ValueError('discovery diversity quotas unmet: '+str(counters))
    return [ids[i] for i in kept[original_count:]],pd.DataFrame(records),similarity[np.ix_(kept,kept)]


def shrunk_error_weights(prediction,target,diagonal_shrink,equal_shrink,minimum):
    """Convex error covariance blend; units remain original raw-return units.

    Finite complete historical rows only. Missing observations below the declared
    minimum use the declared equal-weight warm-up, never in-sample forecasts.
    """
    from scipy.optimize import minimize
    k=prediction.shape[-1];equal=np.ones(k)/k
    p=prediction.reshape(-1,k);y=target.ravel();good=np.isfinite(p).all(axis=1)&np.isfinite(y)
    if not 0<=diagonal_shrink<=1 or not 0<=equal_shrink<=1:raise ValueError('invalid weight shrinkage')
    if good.sum()<minimum:return equal,int(good.sum()),np.eye(k)
    error=p[good]-y[good,None];cov=np.cov(error,rowvar=False,ddof=0)
    matrix=(1-diagonal_shrink)*cov+diagonal_shrink*np.diag(np.diag(cov))
    matrix=matrix/max(float(np.trace(matrix)/k),1e-16)+np.eye(k)*1e-8
    fitted=minimize(lambda w:float(w@matrix@w),equal,jac=lambda w:2*matrix@w,
        bounds=[(0,1)]*k,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)}],
        method='SLSQP',options={'ftol':1e-12,'maxiter':200})
    if not fitted.success:raise ValueError('weight optimization failed: '+fitted.message)
    weights=equal_shrink*equal+(1-equal_shrink)*fitted.x
    sd=np.sqrt(np.maximum(np.diag(cov),1e-20));corr=cov/sd[:,None]/sd[None,:]
    return weights,int(good.sum()),corr


def causal_weighted_average(parts,y,dates,periods,horizon,policy):
    stack=np.stack(parts,axis=-1);result=np.full_like(y,np.nan);records=[]
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    for start,end in periods:
        boundary=pd.to_datetime(start,format='%Y%m%d%H%M',utc=True);stop=pd.to_datetime(end,format='%Y%m%d%H%M',utc=True)
        while boundary<=stop:
            key=boundary.strftime('%Y%m%d%H%M');next_time=boundary+pd.DateOffset(months=1)
            past=mature_window(dates,key,horizon,12);test=(stamp>=boundary)&(stamp<next_time)&(dates<=end)
            w,n,corr=shrunk_error_weights(stack[past],y[past],policy['diagonal_shrink'],policy['equal_shrink'],policy['minimum_labels'])
            result[test]=np.sum(stack[test]*w,axis=-1)
            records.append({'update':key,'mature_labels':n,**{f'weight_{i}':float(v) for i,v in enumerate(w)},
                'error_corr_01':float(corr[0,1]),'error_corr_02':float(corr[0,2]),'error_corr_12':float(corr[1,2])})
            boundary=next_time
    return result,records


def asset_shrunk_predictions(x,y,dates,periods,shared,spec,horizon,seed,threads,clip_bps,share,save=None):
    """Per-asset raw forecast with a fixed shrinkage towards the shared model."""
    if not 0<=share<=1:raise ValueError('invalid asset blend')
    result=np.full_like(y,np.nan);records=[];stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    for start,end in periods:
        boundary=pd.to_datetime(start,format='%Y%m%d%H%M',utc=True);stop=pd.to_datetime(end,format='%Y%m%d%H%M',utc=True)
        while boundary<=stop:
            key=boundary.strftime('%Y%m%d%H%M');next_time=boundary+pd.DateOffset(months=1)
            train=mature_window(dates,key,horizon,12);test=np.asarray((stamp>=boundary)&(stamp<next_time)&(dates<=end))
            for j in range(y.shape[1]):
                model,predict=fit_model(x[train,j:j+1],y[train,j:j+1],spec,seed+j,threads,clip_bps)
                result[test,j]=(1-share)*shared[test,j]+share*predict(x[test,j:j+1]).ravel()
                record={'update':key,'asset_index':j,'mature_labels':int(np.isfinite(y[train,j]).sum()),
                    'last_train_origin':str(dates[train][-1]),'first_train_origin':str(dates[train][0]),'prediction_rows':int(test.sum())}
                records.append(record)
                if save:save(model,key,j,record)
            boundary=next_time
    return result,records
