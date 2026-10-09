"""Minute observations -> causal structured coordinates, with signed metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from scipy.signal import lfilter
from .complex_alpha_primitives import (PRIMITIVE_VERSION, rolling_sum, causal_standardize,
    causal_haar, ilr_counts, signed_log_bins, window_logsignatures,
    relative_covariance_log, fractional_kernel, lie_basis)

DATA_VERSION='complex-minute-representations-20261008-v1'
RAW_COLUMNS=['date','close','high','low','volume','quote_volume','trade_count','taker_buy_quote_volume']
CHANNELS=['r','flow','amount','count','size','range','location','vwap','relative_r']


def fingerprint(path: Path) -> dict:
    s=path.stat()
    return {'path':str(path.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def minute_observations(frame: pd.DataFrame, market: np.ndarray, normal_window: int,
                        normal_min: int) -> tuple[np.ndarray,dict]:
    """Invalid OHLC/amount/count/flow remain missing, not zero activity."""
    v={k:pd.to_numeric(frame[k],errors='coerce').to_numpy(float) for k in RAW_COLUMNS if k!='date'}
    close=v['close']; q=v['quote_volume']; n=v['trade_count']; buy=v['taker_buy_quote_volume']
    price_good=(close>0)&(v['high']>=close)&(v['low']<=close)&(v['low']>0)&(v['high']>=v['low'])
    r=np.r_[np.nan,np.diff(np.log(np.where(close>0,close,np.nan)))]
    r[~price_good]=np.nan
    flow=np.divide(2*buy-q,q,out=np.full(len(q),np.nan),where=(q>0)&(buy>=0)&(buy<=q))
    amount=np.where(q>=0,np.log1p(np.maximum(q,0)),np.nan)
    count=np.where(n>=0,np.log1p(np.maximum(n,0)),np.nan)
    size=np.log(np.divide(q,n,out=np.full(len(q),np.nan),where=(q>0)&(n>0)))
    interval=np.log(np.divide(v['high'],v['low'],out=np.full(len(q),np.nan),where=price_good))
    location=np.divide(2*close-v['high']-v['low'],v['high']-v['low'],out=np.full(len(q),np.nan),where=price_good&(v['high']>v['low']))
    vwap=np.log(np.divide(close*v['volume'],q,out=np.full(len(q),np.nan),where=price_good&(v['volume']>0)&(q>0)))
    # Rolling past-only market loading at minute clock; no future return risk input.
    cov=pd.Series(r*market).shift(1).rolling(normal_window,min_periods=normal_min).mean().to_numpy()
    var=pd.Series(market**2).shift(1).rolling(normal_window,min_periods=normal_min).mean().to_numpy()
    beta=np.divide(cov,var,out=np.full(len(q),np.nan),where=var>1e-14)
    relative=r-np.clip(beta,0,3)*market
    raw=[r,flow,amount,count,size,interval,location,vwap,relative]
    normalized=[]; quality={}
    for k,x in zip(CHANNELS,raw):
        z,clip=causal_standardize(x,normal_window,normal_min,center=k not in {'r','relative_r'})
        normalized.append(z)
        quality[k]={'raw_missing_share':float(np.mean(~np.isfinite(x))),
                    'normalized_missing_share':float(np.mean(~np.isfinite(z))),
                    'clipped_share':float(np.mean(clip))}
    return np.column_stack(normalized),quality


@njit(cache=True)
def closed_activity_buckets(z,q,threshold):
    n=len(q); vectors=np.empty((n,4)); times=np.empty(n,np.int64);duration=np.empty(n);excess=np.empty(n)
    count=0; amount=0.;total=np.zeros(3);age=0;goal=np.nan;valid=True
    for t in range(n):
        if age==0:goal=threshold[t];valid=np.isfinite(goal) and goal>0
        age+=1
        if np.isfinite(q[t]) and q[t]>=0:amount+=q[t]
        else:valid=False
        for j in range(3):
            if np.isfinite(z[t,j]):total[j]+=z[t,j]
            else:valid=False
        if (np.isfinite(goal) and amount>=goal) or age>=120:
            if valid and amount>=goal:
                vectors[count,0]=age/15.
                for j in range(3):vectors[count,j+1]=total[j]
                times[count]=t+1;duration[count]=age;excess[count]=amount/goal-1;count+=1
            age=0;amount=0.;total[:]=0
    return vectors[:count],times[:count],duration[:count],excess[:count]


def representation_bank(z: np.ndarray, ends: np.ndarray, windows: list[int],
                        seed: int, quote_amount: np.ndarray | None = None,
                        min_reliability: float=.90) -> tuple[np.ndarray,list[dict],dict]:
    """All outputs depend only on [0,end); endpoints are completed minutes."""
    cols=[]; cards=[]; reliability={}
    def add(family,name,values,support,order=1,dependency=CHANNELS):
        cols.append(np.asarray(values,dtype=np.float32)); cards.append({
            'name':name,'family':family,'dependencies':list(dependency),'support_minutes':support,
            'order':order,'fitted':False,'version':DATA_VERSION,'clock':'completed_15m',
            'missing':'undefined/full-required support -> NaN; fit-time coverage gate',
            'unit':'dimensionless','direction':'none; discovery readout only'})
    def at(a): return a[ends-1]
    def mean(a,w): return rolling_sum(a,w)/w
    # Same-horizon low-order controls. Each field is a real minute aggregation.
    for w in windows:
        for j,ch in enumerate(CHANNELS):
            x=z[:,j]; m=mean(x,w); m2=mean(x*x,w); sd=np.sqrt(np.maximum(m2-m*m,0))
            with np.errstate(divide='ignore',invalid='ignore'):
                skew=(mean(x**3,w)-3*m*m2+2*m**3)/(sd**3)
            skew[sd<1e-8]=np.nan
            for tag,v in [('mean',m),('std',sd),('skew',np.clip(skew,-10,10))]:
                add('OLD',f'old_{ch}_{tag}_{w}',at(v),w,dependency=[ch])
    # F1: signed compositions, conditioned nonlinear-before-aggregation.
    bins=signed_log_bins(z[:,1])
    for w in windows:
        counts=np.column_stack([at(rolling_sum(np.where(bins>=0,(bins==k).astype(float),np.nan),w)) for k in range(13)])
        ilr=ilr_counts(counts)
        for k in range(12): add('F1',f'ilr_flow_{w}_{k}',ilr[:,k],w,dependency=['flow'])
        for lam,eta in [(.25,.25),(.5,.25),(.5,.5),(1.,.25)]:
            for group in [-1,0,1]:
                valid=np.isfinite(z[:,:4]).all(axis=1)
                inside=(z[:,1]<-.5) if group==-1 else ((np.abs(z[:,1])<=.5) if group==0 else (z[:,1]>.5))
                inside=inside&valid
                n=at(rolling_sum(inside.astype(float),w))
                ks=[]; ne=[]; mx=[]
                for sign in [-1,1]:
                    v=np.clip(sign*lam*z[:,0]+eta*z[:,2],-24,24)
                    ex=np.where(inside,np.exp(np.nan_to_num(v,nan=0)),0.)
                    s=at(rolling_sum(ex,w)); s2=at(rolling_sum(ex*ex,w))
                    with np.errstate(divide='ignore',invalid='ignore'):
                        ks.append(np.log(s/n)); ne.append(s*s/s2)
                    maxw=pd.Series(ex).rolling(w,min_periods=w).max().to_numpy()[ends-1]/np.maximum(s,1e-12)
                    mx.append(maxw)
                good=(n>=max(12,int(w*.025)))&(np.minimum(*ne)>=8)&(np.maximum(*mx)<.40)
                for tag,value in [('odd',ks[1]-ks[0]),('even',ks[1]+ks[0])]:
                    value[~good]=np.nan
                    add('F1',f'tilt_{tag}_{w}_{lam}_{eta}_{group}',value,w,2,['r','flow','amount'])
                reliability[f'tilt_{w}_{lam}_{eta}_{group}']={'median_neff':float(np.nanmedian(np.minimum(*ne))),
                    'pass_share':float(np.mean(good)),'max_weight_p99':float(np.nanquantile(np.maximum(*mx),.99))}
    # F2: signed couplings, energy and second modulation. All filters single-sided.
    for scale in [2,8,32,64]:
        r=causal_haar(z[:,0],scale); f=causal_haar(z[:,1],scale)
        a=causal_haar(z[:,2],scale)
        for w in [windows[0],windows[-1]]:
            energy=mean(np.abs(f),w)
            second=mean(np.abs(causal_haar(np.abs(f),scale*2)),w)
            add('F2',f'flow_envelope_{scale}_{w}',at(energy),w+2*scale-1,2,['flow'])
            add('F2',f'flow_modulation_{scale}_{w}',at(np.log((second+1e-6)/(energy+1e-6))),w+6*scale-2,3,['flow'])
            for lag in [0,scale]:
                delayed=np.r_[np.full(lag,np.nan),np.abs(f)[:-lag]] if lag else np.abs(f)
                m1=mean(r,w); m2=mean(delayed,w)
                denom=np.sqrt(np.maximum(mean(r*r,w)-m1*m1,0)*np.maximum(mean(delayed*delayed,w)-m2*m2,0))
                with np.errstate(divide='ignore',invalid='ignore'): c=(mean(r*delayed,w)-m1*m2)/denom
                c[denom<1e-8]=np.nan
                add('F2',f'price_flow_couple_{scale}_{lag}_{w}',at(c),w+2*scale+lag-1,2,['r','flow'])
                aa=np.r_[np.full(lag,np.nan),np.abs(a)[:-lag]] if lag else np.abs(a)
                add('F2',f'flow_activity_signed_{scale}_{lag}_{w}',at(mean(f*aa,w)),w+2*scale+lag-1,2,['flow','amount'])
    # F3: four dimensions include time; w-normalization fixes statistical scale.
    for w in windows:
        increments=np.column_stack((np.ones(len(z))/w,z[:,0]/np.sqrt(w),z[:,1]/np.sqrt(w),z[:,2]/np.sqrt(w)))
        signatures=window_logsignatures(increments,ends,w)
        names=[f'{i}' for i in range(4)]+[''.join(map(str,x)) for x in lie_basis(4,2)[0]]+[''.join(map(str,x)) for x in lie_basis(4,3)[0]]
        for k,name in enumerate(names):
            add('F3',f'logsig_{w}_{name}',signatures[:,k],w,len(name),['r','flow','amount','time'])
    if quote_amount is not None:
        q=np.asarray(quote_amount,float)
        historical=pd.Series(q).shift(1).rolling(10080,min_periods=1440).mean().to_numpy()*15
        vectors,times,duration,excess=closed_activity_buckets(z[:,[0,1,2]],q,historical)
        location=np.searchsorted(times,ends,side='right')-1
        for count in [16,48]:
            embedded=vectors.copy();embedded[:,0]/=count;embedded[:,1:]/=np.sqrt(15*count)
            signatures=window_logsignatures(embedded,np.arange(1,len(vectors)+1),count)
            values=np.full((len(ends),30),np.nan);good=location>=count-1
            ids=np.flatnonzero(good);lo=location[ids]-count+1
            good_ids=(ends[ids]-times[location[ids]]<=120)&(times[location[ids]]-times[lo]<=4320)
            ids=ids[good_ids];values[ids]=signatures[location[ids]]
            for k,name in enumerate(names):add('F3',f'activity_logsig_{count}_{name}',values[:,k],4440,len(name),['r','flow','amount','quote_volume','closed_bucket_clock'])
        for name,values in [('bucket_duration',duration),('bucket_excess',excess)]:
            result=np.full(len(ends),np.nan);good=(location>=0)
            result[good]=values[location[good]]
            result[good&(ends-np.maximum(times[np.maximum(location,0)],0)>120)]=np.nan
            add('F3',name,result,4440,1,['quote_volume','closed_bucket_clock'])
    # F4: full anti-symmetric basis of a six-state transition matrix.
    state=np.where(z[:,0]>=0,3,0)+np.where(z[:,1]<-.5,0,np.where(z[:,1]>.5,2,1))
    state[~np.isfinite(z[:,:2]).all(axis=1)]=-1
    for w in [windows[-1],windows[-1]*2]:
        occ=np.column_stack([at(rolling_sum(np.where(state>=0,(state==k).astype(float),np.nan),w)) for k in range(6)])
        contrasts=ilr_counts(occ)
        for k in range(5): add('F4',f'occupancy_{w}_{k}',contrasts[:,k],w,1,['r','flow'])
        for lag in [1,4]:
            prev=np.r_[np.full(lag,-1),state[:-lag]]
            good=(prev>=0)&(state>=0)
            q=np.stack([at(rolling_sum(np.where(good,((prev==a)&(state==b)).astype(float),np.nan),w))+.5 for a in range(6) for b in range(6)],axis=1).reshape(-1,6,6)
            q/=q.sum(axis=(1,2),keepdims=True)
            for a in range(6):
                for b in range(a+1,6): add('F4',f'flux_{w}_{lag}_{a}{b}',q[:,a,b]-q[:,b,a],w+lag,2,['r','flow'])
            add('F4',f'reversal_kl_{w}_{lag}',np.sum(q*np.log(q/q.transpose(0,2,1)),axis=(1,2)),w+lag,2,['r','flow'])
    # F5: shrinkage fixed ex ante, finite covariance only. Missing row never repaired.
    x=z[:,:4]
    def cov(w):
        m=at(mean(x,w)); c=np.empty((len(ends),4,4))
        for i in range(4):
            for j in range(4): c[:,i,j]=at(mean(x[:,i]*x[:,j],w))-m[:,i]*m[:,j]
        diag=np.diagonal(c,axis1=1,axis2=2).copy()
        c*=.9
        for i in range(4): c[:,i,i]+=.1*np.maximum(diag[:,i],1e-5)+1e-5
        return c
    long=cov(windows[-1]); good_long=np.isfinite(long).all(axis=(1,2))
    for w in windows[:2]:
        short=cov(w); good=good_long&np.isfinite(short).all(axis=(1,2))
        geom=np.full_like(short,np.nan)
        geom[good]=relative_covariance_log(short[good],long[good])
        for i in range(4):
            for j in range(i,4): add('F5',f'geometry_{w}_{windows[-1]}_{i}{j}',geom[:,i,j],windows[-1],2,['r','flow','amount','count'])
    # F6: causal fractional response and logarithmic-age contrasts.
    for j,ch in enumerate(CHANNELS[:4]):
        for d in [.25,.5]:
            kernel=fractional_kernel(d,256)
            valid=rolling_sum(np.isfinite(z[:,j]).astype(float),256)==256
            y=lfilter(kernel,[1],np.nan_to_num(z[:,j],nan=0.)); y[~valid]=np.nan
            add('F6',f'fractional_{ch}_{d}',at(y),256,1,[ch])
        for short,longw in [(8,64),(64,256),(256,720)]:
            add('F6',f'log_age_{ch}_{short}_{longw}',at(mean(z[:,j],short)-mean(z[:,j],longw)),longw,1,[ch])
    # F7: one registered seed. Finite kernel mean, not an exact distribution.
    rng=np.random.default_rng(seed); omega=rng.normal(size=(4,32)); phase=rng.uniform(0,2*np.pi,32)
    for w in [windows[0],windows[-1]]:
        for k in range(32):
            phi=np.sqrt(2/32)*np.cos(z[:,:4]@omega[:,k]+phase[k])
            add('F7',f'kernel_mean_{w}_{k}',at(mean(phi,w)),w,2,['r','flow','amount','count'])
    values=np.column_stack(cols)
    return values,cards,reliability


def prepare_minute_bank(cfg: dict, cache_root: Path) -> dict:
    """Cache is keyed by config, minute source fingerprints and formula code."""
    data=cfg['data']; spec=cfg['complex_alpha']['representation']
    root=Path(data['root']); codes=data['assets']
    files=[root/code/'1m.parquet' for code in codes]
    source=[fingerprint(p) for p in files]
    source += [fingerprint(root/code/'5m.parquet') for code in codes]
    code_hash=hashlib.sha256((Path(__file__).read_bytes()+Path(__file__).with_name('complex_alpha_primitives.py').read_bytes())).hexdigest()
    signature=hashlib.sha256(json.dumps({'data':data,'spec':spec,'source':source,'code':code_hash},sort_keys=True).encode()).hexdigest()
    dest=cache_root/signature[:20]; dest.mkdir(parents=True,exist_ok=True)
    manifest_path=dest/'bank_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']!='COMPLETED' or manifest['signature']!=signature:
            raise ValueError('invalid representation cache')
        print(f'[complex data] signed cache hit {dest}',flush=True)
        return {**manifest,'cache_dir':str(dest)}
    start=pd.Timestamp(data['start'],tz='UTC'); end=pd.Timestamp(data['end'],tz='UTC')
    clock=None; market=[]; fast_close=[]; fast_clock=None
    for code,p in zip(codes,files):
        frame=pq.read_table(p,columns=['date','close']).to_pandas()
        t=pd.to_datetime(frame.date,utc=True); good=(t>=start)&(t<=end)
        t=pd.DatetimeIndex(t[good]); close=frame.close.to_numpy(float)[good]
        if t.has_duplicates or (np.diff(t.asi8)!=60_000_000_000).any():
            raise ValueError(f'minute duplicate/gap {code}')
        if clock is None: clock=t
        elif not clock.equals(t): raise ValueError(f'minute clocks differ {code}')
        market.append(np.r_[np.nan,np.diff(np.log(np.where(close>0,close,np.nan)))])
        f=pq.read_table(root/code/'5m.parquet',columns=['date','close']).to_pandas()
        ft=pd.to_datetime(f.date,utc=True); fg=(ft>=start)&(ft<=end)
        ft=pd.DatetimeIndex(ft[fg]+pd.Timedelta(minutes=5))
        if fast_clock is None: fast_clock=ft
        elif not fast_clock.equals(ft): raise ValueError(f'5m clocks differ {code}')
        fast_close.append(f.close.to_numpy(float)[fg])
    market=np.nanmean(np.column_stack(market),axis=1)
    available=clock+pd.Timedelta(minutes=1)
    endpoint=np.flatnonzero((available.minute%15==0))+1
    dates=available[endpoint-1].strftime('%Y%m%d%H%M').to_numpy()
    np.save(dest/'dates.npy',dates.astype('U12')); np.save(dest/'fast_dates.npy',fast_clock.strftime('%Y%m%d%H%M').to_numpy().astype('U12'))
    np.save(dest/'fast_close.npy',np.column_stack(fast_close))
    quality=[]; cards=None; reliabilities=[]
    for code,p in zip(codes,files):
        print(f'[complex data] {code}: minute structure',flush=True)
        frame=pq.read_table(p,columns=RAW_COLUMNS).to_pandas()
        t=pd.to_datetime(frame.date,utc=True); frame=frame[(t>=start)&(t<=end)].reset_index(drop=True)
        z,q=minute_observations(frame,market,spec['normalization_minutes'],spec['normalization_min_count'])
        values,coordinate_cards,rel=representation_bank(z,endpoint,spec['windows_minutes'],spec['seed'],frame.quote_volume.to_numpy(float))
        for card in coordinate_cards:
            card['normalization_support_minutes']=spec['normalization_minutes']*2
            card['total_dependency_minutes']=card['support_minutes']+spec['normalization_minutes']*2-1
        if cards is not None and cards!=coordinate_cards: raise ValueError('asset coordinate registry differs')
        cards=coordinate_cards
        # Verify minute endpoint close against native 5m at exact completion.
        idx=np.searchsorted(fast_clock.asi8,available[endpoint-1].asi8)
        if not np.allclose(frame.close.to_numpy(float)[endpoint-1],np.column_stack(fast_close)[idx,codes.index(code)],rtol=1e-10,atol=1e-9):
            raise ValueError(f'1m/5m closing price mismatch {code}')
        np.save(dest/f'{code}.npy',values)
        quality.append({'asset':code,'channels':q,'coordinate_finite_share':float(np.mean(np.isfinite(values))),
                        'minute_rows':len(frame),'duplicates':0,'missing_clock_rows':0})
        reliabilities.append({'asset':code,'tilt':rel})
    manifest={'status':'COMPLETED','signature':signature,'code_sha256':code_hash,'source_files':source,
        'codes':codes,'coordinates':cards,'quality':quality,'reliability':reliabilities,
        'decision_rows':len(dates),'minute_rows':len(clock),'version':DATA_VERSION}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return {**manifest,'cache_dir':str(dest)}
