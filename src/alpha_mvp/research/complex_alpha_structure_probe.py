"""Past-only bounded structural counterfactuals; never a candidate score.

This proves distinguishability, not predictive necessity or significance.
Minute samples are preserved exactly under joint time permutation. Independent
channel shifts preserve marginal samples while removing channel alignment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .complex_alpha_primitives import window_logsignatures,tilt_partition,transition_flux
from .complex_alpha_data import minute_observations,RAW_COLUMNS


def perturb_window(z,mode,rng,block=30):
    if len(z)%block or mode not in {'time_order','channel_alignment'}:
        raise ValueError('counterfactual requires whole blocks and a named mode')
    result=z.copy()
    for start in range(0,len(z),block):
        if mode=='time_order':result[start:start+block]=z[start:start+block][rng.permutation(block)]
        else:
            for j in range(z.shape[1]):
                result[start:start+block,j]=np.roll(z[start:start+block,j],int(rng.integers(1,block)))
    return result


def probe_coordinates(z):
    if z.ndim!=2 or z.shape[1]<3 or not np.isfinite(z[:,:3]).all():
        raise ValueError('probe requires a completely observed normalized path')
    n=len(z);dx=np.column_stack((np.ones(n)/n,z[:,:3]/np.sqrt(n)))
    sig=window_logsignatures(dx,np.array([n]),n)[0]
    tilt=[]
    for group in [-1,0,1]:
        valid=(z[:,1]<-.5) if group==-1 else (np.abs(z[:,1])<=.5 if group==0 else z[:,1]>.5)
        for sign in [-1,1]:
            value,neff,maxw=tilt_partition(z[valid][:,[0,2]],np.array([sign*.5,.5]))
            tilt.append(value if neff>=8 and maxw<=.4 else np.nan)
    states=(z[:,0]>=0)*3+np.where(z[:,1]<-.5,0,np.where(z[:,1]>.5,2,1))
    flux,kl=transition_flux(states,6)
    return {'path_order1':sig[:4],'path_order2':sig[4:10],'path_order3':sig[10:],
        'conditional_tilt':np.array(tilt),'state_flux':flux.ravel(),
        'ordinary_marginals':np.r_[np.mean(z[:,:3],axis=0),np.std(z[:,:3],axis=0)],
        'ordinary_covariance':np.cov(z[:,:3],rowvar=False,ddof=0).ravel()}


def run_structure_probe(cfg,examples=24):
    """Uniform discovery windows, independent of outcomes and factor rankings."""
    data=cfg['data'];end=pd.to_datetime(cfg['split']['discovery_end'],format='%Y%m%d%H%M',utc=True)
    frames={};returns=[]
    for code in data['assets']:
        f=pq.read_table(f"{data['root']}/{code}/1m.parquet",columns=RAW_COLUMNS).to_pandas()
        t=pd.to_datetime(f.date,utc=True);f=f[(t>=pd.Timestamp(data['start'],tz='UTC'))&(t<end)].reset_index(drop=True)
        r=np.r_[np.nan,np.diff(np.log(f.close.to_numpy(float)))];returns.append(r);frames[code]=f
    stacked=np.column_stack(returns);market=np.full(len(stacked),np.nan);valid=np.isfinite(stacked).any(axis=1)
    market[valid]=np.nanmean(stacked[valid],axis=1);rep=cfg['complex_alpha']['representation'];rng=np.random.default_rng(cfg['complex_alpha']['seed']);rows=[]
    for code in [data['assets'][0],'BTCUSDT']:
        if code not in frames:continue
        f=frames[code];z,_=minute_observations(f,market,rep['normalization_minutes'],rep['normalization_min_count'])
        stops=np.linspace(2*rep['normalization_minutes']+720,len(z),examples,dtype=int)
        for stop in stops:
            original=z[stop-720:stop];a=probe_coordinates(original)
            for mode in ['time_order','channel_alignment']:
                b=probe_coordinates(perturb_window(original,mode,rng))
                for kind in a:
                    good=np.isfinite(a[kind])&np.isfinite(b[kind]);difference=b[kind][good]-a[kind][good]
                    rows.append({'asset':code,'completed_minute':str(f.date.iloc[stop-1]),'mode':mode,'object':kind,
                        'rms_change':float(np.sqrt(np.mean(difference**2))) if good.any() else np.nan,
                        'rms_original':float(np.sqrt(np.mean(a[kind][good]**2))) if good.any() else np.nan,
                        'finite_coordinates':int(good.sum()),'role':'identifiability only; no outcomes or model selection'})
    return pd.DataFrame(rows)
