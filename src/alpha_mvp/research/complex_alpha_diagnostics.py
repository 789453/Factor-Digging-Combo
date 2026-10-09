"""Bounded structural identifiability, time-block uncertainty and selection nulls."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .complex_alpha_search import evaluate_candidate,correlation


def block_gain_interval(delta,target,baseline,dates,mask,repeats,seed,block_days=30):
    loss=(target-baseline)**2-(target-baseline-delta)**2
    row=np.nanmean(loss,axis=1);base=np.nanmean((target-baseline)**2,axis=1)
    frame=pd.DataFrame({'gain':row[mask],'base':base[mask]},index=pd.to_datetime(dates[mask],format='%Y%m%d%H%M',utc=True))
    daily=frame.resample('D').mean().dropna()
    if len(daily)<block_days*2:return {'low':np.nan,'high':np.nan,'days':len(daily)}
    rng=np.random.default_rng(seed);samples=[];n=len(daily);values=daily.to_numpy()
    for _ in range(repeats):
        starts=rng.integers(0,n,size=int(np.ceil(n/block_days)))
        index=np.concatenate([(np.arange(s,s+block_days)%n) for s in starts])[:n]
        m=values[index].mean(axis=0);samples.append(m[0]/m[1])
    return {'low':float(np.quantile(samples,.025)),'high':float(np.quantile(samples,.975)),
        'days':n,'repeats':repeats,'block_days':block_days,'unit':'whole market days, assets drawn jointly',
        'limitation':'interval conditional on selected model; not a full search-adjusted significance test'}


def limited_shift_null(x,targets,masks,candidates,seed,repeats=40):
    """Joint circular shifts of survivor labels; preserves cross-asset dependence.

    Does NOT rerun the full 60k search, hence cannot certify winner significance.
    """
    rng=np.random.default_rng(seed);panel=np.stack([evaluate_candidate(x,c) for c in candidates],axis=-1)
    use=masks['fold0']|masks['fold1']|masks['fold2']|masks['fold3'];train=masks['calibration']
    rows=[]
    for h,y in targets.items():
        for repeat in range(repeats):
            shift=int(rng.integers(max(10,len(y)//10),max(11,len(y)*9//10)))
            label=np.roll(y,shift,axis=0);yy=np.nan_to_num(label,nan=0)
            t=panel[train].reshape(-1,len(candidates));a=yy[train].reshape(-1)
            direction=np.sign((t*a[:,None]).sum(axis=0))
            test=panel[use].reshape(-1,len(candidates));b=yy[use].reshape(-1);b-=b.mean()
            ac=test-test.mean(axis=0);den=np.sqrt(np.maximum((ac*ac).sum(axis=0)*np.dot(b,b),1e-12))
            ic=(ac*b[:,None]).sum(axis=0)/den*direction
            rows.append({'repeat':repeat,'horizon_hours':h,'shift_steps':shift,'survivor_winner_ic':float(np.max(ic))})
    return pd.DataFrame(rows)


def matched_structure_windows(x,dates,cards,discovery,asset=0,limit=150):
    """Pairs chosen from past observable statistics only, never future outcomes."""
    old=[i for i,c in enumerate(cards) if c['family']=='OLD']
    structure=[i for i,c in enumerate(cards) if c['family']=='F3' and c['order']>=2]
    index=np.flatnonzero(discovery)[::4]
    a=x[index,asset][:,old];b=x[index,asset][:,structure]
    neighbor=NearestNeighbors(n_neighbors=min(24,len(index)),algorithm='auto').fit(a)
    pick=np.linspace(0,len(index)-1,min(limit,len(index)),dtype=int)
    distance,neighbors=neighbor.kneighbors(a[pick]);rows=[]
    for i,p in enumerate(pick):
        legal=np.flatnonzero(np.abs(neighbors[i]-p)>24*7)
        if not len(legal):continue
        k=legal[0];q=neighbors[i,k]
        rows.append({'left_time':dates[index[p]],'right_time':dates[index[q]],
            'old_distance':float(distance[i,k]/np.sqrt(len(old))),
            'path_distance':float(np.linalg.norm(b[p]-b[q])/np.sqrt(len(structure))),
            'selection':'nearest old observations excluding nearby week; no outcomes used'})
    return pd.DataFrame(rows)
