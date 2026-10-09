"""Additive log-age surfaces/paths over a strictly verified parent minute bank."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from .complex_alpha_data import fingerprint,minute_observations,RAW_COLUMNS
from .complex_alpha_primitives import rolling_sum,ilr_counts,signed_log_bins,window_logsignatures,lie_basis

EXTENSION_VERSION='complex-log-age-extension-20261008-v2-relative-and-joint-controls'


def log_age_coordinates(z,ends,blocks):
    columns=[];cards=[]
    def add(family,name,values,lag,width,order,dependencies):
        columns.append(np.asarray(values,np.float32));cards.append({'family':family,'name':name,
            'support_minutes':lag+width,'order':order,'dependencies':dependencies,
            'fitted':False,'clock':'completed_15m','unit':'dimensionless','version':EXTENSION_VERSION,
            'missing':'full block required; conditional n>=12, neff>=8 and max weight<=.4',
            'direction':'none; discovery readout only','lag_minutes':lag,'block_minutes':width})
    bins=signed_log_bins(z[:,1]);base_valid=np.isfinite(z[:,:3]).all(axis=1)
    directions=[(.25,.25),(.5,.5),(1.,.25)]
    for lag,stop in blocks:
        width=stop-lag;anchor=ends-lag
        def at(a):
            result=np.full(len(ends),np.nan);good=anchor>0
            result[good]=a[anchor[good]-1];return result
        counts=np.column_stack([at(rolling_sum(np.where(bins>=0,(bins==k).astype(float),np.nan),width)) for k in range(13)])
        ilr=ilr_counts(counts)
        for k in range(12):add('F1',f'age_ilr_{lag}_{stop}_{k}',ilr[:,k],lag,width,1,['flow'])
        for lam,eta in directions:
            groups=[99] if width<90 else [-1,0,1]
            for group in groups:
                selected=(z[:,1]<-.5) if group==-1 else ((np.abs(z[:,1])<=.5) if group==0 else (z[:,1]>.5))
                if group==99:selected=np.ones(len(z),bool)
                selected&=base_valid
                n=at(rolling_sum(selected.astype(float),width));ks=[];eff=[];maxweight=[]
                for sign in [-1,1]:
                    exp=np.where(selected,np.exp(np.clip(np.nan_to_num(sign*lam*z[:,0]+eta*z[:,2]),-24,24)),0.)
                    total=at(rolling_sum(exp,width));squares=at(rolling_sum(exp*exp,width))
                    with np.errstate(divide='ignore',invalid='ignore'):
                        ks.append(np.log(total/n));eff.append(total*total/squares)
                    maxweight.append(at(pd.Series(exp).rolling(width,min_periods=width).max().to_numpy())/np.maximum(total,1e-12))
                valid=(n>=12)&(np.minimum(*eff)>=8)&(np.maximum(*maxweight)<=.4)
                for name,value in [('odd',ks[1]-ks[0]),('even',ks[1]+ks[0])]:
                    value[~valid]=np.nan
                    add('F1',f'age_tilt_{name}_{lag}_{stop}_{lam}_{eta}_{group}',value,lag,width,2,['r','flow','amount'])
        names=[str(i) for i in range(4)]+[''.join(map(str,w)) for w in lie_basis(4,2)[0]]+[''.join(map(str,w)) for w in lie_basis(4,3)[0]]
        count=10 if width<90 else 30
        for price_channel in [0,8]:
            variant='raw' if price_channel==0 else 'relative'
            dx=np.column_stack((np.ones(len(z))/width,z[:,price_channel]/np.sqrt(width),z[:,1]/np.sqrt(width),z[:,2]/np.sqrt(width)))
            signature=window_logsignatures(dx,anchor,width)
            for k,name in enumerate(names[:count]):add('F3',f'age_logsig_{variant}_{lag}_{stop}_{name}',signature[:,k],lag,width,len(name),['r' if price_channel==0 else 'relative_r','flow','amount','time'])
    # Give ordinary joint covariances the same history as SPD geometry. This
    # prevents attributing a missing covariance control to matrix-log value.
    for width in [120,240,720]:
        for i in range(4):
            for j in range(i+1,4):
                a=rolling_sum(z[:,i],width)/width;b=rolling_sum(z[:,j],width)/width
                value=rolling_sum(z[:,i]*z[:,j],width)/width-a*b
                add('OLD',f'old_joint_cov_{width}_{i}{j}',value[ends-1],0,width,2,['r','flow','amount','count'])
    return np.column_stack(columns),cards


def extend_verified_bank(cfg):
    extension=cfg['complex_alpha']['extension'];parent_path=Path(extension['parent_bank_manifest'])
    parent=json.loads(parent_path.read_text(encoding='utf-8'));parent_dir=parent_path.parent
    parent_cfg=yaml.safe_load(Path(extension['parent_config']).read_text(encoding='utf-8'))
    if parent['status']!='COMPLETED' or parent_cfg['data']!=cfg['data'] or parent_cfg['complex_alpha']['representation']!=cfg['complex_alpha']['representation']:
        raise ValueError('extension parent data/representation contract does not match')
    for f in parent['source_files']:
        if fingerprint(Path(f['path']))!=f:raise ValueError('extension parent source changed')
    snapshot=parent_dir/'source_snapshot'
    code=hashlib.sha256((snapshot/'complex_alpha_data.py').read_bytes()+(snapshot/'complex_alpha_primitives.py').read_bytes()).hexdigest()
    if code!=parent['code_sha256']:raise ValueError('parent source snapshot differs from signed bank')
    current=hashlib.sha256(Path(__file__).with_name('complex_alpha_data.py').read_bytes()+Path(__file__).with_name('complex_alpha_primitives.py').read_bytes()).hexdigest()
    if current!=code:raise ValueError('parent observation implementation changed; extension cannot reuse it')
    signature=hashlib.sha256((parent['signature']+json.dumps(extension,sort_keys=True)).encode()+Path(__file__).read_bytes()).hexdigest()
    dest=Path(cfg['data']['feature_cache'])/signature[:20];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'bank_manifest.json').exists():
        result=json.loads((dest/'bank_manifest.json').read_text());
        if result['signature']!=signature or result['status']!='COMPLETED':raise ValueError('invalid extension cache')
        return {**result,'cache_dir':str(dest)}
    data=cfg['data'];root=Path(data['root']);spec=cfg['complex_alpha']['representation'];start=pd.Timestamp(data['start'],tz='UTC');end=pd.Timestamp(data['end'],tz='UTC')
    market=[]
    for code in parent['codes']:
        f=pq.read_table(root/code/'1m.parquet',columns=['date','close']).to_pandas();t=pd.to_datetime(f.date,utc=True);good=(t>=start)&(t<=end)
        price=f.close.to_numpy(float)[good];market.append(np.r_[np.nan,np.diff(np.log(price))])
    market=np.nanmean(np.column_stack(market),axis=1);extra_cards=None
    for code in parent['codes']:
        print(f'[complex extension] {code}: disjoint log-age surfaces and paths',flush=True)
        frame=pq.read_table(root/code/'1m.parquet',columns=RAW_COLUMNS).to_pandas();t=pd.to_datetime(frame.date,utc=True)
        frame=frame[(t>=start)&(t<=end)].reset_index(drop=True);clock=pd.to_datetime(frame.date,utc=True)+pd.Timedelta(minutes=1)
        ends=np.flatnonzero(clock.dt.minute%15==0)+1
        z,_=minute_observations(frame,market,spec['normalization_minutes'],spec['normalization_min_count'])
        values,newcards=log_age_coordinates(z,ends,extension['age_blocks'])
        for c in newcards:c['total_dependency_minutes']=c['support_minutes']+2*spec['normalization_minutes']-1
        if extra_cards is not None and newcards!=extra_cards:raise ValueError('extension asset registries differ')
        extra_cards=newcards
        base=np.load(parent_dir/f'{code}.npy',mmap_mode='r')
        np.save(dest/f'{code}.npy',np.column_stack((base,values)))
    for name in ['dates.npy','fast_dates.npy','fast_close.npy']:
        import shutil
        shutil.copyfile(parent_dir/name,dest/name)
    result={**parent,'signature':signature,'parent_signature':parent['signature'],'parent_bank':str(parent_path.resolve()),
        'coordinates':parent['coordinates']+extra_cards,'version':EXTENSION_VERSION,
        'extension_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dest/'bank_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return {**result,'cache_dir':str(dest)}
