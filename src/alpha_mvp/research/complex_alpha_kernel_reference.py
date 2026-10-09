"""Moment-conditioned Gaussian reference for a fixed finite Fourier embedding."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from .complex_alpha_primitives import rolling_sum

VERSION='conditional-gaussian-kernel-20261008-v1'
CHANNELS=['r','flow','amount','count']


def gaussian_fourier_reference(mean,covariance,omega,phase):
    if mean.shape[-1]!=omega.shape[0] or covariance.shape!=(*mean.shape[:-1],mean.shape[-1],mean.shape[-1]):
        raise ValueError('Gaussian reference moment shapes disagree')
    variance=np.einsum('ik,...ij,jk->...k',omega,covariance,omega)
    finite=np.isfinite(variance)
    if np.any(variance[finite]<-1e-4):raise ValueError('Gaussian reference has invalid negative directional variance')
    return np.sqrt(2/omega.shape[1])*np.cos(mean@omega+phase)*np.exp(-.5*np.maximum(variance,0))


def joint_covariance_controls(z,ends,windows):
    values=[];cards=[]
    for width in windows:
        for i in range(4):
            for j in range(i+1,4):
                m1=rolling_sum(z[:,i],width)/width;m2=rolling_sum(z[:,j],width)/width
                values.append((rolling_sum(z[:,i]*z[:,j],width)/width-m1*m2)[ends-1])
                cards.append({'name':f'old_joint_cov_{width}_{i}{j}','family':'OLD','support_minutes':width,'order':2,
                    'dependencies':CHANNELS,'fitted':False,'clock':'completed_15m','unit':'dimensionless',
                    'missing':'any missing minute invalidates block','version':VERSION,'direction':'discovery only'})
    return np.column_stack(values).astype(np.float32),cards


def augment_kernel_reference(bank,cache_root,windows,seed):
    parent=Path(bank['cache_dir']);signature=hashlib.sha256(bank['signature'].encode()+str((windows,seed)).encode()+Path(__file__).read_bytes()).hexdigest()
    dest=Path(cache_root)/signature[:20];manifest=dest/'bank_manifest.json'
    if manifest.exists():
        result=json.loads(manifest.read_text());
        if result['signature']!=signature or result['status']!='COMPLETED':raise ValueError('invalid conditional kernel cache')
        return {**result,'cache_dir':str(dest)}
    dest.mkdir(parents=True,exist_ok=True);index={c['name']:i for i,c in enumerate(bank['coordinates'])}
    rng=np.random.default_rng(seed);omega=rng.normal(size=(4,32));phase=rng.uniform(0,2*np.pi,32);cards=[]
    for width in [windows[0],windows[-1]]:
        for k in range(32):
            cards.append({'name':f'kernel_departure_{width}_{k}','family':'F7','support_minutes':width,'order':2,
                'dependencies':CHANNELS+['same_window_first_second_moments'],'fitted':False,'clock':'completed_15m',
                'unit':'dimensionless','missing':'full moment and empirical kernel support required','version':VERSION,
                'direction':'none; discovery only','formula':'empirical Fourier mean minus moment-matched Gaussian expectation',
                'kernel_seed':seed,'omega':omega[:,k].tolist(),'phase':float(phase[k])})
    quality=[]
    for code in bank['codes']:
        print(f'[complex kernel reference] {code}: same-window mean/covariance conditioned',flush=True)
        raw=np.load(parent/f'{code}.npy',mmap_mode='r');extra=[]
        for width in [windows[0],windows[-1]]:
            mean=np.column_stack([raw[:,index[f'old_{ch}_mean_{width}']] for ch in CHANNELS]).astype(float)
            cov=np.empty((len(raw),4,4),float)
            for i,ch in enumerate(CHANNELS):cov[:,i,i]=raw[:,index[f'old_{ch}_std_{width}']].astype(float)**2
            for i in range(4):
                for j in range(i+1,4):cov[:,i,j]=cov[:,j,i]=raw[:,index[f'old_joint_cov_{width}_{i}{j}']]
            observed=raw[:,[index[f'kernel_mean_{width}_{k}'] for k in range(32)]]
            extra.append(observed-gaussian_fourier_reference(mean,cov,omega,phase))
        values=np.column_stack(extra).astype(np.float32)
        np.save(dest/f'{code}.npy',np.column_stack((raw,values)))
        quality.append({'asset':code,'coordinate_finite_share':np.mean(np.isfinite(values),axis=0).tolist()})
    for name in ['dates.npy','fast_dates.npy','fast_close.npy']:shutil.copyfile(parent/name,dest/name)
    (dest/'source_snapshot').mkdir(exist_ok=True);shutil.copyfile(__file__,dest/'source_snapshot'/Path(__file__).name)
    result={**bank,'signature':signature,'parent_signature':bank['signature'],'parent_cache':str(parent.resolve()),
        'coordinates':bank['coordinates']+cards,'version':VERSION,'kernel_reference_quality':quality,
        'kernel_reference_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result.pop('cache_dir',None);manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return {**result,'cache_dir':str(dest)}
