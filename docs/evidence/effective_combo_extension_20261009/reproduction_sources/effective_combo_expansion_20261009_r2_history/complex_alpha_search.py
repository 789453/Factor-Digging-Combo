"""Typed, deterministic bounded expressions and discovery-only matrix screening."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch

SEARCH_VERSION='complex-typed-search-20261008-v1'


@dataclass(frozen=True)
class Candidate:
    id: str
    family: str
    mode: str
    left: tuple[int,...]
    left_weights: tuple[float,...]
    right: tuple[int,...]
    right_weights: tuple[float,...]
    support_minutes: int
    nonlinear_order: int
    nodes: int

    def record(self): return asdict(self)


def generate_candidates(cards: list[dict], budgets: dict[str,int], seed: int,
                        modes: list[str], prefix: str='main') -> list[Candidate]:
    legal={'linear','tanh_product','asinh_difference','signed_energy','cubic'}
    if set(modes)-legal or not modes: raise ValueError('invalid typed scalar transform')
    rng=np.random.default_rng(seed); result=[]; seen=set()
    for family,n in budgets.items():
        pool=np.array([i for i,c in enumerate(cards) if c['family']==family and c.get('admitted',True)])
        if len(pool)<4 or n<1: raise ValueError(f'insufficient typed coordinates for {family}')
        accepted=0; attempts=0
        while accepted<n:
            attempts+=1
            if attempts>n*100: raise ValueError(f'unique candidate budget exhausted {family}')
            mode=modes[rng.integers(len(modes))]
            def projection():
                indices=tuple(sorted(rng.choice(pool,size=int(rng.integers(2,5)),replace=False).tolist()))
                weights=rng.choice([-2,-1,1,2],size=len(indices))
                weights*=np.sign(weights[0])
                divisor=math.gcd(*np.abs(weights).tolist())
                weights=weights/divisor
                return indices,tuple((weights/np.linalg.norm(weights)).tolist())
            left,lw=projection(); right,rw=projection() if mode!='linear' else ((),())
            if mode=='tanh_product' and (right,rw)<(left,lw): left,lw,right,rw=right,rw,left,lw
            key=(family,mode,left,lw,right,rw)
            if key in seen: continue
            seen.add(key)
            depend=left+right
            support=max(cards[k]['support_minutes'] for k in depend)
            order=1 if mode=='linear' else (3 if mode=='cubic' else 2)
            nodes=3+len(depend)*2+order
            digest=hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            result.append(Candidate(f'{prefix}_{family}_{digest}',family,mode,left,lw,right,rw,support,order,nodes))
            accepted+=1
    return result


def evaluate_candidate(x: np.ndarray, c: Candidate) -> np.ndarray:
    left=x[...,list(c.left)]@np.asarray(c.left_weights)
    right=x[...,list(c.right)]@np.asarray(c.right_weights) if c.right else 0.
    if c.mode=='linear': return left
    if c.mode=='tanh_product': return np.tanh(left)*np.tanh(right)
    if c.mode=='asinh_difference': return np.arcsinh(left)-np.arcsinh(right)
    if c.mode=='signed_energy': return np.tanh(left)*np.logaddexp(right,-right)-np.tanh(left)*np.log(2.)
    if c.mode=='cubic': return np.tanh(left)*np.tanh(right)**2
    raise ValueError('unknown candidate mode')


def _torch_candidates(x, candidates):
    outputs=[]
    for mode in ['linear','tanh_product','asinh_difference','signed_energy','cubic']:
        pick=[i for i,c in enumerate(candidates) if c.mode==mode]
        if not pick: continue
        group=[candidates[i] for i in pick]
        def projection(side):
            value=torch.zeros((len(x),len(group)),device=x.device)
            for k in range(4):
                idx=[getattr(c,side)[k] if k<len(getattr(c,side)) else 0 for c in group]
                weight=[getattr(c,side+'_weights')[k] if k<len(getattr(c,side)) else 0 for c in group]
                value+=x[:,idx]*torch.tensor(weight,device=x.device)[None,:]
            return value
        left=projection('left'); right=projection('right') if mode!='linear' else None
        if mode=='linear': value=left
        elif mode=='tanh_product': value=torch.tanh(left)*torch.tanh(right)
        elif mode=='asinh_difference': value=torch.asinh(left)-torch.asinh(right)
        elif mode=='signed_energy': value=torch.tanh(left)*(torch.logaddexp(right,-right)-math.log(2))
        else: value=torch.tanh(left)*torch.tanh(right)**2
        outputs.append((pick,value))
    result=torch.empty((len(x),len(candidates)),device=x.device)
    for pick,value in outputs: result[:,pick]=value
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    a=np.asarray(x).ravel(); b=np.asarray(y).ravel(); good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<100:return np.nan
    a=a[good]-np.mean(a[good]);b=b[good]-np.mean(b[good])
    d=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/d) if d>1e-12 else np.nan


def screen_candidates(x: np.ndarray, targets: dict[int,np.ndarray], masks: dict[str,np.ndarray],
                      candidates: list[Candidate], device: str, batch: int,
                      checkpoint, resume: pd.DataFrame | None=None) -> pd.DataFrame:
    """Explicit discovery arrays only; no validation/holdout argument accepted.

    Directions and one-dimensional calibration from initial OOF calibration.
    All four later discovery folds independently contribute to robust score.
    """
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('requested CUDA unavailable')
    tensor=torch.as_tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float32,device=device)
    expanded={k:torch.as_tensor(np.repeat(v,x.shape[1]),device=device) for k,v in masks.items()}
    ys={h:torch.as_tensor(np.nan_to_num(y.reshape(-1),nan=0),dtype=torch.float32,device=device) for h,y in targets.items()}
    valid={h:torch.as_tensor(np.isfinite(y.reshape(-1)),device=device) for h,y in targets.items()}
    existing=set(resume.id) if resume is not None and not resume.empty else set()
    rows=[] if resume is None else resume.to_dict('records')
    todo=[c for c in candidates if c.id not in existing]
    for start in range(0,len(todo),batch):
        group=todo[start:start+batch]; raw=_torch_candidates(tensor,group)
        for h,y in ys.items():
            train=expanded['calibration']&valid[h]
            p=raw[train]; label=y[train]
            pc=p-p.mean(dim=0); yc=label-label.mean()
            coeff=(pc*yc[:,None]).sum(dim=0)/torch.clamp((pc*pc).sum(dim=0),min=1e-8)
            direction=torch.sign(coeff); fold_ic=[]; fold_gain=[]
            for fold in ['fold0','fold1','fold2','fold3']:
                use=expanded[fold]&valid[h]
                a=raw[use]-p.mean(dim=0); b=y[use]
                ac=a-a.mean(dim=0); bc=b-b.mean()
                corr=(ac*bc[:,None]).sum(dim=0)/torch.sqrt(torch.clamp((ac*ac).sum(dim=0)*(bc*bc).sum(),min=1e-16))
                gain=(2*a*coeff[None,:]*b[:,None]-(a*coeff[None,:])**2).mean(dim=0)
                fold_ic.append((corr*direction).cpu().numpy()); fold_gain.append(gain.cpu().numpy())
            ic=np.vstack(fold_ic); gain=np.vstack(fold_gain)
            coefficient=coeff.cpu().numpy(); dirs=direction.cpu().numpy()
            robust=np.mean(ic,axis=0)-.35*np.std(ic,axis=0)
            for k,c in enumerate(group):
                rows.append({'id':c.id,'family':c.family,'mode':c.mode,'horizon_hours':h,
                    'direction':int(dirs[k]),'coefficient':float(coefficient[k]),
                    'discovery_score':float(robust[k]),'discovery_mean_ic':float(ic[:,k].mean()),
                    'discovery_min_ic':float(ic[:,k].min()),'positive_folds':int((ic[:,k]>0).sum()),
                    'discovery_loss_gain':float(gain[:,k].mean()),
                    **{f'fold{j}_ic':float(ic[j,k]) for j in range(4)},
                    'nodes':c.nodes,'support_minutes':c.support_minutes})
        if (start//batch)%10==0 or start+batch>=len(todo):
            frame=pd.DataFrame(rows); checkpoint(frame)
            print(f'[complex coarse] {start+len(group)}/{len(todo)} definitions; {len(frame)} target records',flush=True)
    return pd.DataFrame(rows)


def diversity_survivors(frame: pd.DataFrame, n: int, per_family: int) -> pd.DataFrame:
    ranked=frame.sort_values(['discovery_score','id','horizon_hours'],ascending=[False,True,True])
    chosen=[]; counts={}; seen=set()
    for row in ranked.itertuples():
        if row.id in seen or counts.get(row.family,0)>=per_family or row.direction==0: continue
        chosen.append(row._asdict()); seen.add(row.id); counts[row.family]=counts.get(row.family,0)+1
        if len(chosen)==n: break
    return pd.DataFrame(chosen)


def numeric_archive(x,candidates,frame,threshold=.985):
    """Discovery-only output equivalence. Low correlation is not alpha evidence."""
    n=min(10000,x.shape[0]*x.shape[1]); sample=x.reshape(-1,x.shape[-1])[np.linspace(0,x.shape[0]*x.shape[1]-1,n,dtype=int)]
    values=np.column_stack([evaluate_candidate(sample,c) for c in candidates]);values-=values.mean(axis=0)
    norm=np.sqrt(np.sum(values*values,axis=0));unit=values/np.maximum(norm,1e-12)
    similarity=unit.T@unit;selected=[];aliases=[]
    for i,row in enumerate(frame.itertuples()):
        if norm[i]<1e-8:aliases.append({'id':row.id,'representative':'NONE','reason':'constant discovery output'});continue
        duplicate=next((j for j in selected if abs(similarity[i,j])>=threshold),None)
        if duplicate is not None:
            aliases.append({'id':row.id,'representative':frame.iloc[duplicate].id,'reason':'discovery output correlation',
                'correlation':float(similarity[i,duplicate])})
        else:selected.append(i)
    return frame.iloc[selected].copy(),pd.DataFrame(aliases)
