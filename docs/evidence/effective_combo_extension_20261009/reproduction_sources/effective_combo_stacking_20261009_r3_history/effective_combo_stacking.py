"""Convex stacking in prediction units, with causal shared/asset shrinkage."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .effective_combo import mature_window


def predictive_weights(prediction,target,prior,diagonal_shrink,prior_shrink,minimum):
    prior=np.asarray(prior,float);k=len(prior)
    if np.any(prior<0) or not np.isclose(prior.sum(),1) or not 0<=diagonal_shrink<=1 or not 0<=prior_shrink<=1:raise ValueError('invalid convex prior')
    p=prediction.reshape(-1,k);y=target.ravel();good=np.isfinite(p).all(axis=1)&np.isfinite(y);n=int(good.sum())
    if n<minimum:return prior.copy(),n
    p=p[good];y=np.clip(y[good],-.2,.2)
    gram=p.T@p/n;cross=p.T@y/n
    matrix=(1-diagonal_shrink)*gram+diagonal_shrink*np.diag(np.diag(gram))
    scale=max(float(np.trace(matrix)/k),1e-16);matrix=matrix/scale+np.eye(k)*1e-8;cross=cross/scale
    fitted=minimize(lambda w:float(w@matrix@w-2*cross@w),prior,jac=lambda w:2*(matrix@w-cross),
        bounds=[(0,1)]*k,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)}],
        method='SLSQP',options={'maxiter':200,'ftol':1e-12})
    if not fitted.success:raise ValueError('predictive stacking optimization failed')
    return prior_shrink*prior+(1-prior_shrink)*fitted.x,n


def causal_stacking(parts,y,dates,periods,horizon,policy):
    stack=np.stack(parts,axis=-1);global_pred=np.full_like(y,np.nan);asset_pred=np.full_like(y,np.nan);records=[]
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    for start,end in periods:
        boundary=pd.to_datetime(start,format='%Y%m%d%H%M',utc=True);stop=pd.to_datetime(end,format='%Y%m%d%H%M',utc=True)
        while boundary<=stop:
            key=boundary.strftime('%Y%m%d%H%M');next_time=boundary+pd.DateOffset(months=1)
            past=mature_window(dates,key,horizon,12);test=np.asarray((stamp>=boundary)&(stamp<next_time)&(dates<=end))
            w,n=predictive_weights(stack[past],y[past],policy['prior'],policy['diagonal_shrink'],policy['prior_shrink'],policy['minimum_labels'])
            global_pred[test]=np.sum(stack[test]*w,axis=-1)
            records.append({'update':key,'asset':'GLOBAL','mature_labels':n,**{f'weight_{i}':float(v) for i,v in enumerate(w)}})
            for j in range(y.shape[1]):
                local,count=predictive_weights(stack[past,j],y[past,j],policy['prior'],policy['diagonal_shrink'],policy['prior_shrink'],policy['minimum_labels'])
                blended=(1-policy['asset_share'])*w+policy['asset_share']*local
                asset_pred[test,j]=stack[test,j]@blended
                records.append({'update':key,'asset':j,'mature_labels':count,**{f'weight_{i}':float(v) for i,v in enumerate(blended)}})
            boundary=next_time
    return global_pred,asset_pred,records
