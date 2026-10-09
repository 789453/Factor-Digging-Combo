"""Cross-fitted residual prediction and a common native-5m trading contract."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from numba import njit
from .complex_alpha_search import correlation
from .aligned_contract import Split

EVALUATION_VERSION='complex-crossfit-native-contract-20261008-v2'


def past_market_betas(close: np.ndarray, width: int=2880,return_contract='log') -> np.ndarray:
    """15m past-only rolling beta vs leave-one-asset-out basket, 30 days."""
    ret=np.diff(np.log(close),axis=0,prepend=np.full((1,close.shape[1]),np.nan))
    if return_contract=='native_simple':ret=np.expm1(ret)
    elif return_contract!='log':raise ValueError('invalid return contract')
    result=np.full_like(close,np.nan)
    for j in range(close.shape[1]):
        market=np.nanmean(np.delete(ret,j,axis=1),axis=1)
        a=pd.Series(ret[:,j]); b=pd.Series(market)
        cov=a.shift(1).rolling(width,min_periods=max(96,width//4)).cov(b.shift(1),ddof=0).to_numpy()
        var=b.shift(1).rolling(width,min_periods=max(96,width//4)).var(ddof=0).to_numpy()
        result[:,j]=np.divide(cov,var,out=np.full(len(close),np.nan),where=var>1e-14)
    return np.clip(result,0,3)


def causal_risk_scales(close,beta,horizons,window=672,min_count=96,return_contract='log',label_projection='loo'):
    """Prior seven-day residual 15m volatility, restored to price-return units."""
    r=np.diff(np.log(close),axis=0,prepend=np.full((1,close.shape[1]),np.nan))
    if return_contract=='native_simple':r=np.expm1(r)
    elif return_contract!='log':raise ValueError('invalid return contract')
    market=(np.nansum(r,axis=1,keepdims=True)-r)/(r.shape[1]-1)
    innovation=r-beta*market
    if label_projection=='cash_common_beta':innovation=project_basket_returns(r,beta)
    elif label_projection!='loo':raise ValueError('invalid label projection')
    sigma=pd.DataFrame(innovation).shift(1).rolling(window,min_periods=min_count).std(ddof=0).to_numpy(copy=True)
    sigma[~np.isfinite(sigma)|(sigma<1e-8)]=np.nan
    return {h:sigma*np.sqrt(h*4) for h in horizons}


def targets_from_prices(dates,fast_dates,fast_close,horizons,split: Split,beta,
                      funding_event: np.ndarray,include_holdout: bool=False,return_contract='log') -> tuple[dict,dict]:
    """Future observations are labels only; split-crossing labels are missing."""
    idx=np.searchsorted(fast_dates,dates)
    if not np.array_equal(fast_dates[idx],dates): raise ValueError('decision not on native5 clock')
    period=np.where(fast_dates<=split.discovery_end,0,np.where(fast_dates<=split.validation_end,1,2))
    logp=np.log(fast_close); cash=np.cumsum(funding_event,axis=0)
    if return_contract not in {'log','native_simple'}:raise ValueError('invalid return contract')
    funded_notional=np.cumsum(funding_event*fast_close,axis=0) if return_contract=='native_simple' else None
    raw={}; residual={}
    for h in horizons:
        stop=idx+h*12; good=stop<len(fast_dates)
        rows=np.flatnonzero(good); rows=rows[period[idx[rows]]==period[stop[rows]]]
        if not include_holdout: rows=rows[period[idx[rows]]<2]
        y=np.full((len(dates),fast_close.shape[1]),np.nan)
        # Settlement at decision has already happened. Future long holding pays funding.
        y[rows]=logp[stop[rows]]-logp[idx[rows]]-(cash[stop[rows]]-cash[idx[rows]])
        if return_contract=='native_simple':
            initial=fast_close[idx[rows]]
            y[rows]=fast_close[stop[rows]]/initial-1-(funded_notional[stop[rows]]-funded_notional[idx[rows]])/initial
        market=(np.nansum(y,axis=1,keepdims=True)-y)/(y.shape[1]-1)
        raw[h]=y; residual[h]=y-beta*market
    return raw,residual


@dataclass
class RidgeState:
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    target_scale: float

    def predict(self,x):
        if x.shape[-1]!=len(self.center):raise ValueError('readout feature dimension mismatch')
        flat=x.reshape(-1,x.shape[-1]);result=np.empty(len(flat),float)
        for start in range(0,len(flat),32768):
            a=np.clip(np.nan_to_num((flat[start:start+32768]-self.center)/self.scale,nan=0.),-8,8)
            result[start:start+len(a)]=(a@self.coef+self.intercept)*self.target_scale
        return result.reshape(x.shape[:-1])

    def record(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.__dict__.items()}


def fit_ridge(x: np.ndarray,y: np.ndarray,penalty: float,device: str='cpu') -> RidgeState:
    if penalty<=0 or x.ndim!=2 or y.shape!=(len(x),): raise ValueError('invalid ridge inputs')
    good=np.isfinite(y).copy()
    for start in range(0,len(x),16384):good[start:start+16384]&=np.mean(np.isfinite(x[start:start+16384]),axis=1)>=.85
    if good.sum()<max(200,x.shape[1]*4): raise ValueError('insufficient ridge fitting observations')
    count=np.zeros(x.shape[1]);center=np.zeros(x.shape[1]);m2=np.zeros(x.shape[1])
    for start in range(0,len(x),16384):
        sample=x[start:start+16384][good[start:start+16384]].astype(float)
        n=np.isfinite(sample).sum(axis=0);total=count+n
        mean=np.divide(np.nansum(sample,axis=0),n,out=np.zeros(x.shape[1]),where=n>0)
        delta=mean-center
        m2+=np.nansum((sample-mean)**2,axis=0)+np.divide(delta*delta*count*n,total,out=np.zeros(x.shape[1]),where=total>0)
        center+=np.divide(delta*n,total,out=np.zeros(x.shape[1]),where=total>0);count=total
    scale=np.sqrt(np.divide(m2,count,out=np.zeros_like(m2),where=count>0))
    center[~np.isfinite(center)]=0;scale[~np.isfinite(scale)|(scale<1e-8)]=1
    label=y[good].astype(float);target_scale=float(np.std(label))
    if target_scale<1e-10: raise ValueError('degenerate ridge target')
    intercept=float(np.mean(label/target_scale))
    # Bound CUDA working memory independently of the number of minute rows.
    # Accumulate sufficient statistics; the objective remains average Gram.
    gram=np.zeros((x.shape[1],x.shape[1]),float);rhs=np.zeros(x.shape[1],float)
    for start in range(0,len(x),16384):
        valid=good[start:start+16384];sample=x[start:start+16384][valid].astype(float)
        if not len(sample):continue
        a=np.clip(np.nan_to_num((sample-center)/scale,nan=0.),-8,8)
        b=y[start:start+16384][valid]/target_scale-intercept
        t=torch.as_tensor(a,dtype=torch.float32,device=device)
        v=torch.as_tensor(b,dtype=torch.float32,device=device)
        gram+=(t.T@t).cpu().numpy().astype(float)
        rhs+=(t.T@v).cpu().numpy().astype(float)
    gram/=good.sum();rhs/=good.sum()
    coef=np.linalg.solve(gram+np.eye(x.shape[1])*penalty,rhs)
    return RidgeState(center,scale,coef,intercept,target_scale)


def _project_coordinates(x,indices,projection,device):
    if device=='cpu':return x[...,indices]@projection
    if not torch.cuda.is_available():raise ValueError('requested CUDA projection unavailable')
    flat=x.reshape(-1,x.shape[-1]);result=np.empty((len(flat),projection.shape[1]),np.float32)
    matrix=torch.as_tensor(projection,dtype=torch.float32,device=device)
    for start in range(0,len(flat),32768):
        stop=min(len(flat),start+32768)
        part=torch.as_tensor(flat[start:stop][:,indices],dtype=torch.float32,device=device)
        result[start:stop]=(part@matrix).cpu().numpy()
    return result.reshape(*x.shape[:-1],projection.shape[1])


def parameter_design(x: np.ndarray, cards: list[dict], kind: str, train: np.ndarray,
                     target: np.ndarray, seed: int,device: str='cpu') -> tuple[np.ndarray,dict]:
    old=[i for i,c in enumerate(cards) if c['family']=='OLD']
    structural=[i for i,c in enumerate(cards) if c['family'] not in {'OLD','POOL'}]
    if kind=='old': return x[...,old],{'kind':kind,'coordinates':old,'fit_parameters':len(old)+1}
    if kind=='pool':
        idx=old+[i for i,c in enumerate(cards) if c['family']=='POOL']
        return x[...,idx],{'kind':kind,'coordinates':idx,'fit_parameters':len(idx)+1,'outer_folds_reused_for_selection':True}
    if kind.startswith('seed_'):
        family=kind.removeprefix('seed_')
        candidates=[i for i,c in enumerate(cards) if c['family']=='POOL' and c.get('source_family')==family]
        if not candidates:raise ValueError(f'no declared surviving seed for {family}')
        idx=old+[candidates[0]]
        return x[...,idx],{'kind':kind,'coordinates':idx,'seed_coordinate':cards[candidates[0]]['name'],
            'fit_parameters':len(idx)+1,'outer_folds_reused_for_selection':True}
    ablations={'F3_order1':('F3',lambda c:c['order']==1),
        'F3_order2':('F3',lambda c:c['order']<=2),
        'F3_natural':('F3',lambda c:c['name'].startswith('logsig_')),
        'F3_activity':('F3',lambda c:c['name'].startswith('activity_') or c['name'].startswith('bucket_')),
        'F1_composition':('F1',lambda c:c['name'].startswith('ilr_')),
        'F2_energy':('F2',lambda c:'couple' not in c['name'] and 'signed' not in c['name']),
          'F4_occupancy':('F4',lambda c:c['name'].startswith('occupancy_')),
          'F7_raw':('F7',lambda c:c['name'].startswith('kernel_mean_')),
          'F7_reference':('F7',lambda c:c['name'].startswith('kernel_departure_')),
        'F5_diagonal':('F5',lambda c:c['name'][-2:][0]==c['name'][-2:][1])}
    if kind in ablations:
        family,test=ablations[kind];idx=old+[i for i,c in enumerate(cards) if c['family']==family and test(c)]
        return x[...,idx],{'kind':kind,'coordinates':idx,'fit_parameters':len(idx)+1,'ablation':True}
    if kind.startswith('F') and kind!='F8':
        idx=old+[i for i,c in enumerate(cards) if c['family']==kind]
        return x[...,idx],{'kind':kind,'coordinates':idx,'fit_parameters':len(idx)+1}
    rng=np.random.default_rng(seed)
    if kind=='old_nonlinear':
        left=rng.choice(old,size=192);right=rng.choice(old,size=192)
        v=np.tanh(x[...,left])*np.tanh(x[...,right])
        return np.concatenate((x[...,old],v),axis=-1),{'kind':kind,'left':left.tolist(),'right':right.tolist(),'fit_parameters':len(old)+193}
    if kind in {'joint','joint_interaction','joint_interaction_residual'}:
        projection=(rng.normal(size=(len(structural),128))/np.sqrt(len(structural))).astype(np.float32)
        v=_project_coordinates(x,structural,projection,device)
        if kind=='joint': return np.concatenate((x[...,old],v),axis=-1),{'kind':kind,'projection':projection.tolist(),'fit_parameters':len(old)+129}
        path=[i for i,c in enumerate(cards) if c['family']=='F3']; tail=[i for i,c in enumerate(cards) if c['family']=='F1']
        p=(rng.normal(size=(len(path),16))/np.sqrt(len(path))).astype(np.float32);q=(rng.normal(size=(len(tail),16))/np.sqrt(len(tail))).astype(np.float32)
        s=np.tanh(_project_coordinates(x,path,p,device));k=np.tanh(_project_coordinates(x,tail,q,device))
        st=s[train].reshape(-1,16);kt=k[train].reshape(-1,16);y=target[train].reshape(-1)
        if kind=='joint_interaction_residual':
            old_train=x[train][...,old].reshape(-1,len(old))
            nuisance=fit_ridge(old_train,y,.1,device)
            y=y-nuisance.predict(old_train)
        good=np.isfinite(y);yc=y[good]-np.mean(y[good]);
        cross=(st[good].T@(kt[good]*yc[:,None]))/good.sum()
        u,_,vt=np.linalg.svd(cross,full_matrices=False)
        interaction=(s@u[:,:2])*(k@vt.T[:,:2])
        return np.concatenate((x[...,old],v,interaction),axis=-1),{'kind':kind,'projection':projection.tolist(),
            'path_projection':p.tolist(),'tail_projection':q.tolist(),'u':u[:,:2].tolist(),'v':vt.T[:,:2].tolist(),
            'fit_parameters':len(old)+131+64+(len(old)+1 if kind=='joint_interaction_residual' else 0),'interaction_rank':2,
            'interaction_nuisance':nuisance.record() if kind=='joint_interaction_residual' else None,
            'interaction_target':'old_training_residual' if kind=='joint_interaction_residual' else 'risk_residual_target'}
    if kind=='F8':
        # State innovation learned only on training transitions; direct horizon readout.
        idx=structural[:16]; state=x[...,idx]
        previous=np.concatenate((np.zeros_like(state[:1]),state[:-1]),axis=0)
        fit=train&np.r_[False,train[:-1]]
        a=previous[fit].reshape(-1,16); b=state[fit].reshape(-1,16)
        dynamics=np.linalg.solve(a.T@a+np.eye(16)*len(a)*.1,a.T@b)
        innovation=state-previous@dynamics
        return np.concatenate((x[...,old],state,innovation),axis=-1),{'kind':kind,'state_coordinates':idx,'dynamics':dynamics.tolist(),'fit_parameters':len(old)+33+256}
    raise ValueError(f'unknown readout kind {kind}')


def fit_increment_projector(design,old_design,state,base_state,train,target,device):
    """Remove the old linear span from a new readout; all fitting is past-only."""
    raw=state.predict(design[train].reshape(-1,design.shape[-1]))-base_state.predict(old_design[train].reshape(-1,old_design.shape[-1]))
    raw[~np.isfinite(target[train].reshape(-1))]=np.nan
    projector=fit_ridge(old_design[train].reshape(-1,old_design.shape[-1]),raw,1e-4,device)
    residue=raw-projector.predict(old_design[train].reshape(-1,old_design.shape[-1]))
    e=target[train].reshape(-1)-base_state.predict(old_design[train].reshape(-1,old_design.shape[-1]))
    good=np.isfinite(e)&np.isfinite(residue)
    gamma=float(np.clip(np.dot(residue[good],e[good])/max(np.dot(residue[good],residue[good]),1e-16),0.,1.))
    return projector,gamma


def crossfit_readout(x,dates,target,cards,kind,penalty,folds,horizon,seed,device,policy='joint'):
    if policy not in {'joint','orthogonal_increment'}:raise ValueError('unknown residual readout policy')
    prediction=np.full_like(target,np.nan); states=[]; summaries=[]
    stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    fixed_design=None
    for start,end in folds:
        start_time=pd.to_datetime(start,format='%Y%m%d%H%M',utc=True)
        train=(stamp+pd.Timedelta(hours=horizon)<start_time)
        test=(dates>=start)&(dates<=end)
        if train.sum()<100 or test.sum()<10: raise ValueError('crossfit fold too small')
        if fixed_design is None or kind in {'joint_interaction','joint_interaction_residual','F8'}:
            design,metadata=parameter_design(x,cards,kind,train,target,seed,device)
            if kind not in {'joint_interaction','joint_interaction_residual','F8'}:fixed_design=(design,metadata)
        else:design,metadata=fixed_design
        state=fit_ridge(design[train].reshape(-1,design.shape[-1]),target[train].reshape(-1),penalty,device)
        test_prediction=state.predict(design[test].reshape(-1,design.shape[-1]))
        increment_metadata={}
        if policy=='orthogonal_increment' and kind!='old':
            old_design,old_meta=parameter_design(x,cards,'old',train,target,seed,device)
            base_state=fit_ridge(old_design[train].reshape(-1,old_design.shape[-1]),target[train].reshape(-1),.1,device)
            projector,gamma=fit_increment_projector(design,old_design,state,base_state,train,target,device)
            base_prediction=base_state.predict(old_design[test].reshape(-1,old_design.shape[-1]))
            test_prediction=base_prediction+gamma*(test_prediction-base_prediction-projector.predict(old_design[test].reshape(-1,old_design.shape[-1])))
            increment_metadata={'base_state':base_state.record(),'projector':projector.record(),'gamma':gamma}
            metadata={**metadata,'fit_parameters':metadata['fit_parameters']+old_design.shape[-1]+2,'residual_policy':policy}
        elif policy=='orthogonal_increment' and kind=='old':
            # The candidate is entirely in the reference linear span.
            base_state=fit_ridge(design[train].reshape(-1,design.shape[-1]),target[train].reshape(-1),.1,device)
            test_prediction=base_state.predict(design[test].reshape(-1,design.shape[-1]))
        prediction[test]=test_prediction.reshape(test.sum(),x.shape[1])
        summaries.append({'fold_start':start,'fold_end':end,'ic':correlation(prediction[test],target[test]),
            'loss':float(np.nanmean((target[test]-prediction[test])**2)),
            'target_loss':float(np.nanmean(target[test]**2)),'fit_parameters':metadata['fit_parameters']})
        states.append({'start':start,'state':state.record(),'design':metadata,'increment':increment_metadata})
    return prediction,summaries,states


def common_market_loadings(beta):
    n=beta.shape[-1]
    if n<2:raise ValueError('market loading requires multiple assets')
    return beta*(n/(n-1))/(1+beta/(n-1))


def project_basket_returns(returns,beta):
    """Symmetric projection onto exactly the executable cash/beta subspace.

    Beta is known at the forecast origin; returns can be future labels. Missing
    labels or loadings invalidate the whole fixed-universe basket at that row.
    """
    if returns.shape!=beta.shape or returns.ndim!=2:raise ValueError('invalid basket target shape')
    good=np.isfinite(returns).all(axis=1)&np.isfinite(beta).all(axis=1)
    result=np.full_like(returns,np.nan,dtype=float);r=returns[good];b=common_market_loadings(beta[good])
    centered=b-b.mean(axis=1,keepdims=True);r=r-r.mean(axis=1,keepdims=True)
    denominator=np.sum(centered**2,axis=1,keepdims=True)
    coefficient=np.divide(np.sum(r*centered,axis=1,keepdims=True),denominator,out=np.zeros((len(r),1)),where=denominator>1e-8)
    result[good]=r-coefficient*centered
    return result


def neutral_positions(prediction: np.ndarray,beta: np.ndarray,scale: float,budget: float,
                      smoothing: int,deadband: float,edge_floor_bps: float=0.,risk_mapping='legacy_loo',exit_on_inactive=False) -> np.ndarray:
    if scale<=0 or not 0<budget<=1 or smoothing<1 or deadband<0: raise ValueError('invalid frozen execution')
    if risk_mapping=='exact_loo_to_common':
        prediction=prediction/(1+beta/(beta.shape[-1]-1))
        beta=common_market_loadings(beta)
    elif risk_mapping=='common_label':beta=common_market_loadings(beta)
    elif risk_mapping!='legacy_loo':raise ValueError('invalid residual-to-basket risk mapping')
    score=np.clip(np.nan_to_num(prediction/scale,nan=0.),-3,3)
    smooth=pd.DataFrame(score).ewm(span=smoothing,adjust=False).mean().to_numpy()
    proposed=np.tanh(smooth)*budget
    if edge_floor_bps>0:
        smoothed_edge=pd.DataFrame(np.nan_to_num(prediction,nan=0.)).ewm(span=smoothing,adjust=False).mean().to_numpy()
        proposed[np.abs(smoothed_edge)<edge_floor_bps/1e4]=0
    valid=np.isfinite(beta); b=np.where(valid,beta,0)
    proposed[~valid]=0
    count=valid.sum(axis=1,keepdims=True)
    proposed-=np.divide(proposed.sum(axis=1,keepdims=True),count,out=np.zeros((len(beta),1)),where=count>=3)*valid
    bm=np.divide((b*valid).sum(axis=1,keepdims=True),count,out=np.zeros((len(beta),1)),where=count>=3)
    bc=(b-bm)*valid
    coeff=np.divide((proposed*bc).sum(axis=1,keepdims=True),(bc*bc).sum(axis=1,keepdims=True),out=np.zeros((len(beta),1)),where=(bc*bc).sum(axis=1,keepdims=True)>1e-8)
    proposed-=coeff*bc
    # Deadband determines scalar update acceptance for the whole neutral basket.
    # Retaining individual holdings independently would break the constraints.
    position=np.zeros_like(proposed); held=np.zeros(proposed.shape[1])
    for t in range(len(position)):
        if exit_on_inactive and np.max(np.abs(proposed[t]))<1e-12:
            held[:]=0
        elif np.mean(np.abs(proposed[t]-held))>=deadband:
            held=proposed[t].copy()
        else:
            # Re-hedge held cash/beta under today's risk estimate, charge changes.
            held[~valid[t]]=0
            if valid[t].sum()>=3:
                held-=np.mean(held[valid[t]])*valid[t]
                denom=np.dot(bc[t],bc[t])
                if denom>1e-8: held-=np.dot(held,bc[t])/denom*bc[t]
        gross=np.mean(np.abs(held))
        if gross>budget: held*=budget/gross
        if valid[t].sum()<3: held[:]=0
        position[t]=held
    return position


@njit(cache=True)
def _self_financing_path(proposed,rebalance,prices,funding,fee_rate):
    n,d=prices.shape; shares=np.zeros(d);cash=1.;equity=1.
    gross=np.zeros((n,d));fees=np.zeros((n,d));settled=np.zeros((n,d));
    changes=np.zeros((n,d));exposure=np.zeros((n,d));equities=np.zeros(n)
    for t in range(n):
        # Equity at this mark before settlement or new orders.
        equity=cash+np.dot(shares,prices[t])
        if equity<=0: raise ValueError('portfolio insolvency under frozen contract')
        for j in range(d):
            payment=-shares[j]*prices[t,j]*funding[t,j]
            cash+=payment;settled[t,j]=payment/equity*d
        if rebalance[t] or t==n-1:
            current=cash+np.dot(shares,prices[t])
            for j in range(d):
                desired=0. if t==n-1 else proposed[t,j]*current/d/prices[t,j]
                dollars=(desired-shares[j])*prices[t,j]
                fee=abs(dollars)*fee_rate
                cash-=dollars+fee;shares[j]=desired
                fees[t,j]=fee/equity*d;changes[t,j]=abs(dollars)/equity*d
        for j in range(d):
            exposure[t,j]=shares[j]*prices[t,j]/equity*d
            if t<n-1:gross[t,j]=shares[j]*(prices[t+1,j]-prices[t,j])/equity*d
        equities[t]=equity
    return gross,fees,settled,changes,exposure,equities


def five_minute_ledger(position,dates,fast_dates,fast_close,funding,cost_bps,split: Split):
    """Self-financing shares/cash, 15m orders, funding on previous shares.

    Exposures drift between orders. Rebalances and terminal liquidation pay fees.
    Net simple return is actual wealth change divided by pre-settlement equity.
    """
    loc=np.searchsorted(dates,fast_dates,side='right')-1
    holding=np.zeros_like(fast_close); good=loc>=0;holding[good]=position[loc[good]]
    rebalance=np.r_[True,loc[1:]!=loc[:-1]]
    price,fee,cash,change,actual,equity=_self_financing_path(holding,rebalance,fast_close,funding,cost_bps/1e4)
    net=price+cash-fee
    period=np.where(fast_dates<=split.discovery_end,'discovery',np.where(fast_dates<=split.validation_end,'validation','holdout'))
    frame=pd.DataFrame({'completed_5m':fast_dates,'period':period,'price_gross':price.mean(axis=1),
        'funding':cash.mean(axis=1),'fee':fee.mean(axis=1),'net':net.mean(axis=1),
        'turnover':change.mean(axis=1),'mean_abs_position':np.abs(actual).mean(axis=1),
        'cash_exposure':actual.mean(axis=1),'equity_before_cashflows':equity})
    return frame,net,actual


def ledger_metrics(frame: pd.DataFrame,per_asset: np.ndarray,mask: np.ndarray) -> dict:
    f=frame.loc[mask]; r=f.net.to_numpy(); wealth=np.cumprod(1+r)
    daily=pd.Series(r,index=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M',utc=True)).resample('D').sum()
    vol=daily.std(ddof=0)
    return {'net_return_pct':float((wealth[-1]-1)*100) if len(wealth) else 0.,
        'net_sum_pct':float(r.sum()*100),'price_gross_pct':float(f.price_gross.sum()*100),
        'funding_pct':float(f.funding.sum()*100),'fee_pct':float(f.fee.sum()*100),
        'sharpe':float(daily.mean()/vol*np.sqrt(365.25)) if vol>1e-12 else 0.,
        'max_drawdown_pct':float(np.min(wealth/np.maximum.accumulate(np.r_[1,wealth])[1:]-1)*100) if len(wealth) else 0.,
        'mean_abs_position':float(f.mean_abs_position.mean()),'turnover':float(f.turnover.sum()),
        'positive_asset_share':float(np.mean(per_asset[mask].sum(axis=0)>0)),
        'break_even_bps':float((f.price_gross.sum()+f.funding.sum())/max(f.turnover.sum(),1e-10)*1e4)}
