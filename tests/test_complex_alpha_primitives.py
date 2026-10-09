import numpy as np
import pandas as pd
import pytest
from src.alpha_mvp.research.complex_alpha_primitives import (rolling_sum,causal_standardize,
    signed_log_bins,ilr_counts,tilt_partition,causal_haar,window_logsignatures,
    lie_basis,transition_flux,matrix_log_spd,relative_covariance_log)
from src.alpha_mvp.research.complex_alpha_search import generate_candidates,evaluate_candidate,_torch_candidates
from src.alpha_mvp.research.complex_alpha_evaluation import (fit_ridge,five_minute_ledger,targets_from_prices)
from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_data import minute_observations,closed_activity_buckets
import torch


def test_full_support_and_missing_not_zero():
    x=np.array([1.,2.,np.nan,4.,5.]);r=rolling_sum(x,2)
    np.testing.assert_allclose(r,[np.nan,3,np.nan,np.nan,9],equal_nan=True)
    assert np.isnan(causal_haar(np.ones(8),2)[:3]).all()
    np.testing.assert_allclose(causal_haar(np.ones(8),2)[3:],0)


def test_normalization_causality_units_and_extreme():
    x=np.arange(1.,101);a,clip=causal_standardize(x,16,8)
    b,_=causal_standardize(x*100,16,8)
    np.testing.assert_allclose(a,b,equal_nan=True)
    y=x.copy();y[-1]=1e20;c,clip=causal_standardize(y,16,8)
    np.testing.assert_allclose(a[:-1],c[:-1],equal_nan=True);assert c[-1]==8 and clip[-1]
    assert np.isnan(causal_standardize(np.ones(100),16,8)[0]).all()


def test_bins_ilr_and_exponential_stability():
    np.testing.assert_array_equal(signed_log_bins(np.array([0,.25,.5,8,-.25,-8,np.nan])),[0,1,2,6,7,12,-1])
    np.testing.assert_allclose(ilr_counts(np.ones((1,13))),0,atol=1e-12)
    assert np.isnan(ilr_counts(np.zeros((1,13)))).all()
    k,n,m=tilt_partition(np.ones((20,2))*1e3,np.array([.5,.5]))
    assert abs(k-1000)<1e-9 and abs(n-20)<1e-8 and abs(m-.05)<1e-9
    assert tilt_partition(np.ones((4,2)),np.zeros(2))[0]==pytest.approx(0)
    assert np.isnan(tilt_partition(np.full((4,2),np.nan),np.ones(2))[0])


def test_logsignature_straight_line_rectangle_and_basis():
    assert len(lie_basis(4,2)[0])==6 and len(lie_basis(4,3)[0])==20
    straight=np.tile([1.,2.],(10,1))
    s=window_logsignatures(straight,np.array([10]),10)[0]
    np.testing.assert_allclose(s[:2],[10,20]);np.testing.assert_allclose(s[2:],0,atol=1e-10)
    rectangle=np.array([[1.,0],[0,1],[-1,0],[0,-1]])
    a=window_logsignatures(rectangle,np.array([4]),4)[0]
    b=window_logsignatures(rectangle[::-1],np.array([4]),4)[0]
    assert a[2]==pytest.approx(1) and b[2]==pytest.approx(-1)
    broken=rectangle.copy();broken[1,0]=np.nan
    assert np.isnan(window_logsignatures(broken,np.array([4]),4)).all()


def test_logsignature_chunk_and_rolling_reference():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,4))
    result=window_logsignatures(x,np.arange(20,101,10),20,chunk_size=27)
    for j,end in enumerate(range(20,101,10)):
        reference=window_logsignatures(x[end-20:end],np.array([20]),20)
        np.testing.assert_allclose(result[j],reference[0],atol=1e-9)


def test_transition_reverse_and_spd_geometry():
    x=np.array([0,1,2,0,1,2,0]);a,d=transition_flux(x,3);b,e=transition_flux(x[::-1],3)
    np.testing.assert_allclose(a,-b);assert d==pytest.approx(e)
    m=np.diag([1.,4.,9]);np.testing.assert_allclose(matrix_log_spd(m),np.diag(np.log([1,4,9])))
    np.testing.assert_allclose(relative_covariance_log(m,m),0,atol=1e-10)
    with pytest.raises(ValueError):matrix_log_spd(np.diag([0.,1]))
    q,_=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))
    np.testing.assert_allclose(matrix_log_spd(q@m@q.T),q@np.diag(np.log([1,4,9]))@q.T,atol=1e-12)


def test_fractional_kernel_binomial_reference_and_boundary():
    from src.alpha_mvp.research.complex_alpha_primitives import fractional_kernel
    from scipy.special import binom
    for d in [0.,.25,.5,1.]:
        weights=fractional_kernel(d,32);k=np.arange(32)
        np.testing.assert_allclose(weights,(-1.)**k*binom(d,k),atol=1e-14)
    with pytest.raises(ValueError):fractional_kernel(-.1,32)


def test_candidate_determinism_and_gpu_cpu_reference():
    cards=[{'family':'F1','support_minutes':100,'name':str(i)} for i in range(12)]
    a=generate_candidates(cards,{'F1':30},5,['linear','tanh_product','asinh_difference','signed_energy','cubic'])
    b=generate_candidates(cards,{'F1':30},5,['linear','tanh_product','asinh_difference','signed_energy','cubic'])
    assert a==b and len(set(c.id for c in a))==30
    x=np.random.default_rng(6).normal(size=(100,12)).astype(np.float32)
    result=_torch_candidates(torch.tensor(x),a).numpy()
    np.testing.assert_allclose(result,np.column_stack([evaluate_candidate(x,c) for c in a]),atol=3e-6)
    from pathlib import Path
    import yaml
    grammar=yaml.safe_load(Path('configs/research/complex_alpha_grammar_20261008.yaml').read_text())
    assert max(c.nodes for c in a)<=grammar['complexity']['max_bookkeeping_nodes']
    assert set(grammar['templates'])==set(c.mode for c in a)


def test_self_financing_funding_prior_position_and_closing_fee():
    dates=np.array(['202301010015','202301010030','202301010045'])
    fast=np.array(['202301010015','202301010020','202301010025','202301010030','202301010035','202301010040','202301010045'])
    close=np.tile(np.arange(100.,107)[:,None],(1,4));p=np.tile([.4,-.4,0.,0.],(3,1))
    funding=np.zeros_like(close);funding[0,0]=.01;funding[1,0]=.01
    split=Split('202301010025','202301010030','202301010040','202301010045')
    f,net,_=five_minute_ledger(p,dates,fast,close,funding,4.,split)
    assert f.funding.iloc[0]==0 and f.funding.iloc[1]<0
    np.testing.assert_allclose(f.net,f.price_gross+f.funding-f.fee)
    # Product of returns must agree with internal pre-cashflow equity trajectory.
    wealth=np.cumprod(1+f.net.to_numpy())
    np.testing.assert_allclose(wealth[:-1],f.equity_before_cashflows.to_numpy()[1:],rtol=1e-10)
    assert f.fee.iloc[-1]>0


def test_holdout_targets_not_read_before_audit():
    fast=pd.date_range('2023-01-01',periods=2000,freq='5min').strftime('%Y%m%d%H%M').to_numpy()
    dates=fast[::3];prices=np.exp(np.arange(2000)[:,None]*.0001)*np.ones((1,4))
    split=Split(fast[800],fast[801],fast[1400],fast[1401]);beta=np.ones((len(dates),4))
    raw,_=targets_from_prices(dates,fast,prices,[4],split,beta,np.zeros_like(prices))
    assert np.isnan(raw[4][dates>=split.holdout_start]).all()
    changed=prices.copy();changed[fast>=split.holdout_start]*=100
    again,_=targets_from_prices(dates,fast,changed,[4],split,beta,np.zeros_like(prices))
    np.testing.assert_allclose(raw[4],again[4],equal_nan=True)


def test_ridge_nan_and_known_nonlinear_target():
    rng=np.random.default_rng(3);x=rng.normal(size=(1000,3));y=2*x[:,0]-x[:,2];x[4,1]=np.nan
    model=fit_ridge(x,y,.001)
    assert np.mean((model.predict(x)-y)**2)<.01


def test_chunked_ridge_matches_average_gram_reference():
    rng=np.random.default_rng(666);x=rng.normal(size=(17001,4));y=.8*x[:,0]+.3*x[:,1]+rng.normal(0,.1,len(x))
    model=fit_ridge(x,y,.1);z=(x-x.mean(axis=0))/x.std(axis=0);target=(y-y.mean())/y.std()
    expected=np.linalg.solve(z.T@z/len(z)+np.eye(4)*.1,z.T@target/len(z))
    np.testing.assert_allclose(model.coef,expected,atol=2e-6)


def test_minute_fields_missing_extreme_and_quote_direction():
    n=300;rng=np.random.default_rng(4);close=100*np.exp(np.cumsum(rng.normal(0,.001,n)));q=rng.uniform(1000,10000,n)
    frame=pd.DataFrame({'close':close,'high':close*1.001,'low':close*.999,'volume':q/close,
        'quote_volume':q,'trade_count':rng.integers(3,20,n),'taker_buy_quote_volume':q*.8})
    frame.loc[200,'taker_buy_quote_volume']=-1;frame.loc[201,'quote_volume']=np.nan
    frame.loc[202,'trade_count']=0;frame.loc[299,'quote_volume']=1e30
    z,quality=minute_observations(frame,rng.normal(0,.001,n),64,16)
    assert np.isnan(z[200,1]) and np.isnan(z[201,2]) and np.isnan(z[202,4])
    assert quality['amount']['clipped_share']>0
    before=frame.copy();before.loc[299,'quote_volume']=1000
    a,_=minute_observations(before,rng.normal(0,.001,n),64,16)
    # Raw quote amount may change only the current normalized amount.
    np.testing.assert_allclose(z[:-1,2],a[:-1,2],equal_nan=True)


def test_activity_buckets_closed_and_whole_minutes_only():
    z=np.ones((20,3));q=np.ones(20)*6;threshold=np.ones(20)*10
    values,times,duration,excess=closed_activity_buckets(z,q,threshold)
    np.testing.assert_array_equal(times,np.arange(2,21,2));np.testing.assert_array_equal(duration,2)
    np.testing.assert_allclose(excess,.2)
    future=q.copy();future[-1]=100
    other=closed_activity_buckets(z,future,threshold)
    np.testing.assert_allclose(values[:-1],other[0][:-1])


def test_log_age_full_support_and_future_causality():
    from src.alpha_mvp.research.complex_alpha_extension import log_age_coordinates
    rng=np.random.default_rng(33);z=rng.normal(size=(900,9));ends=np.arange(300,901,15)
    a,cards=log_age_coordinates(z,ends,[[0,30],[30,120]])
    changed=z.copy();changed[800:]=1e3
    b,_=log_age_coordinates(changed,ends,[[0,30],[30,120]])
    np.testing.assert_allclose(a[ends<=800],b[ends<=800],equal_nan=True)
    assert any('relative' in c['name'] for c in cards)
    broken=z.copy();broken[799,1]=np.nan
    missing,_=log_age_coordinates(broken,np.array([810]),[[0,30]])
    indices=[i for i,c in enumerate(cards[:a.shape[1]]) if c['name'].startswith('age_ilr_0_30')]
    assert np.isnan(missing[0,indices]).all()


def test_old_span_removed_but_nonlinear_increment_retained():
    from src.alpha_mvp.research.complex_alpha_evaluation import fit_increment_projector
    rng=np.random.default_rng(13);x=rng.normal(size=(800,4,2));train=np.ones(800,bool)
    nonlinear=x[...,0]*x[...,1];y=x[...,0]+.7*nonlinear
    old=fit_ridge(x.reshape(-1,2),y.ravel(),.1)
    design=np.concatenate((x,nonlinear[...,None]),axis=-1)
    state=fit_ridge(design.reshape(-1,3),y.ravel(),.01)
    projector,gamma=fit_increment_projector(design,x,state,old,train,y,'cpu')
    residue=state.predict(design.reshape(-1,3))-old.predict(x.reshape(-1,2))-projector.predict(x.reshape(-1,2))
    assert gamma>.9 and abs(np.corrcoef(residue,x[...,0].ravel())[0,1])<.002
    assert np.corrcoef(residue,nonlinear.ravel())[0,1]>.99
    linear=fit_ridge(x.reshape(-1,2),y.ravel(),1.)
    p,_=fit_increment_projector(x,x,linear,old,train,y,'cpu')
    error=linear.predict(x.reshape(-1,2))-old.predict(x.reshape(-1,2))-p.predict(x.reshape(-1,2))
    assert np.std(error)<1e-4


def test_risk_scaling_and_oof_calibration_do_not_read_future():
    from src.alpha_mvp.research.complex_alpha_evaluation import causal_risk_scales
    from src.alpha_mvp.research.complex_alpha_workflow import _calibrate_oof
    rng=np.random.default_rng(13);p=100*np.exp(np.cumsum(rng.normal(0,.001,(1000,4)),axis=0));beta=np.ones_like(p)
    a=causal_risk_scales(p,beta,[4],32,16)[4];q=p.copy();q[800:]*=100
    b=causal_risk_scales(q,beta,[4],32,16)[4]
    np.testing.assert_allclose(a[:801],b[:801],equal_nan=True)
    dates=pd.date_range('2023-01-01',periods=1000,freq='15min').strftime('%Y%m%d%H%M').to_numpy()
    pred=rng.normal(size=(1000,4));base=np.zeros_like(pred);target=pred*.4
    folds=[(dates[200],dates[399]),(dates[400],dates[599]),(dates[600],dates[799])]
    c,meta=_calibrate_oof(pred,base,target,dates,folds,4)
    target[600:]=pred[600:]*10
    d,again=_calibrate_oof(pred,base,target,dates,folds,4)
    np.testing.assert_allclose(c[:800],d[:800]);assert meta[0]['gamma']==pytest.approx(.4)


def test_structural_counterfactual_preserves_marginals_changes_order():
    from src.alpha_mvp.research.complex_alpha_structure_probe import perturb_window,probe_coordinates
    rng=np.random.default_rng(90);z=rng.normal(size=(720,9));a=probe_coordinates(z)
    b=probe_coordinates(perturb_window(z,'time_order',rng))
    for name in ['ordinary_marginals','ordinary_covariance','path_order1','conditional_tilt']:
        np.testing.assert_allclose(a[name],b[name],atol=1e-12,equal_nan=True)
    assert np.linalg.norm(a['path_order2']-b['path_order2'])>.01
    c=probe_coordinates(perturb_window(z,'channel_alignment',rng))
    np.testing.assert_allclose(a['ordinary_marginals'],c['ordinary_marginals'],atol=1e-12)
    assert np.linalg.norm(a['ordinary_covariance']-c['ordinary_covariance'])>.01


def test_supervised_interaction_residual_target_and_fold_isolation():
    from src.alpha_mvp.research.complex_alpha_evaluation import parameter_design
    rng=np.random.default_rng(440);x=rng.normal(size=(800,4,10)).astype(np.float32)
    cards=[{'family':'OLD' if i<2 else ('F3' if i<6 else 'F1'),'name':str(i)} for i in range(10)]
    y=4*x[...,0]+np.tanh(x[...,2])*np.tanh(x[...,6]);train=np.arange(800)<600
    a,m=parameter_design(x,cards,'joint_interaction_residual',train,y,13)
    changed=y.copy();changed[~train]+=1000
    b,n=parameter_design(x,cards,'joint_interaction_residual',train,changed,13)
    np.testing.assert_allclose(a,b,atol=1e-7)
    assert m['interaction_target']=='old_training_residual' and m['interaction_rank']==2
    assert m['u']==n['u'] and m['fit_parameters']==200


def test_native_label_fixed_shares_funding_and_exact_loo_identity():
    from src.alpha_mvp.research.complex_alpha_evaluation import common_market_loadings,neutral_positions
    fast=pd.date_range('2023-01-01',periods=80,freq='5min').strftime('%Y%m%d%H%M').to_numpy()
    dates=fast[::3];prices=np.tile(np.linspace(100.,140.,80)[:,None],(1,4));funding=np.zeros_like(prices)
    funding[10,0]=.01;funding[0,0]=1
    split=Split(fast[60],fast[61],fast[70],fast[71]);beta=np.ones((len(dates),4))
    raw,_=targets_from_prices(dates,fast,prices,[4],split,beta,funding,return_contract='native_simple')
    expected=prices[48,0]/prices[0,0]-1-.01*prices[10,0]/prices[0,0]
    assert raw[4][0,0]==pytest.approx(expected)
    rng=np.random.default_rng(911);b=rng.uniform(.1,2.5,(100,12));pred=rng.normal(0,.01,b.shape)
    pos=neutral_positions(pred,b,.01,.5,1,0,0,'exact_loo_to_common')
    g=common_market_loadings(b);np.testing.assert_allclose(np.sum(pos*g,axis=1),0,atol=1e-12)
    np.testing.assert_allclose(np.sum(pos,axis=1),0,atol=1e-12)
    r=rng.normal(0,.03,b.shape);loo=(r.sum(axis=1,keepdims=True)-r)/11;residue=r-b*loo
    np.testing.assert_allclose(np.sum(pos*r,axis=1),np.sum(pos*residue/(1+b/11),axis=1),atol=1e-12)


def test_tradable_target_projection_duality_and_missing():
    from src.alpha_mvp.research.complex_alpha_evaluation import project_basket_returns,common_market_loadings,neutral_positions
    rng=np.random.default_rng(312);beta=rng.uniform(.2,2.5,(100,12));r=rng.normal(size=beta.shape);s=rng.normal(size=beta.shape)
    e=project_basket_returns(r,beta);ps=project_basket_returns(s,beta)
    np.testing.assert_allclose(e.sum(axis=1),0,atol=1e-12)
    np.testing.assert_allclose((e*common_market_loadings(beta)).sum(axis=1),0,atol=1e-12)
    np.testing.assert_allclose(project_basket_returns(e,beta),e,atol=1e-12)
    np.testing.assert_allclose((ps*r).sum(axis=1),(s*e).sum(axis=1),atol=1e-12)
    r[50,1]=np.nan;assert np.isnan(project_basket_returns(r,beta)[50]).all()
    positions=neutral_positions(s,beta,1.,.5,1,0,0,'common_label')
    np.testing.assert_allclose((positions*common_market_loadings(beta)).sum(axis=1),0,atol=1e-12)


def test_inactive_signal_closes_small_position_below_deadband():
    from src.alpha_mvp.research.complex_alpha_evaluation import neutral_positions
    a=np.array([1.,-1,1,-1]);v=np.array([1.,1,-1,-1]);b=np.vstack((np.ones(4),1+.3*a+.2*v,1+.3*a+.2*v))
    forecast=np.vstack((np.arctanh(.6)*a,np.arctanh(.6)*a,np.zeros(4)))
    old=neutral_positions(forecast,b,1.,.5,1,.2)
    fixed=neutral_positions(forecast,b,1.,.5,1,.2,exit_on_inactive=True)
    assert np.mean(np.abs(old[1]))<.2 and np.mean(np.abs(old[2]))>.1
    np.testing.assert_allclose(fixed[:2],old[:2]);np.testing.assert_allclose(fixed[2],0)


def test_gaussian_kernel_reference_analytic_missing_and_extreme():
    from src.alpha_mvp.research.complex_alpha_kernel_reference import gaussian_fourier_reference,joint_covariance_controls
    rng=np.random.default_rng(872);omega=rng.normal(size=(4,32));phase=rng.uniform(0,2*np.pi,32)
    mean=np.ones((1,4))*.2;cov=np.zeros((1,4,4))
    deterministic=gaussian_fourier_reference(mean,cov,omega,phase)
    np.testing.assert_allclose(deterministic,np.sqrt(2/32)*np.cos(mean@omega+phase))
    cov[0]=np.eye(4)*.2;samples=rng.multivariate_normal(mean[0],cov[0],size=50000)
    empirical=(np.sqrt(2/32)*np.cos(samples@omega+phase)).mean(axis=0)
    np.testing.assert_allclose(gaussian_fourier_reference(mean,cov,omega,phase)[0],empirical,atol=.002)
    cov[0]*=1e20;assert np.isfinite(gaussian_fourier_reference(mean,cov,omega,phase)).all()
    cov[0,0,0]=np.nan;assert np.isnan(gaussian_fourier_reference(mean,cov,omega,phase)).all()
    z=rng.normal(size=(100,9));v,c=joint_covariance_controls(z,np.array([100]),[30])
    assert v[0,0]==pytest.approx(np.cov(z[-30:,:2],rowvar=False,ddof=0)[0,1],abs=1e-7)
    z[90,0]=np.nan;w,_=joint_covariance_controls(z,np.array([100]),[30]);assert np.isnan(w[0,0])


def test_bounded_bank_loader_matches_full_stack_reference(tmp_path):
    from src.alpha_mvp.research.complex_alpha_workflow import _load_bank
    dates=pd.date_range('2023-01-01',periods=17000,freq='15min').strftime('%Y%m%d%H%M').to_numpy().astype('U12')
    np.save(tmp_path/'dates.npy',dates);np.save(tmp_path/'fast_dates.npy',dates);np.save(tmp_path/'fast_close.npy',np.ones((len(dates),2)))
    rng=np.random.default_rng(842);expected=[]
    for code in ['a','b']:
        raw=rng.normal(size=(len(dates),3)).astype(np.float32);raw[:,2]=1;raw[5,0]=np.nan;np.save(tmp_path/f'{code}.npy',raw)
        sample=raw[:1000];m=np.nanmean(sample,axis=0);s=np.nanstd(sample,axis=0);m[2]=0;s[2]=1
        n=np.clip(np.nan_to_num((raw-m)/s,nan=0),-8,8);n[:,2]=0;expected.append(n)
    actual,*_= _load_bank({'cache_dir':str(tmp_path),'codes':['a','b'],'coordinates':[{}, {}, {}]},dates[999])
    np.testing.assert_array_equal(actual,np.stack(expected,axis=1))
