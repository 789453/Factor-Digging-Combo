import numpy as np
import pandas as pd
import yaml

from src.alpha_mvp.research.aligned_contract import (
    Split, align_hourly_to_5m, completed_hour, exact_pnl, labels,
    return_targets, signal_transform)
from src.alpha_mvp.research.aligned_workflow import _records, _risk_targets, _validate
from src.alpha_mvp.research.aligned_workflow import _role_diversity_pick
from src.alpha_mvp.research.aligned_combo import (
    scheduled_risk_budget, hourly_positions, fit_candidate, forecast)
from src.alpha_mvp.research.native5_aligned import fixed_cadence_positions
from src.alpha_mvp.research.aligned_screen import score_one, phase_rotating_coarse_mask
from src.alpha_mvp.research.templates import generate_expressions, load_template_families


def test_hour_completion_and_no_cross_period_future():
    raw=np.array(['202506302200','202506302300','202507010000'])
    completed=completed_hour(raw)
    assert completed.tolist()==['202506302300','202507010000','202507010100']
    fast=np.array(['202506302300','202506302305','202507010000','202507010005','202507010100'])
    assert align_hourly_to_5m(completed,fast).tolist()==[0,2,4]
    close=np.log(np.array([100.,101.,102.,103.,104.]))[:,None]
    periods=np.array([0,0,1,1,1])
    result=return_targets(close,np.array([0,2]),(2,),periods)[2]
    assert np.isnan(result[0,0])
    assert np.isclose(result[1,0],np.log(104/102))


def test_relative_target_removes_discovery_beta_market():
    raw=np.array([[.01,.02,-.01,.03],[.04,.03,.05,.02]])
    out=labels(raw,np.ones(4))
    assert np.allclose(out['market'],raw.mean(axis=1))
    assert np.allclose(out['relative'][0,0],.01-(.02-.01+.03)/3)


def test_gate_inactive_zero_and_exact_next_bar_cost():
    values=np.array([[0.,2.,np.nan],[-2.,0.,1.]])
    transformed=signal_transform(values,1.,1.,True)
    assert transformed[0,0]==transformed[1,1]==transformed[0,2]==0
    p=np.array([[.5,-.5],[.5,0.]])
    next_r=np.array([[.01,-.02],[.02,.01]])
    result=exact_pnl(p,next_r,10.)
    assert np.isclose(result['gross'][0,0],.005)
    assert np.isclose(result['fee'][0,0],.0005)
    assert np.isclose(result['fee'][1,1],.0005)


def test_risk_label_is_future_market_downside_only():
    close=np.array([[100.,100.],[90.,90.],[80.,80.],[100.,100.]])
    targets=_risk_targets(close,np.array([0]),(2,),np.array([0,0,0,0]))
    expected=(np.log(.9)**2+np.log(80/90)**2)*1e6
    assert np.isclose(targets[2][0],expected,rtol=1e-5)


def test_bounded_template_generation_is_deterministic():
    cfg=yaml.safe_load(open('configs/research/smoke_crypto_aligned_20260930.yaml',encoding='utf8'))
    _validate(cfg)
    records,roles,_=_records(cfg)
    assert len(records)==180
    assert len(set(x.expr_hash for x in records))==180
    assert sum(role=='conditional_alpha' for role in roles.values())==60
    assert all('Gate' in x.expr or 'Mul($us_' in x.expr for x in records if roles[x.expr_hash]=='conditional_alpha')


def test_reserved_screen_survivors_do_not_compare_unlike_role_ic():
    cfg=yaml.safe_load(open('configs/research/smoke_crypto_aligned_20260930.yaml',encoding='utf8'))
    rows=[]
    for role,score in [('alpha',.02),('conditional_alpha',.01),('beta',.10),('risk',.70)]:
        for i in range(40):
            rows.append({'expr_hash':f'{role}-{i}','candidate_id':f'{role}-{i}:12',
                         'role':role,'status':'OK',
                         'coarse_score':score+i/100000,'template_name':role,
                         'fields':f'field{i%12}','horizon_bars':12 if i%2 else 48})
    picked=_role_diversity_pick(pd.DataFrame(rows),80,cfg,5)
    assert set(picked.role)=={'alpha','conditional_alpha','beta','risk'}
    assert len(picked)==80


def test_relative_fee_gate_keeps_realized_basket_neutral():
    dates=pd.date_range('2023-01-01',periods=48,freq='h',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    rng=np.random.default_rng(14)
    pred=rng.normal(size=(48,4)).astype(np.float32)*.002
    pos=hourly_positions(pred,dates,12,.002,.7,True,4.)
    assert np.max(np.abs(pos.mean(axis=1)))<1e-6


def test_declared_horizon_expires_without_renewed_edge():
    dates=pd.date_range('2023-01-01',periods=24,freq='h',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    pred=np.zeros((24,4),dtype=np.float32)
    pred[0]=[.02,-.02,.02,-.02]
    hour=hourly_positions(pred,dates,48,.02,.7,True,4.)
    assert np.any(hour[:4]) and np.all(hour[4:]==0)
    native=fixed_cadence_positions(pred,3,.02,.2,4.)
    assert np.any(native[:3]) and np.all(native[3:]==0)


def test_risk_budget_uses_discovery_scale_and_four_hour_decisions():
    dates=pd.date_range('2023-01-01',periods=240,freq='h',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    pred=np.arange(240,dtype=float)
    budget=scheduled_risk_budget(pred,dates,np.ones(240,dtype=bool),.5)
    for start in range(0,240,4):
        assert np.all(budget[start:start+4]==budget[start])


def test_native5_positions_follow_declared_cadence_and_remain_neutral():
    pred=np.random.default_rng(5).normal(0,.002,size=(60,4)).astype(np.float32)
    pos=fixed_cadence_positions(pred,3,.002,.2,4.)
    assert np.max(np.abs(pos.mean(axis=1)))<1e-6
    assert np.array_equal(pos[0::3],pos[1::3])
    assert np.array_equal(pos[0::3],pos[2::3])


def test_alpha_screen_scores_the_traded_basket_not_only_time_ic():
    rng=np.random.default_rng(8)
    common=rng.normal(size=(500,1))
    relative=rng.normal(size=(500,4))
    relative-=relative.mean(axis=1,keepdims=True)
    values=10*common+relative
    metric=score_one(values,relative,np.ones(500,dtype=bool),'alpha',False)
    assert metric['status']=='OK'
    assert metric['basket_ic']>.99
    assert metric['direction']==1


def test_alpha_calibration_removes_common_signal_before_estimating_payoff():
    rng=np.random.default_rng(11)
    relative=rng.normal(0,1,size=(1000,4))
    relative-=relative.mean(axis=1,keepdims=True)
    values=10*rng.normal(size=(1000,1))+relative
    model=fit_candidate(values,relative,np.ones(1000,dtype=bool),'alpha',False,1)
    pred=forecast(values,model)
    assert np.max(np.abs(pred.mean(axis=1)))<1e-5
    assert np.mean((pred-relative)**2)<.05*np.mean(relative**2)


def test_coarse_sample_rotates_through_all_intraday_phases():
    hour=phase_rotating_coarse_mask(np.ones(24*6,dtype=bool),24,6)
    hour_indices=np.flatnonzero(hour)
    assert len(hour_indices)==24
    assert set(hour_indices%6)==set(range(6))
    fast=phase_rotating_coarse_mask(np.ones(288*12,dtype=bool),288,12)
    assert set(np.flatnonzero(fast)%12)==set(range(12))


def test_declared_raw_field_template_has_bounded_deterministic_attribution():
    cfg=yaml.safe_load(open('configs/research/smoke_crypto_aligned_20260930.yaml',encoding='utf8'))
    families,_=load_template_families('configs/research/templates_crypto_raw_field_baselines_v1.yaml')
    def make():
        return generate_expressions(cfg['search']['fields'],cfg['search']['windows'],families,
            max_expressions=len(cfg['search']['fields']),
            max_per_family=len(cfg['search']['fields']),seed=20260930,diversity_share=.8)
    first,second=make(),make()
    assert len(first)==len(cfg['search']['fields'])
    assert [x.expr_hash for x in first]==[x.expr_hash for x in second]
    assert all(x.template_name=='raw_field_baseline' and x.nodes==1 for x in first)
