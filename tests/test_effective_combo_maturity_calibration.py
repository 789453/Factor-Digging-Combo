import numpy as np
import pandas as pd
from src.alpha_mvp.research.effective_combo_maturity_calibration import causal_skill_states,smooth_targets


def test_negative_skill_is_learned_from_mature_past_without_current_outcomes():
    rng=np.random.default_rng(57);dates=pd.date_range('2023-01-01',periods=600,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    p=rng.normal(size=(600,4))*.005;y=-.5*p
    policy=dict(windows_months=[1,3,6],training_floor_bps=16,minimum_labels=100,skill_deadzone=.05)
    periods=[['202401010000','202404302355']]
    cal,score,r=causal_skill_states(p,y,dates,periods,4,policy)
    month=(dates>='202401010000')&(dates<'202402010000')
    np.testing.assert_allclose(cal[3][month],-.5*p[month]);np.testing.assert_allclose(score[3][month],-p[month])
    altered=y.copy();altered[dates>='202401010000']*=100
    ca,sc,_=causal_skill_states(p,altered,dates,periods,4,policy)
    np.testing.assert_allclose(cal[3][month],ca[3][month]);np.testing.assert_allclose(score[3][month],sc[3][month])
    assert all(v['last_eligible_origin']<v['update'] for v in r)


def test_score_deadzone_and_smoothing_match_explicit_reference():
    target=np.array([[.4,-.3],[.2,-.1],[0.,0.],[-.3,.2],[.1,-.2]])
    actual=smooth_targets(target,3,.05);mean=target[0].copy();held=np.zeros(2);rows=[]
    for i,row in enumerate(target):
        if i:mean=.5*row+.5*mean
        held=np.where(abs(mean-held)>=.05,mean,held);rows.append(held.copy())
    np.testing.assert_allclose(actual,rows)
    np.testing.assert_allclose(actual[:3],smooth_targets(target[:3],3,.05))
    assert np.max(abs(actual))<=np.max(abs(target))
