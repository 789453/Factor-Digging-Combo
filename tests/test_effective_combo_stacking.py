import numpy as np
import pandas as pd
import pytest
from src.alpha_mvp.research.effective_combo_stacking import predictive_weights,causal_stacking


def test_prediction_scale_optimizer_prefers_useful_member_and_prior_limits_overfit():
    rng=np.random.default_rng(33);y=rng.normal(size=4000)*.01
    p=np.column_stack([y+rng.normal(size=len(y))*.0001,rng.normal(size=len(y))*.01])
    w,n=predictive_weights(p,y,[.5,.5],.25,.75,500)
    assert n==4000 and .5<w[0]<=.625+1e-8 and w.sum()==pytest.approx(1)
    assert np.mean((p@w-y)**2)<np.mean((p.mean(axis=1)-y)**2)
    with pytest.raises(ValueError,match='prior'):predictive_weights(p,y,[1,-1],.25,.75,500)


def test_shared_coin_stacking_respects_future_label_firewall_and_convex_budget():
    rng=np.random.default_rng(34);dates=pd.date_range('2023-01-01',periods=600,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    y=rng.normal(size=(600,4))*.01;parts=[y+rng.normal(size=y.shape)*v for v in [.0001,.01,.02,.03]]
    policy=dict(prior=[.4,.35,.2,.05],diagonal_shrink=.25,prior_shrink=.75,minimum_labels=100,asset_share=.5)
    periods=[['202401010000','202404302355']]
    a,b,r=causal_stacking(parts,y,dates,periods,4,policy)
    altered=y.copy();altered[dates>='202401010000']*=100
    aa,bb,_=causal_stacking(parts,altered,dates,periods,4,policy)
    month=(dates>='202401010000')&(dates<'202402010000')
    np.testing.assert_allclose(a[month],aa[month]);np.testing.assert_allclose(b[month],bb[month])
    for record in r:
        weights=[v for k,v in record.items() if k.startswith('weight_')]
        assert sum(weights)==pytest.approx(1) and min(weights)>=0
    low=np.minimum.reduce(parts);high=np.maximum.reduce(parts)
    assert np.all(b[month]>=low[month]-1e-12) and np.all(b[month]<=high[month]+1e-12)
