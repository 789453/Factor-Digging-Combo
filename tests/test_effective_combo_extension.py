import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.alpha_mvp.research.effective_combo_extension import expand_pool,shrunk_error_weights,causal_weighted_average,asset_shrunk_predictions
from src.alpha_mvp.research.effective_combo_extension_workflow import validate_extension_config,protocol


def test_expansion_preserves_original_prefix_and_ignores_future_correlation():
    rng=np.random.default_rng(29);values=rng.normal(size=(140,4,26));ids=['f'+str(i) for i in range(26)]
    rows=pd.DataFrame([{'id':ids[k],'stream':'old' if k<14 else 'mixed',
        'research_tier':'predictive_candidate' if k%2 else 'conditional_or_unstable','family':'F'+str(k%3)} for k in range(2,26)])
    mask=np.arange(140)<100
    a,archive,_=expand_pool(values,ids,rows,mask,8,.95,original_count=2)
    assert len(a)==16 and len(set(a))==16 and not set(a)&set(ids[:2])
    values[~mask]=values[~mask,...,:1]
    b,_,_=expand_pool(values,ids,rows,mask,8,.95,original_count=2)
    assert a==b
    assert archive.loc[archive.reason=='retained'].groupby(['stream','research_tier']).size().eq(4).all()
    with pytest.raises(ValueError,match='quotas'):expand_pool(np.ones_like(values),ids,rows,mask,8,.95,original_count=2)


def test_error_covariance_weights_reduce_noise_and_obey_declared_shrinkage():
    rng=np.random.default_rng(103);y=rng.normal(size=3000);p=y[:,None]+rng.normal(size=(3000,3))*[.05,1,2]
    w,n,_=shrunk_error_weights(p,y,.5,.5,500)
    assert n==3000 and w.sum()==pytest.approx(1) and w.min()>=1/6-1e-8 and w.max()<=2/3+1e-8
    assert np.mean((p@w-y)**2)<np.mean((p.mean(axis=1)-y)**2)
    p[:]=np.nan
    warm,n,_=shrunk_error_weights(p,y,.5,.5,500)
    np.testing.assert_allclose(warm,np.ones(3)/3);assert n==0


def test_weight_updates_only_read_matured_past_forecasts():
    rng=np.random.default_rng(31);dates=pd.date_range('2023-01-01',periods=600,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    y=rng.normal(size=(600,4))*.01;parts=[y+rng.normal(size=y.shape)*v for v in [.001,.02,.03]]
    policy=dict(diagonal_shrink=.5,equal_shrink=.5,minimum_labels=500)
    periods=[['202401010000','202404302355']]
    a,r=causal_weighted_average(parts,y,dates,periods,4,policy)
    mutated=y.copy();mutated[dates>='202401010000']*=100
    b,_=causal_weighted_average(parts,mutated,dates,periods,4,policy)
    np.testing.assert_allclose(a[dates<'202402010000'],b[dates<'202402010000'],equal_nan=True)
    assert all(abs(sum(v for k,v in record.items() if k.startswith('weight_'))-1)<1e-8 for record in r)


def test_asset_adaptation_learns_opposite_coin_relationships_without_future_labels():
    rng=np.random.default_rng(92);dates=pd.date_range('2023-01-01',periods=600,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x=rng.normal(size=(600,4,3));y=x[...,0]*np.array([.01,-.01,.02,-.02]);shared=np.zeros_like(y)
    spec=dict(id='asset',backend='lightgbm',trees=80,leaves=7,min_leaf=16,learning_rate=.1,penalty=1.,window_months=12,update_months=1)
    periods=[['202401010000','202404302355']]
    a,r=asset_shrunk_predictions(x,y,dates,periods,shared,spec,4,73,1,2000,.25)
    m=(dates>='202401010000')&(dates<'202402010000')
    assert np.corrcoef(a[m].ravel(),y[m].ravel())[0,1]>.95 and len(r)==16
    y[dates>='202401010000']*=100
    b,_=asset_shrunk_predictions(x,y,dates,periods,shared,spec,4,73,1,2000,.25)
    np.testing.assert_allclose(a[m],b[m])


def test_strict_phase_freeze_no_configuration_fallback():
    c=yaml.safe_load(Path('configs/research/effective_combo_expansion_20261009_r2_history.yaml').read_text())
    validate_extension_config(c)
    bad=copy.deepcopy(c);bad['complex_alpha']['phase']='oos'
    with pytest.raises(ValueError,match='frozen'):validate_extension_config(bad)
    bad['complex_alpha']['frozen_run']='outputs/frozen'
    assert protocol(bad['complex_alpha'])==protocol(c['complex_alpha'])
    bad['complex_alpha']['weighting']['equal_shrink']=-1
    with pytest.raises(ValueError,match='weighting'):validate_extension_config(bad)
