import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.effective_combo import (
    asset_conditional_design,mature_window,causal_predictions,positions_from_prediction,completed_trades)


def test_asset_identity_is_known_and_missing_input_fails():
    x=np.zeros((5,4,2));z=asset_conditional_design(x)
    assert z.shape==(5,4,6)
    np.testing.assert_array_equal(z[0,:,-4:],np.eye(4))
    x[0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):asset_conditional_design(x)


def test_maturity_window_and_monthly_update_cannot_read_current_future_labels():
    rng=np.random.default_rng(71)
    dates=pd.date_range('2023-01-01',periods=450,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x=asset_conditional_design(rng.normal(size=(450,4,3)));y=.001*x[...,0]+rng.normal(size=(450,4))*.0001
    spec=dict(id='test',backend='lightgbm',trees=30,leaves=7,min_leaf=16,learning_rate=.1,penalty=1.,window_months=12,update_months=1)
    periods=[['202401010000','202403012355']]
    a,records=causal_predictions(x,y,dates,periods,spec,4,7,1,2000)
    assert records[0]['last_train_origin']=='202312310000'
    y[dates>='202401010000']*=1000
    b,_=causal_predictions(x,y,dates,periods,spec,4,7,1,2000)
    np.testing.assert_allclose(a[dates<'202402010000'],b[dates<'202402010000'],equal_nan=True)
    assert np.nanmax(np.abs(a[dates>='202402010000']-b[dates>='202402010000']))>1e-5
    assert not mature_window(dates,'202401010000',24,12)[dates=='202312310000'].item()


def test_nonlinear_readout_can_fit_zero_marginal_interaction_without_future_features():
    rng=np.random.default_rng(3)
    dates=pd.date_range('2023-01-01',periods=800,freq='D',tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x=rng.choice([-1.,1.],size=(800,4,2));y=x[...,0]*x[...,1]*.01
    spec=dict(id='interaction',backend='lightgbm',trees=100,leaves=7,min_leaf=16,learning_rate=.1,penalty=1.,window_months=0,update_months=0)
    p,_=causal_predictions(asset_conditional_design(x),y,dates,[['202401010000','202412312355']],spec,4,5,1,2000)
    mask=np.isfinite(p)
    assert 1-np.mean((p[mask]-y[mask])**2)/np.mean(y[mask]**2)>.99


def test_cost_edge_and_censored_holdings_are_explicit():
    p=np.array([[np.nan,.0007,.0018,-.0018]])
    q=positions_from_prediction(p,.001,dict(name='net_edge',floor_bps=8,amplitude_bps=50),.5)
    assert q[0,0]==0 and q[0,1]==0 and q[0,2]>0 and q[0,3]<0
    np.testing.assert_allclose(positions_from_prediction(p,.001,dict(name='rms'),.5),
                               positions_from_prediction(p*10,.01,dict(name='rms'),.5))
    dates=np.array(['202301010000','202301010005','202301010010','202301010015'])
    trades=completed_trades(dates,np.array([[0],[1],[1],[0]]),np.zeros((4,1)),['BTC'])
    assert trades.censored.item() and trades.hours.item()==pytest.approx(1/6)


def test_strict_model_config_no_silent_fallback():
    import yaml
    from pathlib import Path
    from src.alpha_mvp.research.effective_combo_workflow import validate_effective_config
    cfg=yaml.safe_load(Path('configs/research/effective_combo_20261009_smoke1.yaml').read_text())
    validate_effective_config(cfg);cfg['complex_alpha']['models'][0]['backend']='unknown'
    with pytest.raises(ValueError,match='model'):validate_effective_config(cfg)


def test_fixed_removal_keeps_original_divisor_and_preserves_cancellation():
    from scripts.effective_combo_frozen_review import fixed_average
    parts={'weak':np.array([[1.,-1.]]),'strong':np.array([[5.,-5.]]),'opposed':np.array([[-3.,3.]])}
    np.testing.assert_allclose(fixed_average(parts),[[1.,-1.]])
    np.testing.assert_allclose(fixed_average(parts,'weak'),[[2/3,-2/3]])
    np.testing.assert_allclose(fixed_average(parts,'opposed'),[[2.,-2.]])


def test_report_preserves_discovery_choice_and_distinct_price_prediction_position_axes(tmp_path):
    import json
    from src.alpha_mvp.research.effective_combo_reporting import render_effective_report
    source=tmp_path/'source';source.mkdir();out=tmp_path/'html'
    def write(name,obj):(source/name).write_text(json.dumps(obj),encoding='utf-8')
    key='ridge__edge16';codes=['A','B','C','D'];dates=np.array(['202507010000','202507010400','202507010800'])
    write('manifest.json',{'status':'COMPLETED'})
    write('selection_freeze.json',{'selected':{'id':key,'model':'ridge'},'ranked_discovery':[{'id':key,'model':'ridge'}]})
    write('feature_contract.json',{'assets':codes})
    write('feature_pool_freeze.json',{'features':[]})
    write('overfit_diagnostic_metrics.json',{'ic':.9,'r2_vs_zero':.8})
    write('overfit_diagnostic_strategy.json',{'net_return_pct':100.})
    np.save(source/'dates.npy',dates);np.save(source/'ridge_prediction.npy',np.ones((3,4))*.001)
    pd.DataFrame([{'id':key,'period':'validation','net_return_pct':1.,'sharpe':1.,'max_drawdown_pct':-.5}]).to_csv(source/'portfolio_metrics.csv',index=False)
    pd.DataFrame([{'model':'ridge','period':'historical_validation','ic':.03}]).to_csv(source/'prediction_metrics.csv',index=False)
    pd.DataFrame([{'model':'ridge','update':'202507010000'}]).to_csv(source/'update_records.csv',index=False)
    for name in ['old','mixed']:pd.DataFrame([{'family':'OLD','id':name+'_x'}]).to_csv(source/(name+'_feature_pool.csv'),index=False)
    pd.DataFrame([{'feature':'x','gain':1.}]).to_csv(source/'feature_importance.csv',index=False)
    f=pd.DataFrame({'completed_5m':dates,'net':[.001,.002,-.001],'price_gross':[.002,.003,0.],
         'funding':0.,'fee':.001,'mean_abs_position':.1,'cash_exposure':.02,'turnover':.1})
    f.to_parquet(source/(key+'_validation_ledger.parquet'),index=False)
    f.to_parquet(source/'overfit_diagnostic_in_sample_ledger.parquet',index=False)
    np.savez(source/(key+'_validation_asset_path.npz'),net=np.tile(f.net.to_numpy()[:,None],(1,4)),position=np.ones((3,4))*.1,price=np.ones((3,4))*100,dates=dates)
    pd.DataFrame([{'asset':'A','entry':dates[0],'exit':dates[-1],'direction':1,'hours':8.,'censored':False,'mean_abs_position':.1}]).to_csv(source/(key+'_validation_trades.csv'),index=False)
    render_effective_report(source,out)
    manifest=json.loads((out/'visualization_manifest.json').read_text())
    assert manifest['automatic_selection']==key and manifest['showcase']==key
    text=(out/'assets/A.html').read_text(encoding='utf-8')
    assert '4h组合预测' in text and '实际持仓与长短方向' in text and '完成价格' in text
    assert 'yaxis4' in text
    with pytest.raises(ValueError,match='new visualization'):render_effective_report(source,out)
