import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.render_complex_alpha_review import inspect_evidence,inspect_links,finalize_review_navigation


def _evidence(root):
    snapshot=root/'source_snapshot';snapshot.mkdir()
    (snapshot/'complex_alpha_fake.py').write_text('version = 1\n')
    code=hashlib.sha256((snapshot/'complex_alpha_fake.py').read_bytes()).hexdigest()
    cfg={'version':'synthetic-review','data':{'source':'simulated'}}
    (root/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    sig=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+code.encode()).hexdigest()
    (root/'manifest.json').write_text(json.dumps({'status':'COMPLETED','code_sha256':code}))
    (root/'selection_freeze.json').write_text(json.dumps({'status':'FROZEN','selection_uses':['discovery_loss'],'run_signature':sig,'selected':[]}))
    f=pd.DataFrame({'completed_5m':['202301010000','202301010005','202301010010'],
        'period':['discovery']*3,'price_gross':[.01,.01,0.],'funding':[0.]*3,
        'fee':[0.,0.,.01],'net':[.01,.01,-.01],'equity_before_cashflows':[1.,1.01,1.0201]})
    f.to_parquet(root/'combo_5m_ledger.parquet',index=False)
    pd.DataFrame({'a':f.net,'b':f.net}).to_parquet(root/'combo_asset_pnl.parquet',index=False)
    (root/'combo_metrics.json').write_text(json.dumps({'discovery':{'net_return_pct':(np.prod(1+f.net)-1)*100}}))
    return f


def test_evidence_review_checks_actual_conservation(tmp_path):
    f=_evidence(tmp_path)
    assert inspect_evidence(tmp_path)['status']=='PASS'
    f.loc[1,'equity_before_cashflows']+=.1
    f.to_parquet(tmp_path/'combo_5m_ledger.parquet',index=False)
    with pytest.raises(AssertionError):inspect_evidence(tmp_path)


def test_evidence_review_rejects_source_or_firewall_mismatch(tmp_path):
    _evidence(tmp_path)
    f=tmp_path/'selection_freeze.json';freeze=json.loads(f.read_text());freeze['selection_uses']=['holdout_ic'];f.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='holdout'):inspect_evidence(tmp_path)
    freeze['selection_uses']=['discovery_loss'];f.write_text(json.dumps(freeze))
    (tmp_path/'source_snapshot'/'complex_alpha_fake.py').write_text('version = 2\n')
    with pytest.raises(ValueError,match='snapshot'):inspect_evidence(tmp_path)


def test_static_visualization_review_checks_nested_local_links(tmp_path):
    (tmp_path/'child').mkdir();(tmp_path/'plotly.min.js').write_text('/* fixture */')
    (tmp_path/'child'/'page.html').write_text("<meta charset='utf-8'><title>test</title><script src='../plotly.min.js'></script>")
    assert inspect_links(tmp_path)['html_pages']==1
    (tmp_path/'plotly.min.js').unlink()
    with pytest.raises(ValueError,match='broken'):inspect_links(tmp_path)


def test_composite_atlas_exposes_every_diagnostic_and_accurate_page_manifest(tmp_path):
    names=['index.html','readouts.html','uncertainty.html','structure.html','economics.html','iterations.html','decay.html','audit.html','additive.html']
    for name in names:
        (tmp_path/name).write_text("<html><head><meta charset='utf-8'><title>fixture</title></head><body><nav>draft</nav><h1>fixture</h1></body></html>",encoding='utf-8')
    (tmp_path/'visualization_manifest.json').write_text(json.dumps({'pages':names[:6]}))
    finalize_review_navigation(tmp_path)
    manifest=json.loads((tmp_path/'visualization_manifest.json').read_text())
    assert manifest['pages']==names and manifest['page_count']==9
    for name in names:
        body=(tmp_path/name).read_text(encoding='utf-8')
        assert all(f'href="{file}"' in body for file in names)
        assert 'viewport' in body
    assert '轮换1h' in (tmp_path/'index.html').read_text(encoding='utf-8')
    assert inspect_links(tmp_path)['html_pages']==9


def test_decay_metrics_preserve_direction_and_mark_empty_signal_undefined():
    from scripts.complex_alpha_decay_review import forecast_decay_metrics
    rng=np.random.default_rng(72);p=rng.normal(size=(400,4));mask=np.ones(400,bool)
    active=np.abs(p)>1;phase=np.arange(400)%4==0
    m=forecast_decay_metrics(p,-p,mask,active,phase)
    assert m['mean_asset_ic']==pytest.approx(-1.) and m['nonoverlap_ic']==pytest.approx(-1.)
    z=forecast_decay_metrics(np.zeros_like(p),p,mask,np.zeros_like(p,dtype=bool),phase)
    assert np.isnan(z['mean_asset_ic']) and np.isnan(z['edge_condition_ic'])


def test_saved_old_design_reconstruction_missing_extreme_and_control(tmp_path):
    from scripts.complex_alpha_additive_review import rebuild_old_design
    rng=np.random.default_rng(46);dates=pd.date_range('2023-01-01',periods=300,freq='15min').strftime('%Y%m%d%H%M').to_numpy()
    codes=['a','b'];scalers=[];raws=[]
    for code in codes:
        raw=rng.normal(size=(300,3)).astype(np.float32);raw[5,0]=np.nan;raw[200,2]=1e20
        np.save(tmp_path/f'{code}.npy',raw);raws.append(raw)
        scalers.append({'asset':code,'mean':[1.,1.,1.],'scale':[2.,2.,2.],'valid':[True,True,True]})
    control=np.tile(np.arange(300,dtype=float)[:,None],(1,2));control[5]=np.nan
    bank={'codes':codes,'coordinates':[{'family':'OLD'},{'family':'F3'},{'family':'OLD'}]}
    actual=rebuild_old_design(tmp_path,bank,scalers,dates,{'control':control},dates[99])
    for j,raw in enumerate(raws):
        expected=np.clip(np.nan_to_num((raw[:,[0,2]]-np.float32(1))/np.float32(2),nan=0),-8,8)
        np.testing.assert_array_equal(actual[:,j,:2],expected)
    mean=np.nanmean(control[:100],axis=0);scale=np.nanstd(control[:100],axis=0)
    np.testing.assert_array_equal(actual[:,:,-1],np.clip(np.nan_to_num((control-mean)/scale,nan=0),-8,8).astype(np.float32))


def test_reference_forecasts_use_saved_fold_states_and_final_nuisance():
    from scripts.complex_alpha_additive_review import reference_forecasts
    dates=pd.date_range('2023-01-01',periods=300,freq='15min').strftime('%Y%m%d%H%M').to_numpy()
    design=np.ones((300,4,1));folds=[(dates[50],dates[99]),(dates[100],dates[199])]
    cfg={'split':{'discovery_end':dates[199]},'complex_alpha':{'folds':folds}}
    def record(coef):return {'center':[0.],'scale':[1.],'coef':[coef],'intercept':0.,'target_scale':2.}
    models=[]
    for h in [4,12]:
        states=[{'start':fold[0],'increment':{'base_state':record(i+1)}} for i,fold in enumerate(folds)]
        models.append({'kind':'joint_interaction_residual','horizon':h,'discovery_oof_states':states,'design':{'interaction_nuisance':record(3)}})
    actual=reference_forecasts(cfg,models,design,dates)
    assert np.isnan(actual[4][:50]).all()
    np.testing.assert_array_equal(actual[4][50:100],2.)
    np.testing.assert_array_equal(actual[4][100:200],4.)
    np.testing.assert_array_equal(actual[4][200:],6.)


def test_risk_activation_edges_are_discovery_only_and_separate_units():
    from scripts.complex_alpha_decay_review import volatility_activation
    risk=np.tile(np.linspace(1,5,500)[:,None],(1,4));edge=risk*.0002
    masks={'discovery':np.arange(500)<300,'validation':np.arange(500)>=300}
    rows,cuts=volatility_activation(edge,risk,masks,.0008)
    changed=risk.copy();changed[300:]*=100
    _,other=volatility_activation(edge,changed,masks,.0008)
    assert cuts==other
    good=[r for r in rows if r['observations']]
    assert all(r['mean_abs_forecast_in_risk_units']==pytest.approx(.0002) for r in good)
    assert rows[-1]['forecast_edge_active_share']>rows[0]['forecast_edge_active_share']


def test_evidence_review_rejects_changed_input_fingerprint(tmp_path):
    _evidence(tmp_path);raw=tmp_path/'source.dat';raw.write_bytes(b'input')
    manifest=tmp_path/'manifest.json';m=json.loads(manifest.read_text());st=raw.stat()
    m['source_files']=[{'path':str(raw),'size':st.st_size,'mtime_ns':st.st_mtime_ns}];manifest.write_text(json.dumps(m))
    assert inspect_evidence(tmp_path)['status']=='PASS'
    raw.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='fingerprint'):inspect_evidence(tmp_path)
