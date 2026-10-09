import numpy as np
import pandas as pd
import pytest

from src.alpha_mvp.research.complex_alpha_prediction import (
    discovery_archive, shortlist, deduplicate_pool, fit_direct_baseline)


def evidence():
    rows = []
    for i, fold in enumerate([[.02]*4, [.08, -.03, -.04, -.02], [-.01]*4]):
        rows.append(dict(id=f'main_F1_{i}', family='F1', mode='linear', horizon_hours=4,
            direction=1, discovery_score=np.mean(fold)-.35*np.std(fold),
            discovery_mean_ic=np.mean(fold), discovery_min_ic=min(fold),
            positive_folds=sum(v>0 for v in fold), **{f'fold{j}_ic': v for j,v in enumerate(fold)}))
    return pd.DataFrame(rows)


def test_cost_validation_holdout_cannot_delete_prediction_features():
    frame = evidence()
    before = discovery_archive(frame)
    frame['validation_net_return_pct'] = -1000
    frame['holdout_ic'] = [1, -1, 0]
    frame['validation_positive_asset_share'] = 0
    pd.testing.assert_frame_equal(before, discovery_archive(frame))
    assert before.research_tier.tolist() == ['predictive_candidate', 'conditional_or_unstable', 'diagnostic_only']
    assert len(shortlist(before, 4, 'mixed', 3)) == 2


def test_discovery_only_redundancy_and_constant_archive():
    rng = np.random.default_rng(17)
    v = rng.normal(size=(120, 4))
    values = np.stack([v, v*2, np.zeros_like(v)], axis=-1)
    rows = evidence()
    rows['research_tier'] = ['predictive_candidate']*3
    train = np.arange(120) < 80
    a, kept, reasons = deduplicate_pool(values, rows, train, 3, .98)
    assert kept.id.tolist() == ['main_F1_0']
    assert set(reasons.reason) == {'retained', 'alias', 'constant'}
    values[~train] = rng.normal(size=values[~train].shape)*1000
    _, later, _ = deduplicate_pool(values, rows, train, 3, .98)
    assert later.id.tolist() == kept.id.tolist()
    assert a.shape[-1] == 1


def test_direct_prediction_validation_labels_do_not_change_fit():
    rng = np.random.default_rng(4)
    dates = pd.date_range('2023-01-01', periods=720, freq='h', tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x = rng.normal(size=(720, 4, 3))
    y = .5*x[..., 0] + rng.normal(size=(720, 4))*.1
    folds = [['202301100000', '202301152300'], ['202301160000', '202301202300']]
    end = '202301202300'
    a, _, _, meta = fit_direct_baseline(x, y, dates, folds, end, 4, [.1, 1.], 'cpu')
    y[dates > end] *= -1000
    b, _, _, later = fit_direct_baseline(x, y, dates, folds, end, 4, [.1, 1.], 'cpu')
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert meta['final_state'] == later['final_state']
    assert np.std(a[dates > end]) > .1  # direct forecast is not gated to zero
    # First fold excludes origins whose four-hour labels reach the fold boundary.
    assert meta['fold_states'][0]['train_rows'] == 212


def test_invalid_stage_configuration_rejected():
    import yaml
    from pathlib import Path
    from src.alpha_mvp.research.complex_alpha_prediction_workflow import validate_prediction_config
    cfg = yaml.safe_load(Path('configs/research/complex_alpha_prediction_smoke_20261009.yaml').read_text())
    validate_prediction_config(cfg)
    cfg['complex_alpha']['diagnostics'] = ['bootstrap']
    with pytest.raises(ValueError, match='disabled'):
        validate_prediction_config(cfg)


def test_rolling_window_only_uses_declared_mature_history_and_preserves_oof():
    rng = np.random.default_rng(27)
    dates = pd.date_range('2022-01-01', periods=1260, freq='D', tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x = rng.normal(size=(1260, 4, 3))
    y = .2*x[..., 0] + rng.normal(size=(1260, 4))*.1
    folds = [['202401010000', '202403312355'], ['202404010000', '202406302355']]
    args = (dates, folds, '202406302355', 4, [.1, 1.], 'cpu')
    a, _, _, meta = fit_direct_baseline(x, y, *args, training_window_months=12)
    assert meta['fold_states'][0]['first_origin'] == '202301010000'
    assert meta['final_first_origin'] == '202307010000'
    assert np.isnan(a[dates < '202401010000']).all()  # not filled with final fit
    x[dates < '202301010000'] *= 10000
    y[(dates < '202301010000') | (dates > '202406302355')] *= -10000
    b, _, _, later = fit_direct_baseline(x, y, *args, training_window_months=12)
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert meta == later
    with pytest.raises(ValueError, match='twelve-month'):
        fit_direct_baseline(x, y, *args, training_window_months=6)


def test_final_fit_excludes_unmatured_labels_even_if_caller_provides_them():
    rng = np.random.default_rng(6)
    dates = pd.date_range('2023-01-01', periods=720, freq='h', tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    x = rng.normal(size=(720, 4, 3)); y = .4*x[..., 0] + rng.normal(size=(720,4))*.1
    folds = [['202301100000', '202301202300']]; end='202301202355'
    a, _, _, meta = fit_direct_baseline(x, y, dates, folds, end, 4, [.1], 'cpu')
    y[(dates >= '202301202000') & (dates <= end)] *= 100000
    b, _, _, later = fit_direct_baseline(x, y, dates, folds, end, 4, [.1], 'cpu')
    assert meta['final_last_origin'] == '202301201900'
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert meta == later


def test_study_cannot_silently_accept_other_training_windows():
    import yaml
    from pathlib import Path
    from src.alpha_mvp.research.complex_alpha_prediction_workflow import validate_prediction_config
    cfg = yaml.safe_load(Path('configs/research/complex_alpha_prediction_estimation_20261009_v1.yaml').read_text())
    validate_prediction_config(cfg)
    cfg['complex_alpha']['estimation_study']['training_windows_months'] = [0, 6]
    with pytest.raises(ValueError, match='twelve months'):
        validate_prediction_config(cfg)
