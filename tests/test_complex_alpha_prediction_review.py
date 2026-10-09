import numpy as np
import pytest

from scripts.complex_alpha_prediction_review import loss_decomposition, component_review
from scripts.complex_alpha_prediction_review import training_constant_predictions
from src.alpha_mvp.research.aligned_contract import Split
import pandas as pd


def test_loss_identity_and_zero_target_not_reported_as_infinite_skill():
    rng = np.random.default_rng(71)
    y = rng.normal(size=(300, 4)) - .2
    p = .1*y + .4
    p[0, 0] = np.nan
    row = loss_decomposition(p, y)
    good = np.isfinite(p)
    expected = 1-np.mean((y[good]-p[good])**2)/np.mean(y[good]**2)
    assert row['r2_vs_zero'] == pytest.approx(expected)
    assert (row['variance_gain_pct']+row['mean_gain_pct'])/100 == pytest.approx(expected)
    zero = loss_decomposition(np.ones((300, 4)), np.zeros((300, 4)))
    assert np.isnan(zero['ic']) and np.isnan(zero['r2_vs_zero'])
    assert zero['historical_no_flip_shrink'] == 0


def test_common_relative_orthogonal_loss_and_no_future_weight_use():
    rng = np.random.default_rng(25)
    y = rng.normal(size=(300, 4))
    p = rng.normal(size=y.shape)
    parts = component_review(p, y)
    full = loss_decomposition(p, y)
    assert sum(r['total_label_loss_gain_pct'] for r in parts) == pytest.approx(full['r2_vs_zero']*100)
    assert sum(r['total_label_energy_share'] for r in parts) == pytest.approx(1)
    negative = loss_decomposition(y, -y)
    assert negative['historical_no_flip_shrink'] == 0  # no sign-flip oracle
    assert negative['historical_no_flip_oracle_r2'] == 0


def test_constant_benchmark_uses_mature_past_per_fold_not_final_discovery_mean():
    dates = pd.date_range('2022-01-01', periods=1100, freq='D', tz='UTC').strftime('%Y%m%d%H%M').to_numpy()
    y = np.arange(1100*4, dtype=float).reshape(1100, 4)
    cfg = {'complex_alpha': {'horizon_hours': 4, 'folds': [['202401010000','202406302355']]}}
    split = Split(discovery_end='202406302355', validation_start='202407010000',
                  validation_end='202412312355', holdout_start='202501010000')
    before = training_constant_predictions(y, dates, cfg, {'training_window_months':12}, split)
    y[(dates < '202301010000') | (dates >= '202401010000')] *= -1000
    after = training_constant_predictions(y, dates, cfg, {'training_window_months':12}, split)
    fold = (dates >= '202401010000') & (dates <= '202406302355')
    np.testing.assert_array_equal(before[fold], after[fold])
