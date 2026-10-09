import numpy as np
import pytest
from scripts.complex_alpha_fixed_readout_ablation import coordinate_contribution
from src.alpha_mvp.research.complex_alpha_evaluation import RidgeState


def test_fixed_coordinate_removal_matches_saved_predictor_and_clipping():
    state=RidgeState(np.array([1.,2.]),np.array([2.,3.]),np.array([.5,-.2]),.1,.04)
    x=np.array([[0.,8.],[1000.,2.],[-1000.,-20.]])
    removed=coordinate_contribution(x[:,0],state.record(),0)
    counter=x.copy();counter[:,0]=1.
    np.testing.assert_allclose(state.predict(x)-state.predict(counter),removed,atol=1e-16)
    assert removed[1]==pytest.approx(.16) and removed[2]==pytest.approx(-.16)


def test_fixed_coordinate_removal_rejects_missing_or_bad_frozen_scale():
    state={'center':[0.],'scale':[1.],'coef':[1.],'target_scale':1.}
    with pytest.raises(ValueError,match='nonfinite'):coordinate_contribution(np.array([np.nan]),state,0)
    state['scale']=[0.]
    with pytest.raises(ValueError,match='scale'):coordinate_contribution(np.array([1.]),state,0)
