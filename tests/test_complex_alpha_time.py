import numpy as np
import pandas as pd
import pytest
from src.alpha_mvp.research.complex_alpha_time import utc_epoch_nanoseconds,rotating_screening_masks,nonoverlap_phase


@pytest.mark.parametrize('unit',['s','us','ns'])
def test_calendar_sampling_is_invariant_to_datetime_resolution(unit):
    t=pd.date_range('2025-07-01',periods=4*96,freq='15min',tz='UTC').as_unit(unit)
    a,b=rotating_screening_masks(t)
    ns=utc_epoch_nanoseconds(t)
    assert np.all(np.diff(ns)==900_000_000_000)
    for day in range(4):
        segment=slice(day*96,(day+1)*96)
        expected_phase=(pd.Timestamp('2025-07-01',tz='UTC').value//86_400_000_000_000+day)%4
        assert a[segment].sum()==6 and b[segment].sum()==24
        assert np.all(t[segment][b[segment]].minute==15*expected_phase)
    for h in (1,4,12,24):
        chosen=t[nonoverlap_phase(t,h)]
        assert len(chosen)==4*24//h
        assert np.all(np.diff(utc_epoch_nanoseconds(chosen))==h*3_600_000_000_000)


def test_sampling_uses_utc_not_local_dst_and_rejects_invalid_clocks():
    utc=pd.date_range('2025-11-01',periods=4*96,freq='15min',tz='UTC').as_unit('us')
    np.testing.assert_array_equal(nonoverlap_phase(utc,12),nonoverlap_phase(utc.tz_convert('America/New_York'),12))
    for x,y in zip(rotating_screening_masks(utc),rotating_screening_masks(utc.tz_convert('America/New_York'))):
        np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError,match='timezone'):utc_epoch_nanoseconds(utc.tz_localize(None))
    with pytest.raises(ValueError,match='missing'):utc_epoch_nanoseconds(pd.DatetimeIndex([pd.NaT],tz='UTC'))
    with pytest.raises(ValueError,match='15m'):nonoverlap_phase(utc+pd.Timedelta(minutes=1),4)
    with pytest.raises(ValueError,match='integer'):nonoverlap_phase(utc,1.5)
