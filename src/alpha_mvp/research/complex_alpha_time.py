"""UTC calendar sampling with explicit integer units, independent of pandas dtype."""
from __future__ import annotations
import numpy as np
import pandas as pd

TIME_VERSION='complex-utc-explicit-ns-20261008-v1'


def utc_epoch_nanoseconds(stamp: pd.DatetimeIndex) -> np.ndarray:
    if not isinstance(stamp,pd.DatetimeIndex) or stamp.tz is None:
        raise ValueError('research sampling requires a timezone-aware DatetimeIndex')
    if stamp.hasnans:raise ValueError('missing timestamp in research sampling')
    return stamp.tz_convert('UTC').as_unit('ns').asi8


def rotating_screening_masks(stamp: pd.DatetimeIndex):
    utc=stamp.tz_convert('UTC') if isinstance(stamp,pd.DatetimeIndex) and stamp.tz is not None else stamp
    ns=utc_epoch_nanoseconds(utc)
    if np.any(ns%900_000_000_000):raise ValueError('screening clock must be on completed 15m grid')
    phase=(ns//86_400_000_000_000)%4
    minute_phase=utc.minute//15
    return (utc.hour%4==phase)&(minute_phase==phase),minute_phase==phase


def nonoverlap_phase(stamp: pd.DatetimeIndex,horizon_hours: int) -> np.ndarray:
    if isinstance(horizon_hours,bool) or not isinstance(horizon_hours,(int,np.integer)) or horizon_hours<1:
        raise ValueError('nonoverlap horizon must be positive integer hours')
    ns=utc_epoch_nanoseconds(stamp)
    if np.any(ns%900_000_000_000):raise ValueError('nonoverlap clock must be on completed 15m grid')
    return (ns//900_000_000_000)%(horizon_hours*4)==0
