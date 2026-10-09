"""Bounded conditional readouts and causal update protocols on frozen feature pools."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .complex_alpha_evaluation import fit_ridge


def asset_conditional_design(values):
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('finite time/asset/feature input required')
    identity = np.broadcast_to(np.eye(values.shape[1], dtype=np.float32),
                               (len(values), values.shape[1], values.shape[1]))
    return np.concatenate([values, identity], axis=-1)


def mature_window(dates, boundary, horizon, months):
    stamp = pd.to_datetime(dates, format='%Y%m%d%H%M', utc=True)
    cut = pd.to_datetime(boundary, format='%Y%m%d%H%M', utc=True)
    mask = (stamp + pd.Timedelta(hours=horizon) < cut)
    if months: mask &= stamp >= cut-pd.DateOffset(months=months)
    return np.asarray(mask)


def fit_model(x, y, spec, seed, threads, clip_bps):
    a = x.reshape(-1, x.shape[-1]); b = y.ravel()*1e4
    good = np.isfinite(b) & np.isfinite(a).all(axis=1)
    if good.sum() < max(200, x.shape[-1]*2): raise ValueError('insufficient mature model samples')
    target = np.clip(b[good], -clip_bps, clip_bps)
    if spec['backend'] == 'ridge':
        model = fit_ridge(a[good], target, spec['penalty'], 'cpu')
        return model, lambda z: model.predict(z)/1e4
    if spec['backend'] != 'lightgbm': raise ValueError('unsupported explicit backend')
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=spec['trees'], num_leaves=spec['leaves'],
        min_child_samples=spec['min_leaf'], learning_rate=spec['learning_rate'],
        reg_lambda=spec['penalty'], random_state=seed, n_jobs=threads,
        verbosity=-1, deterministic=True, force_col_wise=True, importance_type='gain')
    model.fit(a[good], target)
    def predict(z):
        return model.booster_.predict(z.reshape(-1,z.shape[-1]),num_threads=threads).reshape(z.shape[:-1])/1e4
    return model, predict


def causal_predictions(x, y, dates, periods, spec, horizon, seed, threads, clip_bps, save=None):
    result = np.full_like(y, np.nan); records = []
    stamp = pd.to_datetime(dates, format='%Y%m%d%H%M', utc=True)
    for start, end in periods:
        start_time = pd.to_datetime(start, format='%Y%m%d%H%M', utc=True)
        end_time = pd.to_datetime(end, format='%Y%m%d%H%M', utc=True)
        boundaries = [start_time]
        if spec['update_months']:
            next_time = start_time+pd.DateOffset(months=spec['update_months'])
            while next_time <= end_time:
                boundaries.append(next_time); next_time += pd.DateOffset(months=spec['update_months'])
        for i, boundary in enumerate(boundaries):
            key = boundary.strftime('%Y%m%d%H%M')
            train = mature_window(dates, key, horizon, spec['window_months'])
            stop = boundaries[i+1] if i+1<len(boundaries) else end_time+pd.Timedelta(minutes=5)
            test = np.asarray((stamp >= boundary) & (stamp < stop) & (dates <= end))
            if not test.any(): continue
            model, predict = fit_model(x[train],y[train],spec,seed,threads,clip_bps)
            result[test] = predict(x[test])
            record = {'update':key,'last_train_origin':str(dates[train][-1]),
                'first_train_origin':str(dates[train][0]),'mature_labels':int(np.isfinite(y[train]).sum()),
                'prediction_rows':int(test.sum())}
            records.append(record)
            if save: save(model,key,record)
    return result, records


def positions_from_prediction(prediction, scale, policy, budget):
    if not np.isfinite(scale) or scale<=0 or not 0<budget<=1: raise ValueError('invalid fixed mapping scale/budget')
    p = np.nan_to_num(prediction, nan=0.)
    if policy['name'] == 'rms': return np.tanh(p/scale)*budget
    if policy['name'] != 'net_edge': raise ValueError('invalid declared execution')
    # Direction and strength are learned predictions; the economic threshold is fixed.
    edge = np.maximum(np.abs(p)-policy['floor_bps']/1e4,0.)
    return np.sign(p)*np.tanh(edge/(policy['amplitude_bps']/1e4))*budget


def completed_trades(dates, actual, per_asset, codes, epsilon=1e-5):
    """Contiguous signed exposure episodes; terminal closes are censored."""
    stamp = pd.to_datetime(dates, format='%Y%m%d%H%M', utc=True)
    rows = []
    for j, code in enumerate(codes):
        sign = np.where(np.abs(actual[:,j])>epsilon,np.sign(actual[:,j]),0).astype(int)
        start = None; direction = 0
        for i in range(len(sign)):
            if sign[i] != direction:
                if direction and start is not None:
                    rows.append({'asset':code,'entry':str(dates[start]),'exit':str(dates[i]),
                        'direction':direction,'hours':(stamp[i]-stamp[start]).total_seconds()/3600,
                        'censored':i==len(sign)-1,
                        'mean_abs_position':float(np.mean(np.abs(actual[start:i,j])))})
                direction = sign[i]; start = i if direction else None
        if direction and start is not None:
            rows.append({'asset':code,'entry':str(dates[start]),'exit':str(dates[-1]),
                'direction':direction,'hours':(stamp[-1]-stamp[start]).total_seconds()/3600,
                'censored':True,
                'mean_abs_position':float(np.mean(np.abs(actual[start:,j])))})
    return pd.DataFrame(rows,columns=['asset','entry','exit','direction','hours','censored','mean_abs_position'])
