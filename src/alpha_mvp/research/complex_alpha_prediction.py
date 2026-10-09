"""Prediction-feature retention and direct combination; no trading admission gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .complex_alpha_evaluation import fit_ridge
from .complex_alpha_search import correlation

DISCOVERY_COLUMNS = [
    'id', 'family', 'mode', 'horizon_hours', 'direction', 'discovery_score',
    'discovery_mean_ic', 'discovery_min_ic', 'positive_folds',
    'fold0_ic', 'fold1_ic', 'fold2_ic', 'fold3_ic',
]


def discovery_archive(frame):
    """Allowlist prevents cost, validation and holdout columns affecting retention."""
    result = frame.loc[:, DISCOVERY_COLUMNS].copy()
    fold = result[[f'fold{i}_ic' for i in range(4)]].to_numpy(float)
    finite = np.isfinite(fold).all(axis=1) & result.direction.ne(0).to_numpy()
    core = finite & result.discovery_mean_ic.gt(0) & result.positive_folds.ge(2)
    conditional = finite & ~core & (np.max(fold, axis=1) > 0)
    result['research_tier'] = np.where(core, 'predictive_candidate',
        np.where(conditional, 'conditional_or_unstable', 'diagnostic_only'))
    result['sign_changes'] = np.sum(np.sign(fold[:, 1:]) != np.sign(fold[:, :-1]), axis=1)
    # This is an archive of searched historical evidence, not an alpha certificate.
    return result


def shortlist(archive, horizon, stream, limit):
    frame = archive.loc[archive.horizon_hours.eq(horizon)].copy()
    if stream == 'old':
        frame = frame.loc[frame.id.str.startswith('control_') & frame.family.eq('OLD')]
    elif stream == 'mixed':
        frame = frame.loc[frame.id.str.startswith('main_')]
    else:
        raise ValueError('unknown candidate stream')
    if frame.empty:
        raise ValueError(f'no source candidates for {stream}/{horizon}')
    # Reserve conditional candidates without requiring them to be independently profitable.
    core = frame.loc[frame.research_tier.eq('predictive_candidate')].sort_values(
        ['discovery_score', 'id'], ascending=[False, True])
    conditional = frame.loc[frame.research_tier.eq('conditional_or_unstable')].copy()
    conditional['peak_ic'] = conditional[[f'fold{i}_ic' for i in range(4)]].max(axis=1)
    conditional = conditional.sort_values(['peak_ic', 'id'], ascending=[False, True])
    def interleave(data, count):
        groups = [g for _, g in data.groupby('family', sort=True)]
        rows = []
        for rank in range(len(data)):
            for group in groups:
                if rank < len(group):
                    rows.append(group.iloc[rank])
                    if len(rows) == count:
                        return pd.DataFrame(rows)
        return pd.DataFrame(rows, columns=data.columns)
    part = interleave(conditional, max(1, limit // 4))
    primary = interleave(core, limit - len(part))
    result = pd.concat([primary, part], ignore_index=True)
    return result.drop_duplicates('id').head(limit).drop(columns=['peak_ic'], errors='ignore')


def deduplicate_pool(values, rows, train, maximum, correlation_limit):
    """Discovery-only correlation, protected family/conditional round-robin priority."""
    if not 0 < correlation_limit <= 1 or maximum < 1:
        raise ValueError('invalid feature pool bounds')
    # Interleave tiers so conditional slots cannot all vanish at a later head(maximum).
    order = []
    core = np.flatnonzero(rows.research_tier.eq('predictive_candidate'))
    conditional = np.flatnonzero(~rows.research_tier.eq('predictive_candidate'))
    for i in range(max(len(core), len(conditional))):
        if i < len(core): order.append(core[i])
        if i < len(conditional): order.append(conditional[i])
    sample = values[train].reshape(-1, values.shape[-1]).astype(float)
    if len(sample) < 100:
        raise ValueError('insufficient discovery rows for redundancy check')
    sample -= sample.mean(axis=0)
    norms = np.sqrt((sample * sample).sum(axis=0))
    unit = sample / np.maximum(norms, 1e-12)
    sim = unit.T @ unit
    kept, reasons = [], []
    for k in order:
        duplicate = next((j for j in kept if abs(sim[k, j]) >= correlation_limit), None)
        reason = ('constant' if norms[k] < 1e-8 else 'alias' if duplicate is not None
                  else 'budget_archive' if len(kept) >= maximum else 'retained')
        reasons.append({'id': rows.iloc[k].id, 'reason': reason,
                        'representative': rows.iloc[duplicate].id if duplicate is not None else ''})
        if reason == 'retained': kept.append(k)
    if not kept:
        raise ValueError('no numerically usable prediction features; inspect archive')
    return values[..., kept], rows.iloc[kept].copy(), pd.DataFrame(reasons)


def prediction_metrics(prediction, target):
    good = np.isfinite(prediction) & np.isfinite(target)
    if good.sum() < 100:
        raise ValueError('insufficient prediction evaluation observations')
    p, y = prediction[good], target[good]
    loss = float(np.mean((y - p) ** 2))
    zero = float(np.mean(y ** 2))
    return {'ic': correlation(p, y), 'mse': loss,
            'r2_vs_zero': 1 - loss / zero if zero > 0 else np.nan,
            'prediction_rms': float(np.sqrt(np.mean(p ** 2))), 'observations': int(good.sum())}


def fit_direct_baseline(values, target, dates, folds, discovery_end, horizon, penalties, device,
                        training_window_months=0):
    """Direct Y prediction. No OLD subtraction, orthogonal-increment or gamma gate.

    Source candidate selection used discovery feedback; these chronological fits
    are not nested out-of-search OOF evidence. Validation never chooses penalty.
    """
    stamp = pd.to_datetime(dates, format='%Y%m%d%H%M', utc=True)
    if training_window_months not in {0, 12}:
        raise ValueError('only expanding or predeclared twelve-month training supported')
    trials, predictions, state_records = [], {}, {}
    for penalty in penalties:
        pred = np.full_like(target, np.nan)
        states = []
        for start, end in folds:
            boundary = pd.to_datetime(start, format='%Y%m%d%H%M', utc=True)
            train = (dates < start) & (stamp + pd.Timedelta(hours=horizon) < boundary)
            if training_window_months:
                train &= stamp >= boundary - pd.DateOffset(months=training_window_months)
            test = (dates >= start) & (dates <= end)
            state = fit_ridge(values[train].reshape(-1, values.shape[-1]), target[train].ravel(), penalty, device)
            pred[test] = state.predict(values[test])
            trials.append({'penalty': penalty, 'fold_start': start, 'fold_end': end,
                           **prediction_metrics(pred[test], target[test])})
            states.append({'start': start, 'train_rows': int(train.sum()),
                'first_origin': str(dates[train][0]), 'last_origin': str(dates[train][-1]),
                'state': state.record()})
        predictions[penalty] = pred
        state_records[penalty] = states
    results = pd.DataFrame(trials)
    score = results.groupby('penalty').mse.mean()
    chosen = float(score.sort_values(kind='stable').index[0])
    boundary = pd.to_datetime(discovery_end, format='%Y%m%d%H%M', utc=True) + pd.Timedelta(minutes=5)
    train = (dates <= discovery_end) & (stamp + pd.Timedelta(hours=horizon) < boundary)
    if training_window_months:
        train &= stamp >= boundary - pd.DateOffset(months=training_window_months)
    state = fit_ridge(values[train].reshape(-1, values.shape[-1]), target[train].ravel(), chosen, device)
    pred = predictions[chosen]
    fitted = state.predict(values)
    # Early history excluded by a rolling window must remain chronological OOF.
    pred[dates > discovery_end] = fitted[dates > discovery_end]
    return pred, fitted, results, {'penalty': chosen, 'final_state': state.record(),
        'fold_states': state_records[chosen], 'features': values.shape[-1],
        'training_window_months': training_window_months, 'final_train_rows': int(train.sum()),
        'final_first_origin': str(dates[train][0]), 'final_last_origin': str(dates[train][-1])}
