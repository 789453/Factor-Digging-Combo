"""Bounded reuse stage of the existing complex-alpha workflow, not a new miner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

from .aligned_contract import Split
from .complex_alpha_search import Candidate, evaluate_candidate
from .complex_alpha_evaluation import (past_market_betas, targets_from_prices,
    project_basket_returns, neutral_positions, five_minute_ledger, ledger_metrics)
from .complex_alpha_prediction import (DISCOVERY_COLUMNS, discovery_archive, shortlist,
    deduplicate_pool, fit_direct_baseline, prediction_metrics)
from .manual_alpha_workflow import _write_json, _write_csv


def validate_prediction_config(cfg):
    if set(cfg) != {'mode', 'version', 'complex_alpha', 'output'} or cfg['mode'] != 'aligned_crypto':
        raise ValueError('prediction reuse requires explicit aligned_crypto stage config')
    s = cfg['complex_alpha']
    keys = {'stage', 'source_run', 'bank_manifest', 'horizon_hours', 'sample_minutes',
            'shortlist_count', 'pool_count', 'correlation_limit', 'penalties', 'folds',
            'device', 'cost_bps', 'gross_budget', 'targets', 'diagnostics'}
    if set(s) not in (keys, keys | {'estimation_study'}) or s['stage'] != 'predictive_pool':
        raise ValueError('invalid prediction reuse declaration')
    if 'estimation_study' in s:
        study = s['estimation_study']
        if set(study) != {'training_windows_months', 'frozen_pool_run'} or study['training_windows_months'] != [0, 12]:
            raise ValueError('estimation study must explicitly compare expanding and twelve months')
    if s['horizon_hours'] not in {4, 12} or s['sample_minutes'] not in {15, 30, 60, 120, 240}:
        raise ValueError('invalid explicit target/update clock')
    if not 1 <= s['pool_count'] <= s['shortlist_count'] <= 256 or not 0 < s['correlation_limit'] <= 1:
        raise ValueError('bounded prediction pool required')
    if not s['penalties'] or len(s['penalties']) > 4 or any(v <= 0 for v in s['penalties']):
        raise ValueError('bounded positive ridge penalties required')
    if len(set(s['penalties'])) != len(s['penalties']): raise ValueError('duplicate penalties')
    if s['targets'] != ['raw_return', 'cash_beta_residual']:
        raise ValueError('raw and residual targets must both be reported')
    if s['diagnostics'] != []:
        raise ValueError('heavy diagnostics disabled in bounded prediction stage')
    if s['device'] not in {'cpu', 'cuda'} or (s['device'] == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('requested prediction device unavailable')
    if s['cost_bps'] < 0 or not 0 < s['gross_budget'] <= 1:
        raise ValueError('invalid portfolio cost/budget')
    if set(cfg['output']) != {'out_dir', 'visualization_dir'}:
        raise ValueError('explicit new output and visualization directories required')


def run_prediction_stage(cfg, config_path):
    validate_prediction_config(cfg)
    # Reuse existing source loaders and financial contract, never run generation.
    from .complex_alpha_workflow import _funding
    s = cfg['complex_alpha']; source = Path(s['source_run']); bank_path = Path(s['bank_manifest'])
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifest = load(source / 'manifest.json'); bank = load(bank_path)
    if manifest['status'] != 'COMPLETED' or bank['status'] != 'COMPLETED':
        raise ValueError('reuse requires completed source evidence')
    if manifest['source_signature'] != bank['signature']:
        raise ValueError('source bank signature mismatch')
    prior = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    split = Split(**prior['split']); codes = prior['data']['assets']; root = bank_path.parent
    frozen_reference = None
    if 'estimation_study' in s:
        frozen_reference = Path(s['estimation_study']['frozen_pool_run'])
        if load(frozen_reference / 'manifest.json')['status'] != 'COMPLETED':
            raise ValueError('frozen prediction pool must be complete')
        reference_cfg = yaml.safe_load((frozen_reference / 'config.yaml').read_text(encoding='utf-8'))
        if reference_cfg['complex_alpha'] != {k: v for k, v in s.items() if k != 'estimation_study'}:
            raise ValueError('fixed-pool estimation study cannot change source, targets or execution')
    if bank['codes'] != codes: raise ValueError('source asset order mismatch')
    if not s['folds'] or len(s['folds']) > 4: raise ValueError('one to four discovery folds required')
    boundary = prior['complex_alpha']['feature_calibration_end']
    for start, end in s['folds']:
        if not boundary < start <= end <= split.discovery_end:
            raise ValueError('prediction fold crosses discovery or overlaps')
        boundary = end
    out = Path(cfg['output']['out_dir']); viz = Path(cfg['output']['visualization_dir'])
    if out.exists() or viz.exists(): raise ValueError('prediction outputs must be new directories')
    if source.resolve() in out.resolve().parents or out.resolve() == source.resolve():
        raise ValueError('cannot write into source evidence')
    out.mkdir(parents=True); viz.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    _write_json(out / 'manifest.json', {'status': 'RUNNING', 'stage': 'predictive_pool'})
    provenance = {}
    for p in [source / n for n in ['manifest.json', 'config.yaml', 'coarse_results.csv',
            'candidate_registry.json', 'initial_coordinate_scaler.json', 'field_registry.json']] + [bank_path]:
        provenance[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    snapshot = out / 'source_snapshot'; snapshot.mkdir()
    for p in Path(__file__).parent.glob('complex_alpha_*.py'): shutil.copyfile(p, snapshot / p.name)
    _write_json(out / 'source_provenance.json', provenance)
    if frozen_reference:
        prior_hashes = load(frozen_reference / 'source_provenance.json')
        if provenance != prior_hashes: raise ValueError('frozen pool source evidence changed')
        _write_json(out / 'frozen_reference_provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [frozen_reference / n for n in ['manifest.json', 'config.yaml', 'feature_pool_freeze.json',
                'prediction_metrics.csv', 'frozen_prediction_models.json']]})
    archive = discovery_archive(pd.read_csv(source / 'coarse_results.csv', usecols=DISCOVERY_COLUMNS))
    _write_csv(out / 'candidate_information_archive.csv', archive)
    rows = {name: shortlist(archive, s['horizon_hours'], name, s['shortlist_count']) for name in ['old', 'mixed']}
    if any(v.empty for v in rows.values()): raise ValueError('no informative shortlist; inspect archive')
    definitions = {c['id']: Candidate(**c) for c in load(source / 'candidate_registry.json')['definitions']}
    cards = load(source / 'field_registry.json')['coordinates']
    old_indices = [i for i, c in enumerate(cards) if c['family'] == 'OLD' and c.get('admitted', True)]
    needed = sorted(set(old_indices).union(*(set(definitions[r].left + definitions[r].right)
                    for frame in rows.values() for r in frame.id)))
    remap = {v: k for k, v in enumerate(needed)}
    all_dates = np.load(root / 'dates.npy')
    # Physically slice out holdout BEFORE reading feature rows or price arrays.
    end = int(np.searchsorted(all_dates, split.validation_end, side='right'))
    full_dates = all_dates[:end]
    stamp = pd.to_datetime(full_dates, format='%Y%m%d%H%M', utc=True)
    sampled = np.flatnonzero((stamp.hour * 60 + stamp.minute) % s['sample_minutes'] == 0)
    dates = full_dates[sampled]
    fast_all = np.load(root / 'fast_dates.npy')
    fast_end = int(np.searchsorted(fast_all, split.validation_end, side='right'))
    fast_dates = fast_all[:fast_end]
    fast_close = np.load(root / 'fast_close.npy', mmap_mode='r')[:fast_end].copy()
    if not len(dates) or dates[-1] >= split.holdout_start: raise ValueError('holdout slicing failed')
    scaler = load(source / 'initial_coordinate_scaler.json'); base_count = len(scaler[0]['mean'])
    x = np.empty((len(dates), len(codes), len(needed)), np.float32)
    original = [i for i in needed if i < base_count]
    for j, code in enumerate(codes):
        if scaler[j]['asset'] != code: raise ValueError('scaler order mismatch')
        cached = np.load(root / f'{code}.npy', mmap_mode='r')
        if cached.shape != (len(all_dates), base_count): raise ValueError('source coordinate dimensions changed')
        values = cached[np.ix_(sampled, original)]
        center = np.array(scaler[j]['mean'])[original]; scale = np.array(scaler[j]['scale'])[original]
        valid = np.array(scaler[j]['valid'])[original]
        values = np.clip(np.nan_to_num((values - center) / scale, nan=0.), -8, 8)
        values[:, ~valid] = 0
        x[:, j, [remap[i] for i in original]] = values
    funding, funding_files, extra = _funding(prior, full_dates, fast_dates, fast_close)
    for name, value in {'clock_sin': np.sin(2*np.pi*(stamp.hour+stamp.minute/60)/24),
        'clock_cos': np.cos(2*np.pi*(stamp.hour+stamp.minute/60)/24),
        'weekday_sin': np.sin(2*np.pi*stamp.dayofweek/7),
        'weekday_cos': np.cos(2*np.pi*stamp.dayofweek/7)}.items():
        extra[name] = np.repeat(np.asarray(value)[:, None], len(codes), axis=1)
    calibration = full_dates <= prior['complex_alpha']['feature_calibration_end']
    for i in needed:
        if i < base_count: continue
        values = extra[cards[i]['name']]
        center = np.nanmean(values[calibration], axis=0); scale = np.nanstd(values[calibration], axis=0)
        valid = np.isfinite(scale) & (scale > 1e-12); center[~valid] = 0; scale[~valid] = 1
        normalized = np.clip(np.nan_to_num((values[sampled] - center) / scale, nan=0.), -8, 8)
        normalized[:, ~valid] = 0; x[..., remap[i]] = normalized
    # Reconstruct funding from exactly the source files; changed inputs are an error.
    expected = load(source / 'input_signature.json')['derivative_files']
    if funding_files != expected: raise ValueError('source derivative fingerprints changed')
    idx = np.searchsorted(fast_dates, full_dates)
    beta_full = past_market_betas(fast_close[idx], width=2880 if prior['data']['source']=='crypto_parquet' else 192,
                                  return_contract='native_simple')
    beta = beta_full[sampled]
    raw, _ = targets_from_prices(dates, fast_dates, fast_close, [s['horizon_hours']], split, beta, funding,
                                return_contract='native_simple')
    y = raw[s['horizon_hours']]
    targets = {'raw_return': y, 'cash_beta_residual': project_basket_returns(y, beta)}
    train = dates <= split.discovery_end
    designs = {'old_coordinates': x[..., [remap[i] for i in old_indices]]}
    registry = []
    for name, frame in rows.items():
        values = []
        for identifier in frame.id:
            c = definitions[identifier]
            transformed = Candidate(c.id, c.family, c.mode, tuple(remap[i] for i in c.left),
                c.left_weights, tuple(remap[i] for i in c.right), c.right_weights,
                c.support_minutes, c.nonlinear_order, c.nodes)
            values.append(evaluate_candidate(x, transformed).astype(np.float32))
        matrix, kept, aliases = deduplicate_pool(np.stack(values, axis=-1), frame, train,
                                               s['pool_count'], s['correlation_limit'])
        if frozen_reference:
            reference_ids = pd.read_csv(frozen_reference / f'{name}_feature_pool.csv').id.tolist()
            if reference_ids != kept.id.tolist(): raise ValueError('study reconstructed a different feature pool')
        designs[name + '_mined'] = matrix
        _write_csv(out / f'{name}_feature_pool.csv', kept)
        _write_csv(out / f'{name}_redundancy_archive.csv', aliases)
        for row in kept.to_dict('records'):
            registry.append({**row, 'stream': name, 'definition': definitions[row['id']].record()})
    _write_json(out / 'feature_pool_freeze.json', {'features': registry, 'source_hashes': provenance,
        'selection_data': 'source discovery only', 'independent_oof': False, 'holdout_opened': False,
        'role': 'prediction candidates, not independently tradable alpha', 'cost_used_in_selection': False})
    metrics, models, trades = [], {}, []
    valid = (dates >= split.validation_start) & (dates <= split.validation_end)
    np.save(out / 'dates.npy', dates)
    windows = s.get('estimation_study', {}).get('training_windows_months', [0])
    for name, design in designs.items(): np.save(out / f'{name}_features.npy', design)
    for name, target in targets.items(): np.save(out / f'{name}_target.npy', target)
    _write_json(out / 'numeric_pool_contract.json', {
        'axes': ['decision_date', 'asset', 'feature'], 'assets': codes,
        'feature_order': {'old_coordinates': [cards[i]['name'] for i in old_indices],
            **{name + '_mined': pd.read_csv(out / f'{name}_feature_pool.csv').id.tolist() for name in ['old', 'mixed']}},
        'decision_clock_minutes': s['sample_minutes'], 'horizon_hours': s['horizon_hours'],
        'entry': 'completed decision close; next complete native5 return',
        'targets': 'simple fixed-share price return minus future funding-price proxy; residual separately projected',
        'feature_preprocessing': 'source initial per-asset normalization, finite fill, clip; Ridge scaler fitted in each training window',
        'candidate_direction': 'metadata only; direct Ridge learns signed coefficients from matured labels',
        'holdout_opened': False, 'independent_oof': False})
    fit_designs = {(name if not frozen_reference else name + ('__expanding' if window == 0 else '__rolling12')):
        (design, window) for name, design in designs.items() for window in windows}
    for target_name, target in targets.items():
        for name, (design, window) in fit_designs.items():
            key = target_name + '__' + name
            prediction, fitted, tuning, model = fit_direct_baseline(design, target, dates, s['folds'],
                split.discovery_end, s['horizon_hours'], s['penalties'], s['device'], window)
            models[key] = model
            _write_csv(out / f'{key}_discovery_tuning.csv', tuning)
            fitted_train = train & (dates >= model['final_first_origin'])
            for period, p, mask in [('discovery_in_sample_diagnostic', fitted, fitted_train),
                                    ('discovery_feedback_oof', prediction, train),
                                    ('historical_validation', prediction, valid)]:
                metrics.append({'model': key, 'period': period, **prediction_metrics(p[mask], target[mask])})
            np.save(out / f'{key}_prediction.npy', prediction.astype(np.float32))
            # Always report a fixed portfolio mapping, regardless of profitability.
            scale = float(np.sqrt(np.nanmean(prediction[train] ** 2)))
            if not np.isfinite(scale) or scale <= 1e-12: raise ValueError('degenerate baseline prediction scale')
            pv = prediction[valid]
            if target_name == 'cash_beta_residual':
                position = neutral_positions(pv, beta[valid], scale, s['gross_budget'], 1, 0., 0., 'common_label')
            else:
                position = np.tanh(np.nan_to_num(pv / scale, nan=0.)) * s['gross_budget']
            fv = fast_dates >= split.validation_start
            ledger, asset, _ = five_minute_ledger(position, dates[valid], fast_dates[fv], fast_close[fv],
                                                 funding[fv], s['cost_bps'], split)
            ledger.to_parquet(out / f'{key}_validation_ledger.parquet', index=False)
            trades.append({'model': key, **ledger_metrics(ledger, asset, np.ones(len(ledger), bool))})
            model['execution_scale'] = scale
            print(f'[prediction baseline] {key}: features={design.shape[-1]} penalty={model["penalty"]}', flush=True)
    _write_json(out / 'frozen_prediction_models.json', models)
    table = pd.DataFrame(metrics); trading = pd.DataFrame(trades)
    _write_csv(out / 'prediction_metrics.csv', table); _write_csv(out / 'portfolio_metrics.csv', trading)
    summary = {'candidate_target_archive': len(archive), 'retained_feature_records': len(registry),
        'baseline_models': len(models), 'decision_rows': len(dates), 'assets': len(codes),
        'holdout_opened': False, 'new_candidates_generated': 0, 'source': str(source),
        'pool_counts': {k: v.shape[-1] for k, v in designs.items()},
        'ridge_fits': len(models) * (len(s['penalties']) * len(s['folds']) + 1),
        'training_windows_months': windows}
    if frozen_reference:
        reference_metrics = pd.read_csv(frozen_reference / 'prediction_metrics.csv')
        replay = table.loc[table.model.str.endswith('__expanding')].copy()
        replay['model'] = replay.model.str.removesuffix('__expanding')
        merged = replay.merge(reference_metrics, on=['model', 'period'], suffixes=('', '_reference'))
        error = max(float(np.max(np.abs(merged[col] - merged[col + '_reference'])))
            for col in ['ic', 'mse', 'r2_vs_zero', 'prediction_rms'])
        if len(merged) != len(reference_metrics) or error > 1e-9:
            raise ValueError(f'expanding baseline replay mismatch: {error}')
        _write_json(out / 'baseline_replay_check.json', {'status': 'PASS', 'max_metric_error': error,
            'frozen_pool_match': True, 'reference': str(frozen_reference)})
    result = {'status': 'COMPLETED', 'stage': 'predictive_pool', 'version': cfg['version'],
              'results': summary, 'source_signature': bank['signature'], 'config': str(config_path),
              'independent_oof': False, 'historical_validation_previously_seen': True,
              'source_data': prior['data']['source'], 'diagnostics': [], 'trade_admission': 'not evaluated'}
    html = '<!doctype html><meta charset="utf-8"><title>预测特征池与组合基线</title><h1>预测特征池与组合基线</h1>'
    html += '<p>历史研究复核；发现期候选已经过搜索，OOF不独立于候选选择。费用仅评价组合，不删除特征。未读取holdout。</p>'
    html += '<h2>预测表现</h2>' + table.to_html(index=False) + '<h2>固定组合映射</h2>' + trading.to_html(index=False)
    (viz / 'index.html').write_text(html, encoding='utf-8'); (out / 'index.html').write_text(html, encoding='utf-8')
    _write_json(out / 'manifest.json', result)
    return result
