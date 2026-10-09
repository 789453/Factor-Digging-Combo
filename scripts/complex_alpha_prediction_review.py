"""Read-only review of saved predictions. Never fit, select, or trade a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from src.alpha_mvp.research.aligned_contract import Split
from src.alpha_mvp.research.complex_alpha_evaluation import (
    past_market_betas, targets_from_prices, project_basket_returns)
from src.alpha_mvp.research.complex_alpha_workflow import _funding


def loss_decomposition(prediction, target, training_mean=0.):
    good = np.isfinite(prediction) & np.isfinite(target)
    p, y = prediction[good].astype(float), target[good].astype(float)
    baseline = np.asarray(training_mean)
    if baseline.ndim: baseline = baseline[good]
    if len(y) < 2:
        raise ValueError('insufficient review observations')
    ey2, ep2, epy = np.mean(y*y), np.mean(p*p), np.mean(p*y)
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = np.mean((p-mp)*(y-my))
    variance_gain = 2*cov-vp
    mean_gain = 2*mp*my-mp*mp
    slope = cov/vp if vp > 1e-24 else np.nan
    # Oracle numbers explain historical failure; they are never deployed weights.
    shrink = float(np.clip(epy/ep2, 0, 1)) if ep2 > 1e-24 else 0.
    denominator = np.mean((y-baseline)**2)
    nonzero = ey2 > 1e-24
    return dict(observations=len(y), label_rms_bps=np.sqrt(ey2)*1e4,
        prediction_rms_bps=np.sqrt(ep2)*1e4, prediction_mean_bps=mp*1e4,
        label_mean_bps=my*1e4, ic=cov/np.sqrt(vp*vy) if vp > 1e-24 and vy > 1e-24 else np.nan,
        r2_vs_zero=(2*epy-ep2)/ey2 if nonzero else np.nan,
        variance_gain_pct=variance_gain/ey2*100 if nonzero else np.nan,
        mean_gain_pct=mean_gain/ey2*100 if nonzero else np.nan,
        r2_vs_training_mean=1-np.mean((y-p)**2)/denominator if denominator > 1e-24 else np.nan,
        historical_centered_slope=slope, historical_no_flip_shrink=shrink,
        historical_no_flip_oracle_r2=(2*shrink*epy-shrink*shrink*ep2)/ey2 if nonzero else np.nan)


def component_review(p, y):
    good = np.isfinite(p).all(axis=1) & np.isfinite(y).all(axis=1)
    p, y = p[good].astype(float), y[good].astype(float)
    if not len(p): raise ValueError('no complete asset basket')
    common_p, common_y = p.mean(axis=1), y.mean(axis=1)
    relative_p, relative_y = p-common_p[:, None], y-common_y[:, None]
    zero = np.mean(y*y)
    rows = []
    for name, a, b in [('common', common_p, common_y), ('relative', relative_p, relative_y)]:
        rows.append({'component': name, **loss_decomposition(a, b),
            'total_label_loss_gain_pct': np.mean(2*a*b-a*a)/zero*100,
            'total_label_energy_share': np.mean(b*b)/zero})
    return rows


def reconstruct_labels(run):
    cfg = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    s = cfg['complex_alpha']; source = Path(s['source_run'])
    prior = yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    split = Split(**prior['split']); root = Path(s['bank_manifest']).parent
    dates = np.load(run/'dates.npy')
    full = np.load(root/'dates.npy'); full = full[:np.searchsorted(full, split.validation_end, side='right')]
    fast = np.load(root/'fast_dates.npy'); end = np.searchsorted(fast, split.validation_end, side='right')
    fast = fast[:end]; close = np.load(root/'fast_close.npy', mmap_mode='r')[:end].copy()
    if dates[-1] >= split.holdout_start: raise ValueError('review would cross holdout')
    funding, fingerprints, _ = _funding(prior, full, fast, close)
    expected = json.loads((source/'input_signature.json').read_text())['derivative_files']
    if fingerprints != expected: raise ValueError('source derivative fingerprint changed')
    beta = past_market_betas(close[np.searchsorted(fast, full)],
        width=2880 if prior['data']['source']=='crypto_parquet' else 192, return_contract='native_simple')
    raw, _ = targets_from_prices(dates, fast, close, [s['horizon_hours']], split,
        beta[np.searchsorted(full, dates)], funding, return_contract='native_simple')
    y = raw[s['horizon_hours']]
    return dates, {'raw_return': y, 'cash_beta_residual': project_basket_returns(y, beta[np.searchsorted(full, dates)])}, split, prior['data']['assets']


def training_constant_predictions(y, dates, cfg, model, split):
    """Each feedback fold's benchmark uses its own matured training labels."""
    stamp = pd.to_datetime(dates, format='%Y%m%d%H%M', utc=True)
    s = cfg['complex_alpha']; months = model.get('training_window_months', 0)
    baseline = np.full_like(y, np.nan)
    for start, end in s['folds'] + [[split.validation_start, split.validation_end]]:
        boundary = pd.to_datetime(start, format='%Y%m%d%H%M', utc=True)
        train = (dates < start) & (stamp + pd.Timedelta(hours=s['horizon_hours']) < boundary)
        if months: train &= stamp >= boundary - pd.DateOffset(months=months)
        baseline[(dates >= start) & (dates <= end)] = np.nanmean(y[train])
    return baseline


def review(run, out, viz=None):
    if out.exists(): raise ValueError('review output must be new')
    if viz is not None and viz.exists(): raise ValueError('review visualization must be new')
    if json.loads((run/'manifest.json').read_text())['status'] != 'COMPLETED':
        raise ValueError('review source must be complete')
    source_files = list(run.glob('*_prediction.npy')) + [run/n for n in
        ['dates.npy', 'config.yaml', 'manifest.json', 'prediction_metrics.csv', 'feature_pool_freeze.json', 'frozen_prediction_models.json']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    dates, targets, split, codes = reconstruct_labels(run)
    cfg = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    models = json.loads((run/'frozen_prediction_models.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/Path(__file__).name)
    np.save(out/'dates.npy', dates)
    for k, y in targets.items(): np.save(out/f'{k}_target.npy', y)
    aggregate, monthly, assets, components = [], [], [], []
    months = pd.to_datetime(dates, format='%Y%m%d%H%M').strftime('%Y-%m').to_numpy()
    expected = pd.read_csv(run/'prediction_metrics.csv'); max_error = 0.
    for path in sorted(run.glob('*_prediction.npy')):
        key = path.name.removesuffix('_prediction.npy'); target_name = key.split('__')[0]
        p = np.load(path); y = targets[target_name]; train = dates <= split.discovery_end
        baseline = training_constant_predictions(y, dates, cfg, models[key], split)
        for period, mask in [('discovery_feedback_oof', train), ('historical_validation', dates >= split.validation_start)]:
            row = {'model': key, 'period': period, **loss_decomposition(p[mask], y[mask], baseline[mask])}
            aggregate.append(row)
            old = expected.loc[expected.model.eq(key) & expected.period.eq(period)].iloc[0]
            max_error = max(max_error, abs(row['r2_vs_zero']-old.r2_vs_zero), abs(row['ic']-old.ic))
            for r in component_review(p[mask], y[mask]): components.append({'model': key, 'period': period, **r})
            for j, code in enumerate(codes):
                assets.append({'model': key, 'period': period, 'asset': code, **loss_decomposition(p[mask,j], y[mask,j], baseline[mask,j])})
            for month in np.unique(months[mask]):
                m = mask & (months==month) & np.isfinite(p).all(axis=1) & np.isfinite(y).all(axis=1)
                if m.sum() < 2: continue
                monthly.append({'model': key, 'period': period, 'month': month, **loss_decomposition(p[m], y[m], baseline[m])})
    if max_error > 1e-7: raise ValueError(f'prediction metric reconstruction failed: {max_error}')
    for name, rows in [('loss_decomposition', aggregate), ('monthly', monthly), ('assets', assets), ('components', components)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
    pool_stats = []
    for path in sorted(run.glob('*_features.npy')):
        x = np.load(path)
        flat = x[dates <= split.discovery_end].reshape(-1, x.shape[-1]).astype(float)
        usable = np.std(flat, axis=0) > 1e-8
        ev = np.linalg.eigvalsh(np.corrcoef(flat[:, usable], rowvar=False))
        pool_stats.append({'pool': path.name.removesuffix('_features.npy'), 'columns': x.shape[-1],
            'nonconstant_columns': int(usable.sum()), 'correlation_participation_rank': ev.sum()**2/(ev*ev).sum(),
            'largest_correlation_eigenvalue': ev[-1], 'data': 'discovery only; no claim of independent signal count'})
    pd.DataFrame(pool_stats).to_csv(out/'pool_geometry.csv', index=False)
    accounting_error = 0.
    for path in run.glob('*_validation_ledger.parquet'):
        ledger = pd.read_parquet(path)
        accounting_error = max(accounting_error, float(np.max(np.abs(ledger.price_gross + ledger.funding - ledger.fee - ledger.net))))
    if accounting_error > 1e-12: raise ValueError('new portfolio ledger accounting mismatch')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p, sha in hashes.items()):
        raise ValueError('source evidence changed during review')
    (out/'manifest.json').write_text(json.dumps({'status':'COMPLETED', 'role':'read-only prediction review',
        'holdout_opened':False, 'fitted_models':0, 'metric_max_error':max_error,
        'ledger_row_max_error': accounting_error, 'baseline': 'constant estimated separately in each matured training window',
        'oracle_values':'historical explanation only; never selected or deployed', 'source_hashes':hashes}, indent=2), encoding='utf-8')
    print(pd.DataFrame(aggregate).to_string(index=False))
    if viz is not None:
        render_review(run, out, viz, pd.DataFrame(aggregate))


def render_review(run, out, viz, decomposition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    viz.mkdir(parents=True)
    metrics = pd.read_csv(run/'prediction_metrics.csv')
    portfolio = pd.read_csv(run/'portfolio_metrics.csv')
    valid = metrics.loc[metrics.period.eq('historical_validation')].copy()
    valid['base'] = valid.model.str.replace(r'__(expanding|rolling12)$', '', regex=True)
    labels = ['Raw / OLD coordinates', 'Raw / OLD mined', 'Raw / Mixed mined',
              'Residual / OLD coordinates', 'Residual / OLD mined', 'Residual / Mixed mined']
    order = ['raw_return__old_coordinates', 'raw_return__old_mined', 'raw_return__mixed_mined',
             'cash_beta_residual__old_coordinates', 'cash_beta_residual__old_mined', 'cash_beta_residual__mixed_mined']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={'width_ratios':[1.2, 1, 1]})
    ypos = np.arange(6)
    for suffix, color, delta, label in [('__expanding','#245b91',-.15,'Expanding'),('__rolling12','#c36b2c',.15,'Rolling 12 months')]:
        part = valid.set_index('model').loc[[b+suffix for b in order]]
        axes[0].barh(ypos+delta, part.r2_vs_zero*100, height=.28, color=color, label=label)
        axes[1].barh(ypos+delta, part.ic, height=.28, color=color)
        trade = portfolio.set_index('model').loc[[b+suffix for b in order]]
        axes[2].barh(ypos+delta, trade.net_return_pct, height=.28, color=color)
    for ax in axes:
        ax.axvline(0,color='#777777',linewidth=.8); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(ypos, labels); axes[0].legend(loc='lower left',fontsize=8)
    for ax in axes[1:]: ax.set_yticks(ypos, ['']*6)
    axes[0].set_title('Prediction R2 vs zero (%)'); axes[1].set_title('Flattened Pearson IC')
    axes[2].set_title('Fixed-map net wealth (%)')
    fig.suptitle('Frozen feature pools: training-span comparison | Historical validation 2025-07 to 2026-01',fontsize=12)
    fig.text(.5,.015,'4h target/update | 12 assets | discovery-only penalties | 4bps per one-way change | holdout excluded',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94)); fig.savefig(viz/'research_evidence.png',dpi=150); plt.close(fig)
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定预测池：估计跨度与弱信号研究</title><style>body{font-family:system-ui;margin:32px;color:#183044}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #cad3dc}img{width:100%;max-width:1500px}p{max-width:1000px}</style>
<h1>固定预测池：估计跨度与弱信号研究</h1><p>复用已有候选，不新搜索，不读取holdout。12个模型完整展示；历史验证已见，发现反馈OOF不独立于源搜索。12个月训练没有普遍修复迁移，固定交易映射全部亏损；特征与预测均保留。</p>
<p>未来原始收益含持有期资金费价格代理；风险残差目标单列。每4h完成快照决策，承担之后完整5m收益，单向变仓4bps，真实份额漂移，净财富逐行复利；候选与惩罚只用发现数据。</p><img src="research_evidence.png" alt="预测与交易的完整跨度对照">'''
    html += '<h2>预测全表</h2>' + metrics.to_html(index=False)
    html += '<h2>毛、费用、净与实际敞口</h2>' + portfolio.to_html(index=False)
    html += '<h2>验证损失分解</h2><p>variance_gain_pct与mean_gain_pct相加为R²百分数。historical/oracle列仅作事后解释，未部署或选择。</p>'
    html += decomposition.loc[decomposition.period.eq('historical_validation')].to_html(index=False)
    html += '<h2>数据与报告</h2><ul>'
    import os
    for label, path in [('详细研究报告',Path('docs/PREDICTIVE_FACTOR_DEEP_RESEARCH_RESULTS_20261009.md')),
                        ('数值池合同',run/'numeric_pool_contract.json'), ('特征池与公式',run/'feature_pool_freeze.json'),
                        ('逐月证据',out/'monthly.csv'), ('共同与相对分量',out/'components.csv'),
                        ('输入相关结构',out/'pool_geometry.csv'), ('复核清单',out/'manifest.json')]:
        href = os.path.relpath(path.resolve(), viz.resolve()).replace('\\','/')
        html += f'<li><a href="{href}">{label}</a></li>'
    html += '</ul></html>'
    (viz/'index.html').write_text(html,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--visualization', type=Path)
    args = parser.parse_args()
    review(args.source, args.out, args.visualization)
