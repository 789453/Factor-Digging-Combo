"""Read-only forensic audit of frozen artifacts; not a research pipeline.

Run from repository root. No candidate selection, fitting, or threshold tuning.
Creates a fresh evidence directory and refuses to overwrite it.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'outputs/crypto_factor_combo_signal_engine_20260929_v9'
OUT = ROOT / 'outputs/factor_information_audit_20260930_v2'


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main():
    OUT.mkdir(exist_ok=False)
    ledger = pd.read_parquet(SRC / 'asset_positions_5m.parquet')
    pred = pd.read_parquet(SRC / 'predictions_5m.parquet')
    assert (ledger[['date', 'asset']].astype(str).to_numpy() ==
            pred[['date', 'asset']].astype(str).to_numpy()).all()
    registry = pd.read_csv(SRC / 'factor_registry.csv')
    metadata = json.loads((SRC / 'model_metadata.json').read_text())
    rows, cumulative, projections, buckets, events_out = [], [], [], [], []
    for period, frame in ledger.groupby('period', sort=False):
        ids = frame.index
        nasset = frame.asset.nunique()
        ntime = len(frame) // nasset
        assert frame.groupby('date').size().eq(nasset).all()
        y = frame.next_raw_log_return.to_numpy().reshape(ntime, nasset)
        ry = pred.loc[ids, 'residual_y'].to_numpy().reshape(ntime, nasset)
        dates = frame.date.to_numpy().reshape(ntime, nasset)[:, 0].astype(str)
        columns = ['dynamic_alpha_strength', 'alpha_unit_target', 'native5_alpha_unit',
                   'hourly_alpha_unit', 'beta_direction', 'target', 'position',
                   'fast_strength', 'medium_strength', 'slow_strength']
        for col in columns:
            s = frame[col].to_numpy().reshape(ntime, nasset)
            for horizon in (1, 3, 12, 24, 48, 144, 288):
                # Cumulative future path starting with next bar; no synthetic trading.
                target = pd.DataFrame(ry).rolling(horizon, min_periods=horizon).sum().shift(1-horizon).to_numpy()
                rawtarget = pd.DataFrame(y).rolling(horizon, min_periods=horizon).sum().shift(1-horizon).to_numpy()
                valid = np.isfinite(target) & np.isfinite(s)
                rows.append(dict(period=period, signal=col, horizon_5m=horizon,
                                 residual_pooled_ic=corr(s.ravel(), target.ravel()),
                                 raw_pooled_ic=corr(s.ravel(), rawtarget.ravel()),
                                 signed_residual_bps=float(np.mean(np.sign(s[valid])*target[valid])*1e4),
                                 active_fraction=float(np.mean(np.abs(s)>1e-8))))
            # Fixed transformations, no calibration: diagnostic gross accounting only.
            # Unit target scales differ; these are not same-risk strategies.
            pnl = np.nan_to_num(s*y).mean(axis=1)
            cumulative.append(dict(period=period, signal=col, gross_log_sum=float(pnl.sum()),
                                   mean_abs=float(np.abs(s).mean()),
                                   signed_next_raw_bps=float(np.nanmean(np.sign(s)*y)*1e4)))
        # Exact pre-shaping decomposition: common vs demeaned unbounded alpha.
        s = frame.dynamic_alpha_strength.to_numpy().reshape(ntime, nasset)
        common = s.mean(axis=1, keepdims=True)
        centered = s-common
        total = np.nan_to_num(s*y).mean(axis=1).sum()
        common_pnl = np.nan_to_num(common*y).mean(axis=1).sum()
        centered_pnl = np.nan_to_num(centered*y).mean(axis=1).sum()
        assert np.isclose(total, common_pnl+centered_pnl, rtol=1e-5, atol=1e-5)
        projections.append(dict(period=period, raw_signal_mean_square=float(np.mean(s*s)),
                                common_mean_square_share=float(np.mean(common*common)/np.mean(s*s)),
                                total_gross_log=float(total), common_gross_log=float(common_pnl),
                                centered_gross_log=float(centered_pnl)))
        # Risk/alpha quality at fixed supplied risk states, no new optimized threshold.
        q = frame.risk_budget.to_numpy()
        for name, mask in [('risk_floor', q <= .500001), ('intermediate', (q>.500001)&(q<.999999)), ('full_budget', q>=.999999)]:
            part = frame.loc[mask]
            buckets.append(dict(period=period, state=name, fraction=float(mask.mean()),
                                net_log_contribution=float(part.net_log_return.sum()/nasset),
                                gross_log_contribution=float(part.gross_log_return.sum()/nasset)))
        # Event study: entry direction at fixed offsets. Equal-event means, overlap allowed.
        pe = pd.read_csv(SRC/'profile_events_5m.csv')
        loc = {d:i for i,d in enumerate(dates)}
        for profile in ('fast','medium','slow'):
            opens = pe[(pe.profile==profile)&pe.event.isin(['open_long','open_short'])]
            opens = opens[opens.date.astype(str).isin(loc)]
            for start, end in ((0,1),(1,3),(3,12),(12,24),(24,48),(48,144)):
                vals=[]
                for event in opens.itertuples():
                    i=loc[str(event.date)]; j=int(event.asset_index)
                    if i+end<=ntime and np.isfinite(y[i+start:i+end,j]).all():
                        vals.append(np.sign(event.new_position)*y[i+start:i+end,j].sum())
                events_out.append(dict(period=period,profile=profile,start_bar=start,end_bar=end,
                                       events=len(vals),signed_raw_bps=float(np.mean(vals)*1e4) if vals else np.nan))
    for name, data in [('signal_horizons',rows),('signal_gross',cumulative),('common_projection',projections),('risk_buckets',buckets),('entry_event_path',events_out)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    # Source selection audit: explicitly exclude holdout columns when reading CSV.
    intake=[]; populations=[]
    for source, group in registry.groupby('source',sort=False):
        path=ROOT/source
        selected=pd.read_csv(path,usecols=lambda c:not c.startswith('holdout_'))
        search=pd.read_csv(path.with_name('search_results.csv'),usecols=lambda c:not c.startswith('holdout_'))
        for pop, data in [('all',search),('fine',search[search.fine_evaluated.fillna(False).astype(bool)]),('selected20',selected),('intake',selected[selected.expr_hash.isin(group.expr_hash)])]:
            populations.append(dict(source=source,population=pop,count=len(data),
                 median_discovery_ic=data.discovery_mean_rank_ic.median(),
                 median_validation_ic=data.validation_mean_rank_ic.median(),
                 median_increment=data.incremental_mean_delta_r2.median(),
                 positive_increment=int((data.incremental_mean_delta_r2>0).sum())))
        intake.append(selected.merge(group[['expr_hash','source_rank','clock']],on='expr_hash'))
    pd.DataFrame(populations).to_csv(OUT/'source_populations.csv',index=False)
    pd.concat(intake).to_csv(OUT/'intake_nonholdout_metrics.csv',index=False)
    weights=pd.read_parquet(SRC/'factor_weights_hourly.parquet')
    weights['year']=weights.date.astype(str).str[:4]
    weight_summary=weights.groupby(['year','clock']).agg(mean_weight=('weight','mean'),active_share=('active','mean'),mean_utility_z=('utility_z','mean'))
    weight_summary.to_csv(OUT/'weight_summary.csv')
    pivot=weights.pivot(index='date',columns='expr_hash',values='weight').fillna(0)
    denom=pivot.sum(axis=1)
    ess=denom**2 / (pivot**2).sum(axis=1).replace(0,np.nan)
    pd.DataFrame({'weight_sum':denom,'effective_factors':ess,'positive_factors':(pivot>0).sum(axis=1)}).groupby(pivot.index.astype(str).str[:4]).mean().to_csv(OUT/'weight_effective.csv')
    # Mathematical reference checks for cumulative horizons and decomposition.
    toy=pd.DataFrame(np.arange(8,dtype=float).reshape(-1,1))
    calc=toy.rolling(3).sum().shift(-2).to_numpy()[:,0]
    assert np.allclose(calc[:6],[3,6,9,12,15,18])
    reconciliation=float(np.max(np.abs(ledger.gross_log_return-ledger.fee_log_return-ledger.net_log_return)))
    assert reconciliation<1e-7
    manifest={'mode':'read_only_frozen_artifact_diagnostic','production_behavior_changed':False,
              'source':str(SRC),'rows':len(ledger),'gross_fee_net_max_error':reconciliation,
              'reference_checks':'forward cumulative sum; linear projection identity; ledger reconciliation passed',
              'limitations':['historical descriptive evidence; not independent confirmation','overlapping event paths; no independence claim','gross signal products are not executable equal-risk strategies'],
              'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [SRC/'factor_registry.csv',SRC/'model_metadata.json',SRC/'forecast_comparison.csv']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(pd.DataFrame(populations).to_string(index=False))
    print(pd.DataFrame(projections).to_string(index=False))
    print(pd.read_csv(OUT/'weight_effective.csv').to_string(index=False))


if __name__=='__main__':
    main()
