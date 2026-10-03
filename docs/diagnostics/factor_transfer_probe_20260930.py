"""Bounded replay of 18 frozen factors, using existing production functions.

No mining, fitting, strategy optimization, or mutation of completed artifacts.
Uses saved v9 scalers/weights; quantifies signal transformations and attribution.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.alpha_mvp.research.factor_combo import _segment, _registry
from factor_information_audit_20260930 import corr

SRC=ROOT/'outputs/crypto_factor_combo_signal_engine_20260929_v9'
OUT=ROOT/'outputs/factor_transfer_probe_20260930_v3'


def main():
    OUT.mkdir(exist_ok=False)
    cfg=yaml.safe_load((ROOT/'configs/research/crypto_factor_combo_baseline_2023_2025.yaml').read_text())
    registry=pd.read_csv(SRC/'factor_registry.csv')
    source_registry=_registry(cfg)
    meta=json.loads((SRC/'model_metadata.json').read_text())
    scaler=meta['scalers']['raw']
    saved_weights=pd.read_parquet(SRC/'factor_weights_5m.parquet')
    saved_ledger=pd.read_parquet(SRC/'asset_positions_5m.parquet')
    predictions=pd.read_parquet(SRC/'predictions_5m.parquet')
    factor_rows=[]; shape_rows=[]; stages=[]; weight_clock=[]; beta_rows=[]
    reference_error=[]
    # Replay the 18-factor inventory over the original three Q1 segments.
    # Identical warmup and saved weights; diagnostic only, no re-selection.
    for spec in cfg['segments']:
        print('reconstruct',spec['name'],flush=True)
        segment=_segment(cfg,spec,source_registry)
        segment['factor']=segment['factor'][:,:,registry.original_column.to_numpy(dtype=int)]
        ntime,nasset,nf=segment['factor'].shape
        dates=segment['dates'].astype(str)
        raw=segment['factor'].astype(float)
        z=np.clip((raw-np.array(scaler['median']))/np.array(scaler['scale']),-5,5)
        z=np.nan_to_num(z,nan=0).astype(np.float32)
        w=saved_weights[saved_weights.date.astype(str).isin(dates)].set_index(saved_weights.columns[0])
        assert np.array_equal(w.index.astype(str),dates)
        wa=w[registry.expr_hash].to_numpy()
        ledger=saved_ledger[saved_ledger.date.astype(str).isin(dates)]
        pred=predictions[predictions.date.astype(str).isin(dates)]
        assert len(ledger)==ntime*nasset
        ry=pred.residual_y.to_numpy().reshape(ntime,nasset)
        y=pred.raw_y.to_numpy().reshape(ntime,nasset)
        dynamic=np.einsum('tnf,tf->tn',z,wa)
        ref=ledger.dynamic_alpha_strength.to_numpy().reshape(ntime,nasset)
        err=float(np.max(np.abs(dynamic-ref)))
        affected=np.abs(dynamic-ref)>2e-5
        reference_error.append({'year':spec['name'],'max_reconstruction_error':err,
                                'affected_asset_bars':int(affected.sum()),
                                'total_asset_bars':int(affected.size),
                                'accepted_for_factor_diagnostics':err<2e-5})
        if err>=2e-5:
            ti,ai=np.where(affected)
            pd.DataFrame({'date':dates[ti],'asset':[segment['codes'][j] for j in ai],
                          'reconstructed':dynamic[ti,ai],'stored':ref[ti,ai]}).to_csv(
                              OUT/f'reconstruction_mismatch_{spec["name"]}.csv',index=False)
            print('excluded nonmatching replay',spec['name'],err,int(affected.sum()),flush=True)
            continue
        period=ledger.period.to_numpy().reshape(ntime,nasset)[:,0]
        for label in dict.fromkeys(period):
            mask=period==label
            zm=z[mask]; wm=wa[mask]; ym=y[mask]; rym=ry[mask]
            native=(registry.clock=='native5').to_numpy()
            for clock, cols in [('native5',native),('hourly',~native)]:
                sums=wm[:,cols].sum(axis=1)
                sig=np.einsum('tnf,tf->tn',zm[:,:,cols],wm[:,cols])
                weight_clock.append(dict(period=label,clock=clock,mean_total_weight=float(sums.mean()),
                                         time_zero_weight=float(np.mean(sums==0)),
                                         raw_signal_rms=float(np.sqrt(np.mean(sig**2)))))
            for j,row in enumerate(registry.itertuples()):
                orig=raw[mask,:,j]; scaled=zm[:,:,j]
                off=np.isfinite(orig)&(orig==0)
                shape_rows.append(dict(period=label,expr_hash=row.expr_hash,clock=row.clock,
                  original_zero_share=float(off.mean()),
                  inactive_scaled_mean=float(scaled[off].mean()) if off.any() else np.nan,
                  clipped_share=float(np.mean(abs(scaled)>=5)),
                  median_training=float(scaler['median'][j]),scale_training=float(scaler['scale'][j])))
                for horizon,lag in [(1,0),(3,0),(12,0),(24,0),(48,0),(48,12),(144,0),(288,0)]:
                    future=pd.DataFrame(rym).rolling(horizon,min_periods=horizon).sum().shift(1-horizon-lag).to_numpy()
                    per_asset=[corr(scaled[:,i],future[:,i]) for i in range(nasset)]
                    factor_rows.append(dict(period=label,expr_hash=row.expr_hash,clock=row.clock,
                         horizon_5m=horizon,lag_5m=lag,
                         per_asset_mean_ic=float(np.nanmean(per_asset)),
                         positive_asset_share=float(np.mean(np.asarray(per_asset)>0)),
                         raw_pooled_ic=corr(orig.ravel(),future.ravel()),
                         scaled_pooled_ic=corr(scaled.ravel(),future.ravel())))
            # Equal gross exposure 0.1 across each evaluation period: descriptive scaling,
            # never a causal trading proposal; turnover costs expose the fast-alpha hurdle.
            family=np.stack([zm[:,:,(registry.family==f).to_numpy()].mean(axis=2)
                             for f in dict.fromkeys(registry.family)],axis=2).mean(axis=2)
            for name,signal in [('raw_equal',zm.mean(axis=2)),('family_equal',family),('dynamic',dynamic[mask])]:
                for center in (False,True):
                    s=signal-signal.mean(axis=1,keepdims=True) if center else signal
                    s=s*(.1/np.abs(s).mean())
                    gross=np.nan_to_num(s*ym).mean(axis=1)
                    turnover=np.abs(np.diff(s,axis=0,prepend=np.zeros((1,nasset)))).mean(axis=1)
                    # flatten segment endpoint; makes round-trip bookkeeping complete
                    turnover[-1]+=np.abs(s[-1]).mean()
                    stages.append(dict(period=label,signal=name,centered=center,gross_log_sum=float(gross.sum()),
                                       turnover=float(turnover.sum()),net_log_4bps=float(gross.sum()-turnover.sum()*.0004),
                                       break_even_bps=float(gross.sum()/turnover.sum()*1e4)))
            # Beta direction component: mean drift vs varying market timing, exact gross identity.
            beta=ledger.beta_direction.to_numpy().reshape(ntime,nasset)[mask,0]
            market=np.nan_to_num(ym).mean(axis=1)
            beta_rows.append(dict(period=label,mean_direction=float(beta.mean()),
                                  long_share=float(np.mean(beta>0)),
                                  gross_total=float(np.sum(beta*market)),
                                  constant_exposure_gross=float(beta.mean()*market.sum()),
                                  centered_timing_gross=float(np.sum((beta-beta.mean())*market)),
                                  next_market_ic=corr(beta,market)))
        print('done',spec['name'],'reconstruction error',err,flush=True)
    for name,rows in [('factor_horizons',factor_rows),('factor_shapes',shape_rows),('linear_combos_fixed_gross',stages),('clock_weights',weight_clock),('beta_decomposition',beta_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    manifest={'kind':'frozen factor replay, no fitting/selection','source':str(SRC),'factors':len(registry),
              'reconstruction':reference_error,'periods':'2023 Q1; 2024 Q1; 2025 Q1',
              'fixed_gross_warning':'ex-post common exposure for diagnostic comparability only; not an executable strategy',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(weight_clock).to_string(index=False))
    print(pd.DataFrame(stages).to_string(index=False))
    print(pd.DataFrame(beta_rows).to_string(index=False))


if __name__=='__main__':
    main()
