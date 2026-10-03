"""Controlled replay of v9 execution from saved signals, not a pipeline.

All thresholds and signal denominators remain frozen to the original run.
No model fitting, parameter selection, or production behavior changes.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.alpha_mvp.research.factor_combo_signal_engine import (
    SignalEngineSpec,_segmented_ema,_simulate_profile,_net_from_positions)

SRC=ROOT/'outputs/crypto_factor_combo_signal_engine_20260929_v9'
OUT=ROOT/'outputs/frozen_execution_ablation_20260930_v1'


def main():
    OUT.mkdir(exist_ok=False)
    cfg=yaml.safe_load((ROOT/'configs/research/crypto_factor_combo_baseline_2023_2025.yaml').read_text())
    spec=SignalEngineSpec.from_dict(cfg['signal_engine'])
    ledger=pd.read_parquet(SRC/'asset_positions_5m.parquet')
    nasset=ledger.asset.nunique(); ntime=len(ledger)//nasset
    def panel(c):return ledger[c].to_numpy().reshape(ntime,nasset)
    dates=panel('date')[:,0].astype(str); period=panel('period')[:,0]
    lengths=[int(np.sum(np.char.startswith(dates,year))) for year in ['2023','2024','2025']]
    q=panel('risk_budget')[:,0]; y=panel('next_raw_log_return')
    cal=pd.read_csv(SRC/'trigger_calibration.csv')
    rows=[]; baseline_error=None
    for case in ['original','remove_beta','remove_hourly','remove_native5','risk_budget_one','min_hold_one_bar']:
        print('replay',case,flush=True)
        total=np.zeros_like(y)
        for profile in spec.profiles:
            strength=panel(profile.name+'_strength').copy()
            scale=panel(profile.name+'_scale')
            removed=None
            if case=='remove_beta':removed=profile.beta_share*panel('beta_direction')
            elif case=='remove_hourly':removed=profile.hourly_share*panel('hourly_alpha_unit')
            elif case=='remove_native5':removed=profile.native5_share*panel('native5_alpha_unit')
            if removed is not None:
                strength-=_segmented_ema(removed,lengths,profile.half_life_bars)/scale
            if case=='min_hold_one_bar':profile=replace(profile,min_hold_bars=1)
            entry=float(cal[(cal.profile==profile.name)&cal.selected].entry_threshold.iloc[0])
            position,_,_,_=_simulate_profile(strength,np.ones_like(q) if case=='risk_budget_one' else q,
                                            lengths,profile,entry,collect=False)
            total+=position
        total=np.clip(total,-spec.max_abs_position,spec.max_abs_position)
        if case=='original':
            baseline_error=float(np.max(np.abs(total-panel('position'))))
            assert baseline_error<1e-6,baseline_error
        delta,net=_net_from_positions(total,y,lengths,4.)
        for label in dict.fromkeys(period):
            mask=period==label
            daily=pd.Series(net[mask].mean(axis=1)).groupby([d[:8] for d in dates[mask]]).sum()
            gross=np.nan_to_num(total[mask]*y[mask]).mean(axis=1).sum()
            netlog=net[mask].mean(axis=1).sum()
            market=np.nan_to_num(y[mask]).mean(axis=1)
            common=total[mask].mean(axis=1)
            rows.append(dict(case=case,period=label,gross_return_pct=float(np.expm1(gross)*100),
                             net_return_pct=float(np.expm1(netlog)*100),
                             daily_sharpe=float(daily.mean()/daily.std(ddof=0)*np.sqrt(365)),
                             turnover=float(np.abs(delta[mask]).mean(axis=1).sum()),
                             mean_abs_position=float(np.abs(total[mask]).mean()),
                             mean_net_position=float(total[mask].mean()),
                             common_gross_log=float(np.sum(common*market)),
                             relative_gross_log=float(gross-np.sum(common*market))))
    pd.DataFrame(rows).to_csv(OUT/'ablation_metrics.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'baseline_max_position_error':baseline_error,
       'source':str(SRC),'cost_bps':4,'frozen':'original thresholds, scale denominators, factor weights, train anchors',
       'changed':'one named component per case','not_equal_risk':True,
       'purpose':'historical mechanistic ablation; no optimized strategy selected'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__=='__main__':main()
