"""Frozen-model execution and post-freeze evidence for aligned crypto research."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aligned_combo import forecast, hourly_positions, scheduled_risk_budget
from .aligned_contract import exact_pnl


def _stats(net: np.ndarray, gross: np.ndarray, fee: np.ndarray,
           position: np.ndarray, mask: np.ndarray) -> dict:
    hourly = net[mask]
    if not len(hourly):
        raise ValueError("empty evidence period")
    daily = pd.Series(hourly).groupby(np.arange(len(hourly)) // 288).sum().to_numpy()
    curve = np.exp(np.cumsum(daily))
    draw = curve / np.maximum.accumulate(np.r_[1., curve])[1:] - 1
    return {
        "bars": int(mask.sum()), "days": float(mask.sum()/288),
        "net_log_return": float(hourly.sum()),
        "simple_return": float(np.expm1(hourly.sum())),
        "gross_log_return": float(gross[mask].sum()),
        "fee_log_return": float(fee[mask].sum()),
        "net_mean_5m_bps": float(hourly.mean()*1e4),
        "daily_sharpe": float(np.sqrt(365)*daily.mean()/daily.std(ddof=1))
            if len(daily)>1 and daily.std(ddof=1)>0 else np.nan,
        "max_drawdown": float(draw.min()),
        "mean_abs_position": float(np.mean(np.abs(position[mask]))),
    }


def finish_aligned_report(out: Path, cfg: dict, selected: pd.DataFrame,
                          models: dict, record_map: dict, evaluator,
                          dates: np.ndarray, fast_dates: np.ndarray,
                          fast_idx: np.ndarray, fast_close: np.ndarray,
                          one_hour: np.ndarray, risk: dict,
                          masks: dict, boundary: np.ndarray,
                          native_position:np.ndarray|None=None) -> dict:
    """Called only after research_freeze.json exists. Never returns ranking inputs."""
    freeze = out/'research_freeze.json'
    if not freeze.exists() or json.loads(freeze.read_text())['status'] != 'FROZEN_BEFORE_HOLDOUT':
        raise ValueError('holdout cannot be opened before the research decision is frozen')
    cost=float(cfg['combo']['cost_bps'])
    alpha=[]; beta=[]; risk_forecasts=[]; risk_baselines=[]; factor_audit=[]
    for row in selected.itertuples():
        values,status=evaluator.eval_expr(record_map[row.expr_hash].expr)
        if status != 'OK' or values is None:
            raise ValueError(f'cannot reevaluate frozen factor {row.expr_hash}: {status}')
        model=models[row.candidate_id]
        pred=forecast(values,model)
        if row.role=='risk':
            target=risk[int(row.horizon_bars)]
            baseline=float(np.nanmean(target[masks['discovery']]))
            if float(row.validation_delta_r2)>0:
                risk_baselines.append(baseline)
                risk_forecasts.append(baseline+pred)
            valid=masks['holdout'] & np.isfinite(target) & np.isfinite(pred)
            mse=float(np.mean((target[valid]-(baseline+pred[valid]))**2)) if valid.any() else np.nan
            base=float(np.mean((target[valid]-baseline)**2)) if valid.any() else np.nan
            factor_audit.append({'candidate_id':row.candidate_id,'expr_hash':row.expr_hash,'role':row.role,
                                 'holdout_delta_r2':1-mse/base if base>0 else np.nan})
            continue
        if row.role=='beta':
            pred=np.broadcast_to(pred[:,None],one_hour.shape)
        budget=float(cfg['combo']['beta_budget' if row.role=='beta' else 'alpha_budget'])
        pos=hourly_positions(pred,dates,int(row.horizon_bars),
                             float(model['prediction_scale']),budget,row.role!='beta',cost)
        if row.validation_trade_enabled:
            (beta if row.role=='beta' else alpha).append(pos)
        # A selected-factor audit is written after freeze; never used to reselect.
        valid=masks['holdout'] & np.isfinite(one_hour).all(axis=1)
        factor_audit.append({'candidate_id':row.candidate_id,'expr_hash':row.expr_hash,'role':row.role,
                             'holdout_gross_hour_bps':float(np.mean((pos[valid]*one_hour[valid]))*1e4)
                                 if valid.any() else np.nan})
    alpha_hour=np.mean(np.stack(alpha),axis=0) if alpha else np.zeros_like(one_hour)
    beta_hour=np.mean(np.stack(beta),axis=0) if beta else np.zeros_like(one_hour)
    # The risk policy and its validation gate were frozen before this function.
    risk_enabled=bool(json.loads(freeze.read_text())['risk_enabled'])
    risk_budget=np.ones(len(dates),dtype=np.float32)
    if risk_enabled:
        pred_risk=np.mean(np.stack(risk_forecasts),axis=0)
        risk_budget=scheduled_risk_budget(pred_risk,dates,masks['discovery'],
                                          cfg['combo']['risk_floor'])
    # Last completed 5m bar before an hourly decision carries the previous state.
    hour_at_bar=np.searchsorted(dates,fast_dates,side='right')-1
    hour_at_bar=np.clip(hour_at_bar,0,len(dates)-1)
    active=fast_dates>=dates[0]
    native_enabled=bool(json.loads(freeze.read_text()).get('native_enabled',False))
    if native_enabled and native_position is None:
        raise ValueError('frozen native5 decision requires its 5m position panel')
    hour_alpha_budget=1.
    if native_enabled:
        hour_alpha_budget=(cfg['combo']['alpha_budget']-cfg['native5']['alpha_budget'])/cfg['combo']['alpha_budget']
    alpha_fast=alpha_hour[hour_at_bar].copy()*hour_alpha_budget
    beta_fast=beta_hour[hour_at_bar].copy()
    alpha_fast[~active]=0;beta_fast[~active]=0
    if native_enabled:alpha_fast+=native_position
    risk_fast=risk_budget[hour_at_bar].copy();risk_fast[~active]=1
    main_fast=(alpha_fast+beta_fast)*risk_fast[:,None]
    max_abs=float(cfg['combo']['max_abs_position'])
    if np.max(np.abs(main_fast))>max_abs+1e-5:
        main_fast=np.clip(main_fast,-max_abs,max_abs)
    log_close=np.log(np.where(fast_close>0,fast_close,np.nan))
    next_bar=np.vstack([np.diff(log_close,axis=0),np.full((1,fast_close.shape[1]),np.nan)])
    period_masks={
        'discovery':fast_dates<=cfg['split']['discovery_end'],
        'validation':(fast_dates>=cfg['split']['validation_start']) &
                     (fast_dates<=cfg['split']['validation_end']),
        'holdout':fast_dates>=cfg['split']['holdout_start'],
    }
    split_boundary=np.zeros(len(fast_dates),dtype=bool)
    for name in ('discovery','validation','holdout'):
        idx=np.flatnonzero(period_masks[name])
        if len(idx):split_boundary[idx[0]]=True
    bars=[]; rows=[]
    strategies=[('alpha',alpha_fast),('beta',beta_fast),('main',main_fast)]
    if native_position is not None:strategies.append(('native5',native_position))
    for name,pos in strategies:
        ledger=exact_pnl(pos,next_bar,cost,split_boundary)
        gross=np.mean(ledger['gross'],axis=1)
        fee=np.mean(ledger['fee'],axis=1)
        net=gross-fee
        # A return crossing into another period is excluded from that period's statistics.
        for period,mask in period_masks.items():
            usable=mask & np.r_[mask[1:],False] & np.isfinite(next_bar).all(axis=1)
            rows.append({'strategy':name,'period':period,**_stats(net,gross,fee,pos,usable)})
        bars.append(pd.DataFrame({'date':fast_dates,'strategy':name,'gross':gross,
                                  'fee':fee,'net':net,'abs_position':np.mean(np.abs(pos),axis=1)}))
        if name=='main':
            audit=pd.DataFrame({'date':np.repeat(fast_dates,fast_close.shape[1]),
                                'asset':np.tile(np.asarray(evaluator.codes),len(fast_dates)),
                                'position':pos.ravel(),
                                'next_log_return':next_bar.ravel(),
                                'gross':ledger['gross'].ravel(),
                                'fee':ledger['fee'].ravel(),
                                'net':ledger['net'].ravel()})
            audit.to_parquet(out/'execution_ledger_5m.parquet',index=False,compression='zstd')
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'strategy_metrics.csv',index=False)
    pd.DataFrame(factor_audit).to_csv(out/'holdout_factor_audit.csv',index=False)
    bar_frame=pd.concat(bars,ignore_index=True)
    bar_frame.to_parquet(out/'portfolio_5m.parquet',index=False,compression='zstd')
    pd.DataFrame({'date':dates,'risk_budget':risk_budget,
                  'alpha_abs':np.mean(np.abs(alpha_fast[fast_idx]),axis=1),
                  'beta_abs':np.mean(np.abs(beta_fast[fast_idx]),axis=1),
                  'native5_abs':np.mean(np.abs(native_position[fast_idx]),axis=1)
                       if native_position is not None else np.zeros(len(dates)),
                  'main_abs':np.mean(np.abs(main_fast[fast_idx]),axis=1)}).to_parquet(
                      out/'hourly_decisions.parquet',index=False)
    _html(out,summary,selected,bar_frame,risk_enabled)
    return {'risk_enabled':risk_enabled,'native5_enabled':native_enabled,'metrics':rows,
            'files':['strategy_metrics.csv','holdout_factor_audit.csv',
                     'portfolio_5m.parquet','execution_ledger_5m.parquet',
                     'hourly_decisions.parquet','field_baselines.csv','index.html']+
                    (['native5_search.csv','native5_fine.csv','native5_selected.csv']
                     if native_position is not None else [])}


def _html(out:Path,summary:pd.DataFrame,selected:pd.DataFrame,
          bars:pd.DataFrame,risk_enabled:bool):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=1,cols=3,subplot_titles=['Discovery','Validation','Holdout'])
    for col,period in enumerate(['discovery','validation','holdout'],1):
        for strategy in ['alpha','beta','main','native5']:
            frame=bars[(bars.strategy==strategy) &
                       ((bars.date<='202506302355') if period=='discovery' else
                        ((bars.date>='202507010000') & (bars.date<='202601312355'))
                         if period=='validation' else (bars.date>='202602010000'))]
            daily=frame.iloc[::288].copy()
            equity=np.exp(np.cumsum(frame.net.to_numpy()))
            fig.add_trace(go.Scatter(x=frame.date.iloc[::288],y=equity[::288],
                           name=strategy,legendgroup=strategy,showlegend=col==1),row=1,col=col)
    fig.update_layout(template='plotly_dark',height=520,title='Frozen factor combo · exact 5m net path',
                      hovermode='x unified')
    chart=fig.to_html(full_html=False,include_plotlyjs='cdn')
    style='''<style>body{background:#0b1020;color:#e9efff;font:15px system-ui;margin:36px auto;max-width:1400px}h1{font-size:30px}p{color:#afbdd6}table{border-collapse:collapse;width:100%;font-size:12px;margin:22px 0}th,td{border-bottom:1px solid #2b344b;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#83baff}.card{background:#131d33;border:1px solid #25314b;padding:18px;margin:18px 0;border-radius:12px}</style>'''
    links=' '.join(f'<a href="{x}">{x}</a>' for x in ['strategy_metrics.csv','selected_factors.csv',
           'search_results.csv','fine_results.csv','field_baselines.csv',
           'holdout_factor_audit.csv','research_freeze.json']+
           (['native5_selected.csv','native5_fine.csv'] if 'native5' in bars.strategy.values else []))
    html=f'''<!doctype html><meta charset="utf-8"><title>Aligned crypto factor research</title>{style}
    <h1>一体化因子挖掘与组合</h1><p>2023-01 至 2025-06 discovery · 2025-07 至 2026-01 validation · 2026-02 后 holdout。全部决策在 holdout 审计前冻结。收益为逐资产等权、双向、5m 下一根对数收益扣变仓费用。</p>
    <div class="card">{chart}</div><h2>分段净值指标</h2>{summary.to_html(index=False,float_format=lambda v:f'{v:.4f}')}
    <h2>冻结入选因子</h2>{selected[['role','expr_hash','expr','horizon_bars','validation_net_mean_bar_bps'] if 'validation_net_mean_bar_bps' in selected else ['role','expr_hash','expr','horizon_bars']].to_html(index=False,escape=True)}
    <p>风险预算启用：{risk_enabled}。风险指标为发现期标定的未来下行路径预测，仅验证期相对常数基准改善时进入预算。</p><p>{links}</p>'''
    (out/'index.html').write_text(html,encoding='utf-8')
