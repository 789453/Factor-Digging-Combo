"""Simple calibrated forecasts and fixed-horizon execution for aligned research."""
from __future__ import annotations

import numpy as np

from .aligned_contract import fit_scale, signal_transform, positions_from_score


def fit_candidate(values: np.ndarray, target: np.ndarray, discovery: np.ndarray,
                  role: str, conditional: bool, direction: int) -> dict:
    """Fit only discovery amplitude. No validation/holdout row enters a coefficient."""
    if direction not in (-1, 1):
        raise ValueError("candidate needs a frozen discovery direction")
    common = role in {"beta", "risk"}
    source = np.nanmedian(values, axis=1) if common else values
    median, scale = fit_scale(source, discovery, conditional)
    z = signal_transform(source, median, scale, conditional) * direction
    if not common:
        z=z-z.mean(axis=1,keepdims=True)
    if common:
        good = discovery & np.isfinite(target) & np.isfinite(source)
        if conditional:
            good &= source != 0
    else:
        good = discovery[:, None] & np.isfinite(target) & np.isfinite(source)
    if good.sum() < 200:
        raise ValueError("too few discovery labels for calibration")
    xx = z[good].astype(float); yy = target[good].astype(float)
    ridge = max(100., .02*len(xx))
    slope = max(0., float(np.dot(xx, yy)/(np.dot(xx, xx)+ridge*np.var(xx))))
    if slope < 1e-12:
        raise ValueError("frozen direction has no positive discovery payoff")
    return {"median": median, "scale": scale, "direction": direction,
            "slope": slope, "conditional": conditional, "role": role}


def forecast(values: np.ndarray, model: dict) -> np.ndarray:
    common = model["role"] in {"beta", "risk"}
    source = np.nanmedian(values, axis=1) if common else values
    z = signal_transform(source, model["median"], model["scale"], model["conditional"])
    if not common:
        z=z-z.mean(axis=1,keepdims=True)
    return model["slope"]*model["direction"]*z


def hourly_positions(prediction: np.ndarray, dates: np.ndarray, horizon: int,
                     training_scale: float, budget: float, relative: bool,
                     fee_bps: float) -> np.ndarray:
    """UTC-anchored nonoverlapping decisions at declared holding horizon."""
    if horizon % 12 or horizon < 12:
        raise ValueError("hourly feature horizon must be a whole number of hours")
    ntime = len(dates)
    nasset = prediction.shape[1] if prediction.ndim == 2 else 1
    out = np.zeros((ntime, nasset), dtype=np.float32)
    h = horizon//12
    proposed = positions_from_score(prediction if prediction.ndim==2 else prediction[:,None],
                                    training_scale, budget, relative)
    for t in range(ntime):
        if int(str(dates[t])[-4:-2]) % h != 0:
            out[t] = out[t-1] if t else 0
            continue
        new = proposed[t]
        # The previous forecast expires after H bars. A fresh forecast must
        # cover its own round-trip fee, otherwise the position goes flat.
        predicted = np.abs(prediction[t])
        if np.ndim(predicted)==0:
            predicted=np.full(nasset,predicted)
        if relative:
            # Trade or flatten the entire basket, preserving neutrality.
            expected_gain=float(np.mean(new*prediction[t]))
            round_trip=float(2*fee_bps/1e4*np.mean(np.abs(new)))
            out[t]=new if expected_gain>=round_trip else 0
        else:
            worthwhile = predicted >= 2*fee_bps/1e4*np.abs(new)
            out[t] = np.where(worthwhile,new,0)
    return out


def hourly_pnl(position: np.ndarray, next_hour_raw: np.ndarray, cost_bps: float,
               boundary: np.ndarray) -> dict:
    if position.shape != next_hour_raw.shape:
        raise ValueError("hourly positions and return panel mismatch")
    previous=np.vstack([np.zeros((1,position.shape[1])),position[:-1]])
    previous[boundary]=0
    change=position-previous
    gross=np.nan_to_num(position*next_hour_raw,nan=0)
    fee=np.abs(change)*cost_bps/1e4
    return {"gross":gross,"fee":fee,"net":gross-fee,"turnover":np.abs(change)}


def utility_metrics(position: np.ndarray, next_hour_raw: np.ndarray,
                    cost_bps: float, mask: np.ndarray, boundary: np.ndarray) -> dict:
    ledger=hourly_pnl(position,next_hour_raw,cost_bps,boundary)
    valid = mask[:,None]
    gross=float(np.mean(ledger["gross"][valid.repeat(position.shape[1],axis=1)]))
    fee=float(np.mean(ledger["fee"][valid.repeat(position.shape[1],axis=1)]))
    turnover=float(np.mean(ledger["turnover"][valid.repeat(position.shape[1],axis=1)]))
    return {"gross_mean_bar_bps":gross*1e4,"fee_mean_bar_bps":fee*1e4,
            "net_mean_bar_bps":(gross-fee)*1e4,"turnover_mean_bar":turnover,
            "break_even_cost_bps":gross/turnover*1e4 if turnover>0 else np.nan,
            "average_abs_position":float(np.mean(np.abs(position[mask])))}


def scheduled_risk_budget(prediction: np.ndarray, dates: np.ndarray,
                          discovery: np.ndarray, floor: float) -> np.ndarray:
    """Discovery-frozen 4h risk gate; no continuous hourly resizing."""
    sample=prediction[discovery & np.isfinite(prediction)]
    if len(sample)<200 or not 0<floor<=1:
        raise ValueError('risk budget needs discovery observations and positive floor')
    q50,q90=np.quantile(sample,[.5,.9])
    width=max(float(q90-q50),1e-9)
    proposed=np.clip(1-(prediction-q50)/width*(1-floor),floor,1)
    proposed[~np.isfinite(proposed)]=1
    hours=np.array([int(str(x)[-4:-2]) for x in dates])
    decision=hours%4==0
    anchor=np.maximum.accumulate(np.where(decision,np.arange(len(dates)),0))
    return proposed[anchor].astype(np.float32)
