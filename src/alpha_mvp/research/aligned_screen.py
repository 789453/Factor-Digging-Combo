"""Purpose- and horizon-aware screening without access to holdout observations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def phase_rotating_coarse_mask(research_mask:np.ndarray,bars_per_day:int,stride:int)->np.ndarray:
    """Deterministic 1/stride sample that covers every intraday clock phase."""
    if bars_per_day<1 or stride<1 or bars_per_day%stride:
        raise ValueError('coarse stride must divide bars per day')
    idx=np.arange(len(research_mask))
    phase=(idx//bars_per_day)%stride
    return research_mask & ((idx+phase)%stride==0)


def conditional_expression(expr: str) -> bool:
    return ("GatePos(" in expr or "GateNeg(" in expr
            or "Mul($us_day_flag" in expr or "Mul($us_evening_flag" in expr
            or "Mul($us_overnight_flag" in expr)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 48:
        return np.nan
    a = x[good].astype(float); b = y[good].astype(float)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt(np.dot(a, a)*np.dot(b, b))
    return float(np.dot(a, b)/den) if den > 1e-12 else np.nan


def score_one(values: np.ndarray, target: np.ndarray, research_mask: np.ndarray,
              role: str, conditional: bool, min_active: int = 240) -> dict:
    """Signed discovery-only information; one common observation per timestamp."""
    if values.ndim != 2 or len(values) != len(research_mask):
        raise ValueError("candidate panel does not align with research dates")
    mask = research_mask[:, None] & np.isfinite(values)
    if conditional:
        mask &= values != 0
    coverage = float(mask.sum() / max(1, research_mask.sum()*values.shape[1]))
    if conditional and (mask.sum() < min_active or coverage < .01 or coverage > .90):
        return {"status": "INSUFFICIENT_ACTIVE", "coverage": coverage}
    if role in {"beta", "risk"}:
        # Common labels are one observation per time, not 12 independent assets.
        keep = mask.sum(axis=1) >= max(3, values.shape[1]//2)
        if keep.sum() < 100:
            return {"status": "LOW_COVERAGE", "coverage": coverage}
        x = np.nanmedian(np.where(mask, values, np.nan)[keep], axis=1)
        y = target[keep]
        ic = _correlation(x, y)
        edge = float(np.nanmean(np.sign(x)*y)*1e4) if role == "beta" else np.nan
        breadth = np.nan
        active_ic=np.nan;basket_ic=np.nan
    else:
        if target.shape != values.shape:
            raise ValueError("relative target must have candidate panel shape")
        per_asset = [_correlation(values[mask[:,j],j],target[mask[:,j],j])
                     for j in range(values.shape[1])]
        good = np.asarray(per_asset)[np.isfinite(per_asset)]
        active_ic = float(good.mean()) if len(good) else np.nan
        breadth = float(np.mean(np.sign(good)==np.sign(active_ic))) if len(good) else np.nan
        # The downstream alpha product is a cash-neutral basket. Preserve
        # inactive gate zeros while removing only that basket's common score.
        count=np.isfinite(values).sum(axis=1,keepdims=True)
        mean=np.divide(np.nansum(values,axis=1,keepdims=True),count,
                       out=np.zeros((len(values),1)),where=count>0)
        basket=values-mean
        valid=research_mask[:,None] & np.isfinite(basket) & np.isfinite(target)
        basket_ic=_correlation(basket[valid],target[valid])
        ic=basket_ic
        edge=float(np.mean(np.sign(basket[valid])*target[valid])*1e4) if valid.any() else np.nan
    if not np.isfinite(ic):
        return {"status": "NO_IC", "coverage": coverage}
    # Coarse ranking is intentionally cheap; validation is not read here.
    strength=max(abs(ic),.5*abs(active_ic)) if conditional and np.isfinite(active_ic) else abs(ic)
    confidence = strength*np.sqrt(max(min(coverage,1), .01))
    return {"status": "OK", "coverage": coverage, "raw_ic": ic,
            "direction": int(np.sign(ic)), "signed_ic": abs(ic),
            "signed_edge_bps": edge*np.sign(ic) if np.isfinite(edge) else np.nan,
            "positive_asset_share": breadth, "active_time_ic": active_ic,
            "basket_ic":basket_ic,"coarse_score": confidence}


def screen_horizons(values: np.ndarray, targets: dict[int, np.ndarray],
                    research_mask: np.ndarray, role: str, conditional: bool) -> dict:
    choices = []
    for horizon, target in targets.items():
        metric = score_one(values, target, research_mask, role, conditional)
        if metric["status"] == "OK":
            choices.append((metric["coarse_score"], -horizon, horizon, metric))
    if not choices:
        return {"status": "NO_VALID_HORIZON", "coarse_score": np.nan}
    _, _, horizon, metric = max(choices)
    return {**metric, "horizon_bars": horizon}


def diversity_pick(frame: pd.DataFrame, n: int, per_family: int,
                   per_field: int, per_horizon: int) -> pd.DataFrame:
    """Quota on mechanism, primary field and target horizon, with stable ties."""
    ranked = frame[frame.status.eq("OK")].sort_values(
        ["coarse_score", "expr_hash"], ascending=[False, True])
    chosen=[]; families={}; fields={}; horizons={}
    for row in ranked.itertuples():
        family=row.template_name
        field=row.fields.split("|")[0] if isinstance(row.fields,str) else "unknown"
        horizon=int(row.horizon_bars)
        if (families.get(family,0)>=per_family or fields.get(field,0)>=per_field
                or horizons.get(horizon,0)>=per_horizon):
            continue
        chosen.append(row.Index)
        families[family]=families.get(family,0)+1
        fields[field]=fields.get(field,0)+1
        horizons[horizon]=horizons.get(horizon,0)+1
        if len(chosen)==n:break
    return frame.loc[chosen].copy().reset_index(drop=True)
