"""Shared causal clock, targets, calibration and PnL for aligned crypto research."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

CONTRACT_VERSION = "aligned-5m-path-20260930-v1"


@dataclass(frozen=True)
class Split:
    discovery_end: str
    validation_start: str
    validation_end: str
    holdout_start: str

    def masks(self, dates: np.ndarray) -> dict[str, np.ndarray]:
        labels = np.asarray(dates).astype(str)
        if not (self.discovery_end < self.validation_start <= self.validation_end < self.holdout_start):
            raise ValueError("invalid chronological research split")
        discovery = labels <= self.discovery_end
        validation = (labels >= self.validation_start) & (labels <= self.validation_end)
        holdout = labels >= self.holdout_start
        if not min(discovery.sum(), validation.sum(), holdout.sum()) > 0:
            raise ValueError("all three research periods require observations")
        return {"discovery": discovery, "validation": validation, "holdout": holdout}


def completed_hour(dates: list[str]) -> np.ndarray:
    values = pd.to_datetime(dates, format="%Y%m%d%H%M", utc=True)
    return (values + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M").to_numpy()


def align_hourly_to_5m(hourly_dates: np.ndarray, fast_dates: np.ndarray) -> np.ndarray:
    """Exact completed-hour lookup; a missing bar is an error, not a prior close."""
    index = np.searchsorted(fast_dates, hourly_dates)
    if np.any(index >= len(fast_dates)) or not np.array_equal(fast_dates[index], hourly_dates):
        raise ValueError("hourly completion times must occur in the native 5m price panel")
    return index


def return_targets(log_close_5m: np.ndarray, start_index: np.ndarray,
                   horizons: tuple[int, ...], periods: np.ndarray) -> dict[int, np.ndarray]:
    """Decision at completed bar t, returns from t+1 through t+H, no cross-split labels."""
    result = {}
    n = len(log_close_5m)
    for horizon in horizons:
        end = start_index + horizon
        valid = end < n
        valid &= np.asarray([periods[a] == periods[b] if b < n else False
                             for a, b in zip(start_index, end)])
        label = np.full((len(start_index), log_close_5m.shape[1]), np.nan, dtype=np.float32)
        idx = np.flatnonzero(valid)
        label[idx] = (log_close_5m[end[idx]] - log_close_5m[start_index[idx]]).astype(np.float32)
        result[horizon] = label
    return result


def discovery_betas(hourly_log_close: np.ndarray, discovery: np.ndarray) -> np.ndarray:
    ret = np.diff(hourly_log_close, axis=0, prepend=np.nan)
    nasset = ret.shape[1]
    betas = np.empty(nasset)
    for j in range(nasset):
        others = np.nanmean(np.delete(ret, j, axis=1), axis=1)
        good = discovery & np.isfinite(ret[:, j]) & np.isfinite(others)
        if good.sum() < 720:
            raise ValueError("insufficient discovery returns for beta")
        betas[j] = np.cov(ret[good, j], others[good], ddof=1)[0, 1] / np.var(others[good], ddof=1)
    return betas


def labels(raw: np.ndarray, betas: np.ndarray) -> dict[str, np.ndarray]:
    nasset = raw.shape[1]
    total = np.nansum(raw, axis=1, keepdims=True)
    count = np.isfinite(raw).sum(axis=1, keepdims=True)
    own = np.where(np.isfinite(raw), raw, 0)
    loo = np.divide(total-own, count-np.isfinite(raw),
                    out=np.full_like(raw, np.nan), where=count-np.isfinite(raw)>=3)
    market = np.nanmean(raw, axis=1)
    return {"relative": raw-betas[None, :]*loo, "market": market,
            "absolute": raw, "market_loo": loo}


def signal_transform(x: np.ndarray, median: float, scale: float, conditional: bool) -> np.ndarray:
    """Training-frozen amplitude; inactive gate zeros stay zero."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate signal scale")
    z = np.clip((x - median) / scale, -5, 5)
    z[~np.isfinite(z)] = 0
    if conditional:
        z[np.isfinite(x) & (x == 0)] = 0
    return z.astype(np.float32)


def fit_scale(x: np.ndarray, discovery: np.ndarray, conditional: bool) -> tuple[float, float]:
    sample = x[discovery]
    if conditional:
        sample = sample[sample != 0]
    sample = sample[np.isfinite(sample)]
    if len(sample) < 200:
        raise ValueError("too few active discovery observations")
    median = float(np.median(sample))
    qlo, qhi = np.quantile(sample, [.25, .75])
    scale = float((qhi-qlo)/1.349)
    if scale < 1e-8:
        scale = float(np.std(sample))
    if scale < 1e-8:
        raise ValueError("degenerate discovery signal")
    return median, scale


def positions_from_score(score: np.ndarray, scale: float, budget: float,
                         relative: bool) -> np.ndarray:
    """Apply frozen bps calibration. Relative sleeve removes only common cash exposure."""
    if scale <= 0 or budget < 0:
        raise ValueError("invalid position scale or budget")
    position = np.clip(np.nan_to_num(score/scale, nan=0), -1, 1) * budget
    if relative:
        position -= position.mean(axis=1, keepdims=True)
    return position.astype(np.float32)


def exact_pnl(position: np.ndarray, next_bar_return: np.ndarray,
              cost_bps: float, boundary: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Position at completed bar t earns next completed 5m bar; exact net delta fee."""
    if position.shape != next_bar_return.shape or cost_bps < 0:
        raise ValueError("position/return/cost mismatch")
    previous = np.vstack([np.zeros((1, position.shape[1])), position[:-1]])
    if boundary is not None:
        previous[boundary] = 0
    change = position-previous
    gross = np.where(np.isfinite(next_bar_return), position*next_bar_return, 0)
    fee = np.abs(change)*cost_bps/1e4
    return {"gross": gross, "fee": fee, "net": gross-fee, "change": change}
