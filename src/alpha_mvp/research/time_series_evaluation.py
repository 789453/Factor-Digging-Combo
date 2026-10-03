from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fastops
from .config import EvaluationConfig, SplitConfig
from .evaluation import EvaluationResult, PeriodMetrics, _period_metrics


def asset_time_ic(
    factor: np.ndarray,
    forward_return: np.ndarray,
    mask: np.ndarray,
    min_observations: int = 168,
) -> np.ndarray:
    """Pearson IC through time for each asset; never mixes asset identities."""
    x = np.asarray(factor, dtype=float)[mask]
    y = np.asarray(forward_return, dtype=float)[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    left = np.where(valid, x, 0.0)
    right = np.where(valid, y, 0.0)
    sx = left.sum(axis=0)
    sy = right.sum(axis=0)
    covariance = (left * right).sum(axis=0) - sx * sy / np.maximum(count, 1)
    variance_x = np.square(left).sum(axis=0) - sx * sx / np.maximum(count, 1)
    variance_y = np.square(right).sum(axis=0) - sy * sy / np.maximum(count, 1)
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    output = np.full(x.shape[1], np.nan, dtype=float)
    usable = (count >= min_observations) & (denominator > 1e-12)
    output[usable] = covariance[usable] / denominator[usable]
    return output


def block_time_ic(
    factor: np.ndarray,
    forward_return: np.ndarray,
    mask: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Non-overlapping block IC, averaged across per-asset temporal ICs."""
    indices = np.flatnonzero(mask)
    values = []
    for start in range(0, len(indices), block_size):
        block = indices[start:start + block_size]
        if len(block) < max(24, block_size // 2):
            continue
        block_mask = np.zeros(len(mask), dtype=bool)
        block_mask[block] = True
        per_asset = asset_time_ic(
            factor,
            forward_return,
            block_mask,
            min_observations=max(24, len(block) // 2),
        )
        finite = per_asset[np.isfinite(per_asset)]
        if len(finite):
            values.append(float(finite.mean()))
    return np.asarray(values, dtype=float)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else np.nan
    std = float(finite.std(ddof=0)) if len(finite) else np.nan
    return {
        "n_days": int(len(finite)),
        "mean_rank_ic": mean,
        "rank_ic_std": std,
        "rank_icir": mean / std if np.isfinite(std) and std > 1e-12 else np.nan,
        "positive_ratio": float(np.mean(finite > 0)) if len(finite) else np.nan,
    }


def evaluate_time_series_factor_coarse(
    factor: np.ndarray,
    forward_return: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
) -> dict[str, float | int | bool | str]:
    date_values = np.asarray(dates).astype(str)
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    research_mask = date_values <= split.validation_end
    discovery_raw = asset_time_ic(factor, forward_return, discovery_mask)
    finite_discovery = discovery_raw[np.isfinite(discovery_raw)]
    direction = 0
    if len(finite_discovery) and abs(float(finite_discovery.mean())) > 1e-12:
        direction = 1 if finite_discovery.mean() > 0 else -1
    validation_raw = asset_time_ic(factor, forward_return, validation_mask)
    discovery = _summary(discovery_raw * direction if direction else discovery_raw)
    validation = _summary(validation_raw * direction if direction else validation_raw)
    research_factor = factor[research_mask]
    coverage = float(np.mean(np.isfinite(research_factor)))
    status = "OK"
    if direction == 0:
        status = "NO_DISCOVERY_DIRECTION"
    elif coverage < config.min_coverage:
        status = "LOW_COVERAGE"
    output: dict[str, float | int | bool | str] = {
        "status": status,
        "direction": direction,
        "coverage": coverage,
        "usable_days": int(np.sum(np.isfinite(research_factor).sum(axis=1) >= 3)),
        "turnover_proxy": np.nan,
        "sign_consistent": bool(
            np.isfinite(discovery["mean_rank_ic"])
            and np.isfinite(validation["mean_rank_ic"])
            and discovery["mean_rank_ic"] > 0
            and validation["mean_rank_ic"] > 0
        ),
        "ic_mode": "per_asset_time_pearson",
    }
    for prefix, values in (("discovery", discovery), ("validation", validation)):
        output.update({f"{prefix}_{key}": value for key, value in values.items()})
        output[f"{prefix}_nw_tstat"] = np.nan
        output[f"{prefix}_ci_low"] = np.nan
        output[f"{prefix}_ci_high"] = np.nan
    return output


def evaluate_time_series_factor_medium(
    factor: np.ndarray,
    forward_return: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
) -> dict[str, float | int | bool | str]:
    """Cheap second-stage stability screen across assets, eras and NY sessions."""
    output = evaluate_time_series_factor_coarse(
        factor, forward_return, dates, split, config
    )
    direction = int(output["direction"])
    date_values = np.asarray(dates).astype(str)
    validation = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    parsed = pd.to_datetime(pd.Series(dates), format="%Y%m%d%H%M", utc=True)
    ny_hour = parsed.dt.tz_convert("America/New_York").dt.hour.to_numpy()
    session_masks = {
        "us_overnight": validation & (ny_hour < 8),
        "us_day": validation & (ny_hour >= 8) & (ny_hour < 16),
        "us_evening": validation & (ny_hour >= 16),
    }
    session_ics = []
    for name, mask in session_masks.items():
        values = asset_time_ic(factor, forward_return, mask, min_observations=72)
        finite = values[np.isfinite(values)] * direction
        mean = float(finite.mean()) if len(finite) else np.nan
        output[f"medium_{name}_ic"] = mean
        session_ics.append(mean)
    validation_assets = asset_time_ic(factor, forward_return, validation) * direction
    finite_assets = validation_assets[np.isfinite(validation_assets)]
    midpoint = np.flatnonzero(validation)
    first_half = np.zeros(len(validation), dtype=bool)
    second_half = np.zeros(len(validation), dtype=bool)
    if len(midpoint):
        split_at = len(midpoint) // 2
        first_half[midpoint[:split_at]] = True
        second_half[midpoint[split_at:]] = True
    half_ics = []
    for mask in (first_half, second_half):
        values = asset_time_ic(factor, forward_return, mask, min_observations=168)
        finite = values[np.isfinite(values)] * direction
        half_ics.append(float(finite.mean()) if len(finite) else np.nan)
    finite_sessions = np.asarray(session_ics, dtype=float)
    finite_sessions = finite_sessions[np.isfinite(finite_sessions)]
    finite_halves = np.asarray(half_ics, dtype=float)
    finite_halves = finite_halves[np.isfinite(finite_halves)]
    output.update({
        "medium_asset_positive_share": (
            float(np.mean(finite_assets > 0)) if len(finite_assets) else 0.0
        ),
        "medium_session_positive_share": (
            float(np.mean(finite_sessions > 0)) if len(finite_sessions) else 0.0
        ),
        "medium_session_worst_ic": (
            float(finite_sessions.min()) if len(finite_sessions) else np.nan
        ),
        "medium_half_positive_share": (
            float(np.mean(finite_halves > 0)) if len(finite_halves) else 0.0
        ),
        "medium_half_worst_ic": (
            float(finite_halves.min()) if len(finite_halves) else np.nan
        ),
    })
    return output


def rank_time_series_medium_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank without holdout, rewarding all-session and temporal breadth."""
    ranked = frame.copy()
    validation = pd.to_numeric(
        ranked.get("validation_mean_rank_ic"), errors="coerce"
    )
    discovery = pd.to_numeric(
        ranked.get("discovery_mean_rank_ic"), errors="coerce"
    )
    ingredients = {
        "edge": validation,
        "discovery": discovery,
        "asset_breadth": pd.to_numeric(
            ranked.get("medium_asset_positive_share"), errors="coerce"
        ),
        "session_breadth": pd.to_numeric(
            ranked.get("medium_session_positive_share"), errors="coerce"
        ),
        "half_breadth": pd.to_numeric(
            ranked.get("medium_half_positive_share"), errors="coerce"
        ),
        "session_worst": pd.to_numeric(
            ranked.get("medium_session_worst_ic"), errors="coerce"
        ),
    }
    weights = {
        "edge": 0.35, "discovery": 0.15, "asset_breadth": 0.20,
        "session_breadth": 0.12, "half_breadth": 0.10, "session_worst": 0.08,
    }
    score = pd.Series(0.0, index=ranked.index)
    for name, values in ingredients.items():
        score += weights[name] * values.rank(pct=True, method="average")
    ranked["research_score"] = score
    ranked["eligible"] = (
        ranked.get("status", "") == "OK"
    ) & (validation > 0) & (discovery > 0) & (
        ingredients["asset_breadth"] >= 0.5
    ) & (ingredients["session_breadth"] >= (1.0 / 3.0)) & (
        ingredients["half_breadth"] >= 0.5
    )
    ranked.loc[~ranked["eligible"], "research_score"] = np.nan
    return ranked.sort_values("research_score", ascending=False, na_position="last")


def _empty_metrics() -> PeriodMetrics:
    return PeriodMetrics(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def _metrics(values: np.ndarray, config: EvaluationConfig, seed: int) -> PeriodMetrics:
    if not len(values):
        return _empty_metrics()
    return _period_metrics(
        values,
        np.ones(len(values), dtype=bool),
        config,
        seed,
    )


def evaluate_time_series_factor(
    factor: np.ndarray,
    forward_return: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    seed: int = 42,
    include_holdout: bool = True,
) -> EvaluationResult:
    date_values = np.asarray(dates).astype(str)
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    holdout_mask = date_values >= split.holdout_start
    research_mask = date_values <= split.validation_end
    discovery_assets = asset_time_ic(factor, forward_return, discovery_mask)
    finite = discovery_assets[np.isfinite(discovery_assets)]
    direction = 0
    if len(finite) and abs(float(finite.mean())) > 1e-12:
        direction = 1 if finite.mean() > 0 else -1
    discovery_blocks = block_time_ic(
        factor, forward_return, discovery_mask, config.time_series_block
    )
    validation_blocks = block_time_ic(
        factor, forward_return, validation_mask, config.time_series_block
    )
    holdout_blocks = (
        block_time_ic(factor, forward_return, holdout_mask, config.time_series_block)
        if include_holdout else np.asarray([], dtype=float)
    )
    if direction:
        discovery_blocks *= direction
        validation_blocks *= direction
        holdout_blocks *= direction
    discovery = _metrics(discovery_blocks, config, seed)
    validation = _metrics(validation_blocks, config, seed + 1)
    holdout = _metrics(holdout_blocks, config, seed + 2) if include_holdout else _empty_metrics()
    research_factor = factor[research_mask]
    coverage = float(np.mean(np.isfinite(research_factor)))
    normalized = fastops.fast_rolling_zscore(
        np.asarray(research_factor, dtype=float),
        config.time_series_position_window,
    )
    turnover = float(np.nanmean(np.abs(normalized[1:] - normalized[:-1])))
    status = "OK"
    if direction == 0:
        status = "NO_DISCOVERY_DIRECTION"
    elif coverage < config.min_coverage:
        status = "LOW_COVERAGE"
    return EvaluationResult(
        status=status,
        direction=direction,
        coverage=coverage,
        usable_days=int(np.sum(np.isfinite(research_factor).sum(axis=1) >= 3)),
        turnover_proxy=turnover,
        discovery=discovery,
        validation=validation,
        holdout=holdout,
        sign_consistent=bool(
            np.isfinite(discovery.mean_rank_ic)
            and np.isfinite(validation.mean_rank_ic)
            and discovery.mean_rank_ic > 0
            and validation.mean_rank_ic > 0
        ),
    )
