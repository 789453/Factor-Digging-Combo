from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .. import fastops
from .config import EvaluationConfig, SplitConfig


def forward_returns(
    close: np.ndarray,
    horizon: int = 5,
    entry_lag: int = 1,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if entry_lag < 0:
        raise ValueError("entry_lag cannot be negative")
    output = np.full_like(close, np.nan, dtype=float)
    end_offset = entry_lag + horizon
    if end_offset < len(close):
        output[:-end_offset] = (
            close[end_offset:] / close[entry_lag:-horizon] - 1.0
        )
    return output


def purge_cross_split_labels(
    values: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    horizon: int,
    entry_lag: int,
) -> np.ndarray:
    """Remove research labels whose exit price lies beyond their split."""
    output = np.asarray(values, dtype=float).copy()
    labels = np.asarray(dates).astype(str)
    if len(output) != len(labels):
        raise ValueError("forward returns and dates must have the same length")
    exit_index = np.arange(len(labels)) + entry_lag + horizon
    in_range = exit_index < len(labels)
    exit_label = np.empty(len(labels), dtype=object)
    exit_label[:] = ""
    exit_label[in_range] = labels[exit_index[in_range]]
    discovery = labels <= split.discovery_end
    validation = (labels >= split.validation_start) & (labels <= split.validation_end)
    crossing = (
        (discovery & (exit_label > split.discovery_end))
        | (validation & (exit_label > split.validation_end))
    )
    output[crossing] = np.nan
    return output


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def daily_rank_ic(
    factor: np.ndarray,
    forward_return: np.ndarray,
    min_names: int,
) -> np.ndarray:
    return fastops.daily_corr(
        np.asarray(factor, dtype=float),
        np.asarray(forward_return, dtype=float),
        rank=True,
        min_valid=min_names,
    )


def newey_west_tstat(values: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(10, lags + 2):
        return np.nan
    centered = x - x.mean()
    long_run_variance = np.dot(centered, centered) / n
    for lag in range(1, min(lags, n - 2) + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / n
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    return float(x.mean() / np.sqrt(variance_of_mean)) if variance_of_mean > 0 else np.nan


def block_bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    block_size: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(10, block_size):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    blocks_needed = int(np.ceil(len(x) / block_size))
    max_start = max(1, len(x) - block_size + 1)
    for sample in range(samples):
        starts = rng.integers(0, max_start, size=blocks_needed)
        boot = np.concatenate([x[s:s + block_size] for s in starts])[:len(x)]
        means[sample] = boot.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def turnover_proxy(factor: np.ndarray) -> float:
    ranked = fastops.rank_cs(np.asarray(factor, dtype=float))
    value = np.nanmean(np.abs(ranked[1:] - ranked[:-1]))
    return float(value) if np.isfinite(value) else np.nan


@dataclass(frozen=True)
class PeriodMetrics:
    n_days: int
    mean_rank_ic: float
    rank_ic_std: float
    rank_icir: float
    nw_tstat: float
    positive_ratio: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    direction: int
    coverage: float
    usable_days: int
    turnover_proxy: float
    discovery: PeriodMetrics
    validation: PeriodMetrics
    holdout: PeriodMetrics
    sign_consistent: bool

    def to_flat_dict(self) -> dict:
        output = {
            "status": self.status,
            "direction": self.direction,
            "coverage": self.coverage,
            "usable_days": self.usable_days,
            "turnover_proxy": self.turnover_proxy,
            "sign_consistent": self.sign_consistent,
        }
        for name in ("discovery", "validation", "holdout"):
            for key, value in asdict(getattr(self, name)).items():
                output[f"{name}_{key}"] = value
        return output


def _period_metrics(
    ic: np.ndarray,
    mask: np.ndarray,
    config: EvaluationConfig,
    seed: int,
) -> PeriodMetrics:
    values = ic[mask]
    values = values[np.isfinite(values)]
    if not len(values):
        return PeriodMetrics(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std())
    low, high = block_bootstrap_mean_ci(
        values,
        config.bootstrap_samples,
        config.bootstrap_block,
        seed,
    )
    return PeriodMetrics(
        n_days=len(values),
        mean_rank_ic=mean,
        rank_ic_std=std,
        rank_icir=mean / std if std > 1e-12 else np.nan,
        nw_tstat=newey_west_tstat(values, config.nw_lags),
        positive_ratio=float(np.mean(values > 0)),
        ci_low=low,
        ci_high=high,
    )


def evaluate_factor(
    factor: np.ndarray,
    forward_return: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
    seed: int = 42,
    include_holdout: bool = True,
) -> EvaluationResult:
    split.validate()
    date_values = np.asarray(dates).astype(str)
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    holdout_mask = date_values >= split.holdout_start
    research_mask = date_values <= split.validation_end
    if not discovery_mask.any() or not validation_mask.any():
        raise ValueError("Discovery and validation periods must both contain dates")

    raw_ic = daily_rank_ic(factor, forward_return, config.min_daily_names)
    discovery_raw = raw_ic[discovery_mask]
    discovery_raw = discovery_raw[np.isfinite(discovery_raw)]
    direction = 0
    if len(discovery_raw):
        discovery_mean = float(discovery_raw.mean())
        if abs(discovery_mean) > 1e-12:
            direction = 1 if discovery_mean > 0 else -1
    oriented_ic = raw_ic * direction if direction else raw_ic
    research_factor = factor[research_mask]
    coverage = float(np.mean(np.isfinite(research_factor)))
    usable_days = int(np.sum(
        np.isfinite(research_factor).sum(axis=1) >= config.min_daily_names
    ))

    discovery = _period_metrics(oriented_ic, discovery_mask, config, seed)
    validation = _period_metrics(oriented_ic, validation_mask, config, seed + 1)
    holdout = (
        _period_metrics(oriented_ic, holdout_mask, config, seed + 2)
        if include_holdout
        else PeriodMetrics(
            n_days=0,
            mean_rank_ic=np.nan,
            rank_ic_std=np.nan,
            rank_icir=np.nan,
            nw_tstat=np.nan,
            positive_ratio=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
        )
    )
    status = "OK"
    if direction == 0:
        status = "NO_DISCOVERY_DIRECTION"
    elif coverage < config.min_coverage:
        status = "LOW_COVERAGE"
    sign_consistent = bool(
        np.isfinite(discovery.mean_rank_ic)
        and np.isfinite(validation.mean_rank_ic)
        and discovery.mean_rank_ic > 0
        and validation.mean_rank_ic > 0
    )
    return EvaluationResult(
        status=status,
        direction=direction,
        coverage=coverage,
        usable_days=usable_days,
        turnover_proxy=turnover_proxy(research_factor),
        discovery=discovery,
        validation=validation,
        holdout=holdout,
        sign_consistent=sign_consistent,
    )


def evaluate_factor_coarse(
    factor: np.ndarray,
    forward_return: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    config: EvaluationConfig,
) -> dict[str, float | int | bool | str]:
    """Cheap first-stage IC screen without bootstrap, NW, or strategy tests."""
    date_values = np.asarray(dates).astype(str)
    discovery_mask = date_values <= split.discovery_end
    validation_mask = (
        (date_values >= split.validation_start)
        & (date_values <= split.validation_end)
    )
    research_mask = date_values <= split.validation_end
    raw_ic = daily_rank_ic(factor, forward_return, config.min_daily_names)
    discovery_raw = raw_ic[discovery_mask]
    discovery_raw = discovery_raw[np.isfinite(discovery_raw)]
    direction = 0
    if len(discovery_raw) and abs(float(discovery_raw.mean())) > 1e-12:
        direction = 1 if discovery_raw.mean() > 0 else -1
    oriented = raw_ic * direction if direction else raw_ic

    def summarize(mask):
        values = oriented[mask]
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=0)) if len(values) else np.nan
        return {
            "n_days": int(len(values)),
            "mean_rank_ic": mean,
            "rank_ic_std": std,
            "rank_icir": mean / std if np.isfinite(std) and std > 1e-12 else np.nan,
            "positive_ratio": float(np.mean(values > 0)) if len(values) else np.nan,
        }

    discovery = summarize(discovery_mask)
    validation = summarize(validation_mask)
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
        "usable_days": int(np.sum(
            np.isfinite(research_factor).sum(axis=1) >= config.min_daily_names
        )),
        "turnover_proxy": np.nan,
        "sign_consistent": bool(
            np.isfinite(discovery["mean_rank_ic"])
            and np.isfinite(validation["mean_rank_ic"])
            and discovery["mean_rank_ic"] > 0
            and validation["mean_rank_ic"] > 0
        ),
    }
    for prefix, values in (("discovery", discovery), ("validation", validation)):
        output.update({f"{prefix}_{key}": value for key, value in values.items()})
        output[f"{prefix}_nw_tstat"] = np.nan
        output[f"{prefix}_ci_low"] = np.nan
        output[f"{prefix}_ci_high"] = np.nan
    return output


def rank_crypto_coarse_results(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["eligible"] = (
        frame["status"].eq("OK") & frame["sign_consistent"].astype(bool)
    )
    components = {
        "validation_mean_rank_ic": (0.35, True),
        "validation_rank_icir": (0.20, True),
        "validation_positive_ratio": (0.10, True),
        "discovery_mean_rank_ic": (0.15, True),
        "coverage": (0.10, True),
        "nodes": (0.10, False),
    }
    frame["research_score"] = 0.0
    for column, (weight, higher_is_better) in components.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if not higher_is_better:
            values = -values
        frame["research_score"] += values.rank(pct=True).fillna(0.0) * weight
    frame.loc[~frame["eligible"], "research_score"] = np.nan
    return frame.sort_values(
        ["eligible", "research_score"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def rank_research_results(
    results: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Rank candidates without using any holdout metric."""
    if results.empty:
        return results.copy()
    frame = results.copy()
    eligible = frame["status"].eq("OK") & frame["sign_consistent"].astype(bool)
    if "horizon_positive_share" in frame:
        eligible &= frame["horizon_positive_share"].fillna(0.0) >= (2.0 / 3.0)
    if "validation_net_sharpe" in frame:
        eligible &= frame["validation_net_sharpe"].fillna(-np.inf) > 0.0
    if "fine_evaluated" in frame:
        eligible &= frame["fine_evaluated"].fillna(False).astype(bool)
    frame["eligible"] = eligible
    component_sources = {
        "validation_edge_rank": ("validation_mean_rank_ic", True),
        "validation_ir_rank": ("validation_rank_icir", True),
        "validation_hit_rank": ("validation_positive_ratio", True),
        "discovery_edge_rank": ("discovery_mean_rank_ic", True),
        "sign_consistency_rank": ("sign_consistent", True),
        "coverage_rank": ("coverage", True),
        "turnover_rank": ("turnover_proxy", False),
        "complexity_rank": ("nodes", False),
        "validation_sharpe_rank": ("validation_net_sharpe", True),
        "discovery_sharpe_rank": ("discovery_net_sharpe", True),
        "horizon_robustness_rank": ("horizon_worst_validation_ic", True),
        "incremental_r2_rank": ("incremental_mean_delta_r2", True),
    }
    requested_components = set(config.score_weights)
    unknown = requested_components - set(component_sources)
    if unknown:
        raise ValueError(f"Unknown research score components: {sorted(unknown)}")
    for target, (source, ascending) in component_sources.items():
        if target not in requested_components:
            continue
        values = frame[source].astype(float)
        if not ascending:
            values = -values
        frame[target] = values.rank(pct=True, method="average")
    frame["research_score"] = 0.0
    for component, weight in config.score_weights.items():
        frame["research_score"] += frame[component].fillna(0.0) * weight
    frame.loc[~eligible, "research_score"] = np.nan
    return frame.sort_values(
        ["eligible", "research_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
