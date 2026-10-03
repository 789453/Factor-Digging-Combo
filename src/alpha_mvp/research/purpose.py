"""Purpose-specific targets and scores for large crypto information searches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EvaluationConfig, SplitConfig
from .evaluation import purge_cross_split_labels
from .time_series_evaluation import (
    evaluate_time_series_factor, evaluate_time_series_factor_coarse,
    evaluate_time_series_factor_medium,
)
from .incremental_information import _ridge_predict


PURPOSE_FORMULA_VERSION = "2026-09-28-residual-common-hybrid-style-v2"
PURPOSE_BASELINE_FORMULA_VERSION = "2026-09-28-purpose-discovery-ridge-v1"


@dataclass(frozen=True)
class PurposeTargets:
    residual: np.ndarray
    common: np.ndarray
    betas: np.ndarray
    style: np.ndarray | None = None


def _forward_sum(values: np.ndarray, horizon: int, lag: int) -> np.ndarray:
    """Future complete-bar sum, requiring every constituent observation."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 2 or horizon < 1 or lag < 0:
        raise ValueError("future sum requires T x N and valid horizon/lag")
    valid = np.isfinite(series)
    clean = np.where(valid, series, 0.0)
    sums = np.vstack([np.zeros((1, series.shape[1])), np.cumsum(clean, axis=0)])
    counts = np.vstack([np.zeros((1, series.shape[1]), dtype=int),
                        np.cumsum(valid.astype(int), axis=0)])
    result = np.full_like(series, np.nan)
    n = len(series) - lag - horizon
    if n > 0:
        start = np.arange(n) + lag + 1
        end = start + horizon
        complete = counts[end] - counts[start] == horizon
        result[:n] = np.where(complete, sums[end] - sums[start], np.nan)
    return result


def build_purpose_targets(
    close: np.ndarray, dates: list[str], split: SplitConfig,
    horizon: int, lag: int,
    style_hhi: np.ndarray | None = None,
) -> PurposeTargets:
    """Fit leave-one-out market betas on discovery; freeze for later periods."""
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 2 or prices.shape[1] < 3 or len(dates) != len(prices):
        raise ValueError("purpose targets require aligned T x N close panel, N >= 3")
    log_price = np.log(np.where(prices > 0, prices, np.nan))
    bar = np.full_like(log_price, np.nan)
    bar[1:] = log_price[1:] - log_price[:-1]
    finite = np.isfinite(bar)
    count = finite.sum(axis=1, keepdims=True)
    total = np.where(finite, bar, 0.0).sum(axis=1, keepdims=True)
    others = count - finite.astype(int)
    market_loo = np.divide(
        total - np.where(finite, bar, 0.0), others,
        out=np.full_like(bar, np.nan), where=others >= 3,
    )
    discovery = np.asarray(dates).astype(str) <= split.discovery_end
    betas = np.full(prices.shape[1], np.nan)
    for asset in range(prices.shape[1]):
        x, y = market_loo[:, asset], bar[:, asset]
        keep = discovery & np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 168:
            continue
        centered_x = x[keep] - x[keep].mean()
        centered_y = y[keep] - y[keep].mean()
        denominator = np.square(centered_x).sum()
        if denominator > 1e-12:
            betas[asset] = np.dot(centered_x, centered_y) / denominator
    future_asset = _forward_sum(bar, horizon, lag)
    future_others = _forward_sum(market_loo, horizon, lag)
    residual = future_asset - future_others * betas[None, :]
    residual = purge_cross_split_labels(residual, dates, split, horizon, lag)
    market = np.divide(total, count, out=np.full((len(bar), 1), np.nan), where=count >= 3)
    downside = np.square(np.minimum(market, 0.0))
    downside[~np.isfinite(market)] = np.nan
    common = _forward_sum(downside, horizon, lag)
    common = purge_cross_split_labels(common, dates, split, horizon, lag)
    style = None
    if style_hhi is not None:
        hhi = np.asarray(style_hhi, dtype=float)
        if hhi.shape != prices.shape:
            raise ValueError("style_hhi must align with close panel")
        style = _forward_sum(hhi, horizon, lag) / horizon
        style = purge_cross_split_labels(style, dates, split, horizon, lag)
    return PurposeTargets(residual=residual, common=common, betas=betas, style=style)


def common_factor(values: np.ndarray) -> np.ndarray:
    """Median across assets, one market observation per timestamp."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("common factor requires T x N")
    count = np.isfinite(array).sum(axis=1)
    median = np.full(len(array), np.nan)
    usable = count >= 3
    if usable.any():
        median[usable] = np.nanmedian(array[usable], axis=1)
    return median[:, None]


def _combine(alpha: dict, common: dict, stage: str) -> dict:
    """Conservative two-axis score: the weaker signed IC is decisive."""
    output = dict(alpha)
    for key in (
        "discovery_mean_rank_ic", "validation_mean_rank_ic",
        "discovery_rank_icir", "validation_rank_icir",
        "discovery_positive_ratio", "validation_positive_ratio",
    ):
        output[f"alpha_{key}"] = alpha.get(key, np.nan)
        output[f"common_{key}"] = common.get(key, np.nan)
        a, b = alpha.get(key, np.nan), common.get(key, np.nan)
        output[key] = min(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
    output["common_direction"] = common.get("direction", 0)
    output["status"] = (
        "OK" if alpha.get("status") == "OK" and common.get("status") == "OK"
        else "INCOMPLETE_HYBRID_AXES"
    )
    output["sign_consistent"] = bool(
        alpha.get("sign_consistent") and common.get("sign_consistent")
    )
    output["coverage"] = min(alpha.get("coverage", 0), common.get("coverage", 0))
    if stage == "medium":
        for key in (
            "medium_asset_positive_share", "medium_session_positive_share",
            "medium_half_positive_share", "medium_session_worst_ic",
            "medium_half_worst_ic",
        ):
            a, b = alpha.get(key, np.nan), common.get(key, np.nan)
            output[key] = min(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
    return output


def evaluate_purpose(
    values: np.ndarray, targets: PurposeTargets, dates: list[str],
    split: SplitConfig, config: EvaluationConfig, stage: str,
    include_holdout: bool = False,
) -> dict:
    if stage not in {"coarse", "medium", "fine"}:
        raise ValueError("purpose evaluation stage must be coarse, medium, or fine")
    evaluator = {
        "coarse": evaluate_time_series_factor_coarse,
        "medium": evaluate_time_series_factor_medium,
        "fine": evaluate_time_series_factor,
    }[stage]

    def assess(x: np.ndarray, y: np.ndarray) -> dict:
        if stage == "fine":
            return evaluator(
                x, y, dates, split, config, include_holdout=include_holdout
            ).to_flat_dict()
        return evaluator(x, y, dates, split, config)

    track = config.research_track
    if track == "residual_alpha":
        return assess(values, targets.residual)
    if track == "style_proxy":
        if targets.style is None:
            raise ValueError("style_proxy requires future HHI target")
        return assess(values, targets.style)
    collapsed = common_factor(values)
    if track == "common_risk":
        return assess(collapsed, targets.common)
    if track == "hybrid":
        return _combine(
            assess(values, targets.residual),
            assess(collapsed, targets.common), stage,
        )
    raise ValueError(f"Unknown purpose track: {track}")


def style_baseline_panels(
    panels: dict[str, np.ndarray], native_5m: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Known-at-decision style controls and one market-wide risk baseline."""
    names = (
        ("ret_5m", "ret_1h", "rv_12", "volume_shock_12", "amihud_shock_12")
        if native_5m else
        ("ret_1h", "ret_24h", "realized_vol_24h",
         "volume_shock_24h", "amihud_shock_24h")
    )
    names += ("us_day_flag", "us_evening_flag")
    missing = sorted(set(names) - set(panels))
    if missing:
        raise ValueError(f"Purpose style baseline missing fields: {missing}")
    shape = np.asarray(panels[names[0]]).shape
    if any(np.asarray(panels[name]).shape != shape for name in names):
        raise ValueError("Purpose baseline panels must be aligned")
    asset = np.stack([panels[name] for name in names], axis=2).astype(float)
    returns = np.asarray(panels[names[0]], dtype=float)
    valid = np.isfinite(returns)
    count = valid.sum(axis=1)
    market_ret = np.divide(
        np.where(valid, returns, 0.0).sum(axis=1), count,
        out=np.full(len(returns), np.nan), where=count >= 3,
    )
    market_risk = np.square(np.minimum(market_ret, 0.0))
    market = np.column_stack([
        market_risk, np.asarray(panels["us_day_flag"])[:, 0],
        np.asarray(panels["us_evening_flag"])[:, 0],
    ])[:, None, :]
    return asset, market


def liquidity_style_baseline_panels(panels: dict[str, np.ndarray]) -> np.ndarray:
    """Current known hourly concentration, activity, and NY clock controls."""
    names = ("micro5_volume_hhi", "micro5_trade_hhi", "volume_shock_24h",
             "trade_count_shock_24h", "us_day_flag", "us_evening_flag")
    missing = sorted(set(names + ("micro5_bar_count",)) - set(panels))
    if missing:
        raise ValueError(f"Liquidity style baseline missing fields: {missing}")
    shape = np.asarray(panels[names[0]]).shape
    if any(np.asarray(panels[name]).shape != shape for name in names):
        raise ValueError("Liquidity baseline panels must be aligned")
    baseline = np.stack([panels[name] for name in names], axis=2).astype(float)
    baseline[np.asarray(panels["micro5_bar_count"]) != 12] = np.nan
    return baseline


def incremental_purpose_information(
    factor: np.ndarray, target: np.ndarray, baseline: np.ndarray,
    dates: list[str], split: SplitConfig,
    min_discovery: int = 168, min_validation: int = 72,
) -> dict[str, float | int]:
    """Discovery-fitted baseline vs baseline+candidate on identical V rows."""
    x = np.asarray(factor, dtype=float)
    y = np.asarray(target, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if x.shape != y.shape or base.shape[:2] != x.shape:
        raise ValueError("incremental purpose panels are misaligned")
    labels = np.asarray(dates).astype(str)
    d = labels <= split.discovery_end
    v = (labels >= split.validation_start) & (labels <= split.validation_end)
    scores, positive, baseline_scores = [], [], []
    for asset in range(x.shape[1]):
        row_valid = np.isfinite(x[:, asset]) & np.isfinite(y[:, asset]) & np.isfinite(base[:, asset]).all(axis=1)
        train, test = row_valid & d, row_valid & v
        if train.sum() < min_discovery or test.sum() < min_validation:
            continue
        scale = max(float(np.median(np.abs(y[train, asset]))), 1e-10)
        train_y = np.arcsinh(y[train, asset] / scale)
        test_y = np.arcsinh(y[test, asset] / scale)
        b_train, b_test = base[train, asset], base[test, asset]
        base_pred = _ridge_predict(b_train, train_y, b_test)
        full_pred = _ridge_predict(
            np.column_stack([b_train, x[train, asset]]), train_y,
            np.column_stack([b_test, x[test, asset]]),
        )
        base_sse = float(np.square(test_y - base_pred).sum())
        full_sse = float(np.square(test_y - full_pred).sum())
        null_sse = float(np.square(test_y - train_y.mean()).sum())
        if base_sse <= 1e-12 or null_sse <= 1e-12:
            continue
        gain = (base_sse - full_sse) / base_sse
        scores.append(gain)
        positive.append(gain > 0)
        baseline_scores.append(1 - base_sse / null_sse)
    return {
        "incremental_mean_delta_r2": float(np.mean(scores)) if scores else np.nan,
        "incremental_median_delta_r2": float(np.median(scores)) if scores else np.nan,
        "incremental_positive_asset_share": float(np.mean(positive)) if scores else np.nan,
        "incremental_baseline_mean_r2": float(np.mean(baseline_scores)) if scores else np.nan,
        "incremental_asset_count": len(scores),
    }
