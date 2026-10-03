"""Discovery-fitted baseline comparison for bounded variance survivors."""

from __future__ import annotations

import numpy as np

from .config import SplitConfig


BASELINE_FORMULA_VERSION = "2026-09-28-same-side-risk-ny-clock-v1"


def variance_baseline_panels(
    realized_vol: np.ndarray,
    upside_share: np.ndarray,
    ny_day: np.ndarray,
    ny_evening: np.ndarray,
    mode: str,
    scope: str,
) -> np.ndarray:
    """Current same-side 5m variance plus two DST-aware NY clock indicators."""
    rv = np.asarray(realized_vol, dtype=float)
    share = np.asarray(upside_share, dtype=float)
    if any(np.asarray(x).shape != rv.shape for x in (share, ny_day, ny_evening)):
        raise ValueError("baseline inputs must have the same panel shape")
    current = np.square(rv)
    if mode == "upside_semivariance":
        current *= share
    elif mode == "downside_semivariance":
        current *= 1.0 - share
    elif mode != "total_variance":
        raise ValueError("baseline requires a variance target")
    current[~np.isfinite(current)] = np.nan
    if scope == "market_relative":
        valid = np.isfinite(current)
        count = valid.sum(axis=1, keepdims=True)
        common = np.divide(
            np.nansum(current, axis=1, keepdims=True), count,
            out=np.full((len(current), 1), np.nan), where=count >= 3,
        )
        current = current - common
    elif scope != "absolute":
        raise ValueError("invalid baseline target scope")
    return np.stack([current, ny_day, ny_evening], axis=2)


def _ridge_predict(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    left = np.column_stack([np.ones(len(train_x)), (train_x - mean) / std])
    right = np.column_stack([np.ones(len(test_x)), (test_x - mean) / std])
    penalty = np.eye(left.shape[1]) * 0.01
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(left.T @ left + penalty, left.T @ train_y)
    return right @ coefficients


def incremental_variance_information(
    factor: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    dates: list[str],
    split: SplitConfig,
    min_discovery: int = 168,
    min_validation: int = 72,
) -> dict[str, float | int]:
    """Compare baseline and baseline+factor on identical validation rows.

    All transformation scales and model coefficients use discovery only.
    The target is compressed with asinh using a discovery-only scale so a
    single jump does not dominate the incremental comparison.
    """
    factor = np.asarray(factor, dtype=float)
    target = np.asarray(target, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if factor.shape != target.shape or baseline.shape[:2] != factor.shape:
        raise ValueError("factor, target, and baseline panel shapes disagree")
    labels = np.asarray(dates).astype(str)
    discovery = labels <= split.discovery_end
    validation = (labels >= split.validation_start) & (labels <= split.validation_end)
    deltas, base_scores, counts = [], [], []
    for asset in range(factor.shape[1]):
        x = baseline[:, asset, :]
        candidate = factor[:, asset]
        y = target[:, asset]
        valid = np.isfinite(y) & np.isfinite(candidate) & np.isfinite(x).all(axis=1)
        train = valid & discovery
        test = valid & validation
        if train.sum() < min_discovery or test.sum() < min_validation:
            continue
        scale = float(np.median(np.abs(y[train])))
        scale = max(scale, 1e-10)
        train_y = np.arcsinh(y[train] / scale)
        test_y = np.arcsinh(y[test] / scale)
        train_base = x[train].copy()
        test_base = x[test].copy()
        train_base[:, 0] = np.arcsinh(train_base[:, 0] / scale)
        test_base[:, 0] = np.arcsinh(test_base[:, 0] / scale)
        base_pred = _ridge_predict(train_base, train_y, test_base)
        full_pred = _ridge_predict(
            np.column_stack([train_base, candidate[train]]), train_y,
            np.column_stack([test_base, candidate[test]]),
        )
        base_error = float(np.square(test_y - base_pred).sum())
        full_error = float(np.square(test_y - full_pred).sum())
        null_error = float(np.square(test_y - train_y.mean()).sum())
        if base_error <= 1e-12 or null_error <= 1e-12:
            continue
        deltas.append((base_error - full_error) / base_error)
        base_scores.append(1.0 - base_error / null_error)
        counts.append(int(test.sum()))
    if not deltas:
        return {
            "incremental_mean_delta_r2": np.nan,
            "incremental_median_delta_r2": np.nan,
            "incremental_positive_asset_share": np.nan,
            "incremental_baseline_mean_r2": np.nan,
            "incremental_asset_count": 0,
            "incremental_validation_rows": 0,
        }
    values = np.asarray(deltas)
    return {
        "incremental_mean_delta_r2": float(values.mean()),
        "incremental_median_delta_r2": float(np.median(values)),
        "incremental_positive_asset_share": float(np.mean(values > 0)),
        "incremental_baseline_mean_r2": float(np.mean(base_scores)),
        "incremental_asset_count": len(values),
        "incremental_validation_rows": sum(counts),
    }
