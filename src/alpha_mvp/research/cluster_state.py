"""One discovery-fitted, causally observed crypto pressure-state field."""

from __future__ import annotations

import numpy as np

from .. import fastops
from ..fields import FieldSpec
from .config import ClusterStateConfig, SplitConfig


CLUSTER_FIELD_FORMULA_VERSION = "2026-09-28-cluster-stress-affinity-v1"
CLUSTER_FIELD_SPEC = FieldSpec(
    "cluster_stress_affinity", "crypto_structured_state",
    ("micro5_taker_imbalance", "micro5_jump_share",
     "session_illiquidity_surprise", "trade_date", "ts_code"),
    "Affinity to a discovery-fitted negative-flow/high-jump/high-illiquidity state",
)


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.square(values).sum(axis=1, keepdims=True)
        + np.square(centers).sum(axis=1)[None, :]
        - 2.0 * values @ centers.T,
        0.0,
    )


def _fit_centers(values: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    pressure = -values[:, 0] + values[:, 1] + values[:, 2]
    ordered = np.argsort(pressure, kind="mergesort")
    quantiles = np.linspace(0.15, 0.85, n_clusters)
    seeds = ordered[(quantiles * (len(values) - 1)).astype(int)]
    centers = values[seeds].copy()
    for _ in range(60):
        labels = _squared_distances(values, centers).argmin(axis=1)
        counts = np.bincount(labels, minlength=n_clusters)
        if (counts == 0).any():
            raise ValueError("cluster_state produced an empty discovery cluster")
        updated = np.vstack([values[labels == group].mean(axis=0)
                             for group in range(n_clusters)])
        if np.max(np.abs(updated - centers)) < 1e-6:
            centers = updated
            break
        centers = updated
    labels = _squared_distances(values, centers).argmin(axis=1)
    return centers, labels


def build_cluster_stress_affinity(
    panels: dict[str, np.ndarray],
    dates: list[str],
    split: SplitConfig,
    config: ClusterStateConfig,
) -> tuple[np.ndarray, dict]:
    """Fit pooled states on discovery, then apply frozen centers causally."""
    config.validate()
    if not config.enabled:
        raise ValueError("cluster_state must be enabled")
    names = (
        config.flow_field, config.jump_field, config.illiquidity_field,
    )
    missing = sorted(set(names) - set(panels))
    if missing:
        raise ValueError(f"cluster_state missing dependencies: {missing}")
    shapes = {np.asarray(panels[name]).shape for name in names}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("cluster_state dependencies must have the same T x N shape")
    if len(dates) != next(iter(shapes))[0]:
        raise ValueError("cluster_state dates and panels do not align")
    normalized = np.stack([
        fastops.fast_rolling_zscore(
            np.asarray(panels[name], dtype=float), config.history_window
        ) for name in names
    ], axis=2)
    normalized = np.clip(normalized, -8.0, 8.0)
    discovery = np.asarray(dates).astype(str) <= split.discovery_end
    training = normalized[discovery][::config.fit_stride].reshape(-1, 3)
    training = training[np.isfinite(training).all(axis=1)]
    if len(training) > config.max_fit_samples:
        sample = np.linspace(
            0, len(training) - 1, config.max_fit_samples, dtype=int
        )
        training = training[sample]
    if len(training) < max(200, 40 * config.n_clusters):
        raise ValueError("cluster_state has too few complete discovery observations")
    centers, labels = _fit_centers(training, config.n_clusters)
    counts = np.bincount(labels, minlength=config.n_clusters)
    shares = counts / len(training)
    if shares.min() < config.min_cluster_share:
        raise ValueError(
            f"cluster_state smallest discovery cluster {shares.min():.4f} "
            f"is below min_cluster_share={config.min_cluster_share}"
        )
    stress = int(np.argmax(-centers[:, 0] + centers[:, 1] + centers[:, 2]))
    within = np.sqrt(_squared_distances(training[labels == stress], centers[[stress]])[:, 0])
    temperature = max(float(np.median(within)), 0.25)
    flat = normalized.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    affinity = np.full(len(flat), np.nan)
    if valid.any():
        distance_sq = _squared_distances(flat[valid], centers[[stress]])[:, 0]
        affinity[valid] = np.exp(-distance_sq / (2.0 * temperature**2))
    metadata = {
        "formula_version": CLUSTER_FIELD_FORMULA_VERSION,
        "field_spec": {
            "name": CLUSTER_FIELD_SPEC.name,
            "dependencies": list(CLUSTER_FIELD_SPEC.dependencies),
        },
        "fit_end": split.discovery_end,
        "fit_samples": int(len(training)),
        "n_clusters": config.n_clusters,
        "cluster_shares": shares.tolist(),
        "centers": centers.tolist(),
        "stress_cluster": stress,
        "temperature": temperature,
        "history_window": config.history_window,
        "fit_stride": config.fit_stride,
        "max_fit_samples": config.max_fit_samples,
    }
    return affinity.reshape(next(iter(shapes))), metadata
