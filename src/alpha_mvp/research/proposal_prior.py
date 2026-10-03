"""Support-smoothed proposal priorities from prior non-holdout evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "status", "fields", "operators", "template_name", "windows",
    "discovery_mean_rank_ic", "validation_mean_rank_ic",
}


def load_proposal_prior(path: str, min_support: int = 20) -> dict[str, float]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Proposal evidence does not exist: {path}")
    columns = set(pd.read_csv(file_path, nrows=0).columns)
    if any(name.startswith("holdout_") for name in columns):
        raise ValueError("Proposal evidence must not contain holdout_* columns")
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f"Proposal evidence missing columns: {sorted(missing)}")
    frame = pd.read_csv(file_path, usecols=sorted(REQUIRED))
    discovery = pd.to_numeric(frame["discovery_mean_rank_ic"], errors="coerce")
    validation = pd.to_numeric(frame["validation_mean_rank_ic"], errors="coerce")
    success = (
        frame["status"].eq("OK") & discovery.gt(0) & validation.gt(0)
    ).to_numpy(dtype=float)
    base_rate = float(success.mean()) if len(success) else 0.5
    observations: dict[str, list[float]] = {}
    for index, row in enumerate(frame.itertuples(index=False)):
        tokens = {f"template:{row.template_name}"}
        for column, prefix in (
            ("fields", "field"), ("operators", "operator"),
            ("windows", "window"),
        ):
            raw = getattr(row, column)
            if isinstance(raw, str):
                tokens.update(f"{prefix}:{token}" for token in raw.split("|") if token)
        for token in tokens:
            entry = observations.setdefault(token, [0.0, 0.0])
            entry[0] += 1.0
            entry[1] += success[index]
    result = {}
    for token, (count, wins) in observations.items():
        if count < min_support:
            continue
        posterior = (wins + 8.0 * base_rate) / (count + 8.0)
        result[token] = float(np.clip(posterior - base_rate, -0.25, 0.25))
    return result


def proposal_bonus(
    prior: dict[str, float], template: str,
    fields: tuple[str, ...], operators: tuple[str, ...],
    windows: tuple[int, ...], strength: float,
) -> float:
    tokens = {f"template:{template}"}
    tokens.update(f"field:{value}" for value in fields)
    tokens.update(f"operator:{value}" for value in operators)
    tokens.update(f"window:{value}" for value in windows)
    values = [prior[token] for token in tokens if token in prior]
    return strength * float(np.mean(values)) if values else 0.0
