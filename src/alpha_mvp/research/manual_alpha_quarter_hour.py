"""Completed-hour summaries of quarter-hour opening 5m order-flow phases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

QUARTER_FIELD_VERSION = "manual-alpha-quarter-hour-20261003-v1"
QUARTER_FIELD_DEPENDENCIES = {
    "quarter_open_imbalance": ("date", "quote_volume", "taker_buy_ratio"),
    "quarter_other_imbalance": ("date", "quote_volume", "taker_buy_ratio"),
    "quarter_open_volume_share": ("date", "quote_volume"),
}


def align_quarter_hour_frame(frame: pd.DataFrame, raw_dates: np.ndarray) -> dict[str, np.ndarray]:
    """Require all four opening and eight other completed 5m bars per hour.

    Bar `date` is its open time. The summary is available only one hour later.
    Missing, duplicated, nonpositive-volume, or invalid-taker bars invalidate
    the entire hour rather than silently filling a phase with zero.
    """
    required = {"date", "quote_volume", "taker_buy_ratio"}
    if not required.issubset(frame):
        raise ValueError(f"quarter-hour source missing {sorted(required - set(frame))}")
    date = pd.to_datetime(frame["date"], utc=True)
    if date.duplicated().any():
        raise ValueError("duplicate native 5m phase timestamps")
    minute = date.dt.minute
    aligned = date.eq(date.dt.floor("5min"))
    if not aligned.all():
        raise ValueError("native 5m phase clock has off-grid observations")
    volume = pd.to_numeric(frame["quote_volume"], errors="coerce")
    ratio = pd.to_numeric(frame["taker_buy_ratio"], errors="coerce")
    valid = volume.gt(0) & np.isfinite(volume) & ratio.between(0, 1)
    data = pd.DataFrame({"hour": date.dt.floor("h"), "open_phase": minute.mod(15).eq(0),
                         "volume": volume.where(valid),
                         "signed_volume": (volume * (2 * ratio - 1)).where(valid),
                         "valid": valid.astype(int)})
    grouped = data.groupby(["hour", "open_phase"], sort=True).agg(
        volume=("volume", "sum"), signed_volume=("signed_volume", "sum"),
        valid=("valid", "sum"))
    hour = pd.to_datetime(raw_dates, format="%Y%m%d%H%M", utc=True)
    present = set(grouped.index.get_level_values("open_phase"))
    empty = pd.DataFrame(columns=["volume", "signed_volume", "valid"])
    opening = (grouped.xs(True, level="open_phase") if True in present else empty).reindex(hour)
    other = (grouped.xs(False, level="open_phase") if False in present else empty).reindex(hour)
    complete = opening["valid"].eq(4) & other["valid"].eq(8)
    with np.errstate(divide="ignore", invalid="ignore"):
        open_flow = opening["signed_volume"] / opening["volume"]
        other_flow = other["signed_volume"] / other["volume"]
        share = opening["volume"] / (opening["volume"] + other["volume"])
    result = {"quarter_open_imbalance": open_flow.where(complete),
              "quarter_other_imbalance": other_flow.where(complete),
              "quarter_open_volume_share": share.where(complete)}
    return {name: series.where(np.isfinite(series)).to_numpy(dtype=np.float32)
            for name, series in result.items()}


def load_quarter_hour_panels(root: str | Path, codes: list[str], raw_dates: np.ndarray) -> dict[str, np.ndarray]:
    root = Path(root)
    fields: dict[str, list[np.ndarray]] = {}
    for code in codes:
        path = root / code / "5m.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pq.read_table(path, columns=["date", "quote_volume", "taker_buy_ratio"]).to_pandas()
        aligned = align_quarter_hour_frame(frame, raw_dates)
        for name, values in aligned.items():
            fields.setdefault(name, []).append(values)
    return {name: np.column_stack(arrays) for name, arrays in fields.items()}
