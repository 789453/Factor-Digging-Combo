"""Causally aligned, optional hourly perpetual-futures fields (version 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DERIVATIVE_FIELD_VERSION = "manual-alpha-derivatives-20261003-v3-null-interval-metadata"
DERIVATIVE_DEPENDENCIES = {
    "derivative_basis": ("date", "mark_close", "mark_close_time", "index_close", "index_close_time"),
    "derivative_trade_mark_gap": ("date", "mark_close", "mark_close_time", "futures_close"),
    "derivative_funding_state": ("date", "funding_rate", "funding_interval_hours"),
}


def _valid_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.log(numerator / denominator)
    result[(numerator <= 0) | (denominator <= 0) | ~np.isfinite(result)] = np.nan
    return result


def align_derivative_frame(frame: pd.DataFrame, raw_dates: np.ndarray,
                           futures_close: np.ndarray) -> dict[str, np.ndarray]:
    """Match completed hourly prices; funding event may arrive milliseconds late.

    A row marked 08:00:00.008 is a funding event, not an 08:00-09:00
    mark/index bar. Funding becomes available at 09:00 and expires after 8h.
    """
    required = set().union(*DERIVATIVE_DEPENDENCIES.values()) - {"futures_close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"derivative columns missing: {sorted(missing)}")
    hour_open = pd.to_datetime(raw_dates, format="%Y%m%d%H%M", utc=True)
    decision = hour_open + pd.Timedelta(hours=1)
    date = pd.to_datetime(frame["date"], utc=True)
    exact = date.eq(date.dt.floor("h"))
    bars = frame.loc[exact].copy()
    bars.index = date[exact]
    if bars.index.has_duplicates:
        raise ValueError("duplicate exact-hour derivative price bars")
    bars = bars.reindex(hour_open)
    mark_end = pd.to_datetime(bars["mark_close_time"], utc=True)
    index_end = pd.to_datetime(bars["index_close_time"], utc=True)
    mark = pd.to_numeric(bars["mark_close"], errors="coerce").to_numpy(dtype=float, copy=True)
    index = pd.to_numeric(bars["index_close"], errors="coerce").to_numpy(dtype=float, copy=True)
    mark[np.asarray(mark_end > decision) | np.asarray(mark_end.isna())] = np.nan
    index[np.asarray(index_end > decision) | np.asarray(index_end.isna())] = np.nan
    basis = _valid_log_ratio(mark, index)
    trade_gap = _valid_log_ratio(np.asarray(futures_close, dtype=float), mark)

    # Funding is event data. Accept scheduled UTC 00/08/16 observations
    # within the first minute, retaining the last event if both exact and
    # millisecond-late records exist. Never forward fill an indefinite state.
    delta = date - date.dt.floor("h")
    scheduled = (date.dt.hour % 8 == 0) & (delta < pd.Timedelta(minutes=1))
    rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
    interval = pd.to_numeric(frame["funding_interval_hours"], errors="coerce")
    unsupported = scheduled & rate.notna() & interval.notna() & interval.ne(8)
    if unsupported.any():
        raise ValueError("non-8h scheduled funding interval requires an explicit clock contract")
    # Some 2026 rows have a real settled rate and an 8h scheduled timestamp,
    # but null interval metadata. Preserve the event and expose that defect.
    event = scheduled & rate.notna() & (interval.eq(8) | interval.isna())
    funding = frame.loc[event, ["funding_rate"]].copy()
    funding["interval_metadata_missing"] = interval[event].isna().to_numpy()
    funding["event_hour"] = date[event].dt.floor("h")
    funding["observation_time"] = date[event]
    funding = funding.sort_values("observation_time").drop_duplicates("event_hour", keep="last")
    values = np.full(len(hour_open), np.nan)
    age = np.full(len(hour_open), np.nan)
    realized_event = np.full(len(hour_open), np.nan)
    missing_interval = np.full(len(hour_open), np.nan)
    if len(funding):
        event_available = funding["event_hour"] + pd.Timedelta(hours=1)
        realized = funding.set_index("event_hour")["funding_rate"].reindex(decision)
        realized_event = pd.to_numeric(realized, errors="coerce").to_numpy(dtype=float, copy=True)
        realized_event[np.abs(realized_event) > .05] = np.nan
        missing_interval = funding.set_index("event_hour")["interval_metadata_missing"].reindex(decision).to_numpy(dtype=float)
        event_ns = event_available.astype("datetime64[ns, UTC]").array.asi8
        decision_ns = decision.astype("datetime64[ns, UTC]").asi8
        loc = np.searchsorted(event_ns, decision_ns, side="right") - 1
        good = loc >= 0
        idx = np.flatnonzero(good)
        elapsed = (decision_ns[idx] - event_ns[loc[idx]]) / 3_600_000_000_000
        rates = pd.to_numeric(funding["funding_rate"], errors="coerce").to_numpy(float)[loc[idx]]
        safe = (elapsed >= 0) & (elapsed < 8) & np.isfinite(rates) & (np.abs(rates) <= .05)
        values[idx[safe]] = rates[safe]
        age[idx[safe]] = elapsed[safe]
    return {"derivative_basis": basis.astype(np.float32),
            "derivative_trade_mark_gap": trade_gap.astype(np.float32),
            "derivative_funding_state": values.astype(np.float32),
            "derivative_funding_age": age.astype(np.float32),
            # Outcome at this completed-hour settlement, never an input to
            # the position made at that same time. Charged to prior holding.
            "derivative_funding_event": realized_event.astype(np.float32),
            "derivative_funding_interval_missing": missing_interval.astype(np.float32)}


def load_derivative_panels(root: str | Path, codes: list[str], raw_dates: np.ndarray,
                           futures_close: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict]]:
    root = Path(root)
    ntime, nasset = futures_close.shape
    by_field: dict[str, list[np.ndarray]] = {}
    fingerprints = []
    for j, code in enumerate(codes):
        path = root / f"{code}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        fingerprints.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        cols = sorted(set().union(*DERIVATIVE_DEPENDENCIES.values()) - {"futures_close"})
        frame = pq.read_table(path, columns=cols).to_pandas()
        result = align_derivative_frame(frame, raw_dates, futures_close[:, j])
        for name, values in result.items():
            if len(values) != ntime:
                raise ValueError(f"derivative clock mismatch: {code}")
            by_field.setdefault(name, []).append(values)
    return {name: np.column_stack(columns) for name, columns in by_field.items()}, fingerprints
