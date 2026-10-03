"""Discovery-calibrated multiscale trigger policy for the factor combo mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGNAL_ENGINE_VERSION = "2026-09-29-multiscale-trigger-v1"


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    native5_share: float
    hourly_share: float
    beta_share: float
    allocation: float
    half_life_bars: int
    min_hold_bars: int
    max_hold_bars: int
    cooldown_bars: int
    rebalance_bars: int
    target_trades_per_asset_week: float
    entry_candidates: tuple[float, ...]
    exit_ratio: float
    size_band: float

    @classmethod
    def from_dict(cls, raw: dict) -> "ProfileSpec":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError(f"signal profile needs exactly {sorted(expected)}; missing={sorted(expected-set(raw))}, unknown={sorted(set(raw)-expected)}")
        values = dict(raw)
        values["entry_candidates"] = tuple(values["entry_candidates"])
        p = cls(**values)
        if (not p.name.isidentifier() or p.allocation <= 0
                or min(p.native5_share, p.hourly_share, p.beta_share) < 0
                or abs(p.native5_share + p.hourly_share + p.beta_share - 1) > 1e-9
                or p.half_life_bars < 1 or p.min_hold_bars < 1
                or p.max_hold_bars < p.min_hold_bars or p.cooldown_bars < 0
                or p.rebalance_bars < 1 or p.target_trades_per_asset_week <= 0
                or not 0 < p.exit_ratio < 1 or not 0 <= p.size_band < 1
                or not p.entry_candidates or any(v <= 0 for v in p.entry_candidates)
                or tuple(sorted(set(p.entry_candidates))) != p.entry_candidates):
            raise ValueError(f"invalid signal profile: {p.name}")
        return p


@dataclass(frozen=True)
class SignalEngineSpec:
    scale_lookback_bars: int
    scale_floor_fraction: float
    max_abs_position: float
    leverage_diagnostic: float
    profiles: tuple[ProfileSpec, ...]

    @classmethod
    def from_dict(cls, raw: dict) -> "SignalEngineSpec":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError(f"signal_engine needs exactly {sorted(expected)}; missing={sorted(expected-set(raw))}, unknown={sorted(set(raw)-expected)}")
        profiles = tuple(ProfileSpec.from_dict(item) for item in raw["profiles"])
        spec = cls(raw["scale_lookback_bars"], raw["scale_floor_fraction"],
                   raw["max_abs_position"], raw["leverage_diagnostic"], profiles)
        if (spec.scale_lookback_bars < 48 or not 0 < spec.scale_floor_fraction <= 1
                or spec.max_abs_position < 1 or not 1 <= spec.leverage_diagnostic <= 2
                or not 2 <= len(profiles) <= 4
                or len({p.name for p in profiles}) != len(profiles)
                or sum(p.allocation for p in profiles) > spec.max_abs_position + 1e-9):
            raise ValueError("invalid signal_engine scale, leverage or profile allocation")
        return spec


def _segmented_ema(signal: np.ndarray, segment_lengths: list[int],
                   half_life_bars: int) -> np.ndarray:
    out = np.empty_like(signal, dtype=np.float32)
    decay = np.exp(-np.log(2) / half_life_bars)
    offset = 0
    for length in segment_lengths:
        previous = np.zeros(signal.shape[1])
        for local in range(length):
            t = offset + local
            previous = decay * previous + (1 - decay) * np.nan_to_num(signal[t], nan=0.0)
            out[t] = previous
        offset += length
    return out


def _adaptive_strength(smoothed: np.ndarray, train_panel: np.ndarray,
                       segment_lengths: list[int], spec: SignalEngineSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ntime, nasset = smoothed.shape
    if train_panel.shape != smoothed.shape or sum(segment_lengths) != ntime:
        raise ValueError("signal strength panels and segments do not align")
    anchor = np.empty(nasset, dtype=float)
    for j in range(nasset):
        sample = np.abs(smoothed[train_panel[:, j], j])
        sample = sample[np.isfinite(sample)]
        if len(sample) < 100:
            raise ValueError("insufficient training signal for adaptive trigger scale")
        anchor[j] = np.quantile(sample, .70)
        if anchor[j] <= 1e-8:
            raise ValueError("degenerate training signal for adaptive trigger scale")
    decay = np.exp(-np.log(2) / spec.scale_lookback_bars)
    scale = np.empty_like(smoothed, dtype=np.float32)
    strength = np.empty_like(smoothed, dtype=np.float32)
    offset = 0
    for length in segment_lengths:
        previous = anchor.copy()
        for local in range(length):
            t = offset + local
            previous = decay * previous + (1 - decay) * np.abs(smoothed[t])
            previous = np.clip(previous, anchor * spec.scale_floor_fraction,
                               anchor * 3)
            scale[t] = previous
            strength[t] = smoothed[t] / previous
        offset += length
    return strength, scale, anchor


def _simulate_profile(strength: np.ndarray, risk_budget: np.ndarray,
                      segment_lengths: list[int], profile: ProfileSpec,
                      entry: float, collect: bool = True) -> tuple[np.ndarray, list[dict], list[dict], int]:
    """A completed bar decides the position that earns the next 5m return."""
    ntime, nasset = strength.shape
    if len(risk_budget) != ntime or sum(segment_lengths) != ntime:
        raise ValueError("trigger panels and segments do not align")
    position = np.zeros_like(strength, dtype=np.float32)
    events: list[dict] = []
    holds: list[dict] = []
    closed = 0
    offset = 0
    for length in segment_lengths:
        current = np.zeros(nasset)
        side = np.zeros(nasset, dtype=np.int8)
        age = np.zeros(nasset, dtype=int)
        cooldown = np.zeros(nasset, dtype=int)
        entered = np.full(nasset, -1, dtype=int)
        for local in range(length):
            t = offset + local
            cooldown = np.maximum(cooldown - 1, 0)
            x = np.nan_to_num(strength[t], nan=0.0)
            for j in range(nasset):
                old = float(current[j])
                if side[j] != 0:
                    age[j] += 1
                    reason = None
                    if local == length - 1:
                        reason = "segment_end"
                    elif age[j] >= profile.max_hold_bars:
                        reason = "time_stop"
                    elif age[j] >= profile.min_hold_bars:
                        if np.sign(x[j]) != side[j] and abs(x[j]) >= entry * profile.exit_ratio:
                            reason = "opposite"
                        elif abs(x[j]) < entry * profile.exit_ratio:
                            reason = "signal_fade"
                    if reason is not None:
                        current[j] = 0
                        if collect:
                            events.append({"t": t, "asset_index": j, "profile": profile.name,
                                           "event": "close", "reason": reason,
                                           "previous_position": old, "new_position": 0.0,
                                           "strength": float(x[j]), "entry_threshold": entry,
                                           "exit_threshold": entry * profile.exit_ratio})
                            holds.append({"profile": profile.name, "asset_index": j,
                                          "start_t": int(entered[j]), "end_t": t,
                                          "direction": int(side[j]),
                                          "holding_hours": (t - entered[j]) * 5 / 60,
                                          "exit_reason": reason,
                                          "right_censored": reason == "segment_end"})
                        if reason != "segment_end":
                            closed += 1
                        side[j] = 0
                        entered[j] = -1
                        cooldown[j] = profile.cooldown_bars
                    elif local % profile.rebalance_bars == 0:
                        fraction = np.clip(abs(x[j]) / (1.5 * entry), .35, 1.0)
                        desired = side[j] * profile.allocation * fraction * risk_budget[t]
                        if abs(desired - current[j]) >= profile.size_band * profile.allocation:
                            current[j] = desired
                            if collect:
                                events.append({"t": t, "asset_index": j, "profile": profile.name,
                                               "event": "resize", "reason": "strength_or_risk",
                                               "previous_position": old,
                                               "new_position": float(desired),
                                               "strength": float(x[j]),
                                               "entry_threshold": entry,
                                               "exit_threshold": entry * profile.exit_ratio})
                elif (cooldown[j] == 0 and local < length - 1
                      and abs(x[j]) >= entry):
                    side[j] = int(np.sign(x[j]))
                    fraction = np.clip(abs(x[j]) / (1.5 * entry), .35, 1.0)
                    current[j] = side[j] * profile.allocation * fraction * risk_budget[t]
                    entered[j] = t
                    age[j] = 0
                    if collect:
                        events.append({"t": t, "asset_index": j, "profile": profile.name,
                                       "event": "open_long" if side[j] > 0 else "open_short",
                                       "reason": "strength_crossing",
                                       "previous_position": old,
                                       "new_position": float(current[j]),
                                       "strength": float(x[j]),
                                       "entry_threshold": entry,
                                       "exit_threshold": entry * profile.exit_ratio})
            position[t] = current
        offset += length
    return position, events, holds, closed


def _net_from_positions(position: np.ndarray, raw_y: np.ndarray,
                        segment_lengths: list[int], cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros_like(position)
    offset = 0
    for length in segment_lengths:
        delta[offset] = position[offset]
        delta[offset + 1:offset + length] = (
            position[offset + 1:offset + length] - position[offset:offset + length - 1])
        offset += length
    gross = np.where(np.isfinite(raw_y), position * raw_y, 0.0)
    net = gross - np.abs(delta) * cost_bps / 1e4
    return delta, net


def run_signal_engine(native_alpha: np.ndarray, hourly_alpha: np.ndarray,
                      beta_direction: np.ndarray, risk_budget: np.ndarray,
                      raw_y: np.ndarray, dates: np.ndarray,
                      train_panel: np.ndarray, segment_lengths: list[int],
                      spec: SignalEngineSpec, cost_bps: float) -> dict:
    ntime, nasset = native_alpha.shape
    if (hourly_alpha.shape != (ntime, nasset) or raw_y.shape != (ntime, nasset)
            or len(beta_direction) != ntime or len(risk_budget) != ntime
            or len(dates) != ntime or train_panel.shape != (ntime, nasset)
            or sum(segment_lengths) != ntime or cost_bps < 0):
        raise ValueError("signal engine inputs do not align")
    train_rows = np.flatnonzero(train_panel.any(axis=1))
    if not len(train_rows) or train_rows[-1] >= segment_lengths[0]:
        raise ValueError("signal engine training rows must end inside first segment")
    train_length = int(train_rows[-1] + 1)
    profile_positions, profile_strengths, profile_scales = {}, {}, {}
    calibration_rows, all_events, all_holds = [], [], []
    weeks = train_length / 288 / 7
    for profile in spec.profiles:
        direction = (profile.native5_share * native_alpha
                     + profile.hourly_share * hourly_alpha
                     + profile.beta_share * beta_direction[:, None])
        smoothed = _segmented_ema(direction, segment_lengths, profile.half_life_bars)
        strength, scale, anchor = _adaptive_strength(smoothed, train_panel,
                                                     segment_lengths, spec)
        candidates = []
        for threshold in profile.entry_candidates:
            _, _, _, closed = _simulate_profile(
                strength[:train_length], risk_budget[:train_length],
                [train_length], profile, threshold, collect=False)
            rate = closed / nasset / weeks
            candidates.append({"profile": profile.name, "entry_threshold": threshold,
                               "train_trades_per_asset_week": rate,
                               "target_trades_per_asset_week": profile.target_trades_per_asset_week,
                               "absolute_rate_gap": abs(rate - profile.target_trades_per_asset_week)})
        selected = min(candidates, key=lambda row: (
            row["absolute_rate_gap"], -row["entry_threshold"]))
        for row in candidates:
            row["selected"] = row is selected
            row["target_within_candidate_range"] = (
                min(item["train_trades_per_asset_week"] for item in candidates)
                <= profile.target_trades_per_asset_week
                <= max(item["train_trades_per_asset_week"] for item in candidates))
            row["training_scale_median"] = float(np.median(anchor))
        calibration_rows.extend(candidates)
        position, events, holds, _ = _simulate_profile(
            strength, risk_budget, segment_lengths, profile,
            selected["entry_threshold"])
        profile_positions[profile.name] = position
        profile_strengths[profile.name] = strength
        profile_scales[profile.name] = scale
        all_events.extend(events)
        all_holds.extend(holds)
    combined = np.sum(np.stack(list(profile_positions.values())), axis=0)
    combined = np.clip(combined, -spec.max_abs_position, spec.max_abs_position).astype(np.float32)
    positions = {"multiscale_trigger": combined,
                 "multiscale_leverage_diagnostic": np.clip(
                     combined * spec.leverage_diagnostic,
                     -spec.max_abs_position * spec.leverage_diagnostic,
                     spec.max_abs_position * spec.leverage_diagnostic).astype(np.float32)}
    positions.update({f"profile_{name}": value for name, value in profile_positions.items()})
    changes, net_returns = {}, {}
    for name, position in positions.items():
        changes[name], net_returns[name] = _net_from_positions(
            position, raw_y, segment_lengths, cost_bps)
    event_frame = pd.DataFrame(all_events)
    hold_frame = pd.DataFrame(all_holds)
    if not event_frame.empty:
        event_frame["date"] = dates[event_frame.t.to_numpy(dtype=int)]
    if not hold_frame.empty:
        hold_frame["start"] = dates[hold_frame.start_t.to_numpy(dtype=int)]
        hold_frame["end"] = dates[hold_frame.end_t.to_numpy(dtype=int)]
    return {"positions": positions, "changes": changes,
            "net_returns": net_returns, "profile_positions": profile_positions,
            "profile_strengths": profile_strengths, "profile_scales": profile_scales,
            "profile_events": event_frame, "profile_holds": hold_frame,
            "calibration": pd.DataFrame(calibration_rows)}
