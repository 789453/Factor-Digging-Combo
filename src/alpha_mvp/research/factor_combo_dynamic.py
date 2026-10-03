"""Causal online combo weights, portfolio sleeves and execution accounting."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DYNAMIC_VERSION = "2026-09-29-online-combo-v3"


@dataclass(frozen=True)
class DynamicSpec:
    lookback_bars: int
    update_bars: int
    min_history_bars: int
    max_factors_per_family: int
    max_active_families: int
    entry_z: float
    exit_z: float
    anchor_weight: float
    weight_speed: float
    max_factor_weight: float
    alpha_budget: float
    beta_budget: float
    risk_floor: float
    target_half_life_bars: int
    event_band: float
    strategy_cost_bps: float
    plot_stride_bars: int

    @classmethod
    def from_dict(cls, raw: dict) -> "DynamicSpec":
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError(f"dynamic_combo needs exactly {sorted(expected)}; missing={sorted(expected-set(raw))}, unknown={sorted(set(raw)-expected)}")
        spec = cls(**raw)
        if (spec.lookback_bars < 48 or spec.min_history_bars < 24
                or spec.min_history_bars > spec.lookback_bars
                or spec.update_bars < 1 or 288 % spec.update_bars != 0
                or spec.max_factors_per_family < 1 or spec.max_active_families < 1
                or spec.target_half_life_bars < 1 or spec.plot_stride_bars < 1):
            raise ValueError("invalid dynamic_combo bars or family limits")
        if not (spec.exit_z < spec.entry_z and 0 <= spec.anchor_weight <= 1
                and 0 < spec.weight_speed <= 1 and 0 < spec.max_factor_weight <= 1
                and 0 <= spec.alpha_budget <= 1 and 0 <= spec.beta_budget <= 1
                and 0 < spec.alpha_budget + spec.beta_budget <= 1
                and 0 <= spec.risk_floor <= 1 and 0 < spec.event_band < 1
                and spec.strategy_cost_bps >= 0):
            raise ValueError("invalid dynamic_combo thresholds, budgets or costs")
        return spec


def _mean_asset(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(np.where(finite, values, 0).sum(axis=1), count,
                     out=np.full(count.shape, np.nan, dtype=float), where=count > 0)


def _quantile_scale(signal: np.ndarray, train: np.ndarray) -> float:
    sample = np.abs(signal[train])
    sample = sample[np.isfinite(sample)]
    if len(sample) < 100:
        raise ValueError("not enough finite training signals for target scale")
    scale = float(np.quantile(sample, .90))
    if scale < 1e-12:
        raise ValueError("degenerate training signal scale")
    return scale


def _alpha_target(signal: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float]:
    scale = _quantile_scale(signal, train)
    shaped = np.tanh(np.nan_to_num(signal / scale, nan=0.0))
    centered = shaped - _mean_asset(shaped)[:, None]
    # Scale the whole cross-section together so clipping cannot reintroduce beta.
    maximum = np.max(np.abs(centered), axis=1, keepdims=True)
    neutral = centered / np.maximum(maximum, 1.0)
    return neutral.astype(np.float32), scale


def _matured_payoffs(raw: np.ndarray, residual_y: np.ndarray,
                     dates: np.ndarray, segment_lengths: list[int],
                     registry: pd.DataFrame, utility_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each factor's payoff to the first bar when its full label is known."""
    if utility_mode not in {"next5", "source_horizon"}:
        raise ValueError(f"unknown factor utility mode: {utility_mode}")
    ntime, _, nfactors = raw.shape
    payoff = np.full((ntime, nfactors), np.nan, dtype=float)
    clocks = registry.clock.to_numpy()
    if utility_mode == "source_horizon" and not set(clocks) <= {"native5", "hourly"}:
        raise ValueError("source_horizon requires native5 or hourly factor clocks")
    if utility_mode == "source_horizon":
        needed = {"lag_bars", "horizon_bars", "stride_bars"}
        if not needed <= set(registry):
            raise ValueError(f"source_horizon registry missing {sorted(needed-set(registry))}")
        stride = registry.stride_bars.to_numpy(dtype=int)
        lag = registry.lag_bars.to_numpy(dtype=int)
        horizon = registry.horizon_bars.to_numpy(dtype=int)
        if (np.any(stride < 1) or np.any(lag < 0) or np.any(horizon < 1)
                or np.any((clocks == "hourly") & (stride != 12))
                or np.any((clocks == "native5") & (stride != 1))):
            raise ValueError("invalid mining source clock or horizon in registry")
        overlap = np.maximum(1, np.ceil(horizon / stride).astype(int))
    else:
        stride = np.ones(nfactors, dtype=int)
        overlap = np.ones(nfactors, dtype=int)
    offset = 0
    for length in segment_lengths:
        segment_raw = raw[offset:offset + length]
        segment_y = residual_y[offset:offset + length]
        if utility_mode == "next5":
            immediate = _mean_asset(segment_raw * segment_y[:, :, None])
            payoff[offset + 1:offset + length] = immediate[:-1]
        else:
            finite = np.isfinite(segment_y)
            cumulative = np.vstack([np.zeros((1, segment_y.shape[1])),
                                    np.cumsum(np.where(finite, segment_y, 0.0), axis=0)])
            valid_count = np.vstack([np.zeros((1, segment_y.shape[1]), dtype=int),
                                     np.cumsum(finite, axis=0)])
            for j in range(nfactors):
                maturity = int(lag[j] + horizon[j])
                if length <= maturity:
                    continue
                starts = np.arange(length - maturity, dtype=int)
                if clocks[j] == "hourly":
                    starts = starts[np.asarray([str(dates[offset + t])[-2:] == "00"
                                                 for t in starts])]
                if not len(starts):
                    continue
                begin = starts + lag[j]
                end = begin + horizon[j]
                forward = (cumulative[end] - cumulative[begin]) / horizon[j]
                complete = valid_count[end] - valid_count[begin] == horizon[j]
                forward = np.where(complete, forward, np.nan)
                realized = _mean_asset(segment_raw[starts, :, j] * forward)
                payoff[offset + end, j] = realized
        offset += length
    return payoff, stride, overlap


def online_factor_weights(raw: np.ndarray, residual_y: np.ndarray,
                          dates: np.ndarray, segment_lengths: list[int],
                          train_end: str, registry: pd.DataFrame,
                          spec: DynamicSpec,
                          utility_mode: str = "next5") -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Update only from matured payoffs; never flip the frozen discovery sign."""
    ntime, nasset, nfactors = raw.shape
    if residual_y.shape != (ntime, nasset) or len(registry) != nfactors:
        raise ValueError("online factor panels and registry do not align")
    payoff, stride, overlap = _matured_payoffs(
        raw, residual_y, dates, segment_lengths, registry, utility_mode)
    weights = np.zeros((ntime, nfactors), dtype=np.float32)
    family_ids = registry.family.to_numpy()
    families = list(dict.fromkeys(family_ids))
    family_members = {f: np.flatnonzero(family_ids == f) for f in families}
    factor_ids = registry.expr_hash.to_numpy()
    decay = np.exp(-stride / spec.lookback_bars)
    lookback_count = np.maximum(1, spec.lookback_bars / stride)
    min_count = np.maximum(1, spec.min_history_bars / stride)
    prior = None
    weight_rows, event_rows = [], []
    offset = 0
    for segment_number, length in enumerate(segment_lengths):
        if segment_number == 0:
            mean = np.zeros(nfactors)
            moment2 = np.zeros(nfactors)
            count = np.zeros(nfactors, dtype=int)
            previous_weight = np.zeros(nfactors)
            previous_active = np.zeros(nfactors, dtype=bool)
        else:
            if prior is None:
                raise ValueError("training state was not frozen before later segments")
            mean, moment2, count, previous_weight, previous_active = (
                prior[0].copy(), prior[1].copy(), prior[2].copy(), prior[3].copy(), prior[4].copy())
        for local in range(length):
            t = offset + local
            observed = payoff[t]
            finite = np.isfinite(observed)
            mean[finite] = decay[finite] * mean[finite] + (1 - decay[finite]) * observed[finite]
            moment2[finite] = decay[finite] * moment2[finite] + (1 - decay[finite]) * observed[finite] ** 2
            count[finite] += 1
            if segment_number == 0 and dates[t] == train_end:
                prior = (mean.copy(), moment2.copy(), count.copy(),
                         previous_weight.copy(), previous_active.copy())
            stamp = str(dates[t])
            minute_of_day = (int(stamp[-4:-2]) * 60 + int(stamp[-2:])) // 5
            if minute_of_day % spec.update_bars == 0:
                dispersion = np.sqrt(np.maximum(moment2 - mean ** 2, 1e-16))
                z = mean / dispersion * np.sqrt(np.minimum(count, lookback_count) / overlap)
                z = np.clip(np.nan_to_num(z), -8, 8)
                selected = np.zeros(nfactors, dtype=bool)
                if np.any(count >= min_count):
                    choices = []
                    for family, members in family_members.items():
                        candidates = [j for j in members if (
                            count[j] >= min_count[j] and (
                                (previous_active[j] and z[j] >= spec.exit_z)
                                or z[j] >= spec.entry_z))]
                        candidates.sort(key=lambda j: (-z[j], str(factor_ids[j])))
                        candidates = candidates[:spec.max_factors_per_family]
                        if candidates:
                            choices.append((max(z[candidates]), family, candidates))
                    choices.sort(key=lambda item: (-item[0], item[1]))
                    choices = choices[:spec.max_active_families]
                    for _, _, members in choices:
                        selected[members] = True
                    target = np.zeros(nfactors)
                    if choices:
                        strengths = np.array([max(item[0], .05) for item in choices])
                        family_weight = (spec.anchor_weight / len(choices)
                                         + (1 - spec.anchor_weight) * strengths / strengths.sum())
                        for family_share, (_, _, members) in zip(family_weight, choices):
                            inside = np.maximum(z[members], .05)
                            inside = inside / inside.sum()
                            target[members] = family_share * inside
                    new_weight = ((1 - spec.weight_speed) * previous_weight
                                  + spec.weight_speed * target)
                    new_weight[~selected] = 0
                    new_weight = np.minimum(new_weight, spec.max_factor_weight)
                else:
                    new_weight = previous_weight.copy()
                    selected = previous_active.copy()
                for j in range(nfactors):
                    if bool(selected[j]) != bool(previous_active[j]):
                        event_rows.append({"date": stamp, "segment": segment_number,
                                           "expr_hash": factor_ids[j], "family": family_ids[j],
                                           "event": "enter" if selected[j] else "exit",
                                           "utility_z": float(z[j]),
                                           "weight_after": float(new_weight[j])})
                    weight_rows.append({"date": stamp, "segment": segment_number,
                                        "expr_hash": factor_ids[j], "family": family_ids[j],
                                        "clock": registry.iloc[j].clock,
                                        "role": registry.iloc[j].role,
                                        "weight": float(new_weight[j]),
                                        "active": bool(selected[j]),
                                        "utility_z": float(z[j]),
                                        "payoff_ewm_bps": float(mean[j] * 1e4)})
                previous_weight, previous_active = new_weight, selected
            weights[t] = previous_weight
        offset += length
    event_columns = ["date", "segment", "expr_hash", "family", "event",
                     "utility_z", "weight_after"]
    return weights, pd.DataFrame(weight_rows), pd.DataFrame(event_rows, columns=event_columns)


def _execute_target(target: np.ndarray, future_return: np.ndarray,
                    segment_lengths: list[int], spec: DynamicSpec,
                    event_band: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Completion-time target, next-bar PnL, exact absolute-position cost."""
    ntime, nasset = target.shape
    held = np.zeros((ntime, nasset), dtype=np.float32)
    delta = np.zeros_like(held)
    net = np.zeros_like(held)
    decay = np.exp(-np.log(2) / spec.target_half_life_bars)
    band = spec.event_band if event_band is None else event_band
    if band <= 0:
        raise ValueError("execution event band must be positive")
    offset = 0
    for length in segment_lengths:
        previous = np.zeros(nasset)
        latent = np.zeros(nasset)
        for local in range(length):
            t = offset + local
            latent = decay * latent + (1 - decay) * target[t]
            # The future return's presence is unknown at decision time.
            # The declared segment endpoint alone is known in advance.
            can_open = local < length - 1
            candidate = np.where((np.abs(latent - previous) >= band) & can_open,
                                 latent, previous)
            move = candidate - previous
            held[t] = candidate
            delta[t] = move
            realized = np.where(np.isfinite(future_return[t]),
                                candidate * future_return[t], 0.0)
            net[t] = realized - np.abs(move) * spec.strategy_cost_bps / 1e4
            previous = candidate
        offset += length
    return held, delta, net


def _strategy_event_band(name: str, spec: DynamicSpec) -> float:
    """Apply the same fractional deadband to each strategy's position budget."""
    budget = (spec.beta_budget if name == "beta_only" else
              spec.alpha_budget if name == "alpha_only" or name in {
                  "single_frozen", "family_equal", "style_B0",
                  "elastic_net", "explicit_gate"} else
              spec.alpha_budget + spec.beta_budget)
    return max(spec.event_band * budget, 1e-12)


def build_dynamic_portfolios(raw: np.ndarray, residual_y: np.ndarray,
                             raw_y: np.ndarray, style: np.ndarray,
                             predictions: dict[str, np.ndarray],
                             q_5m: np.ndarray, dates: np.ndarray,
                             segment_lengths: list[int], train_end: str,
                             train_panel: np.ndarray, registry: pd.DataFrame,
                             spec: DynamicSpec, signal_spec) -> dict:
    weights, weight_rows, weight_events = online_factor_weights(
        raw, residual_y, dates, segment_lengths, train_end, registry, spec,
        utility_mode="source_horizon")
    fast_weights, _, _ = online_factor_weights(
        raw, residual_y, dates, segment_lengths, train_end, registry, spec,
        utility_mode="next5")
    dynamic_signal = np.einsum("tnf,tf->tn", raw, weights, optimize=True)
    dynamic_alpha, dynamic_scale = _alpha_target(dynamic_signal, train_panel)
    native_mask = registry.clock.to_numpy() == "native5"
    hourly_mask = registry.clock.to_numpy() == "hourly"
    native_signal = np.einsum("tnf,tf->tn", raw[:, :, native_mask],
                              weights[:, native_mask], optimize=True)
    hourly_signal = np.einsum("tnf,tf->tn", raw[:, :, hourly_mask],
                              weights[:, hourly_mask], optimize=True)
    native_alpha, native_scale = _alpha_target(native_signal, train_panel)
    hourly_alpha, hourly_scale = _alpha_target(hourly_signal, train_panel)
    fast_signal = np.einsum("tnf,tf->tn", raw, fast_weights, optimize=True)
    fast_alpha, fast_scale = _alpha_target(fast_signal, train_panel)
    reference = {
        "single_frozen": raw[:, :, 0],
        "family_equal": np.stack([
            raw[:, :, (registry.family == f).to_numpy()].mean(axis=2)
            for f in dict.fromkeys(registry.family)], axis=2).mean(axis=2),
        "style_B0": predictions["B0_style"],
        "elastic_net": predictions["B4_elastic_net"],
        "explicit_gate": predictions["B5_explicit_gate"],
    }
    reference_target = {name: _alpha_target(signal, train_panel)[0]
                        for name, signal in reference.items()}
    market_now = _mean_asset(style[:, :, 0])
    market_hour = _mean_asset(style[:, :, 1])
    market_y = _mean_asset(raw_y)
    market_features = np.column_stack([market_now, market_hour])
    train_time = train_panel.any(axis=1) & np.isfinite(market_y)
    if train_time.sum() < 1000:
        raise ValueError("insufficient common beta training timestamps")
    beta_model = Ridge(alpha=1000.0).fit(market_features[train_time], market_y[train_time] * 1e4)
    beta_prediction = beta_model.predict(market_features) / 1e4
    beta_scale = _quantile_scale(beta_prediction, train_time)
    beta_direction = np.tanh(beta_prediction / beta_scale)
    q = np.maximum(np.nan_to_num(q_5m, nan=np.inf), 0)
    q_train = q[train_time & np.isfinite(q)]
    if len(q_train) < 1000:
        raise ValueError("insufficient trained common-risk predictions")
    q50, q90 = np.quantile(q_train, [.50, .90])
    if q90 - q50 <= 1e-12:
        raise ValueError("common-risk forecast has degenerate training quantiles")
    stress = np.clip((q - q50) / (q90 - q50), 0, 1)
    risk_budget = 1 - (1 - spec.risk_floor) * stress
    alpha_sleeve = spec.alpha_budget * dynamic_alpha
    beta_sleeve = spec.beta_budget * beta_direction[:, None] * np.ones_like(dynamic_alpha)
    targets = {
        "alpha_only": alpha_sleeve,
        "beta_only": beta_sleeve,
        "mix_no_risk": alpha_sleeve + beta_sleeve,
        "mix_risk_dynamic": risk_budget[:, None] * (alpha_sleeve + beta_sleeve),
        "mix_risk_5m_utility": risk_budget[:, None] * (
            spec.alpha_budget * fast_alpha + beta_sleeve),
        **{name: spec.alpha_budget * value for name, value in reference_target.items()},
    }
    positions, changes, net_returns = {}, {}, {}
    for name, target in targets.items():
        positions[name], changes[name], net_returns[name] = _execute_target(
            target, raw_y, segment_lengths, spec,
            event_band=_strategy_event_band(name, spec))
    from .factor_combo_signal_engine import run_signal_engine
    signal_engine = run_signal_engine(
        native_alpha, hourly_alpha, beta_direction, risk_budget, raw_y,
        dates, train_panel, segment_lengths, signal_spec,
        spec.strategy_cost_bps)
    positions.update(signal_engine["positions"])
    changes.update(signal_engine["changes"])
    net_returns.update(signal_engine["net_returns"])
    profile_target = np.zeros_like(dynamic_alpha)
    for profile in signal_spec.profiles:
        profile_target += profile.allocation * (
            profile.native5_share * native_alpha
            + profile.hourly_share * hourly_alpha
            + profile.beta_share * beta_direction[:, None])
    targets["multiscale_trigger"] = np.clip(
        profile_target * risk_budget[:, None],
        -signal_spec.max_abs_position, signal_spec.max_abs_position)
    return {
        "weights": weights, "weight_rows": weight_rows,
        "weight_events": weight_events, "dynamic_signal": dynamic_signal,
        "fast_weights": fast_weights,
        "dynamic_alpha": dynamic_alpha,
        "native_alpha": native_alpha, "hourly_alpha": hourly_alpha,
        "native_signal_scale": native_scale, "hourly_signal_scale": hourly_scale,
        "alpha_sleeve": alpha_sleeve, "beta_sleeve": beta_sleeve,
        "beta_prediction": beta_prediction, "beta_direction": beta_direction,
        "risk_budget": risk_budget, "q_5m": q_5m,
        "q50": float(q50), "q90": float(q90),
        "beta_model_coef_bps": beta_model.coef_.tolist(),
        "dynamic_signal_scale": dynamic_scale,
        "fast_dynamic_signal_scale": fast_scale,
        "beta_signal_scale": beta_scale,
        "targets": targets, "positions": positions,
        "changes": changes, "net_returns": net_returns,
        "signal_engine": signal_engine,
    }
