"""Frozen factor-combination baselines on a native 5m decision clock.

This is a research mode of research_cli, not an alternative production entry.
All selection and scaling use the declared training period. Later years are
historical reviews and never feed feature choice or model fitting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from ..crypto_fields import add_crypto_features
from ..data import load_crypto_parquet
from ..evaluator import BatchEvaluator, make_panels
from ..native5_fields import add_crypto_native_5m_features
from ..parser import parse_expr


VERSION = "factor-combo-5m-signal-engine-v4"
STYLE = ("ret_5m", "ret_1h", "rv_12", "volume_shock_12",
         "amihud_shock_12", "us_day_flag", "us_evening_flag")
RISK_EXPR = ("GateNeg(TsZScore($taker_imbalance_change_4h,168),"
             "TsZScore(TsEMA($micro5_realized_vol,12),168))")


def _fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fields_windows(expr: str) -> tuple[set[str], set[int]]:
    fields: set[str] = set()
    windows: set[int] = set()

    def walk(node):
        if node.kind == "field":
            fields.add(node.value)
        elif node.kind == "op":
            for child in node.args:
                walk(child)
            if node.value.startswith(("Ts", "Ref")) and node.args[-1].kind == "const":
                windows.add(int(node.args[-1].value))

    walk(parse_expr(expr))
    return fields, windows


def _registry(cfg: dict) -> pd.DataFrame:
    rows = []
    for source in cfg["factor_sources"]:
        path = Path(source["path"])
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"factor source missing mining manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_clock = {"native5": "5m", "hourly": "1h"}.get(source["clock"])
        evaluation = manifest.get("config", {}).get("evaluation", {})
        if (manifest.get("status") != "COMPLETED"
                or manifest.get("artifacts", {}).get("selected_factors") != path.name
                or manifest.get("data_fingerprint", {}).get("primary_timeframe") != expected_clock
                or evaluation.get("factor_mode") != "time_series"
                or not isinstance(evaluation.get("entry_lag"), int)
                or not isinstance(evaluation.get("horizon"), int)
                or evaluation["entry_lag"] < 0 or evaluation["horizon"] < 1):
            raise ValueError(f"incompatible mining manifest for combo: {manifest_path}")
        clock_stride = 1 if source["clock"] == "native5" else 12
        needed = {"expr", "expr_hash", "template_family", "direction",
                  "rank_equivalence_hash", "template_name", "fields", "operators",
                  "windows", "target_mode", "target_scope"}
        # The holdout firewall applies even to columns we would otherwise ignore.
        frame = pd.read_csv(path, usecols=lambda column: column in needed)
        if not needed <= set(frame):
            raise ValueError(f"factor source missing {sorted(needed - set(frame))}: {source['path']}")
        if not 0 < int(source["limit"]) <= 100:
            raise ValueError("factor source limit must be in [1, 100]")
        frame = frame.head(int(source["limit"]))
        for source_rank, row in enumerate(frame.itertuples(index=False), 1):
            direction = int(row.direction)
            if direction not in (-1, 1):
                raise ValueError(f"missing frozen direction: {row.expr_hash}")
            if (row.target_mode != evaluation.get("target_mode")
                    or row.target_scope != evaluation.get("target_scope")):
                raise ValueError(f"factor target metadata disagrees with manifest: {row.expr_hash}")
            rows.append({"clock": source["clock"], "role": source["role"],
                         "expr": row.expr, "expr_hash": row.expr_hash,
                         "family": f"{source['clock']}:{row.template_family}",
                         "direction": direction, "source": source["path"],
                         "source_rank": source_rank,
                         "source_experiment_id": manifest["experiment_id"],
                         "source_manifest_sha256": _fingerprint(manifest_path),
                         "source_selected_sha256": _fingerprint(path),
                         "source_framework_version": manifest["framework_version"],
                         "source_timeframe": expected_clock,
                         "target_mode": row.target_mode,
                         "target_scope": row.target_scope,
                         "lag_bars": clock_stride * evaluation["entry_lag"],
                         "horizon_bars": clock_stride * evaluation["horizon"],
                         "stride_bars": clock_stride,
                         "rank_equivalence_hash": row.rank_equivalence_hash,
                         "template_name": row.template_name,
                         "fields": row.fields, "operators": row.operators,
                         "windows": row.windows})
    result = pd.DataFrame(rows).drop_duplicates("expr_hash", keep="first")
    if result.empty or not {"native5", "hourly"} <= set(result.clock):
        raise ValueError("combo requires frozen native5 and hourly factors")
    if not set(result.clock) <= {"native5", "hourly"} or not set(result.role) <= {"alpha", "mixed"}:
        raise ValueError("factor clock/role must be native5/hourly and alpha/mixed")
    return result.reset_index(drop=True)


def _evaluate(panel: dict[str, np.ndarray], dates: list[str], codes: list[str],
              expressions: list[str]) -> list[np.ndarray]:
    fields, windows = set(), set()
    for expr in expressions:
        f, w = _fields_windows(expr)
        fields |= f
        windows |= w
    missing = fields - set(panel)
    if missing:
        raise ValueError(f"combo expression fields not available: {sorted(missing)}")
    ev = BatchEvaluator({f: panel[f] for f in fields}, dates, codes,
                        windows=tuple(sorted(windows)), max_depth=12,
                        max_nodes=32, max_ts_ops=8, max_pair_ops=3,
                        max_binary_ops=8, max_cache_items=24)
    values = []
    for expr in expressions:
        value, status = ev.eval_expr(expr)
        if value is None:
            raise ValueError(f"combo expression failed: {expr}: {status}")
        values.append(value.astype(np.float32))
    return values


def align_completed_hourly(hourly_values: np.ndarray, hourly_dates: list[str],
                           fast_dates: list[str]) -> np.ndarray:
    """Hourly rows are open stamped; only expose at open + 60 minutes."""
    h = pd.to_datetime(hourly_dates, format="%Y%m%d%H%M", utc=True).to_numpy(dtype="datetime64[ns]")
    f = pd.to_datetime(fast_dates, format="%Y%m%d%H%M", utc=True).to_numpy(dtype="datetime64[ns]")
    available = h + np.timedelta64(60, "m")
    index = np.searchsorted(available, f, side="right") - 1
    result = np.full((len(f), *hourly_values.shape[1:]), np.nan, dtype=np.float32)
    good = index >= 0
    result[good] = hourly_values[index[good]]
    return result


def _segment(cfg: dict, segment: dict, registry: pd.DataFrame) -> dict:
    root = cfg["data_root"]
    if root == "simulated":
        return _simulated_segment(segment, registry)
    # Warmup is inside the segment's year; no artificial rolling continuity at gaps.
    start = pd.Timestamp(segment["start"], tz="UTC")
    warm = start - pd.Timedelta(days=int(cfg["warmup_days"]))
    end = pd.Timestamp(segment["end"], tz="UTC")
    fast = add_crypto_native_5m_features(load_crypto_parquet(
        root, str(warm), str(end), "5m", "15m", "5m"))
    hour = add_crypto_features(load_crypto_parquet(
        root, str(warm), str(end), "1h", "15m", "5m"))
    nf, nw = set(STYLE) | {"close"}, set()
    hf, hw = {"close"}, set()
    for row in registry.itertuples():
        fields, windows = _fields_windows(row.expr)
        if row.clock == "native5":
            nf |= fields; nw |= windows
        else:
            hf |= fields; hw |= windows
    risk_fields, _ = _fields_windows(RISK_EXPR)
    hf |= risk_fields
    fp, fd, fc = make_panels(fast, sorted(nf - {"close"}))
    hp, hd, hc = make_panels(hour, sorted(hf - {"close"}))
    if fc != hc:
        raise ValueError("native5/hourly asset universes differ")
    native_rows = registry[registry.clock == "native5"]
    hour_rows = registry[registry.clock == "hourly"]
    nv = _evaluate(fp, fd, fc, native_rows.expr.tolist())
    hv = _evaluate(hp, hd, hc, [*hour_rows.expr.tolist(), RISK_EXPR])
    values: dict[str, np.ndarray] = {}
    for row, value in zip(native_rows.itertuples(), nv):
        values[row.expr_hash] = value * row.direction
    for row, value in zip(hour_rows.itertuples(), hv[:-1]):
        values[row.expr_hash] = align_completed_hourly(value, hd, fd) * row.direction
    risk = align_completed_hourly(hv[-1], hd, fd)
    # Remove warmup decisions and the incomplete last forward label.
    keep = np.array([segment["start"].replace("-", "").replace(" ", "").replace(":", "")[:12]
                     <= d <= segment["end"].replace("-", "").replace(" ", "").replace(":", "")[:12]
                     for d in fd])
    return {"dates": np.asarray(fd)[keep], "codes": fc,
            "close": fp["close"][keep].astype(np.float32),
            "style": np.stack([fp[name][keep] for name in STYLE], axis=-1).astype(np.float32),
            "factor": np.stack([values[h][keep] for h in registry.expr_hash], axis=-1),
            "risk": risk[keep].astype(np.float32), "segment": segment["name"]}


def _simulated_segment(segment: dict, registry: pd.DataFrame) -> dict:
    """Small deterministic end-to-end smoke source with completed 5m timestamps."""
    times = pd.date_range(segment["start"], segment["end"], freq="5min", tz="UTC")
    n, assets = len(times), 6
    seed = int(hashlib.sha256(segment["name"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, assets, len(registry))).astype(np.float32)
    style = rng.normal(size=(n, assets, len(STYLE))).astype(np.float32)
    risk = rng.normal(size=(n, assets)).astype(np.float32)
    r = (rng.normal(0, .002, size=(n, assets)) +
         .00005 * factor[:, :, 0]).astype(np.float32)
    close = 100 * np.exp(np.cumsum(r, axis=0))
    return {"dates": times.strftime("%Y%m%d%H%M").to_numpy(),
            "codes": [f"SIM{i}" for i in range(assets)], "close": close,
            "style": style, "factor": factor, "risk": risk,
            "segment": segment["name"]}


def _bar_and_market(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.log(np.where(close > 0, close, np.nan))
    r = np.full_like(p, np.nan)
    r[1:] = np.diff(p, axis=0)
    ok = np.isfinite(r)
    count = ok.sum(axis=1, keepdims=True)
    total = np.where(ok, r, 0).sum(axis=1, keepdims=True)
    loo = np.divide(total - np.where(ok, r, 0), count - ok.astype(int),
                    out=np.full_like(r, np.nan), where=count - ok.astype(int) >= 3)
    return r, loo


def _finite_mean(x: np.ndarray) -> np.ndarray:
    good = np.isfinite(x)
    return np.divide(np.where(good, x, 0).sum(axis=1), good.sum(axis=1),
                     out=np.full(len(x), np.nan), where=good.sum(axis=1) > 0)


def _finite_median(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    good = np.isfinite(x).any(axis=1)
    out[good] = np.nanmedian(x[good], axis=1)
    return out


def _targets(segments: list[dict], train_name: str, train_end: str) -> np.ndarray:
    # Beta uses only completed returns in the declared training slice.
    first = next(s for s in segments if s["segment"] == train_name)
    r, loo = _bar_and_market(first["close"])
    train = first["dates"] <= train_end
    betas = []
    for j in range(r.shape[1]):
        valid = train & np.isfinite(r[:, j]) & np.isfinite(loo[:, j])
        if valid.sum() < 288:
            raise ValueError(f"insufficient beta training data for asset {first['codes'][j]}")
        betas.append(np.cov(r[valid, j], loo[valid, j])[0, 1] /
                     np.var(loo[valid, j], ddof=1))
    beta = np.asarray(betas)
    for s in segments:
        r, loo = _bar_and_market(s["close"])
        s["bar"] = r
        s["market_loo"] = loo
        s["beta"] = beta
        s["residual_bar"] = r - loo * beta
        s["y"] = np.vstack([s["residual_bar"][1:], np.full((1, r.shape[1]), np.nan)])
        s["raw_y"] = np.vstack([r[1:], np.full((1, r.shape[1]), np.nan)])
        s["market_y"] = np.vstack([loo[1:], np.full((1, r.shape[1]), np.nan)])
        # One market observation per timestamp, future 12-bar downside risk.
        market = _finite_mean(r)
        down = np.minimum(market, 0) ** 2
        s["risk_y"] = np.full(len(r), np.nan)
        if len(r) > 12:
            s["risk_y"][:-12] = np.convolve(down[1:], np.ones(12), "valid")
    return beta


def _robust_fit(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    sample = flat[train.reshape(-1)]
    med = np.nanmedian(sample, axis=0)
    lo = np.nanquantile(sample, .25, axis=0)
    hi = np.nanquantile(sample, .75, axis=0)
    scale = np.where(hi - lo > 1e-8, (hi - lo) / 1.349, 1.0)
    if not np.isfinite(med).all():
        raise ValueError("combo feature has no finite training observations")
    z = np.clip((flat - med) / scale, -5, 5)
    z[~np.isfinite(z)] = 0
    return z.astype(np.float32).reshape(x.shape), {"median": med.tolist(), "scale": scale.tolist()}


def deduplicate_factors(x: np.ndarray, registry: pd.DataFrame, train: np.ndarray,
                        threshold: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Discovery-only numerical alias pruning; keep source order fixed."""
    flat = x.reshape(-1, x.shape[-1])
    mask = train.reshape(-1)
    sample = flat[mask][::max(1, mask.sum() // 25000)]
    chosen = []
    notes = []
    for j, row in registry.iterrows():
        alias = None
        for k in chosen:
            valid = np.isfinite(sample[:, j]) & np.isfinite(sample[:, k])
            if valid.sum() > 200 and abs(np.corrcoef(sample[valid, j], sample[valid, k])[0, 1]) >= threshold:
                alias = registry.iloc[k].expr_hash
                break
        notes.append({"expr_hash": row.expr_hash, "alias_of": alias,
                      "kept": alias is None})
        if alias is None:
            chosen.append(j)
    return x[:, :, chosen], registry.iloc[chosen].reset_index(drop=True).assign(
        original_column=chosen), pd.DataFrame(notes)


def _family_values(x: np.ndarray, registry: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = list(dict.fromkeys(registry.family))
    family = np.stack([np.mean(x[:, :, (registry.family == name).to_numpy()], axis=2)
                       for name in names], axis=2)
    return family, names


def _predict_models(style: np.ndarray, raw: np.ndarray, family: np.ndarray,
                    state: np.ndarray, y: np.ndarray, train: np.ndarray,
                    family_names: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    shape = y.shape
    b = style.reshape(-1, style.shape[-1]); a = family.reshape(-1, family.shape[-1])
    z = state.reshape(-1, state.shape[-1]); target = y.reshape(-1) * 1e4
    valid = train.reshape(-1) & np.isfinite(target)
    if valid.sum() < 3000:
        raise ValueError("combo training has fewer than 3000 valid rows")
    fit = np.flatnonzero(valid)[::max(1, valid.sum() // 120000)]
    axz = a[:, :min(4, a.shape[1])] * z[:, :1]
    designs = {
        "B0_style": b,
        "B1_single_frozen": np.column_stack([b, raw.reshape(-1, raw.shape[-1])[:, 0]]),
        "B1_raw_equal": np.column_stack([b, raw.reshape(-1, raw.shape[-1]).mean(axis=1)]),
        "B2_family_equal": np.column_stack([b, a.mean(axis=1)]),
        "B4_ridge": np.column_stack([b, a]),
        "B4_elastic_net": np.column_stack([b, a]),
        "B5_state_main": np.column_stack([b, a, z]),
        "B5_explicit_gate": np.column_stack([b, a, z, axz]),
    }
    predictions, coefficients = {}, {}
    for name, design in designs.items():
        model = (ElasticNet(alpha=.03, l1_ratio=.5, max_iter=3000)
                 if name == "B4_elastic_net" else Ridge(alpha=1000.0))
        model.fit(design[fit], target[fit])
        predictions[name] = (model.predict(design).reshape(shape) / 1e4).astype(np.float32)
        coefficients[name] = {"intercept_bps": float(model.intercept_),
                              "coefficients_bps": model.coef_.tolist(),
                              "training_rows": len(fit)}
    coefficients["gate_interactions"] = family_names[:min(4, len(family_names))]
    return predictions, coefficients


def _metrics(y: np.ndarray, base: np.ndarray, predictions: dict[str, np.ndarray],
             segment: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    rows = []
    for period in dict.fromkeys(segment):
        time = segment == period
        for name, pred in predictions.items():
            mask = np.isfinite(y) & np.isfinite(base) & time[:, None]
            if mask.sum() < 100:
                continue
            mse = np.mean((y[mask] - pred[mask]) ** 2)
            bmse = np.mean((y[mask] - base[mask]) ** 2)
            ic = np.corrcoef(y[mask], pred[mask])[0, 1]
            per_asset = []
            for j in range(y.shape[1]):
                m = mask[:, j]
                if m.sum() >= 100:
                    per_asset.append(np.corrcoef(y[m, j], pred[m, j])[0, 1])
            rows.append({"period": period, "model": name, "rows": int(mask.sum()),
                         "first_utc": dates[time][0], "last_utc": dates[time][-1],
                         "mse": mse, "delta_loss_vs_B0": 1 - mse / bmse,
                         "pooled_ic": ic, "positive_asset_share": np.mean(np.array(per_asset) > 0)})
    return pd.DataFrame(rows)


def _diagnostics(segments: list[dict], predictions: dict[str, np.ndarray],
                 review_period: np.ndarray, train: np.ndarray,
                 family: np.ndarray, family_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    similarity = []
    flat = family.reshape(-1, family.shape[-1])
    sample = flat[train.reshape(-1)][::max(1, train.sum() // 30000)]
    for j, left in enumerate(family_names):
        for k in range(j + 1, len(family_names)):
            similarity.append({"family_a": left, "family_b": family_names[k],
                               "training_signal_corr": np.corrcoef(sample[:, j], sample[:, k])[0, 1]})
    market_y = np.concatenate([s["market_y"] for s in segments])
    raw_y = np.concatenate([s["raw_y"] for s in segments])
    exposure, decay = [], []
    for period in dict.fromkeys(review_period):
        rows = review_period == period
        for name, pred in predictions.items():
            valid = np.isfinite(market_y) & np.isfinite(raw_y) & rows[:, None]
            exposure.append({"period": period, "model": name, "rows": int(valid.sum()),
                             "forecast_vs_future_market_corr": np.corrcoef(
                                 pred[valid], market_y[valid])[0, 1],
                             "forecast_vs_raw_return_ic": np.corrcoef(
                                 pred[valid], raw_y[valid])[0, 1]})
    start = 0
    for s in segments:
        n = len(s["dates"])
        for lag in (1, 2, 3, 6, 12, 24):
            future = np.full_like(s["residual_bar"], np.nan)
            future[:-lag] = s["residual_bar"][lag:]
            same_period = np.zeros(n, dtype=bool)
            same_period[:-lag] = review_period[start:start+n-lag] == review_period[start+lag:start+n]
            for period in dict.fromkeys(review_period[start:start+n]):
                row = (review_period[start:start+n] == period) & same_period
                for name in ("B2_family_equal", "B4_ridge", "B5_explicit_gate"):
                    pred = predictions[name][start:start+n]
                    valid = np.isfinite(future) & row[:, None]
                    if valid.sum() < 100:
                        continue
                    decay.append({"period": period, "model": name, "lag_5m_bars": lag,
                                  "rows": int(valid.sum()), "single_bar_ic": np.corrcoef(
                                      pred[valid], future[valid])[0, 1],
                                  "signed_response_bps": np.mean(np.sign(pred[valid]) * future[valid]) * 1e4})
        start += n
    return pd.DataFrame(similarity), pd.DataFrame(exposure), pd.DataFrame(decay)


def _slow_targets(pred: np.ndarray, raw_y: np.ndarray, dates: np.ndarray,
                  segments: np.ndarray, train: np.ndarray,
                  costs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Train-only scale; target smoothing is distinct from predictive features.
    scale = np.nanquantile(abs(pred[train]), .90)
    if not np.isfinite(scale) or scale < 1e-9:
        raise ValueError("prediction scale is degenerate")
    desired = np.clip(pred / scale, -1, 1)
    rows, curves = [], []
    for period in dict.fromkeys(segments):
        ix = np.flatnonzero(segments == period)
        for rebalance, half_life in ((1, 0), (12, 12), (48, 48), ("event", 12)):
            p = np.zeros((len(ix), pred.shape[1]), dtype=np.float32)
            decay = 0 if half_life == 0 else np.exp(-np.log(2) / half_life)
            latent = np.zeros(pred.shape[1])
            held = np.zeros(pred.shape[1])
            for k, i in enumerate(ix):
                latent = decay * latent + (1 - decay) * desired[i]
                if rebalance == "event":
                    held = np.where(np.abs(latent - held) >= .25, latent, held)
                elif k % rebalance == 0:
                    held = latent.copy()
                p[k] = held
            turnover = np.abs(p - np.vstack([np.zeros((1, p.shape[1])), p[:-1]]))
            gross = _finite_mean(p * raw_y[ix])
            traded = np.nanmean(turnover, axis=1)
            valid = np.isfinite(gross)
            for cost in costs:
                net = np.where(valid, gross - traded * cost / 1e4, 0.0)
                label = (f"event_band0.25_hl{half_life}_cost{cost:g}bps" if rebalance == "event"
                         else f"{rebalance}bars_hl{half_life}_cost{cost:g}bps")
                rows.append({"period": period, "scenario": label,
                             "gross_log_return": float(np.nansum(gross)),
                             "net_log_return": float(np.sum(net)),
                             "turnover": float(traded[valid].sum()),
                             "break_even_cost_bps": float(np.nansum(gross) / traded[valid].sum() * 1e4)
                             if traded[valid].sum() > 0 else np.nan})
                curves.extend({"period": period, "date": dates[i], "scenario": label,
                               "cumulative_net_log_return": float(v)}
                              for i, v in zip(ix[::12], np.cumsum(net)[::12]))
    return pd.DataFrame(rows), pd.DataFrame(curves)


def _risk_comparison(segments: list[dict], train_name: str,
                     train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common risk gets one observation per completed hour, never per asset."""
    records = []
    for s in segments:
        market = _finite_mean(s["bar"])
        past = pd.Series(np.minimum(market, 0) ** 2).rolling(12, min_periods=12).sum().to_numpy()
        factor = _finite_median(s["risk"])
        clock = np.column_stack([_finite_median(s["style"][:, :, j]) for j in (5, 6)])
        x = np.column_stack([past * 1e6, clock])
        z = np.column_stack([x, factor])
        y = s["risk_y"] * 1e6
        hourly = np.array([str(d)[-2:] == "00" for d in s["dates"]])
        good = hourly & np.isfinite(x).all(axis=1) & np.isfinite(z).all(axis=1) & np.isfinite(y)
        records.append((s["segment"], s["dates"], x, z, y, good))
    fit_x, fit_z, fit_y = [], [], []
    for name, dates, x, z, y, good in records:
        if name == train_name:
            risk_cutoff = (pd.to_datetime(train_end, format="%Y%m%d%H%M", utc=True)
                           - pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M")
            mask = good & (dates <= risk_cutoff)
            fit_x.append(x[mask]); fit_z.append(z[mask]); fit_y.append(y[mask])
    if not fit_x or sum(map(len, fit_y)) < 300:
        raise ValueError("insufficient hourly common-risk training observations")
    baseline = Ridge(alpha=100).fit(np.vstack(fit_x), np.concatenate(fit_y))
    augmented = Ridge(alpha=100).fit(np.vstack(fit_z), np.concatenate(fit_y))
    rows, forecasts = [], []
    for name, dates, x, z, y, good in records:
        available = np.array([str(d)[-2:] == "00" for d in dates]) & np.isfinite(z).all(axis=1)
        if available.any():
            b_all = baseline.predict(x[available]) / 1e6
            a_all = augmented.predict(z[available]) / 1e6
            forecasts.extend({"segment": name, "date": date,
                              "future_downside_1h": float(actual / 1e6),
                              "q_style": float(baseline_q), "q_with_risk_expr": float(augmented_q)}
                             for date, actual, baseline_q, augmented_q in zip(
                                 dates[available], y[available], b_all, a_all))
        partitions = (
            [(f"{name}_train", dates <= train_end),
             (f"{name}_review", dates > train_end)]
            if name == train_name else [(name, np.ones(len(dates), dtype=bool))]
        )
        for label, part in partitions:
            mask = good & part
            if mask.sum() < 100:
                continue
            b = baseline.predict(x[mask]); a = augmented.predict(z[mask])
            bmse = np.mean((y[mask] - b) ** 2)
            rows.append({"period": label, "hourly_observations": int(mask.sum()),
                         "baseline_mse_micro_units": bmse,
                         "risk_expr_delta_loss": 1 - np.mean((y[mask] - a) ** 2) / bmse,
                         "risk_expr_ic": np.corrcoef(y[mask], a)[0, 1]})
    return pd.DataFrame(rows), pd.DataFrame(forecasts)


def _dashboard(out: Path, comparison: pd.DataFrame, curves: pd.DataFrame,
               decay: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        "Relative 5m residual forecast loss (vs style B0)",
        "Single future 5m bar IC by information lag",
        "Signed long-short cumulative net log return (fixed target scenarios)"))
    for name, group in comparison.groupby("model", sort=False):
        fig.add_trace(go.Bar(name=name, x=group.period, y=group.delta_loss_vs_B0,
                             legendgroup=name), row=1, col=1)
    for (period, name), group in decay.groupby(["period", "model"], sort=False):
        if period.endswith("_train"):
            continue
        fig.add_trace(go.Scatter(name=f"{period} {name} decay", x=group.lag_5m_bars,
                                 y=group.single_bar_ic, mode="lines+markers"), row=2, col=1)
    for (period, name), group in curves.groupby(["period", "scenario"], sort=False):
        if "cost4bps" not in name:
            continue
        fig.add_trace(go.Scatter(name=f"{period} {name}", x=group.date,
                                 y=group.cumulative_net_log_return, mode="lines"), row=3, col=1)
    fig.update_layout(height=1350, barmode="group", title=(
        "Factor combo historical review | 5m close decision, next completed bar | "
        "2023 frozen direction | log-return, signed long-short, per-asset equal weight"))
    fig.write_html(out / "dashboard.html", include_plotlyjs=True)


def run_factor_combo(cfg: dict, config_path: str) -> dict:
    required = {"version", "data_root", "segments", "training", "factor_sources",
                "out_dir", "warmup_days", "similarity_threshold", "cost_bps",
                "dynamic_combo", "signal_engine"}
    if set(cfg) - required - {"mode"} or required - set(cfg):
        raise ValueError(f"factor_combo unknown/missing keys: {sorted(set(cfg)-required-{'mode'})} / {sorted(required-set(cfg))}")
    if cfg["mode"] != "factor_combo" or not 0 < cfg["similarity_threshold"] < 1:
        raise ValueError("invalid factor_combo mode/similarity_threshold")
    if cfg["warmup_days"] < 30 or not cfg["cost_bps"] or any(x < 0 for x in cfg["cost_bps"]):
        raise ValueError("warmup_days >= 30 and nonnegative cost_bps are required")
    from .factor_combo_dynamic import DynamicSpec
    from .factor_combo_signal_engine import SignalEngineSpec
    dynamic_spec = DynamicSpec.from_dict(cfg["dynamic_combo"])
    signal_spec = SignalEngineSpec.from_dict(cfg["signal_engine"])
    names = [s["name"] for s in cfg["segments"]]
    if len(names) != len(set(names)) or cfg["training"]["segment"] not in names:
        raise ValueError("training segment must be unique and declared")
    if names[0] != cfg["training"]["segment"]:
        raise ValueError("training segment must precede review segments")
    intervals = [(pd.Timestamp(s["start"], tz="UTC"), pd.Timestamp(s["end"], tz="UTC"))
                 for s in cfg["segments"]]
    if any(a >= b for a, b in intervals) or any(intervals[i][1] >= intervals[i+1][0]
                                            for i in range(len(intervals)-1)):
        raise ValueError("segments must be ordered, nonoverlapping positive intervals")
    training_end = pd.Timestamp(cfg["training"]["end"], tz="UTC")
    if not intervals[0][0] < training_end < intervals[0][1]:
        raise ValueError("training.end must lie inside the first segment")
    out = Path(cfg["out_dir"])
    if out.exists():
        raise FileExistsError(f"combo output exists: {out}; use a new directory")
    if cfg["data_root"] != "simulated" and not Path(cfg["data_root"]).is_dir():
        raise FileNotFoundError(cfg["data_root"])
    registry = _registry(cfg)
    segments = [_segment(cfg, s, registry) for s in cfg["segments"]]
    if len({tuple(s["codes"]) for s in segments}) != 1:
        raise ValueError("asset universe differs across segments")
    end = cfg["training"]["end"].replace("-", "").replace(" ", "").replace(":", "")[:12]
    beta = _targets(segments, cfg["training"]["segment"], end)
    dates = np.concatenate([s["dates"] for s in segments])
    period = np.concatenate([np.repeat(s["segment"], len(s["dates"])) for s in segments]).astype(object)
    y = np.concatenate([s["y"] for s in segments]); raw_y = np.concatenate([s["raw_y"] for s in segments])
    raw = np.concatenate([s["factor"] for s in segments])
    style = np.concatenate([s["style"] for s in segments])
    risk = np.concatenate([s["risk"] for s in segments])
    # At t the label is the next completed bar: purge the boundary row.
    return_cutoff = (pd.to_datetime(end, format="%Y%m%d%H%M", utc=True)
                     - pd.Timedelta(minutes=5)).strftime("%Y%m%d%H%M")
    train = (period == cfg["training"]["segment"]) & (dates <= return_cutoff)
    train_panel = np.broadcast_to(train[:, None], y.shape) & np.isfinite(y)
    raw, registry, aliases = deduplicate_factors(raw, registry, train_panel,
                                                  cfg["similarity_threshold"])
    raw, raw_scaler = _robust_fit(raw, train_panel)
    family, names = _family_values(raw, registry)
    style, style_scaler = _robust_fit(style, train_panel)
    state = np.stack([_finite_median(risk)[:, None].repeat(y.shape[1], axis=1),
                      np.concatenate([s["style"][:, :, STYLE.index("rv_12")] for s in segments]),
                      np.concatenate([s["style"][:, :, STYLE.index("amihud_shock_12")] for s in segments])], axis=-1)
    state, state_scaler = _robust_fit(state, train_panel)
    predictions, coefficients = _predict_models(style, raw, family, state, y, train_panel, names)
    review_period = period.copy()
    training_segment = cfg["training"]["segment"]
    review_period[(period == training_segment) & (dates <= end)] = f"{training_segment}_train"
    review_period[(period == training_segment) & (dates > end)] = f"{training_segment}_review"
    comparison = _metrics(y, predictions["B0_style"], predictions, review_period, dates)
    target_table, curves = _slow_targets(predictions["B5_explicit_gate"], raw_y,
                                          dates, review_period, train_panel, cfg["cost_bps"])
    risk_table, risk_forecasts = _risk_comparison(segments, cfg["training"]["segment"], end)
    similarity, exposure, decay = _diagnostics(segments, predictions, review_period,
                                                train_panel, family, names)
    out.mkdir(parents=True)
    registry.to_csv(out / "factor_registry.csv", index=False)
    aliases.to_csv(out / "alias_audit.csv", index=False)
    (out / "factor_lineage.json").write_text(json.dumps({
        "selected_factor_contract": "completed mining manifest plus frozen selected_factors order",
        "direction_source": "upstream discovery only",
        "utility_label_source": "upstream entry_lag and horizon, matured before online update",
        "sources": [{"path": s["path"],
                     "selected_sha256": _fingerprint(s["path"]),
                     "manifest_sha256": _fingerprint(Path(s["path"]).with_name("manifest.json")),
                     "clock": s["clock"], "role": s["role"], "limit": s["limit"]}
                    for s in cfg["factor_sources"]],
        "retained_expr_hashes": registry.expr_hash.tolist(),
        "removed_aliases": aliases.to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    comparison.to_csv(out / "forecast_comparison.csv", index=False)
    risk_table.to_csv(out / "common_risk_comparison.csv", index=False)
    risk_forecasts.to_parquet(out / "common_risk_forecasts_hourly.parquet", index=False)
    similarity.to_csv(out / "family_similarity.csv", index=False)
    exposure.to_csv(out / "exposure_diagnostics.csv", index=False)
    decay.to_csv(out / "forecast_decay.csv", index=False)
    target_table.to_csv(out / "explicit_gate_target_scenarios.csv", index=False)
    curves.to_csv(out / "explicit_gate_target_curves.csv", index=False)
    q_5m = np.full(len(dates), np.nan)
    offset = 0
    for s in segments:
        hourly = risk_forecasts.loc[risk_forecasts.segment == s["segment"]]
        if not hourly.empty:
            location = np.searchsorted(hourly.date.to_numpy(), s["dates"], side="right") - 1
            valid = location >= 0
            chunk = q_5m[offset:offset+len(s["dates"])]
            chunk[valid] = hourly.q_with_risk_expr.to_numpy()[location[valid]]
        offset += len(s["dates"])
    pd.DataFrame({"date": np.repeat(dates, y.shape[1]),
                  "period": np.repeat(review_period, y.shape[1]),
                  "asset": np.tile(segments[0]["codes"], len(dates)),
                  "residual_y": y.ravel(), "raw_y": raw_y.ravel(),
                  "market_y": np.concatenate([s["market_y"] for s in segments]).ravel(),
                  "common_risk_forecast_1h": np.repeat(q_5m, y.shape[1]),
                  **{k: v.ravel() for k, v in predictions.items()}}).to_parquet(
                      out / "predictions_5m.parquet", index=False)
    (out / "model_metadata.json").write_text(json.dumps({
        "version": VERSION, "beta_by_asset": dict(zip(segments[0]["codes"], beta.tolist())),
        "family_names": names, "scalers": {"raw": raw_scaler, "style": style_scaler,
                                             "state": state_scaler},
        "coefficients": coefficients, "label": "next completed 5m log return minus frozen leave-one-out market beta",
        "alignment": "hourly open timestamp + 60m; native5 timestamp is completed-bar availability",
        "direction_source": "2023 selected_factors.csv discovery sign; no recomputation",
        "historical_review": "2024/2025 were previously inspected; not blind confirmation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    from .factor_combo_dynamic import build_dynamic_portfolios
    from .factor_combo_reporting import build_dynamic_reports
    dynamic = build_dynamic_portfolios(
        raw, y, raw_y, style, predictions, q_5m, dates,
        [len(s["dates"]) for s in segments], end, train_panel, registry,
        dynamic_spec, signal_spec)
    dynamic_report = build_dynamic_reports(
        out, dynamic, raw, registry, segments, dates, review_period,
        dynamic_spec, signal_spec, comparison, risk_table)
    manifest = {"status": "COMPLETED", "version": VERSION,
                "config": str(config_path), "config_sha256": _fingerprint(config_path),
                "source_sha256": {
                    "factor_combo": _fingerprint(__file__),
                    "factor_combo_dynamic": _fingerprint(Path(__file__).with_name("factor_combo_dynamic.py")),
                    "factor_combo_reporting": _fingerprint(Path(__file__).with_name("factor_combo_reporting.py")),
                    "factor_combo_signal_engine": _fingerprint(Path(__file__).with_name("factor_combo_signal_engine.py")),
                },
                "factor_sources": {s["path"]: {
                    "selected_sha256": _fingerprint(s["path"]),
                    "manifest_sha256": _fingerprint(Path(s["path"]).with_name("manifest.json"))}
                    for s in cfg["factor_sources"]},
                "summary": {"factor_count": len(registry), "families": len(names),
                            "periods": list(dict.fromkeys(review_period)),
                            "comparison_file": str(out / "forecast_comparison.csv"),
                            "dashboard": str(out / "index.html"),
                            "dynamic_reporting": dynamic_report}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
