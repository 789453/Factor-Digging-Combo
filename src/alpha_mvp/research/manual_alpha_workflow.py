"""Semi-structured manual hypotheses inside the aligned-crypto CLI workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from .aligned_contract import Split, align_hourly_to_5m, completed_hour, discovery_betas, exact_pnl, labels, return_targets
from .aligned_workflow import _real_data, _simulated
from .manual_alpha_derivatives import DERIVATIVE_FIELD_VERSION, load_derivative_panels
from .manual_alpha_factors import FeatureContext, MECHANISMS, MECHANISM_VERSION, evaluate_mechanism
from .manual_alpha_quarter_hour import QUARTER_FIELD_VERSION, QUARTER_FIELD_DEPENDENCIES, load_quarter_hour_panels

BASE_FIELDS = {"btc_ret_4h", "discovery_beta", "market_ret_1h", "market_vol_24h"}
DERIVATIVE_FIELDS = {"derivative_basis", "derivative_trade_mark_gap", "derivative_funding_state",
                     "derivative_funding_age", "derivative_funding_event",
                     "derivative_funding_interval_missing"}
QUARTER_FIELDS = set(QUARTER_FIELD_DEPENDENCIES)
WORKFLOW_VERSION = "manual-alpha-aligned-20261003-v3-quarter-phase"


def validate_manual_config(cfg: dict) -> None:
    if set(cfg) != {"mode", "version", "data", "split", "manual_alpha", "output"} or cfg["mode"] != "aligned_crypto":
        raise ValueError("manual alpha requires mode/version/data/split/manual_alpha/output")
    if set(cfg["data"]) != {"source", "root", "derivative_root", "start", "end", "feature_cache"}:
        raise ValueError("invalid manual-alpha data declaration")
    if cfg["data"]["source"] not in {"crypto_parquet", "simulated"}:
        raise ValueError("manual-alpha source must be crypto_parquet or simulated")
    if set(cfg["split"]) != {"discovery_end", "validation_start", "validation_end", "holdout_start"}:
        raise ValueError("invalid manual-alpha split declaration")
    m = cfg["manual_alpha"]
    expected = {"seed", "trials_per_mechanism", "cost_bps", "gross_budget", "calibration_end",
                "discovery_folds", "max_selected", "mechanisms"}
    optional = {"activation_levels", "direction_policy", "funding_cashflow",
                "require_semantic_direction", "min_validation_net_bps_per_hour", "min_positive_folds"}
    if not expected.issubset(m) or not set(m).issubset(expected | optional) or not isinstance(m["seed"], int) or not 1 <= m["trials_per_mechanism"] <= 200:
        raise ValueError("invalid manual-alpha search declaration")
    if m.get("direction_policy", "initial_fixed") not in {"initial_fixed", "expanding_fold"}:
        raise ValueError("unknown manual-alpha direction policy")
    if not isinstance(m.get("funding_cashflow", False), bool) or not isinstance(m.get("require_semantic_direction", False), bool):
        raise ValueError("manual-alpha funding/semantic switches must be boolean")
    if not isinstance(m.get("min_validation_net_bps_per_hour", 0.), (int, float)) or m.get("min_validation_net_bps_per_hour", 0.) < 0:
        raise ValueError("manual-alpha validation net floor is invalid")
    if not isinstance(m.get("min_positive_folds", 0), int) or not 0 <= m.get("min_positive_folds", 0) <= len(m["discovery_folds"]):
        raise ValueError("manual-alpha fold breadth floor is invalid")
    if "activation_levels" in m and (not m["activation_levels"] or
            any(not isinstance(x, (int, float)) or not 0 <= x < 1 for x in m["activation_levels"])):
        raise ValueError("manual-alpha activation levels outside budget")
    if not 0 < m["gross_budget"] <= 1 or m["cost_bps"] < 0 or not 1 <= m["max_selected"] <= 20:
        raise ValueError("invalid manual-alpha position or cost budget")
    split = Split(**cfg["split"])
    if not (cfg["data"]["start"][:4] == "2023" and m["calibration_end"] < split.discovery_end
            < split.validation_start <= split.validation_end < split.holdout_start):
        raise ValueError("manual-alpha chronological split is invalid")
    if not m["discovery_folds"] or not all(len(x) == 2 for x in m["discovery_folds"]):
        raise ValueError("manual-alpha needs bounded discovery folds")
    earlier = m["calibration_end"]
    for start, end in m["discovery_folds"]:
        if not earlier < start <= end <= split.discovery_end:
            raise ValueError("manual-alpha folds overlap or cross discovery firewall")
        earlier = end
    mechanisms = m["mechanisms"]
    if not mechanisms or (cfg["data"]["source"] == "crypto_parquet" and len(mechanisms) != 20):
        raise ValueError("real manual-alpha research declares exactly 20 mechanisms")
    ids = []
    for item in mechanisms:
        if set(item) != {"id", "hypothesis", "failure", "dependencies", "windows", "thresholds", "horizons_hours", "complexity_nodes"}:
            raise ValueError("manual-alpha mechanism card is incomplete")
        if item["id"] not in MECHANISMS or not item["hypothesis"] or not item["failure"]:
            raise ValueError(f"invalid manual-alpha mechanism: {item.get('id')}")
        if not item["dependencies"] or not all(isinstance(x, str) for x in item["dependencies"]):
            raise ValueError("manual-alpha dependencies must be declared")
        if not item["windows"] or any(not isinstance(x, int) or not 2 <= x <= 168 for x in item["windows"]):
            raise ValueError("manual-alpha window outside budget")
        if not item["thresholds"] or any(not isinstance(x, (int, float)) or x < 0 or x > 10 for x in item["thresholds"]):
            raise ValueError("manual-alpha threshold outside budget")
        if not item["horizons_hours"] or any(x not in (1, 4, 12, 24, 48) for x in item["horizons_hours"]):
            raise ValueError("manual-alpha holding horizon outside budget")
        if not 1 <= item["complexity_nodes"] <= 12:
            raise ValueError("manual-alpha complexity budget exceeded")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("manual-alpha mechanisms must be unique")
    if set(cfg["output"]) != {"out_dir"}:
        raise ValueError("manual-alpha output requires out_dir")


def _fingerprint(path: Path) -> dict:
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def _prepared_data(cfg: dict):
    m = cfg["manual_alpha"]
    requested = {"close", "ret_1h", "ret_4h", "realized_vol_24h"}
    for card in m["mechanisms"]:
        requested.update(card["dependencies"])
    existing = sorted(requested - BASE_FIELDS - DERIVATIVE_FIELDS - QUARTER_FIELDS)
    data_cfg = {"data": {k: v for k, v in cfg["data"].items() if k != "derivative_root"},
                "search": {"seed": m["seed"]}}
    if cfg["data"]["source"] == "crypto_parquet":
        panels, raw_dates, codes, fast_dates, fast_close, source_files = _real_data(data_cfg, existing)
        derivatives, derivative_files = load_derivative_panels(cfg["data"]["derivative_root"],
            codes, raw_dates, panels["close"])
        quarter = (load_quarter_hour_panels(cfg["data"]["root"], codes, raw_dates)
                   if requested & QUARTER_FIELDS else {})
    else:
        all_fields = sorted(set(existing) | (requested & DERIVATIVE_FIELDS) | (requested & QUARTER_FIELDS))
        panels, raw_dates, codes, fast_dates, fast_close, source_files = _simulated(data_cfg, all_fields)
        derivatives = {name: panels[name] for name in requested & DERIVATIVE_FIELDS}
        quarter = {name: panels[name] for name in requested & QUARTER_FIELDS}
        if m.get("funding_cashflow", False):
            stamp = pd.to_datetime(raw_dates, format="%Y%m%d%H%M", utc=True) + pd.Timedelta(hours=1)
            rng = np.random.default_rng(m["seed"] + 7)
            if "derivative_funding_state" in derivatives:
                derivatives["derivative_funding_state"] = rng.normal(0, 1e-4,
                    size=panels["close"].shape).astype(np.float32)
            if "derivative_basis" in derivatives:
                derivatives["derivative_basis"] = rng.normal(0, .001,
                    size=panels["close"].shape).astype(np.float32)
            derivatives["derivative_funding_event"] = np.where(
                (stamp.hour.to_numpy() % 8 == 0)[:, None],
                rng.normal(0, 1e-4, size=panels["close"].shape), np.nan).astype(np.float32)
        derivative_files = [{"simulated": True}]
    panels.update(derivatives)
    panels.update(quarter)
    return panels, raw_dates, codes, fast_dates, fast_close, source_files + derivative_files


def _make_context(panels: dict[str, np.ndarray], codes: list[str], betas: np.ndarray) -> FeatureContext:
    ntime, nasset = panels["close"].shape
    if "BTCUSDT" in codes:
        btc = panels["ret_4h"][:, codes.index("BTCUSDT")]
    else:
        btc = np.full(ntime, np.nan)  # Missing BTC is explicit, never a hidden market proxy.
    market = np.nanmean(panels["ret_1h"], axis=1)
    market_vol = pd.Series(market).rolling(24, min_periods=12).std(ddof=0).to_numpy()
    synthetic = {
        "btc_ret_4h": np.repeat(btc[:, None], nasset, axis=1),
        "discovery_beta": np.repeat(betas[None, :], ntime, axis=0),
        "market_ret_1h": np.repeat(market[:, None], nasset, axis=1),
        "market_vol_24h": np.repeat(market_vol[:, None], nasset, axis=1),
    }
    return FeatureContext({**panels, **synthetic})


def _normalize_signal(values: np.ndarray, calibration: np.ndarray,
                      beta: np.ndarray, budget: float,
                      activation: float = 0.) -> tuple[np.ndarray, dict]:
    """Fit per-coin scale on initial discovery, then remove cash and beta."""
    ntime, nasset = values.shape
    center = np.zeros(nasset)
    scale = np.ones(nasset)
    fitted = np.zeros(nasset, dtype=bool)
    for j in range(nasset):
        sample = values[calibration, j]
        sample = sample[np.isfinite(sample) & (sample != 0)]
        if len(sample) < 96:
            continue
        center[j] = np.median(sample)
        q25, q75 = np.quantile(sample, [.25, .75])
        s = (q75 - q25) / 1.349
        if s <= 1e-9:
            s = np.std(sample)
        if np.isfinite(s) and s > 1e-9:
            scale[j] = s
            fitted[j] = True
    active = np.isfinite(values) & fitted[None, :]
    raw = np.where(active, np.tanh(np.clip((values - center) / scale, -8, 8)), 0)
    # Gated zero means no primary view, although it may still serve as a hedge.
    raw[np.isfinite(values) & (values == 0)] = 0
    raw[np.abs(raw) < activation] = 0
    count = active.sum(axis=1, keepdims=True)
    mean = np.divide((raw * active).sum(axis=1, keepdims=True), count,
                     out=np.zeros((ntime, 1)), where=count > 0)
    bmean = np.divide((active * beta).sum(axis=1, keepdims=True), count,
                      out=np.zeros((ntime, 1)), where=count > 0)
    bcenter = beta[None, :] - bmean
    deviation = (raw - mean) * active
    denominator = (active * bcenter * bcenter).sum(axis=1, keepdims=True)
    coefficient = np.divide((deviation * bcenter).sum(axis=1, keepdims=True), denominator,
                            out=np.zeros((ntime, 1)), where=denominator > 1e-10)
    neutral = np.where(active, deviation - coefficient * bcenter, 0)
    neutral[count[:, 0] < 3] = 0
    gross = np.mean(np.abs(neutral), axis=1, keepdims=True)
    # Cap, never inflate a vanishing score to a full budget.
    scale_budget = np.minimum(1, budget / np.maximum(gross, 1e-10))
    proposed = (neutral * scale_budget).astype(np.float32)
    return proposed, {"center": center.tolist(), "scale": scale.tolist(),
                      "fitted_assets": int(fitted.sum()),
                      "signal_coverage": float(np.mean(active))}


def _positions(proposed: np.ndarray, dates: np.ndarray, horizon_hours: int,
               direction: int | np.ndarray) -> np.ndarray:
    stamp = pd.to_datetime(dates, format="%Y%m%d%H%M", utc=True).astype("datetime64[ns, UTC]")
    decision = (stamp.asi8 // 3_600_000_000_000) % horizon_hours == 0
    index = np.maximum.accumulate(np.where(decision, np.arange(len(dates)), -1))
    result = np.zeros_like(proposed)
    good = index >= 0
    if np.ndim(direction) == 0:
        result[good] = proposed[index[good]] * direction
    else:
        result[good] = proposed[index[good]] * np.asarray(direction)[index[good], None]
    return result


def _ledger(position: np.ndarray, raw_hour: np.ndarray, cost_bps: float,
            reset: np.ndarray, funding_event: np.ndarray | None = None) -> dict[str, np.ndarray]:
    previous = np.vstack([np.zeros((1, position.shape[1])), position[:-1]])
    previous[reset] = 0
    change = position - previous
    gross = np.where(np.isfinite(raw_hour), position * raw_hour, 0)
    fee = np.abs(change) * cost_bps / 1e4
    funding = (np.where(np.isfinite(funding_event), -previous * funding_event, 0)
               if funding_event is not None else np.zeros_like(gross))
    return {"gross": gross, "funding": funding, "fee": fee,
            "net": gross + funding - fee,
            "turnover": np.abs(change)}


def _metric(ledger: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"gross_bps_per_hour": np.nan, "fee_bps_per_hour": np.nan,
                "net_bps_per_hour": np.nan, "turnover_per_hour": np.nan,
                "positive_asset_share": np.nan, "net_total_pct": np.nan}
    gross = ledger["gross"][mask]
    fee = ledger["fee"][mask]
    funding = ledger["funding"][mask]
    net = ledger["net"][mask]
    turnover = ledger["turnover"][mask]
    per_asset = np.mean(net, axis=0)
    return {"gross_bps_per_hour": float(np.mean(gross) * 1e4),
            "funding_bps_per_hour": float(np.mean(funding) * 1e4),
            "fee_bps_per_hour": float(np.mean(fee) * 1e4),
            "net_bps_per_hour": float(np.mean(net) * 1e4),
            "turnover_per_hour": float(np.mean(turnover)),
            "positive_asset_share": float(np.mean(per_asset > 0)),
            "net_total_pct": float(np.sum(np.mean(net, axis=1)) * 100)}


def _direction(proposed: np.ndarray, residual_label: np.ndarray,
               calibration: np.ndarray) -> tuple[int, float]:
    good = calibration[:, None] & np.isfinite(residual_label)
    payoff = float(np.mean((proposed * np.nan_to_num(residual_label))[good])) if good.sum() else 0
    if not np.isfinite(payoff) or abs(payoff) < 1e-12:
        raise ValueError("initial discovery has no signed residual information")
    return (1 if payoff > 0 else -1), payoff * 1e4


def _direction_path(proposed: np.ndarray, residual: np.ndarray,
                    calibration: np.ndarray, folds: list[np.ndarray],
                    policy: str) -> tuple[np.ndarray, int, list[int], float]:
    initial, edge = _direction(proposed, residual, calibration)
    path = np.full(len(proposed), initial, dtype=np.int8)
    if policy == "initial_fixed":
        return path, initial, [initial] * len(folds), edge
    if policy != "expanding_fold":
        raise ValueError("unknown direction policy")
    prior = calibration.copy()
    fold_directions = []
    for fold in folds:
        start = np.flatnonzero(fold)[0]
        direction, _ = _direction(proposed, residual, prior)
        path[start:] = direction
        fold_directions.append(direction)
        prior |= fold
    final, _ = _direction(proposed, residual, prior)
    path[np.flatnonzero(folds[-1])[-1] + 1:] = final
    return path, final, fold_directions, edge


def _evaluate(card: dict, params: dict, context: FeatureContext, dates: np.ndarray,
              beta: np.ndarray, calibration: np.ndarray, folds: list[np.ndarray],
              raw_hour: np.ndarray, residual: np.ndarray, reset: np.ndarray,
              cost_bps: float, budget: float, direction_policy: str = "initial_fixed",
              funding_event: np.ndarray | None = None) -> tuple[dict, np.ndarray, dict]:
    values = evaluate_mechanism(card["id"], context, params["window"], params["threshold"])
    proposed, scale = _normalize_signal(values, calibration, beta, budget,
                                        params.get("activation", 0.))
    if scale["fitted_assets"] < 6 or scale["signal_coverage"] < .2:
        raise ValueError("mechanism has insufficient calibrated asset coverage")
    direction_path, direction, fold_directions, initial_edge = _direction_path(
        proposed, residual, calibration, folds, direction_policy)
    position = _positions(proposed, dates, params["horizon_hours"], direction_path)
    ledger = _ledger(position, raw_hour, cost_bps, reset, funding_event)
    fold_metrics = [_metric(ledger, mask) for mask in folds]
    returns = np.array([x["net_bps_per_hour"] for x in fold_metrics], dtype=float)
    objective = float(np.median(returns) + .25 * np.mean(returns) - .35 * np.std(returns))
    result = {"direction": direction, "fold_directions": fold_directions,
              "direction_policy": direction_policy,
              "initial_residual_edge_bps": initial_edge,
              "objective": objective, "fold_net_bps_per_hour": returns.tolist(),
              "fold_positive_asset_share": [x["positive_asset_share"] for x in fold_metrics],
              "discovery": _metric(ledger, calibration | np.logical_or.reduce(folds)),
              "scale": scale}
    return result, position, ledger


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _five_minute_ledger(hourly_position: np.ndarray, dates: np.ndarray,
                        fast_dates: np.ndarray, fast_close: np.ndarray,
                        fast_period: np.ndarray, cost_bps: float,
                        funding_event: np.ndarray | None) -> tuple[pd.DataFrame, dict]:
    """Broadcast frozen hourly holdings, account at native completed 5m bars."""
    anchor = np.searchsorted(dates, fast_dates, side="right") - 1
    fast_position = np.zeros_like(fast_close, dtype=np.float32)
    available = anchor >= 0
    fast_position[available] = hourly_position[anchor[available]]
    log_price = np.log(np.where(fast_close > 0, fast_close, np.nan))
    next_bar = np.diff(log_price, axis=0, append=np.full((1, log_price.shape[1]), np.nan))
    next_bar[fast_period != np.r_[fast_period[1:], fast_period[-1]]] = np.nan
    boundary = np.r_[True, fast_period[1:] != fast_period[:-1]]
    price = exact_pnl(fast_position, next_bar, cost_bps, boundary)
    funding = np.zeros_like(price["gross"])
    if funding_event is not None:
        hour_fast_idx = align_hourly_to_5m(dates, fast_dates)
        previous = np.vstack((np.zeros((1, fast_position.shape[1])), fast_position[:-1]))
        previous[boundary] = 0
        funding[hour_fast_idx] = np.where(np.isfinite(funding_event),
                                          -previous[hour_fast_idx] * funding_event, 0)
    net = price["gross"] + funding - price["fee"]
    bridge = float(np.max(np.abs(net - (price["gross"] + funding - price["fee"]))))
    frame = pd.DataFrame({"completed_5m": fast_dates, "period": fast_period,
        "price_gross": np.mean(price["gross"], axis=1),
        "funding": np.mean(funding, axis=1),
        "fee": np.mean(price["fee"], axis=1),
        "net": np.mean(net, axis=1),
        "mean_abs_position": np.mean(np.abs(fast_position), axis=1),
        "turnover": np.mean(np.abs(price["change"]), axis=1)})
    return frame, {"bridge_max_error": bridge,
        "price_gross_total_pct": float(np.sum(frame.price_gross) * 100),
        "funding_total_pct": float(np.sum(frame.funding) * 100),
        "fee_total_pct": float(np.sum(frame.fee) * 100),
        "net_total_pct": float(np.sum(frame.net) * 100)}


def run_manual_alpha(cfg: dict, config_path: str) -> dict:
    validate_manual_config(cfg)
    m = cfg["manual_alpha"]
    out = Path(cfg["output"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"completed research evidence exists: {out}")
    if cfg["data"]["source"] == "crypto_parquet":
        root = Path(cfg["data"]["root"])
        derivative_root = Path(cfg["data"]["derivative_root"])
        source_paths = sorted(p for p in root.glob("*/*.parquet") if p.name in {"1h.parquet", "15m.parquet", "5m.parquet"})
        source_paths += sorted(derivative_root.glob("*.parquet"))
        if not source_paths:
            raise FileNotFoundError("manual-alpha source parquet files unavailable")
        source_signature = json.dumps([_fingerprint(p) for p in source_paths], sort_keys=True)
    else:
        source_signature = "simulated"
    dependency_hashes = "".join(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("manual_alpha_factors.py", "manual_alpha_derivatives.py",
                     "manual_alpha_quarter_hour.py", "aligned_contract.py"))
    signature = hashlib.sha256((yaml.safe_dump(cfg, sort_keys=True) + WORKFLOW_VERSION +
        MECHANISM_VERSION + DERIVATIVE_FIELD_VERSION + QUARTER_FIELD_VERSION +
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() +
        dependency_hashes + source_signature).encode()).hexdigest()
    state_path = out / "checkpoint_signature.json"
    if state_path.exists() and json.loads(state_path.read_text())["signature"] != signature:
        raise ValueError("manual-alpha checkpoint signature differs; use a new output directory")
    if not state_path.exists():
        _write_json(state_path, {"signature": signature, "config_path": config_path})
        (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    panels, raw_dates, codes, fast_dates, fast_close, files = _prepared_data(cfg)
    dates = completed_hour(raw_dates)
    split = Split(**cfg["split"])
    masks = split.masks(dates)
    calibration = masks["discovery"] & (dates <= m["calibration_end"])
    folds = [(dates >= start) & (dates <= end) & masks["discovery"]
             for start, end in m["discovery_folds"]]
    if calibration.sum() < 720 or any(x.sum() < 168 for x in folds):
        raise ValueError("manual-alpha calibration or discovery folds too short")
    beta = discovery_betas(np.log(panels["close"]), calibration)
    if any(card["id"] == "btc_lead_catchup" for card in m["mechanisms"]) and "BTCUSDT" not in codes:
        raise ValueError("btc_lead_catchup requires BTCUSDT in the configured universe")
    context = _make_context(panels, codes, beta)
    idx = align_hourly_to_5m(dates, fast_dates)
    fast_period = np.where(fast_dates <= split.discovery_end, 0,
        np.where(fast_dates < split.validation_start, -1,
        np.where(fast_dates <= split.validation_end, 1,
        np.where(fast_dates < split.holdout_start, -1, 2))))
    log_fast = np.log(np.where(fast_close > 0, fast_close, np.nan))
    raw_hour = return_targets(log_fast, idx, (12,), fast_period)[12]
    residual = labels(raw_hour, beta)["relative"]
    funding_event = panels.get("derivative_funding_event") if m.get("funding_cashflow", False) else None
    if m.get("funding_cashflow", False):
        if funding_event is None:
            raise ValueError("funding cashflow requested but settlement events are absent")
        next_rate = np.vstack((funding_event[1:], np.full((1, len(codes)), np.nan)))
        residual = np.where(np.isfinite(residual), residual - np.nan_to_num(next_rate, nan=0), np.nan)
    reset = np.zeros(len(dates), dtype=bool)
    for key in ("validation", "holdout"):
        ix = np.flatnonzero(masks[key])
        if len(ix):
            reset[ix[0]] = True
    # All source coverage is measured without inspecting holdout outcomes.
    coverage = {name: {period: float(np.isfinite(x[mask]).mean()) for period, mask in masks.items()}
                for name, x in panels.items() if name in DERIVATIVE_FIELDS | QUARTER_FIELDS}
    interval_missing = panels.get("derivative_funding_interval_missing")
    interval_audit = ({period: int(np.sum(interval_missing[mask] == 1)) for period, mask in masks.items()}
                      if interval_missing is not None else {})
    _write_json(out / "data_audit.json", {"sources": files, "derivative_field_version": DERIVATIVE_FIELD_VERSION,
        "quarter_field_version": QUARTER_FIELD_VERSION,
        "coverage": coverage, "funding_interval_metadata_missing_events": interval_audit,
        "codes": codes, "calibration_beta": beta.tolist()})
    storage = f"sqlite:///{(out / 'optuna_studies.sqlite3').resolve().as_posix()}"
    results = []
    trial_rows = []
    for number, card in enumerate(m["mechanisms"]):
        name = card["id"]
        study = optuna.create_study(study_name=f"{signature[:10]}_{name}", storage=storage,
            load_if_exists=True, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=m["seed"] + number, multivariate=True))
        missing = m["trials_per_mechanism"] - len(study.trials)
        if missing > 0:
            def objective(trial):
                params = {"window": trial.suggest_categorical("window", card["windows"]),
                          "threshold": trial.suggest_categorical("threshold", card["thresholds"]),
                          "horizon_hours": trial.suggest_categorical("horizon_hours", card["horizons_hours"])}
                if "activation_levels" in m:
                    params["activation"] = trial.suggest_categorical("activation", m["activation_levels"])
                try:
                    measured, _, _ = _evaluate(card, params, context, dates, beta, calibration, folds,
                        raw_hour, residual, reset, m["cost_bps"], m["gross_budget"],
                        m.get("direction_policy", "initial_fixed"), funding_event)
                except (ValueError, FloatingPointError) as exc:
                    trial.set_user_attr("invalid_reason", str(exc))
                    return -1e6
                trial.set_user_attr("fold_net_bps_per_hour", measured["fold_net_bps_per_hour"])
                trial.set_user_attr("direction", measured["direction"])
                trial.set_user_attr("fold_directions", measured["fold_directions"])
                trial.set_user_attr("initial_residual_edge_bps", measured["initial_residual_edge_bps"])
                return measured["objective"]
            study.optimize(objective, n_trials=missing, n_jobs=1, show_progress_bar=False)
        for trial in study.trials:
            trial_rows.append({"mechanism": name, "trial": trial.number, "state": str(trial.state),
                "objective": trial.value, **trial.params, **trial.user_attrs})
        valid_trials = [x for x in study.trials if x.value is not None and x.value > -1e5]
        if not valid_trials:
            results.append({"mechanism": name, "status": "NO_VALID_TRIAL", "hypothesis": card["hypothesis"]})
            print(f"[manual-alpha] {number+1}/{len(m['mechanisms'])} {name}: no valid trial", flush=True)
        else:
            trial = max(valid_trials, key=lambda x: (x.value, -x.number))
            measured, position, ledger = _evaluate(card, trial.params, context, dates, beta,
                calibration, folds, raw_hour, residual, reset, m["cost_bps"], m["gross_budget"],
                m.get("direction_policy", "initial_fixed"), funding_event)
            validation = _metric(ledger, masks["validation"])
            discovery = measured["discovery"]
            results.append({"mechanism": name, "status": "EVALUATED", "hypothesis": card["hypothesis"],
                "failure": card["failure"], "dependencies": "|".join(card["dependencies"]),
                "complexity_nodes": card["complexity_nodes"], "trial": trial.number,
                **trial.params, "direction": measured["direction"],
                "fold_directions": json.dumps(measured["fold_directions"]),
                "initial_residual_edge_bps": measured["initial_residual_edge_bps"],
                "discovery_objective": measured["objective"],
                "fold_net_bps_per_hour": json.dumps(measured["fold_net_bps_per_hour"]),
                **{f"discovery_{k}": v for k, v in discovery.items()},
                **{f"validation_{k}": v for k, v in validation.items()},
                "signal_coverage": measured["scale"]["signal_coverage"],
                "fitted_assets": measured["scale"]["fitted_assets"]})
            print(f"[manual-alpha] {number+1}/{len(m['mechanisms'])} {name}: "
                  f"discovery {discovery['net_bps_per_hour']:.5f} "
                  f"validation {validation['net_bps_per_hour']:.5f} bps/h", flush=True)
        _write_csv(out / "search_trials.csv", pd.DataFrame(trial_rows))
        _write_csv(out / "mechanism_results.csv", pd.DataFrame(results))
    frame = pd.DataFrame(results)
    frame["positive_discovery_folds"] = frame["fold_net_bps_per_hour"].map(
        lambda x: sum(np.asarray(json.loads(x)) > 0) if isinstance(x, str) else 0)
    _write_csv(out / "mechanism_results.csv", frame)
    eligible = frame[frame.status.eq("EVALUATED")
                     & (frame.validation_net_bps_per_hour > m.get("min_validation_net_bps_per_hour", 0.))
                     & (frame.validation_positive_asset_share >= .5)
                     & (frame.discovery_objective > 0)
                     & (frame.positive_discovery_folds >= m.get("min_positive_folds", 0))
                     & ((frame.direction == 1) | (not m.get("require_semantic_direction", False)))].copy()
    eligible = eligible.sort_values(["validation_net_bps_per_hour", "discovery_objective", "mechanism"],
                                   ascending=[False, False, True]).head(m["max_selected"])
    selected = eligible.mechanism.tolist()
    # Freeze the exact research decision before any holdout return is touched.
    freeze = {"signature": signature, "selected": selected,
              "selection_rule": {"min_validation_net_bps_per_hour": m.get("min_validation_net_bps_per_hour", 0.),
                  "min_positive_assets": .5, "positive_discovery_objective": True,
                  "min_positive_folds": m.get("min_positive_folds", 0),
                  "require_semantic_direction": m.get("require_semantic_direction", False)},
              "rows": eligible.to_dict(orient="records"), "holdout_seen_previously_in_project": True}
    freeze_path = out / "selection_freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text())["selected"] != selected:
        raise ValueError("frozen manual-alpha selection differs from checkpoint")
    if not freeze_path.exists():
        _write_json(freeze_path, freeze)
    holdout_rows = []
    selected_positions = []
    for card in m["mechanisms"]:
        if card["id"] not in selected:
            continue
        row = eligible[eligible.mechanism.eq(card["id"])].iloc[0]
        params = {"window": int(row.window), "threshold": float(row.threshold),
                  "horizon_hours": int(row.horizon_hours)}
        if "activation_levels" in m:
            params["activation"] = float(row.activation)
        measured, position, ledger = _evaluate(card, params, context, dates, beta,
            calibration, folds, raw_hour, residual, reset, m["cost_bps"], m["gross_budget"],
            m.get("direction_policy", "initial_fixed"), funding_event)
        selected_positions.append(position)
        holdout_rows.append({"mechanism": card["id"], **_metric(ledger, masks["holdout"])})
    _write_csv(out / "selected_factors.csv", eligible)
    _write_csv(out / "holdout_audit.csv", pd.DataFrame(holdout_rows))
    if selected_positions:
        combo_position = np.mean(selected_positions, axis=0)
    else:
        combo_position = np.zeros_like(raw_hour)
    combo_ledger = _ledger(combo_position, raw_hour, m["cost_bps"], reset, funding_event)
    combo = {period: _metric(combo_ledger, mask) for period, mask in masks.items()}
    # Save compact hourly ledger, including all completed periods and fee bridge.
    hourly = pd.DataFrame({"decision_time": dates,
        "gross": np.mean(combo_ledger["gross"], axis=1),
        "funding": np.mean(combo_ledger["funding"], axis=1),
        "fee": np.mean(combo_ledger["fee"], axis=1),
        "net": np.mean(combo_ledger["net"], axis=1),
        "mean_abs_position": np.mean(np.abs(combo_position), axis=1),
        "turnover": np.mean(combo_ledger["turnover"], axis=1)})
    hourly.to_parquet(out / "combo_hourly_ledger.parquet", index=False, compression="zstd")
    native_ledger, native_totals = _five_minute_ledger(
        combo_position, dates, fast_dates, fast_close, fast_period,
        m["cost_bps"], funding_event)
    native_ledger.to_parquet(out / "combo_5m_ledger.parquet", index=False, compression="zstd")
    manifest = {"status": "COMPLETED", "mode": "aligned_crypto", "lane": "manual_alpha",
                "version": WORKFLOW_VERSION, "signature": signature, "config_path": config_path,
                "results": {"evaluated": int(frame.status.eq("EVALUATED").sum()),
                            "selected": len(selected), "combo": combo,
                            "combo_5m_totals": native_totals},
                "source_clock": "completed 1h plus completed 15m/5m aggregates",
                "decision_clock": "hour end", "entry_lag": "next completed 5m return",
                "fee_bps_per_unit_turnover": m["cost_bps"],
                "funding_cashflow_included": m.get("funding_cashflow", False),
                "holdout_interpretation": "historical audit; this period was seen by earlier project diagnostics"}
    _write_json(out / "manifest.json", manifest)
    return manifest
