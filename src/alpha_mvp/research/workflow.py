from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import (
    load_crypto_parquet,
    load_from_duckdb,
    load_futures_parquet,
    make_simulated_data,
)
from ..evaluator import BatchEvaluator, make_panels
from ..fields import FIELD_FORMULA_VERSION, add_basic_features
from ..crypto_fields import CRYPTO_FIELD_FORMULA_VERSION, add_crypto_features
from ..native5_fields import NATIVE5_FIELD_FORMULA_VERSION, add_crypto_native_5m_features
from ..ops import OPERATOR_SEMANTICS_VERSION
from .attribution import compute_attribution
from .config import ResearchConfig
from .evaluation import (
    evaluate_factor,
    evaluate_factor_coarse,
    forward_returns,
    purge_cross_split_labels,
    rank_crypto_coarse_results,
    rank_research_results,
)
from .futures_reporting import build_futures_data_profile
from .futures_performance import build_futures_factor_return_report
from .crypto_performance import build_crypto_reports, crypto_search_metrics
from .execution_diagnostics import execution_diagnostic_rows
from .crypto_time_series_performance import (
    build_crypto_pair_report,
    build_crypto_time_series_reports,
    crypto_time_series_search_metrics,
)
from .time_series_evaluation import (
    evaluate_time_series_factor,
    evaluate_time_series_factor_coarse,
    evaluate_time_series_factor_medium,
    rank_time_series_medium_results,
)
from .targets import TARGET_FORMULA_VERSION, forward_micro5_variance
from .proposal_prior import load_proposal_prior
from .incremental_information import (
    BASELINE_FORMULA_VERSION, incremental_variance_information,
    variance_baseline_panels,
)
from .cluster_state import (
    CLUSTER_FIELD_FORMULA_VERSION, CLUSTER_FIELD_SPEC,
    build_cluster_stress_affinity,
)
from .purpose import (
    PURPOSE_FORMULA_VERSION, PURPOSE_BASELINE_FORMULA_VERSION,
    build_purpose_targets, common_factor, evaluate_purpose,
    incremental_purpose_information, liquidity_style_baseline_panels,
    style_baseline_panels,
)
from .selection import (
    build_factor_pairs, select_diverse_factors, select_signal_distinct_factors,
)
from .reporting import build_reports
from .store import ExperimentStore
from .templates import generate_expressions, load_template_families


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _stable_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: str | None) -> dict:
    if not path:
        return {"path": None, "exists": False}
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    stat = file_path.stat()
    return {
        "path": str(file_path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_fingerprint() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = (
        "fields.py",
        "data.py",
        "ops.py",
        "fastops.py",
        "evaluator.py",
        "research/evaluation.py",
        "crypto_fields.py",
        "research/crypto_performance.py",
        "research/crypto_time_series_performance.py",
        "research/time_series_evaluation.py",
        "research/execution_diagnostics.py",
        "research/targets.py",
        "research/proposal_prior.py",
        "research/incremental_information.py",
        "research/cluster_state.py",
        "research/purpose.py",
        "native5_fields.py",
        "research/selection.py",
    )
    return {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in names
    }


def _load_data(config: ResearchConfig) -> pd.DataFrame:
    if config.data.source == "simulated":
        return make_simulated_data(
            n_days=config.data.simulated_days,
            n_stocks=config.data.simulated_stocks,
            start=pd.to_datetime(config.data.start).strftime("%Y-%m-%d"),
            seed=config.search.seed,
        )
    if config.data.source == "futures_parquet":
        return load_futures_parquet(
            config.data.parquet_path,
            config.data.futures_universe,
            config.data.start,
            config.data.end,
        )
    if config.data.source == "crypto_parquet":
        return load_crypto_parquet(
            config.data.crypto_parquet_root,
            config.data.start,
            config.data.end,
            config.data.crypto_primary_timeframe,
            config.data.crypto_micro_timeframe,
            config.data.crypto_fast_timeframe,
        )
    return load_from_duckdb(
        config.data.duckdb_path,
        config.data.pool_json,
        config.data.start,
        config.data.end,
    )


def _record_row(record, template_as_family: bool = False) -> dict:
    return {
        "expr_hash": record.expr_hash,
        "rank_equivalence_hash": record.rank_equivalence_hash,
        "expr": record.expr,
        "canonical": record.canonical,
        "template_name": record.template_name,
        "template_family": record.template_name if template_as_family else record.template_family,
        "template_order": record.template_order,
        "fields": "|".join(record.fields),
        "operators": "|".join(record.operators),
        "windows": "|".join(str(w) for w in record.windows),
        "depth": record.depth,
        "nodes": record.nodes,
        "priority_score": record.priority_score,
    }


def _cache_locality_key(record) -> tuple:
    """Keep related expressions adjacent so rolling sub-panels stay hot."""
    return (
        record.template_family,
        record.template_name,
        tuple(record.fields),
        tuple(record.windows),
        record.template_order,
        record.expr_hash,
    )


def run_research(config: ResearchConfig) -> dict:
    config.validate()
    purpose_track = config.evaluation.research_track != "legacy"
    native_5m = config.data.source == "crypto_parquet" and config.data.crypto_primary_timeframe == "5m"
    out = Path(config.output.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    families, template_raw = load_template_families(config.search.template_config)
    proposal_fingerprint = None
    proposal_prior = {}
    if config.search.proposal_evidence_path:
        evidence_path = Path(config.search.proposal_evidence_path)
        proposal_prior = load_proposal_prior(str(evidence_path))
        proposal_fingerprint = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    families = [
        family for family in families
        if (
            config.evaluation.target_mode in family.target_modes
            or (config.evaluation.target_mode == "return" and not family.target_modes)
        )
    ]
    if not families:
        raise ValueError(f"No templates support target_mode={config.evaluation.target_mode}")
    stable_config = asdict(config)
    if config.data.source == "simulated":
        data_fingerprint = {"source": "simulated", "seed": config.search.seed}
    elif config.data.source == "futures_parquet":
        data_fingerprint = {
            "source": "futures_parquet",
            "parquet": _file_fingerprint(config.data.parquet_path),
            "futures_universe": config.data.futures_universe,
        }
    elif config.data.source == "crypto_parquet":
        crypto_root = Path(config.data.crypto_parquet_root)
        crypto_files = sorted(crypto_root.glob("*/*.parquet"))
        data_fingerprint = {
            "source": "crypto_parquet",
            "crypto_parquet_root": str(crypto_root.resolve()),
            "primary_timeframe": config.data.crypto_primary_timeframe,
            "micro_timeframe": config.data.crypto_micro_timeframe,
            "fast_timeframe": config.data.crypto_fast_timeframe,
            "files": [_file_fingerprint(str(path)) for path in crypto_files],
        }
    else:
        data_fingerprint = {
            "source": "duckdb",
            "database": _file_fingerprint(config.data.duckdb_path),
            "pool": _file_fingerprint(config.data.pool_json),
        }
    config_hash = _stable_hash(stable_config)
    experiment_id = _stable_hash({
        "config": stable_config,
        "templates": template_raw,
        "git_hash": _git_hash(),
        "data_fingerprint": data_fingerprint,
        "proposal_evidence_sha256": proposal_fingerprint,
    })[:16]
    completed_manifest = out / "manifest.json"
    if completed_manifest.exists():
        existing = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            raise FileExistsError(
                f"Completed experiment exists at {out}; use a new output directory"
            )
    store = ExperimentStore(config.output.store_path)
    store.initialize(experiment_id, config_hash)

    is_crypto = config.data.source == "crypto_parquet"
    featured_cache_path = None
    if is_crypto and config.data.crypto_feature_cache_dir:
        cache_key = _stable_hash({
            "data": asdict(config.data),
            "data_fingerprint": data_fingerprint,
            "field_formula_version": (
                NATIVE5_FIELD_FORMULA_VERSION if native_5m else CRYPTO_FIELD_FORMULA_VERSION
            ),
        })[:20]
        cache_dir = Path(config.data.crypto_feature_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        featured_cache_path = cache_dir / f"crypto_features_{cache_key}.parquet"
    if featured_cache_path is not None and featured_cache_path.exists():
        featured = pd.read_parquet(featured_cache_path)
        raw = featured
        print(f"[data] feature cache hit: {featured_cache_path}", flush=True)
    else:
        raw = _load_data(config)
        featured = (
            add_crypto_native_5m_features(raw) if native_5m else
            add_crypto_features(raw) if is_crypto else add_basic_features(raw)
        )
        if featured_cache_path is not None:
            temporary = featured_cache_path.with_suffix(".tmp.parquet")
            featured.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(featured_cache_path)
            print(f"[data] feature cache saved: {featured_cache_path}", flush=True)
    source_report_artifacts = (
        build_futures_data_profile(raw, out)
        if config.data.source == "futures_parquet"
        else {}
    )
    is_crypto_time_series = is_crypto and config.evaluation.factor_mode == "time_series"
    requested = [
        field for field in config.search.fields
        if field not in set(config.search.exclude_fields)
    ]
    structured_fields = (
        {config.cluster_state.field_name} if config.cluster_state.enabled else set()
    )
    missing = sorted(set(requested) - structured_fields - set(featured.columns))
    if missing:
        raise ValueError(f"Configured fields were not built: {missing}")
    variance_target = config.evaluation.target_mode != "return"
    label_columns = (
        [
            "micro5_realized_vol", "micro5_upside_share", "micro5_bar_count",
            "us_day_flag", "us_evening_flag",
        ]
        if variance_target and not purpose_track else []
    )
    missing_labels = sorted(set(label_columns) - set(featured.columns))
    if missing_labels:
        raise ValueError(f"Required variance label columns missing: {missing_labels}")
    cluster_dependencies = (
        list(CLUSTER_FIELD_SPEC.dependencies[:3]) if config.cluster_state.enabled else []
    )
    purpose_baseline_fields = (
        ["micro5_volume_hhi", "micro5_trade_hhi", "micro5_bar_count",
         "volume_shock_24h", "trade_count_shock_24h",
         "us_day_flag", "us_evening_flag"]
        if config.evaluation.research_track == "style_proxy" else
        ["ret_5m", "ret_1h", "rv_12", "volume_shock_12", "amihud_shock_12",
         "us_day_flag", "us_evening_flag"]
        if native_5m else
        ["ret_1h", "ret_24h", "realized_vol_24h", "volume_shock_24h",
         "amihud_shock_24h", "us_day_flag", "us_evening_flag"]
    ) if purpose_track else []
    panels, dates, codes = make_panels(
        featured,
        list(dict.fromkeys([
            *(name for name in requested if name not in structured_fields),
            *label_columns, *cluster_dependencies, *purpose_baseline_fields,
        ])),
        value_col="close",
    )
    cluster_model_artifact = None
    if config.cluster_state.enabled:
        missing_dependencies = sorted(set(cluster_dependencies) - set(panels))
        if missing_dependencies:
            raise ValueError(f"cluster_state dependencies were not built: {missing_dependencies}")
        cluster_values, cluster_metadata = build_cluster_stress_affinity(
            panels, dates, config.split, config.cluster_state
        )
        panels[config.cluster_state.field_name] = cluster_values
        cluster_model_artifact = "cluster_state_model.json"
        (out / cluster_model_artifact).write_text(
            json.dumps(cluster_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    purpose_targets = (
        build_purpose_targets(
            panels["close"], dates, config.split,
            config.evaluation.horizon, config.evaluation.entry_lag,
            style_hhi=np.where(
                panels["micro5_bar_count"] == 12,
                panels["micro5_volume_hhi"], np.nan,
            ) if config.evaluation.research_track == "style_proxy" else None,
        ) if purpose_track else None
    )
    forward = (
        purpose_targets.common if config.evaluation.research_track == "common_risk"
        else purpose_targets.style if config.evaluation.research_track == "style_proxy"
        else purpose_targets.residual
    ) if purpose_track else (
        forward_micro5_variance(
            panels["micro5_realized_vol"], panels["micro5_upside_share"],
            panels["micro5_bar_count"], config.evaluation.horizon,
            config.evaluation.entry_lag, config.evaluation.target_mode,
            config.evaluation.target_scope,
        ) if variance_target else forward_returns(
            panels["close"], config.evaluation.horizon,
            config.evaluation.entry_lag,
        )
    )
    audit_forward = forward
    if not purpose_track:
        forward = purge_cross_split_labels(
            forward, dates, config.split,
            config.evaluation.horizon, config.evaluation.entry_lag,
        )
    variance_baseline = (
        variance_baseline_panels(
            np.where(
                panels["micro5_bar_count"] == 12,
                panels["micro5_realized_vol"], np.nan,
            ),
            panels["micro5_upside_share"], panels["us_day_flag"],
            panels["us_evening_flag"], config.evaluation.target_mode,
            config.evaluation.target_scope,
        ) if variance_target and not purpose_track else None
    )
    robustness_forward = (
        {
            horizon: purge_cross_split_labels(
                forward_returns(
                    panels["close"], horizon, config.evaluation.entry_lag
                ), dates, config.split, horizon, config.evaluation.entry_lag,
            )
            for horizon in config.evaluation.robustness_horizons
        }
        if is_crypto and not variance_target and not purpose_track else {}
    )
    field_formula_version = (
        NATIVE5_FIELD_FORMULA_VERSION if native_5m else
        CRYPTO_FIELD_FORMULA_VERSION if is_crypto else FIELD_FORMULA_VERSION
    )
    evaluation_signature = _stable_hash({
        "data": asdict(config.data),
        "data_fingerprint": data_fingerprint,
        "split": asdict(config.split),
        "evaluation": asdict(config.evaluation),
        "fields": requested,
        "field_formula_version": field_formula_version,
        "operator_semantics_version": OPERATOR_SEMANTICS_VERSION,
        "cluster_state": asdict(config.cluster_state),
        "cluster_field_formula_version": (
            CLUSTER_FIELD_FORMULA_VERSION if config.cluster_state.enabled else None
        ),
        "purpose_formula_version": PURPOSE_FORMULA_VERSION if purpose_track else None,
        "source": _source_fingerprint(),
    })

    records = generate_expressions(
        fields=requested,
        windows=config.search.windows,
        families=families,
        max_expressions=config.search.max_expressions,
        max_per_family=config.search.max_per_family,
        seed=config.search.seed,
        priority_fields=config.search.priority_fields,
        priority_operators=config.search.priority_operators,
        priority_templates=config.search.priority_templates,
        priority_windows=config.search.priority_windows,
        diversity_share=config.search.diversity_share,
        evidence_prior=proposal_prior,
        evidence_strength=config.search.proposal_evidence_strength,
    )
    store.enqueue(experiment_id, records)
    completed_results = store.completed_results(experiment_id)
    compatible_results = (
        store.compatible_experiment_results(
            evaluation_signature,
            {record.expr_hash for record in records},
        )
        if is_crypto and config.search.coarse_keep_n else {}
    )
    reusable_results = (
        {}
        if is_crypto and config.search.coarse_keep_n
        else store.cached_results(
            evaluation_signature,
            {record.expr_hash for record in records} - set(completed_results),
        )
    )
    all_windows = set(config.search.windows)
    for family in families:
        all_windows.update(family.short_windows)
        all_windows.update(family.long_windows)
    evaluator = BatchEvaluator(
        panels={
            name: panel for name, panel in panels.items()
            if name in requested and name != "close"
        },
        dates=dates,
        codes=codes,
        windows=tuple(sorted(all_windows)),
        max_depth=max(f.max_depth for f in families if f.enabled),
        max_nodes=max(f.max_nodes for f in families if f.enabled),
        max_ts_ops=max(f.max_ts_ops for f in families if f.enabled),
        max_pair_ops=max(f.max_pair_ops for f in families if f.enabled),
        max_binary_ops=max(f.max_binary_ops for f in families if f.enabled),
        max_cache_items=config.search.max_cache_items,
    )

    result_rows = []
    reused_count = 0
    resumed_count = 0
    evaluated_count = 0
    fine_evaluated_count = len(records)
    medium_evaluated_count = 0
    started = time.perf_counter()
    if is_crypto and config.search.coarse_keep_n:
        batch_size = config.search.checkpoint_batch_size
        coarse_rows: dict[str, dict] = {}
        checkpoint: list[tuple[str, dict]] = []
        evaluation_records = sorted(records, key=_cache_locality_key)
        for index, record in enumerate(evaluation_records, start=1):
            existing = completed_results.get(record.expr_hash)
            if existing is not None:
                row = dict(existing)
                row.update(_record_row(record, purpose_track))
                coarse_rows[record.expr_hash] = row
                resumed_count += 1
                continue
            compatible = compatible_results.get(record.expr_hash)
            if compatible is not None:
                row = dict(compatible)
                row.update(_record_row(record, purpose_track))
                coarse_rows[record.expr_hash] = row
                reused_count += 1
                continue
            row = _record_row(record, purpose_track)
            values, status = evaluator.eval_expr(record.expr)
            if values is None:
                row.update({"status": status, "error": status})
            else:
                if purpose_track:
                    row.update(evaluate_purpose(
                        values, purpose_targets, dates, config.split,
                        config.evaluation, "coarse",
                    ))
                else:
                    coarse_evaluator = (
                        evaluate_time_series_factor_coarse
                        if is_crypto_time_series else evaluate_factor_coarse
                    )
                    row.update(coarse_evaluator(
                        values, forward, dates, config.split, config.evaluation
                    ))
                row["error"] = None
            row["evaluation_stage"] = "coarse"
            row["fine_evaluated"] = False
            coarse_rows[record.expr_hash] = row
            checkpoint.append((record.expr_hash, row))
            evaluated_count += 1
            if len(checkpoint) >= batch_size:
                store.write_results_batch(experiment_id, checkpoint)
                checkpoint.clear()
            if index % 250 == 0 or index == len(evaluation_records):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[coarse] {index}/{len(evaluation_records)} | new={evaluated_count} "
                    f"reused={reused_count} resumed={resumed_count} | "
                    f"{index / elapsed:.2f} expr/s",
                    flush=True,
                )
        store.write_results_batch(experiment_id, checkpoint)

        coarse_frame = rank_crypto_coarse_results(
            pd.DataFrame(coarse_rows.values())
        )
        coarse_selected = select_diverse_factors(
            coarse_frame,
            min(config.search.coarse_keep_n, len(coarse_frame)),
            config.search.coarse_max_per_family,
            config.search.coarse_max_per_primary_field,
        )
        fine_source = coarse_selected
        if is_crypto_time_series and config.search.medium_keep_n:
            medium_hashes = set(coarse_selected["expr_hash"].astype(str))
            medium_signature = f"{evaluation_signature}:crypto_medium_v1"
            medium_cache = store.cached_results(medium_signature, medium_hashes)
            medium_checkpoint: list[tuple[str, dict]] = []
            medium_cache_checkpoint: list[tuple[str, dict]] = []
            medium_records = sorted(
                (item for item in records if item.expr_hash in medium_hashes),
                key=_cache_locality_key,
            )
            medium_started = time.perf_counter()
            for medium_index, record in enumerate(medium_records, start=1):
                base = dict(coarse_rows[record.expr_hash])
                if base.get("evaluation_stage") in {"medium", "fine"}:
                    row = base
                elif record.expr_hash in medium_cache:
                    row = dict(medium_cache[record.expr_hash])
                    row.update(_record_row(record, purpose_track))
                    reused_count += 1
                else:
                    row = _record_row(record, purpose_track)
                    values, status = evaluator.eval_expr(record.expr)
                    if values is None:
                        row.update({"status": status, "error": status})
                    else:
                        row.update(
                            evaluate_purpose(
                                values, purpose_targets, dates, config.split,
                                config.evaluation, "medium",
                            ) if purpose_track else
                            evaluate_time_series_factor_medium(
                                values, forward, dates, config.split, config.evaluation
                            )
                        )
                        row["error"] = None
                    medium_evaluated_count += 1
                row["evaluation_stage"] = "medium"
                row["fine_evaluated"] = False
                coarse_rows[record.expr_hash] = row
                medium_checkpoint.append((record.expr_hash, row))
                medium_cache_checkpoint.append((record.expr_hash, row))
                if len(medium_checkpoint) >= batch_size:
                    store.write_results_batch(experiment_id, medium_checkpoint)
                    store.cache_results_batch(
                        medium_signature, experiment_id, medium_cache_checkpoint
                    )
                    medium_checkpoint.clear()
                    medium_cache_checkpoint.clear()
                if medium_index % 100 == 0 or medium_index == len(medium_records):
                    elapsed = max(time.perf_counter() - medium_started, 1e-9)
                    print(
                        f"[medium] {medium_index}/{len(medium_records)} | "
                        f"new={medium_evaluated_count} | {medium_index / elapsed:.2f} expr/s",
                        flush=True,
                    )
            store.write_results_batch(experiment_id, medium_checkpoint)
            store.cache_results_batch(
                medium_signature, experiment_id, medium_cache_checkpoint
            )
            medium_frame = rank_time_series_medium_results(pd.DataFrame([
                coarse_rows[value] for value in medium_hashes
            ]))
            fine_source = select_diverse_factors(
                medium_frame,
                min(config.search.medium_keep_n, len(medium_frame)),
                config.search.medium_max_per_family,
                config.search.medium_max_per_primary_field,
            )
        fine_hashes = set(fine_source["expr_hash"].astype(str))
        fine_evaluated_count = len(fine_hashes)
        fine_signature = f"{evaluation_signature}:crypto_fine_v1"
        fine_cache = store.cached_results(fine_signature, fine_hashes)
        fine_checkpoint: list[tuple[str, dict]] = []
        cache_checkpoint: list[tuple[str, dict]] = []
        fine_started = time.perf_counter()
        fine_new = 0
        fine_records = sorted(
            (item for item in records if item.expr_hash in fine_hashes),
            key=_cache_locality_key,
        )
        for fine_index, record in enumerate(fine_records, start=1):
            base = dict(coarse_rows[record.expr_hash])
            if base.get("evaluation_stage") == "fine":
                row = base
            elif record.expr_hash in fine_cache:
                row = dict(fine_cache[record.expr_hash])
                row.update(_record_row(record, purpose_track))
                reused_count += 1
            else:
                row = base
                row.update(_record_row(record, purpose_track))
                values, status = evaluator.eval_expr(record.expr)
                if values is None:
                    row.update({"status": status, "error": status})
                else:
                    if purpose_track:
                        row.update(evaluate_purpose(
                            values, purpose_targets, dates, config.split,
                            config.evaluation, "fine",
                        ))
                    else:
                        factor_evaluator = (
                            evaluate_time_series_factor
                            if is_crypto_time_series else evaluate_factor
                        )
                        result = factor_evaluator(
                            values, forward, dates, config.split,
                            config.evaluation, seed=config.search.seed,
                            include_holdout=False,
                        )
                        row.update(result.to_flat_dict())
                    if not purpose_track and result.direction != 0 and not variance_target:
                        search_metric_fn = (
                            crypto_time_series_search_metrics
                            if is_crypto_time_series else crypto_search_metrics
                        )
                        row.update(search_metric_fn(
                            values,
                            panels["close"],
                            dates,
                            config.split,
                            config.evaluation,
                            result.direction,
                            robustness_forward,
                        ))
                    row["error"] = None
                fine_new += 1
            row["evaluation_stage"] = "fine"
            row["fine_evaluated"] = True
            coarse_rows[record.expr_hash] = row
            fine_checkpoint.append((record.expr_hash, row))
            cache_checkpoint.append((record.expr_hash, row))
            if len(fine_checkpoint) >= batch_size:
                store.write_results_batch(experiment_id, fine_checkpoint)
                store.cache_results_batch(
                    fine_signature, experiment_id, cache_checkpoint
                )
                fine_checkpoint.clear()
                cache_checkpoint.clear()
            if fine_index % 50 == 0 or fine_index == len(fine_records):
                elapsed = max(time.perf_counter() - fine_started, 1e-9)
                print(
                    f"[fine] {fine_index}/{len(fine_records)} | "
                    f"new={fine_new} reused={reused_count} | "
                    f"{fine_index / elapsed:.2f} expr/s",
                    flush=True,
                )
        store.write_results_batch(experiment_id, fine_checkpoint)
        store.cache_results_batch(
            fine_signature, experiment_id, cache_checkpoint
        )
        result_rows = list(coarse_rows.values())
    else:
        for index, record in enumerate(records, start=1):
            if record.expr_hash in completed_results:
                result_rows.append(completed_results[record.expr_hash])
                resumed_count += 1
                continue
            if record.expr_hash in reusable_results:
                row = dict(reusable_results[record.expr_hash])
                row.update(_record_row(record, purpose_track))
                store.write_result(experiment_id, record.expr_hash, row)
                result_rows.append(row)
                reused_count += 1
                continue
            row = _record_row(record, purpose_track)
            values, status = evaluator.eval_expr(record.expr)
            if values is None:
                row.update({"status": status, "error": status})
                store.write_failure(experiment_id, record.expr_hash, status, row)
            else:
                if purpose_track:
                    row.update(evaluate_purpose(
                        values, purpose_targets, dates, config.split,
                        config.evaluation, "fine",
                    ))
                else:
                    factor_evaluator = (
                        evaluate_time_series_factor
                        if is_crypto_time_series else evaluate_factor
                    )
                    result = factor_evaluator(
                        values, forward, dates, config.split,
                        config.evaluation, seed=config.search.seed,
                        include_holdout=False,
                    )
                    row.update(result.to_flat_dict())
                if not purpose_track and is_crypto and result.direction != 0 and not variance_target:
                    search_metric_fn = (
                        crypto_time_series_search_metrics
                        if is_crypto_time_series else crypto_search_metrics
                    )
                    row.update(search_metric_fn(
                        values,
                        panels["close"],
                        dates,
                        config.split,
                        config.evaluation,
                        result.direction,
                        robustness_forward,
                    ))
                row["error"] = None
                store.write_result(experiment_id, record.expr_hash, row)
                store.cache_result(
                    evaluation_signature,
                    experiment_id,
                    record.expr_hash,
                    row,
                )
            result_rows.append(row)
            evaluated_count += 1
            if index % 50 == 0 or index == len(records):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[research] {index}/{len(records)} | "
                    f"new={evaluated_count} reused={reused_count} "
                    f"resumed={resumed_count} | {index / elapsed:.2f} expr/s",
                    flush=True,
                )

    incremental_artifact = None
    if purpose_track:
        if config.evaluation.research_track == "style_proxy":
            asset_style = liquidity_style_baseline_panels(panels)
            market_style = None
        else:
            asset_style, market_style = style_baseline_panels(panels, native_5m)
        incremental_rows = []
        for row in result_rows:
            if row.get("status") != "OK" or not row.get("fine_evaluated", True):
                continue
            values, status = evaluator.eval_expr(row["expr"])
            if values is None or status != "OK":
                raise RuntimeError(f"Purpose incremental evaluation failed: {row['expr']}: {status}")
            metrics = {}
            alpha_metrics = common_metrics = None
            if config.evaluation.research_track == "style_proxy":
                alpha_metrics = incremental_purpose_information(
                    values, purpose_targets.style, asset_style, dates, config.split,
                )
                metrics.update({f"style_{key}": value for key, value in alpha_metrics.items()})
            if config.evaluation.research_track in {"residual_alpha", "hybrid"}:
                alpha_metrics = incremental_purpose_information(
                    values, purpose_targets.residual, asset_style, dates, config.split,
                )
                metrics.update({f"alpha_{key}": value for key, value in alpha_metrics.items()})
            if config.evaluation.research_track in {"common_risk", "hybrid"}:
                common_metrics = incremental_purpose_information(
                    common_factor(values), purpose_targets.common, market_style,
                    dates, config.split,
                )
                metrics.update({f"common_{key}": value for key, value in common_metrics.items()})
            if alpha_metrics is None:
                primary = common_metrics
            elif common_metrics is None:
                primary = alpha_metrics
            else:
                primary = {
                    key: min(alpha_metrics[key], common_metrics[key])
                    if np.isfinite(alpha_metrics[key]) and np.isfinite(common_metrics[key])
                    else np.nan
                    for key in ("incremental_mean_delta_r2", "incremental_median_delta_r2",
                                "incremental_positive_asset_share", "incremental_baseline_mean_r2")
                }
                primary["incremental_asset_count"] = min(
                    alpha_metrics["incremental_asset_count"],
                    common_metrics["incremental_asset_count"],
                )
            row.update(metrics)
            row.update(primary)
            incremental_rows.append({
                "expr_hash": row["expr_hash"], "expr": row["expr"],
                "template_name": row["template_name"],
                "research_track": config.evaluation.research_track,
                **metrics, **primary,
            })
        incremental_artifact = "purpose_incremental_information.csv"
        pd.DataFrame(incremental_rows).to_csv(
            out / incremental_artifact, index=False, encoding="utf-8-sig"
        )
    elif variance_target:
        incremental_rows = []
        for row in result_rows:
            if row.get("status") != "OK" or not row.get("fine_evaluated", True):
                continue
            values, status = evaluator.eval_expr(row["expr"])
            if values is None or status != "OK":
                raise RuntimeError(f"Incremental evaluation failed: {row['expr']}: {status}")
            metrics = incremental_variance_information(
                values, forward, variance_baseline, dates, config.split
            )
            row.update(metrics)
            incremental_rows.append({
                "expr_hash": row["expr_hash"], "expr": row["expr"],
                "template_name": row["template_name"],
                "target_mode": config.evaluation.target_mode,
                "target_scope": config.evaluation.target_scope,
                **metrics,
            })
        incremental_artifact = "variance_incremental_information.csv"
        pd.DataFrame(incremental_rows).to_csv(
            out / incremental_artifact, index=False, encoding="utf-8-sig"
        )
    results = rank_research_results(pd.DataFrame(result_rows), config.evaluation)
    results["target_mode"] = config.evaluation.target_mode
    results["target_scope"] = config.evaluation.target_scope
    if is_crypto_time_series and "medium_session_positive_share" in results:
        stability = (
            pd.to_numeric(results["medium_session_positive_share"], errors="coerce")
            + pd.to_numeric(results["medium_asset_positive_share"], errors="coerce")
            + pd.to_numeric(results["medium_half_positive_share"], errors="coerce")
        ) / 3.0
        stability_rank = stability.rank(pct=True, method="average")
        valid = results["research_score"].notna() & stability_rank.notna()
        results.loc[valid, "research_score"] = (
            0.90 * results.loc[valid, "research_score"]
            + 0.10 * stability_rank.loc[valid]
        )
    if is_crypto_time_series and not purpose_track and "validation_net_total_return" in results:
        positive_validation = (
            pd.to_numeric(results["validation_net_total_return"], errors="coerce")
            .fillna(-np.inf)
            .gt(0.0)
        )
        results.loc[~positive_validation, "eligible"] = False
        results.loc[~positive_validation, "research_score"] = np.nan
        discovery_profitable = (
            pd.to_numeric(results["discovery_net_total_return"], errors="coerce") > 0.0
        ) & (
            pd.to_numeric(results["discovery_net_sharpe"], errors="coerce") > 0.0
        )
        session_columns = [
            "validation_us_overnight_sharpe",
            "validation_us_day_sharpe",
            "validation_us_evening_sharpe",
        ]
        session_breadth = results[session_columns].apply(
            pd.to_numeric, errors="coerce"
        ).gt(0.0).sum(axis=1) >= 2
        robust = discovery_profitable & session_breadth
        results.loc[~robust, "eligible"] = False
        results.loc[~robust, "research_score"] = np.nan
    selection_audit_artifact = None
    if purpose_track:
        def factor_provider(expression: str) -> np.ndarray:
            values, status = evaluator.eval_expr(expression)
            if values is None or status != "OK":
                raise RuntimeError(f"Distinctness evaluation failed: {expression}: {status}")
            return values

        selected, selection_audit = select_signal_distinct_factors(
            results, factor_provider,
            np.asarray(dates).astype(str) <= config.split.validation_end,
            config.selection.top_n,
            config.selection.max_per_family,
            config.selection.max_per_primary_field,
            config.selection.max_signal_similarity,
        )
        selection_audit_artifact = "selection_distinct_audit.csv"
        selection_audit.to_csv(
            out / selection_audit_artifact, index=False, encoding="utf-8-sig"
        )
    else:
        selected = select_diverse_factors(
            results,
            config.selection.top_n,
            config.selection.max_per_family,
            config.selection.max_per_primary_field,
        )

    execution_artifact = None
    if is_crypto_time_series and not purpose_track and config.evaluation.execution_horizons:
        micro = results.loc[
            results["fine_evaluated"].fillna(False).astype(bool)
            & results["fields"].fillna("").str.contains(r"micro5_|micro_", regex=True)
            & results["direction"].isin([-1, 1])
        ].sort_values("validation_mean_rank_ic", ascending=False)
        probes = micro.head(config.evaluation.execution_probe_n)
        candidates = pd.concat([selected, probes]).drop_duplicates("expr_hash")
        execution_rows = []
        for _, candidate in candidates.iterrows():
            values, status = evaluator.eval_expr(candidate["expr"])
            if values is None or status != "OK":
                raise RuntimeError(f"Execution probe failed: {candidate['expr']}: {status}")
            fields = str(candidate["fields"])
            feature_frequency = (
                "5m_aggregated_to_1h" if "micro5_" in fields else
                "15m_aggregated_to_1h" if "micro_" in fields else "1h"
            )
            for diagnostic in execution_diagnostic_rows(
                values, panels["close"], dates, config.split,
                config.evaluation, int(candidate["direction"]),
            ):
                execution_rows.append({
                    "expr_hash": candidate["expr_hash"],
                    "expr": candidate["expr"],
                    "feature_frequency": feature_frequency,
                    "decision_frequency": "1h",
                    "probe_source": (
                        "selected" if candidate["expr_hash"] in set(selected["expr_hash"])
                        else "micro_fine_probe"
                    ),
                    **diagnostic,
                })
        execution_artifact = "crypto_execution_diagnostics.csv"
        pd.DataFrame(execution_rows).to_csv(
            out / execution_artifact, index=False, encoding="utf-8-sig"
        )

    futures_performance_artifacts = (
        build_futures_factor_return_report(
            selected,
            evaluator,
            panels["close"],
            dates,
            config.split.validation_end,
            config.evaluation.entry_lag,
            out,
        )
        if config.data.source == "futures_parquet"
        else {}
    )
    crypto_report_artifacts = (
        (
            build_crypto_time_series_reports
            if is_crypto_time_series else build_crypto_reports
        )(
            selected,
            evaluator,
            panels["close"],
            dates,
            config.split,
            config.evaluation,
            out,
        )
        if is_crypto and not variance_target and not purpose_track else {}
    )

    selected_panels = {}
    holdout_rows = []
    for selected_index, (_, row) in enumerate(selected.iterrows()):
        values, status = evaluator.eval_expr(row["expr"])
        if values is not None and status == "OK":
            if selected_index < config.selection.pair_candidate_n:
                selected_panels[row["expr_hash"]] = values.copy()
            if purpose_track:
                audit = evaluate_purpose(
                    values, purpose_targets, dates, config.split,
                    config.evaluation, "fine", include_holdout=True,
                )
            else:
                factor_evaluator = (
                    evaluate_time_series_factor
                    if is_crypto_time_series else evaluate_factor
                )
                audit = factor_evaluator(
                    values, audit_forward, dates, config.split,
                    config.evaluation, seed=config.search.seed,
                    include_holdout=True,
                ).to_flat_dict()
            holdout_rows.append({
                "expr_hash": row["expr_hash"],
                **{
                    key: value
                    for key, value in audit.items()
                    if key.startswith("holdout_")
                },
            })
    pairs = build_factor_pairs(
        selected,
        selected_panels,
        config.selection.pair_top_n,
        config.selection.pair_candidate_n,
        config.selection.pair_max_similarity,
        research_mask=np.asarray(dates).astype(str) <= config.split.validation_end,
        similarity_mode="time_series" if is_crypto_time_series else "cross_sectional",
    )
    if is_crypto_time_series and not variance_target and not purpose_track:
        crypto_report_artifacts.update(build_crypto_pair_report(
            pairs, selected, selected_panels, panels["close"], dates,
            config.split, config.evaluation, out,
        ))
    attribution = compute_attribution(results)
    report_artifacts = {}

    holdout_columns = [column for column in results if column.startswith("holdout_")]
    search_columns = [column for column in results if column not in holdout_columns]
    results[search_columns].to_csv(
        out / "search_results.csv", index=False, encoding="utf-8-sig"
    )
    selected[search_columns].to_csv(
        out / "selected_factors.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        holdout_rows,
        columns=["expr_hash", *holdout_columns],
    ).to_csv(
        out / "holdout_audit.csv", index=False, encoding="utf-8-sig"
    )
    pairs.to_csv(out / "factor_pairs.csv", index=False, encoding="utf-8-sig")
    for name, frame in attribution.items():
        frame.to_csv(
            out / f"attribution_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if config.reporting.enabled:
        report_artifacts = build_reports(
            results,
            selected,
            attribution,
            out,
            top_n=config.reporting.top_n,
        )

    manifest = {
        "experiment_id": experiment_id,
        "framework_version": config.version,
        "status": "COMPLETED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "config_hash": _stable_hash(stable_config),
        "template_hash": _stable_hash(template_raw),
        "data_fingerprint": data_fingerprint,
        "evaluation_signature": evaluation_signature,
        "field_formula_version": field_formula_version,
        "operator_semantics_version": OPERATOR_SEMANTICS_VERSION,
        "target_formula_version": (
            PURPOSE_FORMULA_VERSION if purpose_track else
            TARGET_FORMULA_VERSION if variance_target else "close_return_v1"
        ),
        "baseline_formula_version": (
            PURPOSE_BASELINE_FORMULA_VERSION if purpose_track else
            BASELINE_FORMULA_VERSION if variance_target else None
        ),
        "cluster_field_formula_version": (
            CLUSTER_FIELD_FORMULA_VERSION if config.cluster_state.enabled else None
        ),
        "proposal_evidence_sha256": proposal_fingerprint,
        "config": stable_config,
        "data_summary": {
            "rows": len(raw),
            "dates": len(dates),
            "codes": len(codes),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
        },
        "research_summary": {
            "expressions": len(records),
            "successful": int(results["status"].eq("OK").sum()),
            "eligible": int(results.get("eligible", pd.Series(dtype=bool)).sum()),
            "selected": len(selected),
            "pairs": len(pairs),
            "newly_evaluated": evaluated_count,
            "fine_evaluated": fine_evaluated_count,
            "medium_evaluated": medium_evaluated_count,
            "reused_results": reused_count,
            "resumed_results": resumed_count,
        },
        "artifacts": {
            "search_results": "search_results.csv",
            "selected_factors": "selected_factors.csv",
            "holdout_audit": "holdout_audit.csv",
            "factor_pairs": "factor_pairs.csv",
            "execution_diagnostics": execution_artifact,
            "variance_incremental_information": incremental_artifact,
            "selection_distinct_audit": selection_audit_artifact,
            "cluster_state_model": cluster_model_artifact,
            "attribution": [
                f"attribution_{name}.csv" for name in attribution
            ],
            "reports": {
                **source_report_artifacts,
                **futures_performance_artifacts,
                **crypto_report_artifacts,
                **report_artifacts,
            },
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.complete(experiment_id, manifest)
    store.close()
    return manifest
