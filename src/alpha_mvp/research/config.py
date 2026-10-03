from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str = "simulated"
    duckdb_path: str | None = None
    pool_json: str | None = None
    parquet_path: str | None = None
    futures_universe: str | None = None
    crypto_parquet_root: str | None = None
    crypto_primary_timeframe: str = "1h"
    crypto_micro_timeframe: str = "15m"
    crypto_fast_timeframe: str | None = None
    crypto_feature_cache_dir: str | None = None
    start: str = "20240101"
    end: str = "20260430"
    simulated_days: int = 180
    simulated_stocks: int = 100


@dataclass(frozen=True)
class SplitConfig:
    discovery_end: str = "20241231"
    validation_start: str = "20250101"
    validation_end: str = "20251231"
    holdout_start: str = "20260101"

    def validate(self) -> None:
        if not (
            self.discovery_end < self.validation_start
            <= self.validation_end < self.holdout_start
        ):
            raise ValueError(
                "Require discovery_end < validation_start <= "
                "validation_end < holdout_start"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    horizon: int = 5
    research_track: str = "legacy"
    target_mode: str = "return"
    target_scope: str = "absolute"
    entry_lag: int = 1
    min_daily_names: int = 30
    min_coverage: float = 0.55
    nw_lags: int = 5
    bootstrap_samples: int = 300
    bootstrap_block: int = 20
    periods_per_year: int = 252
    strategy_cost_bps: float = 0.0
    strategy_top_pct: float = 0.25
    robustness_horizons: tuple[int, ...] = ()
    factor_mode: str = "cross_sectional"
    time_series_block: int = 168
    time_series_position_window: int = 168
    time_series_position_clip: float = 2.0
    execution_horizons: tuple[int, ...] = ()
    execution_rebalance_bars: tuple[int, ...] = (1,)
    execution_no_trade_bands: tuple[float, ...] = (0.0,)
    execution_cost_bps: tuple[float, ...] = (4.0,)
    execution_probe_n: int = 0
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "validation_edge_rank": 0.30,
        "validation_ir_rank": 0.20,
        "validation_hit_rank": 0.10,
        "discovery_edge_rank": 0.10,
        "sign_consistency_rank": 0.10,
        "coverage_rank": 0.08,
        "turnover_rank": 0.07,
        "complexity_rank": 0.05,
    })

    def validate(self) -> None:
        if self.research_track not in {
            "legacy", "residual_alpha", "common_risk", "hybrid", "style_proxy"
        }:
            raise ValueError("research_track must be legacy, residual_alpha, common_risk, hybrid, or style_proxy")
        if self.research_track in {"residual_alpha", "hybrid"} and self.target_mode != "return":
            raise ValueError("residual_alpha and hybrid require target_mode=return")
        if self.research_track == "common_risk" and self.target_mode != "downside_semivariance":
            raise ValueError("common_risk requires target_mode=downside_semivariance")
        if self.research_track == "style_proxy" and self.target_mode != "liquidity_concentration":
            raise ValueError("style_proxy requires target_mode=liquidity_concentration")
        if self.target_mode == "liquidity_concentration" and self.research_track != "style_proxy":
            raise ValueError("liquidity_concentration requires research_track=style_proxy")
        if self.target_mode not in {
            "return", "upside_semivariance", "downside_semivariance", "total_variance", "liquidity_concentration"
        }:
            raise ValueError("target_mode must be return, upside_semivariance, downside_semivariance, or total_variance")
        if self.target_scope not in {"absolute", "market_relative"}:
            raise ValueError("target_scope must be absolute or market_relative")
        if self.research_track != "legacy" and self.target_scope != "absolute":
            raise ValueError("purpose research tracks require target_scope=absolute")
        if self.target_mode == "return" and self.target_scope != "absolute":
            raise ValueError("market_relative target_scope requires a variance target")
        if self.target_mode != "return":
            return_only_scores = {
                "validation_sharpe_rank", "discovery_sharpe_rank",
                "horizon_robustness_rank",
            }
            if return_only_scores & set(self.score_weights):
                raise ValueError("variance targets cannot use return strategy score components")
            if self.robustness_horizons:
                raise ValueError("variance targets cannot use return robustness_horizons")
        elif "incremental_r2_rank" in self.score_weights and self.research_track == "legacy":
            raise ValueError("incremental_r2_rank requires a variance target")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.entry_lag < 0:
            raise ValueError("entry_lag cannot be negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if self.strategy_cost_bps < 0:
            raise ValueError("strategy_cost_bps cannot be negative")
        if not 0 < self.strategy_top_pct < 0.5:
            raise ValueError("strategy_top_pct must be in (0, 0.5)")
        if any(h <= 0 for h in self.robustness_horizons):
            raise ValueError("robustness_horizons must contain positive integers")
        if self.factor_mode not in {"cross_sectional", "time_series"}:
            raise ValueError("factor_mode must be cross_sectional or time_series")
        if self.time_series_block < 24:
            raise ValueError("time_series_block must be at least 24")
        if self.time_series_position_window < 24:
            raise ValueError("time_series_position_window must be at least 24")
        if self.time_series_position_clip <= 0:
            raise ValueError("time_series_position_clip must be positive")
        if self.execution_horizons:
            if len(self.execution_horizons) > 8 or any(
                not isinstance(x, int) or x <= 0 for x in self.execution_horizons
            ):
                raise ValueError("execution_horizons must contain at most 8 positive integers")
            if len(self.execution_rebalance_bars) * len(self.execution_no_trade_bands) * len(self.execution_horizons) * len(self.execution_cost_bps) > 96:
                raise ValueError("execution diagnostics grid exceeds 96 scenarios")
        if self.execution_probe_n < 0 or self.execution_probe_n > 100:
            raise ValueError("execution_probe_n must be in [0, 100]")
        if self.execution_probe_n and not self.execution_horizons:
            raise ValueError("execution_probe_n requires execution_horizons")
        for name, values in (
            ("execution_rebalance_bars", self.execution_rebalance_bars),
            ("execution_no_trade_bands", self.execution_no_trade_bands),
            ("execution_cost_bps", self.execution_cost_bps),
        ):
            if not values or any(
                not isinstance(x, (int, float)) or not isfinite(x) or x < 0
                for x in values
            ):
                raise ValueError(f"{name} must contain nonnegative numbers")
        if any(x == 0 or int(x) != x for x in self.execution_rebalance_bars):
            raise ValueError("execution_rebalance_bars must contain positive integers")
        if self.min_daily_names < 3:
            raise ValueError("min_daily_names must be at least 3")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        if abs(sum(self.score_weights.values()) - 1.0) > 1e-9:
            raise ValueError("score_weights must sum to 1")


@dataclass(frozen=True)
class SearchConfig:
    fields: tuple[str, ...] = ()
    exclude_fields: tuple[str, ...] = ()
    windows: tuple[int, ...] = (10, 20, 30, 40, 50)
    max_expressions: int = 5000
    max_per_family: int = 2000
    seed: int = 42
    template_config: str = "configs/research/templates.yaml"
    max_cache_items: int = 256
    priority_fields: tuple[str, ...] = ()
    priority_operators: tuple[str, ...] = ()
    priority_templates: tuple[str, ...] = ()
    priority_windows: tuple[int, ...] = ()
    diversity_share: float = 0.50
    proposal_evidence_path: str | None = None
    proposal_evidence_strength: float = 0.0
    coarse_keep_n: int = 0
    checkpoint_batch_size: int = 50
    coarse_max_per_family: int = 0
    coarse_max_per_primary_field: int = 0
    medium_keep_n: int = 0
    medium_max_per_family: int = 0
    medium_max_per_primary_field: int = 0

    def validate(self) -> None:
        if self.max_expressions <= 0 or self.max_per_family <= 0:
            raise ValueError("expression limits must be positive")
        if not self.windows or any(w <= 1 for w in self.windows):
            raise ValueError("windows must contain integers greater than 1")
        if self.max_cache_items < 0:
            raise ValueError("max_cache_items cannot be negative")
        if not 0 <= self.diversity_share <= 1:
            raise ValueError("diversity_share must be in [0, 1]")
        if not 0 <= self.proposal_evidence_strength <= 1:
            raise ValueError("proposal_evidence_strength must be in [0, 1]")
        if bool(self.proposal_evidence_path) != bool(self.proposal_evidence_strength):
            raise ValueError("proposal_evidence_path and strength must be set together")
        if self.coarse_keep_n < 0:
            raise ValueError("coarse_keep_n cannot be negative")
        if self.checkpoint_batch_size <= 0:
            raise ValueError("checkpoint_batch_size must be positive")
        if self.coarse_keep_n and (
            self.coarse_max_per_family <= 0
            or self.coarse_max_per_primary_field <= 0
        ):
            raise ValueError(
                "coarse diversity caps must be positive when coarse_keep_n is set"
            )
        if self.medium_keep_n < 0:
            raise ValueError("medium_keep_n cannot be negative")
        if self.medium_keep_n:
            if not self.coarse_keep_n or self.medium_keep_n > self.coarse_keep_n:
                raise ValueError("medium_keep_n requires coarse_keep_n >= medium_keep_n")
            if (
                self.medium_max_per_family <= 0
                or self.medium_max_per_primary_field <= 0
            ):
                raise ValueError(
                    "medium diversity caps must be positive when medium_keep_n is set"
                )


@dataclass(frozen=True)
class SelectionConfig:
    top_n: int = 100
    max_per_family: int = 30
    max_per_primary_field: int = 15
    pair_top_n: int = 30
    pair_candidate_n: int = 80
    pair_max_similarity: float = 0.70
    max_signal_similarity: float = 0.985


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "outputs/research"
    store_path: str = "outputs/research_index.duckdb"
    save_factor_panels: bool = False


@dataclass(frozen=True)
class ReportingConfig:
    enabled: bool = True
    top_n: int = 50


@dataclass(frozen=True)
class ClusterStateConfig:
    enabled: bool = False
    field_name: str = "cluster_stress_affinity"
    flow_field: str = "micro5_taker_imbalance"
    jump_field: str = "micro5_jump_share"
    illiquidity_field: str = "session_illiquidity_surprise"
    n_clusters: int = 2
    history_window: int = 168
    fit_stride: int = 6
    max_fit_samples: int = 12000
    min_cluster_share: float = 0.02

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.field_name != "cluster_stress_affinity":
            raise ValueError("cluster_state.field_name must be cluster_stress_affinity")
        if (
            self.flow_field, self.jump_field, self.illiquidity_field
        ) != (
            "micro5_taker_imbalance", "micro5_jump_share",
            "session_illiquidity_surprise",
        ):
            raise ValueError("cluster_state v1 requires the declared flow/jump/illiquidity roles")
        if self.n_clusters not in {2, 3}:
            raise ValueError("cluster_state.n_clusters must be 2 or 3")
        if self.history_window < 24 or self.fit_stride < 1:
            raise ValueError("cluster_state history_window >= 24 and fit_stride >= 1 are required")
        if not 200 <= self.max_fit_samples <= 50000:
            raise ValueError("cluster_state.max_fit_samples must be in [200, 50000]")
        if not 0 < self.min_cluster_share <= 0.2:
            raise ValueError("cluster_state.min_cluster_share must be in (0, 0.2]")


@dataclass(frozen=True)
class ResearchConfig:
    version: str
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    search: SearchConfig
    selection: SelectionConfig
    output: OutputConfig
    reporting: ReportingConfig
    cluster_state: ClusterStateConfig = field(default_factory=ClusterStateConfig)

    def validate(self) -> None:
        if not self.version:
            raise ValueError("version is required")
        self.split.validate()
        self.evaluation.validate()
        if self.evaluation.execution_horizons and (
            self.data.source != "crypto_parquet"
            or self.evaluation.factor_mode != "time_series"
            or self.evaluation.target_mode != "return"
        ):
            raise ValueError("execution diagnostics require crypto_parquet time_series return mode")
        if self.evaluation.target_mode != "return" and self.evaluation.research_track == "legacy" and (
            self.data.source != "crypto_parquet"
            or self.evaluation.factor_mode != "time_series"
            or self.data.crypto_fast_timeframe != "5m"
        ):
            raise ValueError("variance targets require crypto_parquet time_series mode with 5m bars")
        self.search.validate()
        if not 0 < self.selection.max_signal_similarity < 1:
            raise ValueError("selection.max_signal_similarity must be in (0, 1)")
        if self.evaluation.research_track != "legacy" and (
            self.data.source != "crypto_parquet"
            or self.evaluation.factor_mode != "time_series"
        ):
            raise ValueError("research_track requires crypto_parquet time_series mode")
        if self.evaluation.research_track == "style_proxy" and (
            self.data.crypto_primary_timeframe != "1h"
            or self.data.crypto_fast_timeframe != "5m"
        ):
            raise ValueError("style_proxy requires 1h primary and 5m fast crypto bars")
        self.cluster_state.validate()
        if self.cluster_state.enabled and (
            self.data.source != "crypto_parquet"
            or self.evaluation.factor_mode != "time_series"
            or self.data.crypto_fast_timeframe != "5m"
        ):
            raise ValueError("cluster_state requires crypto_parquet time_series with 5m bars")
        if self.cluster_state.enabled and self.cluster_state.field_name not in self.search.fields:
            raise ValueError("cluster_state field must be explicitly listed in search.fields")
        if not self.cluster_state.enabled and "cluster_stress_affinity" in self.search.fields:
            raise ValueError("cluster_stress_affinity requires cluster_state.enabled")
        if self.data.source not in {
            "simulated", "duckdb", "futures_parquet", "crypto_parquet",
        }:
            raise ValueError(
                "data.source must be simulated, duckdb, futures_parquet, "
                "or crypto_parquet"
            )
        if self.data.source == "duckdb" and not self.data.duckdb_path:
            raise ValueError("data.duckdb_path is required for DuckDB data")
        if self.data.source == "futures_parquet":
            if not self.data.parquet_path:
                raise ValueError("data.parquet_path is required for futures_parquet data")
            if self.data.futures_universe not in {
                "commodity_main", "index_continuous",
            }:
                raise ValueError(
                    "data.futures_universe must be commodity_main or "
                    "index_continuous for futures_parquet data"
                )
        if self.data.source == "crypto_parquet":
            if not self.data.crypto_parquet_root:
                raise ValueError(
                    "data.crypto_parquet_root is required for crypto_parquet data"
                )
            if self.data.crypto_primary_timeframe not in {"1h", "5m"}:
                raise ValueError("crypto_primary_timeframe must be 1h or 5m")
            if self.data.crypto_primary_timeframe == "5m" and self.evaluation.research_track == "legacy":
                raise ValueError("native 5m requires an explicit research_track")
            if self.data.crypto_micro_timeframe != "15m":
                raise ValueError("crypto_micro_timeframe currently must be 15m")
            if self.data.crypto_fast_timeframe not in {None, "5m"}:
                raise ValueError("crypto_fast_timeframe must be null or 5m")


def _construct(cls, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_research_config(path: str) -> ResearchConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = ResearchConfig(
        version=str(raw.get("version", "")),
        data=_construct(DataConfig, raw.get("data")),
        split=_construct(SplitConfig, raw.get("split")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        search=_construct(SearchConfig, raw.get("search")),
        selection=_construct(SelectionConfig, raw.get("selection")),
        output=_construct(OutputConfig, raw.get("output")),
        reporting=_construct(ReportingConfig, raw.get("reporting")),
        cluster_state=_construct(ClusterStateConfig, raw.get("cluster_state")),
    )
    cfg.validate()
    return cfg
