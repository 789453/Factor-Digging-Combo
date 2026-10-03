"""Clean research kernel for factor discovery, pairing, and validation."""

from .config import ResearchConfig, load_research_config
from .attribution import compute_attribution
from .evaluation import EvaluationResult, evaluate_factor, rank_research_results
from .selection import build_factor_pairs, select_diverse_factors
from .templates import ExpressionRecord, generate_expressions, load_template_families

__all__ = [
    "ResearchConfig",
    "load_research_config",
    "compute_attribution",
    "EvaluationResult",
    "evaluate_factor",
    "rank_research_results",
    "ExpressionRecord",
    "generate_expressions",
    "load_template_families",
    "select_diverse_factors",
    "build_factor_pairs",
]
