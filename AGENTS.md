# Alpha Research Project Rules

These rules apply to the whole repository.

1. `python -m src.alpha_mvp.research_cli --config ...` is the only production
   research entry point. Compatibility modules may exist temporarily, but new
   features must not add another pipeline.
2. The holdout period is a firewall. Candidate scoring, ranking, template
   tuning, and pair selection must never read a `holdout_*` metric.
3. A new field requires a field knowledge card, formula/version metadata,
   dependency declaration, and unit tests covering missing and extreme input.
4. A new operator requires documented semantics, arity, numerical-domain
   rules, NaN behavior, tie behavior where relevant, and reference tests
   against a simple implementation.
5. A new template must be declared in YAML, have a bounded complexity budget,
   deterministic generation, and representation in template attribution.
6. Generated research artifacts under `outputs/` are immutable evidence.
   Never overwrite a completed experiment; use a new output directory.
7. Direction is inferred from discovery data only. Validation measures
   survival. Holdout is reported only after the research decision is frozen.
8. Every behavior change must pass unit tests plus one simulated smoke run.
   Data/metric changes also require a bounded real-data check when available.
9. Keep domain logic out of orchestration. Data, expressions, evaluation,
   selection, and reporting must remain independently testable.
10. Do not silently fall back from a requested configuration. Invalid fields,
    operators, templates, dates, or data sources must fail with a clear error.

## Current product direction (2026-09-28)

11. The near-term primary workload is crypto per-asset time-series factor
    research. Keep stock and futures cross-sectional interfaces working; asset
    and factor modes must be selected explicitly in configuration.
12. Use coarse, medium, and fine screening for large searches. The medium
    stage must preserve field/template/window/frequency diversity; expensive
    robustness and strategy simulation belong on bounded survivors.
13. Crypto research returns are signed long-short unless an explicitly named
    diagnostic says otherwise. Reports must state direction source, lag,
    horizon, costs, and compounding convention.
14. Treat stable beta/risk/style proxies as valid research products, but do
    not label them pure alpha. Future work must report exposure diagnostics
    and residual performance without using holdout for classification.
15. Do not conclude that 5m/15m information is useless from the current 1h
    decision-panel run. Native multi-frequency work must separate feature,
    signal, rebalance, and holding frequencies and report decay, turnover,
    and cost break-even.
16. Structured or semi-mined factors must still use the same production entry,
    bounded deterministic YAML declarations, attribution, metadata, tests, and
    holdout firewall. Do not create a second research pipeline.
17. Before material research changes, read `docs/README.md`,
    `docs/PROJECT_STATUS_AND_ROADMAP_20260928.md`, and
    `docs/RESEARCH_REQUIREMENTS.md`.
