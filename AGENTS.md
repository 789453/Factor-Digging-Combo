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

## Prediction research priority (2026-10-09)

18. Read `docs/PREDICTIVE_FACTOR_RESEARCH_POLICY_20261009.md` before factor
    selection, combination, or new experiments. The current research product is
    an informative feature pool plus observable multi-factor predictions.
19. Never require individual features to earn positive net trading returns,
    beat fees, or have profitable hedge legs to enter a prediction pool.
    Trading admission is a separate portfolio-level decision. Preserve unstable
    and in-sample candidates in explicitly labelled diagnostic archives.
20. Keep raw-return, risk-residual, and baseline-error prediction objectives
    explicit. An error correction must not silently become a standalone return
    forecast. Report direct prediction baselines before adding control layers.
21. Default near-term work to reuse of frozen candidates, small Ridge baselines,
    and heavy diagnostics disabled. Do not launch another large search until
    the feature pool and baseline have observable results and a new research
    question justifies it. Empty trading admission must not erase predictions.
22. State dependence is admissible evidence, not permission to flip direction
    using future outcomes. Fit coefficients/states using matured past labels.
    Source-selected discovery folds are research feedback, not independent OOF.

## Effective strategy mandate (2026-10-09, user correction)

23. Read `docs/EFFECTIVE_FACTOR_AND_STRATEGY_MANDATE_20261009.md` before continuing
    prediction research. Observable losing baselines and failure explanations
    alone do not satisfy the current task; pursue useful factors and a strategy.
24. Small Ridge baselines are the starting point, not a permanent model ceiling.
    Their observed failures authorize bounded nonlinear, asset-conditional and
    matured-label update research. Preserve explicit hypotheses and all attempts.
25. Strong in-sample and overfit diagnostic products are allowed and must be
    observable, but must never be represented as causal out-of-sample strategies.
    The holdout firewall and real-time label maturity still apply.
26. Do not mark the strategy task successful without passing the mandate's
    historical signal/strategy acceptance. Restore full interactive evidence,
    asset/time breadth, trading paths, and the iteration log, not just tables.
27. A discovery argmax is a research recommendation, not the sole product gate.
    Retain all predeclared predictions and report their validation survival;
    do not rewrite the frozen winner using validation. Fixed member removals
    keep the original divisor, units, execution and budget without refitting.
