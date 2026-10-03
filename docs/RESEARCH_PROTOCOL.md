# Research Protocol

## 1. Before running

Freeze the following in YAML:

- source data and asset universe;
- date range and discovery/validation/holdout boundaries;
- field list and formula version;
- operator and template configuration;
- horizon and minimum coverage;
- entry lag, expression limits, random seed, priority policy, and selection limits.

Changing any of these creates a different experiment.

## 2. Base-variable quality gate

A field may enter search only if:

- its formula uses current or historical information;
- its raw dependencies exist at the stated timestamp;
- division and logarithm domains are protected;
- coverage and extreme-value distributions have been inspected;
- it is not a duplicate or monotonic alias of an existing field;
- its knowledge card is complete.

Size, industry, liquidity, and tradability fields must be marked as either
search inputs, controls, or filters. They must not drift silently between roles.

## 3. Search protocol

1. Generate candidates deterministically.
2. Reserve the configured diversity quota, then apply explicit priorities.
3. Canonicalize and reject invalid structures. Under RankIC, also merge
   root-level strictly monotonic equivalents.
4. Reuse only results with an identical strict evaluation signature.
5. Compute factor values without reading future returns.
6. Infer sign from discovery RankIC only.
7. Reject low-coverage and directionless factors.
8. Use coarse, medium, and fine stages for large searches. Protect structural,
   field, window, and frequency diversity before expensive fine evaluation.
9. Rank surviving factors using discovery and validation only.
10. Apply structural diversity limits.
11. Build low-similarity factor pairs using the mode-appropriate similarity
    definition.
12. Freeze selected candidates before opening holdout results.

## 4. Interpretation rules

ICIR is not a standalone confidence measure. Review together:

- mean RankIC;
- Newey-West t-statistic;
- block-bootstrap confidence interval;
- positive-day ratio;
- discovery-to-validation sign survival;
- coverage and usable days;
- turnover proxy;
- template and field support count.

A single high result with weak support is a lead, not evidence.

For crypto strategy factors, also review net signed long-short cumulative
return and Sharpe, turnover, break-even costs, holding-horizon decay, New York
session stability, and worst subperiod. Report order may use validation net
cumulative return, but selection should remain multi-objective.

Beta, risk, and style proxies are not automatically rejected. When exposure
diagnostics exist, preserve both raw and neutralized evidence and distinguish
stable beta/style candidates from residual-alpha candidates. Exposure model
selection must not read holdout.

## 5. Holdout firewall

`holdout_audit.csv` contains only the frozen selected set and is never joined
into candidate ranking. If holdout results
influence a field, operator, template, threshold, or factor choice, the holdout
has become validation data. A new later holdout must then be designated.
Overall and Top50 reports are also search-only and must not display holdout
columns.

## 6. Promotion levels

- L0 — idea: knowledge card only.
- L1 — computable: formula and tests pass.
- L2 — discovered: positive discovery evidence.
- L3 — validated: survives validation and robustness checks.
- L4 — paired: adds value beside complementary factors.
- L5 — holdout-audited: frozen candidate survives the holdout.
- L6 — production candidate: tradability and execution tests pass.

## 7. Probabilistic-search extension

Future Bayesian, bandit, evolutionary, or probabilistic grammar methods should
change only the proposal policy. They must consume the same expression catalog
and attribution evidence and must return candidates through the same
validation contract.

Recommended proposal state:

```text
P(field | evidence)
P(operator | field, window, template)
P(template | complexity, support, validation survival)
P(pair | individual score, similarity, complementarity)
```

Use smoothed estimates and minimum support. Never train proposal probabilities
on holdout metrics.

## 8. Multi-frequency extension

Fast inputs must not be judged only through a slow strategy's assumptions.
Separate feature sampling, signal refresh, portfolio rebalance, and holding
horizon. Evaluate decay curves, turnover controls, realistic fee/slippage
scenarios, and cost break-even on a bounded survivor set. Fast, medium, and
slow sleeves may be selected separately before combination.

Current 5m/15m crypto fields are aggregated onto an hourly decision panel;
they are not evidence of native 5m/15m execution performance. Native frequency
support must remain in the same CLI and holdout protocol.

## 9. Structured and semi-mined templates

Expert-designed event/state/gate/persistence modules may extend the proposal
space when shallow operator nesting is insufficient. Each module still needs
an economic hypothesis card, dimensional rules, bounded deterministic YAML
parameters, complexity metadata, template attribution, tests, and the same
three-stage evaluation. It must not become a second pipeline.

## 10. Full-market portfolio audit

The long-only audit may run only after factor selection is frozen. It must:

- preserve the discovery-period direction;
- execute no earlier than the configured entry lag;
- rank only stocks with a valid signal and execution-date close;
- report every rebalance/Top-K condition separately;
- retain the full expression beside every displayed curve;
- disclose that an extended history is not a new untouched holdout when it
  overlaps factor discovery or validation.

Gross curves without costs or tradability filters are diagnostics, not
production performance claims.
