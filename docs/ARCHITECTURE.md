# Alpha Research Framework Architecture

## Purpose

This repository is a research operating system for:

1. constructing versioned base variables;
2. generating reproducible factor expressions;
3. screening them without contaminating the final holdout;
4. selecting complementary factor pairs;
5. turning accumulated search evidence into richer, human-designed factors.

It is not a library of final factors. Its primary product is trustworthy
research evidence.

## One production path

```text
Research YAML
  -> data extraction
  -> versioned field construction
  -> T x N panels
  -> deterministic template generation
  -> parse / canonicalize / validate
  -> coarse vectorized screening
  -> medium diversity / segment screening
  -> fine strategy and robustness evaluation
  -> discovery direction
  -> validation survival score
  -> diversity selection
  -> low-similarity pairing
  -> holdout audit
  -> overall / Top50 reports
  -> experiment manifest and artifacts
```

The entry point is:

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/crypto_multiscale_sessions_smoke.yaml
```

## Layer boundaries

### Data and fields

`data.py` only extracts source columns. `fields.py` constructs domain
variables. The research workflow converts them into aligned `(T, N)` panels.

For crypto, `data.py` aligns completed 5-minute and 15-minute bars to each
hourly candle and `crypto_fields.py` builds the versioned multi-scale and
DST-aware New York session vocabulary. The current production panel remains
hourly, so these fast inputs are within-hour aggregates known before the
configured one-hour entry lag. This is not yet a native 5m/15m rebalance
engine; the future multi-frequency lane must separate feature, signal,
rebalance, and holding frequencies.

Crypto production searches use deterministic three-stage evaluation. The
coarse stage evaluates all generated expressions with vectorized primary
metrics and batched checkpoints. It omits bootstrap, Newey-West, full strategy
simulation, and broad sensitivity work. The medium stage applies
template/field diversity caps and discovery/validation segment and session
survival checks. A bounded set then enters the fine stage for net signed
long-short simulation, costs, turnover, multi-horizon IC and full robustness.
Related templates are evaluated adjacently to keep rolling sub-panels in the
bounded LRU cache. Completed results may be reused only when their full
evaluation signature (data fingerprint, split, fields, formula/operator
versions, and evaluation settings) is identical.

A base variable is an input vocabulary item, not a discovered factor. It must
have a stable name, formula version, raw dependencies, expected domain,
missing-value behavior, and an economic interpretation that does not rely on
its backtest result.

### Expression language

Expressions use a compact function syntax:

```text
Rank(TsMean($ret_1d,20))
Rank(TsCorr($flow_imbalance,$ret_1d,20))
```

The parser builds an AST, canonicalization removes structural duplicates, the
validator enforces syntax and complexity, and the evaluator maps the AST to
panel operations.

Cross-sectional configurations may merge root-level strictly monotonic
transforms under RankIC equivalence. Crypto time-series configurations do not
wrap expressions in cross-sectional `Rank`; their transforms and IC operate
along each asset's own history.

### Template search

Generic templates define tree shapes and variable slots. Role-constrained
crypto templates additionally declare which field categories may occupy fast,
slow, state, gate, confirmation, and scale roles. The generic vocabulary is:

```text
U(F)        unary preprocessing
T(U(F), w)  time-series state
X(T)        cross-sectional transform
B(A, B)     binary composition
P(A, B, w)  rolling relationship
```

Enabled families include single state, same/mixed-state binary, rolling
relationship, short/long gap, cross-window state, normalized spread,
relationship change, and triple modulation.
Four-variable balanced templates are intentionally disabled by default.

The current crypto multiscale catalog adds session-adjusted price/flow,
volatility asymmetry, liquidity efficiency, accumulation state, derivative
dynamics, correlation regime, cross-scale state, and tri-state interaction.
Future expert-structured modules may express event-response, recovery,
persistence, decay, or fast-to-slow hypotheses that are not well represented
by shallow enumeration. They still require bounded YAML parameters,
deterministic generation, complexity metadata, attribution, tests, and this
same evaluation path.

Generation is deterministic for a fixed field set, template file, window set,
limit, seed, and priority policy. A configured diversity share is first
allocated across families; the remaining budget is filled by explicit
template/field/operator/window priorities. Grammar expansion uses a bounded,
stable scan rather than materializing the full Cartesian product.

### Evaluation

Data is divided into three roles:

- discovery: infer direction and discover candidates;
- validation: measure survival and rank candidates;
- holdout: final audit only.

The score does not compute or use holdout metrics. Coverage, usable days, and
turnover are also restricted to discovery plus validation. Daily RankIC uses
average ranks for ties. Robustness diagnostics include Newey-West t-statistics
and block-bootstrap confidence intervals. The default entry lag is one trading
day, so a close-based signal is never evaluated as if it could trade at that
close. Holdout metrics are computed only after the selected set is frozen.

Successful expression results are reusable across experiments only when a
strict evaluation signature matches. The signature covers the data files,
date split, evaluation settings, selected fields, field/operator versions,
and relevant source hashes.

Crypto scoring additionally uses discovery/validation-only net signed
long-short Sharpe after configured turnover costs and multi-horizon survival.
The current robust selection also requires positive discovery and validation
net return and Sharpe plus positive Sharpe in at least two of three New York
session buckets. Holdout remains audit-only.

Future exposure diagnostics will classify candidates as alpha candidates,
beta candidates, style proxies, or unclassified. Raw and neutralized evidence
must be retained: beta-like factors are useful research products, but must not
be described as pure alpha. Exposure models and classifications are also
discovery/validation-only decisions.

### Selection and pairing

First, high-scoring factors are selected with caps by template family and
primary field. Then factor pairs are formed only when their configured
similarity metric is below a threshold. For the current crypto time-series
run, similarity is median absolute within-asset time-series correlation, not
daily cross-sectional rank correlation.

Pair selection is a bridge to later probabilistic search: it produces evidence
about which variables, operators, windows, and structures are individually
strong and mutually complementary.

### Human-designed factors

Human factor design happens after search evidence exists. A proposed factor
should cite:

- useful primitive fields and their support;
- operator/window combinations that survived validation;
- complementary factors or failure modes;
- the new hypothesis added by the researcher.

It then enters the same parser, validator, evaluation, and holdout protocol as
machine-generated factors.

### Frozen full-market long-only audit

This downstream diagnostic consumes the already frozen Top50. It preserves
the discovery-period direction, executes one close after signal formation,
and evaluates equal-weight Top-K portfolios across configured rebalance
periods and holding percentages. Curves are selected independently for every
condition using the sum of total-return rank and full-period Sharpe rank.

The default report shows cumulative return and 252-day rolling annualized
Sharpe. It is a gross-return research view: transaction costs, taxes, market
impact, limit-up/limit-down execution, ST filtering, suspension handling, and
point-in-time index membership remain production-stage concerns.

## Artifact contract

Each completed experiment writes the applicable subset of:

- `manifest.json`: immutable configuration and provenance;
- `search_results.csv`: discovery/validation evidence, no holdout columns;
- `selected_factors.csv`: diversity-constrained candidates;
- `factor_pairs.csv`: complementary pair proposals;
- `holdout_audit.csv`: isolated final-period diagnostics.
- `overview.html`: aggregate search diagnostics;
- `top50.html` and `top50_metrics.csv`: selected-factor diagnostics, excluding
  holdout metrics.
- `crypto_time_series_factor_returns.html`: selected-factor cumulative curves,
  top-factor signal groups, and session diagnostics;
- `crypto_time_series_factor_pairs.html`: pair curves and complementarity;
- `crypto_time_series_backtest_metrics.csv` and
  `crypto_time_series_pair_metrics.csv`: the report's machine-readable source.

The crypto return report defaults to validation net cumulative-return display
order, with non-overlapping ranks 1-5, 6-15, and 16-20. Selection remains a
multi-objective discovery/validation decision; display ordering does not alter
selection and never uses holdout.

Completed experiment directories are evidence and must not be overwritten.
