# Template Catalog

The executable source of truth depends on the explicitly selected asset/mode
configuration. `configs/research/templates.yaml` is the generic cross-sectional
catalog; the current crypto production source is
`configs/research/templates_crypto_multiscale_sessions.yaml`.

| Template | Role |
|---|---|
| `single_state` | Learn which transformed field states carry information |
| `binary_same_state` | Compare or combine fields under the same temporal lens |
| `binary_mixed_state` | Discover asymmetric operator relationships |
| `rolling_relationship` | Measure changing pair relationships with `TsCorr` |
| `short_long_gap` | Capture acceleration, reversal, and regime gaps |
| `cross_window_state` | Compare different fields under short and long temporal lenses |
| `normalized_spread` | Scale a state difference by the combined absolute magnitude |
| `relationship_change` | Measure changes in rolling pair correlation |
| `triple_modulation` | Let a third state amplify or normalize a two-state relation |
| `quad_balanced` | Restricted balanced four-state comparison; disabled by default |

Templates define hypothesis classes, not specific economic stories. Evidence
from Attribution should update proposal probabilities for field/operator/window
slots without changing the evaluation protocol.

Every family has an explicit maximum count and complexity budget. Generation
reserves a diversity quota before filling the remaining budget with explicit
priorities. The bounded stable scan prevents large templates from expanding
the complete Cartesian product.

## Historical crypto multi-timeframe catalog

`configs/research/templates_crypto.yaml` keeps the same executable template
kinds, expands hourly windows to 4–336 hours, and enables restricted
four-field balanced expressions. Every family has explicit count and
complexity ceilings, up to depth 9 and 32 nodes. Generation remains
deterministic and uses canonical/RankIC-equivalence de-duplication.

The crypto configuration mixes price, volatility, liquidity, taker-flow, and
15-minute intrahour fields through a 65% inter-family diversity allocation.
No proposal or attribution step reads holdout metrics.

## Historical crypto time-series catalog

`configs/research/templates_crypto_timeseries.yaml` was the first preferred
catalog for the 12-contract crypto universe and is retained as historical
evidence. It does not wrap expressions in cross-sectional
`Rank`. Each formula assigns economic roles to field slots, so price, volume,
taker flow, risk, liquidity, and 15-minute microstructure are combined only in
declared hypotheses rather than an unrestricted Cartesian product.

The families cover rolling state normalization, first and second differences,
fast/slow smoothing gaps, price-volume and price-flow confirmation, volatility
and liquidity regime gates, correlation levels and changes, and micro-to-macro
transitions. Outputs are dimensionless or built from dimensionless inputs;
`TsZScore` standardizes along each contract's own history.

## Current crypto multiscale/session catalog

`configs/research/templates_crypto_multiscale_sessions.yaml` is preferred for
the 12-contract crypto universe. It contains 12 bounded time-series families:

- multiscale structure;
- session state;
- session-adjusted price/flow;
- volatility asymmetry;
- liquidity efficiency;
- flow structure;
- derivative dynamics;
- correlation regime;
- accumulation state;
- level smoothing;
- cross-scale regime;
- tri-state interaction.

The catalog combines structure, efficiency, volatility level and direction,
liquidity, flow, accumulation, 5m/15m within-hour state, and DST-aware New York
sessions. Cross-sectional root `Rank` share is 0%. Fast fields are currently
aggregated to a 1h decision panel; this catalog is not a native 5m/15m
execution claim.

## Planned structured / semi-mined tier

The first executable mechanism catalog is now
`configs/research/templates_crypto_mechanisms_v1.yaml`. It declares
downside-flow absorption, upside price/flow confirmation, DST-aware New York
session transfer, continuous-market liquidity stress, and two simple variance
baselines. Each family declares applicable target modes and a hypothesis card;
the loader fails clearly if a declared card is missing. All families still use
the existing bounded deterministic `time_series_formula` generator and appear
by template name in attribution. This catalog is an initial, bounded mechanism
study, not a claim that its candidates are production alpha. See
`docs/CRYPTO_MECHANISM_RESEARCH_DESIGN_20260928.md`.

The next catalog tier should assign explicit semantic roles to slots: fast
input, slow state, gate, confirmation/divergence, scale, persistence and
decay. It should support hypotheses such as fast-to-slow conversion,
event-intensity × regime × persistence, liquidity-shock recovery, asymmetric
volatility paths, session-conditioned accumulation, and price/flow/OI
confirmation.

These modules may be more structured than simple nested enumeration, but they
must remain bounded deterministic YAML templates, carry an economic hypothesis
card and dimensional rules, appear in attribution, and pass through the same
parser/evaluator/three-stage/holdout protocol.
