# Operator Catalog

The executable source of truth is `OPERATOR_SPECS` in `ops.py`.

## Preferred search operators

- Cross-section: `Rank`
- Safe shape transform: `Abs`, `SLog1p`
- Rolling state: `TsMean`, `TsStd`, `TsIr`, `TsRank`, `TsDelta`, `TsEMA`, `TsWMA`
- Composition: `Add`, `Sub`, guarded `Mul`, guarded `Div`
- Relationship: `TsCorr`

## Restricted operators

`Log`, `Inv`, `Pow`, `Div`, `TsDiv`, `TsPctChange`, and `TsCov` have explicit
domain or scale risks. A template must justify their use and rely on Validator
and runtime safeguards.

`Greater` and `Less` mean element-wise maximum and minimum. They are not
boolean gates. Future conditional structures must introduce an explicitly
named gate operator rather than overloading these semantics.

## Crypto time-series operators

- `TsZScore(x,w)` is `(x - rolling_mean) / rolling_std` along each asset's
  own history. It requires the current value, at least `floor(w/2)+1` finite
  observations, and strictly positive rolling standard deviation; otherwise
  it returns NaN. It is invariant to positive affine rescaling of `x`.
- `GatePos(state,signal)` returns `signal` where `state > 0`, else zero.
- `GateNeg(state,signal)` returns `signal` where `state < 0`, else zero.

The gate comparison is strict: a zero or NaN state produces zero. Gates are
ordered, non-commutative operators. They are intended for explicit regime
conditioning, not as a substitute for cross-sectional selection.

## Rank semantics

Cross-sectional Rank, rolling Rank, and RankIC use average ranks for ties.
Changing tie semantics changes factor values and requires an operator-version
change plus equivalence tests.

## Acceleration rule

Every accelerated implementation must match a transparent reference
implementation on:

- finite random data;
- NaN patterns;
- constant vectors;
- tied values;
- zero denominators;
- minimum-window boundaries.
