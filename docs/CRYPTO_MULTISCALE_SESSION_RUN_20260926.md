# Crypto multiscale/session research run — 2026-09-26

## Preserved evidence

- Previous accepted run remains immutable at
  `outputs/crypto_timeseries_10000_final_20260926`.
- The first 30k run remains immutable at
  `outputs/crypto_multiscale_sessions_30000_20260926`.
- The final robust selection is at
  `outputs/crypto_multiscale_sessions_30000_robust20_20260926`.
- The active reproducible configuration is
  `configs/research/crypto_multiscale_sessions_30000.yaml`.

The final manifest SHA-256 is
`2987CE6BE65420D6DBC0C9CF3ACBB7F2680E0FF5A8F4F1F2EC3DCDDD4E8F05C3`.
The selected-factor CSV SHA-256 is
`D04EAE5C2A5118062DB4069AC3EBE0E6010BA572C152DF686F6B140BC8F64C70`.

## Research design

- Universe: 12 crypto contracts; 1h decision panel enriched by vectorized 15m
  and 5m within-hour aggregates.
- Search dates: 2023-01-01 through 2026-08-31.
- Discovery ends 2024-12-31; validation is 2025-01-01 through 2026-03-31;
  holdout starts 2026-04-01 and is excluded from selection.
- Signal mode: per-asset time series. Cross-sectional `Rank` share is 0%.
- Portfolio: signed long-short continuous positions, four-hour overlapping
  holdings, one-hour signal lag convention, 4 bps per unit turnover. It is
  not a long-only backtest.
- Search: 30,000 generated; 29,852 computed successfully; 2,400 medium-stage;
  650 full fine-stage; 34 robust eligible; 20 selected; 10 pairs.
- Robust final eligibility requires positive discovery and validation net
  return and Sharpe, at least two of three NY-session strategy Sharpes above
  zero, IC sign survival and multi-horizon survival.

## Field dimensions

The field catalog explicitly covers structure, efficiency, volatility level,
volatility direction/asymmetry, liquidity, flow, accumulation and DST-aware
New York sessions. Session surprises compare each observation only with prior
observations at the same NY local hour. The detailed formulas and missing-data
rules are in `docs/CRYPTO_MULTISCALE_SESSION_FIELD_CARDS.md`.

## Final result summary

- All 20 selected factors have positive discovery and validation net returns
  and Sharpes.
- Validation net return across the 20 ranges from 0.18% to 94.97%; validation
  Sharpe ranges from 0.152 to 2.055. Means are 31.19% and 0.883.
- Discovery net return ranges from 0.96% to 48.80%; discovery Sharpe ranges
  from 0.144 to 0.973. Means are 17.22% and 0.465.
- Selected directions: 15 negative orientations and 5 positive orientations;
  direction was learned from discovery only.
- Template coverage: session-adjusted price/flow 6, correlation regime 6,
  liquidity efficiency 3, derivative dynamics 2, cross-scale regime 2 and
  session state 1.
- Top validation factor:
  `TsZScore(TsDelta(TsCorr($ret_24h,$realized_vol_24h,720),12),720)`;
  discovery return/Sharpe 25.44%/0.632 and validation 94.97%/2.055.
- Ten low-similarity pairs have validation Sharpe from 1.19 to 2.44. Pair
  similarity is the median absolute within-asset time-series correlation, not
  a cross-sectional daily rank correlation.

## 5m conclusion and limitations

5m data was not omitted: 5m semivariance, jump share, path efficiency, taker
imbalance, early/late flow and volume, HHI concentration, and 5m–15m gaps were
all searched. Two 5m expressions survived into the 34-factor robust eligible
pool but did not enter the score/diversity-constrained final 20. They remain in
`search_results.csv`; forcing them into the final set would weaken the stated
selection rule.

The selected set is concentrated in 720h correlation dynamics. That is an
empirical result but also a model-risk warning: the requested first pass does
not perform a full neighbouring-window sensitivity sweep. Holdout metrics are
reported only after selection and must not be used to retune this run.

## Performance engineering

- 5m/15m data is aggregated by vectorized symbol-hour operations.
- Rolling correlation and standard deviation use sliding sufficient statistics
  in `O(T*N)` rather than rescanning every window in `O(T*W*N)`.
- Per-asset time IC uses matrix reductions.
- Numba parallelism remains inside numerical kernels; expression-level threads
  are intentionally not stacked on top, avoiding oversubscription and large
  matrix duplication.
- A fingerprinted Zstandard Parquet feature cache avoids repeating the roughly
  4.7-million-row 5m preparation on restart.
- DuckDB checkpoints are written in batches of 250.

Verification: 33 tests passed, including rolling-kernel reference equality,
missing/extreme fields, 5m/15m bar aggregation and the holdout firewall.
