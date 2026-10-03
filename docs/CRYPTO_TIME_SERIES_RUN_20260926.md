# Crypto time-series factor run — 2026-09-26

## Return audit

The earlier CSV and cumulative-return chart appeared inconsistent because the
CSV ranked `validation_net_total_return`, while the chart compounded discovery
and validation together. A top-30 factor audit reproduced every CSV validation
return with maximum absolute error `9.02e-17`. The return engine was
deterministic; the plotted period and ranking period were inconsistent.

The time-series report fixes this structurally: validation-return ranking and
the plotted cumulative curves use the same validation mask. Discovery returns
are separate columns.

## Data limitation

The supplied 15-minute and 1-hour parquet files contain OHLCV, quote volume,
trade count, taker buy/sell volume, taker ratio, VWAP, and candle fields. They
contain no open interest, funding rate, mark/index basis, or liquidation data.
No OI proxy was fabricated. Genuine OI requires an aligned source, field cards,
missing/extreme tests, and formula metadata.

## Time-series design

- Signals are standardized along each asset's own history; no selected
  expression starts with cross-sectional `Rank`.
- Temporal IC is calculated within each asset and fine evaluation uses
  non-overlapping 168-hour blocks.
- Positions are signed: positive signals are long and negative signals are
  short. The portfolio is not long-only and may be net long or net short.
- Four overlapping one-hour tranches align the holding period with the
  four-hour forecast horizon.
- Turnover is charged at 4 bps per unit of signed position change.

Templates include rolling z-scores, EMA/WMA smoothing, first/second
differences, fast/slow gaps, price-volume and price-flow confirmation,
risk/liquidity gates, correlation dynamics, and 15-minute-to-hourly state
transitions.

## Final experiment

- Deterministic upper bound after deduplication: 8,993 expressions.
- Successfully evaluated: 8,978.
- Fine evaluation: 600 expressions.
- Passed IC, horizon, positive validation Sharpe, and positive validation
  cumulative-return gates: 65.
- Final diverse set: 20 factors and 10 low-similarity pairs.
- Direction: 13 positive-orientation and 7 negative-orientation factors.
- Coverage: six concrete time-series families and ten primary fields.
- Nine selected factors are profitable in both discovery and validation. The
  other eleven have discovery-period regime weakness and belong on a watchlist,
  not directly in production.

The highest validation-return factor is
`TsZScore(TsCorr($realized_vol_24h,$downside_vol_24h,336),336)`, with 97.52%
validation cumulative net return and validation Sharpe 1.954, but negative
discovery performance. A stronger two-period candidate is
`TsZScore(TsCorr($micro_return_reversal,$trend_efficiency_24h,72),72)`, with
7.03% discovery return, 76.09% validation return, and validation Sharpe 2.065.

These remain research results. Funding, venue-specific fees, slippage,
capacity, liquidation mechanics, contract changes, and survivorship remain
outside the model.
