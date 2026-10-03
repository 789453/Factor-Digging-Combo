# Crypto multi-timeframe factor run — 2026-09-25

## Scope and outcome

- Universe: 12 crypto assets.
- Inputs: native 1-hour bars plus exactly four aligned 15-minute bars per hour.
- Sample: 2023-01-01 through 2026-08-31 (UTC).
- Search space: 10,000 deterministic expressions from 10 bounded template
  families, 46 scale-free fields, and 4–336 hour windows.
- Coarse stage: 10,000 expressions; 9,960 evaluated successfully.
- Fine stage: 480 discovery/validation-consistent, diversity-capped survivors.
- Frozen research eligibility: 39 factors.
- Final diversified set: 20 factors from four surviving template families and
  six primary fields; 10 low-similarity pairs were also retained.

The final evidence is in
`outputs/crypto_multitimeframe_10000_two_stage_20_20260925`. The earlier
13-factor output is retained unchanged as immutable evidence of the stricter
family/field caps.

## Performance interpretation

The strongest selected expression by the combined research score is:

`Rank(TsCorr($residual_ret_24h,$downside_vol_24h,336))`

Its validation mean rank IC is 0.0280, validation ICIR is 0.0698, worst
validation IC across the 1/4/12-hour horizons is 0.0182, and the cost-aware
validation Sharpe is 0.763. The highest selected validation mean rank IC is
0.0373 for a 168-hour normalized spread between micro path efficiency and the
hourly high-low range.

The selected set is concentrated in economically interpretable clusters:

- residual momentum versus downside volatility;
- taker-flow imbalance versus residual return;
- micro path efficiency versus candle range/downside volatility;
- volatility term structure versus residual or raw 24-hour momentum;
- intrahour volume trend and trade concentration.

This is a useful cross-sectional signal result, but not yet a production
strategy. Although IC direction survives discovery and validation and the
three requested horizons, hourly top-minus-bottom trading with a 4 bps
turnover charge is weak in much of the discovery period. The positive
validation Sharpe gate therefore identifies recent-regime candidates; it does
not establish full-sample cost robustness. Funding, exchange-specific fees,
slippage, capacity, missing venues, and delisting bias are not modeled.

## Speed design

The old single-stage run processed roughly five expressions per second because
every candidate paid for bootstrap, Newey-West, three extra horizon checks,
portfolio construction, and a database write. The new run used:

1. matrix/Numba expression operations on the full time-by-asset panel;
2. cache-local ordering of related templates;
3. a cheap primary-horizon IC coarse screen for all 10,000 expressions;
4. batch DuckDB checkpoints every 200 expressions;
5. full statistics only for 480 survivors (a 95.2% reduction in expensive
   evaluations);
6. evaluation-signature-safe reuse for report/selection changes.

Observed coarse throughput started near 17 expressions/second and averaged
13.9 expressions/second as the search moved into deeper templates. Fine-stage
throughput was about 4.2 expressions/second. Sensitivity beyond the configured
1/4/12-hour check is intentionally deferred.

## Key artifacts

- `selected_factors.csv`: the 20 frozen expressions and research metrics.
- `crypto_selected_backtest_metrics.csv`: cost-aware return, Sharpe, drawdown,
  and turnover summaries.
- `crypto_factor_returns.html`: interactive cumulative net-return curves for
  the leading validation-Sharpe factors, with full expressions.
- `crypto_group_returns.html`: validation quartile heatmap and ensemble group
  cumulative-return curves.
- `search_results.csv`: all 10,000 coarse rows plus the 480 fine-stage rows.
- `holdout_audit.csv`: holdout metrics calculated only after selection froze.
- `manifest.json`: data fingerprints, formula/operator versions, configuration,
  and run counts.
