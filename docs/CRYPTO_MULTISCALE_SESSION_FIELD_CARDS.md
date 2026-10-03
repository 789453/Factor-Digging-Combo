# Crypto multiscale/session field cards

Formula version: `2026-09-26-crypto-multiscale-session-v2`. All rolling
baselines are backward-looking; no holdout metric or future observation enters
feature construction. Ratios use the framework safe division rule and replace
infinite output with missing values.

## 5m and 15m structure

| Fields | Unit | Dependencies | Formula / economic interpretation |
|---|---|---|---|
| `micro*_realized_vol` | return | intrahour log returns | Square root of summed squared returns; volatility level. |
| `micro*_upside_share` | ratio | positive and total squared returns | Positive semivariance / total variance; volatility direction. |
| `micro*_jump_share` | ratio | absolute intrahour returns | Largest absolute bar return / total absolute path; jump concentration. |
| `micro*_path_efficiency` | ratio | hourly displacement, intrahour path | Absolute open-close log return / absolute return path; structural efficiency. |
| `micro*_taker_imbalance` | ratio | volume, taker-buy ratio | Signed taker volume / volume. |
| `micro*_volume_trend` | ratio | intrahour volume | (late-half − early-half volume) / total volume. |
| `micro*_imbalance_trend` | ratio | signed taker volume | (late-half − early-half signed volume) / total volume. |
| `micro*_volume_hhi`, `micro*_trade_hhi` | ratio | volume or trade-count shares | Sum of squared bar shares; liquidity concentration. |
| `micro5_15_*_gap` | ratio | corresponding 5m and 15m measures | Cross-resolution difference; detects whether structure exists only at the finest scale. |

`micro*` denotes 15m and `micro5*` denotes 5m. Aggregation is vectorized by
symbol-hour; expected complete counts are four and twelve bars respectively.

## Volatility direction and efficiency

| Fields | Unit | Dependencies | Formula / economic interpretation |
|---|---|---|---|
| `upside_vol_24h`, `downside_vol_24h` | return | hourly returns | Root mean positive/negative semivariance over 24h. |
| `volatility_direction_24h` | ratio | upside/downside volatility | (up − down) / (up + down), bounded under finite inputs. |
| `return_skew_24h` | dimensionless | hourly returns | Causal rolling skew; asymmetric tail direction. |
| `liquidity_efficiency` | return/log-count | return, trade count | Absolute return / log(1 + trades); price movement per unit activity. |

## DST-aware session fields

The timestamp is converted from UTC to `America/New_York`, so daylight-saving
transitions are handled by the timezone database. Sessions are mutually
exclusive: overnight 00:00–07:59, US day 08:00–15:59, evening 16:00–23:59.

| Fields | Unit | Dependencies | Formula / economic interpretation |
|---|---|---|---|
| `us_*_flag` | binary | timestamp | Ex-ante session regime; never inferred from returns. |
| `ny_hour_*`, `weekday_*` | dimensionless | timestamp | Sine/cosine cyclic encoding, avoiding an artificial boundary. |
| `session_cumulative_return` | return | open, close, session key | Close / first session open − 1. |
| `session_progress` | ratio | session key | Hour index / 7 in the current 8h session. |
| `session_cumulative_volume_ratio` | ratio | quote volume | Cumulative session volume / causal 24h expected volume at the same progress. |
| `session_*_surprise` | z-score | volume/range/illiquidity/flow | Current value standardized using only the prior 10–60 observations at the same NY local hour. |

Missing/degenerate history produces NaN, not zero. Extreme zero volume/trade
count and infinite source prices are sanitized before the search panel.
