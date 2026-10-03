# Crypto Field Cards

Formula version: `2026-09-25-crypto-v1`. Owner/date: crypto research graft,
2026-09-25. All fields are active search inputs, available only after the
hourly candle closes, and executed with at least one full hourly-bar lag.
Division uses a protected denominator; infinities become missing. Rolling
statistics require at least half of their window. No field uses a later bar.

## Price, trend, and risk

| Fields | Raw dependencies | Formula / unit | Hypothesis and failure mode |
|---|---|---|---|
| `ret_1h`, `ret_4h`, `ret_24h`, `ret_168h` | close | lagged percentage returns | momentum/reversal across intraday to weekly scales; jumps dominate tails |
| `residual_ret_1h`, `residual_ret_24h` | asset and universe returns | asset minus same-hour universe mean | relative strength; unstable if one constituent dominates |
| `hl_range`, `oc_ret`, `close_pos` | OHLC | scale-free candle range/body/location | intrabar pressure; bad ticks and flat bars cause missing/extremes |
| `vwap_bias`, `range_body_ratio` | OHLC/VWAP | close/VWAP bias and body/range | traded-consensus deviation and directional candle quality |
| `realized_vol_4h`, `realized_vol_24h`, `realized_vol_168h` | 1h return | rolling population standard deviation | volatility regime; overlapping windows are persistent |
| `downside_vol_24h` | negative 1h return | root mean negative squared return | downside-specific risk; sparse in uninterrupted rallies |
| `trend_efficiency_24h`, `trend_efficiency_168h` | close | net displacement / absolute return path | persistent trend versus noise; gaps may inflate it |
| `short_long_momentum_gap` | 4h/24h return | short minus long return | acceleration/reversal; horizons overlap |
| `volatility_term_slope` | 4h/24h volatility | ratio minus one | short-vol shock versus daily regime |

## Liquidity and taker flow

| Fields | Raw dependencies | Formula / unit | Hypothesis and failure mode |
|---|---|---|---|
| `volume_ratio_24h`, `volume_ratio_168h` | quote volume | current / rolling mean | abnormal activity; exchange regime changes matter |
| `volume_shock_24h` | quote volume | 24h z-score of log volume | normalized volume shock; zero variance is missing |
| `trade_count_shock_24h` | trade count | 24h z-score of log count | participation shock; aggregation rules may change |
| `avg_trade_size_shock_24h` | quote volume/count | 24h z-score of log average size | whale/retail mix proxy, not wallet-level flow |
| `amihud_shock_24h` | absolute return/quote volume | 24h z-score of log scaled impact | liquidity stress; very low-volume bars are unstable |
| `taker_imbalance` | taker-buy ratio | `2*buy_ratio-1` | aggressive buy/sell pressure |
| `taker_imbalance_change_4h`, `taker_imbalance_mean_4h`, `taker_imbalance_mean_24h` | taker imbalance | difference and rolling means | flow acceleration/persistence; noisy in thin hours |
| `taker_price_agreement`, `taker_return_divergence` | imbalance/return | product and difference | confirmation versus absorption; empirical scales differ |
| `taker_micro_divergence` | hourly and 15m imbalance | hourly minus micro aggregate | late/aggregate mismatch; requires exact bar alignment |

## Within-hour 15-minute microstructure

Each field uses exactly four 15-minute bars matched to one hourly bar. Missing
micro bars remain missing and are never silently filled.

| Fields | Formula / unit | Hypothesis and failure mode |
|---|---|---|
| `micro_first_return`, `micro_last_return`, `micro_return_reversal` | first, last, and last-minus-first 15m return | opening/late impulse and intrahour reversal; single-bar noise |
| `micro_realized_vol`, `micro_volatility_ratio` | root sum squared 15m returns; divided by 4h vol | current-hour volatility shock |
| `micro_path_efficiency` | hourly displacement / absolute 15m path | clean trend versus choppy path |
| `micro_volume_concentration`, `micro_volume_trend` | max share; second-half minus first-half balance | burst concentration and late participation |
| `micro_taker_imbalance`, `micro_imbalance_trend` | signed 15m volume ratios | aggressive flow and late rotation |
| `micro_flow_price_agreement` | micro imbalance × last return | late flow confirmation |
| `micro_trade_concentration` | max 15m trade-count share | clustered participation |
| `micro_range_expansion` | last 15m range / hourly mean | late range expansion |
| `micro_vwap_dispersion` | standard deviation of 15m VWAP / hourly close | intrahour price dispersion |

Coverage and extreme-value evidence are recorded in immutable experiment
artifacts. Search evidence and holdout evidence remain separate.
