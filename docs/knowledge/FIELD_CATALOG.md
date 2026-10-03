# Base Field Catalog

Formula version: `2026-09-05-v3`

The executable source of truth is `FIELD_SPECS` in `fields.py`. This document
explains the research roles.

## Default search vocabulary

| Group | Fields |
|---|---|
| Price state | `ret_1d`, `hl_range`, `oc_ret`, `close_pos`, `gap_ret`, `intraday_reversal` |
| Liquidity | `amount_log`, `vol_log`, `turnover_log` |
| Price/liquidity interaction | `price_volume_pressure`, `amplitude_turnover`, `volume_ratio_x_ret` |
| Order flow | `sm_net_ratio`, `md_net_ratio`, `lg_net_ratio`, `main_net_ratio`, `retail_pressure`, `big_vs_small_flow`, `flow_imbalance`, `large_order_intensity`, `active_big_buy_pressure` |
| Chip distribution | `chip_width_90`, `chip_width_70`, `chip_cost_bias`, `chip_median_bias`, `chip_upper_pressure`, `chip_lower_support`, `winner_rate_norm`, `hist_price_position`, `winner_cost_divergence` |

## Added research vocabulary (v2)

| Group | Fields |
|---|---|
| Return/volatility state | `ret_5d`, `ret_20d`, `residual_ret_1d`, `realized_vol_10`, `realized_vol_20`, `downside_vol_20`, `trend_efficiency_20` |
| Liquidity change/impact | `vol_change_5d`, `amount_change_5d`, `turnover_change_5d`, `amihud_20`, `liquidity_shock_20`, `range_vol_ratio` |
| Flow persistence/divergence | `main_flow_persistence_5`, `flow_agreement`, `main_return_divergence` |
| Chip interactions | `chip_pressure_balance`, `chip_concentration`, `winner_position_interaction` |
| Session structure | `overnight_intraday_spread` |

These stock-oriented fields are intended to describe a state before generic
operators and templates are applied.

## Futures open-interest vocabulary (v3)

| Group | Fields |
|---|---|
| OI change | `oi_change_1d`, `oi_change_5d`, `oi_change_acceleration_5`, `oi_relative_20` |
| OI participation | `oi_volume_ratio`, `oi_turnover_pressure` |
| Price/OI interaction | `oi_price_divergence`, `oi_return_agreement` |

These eight fields are available only when the data source supplies
`open_interest`.  They are rates, ratios, interactions, or rolling z-scores;
absolute open interest is deliberately not a search field.  The rapid futures
experiment uses continuous/main commodity contracts because the supplied index
continuous series has sparse OI and volume coverage.

## Built but disabled search fields

- `upper_shadow`, `lower_shadow`: potentially redundant with range and close
  position; require incremental-value evidence before activation.
- `amount_per_vol`: average-price proxy with unit sensitivity.

## Controls, not default search inputs

- `size_log`
- `free_turnover_gap`
- `liquidity_crowding`

Controls must be explicitly enabled if used as search inputs. Their primary
role is exposure diagnosis, neutralization, or tradability filtering.

## Change rule

Crypto structured state field: [`cluster_stress_affinity`](../cards/crypto_cluster_stress_affinity_FIELD.md) is available only with explicit `cluster_state.enabled` in crypto time-series research. Its independent formula version and discovery-fitted metadata are recorded with each run.

Do not edit only `DEFAULT_FEATURES`. Add or update `FieldSpec`, implement the
formula, update the field card, bump `FIELD_FORMULA_VERSION`, and add numerical
tests. A formula change under the same field name is a breaking research-data
change.
