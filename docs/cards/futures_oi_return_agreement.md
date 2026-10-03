# Field Card: `oi_return_agreement`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `ret_1d`, `oi_change_1d`; Formula: \(r_t\times\Delta OI_t\%\).
- Timestamp availability: daily close, entered no earlier than the next trading day.
- Unit/range: product of dimensionless rates; invalid inputs propagate `NaN`.
- Hypothesis: price and open-interest agreement proxies for conviction in the move.
- Failure modes: negative products conflate short build-up and long liquidation.
