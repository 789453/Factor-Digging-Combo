# Field Card: `oi_price_divergence`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `ret_1d`, `oi_change_1d`; Formula: \(r_t-\Delta OI_t\%\).
- Timestamp availability: end-of-day; no forward data is used.
- Unit/range: difference of two dimensionless rates; invalid inputs produce `NaN`.
- Hypothesis: price action unsupported by position change may revert or reflect short covering.
- Failure modes: price and OI timing can differ across venues.
