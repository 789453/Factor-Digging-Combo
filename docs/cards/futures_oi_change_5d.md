# Field Card: `oi_change_5d`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `open_interest`; Formula: \(OI_t/OI_{t-5}-1\).
- Timestamp availability: end-of-day; all inputs are at or before the signal date.
- Unit/range: dimensionless; missing/zero denominator and infinities are `NaN`.
- Hypothesis: multi-day positioning trend is more persistent than a single OI print.
- Failure modes: main-contract switches can introduce discontinuities.
