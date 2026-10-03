# Field Card: `oi_volume_ratio`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `vol`, `open_interest`; Formula: \(Volume_t/OI_t\).
- Timestamp availability: daily close; evaluated with a one-day entry lag.
- Unit/range: contract-to-contract ratio, hence dimensionless; nonpositive/missing OI yields `NaN`.
- Hypothesis: unusually rapid turnover of outstanding positions signals repricing pressure.
- Failure modes: exchange-specific volume conventions may not be comparable.
