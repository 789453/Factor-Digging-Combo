# Field Card: `oi_change_1d`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `open_interest`; Formula: \(OI_t/OI_{t-1}-1\).
- Timestamp availability: end-of-day vendor field; used with one-day entry lag.
- Unit/range: dimensionless return; zero denominator, missing, and infinities become `NaN`.
- Hypothesis: fresh positioning can distinguish price moves with participation from exhaustion.
- Failure modes: contract rolls and exchange reporting revisions can create jumps.
