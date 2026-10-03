# Field Card: `oi_change_acceleration_5`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `oi_change_1d`; Formula: current OI change minus its trailing five-day mean.
- Timestamp availability: current and historical end-of-day values only.
- Unit/range: dimensionless rate residual; fewer than three observations gives `NaN`.
- Hypothesis: accelerating position formation contains incremental information beyond OI trend.
- Failure modes: irregular holidays and rolling contracts reduce comparability.
