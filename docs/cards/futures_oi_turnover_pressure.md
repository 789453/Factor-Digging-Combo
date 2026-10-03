# Field Card: `oi_turnover_pressure`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `oi_volume_ratio`, `oi_change_1d`; Formula: \((Volume/OI)\times\Delta OI\%\).
- Timestamp availability: derived solely from same-day and lagged EOD data.
- Unit/range: dimensionless interaction; invalid components propagate `NaN`.
- Hypothesis: position expansion with fast contract turnover is stronger than either component alone.
- Failure modes: highly leveraged event days can dominate raw values before rank transforms.
