# Field Card: `oi_relative_20`

- Status: active; Formula version: `2026-09-05-v3`; Group: futures OI.
- Raw dependencies: `open_interest`; Formula: trailing 20-day z-score of \(\log(1+OI)\).
- Timestamp availability: current and preceding 19 end-of-day observations only.
- Unit/range: standardized and dimensionless; negative OI, zero variance, and missing values yield `NaN`.
- Hypothesis: unusually high or low positioning relative to a contract's own history changes risk appetite.
- Failure modes: a main-contract roll can reset the local OI level.
