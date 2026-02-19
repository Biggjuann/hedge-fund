# Validation Protocol (No exceptions)

## Walk-forward design (default)
- Rolling:
  - Train: 2–5 years (depending on timeframe)
  - Test: 3–6 months
  - Step: 1–3 months

## Parameter selection
- Choose stable regions:
  - evaluate top-k parameter sets
  - pick cluster center / median
  - penalize turnover + complexity

## Leakage controls
- Features at time t must only use info <= t (or <= t-1) per documented convention.
- For ML / event labeling:
  - Purge overlapping label horizons
  - Embargo buffer after test fold

## Multiple testing reporting
- Log number of trials per strategy
- Output DSR/PBO-style proxy diagnostics if exact methods not implemented

## Stress tests
- slippage x2, x3
- 1 tick adverse on all fills
- delayed fills (next bar)
- random trade drop (simulate missed signals)
- correlation spike scenario (risk haircuts)
- margin shock scenario (reduced leverage cap)

## Regime-conditioned OOS
Report metrics broken down by:
- vol regime: low/med/high (quantiles)
- corr regime: low/high (quantiles of avg pairwise corr)
- trend regime: ADX proxy bucket
- macro tags (evaluation only if available)
