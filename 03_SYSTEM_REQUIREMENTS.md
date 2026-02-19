# System Requirements (Engineering)

## Universe (v1)
Start: ES/MES, NQ/MNQ, RTY/M2K
Then expand: CL, GC, ZN, 6E (and more) once pipeline is stable.

## Data granularity
- For intraday: 5m/15m
- For regime + macro features: 1h/4h/1D
Must support multi-timeframe features without lookahead.

## Must-implement strategies (v1)
1) Trend/TSMOM (multi-horizon ensemble)
2) Carry / term structure (cross-sectional)
3) Vol breakout / convex proxy (risk-off convexity response)
4) Optional (phase 2): Spread mean reversion (strict constraints)

## Portfolio & risk
- Per-strategy vol targeting + portfolio vol targeting
- Correlation-aware scaling (haircuts when corr spikes)
- Leverage caps + margin buffers
- Drawdown de-risk ladder + kill switch + cooldown rules

## Validation deliverables
- Walk-forward optimization (rolling windows)
- Regime-conditioned OOS reporting
- Leakage prevention tests
- Multiple-testing diagnostics report
- Stress tests suite

## Output artifacts
- Standard results JSON + CSV
- Fold-by-fold WFO summaries
- Regime breakdown tables
- Cost attribution report
