# Strategy Library (Implementable Specs)

## Common helpers
- Vol estimator: EWMA variance with 60d center of mass (daily) or equivalent in bars.
- Position sizing baseline: target risk per instrument via 1/vol.
- Rebalance cadence: monthly default unless intraday strategy.

---

## S1: Multi-horizon Trend (TSMOM ensemble)
Inputs:
- lookbacks: [21, 63, 252] trading days (or bar-equivalents)
Signal per horizon:
- s_h = sign( return over lookback )
Combine:
- s = average(s_h) then clamp to [-1, +1]
Sizing:
- w_i = s_i * (target_vol / vol_i)
Portfolio vol target applied after aggregation.

---

## S2: Cross-asset Carry (term structure)
Inputs:
- carry measure per instrument (contract-level curve metric)
Cross-sectional:
- rank carry; long top quantile, short bottom quantile
Weights:
- proportional to rank distance, scaled by 1/vol
Rebalance:
- monthly

Implementation rule:
- returns computed on tradable contract rolls, NOT interpolated trading.

---

## S3: Volatility Breakout / Convex Proxy
Purpose:
- provide crisis-like convex response without options (proxy).
Signal:
- range compression then expansion (e.g., breakout above N-day range) OR ATR percentile regime break.
Sizing:
- small baseline risk; scale up only under high-vol regime with correlation spike haircuts.
Risk:
- tight max loss per trade; avoid "martingale".

---

## S4 (Phase 2): Spread Mean Reversion (calendar spreads only)
Universe constraint:
- only pre-approved spreads (e.g., CL1-CL2, etc.)
Signal:
- zscore of spread vs rolling mean/std
Entry:
- abs(z) >= entry_z
Exit:
- abs(z) <= exit_z
Risk:
- strict stop based on spread ATR; structural break detection; liquidity filters.
