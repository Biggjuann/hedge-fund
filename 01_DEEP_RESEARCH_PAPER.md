# Deep Research Paper — Systematic Futures Strategies That Survive Across Market Regimes

## Core thesis
Regime-robust futures programs share:
- breadth across liquid futures markets
- conservative risk scaling (vol targeting + leverage caps + correlation awareness)
- validation discipline (walk-forward + OOS-by-regime + leakage controls)

## Strategy families (required building blocks)

### A) Trend / Time-Series Momentum (TSMOM)
**Signal**
- Position sign = sign(lookback return) per instrument
- Use multiple horizons (e.g., 1M, 3M, 12M) as an ensemble

**Sizing**
- Ex-ante risk scaling using EWMA volatility (60d center-of-mass baseline)
- Apply vol(t-1) to returns(t) (strict causality)

**Rebalance**
- Monthly baseline (faster increases turnover & cost sensitivity)

**Strength**
- Directional optionality (long or short)
- Diversified across asset classes

**Failure mode**
- Whipsaws and sharp turning points
- Correlation spikes reduce independent bets

---

### B) Carry / Term Structure
**Signal**
- Cross-sectional long high-carry / short low-carry
- Carry inferred from term structure; careful: use contract-level tradable returns
- Monthly rebalance baseline

**Strength**
- Different economic driver than trend
- Often diversifies trend

**Failure mode**
- Tail "carry unwind" / crash-like behavior
- Policy/inventory distortions can change curve dynamics

---

### C) Mean Reversion in Spreads (calendar + relative value)
**Expression**
- Prefer spreads (calendar spreads, inter-commodity spreads) vs outright mean reversion
- Hedge ratio: regression or dynamic (Kalman/filter) — dynamic adds model risk

**Signal**
- Z-score / band deviation (e.g., Bollinger-style)
- Typical entry ±2, exit near 0 (parameters must be WFO-stable)

**Strength**
- Reduced directional beta
- Microstructure/seasonality anchors can persist

**Failure mode**
- Structural breaks, squeezes, liquidity cliffs
- Tail risk under stress; legs correlate more when you least want them to

---

### D) Volatility targeting / Vol-managed overlay (risk control, not alpha)
**Mechanic**
- Scale exposures by inverse realized variance / volatility forecast
- Use caps/leverage constraints

**Strength**
- Reduces exposure in high-vol drawdown regimes
- Improves risk-adjusted consistency

**Failure mode**
- Can increase leverage in "calm before storm"
- Correlation spikes matter; need correlation-aware risk model

---

### E) Multi-premia ensemble
Blend: trend + carry + defensive/convex + (careful) spreads.
Treat "paper Sharpe" results skeptically unless tradable roll + costs + fees are modeled.

---

### F) Regime-aware meta-layer (soft gating)
**Purpose**
- Do NOT hard switch strategies ON/OFF with a classifier.
- Use regime probabilities / continuous "risk state" features to:
  - scale risk budgets
  - cap leverage
  - apply correlation haircuts
  - slow execution / reduce participation

**Models**
- HMM / Markov-switching / clustering as *inputs*, not dictators.

---

## Regime definition (what matters in practice)
Core latent dimensions:
1) Volatility level + change (realized and implied proxies)
2) Correlation regime (cross-market clustering, risk-on/off)
3) Liquidity/funding regime (spreads, depth, OI changes)
4) Macro cycle regime (for evaluation tagging; not real-time hard switching)

---

## Data & engineering requirements

### Contract truth vs continuous
Maintain three layers:
1) Contract-level truth (each expiry)
2) Research continuous series (for stable features)
3) Tradable simulated series (your exact roll + execution assumptions)

### Roll rules
- Calendar roll windows OR liquidity-based rolls (explicit, unit-tested)
- Avoid delivery/first notice windows unless explicitly modeled

### Metadata you must store
multiplier, tick size/value, currency, hours, last trade, first notice, delivery type, roll schedule.

---

## Cost model (must be explicit)
Decompose:
1) fees/commissions
2) bid-ask spread
3) slippage + nonlinear impact (size vs liquidity)
4) financing/margin constraints + collateral yield mechanics

Execution must use participation caps and widen schedules in low liquidity/high vol.

---

## Validation: what "robust" means operationally
- Walk-forward optimization (rolling windows) answers: "does it survive after being tuned in a different regime?"
- If ML or extensive tuning: nested CV + purging + embargo
- Multiple-testing realism: track number of trials and use DSR/PBO/Reality Check/SPA-style diagnostics
- OOS-by-regime is mandatory: performance conditional on vol/corr/macro tags

Robustness suite:
- parameter sensitivity maps
- crisis replay
- slippage x2/x3, delayed fills, missed trades
- stationary/bootstrap resampling preserving dependence

---

## Recommended build sequence (v1)
1) Diversified multi-horizon trend (vol-scaled, conservative roll)
2) Add cross-asset carry (monthly, tradable returns)
3) Add portfolio risk layer: vol target + correlation-aware allocation (ERC option) + leverage/margin buffers
4) Add spreads mean reversion only in well-defined microstructures; require structural-break stress testing
5) Add soft regime gating only as risk tilts; validate with purged/embargoed method and overfit diagnostics
