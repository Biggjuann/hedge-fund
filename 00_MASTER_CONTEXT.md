# MASTER CONTEXT — Regime-Robust Systematic Futures System

## Objective
Build a production-grade systematic futures program that survives regime changes by combining:
1) multi-strategy ensembles (trend + carry + spreads + convex/vol breakout)
2) conservative, correlation-aware risk scaling + margin realism
3) disciplined validation (walk-forward, purged/embargoed CV, multiple-testing controls)
4) realistic data + roll mechanics + execution/cost modeling

## Primary knowledge sources in this repo
- 01_DEEP_RESEARCH_PAPER.md (authoritative design + evidence + workflow)
- 02_BOOK_SYNTHESIS.md (Page, Harvey/Rattray/Van Hemert, Spitznagel, Oppenheimer, Pedersen, Papic, Lopez de Prado)
- 03_SYSTEM_REQUIREMENTS.md (engineering requirements)
- 04_STRATEGY_LIBRARY.md (concrete strategy definitions)
- 05_VALIDATION_PROTOCOL.md (WFO + robustness suite + leakage rules)
- 06_RISK_PORTFOLIO_EXECUTION.md (risk overlays, sizing, execution realism)
- 07_DATA_MODEL.md (contracts, rolls, continuous vs tradable series)

## Non-negotiables
- Portfolio of distinct return drivers (do not ship single strategy).
- OOS regime robustness is required, not implied.
- No-leakage features (strict causality).
- Costs/slippage/margin shocks must be stress-tested.
- Tail/drawdown controls must exist (kill switch + de-risk ladder).

## Deliverables
1) Research pipeline: data → signals → backtest → WFO → robustness reports
2) Strategy library: trend/TSMOM, carry, spread mean reversion, vol breakout/convex proxy
3) Portfolio allocator + risk overlays: vol target + correlation haircuts + ERC/risk parity option
4) Execution/cost modeling: spread + slippage + impact + participation caps (regime-aware)
5) Outputs: standardized JSON/CSV reports + regime breakdowns
