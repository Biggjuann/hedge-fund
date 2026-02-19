# Book Synthesis — Principles that constrain the implementation

## Beyond Diversification (Page)
- Diversification fails in crises; need diversification across *return drivers* and explicit tail awareness.
Implication: must be multi-strategy (trend+carry+convex+spreads), not single-edge.

## Strategic Risk Management (Harvey/Rattray/Van Hemert)
- Harvest persistent premia with robust construction: trend, carry, defensive overlays, vol management.
Implication: these are the v1 strategy library.

## Safe Haven (Spitznagel)
- Survival and compounding require convex/tail protection.
Implication: include drawdown/tail overlay; accept small bleed.

## The Long Good Buy (Oppenheimer)
- Macro cycles and drawdowns are inevitable.
Implication: system must survive recessions/inflation/policy shifts, not optimize for one decade.

## Efficiently Inefficient (Pedersen)
- Premia persist due to constraints; core ones: momentum, carry, value, liquidity.
Implication: blend distinct premia; expect regime-dependent behavior; diversify.

## Geopolitical Alpha (Papic)
- Markets move on policy constraints more than headlines.
Implication: use "soft regime inputs" to adjust risk/execution during policy shock regimes.

## Advances in Financial ML (Lopez de Prado)
- Backtests are fragile; leakage and multiple testing are dominant failure modes.
Implication: purged/embargoed CV, WFO, multiple testing controls, stability selection are mandatory.
