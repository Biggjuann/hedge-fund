# Risk, Portfolio, Execution (Survival Layer)

## Portfolio construction
- Default: risk parity across strategies (equal vol contribution)
- Optional: ERC for instruments if correlation model available
- Correlation haircuts: reduce risk when corr rises (risk-on/off clustering)

## Drawdown ladder
- DD >= 5%: risk * 0.75
- DD >= 8%: risk * 0.50
- DD >= 10%: flat + cooldown N days
Cooldown re-entry requires rolling performance stabilization rule.

## Margin buffers
- Maintain margin utilization cap (e.g., <= 30–40% of equity).
- Apply "margin shock" stress test; enforce leverage cap under stress.

## Execution model (backtest realism)
Cost components:
- spread
- slippage
- impact (nonlinear vs size/liquidity)
Participation caps:
- cap order size as % of ADV (proxy)
- widen execution schedules in high vol / low liquidity regimes
