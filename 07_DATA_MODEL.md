# Data Model

## Store 3 series types
1) contract_bars: per expiry
2) continuous_research: stitched for features (explicit rule)
3) tradable_simulated: follows exact roll + execution assumptions

## Required contract metadata
- symbol, exchange, currency
- multiplier, tick_size, tick_value
- session hours
- last_trade_date, first_notice_date
- delivery flags
- roll calendar rules

## Roll rules
- must be explicit and unit-tested
- avoid delivery window by default
