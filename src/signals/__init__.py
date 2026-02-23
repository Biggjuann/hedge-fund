"""
Signal families for regime-robust systematic futures.

Exports all signal functions for use by the portfolio builder.
"""
from src.signals.base import vol_scale, weekly_rebal
from src.signals.trend import majority_vote_tsm
from src.signals.carry import carry_signal
from src.signals.mean_reversion import mean_reversion_signal
from src.signals.cross_sectional import cross_sectional_momentum
from src.signals.lead_lag import lead_lag_signal
from src.signals.vol_breakout import vol_breakout_signal
from src.signals.seasonality import seasonality_signal

__all__ = [
    "vol_scale",
    "weekly_rebal",
    "majority_vote_tsm",
    "carry_signal",
    "mean_reversion_signal",
    "cross_sectional_momentum",
    "lead_lag_signal",
    "vol_breakout_signal",
    "seasonality_signal",
]
