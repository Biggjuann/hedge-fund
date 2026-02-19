"""
Base strategy interface — all strategies must implement this.

Per 04_STRATEGY_LIBRARY.md:
- Common vol estimator (EWMA 60d CoM)
- Position sizing baseline: target risk per instrument via 1/vol
- Rebalance cadence: monthly default

Per 01_DEEP_RESEARCH_PAPER.md:
- Strict causality: vol(t-1) applied to returns(t)
- Features at time t use only info <= t-1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.volatility import ewma_volatility


@dataclass
class StrategySignal:
    """Output of a strategy's signal generation.

    signals: pd.DataFrame with columns per instrument, values in [-1, +1]
    raw_weights: pd.DataFrame with vol-scaled position sizes
    metadata: dict with any strategy-specific diagnostics
    """

    signals: pd.DataFrame  # direction signals, clipped to [-1, +1]
    raw_weights: pd.DataFrame  # vol-scaled weights (before portfolio overlay)
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all strategy implementations.

    Enforces:
    - Strict causality in signal generation
    - Vol-scaled position sizing
    - Standardized output format
    """

    def __init__(
        self,
        name: str,
        target_vol: float = 0.10,
        vol_lookback: int = 60,
        rebalance_freq: str = "monthly",
        symbol: str = "ES",
    ):
        self.name = name
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.rebalance_freq = rebalance_freq
        self.symbol = symbol
        self._n_params_tested: int = 0  # for multiple testing tracking

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate trading signals from price data.

        MUST enforce:
        - No lookahead: signals at t use data <= t-1
        - Output signals clipped to [-1, +1]

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV daily bars.
        returns : pd.Series
            Log returns (daily).
        params : dict, optional
            Parameter overrides for walk-forward optimization.

        Returns
        -------
        StrategySignal
        """
        ...

    def vol_scale_weights(
        self,
        signals: pd.Series,
        returns: pd.Series,
        target_vol: float | None = None,
    ) -> pd.Series:
        """Scale signal by inverse volatility to target risk.

        Per spec: w_i = s_i * (target_vol / vol_i)
        vol_i is EWMA vol shifted by 1 (causality).

        Parameters
        ----------
        signals : pd.Series
            Raw direction signal in [-1, +1].
        returns : pd.Series
            Return series for vol estimation.
        target_vol : float, optional
            Override target vol.

        Returns
        -------
        pd.Series
            Vol-scaled position weights.
        """
        tv = target_vol or self.target_vol
        vol = ewma_volatility(returns, com=self.vol_lookback)

        # Floor at 50% of target vol → max 2x scaling (not 10x)
        vol_safe = vol.clip(lower=tv * 0.5)

        weights = signals * (tv / vol_safe)

        # Cap position at 2x to prevent leverage bombs
        weights = weights.clip(-2.0, 2.0)

        return weights

    def apply_rebalance_mask(
        self,
        signals: pd.Series,
        freq: str | None = None,
    ) -> pd.Series:
        """Apply rebalance frequency — hold signal between rebalance dates.

        Parameters
        ----------
        signals : pd.Series
            Raw signals (potentially changing every bar).
        freq : str, optional
            'monthly', 'weekly', 'daily'.

        Returns
        -------
        pd.Series
            Signals that only change at rebalance points.
        """
        f = freq or self.rebalance_freq

        if f == "daily":
            return signals

        if f == "monthly":
            # Rebalance on first trading day of each month
            rebal_mask = signals.index.to_series().dt.is_month_start
            # Handle case where month starts on weekend
            if not rebal_mask.any():
                rebal_mask = (
                    signals.index.to_series().dt.month
                    != signals.index.to_series().dt.month.shift(1)
                )
        elif f == "weekly":
            rebal_mask = signals.index.to_series().dt.dayofweek == 0  # Monday
        else:
            return signals

        # Forward-fill signal between rebalance dates
        rebal_signal = signals.where(rebal_mask).ffill()

        return rebal_signal

    @property
    def param_count(self) -> int:
        """Number of parameter combinations tested (for multiple testing)."""
        return self._n_params_tested
