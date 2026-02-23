"""
S4: Williams %R Momentum Strategy.

Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
Range: [-100, 0] where -100 = near lows, 0 = near highs.

Signal logic (momentum — long when near highs, short when near lows):
- %R near 0 (close near highest high): long (+1)
- %R near -100 (close near lowest low): short (-1)
- Linear interpolation between

Multi-horizon ensemble with long lookbacks (126, 189, 252) for capturing
sustained trends. Complementary to S1 trend (which uses return sign).

CAUSALITY:
- All indicator inputs use shift(1) — signal at t uses data up to t-1
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategySignal


class WilliamsRStrategy(BaseStrategy):
    """Williams %R momentum strategy with multi-horizon ensemble."""

    DEFAULT_LOOKBACKS = [126, 189, 252]

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        target_vol: float = 0.10,
        vol_lookback: int = 60,
        rebalance_freq: str = "weekly",
        symbol: str = "ES",
    ):
        super().__init__(
            name="S4_williams_r",
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
            symbol=symbol,
        )
        self.lookbacks = lookbacks or self.DEFAULT_LOOKBACKS

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate Williams %R momentum signals.

        Momentum mode: long when close is near period highs, short when near lows.
        Uses long lookback periods (126-252d) for sustained trend capture.

        CAUSALITY: shift(1) on all inputs — signal at t uses data up to t-1.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV daily bars.
        returns : pd.Series
            Daily log returns.
        params : dict, optional
            Overrides: 'lookbacks', 'target_vol'.

        Returns
        -------
        StrategySignal
        """
        lookbacks = params.get("lookbacks", self.lookbacks) if params else self.lookbacks
        target_vol = (
            params.get("target_vol", self.target_vol) if params else self.target_vol
        )

        if params:
            self._n_params_tested += 1

        high = prices["high"]
        low = prices["low"]
        close = prices["close"]

        horizon_signals = {}
        for lb in lookbacks:
            # CAUSALITY: shift(1) — use data up to t-1
            highest_high = high.shift(1).rolling(lb, min_periods=lb).max()
            lowest_low = low.shift(1).rolling(lb, min_periods=lb).min()
            close_prev = close.shift(1)

            # Williams %R: range [-100, 0]
            hl_range = highest_high - lowest_low
            wr = np.where(
                hl_range > 0,
                (highest_high - close_prev) / hl_range * -100.0,
                -50.0,
            )
            wr = pd.Series(wr, index=prices.index)

            # Momentum signal: map %R to [-1, +1]
            # %R = 0 (near highs) → +1 (long)
            # %R = -100 (near lows) → -1 (short)
            sig = (wr + 50.0) / 50.0  # maps 0→+1, -100→-1
            sig = sig.clip(-1.0, 1.0)

            horizon_signals[f"wr_{lb}"] = sig

        signals_df = pd.DataFrame(horizon_signals, index=prices.index)

        # Ensemble: average across horizons, clamp [-1, +1]
        ensemble_signal = signals_df.mean(axis=1).clip(-1.0, 1.0)

        # Apply rebalance frequency
        ensemble_signal = self.apply_rebalance_mask(ensemble_signal)

        # Vol-scale weights
        raw_weights = self.vol_scale_weights(ensemble_signal, returns, target_vol)

        signal_out = pd.DataFrame({self.symbol: ensemble_signal}, index=prices.index)
        weight_out = pd.DataFrame({self.symbol: raw_weights}, index=prices.index)

        return StrategySignal(
            signals=signal_out,
            raw_weights=weight_out,
            metadata={
                "lookbacks": lookbacks,
                "horizon_signals": signals_df,
                "n_horizons": len(lookbacks),
                "aggregation_method": "wr_momentum",
            },
        )
