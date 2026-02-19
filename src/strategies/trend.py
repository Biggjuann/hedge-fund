"""
S1: Multi-horizon Trend / TSMOM Ensemble — per 04_STRATEGY_LIBRARY.md.

Signal per horizon: s_h = sign(return over lookback)
Combine: s = average(s_h) then clamp to [-1, +1]
Sizing: w_i = s_i * (target_vol / vol_i)

CAUSALITY:
- Lookback return at time t uses close[t-1] - close[t-1-lookback]
- Vol uses EWMA(t-1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategySignal


class TrendStrategy(BaseStrategy):
    """Multi-horizon trend-following (TSMOM ensemble).

    Per spec:
    - lookbacks: [21, 63, 252] trading days
    - signal = sign(lookback return) per horizon
    - ensemble = average of horizon signals, clamped [-1, +1]
    - vol-scaled position sizing
    - monthly rebalance baseline
    """

    DEFAULT_LOOKBACKS = [21, 63, 252]

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        target_vol: float = 0.10,
        vol_lookback: int = 60,
        rebalance_freq: str = "monthly",
    ):
        super().__init__(
            name="S1_trend",
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
        )
        self.lookbacks = lookbacks or self.DEFAULT_LOOKBACKS

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate multi-horizon trend signals.

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

        close = prices["close"]

        # Compute signal per horizon
        # CAUSALITY: use shift(1) so signal at t uses data up to t-1
        horizon_signals = {}
        for lb in lookbacks:
            # Return over lookback period, shifted by 1 for causality
            lookback_return = close.shift(1) / close.shift(1 + lb) - 1
            sig = np.sign(lookback_return)
            horizon_signals[f"trend_{lb}"] = sig

        signals_df = pd.DataFrame(horizon_signals, index=prices.index)

        # Ensemble: average across horizons, clamp to [-1, +1]
        ensemble_signal = signals_df.mean(axis=1).clip(-1.0, 1.0)

        # Apply rebalance frequency
        ensemble_signal = self.apply_rebalance_mask(ensemble_signal)

        # Vol-scale the weights
        raw_weights = self.vol_scale_weights(ensemble_signal, returns, target_vol)

        # Build output DataFrame
        signal_out = pd.DataFrame({"ES": ensemble_signal}, index=prices.index)
        weight_out = pd.DataFrame({"ES": raw_weights}, index=prices.index)

        return StrategySignal(
            signals=signal_out,
            raw_weights=weight_out,
            metadata={
                "lookbacks": lookbacks,
                "horizon_signals": signals_df,
                "n_horizons": len(lookbacks),
            },
        )
