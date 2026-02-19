"""
S2: Cross-asset Carry (Term Structure) — per 04_STRATEGY_LIBRARY.md.

Signal: cross-sectional long high-carry / short low-carry
Carry inferred from term structure (contract-level curve metric)
Weights: proportional to rank distance, scaled by 1/vol
Rebalance: monthly

Implementation rule: returns computed on tradable contract rolls,
NOT interpolated trading.

NOTE: For single-instrument mode (v1 with ES only), carry is approximated
using the slope of the term structure estimated from the unadjusted series.
Full cross-asset carry requires multi-instrument data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategySignal


class CarryStrategy(BaseStrategy):
    """Cross-asset carry strategy based on term structure.

    For multi-instrument: ranks carry across instruments.
    For single-instrument: uses carry signal as directional indicator.
    """

    def __init__(
        self,
        carry_lookback: int = 63,
        rank_long_quantile: float = 0.33,
        rank_short_quantile: float = 0.33,
        target_vol: float = 0.10,
        vol_lookback: int = 60,
        rebalance_freq: str = "monthly",
    ):
        super().__init__(
            name="S2_carry",
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
        )
        self.carry_lookback = carry_lookback
        self.rank_long_quantile = rank_long_quantile
        self.rank_short_quantile = rank_short_quantile

    def estimate_carry_single_instrument(
        self,
        close: pd.Series,
        lookback: int,
    ) -> pd.Series:
        """Estimate carry proxy for a single instrument from price data.

        Uses the annualized roll yield as a proxy for carry:
        carry ~ (close[t-1] - close[t-1-lookback]) / close[t-1-lookback] * (252/lookback)

        This approximates the return from holding the near contract
        vs the deferred, which is the term structure slope.

        CAUSALITY: uses shift(1) so carry at t uses data up to t-1.
        """
        # Rolling return as carry proxy, shifted for causality
        carry = (close.shift(1) / close.shift(1 + lookback) - 1) * (
            252 / lookback
        )
        return carry

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate carry signals.

        Single-instrument mode: directional carry signal.
        Multi-instrument mode (prices has multiple columns): cross-sectional.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV daily bars.
        returns : pd.Series
            Daily log returns.
        params : dict, optional
            Overrides: 'carry_lookback', 'target_vol'.

        Returns
        -------
        StrategySignal
        """
        carry_lb = (
            params.get("carry_lookback", self.carry_lookback)
            if params
            else self.carry_lookback
        )
        target_vol = (
            params.get("target_vol", self.target_vol) if params else self.target_vol
        )

        if params:
            self._n_params_tested += 1

        close = prices["close"]

        # Single instrument carry: directional
        carry = self.estimate_carry_single_instrument(close, carry_lb)

        # Normalize carry to [-1, +1] using expanding rank/percentile
        # CAUSALITY: expanding (not full-sample) percentile
        carry_pctile = carry.expanding(min_periods=60).rank(pct=True)
        # Map [0, 1] percentile to [-1, +1] signal
        signal = (carry_pctile - 0.5) * 2.0
        signal = signal.clip(-1.0, 1.0)

        # Apply rebalance
        signal = self.apply_rebalance_mask(signal)

        # Vol-scale
        raw_weights = self.vol_scale_weights(signal, returns, target_vol)

        signal_out = pd.DataFrame({"ES": signal}, index=prices.index)
        weight_out = pd.DataFrame({"ES": raw_weights}, index=prices.index)

        return StrategySignal(
            signals=signal_out,
            raw_weights=weight_out,
            metadata={
                "carry_lookback": carry_lb,
                "carry_raw": carry,
            },
        )

    def generate_signals_cross_sectional(
        self,
        prices_dict: dict[str, pd.DataFrame],
        returns_dict: dict[str, pd.Series],
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate cross-sectional carry signals across multiple instruments.

        Long top quantile, short bottom quantile by carry rank.
        Weights proportional to rank distance, scaled by 1/vol.

        Parameters
        ----------
        prices_dict : dict
            Maps instrument name -> OHLCV DataFrame.
        returns_dict : dict
            Maps instrument name -> returns Series.
        params : dict, optional
            Overrides.

        Returns
        -------
        StrategySignal
        """
        carry_lb = (
            params.get("carry_lookback", self.carry_lookback)
            if params
            else self.carry_lookback
        )
        target_vol = (
            params.get("target_vol", self.target_vol) if params else self.target_vol
        )

        if params:
            self._n_params_tested += 1

        instruments = sorted(prices_dict.keys())
        n = len(instruments)

        # Compute carry for each instrument
        carry_df = pd.DataFrame()
        for inst in instruments:
            close = prices_dict[inst]["close"]
            carry_df[inst] = self.estimate_carry_single_instrument(close, carry_lb)

        # Cross-sectional rank at each timestamp (expanding percentile)
        # CAUSALITY: carry already uses shift(1)
        ranked = carry_df.rank(axis=1, pct=True)

        # Long top quantile, short bottom quantile
        long_threshold = 1.0 - self.rank_long_quantile
        short_threshold = self.rank_short_quantile

        signals = pd.DataFrame(0.0, index=carry_df.index, columns=instruments)
        signals[ranked >= long_threshold] = 1.0
        signals[ranked <= short_threshold] = -1.0

        # Scale by rank distance from median for proportional weights
        rank_distance = (ranked - 0.5) * 2.0  # [-1, +1]
        signals = signals * rank_distance.abs()
        signals = signals.clip(-1.0, 1.0)

        # Vol-scale per instrument
        weights = pd.DataFrame(index=carry_df.index, columns=instruments)
        for inst in instruments:
            rebal_sig = self.apply_rebalance_mask(signals[inst])
            weights[inst] = self.vol_scale_weights(
                rebal_sig, returns_dict[inst], target_vol
            )

        return StrategySignal(
            signals=signals,
            raw_weights=weights,
            metadata={"carry_lookback": carry_lb, "carry_raw": carry_df},
        )
