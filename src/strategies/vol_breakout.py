"""
S3: Volatility Breakout / Convex Proxy — per 04_STRATEGY_LIBRARY.md.

Purpose: provide crisis-like convex response without options.
Signal: range compression then expansion (breakout above N-day range)
        OR ATR percentile regime break.
Sizing: small baseline risk; scale up only under high-vol regime
        with correlation spike haircuts.
Risk: tight max loss per trade; avoid "martingale".

CAUSALITY: All indicators use data up to t-1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategySignal
from src.features.volatility import ewma_volatility


class VolBreakoutStrategy(BaseStrategy):
    """Volatility breakout / convex proxy strategy.

    Detects volatility regime breaks and positions for continuation
    of the move, providing convex-like payoff in crisis regimes.
    """

    def __init__(
        self,
        range_lookback: int = 20,
        atr_lookback: int = 14,
        breakout_threshold: float = 1.5,
        atr_percentile_threshold: float = 80.0,
        max_loss_per_trade_pct: float = 0.02,
        target_vol: float = 0.05,  # small baseline per spec
        vol_lookback: int = 60,
        rebalance_freq: str = "daily",
    ):
        super().__init__(
            name="S3_vol_breakout",
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
        )
        self.range_lookback = range_lookback
        self.atr_lookback = atr_lookback
        self.breakout_threshold = breakout_threshold
        self.atr_percentile_threshold = atr_percentile_threshold
        self.max_loss_per_trade_pct = max_loss_per_trade_pct

    def _compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """Compute Average True Range, shifted for causality."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, min_periods=period).mean()

        # Shift for causality: ATR at t uses data up to t-1
        return atr.shift(1)

    def _compute_range_compression(
        self,
        high: pd.Series,
        low: pd.Series,
        lookback: int,
    ) -> pd.Series:
        """Detect range compression: ratio of recent range to lookback range.

        Low values = compressed range (potential breakout setup).
        Shifted for causality.
        """
        # Short-term range (5 days)
        short_range = high.rolling(5).max() - low.rolling(5).min()
        # Long-term range
        long_range = high.rolling(lookback).max() - low.rolling(lookback).min()

        compression = short_range / long_range.replace(0, np.nan)

        return compression.shift(1)

    def _compute_atr_percentile(
        self,
        atr: pd.Series,
    ) -> pd.Series:
        """Compute expanding percentile rank of ATR.

        Uses expanding window to prevent lookahead.
        """
        return atr.expanding(min_periods=60).rank(pct=True) * 100

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Generate vol breakout signals.

        Logic:
        1. Detect range compression (setup condition)
        2. Detect ATR percentile breakout (trigger condition)
        3. Direction follows the breakout move direction
        4. Size is small; scaled up only in confirmed high-vol regime

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV daily bars.
        returns : pd.Series
            Daily log returns.
        params : dict, optional
            Overrides.

        Returns
        -------
        StrategySignal
        """
        range_lb = (
            params.get("range_lookback", self.range_lookback)
            if params
            else self.range_lookback
        )
        atr_lb = (
            params.get("atr_lookback", self.atr_lookback)
            if params
            else self.atr_lookback
        )
        breakout_thresh = (
            params.get("breakout_threshold", self.breakout_threshold)
            if params
            else self.breakout_threshold
        )
        atr_pctile_thresh = (
            params.get("atr_percentile_threshold", self.atr_percentile_threshold)
            if params
            else self.atr_percentile_threshold
        )
        target_vol = (
            params.get("target_vol", self.target_vol) if params else self.target_vol
        )

        if params:
            self._n_params_tested += 1

        high = prices["high"]
        low = prices["low"]
        close = prices["close"]

        # Compute indicators (all shifted for causality)
        atr = self._compute_atr(high, low, close, atr_lb)
        atr_pctile = self._compute_atr_percentile(atr)
        compression = self._compute_range_compression(high, low, range_lb)

        # Breakout detection:
        # 1. Price breaks above/below the N-day range
        upper_band = high.shift(1).rolling(range_lb).max()
        lower_band = low.shift(1).rolling(range_lb).min()

        # Use previous close vs bands (causality: close[t-1] vs bands from [t-2..])
        prev_close = close.shift(1)
        breakout_up = prev_close > upper_band.shift(1)
        breakout_down = prev_close < lower_band.shift(1)

        # 2. ATR regime filter: only trade when vol is expanding
        vol_expanding = atr_pctile >= atr_pctile_thresh

        # Combine signals
        signal = pd.Series(0.0, index=prices.index)

        # Breakout up + vol expanding = go long (momentum continuation)
        signal[breakout_up & vol_expanding] = 1.0
        # Breakout down + vol expanding = go short
        signal[breakout_down & vol_expanding] = -1.0

        signal = signal.clip(-1.0, 1.0)

        # Apply rebalance
        signal = self.apply_rebalance_mask(signal)

        # Vol-scale with small baseline
        raw_weights = self.vol_scale_weights(signal, returns, target_vol)

        # Apply max loss cap: limit weight so max 1-day loss < threshold
        vol = ewma_volatility(returns, com=self.vol_lookback)
        max_weight = self.max_loss_per_trade_pct / vol.clip(lower=0.001)
        raw_weights = raw_weights.clip(lower=-max_weight, upper=max_weight)

        signal_out = pd.DataFrame({"ES": signal}, index=prices.index)
        weight_out = pd.DataFrame({"ES": raw_weights}, index=prices.index)

        return StrategySignal(
            signals=signal_out,
            raw_weights=weight_out,
            metadata={
                "atr": atr,
                "atr_percentile": atr_pctile,
                "compression": compression,
                "breakout_up": breakout_up,
                "breakout_down": breakout_down,
            },
        )
