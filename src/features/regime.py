"""
Regime classification — per 01_DEEP_RESEARCH_PAPER.md + 05_VALIDATION_PROTOCOL.md.

Core latent dimensions:
1) Volatility level + change (realized and implied proxies)
2) Correlation regime (cross-market clustering, risk-on/off)
3) Liquidity/funding regime
4) Macro cycle regime (for evaluation tagging only)

CAUSALITY: All regime tags at time t use only info available at t-1.
These are used for:
- OOS regime-conditioned reporting
- Soft risk scaling (NOT hard strategy switching per spec)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.volatility import ewma_volatility, realized_volatility


@dataclass
class RegimeLabels:
    """Container for regime labels across all dimensions."""

    vol_regime: pd.Series  # 'low', 'med', 'high'
    corr_regime: pd.Series  # 'low', 'high'
    trend_regime: pd.Series  # 'trending', 'range_bound'
    composite_risk: pd.Series  # continuous [0, 1] risk score


class RegimeTagger:
    """Assigns regime labels to each timestamp for OOS conditioning.

    Per 05_VALIDATION_PROTOCOL.md:
    - vol regime: low/med/high (quantiles)
    - corr regime: low/high (quantiles of avg pairwise corr)
    - trend regime: ADX proxy bucket

    All outputs are LAGGED to enforce causality.
    """

    def __init__(
        self,
        vol_lookback: int = 63,
        corr_lookback: int = 63,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        vol_quantiles: tuple[float, ...] = (0.33, 0.67),
        corr_quantile: float = 0.50,
    ):
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_quantiles = vol_quantiles
        self.corr_quantile = corr_quantile

    def tag_vol_regime(
        self,
        returns: pd.Series,
        vol: pd.Series | None = None,
    ) -> pd.Series:
        """Classify volatility regime into low/med/high.

        Uses expanding quantiles to avoid lookahead.
        """
        if vol is None:
            vol = ewma_volatility(returns, com=self.vol_lookback)

        # Use EXPANDING quantiles (not full-sample) to prevent lookahead
        q_low = vol.expanding(min_periods=60).quantile(self.vol_quantiles[0])
        q_high = vol.expanding(min_periods=60).quantile(self.vol_quantiles[1])

        regime = pd.Series("med", index=vol.index)
        regime[vol <= q_low] = "low"
        regime[vol >= q_high] = "high"
        regime[vol.isna()] = np.nan

        return regime

    def tag_corr_regime(
        self,
        avg_corr: pd.Series,
    ) -> pd.Series:
        """Classify correlation regime into low/high.

        Uses expanding quantile to prevent lookahead.
        """
        q_mid = avg_corr.expanding(min_periods=60).quantile(self.corr_quantile)

        regime = pd.Series("low", index=avg_corr.index)
        regime[avg_corr >= q_mid] = "high"
        regime[avg_corr.isna()] = np.nan

        return regime

    def tag_trend_regime(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Classify trend regime using ADX proxy.

        ADX > threshold = trending, else range_bound.
        All computations use data up to t-1 (shifted).
        """
        adx = self._compute_adx_proxy(high, low, close, self.adx_period)

        regime = pd.Series("range_bound", index=close.index)
        regime[adx > self.adx_threshold] = "trending"
        regime[adx.isna()] = np.nan

        return regime

    def _compute_adx_proxy(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """Compute ADX (Average Directional Index) proxy.

        Shifted by 1 for causality.
        """
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)

        plus_mask = (up_move > down_move) & (up_move > 0)
        minus_mask = (down_move > up_move) & (down_move > 0)

        plus_dm[plus_mask] = up_move[plus_mask]
        minus_dm[minus_mask] = down_move[minus_mask]

        # Smoothed averages
        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, min_periods=period).mean()

        # Enforce causality
        adx = adx.shift(1)

        return adx

    def compute_composite_risk_score(
        self,
        vol_regime: pd.Series,
        corr_regime: pd.Series,
    ) -> pd.Series:
        """Compute a continuous [0, 1] risk score from regime labels.

        Used for soft risk scaling (not hard switching, per spec).
        Higher = more risk, requiring de-risking.
        """
        vol_score = vol_regime.map({"low": 0.0, "med": 0.5, "high": 1.0}).fillna(0.5)
        corr_score = corr_regime.map({"low": 0.0, "high": 1.0}).fillna(0.5)

        # Weighted combination: vol matters more
        composite = 0.6 * vol_score + 0.4 * corr_score

        return composite

    def tag_all(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        avg_corr: pd.Series | None = None,
    ) -> RegimeLabels:
        """Compute all regime tags for a dataset.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV daily data.
        returns : pd.Series
            Log returns.
        avg_corr : pd.Series, optional
            Average pairwise correlation series.

        Returns
        -------
        RegimeLabels
        """
        vol = ewma_volatility(returns, com=self.vol_lookback)
        vol_regime = self.tag_vol_regime(returns, vol)

        if avg_corr is not None:
            corr_regime = self.tag_corr_regime(avg_corr)
        else:
            # Single instrument: no correlation regime
            corr_regime = pd.Series("low", index=prices.index)

        trend_regime = self.tag_trend_regime(
            prices["high"], prices["low"], prices["close"]
        )

        composite = self.compute_composite_risk_score(vol_regime, corr_regime)

        return RegimeLabels(
            vol_regime=vol_regime,
            corr_regime=corr_regime,
            trend_regime=trend_regime,
            composite_risk=composite,
        )
