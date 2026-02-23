"""
Mean-reversion signal: multi-horizon RSI with reversed sign.
Oversold -> buy, overbought -> sell. Short-term contrarian alpha.
Expected correlation with trend: -0.3 to -0.5 (genuine diversification).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close_s: pd.Series, period: int) -> pd.Series:
    """Compute RSI with strict causality (shift(1) on input)."""
    lagged = close_s.shift(1)
    delta = lagged.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def mean_reversion_signal(
    close_s: pd.Series,
    lookbacks: list[int] | None = None,
) -> pd.Series:
    """Multi-horizon RSI mean-reversion signal.

    Averages RSI across multiple lookback periods, converts to [-1, +1]
    via z-scoring with expanding window. Sign is reversed: low RSI
    (oversold) → positive signal (buy).

    CAUSALITY: RSI computed on shift(1) prices, z-scored with expanding window.

    Parameters
    ----------
    close_s : pd.Series
        Close price series.
    lookbacks : list[int], optional
        RSI lookback periods. Defaults to [3, 5, 10].

    Returns
    -------
    pd.Series
        Mean-reversion signal in [-1, +1].
    """
    if lookbacks is None:
        lookbacks = [3, 5, 10]

    rsi_signals = []
    for lb in lookbacks:
        rsi = _rsi(close_s, lb)
        # Center around 50 and reverse sign: oversold (RSI<30) -> positive
        centered = -(rsi - 50) / 50  # RSI=0 -> +1, RSI=100 -> -1
        rsi_signals.append(centered)

    avg_rsi = pd.concat(rsi_signals, axis=1).mean(axis=1)

    # Z-score with expanding window for stationarity
    em = avg_rsi.expanding(min_periods=63).mean()
    es = avg_rsi.expanding(min_periods=63).std().clip(lower=0.01)
    z = (avg_rsi - em) / es

    return z.clip(-1, 1)
