"""
Carry signal: mean-reversion around rolling mean, z-scored.
Captures roll yield / basis convergence in futures.
"""
from __future__ import annotations

import pandas as pd


def carry_signal(close_s: pd.Series, lookback: int = 63) -> pd.Series:
    """Carry/basis signal using deviation from rolling mean.

    CAUSALITY: All computations use shift(1) — no lookahead.

    Parameters
    ----------
    close_s : pd.Series
        Close price series.
    lookback : int
        Rolling mean window.

    Returns
    -------
    pd.Series
        Z-scored carry signal clipped to [-1, +1].
    """
    rmean = close_s.rolling(lookback, min_periods=lookback // 2).mean()
    carry = -(close_s - rmean).shift(1) / rmean.shift(1)
    cm = carry.rolling(252, min_periods=60).mean().shift(1)
    cs = carry.rolling(252, min_periods=60).std().shift(1).clip(lower=0.001)
    return ((carry - cm) / cs / 2.0).clip(-1, 1)
