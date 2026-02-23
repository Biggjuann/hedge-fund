"""
Vol breakout v2: short-term / long-term vol ratio, expanding percentile ranked.
Amplifies positions during vol expansion in trend direction.
Expected correlation with trend: 0.3-0.5 (modulates trend, not independent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility import ewma_volatility


def vol_breakout_signal(
    close_s: pd.Series,
    returns_s: pd.Series,
    short_windows: list[int] | None = None,
    long_window: int = 63,
) -> pd.Series:
    """Vol breakout signal combining vol expansion with trend direction.

    For each short window:
    1. Compute short_vol / long_vol ratio
    2. Expanding percentile rank the ratio
    3. Combine with price trend direction

    CAUSALITY: Vol uses shift(1) internally, percentile uses expanding window.

    Parameters
    ----------
    close_s : pd.Series
        Close price series.
    returns_s : pd.Series
        Log returns.
    short_windows : list[int], optional
        Short-term vol windows. Defaults to [10, 21].
    long_window : int
        Long-term vol window for comparison.

    Returns
    -------
    pd.Series
        Vol breakout signal in [-1, +1].
    """
    if short_windows is None:
        short_windows = [10, 21]

    lagged_close = close_s.shift(1)

    # Long-term vol
    long_vol = ewma_volatility(returns_s, com=long_window)

    sub_signals = []
    for sw in short_windows:
        # Short-term vol
        short_vol = ewma_volatility(returns_s, com=sw)

        # Vol ratio: high = vol expansion
        ratio = short_vol / long_vol.clip(lower=0.001)

        # Expanding percentile rank (0 to 1)
        pctile = ratio.expanding(min_periods=63).rank(pct=True)

        # Trend direction from price momentum (using lagged close)
        trend_dir = np.sign(lagged_close / lagged_close.shift(sw) - 1)

        # Signal: vol expansion * trend direction
        # High pctile (>0.5 → vol expanding) amplifies trend
        # Low pctile (<0.5 → vol contracting) dampens signal
        signal = (2 * pctile - 1) * trend_dir  # maps [0,1] -> [-1,1] then * direction

        sub_signals.append(signal)

    # Average across horizons
    avg = pd.concat(sub_signals, axis=1).mean(axis=1)

    # Z-score for normalization
    em = avg.expanding(min_periods=63).mean()
    es = avg.expanding(min_periods=63).std().clip(lower=0.01)
    z = (avg - em) / es

    return z.clip(-1, 1)
