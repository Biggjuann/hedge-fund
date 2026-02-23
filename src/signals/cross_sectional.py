"""
Cross-sectional momentum: rank instruments by trailing return,
long top quartile, short bottom quartile.
Expected correlation with trend: 0.2-0.4 (relative vs absolute momentum).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_momentum(
    close_dict: dict[str, pd.Series],
    lookback: int = 63,
    rebal_period: int = 21,
) -> dict[str, pd.Series]:
    """Cross-sectional momentum signal across instruments.

    Ranks all instruments by trailing cumulative return. Top quartile
    gets positive signal, bottom quartile gets negative, with linear
    weighting based on rank distance from median.

    CAUSALITY: Uses shift(1) on close prices, lookback on lagged data.

    Parameters
    ----------
    close_dict : dict[str, pd.Series]
        Close prices keyed by symbol.
    lookback : int
        Trailing return lookback in days.
    rebal_period : int
        Rebalance every N days (monthly=21).

    Returns
    -------
    dict[str, pd.Series]
        Signal per symbol in [-1, +1], rebalanced monthly.
    """
    if len(close_dict) < 4:
        return {sym: pd.Series(0.0, index=s.index) for sym, s in close_dict.items()}

    # Build close DataFrame and compute trailing returns
    close_df = pd.DataFrame(close_dict)
    lagged = close_df.shift(1)
    trailing_ret = (lagged / lagged.shift(lookback) - 1)

    syms = list(close_dict.keys())
    n = len(syms)
    signals = {sym: pd.Series(0.0, index=close_df.index, dtype=float) for sym in syms}

    dates = close_df.index
    current_signal = {sym: 0.0 for sym in syms}

    for i in range(lookback + 2, len(dates)):
        # Only rebalance every rebal_period days
        if i % rebal_period != 0:
            for sym in syms:
                signals[sym].iloc[i] = current_signal[sym]
            continue

        # Get trailing returns for this date
        rets = {}
        for sym in syms:
            val = trailing_ret[sym].iloc[i]
            if pd.notna(val):
                rets[sym] = val

        if len(rets) < 4:
            for sym in syms:
                signals[sym].iloc[i] = current_signal[sym]
            continue

        # Rank: highest return = rank N, lowest = rank 1
        sorted_syms = sorted(rets.keys(), key=lambda s: rets[s])
        ranks = {sym: rank for rank, sym in enumerate(sorted_syms)}
        n_ranked = len(sorted_syms)
        median_rank = (n_ranked - 1) / 2

        # Linear signal based on rank distance from median
        for sym in syms:
            if sym in ranks:
                rank_dist = (ranks[sym] - median_rank) / max(median_rank, 1)
                current_signal[sym] = np.clip(rank_dist, -1, 1)
            else:
                current_signal[sym] = 0.0
            signals[sym].iloc[i] = current_signal[sym]

    return signals
