"""
Trend-following signal: majority-vote time-series momentum.
12 lookback windows, each votes +1/-1, average gives signal strength.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def majority_vote_tsm(close_s: pd.Series) -> pd.Series:
    """Multi-horizon trend signal via majority vote.

    Uses 12 lookback windows (21d to 252d) of price momentum.
    Each window votes +1 (up) or -1 (down). Signal is the average vote.

    CAUSALITY: Uses shift(1) on close prices — no lookahead.

    Parameters
    ----------
    close_s : pd.Series
        Close price series.

    Returns
    -------
    pd.Series
        Signal in [-1, +1] range.
    """
    lagged = close_s.shift(1)
    lookbacks = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]
    votes = [np.sign(lagged / lagged.shift(lb) - 1) for lb in lookbacks]
    return pd.concat(votes, axis=1).mean(axis=1)
