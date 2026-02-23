"""
Shared signal utilities: vol scaling, rebalance frequency control.
Extracted from save_v10_dashboard.py for reuse across signal families.
"""
from __future__ import annotations

import pandas as pd

from src.features.volatility import ewma_volatility


def vol_scale(
    sig: pd.Series,
    ret: pd.Series,
    target_vol: float = 0.15,
    com: int = 60,
) -> pd.Series:
    """Scale signal by inverse vol to target a fixed annualized volatility.

    Parameters
    ----------
    sig : pd.Series
        Raw signal in [-1, +1] range.
    ret : pd.Series
        Return series for vol estimation.
    target_vol : float
        Target annualized volatility.
    com : int
        EWMA center of mass for vol estimate.

    Returns
    -------
    pd.Series
        Vol-scaled signal, clipped to [-5, +5].
    """
    v = ewma_volatility(ret, com=com)
    scale = target_vol / v.clip(lower=0.02)
    return (sig * scale.clip(upper=10.0)).clip(-5, 5)


def weekly_rebal(sig: pd.Series, period: int = 5) -> pd.Series:
    """Reduce rebalance frequency by holding signals for `period` days.

    Only updates the signal every `period` bars; carries forward otherwise.

    Parameters
    ----------
    sig : pd.Series
        Daily signal.
    period : int
        Rebalance period in trading days.

    Returns
    -------
    pd.Series
        Signal with reduced turnover.
    """
    out = sig.copy()
    for i in range(len(out)):
        if i % period != 0 and i > 0:
            out.iloc[i] = out.iloc[i - 1]
    return out
