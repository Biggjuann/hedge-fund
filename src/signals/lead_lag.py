"""
Cross-asset lead-lag signal: uses lagged returns of "leader" asset classes
to predict "follower" asset classes.

Default relationships: bonds→equities, energy→ags, metals→FX.
Rolling 252-day correlation weights leader predictiveness.
EXPERIMENTAL — mark for validation before including in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Default lead-lag relationships: leader_class -> follower_class
DEFAULT_RELATIONSHIPS = {
    'DEBT': 'INDEX',     # bonds lead equities
    'ENERGY': 'AGS',     # energy leads ags
    'METALS': 'FX',      # metals lead FX
}

DEFAULT_ASSET_CLASSES = {
    'INDEX': ['ES', 'NQ', 'FDAX', 'NKD', 'LFT', 'HSI', 'YAP'],
    'DEBT':  ['ZN', 'ZB', 'ZF', 'FGBL'],
    'ENERGY': ['CL', 'NG', 'HO', 'RB'],
    'METALS': ['GC', 'SI', 'HG', 'PL', 'PA'],
    'AGS':   ['ZC', 'ZS', 'ZW', 'KC', 'CT', 'SB', 'CC'],
    'FX':    ['6E', '6J', '6B', '6A', '6C'],
}


def _class_return(
    returns_dict: dict[str, pd.Series],
    symbols: list[str],
) -> pd.Series:
    """Equal-weight average return of an asset class."""
    available = [returns_dict[s] for s in symbols if s in returns_dict]
    if not available:
        return pd.Series(dtype=float)
    return pd.concat(available, axis=1).mean(axis=1)


def lead_lag_signal(
    returns_dict: dict[str, pd.Series],
    asset_classes: dict[str, list[str]] | None = None,
    relationships: dict[str, str] | None = None,
    lag_days: int = 5,
    corr_window: int = 252,
    min_corr: float = 0.05,
) -> dict[str, pd.Series]:
    """Cross-asset lead-lag signal.

    For each leader→follower relationship:
    1. Compute leader class average return, lagged by `lag_days`
    2. Compute rolling correlation between lagged leader and follower
    3. If correlation is significant, use lagged leader return as signal
       for all follower instruments

    CAUSALITY: Leader returns lagged by lag_days, correlation uses
    expanding/rolling window — inherently causal.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Log returns keyed by symbol.
    asset_classes : dict, optional
        Asset class groupings.
    relationships : dict, optional
        Leader class -> follower class mappings.
    lag_days : int
        How many days to lag the leader signal.
    corr_window : int
        Rolling window for correlation estimation.
    min_corr : float
        Minimum absolute correlation to generate signal.

    Returns
    -------
    dict[str, pd.Series]
        Signal per symbol in [-1, +1].
    """
    if asset_classes is None:
        asset_classes = DEFAULT_ASSET_CLASSES
    if relationships is None:
        relationships = DEFAULT_RELATIONSHIPS

    # Compute class-level returns
    class_returns = {}
    for cls, syms in asset_classes.items():
        cr = _class_return(returns_dict, syms)
        if len(cr) > 0:
            class_returns[cls] = cr

    signals = {}

    for leader_cls, follower_cls in relationships.items():
        if leader_cls not in class_returns or follower_cls not in class_returns:
            continue

        leader_ret = class_returns[leader_cls]
        follower_ret = class_returns[follower_cls]

        # Lag leader returns
        lagged_leader = leader_ret.shift(lag_days)

        # Rolling correlation between lagged leader and follower
        common = lagged_leader.dropna().index.intersection(follower_ret.dropna().index)
        if len(common) < corr_window:
            continue

        ll = lagged_leader.reindex(common)
        fr = follower_ret.reindex(common)
        rolling_corr = ll.rolling(corr_window, min_periods=corr_window // 2).corr(fr).shift(1)

        # Signal: lagged leader return * sign(rolling_corr), z-scored
        raw_signal = lagged_leader * np.sign(rolling_corr.reindex(lagged_leader.index).fillna(0))
        # Suppress when correlation is too weak
        weak = rolling_corr.reindex(raw_signal.index).fillna(0).abs() < min_corr
        raw_signal[weak] = 0

        # Z-score the signal
        em = raw_signal.expanding(min_periods=63).mean()
        es = raw_signal.expanding(min_periods=63).std().clip(lower=1e-6)
        z_signal = ((raw_signal - em) / es).clip(-1, 1)

        # Assign to all follower instruments
        follower_syms = asset_classes.get(follower_cls, [])
        for sym in follower_syms:
            if sym in returns_dict:
                signals[sym] = z_signal.reindex(returns_dict[sym].index).fillna(0)

    # Fill any symbols without a signal
    for sym in returns_dict:
        if sym not in signals:
            signals[sym] = pd.Series(0.0, index=returns_dict[sym].index)

    return signals
