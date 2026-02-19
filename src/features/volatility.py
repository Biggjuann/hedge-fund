"""
Volatility estimators — per 04_STRATEGY_LIBRARY.md + 01_DEEP_RESEARCH_PAPER.md.

EWMA variance with 60d center-of-mass baseline (daily).
STRICT CAUSALITY: vol(t-1) applied to returns(t).
Features at time t use only info <= t-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility(
    returns: pd.Series,
    com: int = 60,
    min_periods: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """EWMA volatility estimator.

    CAUSALITY: Returns vol estimate using data up to and including t-1.
    The output at index t is the vol forecast AVAILABLE at time t
    (computed from returns up to t-1).

    Parameters
    ----------
    returns : pd.Series
        Return series (log or simple).
    com : int
        Center of mass for EWMA (60 = ~120 day half-life).
    min_periods : int
        Minimum observations before producing a value.
    annualize : bool
        Whether to annualize (multiply by sqrt(trading_days)).
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    pd.Series
        Volatility series. SHIFTED by 1 to enforce causality:
        vol[t] = f(returns[..., t-1]).
    """
    # Compute EWMA variance on squared returns
    ewma_var = returns.pow(2).ewm(com=com, min_periods=min_periods).mean()

    # Convert to standard deviation
    vol = np.sqrt(ewma_var)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    # CRITICAL: Shift forward by 1 to enforce strict causality
    # vol[t] is now computed from returns up to t-1
    vol = vol.shift(1)

    return vol


def realized_volatility(
    returns: pd.Series,
    window: int = 21,
    min_periods: int = 10,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Simple rolling realized volatility.

    CAUSALITY: Shifted by 1 — available at time t, uses data up to t-1.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Rolling window size.
    min_periods : int
        Minimum periods for valid computation.
    annualize : bool
        Whether to annualize.
    trading_days : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Realized volatility, shifted for causality.
    """
    vol = returns.rolling(window=window, min_periods=min_periods).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    # Enforce causality
    vol = vol.shift(1)

    return vol


def ewma_correlation(
    returns_df: pd.DataFrame,
    com: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """EWMA correlation matrix estimate.

    Returns the correlation matrix available at time t (uses data up to t-1).

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of returns for multiple instruments.
    com : int
        Center of mass for EWMA.
    min_periods : int
        Minimum observations.

    Returns
    -------
    pd.DataFrame
        Most recent correlation matrix (shifted for causality).
    """
    # Shift returns to enforce causality — we compute on data up to t-1
    shifted = returns_df.shift(1)
    cov = shifted.ewm(com=com, min_periods=min_periods).cov()

    # Extract the last timestamp's correlation matrix
    last_ts = shifted.dropna().index[-1]
    cov_matrix = cov.loc[last_ts]

    # Convert covariance to correlation
    std = np.sqrt(np.diag(cov_matrix))
    std_outer = np.outer(std, std)
    std_outer[std_outer == 0] = 1.0  # avoid division by zero
    corr_matrix = cov_matrix / std_outer

    return pd.DataFrame(
        corr_matrix,
        index=cov_matrix.index,
        columns=cov_matrix.columns,
    )


def rolling_ewma_correlation_series(
    returns_df: pd.DataFrame,
    com: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """Compute rolling average pairwise correlation over time.

    Returns a Series of the average off-diagonal correlation at each point.
    CAUSALITY: shifted by 1.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Multi-instrument returns.
    com : int
        EWMA center of mass.
    min_periods : int
        Minimum periods.

    Returns
    -------
    pd.Series
        Average pairwise correlation over time.
    """
    n_assets = returns_df.shape[1]
    if n_assets < 2:
        return pd.Series(0.0, index=returns_df.index)

    # Compute pairwise rolling correlations using EWMA
    avg_corr_list = []
    ewm = returns_df.ewm(com=com, min_periods=min_periods)

    # Use rolling correlation of each pair
    cols = returns_df.columns
    pair_corrs = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            pair_corr = (
                returns_df.iloc[:, i]
                .ewm(com=com, min_periods=min_periods)
                .corr(returns_df.iloc[:, j])
            )
            pair_corrs.append(pair_corr)

    if pair_corrs:
        avg_corr = pd.concat(pair_corrs, axis=1).mean(axis=1)
    else:
        avg_corr = pd.Series(0.0, index=returns_df.index)

    # Enforce causality
    avg_corr = avg_corr.shift(1)

    return avg_corr
