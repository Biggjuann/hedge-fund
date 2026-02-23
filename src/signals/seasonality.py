"""
Seasonality signal: month-of-year average return using only expanding
historical data. Z-scored, constant within each month.
Expected correlation with all others: ~0 (calendar-based, orthogonal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonality_signal(
    returns_s: pd.Series,
    min_years: int = 10,
) -> pd.Series:
    """Calendar seasonality signal.

    For each month, computes the expanding average of historical returns
    for that same month-of-year. Z-scored across all months for
    cross-sectional comparability.

    CAUSALITY: Only uses data from prior completed months. The signal for
    month M of year Y uses data from all months M in years < Y.

    Parameters
    ----------
    returns_s : pd.Series
        Daily log returns with DatetimeIndex.
    min_years : int
        Minimum years of history per month before generating signal.

    Returns
    -------
    pd.Series
        Seasonality signal in [-1, +1], constant within each month.
    """
    if len(returns_s) < 252 * min_years:
        return pd.Series(0.0, index=returns_s.index)

    # Compute monthly returns
    monthly_ret = returns_s.resample('ME').sum()

    signal = pd.Series(0.0, index=returns_s.index)

    # For each date, compute the signal based on expanding history
    years = sorted(returns_s.index.year.unique())
    if len(years) < min_years + 1:
        return signal

    # Build expanding monthly average: for each (year, month),
    # use average of that month in all prior years
    month_history = {m: [] for m in range(1, 13)}

    for year in years:
        # First compute signal values for this year using prior history
        month_signals = {}
        for month in range(1, 13):
            hist = month_history[month]
            if len(hist) >= min_years:
                month_signals[month] = np.mean(hist)
            else:
                month_signals[month] = 0.0

        # Z-score across months for this year
        vals = list(month_signals.values())
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if std_v > 1e-8:
            for month in range(1, 13):
                month_signals[month] = (month_signals[month] - mean_v) / std_v

        # Assign signal to daily dates in this year
        year_mask = returns_s.index.year == year
        for month in range(1, 13):
            mask = year_mask & (returns_s.index.month == month)
            signal[mask] = np.clip(month_signals[month], -1, 1)

        # Update history with this year's monthly returns (for future years)
        for month in range(1, 13):
            month_data = monthly_ret[(monthly_ret.index.year == year) & (monthly_ret.index.month == month)]
            if len(month_data) > 0:
                month_history[month].append(float(month_data.iloc[0]))

    return signal
