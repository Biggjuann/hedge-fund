"""
Data loading and resampling — per 07_DATA_MODEL.md + 03_SYSTEM_REQUIREMENTS.md.

Supports multi-timeframe without lookahead:
- 5m/15m for intraday
- 1h/4h/1D for regime + macro features

Resampling uses label='right', closed='right' so bar timestamps
represent the END of the bar period (no lookahead).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_bars_from_csv(
    file_path: str,
    columns: list[str] | None = None,
    parse_dates: bool = True,
    tz: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV bar data from CSV/TXT file.

    Expected format: datetime,open,high,low,close,volume (no header)
    or with header if columns match.

    Parameters
    ----------
    file_path : str
        Path to the data file.
    columns : list[str], optional
        Column names. Defaults to ['datetime','open','high','low','close','volume'].
    parse_dates : bool
        Whether to parse the datetime column.
    tz : str, optional
        Timezone to localize timestamps to (e.g., 'US/Eastern').

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    if columns is None:
        columns = ["datetime", "open", "high", "low", "close", "volume"]

    # Detect if file has a header
    with open(file_path, "r") as f:
        first_line = f.readline().strip()

    has_header = not first_line[0].isdigit()

    df = pd.read_csv(
        file_path,
        names=None if has_header else columns,
        header=0 if has_header else None,
        parse_dates=["datetime"] if parse_dates else False,
        index_col="datetime" if parse_dates else None,
    )

    if not parse_dates and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    # Ensure correct column names
    expected = ["open", "high", "low", "close", "volume"]
    if list(df.columns) != expected:
        df.columns = expected

    if tz is not None:
        df.index = df.index.tz_localize(tz)

    df = df.sort_index()

    # Basic integrity checks
    assert not df.index.duplicated().any(), "Duplicate timestamps found"
    assert (df["high"] >= df["low"]).all(), "High < Low detected"
    assert (df["volume"] >= 0).all(), "Negative volume detected"

    return df


def resample_bars(
    df: pd.DataFrame,
    rule: str,
    closed: str = "right",
    label: str = "right",
) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser frequency.

    CRITICAL: Uses label='right', closed='right' by default so the
    timestamp represents the END of the bar. This prevents lookahead.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data with DatetimeIndex.
    rule : str
        Pandas offset alias: '5min', '15min', '1h', '4h', '1D', etc.
    closed : str
        Which side of bin interval is closed.
    label : str
        Which bin edge to label the result with.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV bars.
    """
    resampled = df.resample(rule, closed=closed, label=label).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop bars where no data existed (e.g., weekends, holidays)
    resampled = resampled.dropna(subset=["open", "close"])

    return resampled


def compute_returns(
    close: pd.Series,
    method: str = "log",
) -> pd.Series:
    """Compute returns from close prices.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    method : str
        'log' for log returns, 'simple' for arithmetic returns.

    Returns
    -------
    pd.Series
        Returns series (first value is NaN).
    """
    if method == "log":
        return np.log(close / close.shift(1))
    elif method == "simple":
        return close.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")


def get_daily_bars(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Convert 1-minute bars to daily bars for strategy computation.

    Uses standard RTH session aggregation with label='right'.
    """
    return resample_bars(df_1min, "1D")
