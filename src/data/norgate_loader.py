"""
Thin adapter for loading Norgate-exported daily CSV files from data_norgate/.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd


def load_norgate_daily(
    symbol: str,
    data_dir: str = "data_norgate",
    project_root: str | None = None,
) -> pd.DataFrame:
    """Load a single Norgate daily CSV.

    Back-adjusted continuous contracts can have zero/negative prices at old
    roll points. We filter these out to prevent inf/nan returns.

    Parameters
    ----------
    symbol : str
        Internal symbol name (e.g. 'ES', 'ZN', '6E').
    data_dir : str
        Directory containing Norgate CSVs.
    project_root : str, optional
        Project root. Defaults to two levels up from this file.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex, weekday-filtered, deduplicated,
        rows with non-positive close prices removed.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path = os.path.join(project_root, data_dir, f"{symbol}_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No Norgate data for {symbol}: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]

    # Filter out rows with non-positive close (back-adjustment artifacts)
    df = df[df["close"] > 0]

    # Weekday filter and dedup
    if hasattr(df.index, 'dayofweek'):
        df = df[df.index.dayofweek < 5]
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def load_norgate_manifest(
    data_dir: str = "data_norgate",
    project_root: str | None = None,
) -> dict:
    """Load the manifest.json with metadata for all exported symbols."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path = os.path.join(project_root, data_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_all_norgate(
    symbols: list[str] | None = None,
    data_dir: str = "data_norgate",
    project_root: str | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load close prices and log returns for all available Norgate symbols.

    Parameters
    ----------
    symbols : list[str], optional
        Subset of symbols. If None, loads all from manifest.
    data_dir : str
        Norgate data directory.
    project_root : str, optional
        Project root.

    Returns
    -------
    tuple[dict, dict]
        (all_close, all_returns) dicts keyed by symbol.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if symbols is None:
        manifest = load_norgate_manifest(data_dir, project_root)
        symbols = list(manifest.keys())

    all_close = {}
    all_returns = {}

    for sym in symbols:
        try:
            df = load_norgate_daily(sym, data_dir, project_root)
            close = df["close"]
            rets = np.log(close / close.shift(1))
            # Clip extreme returns from back-adjustment roll artifacts
            # 25% threshold preserves real tail events (NG, KC) while removing roll artifacts
            rets = rets.clip(-0.25, 0.25)
            # Replace any remaining inf/nan with 0
            rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0)
            all_close[sym] = close
            all_returns[sym] = rets
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    return all_close, all_returns
