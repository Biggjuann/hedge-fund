"""
Continuous series construction — per 07_DATA_MODEL.md.

Three series types:
1) contract_bars: per expiry (raw data)
2) continuous_research: stitched for features (explicit rule)
3) tradable_simulated: follows exact roll + execution assumptions

For unadjusted continuous data (our ES file), we track roll points
and can compute both research and tradable returns correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.data.rolls import RollCalendar, RollDate


@dataclass
class ContinuousSeries:
    """Container for a continuous futures series with roll metadata."""

    prices: pd.DataFrame  # OHLCV
    returns: pd.Series  # log or simple returns
    active_contract: pd.Series  # contract label per timestamp
    roll_dates: list[date]  # when rolls occurred
    series_type: str  # 'research' or 'tradable'


def build_continuous_series(
    prices: pd.DataFrame,
    roll_calendar: list[RollDate],
    method: str = "unadjusted",
) -> ContinuousSeries:
    """Build a research continuous series from raw price data.

    For unadjusted data: returns are computed carefully at roll boundaries
    to avoid spurious returns from contract switches.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV data (already a continuous unadjusted series).
    roll_calendar : list[RollDate]
        Roll schedule from RollCalendar.
    method : str
        'unadjusted' — use raw prices, null out returns at roll points.
        'ratio_adjusted' — multiply by ratio at roll (preserves returns).

    Returns
    -------
    ContinuousSeries
    """
    roll_dates_set = {r.roll_date for r in roll_calendar}
    roll_dates_list = sorted(roll_dates_set)

    # Compute returns
    log_returns = np.log(prices["close"] / prices["close"].shift(1))

    if method == "unadjusted":
        # Null out returns on roll dates to prevent spurious gap returns
        for rd in roll_dates_list:
            rd_ts = pd.Timestamp(rd)
            # Find the first bar on or after the roll date
            mask = prices.index.date == rd if hasattr(prices.index, "date") else False
            if isinstance(mask, np.ndarray) and mask.any():
                first_idx = prices.index[mask][0]
                log_returns.loc[first_idx] = 0.0

    # Build active contract map
    active = pd.Series(index=prices.index, dtype="object")
    if roll_calendar:
        active[:] = roll_calendar[0].from_contract
        for roll in roll_calendar:
            mask = prices.index.date >= roll.roll_date
            active[mask] = roll.to_contract
    else:
        active[:] = "UNKNOWN"

    return ContinuousSeries(
        prices=prices,
        returns=log_returns,
        active_contract=active,
        roll_dates=roll_dates_list,
        series_type="research",
    )


def build_tradable_series(
    prices: pd.DataFrame,
    roll_calendar: list[RollDate],
    spread_cost_per_roll: float = 0.0,
) -> ContinuousSeries:
    """Build a tradable simulated series that reflects actual execution.

    This series follows the exact roll schedule and deducts roll costs.
    Returns are the returns you would actually earn trading this series.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV data.
    roll_calendar : list[RollDate]
        Roll schedule.
    spread_cost_per_roll : float
        Cost in price units deducted at each roll (bid-ask crossing).

    Returns
    -------
    ContinuousSeries
    """
    roll_dates_set = {r.roll_date for r in roll_calendar}
    roll_dates_list = sorted(roll_dates_set)

    # Start with log returns
    log_returns = np.log(prices["close"] / prices["close"].shift(1))

    # At roll dates: null the gap return and deduct spread cost
    for rd in roll_dates_list:
        mask = prices.index.date == rd
        if isinstance(mask, np.ndarray) and mask.any():
            first_idx = prices.index[mask][0]
            # Zero out the roll gap return
            log_returns.loc[first_idx] = 0.0
            # Deduct roll cost as a return impact
            if spread_cost_per_roll > 0 and prices["close"].loc[first_idx] > 0:
                cost_return = spread_cost_per_roll / prices["close"].loc[first_idx]
                log_returns.loc[first_idx] -= cost_return

    active = pd.Series(index=prices.index, dtype="object")
    if roll_calendar:
        active[:] = roll_calendar[0].from_contract
        for roll in roll_calendar:
            mask = prices.index.date >= roll.roll_date
            active[mask] = roll.to_contract
    else:
        active[:] = "UNKNOWN"

    return ContinuousSeries(
        prices=prices,
        returns=log_returns,
        active_contract=active,
        roll_dates=roll_dates_list,
        series_type="tradable",
    )
