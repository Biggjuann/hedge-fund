"""
Roll calendar and roll rule engine — per 07_DATA_MODEL.md.

Rules:
- Must be explicit and unit-tested.
- Avoid delivery/first notice window by default.
- Calendar roll windows OR liquidity-based rolls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.contracts import ContractMetadata, RollRuleConfig


# CME futures month codes
MONTH_CODE_MAP = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
CODE_MONTH_MAP = {v: k for k, v in MONTH_CODE_MAP.items()}


@dataclass
class RollDate:
    """A single roll event."""

    roll_date: date  # date to execute the roll
    from_contract: str  # e.g., 'ESH24'
    to_contract: str  # e.g., 'ESM24'
    from_expiry_month: int
    to_expiry_month: int
    from_year: int
    to_year: int


class RollCalendar:
    """Generates roll dates for a futures contract based on explicit rules.

    Per spec: calendar roll method, N days before expiry, avoiding
    delivery/first-notice windows.
    """

    def __init__(self, metadata: ContractMetadata):
        self.metadata = metadata
        self.rules = metadata.roll_rules

    def get_expiry_months(self, year: int) -> list[tuple[int, int]]:
        """Get all (year, month) expiry pairs for a given year."""
        return [(year, m) for m in self.rules.expiry_months]

    def estimate_expiry_date(self, year: int, month: int) -> date:
        """Estimate the third Friday of expiry month (standard CME).

        This is an approximation; real systems use exchange calendars.
        """
        # Third Friday: find first day of month, then third Friday
        first_day = date(year, month, 1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)
        return third_friday

    def get_roll_date(self, year: int, month: int) -> date:
        """Compute the roll date: N business days before estimated expiry.

        Per spec: avoid delivery window.
        """
        expiry = self.estimate_expiry_date(year, month)
        roll_date = expiry - timedelta(days=self.rules.roll_days_before_expiry)

        # Skip weekends
        while roll_date.weekday() >= 5:
            roll_date -= timedelta(days=1)

        return roll_date

    def generate_roll_calendar(
        self,
        start_year: int,
        end_year: int,
    ) -> list[RollDate]:
        """Generate full roll calendar between start and end years.

        Returns ordered list of RollDate objects.
        """
        # Collect all expiry (year, month) pairs
        all_expiries: list[tuple[int, int]] = []
        for year in range(start_year, end_year + 1):
            for month in self.rules.expiry_months:
                all_expiries.append((year, month))

        if len(all_expiries) < 2:
            return []

        rolls: list[RollDate] = []
        for i in range(len(all_expiries) - 1):
            from_year, from_month = all_expiries[i]
            to_year, to_month = all_expiries[i + 1]

            from_code = CODE_MONTH_MAP[from_month]
            to_code = CODE_MONTH_MAP[to_month]

            from_yr_str = str(from_year)[-2:]
            to_yr_str = str(to_year)[-2:]

            roll_dt = self.get_roll_date(from_year, from_month)

            rolls.append(
                RollDate(
                    roll_date=roll_dt,
                    from_contract=f"{self.metadata.symbol}{from_code}{from_yr_str}",
                    to_contract=f"{self.metadata.symbol}{to_code}{to_yr_str}",
                    from_expiry_month=from_month,
                    to_expiry_month=to_month,
                    from_year=from_year,
                    to_year=to_year,
                )
            )

        return rolls


def apply_roll_schedule(
    prices: pd.DataFrame,
    roll_calendar: list[RollDate],
) -> pd.Series:
    """Create a series indicating the active contract at each timestamp.

    For continuous unadjusted data, this maps each date to its active
    contract symbol (used for cost attribution and roll tracking).

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex.
    roll_calendar : list[RollDate]
        Ordered list of roll events.

    Returns
    -------
    pd.Series
        Series with same index as prices, values are contract symbols.
    """
    if not roll_calendar:
        return pd.Series("UNKNOWN", index=prices.index)

    dates = prices.index.date if hasattr(prices.index, "date") else prices.index
    contract_map = pd.Series(index=prices.index, dtype="object")

    # Before first roll: use the from_contract of first roll
    contract_map[:] = roll_calendar[0].from_contract

    for roll in roll_calendar:
        mask = dates >= roll.roll_date
        contract_map[mask] = roll.to_contract

    return contract_map


def detect_roll_gaps(
    close: pd.Series,
    roll_dates: list[date],
    max_gap_pct: float = 0.05,
) -> list[dict]:
    """Detect abnormal price gaps at roll dates (quality check).

    Parameters
    ----------
    close : pd.Series
        Close price series with DatetimeIndex.
    roll_dates : list[date]
        Dates when rolls occur.
    max_gap_pct : float
        Maximum acceptable gap as fraction of price.

    Returns
    -------
    list[dict]
        List of problematic roll gaps with details.
    """
    issues = []
    daily_close = close.resample("1D").last().dropna()

    for rd in roll_dates:
        rd_ts = pd.Timestamp(rd)
        # Find nearest trading days
        idx = daily_close.index.searchsorted(rd_ts)
        if idx < 1 or idx >= len(daily_close):
            continue

        pre_price = daily_close.iloc[idx - 1]
        post_price = daily_close.iloc[idx]
        gap_pct = abs(post_price - pre_price) / pre_price

        if gap_pct > max_gap_pct:
            issues.append(
                {
                    "roll_date": rd,
                    "pre_price": pre_price,
                    "post_price": post_price,
                    "gap_pct": gap_pct,
                }
            )

    return issues
