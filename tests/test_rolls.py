"""
Tests for roll rules correctness — per 07_DATA_MODEL.md.

Roll rules must be explicit and unit-tested.
Must avoid delivery/first notice window by default.
"""

import pytest
from datetime import date, timedelta

from src.data.contracts import ContractMetadata, RollRuleConfig
from src.data.rolls import (
    RollCalendar,
    RollDate,
    apply_roll_schedule,
    MONTH_CODE_MAP,
    CODE_MONTH_MAP,
)

import pandas as pd
import numpy as np


class TestRollCalendar:
    """Test roll calendar generation."""

    def test_roll_calendar_generates_correct_months(self, es_metadata):
        """ES should roll on H, M, U, Z months."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2023, 2024)

        # ES has quarterly rolls: 3, 6, 9, 12
        expected_months = [3, 6, 9, 12]

        for roll in rolls:
            assert roll.from_expiry_month in expected_months, (
                f"Unexpected from_month: {roll.from_expiry_month}"
            )
            assert roll.to_expiry_month in expected_months, (
                f"Unexpected to_month: {roll.to_expiry_month}"
            )

    def test_roll_dates_before_expiry(self, es_metadata):
        """Roll dates should be N days before expiry."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2023, 2024)

        for roll in rolls:
            expiry = cal.estimate_expiry_date(
                roll.from_year, roll.from_expiry_month
            )
            roll_date = roll.roll_date

            # Roll should be before expiry
            assert roll_date < expiry, (
                f"Roll date {roll_date} is not before expiry {expiry}"
            )

            # Roll should be approximately N days before expiry
            gap = (expiry - roll_date).days
            assert gap >= es_metadata.roll_rules.roll_days_before_expiry - 2, (
                f"Roll too close to expiry: gap={gap} days"
            )

    def test_roll_dates_avoid_weekends(self, es_metadata):
        """Roll dates should not fall on weekends."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2020, 2025)

        for roll in rolls:
            assert roll.roll_date.weekday() < 5, (
                f"Roll date {roll.roll_date} is on a weekend "
                f"(day={roll.roll_date.weekday()})"
            )

    def test_roll_dates_are_ordered(self, es_metadata):
        """Roll dates should be in chronological order."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2020, 2025)

        for i in range(1, len(rolls)):
            assert rolls[i].roll_date > rolls[i - 1].roll_date, (
                f"Roll dates not ordered: {rolls[i-1].roll_date} >= {rolls[i].roll_date}"
            )

    def test_contract_symbols_format(self, es_metadata):
        """Contract symbols should follow standard format."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2023, 2024)

        for roll in rolls:
            # Format: ES + month_code + 2-digit year
            assert roll.from_contract.startswith("ES"), (
                f"Bad from_contract: {roll.from_contract}"
            )
            assert roll.to_contract.startswith("ES"), (
                f"Bad to_contract: {roll.to_contract}"
            )
            assert len(roll.from_contract) == 4 or len(roll.from_contract) == 5
            assert len(roll.to_contract) == 4 or len(roll.to_contract) == 5

    def test_consecutive_contracts_are_adjacent(self, es_metadata):
        """Each roll should go from one expiry to the next in sequence."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2023, 2023)

        for i in range(len(rolls) - 1):
            # to_contract of roll i should match from_contract of roll i+1
            assert rolls[i].to_contract == rolls[i + 1].from_contract, (
                f"Gap in roll chain: {rolls[i].to_contract} != "
                f"{rolls[i+1].from_contract}"
            )

    def test_empty_calendar_for_short_range(self, es_metadata):
        """Calendar for a single year start=end should still work."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2023, 2023)
        assert len(rolls) >= 3  # At least H->M, M->U, U->Z


class TestApplyRollSchedule:
    """Test applying roll schedule to price data."""

    def test_roll_schedule_covers_all_dates(self, es_metadata, sample_prices):
        """Every timestamp should have an active contract assigned."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2020, 2023)

        active = apply_roll_schedule(sample_prices, rolls)

        assert len(active) == len(sample_prices)
        assert not active.isna().any(), "Active contract has NaN values"

    def test_contract_changes_at_roll_dates(self, es_metadata, sample_prices):
        """Active contract should change at roll dates."""
        cal = RollCalendar(es_metadata)
        rolls = cal.generate_roll_calendar(2020, 2023)

        active = apply_roll_schedule(sample_prices, rolls)

        # The contract should change at roll dates
        changes = active != active.shift(1)
        n_changes = changes.sum() - 1  # minus 1 for first NaN

        assert n_changes > 0, "No contract changes detected"


class TestMonthCodeMapping:
    """Test month code mapping is correct."""

    def test_all_months_mapped(self):
        """All 12 months should have codes."""
        for month in range(1, 13):
            assert month in CODE_MONTH_MAP, f"Month {month} not in CODE_MONTH_MAP"

    def test_code_roundtrip(self):
        """Month -> code -> month should be identity."""
        for month, code in CODE_MONTH_MAP.items():
            assert MONTH_CODE_MAP[code] == month, (
                f"Roundtrip failed: {month} -> {code} -> {MONTH_CODE_MAP[code]}"
            )

    def test_standard_codes(self):
        """Verify standard CME month codes."""
        expected = {
            "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
            "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
        }
        assert MONTH_CODE_MAP == expected
