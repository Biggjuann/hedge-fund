"""
Tests for Walk-Forward Optimization split integrity — per 05_VALIDATION_PROTOCOL.md.

Must verify:
- No overlap between train and test sets
- Purge gap is maintained
- Embargo is respected
- Folds are ordered and cover the data
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.wfo import WalkForwardOptimizer, WFOFold


class TestWFOFoldGeneration:
    """Test WFO fold boundary generation."""

    def test_no_train_test_overlap(self, sample_dates):
        """Train and test periods must not overlap."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_id}: train_end ({fold.train_end}) >= "
                f"test_start ({fold.test_start})"
            )

    def test_purge_gap_maintained(self, sample_dates):
        """Purge gap between train end and test start must exist."""
        purge_days = 5
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=purge_days
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            # Gap in business days (index positions)
            train_end_pos = sample_dates.get_loc(fold.train_end)
            test_start_pos = sample_dates.get_loc(fold.test_start)
            gap = test_start_pos - train_end_pos

            assert gap > purge_days, (
                f"Fold {fold.fold_id}: purge gap = {gap} positions, "
                f"need > {purge_days}"
            )

    def test_folds_are_ordered(self, sample_dates):
        """Fold start dates should be monotonically increasing."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        for i in range(1, len(folds)):
            assert folds[i].train_start > folds[i - 1].train_start, (
                f"Folds not ordered: fold {i-1} train_start >= fold {i}"
            )

    def test_train_within_data_bounds(self, sample_dates):
        """Train periods should be within data bounds."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            assert fold.train_start >= sample_dates[0], (
                f"Fold {fold.fold_id}: train starts before data"
            )
            assert fold.train_end <= sample_dates[-1], (
                f"Fold {fold.fold_id}: train extends beyond data"
            )

    def test_test_within_data_bounds(self, sample_dates):
        """Test periods should be within data bounds."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            assert fold.test_start >= sample_dates[0], (
                f"Fold {fold.fold_id}: test starts before data"
            )
            assert fold.test_end <= sample_dates[-1], (
                f"Fold {fold.fold_id}: test extends beyond data"
            )

    def test_minimum_train_size(self, sample_dates):
        """Each fold must have at least min_train_days of training data."""
        min_train = 504
        wfo = WalkForwardOptimizer(
            train_days=756, test_days=126, step_days=63,
            purge_days=5, min_train_days=min_train,
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            train_start_pos = sample_dates.get_loc(fold.train_start)
            train_end_pos = sample_dates.get_loc(fold.train_end)
            train_size = train_end_pos - train_start_pos + 1

            assert train_size >= min_train, (
                f"Fold {fold.fold_id}: train size = {train_size} < {min_train}"
            )

    def test_generates_multiple_folds(self, sample_dates):
        """Should generate at least 1 fold for 3 years of data."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        assert len(folds) >= 1, f"Expected at least 1 fold, got {len(folds)}"

    def test_empty_data_returns_empty(self):
        """Empty data should return no folds."""
        wfo = WalkForwardOptimizer()
        folds = wfo.generate_folds(pd.DatetimeIndex([]))
        assert len(folds) == 0

    def test_purge_window_is_correct(self, sample_dates):
        """Purge window boundaries should be correct."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)

        for fold in folds:
            assert fold.purge_start > fold.train_end, (
                f"Fold {fold.fold_id}: purge starts before train ends"
            )
            assert fold.purge_end < fold.test_start, (
                f"Fold {fold.fold_id}: purge ends after test starts"
            )


class TestWFOValidation:
    """Test WFO integrity validation."""

    def test_validate_good_folds(self, sample_dates):
        """Valid folds should pass validation."""
        wfo = WalkForwardOptimizer(
            train_days=504, test_days=126, step_days=63, purge_days=5
        )
        folds = wfo.generate_folds(sample_dates)
        issues = wfo.validate_fold_integrity(folds)

        assert len(issues) == 0, f"Unexpected issues: {issues}"

    def test_detects_overlap(self):
        """Should detect train/test overlap."""
        wfo = WalkForwardOptimizer(purge_days=5)

        bad_fold = WFOFold(
            fold_id=0,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-15"),
            test_start=pd.Timestamp("2022-01-10"),  # overlaps!
            test_end=pd.Timestamp("2022-06-30"),
            purge_start=pd.Timestamp("2022-01-11"),
            purge_end=pd.Timestamp("2022-01-14"),
        )

        issues = wfo.validate_fold_integrity([bad_fold])
        assert len(issues) > 0, "Should detect overlap"

    def test_trial_counting(self):
        """WFO should track unique param sets for multiple testing."""
        wfo = WalkForwardOptimizer()
        assert wfo.total_trials == 1  # max(0, 1)

        # Simulate adding unique param sets
        wfo._unique_param_sets.add("param_set_1")
        wfo._unique_param_sets.add("param_set_2")
        wfo._unique_param_sets.add("param_set_1")  # duplicate
        assert wfo.total_trials == 2  # only unique sets count


class TestWFORun:
    """Test WFO execution with a simple strategy function."""

    def test_wfo_produces_results(self, sample_prices, sample_returns):
        """WFO should produce fold results."""
        wfo = WalkForwardOptimizer(
            train_days=400, test_days=100, step_days=100,
            purge_days=5, min_train_days=300,
        )

        def simple_strategy(data, returns, params):
            lookback = params.get("lookback", 21)
            signal = np.sign(returns.rolling(lookback).mean().shift(1))
            return (signal * returns).dropna()

        param_grid = [
            {"lookback": 21},
            {"lookback": 63},
        ]

        folds = wfo.run(sample_prices, sample_returns, simple_strategy, param_grid)

        # Should produce at least 1 fold
        assert len(folds) >= 1

        for fold in folds:
            assert fold.best_params is not None
            assert fold.test_returns is not None
            assert len(fold.test_returns) > 0
