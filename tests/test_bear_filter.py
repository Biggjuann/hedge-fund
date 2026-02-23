"""
Tests for bear probability filter — per Jungle Rock Paper Part 3.

Verify:
- Bear probability is in [0, 1]
- Causality: truncating last N bars doesn't change earlier values
- High in declining market, low in rally
"""

import numpy as np
import pandas as pd
import pytest

from src.features.regime import RegimeTagger


class TestBearProbability:
    """Test bear probability computation."""

    def test_bear_prob_in_valid_range(self, sample_prices):
        """Bear probability must be in [0, 1]."""
        tagger = RegimeTagger()
        bear_prob = tagger.compute_bear_probability(sample_prices["close"])

        valid = bear_prob.dropna()
        assert len(valid) > 0, "No valid bear probability values"
        assert (valid >= 0.0).all(), f"Bear prob below 0: min={valid.min()}"
        assert (valid <= 1.0).all(), f"Bear prob above 1: max={valid.max()}"

    def test_causality_truncation(self, sample_prices):
        """Truncating last 50 bars should not change earlier values."""
        tagger = RegimeTagger()

        full = tagger.compute_bear_probability(sample_prices["close"])
        truncated = tagger.compute_bear_probability(sample_prices["close"].iloc[:-50])

        # Compare on the overlap
        overlap_idx = truncated.dropna().index
        if len(overlap_idx) > 100:
            full_aligned = full.reindex(overlap_idx)
            diff = (full_aligned - truncated.reindex(overlap_idx)).abs()
            # Small tolerance for floating point in expanding operations
            assert diff.max() < 1e-10, (
                f"Causality violated: max diff = {diff.max()}"
            )

    def test_high_in_decline(self):
        """Bear probability should be higher during declining markets."""
        np.random.seed(42)
        dates = pd.bdate_range("2018-01-02", "2023-01-02", freq="B")
        n = len(dates)

        # Create market with rally then crash
        rally = np.random.normal(0.001, 0.008, n // 2)
        crash = np.random.normal(-0.002, 0.018, n - n // 2)
        returns = np.concatenate([rally, crash])
        close = 4000 * np.exp(np.cumsum(returns))
        close = pd.Series(close, index=dates)

        tagger = RegimeTagger()
        bear_prob = tagger.compute_bear_probability(close)

        # Compare mean bear prob in rally vs crash period (skip warmup)
        warmup = 300
        rally_bear = bear_prob.iloc[warmup : n // 2].mean()
        crash_bear = bear_prob.iloc[n // 2 + 50 :].mean()

        assert crash_bear > rally_bear, (
            f"Bear prob should be higher in crash ({crash_bear:.3f}) "
            f"than rally ({rally_bear:.3f})"
        )

    def test_low_in_rally(self, sample_prices):
        """Bear probability should be moderate-to-low in upward trending market."""
        tagger = RegimeTagger()
        bear_prob = tagger.compute_bear_probability(sample_prices["close"])

        valid = bear_prob.dropna()
        # Sample prices have slight positive drift (0.0003), so bear prob
        # should not be persistently near 1
        assert valid.mean() < 0.80, (
            f"Mean bear prob too high for upward market: {valid.mean():.3f}"
        )

    def test_integrated_in_tag_all(self, sample_prices, sample_returns):
        """Bear probability should be included in RegimeLabels from tag_all."""
        tagger = RegimeTagger()
        labels = tagger.tag_all(sample_prices, sample_returns)

        assert labels.bear_probability is not None
        valid = labels.bear_probability.dropna()
        assert len(valid) > 0
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()
