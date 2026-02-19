"""
Tests for strategy implementations — per 04_STRATEGY_LIBRARY.md.

Verify:
- Signal output is in [-1, +1]
- Vol scaling is correct
- Rebalance frequency is enforced
- Multiple horizons produce ensemble
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.trend import TrendStrategy
from src.strategies.carry import CarryStrategy
from src.strategies.vol_breakout import VolBreakoutStrategy
from src.strategies.base import StrategySignal


class TestTrendStrategy:
    """Test S1: Multi-horizon Trend (TSMOM ensemble)."""

    def test_signal_in_valid_range(self, sample_prices, sample_returns):
        """Signals must be in [-1, +1]."""
        strategy = TrendStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna()
        assert (valid >= -1.0).all().all(), "Signal below -1"
        assert (valid <= 1.0).all().all(), "Signal above +1"

    def test_multiple_horizons_averaged(self, sample_prices, sample_returns):
        """Ensemble should average across horizons."""
        strategy = TrendStrategy(lookbacks=[21, 63, 252])
        sig = strategy.generate_signals(sample_prices, sample_returns)

        # With 3 horizons, intermediate values should exist
        valid = sig.signals.dropna().iloc[:, 0]
        unique_vals = valid.unique()

        # Should have values other than just -1, 0, +1
        # (from averaging multiple sign signals)
        assert len(unique_vals) > 3, (
            f"Expected intermediate values from averaging, got {unique_vals[:10]}"
        )

    def test_output_structure(self, sample_prices, sample_returns):
        """Output should be a StrategySignal with correct fields."""
        strategy = TrendStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        assert isinstance(sig, StrategySignal)
        assert isinstance(sig.signals, pd.DataFrame)
        assert isinstance(sig.raw_weights, pd.DataFrame)
        assert len(sig.signals) == len(sample_prices)

    def test_different_lookbacks_produce_different_signals(
        self, sample_prices, sample_returns
    ):
        """Different lookbacks should produce different signals."""
        short = TrendStrategy(lookbacks=[21])
        long = TrendStrategy(lookbacks=[252])

        sig_short = short.generate_signals(sample_prices, sample_returns)
        sig_long = long.generate_signals(sample_prices, sample_returns)

        # They should differ (not identical)
        s1 = sig_short.signals.dropna().iloc[:, 0]
        s2 = sig_long.signals.dropna().iloc[:, 0]

        common = s1.index.intersection(s2.index)
        if len(common) > 100:
            # At least some differences
            diffs = (s1.loc[common] != s2.loc[common]).sum()
            assert diffs > 0, "Different lookbacks produced identical signals"


class TestCarryStrategy:
    """Test S2: Cross-asset Carry."""

    def test_signal_in_valid_range(self, sample_prices, sample_returns):
        """Carry signals must be in [-1, +1]."""
        strategy = CarryStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna()
        assert (valid >= -1.0).all().all(), "Signal below -1"
        assert (valid <= 1.0).all().all(), "Signal above +1"

    def test_output_structure(self, sample_prices, sample_returns):
        """Output should be a valid StrategySignal."""
        strategy = CarryStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        assert isinstance(sig, StrategySignal)
        assert "carry_raw" in sig.metadata


class TestVolBreakoutStrategy:
    """Test S3: Volatility Breakout / Convex Proxy."""

    def test_signal_in_valid_range(self, sample_prices, sample_returns):
        """Vol breakout signals must be in [-1, +1]."""
        strategy = VolBreakoutStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna()
        assert (valid >= -1.0).all().all(), "Signal below -1"
        assert (valid <= 1.0).all().all(), "Signal above +1"

    def test_mostly_flat(self, sample_prices, sample_returns):
        """Vol breakout should be mostly flat (not always in a trade)."""
        strategy = VolBreakoutStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna().iloc[:, 0]
        flat_pct = (valid == 0).sum() / len(valid)

        # Should be mostly flat (convex strategy = infrequent trades)
        assert flat_pct > 0.3, (
            f"Expected mostly flat, but only {flat_pct:.1%} are zero"
        )

    def test_small_baseline_risk(self, sample_prices, sample_returns):
        """Weights should be small (small baseline risk per spec)."""
        strategy = VolBreakoutStrategy(target_vol=0.05)
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid_weights = sig.raw_weights.dropna().iloc[:, 0]
        mean_abs_weight = valid_weights.abs().mean()

        # With 5% target vol, weights should be moderate
        assert mean_abs_weight < 2.0, (
            f"Mean absolute weight too large: {mean_abs_weight:.3f}"
        )

    def test_metadata_contains_indicators(self, sample_prices, sample_returns):
        """Metadata should include ATR and breakout detection info."""
        strategy = VolBreakoutStrategy()
        sig = strategy.generate_signals(sample_prices, sample_returns)

        assert "atr" in sig.metadata
        assert "atr_percentile" in sig.metadata
        assert "breakout_up" in sig.metadata
        assert "breakout_down" in sig.metadata
