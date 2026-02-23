"""
Tests for trend signal aggregation — per 04_STRATEGY_LIBRARY.md spec.

V3: Updated to test spec-conforming sign(return) + mean aggregation.

Verify:
- Signal in [-1, +1]
- sign(return) produces discrete levels {-1, 0, +1} per horizon
- Ensemble mean of sign signals produces correct discrete values
- Causality via truncation
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.trend import TrendStrategy


class TestSignMeanAggregation:
    """Test spec-conforming sign(return) mean aggregation for TrendStrategy."""

    def test_signal_in_valid_range(self, sample_prices, sample_returns):
        """Signals must be in [-1, +1]."""
        strategy = TrendStrategy(
            lookbacks=[21, 63, 252],
        )
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna()
        assert (valid >= -1.0).all().all(), "Signal below -1"
        assert (valid <= 1.0).all().all(), "Signal above +1"

    def test_produces_discrete_levels(self, sample_prices, sample_returns):
        """With 3 horizons of sign(), mean should produce specific discrete levels."""
        strategy = TrendStrategy(
            lookbacks=[21, 63, 252],
        )
        sig = strategy.generate_signals(sample_prices, sample_returns)

        valid = sig.signals.dropna().iloc[:, 0]
        unique_vals = sorted(valid.unique())

        # sign() with 3 horizons: mean of {-1,0,+1} values
        # Possible: -1, -0.33, +0.33, +1 (and 0 from NaN periods)
        assert len(unique_vals) <= 8, (
            f"Expected ~4-5 discrete levels, got {len(unique_vals)}: {unique_vals}"
        )
        assert len(unique_vals) >= 2, (
            f"Expected at least 2 levels, got {len(unique_vals)}"
        )

    def test_sign_signal_values(self, sample_prices, sample_returns):
        """Per-horizon signals should be exactly {-1, 0, +1} from sign()."""
        strategy = TrendStrategy(
            lookbacks=[21, 63, 252],
        )
        sig = strategy.generate_signals(sample_prices, sample_returns)

        horizon_signals = sig.metadata["horizon_signals"]
        for col in horizon_signals.columns:
            valid = horizon_signals[col].dropna()
            unique = set(valid.unique())
            expected = {-1.0, 0.0, 1.0}
            assert unique.issubset(expected), (
                f"Horizon {col} has non-sign values: {unique - expected}"
            )

    def test_causality_truncation(self, sample_prices, sample_returns):
        """Truncating last 50 bars should not change earlier signals."""
        strategy = TrendStrategy(
            lookbacks=[21, 63, 252],
            rebalance_freq="daily",
        )

        full_sig = strategy.generate_signals(sample_prices, sample_returns)
        trunc_sig = strategy.generate_signals(
            sample_prices.iloc[:-50], sample_returns.iloc[:-50]
        )

        full_vals = full_sig.signals.iloc[:, 0]
        trunc_vals = trunc_sig.signals.iloc[:, 0]

        overlap = trunc_vals.dropna().index
        if len(overlap) > 100:
            full_aligned = full_vals.reindex(overlap)
            diff = (full_aligned - trunc_vals.reindex(overlap)).abs()
            assert diff.max() < 1e-10, (
                f"Causality violated: max diff = {diff.max()}"
            )

    def test_metadata_contains_aggregation_method(self, sample_prices, sample_returns):
        """Metadata should indicate the aggregation method used."""
        strategy = TrendStrategy(
            lookbacks=[21, 63, 252],
        )
        sig = strategy.generate_signals(sample_prices, sample_returns)

        assert sig.metadata.get("aggregation_method") == "sign_mean"
        assert sig.metadata.get("n_horizons") == 3

    def test_different_lookbacks_produce_different_signals(self, sample_prices, sample_returns):
        """Different lookback sets should produce different signal series."""
        strat_a = TrendStrategy(lookbacks=[21, 63, 252])
        strat_b = TrendStrategy(lookbacks=[10, 42, 126])

        sig_a = strat_a.generate_signals(sample_prices, sample_returns)
        sig_b = strat_b.generate_signals(sample_prices, sample_returns)

        a_vals = sig_a.signals.dropna().iloc[:, 0]
        b_vals = sig_b.signals.dropna().iloc[:, 0]

        common = a_vals.index.intersection(b_vals.index)
        if len(common) > 100:
            diffs = (a_vals.loc[common] != b_vals.loc[common]).sum()
            assert diffs > 0, "Different lookbacks produced identical signals"


class TestVolManagement:
    """Test portfolio-level volatility management."""

    def test_vol_management_scales_weights(self):
        """Vol management should scale weights toward vol target."""
        from src.portfolio.construction import apply_vol_management

        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", "2022-01-02", freq="B")
        n = len(dates)

        # Create weights and returns
        weights = pd.DataFrame({"ES": np.ones(n) * 0.5}, index=dates)
        returns = pd.Series(np.random.normal(0.0003, 0.015, n), index=dates)

        scaled = apply_vol_management(weights, returns, vol_target=0.22)

        assert isinstance(scaled, pd.DataFrame)
        assert len(scaled) == len(weights)
        # Weights should be different from original (scaling applied)
        diff = (scaled["ES"] - weights["ES"]).abs()
        assert diff.sum() > 0, "Vol management didn't change weights"

    def test_vol_management_respects_cap(self):
        """Vol management should not scale above cap multiplier."""
        from src.portfolio.construction import apply_vol_management

        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", "2022-01-02", freq="B")
        n = len(dates)

        weights = pd.DataFrame({"ES": np.ones(n) * 0.1}, index=dates)
        # Very low vol returns -> would want to scale up a lot
        returns = pd.Series(np.random.normal(0, 0.001, n), index=dates)

        scaled = apply_vol_management(
            weights, returns, vol_target=0.22, vol_cap_multiplier=2.0
        )

        # Max scale should be 2.0x the original
        max_ratio = (scaled["ES"] / weights["ES"]).max()
        assert max_ratio <= 2.01, f"Exceeded cap: max ratio = {max_ratio:.3f}"
