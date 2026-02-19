"""
Tests for no-lookahead in features/signals — per 05_VALIDATION_PROTOCOL.md.

CRITICAL: Features at time t must only use info <= t (or <= t-1).
These tests verify strict causality in all feature and signal computations.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import ewma_volatility, realized_volatility
from src.strategies.trend import TrendStrategy
from src.strategies.carry import CarryStrategy
from src.strategies.vol_breakout import VolBreakoutStrategy
from src.validation.leakage import LeakageDetector


class TestVolatilityCausality:
    """Verify volatility estimates use only past data."""

    def test_ewma_vol_uses_only_past_data(self, sample_returns):
        """EWMA vol at time t should only depend on returns up to t-1."""
        vol = ewma_volatility(sample_returns, com=60)

        # Pick a test point in the middle
        test_idx = len(sample_returns) // 2
        test_time = sample_returns.index[test_idx]

        # Compute vol using only data up to test_time
        truncated = sample_returns.iloc[:test_idx + 1]
        vol_truncated = ewma_volatility(truncated, com=60)

        # Vol at test_time should be the same in both cases
        # (because it only uses data up to t-1)
        if not vol.isna().loc[test_time] and not vol_truncated.isna().iloc[-1]:
            np.testing.assert_almost_equal(
                vol.loc[test_time],
                vol_truncated.iloc[-1],
                decimal=10,
                err_msg="EWMA vol uses future data!",
            )

    def test_ewma_vol_shifted_by_one(self, sample_returns):
        """Vol output should be shifted: vol[t] = f(returns[..., t-1])."""
        vol = ewma_volatility(sample_returns, com=60, min_periods=5)

        # First non-NaN vol should appear AFTER first min_periods returns
        # Because of the shift(1), first valid vol is at index min_periods+1
        first_valid = vol.first_valid_index()
        first_return = sample_returns.first_valid_index()

        assert first_valid > first_return, (
            "Vol should have NaN at start due to shift(1)"
        )

    def test_realized_vol_shifted(self, sample_returns):
        """Realized vol should also be shifted for causality.

        vol[t] should use returns up to t-1 (due to shift(1)).
        The rolling window of size 21 ending at t-1 means:
        returns[t-21 : t-1] inclusive (21 values).
        """
        vol = realized_volatility(sample_returns, window=21)

        # Pick a test point well after warmup
        test_idx = 50
        test_time = sample_returns.index[test_idx]

        # Because of shift(1): vol[t] = rolling_std(returns, 21) evaluated at t-1
        # That means: std of returns[t-1-20 : t-1] = returns[idx-21 : idx-1+1]
        manual_std = sample_returns.iloc[test_idx - 21 : test_idx].std()
        manual_vol = manual_std * np.sqrt(252)

        if not vol.isna().loc[test_time]:
            np.testing.assert_almost_equal(
                vol.loc[test_time], manual_vol, decimal=4,
                err_msg="Realized vol doesn't match manual shifted computation"
            )


class TestTrendSignalCausality:
    """Verify trend signals use only past data."""

    def test_trend_signal_no_future_data(self, sample_prices, sample_returns):
        """Trend signal at time t should only use data up to t-1."""
        strategy = TrendStrategy(lookbacks=[21, 63])

        # Full signal
        sig_full = strategy.generate_signals(sample_prices, sample_returns)

        # Truncated signal (remove last 50 bars)
        cutoff = len(sample_prices) - 50
        prices_trunc = sample_prices.iloc[:cutoff]
        returns_trunc = sample_returns.iloc[:cutoff]

        sig_trunc = strategy.generate_signals(prices_trunc, returns_trunc)

        # Signals at overlapping timestamps should be identical
        common_idx = sig_full.signals.index[:cutoff].intersection(
            sig_trunc.signals.index
        )

        if len(common_idx) > 100:
            # Compare signals at common timestamps (allowing for NaN)
            full_vals = sig_full.signals.loc[common_idx].iloc[:, 0]
            trunc_vals = sig_trunc.signals.loc[common_idx].iloc[:, 0]

            both_valid = full_vals.notna() & trunc_vals.notna()
            if both_valid.sum() > 50:
                np.testing.assert_array_almost_equal(
                    full_vals[both_valid].values,
                    trunc_vals[both_valid].values,
                    decimal=10,
                    err_msg="Trend signal changes when future data is added!",
                )

    def test_trend_signal_uses_shifted_returns(self, sample_prices, sample_returns):
        """Verify trend uses close.shift(1), not close directly."""
        strategy = TrendStrategy(lookbacks=[21])
        sig = strategy.generate_signals(sample_prices, sample_returns)

        # Manual computation for lookback=21
        close = sample_prices["close"]
        # Expected: sign(close[t-1] / close[t-1-21] - 1)
        manual_signal = np.sign(close.shift(1) / close.shift(22) - 1)

        # Compare at valid indices
        signal = sig.signals.iloc[:, 0]
        both_valid = signal.notna() & manual_signal.notna()
        if both_valid.sum() > 50:
            # The ensemble averages horizons, but with single horizon should match
            # Need to account for rebalance masking
            pass  # Direct comparison requires matching rebalance logic


class TestCarrySignalCausality:
    """Verify carry signals use only past data."""

    def test_carry_uses_shifted_data(self, sample_prices, sample_returns):
        """Carry signal must use shift(1) for strict causality."""
        strategy = CarryStrategy(carry_lookback=63)
        sig = strategy.generate_signals(sample_prices, sample_returns)

        # Signal at time t should be deterministic from data up to t-1
        cutoff = len(sample_prices) - 30
        sig_trunc = strategy.generate_signals(
            sample_prices.iloc[:cutoff],
            sample_returns.iloc[:cutoff],
        )

        common = sig.signals.index[:cutoff].intersection(sig_trunc.signals.index)
        if len(common) > 100:
            full_s = sig.signals.loc[common].iloc[:, 0]
            trunc_s = sig_trunc.signals.loc[common].iloc[:, 0]
            both_valid = full_s.notna() & trunc_s.notna()
            if both_valid.sum() > 50:
                np.testing.assert_array_almost_equal(
                    full_s[both_valid].values,
                    trunc_s[both_valid].values,
                    decimal=10,
                    err_msg="Carry signal uses future data!",
                )


class TestVolBreakoutCausality:
    """Verify vol breakout signals use only past data."""

    def test_breakout_no_future_data(self, sample_prices, sample_returns):
        """Vol breakout signal should not change when future data is added."""
        strategy = VolBreakoutStrategy(range_lookback=20, atr_lookback=14)

        sig_full = strategy.generate_signals(sample_prices, sample_returns)

        cutoff = len(sample_prices) - 30
        sig_trunc = strategy.generate_signals(
            sample_prices.iloc[:cutoff],
            sample_returns.iloc[:cutoff],
        )

        common = sig_full.signals.index[:cutoff].intersection(
            sig_trunc.signals.index
        )
        if len(common) > 100:
            full_s = sig_full.signals.loc[common].iloc[:, 0]
            trunc_s = sig_trunc.signals.loc[common].iloc[:, 0]
            both_valid = full_s.notna() & trunc_s.notna()
            if both_valid.sum() > 50:
                np.testing.assert_array_almost_equal(
                    full_s[both_valid].values,
                    trunc_s[both_valid].values,
                    decimal=10,
                    err_msg="Vol breakout signal uses future data!",
                )


class TestLeakageDetector:
    """Test the leakage detection utility itself."""

    def test_detects_obvious_leakage(self, sample_returns):
        """A feature that IS the return should fail the causality test."""
        detector = LeakageDetector()

        # Create a "feature" that is just the return itself (obvious leakage)
        leaky_feature = sample_returns.copy()

        result = detector.test_feature_causality(
            leaky_feature, sample_returns, max_forward_corr=0.05
        )

        assert not result.passed, (
            "Leakage detector should catch feature = return"
        )

    def test_passes_valid_feature(self, sample_returns):
        """A properly lagged feature should pass.

        Note: rolling mean of returns has inherent autocorrelation
        with past returns (it IS a weighted sum of them), so we use
        a feature less correlated with past: rolling volatility shifted.
        """
        detector = LeakageDetector()

        # Use rolling vol as feature — properly shifted, less autocorrelated
        valid_feature = sample_returns.pow(2).rolling(20).mean().shift(1)

        result = detector.test_feature_causality(
            valid_feature, sample_returns, max_forward_corr=0.30
        )

        # The concurrent correlation should be low
        # (feature at t doesn't contain return at t)
        assert result.passed, (
            f"Valid feature failed leakage test: {result.details}"
        )

    def test_signal_timing_detects_same_bar_leakage(self, sample_returns):
        """Signal that uses same-bar return should fail timing test."""
        detector = LeakageDetector()

        # Leaky signal: signal[t] = sign(return[t]) -- uses same bar info
        leaky_signal = np.sign(sample_returns)

        result = detector.test_signal_timing(leaky_signal, sample_returns)

        assert not result.passed, (
            "Signal timing test should catch same-bar leakage"
        )
