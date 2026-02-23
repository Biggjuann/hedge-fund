"""
Tests for chop regime detection — per Jungle Rock Paper Part 2.

Verify:
- Valid labels ('chop' or 'not_chop')
- Detected in sideways, volatile market
- Causality via truncation test
"""

import numpy as np
import pandas as pd
import pytest

from src.features.regime import RegimeTagger


class TestChopDetector:
    """Test chop regime detection."""

    def test_valid_labels(self, sample_prices, sample_returns):
        """Chop regime must produce valid labels."""
        tagger = RegimeTagger()
        labels = tagger.tag_all(sample_prices, sample_returns)

        assert labels.chop_regime is not None
        valid = labels.chop_regime.dropna()
        unique = set(valid.unique())
        assert unique.issubset({"chop", "not_chop"}), (
            f"Invalid chop labels: {unique}"
        )

    def test_detected_in_sideways_market(self):
        """Chop should be detected in volatile, sideways market."""
        np.random.seed(123)
        dates = pd.bdate_range("2018-01-02", "2023-01-02", freq="B")
        n = len(dates)

        # Sideways market: mean-reverting with moderate volatility
        close_vals = [4000.0]
        for i in range(1, n):
            # Mean revert to 4000 with noise
            shock = np.random.normal(0, 0.015)
            reversion = -0.01 * (close_vals[-1] - 4000) / 4000
            ret = reversion + shock
            close_vals.append(close_vals[-1] * np.exp(ret))

        close = np.array(close_vals)
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = close * (1 + np.random.normal(0, 0.003, n))
        volume = np.random.randint(500000, 2000000, n).astype(float)

        prices = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        returns = np.log(prices["close"] / prices["close"].shift(1)).dropna()

        tagger = RegimeTagger()
        labels = tagger.tag_all(prices, returns)

        # In a sideways volatile market, there should be some chop detected
        chop_count = (labels.chop_regime == "chop").sum()
        assert chop_count > 0, "No chop detected in sideways volatile market"

    def test_causality_truncation(self, sample_prices, sample_returns):
        """Truncating last 50 bars should not change earlier chop labels."""
        tagger = RegimeTagger()

        full_labels = tagger.tag_all(sample_prices, sample_returns)
        trunc_prices = sample_prices.iloc[:-50]
        trunc_returns = sample_returns.iloc[:-50]
        trunc_labels = tagger.tag_all(trunc_prices, trunc_returns)

        # Compare on overlap (skip some warmup for expanding windows)
        overlap_idx = trunc_labels.chop_regime.dropna().index[300:]
        if len(overlap_idx) > 100:
            full_chop = full_labels.chop_regime.reindex(overlap_idx)
            trunc_chop = trunc_labels.chop_regime.reindex(overlap_idx)
            mismatches = (full_chop != trunc_chop).sum()
            assert mismatches == 0, (
                f"Causality violated: {mismatches} chop label mismatches"
            )

    def test_composite_risk_4factor(self, sample_prices, sample_returns):
        """Composite risk score should use 4-factor model with bear/chop."""
        tagger = RegimeTagger()
        labels = tagger.tag_all(sample_prices, sample_returns)

        # Composite should be bounded [0, 1]
        valid = labels.composite_risk.dropna()
        assert (valid >= 0.0).all(), f"Composite below 0: min={valid.min()}"
        assert (valid <= 1.0).all(), f"Composite above 1: max={valid.max()}"

        # Verify it uses 4-factor by checking it differs from 2-factor
        vol_score = labels.vol_regime.map({"low": 0.0, "med": 0.5, "high": 1.0}).fillna(0.5)
        corr_score = labels.corr_regime.map({"low": 0.0, "high": 1.0}).fillna(0.5)
        two_factor = 0.6 * vol_score + 0.4 * corr_score

        # 4-factor should differ from 2-factor
        diff = (labels.composite_risk - two_factor).abs()
        assert diff.sum() > 0, "Composite risk appears to be 2-factor only"
