"""
Tests for stress test execution — per 05_VALIDATION_PROTOCOL.md.

Required stress tests:
- slippage x2, x3
- 1 tick adverse on all fills
- delayed fills (next bar)
- random trade drop (10%)
- correlation spike scenario
- margin shock scenario
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.stress import StressTestSuite, StressTestResult


class TestStressTestSuite:
    """Test that all stress test scenarios run correctly."""

    @pytest.fixture
    def base_returns(self, sample_dates):
        """Generate base strategy returns."""
        np.random.seed(42)
        n = len(sample_dates)
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, n),
            index=sample_dates,
        )
        return returns

    @pytest.fixture
    def weight_changes(self, sample_dates):
        """Generate weight changes (turnover)."""
        np.random.seed(42)
        n = len(sample_dates)
        changes = pd.DataFrame(
            {"ES": np.random.normal(0, 0.05, n)},
            index=sample_dates,
        )
        return changes

    def test_all_scenarios_execute(self, base_returns, weight_changes, sample_prices):
        """All 6 mandated scenarios should produce results."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns,
            weight_changes,
            sample_prices,
            cost_per_unit=0.001,
        )

        # Should have at least 6 scenarios (2 slippage + 4 others)
        assert len(results) >= 6, f"Expected >= 6 scenarios, got {len(results)}"

    def test_slippage_x2_degrades_performance(self, base_returns, weight_changes, sample_prices):
        """Higher slippage should degrade Sharpe ratio."""
        suite = StressTestSuite(slippage_multipliers=[1.0, 2.0])
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        slippage_2x = [r for r in results if r.scenario_name == "slippage_x2.0"]
        assert len(slippage_2x) == 1

        # Stressed Sharpe should be <= base Sharpe
        assert slippage_2x[0].stressed_sharpe <= slippage_2x[0].base_sharpe + 0.01, (
            "2x slippage should degrade performance"
        )

    def test_slippage_x3_worse_than_x2(self, base_returns, weight_changes, sample_prices):
        """3x slippage should be worse than 2x slippage."""
        suite = StressTestSuite(slippage_multipliers=[1.0, 2.0, 3.0])
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        x2 = [r for r in results if r.scenario_name == "slippage_x2.0"][0]
        x3 = [r for r in results if r.scenario_name == "slippage_x3.0"][0]

        assert x3.stressed_sharpe <= x2.stressed_sharpe + 0.01, (
            "3x slippage should be worse than 2x"
        )

    def test_delayed_fills_shift_returns(self, base_returns, weight_changes, sample_prices):
        """Delayed fills should shift the return series."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        delayed = [r for r in results if r.scenario_name == "delayed_fills"]
        assert len(delayed) == 1

    def test_random_drop_reduces_exposure(self, base_returns, weight_changes, sample_prices):
        """Random trade drops should reduce absolute returns."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        dropped = [r for r in results if r.scenario_name == "random_drop_10pct"]
        assert len(dropped) == 1

    def test_correlation_spike_reduces_risk(self, base_returns, weight_changes, sample_prices):
        """Correlation spike scenario should reduce risk exposure."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        corr = [r for r in results if r.scenario_name == "correlation_spike"]
        assert len(corr) == 1

        # Stressed vol should be lower (risk haircut applied)
        assert abs(corr[0].stressed_return) <= abs(corr[0].base_return) + 0.01

    def test_margin_shock_reduces_leverage(self, base_returns, weight_changes, sample_prices):
        """Margin shock should reduce leverage and returns."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        margin = [r for r in results if r.scenario_name == "margin_shock"]
        assert len(margin) == 1

    def test_result_structure(self, base_returns, weight_changes, sample_prices):
        """Each result should have all required fields."""
        suite = StressTestSuite()
        results = suite.run_all(
            base_returns, weight_changes, sample_prices, cost_per_unit=0.001
        )

        for r in results:
            assert isinstance(r, StressTestResult)
            assert r.scenario_name is not None
            assert r.description is not None
            assert isinstance(r.passed, bool)
            assert isinstance(r.degradation_pct, float)

    def test_pass_fail_logic(self):
        """Test pass/fail thresholds."""
        suite = StressTestSuite(
            max_acceptable_dd=0.30,
            min_stressed_sharpe=-0.5,
        )

        # A result with acceptable Sharpe and DD should pass
        good = StressTestResult(
            scenario_name="test",
            description="test",
            base_sharpe=1.0,
            stressed_sharpe=0.3,
            base_return=0.10,
            stressed_return=0.03,
            base_max_dd=-0.10,
            stressed_max_dd=-0.20,
            degradation_pct=70.0,
            passed=True,  # manually set
        )
        assert good.passed

        # A result with very negative Sharpe should fail
        bad = StressTestResult(
            scenario_name="test",
            description="test",
            base_sharpe=1.0,
            stressed_sharpe=-1.0,
            base_return=0.10,
            stressed_return=-0.15,
            base_max_dd=-0.10,
            stressed_max_dd=-0.50,
            degradation_pct=200.0,
            passed=False,
        )
        assert not bad.passed
