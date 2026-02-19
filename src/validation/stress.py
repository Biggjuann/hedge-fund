"""
Stress test suite — per 05_VALIDATION_PROTOCOL.md.

Required stress tests:
- slippage x2, x3
- 1 tick adverse on all fills
- delayed fills (next bar)
- random trade drop (simulate missed signals)
- correlation spike scenario (risk haircuts)
- margin shock scenario (reduced leverage cap)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.execution.costs import CostModel, TransactionCostBreakdown
from src.execution.fills import FillSimulator, ParticipationCap


@dataclass
class StressTestResult:
    """Result of a single stress test scenario."""

    scenario_name: str
    description: str
    base_sharpe: float
    stressed_sharpe: float
    base_return: float
    stressed_return: float
    base_max_dd: float
    stressed_max_dd: float
    degradation_pct: float  # (base - stressed) / base * 100
    passed: bool  # True if strategy survives (positive Sharpe or acceptable DD)
    details: dict = field(default_factory=dict)


class StressTestSuite:
    """Run all mandated stress tests on strategy returns.

    Per spec: these tests are non-negotiable.
    """

    def __init__(
        self,
        slippage_multipliers: list[float] | None = None,
        max_acceptable_dd: float = 0.30,
        min_stressed_sharpe: float = -0.5,
    ):
        self.slippage_multipliers = slippage_multipliers or [1.0, 2.0, 3.0]
        self.max_acceptable_dd = max_acceptable_dd
        self.min_stressed_sharpe = min_stressed_sharpe

    def run_all(
        self,
        base_returns: pd.Series,
        weight_changes: pd.DataFrame,
        prices: pd.DataFrame,
        cost_per_unit: float,
        equity: float = 1_000_000,
    ) -> list[StressTestResult]:
        """Run full stress test suite.

        Parameters
        ----------
        base_returns : pd.Series
            Strategy returns before additional stress costs.
        weight_changes : pd.DataFrame
            Weight changes (for sizing cost stress).
        prices : pd.DataFrame
            Price data.
        cost_per_unit : float
            Baseline cost per unit of turnover.
        equity : float
            Account equity.

        Returns
        -------
        list[StressTestResult]
        """
        results = []

        base_sharpe = self._sharpe(base_returns)
        base_ret = self._annualized_return(base_returns)
        base_dd = self._max_drawdown(base_returns)

        # 1. Slippage multiplier tests
        for mult in self.slippage_multipliers:
            if mult == 1.0:
                continue
            extra_cost = self._compute_extra_cost(
                weight_changes, cost_per_unit * (mult - 1)
            )
            stressed = base_returns - extra_cost
            results.append(
                self._build_result(
                    f"slippage_x{mult}",
                    f"Slippage costs multiplied by {mult}x",
                    base_sharpe,
                    base_ret,
                    base_dd,
                    stressed,
                )
            )

        # 2. Adverse tick on all fills
        tick_cost = self._compute_adverse_tick_cost(weight_changes, prices, cost_per_unit)
        stressed = base_returns - tick_cost
        results.append(
            self._build_result(
                "adverse_tick",
                "1 tick adverse on all fills",
                base_sharpe,
                base_ret,
                base_dd,
                stressed,
            )
        )

        # 3. Delayed fills (next bar)
        delayed_returns = self._simulate_delayed_fills(base_returns)
        results.append(
            self._build_result(
                "delayed_fills",
                "All fills delayed by 1 bar",
                base_sharpe,
                base_ret,
                base_dd,
                delayed_returns,
            )
        )

        # 4. Random trade drop (10%)
        dropped = self._simulate_random_drops(base_returns, drop_rate=0.10)
        results.append(
            self._build_result(
                "random_drop_10pct",
                "10% of signals randomly missed",
                base_sharpe,
                base_ret,
                base_dd,
                dropped,
            )
        )

        # 5. Correlation spike scenario
        corr_stressed = base_returns * 0.60  # risk haircut to 60%
        results.append(
            self._build_result(
                "correlation_spike",
                "Correlation spike: risk scaled to 60%",
                base_sharpe,
                base_ret,
                base_dd,
                corr_stressed,
            )
        )

        # 6. Margin shock scenario
        margin_stressed = base_returns * 0.50  # leverage cap halved
        results.append(
            self._build_result(
                "margin_shock",
                "Margin shock: leverage cap reduced by 50%",
                base_sharpe,
                base_ret,
                base_dd,
                margin_stressed,
            )
        )

        return results

    def _build_result(
        self,
        name: str,
        description: str,
        base_sharpe: float,
        base_ret: float,
        base_dd: float,
        stressed_returns: pd.Series,
    ) -> StressTestResult:
        stressed_sharpe = self._sharpe(stressed_returns)
        stressed_ret = self._annualized_return(stressed_returns)
        stressed_dd = self._max_drawdown(stressed_returns)

        degradation = (
            (base_sharpe - stressed_sharpe) / abs(base_sharpe) * 100
            if abs(base_sharpe) > 0.01
            else 0.0
        )

        passed = (
            stressed_sharpe >= self.min_stressed_sharpe
            and stressed_dd <= self.max_acceptable_dd
        )

        return StressTestResult(
            scenario_name=name,
            description=description,
            base_sharpe=base_sharpe,
            stressed_sharpe=stressed_sharpe,
            base_return=base_ret,
            stressed_return=stressed_ret,
            base_max_dd=base_dd,
            stressed_max_dd=stressed_dd,
            degradation_pct=degradation,
            passed=passed,
        )

    def _compute_extra_cost(
        self,
        weight_changes: pd.DataFrame,
        extra_cost_per_unit: float,
    ) -> pd.Series:
        """Compute additional cost from slippage multiplier."""
        turnover = weight_changes.abs().sum(axis=1)
        return turnover * extra_cost_per_unit

    def _compute_adverse_tick_cost(
        self,
        weight_changes: pd.DataFrame,
        prices: pd.DataFrame,
        base_cost: float,
    ) -> pd.Series:
        """Simulate 1 tick adverse on every fill."""
        turnover = weight_changes.abs().sum(axis=1)
        # Approximate: 1 tick = ~0.5-1 bps for typical futures
        tick_cost = turnover * base_cost * 0.5
        return tick_cost

    def _simulate_delayed_fills(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Simulate delayed execution by shifting returns."""
        # Delayed fill means signal from bar t is executed on bar t+1
        # Return attribution shifts by 1
        return returns.shift(1).fillna(0.0)

    def _simulate_random_drops(
        self,
        returns: pd.Series,
        drop_rate: float = 0.10,
        seed: int = 42,
    ) -> pd.Series:
        """Randomly drop a fraction of signals (simulating missed trades)."""
        rng = np.random.RandomState(seed)
        mask = rng.random(len(returns)) > drop_rate
        return returns * mask

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @staticmethod
    def _annualized_return(returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        return float(returns.mean() * 252)

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0
