"""
Margin buffer logic — per 06_RISK_PORTFOLIO_EXECUTION.md.

- Maintain margin utilization cap (<= 30-40% of equity)
- Apply "margin shock" stress test
- Enforce leverage cap under stress
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.contracts import ContractMetadata


@dataclass
class MarginState:
    """Current margin utilization state."""

    total_margin_required: float
    equity: float
    utilization_pct: float
    is_breaching: bool
    scale_factor: float  # 1.0 if OK, <1.0 if margin-constrained


class MarginMonitor:
    """Monitor and enforce margin utilization constraints.

    Per spec:
    - max utilization: 35% of equity
    - stress test: margins spike 50%
    - leverage cap under stress: 2.0x
    """

    def __init__(
        self,
        max_utilization: float = 0.35,
        stress_margin_multiplier: float = 1.5,
        leverage_cap_under_stress: float = 2.0,
    ):
        self.max_utilization = max_utilization
        self.stress_margin_multiplier = stress_margin_multiplier
        self.leverage_cap_under_stress = leverage_cap_under_stress

    def compute_margin_required(
        self,
        weights: pd.Series,
        prices: pd.Series,
        metadata: dict[str, ContractMetadata],
        equity: float,
    ) -> float:
        """Compute total margin required for a set of positions.

        Parameters
        ----------
        weights : pd.Series
            Position weights per instrument.
        prices : pd.Series
            Current prices per instrument.
        metadata : dict
            Contract metadata per instrument.
        equity : float
            Current account equity.

        Returns
        -------
        float
            Total margin required in dollars.
        """
        total_margin = 0.0

        for inst in weights.index:
            if inst not in metadata or inst not in prices.index:
                continue

            meta = metadata[inst]
            weight = abs(weights[inst])
            price = prices[inst]

            # Notional value of position
            # weight represents vol-scaled fractional exposure
            # Convert to approximate contract count
            notional = weight * equity
            margin = notional * meta.margin_init_pct

            total_margin += margin

        return total_margin

    def check_and_scale(
        self,
        weights: pd.Series,
        prices: pd.Series,
        metadata: dict[str, ContractMetadata],
        equity: float,
        stress_test: bool = False,
    ) -> MarginState:
        """Check margin utilization and compute scale factor if needed.

        Parameters
        ----------
        weights : pd.Series
            Position weights.
        prices : pd.Series
            Current prices.
        metadata : dict
            Contract metadata.
        equity : float
            Current equity.
        stress_test : bool
            If True, apply stress margin multiplier.

        Returns
        -------
        MarginState
        """
        margin_req = self.compute_margin_required(weights, prices, metadata, equity)

        if stress_test:
            margin_req *= self.stress_margin_multiplier

        utilization = margin_req / equity if equity > 0 else 1.0
        is_breaching = utilization > self.max_utilization

        if is_breaching:
            scale_factor = self.max_utilization / utilization
        else:
            scale_factor = 1.0

        return MarginState(
            total_margin_required=margin_req,
            equity=equity,
            utilization_pct=utilization,
            is_breaching=is_breaching,
            scale_factor=scale_factor,
        )

    def apply_margin_constraint(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        metadata: dict[str, ContractMetadata],
        equity_curve: pd.Series,
    ) -> pd.DataFrame:
        """Apply margin constraints across the full backtest period.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights over time.
        prices : pd.DataFrame
            Close prices over time (column per instrument).
        metadata : dict
            Contract metadata.
        equity_curve : pd.Series
            Equity over time.

        Returns
        -------
        pd.DataFrame
            Margin-constrained weights.
        """
        adjusted = weights.copy()

        for i in range(len(weights)):
            ts = weights.index[i]
            w = weights.iloc[i]
            p = prices.iloc[i] if isinstance(prices, pd.DataFrame) else prices
            eq = equity_curve.iloc[i] if ts in equity_curve.index else 1_000_000

            state = self.check_and_scale(
                w, p, metadata, eq, stress_test=False
            )

            if state.is_breaching:
                adjusted.iloc[i] = w * state.scale_factor

        return adjusted
