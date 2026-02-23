"""
Cost model — per 01_DEEP_RESEARCH_PAPER.md + 06_RISK_PORTFOLIO_EXECUTION.md.

Decompose:
1) fees/commissions
2) bid-ask spread
3) slippage + nonlinear impact (size vs liquidity)
4) financing/margin constraints

Must be explicit. Execution uses participation caps and widens
schedules in low liquidity / high vol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.contracts import ContractMetadata


@dataclass
class TransactionCostBreakdown:
    """Decomposed transaction costs for a single trade."""

    commission: float
    spread_cost: float
    slippage: float
    impact: float
    total: float


class CostModel:
    """Realistic transaction cost model for futures.

    Per spec: explicit decomposition of all cost components.
    Impact uses square-root model: impact ~ coeff * sqrt(size/ADV).
    """

    def __init__(
        self,
        slippage_base_bps: float = 1.0,
        slippage_size_coefficient: float = 5.0,
        impact_power: float = 0.5,
    ):
        self.slippage_base_bps = slippage_base_bps
        self.slippage_size_coefficient = slippage_size_coefficient
        self.impact_power = impact_power

    def compute_trade_cost(
        self,
        trade_value: float,
        n_contracts: float,
        metadata: ContractMetadata,
        price: float,
        vol_regime_high: bool = False,
    ) -> TransactionCostBreakdown:
        """Compute full transaction cost for a trade.

        Parameters
        ----------
        trade_value : float
            Dollar value of the trade.
        n_contracts : float
            Number of contracts traded (can be fractional for sizing).
        metadata : ContractMetadata
            Instrument metadata.
        price : float
            Current price.
        vol_regime_high : bool
            If True, widen costs for high-vol regime.

        Returns
        -------
        TransactionCostBreakdown
        """
        abs_contracts = abs(n_contracts)
        abs_value = abs(trade_value)

        # 1. Commission
        commission = abs_contracts * metadata.commission_per_contract

        # 2. Spread cost: half-spread per side
        spread_points = metadata.avg_spread_ticks * metadata.tick_size
        spread_cost = abs_contracts * spread_points * metadata.multiplier / 2.0

        # Widen in high vol
        if vol_regime_high:
            spread_cost *= 1.5

        # 3. Slippage: base + size-dependent
        adv_value = (
            metadata.avg_daily_volume * price * metadata.multiplier
        )
        participation_rate = abs_value / adv_value if adv_value > 0 else 0.0

        slippage_bps = self.slippage_base_bps + (
            self.slippage_size_coefficient * participation_rate * 100
        )

        # Widen in high vol
        if vol_regime_high:
            slippage_bps *= 1.5

        slippage = abs_value * slippage_bps / 10000.0

        # 4. Market impact: square-root model
        impact = 0.0
        if participation_rate > 0.001:  # only meaningful for larger trades
            impact = (
                abs_value
                * self.slippage_size_coefficient
                * (participation_rate ** self.impact_power)
                / 10000.0
            )

        total = commission + spread_cost + slippage + impact

        return TransactionCostBreakdown(
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            impact=impact,
            total=total,
        )


def compute_transaction_costs(
    weight_changes: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: dict[str, ContractMetadata],
    equity: float | pd.Series,
    cost_model: CostModel,
    vol_regime: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute transaction costs for a full weight change series.

    Parameters
    ----------
    weight_changes : pd.DataFrame
        Change in portfolio weights (delta from previous period).
    prices : pd.DataFrame
        Close prices per instrument.
    metadata : dict
        Contract metadata.
    equity : float or pd.Series
        Account equity for sizing. If Series, uses time-varying equity.
    cost_model : CostModel
        Cost model instance.
    vol_regime : pd.Series, optional
        Vol regime labels ('low', 'med', 'high').

    Returns
    -------
    pd.DataFrame
        Cost breakdown per instrument per timestamp.
    """
    costs = pd.DataFrame(0.0, index=weight_changes.index, columns=weight_changes.columns)
    equity_is_series = isinstance(equity, pd.Series)

    for i in range(len(weight_changes)):
        ts = weight_changes.index[i]
        is_high_vol = False
        if vol_regime is not None and ts in vol_regime.index:
            is_high_vol = vol_regime.loc[ts] == "high"

        eq_i = float(equity.iloc[i]) if equity_is_series else float(equity)

        for inst in weight_changes.columns:
            dw = weight_changes.iloc[i][inst]
            if abs(dw) < 1e-10:
                continue

            if inst not in metadata:
                continue

            meta = metadata[inst]
            price = prices.iloc[i][inst] if inst in prices.columns else float(prices.iloc[i].iloc[0])

            trade_value = abs(dw) * eq_i
            n_contracts = trade_value / (price * meta.multiplier) if price > 0 else 0

            breakdown = cost_model.compute_trade_cost(
                trade_value=trade_value,
                n_contracts=n_contracts,
                metadata=meta,
                price=price,
                vol_regime_high=is_high_vol,
            )

            # Store as fraction of equity
            costs.iloc[i, costs.columns.get_loc(inst)] = breakdown.total / eq_i if eq_i > 0 else 0

    return costs
