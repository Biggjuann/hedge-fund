"""
Backtest engine — NautilusTrader integration.

Wraps NautilusTrader's BacktestEngine to run our strategies with
realistic execution simulation. Also provides a lightweight vectorized
engine for fast WFO parameter sweeps.

Per spec: prioritize correctness, leakage prevention, and cost realism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import ContractMetadata
from src.execution.costs import CostModel, compute_transaction_costs
from src.features.volatility import ewma_volatility
from src.reporting.metrics import PerformanceMetrics


@dataclass
class BacktestResult:
    """Container for backtest results."""

    strategy_name: str
    returns: pd.Series  # daily strategy returns after costs
    gross_returns: pd.Series  # returns before costs
    costs: pd.Series  # total costs per day
    weights: pd.DataFrame  # position weights over time
    equity_curve: pd.Series  # cumulative equity
    metrics: dict  # performance metrics
    metadata: dict = field(default_factory=dict)


class VectorizedBacktestEngine:
    """Lightweight vectorized backtest engine for fast WFO sweeps.

    This engine computes strategy returns from pre-computed signals
    and weights without full event-driven simulation. Used for
    in-sample parameter optimization where speed matters.

    For final OOS evaluation, use NautilusBacktestEngine for full
    execution realism.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        initial_equity: float = 1_000_000,
    ):
        self.cost_model = cost_model or CostModel()
        self.initial_equity = initial_equity
        self.perf = PerformanceMetrics()

    def run(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame | pd.Series,
        metadata: dict[str, ContractMetadata] | None = None,
        prices: pd.DataFrame | None = None,
        vol_regime: pd.Series | None = None,
    ) -> BacktestResult:
        """Run vectorized backtest.

        Strategy return = sum(weight_i * return_i) - costs

        CAUSALITY: weights at time t are applied to returns at time t.
        Since weights are computed from data up to t-1 (enforced in
        strategy code), this is correct.

        Parameters
        ----------
        weights : pd.DataFrame
            Position weights per instrument (columns).
        returns : pd.DataFrame or pd.Series
            Returns per instrument.
        metadata : dict, optional
            Contract metadata for cost computation.
        prices : pd.DataFrame, optional
            Price data for cost computation.
        vol_regime : pd.Series, optional
            Vol regime for cost widening.

        Returns
        -------
        BacktestResult
        """
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=weights.columns[0])

        # Align indices
        common_idx = weights.index.intersection(returns.index)
        w = weights.loc[common_idx]
        r = returns.loc[common_idx]

        # Gross returns: weight * return for each instrument, summed
        common_cols = w.columns.intersection(r.columns)
        gross = (w[common_cols] * r[common_cols]).sum(axis=1)

        # Compute costs from weight changes (turnover)
        weight_changes = w.diff().fillna(0.0)
        if metadata is not None and prices is not None:
            cost_df = compute_transaction_costs(
                weight_changes,
                prices.loc[common_idx] if isinstance(prices, pd.DataFrame) else prices,
                metadata,
                self.initial_equity,
                self.cost_model,
                vol_regime,
            )
            costs = cost_df.sum(axis=1)
        else:
            # Simple cost: proportional to turnover
            turnover = weight_changes.abs().sum(axis=1)
            costs = turnover * 0.001  # 10bps default

        # Net returns
        net = gross - costs

        # Equity curve
        equity = self.initial_equity * (1 + net).cumprod()

        # Metrics
        metrics = self.perf.compute_all(net)

        return BacktestResult(
            strategy_name="vectorized",
            returns=net,
            gross_returns=gross,
            costs=costs,
            weights=w,
            equity_curve=equity,
            metrics=metrics,
        )


class NautilusBacktestEngine:
    """Full event-driven backtest engine using NautilusTrader.

    Provides realistic execution simulation including:
    - Proper order matching against bar data
    - Margin tracking
    - Position management
    - Fill models with slippage

    Used for final OOS evaluation and production validation.
    """

    def __init__(
        self,
        initial_equity: float = 1_000_000,
        venue_name: str = "SIM",
    ):
        self.initial_equity = initial_equity
        self.venue_name = venue_name
        self._engine = None
        self._is_initialized = False

    def initialize(
        self,
        instruments: dict[str, ContractMetadata],
        bar_data: dict[str, pd.DataFrame],
    ) -> None:
        """Initialize the NautilusTrader engine with instruments and data.

        Parameters
        ----------
        instruments : dict
            Maps symbol -> ContractMetadata.
        bar_data : dict
            Maps symbol -> DataFrame of OHLCV bars.
        """
        try:
            from nautilus_trader.backtest.engine import BacktestEngine
            from nautilus_trader.backtest.config import BacktestEngineConfig
            from nautilus_trader.model import (
                InstrumentId,
                Money,
                Symbol,
                TraderId,
                Venue,
            )
            from nautilus_trader.model.currencies import USD
            from nautilus_trader.model.enums import AccountType, AssetClass, OmsType
            from nautilus_trader.model.instruments import FuturesContract
            from nautilus_trader.model.objects import Price, Quantity
            from nautilus_trader.persistence.wranglers import BarDataWrangler
            from nautilus_trader.model import BarType
        except ImportError:
            raise ImportError(
                "nautilus_trader is required for NautilusBacktestEngine. "
                "Install with: pip install nautilus_trader"
            )

        # Create engine
        config = BacktestEngineConfig(
            trader_id=TraderId("REGIME-ROBUST-001")
        )
        self._engine = BacktestEngine(config=config)

        # Add venue
        venue = Venue(self.venue_name)
        self._engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self.initial_equity, USD)],
            bar_adaptive_high_low_ordering=True,
        )

        # Asset class mapping
        asset_class_map = {
            "INDEX": AssetClass.INDEX,
            "COMMODITY": AssetClass.COMMODITY,
            "DEBT": AssetClass.DEBT,
            "FX": AssetClass.FX,
        }

        # Add instruments and data
        self._nautilus_instruments = {}
        for symbol, meta in instruments.items():
            if symbol not in bar_data:
                continue

            df = bar_data[symbol]
            if df.empty:
                continue

            # Create FuturesContract
            ac = asset_class_map.get(meta.asset_class, AssetClass.INDEX)
            inst_id = InstrumentId(
                symbol=Symbol(symbol), venue=Venue(self.venue_name)
            )

            activation_ns = int(
                pd.Timestamp(df.index[0]).value
                if not hasattr(df.index[0], "value")
                else df.index[0].value
            )
            expiration_ns = int(
                pd.Timestamp(df.index[-1]).value
                if not hasattr(df.index[-1], "value")
                else df.index[-1].value
            )

            nautilus_inst = FuturesContract(
                instrument_id=inst_id,
                raw_symbol=Symbol(symbol),
                asset_class=ac,
                currency=USD,
                price_precision=meta.price_precision,
                price_increment=Price.from_str(str(meta.tick_size)),
                multiplier=Quantity.from_int(int(meta.multiplier)),
                lot_size=Quantity.from_int(1),
                underlying=symbol,
                activation_ns=activation_ns,
                expiration_ns=expiration_ns,
                ts_event=activation_ns,
                ts_init=activation_ns,
                margin_init=Decimal(str(meta.margin_init_pct)),
                margin_maint=Decimal(str(meta.margin_maint_pct)),
                maker_fee=Decimal("0"),
                taker_fee=Decimal(str(meta.commission_per_contract)),
            )

            self._engine.add_instrument(nautilus_inst)
            self._nautilus_instruments[symbol] = nautilus_inst

            # Wrangle bar data
            bar_type = BarType.from_str(
                f"{symbol}.{self.venue_name}-1-DAY-LAST-EXTERNAL"
            )
            wrangler = BarDataWrangler(
                bar_type=bar_type, instrument=nautilus_inst
            )

            # Prepare DataFrame for wrangler
            bar_df = df[["open", "high", "low", "close", "volume"]].copy()
            bars = wrangler.process(data=bar_df, ts_init_delta=0)
            self._engine.add_data(bars, sort=False)

        self._engine.sort_data()
        self._is_initialized = True

    def run_strategy(self, strategy) -> dict:
        """Run a NautilusTrader strategy and return results.

        Parameters
        ----------
        strategy : nautilus_trader.trading.strategy.Strategy
            Strategy instance to run.

        Returns
        -------
        dict
            Results including positions report and account report.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._engine.add_strategy(strategy)
        self._engine.run()

        # Extract results
        from nautilus_trader.model import Venue

        venue = Venue(self.venue_name)
        results = {
            "account_report": self._engine.trader.generate_account_report(venue),
            "order_fills_report": self._engine.trader.generate_order_fills_report(),
            "positions_report": self._engine.trader.generate_positions_report(),
        }

        return results

    def reset(self) -> None:
        """Reset engine for next run (preserves instruments and data)."""
        if self._engine is not None:
            self._engine.reset()

    def dispose(self) -> None:
        """Dispose engine and release resources."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._is_initialized = False
