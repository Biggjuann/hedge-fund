"""
Fill simulation and participation caps — per 06_RISK_PORTFOLIO_EXECUTION.md.

Participation caps:
- Cap order size as % of ADV (proxy)
- Widen execution schedules in high vol / low liquidity regimes

For backtest realism: simulate partial fills and execution delays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.contracts import ContractMetadata


@dataclass
class FillResult:
    """Result of fill simulation for a single order."""

    requested_contracts: float
    filled_contracts: float
    fill_price: float
    slippage_ticks: float
    is_partial: bool
    execution_delay_bars: int


class ParticipationCap:
    """Enforce participation rate caps on order sizes.

    Per spec: max 1% of ADV normally, 0.5% in high vol.
    """

    def __init__(
        self,
        max_adv_pct: float = 0.01,
        high_vol_adv_pct: float = 0.005,
        vol_regime_threshold_pctile: float = 0.75,
    ):
        self.max_adv_pct = max_adv_pct
        self.high_vol_adv_pct = high_vol_adv_pct
        self.vol_regime_threshold_pctile = vol_regime_threshold_pctile

    def cap_order_size(
        self,
        desired_contracts: float,
        metadata: ContractMetadata,
        is_high_vol: bool = False,
    ) -> float:
        """Cap order size based on ADV participation limits.

        Parameters
        ----------
        desired_contracts : float
            Desired number of contracts.
        metadata : ContractMetadata
            Instrument metadata with avg_daily_volume.
        is_high_vol : bool
            Whether current regime is high vol.

        Returns
        -------
        float
            Capped number of contracts.
        """
        pct = self.high_vol_adv_pct if is_high_vol else self.max_adv_pct
        max_contracts = metadata.avg_daily_volume * pct

        return min(abs(desired_contracts), max_contracts) * np.sign(desired_contracts)


class FillSimulator:
    """Simulate realistic order fills for backtesting.

    Supports:
    - Participation caps
    - Slippage based on size and liquidity
    - Delayed fills (next bar)
    - Partial fills
    """

    def __init__(
        self,
        participation_cap: ParticipationCap | None = None,
        adverse_tick: bool = False,
        delay_bars: int = 0,
        random_drop_rate: float = 0.0,
        random_seed: int | None = None,
    ):
        self.participation_cap = participation_cap or ParticipationCap()
        self.adverse_tick = adverse_tick
        self.delay_bars = delay_bars
        self.random_drop_rate = random_drop_rate
        self.rng = np.random.RandomState(random_seed)

    def simulate_fill(
        self,
        desired_contracts: float,
        price: float,
        metadata: ContractMetadata,
        is_high_vol: bool = False,
    ) -> FillResult:
        """Simulate a single order fill.

        Parameters
        ----------
        desired_contracts : float
            Desired trade size.
        price : float
            Reference price (close of signal bar).
        metadata : ContractMetadata
            Instrument metadata.
        is_high_vol : bool
            Current vol regime.

        Returns
        -------
        FillResult
        """
        # Random trade drop simulation
        if self.random_drop_rate > 0 and self.rng.random() < self.random_drop_rate:
            return FillResult(
                requested_contracts=desired_contracts,
                filled_contracts=0.0,
                fill_price=price,
                slippage_ticks=0.0,
                is_partial=False,
                execution_delay_bars=0,
            )

        # Apply participation cap
        capped = self.participation_cap.cap_order_size(
            desired_contracts, metadata, is_high_vol
        )

        is_partial = abs(capped) < abs(desired_contracts)

        # Compute slippage
        slippage_ticks = 0.0
        if self.adverse_tick:
            slippage_ticks = 1.0

        # Direction of slippage
        if capped > 0:  # buying: price goes up
            fill_price = price + slippage_ticks * metadata.tick_size
        else:  # selling: price goes down
            fill_price = price - slippage_ticks * metadata.tick_size

        return FillResult(
            requested_contracts=desired_contracts,
            filled_contracts=capped,
            fill_price=fill_price,
            slippage_ticks=slippage_ticks,
            is_partial=is_partial,
            execution_delay_bars=self.delay_bars,
        )

    def apply_to_weight_changes(
        self,
        weight_changes: pd.DataFrame,
        prices: pd.DataFrame,
        metadata: dict[str, ContractMetadata],
        equity: float,
        vol_regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply fill simulation to a full series of weight changes.

        Handles delayed fills by shifting weight changes forward.

        Parameters
        ----------
        weight_changes : pd.DataFrame
            Desired weight changes.
        prices : pd.DataFrame
            Price data.
        metadata : dict
            Contract metadata.
        equity : float
            Account equity.
        vol_regime : pd.Series, optional
            Vol regime labels.

        Returns
        -------
        pd.DataFrame
            Actual weight changes after fill simulation.
        """
        actual = weight_changes.copy()

        if self.delay_bars > 0:
            # Shift weight changes forward to simulate delayed execution
            actual = actual.shift(self.delay_bars).fillna(0.0)

        for i in range(len(actual)):
            ts = actual.index[i]
            is_high = False
            if vol_regime is not None and ts in vol_regime.index:
                is_high = vol_regime.loc[ts] == "high"

            for inst in actual.columns:
                dw = actual.iloc[i][inst]
                if abs(dw) < 1e-10:
                    continue

                if inst not in metadata:
                    continue

                meta = metadata[inst]
                price_val = (
                    prices.iloc[i][inst]
                    if inst in prices.columns
                    else prices.iloc[i]
                )

                # Convert weight change to contracts
                desired = dw * equity / (price_val * meta.multiplier) if price_val > 0 else 0

                fill = self.simulate_fill(desired, price_val, meta, is_high)

                # Convert back to weight
                if fill.filled_contracts != 0 and desired != 0:
                    fill_ratio = fill.filled_contracts / desired
                else:
                    fill_ratio = 0.0

                actual.iloc[i][inst] = dw * fill_ratio

        return actual
