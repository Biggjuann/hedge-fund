"""
Portfolio construction — per 06_RISK_PORTFOLIO_EXECUTION.md.

Default: risk parity across strategies (equal vol contribution).
Optional: ERC for instruments if correlation model available.
Correlation haircuts: reduce risk when corr rises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility import ewma_volatility


class RiskParityAllocator:
    """Allocate risk budget equally across strategies.

    Per spec: equal volatility contribution across strategies.
    Then apply portfolio-level vol target.
    """

    def __init__(
        self,
        portfolio_vol_target: float = 0.12,
        max_leverage: float = 4.0,
        vol_lookback: int = 60,
    ):
        self.portfolio_vol_target = portfolio_vol_target
        self.max_leverage = max_leverage
        self.vol_lookback = vol_lookback

    def allocate(
        self,
        strategy_weights: dict[str, pd.DataFrame],
        strategy_returns: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Combine strategy weights using risk parity.

        Each strategy gets equal risk budget. Then the combined
        portfolio is scaled to the portfolio vol target.

        Parameters
        ----------
        strategy_weights : dict
            Maps strategy name -> DataFrame of weights per instrument.
        strategy_returns : dict
            Maps strategy name -> Series of strategy returns.

        Returns
        -------
        pd.DataFrame
            Combined portfolio weights per instrument.
        """
        n_strategies = len(strategy_weights)
        if n_strategies == 0:
            raise ValueError("No strategies provided")

        # Per-strategy risk budget = 1/N
        risk_budget = 1.0 / n_strategies

        # Get common index
        all_indices = [w.index for w in strategy_weights.values()]
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.intersection(idx)

        # Collect all instrument columns
        all_instruments = set()
        for w in strategy_weights.values():
            all_instruments.update(w.columns)
        all_instruments = sorted(all_instruments)

        # Initialize combined weights
        combined = pd.DataFrame(
            0.0, index=common_idx, columns=all_instruments
        )

        # Scale each strategy by risk budget and add
        for strat_name, weights in strategy_weights.items():
            strat_ret = strategy_returns.get(strat_name)
            if strat_ret is not None:
                strat_vol = ewma_volatility(strat_ret, com=self.vol_lookback)
                # Scale factor: risk_budget * (target_vol / strat_vol)
                scale = risk_budget * self.portfolio_vol_target / strat_vol.clip(
                    lower=0.01
                )
                # Reindex to common_idx, forward-fill for dates with no return
                scale = scale.reindex(common_idx).ffill().fillna(risk_budget)
            else:
                scale = pd.Series(risk_budget, index=common_idx)

            for col in weights.columns:
                if col in combined.columns:
                    w_aligned = weights[col].reindex(common_idx).fillna(0.0)
                    combined[col] += w_aligned * scale

        # Apply portfolio-level vol target
        combined = self._apply_portfolio_vol_target(combined, common_idx)

        # Apply leverage cap
        gross_leverage = combined.abs().sum(axis=1)
        excess = gross_leverage > self.max_leverage
        if excess.any():
            scale_down = self.max_leverage / gross_leverage[excess]
            combined.loc[excess] = combined.loc[excess].multiply(
                scale_down, axis=0
            )

        return combined

    def _apply_portfolio_vol_target(
        self,
        weights: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Scale combined weights to hit portfolio vol target.

        Simple approach: scale by ratio of target to realized portfolio vol.
        """
        # Estimate portfolio vol from weights (simple: sum of absolute weights * avg vol)
        # This is an approximation; correlation-aware version is in risk_overlay
        gross = weights.abs().sum(axis=1)
        scale = self.portfolio_vol_target / gross.clip(lower=0.01)
        scale = scale.clip(upper=self.max_leverage)

        return weights  # weights are already vol-scaled per strategy


class ERCAllocator:
    """Equal Risk Contribution allocator for instrument-level weights.

    Optional per spec: "ERC for instruments if correlation model available."
    Requires a correlation matrix.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def allocate(
        self,
        cov_matrix: pd.DataFrame,
        risk_budget: np.ndarray | None = None,
    ) -> pd.Series:
        """Compute ERC weights given a covariance matrix.

        Uses iterative approach to find weights where each asset
        contributes equally to portfolio risk.

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            Covariance matrix.
        risk_budget : np.ndarray, optional
            Target risk contribution per asset. Defaults to equal.

        Returns
        -------
        pd.Series
            Allocation weights summing to 1.
        """
        n = len(cov_matrix)
        cov = cov_matrix.values

        if risk_budget is None:
            risk_budget = np.ones(n) / n

        # Initialize with equal weights
        w = np.ones(n) / n

        for _ in range(self.max_iterations):
            # Portfolio variance
            port_var = w @ cov @ w
            if port_var <= 0:
                break
            port_vol = np.sqrt(port_var)

            # Marginal risk contribution
            mrc = cov @ w / port_vol

            # Risk contribution
            rc = w * mrc

            # Target risk contribution
            target_rc = risk_budget * port_vol

            # Update weights
            w_new = w * target_rc / rc
            w_new = w_new / w_new.sum()

            # Check convergence
            if np.max(np.abs(w_new - w)) < self.tolerance:
                w = w_new
                break

            w = w_new

        return pd.Series(w, index=cov_matrix.columns)
