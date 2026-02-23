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

        # Get common index
        all_indices = [w.index for w in strategy_weights.values()]
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.intersection(idx)

        # Compute inverse-vol budgets (proportional to realized vol, with floor)
        risk_budgets = self._compute_inverse_vol_budgets(
            strategy_returns, common_idx
        )

        # Collect all instrument columns
        all_instruments = set()
        for w in strategy_weights.values():
            all_instruments.update(w.columns)
        all_instruments = sorted(all_instruments)

        # Initialize combined weights
        combined = pd.DataFrame(
            0.0, index=common_idx, columns=all_instruments
        )

        # Weight strategies by inverse-vol budget — active strategies get more budget
        for strat_name, weights in strategy_weights.items():
            budget = risk_budgets.get(strat_name, 1.0 / n_strategies)
            for col in weights.columns:
                if col in combined.columns:
                    w_aligned = weights[col].reindex(common_idx).fillna(0.0)
                    combined[col] += w_aligned * budget

        # Apply portfolio-level vol target ONCE at the end
        combined = self._apply_portfolio_vol_target(
            combined, common_idx, strategy_returns, risk_budgets
        )

        # Apply leverage cap
        gross_leverage = combined.abs().sum(axis=1)
        excess = gross_leverage > self.max_leverage
        if excess.any():
            scale_down = self.max_leverage / gross_leverage[excess]
            combined.loc[excess] = combined.loc[excess].multiply(
                scale_down, axis=0
            )

        return combined

    def _compute_inverse_vol_budgets(
        self,
        strategy_returns: dict[str, pd.Series],
        index: pd.DatetimeIndex,
        floor: float = 0.05,
    ) -> dict[str, float]:
        """Compute risk budgets for strategy combination.

        Uses equal weight (1/n) since strategies already vol-target their
        own positions internally. The portfolio vol target scaling in
        _apply_portfolio_vol_target handles overall risk sizing.

        Parameters
        ----------
        strategy_returns : dict
            Maps strategy name -> Series of returns.
        index : pd.DatetimeIndex
            Common index for alignment.
        floor : float
            Minimum budget per strategy.

        Returns
        -------
        dict mapping strategy name -> risk budget (sums to 1.0)
        """
        n = len(strategy_returns)
        if n == 0:
            return {}

        # Equal weight: each strategy manages its own vol targeting
        budgets = {name: 1.0 / n for name in strategy_returns}

        return budgets

    def _apply_portfolio_vol_target(
        self,
        weights: pd.DataFrame,
        index: pd.DatetimeIndex,
        strategy_returns: dict[str, pd.Series] | None = None,
        risk_budgets: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Scale combined weights to hit portfolio vol target.

        Uses realized portfolio vol from budget-weighted combined strategy returns,
        matching the actual allocation.
        """
        if strategy_returns:
            n_strategies = len(strategy_returns)
            # Estimate realized portfolio vol using actual risk budgets
            combined_ret = pd.Series(0.0, index=index)
            for strat_name, strat_ret in strategy_returns.items():
                if risk_budgets:
                    budget = risk_budgets.get(strat_name, 1.0 / n_strategies)
                else:
                    budget = 1.0 / n_strategies
                aligned = strat_ret.reindex(index).fillna(0.0)
                combined_ret = combined_ret + aligned * budget
            port_vol = ewma_volatility(combined_ret, com=self.vol_lookback)
            port_vol_safe = port_vol.clip(lower=0.01)
            scale = (self.portfolio_vol_target / port_vol_safe).clip(
                upper=self.max_leverage
            )
            return weights.multiply(scale, axis=0)

        return weights


def apply_vol_management(
    weights: pd.DataFrame,
    returns: pd.Series,
    vol_target: float = 0.22,
    vol_floor_pct: float = 0.50,
    vol_cap_multiplier: float = 2.0,
    ewma_com: int = 60,
) -> pd.DataFrame:
    """Scale entire portfolio to target a specific realized volatility.

    Uses expanding EWMA blend (70% EWMA + 30% expanding) for robustness.
    vol_scale = (vol_target / realized_vol).clip(upper=vol_cap_multiplier)
    with vol floored at vol_floor_pct * vol_target.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (instruments as columns).
    returns : pd.Series
        Portfolio returns for vol estimation.
    vol_target : float
        Annualized vol target.
    vol_floor_pct : float
        Floor realized vol at this fraction of vol_target.
    vol_cap_multiplier : float
        Max upward scaling (caps leverage increase).
    ewma_com : int
        EWMA center-of-mass for vol estimation.

    Returns
    -------
    pd.DataFrame
        Scaled portfolio weights.
    """
    # EWMA vol (annualized)
    ewma_vol = ewma_volatility(returns, com=ewma_com)
    # Expanding vol (annualized), shifted for causality (match EWMA shift)
    expanding_vol = returns.expanding(min_periods=60).std() * np.sqrt(252)
    expanding_vol = expanding_vol.shift(1)

    # Blend: 70% EWMA + 30% expanding for robustness
    blended_vol = 0.70 * ewma_vol + 0.30 * expanding_vol.reindex(ewma_vol.index).fillna(ewma_vol)

    # Floor vol at fraction of target
    vol_floor = vol_target * vol_floor_pct
    safe_vol = blended_vol.clip(lower=vol_floor)

    # Scale factor
    vol_scale = (vol_target / safe_vol).clip(upper=vol_cap_multiplier)

    # Align and apply
    vol_scale = vol_scale.reindex(weights.index).fillna(1.0)
    return weights.multiply(vol_scale, axis=0)


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
