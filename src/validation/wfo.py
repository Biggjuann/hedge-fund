"""
Walk-Forward Optimization — per 05_VALIDATION_PROTOCOL.md.

Rolling design:
- Train: 2-5 years
- Test: 3-6 months
- Step: 1-3 months

Parameter selection: stable regions (top-k, cluster center, penalize turnover).
Purge gap between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class WFOFold:
    """A single walk-forward fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_start: pd.Timestamp  # start of purge gap
    purge_end: pd.Timestamp  # end of purge gap
    best_params: dict = field(default_factory=dict)
    train_metric: float = 0.0
    test_metric: float = 0.0
    test_returns: pd.Series | None = None
    all_param_results: list[dict] = field(default_factory=list)


class WalkForwardOptimizer:
    """Rolling walk-forward optimization engine.

    Per spec:
    - Rolling windows
    - Purge gap between train and test
    - Parameter selection from stable regions
    - Tracks number of trials for multiple testing diagnostics
    """

    def __init__(
        self,
        train_days: int = 756,
        test_days: int = 126,
        step_days: int = 63,
        purge_days: int = 5,
        embargo_days: int = 5,
        min_train_days: int = 504,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self._total_trials = 0

    def generate_folds(
        self,
        data_index: pd.DatetimeIndex,
    ) -> list[WFOFold]:
        """Generate walk-forward fold boundaries.

        Parameters
        ----------
        data_index : pd.DatetimeIndex
            Full date index of available data.

        Returns
        -------
        list[WFOFold]
            Ordered list of folds.
        """
        if len(data_index) == 0:
            return []

        # Convert to business day counts
        dates = data_index.sort_values()
        n = len(dates)

        folds = []
        fold_id = 0
        train_start_idx = 0

        while True:
            train_end_idx = train_start_idx + self.train_days - 1
            purge_start_idx = train_end_idx + 1
            purge_end_idx = purge_start_idx + self.purge_days - 1
            test_start_idx = purge_end_idx + 1
            test_end_idx = test_start_idx + self.test_days - 1

            # Check bounds
            if test_end_idx >= n:
                break

            # Check minimum train size
            actual_train = train_end_idx - train_start_idx + 1
            if actual_train < self.min_train_days:
                train_start_idx += self.step_days
                continue

            fold = WFOFold(
                fold_id=fold_id,
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx],
                purge_start=dates[purge_start_idx],
                purge_end=dates[purge_end_idx],
            )
            folds.append(fold)
            fold_id += 1

            # Step forward
            train_start_idx += self.step_days

        return folds

    def run(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        strategy_fn: Callable,
        param_grid: list[dict[str, Any]],
        metric_fn: Callable[[pd.Series], float] | None = None,
    ) -> list[WFOFold]:
        """Run full walk-forward optimization.

        Parameters
        ----------
        data : pd.DataFrame
            Full OHLCV data.
        returns : pd.Series
            Full return series.
        strategy_fn : Callable
            Function: (data_slice, returns_slice, params) -> pd.Series of strategy returns.
        param_grid : list[dict]
            List of parameter combinations to test.
        metric_fn : Callable, optional
            Function to compute optimization metric from returns.
            Defaults to Sharpe ratio.

        Returns
        -------
        list[WFOFold]
            Completed folds with results.
        """
        if metric_fn is None:
            metric_fn = self._sharpe_ratio

        folds = self.generate_folds(data.index)

        for fold in folds:
            # Extract train data
            train_data = data.loc[fold.train_start : fold.train_end]
            train_returns = returns.loc[fold.train_start : fold.train_end]

            # Optimize on train set
            best_metric = -np.inf
            best_params = {}
            all_results = []

            for params in param_grid:
                self._total_trials += 1

                strat_returns = strategy_fn(train_data, train_returns, params)
                metric = metric_fn(strat_returns)

                all_results.append({"params": params, "metric": metric})

                if metric > best_metric:
                    best_metric = metric
                    best_params = params.copy()

            # Select from stable region (cluster center of top-k)
            best_params = self._select_stable_params(all_results)

            # Evaluate on test set (NO re-optimization)
            test_data = data.loc[fold.test_start : fold.test_end]
            test_returns_raw = returns.loc[fold.test_start : fold.test_end]

            test_strat_returns = strategy_fn(test_data, test_returns_raw, best_params)
            test_metric = metric_fn(test_strat_returns)

            fold.best_params = best_params
            fold.train_metric = best_metric
            fold.test_metric = test_metric
            fold.test_returns = test_strat_returns
            fold.all_param_results = all_results

        return folds

    def _select_stable_params(
        self,
        results: list[dict],
        top_k: int = 10,
    ) -> dict:
        """Select parameters from stable region.

        Per spec: evaluate top-k, pick cluster center/median.
        """
        if not results:
            return {}

        # Sort by metric descending
        sorted_results = sorted(results, key=lambda x: x["metric"], reverse=True)

        # Take top-k
        top = sorted_results[: min(top_k, len(sorted_results))]

        if not top:
            return {}

        # Find median parameter set (cluster center proxy)
        # For numeric parameters: take median
        # For non-numeric: take mode
        median_params = {}
        all_params = [r["params"] for r in top]

        if all_params:
            for key in all_params[0]:
                values = [p[key] for p in all_params if key in p]
                if all(isinstance(v, (int, float)) for v in values):
                    median_params[key] = float(np.median(values))
                    # Round to nearest grid value
                    grid_values = [
                        r["params"][key] for r in results if key in r["params"]
                    ]
                    if grid_values:
                        median_params[key] = min(
                            grid_values,
                            key=lambda x: abs(x - median_params[key]),
                        )
                elif all(isinstance(v, list) for v in values):
                    # For list params (e.g., lookbacks): pick from the best
                    median_params[key] = values[len(values) // 2]
                else:
                    # Mode for categorical — convert unhashable to str
                    from collections import Counter

                    str_values = [str(v) for v in values]
                    counter = Counter(str_values)
                    most_common_str = counter.most_common(1)[0][0]
                    # Map back to original value
                    for v, s in zip(values, str_values):
                        if s == most_common_str:
                            median_params[key] = v
                            break

        return median_params

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns - risk_free / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))

    @property
    def total_trials(self) -> int:
        """Total parameter combinations tested (for multiple testing)."""
        return self._total_trials

    def validate_fold_integrity(self, folds: list[WFOFold]) -> list[str]:
        """Validate that folds don't overlap and maintain purge gaps.

        Returns list of issues found (empty = OK).
        """
        issues = []

        for i, fold in enumerate(folds):
            # Check ordering
            if fold.train_start >= fold.train_end:
                issues.append(f"Fold {i}: train_start >= train_end")
            if fold.test_start >= fold.test_end:
                issues.append(f"Fold {i}: test_start >= test_end")
            if fold.train_end >= fold.test_start:
                issues.append(f"Fold {i}: train overlaps test (no purge gap)")
            if fold.purge_start > fold.purge_end:
                issues.append(f"Fold {i}: invalid purge window")

            # Check purge gap exists
            purge_gap = (fold.test_start - fold.train_end).days
            if purge_gap < self.purge_days:
                issues.append(
                    f"Fold {i}: purge gap too small ({purge_gap} < {self.purge_days})"
                )

            # Check that each fold's test doesn't overlap its OWN train
            # Note: In rolling WFO, consecutive folds' test windows MAY overlap
            # (this is expected when step_days < test_days).
            # The critical constraint is: within each fold, train and test
            # are separated by the purge gap.
            if i < len(folds) - 1:
                next_fold = folds[i + 1]
                # Next fold's train must not overlap THIS fold's test
                # in a way that would leak info (but this is OK in rolling WFO
                # since each fold is independent)
                if next_fold.train_start < fold.train_start:
                    issues.append(
                        f"Fold {i}: next fold starts before this fold"
                    )

        return issues
