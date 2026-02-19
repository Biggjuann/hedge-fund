"""
Regime-conditioned OOS reporting — per 05_VALIDATION_PROTOCOL.md.

Report metrics broken down by:
- vol regime: low/med/high (quantiles)
- corr regime: low/high (quantiles of avg pairwise corr)
- trend regime: ADX proxy bucket
- macro tags (evaluation only if available)

OOS-by-regime is MANDATORY per spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.regime import RegimeLabels, RegimeTagger
from src.reporting.metrics import PerformanceMetrics


@dataclass
class RegimeBreakdown:
    """Performance metrics for a single regime bucket."""

    regime_dimension: str  # 'vol', 'corr', 'trend'
    regime_label: str  # 'low', 'med', 'high', etc.
    n_observations: int
    metrics: dict  # Sharpe, return, vol, drawdown, etc.


@dataclass
class RegimeOOSReport:
    """Full regime-conditioned OOS report."""

    fold_id: int | None
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    overall_metrics: dict
    regime_breakdowns: list[RegimeBreakdown]
    warnings: list[str] = field(default_factory=list)


class RegimeConditionedOOS:
    """Generate regime-conditioned out-of-sample reports.

    Per spec: performance must be reported conditional on
    vol/corr/macro regime tags. This is mandatory, not optional.
    """

    def __init__(
        self,
        regime_tagger: RegimeTagger | None = None,
        min_observations_per_bucket: int = 30,
    ):
        self.regime_tagger = regime_tagger or RegimeTagger()
        self.min_obs = min_observations_per_bucket
        self.perf = PerformanceMetrics()

    def generate_report(
        self,
        returns: pd.Series,
        prices: pd.DataFrame,
        regime_labels: RegimeLabels | None = None,
        avg_corr: pd.Series | None = None,
        fold_id: int | None = None,
    ) -> RegimeOOSReport:
        """Generate full regime-conditioned OOS report.

        Parameters
        ----------
        returns : pd.Series
            OOS strategy returns.
        prices : pd.DataFrame
            OHLCV price data for computing regime tags.
        regime_labels : RegimeLabels, optional
            Pre-computed regime labels. If None, computed from data.
        avg_corr : pd.Series, optional
            Average pairwise correlation.
        fold_id : int, optional
            WFO fold identifier.

        Returns
        -------
        RegimeOOSReport
        """
        warnings = []

        # Compute regime labels if not provided
        if regime_labels is None:
            regime_labels = self.regime_tagger.tag_all(
                prices, returns, avg_corr
            )

        # Overall metrics
        overall = self.perf.compute_all(returns)

        # Breakdowns per dimension
        breakdowns = []

        # Vol regime breakdown
        breakdowns.extend(
            self._breakdown_by_regime(
                returns, regime_labels.vol_regime, "vol", warnings
            )
        )

        # Corr regime breakdown
        breakdowns.extend(
            self._breakdown_by_regime(
                returns, regime_labels.corr_regime, "corr", warnings
            )
        )

        # Trend regime breakdown
        breakdowns.extend(
            self._breakdown_by_regime(
                returns, regime_labels.trend_regime, "trend", warnings
            )
        )

        return RegimeOOSReport(
            fold_id=fold_id,
            period_start=returns.index[0] if len(returns) > 0 else None,
            period_end=returns.index[-1] if len(returns) > 0 else None,
            overall_metrics=overall,
            regime_breakdowns=breakdowns,
            warnings=warnings,
        )

    def _breakdown_by_regime(
        self,
        returns: pd.Series,
        regime_series: pd.Series,
        dimension_name: str,
        warnings: list[str],
    ) -> list[RegimeBreakdown]:
        """Compute metrics for each regime bucket."""
        breakdowns = []

        # Align indices
        common = returns.index.intersection(regime_series.index)
        ret = returns.loc[common]
        reg = regime_series.loc[common]

        for label in reg.dropna().unique():
            mask = reg == label
            bucket_returns = ret[mask]

            n_obs = len(bucket_returns)
            if n_obs < self.min_obs:
                warnings.append(
                    f"{dimension_name}/{label}: only {n_obs} observations "
                    f"(min {self.min_obs})"
                )

            if n_obs > 0:
                metrics = self.perf.compute_all(bucket_returns)
            else:
                metrics = {}

            breakdowns.append(
                RegimeBreakdown(
                    regime_dimension=dimension_name,
                    regime_label=str(label),
                    n_observations=n_obs,
                    metrics=metrics,
                )
            )

        return breakdowns

    def aggregate_fold_reports(
        self,
        fold_reports: list[RegimeOOSReport],
    ) -> dict:
        """Aggregate regime breakdowns across all WFO folds.

        Returns summary statistics per regime bucket.
        """
        agg = {}

        for report in fold_reports:
            for bd in report.regime_breakdowns:
                key = f"{bd.regime_dimension}/{bd.regime_label}"
                if key not in agg:
                    agg[key] = {
                        "sharpe_values": [],
                        "return_values": [],
                        "n_total": 0,
                    }
                if "sharpe_ratio" in bd.metrics:
                    agg[key]["sharpe_values"].append(bd.metrics["sharpe_ratio"])
                if "annualized_return" in bd.metrics:
                    agg[key]["return_values"].append(bd.metrics["annualized_return"])
                agg[key]["n_total"] += bd.n_observations

        # Compute summary stats
        summary = {}
        for key, vals in agg.items():
            sharpes = vals["sharpe_values"]
            returns = vals["return_values"]
            summary[key] = {
                "mean_sharpe": np.mean(sharpes) if sharpes else None,
                "std_sharpe": np.std(sharpes) if len(sharpes) > 1 else None,
                "mean_return": np.mean(returns) if returns else None,
                "n_total_obs": vals["n_total"],
                "n_folds": len(sharpes),
            }

        return summary
