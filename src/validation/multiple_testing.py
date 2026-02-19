"""
Multiple testing diagnostics — per 05_VALIDATION_PROTOCOL.md.

Per Lopez de Prado (02_BOOK_SYNTHESIS.md):
- Log number of trials per strategy
- Output DSR/PBO-style proxy diagnostics

Track and report the inflation of statistical significance
due to multiple hypothesis testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MultipleTestingReport:
    """Report on multiple testing adjustment."""

    n_trials: int
    best_sharpe_observed: float
    expected_max_sharpe_under_null: float
    deflated_sharpe_ratio: float
    haircut_pct: float  # how much the Sharpe is deflated
    pbo_proxy: float  # probability of backtest overfitting proxy
    is_likely_overfit: bool
    details: dict


class MultipleTestingDiagnostics:
    """Compute multiple testing diagnostics.

    Implements:
    - Deflated Sharpe Ratio (DSR) proxy
    - Probability of Backtest Overfitting (PBO) proxy
    - Expected maximum Sharpe under null hypothesis

    Per spec: if exact methods not implemented, output proxy diagnostics.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def expected_max_sharpe_null(
        self,
        n_trials: int,
        n_observations: int = 252,
    ) -> float:
        """Expected maximum Sharpe ratio under the null hypothesis.

        Under null (no skill), the expected max Sharpe from N independent
        trials follows approximately:

        E[max(SR)] ≈ sqrt(2 * ln(N)) - (ln(π) + ln(ln(N))) / (2 * sqrt(2 * ln(N)))

        This is the Euler correction to the Gumbel distribution.

        Parameters
        ----------
        n_trials : int
            Number of strategy/parameter combinations tested.
        n_observations : int
            Number of return observations.

        Returns
        -------
        float
            Expected maximum annualized Sharpe under null.
        """
        if n_trials <= 1:
            return 0.0

        ln_n = np.log(n_trials)
        if ln_n <= 0:
            return 0.0

        # Gumbel approximation for expected maximum of N standard normals
        e_max = np.sqrt(2 * ln_n) - (np.log(np.pi) + np.log(ln_n)) / (
            2 * np.sqrt(2 * ln_n)
        )

        return float(e_max)

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Compute Deflated Sharpe Ratio.

        Adjusts the observed Sharpe for:
        - Number of trials (multiple testing)
        - Non-normality of returns (skewness, kurtosis)

        Parameters
        ----------
        observed_sharpe : float
            Best observed annualized Sharpe ratio.
        n_trials : int
            Number of trials conducted.
        n_observations : int
            Number of return observations.
        skewness : float
            Skewness of the return distribution.
        kurtosis : float
            Kurtosis of the return distribution.

        Returns
        -------
        float
            Deflated Sharpe Ratio (probability that observed Sharpe
            exceeds the expected max under null).
        """
        e_max_sr = self.expected_max_sharpe_null(n_trials, n_observations)

        # Standard error of Sharpe ratio (Lo, 2002)
        # SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1))
        sr = observed_sharpe / np.sqrt(252)  # convert to per-period
        se_sr = np.sqrt(
            (1 - skewness * sr + (kurtosis - 1) / 4 * sr**2)
            / max(1, n_observations - 1)
        )

        if se_sr <= 0:
            return 0.0

        # DSR = P(SR > E[max(SR)]) using t-distribution
        e_max_per_period = e_max_sr / np.sqrt(252)
        t_stat = (sr - e_max_per_period) / se_sr
        dsr = float(stats.t.cdf(t_stat, df=max(1, n_observations - 1)))

        return dsr

    def pbo_proxy(
        self,
        in_sample_sharpes: list[float],
        out_sample_sharpes: list[float],
    ) -> float:
        """Compute Probability of Backtest Overfitting proxy.

        PBO ≈ fraction of times the IS-optimal strategy
        underperforms the median OOS strategy.

        This is a simplified proxy; full PBO requires CSCV.

        Parameters
        ----------
        in_sample_sharpes : list[float]
            In-sample Sharpe ratios across WFO folds.
        out_sample_sharpes : list[float]
            Out-of-sample Sharpe ratios across WFO folds.

        Returns
        -------
        float
            PBO proxy in [0, 1]. Higher = more likely overfit.
        """
        if not in_sample_sharpes or not out_sample_sharpes:
            return 0.5

        n = min(len(in_sample_sharpes), len(out_sample_sharpes))
        oos_median = np.median(out_sample_sharpes[:n])

        # Count folds where OOS performance is below median
        underperform_count = sum(
            1 for s in out_sample_sharpes[:n] if s < oos_median
        )

        # Weight by IS confidence: higher IS Sharpe but low OOS = more overfit
        is_max = max(in_sample_sharpes) if in_sample_sharpes else 0
        oos_of_best_is = out_sample_sharpes[
            in_sample_sharpes.index(is_max)
        ] if is_max in in_sample_sharpes and len(out_sample_sharpes) > in_sample_sharpes.index(is_max) else 0

        # PBO proxy: probability that the IS-best performs below OOS median
        pbo = 1.0 if oos_of_best_is < oos_median else 0.0

        # Softer version: fraction of folds underperforming
        pbo_soft = underperform_count / n if n > 0 else 0.5

        return (pbo + pbo_soft) / 2.0

    def generate_report(
        self,
        n_trials: int,
        observed_sharpe: float,
        returns: pd.Series | None = None,
        in_sample_sharpes: list[float] | None = None,
        out_sample_sharpes: list[float] | None = None,
    ) -> MultipleTestingReport:
        """Generate comprehensive multiple testing report.

        Parameters
        ----------
        n_trials : int
            Total parameter combinations tested.
        observed_sharpe : float
            Best observed Sharpe ratio.
        returns : pd.Series, optional
            Returns for computing higher moments.
        in_sample_sharpes : list, optional
            IS Sharpes across folds.
        out_sample_sharpes : list, optional
            OOS Sharpes across folds.

        Returns
        -------
        MultipleTestingReport
        """
        n_obs = len(returns) if returns is not None else 252
        skew = float(returns.skew()) if returns is not None else 0.0
        kurt = float(returns.kurtosis() + 3) if returns is not None else 3.0

        e_max = self.expected_max_sharpe_null(n_trials, n_obs)
        dsr = self.deflated_sharpe_ratio(
            observed_sharpe, n_trials, n_obs, skew, kurt
        )

        haircut = (
            (1 - dsr) * 100 if dsr < 1 else 0.0
        )

        pbo = 0.5
        if in_sample_sharpes and out_sample_sharpes:
            pbo = self.pbo_proxy(in_sample_sharpes, out_sample_sharpes)

        is_overfit = dsr < (1 - self.confidence_level) or pbo > 0.5

        return MultipleTestingReport(
            n_trials=n_trials,
            best_sharpe_observed=observed_sharpe,
            expected_max_sharpe_under_null=e_max,
            deflated_sharpe_ratio=dsr,
            haircut_pct=haircut,
            pbo_proxy=pbo,
            is_likely_overfit=is_overfit,
            details={
                "skewness": skew,
                "kurtosis": kurt,
                "n_observations": n_obs,
                "confidence_level": self.confidence_level,
            },
        )
