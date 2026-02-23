"""
Report generation — per 00_MASTER_CONTEXT.md deliverables.

Outputs:
- Standard results JSON + CSV
- Fold-by-fold WFO summaries
- Regime breakdown tables
- Cost attribution report
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.reporting.metrics import PerformanceMetrics

if TYPE_CHECKING:
    from src.validation.regime_oos import RegimeOOSReport
    from src.validation.stress import StressTestResult
    from src.validation.multiple_testing import MultipleTestingReport
    from src.validation.wfo import WFOFold


class ReportGenerator:
    """Generate standardized JSON/CSV reports."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.perf = PerformanceMetrics()
        os.makedirs(output_dir, exist_ok=True)

    def generate_wfo_report(
        self,
        folds: list[WFOFold],
        run_id: str | None = None,
    ) -> dict:
        """Generate fold-by-fold WFO summary.

        Parameters
        ----------
        folds : list[WFOFold]
            Completed WFO folds.
        run_id : str, optional
            Unique run identifier.

        Returns
        -------
        dict
            Report data (also saved to disk).
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "n_folds": len(folds),
            "folds": [],
        }

        all_oos_returns = []

        for fold in folds:
            fold_data = {
                "fold_id": fold.fold_id,
                "train_period": f"{fold.train_start} to {fold.train_end}",
                "test_period": f"{fold.test_start} to {fold.test_end}",
                "best_params": fold.best_params,
                "train_metric": fold.train_metric,
                "test_metric": fold.test_metric,
            }

            if fold.test_returns is not None:
                metrics = self.perf.compute_all(fold.test_returns)
                fold_data["test_metrics"] = metrics
                all_oos_returns.append(fold.test_returns)

            report["folds"].append(fold_data)

        # Aggregate OOS metrics
        if all_oos_returns:
            combined_oos = pd.concat(all_oos_returns)
            report["aggregate_oos_metrics"] = self.perf.compute_all(combined_oos)

        # Save
        json_path = os.path.join(self.output_dir, f"wfo_report_{run_id}.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # CSV summary
        csv_data = []
        for fold in report["folds"]:
            row = {
                "fold_id": fold["fold_id"],
                "train_period": fold["train_period"],
                "test_period": fold["test_period"],
                "train_sharpe": fold["train_metric"],
                "test_sharpe": fold["test_metric"],
            }
            if "test_metrics" in fold:
                row.update(
                    {
                        f"test_{k}": v
                        for k, v in fold["test_metrics"].items()
                    }
                )
            csv_data.append(row)

        csv_path = os.path.join(self.output_dir, f"wfo_summary_{run_id}.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        return report

    def generate_regime_report(
        self,
        regime_reports: list[RegimeOOSReport],
        run_id: str | None = None,
    ) -> dict:
        """Generate regime breakdown tables.

        Parameters
        ----------
        regime_reports : list[RegimeOOSReport]
            Per-fold regime reports.
        run_id : str, optional
            Unique run identifier.

        Returns
        -------
        dict
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "n_folds": len(regime_reports),
            "regime_breakdowns": [],
            "warnings": [],
        }

        for rr in regime_reports:
            for bd in rr.regime_breakdowns:
                report["regime_breakdowns"].append(
                    {
                        "fold_id": rr.fold_id,
                        "dimension": bd.regime_dimension,
                        "label": bd.regime_label,
                        "n_observations": bd.n_observations,
                        "metrics": bd.metrics,
                    }
                )
            report["warnings"].extend(rr.warnings)

        # Save
        json_path = os.path.join(
            self.output_dir, f"regime_report_{run_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # CSV pivot table
        rows = []
        for bd in report["regime_breakdowns"]:
            row = {
                "fold_id": bd["fold_id"],
                "dimension": bd["dimension"],
                "label": bd["label"],
                "n_obs": bd["n_observations"],
            }
            if bd["metrics"]:
                row.update(bd["metrics"])
            rows.append(row)

        csv_path = os.path.join(
            self.output_dir, f"regime_breakdown_{run_id}.csv"
        )
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        return report

    def generate_stress_report(
        self,
        stress_results: list[StressTestResult],
        run_id: str | None = None,
    ) -> dict:
        """Generate stress test report.

        Parameters
        ----------
        stress_results : list[StressTestResult]
            Results from StressTestSuite.
        run_id : str, optional
            Unique run identifier.

        Returns
        -------
        dict
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "n_scenarios": len(stress_results),
            "all_passed": all(r.passed for r in stress_results),
            "scenarios": [],
        }

        for sr in stress_results:
            report["scenarios"].append(
                {
                    "name": sr.scenario_name,
                    "description": sr.description,
                    "passed": sr.passed,
                    "base_sharpe": sr.base_sharpe,
                    "stressed_sharpe": sr.stressed_sharpe,
                    "base_return": sr.base_return,
                    "stressed_return": sr.stressed_return,
                    "base_max_dd": sr.base_max_dd,
                    "stressed_max_dd": sr.stressed_max_dd,
                    "degradation_pct": sr.degradation_pct,
                }
            )

        # Save
        json_path = os.path.join(
            self.output_dir, f"stress_report_{run_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        csv_path = os.path.join(
            self.output_dir, f"stress_summary_{run_id}.csv"
        )
        pd.DataFrame(report["scenarios"]).to_csv(csv_path, index=False)

        return report

    def generate_multiple_testing_report(
        self,
        mt_report: MultipleTestingReport,
        run_id: str | None = None,
    ) -> dict:
        """Generate multiple testing diagnostics report."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "n_trials": mt_report.n_trials,
            "best_sharpe_observed": mt_report.best_sharpe_observed,
            "expected_max_sharpe_null": mt_report.expected_max_sharpe_under_null,
            "deflated_sharpe_ratio": mt_report.deflated_sharpe_ratio,
            "haircut_pct": mt_report.haircut_pct,
            "pbo_proxy": mt_report.pbo_proxy,
            "is_likely_overfit": mt_report.is_likely_overfit,
            "details": mt_report.details,
        }

        json_path = os.path.join(
            self.output_dir, f"multiple_testing_{run_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def save_equity_curve(
        self,
        returns: pd.Series,
        initial_equity: float = 1_000_000,
        label: str = "portfolio",
        run_id: str | None = None,
    ) -> str:
        """Save equity curve to CSV.

        Returns
        -------
        str
            Path to saved file.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        equity = initial_equity * (1 + returns).cumprod()
        df = pd.DataFrame(
            {
                "timestamp": equity.index,
                "equity": equity.values,
                "returns": returns.values,
                "cumulative_return": (1 + returns).cumprod().values - 1,
            }
        )

        csv_path = os.path.join(
            self.output_dir, f"equity_curve_{label}_{run_id}.csv"
        )
        df.to_csv(csv_path, index=False)

        return csv_path

    def save_strategy_data(
        self,
        signals: dict[str, 'StrategySignal'],
        strategy_returns: dict[str, pd.Series],
        strategy_weights: dict[str, pd.DataFrame],
        run_id: str | None = None,
    ) -> str:
        """Save per-strategy signals, weights, and returns as a single CSV.

        Columns: date, S1_signal, S1_weight, S1_return, S2_signal, ...
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        frames = {}
        for name, sig in signals.items():
            prefix = name
            # Signal: first column of signals df
            frames[f"{prefix}_signal"] = sig.signals.iloc[:, 0]
            # Weight: first column of raw_weights df
            if name in strategy_weights:
                frames[f"{prefix}_weight"] = strategy_weights[name].iloc[:, 0]
            else:
                frames[f"{prefix}_weight"] = sig.raw_weights.iloc[:, 0]
            # Return
            if name in strategy_returns:
                frames[f"{prefix}_return"] = strategy_returns[name]

        df = pd.DataFrame(frames)
        df.index.name = "date"

        csv_path = os.path.join(
            self.output_dir, f"strategy_data_{run_id}.csv"
        )
        df.to_csv(csv_path)
        return csv_path

    def save_risk_timeseries(
        self,
        dd_multipliers: pd.Series,
        margin_util: pd.Series,
        composite_risk: pd.Series,
        leverage: pd.Series,
        run_id: str | None = None,
    ) -> str:
        """Save risk overlay time series as CSV.

        Columns: date, dd_multiplier, margin_util, composite_risk, leverage
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = pd.DataFrame({
            "dd_multiplier": dd_multipliers,
            "margin_util": margin_util,
            "composite_risk": composite_risk,
            "leverage": leverage,
        })
        df.index.name = "date"

        csv_path = os.path.join(
            self.output_dir, f"risk_timeseries_{run_id}.csv"
        )
        df.to_csv(csv_path)
        return csv_path
