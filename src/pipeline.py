"""
Full pipeline orchestrator — per 00_MASTER_CONTEXT.md deliverables.

Pipeline: data → signals → backtest → WFO → robustness reports

Coordinates all layers:
1) Load and prepare data
2) Run strategies to generate signals
3) Combine via portfolio allocator + risk overlays
4) Backtest with cost model
5) Walk-forward optimize
6) Run stress tests and regime conditioning
7) Generate reports
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.contracts import ContractMetadata, load_instrument_configs
from src.data.loader import load_bars_from_csv, resample_bars, compute_returns
from src.data.rolls import RollCalendar
from src.data.continuous import build_continuous_series, build_tradable_series
from src.features.volatility import ewma_volatility, rolling_ewma_correlation_series
from src.features.regime import RegimeTagger, RegimeLabels
from src.strategies.base import BaseStrategy, StrategySignal
from src.strategies.trend import TrendStrategy
from src.strategies.carry import CarryStrategy
from src.strategies.vol_breakout import VolBreakoutStrategy
from src.portfolio.construction import RiskParityAllocator
from src.portfolio.risk_overlay import (
    RiskOverlay,
    DrawdownLadder,
    CorrelationHaircut,
)
from src.portfolio.margin import MarginMonitor
from src.execution.costs import CostModel
from src.engine import VectorizedBacktestEngine, BacktestResult
from src.validation.wfo import WalkForwardOptimizer, WFOFold
from src.validation.regime_oos import RegimeConditionedOOS
from src.validation.stress import StressTestSuite
from src.validation.leakage import LeakageDetector
from src.validation.multiple_testing import MultipleTestingDiagnostics
from src.reporting.metrics import PerformanceMetrics
from src.reporting.reports import ReportGenerator


class Pipeline:
    """Full research pipeline orchestrator.

    Implements the complete workflow:
    data → signals → backtest → WFO → robustness reports
    """

    def __init__(
        self,
        project_root: str | None = None,
        output_dir: str = "output",
    ):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.project_root = project_root
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load configs
        self.instruments = load_instrument_configs(
            os.path.join(project_root, "config", "instruments.yaml")
        )

        with open(os.path.join(project_root, "config", "strategies.yaml")) as f:
            self.strategy_config = yaml.safe_load(f)

        with open(os.path.join(project_root, "config", "risk.yaml")) as f:
            self.risk_config = yaml.safe_load(f)

        with open(os.path.join(project_root, "config", "validation.yaml")) as f:
            self.validation_config = yaml.safe_load(f)

        # Initialize components
        self._init_strategies()
        self._init_portfolio()
        self._init_validation()
        self._init_reporting()

    def _init_strategies(self, symbol: str = "ES") -> None:
        """Initialize strategy instances from config."""
        sc = self.strategy_config["strategies"]

        self.strategies: dict[str, BaseStrategy] = {}

        if sc["S1_trend"]["enabled"]:
            self.strategies["S1_trend"] = TrendStrategy(
                lookbacks=sc["S1_trend"]["lookback_periods"],
                target_vol=sc["S1_trend"]["target_vol_per_instrument"],
                vol_lookback=sc["S1_trend"]["vol_lookback"],
                rebalance_freq=sc["S1_trend"]["rebalance_frequency"],
                symbol=symbol,
            )

        if sc["S2_carry"]["enabled"]:
            self.strategies["S2_carry"] = CarryStrategy(
                carry_lookback=sc["S2_carry"]["carry_lookback"],
                target_vol=sc["S2_carry"]["target_vol_per_instrument"],
                vol_lookback=sc["S2_carry"]["vol_lookback"],
                rebalance_freq=sc["S2_carry"]["rebalance_frequency"],
                symbol=symbol,
            )

        if sc["S3_vol_breakout"]["enabled"]:
            self.strategies["S3_vol_breakout"] = VolBreakoutStrategy(
                range_lookback=sc["S3_vol_breakout"]["range_lookback"],
                atr_lookback=sc["S3_vol_breakout"]["atr_lookback"],
                breakout_threshold=sc["S3_vol_breakout"]["breakout_threshold_pct"],
                atr_percentile_threshold=sc["S3_vol_breakout"][
                    "atr_percentile_threshold"
                ],
                max_loss_per_trade_pct=sc["S3_vol_breakout"][
                    "max_loss_per_trade_pct"
                ],
                target_vol=sc["S3_vol_breakout"]["target_vol_per_instrument"],
                vol_lookback=sc["S3_vol_breakout"]["vol_lookback"],
                rebalance_freq=sc["S3_vol_breakout"]["rebalance_frequency"],
                symbol=symbol,
            )

    def _init_portfolio(self) -> None:
        """Initialize portfolio and risk components."""
        pc = self.risk_config["portfolio"]
        dc = self.risk_config["drawdown_ladder"]
        mc = self.risk_config["margin"]

        self.allocator = RiskParityAllocator(
            portfolio_vol_target=pc["portfolio_vol_target"],
            max_leverage=pc["max_leverage"],
            vol_lookback=pc["correlation_lookback"],
        )

        thresholds = [
            {"threshold": t["threshold"], "risk_multiplier": t["risk_multiplier"]}
            for t in dc.get("thresholds", [])
        ]

        cooldown = dc.get("cooldown_days", 10)
        recovery = dc.get("recovery_window", 20)
        recovery_sharpe = dc.get("recovery_min_sharpe", -0.5)
        max_cooldown = dc.get("max_cooldown_days", 30)

        self.drawdown_ladder = DrawdownLadder(
            thresholds=thresholds if thresholds else None,
            cooldown_days=cooldown,
            recovery_window=recovery,
            recovery_min_sharpe=recovery_sharpe,
            max_cooldown_days=max_cooldown,
        )

        cc = pc["correlation_haircuts"]
        self.corr_haircut = CorrelationHaircut(
            avg_corr_threshold_high=cc["avg_corr_threshold_high"],
            risk_scale_at_high_corr=cc["risk_scale_at_high_corr"],
            avg_corr_threshold_extreme=cc["avg_corr_threshold_extreme"],
            risk_scale_at_extreme_corr=cc["risk_scale_at_extreme_corr"],
        )

        self.risk_overlay = RiskOverlay(
            drawdown_ladder=self.drawdown_ladder,
            corr_haircut=self.corr_haircut,
            max_leverage=pc["max_leverage"],
        )

        self.margin_monitor = MarginMonitor(
            max_utilization=mc["max_utilization"],
            stress_margin_multiplier=mc["stress_margin_multiplier"],
            leverage_cap_under_stress=mc["leverage_cap_under_stress"],
        )

        # Regime-adaptive scaling config
        rs = self.risk_config.get("regime_scaling", {})
        self.regime_scaling_enabled = rs.get("enabled", False)
        self.regime_scaling_factor = rs.get("scaling_factor", 0.5)
        self.regime_min_scale = rs.get("min_scale", 0.5)

        ec = self.risk_config["execution"]["cost_model"]
        self.cost_model = CostModel(
            slippage_base_bps=ec["slippage_base_bps"],
            slippage_size_coefficient=ec["slippage_size_coefficient"],
            impact_power=ec["impact_power"],
        )

    def _init_validation(self) -> None:
        """Initialize validation components."""
        wc = self.validation_config["walk_forward"]
        self.wfo = WalkForwardOptimizer(
            train_days=wc["train_days"],
            test_days=wc["test_days"],
            step_days=wc["step_days"],
            purge_days=wc["purge_days"],
            embargo_days=wc["embargo_days"],
            min_train_days=wc["min_train_days"],
        )

        rc = self.validation_config["regime_conditioning"]
        self.regime_tagger = RegimeTagger(
            vol_lookback=rc["vol_regime"]["lookback"],
            corr_lookback=rc["corr_regime"]["lookback"],
            adx_period=rc["trend_regime"]["adx_period"],
            adx_threshold=rc["trend_regime"]["threshold"],
        )

        self.regime_oos = RegimeConditionedOOS(
            regime_tagger=self.regime_tagger,
        )

        sc = self.validation_config["stress_tests"]
        self.stress_suite = StressTestSuite(
            slippage_multipliers=sc["slippage_multipliers"],
        )

        self.leakage_detector = LeakageDetector()
        self.mt_diagnostics = MultipleTestingDiagnostics(
            confidence_level=self.validation_config["multiple_testing"][
                "confidence_level"
            ]
        )

    def _init_reporting(self) -> None:
        """Initialize reporting components."""
        self.report_gen = ReportGenerator(output_dir=self.output_dir)
        self.perf = PerformanceMetrics()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        symbols: list[str] | None = None,
    ) -> dict[str, dict]:
        """Load and prepare data for all instruments.

        Returns dict of symbol -> {
            'raw_1min': DataFrame,
            'daily': DataFrame,
            'returns': Series,
            'continuous': ContinuousSeries,
            'tradable': ContinuousSeries,
        }
        """
        if symbols is None:
            symbols = [s for s in self.instruments if self.instruments[s].data_file]

        data = {}
        for symbol in symbols:
            meta = self.instruments[symbol]
            if meta.data_file is None:
                continue

            file_path = os.path.join(self.project_root, meta.data_file)
            if not os.path.exists(file_path):
                print(f"Warning: data file not found for {symbol}: {file_path}")
                continue

            print(f"Loading {symbol} from {file_path}...")

            # Load 1-minute bars
            raw_1min = load_bars_from_csv(file_path)

            # Resample to daily
            daily = resample_bars(raw_1min, "1D")

            # Compute returns
            returns = compute_returns(daily["close"], method="log")

            # Build roll calendar
            start_year = daily.index[0].year
            end_year = daily.index[-1].year
            roll_cal = RollCalendar(meta)
            roll_dates = roll_cal.generate_roll_calendar(start_year, end_year)

            # Build continuous series
            continuous = build_continuous_series(daily, roll_dates)
            tradable = build_tradable_series(
                daily,
                roll_dates,
                spread_cost_per_roll=meta.avg_spread_ticks * meta.tick_size,
            )

            data[symbol] = {
                "raw_1min": raw_1min,
                "daily": daily,
                "returns": returns,
                "continuous": continuous,
                "tradable": tradable,
                "metadata": meta,
                "roll_calendar": roll_dates,
            }

            print(
                f"  {symbol}: {len(raw_1min)} 1min bars, "
                f"{len(daily)} daily bars, "
                f"{len(roll_dates)} rolls"
            )

        return data

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------

    def run_strategies(
        self,
        data: dict[str, dict],
        params_override: dict[str, dict] | None = None,
    ) -> dict[str, StrategySignal]:
        """Run all enabled strategies on the data.

        Parameters
        ----------
        data : dict
            Output of load_data().
        params_override : dict, optional
            Maps strategy_name -> params dict for WFO overrides.

        Returns
        -------
        dict mapping strategy_name -> StrategySignal
        """
        signals = {}

        # v1: single instrument — use first symbol directly (no inner loop)
        first_symbol = list(data.keys())[0]
        d = data[first_symbol]

        for strat_name, strategy in self.strategies.items():
            params = (
                params_override.get(strat_name) if params_override else None
            )

            sig = strategy.generate_signals(
                d["daily"], d["returns"], params
            )
            signals[strat_name] = sig

        return signals

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str] | None = None,
        run_wfo: bool = True,
        run_stress: bool = True,
        run_id: str | None = None,
    ) -> dict:
        """Run the full pipeline.

        Returns
        -------
        dict
            Complete results including all reports.
        """
        import datetime as dt

        if run_id is None:
            run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 60)
        print("REGIME-ROBUST SYSTEMATIC FUTURES SYSTEM")
        print(f"Run ID: {run_id}")
        print("=" * 60)

        # 1. Load data
        print("\n[1/7] Loading data...")
        data = self.load_data(symbols)
        if not data:
            raise ValueError("No data loaded. Check instrument configs and data files.")

        # Re-init strategies with the actual symbol name
        first_symbol = list(data.keys())[0]
        self._init_strategies(symbol=first_symbol)

        # 2. Generate strategy signals
        print("\n[2/7] Generating strategy signals...")
        signals = self.run_strategies(data)

        # 3. Combine into portfolio weights
        print("\n[3/7] Building portfolio weights...")
        # For v1: single instrument, combine strategy weights directly
        first_symbol = list(data.keys())[0]
        d = data[first_symbol]

        strategy_weights = {}
        strategy_returns_for_alloc = {}
        for strat_name, sig in signals.items():
            strategy_weights[strat_name] = sig.raw_weights
            # Compute strategy returns for allocation
            strat_ret = (sig.raw_weights.iloc[:, 0] * d["returns"]).dropna()
            strategy_returns_for_alloc[strat_name] = strat_ret

        combined_weights = self.allocator.allocate(
            strategy_weights, strategy_returns_for_alloc
        )

        # Regime-adaptive position sizing: scale down in high-risk regimes
        regime_labels = self.regime_tagger.tag_all(d["daily"], d["returns"])
        if self.regime_scaling_enabled:
            composite = regime_labels.composite_risk.reindex(
                combined_weights.index
            ).fillna(0.0)
            regime_scale = (
                1.0 - self.regime_scaling_factor * composite
            ).clip(lower=self.regime_min_scale)
            combined_weights = combined_weights.multiply(regime_scale, axis=0)

        # 4. Backtest with costs
        print("\n[4/7] Running backtest...")
        engine = VectorizedBacktestEngine(
            cost_model=self.cost_model,
            initial_equity=1_000_000,
        )

        result = engine.run(
            weights=combined_weights,
            returns=d["returns"],
            metadata={first_symbol: d["metadata"]},
            prices=d["daily"][["close"]].rename(columns={"close": first_symbol}),
        )

        # Apply risk overlays
        print("  Applying risk overlays...")

        # Compute average pairwise correlation of strategy returns
        avg_corr = None
        if len(strategy_returns_for_alloc) >= 2:
            strat_ret_df = pd.DataFrame(strategy_returns_for_alloc)
            strat_ret_df = strat_ret_df.reindex(combined_weights.index).fillna(0.0)
            avg_corr = rolling_ewma_correlation_series(
                strat_ret_df,
                com=self.risk_config["portfolio"]["correlation_lookback"],
            )

        adjusted_weights = self.risk_overlay.apply(
            combined_weights,
            result.equity_curve,
            avg_corr=avg_corr,
        )

        # Re-run with adjusted weights
        result_adjusted = engine.run(
            weights=adjusted_weights,
            returns=d["returns"],
            metadata={first_symbol: d["metadata"]},
            prices=d["daily"][["close"]].rename(columns={"close": first_symbol}),
        )

        print(f"  Sharpe (raw):      {result.metrics['sharpe_ratio']:.3f}")
        print(f"  Sharpe (adjusted): {result_adjusted.metrics['sharpe_ratio']:.3f}")
        print(f"  Max DD (adjusted): {result_adjusted.metrics['max_drawdown']:.3%}")

        # 5. Walk-forward optimization
        wfo_folds = []
        if run_wfo:
            print("\n[5/7] Running walk-forward optimization...")
            wfo_folds = self._run_wfo(d, first_symbol, run_id)

        # 6. Regime-conditioned OOS + stress tests
        regime_reports = []
        stress_results = []

        print("\n[6/7] Running regime conditioning and stress tests...")
        regime_report = self.regime_oos.generate_report(
            result_adjusted.returns,
            d["daily"],
            regime_labels,
        )
        regime_reports.append(regime_report)

        if run_stress:
            weight_changes = adjusted_weights.diff().fillna(0.0)
            cost_per_unit = 0.001  # 10bps baseline
            stress_results = self.stress_suite.run_all(
                result_adjusted.returns,
                weight_changes,
                d["daily"],
                cost_per_unit,
            )

        # 7. Generate reports
        print("\n[7/7] Generating reports...")
        reports = self._generate_all_reports(
            result_adjusted,
            wfo_folds,
            regime_reports,
            stress_results,
            run_id,
        )

        # Leakage checks
        print("\nRunning leakage detection...")
        leakage_results = self._run_leakage_checks(signals, d, wfo_folds)
        reports["leakage"] = [
            {"test": r.test_name, "passed": r.passed, "details": r.details}
            for r in leakage_results
        ]

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Reports saved to: {self.output_dir}")
        print("=" * 60)

        return {
            "result": result_adjusted,
            "wfo_folds": wfo_folds,
            "regime_reports": regime_reports,
            "stress_results": stress_results,
            "reports": reports,
            "data": data,
        }

    def _run_wfo(
        self,
        data_dict: dict,
        symbol: str,
        run_id: str,
    ) -> list[WFOFold]:
        """Run walk-forward optimization across all strategies."""
        daily = data_dict["daily"]
        returns = data_dict["returns"]
        meta = data_dict["metadata"]

        # Define strategy function for WFO — maps params to each strategy type
        def strategy_fn(
            data_slice: pd.DataFrame,
            returns_slice: pd.Series,
            params: dict,
        ) -> pd.Series:
            """Run all strategies with per-strategy params and return combined returns."""
            all_weights = []

            for strat_name, strategy in self.strategies.items():
                # Map global WFO params to per-strategy params
                strat_params = {}
                if "S1_trend" in strat_name:
                    if "lookbacks" in params:
                        strat_params["lookbacks"] = params["lookbacks"]
                    if "target_vol" in params:
                        strat_params["target_vol"] = params["target_vol"]
                elif "S2_carry" in strat_name:
                    if "carry_lookback" in params:
                        strat_params["carry_lookback"] = params["carry_lookback"]
                    if "target_vol" in params:
                        strat_params["target_vol"] = params["target_vol"]
                elif "S3_vol_breakout" in strat_name:
                    if "range_lookback" in params:
                        strat_params["range_lookback"] = params["range_lookback"]
                    if "atr_lookback" in params:
                        strat_params["atr_lookback"] = params["atr_lookback"]
                    if "target_vol" in params:
                        strat_params["target_vol"] = params["target_vol"]

                sig = strategy.generate_signals(
                    data_slice, returns_slice, strat_params or None
                )
                all_weights.append(sig.raw_weights.iloc[:, 0])

            if all_weights:
                combined = pd.concat(all_weights, axis=1).mean(axis=1)
                strat_returns = (combined * returns_slice).dropna()
                return strat_returns
            return pd.Series(0.0, index=data_slice.index)

        # Parameter grid — covers per-strategy params
        param_grid = [
            {"lookbacks": [21, 63, 252], "target_vol": 0.08,
             "carry_lookback": 63, "range_lookback": 20, "atr_lookback": 14},
            {"lookbacks": [21, 63, 252], "target_vol": 0.10,
             "carry_lookback": 63, "range_lookback": 20, "atr_lookback": 14},
            {"lookbacks": [21, 63, 252], "target_vol": 0.12,
             "carry_lookback": 63, "range_lookback": 20, "atr_lookback": 14},
            {"lookbacks": [21, 63, 126], "target_vol": 0.10,
             "carry_lookback": 42, "range_lookback": 15, "atr_lookback": 10},
            {"lookbacks": [42, 126, 252], "target_vol": 0.10,
             "carry_lookback": 126, "range_lookback": 30, "atr_lookback": 20},
        ]

        folds = self.wfo.run(daily, returns, strategy_fn, param_grid)

        # Validate fold integrity
        issues = self.wfo.validate_fold_integrity(folds)
        if issues:
            print(f"  WFO INTEGRITY ISSUES: {issues}")

        print(f"  {len(folds)} folds completed")
        if folds:
            oos_sharpes = [f.test_metric for f in folds]
            print(f"  OOS Sharpe: mean={np.mean(oos_sharpes):.3f}, "
                  f"std={np.std(oos_sharpes):.3f}")

        return folds

    def _run_leakage_checks(
        self,
        signals: dict[str, StrategySignal],
        data_dict: dict,
        folds: list[WFOFold],
    ) -> list:
        """Run leakage detection on all strategies."""
        results = []

        for strat_name, sig in signals.items():
            signal_series = sig.signals.iloc[:, 0]
            returns = data_dict["returns"]

            # Test signal timing
            timing_result = self.leakage_detector.test_signal_timing(
                signal_series, returns
            )
            timing_result.test_name = f"{strat_name}_{timing_result.test_name}"
            results.append(timing_result)

        # Test WFO fold integrity
        if folds:
            fold_result = self.leakage_detector.test_wfo_fold_integrity(
                folds, self.wfo.purge_days
            )
            results.append(fold_result)

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.test_name}: {r.details}")

        return results

    def _generate_all_reports(
        self,
        result: BacktestResult,
        folds: list[WFOFold],
        regime_reports: list,
        stress_results: list,
        run_id: str,
    ) -> dict:
        """Generate all output reports."""
        reports = {}

        # WFO report
        if folds:
            reports["wfo"] = self.report_gen.generate_wfo_report(folds, run_id)

        # Regime report
        if regime_reports:
            reports["regime"] = self.report_gen.generate_regime_report(
                regime_reports, run_id
            )

        # Stress report
        if stress_results:
            reports["stress"] = self.report_gen.generate_stress_report(
                stress_results, run_id
            )

        # Multiple testing
        if folds:
            is_sharpes = [f.train_metric for f in folds]
            oos_sharpes = [f.test_metric for f in folds]
            mt_report = self.mt_diagnostics.generate_report(
                n_trials=self.wfo.total_trials,
                observed_sharpe=max(is_sharpes) if is_sharpes else 0,
                returns=result.returns,
                in_sample_sharpes=is_sharpes,
                out_sample_sharpes=oos_sharpes,
            )
            reports["multiple_testing"] = (
                self.report_gen.generate_multiple_testing_report(mt_report, run_id)
            )

        # Equity curve
        self.report_gen.save_equity_curve(
            result.returns,
            label="portfolio_adjusted",
            run_id=run_id,
        )

        return reports
