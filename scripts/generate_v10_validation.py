"""
Generate stress test & WFO validation data for V10 + V11 configs.

For each run (V10 baseline/conservative/balanced/aggressive + V11 expanded):
1. Stress tests — via StressTestSuite.run_all()
2. WFO stability — rolling 1-year windows with 3-year lookback

Run once: python scripts/generate_v10_validation.py
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.validation.stress import StressTestSuite
from src.validation.wfo import WFOFold
from src.reporting.reports import ReportGenerator
from src.reporting.metrics import PerformanceMetrics

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
V10_RUN_IDS = [
    "v10_baseline", "v10_conservative", "v10_balanced", "v10_aggressive",
    "v10_lev_2x", "v10_lev_4x", "v10_lev_8x", "v10_lev_10x",
    # V11 configs
    "v11_expanded_trendcarry", "v11_all_moderate", "v11_all_growth", "v11_all_aggressive",
]

perf = PerformanceMetrics()
reporter = ReportGenerator(output_dir=OUTPUT_DIR)
stress_suite = StressTestSuite()


def load_equity_returns(run_id: str) -> pd.Series:
    """Load portfolio returns from equity curve CSV."""
    path = os.path.join(OUTPUT_DIR, f"equity_curve_portfolio_adjusted_{run_id}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    return df["returns"].dropna()


def load_strategy_weights(run_id: str) -> pd.DataFrame:
    """Load strategy data and extract signal columns as weights proxy."""
    path = os.path.join(OUTPUT_DIR, f"strategy_data_{run_id}.csv")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    signal_cols = [c for c in df.columns if c.endswith("_signal")]
    return df[signal_cols]


def load_es_daily_prices() -> pd.DataFrame:
    """Load ES daily OHLCV for adverse tick cost computation."""
    from src.data.loader import load_bars_from_csv, resample_bars
    es_path = os.path.join(
        PROJECT_ROOT,
        "ES_full_1min_continuous_UNadjusted_13wjmr",
        "ES_full_1min_continuous_UNadjusted.txt",
    )
    raw = load_bars_from_csv(es_path)
    raw = resample_bars(raw, "1D")
    if hasattr(raw.index, 'dayofweek'):
        raw = raw[raw.index.dayofweek < 5]
    raw.index = raw.index.normalize()
    raw = raw[~raw.index.duplicated(keep='first')]
    return raw[["open", "high", "low", "close", "volume"]]


def generate_stress(run_id: str, returns: pd.Series, weights: pd.DataFrame,
                    prices: pd.DataFrame) -> None:
    """Run stress tests and save report."""
    weight_changes = weights.diff().fillna(0)

    # Align all to common index
    common = returns.index.intersection(weight_changes.index).intersection(prices.index)
    returns_a = returns.loc[common]
    wc_a = weight_changes.loc[common]
    prices_a = prices.loc[common]

    results = stress_suite.run_all(
        base_returns=returns_a,
        weight_changes=wc_a,
        prices=prices_a,
        cost_per_unit=0.001,
    )

    report = reporter.generate_stress_report(results, run_id=run_id)
    passed = report["all_passed"]
    n = report["n_scenarios"]
    print(f"  Stress: {n} scenarios, all_passed={passed}")


def generate_wfo(run_id: str, returns: pd.Series) -> None:
    """Build rolling-window stability folds and save WFO report.

    V10 has no free parameters — methodology is fixed. So we do
    rolling window stability analysis instead of traditional WFO:
    - 756-day (3-year) train window -> compute Sharpe as train_metric
    - 252-day (1-year) test window -> compute full metrics as test_metrics
    - Step forward by 252 days (non-overlapping test windows)
    - Purge gap = 5 days between train and test
    """
    train_days = 756
    test_days = 252
    step_days = 252
    purge_days = 5

    dates = returns.index.sort_values()
    n = len(dates)

    folds = []
    fold_id = 0
    pos = 0

    while True:
        train_start_idx = pos
        train_end_idx = pos + train_days - 1
        purge_start_idx = train_end_idx + 1
        purge_end_idx = purge_start_idx + purge_days - 1
        test_start_idx = purge_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            break

        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        purge_start = dates[purge_start_idx]
        purge_end = dates[purge_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        train_rets = returns.loc[train_start:train_end]
        test_rets = returns.loc[test_start:test_end]

        # Train metric: Sharpe of preceding window
        train_sharpe = _sharpe(train_rets)

        # Test metric: Sharpe of test window
        test_sharpe = _sharpe(test_rets)

        fold = WFOFold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            purge_start=purge_start,
            purge_end=purge_end,
            best_params={"method": "fixed_v10", "note": "no free parameters"},
            train_metric=train_sharpe,
            test_metric=test_sharpe,
            test_returns=test_rets,
        )
        folds.append(fold)
        fold_id += 1

        # Step forward
        pos += step_days

    report = reporter.generate_wfo_report(folds, run_id=run_id)
    n_folds = report["n_folds"]
    if folds:
        oos_sharpes = [f.test_metric for f in folds]
        print(f"  WFO: {n_folds} folds, OOS Sharpe: {np.mean(oos_sharpes):.3f} +/- {np.std(oos_sharpes):.3f}")
    else:
        print(f"  WFO: 0 folds (insufficient data)")


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n=== Generating V10 Validation Data ===\n")

    # Load ES prices once (shared across all runs)
    print("Loading ES daily prices...")
    es_prices = load_es_daily_prices()
    print(f"  ES: {len(es_prices)} daily bars\n")

    for run_id in V10_RUN_IDS:
        eq_path = os.path.join(OUTPUT_DIR, f"equity_curve_portfolio_adjusted_{run_id}.csv")
        if not os.path.exists(eq_path):
            print(f"SKIP {run_id}: no equity curve CSV")
            continue

        print(f"{run_id}:")
        returns = load_equity_returns(run_id)
        weights = load_strategy_weights(run_id)
        print(f"  Data: {len(returns)} days, {len(weights.columns)} instruments")

        generate_stress(run_id, returns, weights, es_prices)
        generate_wfo(run_id, returns)
        print()

    print("=== Done! Refresh dashboard to see Validation tab ===")
