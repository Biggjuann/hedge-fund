#!/usr/bin/env python3
"""
Main entry point for the Regime-Robust Systematic Futures System.

Usage:
    python run.py                    # Full pipeline
    python run.py --no-wfo           # Skip walk-forward (fast run)
    python run.py --no-stress        # Skip stress tests
    python run.py --symbols ES       # Specific instruments
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Regime-Robust Systematic Futures System"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Instruments to run (default: all with data files)",
    )
    parser.add_argument(
        "--no-wfo",
        action="store_true",
        help="Skip walk-forward optimization",
    )
    parser.add_argument(
        "--no-stress",
        action="store_true",
        help="Skip stress tests",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier",
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        project_root=os.path.dirname(os.path.abspath(__file__)),
        output_dir=args.output_dir,
    )

    results = pipeline.run(
        symbols=args.symbols,
        run_wfo=not args.no_wfo,
        run_stress=not args.no_stress,
        run_id=args.run_id,
    )

    # Print summary
    result = results["result"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    if results["stress_results"]:
        print("\nStress Tests:")
        for sr in results["stress_results"]:
            status = "PASS" if sr.passed else "FAIL"
            print(f"  [{status}] {sr.scenario_name}: "
                  f"Sharpe {sr.base_sharpe:.3f} -> {sr.stressed_sharpe:.3f}")

    return results


if __name__ == "__main__":
    main()
