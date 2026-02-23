"""Data loading — discover runs, load output CSVs/JSONs, caching."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import streamlit as st


def discover_runs(output_dir: str = "output") -> list[str]:
    """Scan output/ for equity_curve_portfolio_adjusted_*.csv, extract run_ids.

    Supports both timestamp IDs (20260219_171103) and named IDs (jungle_rock_v1).
    Returns list of run_ids sorted newest-first for timestamps, then named runs.
    """
    pattern = re.compile(r"equity_curve_portfolio_adjusted_(.+)\.csv")
    ts_ids = []
    named_ids = []
    if not os.path.isdir(output_dir):
        return []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            run_id = m.group(1)
            if re.fullmatch(r"\d{8}_\d{6}", run_id):
                ts_ids.append(run_id)
            else:
                named_ids.append(run_id)
    # Named runs first (more interesting), then timestamp runs newest-first
    return sorted(named_ids, reverse=True) + sorted(ts_ids, reverse=True)


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: str, index_col: str | None = None, parse_dates: bool = False) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=[index_col] if (parse_dates and index_col) else False)
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def _find_fallback(output_dir: str, prefix: str, suffix: str, all_run_ids: list[str]) -> str | None:
    """Find the most recent file matching prefix_*suffix across all runs."""
    for rid in all_run_ids:
        path = os.path.join(output_dir, f"{prefix}{rid}{suffix}")
        if os.path.exists(path):
            return path
    return None


@st.cache_data(ttl=300)
def load_run(run_id: str, output_dir: str = "output") -> dict[str, Any]:
    """Load all available files for a run.

    For validation files (stress_report, wfo_report, wfo_summary,
    multiple_testing) that are missing from the selected run, falls back
    to the most recent older run that has them.

    Returns dict with keys:
    - equity_curve: DataFrame (timestamp indexed)
    - regime_report: dict | None
    - regime_breakdown: DataFrame | None
    - stress_report: dict | None
    - wfo_report: dict | None
    - multiple_testing: dict | None
    - strategy_data: DataFrame | None
    - risk_timeseries: DataFrame | None
    """
    base = output_dir
    all_runs = discover_runs(output_dir)

    # Equity curve
    eq_path = os.path.join(base, f"equity_curve_portfolio_adjusted_{run_id}.csv")
    equity_df = _load_csv(eq_path)
    if equity_df is not None and "timestamp" in equity_df.columns:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        equity_df = equity_df.set_index("timestamp")
    elif equity_df is not None and equity_df.index.name != "timestamp":
        # Try to parse the first column as dates
        equity_df.index = pd.to_datetime(equity_df.index)

    # Direct load from this run
    result = {
        "equity_curve": equity_df,
        "regime_report": _load_json(os.path.join(base, f"regime_report_{run_id}.json")),
        "regime_breakdown": _load_csv(os.path.join(base, f"regime_breakdown_{run_id}.csv")),
        "stress_report": _load_json(os.path.join(base, f"stress_report_{run_id}.json")),
        "wfo_report": _load_json(os.path.join(base, f"wfo_report_{run_id}.json")),
        "wfo_summary": _load_csv(os.path.join(base, f"wfo_summary_{run_id}.csv")),
        "multiple_testing": _load_json(os.path.join(base, f"multiple_testing_{run_id}.json")),
        "strategy_data": _load_csv(
            os.path.join(base, f"strategy_data_{run_id}.csv"),
            index_col="date", parse_dates=True,
        ),
        "risk_timeseries": _load_csv(
            os.path.join(base, f"risk_timeseries_{run_id}.csv"),
            index_col="date", parse_dates=True,
        ),
    }

    # Fallback: for validation files missing from this run, check same-generation runs only
    # (don't pull V1-V5 data into V10 runs)
    same_gen = [r for r in all_runs if r.startswith(run_id.split("_")[0])]
    fallback_keys = {
        "stress_report": ("stress_report_", ".json", "json"),
        "wfo_report": ("wfo_report_", ".json", "json"),
        "wfo_summary": ("wfo_summary_", ".csv", "csv"),
        "multiple_testing": ("multiple_testing_", ".json", "json"),
    }
    for key, (prefix, suffix, fmt) in fallback_keys.items():
        if result[key] is None:
            fb_path = _find_fallback(base, prefix, suffix, same_gen)
            if fb_path:
                if fmt == "json":
                    result[key] = _load_json(fb_path)
                else:
                    result[key] = _load_csv(fb_path)

    return result


def get_output_dir() -> str:
    """Resolve the output directory relative to project root."""
    # dashboard/ is one level below project root
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(here, "output")
    if os.path.isdir(output):
        return output
    # fallback: cwd
    return os.path.join(os.getcwd(), "output")
