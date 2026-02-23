"""Reusable HTML components — KPI cards, styled tables, regime pills."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.lib.theme import (
    ACCENT_TEAL, ACCENT_RED, TEXT_PRIMARY, TEXT_SECONDARY,
    REGIME_COLORS, STRATEGY_COLORS, BG_CARD,
)


def kpi_card(label: str, value: str, positive: bool | None = None) -> str:
    """Return HTML for a single KPI card.

    positive=True  → teal
    positive=False → red
    positive=None  → neutral white
    """
    cls = ""
    if positive is True:
        cls = " positive"
    elif positive is False:
        cls = " negative"

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{cls}">{value}</div>
    </div>
    """


def render_kpi_row(kpis: list[tuple[str, str, bool | None]], cols_per_row: int = 6) -> None:
    """Render a row of KPI cards using st.columns."""
    cols = st.columns(cols_per_row)
    for i, (label, value, positive) in enumerate(kpis):
        with cols[i % cols_per_row]:
            st.markdown(kpi_card(label, value, positive), unsafe_allow_html=True)


def section_header(text: str) -> None:
    """Render a styled section divider."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def regime_pill(label: str) -> str:
    """Return HTML for a regime state pill."""
    css_class = f"regime-{label.lower().replace(' ', '_')}"
    return f'<span class="regime-pill {css_class}">{label}</span>'


def pass_fail_banner(passed: bool, text: str = "") -> None:
    """Render a pass/fail banner."""
    if passed:
        msg = text or "ALL TESTS PASSED"
        st.markdown(f'<div class="pass-banner">{msg}</div>', unsafe_allow_html=True)
    else:
        msg = text or "TESTS FAILED"
        st.markdown(f'<div class="fail-banner">{msg}</div>', unsafe_allow_html=True)


def styled_metrics_table(metrics: dict, title: str = "") -> None:
    """Render a dict of metrics as a styled HTML table."""
    rows = ""
    for key, val in metrics.items():
        display_key = key.replace("_", " ").title()
        if isinstance(val, float):
            if "pct" in key or "return" in key or "drawdown" in key or "var" in key or "cvar" in key or "rate" in key or "utilization" in key:
                display_val = f"{val:.2%}"
            elif "ratio" in key or "factor" in key or "sharpe" in key or "sortino" in key or "calmar" in key:
                display_val = f"{val:.3f}"
            else:
                display_val = f"{val:.4f}"
        else:
            display_val = str(val)
        rows += f"<tr><td>{display_key}</td><td style='text-align:right;'>{display_val}</td></tr>"

    html = f"""
    <table class="styled-table">
        <thead><tr><th>{title or 'Metric'}</th><th style='text-align:right;'>Value</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


def tearsheet_table(metrics: dict[str, tuple], title: str = "Key Performance Metrics") -> None:
    """Render QuantStats-style tear sheet metrics as a styled two-column table.

    metrics: {name: (value, format_str)} where format_str is ".2f", ".2%", or "d".
    """
    # Group definitions matching QuantStats layout
    groups = [
        (None, ["Risk-Free Rate", "Time in Market", "Cumulative Return", "CAGR%"]),
        (None, ["Sharpe", "Prob. Sharpe Ratio", "Smart Sharpe", "Sortino",
                 "Smart Sortino", "Sortino/\u221a2", "Smart Sortino/\u221a2", "Omega"]),
        (None, ["Max Drawdown", "Longest DD Days", "Volatility (ann.)",
                 "R\u00b2", "Calmar", "Skew", "Kurtosis"]),
        (None, ["Expected Daily", "Expected Monthly", "Expected Yearly",
                 "Kelly Criterion", "Risk of Ruin", "Daily Value-at-Risk",
                 "Expected Shortfall (cVaR)"]),
        (None, ["Max Consecutive Wins", "Max Consecutive Losses"]),
        (None, ["Gain/Pain Ratio", "Gain/Pain (1M)", "Payoff Ratio",
                 "Profit Factor", "Common Sense Ratio", "CPC Index",
                 "Tail Ratio", "Outlier Win Ratio", "Outlier Loss Ratio"]),
    ]

    rows = ""
    for _, keys in groups:
        for key in keys:
            if key not in metrics:
                continue
            val, fmt = metrics[key]
            if fmt == "d":
                display = f"{int(val):,}"
            elif fmt.endswith("%"):
                display = f"{val:{fmt}}"
            else:
                display = f"{val:{fmt}}"
            rows += f'<tr><td style="padding:5px 12px;">{key}</td><td style="text-align:right;padding:5px 12px;">{display}</td></tr>'
        # separator between groups
        rows += '<tr><td colspan="2" style="border-bottom:1px solid rgba(123,134,152,0.1);padding:0;height:4px;"></td></tr>'

    html = f"""
    <div style="display:flex;justify-content:center;">
    <table class="styled-table" style="max-width:520px;width:100%;">
        <thead><tr>
            <th style="min-width:200px;">Metric</th>
            <th style="text-align:right;">Strategy</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def stress_scenario_table(scenarios: list[dict]) -> None:
    """Render stress test scenarios as a styled table with degradation bars."""
    rows = ""
    for s in scenarios:
        passed_icon = f'<span style="color:{ACCENT_TEAL};">PASS</span>' if s["passed"] else f'<span style="color:{ACCENT_RED};">FAIL</span>'
        deg = s.get("degradation_pct", 0)
        deg_color = ACCENT_TEAL if deg < 20 else (ACCENT_RED if deg > 40 else "#FFB347")
        bar_width = min(abs(deg), 100)
        bar = f'<div style="background:{deg_color}30;border-radius:3px;height:16px;width:{bar_width}%;"><div style="background:{deg_color};border-radius:3px;height:100%;width:100%;"></div></div>'

        rows += f"""<tr>
            <td>{s['name']}</td>
            <td style="text-align:center;">{passed_icon}</td>
            <td style="text-align:right;">{s['base_sharpe']:.3f}</td>
            <td style="text-align:right;">{s['stressed_sharpe']:.3f}</td>
            <td style="text-align:right;color:{deg_color};">{deg:.1f}%</td>
            <td style="width:100px;">{bar}</td>
        </tr>"""

    html = f"""
    <table class="styled-table">
        <thead><tr>
            <th>Scenario</th><th style="text-align:center;">Status</th>
            <th style="text-align:right;">Base Sharpe</th>
            <th style="text-align:right;">Stressed</th>
            <th style="text-align:right;">Degradation</th>
            <th>Impact</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


def strategy_kpi_table(strategy_data: pd.DataFrame) -> None:
    """Per-strategy KPI table."""
    return_cols = [c for c in strategy_data.columns if c.endswith("_return")]
    if not return_cols:
        st.info("No strategy return data available.")
        return

    rows = ""
    for col in return_cols:
        name = col.replace("_return", "")
        rets = strategy_data[col].dropna()
        if len(rets) < 2:
            continue

        ann_ret = ((1 + rets).prod() ** (252 / len(rets))) - 1
        ann_vol = rets.std() * (252 ** 0.5)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        max_dd = ((cum - peak) / peak).min()
        sortino_denom = rets[rets < 0].std() * (252 ** 0.5)
        sortino = ann_ret / sortino_denom if sortino_denom > 0 else 0

        color = STRATEGY_COLORS.get(name, TEXT_PRIMARY)
        name_html = f'<span style="color:{color};font-weight:600;">{name}</span>'

        sharpe_color = ACCENT_TEAL if sharpe > 0.5 else (ACCENT_RED if sharpe < 0 else TEXT_PRIMARY)

        rows += f"""<tr>
            <td>{name_html}</td>
            <td style="text-align:right;color:{sharpe_color};">{sharpe:.3f}</td>
            <td style="text-align:right;">{ann_ret:.2%}</td>
            <td style="text-align:right;">{ann_vol:.2%}</td>
            <td style="text-align:right;color:{ACCENT_RED};">{max_dd:.2%}</td>
            <td style="text-align:right;">{sortino:.3f}</td>
        </tr>"""

    html = f"""
    <table class="styled-table">
        <thead><tr>
            <th>Strategy</th><th style="text-align:right;">Sharpe</th>
            <th style="text-align:right;">Ann Return</th><th style="text-align:right;">Vol</th>
            <th style="text-align:right;">Max DD</th><th style="text-align:right;">Sortino</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


def regime_metrics_table(regime_report: dict) -> None:
    """Styled regime breakdown table with conditional Sharpe coloring."""
    breakdowns = regime_report.get("regime_breakdowns", [])
    if not breakdowns:
        st.info("No regime data available.")
        return

    rows = ""
    for bd in breakdowns:
        label = bd.get("label", "?")
        dim = bd.get("dimension", "?")
        m = bd.get("metrics", {})
        if not m:
            continue

        sharpe = m.get("sharpe_ratio", 0)
        sharpe_color = ACCENT_TEAL if sharpe > 0.5 else (ACCENT_RED if sharpe < 0 else TEXT_PRIMARY)
        regime_color = REGIME_COLORS.get(label, TEXT_SECONDARY)
        pill = f'<span class="regime-pill regime-{label}" style="background:{regime_color}25;color:{regime_color};">{label}</span>'

        rows += f"""<tr>
            <td>{dim}</td>
            <td>{pill}</td>
            <td style="text-align:right;">{m.get('n_observations', 0)}</td>
            <td style="text-align:right;color:{sharpe_color};">{sharpe:.3f}</td>
            <td style="text-align:right;">{m.get('annualized_return', 0):.2%}</td>
            <td style="text-align:right;">{m.get('annualized_volatility', 0):.2%}</td>
            <td style="text-align:right;color:{ACCENT_RED};">{m.get('max_drawdown', 0):.2%}</td>
            <td style="text-align:right;">{m.get('sortino_ratio', 0):.3f}</td>
        </tr>"""

    html = f"""
    <table class="styled-table">
        <thead><tr>
            <th>Dimension</th><th>Regime</th><th style="text-align:right;">Obs</th>
            <th style="text-align:right;">Sharpe</th><th style="text-align:right;">Return</th>
            <th style="text-align:right;">Vol</th><th style="text-align:right;">Max DD</th>
            <th style="text-align:right;">Sortino</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


def gauge_html(value: float, label: str, fmt: str = ".1%") -> str:
    """Simple text-based gauge display."""
    if fmt == ".1%":
        display = f"{value:.1%}"
    elif fmt == ".2f":
        display = f"{value:.2f}"
    else:
        display = f"{value:{fmt}}"

    if value < 0.3:
        color = ACCENT_TEAL
    elif value < 0.6:
        color = "#FFB347"
    else:
        color = ACCENT_RED

    return f"""
    <div class="gauge-container">
        <div class="gauge-value" style="color:{color};">{display}</div>
        <div class="gauge-label">{label}</div>
    </div>
    """


def multiple_testing_cards(mt_report: dict) -> None:
    """Render multiple testing diagnostics as KPI cards."""
    dsr = mt_report.get("deflated_sharpe_ratio", 0)
    pbo = mt_report.get("pbo_proxy", 0)
    haircut = mt_report.get("haircut_pct", 0)
    overfit = mt_report.get("is_likely_overfit", False)

    kpis = [
        ("Deflated Sharpe Ratio", f"{dsr:.4f}", dsr > 0.05),
        ("PBO Proxy", f"{pbo:.2f}", pbo < 0.5),
        ("Haircut %", f"{haircut:.1f}%", haircut < 50),
        ("Overfit?", "YES" if overfit else "NO", not overfit),
    ]
    render_kpi_row(kpis, cols_per_row=4)
