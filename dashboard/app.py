"""
Regime-Robust Systematic Futures -- Dashboard
Single-page app with sidebar navigation. Auto-loads latest run.
Launch: streamlit run dashboard/app.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.lib.theme import (
    CUSTOM_CSS, PLOTLY_CONFIG,
    ACCENT_TEAL, ACCENT_RED, ACCENT_BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY,
    BG_PAGE, BG_CARD, BG_ELEVATED,
    COLOR_TREND, COLOR_CARRY, COLOR_VOL_BREAKOUT,
    STRATEGY_COLORS, REGIME_COLORS,
    ASSET_CLASS_COLORS, V11_ASSET_CLASSES, INSTRUMENT_ASSET_CLASS,
)
from dashboard.lib.data_loader import discover_runs, load_run, get_output_dir
from dashboard.lib.components import (
    render_kpi_row, section_header, regime_pill, styled_metrics_table,
    pass_fail_banner, stress_scenario_table, strategy_kpi_table,
    regime_metrics_table, gauge_html, multiple_testing_cards,
    tearsheet_table,
)
from dashboard.lib.metrics import compute_tearsheet
from dashboard.lib.charts import (
    equity_curve_chart, rolling_sharpe_chart, drawdown_chart,
    return_distribution_chart, monthly_heatmap, annual_bar_chart,
    dd_ladder_timeseries, margin_timeseries, leverage_timeseries,
    composite_risk_overlay, strategy_contribution_area, correlation_heatmap,
    regime_timeline, stress_sharpe_comparison, wfo_sharpe_boxplot,
    wfo_is_oos_scatter,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
_chart_id = 0
def chart(fig, **kwargs):
    """Plotly chart with auto-unique key."""
    global _chart_id
    _chart_id += 1
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"c{_chart_id}", **kwargs)


def _is_v11_run(run_id: str) -> bool:
    """Detect V11 runs by run_id prefix."""
    return run_id.startswith("v11_")


def _aggregate_by_asset_class(strat_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-instrument strategy columns into per-asset-class columns.

    For V11 runs with 30+ instruments, aggregates signal and return columns
    by asset class (INDEX, DEBT, ENERGY, METALS, AGS, FX) using equal-weight
    averaging for signals and summing for returns.
    """
    signal_cols = [c for c in strat_data.columns if c.endswith("_signal")]
    return_cols = [c for c in strat_data.columns if c.endswith("_return")]

    # Group columns by asset class
    agg = pd.DataFrame(index=strat_data.index)
    for cls in V11_ASSET_CLASSES:
        cls_syms = V11_ASSET_CLASSES[cls]
        # Signals: average across instruments in this class
        cls_sig_cols = [f"{s}_signal" for s in cls_syms if f"{s}_signal" in strat_data.columns]
        if cls_sig_cols:
            agg[f"{cls}_signal"] = strat_data[cls_sig_cols].mean(axis=1)
        # Returns: sum across instruments (portfolio return attribution)
        cls_ret_cols = [f"{s}_return" for s in cls_syms if f"{s}_return" in strat_data.columns]
        if cls_ret_cols:
            agg[f"{cls}_return"] = strat_data[cls_ret_cols].sum(axis=1)

    return agg

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime-Robust Fund",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
PAGES = ["Overview", "Performance", "Risk", "Strategies", "Regimes", "Validation"]

with st.sidebar:
    st.markdown(
        f'<div style="font-weight:800;font-size:0.85rem;color:{ACCENT_TEAL};'
        f'letter-spacing:0.1em;text-transform:uppercase;padding:16px 0 24px 0;'
        f'text-shadow:0 0 24px rgba(0,229,176,0.15);">REGIME-ROBUST FUND</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("NAVIGATE", PAGES, label_visibility="collapsed")

# ── Run labels & config ──────────────────────────────────────────────────────
_RUN_LABELS = {
    # V10 configs (13 instruments, trend+carry)
    "v10_conservative": "V10 Conservative — Roll Reg+Carry, 20% vol target",
    "v10_balanced": "V10 Balanced — Roll Reg+Carry, 25% vol, Sh>0.20 filter",
    "v10_aggressive": "V10 Aggressive — Roll NoReg+Carry, 30% vol target",
    "v10_baseline": "V10 Baseline — Fixed EqWt Trend-only, 20% vol",
    "v10_lev_2x": "Leverage Tier: 2x — Conservative base, 20% vol",
    "v10_lev_4x": "Leverage Tier: 4x — Conservative base, 20% vol",
    "v10_lev_8x": "Leverage Tier: 8x — Conservative base, 20% vol",
    "v10_lev_10x": "Leverage Tier: 10x — Conservative base, 20% vol",
    # V11 configs (32 instruments, 7 signal families)
    "v11_expanded_trendcarry": "V11 Expanded — 32 inst, trend+carry only, 50% vol, 8x lev",
    "v11_all_moderate": "V11 Moderate — 32 inst, 7 signals, 35% vol, 6x lev (~11% CAGR)",
    "v11_all_growth": "V11 Growth — 32 inst, 7 signals, 50% vol, 8x lev (~15% CAGR)",
    "v11_all_aggressive": "V11 Aggressive — 32 inst, 7 signals, 55% vol, 9x lev (~16% CAGR)",
}
_V10_RUN_IDS = set(_RUN_LABELS.keys())

# ── Discover runs & build selector ────────────────────────────────────────────
output_dir = get_output_dir()
_all_runs = discover_runs(output_dir)
runs = [r for r in _all_runs if r in _V10_RUN_IDS]
if not runs:
    st.error("No V10 runs found in output/. Run scripts/save_v10_dashboard.py first.")
    st.stop()

# Load SPY buy-and-hold for overlay
_spy_eq = None
_spy_path = os.path.join(output_dir, "equity_curve_portfolio_adjusted_spy_buyhold.csv")
if os.path.exists(_spy_path):
    _spy_df = pd.read_csv(_spy_path, parse_dates=["timestamp"])
    _spy_df = _spy_df.set_index("timestamp")
    _spy_eq = _spy_df

def _run_label(rid: str) -> str:
    return _RUN_LABELS.get(rid, rid)

with st.sidebar:
    st.markdown(
        f'<div style="margin-top:20px;margin-bottom:4px;font-size:0.7rem;'
        f'color:{TEXT_SECONDARY};letter-spacing:0.05em;text-transform:uppercase;">'
        f'SELECT RUN</div>',
        unsafe_allow_html=True,
    )
    selected_run = st.selectbox(
        "Run",
        runs,
        format_func=_run_label,
        label_visibility="collapsed",
    )
    # Comparison toggle
    compare_enabled = st.checkbox("Compare with another run", value=False)
    compare_run = None
    if compare_enabled:
        other_runs = [r for r in runs if r != selected_run]
        if other_runs:
            compare_run = st.selectbox(
                "Compare to",
                other_runs,
                format_func=_run_label,
                label_visibility="collapsed",
            )

data = load_run(selected_run, output_dir)
compare_data = load_run(compare_run, output_dir) if compare_run else None
eq = data.get("equity_curve")
if eq is None or len(eq) == 0:
    st.error("No equity curve data.")
    st.stop()

returns = eq["returns"].dropna()
regime_report = data.get("regime_report")
stress_report = data.get("stress_report")
wfo_report = data.get("wfo_report")
mt_report = data.get("multiple_testing")
strat_data = data.get("strategy_data")
risk_ts = data.get("risk_timeseries")
regime_bd = data.get("regime_breakdown")

# ── Precompute common metrics ─────────────────────────────────────────────────
def _compute_core_metrics(rets):
    """Compute standard metrics from a return series."""
    n = len(rets)
    _ann_ret = ((1 + rets).prod() ** (252 / n)) - 1 if n > 10 else 0
    _ann_vol = rets.std() * np.sqrt(252) if n > 10 else 0
    _sharpe = _ann_ret / _ann_vol if _ann_vol > 0 else 0
    _down = rets[rets < 0]
    _dvol = _down.std() * np.sqrt(252) if len(_down) > 0 else 0
    _sortino = _ann_ret / _dvol if _dvol > 0 else 0
    _cum = (1 + rets).cumprod()
    _peak = _cum.cummax()
    _dd = (_cum - _peak) / _peak
    _max_dd = _dd.min()
    _calmar = _ann_ret / abs(_max_dd) if _max_dd != 0 else 0
    _total = _cum.iloc[-1] - 1
    _var95 = rets.quantile(0.05)
    _tail = rets[rets <= _var95]
    _cvar95 = _tail.mean() if len(_tail) > 0 else _var95
    return dict(
        ann_ret=_ann_ret, ann_vol=_ann_vol, sharpe=_sharpe, sortino=_sortino,
        cum=_cum, dd=_dd, max_dd=_max_dd, calmar=_calmar, total_ret=_total,
        var_95=_var95, cvar_95=_cvar95,
    )

ann_ret = ((1 + returns).prod() ** (252 / len(returns))) - 1 if len(returns) > 10 else 0
ann_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else 0
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
downside = returns[returns < 0]
downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
sortino = ann_ret / downside_vol if downside_vol > 0 else 0
cum = (1 + returns).cumprod()
peak = cum.cummax()
dd = (cum - peak) / peak
max_dd = dd.min()
calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
total_ret = cum.iloc[-1] - 1
var_95 = returns.quantile(0.05)
tail = returns[returns <= var_95]
cvar_95 = tail.mean() if len(tail) > 0 else var_95

# Comparison metrics
cmp_metrics = None
cmp_eq = None
cmp_returns = None
if compare_data:
    cmp_eq = compare_data.get("equity_curve")
    if cmp_eq is not None and len(cmp_eq) > 0:
        cmp_returns = cmp_eq["returns"].dropna()
        cmp_metrics = _compute_core_metrics(cmp_returns)


# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    # Show comparison table if comparing
    if cmp_metrics:
        section_header("Run Comparison")
        me = _compute_core_metrics(returns)
        cmp_label = _run_label(compare_run) if compare_run else "Compare"
        sel_label = _run_label(selected_run)
        comparison_html = f"""
        <table class="styled-table" style="max-width:700px;">
            <thead><tr>
                <th>Metric</th>
                <th style="text-align:right;">{sel_label}</th>
                <th style="text-align:right;">{cmp_label}</th>
                <th style="text-align:right;">Delta</th>
            </tr></thead>
            <tbody>
                <tr><td>Sharpe</td>
                    <td style="text-align:right;">{me['sharpe']:.3f}</td>
                    <td style="text-align:right;">{cmp_metrics['sharpe']:.3f}</td>
                    <td style="text-align:right;color:{ACCENT_TEAL if me['sharpe']>=cmp_metrics['sharpe'] else ACCENT_RED};">{me['sharpe']-cmp_metrics['sharpe']:+.3f}</td></tr>
                <tr><td>Ann. Return</td>
                    <td style="text-align:right;">{me['ann_ret']:.1%}</td>
                    <td style="text-align:right;">{cmp_metrics['ann_ret']:.1%}</td>
                    <td style="text-align:right;color:{ACCENT_TEAL if me['ann_ret']>=cmp_metrics['ann_ret'] else ACCENT_RED};">{me['ann_ret']-cmp_metrics['ann_ret']:+.1%}</td></tr>
                <tr><td>Max Drawdown</td>
                    <td style="text-align:right;">{me['max_dd']:.1%}</td>
                    <td style="text-align:right;">{cmp_metrics['max_dd']:.1%}</td>
                    <td style="text-align:right;color:{ACCENT_TEAL if me['max_dd']>=cmp_metrics['max_dd'] else ACCENT_RED};">{me['max_dd']-cmp_metrics['max_dd']:+.1%}</td></tr>
                <tr><td>Calmar</td>
                    <td style="text-align:right;">{me['calmar']:.2f}</td>
                    <td style="text-align:right;">{cmp_metrics['calmar']:.2f}</td>
                    <td style="text-align:right;color:{ACCENT_TEAL if me['calmar']>=cmp_metrics['calmar'] else ACCENT_RED};">{me['calmar']-cmp_metrics['calmar']:+.2f}</td></tr>
                <tr><td>Sortino</td>
                    <td style="text-align:right;">{me['sortino']:.2f}</td>
                    <td style="text-align:right;">{cmp_metrics['sortino']:.2f}</td>
                    <td style="text-align:right;color:{ACCENT_TEAL if me['sortino']>=cmp_metrics['sortino'] else ACCENT_RED};">{me['sortino']-cmp_metrics['sortino']:+.2f}</td></tr>
                <tr><td>Ann. Vol</td>
                    <td style="text-align:right;">{me['ann_vol']:.1%}</td>
                    <td style="text-align:right;">{cmp_metrics['ann_vol']:.1%}</td>
                    <td style="text-align:right;">{me['ann_vol']-cmp_metrics['ann_vol']:+.1%}</td></tr>
            </tbody>
        </table>
        """
        st.markdown(comparison_html, unsafe_allow_html=True)
        st.markdown("")

    render_kpi_row([
        ("Sharpe", f"{sharpe:.2f}", sharpe > 0),
        ("Ann. Return", f"{ann_ret:.1%}", ann_ret > 0),
        ("Max Drawdown", f"{max_dd:.1%}", None),
        ("Sortino", f"{sortino:.2f}", sortino > 0),
        ("Calmar", f"{calmar:.2f}", calmar > 0),
        ("Total Return", f"{total_ret:,.0%}", total_ret > 0),
    ])

    st.markdown("")

    col_eq, col_state = st.columns([5, 1])
    with col_eq:
        # Always build custom chart to include SPY overlay
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            mode="lines", name=_run_label(selected_run),
            line=dict(color=ACCENT_TEAL, width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        if cmp_eq is not None:
            fig.add_trace(go.Scatter(
                x=cmp_eq.index, y=cmp_eq["equity"],
                mode="lines", name=_run_label(compare_run),
                line=dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
            ))
        # SPY buy-and-hold overlay (always shown)
        if _spy_eq is not None:
            # Align SPY to same start date as selected run
            spy_start = eq.index[0]
            spy_slice = _spy_eq[_spy_eq.index >= spy_start]
            if len(spy_slice) > 0:
                spy_rebased = 1_000_000 * (1 + spy_slice["returns"]).cumprod()
                fig.add_trace(go.Scatter(
                    x=spy_rebased.index, y=spy_rebased.values,
                    mode="lines", name="S&P 500 Buy & Hold",
                    line=dict(color="#FFD700", width=1.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
                ))
        fig.update_layout(
            height=400,
            yaxis=dict(type="log", title="Equity ($)", tickprefix="$", separatethousands=True),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)",
        )
        chart(fig)

    with col_state:
        current_dd = dd.iloc[-1]
        dd_color = ACCENT_TEAL if current_dd > -0.05 else (ACCENT_RED if current_dd < -0.1 else COLOR_CARRY)
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<div class="kpi-label">Current DD</div>'
            f'<div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:700;color:{dd_color};">{current_dd:.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if regime_report:
            breakdowns = regime_report.get("regime_breakdowns", [])
            for dim in ["vol", "trend"]:
                dim_bds = [b for b in breakdowns if b.get("dimension") == dim]
                if dim_bds:
                    best = max(dim_bds, key=lambda x: x.get("n_observations", 0))
                    label = best.get("label", "?")
                    st.markdown(
                        f'<div style="margin-top:14px;">'
                        f'<div class="kpi-label">{dim}</div>'
                        f'{regime_pill(label)}</div>',
                        unsafe_allow_html=True,
                    )

        if risk_ts is not None and len(risk_ts) > 0:
            last = risk_ts.iloc[-1]
            for key, label in [("dd_multiplier", "Risk Scale"), ("margin_util", "Margin"), ("composite_risk", "Risk Score")]:
                if key in risk_ts.columns:
                    val = float(last.get(key, 0))
                    if key == "dd_multiplier":
                        c = ACCENT_TEAL if val > 0.7 else (ACCENT_RED if val < 0.3 else COLOR_CARRY)
                    else:
                        c = ACCENT_TEAL if val < 0.3 else (ACCENT_RED if val > 0.7 else COLOR_CARRY)
                    st.markdown(
                        f'<div style="margin-top:14px;">'
                        f'<div class="kpi-label">{label}</div>'
                        f'<div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:700;color:{c};">{val:.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # SPY comparison summary
    if _spy_eq is not None:
        spy_start = eq.index[0]
        spy_slice = _spy_eq[_spy_eq.index >= spy_start]
        if len(spy_slice) > 10:
            spy_rets = spy_slice["returns"].dropna()
            spy_m = _compute_core_metrics(spy_rets)
            section_header("vs S&P 500 Buy & Hold")
            me = _compute_core_metrics(returns)
            vs_html = f"""
            <table class="styled-table" style="max-width:700px;">
                <thead><tr>
                    <th>Metric</th>
                    <th style="text-align:right;">Strategy</th>
                    <th style="text-align:right;">S&P 500</th>
                    <th style="text-align:right;">Alpha</th>
                </tr></thead>
                <tbody>
                    <tr><td>Sharpe</td>
                        <td style="text-align:right;">{me['sharpe']:.3f}</td>
                        <td style="text-align:right;">{spy_m['sharpe']:.3f}</td>
                        <td style="text-align:right;color:{ACCENT_TEAL if me['sharpe']>=spy_m['sharpe'] else ACCENT_RED};">{me['sharpe']-spy_m['sharpe']:+.3f}</td></tr>
                    <tr><td>Ann. Return</td>
                        <td style="text-align:right;">{me['ann_ret']:.1%}</td>
                        <td style="text-align:right;">{spy_m['ann_ret']:.1%}</td>
                        <td style="text-align:right;color:{ACCENT_TEAL if me['ann_ret']>=spy_m['ann_ret'] else ACCENT_RED};">{me['ann_ret']-spy_m['ann_ret']:+.1%}</td></tr>
                    <tr><td>Max Drawdown</td>
                        <td style="text-align:right;">{me['max_dd']:.1%}</td>
                        <td style="text-align:right;">{spy_m['max_dd']:.1%}</td>
                        <td style="text-align:right;color:{ACCENT_TEAL if me['max_dd']>=spy_m['max_dd'] else ACCENT_RED};">{me['max_dd']-spy_m['max_dd']:+.1%}</td></tr>
                    <tr><td>Calmar</td>
                        <td style="text-align:right;">{me['calmar']:.2f}</td>
                        <td style="text-align:right;">{spy_m['calmar']:.2f}</td>
                        <td style="text-align:right;color:{ACCENT_TEAL if me['calmar']>=spy_m['calmar'] else ACCENT_RED};">{me['calmar']-spy_m['calmar']:+.2f}</td></tr>
                    <tr><td>Total Return</td>
                        <td style="text-align:right;">{me['total_ret']:.0%}</td>
                        <td style="text-align:right;">{spy_m['total_ret']:.0%}</td>
                        <td style="text-align:right;color:{ACCENT_TEAL if me['total_ret']>=spy_m['total_ret'] else ACCENT_RED};">{me['total_ret']-spy_m['total_ret']:+.0%}</td></tr>
                </tbody>
            </table>
            """
            st.markdown(vs_html, unsafe_allow_html=True)
            st.markdown("")

    section_header("Key Performance Metrics")
    ts_metrics = compute_tearsheet(returns, rf_annual=0.0)
    tearsheet_table(ts_metrics)


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Performance":
    section_header("Rolling Sharpe Ratio")
    # Build rolling Sharpe chart with SPY overlay
    fig = go.Figure()
    w = 252
    traces = [(returns, _run_label(selected_run), ACCENT_TEAL, None)]
    if cmp_returns is not None:
        traces.append((cmp_returns, _run_label(compare_run), ACCENT_BLUE, "dot"))
    # Add SPY rolling Sharpe
    if _spy_eq is not None:
        spy_start = returns.index[0]
        spy_rets = _spy_eq["returns"].dropna()
        spy_rets = spy_rets[spy_rets.index >= spy_start]
        if len(spy_rets) >= w:
            traces.append((spy_rets, "S&P 500", "#FFD700", "dash"))
    for rets_i, label_i, color_i, dash_i in traces:
        if len(rets_i) >= w:
            rm = rets_i.rolling(w).mean()
            rs = rets_i.rolling(w).std()
            rsr = (rm / rs * np.sqrt(252)).dropna()
            fig.add_trace(go.Scatter(
                x=rsr.index, y=rsr.values,
                mode="lines", name=f"{label_i} (252d)",
                line=dict(color=color_i, width=1.2, dash=dash_i),
            ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, opacity=0.4)
    fig.update_layout(height=350, yaxis_title="Sharpe Ratio",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)")
    chart(fig)

    section_header("Drawdown")
    # Build drawdown chart with SPY overlay
    fig = go.Figure()
    dd_traces = [(returns, _run_label(selected_run), ACCENT_RED, None)]
    if cmp_returns is not None:
        dd_traces.append((cmp_returns, _run_label(compare_run), ACCENT_BLUE, "dot"))
    # Add SPY drawdown
    if _spy_eq is not None:
        spy_start = returns.index[0]
        spy_rets = _spy_eq["returns"].dropna()
        spy_rets = spy_rets[spy_rets.index >= spy_start]
        if len(spy_rets) > 0:
            dd_traces.append((spy_rets, "S&P 500", "#FFD700", "dash"))
    for rets_i, label_i, color_i, dash_i in dd_traces:
        c = (1 + rets_i).cumprod()
        p = c.cummax()
        d = (c - p) / p
        fig.add_trace(go.Scatter(
            x=d.index, y=d.values, mode="lines", name=label_i,
            line=dict(color=color_i, width=1.2, dash=dash_i),
        ))
    for level in [-0.05, -0.08, -0.10, -0.12, -0.20]:
        fig.add_hline(y=level, line_dash="dot", line_color=TEXT_SECONDARY, opacity=0.4,
                      annotation_text=f"{level:.0%}", annotation_position="bottom right",
                      annotation_font_color=TEXT_SECONDARY, annotation_font_size=9)
    fig.update_layout(height=300, yaxis_title="Drawdown", yaxis_tickformat=".0%",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)")
    chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        section_header("Return Distribution")
        chart(return_distribution_chart(returns))
    with c2:
        section_header("Annual Returns vs S&P 500")
        # Custom annual bar chart with SPY comparison
        annual_rets = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual_rets.index.year, y=annual_rets.values,
            name="Strategy", marker_color=ACCENT_TEAL, opacity=0.85,
            text=[f"{v:.0%}" for v in annual_rets.values],
            textposition="outside", textfont=dict(size=9),
        ))
        if _spy_eq is not None:
            spy_start = returns.index[0]
            spy_rets = _spy_eq["returns"].dropna()
            spy_rets = spy_rets[spy_rets.index >= spy_start]
            if len(spy_rets) > 0:
                spy_annual = spy_rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
                fig.add_trace(go.Bar(
                    x=spy_annual.index.year, y=spy_annual.values,
                    name="S&P 500", marker_color="#FFD700", opacity=0.6,
                    text=[f"{v:.0%}" for v in spy_annual.values],
                    textposition="outside", textfont=dict(size=9),
                ))
        fig.add_hline(y=0, line_color=TEXT_SECONDARY, opacity=0.4)
        fig.update_layout(
            height=350, barmode="group", yaxis_tickformat=".0%",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        chart(fig)

    section_header("Monthly Returns")
    chart(monthly_heatmap(returns, height=max(350, len(set(returns.index.year)) * 28 + 100)))


# ═════════════════════════════════════════════════════════════════════════════
# RISK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Risk":
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(gauge_html(abs(float(dd.iloc[-1])), "Current Drawdown"), unsafe_allow_html=True)
    with g2:
        mu = float(risk_ts["margin_util"].iloc[-1]) if risk_ts is not None and "margin_util" in risk_ts.columns else 0
        st.markdown(gauge_html(mu, "Margin Utilization"), unsafe_allow_html=True)
    with g3:
        cr = float(risk_ts["composite_risk"].iloc[-1]) if risk_ts is not None and "composite_risk" in risk_ts.columns else 0
        st.markdown(gauge_html(cr, "Composite Risk"), unsafe_allow_html=True)

    st.markdown("")

    if risk_ts is not None and "dd_multiplier" in risk_ts.columns:
        section_header("Drawdown Ladder Multiplier")
        chart(dd_ladder_timeseries(risk_ts["dd_multiplier"]))

    if risk_ts is not None:
        c1, c2 = st.columns(2)
        with c1:
            section_header("Margin Utilization")
            chart(margin_timeseries(risk_ts))
        with c2:
            section_header("Gross Leverage")
            chart(leverage_timeseries(risk_ts))

    section_header("Equity + Composite Risk")
    chart(composite_risk_overlay(eq, risk_ts))


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Strategies":
    if strat_data is None or len(strat_data) == 0:
        st.info("No strategy data. Re-run pipeline with latest code.")
    else:
        # For V11 runs with many instruments, aggregate by asset class
        is_v11 = _is_v11_run(selected_run)
        display_data = _aggregate_by_asset_class(strat_data) if is_v11 else strat_data
        color_map = ASSET_CLASS_COLORS if is_v11 else STRATEGY_COLORS

        if is_v11:
            n_instruments = len([c for c in strat_data.columns if c.endswith("_signal")])
            st.caption(f"Showing asset class aggregation of {n_instruments} instruments")

        section_header("Strategy Performance")
        strategy_kpi_table(display_data)
        st.markdown("")

        signal_cols = [c for c in display_data.columns if c.endswith("_signal")]
        if signal_cols:
            section_header("Signals")
            n_sigs = len(signal_cols)
            fig = make_subplots(
                rows=n_sigs, cols=1, shared_xaxes=True,
                vertical_spacing=0.04,
                subplot_titles=[c.replace("_signal", "") for c in signal_cols],
            )
            for i, col in enumerate(signal_cols):
                name = col.replace("_signal", "")
                color = color_map.get(name, TEXT_SECONDARY)
                fig.add_trace(go.Scatter(
                    x=display_data.index, y=display_data[col],
                    mode="lines", name=name,
                    line=dict(color=color, width=1),
                ), row=i + 1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, opacity=0.2, row=i + 1, col=1)
            fig.update_layout(height=180 * n_sigs, showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)")
            for ann in fig.layout.annotations:
                ann.font.size = 10
                ann.font.color = TEXT_SECONDARY
            chart(fig)

        c1, c2 = st.columns([2, 1])
        with c1:
            section_header("Return Contribution")
            chart(strategy_contribution_area(display_data))
        with c2:
            section_header("Vol Budget Allocation")
            return_cols = [c for c in display_data.columns if c.endswith("_return")]
            if return_cols:
                vols = {}
                for col in return_cols:
                    name = col.replace("_return", "")
                    vol = display_data[col].dropna().std() * np.sqrt(252)
                    vols[name] = 1 / vol if vol > 0 else 0
                total = sum(vols.values())
                if total > 0:
                    labels = list(vols.keys())
                    values = [v / total for v in vols.values()]
                    colors = [color_map.get(n, TEXT_SECONDARY) for n in labels]
                    fig = go.Figure(data=go.Pie(
                        labels=labels, values=values,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                        textfont=dict(size=10),
                        hole=0.45,
                    ))
                    fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)")
                    chart(fig)

        section_header("Signal Correlation")
        chart(correlation_heatmap(display_data))


# ═════════════════════════════════════════════════════════════════════════════
# REGIMES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Regimes":
    if regime_report is None:
        st.info("No regime report for this run.")
    else:
        section_header("Regime Distribution")
        chart(regime_timeline(regime_bd, eq))

        breakdowns = regime_report.get("regime_breakdowns", [])
        for dim in ["vol", "trend", "corr"]:
            dim_data = [b for b in breakdowns if b.get("dimension") == dim]
            if not dim_data:
                continue

            section_header(f"{dim.title()} Regime Performance")
            labels = [b["label"] for b in dim_data]
            sharpes_r = [b["metrics"]["sharpe_ratio"] for b in dim_data if b.get("metrics")]
            returns_r = [b["metrics"]["annualized_return"] for b in dim_data if b.get("metrics")]
            max_dds_r = [abs(b["metrics"]["max_drawdown"]) for b in dim_data if b.get("metrics")]
            colors_r = [REGIME_COLORS.get(l, TEXT_SECONDARY) for l in labels]

            c1, c2, c3 = st.columns(3)
            for col_w, ydata, title, fmt in [
                (c1, sharpes_r, "Sharpe", ""),
                (c2, returns_r, "Return", ".0%"),
                (c3, max_dds_r, "|Max DD|", ".0%"),
            ]:
                with col_w:
                    fig = go.Figure(data=go.Bar(
                        x=labels, y=ydata, marker_color=colors_r,
                        text=[f"{v:.3f}" if not fmt else f"{v:{fmt}}" for v in ydata],
                        textposition="outside", textfont=dict(size=9),
                    ))
                    fig.update_layout(height=220, title=title, yaxis_tickformat=fmt or "",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,1)")
                    chart(fig)

        section_header("Regime Detail")
        regime_metrics_table(regime_report)

        if eq is not None:
            section_header("Risk Overlay")
            chart(composite_risk_overlay(eq, risk_ts, height=320))


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Validation":
    if stress_report:
        all_passed = stress_report.get("all_passed", False)
        pass_fail_banner(all_passed, f"STRESS TESTS: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
        st.markdown("")

        section_header("Scenario Comparison")
        scenarios = stress_report.get("scenarios", [])
        stress_scenario_table(scenarios)
        st.markdown("")

        section_header("Sharpe Degradation")
        chart(stress_sharpe_comparison(scenarios))
    else:
        st.info("No stress test data.")

    st.markdown("")

    if wfo_report:
        section_header("Walk-Forward Optimization")
        folds = wfo_report.get("folds", [])
        if folds:
            oos = [f["test_metric"] for f in folds]
            iss = [f["train_metric"] for f in folds]
            st.markdown(
                f'<p style="font-family:JetBrains Mono;font-size:0.8rem;color:{TEXT_PRIMARY};">'
                f'{len(folds)} folds &nbsp;&middot;&nbsp; OOS Sharpe: {np.mean(oos):.3f} &plusmn; {np.std(oos):.3f}'
                f' &nbsp;&middot;&nbsp; IS Sharpe: {np.mean(iss):.3f} &plusmn; {np.std(iss):.3f}</p>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                chart(wfo_sharpe_boxplot(wfo_report))
            with c2:
                chart(wfo_is_oos_scatter(wfo_report))
    else:
        st.info("No WFO data.")

    if mt_report:
        st.markdown("")
        section_header("Multiple Testing Diagnostics")
        multiple_testing_cards(mt_report)
