"""Reusable Plotly chart factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.lib.theme import (
    ACCENT_TEAL, ACCENT_RED, COLOR_TREND, COLOR_CARRY, COLOR_VOL_BREAKOUT,
    BG_PAGE, BG_CARD, BG_CHART, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    GRID_COLOR, STRATEGY_COLORS, REGIME_COLORS,
)

# Standard chart layout kwargs (same approach as backtester)
_CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,1)",
)


def equity_curve_chart(
    equity_df: pd.DataFrame,
    log_scale: bool = True,
    height: int = 420,
) -> go.Figure:
    """Full-width equity curve with gradient fill."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df["equity"],
        mode="lines",
        name="Portfolio",
        line=dict(color=ACCENT_TEAL, width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 176, 0.06)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    yaxis_type = "log" if log_scale else "linear"
    fig.update_layout(
        height=height,
        yaxis=dict(type=yaxis_type, title="Equity ($)", tickprefix="$", separatethousands=True),
        xaxis=dict(title=""),
        showlegend=False,
        **_CHART_BG,
    )
    return fig


def rolling_sharpe_chart(
    returns: pd.Series,
    windows: list[int] | None = None,
    height: int = 350,
) -> go.Figure:
    """Multi-window rolling Sharpe."""
    if windows is None:
        windows = [126, 252, 756]

    colors = [ACCENT_TEAL, COLOR_TREND, COLOR_CARRY]
    fig = go.Figure()
    for i, w in enumerate(windows):
        if len(returns) < w:
            continue
        rolling_mean = returns.rolling(w).mean()
        rolling_std = returns.rolling(w).std()
        rolling_sr = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
        fig.add_trace(go.Scatter(
            x=rolling_sr.index,
            y=rolling_sr.values,
            mode="lines",
            name=f"{w}d",
            line=dict(color=colors[i % len(colors)], width=1.2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, opacity=0.4)
    fig.update_layout(height=height, yaxis_title="Sharpe Ratio", **_CHART_BG)
    return fig


def drawdown_chart(
    returns: pd.Series,
    ladder_levels: list[float] | None = None,
    height: int = 300,
) -> go.Figure:
    """Drawdown chart with ladder lines."""
    if ladder_levels is None:
        ladder_levels = [-0.05, -0.08, -0.10, -0.12]

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        line=dict(color=ACCENT_RED, width=1.2),
        fill="tozeroy",
        fillcolor="rgba(255, 71, 87, 0.12)",
    ))

    for level in ladder_levels:
        fig.add_hline(
            y=level, line_dash="dot",
            line_color=TEXT_SECONDARY, opacity=0.5,
            annotation_text=f"{level:.0%}",
            annotation_position="bottom right",
            annotation_font_color=TEXT_SECONDARY,
            annotation_font_size=9,
        )

    fig.update_layout(height=height, yaxis_title="Drawdown", yaxis_tickformat=".0%", **_CHART_BG)
    return fig


def return_distribution_chart(
    returns: pd.Series,
    height: int = 300,
) -> go.Figure:
    """Histogram with VaR/CVaR markers."""
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=80,
        marker_color=ACCENT_TEAL,
        opacity=0.7,
        name="Returns",
    ))

    fig.add_vline(x=var_95, line_dash="dash", line_color=ACCENT_RED,
                  annotation_text=f"VaR 95: {var_95:.2%}",
                  annotation_font_color=ACCENT_RED, annotation_font_size=9)
    fig.add_vline(x=cvar_95, line_dash="dot", line_color=COLOR_CARRY,
                  annotation_text=f"CVaR 95: {cvar_95:.2%}",
                  annotation_font_color=COLOR_CARRY, annotation_font_size=9)

    fig.update_layout(
        height=height, xaxis_title="Daily Return",
        yaxis_title="Frequency", showlegend=False,
        xaxis_tickformat=".1%",
        **_CHART_BG,
    )
    return fig


def monthly_heatmap(
    returns: pd.Series,
    height: int = 400,
) -> go.Figure:
    """Monthly return heatmap."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })

    pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, ACCENT_RED],
            [0.5, "#0e1117"],
            [1.0, ACCENT_TEAL],
        ],
        zmid=0,
        text=[[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title=dict(text="Return", font=dict(size=10, color=TEXT_SECONDARY)),
            tickformat=".0%",
            tickfont=dict(size=9, color=TEXT_SECONDARY),
        ),
    ))

    fig.update_layout(height=height, yaxis=dict(autorange="reversed"), **_CHART_BG)
    return fig


def annual_bar_chart(
    returns: pd.Series,
    height: int = 300,
) -> go.Figure:
    """Annual return bar chart."""
    annual = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    colors = [ACCENT_TEAL if v >= 0 else ACCENT_RED for v in annual.values]

    fig = go.Figure(data=go.Bar(
        x=[str(d.year) for d in annual.index],
        y=annual.values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in annual.values],
        textposition="outside",
        textfont=dict(size=9, color=TEXT_SECONDARY),
    ))
    fig.update_layout(height=height, yaxis_title="Return", yaxis_tickformat=".0%", **_CHART_BG)
    return fig


def dd_ladder_timeseries(
    dd_multipliers: pd.Series,
    height: int = 280,
) -> go.Figure:
    """DD ladder multiplier step chart."""
    fig = go.Figure()

    # Color bands
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor=ACCENT_TEAL, opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.50, y1=0.75, fillcolor=COLOR_CARRY, opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.50, fillcolor=ACCENT_RED, opacity=0.06, line_width=0)

    fig.add_trace(go.Scatter(
        x=dd_multipliers.index, y=dd_multipliers.values,
        mode="lines", name="Risk Multiplier",
        line=dict(color=ACCENT_TEAL, width=1.5, shape="hv"),
    ))

    fig.update_layout(
        height=height, yaxis_title="Risk Multiplier",
        yaxis=dict(range=[-0.05, 1.05]),
        **_CHART_BG,
    )
    return fig


def gauge_indicator(value: float, label: str, max_val: float = 1.0) -> go.Figure:
    """Gauge-style indicator chart."""
    color = ACCENT_TEAL if value / max_val < 0.5 else (COLOR_CARRY if value / max_val < 0.75 else ACCENT_RED)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(suffix="%", font=dict(size=28, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, max_val * 100], tickfont=dict(size=9, color=TEXT_SECONDARY)),
            bar=dict(color=color, thickness=0.7),
            bgcolor=BG_CARD,
            borderwidth=0,
            steps=[
                dict(range=[0, max_val * 50], color="rgba(0,212,170,0.08)"),
                dict(range=[max_val * 50, max_val * 75], color="rgba(255,179,71,0.08)"),
                dict(range=[max_val * 75, max_val * 100], color="rgba(255,71,87,0.08)"),
            ],
        ),
        title=dict(text=label, font=dict(size=11, color=TEXT_SECONDARY)),
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), **_CHART_BG)
    return fig


def strategy_contribution_area(
    strategy_data: pd.DataFrame,
    height: int = 350,
) -> go.Figure:
    """Stacked area of strategy return contributions."""
    fig = go.Figure()
    return_cols = [c for c in strategy_data.columns if c.endswith("_return")]

    for col in return_cols:
        name = col.replace("_return", "")
        cum = (1 + strategy_data[col].fillna(0)).cumprod() - 1
        color = STRATEGY_COLORS.get(name, TEXT_SECONDARY)
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=name,
            line=dict(color=color, width=1.2),
            stackgroup="one",
        ))

    fig.update_layout(height=height, yaxis_title="Cumulative Return", yaxis_tickformat=".0%", **_CHART_BG)
    return fig


def correlation_heatmap(
    strategy_data: pd.DataFrame,
    height: int = 350,
) -> go.Figure:
    """Signal correlation matrix."""
    signal_cols = [c for c in strategy_data.columns if c.endswith("_signal")]
    if not signal_cols:
        return go.Figure()

    corr = strategy_data[signal_cols].corr()
    labels = [c.replace("_signal", "") for c in signal_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[[0, ACCENT_RED], [0.5, "#0e1117"], [1, ACCENT_TEAL]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig.update_layout(height=height, **_CHART_BG)
    return fig


def regime_timeline(
    regime_breakdown: pd.DataFrame,
    equity_df: pd.DataFrame,
    height: int = 200,
) -> go.Figure:
    """Colored horizontal regime blocks on timeline."""
    fig = go.Figure()

    if regime_breakdown is not None and "dimension" in regime_breakdown.columns:
        vol_rows = regime_breakdown[regime_breakdown["dimension"] == "vol"]
        for _, row in vol_rows.iterrows():
            label = row.get("label", "unknown")
            n_obs = row.get("n_obs", row.get("n_observations", 0))
            color = REGIME_COLORS.get(label, TEXT_SECONDARY)
            fig.add_trace(go.Bar(
                x=[n_obs], y=["Vol Regime"],
                orientation="h",
                name=f"{label} ({n_obs}d)",
                marker_color=color,
                text=f"{label}: {n_obs}d",
                textposition="inside",
                textfont=dict(size=10),
            ))

        trend_rows = regime_breakdown[regime_breakdown["dimension"] == "trend"]
        for _, row in trend_rows.iterrows():
            label = row.get("label", "unknown")
            n_obs = row.get("n_obs", row.get("n_observations", 0))
            color = REGIME_COLORS.get(label, TEXT_SECONDARY)
            fig.add_trace(go.Bar(
                x=[n_obs], y=["Trend Regime"],
                orientation="h",
                name=f"{label} ({n_obs}d)",
                marker_color=color,
                text=f"{label}: {n_obs}d",
                textposition="inside",
                textfont=dict(size=10),
            ))

    fig.update_layout(
        barmode="stack", height=height,
        xaxis_title="Trading Days", showlegend=True,
        **_CHART_BG,
    )
    return fig


def stress_sharpe_comparison(
    scenarios: list[dict],
    height: int = 350,
) -> go.Figure:
    """Grouped bar: base vs stressed Sharpe per scenario."""
    names = [s["name"] for s in scenarios]
    base = [s["base_sharpe"] for s in scenarios]
    stressed = [s["stressed_sharpe"] for s in scenarios]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=names, y=base, marker_color=ACCENT_TEAL))
    fig.add_trace(go.Bar(name="Stressed", x=names, y=stressed, marker_color=ACCENT_RED))

    fig.update_layout(
        barmode="group", height=height,
        yaxis_title="Sharpe Ratio",
        **_CHART_BG,
    )
    return fig


def wfo_sharpe_boxplot(
    wfo_report: dict,
    height: int = 300,
) -> go.Figure:
    """Box plot of IS vs OOS Sharpe across folds."""
    folds = wfo_report.get("folds", [])
    is_sharpes = [f["train_metric"] for f in folds]
    oos_sharpes = [f["test_metric"] for f in folds]

    fig = go.Figure()
    fig.add_trace(go.Box(y=is_sharpes, name="In-Sample", marker_color=COLOR_TREND, boxmean=True))
    fig.add_trace(go.Box(y=oos_sharpes, name="Out-of-Sample", marker_color=ACCENT_TEAL, boxmean=True))
    fig.update_layout(height=height, yaxis_title="Sharpe Ratio", showlegend=False, **_CHART_BG)
    return fig


def wfo_is_oos_scatter(
    wfo_report: dict,
    height: int = 300,
) -> go.Figure:
    """IS vs OOS Sharpe scatter with 45-degree line."""
    folds = wfo_report.get("folds", [])
    is_sharpes = [f["train_metric"] for f in folds]
    oos_sharpes = [f["test_metric"] for f in folds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=is_sharpes, y=oos_sharpes,
        mode="markers",
        marker=dict(color=ACCENT_TEAL, size=6, opacity=0.7),
        name="Folds",
    ))

    all_vals = is_sharpes + oos_sharpes
    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(dash="dash", color=TEXT_SECONDARY, width=1),
            showlegend=False,
        ))

    fig.update_layout(
        height=height,
        xaxis_title="In-Sample Sharpe",
        yaxis_title="Out-of-Sample Sharpe",
        **_CHART_BG,
    )
    return fig


def leverage_timeseries(
    risk_ts: pd.DataFrame,
    height: int = 280,
) -> go.Figure:
    """Gross leverage over time."""
    fig = go.Figure()
    lev_col = "leverage" if "leverage" in risk_ts.columns else ("gross_leverage" if "gross_leverage" in risk_ts.columns else None)
    if lev_col is not None:
        fig.add_trace(go.Scatter(
            x=risk_ts.index, y=risk_ts[lev_col],
            mode="lines", name="Gross Leverage",
            line=dict(color=COLOR_TREND, width=1.2),
        ))
    fig.update_layout(height=height, yaxis_title="Leverage (x)", **_CHART_BG)
    return fig


def margin_timeseries(
    risk_ts: pd.DataFrame,
    stress_mult: float = 1.5,
    height: int = 280,
) -> go.Figure:
    """Margin utilization with stress overlay."""
    fig = go.Figure()
    if "margin_util" in risk_ts.columns:
        fig.add_trace(go.Scatter(
            x=risk_ts.index, y=risk_ts["margin_util"],
            mode="lines", name="Margin Util",
            line=dict(color=ACCENT_TEAL, width=1.2),
        ))
        fig.add_trace(go.Scatter(
            x=risk_ts.index, y=risk_ts["margin_util"] * stress_mult,
            mode="lines", name=f"Stress ({stress_mult}x)",
            line=dict(color=ACCENT_RED, width=1, dash="dot"),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=ACCENT_RED, opacity=0.6,
                  annotation_text="Limit", annotation_font_color=ACCENT_RED)
    fig.update_layout(height=height, yaxis_title="Utilization", yaxis_tickformat=".0%", **_CHART_BG)
    return fig


def composite_risk_overlay(
    equity_df: pd.DataFrame,
    risk_ts: pd.DataFrame,
    height: int = 350,
) -> go.Figure:
    """Equity curve with composite risk score on dual y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        mode="lines", name="Equity",
        line=dict(color=ACCENT_TEAL, width=1.2),
    ), secondary_y=False)

    if risk_ts is not None and "composite_risk" in risk_ts.columns:
        fig.add_trace(go.Scatter(
            x=risk_ts.index, y=risk_ts["composite_risk"],
            mode="lines", name="Composite Risk",
            line=dict(color=ACCENT_RED, width=1, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(255,71,87,0.06)",
        ), secondary_y=True)

    fig.update_yaxes(title_text="Equity ($)", secondary_y=False)
    fig.update_yaxes(title_text="Risk Score", secondary_y=True, range=[0, 1])
    fig.update_layout(height=height, **_CHART_BG)
    return fig
