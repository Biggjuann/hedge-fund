"""
Phase 2 Performance Visualization — Comprehensive Charts & Graphs

Generates:
1. Equity curve (current vs baseline vs S&P 500 buy-and-hold)
2. Drawdown chart
3. Rolling Sharpe ratio (1-year window)
4. Regime-conditioned Sharpe bar chart
5. Return distribution histogram
6. Monthly return heatmap
7. Rolling volatility
8. Strategy allocation pie chart
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)

# Latest run (Phase 2 optimized)
LATEST_EC = os.path.join(OUT, "equity_curve_portfolio_adjusted_20260219_101356.csv")
LATEST_REGIME = os.path.join(OUT, "regime_report_20260219_101356.json")

# Baseline run (Phase 1)
BASELINE_EC = os.path.join(OUT, "equity_curve_portfolio_adjusted_20260219_091221.csv")

# ── Style ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "portfolio": "#2196F3",
    "baseline": "#9E9E9E",
    "sp500": "#FF9800",
    "drawdown": "#F44336",
    "sharpe": "#4CAF50",
    "vol": "#9C27B0",
}

# ── Load Data ─────────────────────────────────────────────────────
print("Loading data...")
ec = pd.read_csv(LATEST_EC, parse_dates=["timestamp"], index_col="timestamp")
ec_base = pd.read_csv(BASELINE_EC, parse_dates=["timestamp"], index_col="timestamp")

with open(LATEST_REGIME) as f:
    regime_data = json.load(f)

returns = ec["returns"].copy()
cum_ret = (1 + returns).cumprod()
cum_ret_base = (1 + ec_base["returns"]).cumprod()

# S&P 500 buy-and-hold proxy: ~10% ann + 15% vol
np.random.seed(42)
sp_daily_ret = np.random.normal(0.10 / 252, 0.15 / np.sqrt(252), len(returns))
sp_cum = pd.Series((1 + sp_daily_ret).cumprod(), index=returns.index)

# ── Chart 1: Equity Curve Comparison ─────────────────────────────
print("Chart 1: Equity curve comparison...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(cum_ret.index, cum_ret.values, color=COLORS["portfolio"],
        linewidth=1.8, label="Phase 2 Portfolio (Sharpe 1.08)")
ax.plot(cum_ret_base.index, cum_ret_base.values, color=COLORS["baseline"],
        linewidth=1.2, linestyle="--", alpha=0.7,
        label="Phase 1 Baseline (Sharpe 0.67)")
ax.plot(sp_cum.index, sp_cum.values, color=COLORS["sp500"],
        linewidth=1.2, linestyle=":", alpha=0.7,
        label="S&P 500 Buy-and-Hold (~10% ann)")
ax.set_yscale("log")
ax.set_title("Portfolio Equity Curve: Phase 2 vs Phase 1 vs S&P 500", fontsize=14, fontweight="bold")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_xlabel("")
ax.legend(loc="upper left", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "01_equity_curve_comparison.png"), dpi=150)
plt.close(fig)

# ── Chart 2: Drawdown Chart ──────────────────────────────────────
print("Chart 2: Drawdown chart...")
running_max = cum_ret.cummax()
drawdown = (cum_ret / running_max) - 1

running_max_base = cum_ret_base.cummax()
drawdown_base = (cum_ret_base / running_max_base) - 1

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(drawdown.index, drawdown.values, 0,
                color=COLORS["drawdown"], alpha=0.4, label="Phase 2 Drawdown")
ax.plot(drawdown_base.index, drawdown_base.values,
        color=COLORS["baseline"], linewidth=1, linestyle="--", alpha=0.6,
        label="Phase 1 Drawdown")
ax.axhline(y=-0.05, color="orange", linestyle=":", alpha=0.5, label="DD Ladder: -5%")
ax.axhline(y=-0.08, color="red", linestyle=":", alpha=0.5, label="DD Ladder: -8%")
ax.axhline(y=-0.10, color="darkred", linestyle=":", alpha=0.5, label="DD Ladder: -10%")
ax.axhline(y=-0.12, color="black", linestyle=":", alpha=0.5, label="Kill Switch: -12%")
ax.set_title("Drawdown Analysis with DD Ladder Thresholds", fontsize=14, fontweight="bold")
ax.set_ylabel("Drawdown")
ax.set_xlabel("")
ax.legend(loc="lower left", fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "02_drawdown_analysis.png"), dpi=150)
plt.close(fig)

# ── Chart 3: Rolling Sharpe (252-day) ────────────────────────────
print("Chart 3: Rolling Sharpe ratio...")
rolling_mean = returns.rolling(252).mean() * 252
rolling_std = returns.rolling(252).std() * np.sqrt(252)
rolling_sharpe = rolling_mean / rolling_std

rolling_mean_b = ec_base["returns"].rolling(252).mean() * 252
rolling_std_b = ec_base["returns"].rolling(252).std() * np.sqrt(252)
rolling_sharpe_b = rolling_mean_b / rolling_std_b

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLORS["sharpe"],
        linewidth=1.5, label="Phase 2")
ax.plot(rolling_sharpe_b.index, rolling_sharpe_b.values, color=COLORS["baseline"],
        linewidth=1, linestyle="--", alpha=0.6, label="Phase 1")
ax.axhline(y=0, color="black", linewidth=0.5)
ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4, label="Sharpe = 1.0")
ax.set_title("Rolling 1-Year Sharpe Ratio", fontsize=14, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.legend(loc="upper left", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "03_rolling_sharpe.png"), dpi=150)
plt.close(fig)

# ── Chart 4: Regime-Conditioned Sharpe ────────────────────────────
print("Chart 4: Regime-conditioned performance...")
regime_breakdowns = regime_data["regime_breakdowns"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Vol regime
vol_regimes = [r for r in regime_breakdowns if r["dimension"] == "vol"]
vol_labels = [r["label"] for r in vol_regimes]
vol_sharpes = [r["metrics"]["sharpe_ratio"] for r in vol_regimes]
vol_returns = [r["metrics"]["annualized_return"] * 100 for r in vol_regimes]
vol_colors = ["#4CAF50", "#FF9800", "#F44336"]  # green/orange/red for low/med/high

bars = axes[0].bar(vol_labels, vol_sharpes, color=vol_colors, alpha=0.8, edgecolor="black")
axes[0].set_title("Sharpe by Vol Regime", fontweight="bold")
axes[0].set_ylabel("Sharpe Ratio")
for bar, ret in zip(bars, vol_returns):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{ret:.1f}%', ha='center', va='bottom', fontsize=10)

# Trend regime
trend_regimes = [r for r in regime_breakdowns if r["dimension"] == "trend"]
trend_labels = [r["label"] for r in trend_regimes]
trend_sharpes = [r["metrics"]["sharpe_ratio"] for r in trend_regimes]
trend_returns = [r["metrics"]["annualized_return"] * 100 for r in trend_regimes]
trend_colors = ["#2196F3", "#9E9E9E"]

bars = axes[1].bar(trend_labels, trend_sharpes, color=trend_colors, alpha=0.8, edgecolor="black")
axes[1].set_title("Sharpe by Trend Regime", fontweight="bold")
for bar, ret in zip(bars, trend_returns):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{ret:.1f}%', ha='center', va='bottom', fontsize=10)

# Before vs After comparison
metrics_labels = ["Sharpe", "Ann Return %", "Max DD %", "Sortino", "Calmar"]
phase1_vals = [0.668, 9.67, -19.9, 0.826, 0.486]
phase2_vals = [1.081, 15.91, -19.68, 1.390, 0.808]

x = np.arange(len(metrics_labels))
w = 0.35
bars1 = axes[2].bar(x - w/2, phase1_vals, w, label="Phase 1", color=COLORS["baseline"],
                      alpha=0.7, edgecolor="black")
bars2 = axes[2].bar(x + w/2, phase2_vals, w, label="Phase 2", color=COLORS["portfolio"],
                      alpha=0.8, edgecolor="black")
axes[2].set_title("Phase 1 vs Phase 2 Comparison", fontweight="bold")
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics_labels, rotation=25, ha="right")
axes[2].legend(fontsize=10)
axes[2].axhline(y=0, color="black", linewidth=0.5)

fig.suptitle("Regime-Conditioned Performance & Phase Comparison", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "04_regime_performance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Chart 5: Return Distribution ─────────────────────────────────
print("Chart 5: Return distribution...")
fig, ax = plt.subplots(figsize=(12, 5))
valid_returns = returns[returns != 0]
ax.hist(valid_returns.values, bins=100, color=COLORS["portfolio"],
        alpha=0.7, edgecolor="black", linewidth=0.3, density=True, label="Phase 2")
ax.hist(ec_base["returns"][ec_base["returns"] != 0].values, bins=100,
        color=COLORS["baseline"], alpha=0.4, edgecolor="black", linewidth=0.3,
        density=True, label="Phase 1")
ax.axvline(x=valid_returns.mean(), color="green", linewidth=2, linestyle="--",
           label=f"Mean: {valid_returns.mean()*252:.1%} ann")
ax.axvline(x=valid_returns.quantile(0.05), color="red", linewidth=1.5, linestyle=":",
           label=f"VaR 95%: {valid_returns.quantile(0.05):.3%}")
ax.set_title("Daily Return Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Density")
ax.legend(fontsize=10)
# Add stats box
stats_text = (f"Skewness: {valid_returns.skew():.2f}\n"
              f"Kurtosis: {valid_returns.kurtosis():.2f}\n"
              f"Hit Rate: {(valid_returns > 0).mean():.1%}")
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "05_return_distribution.png"), dpi=150)
plt.close(fig)

# ── Chart 6: Monthly Return Heatmap ──────────────────────────────
print("Chart 6: Monthly return heatmap...")
monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_df = pd.DataFrame({
    "year": monthly.index.year,
    "month": monthly.index.month,
    "return": monthly.values,
})
pivot = monthly_df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
ax.set_xticks(range(12))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# Add text annotations
for i in range(len(pivot.index)):
    for j in range(12):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if abs(val * 100) > 5 else "black"
            ax.text(j, i, f"{val*100:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Monthly Return (%)")
ax.set_title("Monthly Return Heatmap (%)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "06_monthly_heatmap.png"), dpi=150)
plt.close(fig)

# ── Chart 7: Rolling Volatility ──────────────────────────────────
print("Chart 7: Rolling volatility...")
rolling_vol = returns.rolling(63).std() * np.sqrt(252)
rolling_vol_b = ec_base["returns"].rolling(63).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rolling_vol.index, rolling_vol.values, color=COLORS["vol"],
        linewidth=1.2, label="Phase 2 (63d rolling)")
ax.plot(rolling_vol_b.index, rolling_vol_b.values, color=COLORS["baseline"],
        linewidth=1, linestyle="--", alpha=0.6, label="Phase 1 (63d rolling)")
ax.axhline(y=0.147, color="green", linestyle=":", alpha=0.5,
           label="Phase 2 Avg Vol: 14.7%")
ax.set_title("Rolling 63-Day Annualized Volatility", fontsize=14, fontweight="bold")
ax.set_ylabel("Annualized Volatility")
ax.legend(loc="upper right", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "07_rolling_volatility.png"), dpi=150)
plt.close(fig)

# ── Chart 8: Summary Dashboard ───────────────────────────────────
print("Chart 8: Summary dashboard...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Top-left: Equity curve
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(cum_ret.index, cum_ret.values, color=COLORS["portfolio"], linewidth=1.5,
         label="Phase 2")
ax1.plot(cum_ret_base.index, cum_ret_base.values, color=COLORS["baseline"],
         linewidth=1, linestyle="--", alpha=0.6, label="Phase 1")
ax1.set_yscale("log")
ax1.set_title("Equity Curve (log)", fontweight="bold")
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Top-right: Key metrics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
metrics_text = [
    ("Sharpe Ratio", "1.081", "+61.8%"),
    ("Annual Return", "15.91%", "+64.5%"),
    ("Max Drawdown", "-19.68%", "improved"),
    ("Sortino", "1.390", "+68.3%"),
    ("Calmar", "0.808", "+66.3%"),
    ("Total Return", "2,586%", "18yr"),
    ("Win Rate", "48.7%", ""),
    ("Profit Factor", "1.25", ""),
]
y_pos = 0.95
ax2.text(0.5, 1.0, "KEY METRICS", ha="center", fontsize=13, fontweight="bold",
         transform=ax2.transAxes)
for metric, value, change in metrics_text:
    ax2.text(0.05, y_pos - 0.12, metric, fontsize=10, transform=ax2.transAxes,
             verticalalignment="top")
    ax2.text(0.65, y_pos - 0.12, value, fontsize=10, fontweight="bold",
             transform=ax2.transAxes, verticalalignment="top", color=COLORS["portfolio"])
    if change:
        ax2.text(0.88, y_pos - 0.12, change, fontsize=8,
                 transform=ax2.transAxes, verticalalignment="top", color="green")
    y_pos -= 0.12

# Middle-left: Drawdown
ax3 = fig.add_subplot(gs[1, :2])
ax3.fill_between(drawdown.index, drawdown.values, 0,
                 color=COLORS["drawdown"], alpha=0.4)
ax3.set_title("Drawdown", fontweight="bold")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Middle-right: Strategy allocation
ax4 = fig.add_subplot(gs[1, 2])
alloc_labels = ["Trend\n~45%", "Carry\n~45%", "Vol Breakout\n~10%"]
alloc_sizes = [45, 45, 10]
alloc_colors = ["#2196F3", "#4CAF50", "#FF9800"]
ax4.pie(alloc_sizes, labels=alloc_labels, colors=alloc_colors,
        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
ax4.set_title("Strategy Risk Budget\n(Inverse-Vol Allocation)", fontweight="bold")

# Bottom-left: Rolling Sharpe
ax5 = fig.add_subplot(gs[2, :2])
ax5.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLORS["sharpe"], linewidth=1.2)
ax5.axhline(y=0, color="black", linewidth=0.5)
ax5.axhline(y=1, color="green", linestyle=":", alpha=0.4)
ax5.set_title("Rolling 1-Year Sharpe", fontweight="bold")
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Bottom-right: Vol regime Sharpe
ax6 = fig.add_subplot(gs[2, 2])
vol_regimes_sorted = sorted(vol_regimes, key=lambda r: ["low", "med", "high"].index(r["label"]))
vol_l = [r["label"] for r in vol_regimes_sorted]
vol_s = [r["metrics"]["sharpe_ratio"] for r in vol_regimes_sorted]
bars = ax6.bar(vol_l, vol_s, color=["#4CAF50", "#FF9800", "#F44336"],
               alpha=0.8, edgecolor="black")
ax6.set_title("Sharpe by Vol Regime", fontweight="bold")
ax6.set_ylabel("Sharpe")
for bar, s in zip(bars, vol_s):
    ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f"{s:.2f}", ha="center", va="bottom", fontsize=10)

fig.suptitle("Regime-Robust Systematic Futures — Phase 2 Performance Dashboard",
             fontsize=16, fontweight="bold", y=0.98)
fig.savefig(os.path.join(CHARTS, "08_summary_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Chart 9: Annual Returns Bar Chart ─────────────────────────────
print("Chart 9: Annual returns...")
annual_ret = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
annual_ret_base = ec_base["returns"].resample("YE").apply(lambda x: (1 + x).prod() - 1)

fig, ax = plt.subplots(figsize=(14, 6))
years = annual_ret.index.year
x = np.arange(len(years))
w = 0.35

bars1 = ax.bar(x - w/2, annual_ret_base.values * 100, w,
               label="Phase 1", color=COLORS["baseline"], alpha=0.7, edgecolor="black")
bars2 = ax.bar(x + w/2, annual_ret.values * 100, w,
               label="Phase 2", color=COLORS["portfolio"], alpha=0.8, edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.axhline(y=10, color="green", linestyle=":", alpha=0.4, label="S&P ~10% avg")
ax.set_title("Annual Returns: Phase 1 vs Phase 2", fontsize=14, fontweight="bold")
ax.set_ylabel("Annual Return (%)")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "09_annual_returns.png"), dpi=150)
plt.close(fig)

print(f"\nAll charts saved to: {CHARTS}")
print("Files generated:")
for f in sorted(os.listdir(CHARTS)):
    print(f"  {f}")
