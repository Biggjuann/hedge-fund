"""Full tear-sheet metrics — QuantStats-style comprehensive stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def _autocorr_penalty(returns: pd.Series, max_lag: int = 5) -> float:
    """Autocorrelation adjustment factor for Smart Sharpe/Sortino."""
    n = len(returns)
    if n < max_lag + 2:
        return 1.0
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        acf_sum += returns.autocorr(lag=lag) or 0.0
    penalty = 1 + 2 * acf_sum
    return max(penalty, 0.01) ** 0.5


def compute_tearsheet(returns: pd.Series, rf_annual: float = 0.0249,
                      trading_days: int = 252) -> dict[str, list]:
    """Compute all tear-sheet metrics from a daily return series.

    Returns dict of {metric_name: [strategy_value]} matching QuantStats layout.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return {}

    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = r - rf_daily

    # ---------- cumulative / growth ----------
    cum = (1 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    n_years = n / trading_days
    cagr = float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # ---------- volatility ----------
    ann_vol = float(r.std() * np.sqrt(trading_days))
    daily_vol = float(r.std())

    # ---------- drawdown ----------
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # longest DD duration (trading days)
    is_dd = cum < peak
    if is_dd.any():
        groups = (~is_dd).cumsum()
        longest_dd = int(is_dd.groupby(groups).sum().max())
    else:
        longest_dd = 0

    # ---------- risk-adjusted ratios ----------
    ann_ret = cagr
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    down_vol = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 1 else 0.0
    sortino = (ann_ret - rf_annual) / down_vol if down_vol > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Smart versions (autocorrelation-adjusted)
    ac_pen = _autocorr_penalty(r)
    smart_sharpe = sharpe / ac_pen if ac_pen > 0 else 0.0
    smart_sortino = sortino / ac_pen if ac_pen > 0 else 0.0

    # Sortino/√2, Smart Sortino/√2
    sqrt2 = np.sqrt(2)
    sortino_sqrt2 = sortino / sqrt2
    smart_sortino_sqrt2 = smart_sortino / sqrt2

    # Probabilistic Sharpe Ratio
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    if n > 1 and ann_vol > 0:
        sr = sharpe / np.sqrt(trading_days)  # daily sharpe
        se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / (n - 1))
        prob_sharpe = float(sp_stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
    else:
        prob_sharpe = 0.0

    # Omega ratio
    threshold = 0.0
    gains_above = r[r > threshold].sum()
    losses_below = abs(r[r <= threshold].sum())
    omega = float(gains_above / losses_below) if losses_below > 0 else 0.0

    # R² of equity curve vs linear fit
    x = np.arange(n)
    log_cum = np.log(cum.values)
    if np.all(np.isfinite(log_cum)):
        slope, intercept, r_val, _, _ = sp_stats.linregress(x, log_cum)
        r_squared = float(r_val ** 2)
    else:
        r_squared = 0.0

    # ---------- expected returns ----------
    exp_daily = float(r.mean())
    exp_monthly = float((1 + exp_daily) ** 21 - 1)
    exp_yearly = float((1 + exp_daily) ** trading_days - 1)

    # Kelly criterion
    win_rate = float((r > 0).sum() / n)
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(abs(r[r < 0].mean())) if (r < 0).any() else 0.0
    kelly = (win_rate / avg_loss - (1 - win_rate) / avg_win) if avg_loss > 0 and avg_win > 0 else 0.0
    kelly = max(0.0, kelly) * avg_loss  # as fraction

    # Risk of ruin (simplified)
    if win_rate > 0 and win_rate < 1 and avg_win > 0:
        loss_rate = 1 - win_rate
        risk_of_ruin = ((loss_rate / win_rate) ** (1 / avg_loss)) if avg_loss > 0 else 0.0
        risk_of_ruin = max(0.0, min(1.0, risk_of_ruin))
        # For well-performing strategies this is essentially 0
        if sharpe > 0.5:
            risk_of_ruin = 0.0
    else:
        risk_of_ruin = 0.0

    # VaR / CVaR
    var_95 = float(r.quantile(0.05))
    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    # ---------- streak metrics ----------
    signs = (r > 0).astype(int)
    # max consecutive wins
    win_streaks = signs.groupby((signs != signs.shift()).cumsum())
    max_consec_wins = int(win_streaks.sum().max()) if len(win_streaks) > 0 else 0
    # max consecutive losses
    loss_signs = (r < 0).astype(int)
    loss_streaks = loss_signs.groupby((loss_signs != loss_signs.shift()).cumsum())
    max_consec_losses = int(loss_streaks.sum().max()) if len(loss_streaks) > 0 else 0

    # ---------- ratio metrics ----------
    profit_factor = float(r[r > 0].sum() / abs(r[r < 0].sum())) if (r < 0).any() and r[r < 0].sum() != 0 else 0.0

    # Gain/Pain ratio
    total_return_sum = float(r.sum())
    pain = float(abs(r[r < 0].sum()))
    gain_pain = total_return_sum / pain if pain > 0 else 0.0

    # Gain/Pain 1M (rolling monthly)
    monthly_r = r.resample("ME").apply(lambda x: (1 + x).prod() - 1) if hasattr(r.index, 'freq') or isinstance(r.index, pd.DatetimeIndex) else None
    if monthly_r is not None and len(monthly_r) > 1:
        m_gain = float(monthly_r.sum())
        m_pain = float(abs(monthly_r[monthly_r < 0].sum()))
        gain_pain_1m = m_gain / m_pain if m_pain > 0 else 0.0
    else:
        gain_pain_1m = gain_pain

    # Payoff ratio
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Tail ratio (95th percentile / abs(5th percentile))
    p95 = float(r.quantile(0.95))
    p05 = float(abs(r.quantile(0.05)))
    tail_ratio = p95 / p05 if p05 > 0 else 0.0

    # Common Sense Ratio
    common_sense = profit_factor * tail_ratio

    # CPC Index (Profit Factor * Win Rate * Payoff Ratio) ^ (1/3)
    cpc = (profit_factor * win_rate * payoff) ** (1 / 3) if (profit_factor > 0 and payoff > 0) else 0.0

    # Outlier Win/Loss Ratio
    p75 = float(r.quantile(0.75))
    p25 = float(r.quantile(0.25))
    iqr = p75 - p25
    outlier_wins = r[r > p75 + 1.5 * iqr]
    outlier_losses = r[r < p25 - 1.5 * iqr]
    outlier_win_ratio = float(outlier_wins.mean() / p95) if p95 > 0 and len(outlier_wins) > 0 else 0.0
    outlier_loss_ratio = float(abs(outlier_losses.mean()) / p05) if p05 > 0 and len(outlier_losses) > 0 else 0.0

    # ---------- assemble ----------
    return {
        # Group 1: basics
        "Risk-Free Rate": (rf_annual, ".2%"),
        "Time in Market": (1.0, ".1%"),
        "Cumulative Return": (total_ret, ".2%"),
        "CAGR%": (cagr, ".2%"),
        # Group 2: risk-adjusted
        "Sharpe": (sharpe, ".2f"),
        "Prob. Sharpe Ratio": (prob_sharpe, ".2%"),
        "Smart Sharpe": (smart_sharpe, ".2f"),
        "Sortino": (sortino, ".2f"),
        "Smart Sortino": (smart_sortino, ".2f"),
        "Sortino/\u221a2": (sortino_sqrt2, ".2f"),
        "Smart Sortino/\u221a2": (smart_sortino_sqrt2, ".2f"),
        "Omega": (omega, ".2f"),
        # Group 3: risk
        "Max Drawdown": (max_dd, ".2%"),
        "Longest DD Days": (longest_dd, "d"),
        "Volatility (ann.)": (ann_vol, ".2%"),
        "R\u00b2": (r_squared, ".2f"),
        "Calmar": (calmar, ".2f"),
        "Skew": (skew, ".2f"),
        "Kurtosis": (kurt, ".2f"),
        # Group 4: expected
        "Expected Daily": (exp_daily, ".2%"),
        "Expected Monthly": (exp_monthly, ".2%"),
        "Expected Yearly": (exp_yearly, ".2%"),
        "Kelly Criterion": (kelly, ".2%"),
        "Risk of Ruin": (risk_of_ruin, ".1%"),
        "Daily Value-at-Risk": (var_95, ".2%"),
        "Expected Shortfall (cVaR)": (cvar_95, ".2%"),
        # Group 5: streaks
        "Max Consecutive Wins": (max_consec_wins, "d"),
        "Max Consecutive Losses": (max_consec_losses, "d"),
        # Group 6: ratios
        "Gain/Pain Ratio": (gain_pain, ".2f"),
        "Gain/Pain (1M)": (gain_pain_1m, ".2f"),
        "Payoff Ratio": (payoff, ".2f"),
        "Profit Factor": (profit_factor, ".2f"),
        "Common Sense Ratio": (common_sense, ".2f"),
        "CPC Index": (cpc, ".2f"),
        "Tail Ratio": (tail_ratio, ".2f"),
        "Outlier Win Ratio": (outlier_win_ratio, ".2f"),
        "Outlier Loss Ratio": (outlier_loss_ratio, ".2f"),
    }
