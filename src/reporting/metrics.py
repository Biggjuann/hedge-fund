"""
Performance metrics — standardized computations for reporting.

Per 00_MASTER_CONTEXT.md: outputs must include standardized reports
with regime breakdowns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Compute standard performance metrics from a return series."""

    def __init__(self, trading_days: int = 252, risk_free_rate: float = 0.0):
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate

    def compute_all(self, returns: pd.Series) -> dict:
        """Compute all metrics for a return series.

        Returns
        -------
        dict
            Dictionary of named metrics.
        """
        if len(returns) < 2 or returns.isna().all():
            return self._empty_metrics()

        clean = returns.dropna()
        if len(clean) < 2:
            return self._empty_metrics()

        cumulative = (1 + clean).cumprod()

        return {
            "annualized_return": self.annualized_return(clean),
            "annualized_volatility": self.annualized_volatility(clean),
            "sharpe_ratio": self.sharpe_ratio(clean),
            "sortino_ratio": self.sortino_ratio(clean),
            "max_drawdown": self.max_drawdown(clean),
            "max_drawdown_duration_days": self.max_drawdown_duration(clean),
            "calmar_ratio": self.calmar_ratio(clean),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
            "hit_rate": self.hit_rate(clean),
            "profit_factor": self.profit_factor(clean),
            "avg_win": self.avg_win(clean),
            "avg_loss": self.avg_loss(clean),
            "total_return": float(cumulative.iloc[-1] - 1),
            "n_observations": len(clean),
            "n_positive_days": int((clean > 0).sum()),
            "n_negative_days": int((clean < 0).sum()),
            "best_day": float(clean.max()),
            "worst_day": float(clean.min()),
            "var_95": self.value_at_risk(clean, 0.05),
            "cvar_95": self.conditional_var(clean, 0.05),
        }

    def annualized_return(self, returns: pd.Series) -> float:
        total = (1 + returns).prod()
        n_years = len(returns) / self.trading_days
        if n_years <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    def annualized_volatility(self, returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(self.trading_days))

    def sharpe_ratio(self, returns: pd.Series) -> float:
        ann_ret = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        if ann_vol == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / ann_vol)

    def sortino_ratio(self, returns: pd.Series) -> float:
        ann_ret = self.annualized_return(returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf") if ann_ret > 0 else 0.0
        downside_vol = float(downside.std() * np.sqrt(self.trading_days))
        if downside_vol == 0:
            return 0.0
        return float((ann_ret - self.risk_free_rate) / downside_vol)

    def max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def max_drawdown_duration(self, returns: pd.Series) -> int:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        is_dd = cumulative < peak

        if not is_dd.any():
            return 0

        # Find longest consecutive drawdown
        groups = (~is_dd).cumsum()
        dd_groups = is_dd.groupby(groups)
        max_duration = 0
        for _, group in dd_groups:
            if group.any():
                duration = group.sum()
                max_duration = max(max_duration, duration)

        return int(max_duration)

    def calmar_ratio(self, returns: pd.Series) -> float:
        ann_ret = self.annualized_return(returns)
        mdd = abs(self.max_drawdown(returns))
        if mdd == 0:
            return 0.0
        return float(ann_ret / mdd)

    def hit_rate(self, returns: pd.Series) -> float:
        n_positive = (returns > 0).sum()
        return float(n_positive / len(returns)) if len(returns) > 0 else 0.0

    def profit_factor(self, returns: pd.Series) -> float:
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def avg_win(self, returns: pd.Series) -> float:
        wins = returns[returns > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    def avg_loss(self, returns: pd.Series) -> float:
        losses = returns[returns < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    def value_at_risk(self, returns: pd.Series, alpha: float = 0.05) -> float:
        return float(returns.quantile(alpha))

    def conditional_var(self, returns: pd.Series, alpha: float = 0.05) -> float:
        var = self.value_at_risk(returns, alpha)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def _empty_metrics(self) -> dict:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0,
            "calmar_ratio": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return": 0.0,
            "n_observations": 0,
            "n_positive_days": 0,
            "n_negative_days": 0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }
