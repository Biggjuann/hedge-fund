"""
Risk overlays — per 06_RISK_PORTFOLIO_EXECUTION.md.

1. Vol targeting (per-strategy + portfolio level)
2. Correlation haircuts (reduce risk when corr spikes)
3. Drawdown ladder (5%/8%/10% de-risk + kill switch + cooldown)

Per 01_DEEP_RESEARCH_PAPER.md: these are risk controls, NOT alpha.
Use regime probabilities as soft gating inputs, NOT hard switches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.volatility import ewma_volatility, rolling_ewma_correlation_series


@dataclass
class DrawdownState:
    """Current drawdown state for the ladder."""

    peak_equity: float
    current_equity: float
    drawdown_pct: float
    risk_multiplier: float
    is_killed: bool
    cooldown_remaining: int
    recovery_window_returns: list[float]


class DrawdownLadder:
    """Drawdown-based de-risk ladder — per 06_RISK_PORTFOLIO_EXECUTION.md.

    DD >= 5%: risk * 0.75
    DD >= 8%: risk * 0.50
    DD >= 10%: flat + cooldown N days
    Cooldown re-entry requires rolling performance stabilization.
    """

    def __init__(
        self,
        thresholds: list[dict] | None = None,
        cooldown_days: int = 10,
        recovery_window: int = 20,
        recovery_min_sharpe: float = -0.5,
        max_cooldown_days: int = 30,
    ):
        if thresholds is None:
            thresholds = [
                {"threshold": 0.08, "risk_multiplier": 0.75},
                {"threshold": 0.12, "risk_multiplier": 0.50},
                {"threshold": 0.15, "risk_multiplier": 0.00},
            ]
        self.thresholds = sorted(thresholds, key=lambda x: x["threshold"])
        self.cooldown_days = cooldown_days
        self.recovery_window = recovery_window
        self.recovery_min_sharpe = recovery_min_sharpe
        self.max_cooldown_days = max_cooldown_days

    def compute_risk_multipliers(
        self,
        equity_curve: pd.Series,
    ) -> pd.Series:
        """Compute risk multiplier series from equity curve.

        Tracks peak equity, drawdown, and applies ladder rules.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity curve (starting from initial capital).

        Returns
        -------
        pd.Series
            Risk multiplier at each timestamp (0.0 to 1.0).
        """
        multipliers = pd.Series(1.0, index=equity_curve.index)
        peak = equity_curve.iloc[0]
        cooldown_remaining = 0
        total_cooldown_elapsed = 0
        in_cooldown = False
        cooldown_start_idx = 0

        for i in range(len(equity_curve)):
            current = equity_curve.iloc[i]

            if not in_cooldown:
                peak = max(peak, current)

            dd_pct = (peak - current) / peak if peak > 0 else 0.0

            if in_cooldown:
                cooldown_remaining -= 1
                total_cooldown_elapsed += 1

                # Hard max cooldown cap — force re-entry
                if total_cooldown_elapsed >= self.max_cooldown_days:
                    in_cooldown = False
                    peak = current  # reset peak to avoid immediate re-trigger
                    multipliers.iloc[i] = 1.0
                    continue

                if cooldown_remaining <= 0:
                    # Recovery window starts AFTER cooldown ends (not during DD)
                    window_start = i
                    window_end = i
                    # Look at returns from cooldown end onward (the last few days)
                    lookback = min(self.recovery_window, i - cooldown_start_idx)
                    if lookback >= 5:
                        recent_returns = equity_curve.iloc[
                            max(0, i - lookback) : i + 1
                        ].pct_change().dropna()
                        if len(recent_returns) > 0:
                            mean_ret = recent_returns.mean()
                            std_ret = recent_returns.std()
                            rolling_sharpe = (
                                mean_ret / std_ret * np.sqrt(252)
                                if std_ret > 0
                                else 0.0
                            )
                            if rolling_sharpe >= self.recovery_min_sharpe:
                                in_cooldown = False
                                peak = current  # reset peak on re-entry
                                multipliers.iloc[i] = 1.0
                                continue

                if in_cooldown:
                    multipliers.iloc[i] = 0.0
                    continue

            # Apply ladder
            mult = 1.0
            for level in self.thresholds:
                if dd_pct >= level["threshold"]:
                    mult = level["risk_multiplier"]

            if mult == 0.0:
                in_cooldown = True
                cooldown_remaining = self.cooldown_days
                total_cooldown_elapsed = 0
                cooldown_start_idx = i

            multipliers.iloc[i] = mult

        return multipliers


class CorrelationHaircut:
    """Reduce risk budget when correlations spike.

    Per spec: risk-on/off clustering → correlation haircuts.
    Uses continuous scaling, NOT hard binary switch.
    """

    def __init__(
        self,
        avg_corr_threshold_high: float = 0.6,
        risk_scale_at_high_corr: float = 0.60,
        avg_corr_threshold_extreme: float = 0.8,
        risk_scale_at_extreme_corr: float = 0.30,
        corr_lookback: int = 63,
    ):
        self.avg_corr_threshold_high = avg_corr_threshold_high
        self.risk_scale_at_high_corr = risk_scale_at_high_corr
        self.avg_corr_threshold_extreme = avg_corr_threshold_extreme
        self.risk_scale_at_extreme_corr = risk_scale_at_extreme_corr
        self.corr_lookback = corr_lookback

    def compute_haircut(
        self,
        avg_corr: pd.Series,
    ) -> pd.Series:
        """Compute correlation-based risk scaling factor.

        Interpolates linearly between thresholds.

        Parameters
        ----------
        avg_corr : pd.Series
            Average pairwise correlation (already shifted for causality).

        Returns
        -------
        pd.Series
            Risk scaling factor in [risk_scale_at_extreme, 1.0].
        """
        haircut = pd.Series(1.0, index=avg_corr.index)

        # Linear interpolation between normal and high
        high_mask = avg_corr >= self.avg_corr_threshold_high
        extreme_mask = avg_corr >= self.avg_corr_threshold_extreme

        # Between high and extreme: interpolate
        mid_mask = high_mask & ~extreme_mask
        if mid_mask.any():
            frac = (avg_corr[mid_mask] - self.avg_corr_threshold_high) / (
                self.avg_corr_threshold_extreme - self.avg_corr_threshold_high
            )
            haircut[mid_mask] = (
                self.risk_scale_at_high_corr
                + (self.risk_scale_at_extreme_corr - self.risk_scale_at_high_corr)
                * frac
            )

        # Below high but above normal: interpolate from 1.0 to high_scale
        # Use a buffer zone starting at threshold - 0.1
        buffer_start = max(0, self.avg_corr_threshold_high - 0.1)
        buffer_mask = (avg_corr >= buffer_start) & ~high_mask
        if buffer_mask.any():
            frac = (avg_corr[buffer_mask] - buffer_start) / (
                self.avg_corr_threshold_high - buffer_start
            )
            haircut[buffer_mask] = 1.0 + (self.risk_scale_at_high_corr - 1.0) * frac

        # Above extreme
        haircut[extreme_mask] = self.risk_scale_at_extreme_corr

        return haircut.clip(lower=self.risk_scale_at_extreme_corr, upper=1.0)


class RiskOverlay:
    """Combined risk overlay applying all risk controls.

    Orchestrates:
    1. Drawdown ladder
    2. Correlation haircuts
    3. Leverage caps
    """

    def __init__(
        self,
        drawdown_ladder: DrawdownLadder,
        corr_haircut: CorrelationHaircut,
        max_leverage: float = 4.0,
    ):
        self.drawdown_ladder = drawdown_ladder
        self.corr_haircut = corr_haircut
        self.max_leverage = max_leverage

    def apply(
        self,
        weights: pd.DataFrame,
        equity_curve: pd.Series,
        avg_corr: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply all risk overlays to portfolio weights.

        Parameters
        ----------
        weights : pd.DataFrame
            Raw portfolio weights per instrument.
        equity_curve : pd.Series
            Running equity curve.
        avg_corr : pd.Series, optional
            Average pairwise correlation.

        Returns
        -------
        pd.DataFrame
            Risk-adjusted weights.
        """
        adjusted = weights.copy()

        # 1. Drawdown ladder
        dd_mult = self.drawdown_ladder.compute_risk_multipliers(equity_curve)
        # Align index
        common_idx = adjusted.index.intersection(dd_mult.index)
        for col in adjusted.columns:
            adjusted.loc[common_idx, col] *= dd_mult.loc[common_idx]

        # 2. Correlation haircuts
        if avg_corr is not None:
            corr_scale = self.corr_haircut.compute_haircut(avg_corr)
            common_idx = adjusted.index.intersection(corr_scale.index)
            for col in adjusted.columns:
                adjusted.loc[common_idx, col] *= corr_scale.loc[common_idx]

        # 3. Leverage cap
        gross = adjusted.abs().sum(axis=1)
        excess = gross > self.max_leverage
        if excess.any():
            scale_down = self.max_leverage / gross[excess]
            adjusted.loc[excess] = adjusted.loc[excess].multiply(
                scale_down, axis=0
            )

        return adjusted
