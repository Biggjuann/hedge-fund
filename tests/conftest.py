"""
Shared test fixtures for the test suite.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import ContractMetadata, RollRuleConfig


@pytest.fixture
def sample_dates():
    """Generate 3 years of business days."""
    return pd.bdate_range("2020-01-02", "2023-01-02", freq="B")


@pytest.fixture
def sample_prices(sample_dates):
    """Generate synthetic OHLCV daily data with realistic properties."""
    np.random.seed(42)
    n = len(sample_dates)

    # Random walk for close prices starting at 4000
    returns = np.random.normal(0.0003, 0.012, n)
    close = 4000 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(500000, 2000000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=sample_dates,
    )

    return df


@pytest.fixture
def sample_returns(sample_prices):
    """Log returns from sample prices."""
    return np.log(sample_prices["close"] / sample_prices["close"].shift(1)).dropna()


@pytest.fixture
def es_metadata():
    """ES contract metadata."""
    return ContractMetadata(
        symbol="ES",
        description="E-mini S&P 500",
        exchange="CME",
        venue="GLBX",
        asset_class="INDEX",
        currency="USD",
        multiplier=50,
        tick_size=0.25,
        tick_value=12.50,
        price_precision=2,
        session_hours="18:00-17:00",
        margin_init_pct=0.05,
        margin_maint_pct=0.04,
        commission_per_contract=2.25,
        avg_spread_ticks=1.0,
        avg_daily_volume=1500000,
        roll_rules=RollRuleConfig(
            method="calendar",
            roll_days_before_expiry=8,
            expiry_months=(3, 6, 9, 12),
            month_codes=("H", "M", "U", "Z"),
        ),
    )


@pytest.fixture
def multi_instrument_returns(sample_dates):
    """Returns for 3 instruments (for correlation tests)."""
    np.random.seed(42)
    n = len(sample_dates) - 1  # one less due to returns
    dates = sample_dates[1:]

    # Correlated returns
    common_factor = np.random.normal(0, 0.008, n)
    es_ret = common_factor + np.random.normal(0, 0.006, n)
    nq_ret = common_factor * 1.2 + np.random.normal(0, 0.008, n)
    rty_ret = common_factor * 0.8 + np.random.normal(0, 0.010, n)

    return pd.DataFrame(
        {"ES": es_ret, "NQ": nq_ret, "RTY": rty_ret},
        index=dates,
    )
