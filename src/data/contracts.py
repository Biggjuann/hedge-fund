"""
Contract metadata management — per 07_DATA_MODEL.md.

Stores: symbol, exchange, currency, multiplier, tick_size, tick_value,
session hours, last_trade_date, first_notice_date, delivery flags, roll rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class RollRuleConfig:
    """Explicit roll rule configuration — must be unit-tested per spec."""

    method: str  # 'calendar' or 'liquidity'
    roll_days_before_expiry: int
    expiry_months: tuple[int, ...]
    month_codes: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict) -> RollRuleConfig:
        return cls(
            method=d["method"],
            roll_days_before_expiry=d["roll_days_before_expiry"],
            expiry_months=tuple(d["expiry_months"]),
            month_codes=tuple(d["month_codes"]),
        )


@dataclass(frozen=True)
class ContractMetadata:
    """Immutable metadata for a single futures root symbol."""

    symbol: str
    description: str
    exchange: str
    venue: str
    asset_class: str
    currency: str
    multiplier: float
    tick_size: float
    tick_value: float
    price_precision: int
    session_hours: str
    margin_init_pct: float
    margin_maint_pct: float
    commission_per_contract: float
    avg_spread_ticks: float
    avg_daily_volume: int
    roll_rules: RollRuleConfig
    data_file: str | None = None

    @classmethod
    def from_dict(cls, symbol: str, d: dict) -> ContractMetadata:
        return cls(
            symbol=symbol,
            description=d["description"],
            exchange=d["exchange"],
            venue=d["venue"],
            asset_class=d["asset_class"],
            currency=d["currency"],
            multiplier=d["multiplier"],
            tick_size=d["tick_size"],
            tick_value=d["tick_value"],
            price_precision=d["price_precision"],
            session_hours=d["session_hours"],
            margin_init_pct=d["margin_init_pct"],
            margin_maint_pct=d["margin_maint_pct"],
            commission_per_contract=d["commission_per_contract"],
            avg_spread_ticks=d["avg_spread_ticks"],
            avg_daily_volume=d["avg_daily_volume"],
            roll_rules=RollRuleConfig.from_dict(d["roll_rules"]),
            data_file=d.get("data_file"),
        )

    @property
    def point_value(self) -> float:
        """Dollar value of one full point move per contract."""
        return self.multiplier

    @property
    def tick_dollar_value(self) -> float:
        """Dollar value of one tick move per contract."""
        return self.tick_value

    def notional_value(self, price: float, contracts: float = 1.0) -> float:
        """Full notional value of position."""
        return price * self.multiplier * contracts

    def margin_required(self, price: float, contracts: float = 1.0) -> float:
        """Initial margin required for position."""
        return self.notional_value(price, contracts) * self.margin_init_pct


def load_instrument_configs(
    config_path: str | None = None,
) -> dict[str, ContractMetadata]:
    """Load all instrument configs from YAML.

    Parameters
    ----------
    config_path : str, optional
        Path to instruments.yaml. Defaults to config/instruments.yaml
        relative to the project root.

    Returns
    -------
    dict mapping symbol -> ContractMetadata
    """
    if config_path is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(project_root, "config", "instruments.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    instruments: dict[str, ContractMetadata] = {}
    for symbol, cfg in raw["instruments"].items():
        instruments[symbol] = ContractMetadata.from_dict(symbol, cfg)

    return instruments
