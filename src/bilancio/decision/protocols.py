"""Decision protocol interfaces and default implementations.

Defines the four decision levels as Protocol classes with trivial
default implementations that capture current hard-coded behavior.
Wired into the lending engine via ``_resolve_protocols()`` in
``engines/lending.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable


# ── Level 1: Portfolio Strategy ─────────────────────────────────────

@runtime_checkable
class PortfolioStrategy(Protocol):
    """Decides overall portfolio allocation and exposure limits."""

    def max_exposure(self, total_assets: int) -> int:
        """Maximum total lending exposure given current assets."""
        ...

    def target_return(self) -> Decimal:
        """Target portfolio return rate."""
        ...


@dataclass(frozen=True)
class FixedPortfolioStrategy:
    """Default: fixed fraction of assets as max exposure.

    Captures current LendingConfig.max_total_exposure behavior.
    """

    max_exposure_fraction: Decimal = Decimal("0.80")
    base_return: Decimal = Decimal("0.05")

    def max_exposure(self, total_assets: int) -> int:
        return int(total_assets * self.max_exposure_fraction)

    def target_return(self) -> Decimal:
        return self.base_return


# ── Level 2: Counterparty Screener ──────────────────────────────────

@runtime_checkable
class CounterpartyScreener(Protocol):
    """Screens counterparties for eligibility."""

    def is_eligible(self, default_probability: Decimal) -> bool:
        """Whether a counterparty passes the screening threshold."""
        ...


@dataclass(frozen=True)
class ThresholdScreener:
    """Default: reject if default probability exceeds threshold.

    Captures current max_default_prob filter in lending.
    """

    max_default_prob: Decimal = Decimal("0.5")

    def is_eligible(self, default_probability: Decimal) -> bool:
        return default_probability <= self.max_default_prob


# ── Level 3: Instrument Selector ────────────────────────────────────

@runtime_checkable
class InstrumentSelector(Protocol):
    """Selects instrument parameters (maturity, collateral, etc.)."""

    def select_maturity(self) -> int:
        """Choose maturity in days for new instruments."""
        ...


@dataclass(frozen=True)
class FixedMaturitySelector:
    """Default: always use the same maturity.

    Captures current maturity_days in LendingConfig.
    """

    maturity_days: int = 2

    def select_maturity(self) -> int:
        return self.maturity_days


# ── Level 4: Transaction Pricer ─────────────────────────────────────

@runtime_checkable
class TransactionPricer(Protocol):
    """Prices individual transactions based on risk."""

    def price(self, base_rate: Decimal, default_probability: Decimal) -> Decimal:
        """Compute the interest rate for a transaction."""
        ...


@dataclass(frozen=True)
class LinearPricer:
    """Default: rate = base_rate + scale * default_probability.

    Captures current non-bank lender pricing rule.
    """

    risk_premium_scale: Decimal = Decimal("0.20")

    def price(self, base_rate: Decimal, default_probability: Decimal) -> Decimal:
        return base_rate + self.risk_premium_scale * default_probability


# ── Level 5: Instrument Valuer ─────────────────────────────────────

@runtime_checkable
class InstrumentValuer(Protocol):
    """Values a tradable instrument — an agent's belief about worth.

    Two-method protocol:
    - value_decimal(): fast path returning bare Decimal (backward-compat)
    - value(): full path returning Estimate with provenance
    """

    def value_decimal(self, ticket: Any, day: int) -> Decimal: ...
    def value(self, ticket: Any, day: int) -> Any: ...


__all__ = [
    "PortfolioStrategy",
    "FixedPortfolioStrategy",
    "CounterpartyScreener",
    "ThresholdScreener",
    "InstrumentSelector",
    "FixedMaturitySelector",
    "TransactionPricer",
    "LinearPricer",
    "InstrumentValuer",
]
