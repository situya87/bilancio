"""Jurisdiction domain models for multi-currency/multi-jurisdiction scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum


class InterbankSettlementMode(str, Enum):
    """How interbank settlement is conducted within a jurisdiction."""
    RTGS = "RTGS"
    DNS = "DNS"
    HYBRID = "HYBRID"

    def __str__(self) -> str:
        return self.value


class CapitalFlowPurpose(str, Enum):
    """Purpose classification for cross-border capital flows."""
    TRADE = "TRADE"
    PORTFOLIO = "PORTFOLIO"
    FDI = "FDI"
    INTERBANK = "INTERBANK"
    REMITTANCE = "REMITTANCE"
    OTHER = "OTHER"

    def __str__(self) -> str:
        return self.value


class CapitalControlAction(str, Enum):
    """Action to take when a capital flow matches a control rule."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    TAX = "TAX"

    def __str__(self) -> str:
        return self.value


@dataclass
class CapitalControlRule:
    """A single capital control rule evaluated against cross-border flows."""
    purpose: CapitalFlowPurpose
    direction: str  # "inflow", "outflow", or "both"
    action: CapitalControlAction
    tax_rate: Decimal = Decimal("0")
    description: str = ""


@dataclass
class CapitalControls:
    """Set of capital control rules with a default action.

    Rules are evaluated in order (first-match-wins). If no rule matches,
    ``default_action`` is used.
    """
    rules: list[CapitalControlRule] = field(default_factory=list)
    default_action: CapitalControlAction = CapitalControlAction.ALLOW

    def evaluate(
        self, purpose: CapitalFlowPurpose, direction: str
    ) -> tuple[CapitalControlAction, Decimal]:
        """Evaluate capital controls for a given flow.

        Args:
            purpose: The purpose of the capital flow.
            direction: ``"inflow"`` or ``"outflow"``.

        Returns:
            Tuple of (action, tax_rate). Tax rate is ``Decimal("0")``
            unless the action is ``TAX``.
        """
        for rule in self.rules:
            if rule.purpose == purpose and rule.direction in (direction, "both"):
                return rule.action, rule.tax_rate
        return self.default_action, Decimal("0")


@dataclass
class BankingRules:
    """Banking regulations within a jurisdiction."""
    reserve_requirement_ratio: Decimal = Decimal("0")
    interbank_settlement_mode: InterbankSettlementMode = InterbankSettlementMode.RTGS
    deposit_convertibility: bool = True
    cb_lending_enabled: bool = True


@dataclass
class ExchangeRatePair:
    """A quoted exchange rate between two currencies.

    Convention: ``rate`` is the price of 1 unit of ``base_currency``
    expressed in ``quote_currency`` (e.g., EUR/USD = 1.10 means
    1 EUR = 1.10 USD).
    """
    base_currency: str
    quote_currency: str
    rate: Decimal
    spread: Decimal = Decimal("0")

    @property
    def bid(self) -> Decimal:
        """Price at which the market buys the base currency."""
        return self.rate - self.spread / 2

    @property
    def ask(self) -> Decimal:
        """Price at which the market sells the base currency."""
        return self.rate + self.spread / 2

    def convert(self, amount: int, from_currency: str) -> int:
        """Convert an integer amount from one currency to the other.

        Uses mid rate and rounds half-up to nearest integer.

        Args:
            amount: Amount in minor units of ``from_currency``.
            from_currency: Must be either ``base_currency`` or ``quote_currency``.

        Returns:
            Converted amount in minor units of the other currency.

        Raises:
            ValueError: If ``from_currency`` is neither base nor quote.
        """
        if from_currency == self.base_currency:
            # base -> quote: multiply by rate
            result = Decimal(amount) * self.rate
        elif from_currency == self.quote_currency:
            # quote -> base: divide by rate
            result = Decimal(amount) / self.rate
        else:
            raise ValueError(
                f"Currency {from_currency} is neither base ({self.base_currency}) "
                f"nor quote ({self.quote_currency})"
            )
        return int(result.to_integral_value(rounding=ROUND_HALF_UP))


@dataclass
class FXMarket:
    """Foreign exchange market holding exchange rate pairs.

    Supports automatic rate inversion: if only EUR/USD is registered,
    querying USD/EUR will return the inverse rate.
    """
    _rates: dict[tuple[str, str], ExchangeRatePair] = field(default_factory=dict)

    def add_rate(self, pair: ExchangeRatePair) -> None:
        """Register an exchange rate pair."""
        self._rates[(pair.base_currency, pair.quote_currency)] = pair

    def get_rate(self, base: str, quote: str) -> ExchangeRatePair:
        """Look up exchange rate, with auto-inversion.

        Args:
            base: Base currency code.
            quote: Quote currency code.

        Returns:
            The ``ExchangeRatePair`` (possibly a synthetic inverse).

        Raises:
            KeyError: If no rate exists in either direction.
        """
        if base == quote:
            return ExchangeRatePair(
                base_currency=base,
                quote_currency=quote,
                rate=Decimal("1"),
                spread=Decimal("0"),
            )
        direct = self._rates.get((base, quote))
        if direct is not None:
            return direct
        inverse = self._rates.get((quote, base))
        if inverse is not None:
            inv_rate = Decimal("1") / inverse.rate
            inv_spread = inverse.spread / (inverse.rate ** 2) if inverse.rate else Decimal("0")
            return ExchangeRatePair(
                base_currency=base,
                quote_currency=quote,
                rate=inv_rate,
                spread=inv_spread,
            )
        raise KeyError(f"No exchange rate found for {base}/{quote}")

    def convert(self, amount: int, from_currency: str, to_currency: str) -> int:
        """Convert an amount between currencies.

        Args:
            amount: Amount in minor units.
            from_currency: Source currency code.
            to_currency: Target currency code.

        Returns:
            Converted amount in minor units.
        """
        if from_currency == to_currency:
            return amount
        pair = self.get_rate(from_currency, to_currency)
        return pair.convert(amount, from_currency)


@dataclass
class Jurisdiction:
    """A monetary/regulatory jurisdiction.

    Groups agents under a common currency and set of banking rules.
    """
    id: str
    name: str
    domestic_currency: str
    institutional_agent_ids: list[str] = field(default_factory=list)
    banking_rules: BankingRules = field(default_factory=BankingRules)
    capital_controls: CapitalControls = field(default_factory=CapitalControls)
