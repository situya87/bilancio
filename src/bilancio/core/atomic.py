"""Atomic value types for bilancio."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


class AtomicValue(Protocol):
    """Protocol for atomic values that have a value property."""

    @property
    def value(self) -> object:
        """Return the atomic value."""
        ...


@dataclass
class Money:
    """Represents a monetary amount with currency."""

    amount: Decimal
    currency: str

    @property
    def value(self) -> Decimal:
        """Return the monetary amount as the atomic value."""
        return self.amount


@dataclass
class Quantity:
    """Represents a quantity with a unit."""

    value: float
    unit: str


@dataclass
class Rate:
    """Represents a rate with a basis."""

    value: Decimal
    basis: str
