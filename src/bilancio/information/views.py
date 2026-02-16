"""Contextual view objects returned by InformationService queries.

Frozen dataclasses representing information snapshots at a point in time.
Fields are ``None`` when the observer lacks access (AccessLevel.NONE).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class SystemView:
    """Snapshot of system-wide conditions."""

    day: int
    aggregate_default_rate: Decimal | None = None
    system_liquidity: int | None = None


@dataclass(frozen=True)
class CounterpartyView:
    """Snapshot of a single counterparty's observable state."""

    agent_id: str
    day: int
    cash: int | None = None
    obligations: int | None = None
    net_worth: int | None = None
    default_probability: Decimal | None = None


@dataclass(frozen=True)
class InstrumentView:
    """Snapshot of instrument/market information.

    Sparse in Phase 1 — dealer quote data not yet queryable
    through the service.
    """

    day: int


@dataclass(frozen=True)
class TransactionView:
    """Snapshot of counterparty x instrument specific information."""

    agent_id: str
    day: int
    bilateral_exposure: int | None = None
    total_exposure: int | None = None


__all__ = [
    "SystemView",
    "CounterpartyView",
    "InstrumentView",
    "TransactionView",
]
