"""Contextual view objects returned by InformationService queries.

Frozen dataclasses representing information snapshots at a point in time.
Fields are ``None`` when the observer lacks access (AccessLevel.NONE).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class SystemView:
    """Snapshot of system-wide conditions."""
    day: int
    aggregate_default_rate: Optional[Decimal] = None
    system_liquidity: Optional[int] = None


@dataclass(frozen=True)
class CounterpartyView:
    """Snapshot of a single counterparty's observable state."""
    agent_id: str
    day: int
    cash: Optional[int] = None
    obligations: Optional[int] = None
    net_worth: Optional[int] = None
    default_probability: Optional[Decimal] = None


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
    bilateral_exposure: Optional[int] = None
    total_exposure: Optional[int] = None


__all__ = [
    "SystemView",
    "CounterpartyView",
    "InstrumentView",
    "TransactionView",
]
