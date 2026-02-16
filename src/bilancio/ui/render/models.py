"""View model classes for Bilancio UI rendering."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class BalanceItemView:
    """View model for a single balance sheet item."""

    category: str  # e.g., "Assets", "Liabilities"
    instrument: str  # e.g., "Cash", "Reserves", "Stock"
    amount: int | Decimal  # Quantity or monetary amount
    value: int | Decimal  # Monetary value


@dataclass
class AgentBalanceView:
    """View model for agent balance information."""

    agent_id: str
    agent_name: str
    agent_kind: str
    items: list[BalanceItemView]


@dataclass
class EventView:
    """View model for a single event."""

    kind: str
    title: str
    lines: list[str]
    icon: str
    raw_event: dict[str, Any]  # Original event data for reference


@dataclass
class DayEventsView:
    """View model for events in a simulation day."""

    day: int
    phases: dict[str, list[EventView]]  # phase -> list of events


@dataclass
class DaySummaryView:
    """View model for a complete day summary."""

    day: int
    events_view: DayEventsView
    agent_balances: list[AgentBalanceView]
