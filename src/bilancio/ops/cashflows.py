"""Cash flow operations and data structures."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

# Placeholder types - these should be replaced with actual implementations from the domain layer
Money = Decimal  # Placeholder for Money type
TimeCoordinate = datetime  # Placeholder for TimeCoordinate type
Agent = str  # Placeholder for Agent type


@dataclass
class CashFlow:
    """Represents a single cash flow between two agents."""

    amount: Money
    time: TimeCoordinate
    payer: Agent
    payee: Agent


class CashFlowStream:
    """Manages a collection of cash flows with basic operations."""

    def __init__(self, flows: list[CashFlow] | None = None):
        """Initialize with an optional list of cash flows."""
        self.flows = flows or []

    def add_flow(self, flow: CashFlow) -> None:
        """Add a cash flow to the stream."""
        self.flows.append(flow)

    def get_flows_at_time(self, time: TimeCoordinate) -> list[CashFlow]:
        """Get all cash flows occurring at a specific time."""
        return [flow for flow in self.flows if flow.time == time]

    def get_all_flows(self) -> list[CashFlow]:
        """Get all cash flows in the stream."""
        return self.flows.copy()

    def __len__(self) -> int:
        """Return the number of flows in the stream."""
        return len(self.flows)
