"""Network graph data extraction from Bilancio system state.

This module provides functions to extract network graph data representing
the relationships between agents through financial instruments/contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bilancio.engines.system import System


@dataclass
class NetworkNode:
    """Represents an agent node in the network graph."""

    id: str
    name: str
    kind: str


@dataclass
class NetworkEdge:
    """Represents a financial relationship (contract) between agents."""

    source: str
    target: str
    amount: int
    instrument_type: str
    contract_id: str


@dataclass
class NetworkSnapshot:
    """Complete network graph snapshot for a specific day."""

    day: int
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


def build_network_data(
    system: System, day: int, instrument_types: list[str] | None = None
) -> NetworkSnapshot:
    """Build network graph data from system state for a specific day.

    Args:
        system: The System instance containing state and agents
        day: The day number for this snapshot
        instrument_types: Optional list of instrument types to filter by.
                         If None, all contracts are included.

    Returns:
        NetworkSnapshot containing nodes (agents) and edges (contracts)
    """
    # Extract all agents as NetworkNode objects
    nodes = []
    for agent in system.state.agents.values():
        nodes.append(NetworkNode(id=agent.id, name=agent.name, kind=agent.kind))

    # Extract contracts as NetworkEdge objects
    edges = []
    for contract in system.state.contracts.values():
        # Apply instrument type filter if provided
        if instrument_types is not None and contract.kind not in instrument_types:
            continue

        edges.append(
            NetworkEdge(
                source=contract.asset_holder_id,
                target=contract.liability_issuer_id,
                amount=contract.amount,
                instrument_type=contract.kind,
                contract_id=contract.id,
            )
        )

    # Also check for dealer tickets if dealer subsystem exists
    if hasattr(system.state, "dealer_subsystem") and system.state.dealer_subsystem is not None:
        dealer_subsystem = system.state.dealer_subsystem
        if hasattr(dealer_subsystem, "tickets"):
            for ticket in dealer_subsystem.tickets.values():
                # Tickets have: issuer_id (debtor), owner_id (creditor), face (amount)
                ticket_type = "dealer_ticket"

                # Apply instrument type filter if provided
                if instrument_types is not None and ticket_type not in instrument_types:
                    continue

                # Convert Decimal face value to int (minor units)
                # Assuming face is in major units, multiply by 100 for minor units
                amount_minor = int(float(ticket.face) * 100)

                edges.append(
                    NetworkEdge(
                        source=ticket.owner_id,  # Creditor/holder as source (asset holder)
                        target=ticket.issuer_id,  # Debtor as target (liability issuer)
                        amount=amount_minor,
                        instrument_type=ticket_type,
                        contract_id=ticket.id,
                    )
                )

    return NetworkSnapshot(day=day, nodes=nodes, edges=edges)


def build_network_time_series(
    system: System, days: list[int], instrument_types: list[str] | None = None
) -> list[NetworkSnapshot]:
    """Build network graph data for multiple days.

    Args:
        system: The System instance containing state and agents
        days: List of day numbers to create snapshots for
        instrument_types: Optional list of instrument types to filter by.
                         If None, all contracts are included.

    Returns:
        List of NetworkSnapshot objects, one for each day
    """
    snapshots = []
    for day in days:
        snapshot = build_network_data(system, day, instrument_types)
        snapshots.append(snapshot)

    return snapshots


def _snapshot_to_dict(snapshot: NetworkSnapshot) -> dict[str, Any]:
    """Convert NetworkSnapshot to JSON-serializable dictionary.

    Args:
        snapshot: The NetworkSnapshot to convert

    Returns:
        Dictionary representation of the snapshot
    """
    return asdict(snapshot)
