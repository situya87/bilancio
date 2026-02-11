"""
Bridge between main Bilancio system and dealer module.

This module provides conversion functions between the main system's Payable
contracts and the dealer module's Ticket instruments. It enables:

- Converting Payable contracts to tradeable Tickets
- Mapping ticket ownership back to Payable holders
- Applying trade results to update contract ownership

The bridge maintains consistency between the two representations while keeping
the systems decoupled.
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Set, Optional

from bilancio.dealer.models import Ticket, TicketId, BucketConfig, DEFAULT_BUCKETS
from bilancio.domain.instruments.credit import Payable
from bilancio.core.ids import new_id


def assign_bucket(remaining_tau: int, bucket_configs: List[BucketConfig]) -> str:
    """
    Determine which maturity bucket a ticket belongs to based on remaining maturity.

    Buckets partition tickets by remaining days to maturity τ (tau). Each bucket
    defines an inclusive range [tau_min, tau_max], with tau_max=None representing
    unbounded (τ ≥ tau_min).

    Args:
        remaining_tau: Remaining days until maturity (due_day - current_day)
        bucket_configs: List of bucket configurations to check

    Returns:
        bucket_id: The name of the matching bucket (e.g., "short", "mid", "long")

    Raises:
        ValueError: If remaining_tau doesn't match any bucket

    Examples:
        >>> assign_bucket(2, DEFAULT_BUCKETS)
        'short'
        >>> assign_bucket(5, DEFAULT_BUCKETS)
        'mid'
        >>> assign_bucket(15, DEFAULT_BUCKETS)
        'long'
    """
    for bucket in bucket_configs:
        if remaining_tau < bucket.tau_min:
            continue
        if bucket.tau_max is None or remaining_tau <= bucket.tau_max:
            return bucket.name

    raise ValueError(
        f"No bucket found for remaining_tau={remaining_tau}. "
        f"Available buckets: {[b.name for b in bucket_configs]}"
    )


def payables_to_tickets(
    payables: Dict[str, Payable],
    current_day: int,
    bucket_configs: List[BucketConfig],
    ticket_size: Decimal = Decimal(1),
) -> Tuple[Dict[TicketId, Ticket], Dict[str, List[str]]]:
    """
    Convert outstanding Payable contracts to tradeable Tickets.

    Each payable is "ticketized" into (face_value / ticket_size) tickets.
    Tickets are assigned to maturity buckets based on remaining days to maturity.

    Args:
        payables: Dictionary mapping payable_id to Payable contracts
        current_day: Current simulation day
        bucket_configs: Maturity bucket configurations
        ticket_size: Face value per ticket (default: 1)

    Returns:
        Tuple of:
        - ticket_registry: Dict mapping ticket_id to Ticket objects
        - payable_to_tickets: Dict mapping payable_id to list of ticket_ids

    Raises:
        ValueError: If payable amount is not divisible by ticket_size
        ValueError: If payable has already matured (due_day <= current_day)

    Notes:
        - Ticket IDs are generated using new_id() for uniqueness
        - Serial numbers are assigned sequentially for deterministic tie-breaking
        - Amount is converted from minor units to major units by dividing by 100
    """
    ticket_registry: Dict[TicketId, Ticket] = {}
    payable_to_tickets: Dict[str, List[str]] = {}

    for payable_id, payable in payables.items():
        # Validate payable hasn't matured
        if payable.due_day is None:
            raise ValueError(f"Payable {payable_id} has no due_day")

        if payable.due_day <= current_day:
            raise ValueError(
                f"Payable {payable_id} has already matured "
                f"(due_day={payable.due_day}, current_day={current_day})"
            )

        # Convert amount from minor units to major units (cents to dollars)
        # Payable.amount is in minor units (cents), tickets use major units (dollars)
        face_value = Decimal(payable.amount) / Decimal(100)

        # Calculate number of tickets
        num_tickets_dec = face_value / ticket_size
        if num_tickets_dec != int(num_tickets_dec):
            raise ValueError(
                f"Payable {payable_id} face value {face_value} is not divisible "
                f"by ticket size {ticket_size}"
            )
        num_tickets = int(num_tickets_dec)

        # Calculate remaining maturity and assign bucket
        remaining_tau = payable.due_day - current_day
        bucket_id = assign_bucket(remaining_tau, bucket_configs)

        # Create tickets for this payable
        ticket_ids: List[str] = []
        for serial in range(num_tickets):
            ticket_id = new_id()
            ticket = Ticket(
                id=ticket_id,
                issuer_id=payable.liability_issuer_id,
                owner_id=payable.asset_holder_id,
                face=ticket_size,
                maturity_day=payable.due_day,
                remaining_tau=remaining_tau,
                bucket_id=bucket_id,
                serial=serial,
            )
            ticket_registry[ticket_id] = ticket
            ticket_ids.append(ticket_id)

        payable_to_tickets[payable_id] = ticket_ids

    return ticket_registry, payable_to_tickets


def tickets_to_trader_holdings(
    tickets: Dict[TicketId, Ticket],
    agent_ids: Set[str],
) -> Dict[str, List[Ticket]]:
    """
    Group tickets by their owner for agents that are traders.

    This function filters tickets owned by specific agents and groups them
    by owner. Useful for initializing trader holdings from a ticket registry.

    Args:
        tickets: Dictionary mapping ticket_id to Ticket objects
        agent_ids: Set of agent IDs to include (typically ring traders)

    Returns:
        Dict mapping agent_id to list of tickets they hold as assets

    Notes:
        - Only includes tickets owned by agents in agent_ids
        - Returns empty list for agents with no tickets
        - Tickets are references to original objects (not copies)
    """
    holdings: Dict[str, List[Ticket]] = {agent_id: [] for agent_id in agent_ids}

    for ticket in tickets.values():
        if ticket.owner_id in agent_ids:
            holdings[ticket.owner_id].append(ticket)

    return holdings


def apply_trade_results_to_payables(
    payables: Dict[str, Payable],
    ticket_to_payable: Dict[str, str],
    trade_results: List[Dict[str, str]],
) -> None:
    """
    Apply dealer trade results back to main system by updating Payable holders.

    After trading in the dealer system, ticket ownership may change. This function
    propagates those changes back to the original Payable contracts by updating
    the asset_holder_id field.

    Args:
        payables: Dictionary mapping payable_id to Payable contracts (mutated in-place)
        ticket_to_payable: Dictionary mapping ticket_id to payable_id
        trade_results: List of trade result dicts with structure:
            {
                'ticket_id': str,
                'old_owner': str,
                'new_owner': str,
                'price': Decimal,
                ...
            }

    Notes:
        - Payables are modified in-place
        - Only tickets that map to payables are processed
        - If multiple tickets from same payable are traded, updates holder multiple times
        - Assumes all tickets from a payable have same owner (enforced by ticketization)

    Raises:
        KeyError: If ticket_id or payable_id not found in mappings
    """
    for result in trade_results:
        ticket_id = result['ticket_id']
        new_owner = result['new_owner']

        # Find the corresponding payable
        if ticket_id not in ticket_to_payable:
            # Ticket may not correspond to a payable (e.g., dealer-created)
            continue

        payable_id = ticket_to_payable[ticket_id]
        if payable_id not in payables:
            raise KeyError(f"Payable {payable_id} not found in payables dict")

        # Update the payable's secondary market holder (not asset_holder_id)
        # holder_id tracks the current owner after secondary market transfers
        # asset_holder_id should remain the original creditor
        payable = payables[payable_id]
        payable.holder_id = new_owner
