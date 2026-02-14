"""Dealer state synchronization, maturity management, and metrics capture.

Functions that synchronize state between the dealer subsystem and the main
system, manage ticket maturities, and capture metric snapshots.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System
    from bilancio.engines.dealer_integration import DealerSubsystem

from bilancio.domain.instruments.base import InstrumentKind
from bilancio.dealer.models import (
    DealerState,
    VBTState,
    Ticket,
)
from bilancio.dealer.kernel import recompute_dealer_state
from bilancio.dealer.metrics import (
    DealerSnapshot,
    TraderSnapshot,
    SystemStateSnapshot,
    compute_safety_margin,
    compute_saleable_value,
)


def _reassign_payable_owner(system: "System", contract_id: str, old_owner: str, new_owner: str) -> None:
    """Transfer payable ownership in the main system when bucket changes.

    Called during ingestion when a VBT/Dealer payable rolled over to a
    different maturity bucket. Updates asset_holder_id, holder_id, and
    agent asset_ids lists.
    """
    from bilancio.domain.instruments.credit import Payable

    payable = system.state.contracts.get(contract_id)
    if not isinstance(payable, Payable):
        return

    # Move from old owner's asset_ids to new owner's
    old_agent = system.state.agents.get(old_owner)
    new_agent = system.state.agents.get(new_owner)

    if old_agent and contract_id in old_agent.asset_ids:
        old_agent.asset_ids.remove(contract_id)
    if new_agent and contract_id not in new_agent.asset_ids:
        new_agent.asset_ids.append(contract_id)

    # Update the payable's ownership fields
    payable.asset_holder_id = new_owner
    payable.holder_id = None  # Reset secondary holder since asset_holder_id is now correct


def _ingest_new_payables(subsystem: "DealerSubsystem", system: "System", current_day: int) -> int:
    """Create tickets for new payables (from rollover) that have no ticket yet.

    Called during the dealer trading phase to pick up payables created by
    rollover that need to enter the dealer subsystem as tradable tickets.

    Args:
        subsystem: Dealer subsystem state
        system: Main System instance
        current_day: Current simulation day

    Returns:
        Number of new tickets created
    """
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.dealer_integration import _assign_bucket
    from bilancio.dealer.models import Ticket

    new_count = 0
    for contract_id, contract in list(system.state.contracts.items()):
        if not isinstance(contract, Payable):
            continue
        if contract_id in subsystem.payable_to_ticket:
            continue  # Already tracked
        if contract.due_day is None:
            continue

        remaining_tau = max(0, contract.due_day - current_day)
        if remaining_tau == 0:
            continue  # Already matured
        if contract.amount <= 0:
            continue  # Skip zero-face payables (Dirichlet rounding artifacts)

        # Skip defaulted issuers
        issuer = system.state.agents.get(contract.liability_issuer_id)
        if issuer is None or issuer.defaulted:
            continue

        # Create ticket
        ticket_id = f"TKT_{contract_id}"
        ticket = Ticket(
            id=ticket_id,
            issuer_id=contract.liability_issuer_id,
            owner_id=contract.effective_creditor,
            face=Decimal(contract.amount),
            maturity_day=contract.due_day,
            remaining_tau=remaining_tau,
            bucket_id="",
            serial=subsystem._ticket_serial_counter,
        )
        subsystem._ticket_serial_counter += 1
        ticket.bucket_id = _assign_bucket(ticket.remaining_tau, subsystem.bucket_configs)

        subsystem.tickets[ticket_id] = ticket
        subsystem.ticket_to_payable[ticket_id] = contract_id
        subsystem.payable_to_ticket[contract_id] = ticket_id

        # Assign to correct inventory based on owner.
        # For VBT/Dealer: use the ticket's bucket_id (not the owner's name suffix),
        # because rollover may have shifted the maturity to a different bucket.
        owner = ticket.owner_id
        target_bucket = ticket.bucket_id
        if owner.startswith("vbt_"):
            correct_owner = f"vbt_{target_bucket}"
            vbt = subsystem.vbts.get(target_bucket)
            if vbt:
                # Reassign ownership to the correct bucket's VBT
                ticket.owner_id = correct_owner
                vbt.inventory.append(ticket)
                # Update payable ownership in the main system
                if correct_owner != owner:
                    _reassign_payable_owner(system, contract_id, owner, correct_owner)
        elif owner.startswith("dealer_"):
            correct_owner = f"dealer_{target_bucket}"
            dealer = subsystem.dealers.get(target_bucket)
            if dealer:
                ticket.owner_id = correct_owner
                dealer.inventory.append(ticket)
                if correct_owner != owner:
                    _reassign_payable_owner(system, contract_id, owner, correct_owner)
        else:
            trader = subsystem.traders.get(owner)
            if trader:
                trader.tickets_owned.append(ticket)
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = ticket.issuer_id

        # Link obligation to issuer
        issuer_trader = subsystem.traders.get(ticket.issuer_id)
        if issuer_trader:
            issuer_trader.obligations.append(ticket)

        new_count += 1

    return new_count


def _pool_desk_cash(subsystem: "DealerSubsystem") -> None:
    """Redistribute cash equally across dealer desks and VBT desks.

    The three dealer desks (short/mid/long) are conceptually one firm with
    a shared balance sheet. Similarly for VBT desks. This function pools
    all desk cash and redistributes it equally so that:
    - Default losses (absorbed by the short desk at maturity) are shared
    - Settlement revenue (received by short desk) is shared
    - No interdealer payment is needed for bucket transitions

    Called between Phase 1 (maturity aging) and Phase 2 (quote recomputation).
    """
    # Pool dealer cash
    num_dealers = len(subsystem.dealers)
    if num_dealers > 0:
        total_dealer_cash = sum((d.cash for d in subsystem.dealers.values()), Decimal(0))
        per_desk = total_dealer_cash / Decimal(num_dealers)
        for dealer in subsystem.dealers.values():
            dealer.cash = per_desk

    # Pool VBT cash
    num_vbts = len(subsystem.vbts)
    if num_vbts > 0:
        total_vbt_cash = sum((v.cash for v in subsystem.vbts.values()), Decimal(0))
        per_desk = total_vbt_cash / Decimal(num_vbts)
        for vbt in subsystem.vbts.values():
            vbt.cash = per_desk


def _update_vbt_credit_mids(subsystem: "DealerSubsystem", current_day: int) -> None:
    """Update VBT mid prices to reflect the risk assessor's current default estimate.

    When a VBT pricing model is attached to the subsystem, delegates to it.
    Otherwise falls back to inline computation for backward compatibility.

    Called once per day before dealer quote recomputation.
    """
    if not subsystem.risk_assessor:
        return

    # Use system-wide default estimate (no specific issuer)
    p_default = subsystem.risk_assessor.estimate_default_prob("_system_", current_day)

    pricing_model = getattr(subsystem, "vbt_pricing_model", None)

    if pricing_model is not None:
        # Delegate to the VBT's own pricing heuristic
        initial_prior = subsystem.risk_assessor.params.initial_prior
        new_M = pricing_model.compute_mid(p_default, initial_prior)

        for bucket_id, vbt in subsystem.vbts.items():
            vbt.M = new_M
            base_O = subsystem.initial_spread_by_bucket.get(bucket_id, vbt.O)
            vbt.O = pricing_model.compute_spread(base_O, p_default)
            vbt.recompute_quotes()
    else:
        # Legacy inline logic (backward compatibility)
        raw_M = subsystem.outside_mid_ratio * (Decimal(1) - p_default)
        initial_prior = subsystem.risk_assessor.params.initial_prior
        initial_M = subsystem.outside_mid_ratio * (Decimal(1) - initial_prior)
        sens = subsystem.vbt_profile.mid_sensitivity
        new_M = initial_M + sens * (raw_M - initial_M)

        spread_sens = subsystem.vbt_profile.spread_sensitivity

        for bucket_id, vbt in subsystem.vbts.items():
            vbt.M = new_M

            # Update spread if spread_sensitivity > 0
            if spread_sens > 0:
                base_O = subsystem.initial_spread_by_bucket.get(bucket_id, vbt.O)
                vbt.O = base_O * (Decimal(1) + spread_sens * p_default)

            vbt.recompute_quotes()


def _sync_trader_cash_from_system(
    subsystem: "DealerSubsystem",
    system: "System",
) -> None:
    """Sync trader cash balances from the main system.

    Ensures traders have up-to-date cash balances for eligibility checks
    before a trading phase begins.
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    for trader_id, trader in subsystem.traders.items():
        trader.cash = _get_agent_cash(system, trader_id)


def _cleanup_orphaned_tickets(
    subsystem: "DealerSubsystem",
    system: "System",
) -> None:
    """Remove tickets whose payables were removed from the system.

    This can happen when agents default and get expelled (expel-agent mode).
    Cleans up inventories, trader holdings, and ticket/payable mappings.
    """
    from bilancio.domain.instruments.credit import Payable

    orphaned_ticket_ids = []
    for ticket_id, payable_id in subsystem.ticket_to_payable.items():
        payable = system.state.contracts.get(payable_id)
        if payable is None or not isinstance(payable, Payable):
            orphaned_ticket_ids.append(ticket_id)

    for ticket_id in orphaned_ticket_ids:
        ticket = subsystem.tickets.get(ticket_id)
        if ticket:
            # Remove from inventories
            bucket = ticket.bucket_id
            dealer = subsystem.dealers.get(bucket)
            vbt = subsystem.vbts.get(bucket)
            if dealer and ticket in dealer.inventory:
                dealer.inventory.remove(ticket)
            if vbt and ticket in vbt.inventory:
                vbt.inventory.remove(ticket)
            # Remove from trader holdings
            for trader in subsystem.traders.values():
                if ticket in trader.tickets_owned:
                    trader.tickets_owned.remove(ticket)
                if ticket in trader.obligations:
                    trader.obligations.remove(ticket)
            # Remove ticket
            del subsystem.tickets[ticket_id]

        # Clean up mappings
        if ticket_id in subsystem.ticket_to_payable:
            payable_id = subsystem.ticket_to_payable.pop(ticket_id)
            if payable_id in subsystem.payable_to_ticket:
                del subsystem.payable_to_ticket[payable_id]


def _remove_ticket_from_holdings(
    ticket: "Ticket",
    subsystem: "DealerSubsystem",
) -> None:
    """Remove a ticket from all dealer/VBT inventories and trader holdings."""
    bucket = ticket.bucket_id
    dealer = subsystem.dealers.get(bucket)
    vbt = subsystem.vbts.get(bucket)
    if dealer and ticket in dealer.inventory:
        dealer.inventory.remove(ticket)
    if vbt and ticket in vbt.inventory:
        vbt.inventory.remove(ticket)
    for trader in subsystem.traders.values():
        if ticket in trader.tickets_owned:
            trader.tickets_owned.remove(ticket)
        if ticket in trader.obligations:
            trader.obligations.remove(ticket)


def _update_ticket_maturities(
    subsystem: "DealerSubsystem",
    system: "System",
    current_day: int,
) -> None:
    """Update ticket maturities, reassign buckets, and remove matured tickets.

    For each ticket:
    - Updates remaining_tau based on current_day
    - Removes matured tickets (remaining_tau == 0) from all holdings
    - Reassigns tickets to new maturity buckets when they cross boundaries
    - Transfers dealer/VBT-owned tickets to the new bucket's desk
    """
    from bilancio.engines.dealer_integration import _assign_bucket

    matured_ticket_ids = []

    for ticket in subsystem.tickets.values():
        old_bucket = ticket.bucket_id
        ticket.remaining_tau = max(0, ticket.maturity_day - current_day)

        # Mark matured tickets for cleanup (remaining_tau = 0 means due today or past)
        if ticket.remaining_tau == 0:
            matured_ticket_ids.append(ticket.id)
            _remove_ticket_from_holdings(ticket, subsystem)
            continue

        new_bucket = _assign_bucket(ticket.remaining_tau, subsystem.bucket_configs)

        # If bucket changed, move ticket to new dealer/VBT inventory
        if new_bucket != old_bucket:
            _move_ticket_to_new_bucket(subsystem, system, ticket, old_bucket, new_bucket)

    # Clean up matured tickets to prevent unbounded memory growth
    for ticket_id in matured_ticket_ids:
        del subsystem.tickets[ticket_id]


def _move_ticket_to_new_bucket(
    subsystem: "DealerSubsystem",
    system: "System",
    ticket: Ticket,
    old_bucket: str,
    new_bucket: str,
) -> None:
    """Move a ticket from one maturity bucket to another.

    Removes the ticket from the old bucket's inventories, updates the
    bucket_id, and adds it to the new bucket. For dealer/VBT-owned tickets,
    this is an internal transfer within the same firm (shared cash pool)
    so no payment is needed.
    """
    # Remove from old bucket's inventories
    old_dealer = subsystem.dealers.get(old_bucket)
    old_vbt = subsystem.vbts.get(old_bucket)
    if old_dealer and ticket in old_dealer.inventory:
        old_dealer.inventory.remove(ticket)
    if old_vbt and ticket in old_vbt.inventory:
        old_vbt.inventory.remove(ticket)

    # Add to new bucket's appropriate inventory
    ticket.bucket_id = new_bucket
    new_dealer = subsystem.dealers.get(new_bucket)
    new_vbt = subsystem.vbts.get(new_bucket)

    # Reassign dealer/VBT-owned tickets to the new bucket's desk.
    # This is an internal transfer within the same firm (shared cash pool),
    # so no payment is needed — just move inventory and update ownership.
    if ticket.owner_id.startswith("dealer_") and new_dealer:
        old_owner = ticket.owner_id
        new_owner = f"dealer_{new_bucket}"
        ticket.owner_id = new_owner
        new_dealer.inventory.append(ticket)
        if old_owner != new_owner:
            payable_id = subsystem.ticket_to_payable.get(ticket.id)
            if payable_id:
                _reassign_payable_owner(system, payable_id, old_owner, new_owner)
    elif ticket.owner_id.startswith("vbt_") and new_vbt:
        old_owner = ticket.owner_id
        new_owner = f"vbt_{new_bucket}"
        ticket.owner_id = new_owner
        new_vbt.inventory.append(ticket)
        if old_owner != new_owner:
            payable_id = subsystem.ticket_to_payable.get(ticket.id)
            if payable_id:
                _reassign_payable_owner(system, payable_id, old_owner, new_owner)
    # Trader-owned tickets: bucket_id updated above, ticket stays
    # in trader.tickets_owned (no inventory move needed)


def _sync_payable_ownership(
    subsystem: "DealerSubsystem",
    system: "System",
) -> None:
    """Update Payable ownership in the main system to match ticket ownership.

    For each ticket whose owner changed during trading, updates the
    corresponding Payable's holder_id and agent asset_ids lists.

    Note: Payable has two holder fields:
    - asset_holder_id: original creditor (from base Instrument class)
    - holder_id: secondary market holder (specific to Payable)
    Settlement uses effective_creditor which returns holder_id if set,
    else asset_holder_id.
    """
    from bilancio.domain.instruments.credit import Payable

    for ticket_id, ticket in subsystem.tickets.items():
        payable_id = subsystem.ticket_to_payable.get(ticket_id)
        if not payable_id:
            continue

        payable = system.state.contracts.get(payable_id)
        if not isinstance(payable, Payable):
            continue

        # Check if ownership changed (compare with effective_creditor)
        current_holder = payable.effective_creditor
        new_holder = ticket.owner_id

        if current_holder != new_holder:
            # Update agent asset_ids lists
            old_holder_agent = system.state.agents.get(current_holder)
            new_holder_agent = system.state.agents.get(new_holder)

            if old_holder_agent and payable_id in old_holder_agent.asset_ids:
                old_holder_agent.asset_ids.remove(payable_id)

            if new_holder_agent and payable_id not in new_holder_agent.asset_ids:
                new_holder_agent.asset_ids.append(payable_id)

            # Update payable's holder_id (secondary market holder)
            # Keep asset_holder_id as the original creditor
            payable.holder_id = new_holder

            # Log the transfer
            system.log(
                "ClaimTransferredDealer",
                payable_id=payable_id,
                from_holder=current_holder,
                to_holder=new_holder,
                amount=payable.amount,
                due_day=payable.due_day
            )


def _sync_trader_cash_to_system(
    subsystem: "DealerSubsystem",
    system: "System",
) -> None:
    """Sync trader cash balances from dealer subsystem back to main system.

    Compares trader.cash in the dealer subsystem to the actual cash in the
    main system and applies the delta as minting (if trader gained cash)
    or burning (if trader spent cash).
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    for trader_id, trader in subsystem.traders.items():
        main_system_cash = _get_agent_cash(system, trader_id)
        dealer_cash = trader.cash
        delta = dealer_cash - main_system_cash

        if delta > 0:
            # Trader gained cash from selling tickets - mint cash to them
            # This represents money coming from outside the system (dealer/VBT)
            system.mint_cash(to_agent_id=trader_id, amount=round(delta))
        elif delta < 0:
            # Trader spent cash buying tickets - need to reduce their cash
            # Find and reduce their cash contracts
            agent = system.state.agents.get(trader_id)
            if agent:
                # Cap burn at available cash to prevent negative balances
                remaining_to_burn = min(abs(delta), main_system_cash)
                for contract_id in list(agent.asset_ids):
                    if remaining_to_burn <= 0:
                        break
                    contract = system.state.contracts.get(contract_id)
                    if contract and contract.kind == InstrumentKind.CASH:
                        if contract.amount <= remaining_to_burn:
                            # Remove entire contract
                            remaining_to_burn -= contract.amount
                            agent.asset_ids.remove(contract_id)
                            del system.state.contracts[contract_id]
                        else:
                            # Reduce contract amount
                            contract.amount -= round(remaining_to_burn)
                            remaining_to_burn = Decimal(0)


def _capture_dealer_snapshots(
    subsystem: "DealerSubsystem",
    current_day: int
) -> None:
    """
    Capture dealer state snapshots for metrics (Section 8.1).

    Records inventory, cash, quotes, and mark-to-mid equity for each bucket.

    Plan 022 extensions:
    - max_capacity, is_at_zero for capacity tracking
    - hit_vbt_this_step from trade records
    - total_system_face, dealer_share_pct for system context
    """
    # Check if any VBT trades happened this step (for hit_vbt_this_step flag)
    # Look at trades from today that are passthroughs
    vbt_used_buckets = set()
    for trade in subsystem.metrics.trades:
        if trade.day == current_day and trade.is_passthrough:
            vbt_used_buckets.add(trade.bucket)

    for bucket_id, dealer in subsystem.dealers.items():
        vbt = subsystem.vbts[bucket_id]

        # Calculate total system face value for this bucket
        total_face = Decimal(0)
        dealer_face = Decimal(0)
        for trader in subsystem.traders.values():
            for ticket in trader.tickets_owned:
                if ticket.bucket_id == bucket_id:
                    total_face += subsystem.params.S
        # Dealer inventory contribution
        dealer_face = Decimal(dealer.a) * subsystem.params.S

        # Dealer share percentage
        dealer_share = (dealer_face / total_face * 100) if total_face > 0 else Decimal(0)

        snapshot = DealerSnapshot(
            day=current_day,
            bucket=bucket_id,
            inventory=dealer.a,
            cash=dealer.cash,
            bid=dealer.bid,
            ask=dealer.ask,
            midline=dealer.midline,
            vbt_mid=vbt.M,
            vbt_spread=vbt.O,
            ticket_size=subsystem.params.S,
            # Plan 022 extensions
            max_capacity=int(dealer.X_star),
            is_at_zero=(dealer.a == 0),
            hit_vbt_this_step=(bucket_id in vbt_used_buckets),
            total_system_face=total_face,
            dealer_share_pct=dealer_share,
        )
        subsystem.metrics.dealer_snapshots.append(snapshot)


def _capture_trader_snapshots(
    subsystem: "DealerSubsystem",
    current_day: int
) -> None:
    """
    Capture trader state snapshots for metrics (Section 8.1).

    Records cash, tickets, obligations, and safety margins.
    """
    for trader_id, trader in subsystem.traders.items():
        # Calculate safety margin
        safety_margin = compute_safety_margin(
            cash=trader.cash,
            tickets_held=trader.tickets_owned,
            obligations=trader.obligations,
            dealers=subsystem.dealers,
            ticket_size=subsystem.params.S
        )

        # Calculate saleable value
        saleable = compute_saleable_value(trader.tickets_owned, subsystem.dealers)

        # Calculate remaining obligations
        obligations_remaining = sum((t.face for t in trader.obligations), Decimal(0))

        snapshot = TraderSnapshot(
            day=current_day,
            trader_id=trader_id,
            cash=trader.cash,
            tickets_held_count=len(trader.tickets_owned),
            tickets_held_ids=[t.id for t in trader.tickets_owned],
            total_face_held=sum((t.face for t in trader.tickets_owned), Decimal(0)),
            obligations_remaining=obligations_remaining,
            saleable_value=saleable,
            safety_margin=safety_margin,
            defaulted=trader.defaulted,
        )
        subsystem.metrics.trader_snapshots.append(snapshot)


def _capture_system_state_snapshot(
    subsystem: "DealerSubsystem",
    current_day: int
) -> None:
    """
    Capture system-level state snapshot for metrics (Plan 022 - Phase 4).

    Records total face value, cash, and debt-to-money ratio across the system.
    """
    # Calculate total face value by bucket
    face_by_bucket: Dict[str, Decimal] = {}
    total_face = Decimal(0)

    for bucket_id in subsystem.dealers.keys():
        face_by_bucket[bucket_id] = Decimal(0)

    # Sum face value from all tickets held by traders
    for trader in subsystem.traders.values():
        for ticket in trader.tickets_owned:
            bucket = ticket.bucket_id
            face = subsystem.params.S
            face_by_bucket[bucket] = face_by_bucket.get(bucket, Decimal(0)) + face
            total_face += face

    # Also add dealer inventory
    for bucket_id, dealer in subsystem.dealers.items():
        dealer_face = Decimal(dealer.a) * subsystem.params.S
        face_by_bucket[bucket_id] = face_by_bucket.get(bucket_id, Decimal(0)) + dealer_face
        total_face += dealer_face

    # Calculate total cash in system
    total_cash = Decimal(0)
    for trader in subsystem.traders.values():
        total_cash += trader.cash
    for dealer in subsystem.dealers.values():
        total_cash += dealer.cash
    for vbt in subsystem.vbts.values():
        total_cash += vbt.cash

    # Create snapshot
    snapshot = SystemStateSnapshot(
        run_id="",  # Will be set at export time
        regime="",  # Will be set at export time
        day=current_day,
        total_face_value=total_face,
        face_bucket_short=face_by_bucket.get("short", Decimal(0)),
        face_bucket_mid=face_by_bucket.get("mid", Decimal(0)),
        face_bucket_long=face_by_bucket.get("long", Decimal(0)),
        total_cash=total_cash,
    )
    subsystem.metrics.system_state_snapshots.append(snapshot)
