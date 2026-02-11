"""
Dealer subsystem integration for the main simulation engine.

This module provides a bridge between the main bilancio simulation engine
(System/Payables) and the dealer module (Tickets/States). It wraps the dealer
module's components to provide a clean interface for:

1. Converting Payables to Tickets for trading
2. Running dealer trading phases within the main simulation loop
3. Syncing trade results back to the main system

Architecture:
    Main System (Payables) <--> DealerSubsystem <--> Dealer Module (Tickets)

    - DealerSubsystem: Maintains parallel state (tickets, trader states)
    - Bridge functions: Convert between Payable and Ticket representations
    - Integration functions: Initialize, run trading, sync results

The dealer subsystem operates as a secondary market where agents can trade
their existing claims (Payables) to manage liquidity needs. Trades are
executed at market-determined prices through a dealer ring with value-based
traders (VBTs) providing outside liquidity.

References:
    - Dealer module docs: docs/dealer_ring.md
    - Dealer specification: Full specification document
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Any
import random

from bilancio.core.ids import AgentId, InstrId
from bilancio.dealer.models import (
    DealerState,
    VBTState,
    TraderState,
    Ticket,
    BucketConfig,
    DEFAULT_BUCKETS,
    TicketId,
)
from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.trading import TradeExecutor
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.dealer.metrics import (
    RunMetrics,
    TradeRecord,
    DealerSnapshot,
    TraderSnapshot,
    TicketOutcome,
    SystemStateSnapshot,  # Plan 022
    compute_safety_margin,
    compute_saleable_value,
)
from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams


@dataclass
class DealerSubsystem:
    """
    Wrapper that adapts dealer module for main simulation engine.

    This dataclass maintains all state needed for dealer trading operations,
    providing a clean boundary between the main simulation and the dealer
    subsystem.

    State Management:
        - Tickets: Tradable representations of Payables
        - Dealers: Per-bucket market makers with inventory and quotes
        - VBTs: Per-bucket outside liquidity providers
        - Traders: Per-agent trading states with single-issuer constraint

    Mapping Tables:
        - ticket_to_payable: Links tickets back to their source Payables
        - payable_to_ticket: Links Payables to their tradable tickets

    Trading Infrastructure:
        - executor: Handles trade execution and balance sheet updates
        - params: Kernel parameters for pricing computations
        - bucket_configs: Maturity-based grouping definitions

    Attributes:
        dealers: Per-bucket dealer states (bucket_id -> DealerState)
        vbts: Per-bucket VBT states (bucket_id -> VBTState)
        traders: Per-agent trader states (agent_id -> TraderState)
        tickets: All tradable tickets (ticket_id -> Ticket)
        ticket_to_payable: Mapping from ticket IDs to Payable contract IDs
        payable_to_ticket: Mapping from Payable contract IDs to ticket IDs
        bucket_configs: Maturity bucket configurations
        params: Kernel parameters for dealer pricing
        executor: Trade execution engine
        enabled: Whether dealer trading is active
        rng: Random number generator for order flow
    """

    dealers: Dict[str, DealerState] = field(default_factory=dict)
    vbts: Dict[str, VBTState] = field(default_factory=dict)
    traders: Dict[AgentId, TraderState] = field(default_factory=dict)
    tickets: Dict[TicketId, Ticket] = field(default_factory=dict)
    ticket_to_payable: Dict[TicketId, InstrId] = field(default_factory=dict)
    payable_to_ticket: Dict[InstrId, TicketId] = field(default_factory=dict)
    bucket_configs: List[BucketConfig] = field(default_factory=lambda: list(DEFAULT_BUCKETS))
    params: KernelParams = field(default_factory=lambda: KernelParams())
    executor: Optional[TradeExecutor] = None
    enabled: bool = True
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    # Face value for scaling eligibility thresholds
    face_value: Decimal = Decimal(1)

    # Section 8 Metrics (functional dealer analysis)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    # Risk assessment module (optional)
    risk_assessor: RiskAssessor | None = None

    # Serial counter for ticket IDs (continues from initialization)
    _ticket_serial_counter: int = 0


def initialize_dealer_subsystem(
    system,
    dealer_config: DealerRingConfig,
    current_day: int = 0,
    risk_params: RiskAssessmentParams | None = None,
) -> DealerSubsystem:
    """
    Initialize dealer subsystem from system state and configuration.

    This function performs the initial setup of the dealer subsystem:

    1. Create dealer/VBT agents in main system:
       - Add actual Household agents for each dealer and VBT
       - This allows proper ownership tracking in the main system

    2. Convert existing Payables to Tickets:
       - Extract all payables from system contracts
       - Create corresponding Ticket objects with maturity info
       - Assign tickets to maturity buckets
       - Build bidirectional mappings

    3. Initialize market makers:
       - Create DealerState for each bucket with initial capital
       - Create VBTState for each bucket with anchors
       - Allocate initial ticket inventory based on dealer_share/vbt_share

    4. Initialize traders:
       - Create TraderState for each household agent
       - Set initial cash from agent balance sheets
       - Link tickets to trader ownership

    5. Compute initial quotes:
       - Run kernel computation for each dealer
       - Generate initial bid/ask spreads

    Capital Allocation (NEW OUTSIDE MONEY):
        dealer_share: Fraction of system cash injected as dealer capital (e.g., 0.25)
        vbt_share: Fraction of system cash injected as VBT capital (e.g., 0.50)
        NOTE: Traders keep 100% of their tickets. Dealer/VBT start with EMPTY
              inventory and build it by BUYING from traders who want to sell.
              This is NEW MONEY from outside the system, not taken from traders.

    Args:
        system: Main System instance with agents and contracts
        dealer_config: Configuration for dealer subsystem
        current_day: Current simulation day for maturity calculations

    Returns:
        Initialized DealerSubsystem ready for trading

    Raises:
        ValueError: If configuration is invalid or system state inconsistent

    Example:
        >>> system = System()
        >>> # ... set up agents and create payables ...
        >>> config = DealerRingConfig(
        ...     ticket_size=Decimal(1),
        ...     dealer_share=Decimal("0.25"),
        ...     vbt_share=Decimal("0.50"),
        ... )
        >>> subsystem = initialize_dealer_subsystem(system, config)
        >>> # Now ready for run_dealer_trading_phase()
    """
    from bilancio.domain.agents import Dealer, VBT

    subsystem = DealerSubsystem(
        bucket_configs=dealer_config.buckets,
        params=KernelParams(S=dealer_config.ticket_size),
        rng=random.Random(dealer_config.seed),
    )

    # Initialize risk assessor if params provided
    if risk_params:
        subsystem.risk_assessor = RiskAssessor(risk_params)

    # Step 0: Create dealer/VBT agents in main system
    # These agents allow proper ownership tracking when claims transfer to dealers
    for bucket_config in dealer_config.buckets:
        bucket_id = bucket_config.name
        dealer_agent_id = f"dealer_{bucket_id}"
        vbt_agent_id = f"vbt_{bucket_id}"

        # Create dealer agent if not exists
        if dealer_agent_id not in system.state.agents:
            dealer_agent = Dealer(
                id=dealer_agent_id,
                name=f"Dealer ({bucket_id})",
            )
            system.state.agents[dealer_agent_id] = dealer_agent

        # Create VBT agent if not exists
        if vbt_agent_id not in system.state.agents:
            vbt_agent = VBT(
                id=vbt_agent_id,
                name=f"VBT ({bucket_id})",
            )
            system.state.agents[vbt_agent_id] = vbt_agent

    # Initialize trade executor
    subsystem.executor = TradeExecutor(subsystem.params, subsystem.rng)

    # Step 1: Convert Payables to Tickets
    from bilancio.domain.instruments.credit import Payable

    serial_counter = 0
    for contract_id, contract in system.state.contracts.items():
        if not isinstance(contract, Payable):
            continue

        # Create ticket from payable
        ticket_id = f"TKT_{contract_id}"
        remaining_tau = max(0, contract.due_day - current_day)

        ticket = Ticket(
            id=ticket_id,
            issuer_id=contract.liability_issuer_id,  # Debtor
            owner_id=contract.effective_creditor,    # Current creditor
            face=Decimal(contract.amount),
            maturity_day=contract.due_day,
            remaining_tau=remaining_tau,
            bucket_id="",  # Will be assigned below
            serial=serial_counter,
        )
        serial_counter += 1

        # Assign to bucket
        ticket.bucket_id = _assign_bucket(ticket.remaining_tau, subsystem.bucket_configs)

        # Register ticket
        subsystem.tickets[ticket_id] = ticket
        subsystem.ticket_to_payable[ticket_id] = contract_id
        subsystem.payable_to_ticket[contract_id] = ticket_id

    # Step 2: Calculate total system cash for capital allocation
    # Dealer and VBT get NEW CASH from outside the system (not taken from traders)
    total_system_cash = Decimal(0)
    for agent_id, agent in system.state.agents.items():
        # Skip dealer/VBT agents we just created
        if agent.kind in ("dealer", "vbt"):
            continue
        total_system_cash += _get_agent_cash(system, agent_id)

    # Calculate dealer and VBT capital (NEW outside money)
    # Split evenly across buckets
    num_buckets = len(subsystem.bucket_configs)
    dealer_capital_per_bucket = (total_system_cash * dealer_config.dealer_share) / num_buckets
    vbt_capital_per_bucket = (total_system_cash * dealer_config.vbt_share) / num_buckets

    # Step 3: Initialize market makers
    for bucket_config in subsystem.bucket_configs:
        bucket_id = bucket_config.name

        # Get anchor prices from config
        M, O = dealer_config.vbt_anchors.get(
            bucket_id,
            (Decimal(1), Decimal("0.30"))
        )

        # Create dealer state with NEW capital (no inventory yet)
        dealer = DealerState(
            bucket_id=bucket_id,
            agent_id=f"dealer_{bucket_id}",
            inventory=[],  # Empty! Dealers build inventory by buying from traders
            cash=dealer_capital_per_bucket,  # NEW outside money
        )
        subsystem.dealers[bucket_id] = dealer

        # Create VBT state with NEW capital (no inventory yet)
        vbt = VBTState(
            bucket_id=bucket_id,
            agent_id=f"vbt_{bucket_id}",
            M=M,
            O=O,
            phi_M=dealer_config.phi_M,
            phi_O=dealer_config.phi_O,
            clip_nonneg_B=dealer_config.clip_nonneg_B,
            inventory=[],  # Empty! VBTs build inventory by buying from traders
            cash=vbt_capital_per_bucket,  # NEW outside money
        )
        vbt.recompute_quotes()
        subsystem.vbts[bucket_id] = vbt

        # NOTE: Traders keep 100% of their tickets (no allocation to dealer/VBT)
        # Dealer/VBT will acquire inventory by buying from traders during trading

        # Run kernel to compute initial quotes
        recompute_dealer_state(dealer, vbt, subsystem.params)

        # Capture initial equity for P&L calculation (Section 8.2)
        # E_0^(b) = C_0^(b) + M * a_0^(b) * S
        initial_equity = dealer.cash + vbt.M * dealer.a * subsystem.params.S
        subsystem.metrics.initial_equity_by_bucket[bucket_id] = initial_equity

    # Step 3: Initialize traders (households only for now)
    for agent_id, agent in system.state.agents.items():
        if agent.kind != "household":
            continue

        # Calculate trader's cash from their cash holdings in main system
        trader_cash = _get_agent_cash(system, agent_id)

        trader = TraderState(
            agent_id=agent_id,
            cash=trader_cash,
            tickets_owned=[],
            obligations=[],
            asset_issuer_id=None,
        )

        # Link trader to their tickets
        for ticket in subsystem.tickets.values():
            if ticket.owner_id == agent_id:
                trader.tickets_owned.append(ticket)
                # Set asset_issuer_id based on first ticket held
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = ticket.issuer_id

            if ticket.issuer_id == agent_id:
                trader.obligations.append(ticket)

        subsystem.traders[agent_id] = trader

    # Step 4: Capture initial debt-to-money ratio (Plan 020 - Phase B)
    # This is a key control variable - results only make sense given this ratio
    from bilancio.domain.instruments.credit import Payable as PayableClass

    # Sum all payable amounts (total debt in system)
    total_debt = Decimal(0)
    for contract in system.state.contracts.values():
        if isinstance(contract, PayableClass):
            total_debt += Decimal(contract.amount)

    # Sum all cash holdings (total money in system, excluding dealer/VBT)
    total_money = Decimal(0)
    for agent_id_iter, agent in system.state.agents.items():
        if agent.kind not in ("dealer", "vbt"):
            total_money += _get_agent_cash(system, agent_id_iter)

    # Store in metrics
    subsystem.metrics.initial_total_debt = total_debt
    subsystem.metrics.initial_total_money = total_money

    subsystem._ticket_serial_counter = serial_counter

    return subsystem


def initialize_balanced_dealer_subsystem(
    system,
    dealer_config: DealerRingConfig,
    face_value: Decimal = Decimal("20"),
    outside_mid_ratio: Decimal = Decimal("0.75"),
    big_entity_share: Decimal = Decimal("0.25"),  # DEPRECATED
    vbt_share_per_bucket: Decimal = Decimal("0.25"),
    dealer_share_per_bucket: Decimal = Decimal("0.125"),
    mode: str = "active",
    current_day: int = 0,
    risk_params: RiskAssessmentParams | None = None,
) -> DealerSubsystem:
    """
    Initialize dealer subsystem for balanced scenarios (C vs D comparison).

    Per PDF specification (Plan 024):
    - VBT agents START with 25% of claims per maturity bucket + matching cash
    - Dealer agents START with 12.5% of claims per maturity bucket + matching cash
    - For mode="passive": trading is disabled (entities just hold)
    - For mode="active": trading is enabled as normal

    The scenario generator (compile_ring_explorer_balanced) creates payables to:
    - vbt_short, vbt_mid, vbt_long agents
    - dealer_short, dealer_mid, dealer_long agents

    This function:
    1. Converts those payables to tickets
    2. Assigns tickets to VBT/Dealer inventory based on who holds them
    3. Sets up proper cash balances (already minted in scenario)

    Args:
        system: Main System instance with agents and contracts
        dealer_config: Configuration for dealer subsystem
        face_value: Face value S (cashflow at maturity)
        outside_mid_ratio: M/S ratio (outside mid as fraction of face)
        big_entity_share: DEPRECATED - use vbt_share_per_bucket and dealer_share_per_bucket
        vbt_share_per_bucket: VBT holds 25% of claims per maturity bucket
        dealer_share_per_bucket: Dealer holds 12.5% of claims per maturity bucket
        mode: "passive" (no trading) or "active" (trading enabled)
        current_day: Current simulation day

    Returns:
        Initialized DealerSubsystem ready for trading (or holding if passive)
    """
    from bilancio.domain.agents import Dealer, VBT
    from bilancio.domain.instruments.credit import Payable

    # Calculate outside mid M from ratio (per-unit-of-face convention, S=1)
    outside_mid = outside_mid_ratio

    subsystem = DealerSubsystem(
        bucket_configs=dealer_config.buckets,
        params=KernelParams(S=face_value),  # S=face_value: kernel accounts for real ticket size
        rng=random.Random(dealer_config.seed),
        enabled=(mode == "active"),  # Disable trading for passive mode
        face_value=face_value,
    )

    # Initialize risk assessor if params provided
    if risk_params:
        subsystem.risk_assessor = RiskAssessor(risk_params)

    # Step 0: Ensure dealer/VBT agents exist in main system
    for bucket_config in dealer_config.buckets:
        bucket_id = bucket_config.name
        dealer_agent_id = f"dealer_{bucket_id}"
        vbt_agent_id = f"vbt_{bucket_id}"

        if dealer_agent_id not in system.state.agents:
            dealer_agent = Dealer(
                id=dealer_agent_id,
                name=f"Dealer ({bucket_id})",
            )
            system.state.agents[dealer_agent_id] = dealer_agent

        if vbt_agent_id not in system.state.agents:
            vbt_agent = VBT(
                id=vbt_agent_id,
                name=f"VBT ({bucket_id})",
            )
            system.state.agents[vbt_agent_id] = vbt_agent

    # Initialize trade executor
    subsystem.executor = TradeExecutor(subsystem.params, subsystem.rng)

    # Step 1: Convert Payables to Tickets and categorize by holder
    serial_counter = 0
    vbt_tickets: Dict[str, List[Ticket]] = {bc.name: [] for bc in dealer_config.buckets}
    dealer_tickets: Dict[str, List[Ticket]] = {bc.name: [] for bc in dealer_config.buckets}
    trader_tickets: Dict[str, List[Ticket]] = {}

    for contract_id, contract in system.state.contracts.items():
        if not isinstance(contract, Payable):
            continue

        ticket_id = f"TKT_{contract_id}"
        remaining_tau = max(0, contract.due_day - current_day)

        ticket = Ticket(
            id=ticket_id,
            issuer_id=contract.liability_issuer_id,
            owner_id=contract.effective_creditor,
            face=Decimal(contract.amount),
            maturity_day=contract.due_day,
            remaining_tau=remaining_tau,
            bucket_id="",
            serial=serial_counter,
        )
        serial_counter += 1

        ticket.bucket_id = _assign_bucket(ticket.remaining_tau, subsystem.bucket_configs)

        subsystem.tickets[ticket_id] = ticket
        subsystem.ticket_to_payable[ticket_id] = contract_id
        subsystem.payable_to_ticket[contract_id] = ticket_id

        # Categorize by holder
        owner = ticket.owner_id
        if owner.startswith("vbt_"):
            # VBT holds this ticket - extract bucket from agent id (vbt_short -> short)
            bucket_name = owner.replace("vbt_", "")
            if bucket_name in vbt_tickets:
                vbt_tickets[bucket_name].append(ticket)
        elif owner.startswith("dealer_"):
            # Dealer holds this ticket
            bucket_name = owner.replace("dealer_", "")
            if bucket_name in dealer_tickets:
                dealer_tickets[bucket_name].append(ticket)
        elif owner.startswith("big_"):
            # DEPRECATED: old big_ entities - assign to dealer
            bucket_name = owner.replace("big_", "")
            if bucket_name in dealer_tickets:
                dealer_tickets[bucket_name].append(ticket)
        else:
            # Regular trader holds this ticket
            if owner not in trader_tickets:
                trader_tickets[owner] = []
            trader_tickets[owner].append(ticket)

    # Step 2: Initialize VBT and Dealer states per bucket
    for bucket_config in subsystem.bucket_configs:
        bucket_id = bucket_config.name

        # Get VBT's tickets for this bucket
        vbt_bucket_tickets = vbt_tickets.get(bucket_id, [])
        vbt_face_value = sum(t.face for t in vbt_bucket_tickets)

        # Get Dealer's tickets for this bucket
        dealer_bucket_tickets = dealer_tickets.get(bucket_id, [])
        dealer_face_value = sum(t.face for t in dealer_bucket_tickets)

        # Get cash for VBT and Dealer from main system
        vbt_cash = _get_agent_cash(system, f"vbt_{bucket_id}")
        dealer_cash = _get_agent_cash(system, f"dealer_{bucket_id}")

        # Create VBT state WITH inventory
        # VBT anchors based on outside_mid
        M = outside_mid
        O = Decimal("0.50") * outside_mid_ratio  # Wider spread for price sensitivity

        vbt = VBTState(
            bucket_id=bucket_id,
            agent_id=f"vbt_{bucket_id}",
            M=M,
            O=O,
            phi_M=dealer_config.phi_M,
            phi_O=dealer_config.phi_O,
            clip_nonneg_B=dealer_config.clip_nonneg_B,
            inventory=list(vbt_bucket_tickets),  # VBT starts WITH inventory!
            cash=vbt_cash,
        )
        vbt.recompute_quotes()
        subsystem.vbts[bucket_id] = vbt

        # Create Dealer state WITH inventory
        dealer = DealerState(
            bucket_id=bucket_id,
            agent_id=f"dealer_{bucket_id}",
            inventory=list(dealer_bucket_tickets),  # Dealer starts WITH inventory!
            cash=dealer_cash,
        )
        subsystem.dealers[bucket_id] = dealer

        # Compute dealer quotes
        recompute_dealer_state(dealer, vbt, subsystem.params)

        # Capture initial equity (dealer cash + VBT mid × dealer inventory × S)
        initial_equity = dealer.cash + vbt.M * dealer.a * subsystem.params.S
        subsystem.metrics.initial_equity_by_bucket[bucket_id] = initial_equity

    # Step 3: Initialize traders (regular household agents)
    for agent_id, agent in system.state.agents.items():
        if agent.kind != "household":
            continue
        # Skip VBT/Dealer/big entity agents
        if agent_id.startswith("vbt_") or agent_id.startswith("dealer_") or agent_id.startswith("big_"):
            continue

        trader_cash = _get_agent_cash(system, agent_id)

        trader = TraderState(
            agent_id=agent_id,
            cash=trader_cash,
            tickets_owned=[],
            obligations=[],
            asset_issuer_id=None,
        )

        # Link trader to their tickets (tickets they own as assets)
        for ticket in subsystem.tickets.values():
            if ticket.owner_id == agent_id:
                trader.tickets_owned.append(ticket)
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = ticket.issuer_id

            # Tickets where trader is the debtor
            if ticket.issuer_id == agent_id:
                trader.obligations.append(ticket)

        subsystem.traders[agent_id] = trader

    # Step 4: Capture initial debt-to-money ratio
    total_debt = Decimal(0)
    for contract in system.state.contracts.values():
        if isinstance(contract, Payable):
            total_debt += Decimal(contract.amount)

    # Total money held by traders only (not VBT/Dealer)
    total_money = Decimal(0)
    for agent_id_iter, agent in system.state.agents.items():
        if agent.kind == "household" and not (
            agent_id_iter.startswith("vbt_") or
            agent_id_iter.startswith("dealer_") or
            agent_id_iter.startswith("big_")
        ):
            total_money += _get_agent_cash(system, agent_id_iter)

    subsystem.metrics.initial_total_debt = total_debt
    subsystem.metrics.initial_total_money = total_money

    # Store serial counter so ingestion can continue the sequence
    subsystem._ticket_serial_counter = serial_counter

    return subsystem


def _get_agent_cash(system, agent_id: str) -> Decimal:
    """
    Get total cash balance for an agent from the main system.

    Sums all cash contracts where the agent is the asset holder.

    Args:
        system: Main System instance
        agent_id: Agent ID to get cash for

    Returns:
        Total cash balance as Decimal
    """
    agent = system.state.agents.get(agent_id)
    if not agent:
        return Decimal(0)

    total_cash = Decimal(0)
    for contract_id in agent.asset_ids:
        contract = system.state.contracts.get(contract_id)
        if contract and contract.kind == "cash":
            total_cash += Decimal(contract.amount)

    return total_cash


def _reassign_payable_owner(system, contract_id: str, old_owner: str, new_owner: str) -> None:
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


def _ingest_new_payables(subsystem: DealerSubsystem, system, current_day: int) -> int:
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

    new_count = 0
    for contract_id, contract in list(system.state.contracts.items()):
        if not isinstance(contract, Payable):
            continue
        if contract_id in subsystem.payable_to_ticket:
            continue  # Already tracked

        remaining_tau = max(0, contract.due_day - current_day)
        if remaining_tau == 0:
            continue  # Already matured

        # Skip defaulted issuers
        issuer = system.state.agents.get(contract.liability_issuer_id)
        if issuer is None or getattr(issuer, 'defaulted', False):
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


def _execute_sell_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    current_day: int,
    events: List[dict],
    dealer_budgets: Dict[str, Decimal] | None = None,
) -> Decimal:
    """Process a single sell trade attempt for a trader. Returns cash injected into ring."""
    trader = subsystem.traders[trader_id]
    if not trader.tickets_owned:
        return Decimal(0)

    # Select ticket to sell (first in list for simplicity)
    ticket = trader.tickets_owned[0]
    bucket_id = ticket.bucket_id
    dealer = subsystem.dealers[bucket_id]
    vbt = subsystem.vbts[bucket_id]

    # Pre-trade solvency check: estimate cost and verify payer can afford it
    # For a customer sell, the dealer (or VBT) is the buyer/payer
    if dealer_budgets is not None:
        from bilancio.dealer.kernel import can_interior_buy
        is_interior = can_interior_buy(dealer, subsystem.params)
        payer_id = dealer.agent_id if is_interior else vbt.agent_id
        estimated_cost = Decimal(str(dealer.bid if is_interior else vbt.B)) * ticket.face
        if dealer_budgets.get(payer_id, Decimal(0)) < estimated_cost:
            return Decimal(0)  # Skip: payer can't afford this trade

    # Capture pre-trade state for metrics (Section 8.1, 8.4)
    pre_dealer_inventory = dealer.a
    pre_dealer_cash = dealer.cash
    pre_dealer_bid = dealer.bid
    pre_dealer_ask = dealer.ask
    pre_trader_cash = trader.cash
    pre_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

    # Check if liquidity-driven (Section 8.3)
    is_liquidity_driven = trader.shortfall(current_day) > 0

    # Risk assessment check (Plan 032)
    if subsystem.risk_assessor:
        # Compute asset value as sum of EVs of owned tickets
        asset_value = sum(
            subsystem.risk_assessor.expected_value(t, current_day)
            for t in trader.tickets_owned
        )
        if not subsystem.risk_assessor.should_sell(
            ticket=ticket,
            dealer_bid=dealer.bid,
            current_day=current_day,
            trader_cash=trader.cash,
            trader_shortfall=trader.shortfall(current_day),
            trader_asset_value=asset_value,
        ):
            # Trader rejects price - log event and skip
            events.append({
                "kind": "sell_rejected",
                "day": current_day,
                "phase": "simulation",
                "trader_id": trader_id,
                "ticket_id": ticket.id,
                "bucket": bucket_id,
                "offered_price": float(dealer.bid),
                "expected_value": float(subsystem.risk_assessor.expected_value(ticket, current_day)),
                "threshold": float(subsystem.risk_assessor.params.base_risk_premium),
                "reason": "price_below_ev_threshold",
            })
            return Decimal(0)

    # Execute customer sell
    result = subsystem.executor.execute_customer_sell(
        dealer, vbt, ticket, check_assertions=False
    )

    if result.executed:
        # Scale price by ticket face value
        # The dealer module returns unit price (per S=1), but our tickets have actual face values
        scaled_price = result.price * ticket.face

        # Update trader state
        trader.tickets_owned.remove(ticket)
        trader.cash += scaled_price

        # Capture post-trade state
        post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

        # Create detailed trade record for metrics (Section 8)
        trade_record = TradeRecord(
            day=current_day,
            bucket=bucket_id,
            side="SELL",
            trader_id=trader_id,
            ticket_id=ticket.id,
            issuer_id=ticket.issuer_id,
            maturity_day=ticket.maturity_day,
            face_value=ticket.face,
            price=scaled_price,
            unit_price=result.price,
            is_passthrough=result.is_passthrough,
            dealer_inventory_before=pre_dealer_inventory,
            dealer_cash_before=pre_dealer_cash,
            dealer_bid_before=pre_dealer_bid,
            dealer_ask_before=pre_dealer_ask,
            vbt_mid_before=vbt.M,
            trader_cash_before=pre_trader_cash,
            trader_safety_margin_before=pre_safety_margin,
            dealer_inventory_after=dealer.a,
            dealer_cash_after=dealer.cash,
            dealer_bid_after=dealer.bid,
            dealer_ask_after=dealer.ask,
            trader_cash_after=trader.cash,
            trader_safety_margin_after=post_safety_margin,
            is_liquidity_driven=is_liquidity_driven,
            reduces_margin_below_zero=False,  # Only for BUYs
        )
        subsystem.metrics.trades.append(trade_record)

        # Update ticket outcome for return tracking (Section 8.3)
        if ticket.id not in subsystem.metrics.ticket_outcomes:
            subsystem.metrics.ticket_outcomes[ticket.id] = TicketOutcome(
                ticket_id=ticket.id,
                issuer_id=ticket.issuer_id,
                maturity_day=ticket.maturity_day,
                face_value=ticket.face,
            )
        subsystem.metrics.ticket_outcomes[ticket.id].sold_to_dealer = True
        subsystem.metrics.ticket_outcomes[ticket.id].sale_day = current_day
        subsystem.metrics.ticket_outcomes[ticket.id].sale_price = scaled_price
        subsystem.metrics.ticket_outcomes[ticket.id].seller_id = trader_id

        events.append({
            "kind": "dealer_trade",
            "day": current_day,
            "phase": "simulation",
            "trader": trader_id,
            "side": "sell",
            "ticket_id": ticket.id,
            "bucket": bucket_id,
            "price": float(scaled_price),
            "unit_price": float(result.price),
            "face": float(ticket.face),
            "is_passthrough": result.is_passthrough,
            "is_liquidity_driven": is_liquidity_driven,
        })

        # Update dealer/VBT cash budget after successful sell trade
        if dealer_budgets is not None:
            payer_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
            dealer_budgets[payer_id] = dealer_budgets.get(payer_id, Decimal(0)) - scaled_price

        return scaled_price

    return Decimal(0)


def _execute_buy_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    current_day: int,
    events: List[dict],
    dealer_budgets: Dict[str, Decimal] | None = None,
) -> Decimal:
    """Process a single buy trade attempt for a trader. Returns cash drained from ring."""
    trader = subsystem.traders[trader_id]

    # Try to buy from first available bucket
    for bucket_id, dealer in subsystem.dealers.items():
        vbt = subsystem.vbts[bucket_id]

        # Check if dealer or VBT has inventory
        if not dealer.inventory and not vbt.inventory:
            continue

        # Capture pre-trade state for metrics (Section 8.1, 8.4)
        pre_dealer_inventory = dealer.a
        pre_dealer_cash = dealer.cash
        pre_dealer_bid = dealer.bid
        pre_dealer_ask = dealer.ask
        pre_trader_cash = trader.cash
        pre_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

        # Risk assessment pre-check for buy (approximate - we don't know exact ticket yet)
        # We'll do a post-execution check if risk_assessor exists
        should_reverse = False

        # Execute customer buy
        result = subsystem.executor.execute_customer_buy(
            dealer, vbt, trader_id, check_assertions=False
        )

        if result.executed and result.ticket:
            # Post-execution risk assessment check (Plan 032)
            if subsystem.risk_assessor:
                # Compute asset value (including the ticket we just bought)
                asset_value = sum(
                    subsystem.risk_assessor.expected_value(t, current_day)
                    for t in trader.tickets_owned
                )
                # Check if trader would accept this buy
                if not subsystem.risk_assessor.should_buy(
                    ticket=result.ticket,
                    dealer_ask=result.price,  # Unit price
                    current_day=current_day,
                    trader_cash=trader.cash,
                    trader_shortfall=trader.shortfall(current_day),
                    trader_asset_value=asset_value,
                ):
                    # Trader rejects - reverse the transaction
                    should_reverse = True
                    # Put ticket back to dealer/VBT
                    # The kernel added unit price to cash, so reverse by subtracting unit price only
                    if result.is_passthrough:
                        vbt.inventory.append(result.ticket)
                        result.ticket.owner_id = f"vbt_{bucket_id}"
                        vbt.cash -= result.price
                    else:
                        dealer.inventory.append(result.ticket)
                        result.ticket.owner_id = f"dealer_{bucket_id}"
                        dealer.cash -= result.price
                    events.append({
                        "kind": "buy_rejected",
                        "day": current_day,
                        "phase": "simulation",
                        "trader_id": trader_id,
                        "ticket_id": result.ticket.id,
                        "bucket": bucket_id,
                        "offered_price": float(result.price),
                        "expected_value": float(subsystem.risk_assessor.expected_value(result.ticket, current_day)),
                        "threshold": float(subsystem.risk_assessor.params.base_risk_premium * subsystem.risk_assessor.params.buy_premium_multiplier),
                        "reason": "ev_below_price_threshold",
                    })
                    continue

        if result.executed and result.ticket and not should_reverse:
            # Scale price by ticket face value
            scaled_price = result.price * result.ticket.face

            # Pre-trade solvency check: trader must be able to afford scaled price
            if trader.cash < scaled_price:
                # Reverse the kernel trade — put ticket back
                if result.is_passthrough:
                    vbt.inventory.append(result.ticket)
                    result.ticket.owner_id = f"vbt_{bucket_id}"
                    vbt.cash -= result.price
                else:
                    dealer.inventory.append(result.ticket)
                    result.ticket.owner_id = f"dealer_{bucket_id}"
                    dealer.cash -= result.price
                recompute_dealer_state(dealer, vbt, subsystem.params)
                continue  # Try next bucket

            # Update trader state
            trader.tickets_owned.append(result.ticket)
            trader.cash -= scaled_price

            # Update asset issuer if first ticket
            if trader.asset_issuer_id is None:
                trader.asset_issuer_id = result.ticket.issuer_id

            # Capture post-trade state
            post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

            # Check if BUY reduced margin below zero (Section 8.4)
            reduces_margin_below_zero = (
                pre_safety_margin >= 0 and post_safety_margin < 0
            )

            # Create detailed trade record for metrics (Section 8)
            trade_record = TradeRecord(
                day=current_day,
                bucket=bucket_id,
                side="BUY",
                trader_id=trader_id,
                ticket_id=result.ticket.id,
                issuer_id=result.ticket.issuer_id,
                maturity_day=result.ticket.maturity_day,
                face_value=result.ticket.face,
                price=scaled_price,
                unit_price=result.price,
                is_passthrough=result.is_passthrough,
                dealer_inventory_before=pre_dealer_inventory,
                dealer_cash_before=pre_dealer_cash,
                dealer_bid_before=pre_dealer_bid,
                dealer_ask_before=pre_dealer_ask,
                vbt_mid_before=vbt.M,
                trader_cash_before=pre_trader_cash,
                trader_safety_margin_before=pre_safety_margin,
                dealer_inventory_after=dealer.a,
                dealer_cash_after=dealer.cash,
                dealer_bid_after=dealer.bid,
                dealer_ask_after=dealer.ask,
                trader_cash_after=trader.cash,
                trader_safety_margin_after=post_safety_margin,
                is_liquidity_driven=False,  # BUYs are never liquidity-driven
                reduces_margin_below_zero=reduces_margin_below_zero,
            )
            subsystem.metrics.trades.append(trade_record)

            # Update ticket outcome for return tracking (Section 8.3)
            ticket = result.ticket
            if ticket.id not in subsystem.metrics.ticket_outcomes:
                subsystem.metrics.ticket_outcomes[ticket.id] = TicketOutcome(
                    ticket_id=ticket.id,
                    issuer_id=ticket.issuer_id,
                    maturity_day=ticket.maturity_day,
                    face_value=ticket.face,
                )
            subsystem.metrics.ticket_outcomes[ticket.id].purchased_from_dealer = True
            subsystem.metrics.ticket_outcomes[ticket.id].purchase_day = current_day
            subsystem.metrics.ticket_outcomes[ticket.id].purchase_price = scaled_price
            subsystem.metrics.ticket_outcomes[ticket.id].purchaser_id = trader_id

            events.append({
                "kind": "dealer_trade",
                "day": current_day,
                "phase": "simulation",
                "trader": trader_id,
                "side": "buy",
                "ticket_id": result.ticket.id,
                "bucket": bucket_id,
                "price": float(scaled_price),
                "unit_price": float(result.price),
                "face": float(result.ticket.face),
                "is_passthrough": result.is_passthrough,
                "reduces_margin_below_zero": reduces_margin_below_zero,
            })

            # Update dealer/VBT cash budget after successful buy trade
            # For a buy, the dealer/VBT received cash (sold a ticket)
            if dealer_budgets is not None:
                payee_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
                dealer_budgets[payee_id] = dealer_budgets.get(payee_id, Decimal(0)) + scaled_price

            return scaled_price  # One buy per trader per batch

    return Decimal(0)


def _pool_desk_cash(subsystem: DealerSubsystem) -> None:
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
        total_dealer_cash = sum(d.cash for d in subsystem.dealers.values())
        per_desk = total_dealer_cash / num_dealers
        for dealer in subsystem.dealers.values():
            dealer.cash = per_desk

    # Pool VBT cash
    num_vbts = len(subsystem.vbts)
    if num_vbts > 0:
        total_vbt_cash = sum(v.cash for v in subsystem.vbts.values())
        per_desk = total_vbt_cash / num_vbts
        for vbt in subsystem.vbts.values():
            vbt.cash = per_desk


def run_dealer_trading_phase(
    subsystem: DealerSubsystem,
    system,
    current_day: int
) -> List[dict]:
    """
    Execute one dealer trading phase for the current day.

    This function orchestrates a complete trading period:

    Phase 1: Update maturities
        - Increment day counters
        - Update remaining_tau for all tickets
        - Reassign tickets to buckets based on new maturities

    Phase 2: Recompute dealer quotes
        - Run kernel for each dealer based on new inventory
        - Update bid/ask prices

    Phase 3: Build eligibility sets
        - Identify traders with shortfalls (need cash)
        - Identify traders with surplus (can invest)
        - Apply policy constraints (single-issuer, horizon)

    Phase 4: Randomized order flow
        - Generate random order of eligible traders
        - Process sell orders (traders need cash)
        - Process buy orders (traders want to invest)
        - Execute trades through dealers or VBTs

    Phase 5: Record events
        - Log all trades with prices and counterparties
        - Track interior vs passthrough execution
        - Record dealer quote evolution

    Note: This is a simplified implementation. The full dealer simulation
    includes settlement, default handling, and VBT anchor updates which
    may be added in future iterations.

    Args:
        subsystem: Dealer subsystem state
        system: Main System instance (for accessing agent cash balances)
        current_day: Current simulation day

    Returns:
        List of trade event dictionaries for logging

    Example:
        >>> events = run_dealer_trading_phase(subsystem, system, day=5)
        >>> for event in events:
        ...     print(f"Trade: {event['trader']} {event['side']} at {event['price']}")
    """
    if not subsystem.enabled:
        return []

    events = []

    # Phase 0: Sync trader cash from main system
    # This ensures traders have up-to-date cash balances for eligibility checks
    for trader_id, trader in subsystem.traders.items():
        trader.cash = _get_agent_cash(system, trader_id)

    # Phase 0.5: Clean up tickets whose payables were removed
    # This can happen when agents default and get expelled (expel-agent mode)
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
        payable_id = subsystem.ticket_to_payable.pop(ticket_id, None)
        if payable_id:
            subsystem.payable_to_ticket.pop(payable_id, None)

    # Phase 0.75: Ingest new payables as tickets (from rollover)
    _ingest_new_payables(subsystem, system, current_day)

    # Phase 1: Update ticket maturities and buckets
    # Collect matured tickets to remove after iteration
    matured_ticket_ids = []

    for ticket in subsystem.tickets.values():
        # Update remaining maturity
        old_bucket = ticket.bucket_id
        ticket.remaining_tau = max(0, ticket.maturity_day - current_day)

        # Mark matured tickets for cleanup (remaining_tau = 0 means due today or past)
        if ticket.remaining_tau == 0:
            matured_ticket_ids.append(ticket.id)
            # Remove from inventories before deletion
            old_dealer = subsystem.dealers.get(old_bucket)
            old_vbt = subsystem.vbts.get(old_bucket)
            if old_dealer and ticket in old_dealer.inventory:
                old_dealer.inventory.remove(ticket)
            if old_vbt and ticket in old_vbt.inventory:
                old_vbt.inventory.remove(ticket)
            # Remove from trader holdings
            for trader in subsystem.traders.values():
                if ticket in trader.tickets_owned:
                    trader.tickets_owned.remove(ticket)
                if ticket in trader.obligations:
                    trader.obligations.remove(ticket)
            continue

        new_bucket = _assign_bucket(ticket.remaining_tau, subsystem.bucket_configs)

        # If bucket changed, move ticket to new dealer/VBT inventory
        if new_bucket != old_bucket:
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

    # Clean up matured tickets to prevent unbounded memory growth
    for ticket_id in matured_ticket_ids:
        del subsystem.tickets[ticket_id]

    # Phase 1.5: Cash pooling — redistribute cash equally across desks.
    # The three dealer desks (short/mid/long) are one firm sharing a balance sheet.
    # Same for VBT desks. Pooling ensures default losses and settlement revenue
    # are amortized across all desks, not concentrated in the short bucket.
    _pool_desk_cash(subsystem)

    # Phase 2: Recompute dealer quotes for all buckets
    for bucket_id, dealer in subsystem.dealers.items():
        vbt = subsystem.vbts[bucket_id]
        recompute_dealer_state(dealer, vbt, subsystem.params)

    # Capture daily snapshots for metrics (Section 8.1)
    _capture_dealer_snapshots(subsystem, current_day)
    _capture_trader_snapshots(subsystem, current_day)
    _capture_system_state_snapshot(subsystem, current_day)  # Plan 022

    # Phase 3: Build eligibility sets (simplified)
    # Traders who need cash (have shortfall coming in next few days)
    eligible_sellers = []
    horizon = 10  # Look ahead this many days for upcoming obligations
    for trader_id, trader in subsystem.traders.items():
        # Check for shortfall on any of the next 'horizon' days
        upcoming_shortfall = Decimal(0)
        for day_offset in range(horizon + 1):
            upcoming_shortfall = max(upcoming_shortfall, trader.shortfall(current_day + day_offset))
        if upcoming_shortfall > 0 and trader.tickets_owned:
            eligible_sellers.append(trader_id)

    # Traders who can buy (have surplus cash beyond needs)
    # Re-enabled per spec Section 11.2: Investment policy (buying tickets)
    eligible_buyers = []
    # Only allow buying if trader has genuine surplus above ALL upcoming obligations
    for trader_id, trader in subsystem.traders.items():
        total_upcoming_dues = Decimal(0)
        for day_offset in range(horizon + 1):
            total_upcoming_dues += trader.payment_due(current_day + day_offset)
        surplus = trader.cash - total_upcoming_dues
        if surplus > subsystem.face_value:  # Buffer of one face-value unit after covering all dues
            eligible_buyers.append(trader_id)

    # Phase 4: Interleaved order flow in batches
    # Sellers and buyers alternate in batches of BATCH_SIZE.
    # Sellers go first (urgent liquidity needs), then buyers, repeating
    # until both lists are exhausted.
    # Cash neutrality: total buy cash cannot exceed total sell cash,
    # preventing the dealer from being a net cash drain on the ring.
    BATCH_SIZE = 10
    subsystem.rng.shuffle(eligible_sellers)
    subsystem.rng.shuffle(eligible_buyers)

    # Initialize cash budgets for dealers/VBTs from system-level cash
    # This prevents trades that would make dealer/VBT cash go negative
    dealer_budgets: Dict[str, Decimal] = {}
    for bucket_id, dealer in subsystem.dealers.items():
        dealer_budgets[dealer.agent_id] = _get_agent_cash(system, dealer.agent_id)
    for bucket_id, vbt in subsystem.vbts.items():
        dealer_budgets[vbt.agent_id] = _get_agent_cash(system, vbt.agent_id)

    seller_idx = 0
    buyer_idx = 0
    sell_cash = Decimal(0)  # Cumulative cash injected by sells
    buy_cash = Decimal(0)   # Cumulative cash drained by buys

    while seller_idx < len(eligible_sellers) or buyer_idx < len(eligible_buyers):
        # --- Seller batch ---
        for trader_id in eligible_sellers[seller_idx:seller_idx + BATCH_SIZE]:
            sell_cash += _execute_sell_trade(subsystem, trader_id, current_day, events, dealer_budgets)
        seller_idx += BATCH_SIZE

        # --- Buyer batch (constrained by cash neutrality) ---
        for trader_id in eligible_buyers[buyer_idx:buyer_idx + BATCH_SIZE]:
            if buy_cash >= sell_cash:
                break  # Would drain more cash than sells injected
            buy_cash += _execute_buy_trade(subsystem, trader_id, current_day, events, dealer_budgets)
        buyer_idx += BATCH_SIZE

    return events


def sync_dealer_to_system(
    subsystem: DealerSubsystem,
    system
) -> None:
    """
    Sync dealer trade results back to main system state.

    This function bridges the dealer subsystem state back to the main
    simulation system, updating:

    1. Payable ownership:
       - Update Payable.asset_holder_id for transferred claims
       - Update agent.asset_ids lists (remove from old holder, add to new)
       - Maintain double-entry consistency

    2. Agent cash balances:
       - Apply cash changes from trader.cash to agent balance sheets
       - Update Cash contract amounts in system.state.contracts

    3. Consistency checks:
       - Verify ticket ownership matches payable asset_holder_id
       - Ensure cash changes sum to zero (closed system)

    Implementation approach:
        - Use ticket_to_payable mapping to find contracts
        - Update asset_holder_id based on ticket.owner_id
        - Calculate cash deltas by comparing trader.cash to agent balance

    Note: This is a simplified implementation. A full version would:
        - Handle cash contract splits/merges atomically
        - Track exact cash changes through trade history
        - Support rollback on validation failures
        - Log all sync operations for audit trail

    Args:
        subsystem: Dealer subsystem with updated state
        system: Main System instance to update

    Raises:
        ValidationError: If sync would violate system invariants

    Example:
        >>> # After trading phase
        >>> events = run_dealer_trading_phase(subsystem, system, day=5)
        >>> # Sync results back to main system
        >>> sync_dealer_to_system(subsystem, system)
        >>> # Now system.state reflects all trades
    """
    from bilancio.domain.instruments.credit import Payable

    # Step 1: Update Payable ownership for transferred claims
    # Note: Payable has two holder fields:
    # - asset_holder_id: original creditor (from base Instrument class)
    # - holder_id: secondary market holder (specific to Payable)
    # Settlement uses effective_creditor which returns holder_id if set, else asset_holder_id
    for ticket_id, ticket in subsystem.tickets.items():
        # Get corresponding payable
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

    # Step 2: Sync trader cash balances
    # Compare trader.cash in dealer subsystem to actual cash in main system
    # and apply the delta as minting (if trader gained) or burning (if trader paid)
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
                    if contract and contract.kind == "cash":
                        if contract.amount <= remaining_to_burn:
                            # Remove entire contract
                            remaining_to_burn -= contract.amount
                            agent.asset_ids.remove(contract_id)
                            del system.state.contracts[contract_id]
                        else:
                            # Reduce contract amount
                            contract.amount -= round(remaining_to_burn)
                            remaining_to_burn = 0


def _capture_dealer_snapshots(
    subsystem: DealerSubsystem,
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
            max_capacity=dealer.X_star,
            is_at_zero=(dealer.a == 0),
            hit_vbt_this_step=(bucket_id in vbt_used_buckets),
            total_system_face=total_face,
            dealer_share_pct=dealer_share,
        )
        subsystem.metrics.dealer_snapshots.append(snapshot)


def _capture_trader_snapshots(
    subsystem: DealerSubsystem,
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
        obligations_remaining = sum(t.face for t in trader.obligations)

        snapshot = TraderSnapshot(
            day=current_day,
            trader_id=trader_id,
            cash=trader.cash,
            tickets_held_count=len(trader.tickets_owned),
            tickets_held_ids=[t.id for t in trader.tickets_owned],
            total_face_held=sum(t.face for t in trader.tickets_owned),
            obligations_remaining=obligations_remaining,
            saleable_value=saleable,
            safety_margin=safety_margin,
            defaulted=trader.defaulted,
        )
        subsystem.metrics.trader_snapshots.append(snapshot)


def _capture_system_state_snapshot(
    subsystem: DealerSubsystem,
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


def _compute_trader_safety_margin(
    subsystem: DealerSubsystem,
    trader_id: str
) -> Decimal:
    """Compute safety margin for a specific trader."""
    trader = subsystem.traders.get(trader_id)
    if not trader:
        return Decimal(0)

    return compute_safety_margin(
        cash=trader.cash,
        tickets_held=trader.tickets_owned,
        obligations=trader.obligations,
        dealers=subsystem.dealers,
        ticket_size=subsystem.params.S
    )


def _assign_bucket(remaining_tau: int, bucket_configs: List[BucketConfig]) -> str:
    """
    Assign a ticket to a maturity bucket based on remaining tau.

    Args:
        remaining_tau: Days remaining until maturity
        bucket_configs: List of bucket definitions

    Returns:
        Bucket name (e.g., "short", "mid", "long")

    Example:
        >>> _assign_bucket(2, DEFAULT_BUCKETS)
        'short'
        >>> _assign_bucket(15, DEFAULT_BUCKETS)
        'long'
    """
    for bucket in bucket_configs:
        if remaining_tau < bucket.tau_min:
            continue
        if bucket.tau_max is None or remaining_tau <= bucket.tau_max:
            return bucket.name

    # Default to last bucket (usually "long")
    return bucket_configs[-1].name if bucket_configs else "default"
