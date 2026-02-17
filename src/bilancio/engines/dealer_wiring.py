"""Dealer subsystem initialization and wiring.

Functions that build the initial DealerSubsystem state: creating agents,
converting payables to tickets, and setting up market makers and traders.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.dealer_integration import DealerSubsystem
    from bilancio.engines.system import System

from bilancio.dealer.kernel import recompute_dealer_state
from bilancio.dealer.models import (
    BucketConfig,
    DealerState,
    Ticket,
    TraderState,
    VBTState,
)
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agent import AgentKind


def _compute_corridor_spreads(
    banking_sub: object,
    bucket_configs: list,
) -> dict[str, Decimal]:
    """Derive VBT base spreads from corridor width when banking is active.

    O_bucket = Omega * tau_avg_bucket

    Falls back to hardcoded spreads if banking subsystem is not available.
    """
    from decimal import Decimal as D

    try:
        omega = banking_sub.bank_profile.corridor_width(banking_sub.kappa)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return {}

    spreads = {}
    for bc in bucket_configs:
        # Estimate average tau for this bucket
        tau_min = bc.get("tau_min", 1) if isinstance(bc, dict) else getattr(bc, "tau_min", 1)
        tau_max = bc.get("tau_max", tau_min) if isinstance(bc, dict) else getattr(bc, "tau_max", tau_min)
        tau_avg = D(str((tau_min + tau_max) / 2))
        bucket_id = bc.get("bucket_id", "") if isinstance(bc, dict) else getattr(bc, "bucket_id", "")
        spreads[bucket_id] = omega * tau_avg

    return spreads


def _ensure_dealer_vbt_agents(system: System, bucket_configs: list[BucketConfig]) -> None:
    """Create dealer and VBT agents in the main system if they don't exist.

    These agents allow proper ownership tracking when claims transfer to
    dealers/VBTs during trading.

    Args:
        system: Main System instance
        bucket_configs: List of bucket configurations defining which
            dealer/VBT agents to create
    """
    from bilancio.domain.agents import VBT, Dealer

    for bucket_config in bucket_configs:
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


def _convert_payables_to_tickets(
    subsystem: DealerSubsystem,
    system: System,
    current_day: int,
) -> int:
    """Convert all Payables in the system to Tickets and register them.

    Creates a Ticket for each Payable contract, assigns it to a maturity
    bucket, and builds bidirectional mappings.

    Args:
        subsystem: Dealer subsystem to register tickets in
        system: Main System instance with contracts
        current_day: Current simulation day for maturity calculations

    Returns:
        Serial counter value after all tickets are created
    """
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.dealer_integration import _assign_bucket

    serial_counter = 0
    for contract_id, contract in system.state.contracts.items():
        if not isinstance(contract, Payable):
            continue
        if contract.due_day is None:
            continue
        if contract.amount <= 0:
            continue  # Skip zero-face payables (Dirichlet rounding artifacts)

        ticket_id = f"TKT_{contract_id}"
        remaining_tau = max(0, contract.due_day - current_day)

        ticket = Ticket(
            id=ticket_id,
            issuer_id=contract.liability_issuer_id,  # Debtor
            owner_id=contract.effective_creditor,  # Current creditor
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

    return serial_counter


def _initialize_market_makers(
    subsystem: DealerSubsystem,
    dealer_config: DealerRingConfig,
    dealer_capital_per_bucket: Decimal,
    vbt_capital_per_bucket: Decimal,
) -> None:
    """Initialize dealer and VBT states for each maturity bucket.

    Creates DealerState and VBTState with initial capital (no inventory),
    computes initial quotes via the kernel, and captures initial equity.

    Args:
        subsystem: Dealer subsystem to populate
        dealer_config: Configuration with VBT anchors and parameters
        dealer_capital_per_bucket: NEW outside money for each dealer desk
        vbt_capital_per_bucket: NEW outside money for each VBT desk
    """
    for bucket_config in subsystem.bucket_configs:
        bucket_id = bucket_config.name

        # Get anchor prices from config
        mid_anchor, spread_anchor = dealer_config.vbt_anchors.get(
            bucket_id, (Decimal(1), Decimal("0.30"))
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
            M=mid_anchor,
            O=spread_anchor,
            phi_M=dealer_config.phi_M,
            phi_O=dealer_config.phi_O,
            clip_nonneg_B=dealer_config.clip_nonneg_B,
            inventory=[],  # Empty! VBTs build inventory by buying from traders
            cash=vbt_capital_per_bucket,  # NEW outside money
        )
        vbt.recompute_quotes()
        subsystem.vbts[bucket_id] = vbt

        # Store initial spread for spread_sensitivity computation
        subsystem.initial_spread_by_bucket[bucket_id] = spread_anchor

        # NOTE: Traders keep 100% of their tickets (no allocation to dealer/VBT)
        # Dealer/VBT will acquire inventory by buying from traders during trading

        # Run kernel to compute initial quotes
        recompute_dealer_state(dealer, vbt, subsystem.params)

        # Capture initial equity for P&L calculation (Section 8.2)
        # E_0^(b) = C_0^(b) + M * a_0^(b) * S
        initial_equity = dealer.cash + vbt.M * dealer.a * subsystem.params.S
        subsystem.metrics.initial_equity_by_bucket[bucket_id] = initial_equity


def _initialize_traders(
    subsystem: DealerSubsystem,
    system: System,
    skip_prefixes: tuple[str, ...] = (),
) -> None:
    """Initialize TraderState for each household agent in the system.

    Creates trader states with cash balances from the main system and
    links owned/obligated tickets.

    Args:
        subsystem: Dealer subsystem to populate with traders
        system: Main System instance with agents
        skip_prefixes: Agent ID prefixes to skip (e.g., VBT/dealer agents)
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    for agent_id, agent in system.state.agents.items():
        if agent.kind != AgentKind.HOUSEHOLD:
            continue
        if skip_prefixes and agent_id.startswith(skip_prefixes):
            continue

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


def _capture_initial_debt_to_money(
    subsystem: DealerSubsystem,
    system: System,
    exclude_kinds: tuple[str, ...] = ("dealer", "vbt"),
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    """Capture initial debt-to-money ratio for metrics (Plan 020 - Phase B).

    This is a key control variable: results only make sense given this ratio.

    Args:
        subsystem: Dealer subsystem to store metrics in
        system: Main System instance
        exclude_kinds: Agent kinds to exclude from total money calculation
        exclude_prefixes: Agent ID prefixes to exclude from total money
    """
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.dealer_integration import _get_agent_cash

    # Sum all payable amounts (total debt in system)
    total_debt = Decimal(0)
    for contract in system.state.contracts.values():
        if isinstance(contract, Payable):
            total_debt += Decimal(contract.amount)

    # Sum all cash holdings (total money in system, excluding specified kinds)
    total_money = Decimal(0)
    for agent_id_iter, agent in system.state.agents.items():
        if agent.kind in exclude_kinds:
            continue
        if exclude_prefixes and agent_id_iter.startswith(exclude_prefixes):
            continue
        total_money += _get_agent_cash(system, agent_id_iter)

    subsystem.metrics.initial_total_debt = total_debt
    subsystem.metrics.initial_total_money = total_money


def _categorize_tickets_by_holder(
    subsystem: DealerSubsystem,
    bucket_names: list[str],
) -> tuple[dict[str, list[Ticket]], dict[str, list[Ticket]], dict[str, list[Ticket]]]:
    """Categorize tickets by holder type: VBT, dealer, or regular trader.

    Examines each ticket's owner_id prefix to determine which entity holds it.

    Args:
        subsystem: Dealer subsystem with tickets already created
        bucket_names: Names of all maturity buckets

    Returns:
        Tuple of (vbt_tickets, dealer_tickets, trader_tickets) dicts.
        vbt_tickets and dealer_tickets are keyed by bucket name,
        trader_tickets is keyed by agent_id.
    """
    vbt_tickets: dict[str, list[Ticket]] = {name: [] for name in bucket_names}
    dealer_tickets: dict[str, list[Ticket]] = {name: [] for name in bucket_names}
    trader_tickets: dict[str, list[Ticket]] = {}

    for ticket in subsystem.tickets.values():
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

    return vbt_tickets, dealer_tickets, trader_tickets


def _initialize_balanced_market_makers(
    subsystem: DealerSubsystem,
    system: System,
    dealer_config: DealerRingConfig,
    vbt_tickets: dict[str, list[Ticket]],
    dealer_tickets: dict[str, list[Ticket]],
    shared_prior: Decimal = Decimal("0.15"),
) -> None:
    """Initialize VBT and Dealer states per bucket for balanced scenarios.

    Unlike the standard initialization (empty inventory), balanced scenarios
    start VBTs and dealers WITH inventory allocated by the scenario generator.

    Uses per-bucket base spreads (short=0.04, mid=0.08, long=0.12) with
    additive credit risk widening.

    Args:
        subsystem: Dealer subsystem to populate
        system: Main System instance (for reading agent cash)
        dealer_config: Dealer ring configuration
        vbt_tickets: Pre-categorized VBT tickets per bucket
        dealer_tickets: Pre-categorized dealer tickets per bucket
        shared_prior: Kappa-informed default prior shared by VBT and traders
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    # Per-bucket base spreads: term risk premium
    BASE_SPREAD_BY_BUCKET = {
        "short": Decimal("0.04"),
        "mid": Decimal("0.08"),
        "long": Decimal("0.12"),
    }

    # Override with corridor-derived spreads if banking subsystem is active
    banking_sub = getattr(system.state, "banking_subsystem", None)
    if banking_sub is not None:
        try:
            omega = banking_sub.bank_profile.corridor_width(banking_sub.kappa)
            # Derive: O_bucket = omega * tau_avg_bucket
            BUCKET_TAU_AVG = {"short": Decimal("2"), "mid": Decimal("6"), "long": Decimal("12")}
            for bucket_name, tau_avg in BUCKET_TAU_AVG.items():
                if bucket_name in BASE_SPREAD_BY_BUCKET:
                    derived = omega * tau_avg
                    if derived > Decimal("0"):
                        BASE_SPREAD_BY_BUCKET[bucket_name] = derived
        except (AttributeError, TypeError):
            pass  # Fall back to hardcoded spreads

    # VBT mid = pure credit discount: M = 1 - P_prior
    credit_adjusted_mid = Decimal(1) - shared_prior

    for bucket_config in subsystem.bucket_configs:
        bucket_id = bucket_config.name

        # Get VBT's tickets for this bucket
        vbt_bucket_tickets = vbt_tickets.get(bucket_id, [])

        # Get Dealer's tickets for this bucket
        dealer_bucket_tickets = dealer_tickets.get(bucket_id, [])

        # Get cash for VBT and Dealer from main system
        vbt_cash = _get_agent_cash(system, f"vbt_{bucket_id}")
        dealer_cash = _get_agent_cash(system, f"dealer_{bucket_id}")

        # M = 1 - P_prior (pure credit discount, no arbitrary haircut)
        M = credit_adjusted_mid

        # Per-bucket spread: base_spread + spread_sensitivity × P_prior
        # spread_sensitivity is from the VBT pricing model (default 0.6)
        base_O = BASE_SPREAD_BY_BUCKET.get(bucket_id, Decimal("0.08"))
        pricing_model = subsystem.vbt_pricing_model
        if pricing_model is not None:
            spread_anchor = pricing_model.compute_spread(base_O, shared_prior)
        else:
            spread_anchor = base_O + Decimal("0.6") * shared_prior

        # Store base spread for daily per-bucket VBT updates
        subsystem.base_spread_by_bucket[bucket_id] = base_O

        vbt = VBTState(
            bucket_id=bucket_id,
            agent_id=f"vbt_{bucket_id}",
            M=M,
            O=spread_anchor,
            phi_M=dealer_config.phi_M,
            phi_O=dealer_config.phi_O,
            clip_nonneg_B=dealer_config.clip_nonneg_B,
            inventory=list(vbt_bucket_tickets),  # VBT starts WITH inventory!
            cash=vbt_cash,
        )
        vbt.recompute_quotes()
        subsystem.vbts[bucket_id] = vbt

        # Store initial spread for spread_sensitivity computation
        subsystem.initial_spread_by_bucket[bucket_id] = spread_anchor

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

        # Capture initial equity (dealer cash + VBT mid x dealer inventory x S)
        initial_equity = dealer.cash + vbt.M * dealer.a * subsystem.params.S
        subsystem.metrics.initial_equity_by_bucket[bucket_id] = initial_equity
