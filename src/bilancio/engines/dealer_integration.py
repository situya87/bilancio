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

Implementation is split across four modules:
    - dealer_integration.py: Orchestration, public API, core types
    - dealer_wiring.py: Initialization and setup
    - dealer_trades.py: Trade execution logic
    - dealer_sync.py: State sync, maturity management, metrics

References:
    - Dealer module docs: docs/dealer_ring.md
    - Dealer specification: Full specification document
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bilancio.decision.protocols import VBTPricingModel
    from bilancio.engines.system import System
    from bilancio.information.profile import InformationProfile

from bilancio.core.ids import AgentId, InstrId
from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.metrics import RunMetrics
from bilancio.dealer.models import (
    DEFAULT_BUCKETS,
    BucketConfig,
    DealerState,
    Ticket,
    TicketId,
    TraderState,
    VBTState,
)
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.dealer.trading import TradeExecutor
from bilancio.decision.intentions import (
    collect_buy_intentions,
    collect_sell_intentions,
)
from bilancio.decision.profiles import TraderProfile, VBTProfile
from bilancio.domain.agent import AgentKind
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.dealer_sync import (
    _capture_dealer_snapshots,
    _capture_system_state_snapshot,
    _capture_trader_snapshots,
    _cleanup_orphaned_tickets,
    _ingest_new_payables,
    _pool_desk_cash,
    _sync_dealer_vbt_cash_from_system,
    _sync_dealer_vbt_cash_to_system,
    _sync_payable_ownership,
    _sync_trader_cash_from_system,
    _sync_trader_cash_to_system,
    _update_ticket_maturities,
    _update_vbt_credit_mids,
)

# --- Submodule imports (implementation) ---
from bilancio.engines.dealer_wiring import (
    _capture_initial_debt_to_money,
    _categorize_tickets_by_holder,
    _convert_payables_to_tickets,
    _ensure_dealer_vbt_agents,
    _initialize_balanced_market_makers,
    _initialize_market_makers,
    _initialize_traders,
)
from bilancio.engines.matching import DealerMatchingEngine


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

    dealers: dict[str, DealerState] = field(default_factory=dict)
    vbts: dict[str, VBTState] = field(default_factory=dict)
    traders: dict[AgentId, TraderState] = field(default_factory=dict)
    tickets: dict[TicketId, Ticket] = field(default_factory=dict)
    ticket_to_payable: dict[TicketId, InstrId] = field(default_factory=dict)
    payable_to_ticket: dict[InstrId, TicketId] = field(default_factory=dict)
    bucket_configs: list[BucketConfig] = field(default_factory=lambda: list(DEFAULT_BUCKETS))
    params: KernelParams = field(default_factory=lambda: KernelParams())
    executor: TradeExecutor | None = None
    enabled: bool = True
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    # Face value for scaling eligibility thresholds
    face_value: Decimal = Decimal(1)

    # Section 8 Metrics (functional dealer analysis)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    # Risk assessment module (optional)
    risk_assessor: RiskAssessor | None = None

    # Per-trader risk assessors (optional; when empty, shared risk_assessor is used)
    trader_assessors: dict[AgentId, RiskAssessor] = field(default_factory=dict)

    # Base outside-mid ratio (stored for daily VBT M recalculation)
    outside_mid_ratio: Decimal = Decimal(1)

    # Informedness parameters (Plan: credit-informed pricing)
    alpha_vbt: Decimal = Decimal(0)
    alpha_trader: Decimal = Decimal(0)
    kappa: Decimal | None = None

    # VBT credit facility: inventory ratio below which VBT injects cash
    layoff_threshold: Decimal = Decimal("0.7")

    # Serial counter for ticket IDs (continues from initialization)
    _ticket_serial_counter: int = 0

    # Decision module profiles
    trader_profile: TraderProfile = field(default_factory=TraderProfile)
    vbt_profile: VBTProfile = field(default_factory=VBTProfile)

    # Initial spread per bucket (for spread_sensitivity computation)
    initial_spread_by_bucket: dict[str, Decimal] = field(default_factory=dict)

    # Base spread per bucket (for per-bucket daily VBT spread updates)
    base_spread_by_bucket: dict[str, Decimal] = field(default_factory=dict)

    # VBT pricing model (optional — when None, uses inline logic for backward compat)
    vbt_pricing_model: VBTPricingModel | None = None

    # Number of trading sub-rounds per day.  High default lets the loop
    # run until no intentions remain (early-termination at line 670).
    trading_rounds: int = 100

    # Issuer-specific pricing overlay (Feature 1)
    issuer_specific_pricing: bool = False
    issuer_default_probs: dict[str, Decimal] = field(default_factory=dict)
    system_default_prob: Decimal = Decimal(0)

    # Concentration limit (Feature 3): max fraction of total dealer inventory
    # from a single issuer.  0 = disabled (no limit).
    dealer_concentration_limit: Decimal = Decimal(0)



def get_trader_assessor(subsystem: DealerSubsystem, trader_id: AgentId) -> RiskAssessor | None:
    """Return the per-trader assessor if available, else the shared one."""
    return subsystem.trader_assessors.get(trader_id, subsystem.risk_assessor)


def _get_agent_cash(system: System, agent_id: str) -> Decimal:
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
        if contract and contract.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT):
            total_cash += Decimal(contract.amount)

    return total_cash


def _assign_bucket(remaining_tau: int, bucket_configs: list[BucketConfig]) -> str:
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


# ---------------------------------------------------------------------------
# Public orchestration functions
# ---------------------------------------------------------------------------


def initialize_dealer_subsystem(
    system: System,
    dealer_config: DealerRingConfig,
    current_day: int = 0,
    risk_params: RiskAssessmentParams | None = None,
    trader_information_profile: InformationProfile | None = None,
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
    subsystem = DealerSubsystem(
        bucket_configs=dealer_config.buckets,
        params=KernelParams(S=dealer_config.ticket_size),
        rng=random.Random(dealer_config.seed),
    )

    # Initialize risk assessor if params provided
    if risk_params:
        subsystem.risk_assessor = RiskAssessor(risk_params)

    # Step 0: Create dealer/VBT agents in main system
    _ensure_dealer_vbt_agents(system, dealer_config.buckets)

    # Initialize trade executor
    subsystem.executor = TradeExecutor(subsystem.params, subsystem.rng)

    # Step 1: Convert Payables to Tickets
    serial_counter = _convert_payables_to_tickets(subsystem, system, current_day)

    # Step 2: Calculate total system cash for capital allocation
    # Dealer and VBT get NEW CASH from outside the system (not taken from traders)
    total_system_cash = Decimal(0)
    for agent_id, agent in system.state.agents.items():
        if agent.kind in (AgentKind.DEALER, AgentKind.VBT):
            continue
        total_system_cash += _get_agent_cash(system, agent_id)

    # Calculate dealer and VBT capital (NEW outside money)
    # Split evenly across buckets
    num_buckets = len(subsystem.bucket_configs)
    dealer_capital_per_bucket = (total_system_cash * dealer_config.dealer_share) / num_buckets
    vbt_capital_per_bucket = (total_system_cash * dealer_config.vbt_share) / num_buckets

    # Step 3: Initialize market makers
    _initialize_market_makers(
        subsystem,
        dealer_config,
        dealer_capital_per_bucket,
        vbt_capital_per_bucket,
    )

    # Step 4: Initialize traders (households only)
    _initialize_traders(subsystem, system)

    # Step 4b: Create per-trader assessors if information profile provided
    if risk_params and trader_information_profile:
        from bilancio.decision.factories import create_assessor
        from bilancio.information.service import InformationService

        for trader_id in subsystem.traders:
            info_service = InformationService(
                system=system,
                profile=trader_information_profile,
                observer_id=trader_id,
                rng=random.Random(subsystem.rng.randint(0, 2**31)),
            )
            subsystem.traders[trader_id].information_service = info_service
            subsystem.trader_assessors[trader_id] = create_assessor(
                risk_params, trader_information_profile, information_service=info_service
            )

    # Step 5: Capture initial debt-to-money ratio
    _capture_initial_debt_to_money(subsystem, system)

    subsystem._ticket_serial_counter = serial_counter

    return subsystem


def initialize_balanced_dealer_subsystem(
    system: System,
    dealer_config: DealerRingConfig,
    face_value: Decimal = Decimal("20"),
    outside_mid_ratio: Decimal = Decimal("0.75"),
    big_entity_share: Decimal = Decimal("0.25"),  # DEPRECATED
    vbt_share_per_bucket: Decimal = Decimal("0.20"),
    dealer_share_per_bucket: Decimal = Decimal("0.05"),
    mode: str = "active",
    current_day: int = 0,
    risk_params: RiskAssessmentParams | None = None,
    alpha_vbt: Decimal = Decimal("0"),
    alpha_trader: Decimal = Decimal("0"),
    kappa: Decimal | None = None,
    trader_profile: TraderProfile | None = None,
    vbt_profile: VBTProfile | None = None,
    trader_information_profile: InformationProfile | None = None,
) -> DealerSubsystem:
    """
    Initialize dealer subsystem for balanced scenarios (C vs D comparison).

    Per PDF specification (Plan 024):
    - VBT agents START with 20% of claims per maturity bucket + matching cash
    - Dealer agents START with 5% of claims per maturity bucket + matching cash
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
    # Compute shared kappa-informed prior (replaces alpha blending).
    # Both VBT and traders use the same prior to prevent adverse selection.
    from bilancio.dealer.priors import kappa_informed_prior

    if kappa is not None:
        shared_prior = kappa_informed_prior(kappa)
    else:
        shared_prior = Decimal("0.15")

    # Override risk_params from trader_profile and shared prior
    if risk_params is not None:
        from dataclasses import replace as dc_replace

        if trader_profile is not None:
            risk_params = dc_replace(
                risk_params,
                initial_prior=shared_prior,
                base_risk_premium=trader_profile.base_risk_premium,
                buy_risk_premium=trader_profile.buy_risk_premium,
                buy_premium_multiplier=trader_profile.buy_premium_multiplier,
                default_observability=trader_profile.default_observability,
            )
        else:
            risk_params = dc_replace(risk_params, initial_prior=shared_prior)

    subsystem = DealerSubsystem(
        bucket_configs=dealer_config.buckets,
        params=KernelParams(S=face_value),  # S=face_value: kernel accounts for real ticket size
        rng=random.Random(dealer_config.seed),
        enabled=(mode == "active"),  # Disable trading for passive mode
        face_value=face_value,
        outside_mid_ratio=outside_mid_ratio,
        alpha_vbt=alpha_vbt,
        alpha_trader=alpha_trader,
        kappa=kappa,
    )

    # Attach decision profiles
    if trader_profile is not None:
        subsystem.trader_profile = trader_profile
    if vbt_profile is not None:
        subsystem.vbt_profile = vbt_profile

    # Construct VBT pricing model from config params
    from bilancio.decision.valuers import CreditAdjustedVBTPricing

    effective_vbt_profile = vbt_profile or VBTProfile()
    subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
        mid_sensitivity=effective_vbt_profile.mid_sensitivity,
        spread_sensitivity=effective_vbt_profile.spread_sensitivity,
        outside_mid_ratio=outside_mid_ratio,
    )

    # Initialize risk assessor if params provided
    if risk_params:
        subsystem.risk_assessor = RiskAssessor(risk_params)

    # Step 0: Ensure dealer/VBT agents exist in main system
    _ensure_dealer_vbt_agents(system, dealer_config.buckets)

    # Initialize trade executor (with layoff threshold for VBT credit facility)
    subsystem.executor = TradeExecutor(
        subsystem.params, subsystem.rng, layoff_threshold=subsystem.layoff_threshold
    )

    # Step 1: Convert Payables to Tickets
    serial_counter = _convert_payables_to_tickets(subsystem, system, current_day)

    # Categorize tickets by holder type
    bucket_names = [bc.name for bc in dealer_config.buckets]
    vbt_tickets, dealer_tickets, _ = _categorize_tickets_by_holder(
        subsystem,
        bucket_names,
    )

    # Step 2: Initialize VBT and Dealer states per bucket (with inventory)
    _initialize_balanced_market_makers(
        subsystem,
        system,
        dealer_config,
        vbt_tickets,
        dealer_tickets,
        shared_prior=shared_prior,
    )

    # Step 3: Initialize traders (regular household agents, skip VBT/Dealer/big)
    _initialize_traders(
        subsystem,
        system,
        skip_prefixes=("vbt_", "dealer_", "big_"),
    )

    # Step 3b: Create per-trader assessors if information profile provided
    if risk_params and trader_information_profile:
        from bilancio.decision.factories import create_assessor
        from bilancio.information.service import InformationService

        for trader_id in subsystem.traders:
            info_service = InformationService(
                system=system,
                profile=trader_information_profile,
                observer_id=trader_id,
                rng=random.Random(subsystem.rng.randint(0, 2**31)),
            )
            subsystem.traders[trader_id].information_service = info_service
            subsystem.trader_assessors[trader_id] = create_assessor(
                risk_params, trader_information_profile, information_service=info_service
            )

    # Step 4: Capture initial debt-to-money ratio
    _capture_initial_debt_to_money(
        subsystem,
        system,
        exclude_kinds=("dealer", "vbt"),
        exclude_prefixes=("vbt_", "dealer_", "big_"),
    )

    # Store serial counter so ingestion can continue the sequence
    subsystem._ticket_serial_counter = serial_counter

    return subsystem


def run_dealer_trading_phase(
    subsystem: DealerSubsystem, system: System, current_day: int
) -> list[dict[str, object]]:
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
    # Always sync cash from main system so that the corresponding
    # _sync_*_to_system (called later in sync_dealer_to_system)
    # sees the current cash and computes delta=0 when no trades occurred.
    # Without this, other-phase cash changes get erroneously reversed.
    _sync_trader_cash_from_system(subsystem, system)
    _sync_dealer_vbt_cash_from_system(subsystem, system)

    if not subsystem.enabled:
        return []

    events: list[dict[str, object]] = []

    # Phase 0.5: Clean up tickets whose payables were removed
    _cleanup_orphaned_tickets(subsystem, system)

    # Phase 0.75: Ingest new payables as tickets (from rollover)
    _ingest_new_payables(subsystem, system, current_day)

    # Phase 1: Update ticket maturities and buckets
    _update_ticket_maturities(subsystem, system, current_day)

    # Phase 1.5: Cash pooling — redistribute cash equally across desks.
    # The three dealer desks (short/mid/long) are one firm sharing a balance sheet.
    # Same for VBT desks. Pooling ensures default losses and settlement revenue
    # are amortized across all desks, not concentrated in the short bucket.
    _pool_desk_cash(subsystem)

    # Phase 1.75: Update VBT mid prices from risk assessor's credit estimate.
    # VBTs are credit-aware holders: M = outside_mid_ratio × (1 - P_default).
    # As the system learns actual default rates, VBT mid drifts accordingly,
    # keeping dealer asks below par so rational buys remain viable.
    _update_vbt_credit_mids(subsystem, current_day, system)

    # Phase 2: Recompute dealer quotes for all buckets
    for bucket_id, dealer in subsystem.dealers.items():
        vbt = subsystem.vbts[bucket_id]
        recompute_dealer_state(dealer, vbt, subsystem.params)

    # Capture daily snapshots for metrics (Section 8.1)
    _capture_dealer_snapshots(subsystem, current_day)
    _capture_trader_snapshots(subsystem, current_day)
    _capture_system_state_snapshot(subsystem, current_day)  # Plan 022

    # Phase 3+4: Trading sub-rounds
    # Each sub-round re-collects intentions (so a trader who sold in round 1
    # can sell another ticket in round 2 if still short on cash) and
    # re-matches against dealer quotes.  Dealer state is recomputed between
    # rounds to reflect updated inventory/cash.
    n_rounds = getattr(subsystem, "trading_rounds", 1)
    for _round in range(n_rounds):
        sell_intentions = collect_sell_intentions(subsystem, current_day)
        buy_intentions = collect_buy_intentions(subsystem, current_day)

        if not sell_intentions and not buy_intentions:
            break  # No one wants to trade — stop early

        DealerMatchingEngine().execute(
            subsystem,
            system,
            current_day,
            sell_intentions,
            buy_intentions,
            events,
        )

        # Recompute dealer quotes between rounds (updated inventory/cash)
        if _round < n_rounds - 1:
            for bucket_id, dealer in subsystem.dealers.items():
                vbt = subsystem.vbts[bucket_id]
                recompute_dealer_state(dealer, vbt, subsystem.params)

    return events


def compute_passive_pnl(
    subsystem: DealerSubsystem,
    system: System,
) -> dict[str, Any]:
    """Compute PnL for passive dealer entities (hold-only, no trading).

    In passive mode, dealers and VBTs hold their initial inventory but never
    trade. This function computes their mark-to-mid equity at the end of the
    simulation and compares it to their initial equity.

    Uses the same formula as active PnL (mark-to-mid) for fair comparison:
        final_equity = dealer_cash + VBT_M × inventory_count × S
        passive_pnl = final_equity - initial_equity

    VBT_M in passive mode is the initial value (never updated since no trading
    phase runs).

    Args:
        subsystem: Dealer subsystem (with enabled=False for passive)
        system: Main System instance

    Returns:
        Dictionary compatible with dealer_metrics.json format (summary dict)
    """
    pnl_by_bucket: dict[str, float] = {}
    return_by_bucket: dict[str, float] = {}
    total_pnl = Decimal(0)
    total_initial_equity = Decimal(0)

    for bucket_id, dealer in subsystem.dealers.items():
        vbt = subsystem.vbts[bucket_id]
        initial_equity = subsystem.metrics.initial_equity_by_bucket.get(bucket_id, Decimal(0))

        # Read current dealer cash from system
        dealer_cash = _get_agent_cash(system, dealer.agent_id)

        # Count remaining inventory tickets whose underlying payable still
        # exists in the system.  In passive mode _cleanup_orphaned_tickets is
        # never called, so dealer.inventory still references tickets whose
        # payables have settled or defaulted (and been removed from contracts).
        inventory_count = 0
        for ticket in dealer.inventory:
            payable_id = subsystem.ticket_to_payable.get(ticket.id)
            if payable_id and payable_id in system.state.contracts:
                inventory_count += 1

        # Mark to mid: final_equity = cash + M × inventory × S
        final_equity = dealer_cash + vbt.M * inventory_count * subsystem.params.S

        bucket_pnl = final_equity - initial_equity
        pnl_by_bucket[bucket_id] = float(bucket_pnl)

        if initial_equity > 0:
            return_by_bucket[bucket_id] = float(bucket_pnl / initial_equity)
        else:
            return_by_bucket[bucket_id] = 0.0

        total_pnl += bucket_pnl
        total_initial_equity += initial_equity

    total_return = float(total_pnl / total_initial_equity) if total_initial_equity > 0 else 0.0

    return {
        "dealer_total_pnl": float(total_pnl),
        "dealer_total_return": total_return,
        "dealer_profitable": total_pnl >= 0,
        "dealer_pnl_by_bucket": pnl_by_bucket,
        "dealer_return_by_bucket": return_by_bucket,
        "total_trades": 0,
        "total_sell_trades": 0,
        "total_buy_trades": 0,
        "interior_trades": 0,
        "passthrough_trades": 0,
        "spread_income_total": 0.0,
        "initial_total_debt": float(subsystem.metrics.initial_total_debt),
        "initial_total_money": float(subsystem.metrics.initial_total_money),
        "debt_to_money_ratio": float(subsystem.metrics.debt_to_money_ratio),
    }


def sync_dealer_to_system(
    subsystem: DealerSubsystem,
    system: System,
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
    # Step 1: Update Payable ownership for transferred claims
    _sync_payable_ownership(subsystem, system)

    # Step 2: Sync trader cash balances
    _sync_trader_cash_to_system(subsystem, system)

    # Step 3: Sync dealer/VBT cash balances
    # The VBT credit facility and normal trading modify dealer/VBT cash within
    # the subsystem. These changes must be reflected in the main system to
    # maintain the CB cash invariant.
    _sync_dealer_vbt_cash_to_system(subsystem, system)
