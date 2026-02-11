"""
Dealer ring simulation orchestrator.

This module implements the full dealer ring event loop per Section 11 of the
specification. It coordinates all components of the dealer system:

- Ticket maturity updates and rebucketing (Phase 1)
- Dealer pre-computation (Phase 2)
- Trading eligibility sets (Phase 3)
- Randomized order flow (Phase 4)
- Settlement with proportional recovery (Phase 5)
- VBT anchor updates (Phase 6)

The simulation uses Decimal throughout for precision and implements all
programmatic assertions (C1-C6) to ensure correctness.

References:
- Specification Section 11: Full Event Loop
- Specification Section 10: Ring Trader Policies
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import random
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

# Cash precision for settlement calculations to avoid floating-point accumulation errors
# Uses 6 decimal places which is standard for financial calculations
CASH_PRECISION = Decimal("0.000001")

from bilancio.core.ids import AgentId, new_id
from .models import (
    Ticket, DealerState, VBTState, TraderState,
    BucketConfig, DEFAULT_BUCKETS, TicketId,
)
from .kernel import KernelParams, recompute_dealer_state
from .trading import TradeExecutor
from .events import EventLog
from .assertions import run_all_assertions, assert_c6_anchor_timing
from .risk_assessment import RiskAssessor, RiskAssessmentParams


@dataclass
class DaySnapshot:
    """
    Snapshot of simulation state at end of day.

    Captures deep copies of all mutable state for reporting.

    Attributes:
        day: Day number (0 for initial setup, 1+ for each run_day)
        dealers: Per-bucket dealer states as serializable dicts
        vbts: Per-bucket VBT states as serializable dicts
        traders: Trader states as serializable dicts
        tickets: All tickets as serializable dicts
        events: Events that occurred on this day
    """
    day: int
    dealers: dict[str, dict[str, Any]]  # bucket_id -> dealer state dict
    vbts: dict[str, dict[str, Any]]      # bucket_id -> VBT state dict
    traders: dict[str, dict[str, Any]]   # agent_id -> trader state dict
    tickets: dict[str, dict[str, Any]]   # ticket_id -> ticket dict
    events: list[dict[str, Any]]         # Events for this day only


@dataclass
class DealerRingConfig:
    """
    Configuration for dealer ring simulation.

    Attributes:
        ticket_size: Standard ticket size S (face value)
        buckets: List of maturity bucket configurations
        vbt_anchors: VBT anchors per bucket {bucket_name: (M, O)}
        phi_M: VBT mid sensitivity to loss
        phi_O: VBT spread sensitivity to loss
        M_min: Guard threshold for dealer collapse
        clip_nonneg_B: Clip VBT bid to be non-negative
        pi_sell: Probability that next arrival is SELL (vs BUY)
        N_max: Maximum arrivals per period
        horizon_H: Minimum days to liability for buy eligibility
        buffer_B: Cash buffer for buy eligibility
        dealer_share: Initial ticket allocation to dealers (0-1)
        vbt_share: Initial ticket allocation to VBTs (0-1)
        seed: Random seed for reproducibility
        max_days: Default simulation duration
        enable_vbt_anchor_updates: Whether to update VBT anchors based on losses

    References:
        - Section 8: Kernel parameters (ticket_size, M_min)
        - Section 9: VBT anchor parameters (phi_M, phi_O)
        - Section 10: Trading policies (horizon_H, buffer_B)
        - Section 11: Order flow (pi_sell, N_max)
    """
    # Tickets
    ticket_size: Decimal = Decimal(1)  # S

    # Buckets
    buckets: list[BucketConfig] = field(default_factory=lambda: list(DEFAULT_BUCKETS))

    # VBT anchors per bucket: {bucket_name: (M, O)}
    vbt_anchors: dict[str, tuple[Decimal, Decimal]] = field(default_factory=lambda: {
        "short": (Decimal(1), Decimal("0.20")),
        "mid": (Decimal(1), Decimal("0.30")),
        "long": (Decimal(1), Decimal("0.40")),
    })

    # VBT sensitivity
    phi_M: Decimal = Decimal(1)
    phi_O: Decimal = Decimal("0.6")

    # Guard threshold
    M_min: Decimal = Decimal("0.02")
    clip_nonneg_B: bool = True

    # Order flow
    # NOTE: pi_sell is deprecated and no longer used. Order flow direction is now
    # determined endogenously by trader economic conditions (sell-eligible vs buy-eligible).
    # Kept for backward compatibility but has no effect.
    pi_sell: Decimal = Decimal("0.5")  # DEPRECATED: P(next arrival is SELL)
    N_max: int = 3  # Max arrivals per period

    # Trader policies
    horizon_H: int = 3  # Min days to liability for investment
    buffer_B: Decimal = Decimal(1)  # Cash buffer for investment eligibility

    # Initial allocation (for setup)
    dealer_share: Decimal = Decimal("0.25")
    vbt_share: Decimal = Decimal("0.50")

    # RNG
    seed: int = 42

    # Simulation control
    max_days: int = 30
    enable_vbt_anchor_updates: bool = True


class DealerRingSimulation:
    """
    Main simulation orchestrator for the dealer ring.

    Event loop per specification Section 11:
    1. Update maturities and buckets
    2. Dealer pre-computation (per bucket)
    3. Compute trading eligibility sets
    4. Randomized one-ticket order flow
    5. Settlement with proportional recovery
    6. VBT anchor update (if enabled)

    The simulation maintains:
    - Per-bucket dealers and VBTs (market makers)
    - Ring traders with single-issuer constraint
    - Global ticket registry
    - Event log for observability

    Attributes:
        config: Simulation configuration
        rng: Random number generator
        day: Current simulation day
        dealers: Per-bucket dealer states
        vbts: Per-bucket VBT states
        traders: Ring trader states
        all_tickets: Global ticket registry by ID
        events: Event log for observability
        params: Kernel parameters
        executor: Trade executor

    References:
        - Section 11: Full Event Loop
        - Section 10: Ring Trader Policies
    """

    def __init__(self, config: DealerRingConfig, risk_assessor: RiskAssessor | None = None):
        """
        Initialize simulation orchestrator.

        Args:
            config: Simulation configuration
            risk_assessor: Optional risk assessor for trader decisions
        """
        self.config = config
        self.rng = random.Random(config.seed)

        # State
        self.day: int = 0
        self.dealers: dict[str, DealerState] = {}  # bucket -> dealer
        self.vbts: dict[str, VBTState] = {}        # bucket -> VBT
        self.traders: dict[AgentId, TraderState] = {}
        self.all_tickets: dict[TicketId, Ticket] = {}  # Global ticket registry

        # Event log
        self.events = EventLog()

        # Snapshots for reporting
        self.snapshots: list[DaySnapshot] = []

        # Kernel params
        self.params = KernelParams(S=config.ticket_size)
        self.executor = TradeExecutor(self.params, self.rng)
        self.risk_assessor = risk_assessor

        # Initialize dealers and VBTs
        self._init_market_makers()

    def _init_market_makers(self) -> None:
        """
        Initialize dealers and VBTs for each bucket.

        Creates one dealer and one VBT per bucket, with anchors from config.

        References:
            - Section 8: Dealer state initialization
            - Section 9: VBT anchor initialization
        """
        for bucket in self.config.buckets:
            bucket_id = bucket.name

            # Create dealer
            dealer = DealerState(
                bucket_id=bucket_id,
                agent_id=new_id(f"dealer_{bucket_id}"),
            )
            self.dealers[bucket_id] = dealer

            # Create VBT with configured anchors
            M, O = self.config.vbt_anchors.get(bucket_id, (Decimal(1), Decimal("0.30")))
            vbt = VBTState(
                bucket_id=bucket_id,
                agent_id=new_id(f"vbt_{bucket_id}"),
                M=M,
                O=O,
                phi_M=self.config.phi_M,
                phi_O=self.config.phi_O,
                O_min=Decimal(0),
                clip_nonneg_B=self.config.clip_nonneg_B,
            )
            vbt.recompute_quotes()
            self.vbts[bucket_id] = vbt

    def _capture_snapshot(self) -> None:
        """
        Capture deep copy of current state for reporting.

        Creates serializable dicts from all state objects.
        Called at end of setup and after each day.
        """
        # Serialize dealer states
        dealers_dict = {}
        for bucket_id, dealer in self.dealers.items():
            dealers_dict[bucket_id] = {
                "bucket_id": dealer.bucket_id,
                "agent_id": dealer.agent_id,
                "cash": dealer.cash,
                "a": dealer.a,
                "x": dealer.x,
                "V": dealer.V,
                "K_star": dealer.K_star,
                "X_star": dealer.X_star,
                "N": dealer.N,
                "lambda_": dealer.lambda_,
                "I": dealer.I,
                "midline": dealer.midline,
                "bid": dealer.bid,
                "ask": dealer.ask,
                "is_pinned_bid": dealer.is_pinned_bid,
                "is_pinned_ask": dealer.is_pinned_ask,
                "inventory": [
                    {"id": t.id, "issuer_id": t.issuer_id, "face": t.face, "remaining_tau": t.remaining_tau}
                    for t in dealer.inventory
                ],
            }

        # Serialize VBT states
        vbts_dict = {}
        for bucket_id, vbt in self.vbts.items():
            vbts_dict[bucket_id] = {
                "bucket_id": vbt.bucket_id,
                "agent_id": vbt.agent_id,
                "M": vbt.M,
                "O": vbt.O,
                "A": vbt.A,
                "B": vbt.B,
                "cash": vbt.cash,
                "inventory": [
                    {"id": t.id, "issuer_id": t.issuer_id, "face": t.face, "remaining_tau": t.remaining_tau}
                    for t in vbt.inventory
                ],
            }

        # Serialize trader states
        traders_dict = {}
        for agent_id, trader in self.traders.items():
            traders_dict[agent_id] = {
                "agent_id": trader.agent_id,
                "cash": trader.cash,
                "defaulted": trader.defaulted,
                "asset_issuer_id": trader.asset_issuer_id,
                "tickets_owned": [
                    {"id": t.id, "issuer_id": t.issuer_id, "face": t.face, "remaining_tau": t.remaining_tau, "bucket_id": t.bucket_id}
                    for t in trader.tickets_owned
                ],
                "obligations": [
                    {"id": t.id, "owner_id": t.owner_id, "face": t.face, "maturity_day": t.maturity_day}
                    for t in trader.obligations
                ],
            }

        # Serialize all tickets
        tickets_dict = {}
        for ticket_id, ticket in self.all_tickets.items():
            tickets_dict[ticket_id] = {
                "id": ticket.id,
                "issuer_id": ticket.issuer_id,
                "owner_id": ticket.owner_id,
                "face": ticket.face,
                "maturity_day": ticket.maturity_day,
                "remaining_tau": ticket.remaining_tau,
                "bucket_id": ticket.bucket_id,
            }

        # Get events for this day only
        day_events = [e for e in self.events.events if e.get("day") == self.day]

        snapshot = DaySnapshot(
            day=self.day,
            dealers=dealers_dict,
            vbts=vbts_dict,
            traders=traders_dict,
            tickets=tickets_dict,
            events=day_events,
        )
        self.snapshots.append(snapshot)

    def setup_ring(
        self,
        traders: list[TraderState],
        tickets: list[Ticket],
    ) -> None:
        """
        Set up the ring with traders and initial tickets.

        Allocates tickets to dealers/VBTs/traders per dealer_share/vbt_share.
        All tickets must have maturity_day and remaining_tau set.

        Allocation strategy:
        - dealer_share fraction goes to dealers (by bucket)
        - vbt_share fraction goes to VBTs (by bucket)
        - Remainder goes to traders (owners = issuers' assets)

        Args:
            traders: List of trader states (registers them in the simulation)
            tickets: List of tickets to allocate (must be bucketed)

        References:
            - Section 11: Initial setup before event loop
        """
        # Register traders
        for trader in traders:
            self.traders[trader.agent_id] = trader

        # Register tickets and assign buckets
        for ticket in tickets:
            self.all_tickets[ticket.id] = ticket

            # Determine bucket based on remaining maturity
            computed_bucket = self._compute_bucket(ticket.remaining_tau)

            if computed_bucket is None:
                raise ValueError(
                    f"Ticket {ticket.id} has remaining_tau={ticket.remaining_tau} "
                    f"which does not fit any configured bucket. "
                    f"Buckets: {[(b.name, b.tau_min, b.tau_max) for b in self.config.buckets]}"
                )

            ticket.bucket_id = computed_bucket

        # Sort tickets by bucket for allocation
        tickets_by_bucket: dict[str, list[Ticket]] = {}
        for ticket in tickets:
            bucket_id = ticket.bucket_id
            if bucket_id not in tickets_by_bucket:
                tickets_by_bucket[bucket_id] = []
            tickets_by_bucket[bucket_id].append(ticket)

        # Allocate tickets to dealers, VBTs, and traders
        for bucket_id, bucket_tickets in tickets_by_bucket.items():
            n_total = len(bucket_tickets)

            # Compute allocation counts
            n_dealer = int(n_total * self.config.dealer_share)
            n_vbt = int(n_total * self.config.vbt_share)
            n_trader = n_total - n_dealer - n_vbt  # Remainder to traders

            # Allocate to dealer
            dealer = self.dealers[bucket_id]
            for i in range(n_dealer):
                ticket = bucket_tickets[i]
                ticket.owner_id = dealer.agent_id
                dealer.inventory.append(ticket)

            # Allocate to VBT
            vbt = self.vbts[bucket_id]
            for i in range(n_dealer, n_dealer + n_vbt):
                ticket = bucket_tickets[i]
                ticket.owner_id = vbt.agent_id
                vbt.inventory.append(ticket)

            # Allocate to traders (owner = issuer for initial distribution)
            for i in range(n_dealer + n_vbt, n_total):
                ticket = bucket_tickets[i]
                issuer_id = ticket.issuer_id

                if issuer_id not in self.traders:
                    raise ValueError(
                        f"Ticket {ticket.id} has issuer_id {issuer_id} "
                        f"which is not a registered trader."
                    )

                trader = self.traders[issuer_id]
                ticket.owner_id = issuer_id
                trader.tickets_owned.append(ticket)

                # Set asset issuer constraint
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = issuer_id

        # Recompute dealer states after initial allocation
        for bucket_id in self.dealers:
            recompute_dealer_state(
                self.dealers[bucket_id],
                self.vbts[bucket_id],
                self.params,
            )

        # Capture initial state snapshot (Day 0)
        self._capture_snapshot()

    def run_day(self) -> None:
        """
        Execute one simulation day (Section 11 event loop).

        Event loop phases:
        1. Update maturities and buckets
        2. Dealer pre-computation (per bucket)
        3. Compute trading eligibility sets
        4. Randomized order flow
        5. Settlement with proportional recovery
        6. VBT anchor update (if enabled)

        References:
            - Section 11: Full Event Loop
        """
        self.day += 1
        logger.debug("dealer day %d: starting event loop (%d tickets)", self.day, len(self.all_tickets))
        self.events.log_day_start(self.day)

        # Phase 1: Update maturities and buckets
        self._update_maturities()
        self._rebucket_tickets()

        # Phase 2: Dealer pre-computation
        for bucket_id in self.dealers:
            recompute_dealer_state(
                self.dealers[bucket_id],
                self.vbts[bucket_id],
                self.params,
            )
            # Log quotes for observability
            self._log_quotes(bucket_id)

        # Phase 3: Compute eligibility sets
        sell_eligible = self._compute_sell_eligible()
        buy_eligible = self._compute_buy_eligible()

        # Phase 4: Randomized order flow
        self._process_order_flow(sell_eligible, buy_eligible)

        # Phase 5: Settlement
        self._settle_maturing_debt()

        # Phase 6: VBT anchor update
        if self.config.enable_vbt_anchor_updates:
            self._update_vbt_anchors()

        # Capture end-of-day snapshot
        self._capture_snapshot()

    def run(self, max_days: int | None = None) -> None:
        """
        Run simulation for specified days.

        Args:
            max_days: Number of days to simulate (defaults to config.max_days)
        """
        days = max_days or self.config.max_days
        logger.info("dealer simulation: running %d days (%d traders, %d tickets)",
                     days, len(self.traders), len(self.all_tickets))
        for _ in range(days):
            self.run_day()

    # =====================================================================
    # Phase 1: Update maturities and rebucket
    # =====================================================================

    def _update_maturities(self) -> None:
        """
        Decrement remaining_tau for all tickets.

        This reflects the passage of one day. Tickets with remaining_tau
        reaching 0 will mature during settlement (Phase 5).

        References:
            - Section 11.1: Maturity updates
        """
        for ticket in self.all_tickets.values():
            if ticket.remaining_tau > 0:
                ticket.remaining_tau -= 1

    def _rebucket_tickets(self) -> None:
        """
        Reassign bucket_id based on remaining_tau.

        If dealer/VBT holds migrating ticket, execute internal sale:
        - Event 11: Dealer-to-dealer internal sale at old-bucket ask
        - Event 12: VBT-to-VBT internal sale at VBT mid M

        References:
            - Section 11.1: Rebucketing logic
            - Section 6.11: Internal dealer sale
            - Section 6.12: Internal VBT sale
        """
        for ticket in self.all_tickets.values():
            old_bucket = ticket.bucket_id
            new_bucket = self._compute_bucket(ticket.remaining_tau)

            # Skip if bucket unchanged or ticket matured
            if new_bucket is None or new_bucket == old_bucket:
                continue

            # Check if dealer holds this ticket
            old_dealer = self.dealers.get(old_bucket)
            if old_dealer and ticket.owner_id == old_dealer.agent_id:
                self._rebucket_dealer_ticket(ticket, old_bucket, new_bucket)
                continue

            # Check if VBT holds this ticket
            old_vbt = self.vbts.get(old_bucket)
            if old_vbt and ticket.owner_id == old_vbt.agent_id:
                self._rebucket_vbt_ticket(ticket, old_bucket, new_bucket)
                continue

            # Trader-held ticket: just update bucket_id
            ticket.bucket_id = new_bucket

    def _compute_bucket(self, remaining_tau: int) -> str | None:
        """
        Determine bucket for given remaining maturity.

        Args:
            remaining_tau: Remaining days to maturity

        Returns:
            Bucket name, or None if ticket has matured (tau <= 0)

        References:
            - Section 11.1: Bucket assignment rules
        """
        if remaining_tau <= 0:
            return None

        for bucket in self.config.buckets:
            if bucket.tau_max is None:
                # Unbounded upper range
                if remaining_tau >= bucket.tau_min:
                    return bucket.name
            else:
                # Bounded range
                if bucket.tau_min <= remaining_tau <= bucket.tau_max:
                    return bucket.name

        # No bucket found - configuration error
        return None

    def _rebucket_dealer_ticket(
        self,
        ticket: Ticket,
        old_bucket: str,
        new_bucket: str,
    ) -> None:
        """
        Event 11: Internal sale from old-bucket dealer to new-bucket dealer.

        Transfer occurs at the receiving bucket's VBT mid anchor M. This
        preserves equity for both dealers (E = C + M*a is unchanged).

        Args:
            ticket: Ticket to transfer
            old_bucket: Source bucket
            new_bucket: Destination bucket

        References:
            - Section 6.11: Internal dealer sale
            - PDF spec: "internal sale at the Mid bucket mid M_M"
        """
        old_dealer = self.dealers[old_bucket]
        new_dealer = self.dealers[new_bucket]

        # Execute at receiving bucket's VBT mid anchor (preserves equity)
        price = self.vbts[new_bucket].M

        # Update balance sheets
        old_dealer.inventory.remove(ticket)
        old_dealer.cash += price

        new_dealer.inventory.append(ticket)
        new_dealer.cash -= price

        # Transfer ownership
        ticket.owner_id = new_dealer.agent_id
        ticket.bucket_id = new_bucket

        # Recompute both dealers
        recompute_dealer_state(old_dealer, self.vbts[old_bucket], self.params)
        recompute_dealer_state(new_dealer, self.vbts[new_bucket], self.params)

        # Log event
        self.events.log_rebucket(
            day=self.day,
            ticket_id=ticket.id,
            old_bucket=old_bucket,
            new_bucket=new_bucket,
            price=price,
            holder_type="dealer",
        )

    def _rebucket_vbt_ticket(
        self,
        ticket: Ticket,
        old_bucket: str,
        new_bucket: str,
    ) -> None:
        """
        Event 12: Internal sale from old-bucket VBT to new-bucket VBT.

        Transfer occurs at old-bucket VBT mid M. This implements VBT-to-VBT
        internal liquidity provision.

        Args:
            ticket: Ticket to transfer
            old_bucket: Source bucket
            new_bucket: Destination bucket

        References:
            - Section 6.12: Internal VBT sale
        """
        old_vbt = self.vbts[old_bucket]
        new_vbt = self.vbts[new_bucket]

        # Execute at old-bucket VBT mid
        price = old_vbt.M

        # Update balance sheets
        old_vbt.inventory.remove(ticket)
        old_vbt.cash += price

        new_vbt.inventory.append(ticket)
        new_vbt.cash -= price

        # Transfer ownership
        ticket.owner_id = new_vbt.agent_id
        ticket.bucket_id = new_bucket

        # Log event
        self.events.log_rebucket(
            day=self.day,
            ticket_id=ticket.id,
            old_bucket=old_bucket,
            new_bucket=new_bucket,
            price=price,
            holder_type="vbt",
        )

    # =====================================================================
    # Phase 2: Dealer pre-computation
    # =====================================================================

    def _log_quotes(self, bucket_id: str) -> None:
        """
        Log dealer quote state for observability.

        Args:
            bucket_id: Bucket identifier
        """
        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        self.events.log_quote(
            day=self.day,
            bucket=bucket_id,
            dealer_bid=dealer.bid,
            dealer_ask=dealer.ask,
            vbt_bid=vbt.B,
            vbt_ask=vbt.A,
            inventory=dealer.a,
            capacity=dealer.X_star,
            is_pinned_bid=dealer.is_pinned_bid,
            is_pinned_ask=dealer.is_pinned_ask,
        )

    # =====================================================================
    # Phase 3: Compute eligibility sets
    # =====================================================================

    def _compute_sell_eligible(self) -> set[AgentId]:
        """
        Traders with shortfall > 0 and at least one ticket owned.

        Sell eligibility reflects liquidity pressure: traders who face
        payment obligations today and own tickets they can sell.

        Returns:
            Set of agent IDs eligible to sell

        References:
            - Section 10.3: Sell eligibility rule
            - Section 11.3: Eligibility computation
        """
        eligible = set()
        for trader in self.traders.values():
            has_shortfall = trader.shortfall(self.day) > 0
            has_tickets = len(trader.tickets_owned) > 0

            if has_shortfall and has_tickets:
                eligible.add(trader.agent_id)

        return eligible

    def _compute_buy_eligible(self) -> set[AgentId]:
        """
        Traders with horizon >= H and cash > buffer.

        Buy eligibility reflects investment capacity: traders with sufficient
        cash buffer and distant enough liabilities to invest safely.

        The horizon H is defined as the number of days until the trader's
        next liability. If no future liabilities exist, horizon is infinite.

        Returns:
            Set of agent IDs eligible to buy

        References:
            - Section 10.4: Buy eligibility rule
            - Section 11.3: Eligibility computation
        """
        eligible = set()
        for trader in self.traders.values():
            # Check cash buffer
            has_buffer = trader.cash > self.config.buffer_B

            # Check horizon (days to next liability)
            next_liability_day = trader.earliest_liability_day(self.day)
            if next_liability_day is None:
                # No future liabilities - infinite horizon
                has_horizon = True
            else:
                horizon = next_liability_day - self.day
                has_horizon = horizon >= self.config.horizon_H

            if has_buffer and has_horizon:
                eligible.add(trader.agent_id)

        return eligible

    # =====================================================================
    # Phase 4: Randomized order flow
    # =====================================================================

    def _process_order_flow(
        self,
        sell_eligible: set[AgentId],
        buy_eligible: set[AgentId],
    ) -> None:
        """
        Order flow with crossing: match offsetting orders before taking inventory.

        Two-phase approach that minimizes dealer inventory risk:

        Phase 1 - Order Crossing:
        Match sellers with buyers. For each matched pair:
        - Seller sells to dealer at bid
        - Dealer immediately sells to buyer at ask
        - Dealer earns spread but inventory stays flat

        Phase 2 - Residual Flow:
        Remaining unmatched traders trade with dealer:
        - Dealer takes inventory positions
        - When capacity exhausted, passthrough to VBT

        This is more realistic: real dealers minimize inventory risk by
        crossing offsetting flow before taking principal positions.

        Args:
            sell_eligible: Set of agents eligible to sell
            buy_eligible: Set of agents eligible to buy

        References:
            - Section 11.4: Order flow generation
            - Treynor (1987): Dealer as intermediary, not speculator
        """
        # Build working sets (exclude traders who are both - they sell, not buy)
        sellers = sell_eligible.copy()
        buyers = (buy_eligible - sell_eligible).copy()

        # =====================================================================
        # Phase 1: Order Crossing - match offsetting orders
        # =====================================================================
        # Match as many seller-buyer pairs as possible
        n_crossable = min(len(sellers), len(buyers))

        if n_crossable > 0:
            # Randomly select which sellers and buyers get matched
            sellers_list = list(sellers)
            buyers_list = list(buyers)
            self.rng.shuffle(sellers_list)
            self.rng.shuffle(buyers_list)

            for i in range(n_crossable):
                seller_id = sellers_list[i]
                buyer_id = buyers_list[i]

                # Execute crossed trade
                crossed = self._process_crossed_trade(
                    seller_id, buyer_id, sell_eligible, buy_eligible
                )

                if crossed:
                    # Remove from working sets
                    sellers.discard(seller_id)
                    buyers.discard(buyer_id)

        # =====================================================================
        # Phase 2: Residual Flow - unmatched traders go to dealer
        # =====================================================================
        # Remaining traders trade with dealer (inventory accumulation)
        # Limited by N_max arrivals
        if self.config.N_max <= 0:
            return

        # Combine remaining into pool
        remaining_pool = list(sellers) + list(buyers)
        if not remaining_pool:
            return

        # Process up to N_max unmatched arrivals
        n_arrivals = min(self.rng.randint(1, self.config.N_max), len(remaining_pool))
        self.rng.shuffle(remaining_pool)

        for i in range(n_arrivals):
            agent_id = remaining_pool[i]

            if agent_id in sellers:
                self._process_sell_for_agent(agent_id, sell_eligible)
            else:
                self._process_buy_for_agent(agent_id, buy_eligible)

    def _process_crossed_trade(
        self,
        seller_id: AgentId,
        buyer_id: AgentId,
        sell_eligible: set[AgentId],
        buy_eligible: set[AgentId],
    ) -> bool:
        """
        Execute a crossed trade: seller → dealer → buyer (atomic).

        The dealer intermediates between seller and buyer:
        - Buys from seller at dealer's bid
        - Sells to buyer at dealer's ask
        - Net inventory change: 0
        - Dealer earns: ask - bid (the spread)

        This happens atomically - if either leg fails, the whole trade fails.

        Args:
            seller_id: Agent ID of the seller
            buyer_id: Agent ID of the buyer
            sell_eligible: Set for updating after trade
            buy_eligible: Set for updating after trade

        Returns:
            True if crossed trade executed, False otherwise

        References:
            - Treynor (1987): Dealer intermediates between hurried buyers and sellers
        """
        seller = self.traders[seller_id]
        buyer = self.traders[buyer_id]

        # Select ticket seller wants to sell
        ticket = self._select_ticket_to_sell(seller)
        if ticket is None:
            return False

        bucket_id = ticket.bucket_id
        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Check if buyer can afford the ask price
        if buyer.cash < dealer.ask:
            return False  # Buyer can't afford, don't cross

        # Risk-based sell decision (if risk assessor configured)
        if self.risk_assessor:
            asset_value = sum(
                (self.risk_assessor.expected_value(t, self.day)
                for t in seller.tickets_owned),
                Decimal(0),
            )
            if not self.risk_assessor.should_sell(
                ticket=ticket,
                dealer_bid=dealer.bid,
                current_day=self.day,
                trader_cash=seller.cash,
                trader_shortfall=seller.shortfall(self.day),
                trader_asset_value=asset_value,
            ):
                return False  # Seller rejects price

        # Risk-based buy decision (if risk assessor configured)
        if self.risk_assessor:
            buyer_asset_value = sum(
                (self.risk_assessor.expected_value(t, self.day)
                for t in buyer.tickets_owned),
                Decimal(0),
            )
            if not self.risk_assessor.should_buy(
                ticket=ticket,
                dealer_ask=dealer.ask,
                current_day=self.day,
                trader_cash=buyer.cash,
                trader_shortfall=buyer.shortfall(self.day),
                trader_asset_value=buyer_asset_value,
            ):
                return False  # Buyer rejects price

        # Execute the crossed trade atomically
        # Leg 1: Seller sells to dealer at bid
        sell_price = dealer.bid
        seller.tickets_owned.remove(ticket)
        seller.cash += sell_price

        # Leg 2: Dealer sells to buyer at ask
        buy_price = dealer.ask
        ticket.owner_id = buyer_id
        buyer.tickets_owned.append(ticket)
        buyer.cash -= buy_price

        # Dealer's net position: bought at bid, sold at ask
        # Cash: +buy_price - sell_price = spread (earned)
        # Inventory: +1 - 1 = 0 (unchanged)
        dealer.cash += (buy_price - sell_price)

        # Update seller's asset issuer constraint
        if len(seller.tickets_owned) == 0:
            seller.asset_issuer_id = None

        # Update buyer's asset issuer constraint
        if buyer.asset_issuer_id is None:
            buyer.asset_issuer_id = ticket.issuer_id

        # Log the crossed trade
        self.events.log_trade(
            day=self.day,
            side="CROSS_SELL",
            trader_id=seller_id,
            ticket_id=ticket.id,
            bucket=bucket_id,
            price=sell_price,
            is_passthrough=False,
        )
        self.events.log_trade(
            day=self.day,
            side="CROSS_BUY",
            trader_id=buyer_id,
            ticket_id=ticket.id,
            bucket=bucket_id,
            price=buy_price,
            is_passthrough=False,
        )

        # Update eligibility sets
        if seller.shortfall(self.day) <= 0 or len(seller.tickets_owned) == 0:
            sell_eligible.discard(seller_id)
        if buyer.cash <= self.config.buffer_B:
            buy_eligible.discard(buyer_id)

        return True

    def _process_sell(self, eligible: set[AgentId]) -> None:
        """
        Process a SELL order from a randomly selected eligible trader.

        Args:
            eligible: Set of agent IDs eligible to sell

        References:
            - Section 11.4: SELL order processing
        """
        if not eligible:
            return  # No eligible sellers

        # Sample uniformly from eligible
        agent_id = self.rng.choice(list(eligible))
        self._process_sell_for_agent(agent_id, eligible)

    def _process_sell_for_agent(self, agent_id: AgentId, eligible: set[AgentId]) -> None:
        """
        Process a SELL order for a specific trader.

        Args:
            agent_id: Agent ID of the trader who wants to sell
            eligible: Set of agent IDs eligible to sell (for updating after trade)

        References:
            - Section 11.4: SELL order processing
        """
        trader = self.traders[agent_id]

        # Select ticket to sell (shortest maturity first)
        ticket = self._select_ticket_to_sell(trader)
        if ticket is None:
            return  # No ticket available (should not happen if eligible)

        # Determine bucket and execute trade
        bucket_id = ticket.bucket_id
        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Risk-based sell decision
        if self.risk_assessor:
            # Compute total asset value for urgency calculation
            asset_value = sum(
                (self.risk_assessor.expected_value(t, self.day)
                for t in trader.tickets_owned),
                Decimal(0),
            )

            if not self.risk_assessor.should_sell(
                ticket=ticket,
                dealer_bid=dealer.bid,
                current_day=self.day,
                trader_cash=trader.cash,
                trader_shortfall=trader.shortfall(self.day),
                trader_asset_value=asset_value,
            ):
                # Log rejection and skip trade
                ev = self.risk_assessor.expected_value(ticket, self.day)
                threshold = self.risk_assessor.compute_effective_threshold(
                    cash=trader.cash,
                    shortfall=trader.shortfall(self.day),
                    asset_value=asset_value,
                )
                self.events.log_sell_rejected(
                    day=self.day,
                    trader_id=agent_id,
                    ticket_id=ticket.id,
                    bucket=bucket_id,
                    offered_price=dealer.bid,
                    expected_value=ev,
                    threshold=threshold,
                    reason="price_below_ev_plus_threshold",
                )
                return

        # Execute customer sell (trader sells to dealer)
        result = self.executor.execute_customer_sell(
            dealer=dealer,
            vbt=vbt,
            ticket=ticket,
            check_assertions=True,
        )

        # Update trader state
        trader.tickets_owned.remove(ticket)
        trader.cash += result.price

        # Update asset issuer constraint if no more tickets
        if len(trader.tickets_owned) == 0:
            trader.asset_issuer_id = None

        # Log trade
        self.events.log_trade(
            day=self.day,
            side="SELL",
            trader_id=agent_id,
            ticket_id=ticket.id,
            bucket=bucket_id,
            price=result.price,
            is_passthrough=result.is_passthrough,
        )

        # Remove from eligible set if no longer eligible
        if trader.shortfall(self.day) <= 0 or len(trader.tickets_owned) == 0:
            eligible.discard(agent_id)

    def _process_buy(self, eligible: set[AgentId]) -> None:
        """
        Process a BUY order from a randomly selected eligible trader.

        Args:
            eligible: Set of agent IDs eligible to buy

        References:
            - Section 11.4: BUY order processing
        """
        if not eligible:
            return  # No eligible buyers

        # Sample uniformly from eligible
        agent_id = self.rng.choice(list(eligible))
        self._process_buy_for_agent(agent_id, eligible)

    def _process_buy_for_agent(self, agent_id: AgentId, eligible: set[AgentId]) -> None:
        """
        Process a BUY order for a specific trader.

        Args:
            agent_id: Agent ID of the trader who wants to buy
            eligible: Set of agent IDs eligible to buy (for updating after trade)

        References:
            - Section 11.4: BUY order processing
        """
        trader = self.traders[agent_id]

        # Select bucket to buy from (Short -> Mid -> Long preference)
        bucket_id = self._select_bucket_to_buy(trader)
        if bucket_id is None:
            return  # No bucket available

        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Execute customer buy (trader buys from dealer)
        # Note: May fail if issuer preference cannot be satisfied
        try:
            result = self.executor.execute_customer_buy(
                dealer=dealer,
                vbt=vbt,
                buyer_id=agent_id,
                issuer_preference=trader.asset_issuer_id,
                check_assertions=True,
            )
        except ValueError as e:
            # Cannot satisfy single-issuer constraint - skip this trade
            # This happens when trader has asset_issuer_id set but neither
            # dealer nor VBT has tickets from that issuer
            return

        # Risk-based buy validation (post-execution since we need actual ticket)
        if result.ticket and self.risk_assessor:
            asset_value = sum(
                (self.risk_assessor.expected_value(t, self.day)
                for t in trader.tickets_owned),
                Decimal(0),
            )

            # Convert price to unit price for should_buy
            unit_price = result.price  # Already unit price from executor

            if not self.risk_assessor.should_buy(
                ticket=result.ticket,
                dealer_ask=unit_price,
                current_day=self.day,
                trader_cash=trader.cash,
                trader_shortfall=trader.shortfall(self.day),
                trader_asset_value=asset_value,
            ):
                # Reverse the transaction
                # Return ticket to source (dealer or VBT)
                if result.is_passthrough:
                    vbt.inventory.append(result.ticket)
                    vbt.cash -= result.price
                else:
                    dealer.inventory.append(result.ticket)
                    dealer.cash -= result.price

                # Recompute dealer state
                recompute_dealer_state(dealer, vbt, self.params)

                # Log rejection
                ev = self.risk_assessor.expected_value(result.ticket, self.day)
                threshold = self.risk_assessor.params.base_risk_premium * self.risk_assessor.params.buy_premium_multiplier
                self.events.log_buy_rejected(
                    day=self.day,
                    trader_id=agent_id,
                    ticket_id=result.ticket.id,
                    bucket=bucket_id,
                    offered_price=unit_price,
                    expected_value=ev,
                    threshold=threshold,
                    reason="ev_below_price_plus_threshold",
                )
                return

        # Update trader state
        if result.ticket:
            trader.tickets_owned.append(result.ticket)
            trader.cash -= result.price

            # Set asset issuer constraint
            if trader.asset_issuer_id is None:
                trader.asset_issuer_id = result.ticket.issuer_id

        # Log trade
        if result.ticket:
            self.events.log_trade(
                day=self.day,
                side="BUY",
                trader_id=agent_id,
                ticket_id=result.ticket.id,
                bucket=bucket_id,
                price=result.price,
                is_passthrough=result.is_passthrough,
            )

        # Remove from eligible set if no longer eligible
        if trader.cash <= self.config.buffer_B:
            eligible.discard(agent_id)

    def _select_ticket_to_sell(self, trader: TraderState) -> Ticket | None:
        """
        Select ticket for trader to sell (shortest maturity first).

        Priority:
        1. Lowest remaining_tau (sell soonest-to-mature first)
        2. Lowest serial (deterministic tie-breaker)

        Args:
            trader: Trader state

        Returns:
            Selected ticket, or None if no tickets available

        References:
            - Section 10.3: Sell policy (shortest maturity first)
        """
        if not trader.tickets_owned:
            return None

        # Sort by (remaining_tau, serial) and return first
        return min(trader.tickets_owned, key=lambda t: (t.remaining_tau, t.serial))

    def _select_bucket_to_buy(self, trader: TraderState) -> str | None:
        """
        Select bucket to buy from (Short -> Mid -> Long preference).

        This implements a maturity preference: traders prefer shorter-maturity
        tickets for faster turnover and liquidity.

        Args:
            trader: Trader state

        Returns:
            Bucket name, or None if no buckets available

        References:
            - Section 10.4: Buy policy (maturity preference)
        """
        # Try buckets in order of preference
        for bucket in self.config.buckets:
            bucket_id = bucket.name

            # Check if dealer or VBT has inventory in this bucket
            dealer = self.dealers[bucket_id]
            vbt = self.vbts[bucket_id]

            if len(dealer.inventory) > 0 or len(vbt.inventory) > 0:
                return bucket_id

        return None

    # =====================================================================
    # Phase 5: Settlement with proportional recovery
    # =====================================================================

    def _settle_maturing_debt(self) -> None:
        """
        Settlement with proportional recovery (Section 11, Step 5).

        For each issuer with maturing tickets:
        1. Compute total obligations due
        2. Compute recovery rate R = min(1, cash / obligations)
        3. Pay proportional recovery to all holders
        4. Mark issuer as defaulted if R < 1

        References:
            - Section 11.5: Settlement phase
            - Section 6.6-6.8: Settlement events
        """
        # Group maturing tickets by issuer
        maturing_by_issuer: dict[AgentId, list[Ticket]] = {}

        for ticket in self.all_tickets.values():
            if ticket.maturity_day == self.day:
                issuer_id = ticket.issuer_id
                if issuer_id not in maturing_by_issuer:
                    maturing_by_issuer[issuer_id] = []
                maturing_by_issuer[issuer_id].append(ticket)

        # Settle each issuer
        if maturing_by_issuer:
            logger.debug("dealer day %d: settling %d issuers with maturing debt",
                         self.day, len(maturing_by_issuer))
        for issuer_id, tickets in maturing_by_issuer.items():
            self._settle_issuer(issuer_id, tickets)

    def _settle_issuer(self, issuer_id: AgentId, tickets: list[Ticket]) -> None:
        """
        Settle all maturing tickets for one issuer.

        Implements proportional recovery:
        - R = min(1, cash / total_due)
        - Payment to each holder = R * face

        Args:
            issuer_id: Agent ID of issuer
            tickets: List of maturing tickets issued by this agent

        References:
            - Section 6.6: Full settlement (R = 1)
            - Section 6.7-6.8: Partial settlement (R < 1)
        """
        if issuer_id not in self.traders:
            raise ValueError(f"Issuer {issuer_id} is not a registered trader")

        issuer = self.traders[issuer_id]

        # Compute total obligations
        total_due: Decimal = sum((ticket.face for ticket in tickets), Decimal(0))

        # Compute recovery rate
        if total_due == 0:
            return  # No obligations

        recovery_rate = min(Decimal(1), issuer.cash / total_due)

        # Compute total payment - use exact value to avoid precision issues
        # When R < 1, total_paid equals issuer.cash exactly
        if recovery_rate < Decimal(1):
            total_paid = issuer.cash
        else:
            total_paid = total_due

        # Deduct from issuer cash
        issuer.cash -= total_paid

        # Distribute payments to holders using remainder-based allocation
        # to ensure total distributed equals total_paid exactly
        remaining = total_paid
        for i, ticket in enumerate(tickets):
            if i < len(tickets) - 1:
                # For all but last ticket: compute proportional payment and quantize
                payment = (recovery_rate * ticket.face).quantize(
                    CASH_PRECISION, rounding=ROUND_HALF_UP
                )
                remaining -= payment
            else:
                # Last ticket gets remainder to ensure exact balance
                payment = remaining
            self._pay_holder(ticket.owner_id, payment, ticket)

        # Log settlement or default
        if recovery_rate >= Decimal(1):
            # Full settlement
            self.events.log_settlement(
                day=self.day,
                issuer_id=issuer_id,
                total_paid=total_paid,
                n_tickets=len(tickets),
            )
        else:
            # Default with partial recovery
            issuer.defaulted = True

            # Group tickets by bucket for loss rate computation
            tickets_by_bucket: dict[str, list[Ticket]] = {}
            for ticket in tickets:
                bucket_id = ticket.bucket_id
                if bucket_id not in tickets_by_bucket:
                    tickets_by_bucket[bucket_id] = []
                tickets_by_bucket[bucket_id].append(ticket)

            # Log default per bucket
            for bucket_id, bucket_tickets in tickets_by_bucket.items():
                bucket_due: Decimal = sum((t.face for t in bucket_tickets), Decimal(0))
                bucket_paid = recovery_rate * bucket_due

                self.events.log_default(
                    day=self.day,
                    issuer_id=issuer_id,
                    recovery_rate=recovery_rate,
                    total_due=bucket_due,
                    total_paid=bucket_paid,
                    n_tickets=len(bucket_tickets),
                    bucket=bucket_id,
                )

        # Update risk assessor history
        if self.risk_assessor:
            defaulted = recovery_rate < Decimal(1)
            self.risk_assessor.update_history(
                day=self.day,
                issuer_id=issuer_id,
                defaulted=defaulted
            )

        # Remove tickets from issuer's obligations
        for ticket in tickets:
            if ticket in issuer.obligations:
                issuer.obligations.remove(ticket)

    def _pay_holder(
        self,
        holder_id: AgentId,
        amount: Decimal,
        ticket: Ticket,
    ) -> None:
        """
        Credit payment to holder (trader, dealer, or VBT).

        Args:
            holder_id: Agent ID of ticket holder
            amount: Payment amount (may be partial if R < 1)
            ticket: Ticket being settled
        """
        # Check if holder is a trader
        if holder_id in self.traders:
            self.traders[holder_id].cash += amount
            # Quantize to avoid floating-point accumulation (e.g., 1/3 + 1/3 + 1/3 != 1)
            self.traders[holder_id].cash = self.traders[holder_id].cash.quantize(
                CASH_PRECISION, rounding=ROUND_HALF_UP
            )
            # Remove ticket from trader's owned tickets
            if ticket in self.traders[holder_id].tickets_owned:
                self.traders[holder_id].tickets_owned.remove(ticket)
            return

        # Check if holder is a dealer
        for dealer in self.dealers.values():
            if dealer.agent_id == holder_id:
                dealer.cash += amount
                # Quantize to avoid floating-point accumulation
                dealer.cash = dealer.cash.quantize(CASH_PRECISION, rounding=ROUND_HALF_UP)
                # Remove ticket from dealer inventory
                if ticket in dealer.inventory:
                    dealer.inventory.remove(ticket)
                return

        # Check if holder is a VBT
        for vbt in self.vbts.values():
            if vbt.agent_id == holder_id:
                vbt.cash += amount
                # Quantize to avoid floating-point accumulation
                vbt.cash = vbt.cash.quantize(CASH_PRECISION, rounding=ROUND_HALF_UP)
                # Remove ticket from VBT inventory
                if ticket in vbt.inventory:
                    vbt.inventory.remove(ticket)
                return

        raise ValueError(f"Holder {holder_id} not found in traders, dealers, or VBTs")

    # =====================================================================
    # Phase 6: VBT anchor updates
    # =====================================================================

    def _update_vbt_anchors(self) -> None:
        """
        Update VBT anchors based on bucket loss rates (Section 10).

        For each bucket:
        1. Compute loss rate from today's defaults
        2. Apply update rule: M -= phi_M * loss, O += phi_O * loss
        3. Recompute VBT quotes and dealer states

        References:
            - Section 9: Loss-based anchor updates
            - Section 11.6: Anchor update phase
        """
        for bucket_id in self.vbts:
            vbt = self.vbts[bucket_id]

            # Snapshot for C6 assertion
            M_old = vbt.M
            O_old = vbt.O

            # Compute loss rate from today's defaults
            loss_rate = self.events.get_bucket_loss_rate(self.day, bucket_id)

            if loss_rate > 0:
                # Apply update rule
                vbt.update_from_loss(loss_rate)

                # Log anchor update
                self.events.log_vbt_anchor_update(
                    day=self.day,
                    bucket=bucket_id,
                    M_old=M_old,
                    M_new=vbt.M,
                    O_old=O_old,
                    O_new=vbt.O,
                    loss_rate=loss_rate,
                )

                # Recompute dealer state with new anchors
                dealer = self.dealers[bucket_id]
                recompute_dealer_state(dealer, vbt, self.params)

    def to_html(
        self,
        path: "Path | str",
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        """
        Export simulation to HTML report.

        Args:
            path: Output file path
            title: Report title (default: "Dealer Ring Simulation")
            subtitle: Report subtitle (optional)
        """
        from .report import export_dealer_ring_html
        export_dealer_ring_html(
            snapshots=self.snapshots,
            config=self.config,
            path=path,
            title=title,
            subtitle=subtitle,
        )
