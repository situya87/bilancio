"""
Integrated bank-dealer simulation orchestrator.

This module combines the dealer ring simulation with banking components,
replacing cash-based trader state with deposit-based bank interactions.

Key differences from DealerRingSimulation:
- Traders hold deposits at banks (no cash)
- Trading decisions are rate-sensitive (compare yields)
- Interbank settlement at end of day
- Banks set rates based on inventory and CB corridor

Day Loop Phases:
1. Interest accrual on deposits
2. Loan maturity processing
3. Update maturities and buckets (same as dealer ring)
4. Dealer and bank pre-computation
5. Compute trading eligibility (bank-aware)
6. Rate-sensitive order flow
7. Settlement with default waterfall
8. Interbank settlement
9. VBT anchor update

References:
- docs/plans/banks_dealers_integration.md
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import random
from copy import deepcopy
from typing import Any

from bilancio.core.ids import AgentId, new_id

from .models import (
    Ticket, DealerState, VBTState, TraderState,
    BucketConfig, DEFAULT_BUCKETS, TicketId,
)
from .kernel import KernelParams, recompute_dealer_state
from .trading import TradeExecutor
from .events import EventLog
from .risk_assessment import RiskAssessor, RiskAssessmentParams
from .bank_integration import (
    BankAwareTraderState,
    BankDealerRingConfig,
    IntegratedBankState,
    InterbankLedger,
    TraderLoan,
    DepositCohort,
    accrue_deposit_interest,
    settle_interbank_positions,
    resolve_default,
    compute_borrow_vs_sell_decision,
    compute_yield_sell_decision,
    compute_yield_buy_decision,
)
from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_quotes,
    compute_inventory,
)


# Cash precision for settlement calculations
CASH_PRECISION = Decimal("0.000001")


@dataclass
class BankDealerDaySnapshot:
    """
    Snapshot of bank-dealer simulation state at end of day.

    Extended from DaySnapshot to include bank states.
    """
    day: int
    dealers: dict[str, dict[str, Any]]  # bucket_id -> dealer state dict
    vbts: dict[str, dict[str, Any]]     # bucket_id -> VBT state dict
    traders: dict[str, dict[str, Any]]  # agent_id -> bank-aware trader state dict
    banks: dict[str, dict[str, Any]]    # bank_id -> bank state dict
    tickets: dict[str, dict[str, Any]]  # ticket_id -> ticket dict
    events: list[dict[str, Any]]        # Events for this day only


class BankDealerSimulation:
    """
    Integrated bank-dealer simulation orchestrator.

    This extends DealerRingSimulation by:
    - Using BankAwareTraderState (deposits instead of cash)
    - Integrating bank rate computation
    - Supporting borrowing vs selling decisions
    - Implementing interbank settlement

    Attributes:
        config: Combined bank+dealer configuration
        rng: Random number generator
        day: Current simulation day
        dealers: Per-bucket dealer states
        vbts: Per-bucket VBT states
        traders: Bank-aware trader states
        banks: Bank states (includes trader banks + dealer bank)
        interbank_ledger: Tracks interbank positions for settlement
        all_tickets: Global ticket registry
        events: Event log
        params: Kernel parameters
        executor: Trade executor
        risk_assessor: Risk assessment for trading decisions
    """

    def __init__(
        self,
        config: BankDealerRingConfig,
        risk_assessor: RiskAssessor | None = None,
    ):
        """
        Initialize the integrated simulation.

        Args:
            config: Combined bank+dealer configuration
            risk_assessor: Optional risk assessor for credit risk
        """
        self.config = config
        self.rng = random.Random(config.seed)

        # State
        self.day: int = 0
        self.dealers: dict[str, DealerState] = {}
        self.vbts: dict[str, VBTState] = {}
        self.traders: dict[AgentId, BankAwareTraderState] = {}
        self.banks: dict[str, IntegratedBankState] = {}
        self.interbank_ledger = InterbankLedger()
        self.all_tickets: dict[TicketId, Ticket] = {}

        # Event log
        self.events = EventLog()

        # Snapshots
        self.snapshots: list[BankDealerDaySnapshot] = []

        # Kernel params
        self.params = KernelParams(S=config.ticket_size)
        self.executor = TradeExecutor(self.params, self.rng)
        self.risk_assessor = risk_assessor or RiskAssessor(RiskAssessmentParams())

        # Bucket config (use defaults if not specified)
        self.buckets = list(DEFAULT_BUCKETS)

        # VBT anchors (defaults)
        self.vbt_anchors = {
            "short": (Decimal(1), Decimal("0.20")),
            "mid": (Decimal(1), Decimal("0.30")),
            "long": (Decimal(1), Decimal("0.40")),
        }

        # VBT sensitivity
        self.phi_M = Decimal(1)
        self.phi_O = Decimal("0.6")
        self.M_min = Decimal("0.02")

        # Trading parameters
        self.horizon_H = 3
        self.buffer_B = Decimal(1)
        self.N_max = 3

        # Initialize market makers
        self._init_market_makers()

        # Initialize banks
        self._init_banks()

    def _init_market_makers(self) -> None:
        """Initialize dealers and VBTs for each bucket."""
        for bucket in self.buckets:
            bucket_id = bucket.name

            # Create dealer
            dealer = DealerState(
                bucket_id=bucket_id,
                agent_id=new_id(f"dealer_{bucket_id}"),
            )
            self.dealers[bucket_id] = dealer

            # Create VBT
            M, O = self.vbt_anchors.get(bucket_id, (Decimal(1), Decimal("0.30")))
            vbt = VBTState(
                bucket_id=bucket_id,
                agent_id=new_id(f"vbt_{bucket_id}"),
                M=M,
                O=O,
                phi_M=self.phi_M,
                phi_O=self.phi_O,
                O_min=Decimal(0),
                clip_nonneg_B=True,
            )
            vbt.recompute_quotes()
            self.vbts[bucket_id] = vbt

    def _init_banks(self) -> None:
        """Initialize trader banks and dealer bank."""
        # Trader banks
        for i in range(1, self.config.n_trader_banks + 1):
            bank_id = f"bank_{i}"
            self.banks[bank_id] = IntegratedBankState(
                bank_id=bank_id,
                current_day=0,
                reserves=Decimal(0),
                current_deposit_rate=self.config.corridor_mid - Decimal("0.005"),
                current_loan_rate=self.config.corridor_mid + Decimal("0.005"),
            )

        # Dealer bank
        dealer_bank_id = self.config.dealer_bank_id
        self.banks[dealer_bank_id] = IntegratedBankState(
            bank_id=dealer_bank_id,
            current_day=0,
            reserves=Decimal(0),
            current_deposit_rate=self.config.corridor_mid - Decimal("0.005"),
            current_loan_rate=self.config.corridor_mid + Decimal("0.005"),
        )

    def _get_bank_pricing_params(self, bank: IntegratedBankState) -> PricingParams:
        """
        Create pricing parameters for a bank.

        Args:
            bank: The bank to create params for

        Returns:
            PricingParams for the banking kernel
        """
        # Compute total deposits from traders at this bank
        total_deposits = sum(
            t.deposit_balance for t in self.traders.values()
            if t.bank_id == bank.bank_id
        )

        # Compute targets from config ratios
        reserve_target = int(
            total_deposits * self.config.reserve_target_ratio
        ) if total_deposits > 0 else self.config.bank_ticket_size

        symmetric_capacity = int(
            reserve_target * self.config.symmetric_capacity_ratio
        ) if reserve_target > 0 else self.config.bank_ticket_size

        return PricingParams(
            reserve_remuneration_rate=self.config.cb_deposit_rate,
            cb_borrowing_rate=self.config.cb_lending_rate,
            reserve_target=max(1, reserve_target),
            symmetric_capacity=max(1, symmetric_capacity),
            ticket_size=self.config.bank_ticket_size,
            reserve_floor=0,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
        )

    def _update_bank_rates(self) -> None:
        """Update all bank rates based on current inventory."""
        for bank_id, bank in self.banks.items():
            # Get pricing params
            params = self._get_bank_pricing_params(bank)

            # Compute inventory position (simplified: reserves - target)
            inventory = compute_inventory(
                int(bank.reserves),
                params.reserve_target,
            )

            # Cash tightness and risk (simplified for now)
            cash_tightness = Decimal(0)
            risk_index = Decimal(0)

            # Compute quotes
            quote = compute_quotes(
                inventory=inventory,
                cash_tightness=cash_tightness,
                risk_index=risk_index,
                params=params,
                day=self.day,
            )

            # Update bank rates
            bank.current_deposit_rate = quote.deposit_rate
            bank.current_loan_rate = quote.loan_rate

    def setup_ring(
        self,
        n_traders: int,
        initial_deposits: Decimal,
        tickets: list[Ticket],
        max_maturity_days: int | None = None,
    ) -> None:
        """
        Set up the ring with traders, banks, and initial tickets.

        Creates bank-aware traders distributed across trader banks.
        Initial deposits are placed in each trader's assigned bank.

        Args:
            n_traders: Number of ring traders (H1, H2, ..., Hn)
            initial_deposits: Initial deposit amount per trader
            tickets: List of tickets to allocate
            max_maturity_days: Max maturity for loan tenor (auto-detect if None)
        """
        # Auto-detect max maturity if not specified
        if max_maturity_days is None:
            max_maturity_days = max(t.maturity_day for t in tickets) if tickets else 10

        # Set loan tenor
        self.loan_tenor = max_maturity_days

        # Create traders with bank assignments
        for i in range(1, n_traders + 1):
            agent_id = f"H{i}"
            bank_id = self.config.get_bank_id_for_trader(i)

            trader = BankAwareTraderState(
                agent_id=agent_id,
                bank_id=bank_id,
            )

            # Add initial deposits with current bank rate
            bank = self.banks[bank_id]
            trader.add_deposit(
                amount=initial_deposits,
                rate=bank.current_deposit_rate,
                day=0,
            )

            # Register with bank
            bank.client_ids.add(agent_id)

            # Add to registry
            self.traders[agent_id] = trader

        # Register tickets and assign buckets
        for ticket in tickets:
            self.all_tickets[ticket.id] = ticket
            computed_bucket = self._compute_bucket(ticket.remaining_tau)

            if computed_bucket is None:
                raise ValueError(
                    f"Ticket {ticket.id} has remaining_tau={ticket.remaining_tau} "
                    f"which does not fit any configured bucket."
                )

            ticket.bucket_id = computed_bucket

        # Allocate tickets (dealers, VBTs, traders)
        self._allocate_tickets(tickets)

        # Recompute dealer states
        for bucket_id in self.dealers:
            recompute_dealer_state(
                self.dealers[bucket_id],
                self.vbts[bucket_id],
                self.params,
            )

        # Capture initial snapshot
        self._capture_snapshot()

    def _allocate_tickets(
        self,
        tickets: list[Ticket],
        dealer_share: Decimal = Decimal("0.25"),
        vbt_share: Decimal = Decimal("0.50"),
    ) -> None:
        """
        Allocate tickets to dealers, VBTs, and traders.

        Args:
            tickets: Tickets to allocate
            dealer_share: Fraction to dealers
            vbt_share: Fraction to VBTs
        """
        # Group by bucket
        tickets_by_bucket: dict[str, list[Ticket]] = {}
        for ticket in tickets:
            bucket_id = ticket.bucket_id
            if bucket_id not in tickets_by_bucket:
                tickets_by_bucket[bucket_id] = []
            tickets_by_bucket[bucket_id].append(ticket)

        # Allocate per bucket
        for bucket_id, bucket_tickets in tickets_by_bucket.items():
            n_total = len(bucket_tickets)
            n_dealer = int(n_total * dealer_share)
            n_vbt = int(n_total * vbt_share)

            # To dealers
            dealer = self.dealers[bucket_id]
            for i in range(n_dealer):
                ticket = bucket_tickets[i]
                ticket.owner_id = dealer.agent_id
                dealer.inventory.append(ticket)

            # To VBTs
            vbt = self.vbts[bucket_id]
            for i in range(n_dealer, n_dealer + n_vbt):
                ticket = bucket_tickets[i]
                ticket.owner_id = vbt.agent_id
                vbt.inventory.append(ticket)

            # To traders (owner = issuer)
            for i in range(n_dealer + n_vbt, n_total):
                ticket = bucket_tickets[i]
                issuer_id = ticket.issuer_id

                if issuer_id not in self.traders:
                    continue

                trader = self.traders[issuer_id]
                ticket.owner_id = issuer_id
                trader.tickets_owned.append(ticket)

                # Set single-issuer constraint
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = issuer_id

    def _compute_bucket(self, remaining_tau: int) -> str | None:
        """Determine bucket for given remaining maturity."""
        if remaining_tau <= 0:
            return None

        for bucket in self.buckets:
            if bucket.tau_max is None:
                if remaining_tau >= bucket.tau_min:
                    return bucket.name
            else:
                if bucket.tau_min <= remaining_tau <= bucket.tau_max:
                    return bucket.name

        return None

    def run_day(self) -> None:
        """
        Execute one simulation day.

        Extended phases:
        1. Interest accrual on deposits
        2. Loan maturity processing
        3. Update maturities and buckets
        4. Dealer and bank pre-computation
        5. Compute trading eligibility
        6. Rate-sensitive order flow
        7. Settlement with default waterfall
        8. Interbank settlement
        9. VBT anchor update
        """
        self.day += 1
        self.events.log_day_start(self.day)

        # Update bank day
        for bank in self.banks.values():
            bank.current_day = self.day

        # Phase 1: Interest accrual
        self._accrue_interest()

        # Phase 2: Loan maturity processing
        self._process_loan_maturities()

        # Phase 3: Update maturities and buckets
        self._update_maturities()
        self._rebucket_tickets()

        # Phase 4: Dealer and bank pre-computation
        self._update_bank_rates()
        for bucket_id in self.dealers:
            recompute_dealer_state(
                self.dealers[bucket_id],
                self.vbts[bucket_id],
                self.params,
            )

        # Phase 5: Compute eligibility sets
        sell_eligible = self._compute_sell_eligible()
        buy_eligible = self._compute_buy_eligible()

        # Phase 6: Order flow with rate sensitivity
        self._process_order_flow(sell_eligible, buy_eligible)

        # Phase 7: Settlement
        self._settle_maturing_debt()

        # Phase 8: Interbank settlement
        self._settle_interbank()

        # Phase 9: VBT anchor update
        self._update_vbt_anchors()

        # Capture snapshot
        self._capture_snapshot()

    def run(self, max_days: int | None = None) -> None:
        """Run simulation for specified days."""
        days = max_days or self.config.max_days
        for _ in range(days):
            self.run_day()

    # =========================================================================
    # Phase 1: Interest Accrual
    # =========================================================================

    def _accrue_interest(self) -> None:
        """Accrue interest on all trader deposits."""
        for trader in self.traders.values():
            interest = accrue_deposit_interest(trader, self.day)
            if interest > 0:
                self.events.log(
                    "interest_accrual",
                    self.day,
                    trader_id=trader.agent_id,
                    amount=float(interest),
                )

    # =========================================================================
    # Phase 2: Loan Maturity Processing
    # =========================================================================

    def _process_loan_maturities(self) -> None:
        """Process loan repayments due today."""
        for trader in self.traders.values():
            if trader.defaulted:
                continue

            # Find loans due today
            loans_due = [
                loan for loan in trader.loans
                if loan.maturity_day == self.day
            ]

            for loan in loans_due:
                repayment = loan.repayment_amount

                # Check if trader can repay
                if trader.deposit_balance >= repayment:
                    # Withdraw from deposits
                    trader.withdraw_deposits(repayment)

                    # Add to bank reserves
                    bank = self.banks[loan.bank_id]
                    bank.reserves += repayment

                    # Remove loan from both
                    trader.repay_loan(loan)
                    bank.receive_loan_repayment(loan.loan_id)

                    self.events.log(
                        "loan_repayment",
                        self.day,
                        trader_id=trader.agent_id,
                        bank_id=loan.bank_id,
                        amount=float(repayment),
                    )
                else:
                    # Loan default - trigger full default waterfall
                    self._handle_trader_default(trader)
                    break  # Exit loan loop for this trader

    # =========================================================================
    # Phase 3: Update Maturities and Rebucket
    # =========================================================================

    def _update_maturities(self) -> None:
        """Decrement remaining_tau for all tickets."""
        for ticket in self.all_tickets.values():
            if ticket.remaining_tau > 0:
                ticket.remaining_tau -= 1

    def _rebucket_tickets(self) -> None:
        """Reassign bucket_id based on remaining_tau."""
        for ticket in self.all_tickets.values():
            old_bucket = ticket.bucket_id
            new_bucket = self._compute_bucket(ticket.remaining_tau)

            if new_bucket is None or new_bucket == old_bucket:
                continue

            # Check if dealer holds
            old_dealer = self.dealers.get(old_bucket)
            if old_dealer and ticket.owner_id == old_dealer.agent_id:
                self._rebucket_dealer_ticket(ticket, old_bucket, new_bucket)
                continue

            # Check if VBT holds
            old_vbt = self.vbts.get(old_bucket)
            if old_vbt and ticket.owner_id == old_vbt.agent_id:
                self._rebucket_vbt_ticket(ticket, old_bucket, new_bucket)
                continue

            # Trader-held: just update bucket
            ticket.bucket_id = new_bucket

    def _rebucket_dealer_ticket(
        self,
        ticket: Ticket,
        old_bucket: str,
        new_bucket: str,
    ) -> None:
        """Internal sale from old dealer to new dealer."""
        old_dealer = self.dealers[old_bucket]
        new_dealer = self.dealers[new_bucket]
        price = self.vbts[new_bucket].M

        old_dealer.inventory.remove(ticket)
        old_dealer.cash += price
        new_dealer.inventory.append(ticket)
        new_dealer.cash -= price

        ticket.owner_id = new_dealer.agent_id
        ticket.bucket_id = new_bucket

        recompute_dealer_state(old_dealer, self.vbts[old_bucket], self.params)
        recompute_dealer_state(new_dealer, self.vbts[new_bucket], self.params)

    def _rebucket_vbt_ticket(
        self,
        ticket: Ticket,
        old_bucket: str,
        new_bucket: str,
    ) -> None:
        """Internal sale from old VBT to new VBT."""
        old_vbt = self.vbts[old_bucket]
        new_vbt = self.vbts[new_bucket]
        price = old_vbt.M

        old_vbt.inventory.remove(ticket)
        old_vbt.cash += price
        new_vbt.inventory.append(ticket)
        new_vbt.cash -= price

        ticket.owner_id = new_vbt.agent_id
        ticket.bucket_id = new_bucket

    # =========================================================================
    # Phase 5: Trading Eligibility
    # =========================================================================

    def _compute_sell_eligible(self) -> set[AgentId]:
        """
        Traders eligible to sell (shortfall > 0 and has tickets).

        With banks: includes rate-sensitive selling (deposit > hold).
        """
        eligible = set()
        for trader in self.traders.values():
            if trader.defaulted:
                continue

            has_tickets = len(trader.tickets_owned) > 0
            if not has_tickets:
                continue

            # Distress selling: shortfall > 0
            has_shortfall = trader.shortfall(self.day) > 0

            # Yield selling: deposit rate > payable yield
            # Check if any ticket yields less than deposit rate
            bank = self.banks[trader.bank_id]
            has_yield_motive = False

            for ticket in trader.tickets_owned:
                ev = self.risk_assessor.expected_value(ticket, self.day)
                should_sell = compute_yield_sell_decision(
                    payable_expected_value=ev,
                    dealer_bid=self.vbts[ticket.bucket_id].B,  # Use VBT bid as proxy
                    deposit_rate=bank.current_deposit_rate,
                    days_to_maturity=ticket.remaining_tau,
                )
                if should_sell:
                    has_yield_motive = True
                    break

            if has_shortfall or has_yield_motive:
                eligible.add(trader.agent_id)

        return eligible

    def _compute_buy_eligible(self) -> set[AgentId]:
        """
        Traders eligible to buy (horizon >= H, deposits > buffer).

        With banks: includes rate-sensitive buying (payable > deposit).
        """
        eligible = set()
        for trader in self.traders.values():
            if trader.defaulted:
                continue

            # Check deposit buffer
            has_buffer = trader.deposit_balance > self.buffer_B

            # Check horizon
            next_liability_day = trader.earliest_liability_day(self.day)
            if next_liability_day is None:
                has_horizon = True
            else:
                horizon = next_liability_day - self.day
                has_horizon = horizon >= self.horizon_H

            if has_buffer and has_horizon:
                eligible.add(trader.agent_id)

        return eligible

    # =========================================================================
    # Phase 6: Order Flow
    # =========================================================================

    def _process_order_flow(
        self,
        sell_eligible: set[AgentId],
        buy_eligible: set[AgentId],
    ) -> None:
        """Process order flow with order crossing."""
        sellers = sell_eligible.copy()
        buyers = (buy_eligible - sell_eligible).copy()

        # Phase 1: Order crossing
        n_crossable = min(len(sellers), len(buyers))

        if n_crossable > 0:
            sellers_list = list(sellers)
            buyers_list = list(buyers)
            self.rng.shuffle(sellers_list)
            self.rng.shuffle(buyers_list)

            for i in range(n_crossable):
                seller_id = sellers_list[i]
                buyer_id = buyers_list[i]

                crossed = self._process_crossed_trade(
                    seller_id, buyer_id, sell_eligible, buy_eligible
                )

                if crossed:
                    sellers.discard(seller_id)
                    buyers.discard(buyer_id)

        # Phase 2: Residual flow
        if self.N_max <= 0:
            return

        remaining_pool = list(sellers) + list(buyers)
        if not remaining_pool:
            return

        n_arrivals = min(self.rng.randint(1, self.N_max), len(remaining_pool))
        self.rng.shuffle(remaining_pool)

        for i in range(n_arrivals):
            agent_id = remaining_pool[i]
            if agent_id in sellers:
                self._process_sell(agent_id, sell_eligible)
            else:
                self._process_buy(agent_id, buy_eligible)

    def _process_crossed_trade(
        self,
        seller_id: AgentId,
        buyer_id: AgentId,
        sell_eligible: set[AgentId],
        buy_eligible: set[AgentId],
    ) -> bool:
        """Execute a crossed trade: seller → dealer → buyer."""
        seller = self.traders[seller_id]
        buyer = self.traders[buyer_id]

        # Select ticket
        ticket = self._select_ticket_to_sell(seller)
        if ticket is None:
            return False

        bucket_id = ticket.bucket_id
        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Check affordability
        if buyer.deposit_balance < dealer.ask:
            return False

        # Risk-based decisions
        asset_value = sum(
            (self.risk_assessor.expected_value(t, self.day)
            for t in seller.tickets_owned),
            Decimal(0),
        )
        if not self.risk_assessor.should_sell(
            ticket=ticket,
            dealer_bid=dealer.bid,
            current_day=self.day,
            trader_cash=seller.deposit_balance,
            trader_shortfall=seller.shortfall(self.day),
            trader_asset_value=asset_value,
        ):
            return False

        buyer_asset_value = sum(
            (self.risk_assessor.expected_value(t, self.day)
            for t in buyer.tickets_owned),
            Decimal(0),
        )
        if not self.risk_assessor.should_buy(
            ticket=ticket,
            dealer_ask=dealer.ask,
            current_day=self.day,
            trader_cash=buyer.deposit_balance,
            trader_shortfall=buyer.shortfall(self.day),
            trader_asset_value=buyer_asset_value,
        ):
            return False

        # Execute trade
        sell_price = dealer.bid
        buy_price = dealer.ask

        # Seller gets deposits
        seller.tickets_owned.remove(ticket)
        seller_bank = self.banks[seller.bank_id]
        seller.add_deposit(sell_price, seller_bank.current_deposit_rate, self.day)

        # Buyer pays from deposits
        buyer.withdraw_deposits(buy_price)
        ticket.owner_id = buyer_id
        buyer.tickets_owned.append(ticket)

        # Dealer spread
        dealer.cash += (buy_price - sell_price)

        # Interbank tracking (if different banks)
        if seller.bank_id != buyer.bank_id:
            self.interbank_ledger.record_payment(
                buyer.bank_id,
                seller.bank_id,
                sell_price,
            )

        # Update constraints
        if len(seller.tickets_owned) == 0:
            seller.asset_issuer_id = None
        # Log trades
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

        # Update eligibility
        if seller.shortfall(self.day) <= 0 or len(seller.tickets_owned) == 0:
            sell_eligible.discard(seller_id)
        if buyer.deposit_balance <= self.buffer_B:
            buy_eligible.discard(buyer_id)

        return True

    def _process_sell(
        self,
        agent_id: AgentId,
        eligible: set[AgentId],
    ) -> None:
        """Process a SELL order for a trader."""
        trader = self.traders[agent_id]
        if trader.defaulted:
            return

        # Check if borrowing is better than selling
        shortfall = trader.shortfall(self.day)
        if shortfall > 0 and len(trader.tickets_owned) > 0:
            ticket = self._select_ticket_to_sell(trader)
            if ticket:
                ev = self.risk_assessor.expected_value(ticket, self.day)
                dealer = self.dealers[ticket.bucket_id]
                bank = self.banks[trader.bank_id]

                decision = compute_borrow_vs_sell_decision(
                    shortfall=shortfall,
                    payable_expected_value=ev,
                    dealer_bid=dealer.bid,
                    loan_rate=bank.current_loan_rate,
                    loan_tenor=self.loan_tenor,
                )

                if decision == "BORROW":
                    # Try to borrow instead
                    borrowed = self._try_borrow(trader, shortfall)
                    if borrowed:
                        eligible.discard(agent_id)
                        return

        # Fall through to sell
        ticket = self._select_ticket_to_sell(trader)
        if ticket is None:
            return

        bucket_id = ticket.bucket_id
        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Risk-based sell decision
        asset_value = sum(
            (self.risk_assessor.expected_value(t, self.day)
            for t in trader.tickets_owned),
            Decimal(0),
        )
        if not self.risk_assessor.should_sell(
            ticket=ticket,
            dealer_bid=dealer.bid,
            current_day=self.day,
            trader_cash=trader.deposit_balance,
            trader_shortfall=trader.shortfall(self.day),
            trader_asset_value=asset_value,
        ):
            return

        # Execute sell via executor
        result = self.executor.execute_customer_sell(
            dealer=dealer,
            vbt=vbt,
            ticket=ticket,
            check_assertions=True,
        )

        # Update trader state - add deposits instead of cash
        trader.tickets_owned.remove(ticket)
        bank = self.banks[trader.bank_id]
        trader.add_deposit(result.price, bank.current_deposit_rate, self.day)

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

        # Update eligibility
        if trader.shortfall(self.day) <= 0 or len(trader.tickets_owned) == 0:
            eligible.discard(agent_id)

    def _process_buy(
        self,
        agent_id: AgentId,
        eligible: set[AgentId],
    ) -> None:
        """Process a BUY order for a trader."""
        trader = self.traders[agent_id]
        if trader.defaulted:
            return

        # Select bucket
        bucket_id = self._select_bucket_to_buy(trader)
        if bucket_id is None:
            return

        dealer = self.dealers[bucket_id]
        vbt = self.vbts[bucket_id]

        # Check if trader can afford
        if trader.deposit_balance < dealer.ask:
            return

        # Execute buy
        result = self.executor.execute_customer_buy(
            dealer=dealer,
            vbt=vbt,
            buyer_id=agent_id,
            check_assertions=True,
        )

        if result.ticket is None:
            return

        # Risk-based buy check
        buyer_asset_value = sum(
            (self.risk_assessor.expected_value(t, self.day)
            for t in trader.tickets_owned),
            Decimal(0),
        )
        if not self.risk_assessor.should_buy(
            ticket=result.ticket,
            dealer_ask=result.price,
            current_day=self.day,
            trader_cash=trader.deposit_balance,
            trader_shortfall=trader.shortfall(self.day),
            trader_asset_value=buyer_asset_value,
        ):
            # Reverse transaction
            if result.is_passthrough:
                vbt.inventory.append(result.ticket)
                vbt.cash -= result.price
            else:
                dealer.inventory.append(result.ticket)
                dealer.cash -= result.price
            recompute_dealer_state(dealer, vbt, self.params)
            return

        # Update trader - withdraw deposits
        trader.withdraw_deposits(result.price)
        trader.tickets_owned.append(result.ticket)

        # Log trade
        self.events.log_trade(
            day=self.day,
            side="BUY",
            trader_id=agent_id,
            ticket_id=result.ticket.id,
            bucket=bucket_id,
            price=result.price,
            is_passthrough=result.is_passthrough,
        )

        # Update eligibility
        if trader.deposit_balance <= self.buffer_B:
            eligible.discard(agent_id)

    def _try_borrow(
        self,
        trader: BankAwareTraderState,
        amount: Decimal,
    ) -> bool:
        """
        Try to borrow from trader's bank to cover shortfall.

        Args:
            trader: Trader requesting loan
            amount: Amount to borrow (≤ shortfall)

        Returns:
            True if loan granted, False otherwise
        """
        bank = self.banks[trader.bank_id]

        # Issue loan (purpose-bound to shortfall)
        loan = bank.issue_loan(
            borrower_id=trader.agent_id,
            amount=amount,
            tenor=self.loan_tenor,
            day=self.day,
        )

        # Add loan to trader
        trader.add_loan(loan)

        # Credit deposits (bank creates money)
        trader.add_deposit(amount, bank.current_deposit_rate, self.day)

        self.events.log(
            "loan_issued",
            self.day,
            trader_id=trader.agent_id,
            bank_id=bank.bank_id,
            amount=float(amount),
            rate=float(bank.current_loan_rate),
            maturity_day=loan.maturity_day,
        )

        return True

    def _select_ticket_to_sell(
        self,
        trader: BankAwareTraderState,
    ) -> Ticket | None:
        """Select ticket to sell (shortest maturity first)."""
        if not trader.tickets_owned:
            return None
        result: Ticket = min(trader.tickets_owned, key=lambda t: (t.remaining_tau, t.serial))
        return result

    def _select_bucket_to_buy(
        self,
        trader: BankAwareTraderState,
    ) -> str | None:
        """Select bucket to buy from (short preference)."""
        for bucket in self.buckets:
            bucket_id = bucket.name
            dealer = self.dealers[bucket_id]
            vbt = self.vbts[bucket_id]

            if len(dealer.inventory) > 0 or len(vbt.inventory) > 0:
                return bucket_id

        return None

    # =========================================================================
    # Phase 7: Settlement
    # =========================================================================

    def _settle_maturing_debt(self) -> None:
        """Settle maturing payables with default waterfall."""
        # Group maturing by issuer
        maturing_by_issuer: dict[AgentId, list[Ticket]] = {}

        for ticket in self.all_tickets.values():
            if ticket.maturity_day == self.day:
                issuer_id = ticket.issuer_id
                if issuer_id not in maturing_by_issuer:
                    maturing_by_issuer[issuer_id] = []
                maturing_by_issuer[issuer_id].append(ticket)

        # Settle each issuer
        for issuer_id, tickets in maturing_by_issuer.items():
            self._settle_issuer(issuer_id, tickets)

    def _settle_issuer(
        self,
        issuer_id: AgentId,
        tickets: list[Ticket],
    ) -> None:
        """Settle maturing tickets for one issuer."""
        if issuer_id not in self.traders:
            return

        issuer = self.traders[issuer_id]
        if issuer.defaulted:
            # Already defaulted - no further settlement
            for ticket in tickets:
                if ticket in issuer.obligations:
                    issuer.obligations.remove(ticket)
            return

        total_due: Decimal = sum((ticket.face for ticket in tickets), Decimal(0))

        if total_due == 0:
            return

        # Check if issuer can pay
        if issuer.deposit_balance >= total_due:
            # Full settlement
            recovery_rate = Decimal(1)
            issuer.withdraw_deposits(total_due)

            # Pay holders
            for ticket in tickets:
                self._pay_holder(ticket.owner_id, ticket.face, ticket)

            self.events.log_settlement(
                day=self.day,
                issuer_id=issuer_id,
                total_paid=total_due,
                n_tickets=len(tickets),
            )
        else:
            # Default - use waterfall
            self._handle_trader_default(issuer)

        # Update risk history
        defaulted = issuer.defaulted
        self.risk_assessor.update_history(
            day=self.day,
            issuer_id=issuer_id,
            defaulted=defaulted,
        )

        # Remove from obligations
        for ticket in tickets:
            if ticket in issuer.obligations:
                issuer.obligations.remove(ticket)

    def _handle_trader_default(
        self,
        trader: BankAwareTraderState,
    ) -> None:
        """
        Handle trader default with full waterfall.

        Steps:
        1. Liquidate all held payables at bid
        2. Pool with remaining deposits
        3. Distribute pro-rata weighted by maturity
        """
        trader.defaulted = True

        # Build liquidation prices
        liquidation_prices = {}
        for ticket in trader.tickets_owned:
            bucket_id = ticket.bucket_id
            if bucket_id:
                dealer = self.dealers[bucket_id]
                liquidation_prices[ticket.id] = dealer.bid

        # Resolve default
        resolution = resolve_default(
            trader=trader,
            current_day=self.day,
            liquidation_prices=liquidation_prices,
        )

        # Distribute payments
        for claimant_id, amount in resolution.payments.items():
            # Credit claimant's deposits/cash
            if claimant_id in self.traders:
                claimant = self.traders[claimant_id]
                bank = self.banks[claimant.bank_id]
                claimant.add_deposit(amount, bank.current_deposit_rate, self.day)
            elif claimant_id in self.banks:
                # Bank claim (loan recovery)
                bank = self.banks[claimant_id]
                bank.reserves += amount
            else:
                # Dealer/VBT
                for dealer in self.dealers.values():
                    if dealer.agent_id == claimant_id:
                        dealer.cash += amount
                        break
                else:
                    for vbt in self.vbts.values():
                        if vbt.agent_id == claimant_id:
                            vbt.cash += amount
                            break

        # Transfer liquidated assets to dealer
        for ticket in list(trader.tickets_owned):
            if ticket.bucket_id:
                dealer = self.dealers[ticket.bucket_id]
                ticket.owner_id = dealer.agent_id
                dealer.inventory.append(ticket)
                dealer.cash -= liquidation_prices.get(ticket.id, Decimal(0))
                trader.tickets_owned.remove(ticket)

        # Clear deposits
        trader.deposit_cohorts = []

        # Log default
        self.events.log(
            "trader_default",
            self.day,
            trader_id=trader.agent_id,
            total_pool=float(resolution.total_pool),
            liquidation_proceeds=float(resolution.liquidation_proceeds),
            recovery_rate=float(resolution.recovery_rate),
            n_claims=len(resolution.claims),
        )

    def _pay_holder(
        self,
        holder_id: AgentId,
        amount: Decimal,
        ticket: Ticket,
    ) -> None:
        """Credit payment to holder."""
        if holder_id in self.traders:
            trader = self.traders[holder_id]
            bank = self.banks[trader.bank_id]
            trader.add_deposit(amount, bank.current_deposit_rate, self.day)

            # Track interbank payment
            issuer = self.traders.get(ticket.issuer_id)
            if issuer and trader.bank_id != issuer.bank_id:
                self.interbank_ledger.record_payment(
                    issuer.bank_id,
                    trader.bank_id,
                    amount,
                )

            if ticket in trader.tickets_owned:
                trader.tickets_owned.remove(ticket)
            return

        # Dealer
        for dealer in self.dealers.values():
            if dealer.agent_id == holder_id:
                dealer.cash += amount
                if ticket in dealer.inventory:
                    dealer.inventory.remove(ticket)
                return

        # VBT
        for vbt in self.vbts.values():
            if vbt.agent_id == holder_id:
                vbt.cash += amount
                if ticket in vbt.inventory:
                    vbt.inventory.remove(ticket)
                return

    # =========================================================================
    # Phase 8: Interbank Settlement
    # =========================================================================

    def _settle_interbank(self) -> None:
        """Settle interbank positions at end of day."""
        bank_reserves = {
            bank_id: bank.reserves for bank_id, bank in self.banks.items()
        }

        updated_reserves, results, cb_borrowing = settle_interbank_positions(
            ledger=self.interbank_ledger,
            bank_reserves=bank_reserves,
            cb_lending_rate=self.config.cb_lending_rate,
            current_day=self.day,
        )

        # Update bank reserves
        for bank_id, reserves in updated_reserves.items():
            if bank_id in self.banks:
                self.banks[bank_id].reserves = reserves

        # Record CB borrowing
        for bank_id, amount in cb_borrowing.items():
            if bank_id in self.banks and amount > 0:
                self.banks[bank_id].add_cb_borrowing(
                    amount=amount,
                    rate=self.config.cb_lending_rate,
                    day=self.day,
                )
                self.events.log(
                    "cb_borrowing",
                    self.day,
                    bank_id=bank_id,
                    amount=float(amount),
                )

        # Log settlements
        for result in results:
            if result.net_amount > 0:
                self.events.log(
                    "interbank_settlement",
                    self.day,
                    payer=result.payer_bank_id,
                    receiver=result.receiver_bank_id,
                    amount=float(result.net_amount),
                    cb_borrowing=float(result.payer_cb_borrowing),
                )

    # =========================================================================
    # Phase 9: VBT Anchor Update
    # =========================================================================

    def _update_vbt_anchors(self) -> None:
        """Update VBT anchors based on loss rates."""
        for bucket_id in self.vbts:
            vbt = self.vbts[bucket_id]

            M_old = vbt.M
            O_old = vbt.O

            loss_rate = self.events.get_bucket_loss_rate(self.day, bucket_id)

            if loss_rate > 0:
                vbt.update_from_loss(loss_rate)

                dealer = self.dealers[bucket_id]
                recompute_dealer_state(dealer, vbt, self.params)

    # =========================================================================
    # Snapshot
    # =========================================================================

    def _capture_snapshot(self) -> None:
        """Capture deep copy of current state."""
        dealers_dict = {}
        for bucket_id, dealer in self.dealers.items():
            dealers_dict[bucket_id] = {
                "bucket_id": dealer.bucket_id,
                "agent_id": dealer.agent_id,
                "cash": float(dealer.cash),
                "a": dealer.a,
                "bid": float(dealer.bid),
                "ask": float(dealer.ask),
            }

        vbts_dict = {}
        for bucket_id, vbt in self.vbts.items():
            vbts_dict[bucket_id] = {
                "bucket_id": vbt.bucket_id,
                "agent_id": vbt.agent_id,
                "M": float(vbt.M),
                "O": float(vbt.O),
                "A": float(vbt.A),
                "B": float(vbt.B),
                "cash": float(vbt.cash),
            }

        traders_dict = {}
        for agent_id, trader in self.traders.items():
            traders_dict[agent_id] = {
                "agent_id": trader.agent_id,
                "bank_id": trader.bank_id,
                "deposit_balance": float(trader.deposit_balance),
                "defaulted": trader.defaulted,
                "asset_issuer_id": trader.asset_issuer_id,
                "n_tickets": len(trader.tickets_owned),
                "n_loans": len(trader.loans),
                "total_loan_principal": float(trader.total_loan_principal),
            }

        banks_dict = {}
        for bank_id, bank in self.banks.items():
            banks_dict[bank_id] = {
                "bank_id": bank.bank_id,
                "reserves": float(bank.reserves),
                "deposit_rate": float(bank.current_deposit_rate),
                "loan_rate": float(bank.current_loan_rate),
                "total_loans": float(bank.total_loans),
                "total_cb_borrowing": float(bank.total_cb_borrowing),
            }

        tickets_dict = {}
        for ticket_id, ticket in self.all_tickets.items():
            tickets_dict[ticket_id] = {
                "id": ticket.id,
                "issuer_id": ticket.issuer_id,
                "owner_id": ticket.owner_id,
                "face": float(ticket.face),
                "maturity_day": ticket.maturity_day,
                "remaining_tau": ticket.remaining_tau,
                "bucket_id": ticket.bucket_id,
            }

        day_events = [e for e in self.events.events if e.get("day") == self.day]

        snapshot = BankDealerDaySnapshot(
            day=self.day,
            dealers=dealers_dict,
            vbts=vbts_dict,
            traders=traders_dict,
            banks=banks_dict,
            tickets=tickets_dict,
            events=day_events,
        )
        self.snapshots.append(snapshot)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get summary metrics for the simulation.

        Returns dict with:
        - total_defaults: Number of traders who defaulted
        - total_loans: Total loan volume issued
        - total_cb_borrowing: Total CB borrowing
        - avg_recovery_rate: Average recovery rate on defaults
        """
        n_defaults = sum(1 for t in self.traders.values() if t.defaulted)
        total_loans = sum(bank.total_loans for bank in self.banks.values())
        total_cb = sum(bank.total_cb_borrowing for bank in self.banks.values())

        # Compute average recovery from events
        default_events = [
            e for e in self.events.events
            if e.get("type") == "trader_default"
        ]
        if default_events:
            avg_recovery = sum(
                e.get("recovery_rate", 0) for e in default_events
            ) / len(default_events)
        else:
            avg_recovery = 1.0

        return {
            "total_defaults": n_defaults,
            "total_loans": float(total_loans),
            "total_cb_borrowing": float(total_cb),
            "avg_recovery_rate": avg_recovery,
            "n_traders": len(self.traders),
            "n_banks": len(self.banks),
            "simulation_days": self.day,
        }
