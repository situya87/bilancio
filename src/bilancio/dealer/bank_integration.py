"""
Bank-dealer integration models.

This module extends the dealer ring simulation to use bank deposits
instead of cash, enabling interest-bearing deposits, bank lending,
and interbank settlement.

References:
- docs/plans/banks_dealers_integration.md
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, List, Optional

from bilancio.core.ids import AgentId, new_id


# =============================================================================
# Loan Types
# =============================================================================


@dataclass
class TraderLoan:
    """
    A loan from a bank to a trader.

    Loans are used when traders need liquidity but prefer borrowing
    over selling payables (when borrow cost < sell cost).

    All loans have fixed tenor = max payable maturity in the system.
    """
    loan_id: str
    borrower_id: AgentId
    bank_id: str
    principal: Decimal
    rate: Decimal  # 2-day rate, locked at issuance
    issuance_day: int
    maturity_day: int

    @property
    def repayment_amount(self) -> Decimal:
        """Amount due at maturity: principal × (1 + rate)."""
        return self.principal * (Decimal(1) + self.rate)

    def days_to_maturity(self, current_day: int) -> int:
        """Days remaining until maturity."""
        return max(0, self.maturity_day - current_day)


@dataclass
class DepositCohort:
    """
    A cohort of deposits created on a specific day.

    Deposits are tracked by cohort for interest calculation:
    - Interest accrues after 2-day grace period
    - Rate is locked at deposit time

    This is a trader-facing view; the bank tracks these in BankDealerState.
    """
    cohort_id: str
    bank_id: str
    depositor_id: AgentId
    principal: Decimal
    rate: Decimal  # Locked at deposit time
    issuance_day: int
    interest_accrued: Decimal = Decimal(0)
    last_interest_day: Optional[int] = None

    @property
    def total_balance(self) -> Decimal:
        """Principal plus accrued interest."""
        return self.principal + self.interest_accrued

    def next_interest_day(self) -> int:
        """Next day when interest should be credited."""
        if self.last_interest_day is None:
            return self.issuance_day + 2  # Grace period
        return self.last_interest_day + 2

    def compute_interest(self) -> Decimal:
        """Compute interest for one 2-day period (simple interest)."""
        return self.rate * self.principal


# =============================================================================
# Extended Trader State
# =============================================================================


@dataclass
class BankAwareTraderState:
    """
    Extended trader state with bank deposits instead of cash.

    This replaces TraderState.cash with deposit tracking at a specific bank.
    All payments flow through deposits.
    """
    agent_id: AgentId
    bank_id: str  # Which bank holds this trader's deposits

    # Payables held (assets) - same as TraderState
    tickets_owned: List[Any] = field(default_factory=list)

    # Payables issued (liabilities) - same as TraderState
    obligations: List[Any] = field(default_factory=list)

    # Single-issuer constraint - same as TraderState
    asset_issuer_id: AgentId | None = None

    # Deposit cohorts (replaces cash)
    deposit_cohorts: list[DepositCohort] = field(default_factory=list)

    # Outstanding loans from bank
    loans: list[TraderLoan] = field(default_factory=list)

    # Default status
    defaulted: bool = False

    @property
    def deposit_balance(self) -> Decimal:
        """Total deposit balance across all cohorts."""
        return sum((c.total_balance for c in self.deposit_cohorts), Decimal(0))

    @property
    def total_loan_principal(self) -> Decimal:
        """Total outstanding loan principal."""
        return sum((loan.principal for loan in self.loans), Decimal(0))

    def payment_due(self, day: int) -> Decimal:
        """
        Total payment obligations due on a given day.

        Includes both payable maturities AND loan repayments.
        """
        payable_due: Decimal = sum(
            (t.face for t in self.obligations if t.maturity_day == day),
            Decimal(0),
        )
        loan_due: Decimal = sum(
            (loan.repayment_amount for loan in self.loans
            if loan.maturity_day == day),
            Decimal(0),
        )
        return payable_due + loan_due

    def payable_due(self, day: int) -> Decimal:
        """Payable obligations only (excluding loans)."""
        return sum(
            (t.face for t in self.obligations if t.maturity_day == day),
            Decimal(0),
        )

    def loan_due(self, day: int) -> Decimal:
        """Loan repayments due on a given day."""
        return sum(
            (loan.repayment_amount for loan in self.loans
            if loan.maturity_day == day),
            Decimal(0),
        )

    def shortfall(self, day: int) -> Decimal:
        """
        Payment shortfall on a given day.

        shortfall = max(0, payment_due - deposit_balance)
        """
        due = self.payment_due(day)
        return max(Decimal(0), due - self.deposit_balance)

    def earliest_liability_day(self, after_day: int) -> int | None:
        """
        Find the earliest liability date after the given day.

        Considers both payable obligations and loan maturities.
        """
        future_payable_days = [
            t.maturity_day for t in self.obligations
            if t.maturity_day > after_day
        ]
        future_loan_days = [
            loan.maturity_day for loan in self.loans
            if loan.maturity_day > after_day
        ]
        all_future = future_payable_days + future_loan_days
        return min(all_future) if all_future else None

    def add_deposit(
        self,
        amount: Decimal,
        rate: Decimal,
        day: int,
        cohort_id: str | None = None,
    ) -> DepositCohort:
        """
        Add a new deposit cohort.

        Args:
            amount: Deposit amount
            rate: Interest rate (locked)
            day: Issuance day
            cohort_id: Optional ID (auto-generated if None)

        Returns:
            The created deposit cohort
        """
        if cohort_id is None:
            cohort_id = f"dep_{self.agent_id}_{day}_{len(self.deposit_cohorts)}"

        cohort = DepositCohort(
            cohort_id=cohort_id,
            bank_id=self.bank_id,
            depositor_id=self.agent_id,
            principal=amount,
            rate=rate,
            issuance_day=day,
        )
        self.deposit_cohorts.append(cohort)
        return cohort

    def withdraw_deposits(self, amount: Decimal) -> Decimal:
        """
        Withdraw from deposits using FIFO (oldest first).

        Args:
            amount: Amount to withdraw

        Returns:
            Actual amount withdrawn (may be less if insufficient)
        """
        remaining = amount
        withdrawn = Decimal(0)

        # Sort by issuance day (FIFO)
        sorted_cohorts = sorted(
            self.deposit_cohorts,
            key=lambda c: c.issuance_day
        )

        for cohort in sorted_cohorts:
            if remaining <= 0:
                break

            available = cohort.total_balance
            if available <= 0:
                continue

            take = min(remaining, available)

            # First take from accrued interest, then principal
            if cohort.interest_accrued >= take:
                cohort.interest_accrued -= take
            else:
                from_interest = cohort.interest_accrued
                from_principal = take - from_interest
                cohort.interest_accrued = Decimal(0)
                cohort.principal -= from_principal

            withdrawn += take
            remaining -= take

        # Remove empty cohorts
        self.deposit_cohorts = [
            c for c in self.deposit_cohorts if c.total_balance > 0
        ]

        return withdrawn

    def add_loan(self, loan: TraderLoan) -> None:
        """Add a loan to this trader's outstanding loans."""
        self.loans.append(loan)

    def repay_loan(self, loan: TraderLoan) -> Decimal:
        """
        Repay a loan (remove from outstanding).

        Returns the repayment amount.
        """
        if loan in self.loans:
            self.loans.remove(loan)
        return loan.repayment_amount


# =============================================================================
# Interbank Position Tracking
# =============================================================================


@dataclass
class InterbankPosition:
    """
    Tracks bilateral interbank obligations.

    During the day, payments between clients at different banks
    create interbank payables/receivables. These are netted
    at end of day and settled via reserve transfers.
    """
    from_bank_id: str
    to_bank_id: str
    amount: Decimal = Decimal(0)

    def add(self, amount: Decimal) -> None:
        """Add to the position."""
        self.amount += amount

    def clear(self) -> Decimal:
        """Clear and return the amount."""
        amt = self.amount
        self.amount = Decimal(0)
        return amt


@dataclass
class InterbankLedger:
    """
    Ledger tracking all interbank positions for settlement.

    Positions are keyed by (from_bank, to_bank) tuples.
    At end of day, we net bilateral positions and settle.
    """
    positions: dict[tuple[str, str], InterbankPosition] = field(
        default_factory=dict
    )

    def record_payment(
        self,
        from_bank_id: str,
        to_bank_id: str,
        amount: Decimal,
    ) -> None:
        """
        Record an interbank payment obligation.

        Called when a client at from_bank pays a client at to_bank.
        """
        if from_bank_id == to_bank_id:
            return  # Intrabank, no interbank position

        key = (from_bank_id, to_bank_id)
        if key not in self.positions:
            self.positions[key] = InterbankPosition(
                from_bank_id=from_bank_id,
                to_bank_id=to_bank_id,
            )
        self.positions[key].add(amount)

    def net_bilateral(self, bank_a: str, bank_b: str) -> tuple[str, str, Decimal]:
        """
        Net bilateral positions between two banks.

        Returns (payer_bank, receiver_bank, net_amount).
        If net is zero, payer and receiver are both bank_a (arbitrary).
        """
        a_to_b = self.positions.get((bank_a, bank_b), InterbankPosition(bank_a, bank_b)).amount
        b_to_a = self.positions.get((bank_b, bank_a), InterbankPosition(bank_b, bank_a)).amount

        net = a_to_b - b_to_a

        if net >= 0:
            return (bank_a, bank_b, net)
        else:
            return (bank_b, bank_a, -net)

    def clear_bilateral(self, bank_a: str, bank_b: str) -> None:
        """Clear bilateral positions after settlement."""
        if (bank_a, bank_b) in self.positions:
            self.positions[(bank_a, bank_b)].clear()
        if (bank_b, bank_a) in self.positions:
            self.positions[(bank_b, bank_a)].clear()

    def get_all_bank_ids(self) -> set[str]:
        """Get all bank IDs involved in positions."""
        banks = set()
        for from_bank, to_bank in self.positions.keys():
            banks.add(from_bank)
            banks.add(to_bank)
        return banks


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BankDealerRingConfig:
    """
    Configuration for the integrated bank-dealer ring simulation.

    Extends dealer ring config with banking parameters.
    """
    # === Existing dealer config (to be inherited) ===
    ticket_size: Decimal = Decimal(1)
    max_days: int = 30
    seed: int = 42

    # === Bank structure ===
    n_trader_banks: int = 3  # Plus 1 dealer bank = 4 total

    # === Central Bank corridor ===
    cb_deposit_rate: Decimal = Decimal("0.01")  # r_floor (2-day)
    cb_lending_rate: Decimal = Decimal("0.03")  # r_ceiling (2-day)

    # === Loan parameters ===
    # Tenor is set dynamically = max payable maturity
    loan_tenor_days: int | None = None  # If None, use max payable maturity

    # === Interest ===
    deposit_grace_period: int = 2  # Days before interest accrues

    # === Bank internal parameters ===
    reserve_target_ratio: Decimal = Decimal("0.10")  # R^tar / total_deposits
    symmetric_capacity_ratio: Decimal = Decimal("0.50")  # X* / R^tar
    bank_ticket_size: int = 10000  # S_fund for bank pricing

    # === Risk parameters ===
    alpha: Decimal = Decimal("0.005")  # Midline sensitivity to L*
    gamma: Decimal = Decimal("0.002")  # Midline sensitivity to ρ

    @property
    def corridor_width(self) -> Decimal:
        """Ω = r_ceiling - r_floor."""
        return self.cb_lending_rate - self.cb_deposit_rate

    @property
    def corridor_mid(self) -> Decimal:
        """M = (r_floor + r_ceiling) / 2."""
        return (self.cb_deposit_rate + self.cb_lending_rate) / 2

    def get_bank_id_for_trader(self, trader_index: int) -> str:
        """
        Determine which bank a trader belongs to.

        Rule: H_i → Bank ((i-1) mod n_trader_banks) + 1

        Args:
            trader_index: 1-based trader index (H1, H2, ...)

        Returns:
            Bank ID string (e.g., "bank_1", "bank_2", ...)
        """
        bank_num = ((trader_index - 1) % self.n_trader_banks) + 1
        return f"bank_{bank_num}"

    @property
    def dealer_bank_id(self) -> str:
        """ID of the dealer bank (last bank)."""
        return f"bank_{self.n_trader_banks + 1}"


# =============================================================================
# Utility Functions
# =============================================================================


def compute_borrow_vs_sell_decision(
    shortfall: Decimal,
    payable_expected_value: Decimal,
    dealer_bid: Decimal,
    loan_rate: Decimal,
    loan_tenor: int,
) -> str:
    """
    Decide whether to borrow or sell to cover shortfall.

    Args:
        shortfall: Amount needed
        payable_expected_value: E[payoff] of payable that could be sold
        dealer_bid: Price dealer offers for payable
        loan_rate: Bank's loan rate (2-day)
        loan_tenor: Loan maturity in days

    Returns:
        "BORROW" or "SELL"
    """
    # Cost of selling: opportunity cost
    sell_cost = payable_expected_value - dealer_bid

    # Cost of borrowing: interest over tenor
    # Simple interest approximation for small rates/tenors
    borrow_cost = shortfall * loan_rate * Decimal(loan_tenor) / Decimal(2)

    if borrow_cost < sell_cost:
        return "BORROW"
    else:
        return "SELL"


def compute_yield_sell_decision(
    payable_expected_value: Decimal,
    dealer_bid: Decimal,
    deposit_rate: Decimal,
    days_to_maturity: int,
) -> bool:
    """
    Decide whether to sell payable for yield (deposit > hold).

    Args:
        payable_expected_value: E[payoff] = (1 - P_default) × face
        dealer_bid: Price dealer offers
        deposit_rate: Bank's deposit rate (2-day)
        days_to_maturity: Days until payable matures

    Returns:
        True if should sell and deposit, False if should hold
    """
    # Value of holding to maturity
    hold_value = payable_expected_value

    # Value of selling and depositing
    # Compound over days_to_maturity / 2 periods (2-day rate)
    periods = Decimal(days_to_maturity) / Decimal(2)
    sell_deposit_value = dealer_bid * (Decimal(1) + deposit_rate) ** periods

    return sell_deposit_value > hold_value


def compute_yield_buy_decision(
    payable_expected_value: Decimal,
    dealer_ask: Decimal,
    deposit_rate: Decimal,
    days_to_maturity: int,
) -> bool:
    """
    Decide whether to buy payable for yield (payable > deposit).

    Args:
        payable_expected_value: E[payoff] = (1 - P_default) × face
        dealer_ask: Price dealer charges
        deposit_rate: Bank's deposit rate (2-day)
        days_to_maturity: Days until payable matures

    Returns:
        True if should buy payable, False if should keep deposits
    """
    # Value of buying and holding to maturity
    buy_value = payable_expected_value

    # Value of keeping deposits
    periods = Decimal(days_to_maturity) / Decimal(2)
    hold_deposit_value = dealer_ask * (Decimal(1) + deposit_rate) ** periods

    return buy_value > hold_deposit_value


# =============================================================================
# Interbank Settlement
# =============================================================================


@dataclass
class InterbankSettlementResult:
    """Result of settling interbank positions between two banks."""
    payer_bank_id: str
    receiver_bank_id: str
    net_amount: Decimal
    payer_cb_borrowing: Decimal = Decimal(0)  # If payer needed CB funds


def settle_interbank_positions(
    ledger: InterbankLedger,
    bank_reserves: dict[str, Decimal],
    cb_lending_rate: Decimal,
    current_day: int,
) -> tuple[dict[str, Decimal], list[InterbankSettlementResult], dict[str, Decimal]]:
    """
    Settle all interbank positions at end of day.

    Process:
    1. For each bank pair, compute net bilateral position
    2. Transfer reserves from net payer to net receiver
    3. If payer short on reserves, borrow from CB

    Args:
        ledger: Interbank ledger with positions
        bank_reserves: Current reserves by bank_id
        cb_lending_rate: CB lending rate for LOLR
        current_day: Current simulation day

    Returns:
        Tuple of:
        - Updated bank_reserves dict
        - List of settlement results
        - Dict of CB borrowing by bank_id
    """
    results = []
    cb_borrowing: dict[str, Decimal] = {}
    updated_reserves = dict(bank_reserves)

    # Get all banks involved
    all_banks = sorted(ledger.get_all_bank_ids())

    # Settle bilaterally in order
    for i, bank_a in enumerate(all_banks):
        for bank_b in all_banks[i + 1:]:
            payer, receiver, net_amount = ledger.net_bilateral(bank_a, bank_b)

            if net_amount <= 0:
                # No settlement needed
                ledger.clear_bilateral(bank_a, bank_b)
                continue

            # Check if payer has enough reserves
            payer_reserves = updated_reserves.get(payer, Decimal(0))

            if payer_reserves >= net_amount:
                # Normal settlement
                updated_reserves[payer] = payer_reserves - net_amount
                updated_reserves[receiver] = updated_reserves.get(
                    receiver, Decimal(0)
                ) + net_amount
                cb_borrow = Decimal(0)
            else:
                # Payer needs CB borrowing
                shortfall = net_amount - payer_reserves
                cb_borrow = shortfall

                # CB creates reserves for payer
                if payer not in cb_borrowing:
                    cb_borrowing[payer] = Decimal(0)
                cb_borrowing[payer] += shortfall

                # Now payer can settle
                updated_reserves[payer] = Decimal(0)  # Used all reserves
                updated_reserves[receiver] = updated_reserves.get(
                    receiver, Decimal(0)
                ) + net_amount

            results.append(InterbankSettlementResult(
                payer_bank_id=payer,
                receiver_bank_id=receiver,
                net_amount=net_amount,
                payer_cb_borrowing=cb_borrow,
            ))

            # Clear the bilateral positions
            ledger.clear_bilateral(bank_a, bank_b)

    return updated_reserves, results, cb_borrowing


# =============================================================================
# Default Waterfall
# =============================================================================


@dataclass
class Claim:
    """A claim on a defaulting trader."""
    claimant_id: AgentId  # Bank or payable holder
    claim_type: str  # "loan" or "payable"
    amount: Decimal  # Face value / principal
    days_to_maturity: int  # 0 = due today

    @property
    def maturity_weight(self) -> Decimal:
        """Weight for distribution: 1 / (days_to_maturity + 1)."""
        return Decimal(1) / Decimal(self.days_to_maturity + 1)


@dataclass
class DefaultResolution:
    """Result of resolving a default."""
    defaulter_id: AgentId
    total_pool: Decimal  # Liquidation + deposits
    liquidation_proceeds: Decimal
    remaining_deposits: Decimal
    claims: list[Claim]
    payments: dict[AgentId, Decimal]  # claimant -> amount received
    recovery_rate: Decimal  # Weighted average


def resolve_default(
    trader: BankAwareTraderState,
    current_day: int,
    liquidation_prices: dict[str, Decimal],  # ticket_id -> bid price
) -> DefaultResolution:
    """
    Resolve a trader default using the waterfall procedure.

    Steps:
    1. Liquidate all held payables at bid prices
    2. Pool proceeds with remaining deposits
    3. List all claims (loans + payables)
    4. Weight by maturity (shorter = higher weight)
    5. Distribute pro-rata within weights

    Args:
        trader: The defaulting trader
        current_day: Current simulation day
        liquidation_prices: Bid prices for each ticket

    Returns:
        DefaultResolution with payment allocations
    """
    # Step 1: Liquidate assets
    liquidation_proceeds = Decimal(0)
    for ticket in trader.tickets_owned:
        price = liquidation_prices.get(ticket.id, Decimal(0))
        liquidation_proceeds += price

    # Step 2: Pool with deposits
    remaining_deposits = trader.deposit_balance
    total_pool = liquidation_proceeds + remaining_deposits

    # Step 3: List all claims
    claims = []

    # Loan claims (from bank)
    for loan in trader.loans:
        days_to_mat = max(0, loan.maturity_day - current_day)
        claims.append(Claim(
            claimant_id=loan.bank_id,
            claim_type="loan",
            amount=loan.repayment_amount,
            days_to_maturity=days_to_mat,
        ))

    # Payable claims (from holders)
    for ticket in trader.obligations:
        days_to_mat = max(0, ticket.maturity_day - current_day)
        claims.append(Claim(
            claimant_id=ticket.owner_id,
            claim_type="payable",
            amount=ticket.face,
            days_to_maturity=days_to_mat,
        ))

    # Step 4: Compute weights
    total_weight: Decimal = sum((c.maturity_weight * c.amount for c in claims), Decimal(0))

    # Step 5: Distribute pro-rata
    payments: dict[AgentId, Decimal] = {}

    if total_weight > 0 and total_pool > 0:
        for claim in claims:
            weighted_share = (claim.maturity_weight * claim.amount) / total_weight
            payment = weighted_share * total_pool

            if claim.claimant_id not in payments:
                payments[claim.claimant_id] = Decimal(0)
            payments[claim.claimant_id] += payment

    # Compute recovery rate
    total_claims: Decimal = sum((c.amount for c in claims), Decimal(0))
    recovery_rate = total_pool / total_claims if total_claims > 0 else Decimal(0)

    return DefaultResolution(
        defaulter_id=trader.agent_id,
        total_pool=total_pool,
        liquidation_proceeds=liquidation_proceeds,
        remaining_deposits=remaining_deposits,
        claims=claims,
        payments=payments,
        recovery_rate=recovery_rate,
    )


# =============================================================================
# Interest Accrual
# =============================================================================


def accrue_deposit_interest(
    trader: BankAwareTraderState,
    current_day: int,
) -> Decimal:
    """
    Accrue interest on all eligible deposit cohorts.

    Interest accrues after 2-day grace period, every 2 days thereafter.

    Args:
        trader: Trader with deposit cohorts
        current_day: Current simulation day

    Returns:
        Total interest credited
    """
    total_interest = Decimal(0)

    for cohort in trader.deposit_cohorts:
        if cohort.principal <= 0:
            continue

        if cohort.next_interest_day() == current_day:
            interest = cohort.compute_interest()
            cohort.interest_accrued += interest
            cohort.last_interest_day = current_day
            total_interest += interest

    return total_interest


# =============================================================================
# Bank State for Integrated Simulation
# =============================================================================


@dataclass
class IntegratedBankState:
    """
    Bank state for the integrated bank-dealer simulation.

    Extends the basic bank concept with client tracking and interbank positions.
    """
    bank_id: str
    current_day: int = 0

    # Reserves at Central Bank
    reserves: Decimal = Decimal(0)

    # Client deposits (by client_id)
    # This is a summary view; actual deposits are in BankAwareTraderState
    client_ids: set[AgentId] = field(default_factory=set)

    # Loans outstanding (by loan_id)
    loans_outstanding: dict[str, TraderLoan] = field(default_factory=dict)

    # CB borrowing cohorts
    cb_borrowing: list[tuple[int, Decimal, Decimal]] = field(default_factory=list)
    # (issuance_day, principal, rate)

    # Pricing state (from banking kernel)
    current_deposit_rate: Decimal = Decimal("0.02")
    current_loan_rate: Decimal = Decimal("0.025")

    @property
    def total_loans(self) -> Decimal:
        """Total outstanding loan principal."""
        return sum((loan.principal for loan in self.loans_outstanding.values()), Decimal(0))

    @property
    def total_cb_borrowing(self) -> Decimal:
        """Total CB borrowing principal."""
        return sum((principal for _, principal, _ in self.cb_borrowing), Decimal(0))

    def issue_loan(
        self,
        borrower_id: AgentId,
        amount: Decimal,
        tenor: int,
        day: int,
    ) -> TraderLoan:
        """
        Issue a loan to a trader.

        Args:
            borrower_id: Trader receiving the loan
            amount: Loan principal
            tenor: Days until maturity
            day: Current day (issuance)

        Returns:
            The created loan
        """
        loan_id = f"loan_{self.bank_id}_{borrower_id}_{day}"
        loan = TraderLoan(
            loan_id=loan_id,
            borrower_id=borrower_id,
            bank_id=self.bank_id,
            principal=amount,
            rate=self.current_loan_rate,
            issuance_day=day,
            maturity_day=day + tenor,
        )
        self.loans_outstanding[loan_id] = loan
        return loan

    def receive_loan_repayment(self, loan_id: str) -> Decimal:
        """
        Process loan repayment.

        Returns the amount received.
        """
        if loan_id not in self.loans_outstanding:
            return Decimal(0)

        loan = self.loans_outstanding.pop(loan_id)
        return loan.repayment_amount

    def add_cb_borrowing(
        self,
        amount: Decimal,
        rate: Decimal,
        day: int,
    ) -> None:
        """Record CB borrowing."""
        self.cb_borrowing.append((day, amount, rate))
        self.reserves += amount

    def process_cb_repayments(self, day: int) -> Decimal:
        """
        Process CB borrowing repayments due today.

        CB borrowing matures at issuance + 2.

        Returns total amount repaid.
        """
        total_repaid = Decimal(0)
        remaining = []

        for issuance_day, principal, rate in self.cb_borrowing:
            maturity = issuance_day + 2
            if maturity == day:
                repayment = principal * (Decimal(1) + rate)
                total_repaid += repayment
                self.reserves -= repayment
            else:
                remaining.append((issuance_day, principal, rate))

        self.cb_borrowing = remaining
        return total_repaid
