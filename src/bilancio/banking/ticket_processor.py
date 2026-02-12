"""
Ticket processor for Banks-as-Dealers.

Implements Section 6.2: Execution given quotes - one ticket at a time.

At any instant within day t, the bank sits in a state and processes
a single client ticket: either a payment-credit, loan, or withdrawal.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

from bilancio.banking.types import Ticket, TicketType, Quote
from bilancio.banking.state import BankDealerState, CentralBankParams
from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_quotes,
    compute_inventory,
    simple_risk_index,
)
from bilancio.banking.reserve_projection import (
    project_reserves,
    compute_cash_tightness,
    ReserveProjection,
)


@dataclass
class TicketResult:
    """Result of processing a single ticket."""
    ticket: Ticket
    success: bool
    message: str

    # Balance changes
    reserve_delta: int = 0
    deposit_delta: int = 0
    loan_delta: int = 0

    # Updated quotes (after ticket)
    new_quote: Optional[Quote] = None

    # For inter-bank: the counterparty bank's reserve change
    counterparty_reserve_delta: int = 0


@dataclass
class TicketProcessor:
    """
    Processes client tickets for a BankDealer.

    Maintains references to the bank state and pricing parameters,
    and handles the ticket-by-ticket processing loop.
    """
    state: BankDealerState
    cb_params: CentralBankParams
    pricing_params: PricingParams

    # Current projection (updated after each ticket)
    _projection: Optional[ReserveProjection] = None

    def refresh_projection(self) -> ReserveProjection:
        """Rebuild the 10-day reserve projection."""
        self._projection = project_reserves(
            self.state,
            self.cb_params,
            horizon=10,
        )
        return self._projection

    def refresh_quotes(self) -> Quote:
        """
        Recompute quotes based on current state and projection.

        This is called after each ticket to update (r_D, r_L).
        """
        if self._projection is None:
            self.refresh_projection()

        assert self._projection is not None  # guaranteed by refresh_projection above

        # Compute cash-tightness L*
        cash_tightness = compute_cash_tightness(
            self._projection,
            self.pricing_params.reserve_floor,
        )

        # Compute inventory x = R̂(t+2) - R^tar
        t = self.state.current_day
        r_t2 = self._projection.at_horizon(t + 2)
        inventory = compute_inventory(r_t2, self.pricing_params.reserve_target)

        # Simple risk index (just cash-tightness for now)
        risk_index = simple_risk_index(cash_tightness)

        # Get quotes
        quote = compute_quotes(
            inventory=inventory,
            cash_tightness=cash_tightness,
            risk_index=risk_index,
            params=self.pricing_params,
            day=self.state.current_day,
            ticket_number=self.state.ticket_count,
        )

        # Store in state
        self.state.current_quote = quote
        self.state.quote_history.append(quote)

        return quote

    # =========================================================================
    # E1: Book the next ticket and update cohorts
    # =========================================================================

    def process_ticket(self, ticket: Ticket) -> TicketResult:
        """
        Process a single ticket.

        Dispatches to the appropriate handler based on ticket type.
        After processing, refreshes projection and quotes.
        """
        # Ensure we have current quotes
        if self.state.current_quote is None:
            self.refresh_quotes()

        assert self.state.current_quote is not None  # guaranteed by refresh_quotes above

        # Stamp the ticket with current rates
        ticket.stamped_deposit_rate = self.state.current_quote.deposit_rate
        ticket.stamped_loan_rate = self.state.current_quote.loan_rate

        # Dispatch by type
        if ticket.ticket_type == TicketType.PAYMENT_CREDIT:
            result = self._process_payment_credit(ticket)
        elif ticket.ticket_type == TicketType.LOAN:
            result = self._process_loan(ticket)
        elif ticket.ticket_type == TicketType.WITHDRAWAL:
            result = self._process_withdrawal(ticket)
        else:  # pragma: no cover
            result = TicketResult(  # type: ignore[unreachable]
                ticket=ticket,
                success=False,
                message=f"Unknown ticket type: {ticket.ticket_type}",
            )

        if result.success:
            # E2-E4: Update projection and refresh quotes
            self.refresh_projection()
            result.new_quote = self.refresh_quotes()

        return result

    def _process_payment_credit(self, ticket: Ticket) -> TicketResult:
        """
        Process a payment-credit ticket.

        Case (pay): A client receives a payment from another bank's client.
        - Create payment-origin deposit ticket
        - Receive reserves via RTGS (if inter-bank)

        What moves:
        - D^pay += S^pay
        - R += S^pay (RTGS inflow, if inter-bank)
        """
        amount = ticket.amount
        day = self.state.current_day

        # Create deposit cohort
        assert ticket.stamped_deposit_rate is not None
        self.state.add_deposit_cohort(
            day=day,
            origin="payment",
            amount=amount,
            rate=ticket.stamped_deposit_rate,
        )

        # Reserve inflow (RTGS from counterparty bank)
        # For intra-bank transfers, this would be 0
        reserve_inflow = amount if ticket.counterparty_bank_id else 0
        self.state.reserves += reserve_inflow

        return TicketResult(
            ticket=ticket,
            success=True,
            message=f"Payment credit: +{amount} deposit, +{reserve_inflow} reserves",
            reserve_delta=reserve_inflow,
            deposit_delta=amount,
            counterparty_reserve_delta=-reserve_inflow,  # Other bank loses reserves
        )

    def _process_loan(self, ticket: Ticket) -> TicketResult:
        """
        Process a loan ticket.

        Case (loan): A client takes out a loan.
        - Create loan-origin deposit (credit borrower's account)
        - Create loan asset

        What moves:
        - D^loan += S^loan
        - L += S^loan
        - R: NO CHANGE (loan-created deposits are internal)
        """
        amount = ticket.amount
        day = self.state.current_day

        # Create loan cohort
        assert ticket.stamped_loan_rate is not None
        self.state.add_loan_cohort(
            day=day,
            amount=amount,
            rate=ticket.stamped_loan_rate,
        )

        # Create loan-origin deposit cohort
        assert ticket.stamped_deposit_rate is not None
        self.state.add_deposit_cohort(
            day=day,
            origin="loan",
            amount=amount,
            rate=ticket.stamped_deposit_rate,
        )

        # No reserve movement at issuance

        return TicketResult(
            ticket=ticket,
            success=True,
            message=f"Loan issued: +{amount} loan, +{amount} deposit (loan-origin)",
            reserve_delta=0,
            deposit_delta=amount,
            loan_delta=amount,
        )

    def _process_withdrawal(self, ticket: Ticket) -> TicketResult:
        """
        Process a withdrawal ticket.

        Case (wd): A client withdraws/pays out to another bank's client.
        - Cancel deposits
        - Transfer reserves via RTGS

        What moves:
        - D -= S^wd
        - R -= S^wd (RTGS outflow)

        Reserves can go negative (intraday overdraft), which is converted
        to CB borrowing at end of day.
        """
        amount = ticket.amount
        day = self.state.current_day

        # Withdraw from deposits (FIFO by default)
        withdrawn = self.state.withdraw_from_deposits(
            amount=amount,
            allocation_rule="fifo",
        )

        if withdrawn < amount:
            # Insufficient deposits
            return TicketResult(
                ticket=ticket,
                success=False,
                message=f"Insufficient deposits: needed {amount}, have {withdrawn}",
                deposit_delta=-withdrawn,
            )

        # Reserve outflow (RTGS to counterparty bank)
        reserve_outflow = amount if ticket.counterparty_bank_id else 0
        self.state.reserves -= reserve_outflow

        # Update realized withdrawals
        self.state.withdrawals_realized += amount

        return TicketResult(
            ticket=ticket,
            success=True,
            message=f"Withdrawal: -{amount} deposit, -{reserve_outflow} reserves",
            reserve_delta=-reserve_outflow,
            deposit_delta=-amount,
            counterparty_reserve_delta=reserve_outflow,  # Other bank gains reserves
        )

    # =========================================================================
    # Day initialization
    # =========================================================================

    def initialize_day(
        self,
        day: int,
        withdrawal_forecast: int = 0,
    ) -> Quote:
        """
        Initialize for a new day.

        Sets up the day's state, withdrawal forecast, and computes
        initial quotes.
        """
        self.state.advance_to_day(day)
        self.state.withdrawal_forecast = withdrawal_forecast

        # Build projection and compute initial quotes
        self.refresh_projection()
        return self.refresh_quotes()


# =============================================================================
# Multi-bank settlement utilities
# =============================================================================


@dataclass
class InterBankSettlement:
    """
    Result of settling inter-bank payment.

    When Client A (at Bank 1) pays Client B (at Bank 2):
    1. Bank 1 processes withdrawal ticket
    2. Bank 2 processes payment-credit ticket
    3. Reserves flow from Bank 1 to Bank 2
    """
    payer_bank_id: str
    payee_bank_id: str
    amount: int

    payer_result: Optional[TicketResult] = None
    payee_result: Optional[TicketResult] = None

    @property
    def success(self) -> bool:
        return (
            self.payer_result is not None
            and self.payee_result is not None
            and self.payer_result.success
            and self.payee_result.success
        )


def process_inter_bank_payment(
    payer_processor: TicketProcessor,
    payee_processor: TicketProcessor,
    payer_client_id: str,
    payee_client_id: str,
    amount: int,
) -> InterBankSettlement:
    """
    Process a payment from a client at one bank to a client at another bank.

    This creates:
    1. Withdrawal ticket at payer's bank
    2. Payment-credit ticket at payee's bank
    3. Reserve transfer between banks (RTGS)

    Args:
        payer_processor: TicketProcessor for payer's bank
        payee_processor: TicketProcessor for payee's bank
        payer_client_id: Client making the payment
        payee_client_id: Client receiving the payment
        amount: Payment amount

    Returns:
        InterBankSettlement with results from both sides
    """
    payer_bank_id = payer_processor.state.bank_id
    payee_bank_id = payee_processor.state.bank_id
    day = payer_processor.state.current_day

    settlement = InterBankSettlement(
        payer_bank_id=payer_bank_id,
        payee_bank_id=payee_bank_id,
        amount=amount,
    )

    # Step 1: Process withdrawal at payer's bank
    withdrawal_ticket = Ticket(
        id=payer_processor.state.new_ticket_id(),
        ticket_type=TicketType.WITHDRAWAL,
        amount=amount,
        client_id=payer_client_id,
        created_day=day,
        counterparty_bank_id=payee_bank_id,
    )

    payer_result = payer_processor.process_ticket(withdrawal_ticket)
    settlement.payer_result = payer_result

    if not payer_result.success:
        # Withdrawal failed (e.g., insufficient funds)
        return settlement

    # Step 2: Process payment credit at payee's bank
    credit_ticket = Ticket(
        id=payee_processor.state.new_ticket_id(),
        ticket_type=TicketType.PAYMENT_CREDIT,
        amount=amount,
        client_id=payee_client_id,
        created_day=day,
        counterparty_bank_id=payer_bank_id,
    )

    payee_result = payee_processor.process_ticket(credit_ticket)
    settlement.payee_result = payee_result

    # Reserve transfer is implicit in the ticket processing:
    # - Payer bank: reserves -= amount
    # - Payee bank: reserves += amount

    return settlement


def process_intra_bank_payment(
    processor: TicketProcessor,
    payer_client_id: str,
    payee_client_id: str,
    amount: int,
) -> Tuple[TicketResult, TicketResult]:
    """
    Process a payment between two clients at the same bank.

    This creates:
    1. Withdrawal ticket (payer's deposits decrease)
    2. Payment-credit ticket (payee's deposits increase)

    No reserve movement (internal transfer).

    Returns:
        Tuple of (withdrawal_result, credit_result)
    """
    day = processor.state.current_day

    # Step 1: Process withdrawal
    withdrawal_ticket = Ticket(
        id=processor.state.new_ticket_id(),
        ticket_type=TicketType.WITHDRAWAL,
        amount=amount,
        client_id=payer_client_id,
        created_day=day,
        counterparty_bank_id=None,  # Same bank
    )

    withdrawal_result = processor.process_ticket(withdrawal_ticket)

    if not withdrawal_result.success:
        # Create a dummy failed credit result
        credit_result = TicketResult(
            ticket=Ticket(
                id="FAILED",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=amount,
                client_id=payee_client_id,
                created_day=day,
            ),
            success=False,
            message="Withdrawal failed, credit not processed",
        )
        return (withdrawal_result, credit_result)

    # Step 2: Process credit
    credit_ticket = Ticket(
        id=processor.state.new_ticket_id(),
        ticket_type=TicketType.PAYMENT_CREDIT,
        amount=amount,
        client_id=payee_client_id,
        created_day=day,
        counterparty_bank_id=None,  # Same bank
    )

    credit_result = processor.process_ticket(credit_ticket)

    return (withdrawal_result, credit_result)
