"""
Unit tests for ticket processing in the banking kernel.

Tests verify:
- Payment credit ticket processing
- Loan ticket processing
- Withdrawal ticket processing
- Quote refresh after tickets
- Inter-bank and intra-bank payment flows

References:
- "Banks-as-Dealers with deposits on demand" specification Section 6.2
"""

from decimal import Decimal

from bilancio.banking.pricing_kernel import PricingParams
from bilancio.banking.state import BankDealerState, CentralBankParams
from bilancio.banking.ticket_processor import (
    TicketProcessor,
    process_inter_bank_payment,
    process_intra_bank_payment,
)
from bilancio.banking.types import Ticket, TicketType


def create_standard_params() -> tuple[CentralBankParams, PricingParams]:
    """Create standard test parameters."""
    cb_params = CentralBankParams(
        reserve_remuneration_rate=Decimal("0.01"),
        cb_borrowing_rate=Decimal("0.03"),
    )

    pricing_params = PricingParams(
        reserve_remuneration_rate=cb_params.reserve_remuneration_rate,
        cb_borrowing_rate=cb_params.cb_borrowing_rate,
        reserve_target=100000,
        symmetric_capacity=50000,
        ticket_size=10000,
        reserve_floor=10000,
    )

    return cb_params, pricing_params


def create_processor(
    reserves: int = 100000,
    deposits: int = 50000,
) -> TicketProcessor:
    """Create a TicketProcessor with initial state."""
    cb_params, pricing_params = create_standard_params()

    state = BankDealerState(
        bank_id="TestBank",
        reserves=reserves,
        current_day=0,
    )

    if deposits > 0:
        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=deposits,
            rate=Decimal("0.015"),
        )

    return TicketProcessor(
        state=state,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )


class TestTicketProcessorInitialization:
    """Tests for TicketProcessor setup."""

    def test_processor_creation(self):
        """Processor can be created with valid state."""
        processor = create_processor()

        assert processor.state.bank_id == "TestBank"
        assert processor.state.reserves == 100000

    def test_initialize_day(self):
        """Initialize day sets up state and computes quotes."""
        processor = create_processor()

        quote = processor.initialize_day(day=1, withdrawal_forecast=10000)

        assert processor.state.current_day == 1
        assert processor.state.withdrawal_forecast == 10000
        assert quote.day == 1
        assert quote.deposit_rate is not None
        assert quote.loan_rate is not None

    def test_refresh_quotes(self):
        """Refresh quotes updates state and returns quote."""
        processor = create_processor()
        processor.initialize_day(day=1)

        quote = processor.refresh_quotes()

        assert processor.state.current_quote == quote
        assert quote in processor.state.quote_history


class TestPaymentCreditTicket:
    """Tests for payment credit (incoming payment) tickets."""

    def test_payment_credit_creates_deposit(self):
        """Payment credit creates deposit cohort."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",  # Inter-bank
        )

        result = processor.process_ticket(ticket)

        assert result.success
        assert result.deposit_delta == 10000
        assert processor.state.total_deposits == 60000

    def test_payment_credit_increases_reserves(self):
        """Inter-bank payment credit increases reserves."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        assert result.reserve_delta == 10000
        assert processor.state.reserves == 110000

    def test_intra_bank_payment_credit_no_reserve_change(self):
        """Intra-bank payment credit doesn't change reserves."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id=None,  # Same bank
        )

        result = processor.process_ticket(ticket)

        assert result.reserve_delta == 0
        assert processor.state.reserves == 100000

    def test_payment_credit_updates_quotes(self):
        """Processing ticket updates quotes."""
        processor = create_processor()
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        # Quote should be updated (higher reserves = lower rates)
        assert result.new_quote is not None


class TestLoanTicket:
    """Tests for loan issuance tickets."""

    def test_loan_creates_loan_cohort(self):
        """Loan ticket creates loan asset."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.LOAN,
            amount=30000,
            client_id="client_alice",
            created_day=1,
        )

        result = processor.process_ticket(ticket)

        assert result.success
        assert result.loan_delta == 30000
        assert processor.state.total_loans == 30000

    def test_loan_creates_deposit(self):
        """Loan creates loan-origin deposit."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.LOAN,
            amount=30000,
            client_id="client_alice",
            created_day=1,
        )

        result = processor.process_ticket(ticket)

        assert result.deposit_delta == 30000
        assert processor.state.deposits_loan_origin == 30000
        assert processor.state.total_deposits == 80000

    def test_loan_no_reserve_change(self):
        """Loan issuance doesn't change reserves."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.LOAN,
            amount=30000,
            client_id="client_alice",
            created_day=1,
        )

        result = processor.process_ticket(ticket)

        assert result.reserve_delta == 0
        assert processor.state.reserves == 100000

    def test_loan_rate_stamped(self):
        """Loan ticket stamped with current loan rate."""
        processor = create_processor()
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.LOAN,
            amount=30000,
            client_id="client_alice",
            created_day=1,
        )

        processor.process_ticket(ticket)

        assert ticket.stamped_loan_rate is not None
        assert ticket.stamped_loan_rate > Decimal("0")


class TestWithdrawalTicket:
    """Tests for withdrawal tickets."""

    def test_withdrawal_reduces_deposits(self):
        """Withdrawal decreases deposits."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=20000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        assert result.success
        assert result.deposit_delta == -20000
        assert processor.state.total_deposits == 30000

    def test_withdrawal_reduces_reserves(self):
        """Inter-bank withdrawal decreases reserves."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=20000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        assert result.reserve_delta == -20000
        assert processor.state.reserves == 80000

    def test_intra_bank_withdrawal_no_reserve_change(self):
        """Intra-bank withdrawal doesn't change reserves."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=20000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id=None,  # Same bank
        )

        result = processor.process_ticket(ticket)

        assert result.reserve_delta == 0
        assert processor.state.reserves == 100000

    def test_withdrawal_tracks_realized(self):
        """Withdrawal increments withdrawals_realized."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1, withdrawal_forecast=30000)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=15000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        processor.process_ticket(ticket)

        assert processor.state.withdrawals_realized == 15000
        assert processor.state.withdrawals_remaining == 15000

    def test_withdrawal_fails_insufficient_deposits(self):
        """Withdrawal fails if insufficient deposits."""
        processor = create_processor(reserves=100000, deposits=10000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=50000,  # More than available
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        assert not result.success
        assert "Insufficient deposits" in result.message


class TestReserveOverdraft:
    """Tests for reserve overdraft handling."""

    def test_withdrawal_allows_negative_reserves(self):
        """Reserves can go negative (intraday overdraft)."""
        processor = create_processor(reserves=30000, deposits=100000)
        processor.initialize_day(day=1)

        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=50000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        result = processor.process_ticket(ticket)

        assert result.success
        assert processor.state.reserves == -20000  # 30000 - 50000


class TestInterBankPayment:
    """Tests for inter-bank payment processing."""

    def test_inter_bank_payment_both_sides(self):
        """Inter-bank payment affects both banks correctly."""
        cb_params, pricing_params = create_standard_params()

        # Bank A (payer)
        state_a = BankDealerState(bank_id="BankA", reserves=100000)
        state_a.add_deposit_cohort(day=0, origin="payment", amount=80000, rate=Decimal("0.015"))
        processor_a = TicketProcessor(state_a, cb_params, pricing_params)
        processor_a.initialize_day(day=1)

        # Bank B (payee)
        state_b = BankDealerState(bank_id="BankB", reserves=50000)
        state_b.add_deposit_cohort(day=0, origin="payment", amount=40000, rate=Decimal("0.015"))
        processor_b = TicketProcessor(state_b, cb_params, pricing_params)
        processor_b.initialize_day(day=1)

        settlement = process_inter_bank_payment(
            payer_processor=processor_a,
            payee_processor=processor_b,
            payer_client_id="client_alice",
            payee_client_id="client_bob",
            amount=25000,
        )

        assert settlement.success

        # Bank A: reserves decreased, deposits decreased
        assert state_a.reserves == 75000  # 100000 - 25000
        assert state_a.total_deposits == 55000  # 80000 - 25000

        # Bank B: reserves increased, deposits increased
        assert state_b.reserves == 75000  # 50000 + 25000
        assert state_b.total_deposits == 65000  # 40000 + 25000

    def test_inter_bank_payment_reserve_conservation(self):
        """Total reserves are conserved in inter-bank payment."""
        cb_params, pricing_params = create_standard_params()

        state_a = BankDealerState(bank_id="BankA", reserves=100000)
        state_a.add_deposit_cohort(day=0, origin="payment", amount=80000, rate=Decimal("0.015"))
        processor_a = TicketProcessor(state_a, cb_params, pricing_params)
        processor_a.initialize_day(day=1)

        state_b = BankDealerState(bank_id="BankB", reserves=50000)
        state_b.add_deposit_cohort(day=0, origin="payment", amount=40000, rate=Decimal("0.015"))
        processor_b = TicketProcessor(state_b, cb_params, pricing_params)
        processor_b.initialize_day(day=1)

        initial_total = state_a.reserves + state_b.reserves

        process_inter_bank_payment(
            payer_processor=processor_a,
            payee_processor=processor_b,
            payer_client_id="client_alice",
            payee_client_id="client_bob",
            amount=25000,
        )

        final_total = state_a.reserves + state_b.reserves

        assert final_total == initial_total

    def test_inter_bank_payment_fails_insufficient_deposits(self):
        """Inter-bank payment fails if payer has insufficient deposits."""
        cb_params, pricing_params = create_standard_params()

        state_a = BankDealerState(bank_id="BankA", reserves=100000)
        state_a.add_deposit_cohort(day=0, origin="payment", amount=10000, rate=Decimal("0.015"))
        processor_a = TicketProcessor(state_a, cb_params, pricing_params)
        processor_a.initialize_day(day=1)

        state_b = BankDealerState(bank_id="BankB", reserves=50000)
        processor_b = TicketProcessor(state_b, cb_params, pricing_params)
        processor_b.initialize_day(day=1)

        settlement = process_inter_bank_payment(
            payer_processor=processor_a,
            payee_processor=processor_b,
            payer_client_id="client_alice",
            payee_client_id="client_bob",
            amount=50000,  # More than Alice's deposits
        )

        assert not settlement.success
        assert not settlement.payer_result.success


class TestIntraBankPayment:
    """Tests for intra-bank payment processing."""

    def test_intra_bank_payment_no_reserve_change(self):
        """Intra-bank payment doesn't change reserves."""
        processor = create_processor(reserves=100000, deposits=80000)
        processor.initialize_day(day=1)

        initial_reserves = processor.state.reserves

        wd_result, cr_result = process_intra_bank_payment(
            processor=processor,
            payer_client_id="client_alice",
            payee_client_id="client_bob",
            amount=20000,
        )

        assert wd_result.success
        assert cr_result.success
        assert processor.state.reserves == initial_reserves

    def test_intra_bank_payment_deposits_unchanged_total(self):
        """Intra-bank payment: total deposits unchanged (just internal transfer)."""
        processor = create_processor(reserves=100000, deposits=80000)
        processor.initialize_day(day=1)

        initial_deposits = processor.state.total_deposits

        process_intra_bank_payment(
            processor=processor,
            payer_client_id="client_alice",
            payee_client_id="client_bob",
            amount=20000,
        )

        # Total deposits unchanged (withdrawal + credit cancel out)
        assert processor.state.total_deposits == initial_deposits


class TestQuoteEvolution:
    """Tests for quote evolution as tickets are processed."""

    def test_quotes_change_with_tickets(self):
        """Quotes evolve as tickets change state."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        # Process multiple tickets
        for i in range(3):
            ticket = Ticket(
                id=f"T00{i}",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=10000,
                client_id=f"client_{i}",
                created_day=1,
                counterparty_bank_id="OtherBank",
            )
            processor.process_ticket(ticket)

        # Quotes should have evolved
        assert len(processor.state.quote_history) > 1

    def test_inflow_lowers_rates(self):
        """Reserve inflows tend to lower rates."""
        processor = create_processor(reserves=100000, deposits=50000)
        processor.initialize_day(day=1)

        initial_quote = processor.state.current_quote

        # Large inflow
        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=50000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )
        result = processor.process_ticket(ticket)

        # With more reserves, bank should be willing to pay less on deposits
        # (lower deposit rate)
        assert result.new_quote.deposit_rate <= initial_quote.deposit_rate

    def test_outflow_raises_rates(self):
        """Reserve outflows tend to raise rates."""
        processor = create_processor(reserves=100000, deposits=80000)
        processor.initialize_day(day=1)

        initial_quote = processor.state.current_quote

        # Large outflow
        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.WITHDRAWAL,
            amount=50000,
            client_id="client_alice",
            created_day=1,
            counterparty_bank_id="BankB",
        )
        result = processor.process_ticket(ticket)

        # With less reserves, bank should be willing to pay more on deposits
        # (higher deposit rate)
        assert result.new_quote.deposit_rate >= initial_quote.deposit_rate
