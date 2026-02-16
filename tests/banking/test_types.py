"""
Unit tests for the banking kernel core types.

Tests verify the fundamental data structures:
- DepositCohort: On-demand deposits with interest accrual
- LoanCohort: 10-day bullet loans
- CBBorrowingCohort: 2-day CB borrowing
- Ticket: Client event records
- Quote: Bank's current quoted rates

References:
- "Banks-as-Dealers with deposits on demand" specification
"""

from decimal import Decimal

from bilancio.banking.types import (
    CBBorrowingCohort,
    DepositCohort,
    LoanCohort,
    Quote,
    ScheduledLeg,
    Ticket,
    TicketType,
)


class TestDepositCohort:
    """Tests for DepositCohort mechanics."""

    def test_initial_balance_equals_principal(self):
        """New deposit has balance = principal."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
        )

        assert cohort.principal == 10000
        assert cohort.interest_accrued == 0
        assert cohort.total_balance == 10000

    def test_total_balance_includes_interest(self):
        """Total balance = principal + accrued interest."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
            interest_accrued=100,
        )

        assert cohort.total_balance == 10100

    def test_cohort_key_format(self):
        """Cohort key is (issuance_day, origin)."""
        cohort = DepositCohort(
            issuance_day=5,
            origin="loan",
            principal=10000,
            stamped_rate=Decimal("0.02"),
        )

        assert cohort.cohort_key == (5, "loan")

    def test_next_interest_day_no_history(self):
        """First interest day is issuance + 2."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
        )

        assert cohort.next_interest_day() == 2

    def test_next_interest_day_after_credit(self):
        """After interest credit, next day is last_interest_day + 2."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
            last_interest_day=2,
        )

        assert cohort.next_interest_day() == 4

    def test_compute_interest_simple(self):
        """Interest = stamped_rate × principal (simple interest)."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),  # 1% per 2-day period
        )

        # Interest = 0.01 × 10000 = 100
        assert cohort.compute_interest() == 100

    def test_compute_interest_fractional_rate(self):
        """Interest computation handles fractional rates."""
        cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.015"),  # 1.5% per period
        )

        # Interest = 0.015 × 10000 = 150
        assert cohort.compute_interest() == 150

    def test_payment_vs_loan_origin(self):
        """Deposits track their origin (payment vs loan)."""
        payment_cohort = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
        )

        loan_cohort = DepositCohort(
            issuance_day=0,
            origin="loan",
            principal=10000,
            stamped_rate=Decimal("0.01"),
        )

        assert payment_cohort.origin == "payment"
        assert loan_cohort.origin == "loan"
        assert payment_cohort.cohort_key != loan_cohort.cohort_key


class TestLoanCohort:
    """Tests for LoanCohort mechanics."""

    def test_maturity_at_issuance_plus_10(self):
        """Loans mature at issuance_day + 10."""
        cohort = LoanCohort(
            issuance_day=5,
            principal=100000,
            stamped_rate=Decimal("0.03"),  # 3% loan rate
        )

        assert cohort.maturity_day == 15

    def test_repayment_amount_includes_interest(self):
        """Repayment = (1 + rate) × principal."""
        cohort = LoanCohort(
            issuance_day=0,
            principal=100000,
            stamped_rate=Decimal("0.03"),
        )

        # Repayment = (1 + 0.03) × 100000 = 103000
        assert cohort.repayment_amount == 103000

    def test_repayment_amount_zero_rate(self):
        """Zero rate means repayment = principal."""
        cohort = LoanCohort(
            issuance_day=0,
            principal=100000,
            stamped_rate=Decimal("0.00"),
        )

        assert cohort.repayment_amount == 100000

    def test_repayment_truncates_to_int(self):
        """Repayment amount is truncated to integer."""
        cohort = LoanCohort(
            issuance_day=0,
            principal=10001,
            stamped_rate=Decimal("0.03"),
        )

        # Repayment = (1.03) × 10001 = 10301.03 → 10301
        expected = int(Decimal("1.03") * 10001)
        assert cohort.repayment_amount == expected


class TestCBBorrowingCohort:
    """Tests for CBBorrowingCohort mechanics."""

    def test_maturity_at_issuance_plus_2(self):
        """CB borrowing matures at issuance_day + 2."""
        cohort = CBBorrowingCohort(
            issuance_day=5,
            principal=50000,
            cb_rate=Decimal("0.03"),
        )

        assert cohort.maturity_day == 7

    def test_repayment_amount_includes_cb_rate(self):
        """Repayment = (1 + cb_rate) × principal."""
        cohort = CBBorrowingCohort(
            issuance_day=0,
            principal=50000,
            cb_rate=Decimal("0.03"),
        )

        # Repayment = (1 + 0.03) × 50000 = 51500
        assert cohort.repayment_amount == 51500

    def test_cb_borrowing_maturity_different_from_loan(self):
        """CB borrowing (2 days) differs from loan (10 days)."""
        cb_cohort = CBBorrowingCohort(
            issuance_day=0,
            principal=50000,
            cb_rate=Decimal("0.03"),
        )

        loan_cohort = LoanCohort(
            issuance_day=0,
            principal=50000,
            stamped_rate=Decimal("0.03"),
        )

        assert cb_cohort.maturity_day == 2
        assert loan_cohort.maturity_day == 10


class TestTicket:
    """Tests for Ticket creation and attributes."""

    def test_payment_credit_ticket(self):
        """Payment credit ticket has correct type."""
        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.PAYMENT_CREDIT,
            amount=10000,
            client_id="client_alice",
            created_day=1,
        )

        assert ticket.ticket_type == TicketType.PAYMENT_CREDIT
        assert ticket.amount == 10000

    def test_loan_ticket(self):
        """Loan ticket has correct type."""
        ticket = Ticket(
            id="T002",
            ticket_type=TicketType.LOAN,
            amount=50000,
            client_id="client_bob",
            created_day=1,
        )

        assert ticket.ticket_type == TicketType.LOAN

    def test_withdrawal_ticket(self):
        """Withdrawal ticket has correct type."""
        ticket = Ticket(
            id="T003",
            ticket_type=TicketType.WITHDRAWAL,
            amount=15000,
            client_id="client_carol",
            created_day=1,
            counterparty_bank_id="BankB",
        )

        assert ticket.ticket_type == TicketType.WITHDRAWAL
        assert ticket.counterparty_bank_id == "BankB"

    def test_ticket_rate_stamping(self):
        """Tickets can be rate-stamped."""
        ticket = Ticket(
            id="T001",
            ticket_type=TicketType.LOAN,
            amount=50000,
            client_id="client_alice",
            created_day=1,
        )

        # Initially no rates
        assert ticket.stamped_deposit_rate is None
        assert ticket.stamped_loan_rate is None

        # Stamp rates
        ticket.stamped_deposit_rate = Decimal("0.01")
        ticket.stamped_loan_rate = Decimal("0.03")

        assert ticket.stamped_deposit_rate == Decimal("0.01")
        assert ticket.stamped_loan_rate == Decimal("0.03")


class TestScheduledLeg:
    """Tests for ScheduledLeg (reserve flow schedule)."""

    def test_positive_amount_is_inflow(self):
        """Positive amount represents reserve inflow."""
        leg = ScheduledLeg(
            day=10,
            amount=103000,
            leg_type="loan_repay",
            source_cohort="Loan_0",
        )

        assert leg.amount > 0
        assert leg.leg_type == "loan_repay"

    def test_negative_amount_is_outflow(self):
        """Negative amount represents reserve outflow."""
        leg = ScheduledLeg(
            day=2,
            amount=-51500,
            leg_type="cb_repay",
            source_cohort="CB_borrow_0",
        )

        assert leg.amount < 0
        assert leg.leg_type == "cb_repay"


class TestQuote:
    """Tests for Quote structure."""

    def test_quote_basic_structure(self):
        """Quote contains deposit and loan rates."""
        quote = Quote(
            deposit_rate=Decimal("0.015"),
            loan_rate=Decimal("0.025"),
            day=1,
            ticket_number=0,
        )

        assert quote.deposit_rate == Decimal("0.015")
        assert quote.loan_rate == Decimal("0.025")
        assert quote.day == 1

    def test_quote_diagnostic_fields(self):
        """Quote can include diagnostic info."""
        quote = Quote(
            deposit_rate=Decimal("0.015"),
            loan_rate=Decimal("0.025"),
            day=1,
            ticket_number=5,
            inventory=10000,
            cash_tightness=Decimal("0.5"),
            risk_index=Decimal("0.3"),
            midline=Decimal("0.02"),
        )

        assert quote.inventory == 10000
        assert quote.cash_tightness == Decimal("0.5")
        assert quote.risk_index == Decimal("0.3")
        assert quote.midline == Decimal("0.02")

    def test_spread_computation(self):
        """Spread = loan_rate - deposit_rate."""
        quote = Quote(
            deposit_rate=Decimal("0.015"),
            loan_rate=Decimal("0.025"),
            day=1,
        )

        spread = quote.loan_rate - quote.deposit_rate
        assert spread == Decimal("0.01")


class TestTicketTypeEnum:
    """Tests for TicketType enumeration."""

    def test_all_ticket_types_defined(self):
        """All expected ticket types exist."""
        assert TicketType.PAYMENT_CREDIT.value == "payment_credit"
        assert TicketType.LOAN.value == "loan"
        assert TicketType.WITHDRAWAL.value == "withdrawal"

    def test_ticket_types_are_distinct(self):
        """Ticket types are distinct values."""
        types = [TicketType.PAYMENT_CREDIT, TicketType.LOAN, TicketType.WITHDRAWAL]
        values = [t.value for t in types]
        assert len(values) == len(set(values))
