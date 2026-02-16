"""
Unit tests for BankDealerState (cohortized balance sheet).

Tests verify balance sheet mechanics:
- Balance sheet identity: Assets = Liabilities + Equity
- Cohort operations (add, withdraw, process repayments)
- Scheduled leg generation
- Deposit interest crediting

References:
- "Banks-as-Dealers with deposits on demand" specification Section 5
"""

from decimal import Decimal

from bilancio.banking.state import BankDealerState, CentralBankParams
from bilancio.banking.types import DepositCohort


class TestBankDealerStateInitialization:
    """Tests for initial state creation."""

    def test_empty_state(self):
        """New state has zero balances."""
        state = BankDealerState(bank_id="TestBank")

        assert state.reserves == 0
        assert state.total_loans == 0
        assert state.total_deposits == 0
        assert state.total_cb_borrowing == 0
        assert state.equity == 0

    def test_state_with_initial_reserves(self):
        """State can be initialized with reserves."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=100000,
        )

        assert state.reserves == 100000
        assert state.total_assets == 100000
        assert state.equity == 100000  # Assets - Liabilities

    def test_day_tracking(self):
        """State tracks current day."""
        state = BankDealerState(bank_id="TestBank", current_day=5)
        assert state.current_day == 5


class TestBalanceSheetIdentity:
    """Tests for Assets = Liabilities + Equity identity."""

    def test_identity_with_reserves_only(self):
        """Identity holds with only reserves."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=100000,
        )

        # Assets = 100000, Liabilities = 0, Equity = 100000
        assert state.total_assets == state.total_liabilities + state.equity
        assert state.equity == 100000

    def test_identity_with_deposits(self):
        """Identity holds with deposits."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=100000,
        )

        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=50000,
            rate=Decimal("0.01"),
        )

        # Assets = 100000, Liabilities = 50000, Equity = 50000
        assert state.total_assets == 100000
        assert state.total_liabilities == 50000
        assert state.equity == 50000
        assert state.total_assets == state.total_liabilities + state.equity

    def test_identity_with_loans(self):
        """Identity holds with loans and deposits."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=50000,
        )

        # Add loan (asset) and corresponding deposit (liability)
        state.add_loan_cohort(day=0, amount=100000, rate=Decimal("0.03"))
        state.add_deposit_cohort(
            day=0,
            origin="loan",
            amount=100000,
            rate=Decimal("0.01"),
        )

        # Assets = 50000 (R) + 100000 (L) = 150000
        # Liabilities = 100000 (D)
        # Equity = 50000
        assert state.total_assets == 150000
        assert state.total_liabilities == 100000
        assert state.equity == 50000
        assert state.total_assets == state.total_liabilities + state.equity

    def test_identity_with_cb_borrowing(self):
        """Identity holds with CB borrowing."""
        state = BankDealerState(
            bank_id="TestBank",
            reserves=100000,
        )

        state.add_cb_borrowing(day=0, amount=50000, cb_rate=Decimal("0.03"))

        # Assets = 100000, Liabilities = 50000 (CB), Equity = 50000
        assert state.total_assets == 100000
        assert state.total_liabilities == 50000
        assert state.equity == 50000
        assert state.total_assets == state.total_liabilities + state.equity


class TestDepositOperations:
    """Tests for deposit cohort operations."""

    def test_add_single_deposit_cohort(self):
        """Adding a deposit cohort increases deposits."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        key = state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=50000,
            rate=Decimal("0.01"),
        )

        assert key == (0, "payment")
        assert state.total_deposits == 50000
        assert state.deposits_payment_origin == 50000
        assert state.deposits_loan_origin == 0

    def test_add_multiple_deposit_cohorts(self):
        """Multiple cohorts sum correctly."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(day=0, origin="payment", amount=30000, rate=Decimal("0.01"))
        state.add_deposit_cohort(day=0, origin="loan", amount=20000, rate=Decimal("0.01"))
        state.add_deposit_cohort(day=1, origin="payment", amount=10000, rate=Decimal("0.012"))

        assert state.total_deposits == 60000
        assert state.deposits_payment_origin == 40000
        assert state.deposits_loan_origin == 20000

    def test_add_to_existing_cohort(self):
        """Adding to same (day, origin) increases existing cohort."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(day=0, origin="payment", amount=30000, rate=Decimal("0.01"))
        state.add_deposit_cohort(day=0, origin="payment", amount=20000, rate=Decimal("0.01"))

        # Should merge into one cohort
        assert len(state.deposit_cohorts) == 1
        assert state.total_deposits == 50000

    def test_withdraw_from_deposits_fifo(self):
        """Withdrawal uses FIFO allocation."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        # Add cohorts on different days
        state.add_deposit_cohort(day=0, origin="payment", amount=30000, rate=Decimal("0.01"))
        state.add_deposit_cohort(day=1, origin="payment", amount=20000, rate=Decimal("0.012"))

        # Withdraw 25000 - should take from day 0 first
        withdrawn = state.withdraw_from_deposits(25000, allocation_rule="fifo")

        assert withdrawn == 25000
        # Day 0 cohort should have 5000 left
        assert state.deposit_cohorts[(0, "payment")].principal == 5000
        # Day 1 cohort unchanged
        assert state.deposit_cohorts[(1, "payment")].principal == 20000

    def test_withdraw_spanning_cohorts(self):
        """Withdrawal can span multiple cohorts."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(day=0, origin="payment", amount=10000, rate=Decimal("0.01"))
        state.add_deposit_cohort(day=1, origin="payment", amount=20000, rate=Decimal("0.012"))

        # Withdraw 25000 - should deplete day 0 and take 15000 from day 1
        withdrawn = state.withdraw_from_deposits(25000, allocation_rule="fifo")

        assert withdrawn == 25000
        assert state.deposit_cohorts[(0, "payment")].principal == 0
        assert state.deposit_cohorts[(1, "payment")].principal == 5000

    def test_withdraw_more_than_available(self):
        """Withdrawal capped at available deposits."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(day=0, origin="payment", amount=10000, rate=Decimal("0.01"))

        # Try to withdraw more than available
        withdrawn = state.withdraw_from_deposits(50000, allocation_rule="fifo")

        assert withdrawn == 10000  # Only 10000 available
        assert state.total_deposits == 0

    def test_withdraw_from_interest_first(self):
        """Withdrawal takes from accrued interest before principal."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        # Add cohort with accrued interest
        state.deposit_cohorts[(0, "payment")] = DepositCohort(
            issuance_day=0,
            origin="payment",
            principal=10000,
            stamped_rate=Decimal("0.01"),
            interest_accrued=500,
        )

        # Withdraw 300 - should come from interest first
        withdrawn = state.withdraw_from_deposits(300, allocation_rule="fifo")

        assert withdrawn == 300
        assert state.deposit_cohorts[(0, "payment")].interest_accrued == 200
        assert state.deposit_cohorts[(0, "payment")].principal == 10000


class TestLoanOperations:
    """Tests for loan cohort operations."""

    def test_add_loan_cohort(self):
        """Adding loan cohort increases loans."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        key = state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        assert key == 0
        assert state.total_loans == 50000
        assert len(state.loan_cohorts) == 1

    def test_multiple_loan_cohorts(self):
        """Multiple loan cohorts sum correctly."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))
        state.add_loan_cohort(day=1, amount=30000, rate=Decimal("0.025"))

        assert state.total_loans == 80000
        assert len(state.loan_cohorts) == 2

    def test_process_loan_repayment(self):
        """Loan repayment removes cohort and returns amount."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        # Loan matures at day 0 + 10 = day 10
        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        # Repayment amount = (1 + 0.03) × 50000 = 51500
        received = state.process_loan_repayment(day=10)

        assert received == 51500
        assert state.total_loans == 0
        assert len(state.loan_cohorts) == 0

    def test_no_repayment_before_maturity(self):
        """No repayment if loan not yet mature."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        # Try day 5 - loan doesn't mature until day 10
        received = state.process_loan_repayment(day=5)

        assert received == 0
        assert state.total_loans == 50000


class TestCBBorrowingOperations:
    """Tests for CB borrowing cohort operations."""

    def test_add_cb_borrowing(self):
        """Adding CB borrowing increases liability."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        key = state.add_cb_borrowing(day=0, amount=50000, cb_rate=Decimal("0.03"))

        assert key == 0
        assert state.total_cb_borrowing == 50000
        assert len(state.cb_borrowing_cohorts) == 1

    def test_process_cb_repayment(self):
        """CB repayment removes cohort and returns amount."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        # CB borrowing matures at day 0 + 2 = day 2
        state.add_cb_borrowing(day=0, amount=50000, cb_rate=Decimal("0.03"))

        # Repayment = (1 + 0.03) × 50000 = 51500
        repaid = state.process_cb_repayment(day=2)

        assert repaid == 51500
        assert state.total_cb_borrowing == 0
        assert len(state.cb_borrowing_cohorts) == 0


class TestScheduledLegs:
    """Tests for scheduled leg generation."""

    def test_loan_repayment_leg(self):
        """Loan generates positive scheduled leg at maturity."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        legs = state.get_scheduled_legs(from_day=1, to_day=15)

        # Should have one leg at day 10
        assert len(legs) == 1
        assert legs[0].day == 10
        assert legs[0].amount == 51500  # (1 + 0.03) × 50000
        assert legs[0].leg_type == "loan_repay"

    def test_cb_repayment_leg(self):
        """CB borrowing generates negative scheduled leg at maturity."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_cb_borrowing(day=0, amount=50000, cb_rate=Decimal("0.03"))

        legs = state.get_scheduled_legs(from_day=1, to_day=5)

        # Should have one leg at day 2
        assert len(legs) == 1
        assert legs[0].day == 2
        assert legs[0].amount == -51500  # Outflow
        assert legs[0].leg_type == "cb_repay"

    def test_multiple_legs_sorted(self):
        """Multiple legs are sorted by day."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))  # Matures day 10
        state.add_cb_borrowing(day=0, amount=20000, cb_rate=Decimal("0.03"))  # Matures day 2
        state.add_loan_cohort(day=5, amount=30000, rate=Decimal("0.025"))  # Matures day 15

        legs = state.get_scheduled_legs(from_day=1, to_day=20)

        assert len(legs) == 3
        assert legs[0].day == 2  # CB repay
        assert legs[1].day == 10  # Loan 1
        assert legs[2].day == 15  # Loan 2


class TestDepositInterest:
    """Tests for deposit interest crediting."""

    def test_credit_interest_on_schedule(self):
        """Interest credited on day issuance + 2."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=10000,
            rate=Decimal("0.01"),
        )

        # Credit interest on day 2
        interest = state.credit_deposit_interest(day=2)

        # Interest = 0.01 × 10000 = 100
        assert interest == 100
        assert state.deposit_cohorts[(0, "payment")].interest_accrued == 100
        assert state.deposit_cohorts[(0, "payment")].last_interest_day == 2

    def test_no_interest_before_schedule(self):
        """No interest before next interest day."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=10000,
            rate=Decimal("0.01"),
        )

        # Try day 1 - interest due on day 2
        interest = state.credit_deposit_interest(day=1)

        assert interest == 0
        assert state.deposit_cohorts[(0, "payment")].interest_accrued == 0

    def test_multiple_interest_periods(self):
        """Interest credited every 2 days."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)

        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=10000,
            rate=Decimal("0.01"),
        )

        # Day 2: first interest
        state.credit_deposit_interest(day=2)
        assert state.deposit_cohorts[(0, "payment")].interest_accrued == 100

        # Day 4: second interest
        interest = state.credit_deposit_interest(day=4)
        assert interest == 100  # Still 1% of principal
        assert state.deposit_cohorts[(0, "payment")].interest_accrued == 200


class TestDayManagement:
    """Tests for day management functions."""

    def test_advance_to_day(self):
        """Advancing day resets intraday counters."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)
        state.withdrawals_realized = 5000
        state.ticket_count = 10

        state.advance_to_day(5)

        assert state.current_day == 5
        assert state.withdrawals_realized == 0
        assert state.ticket_count == 0
        assert state.current_quote is None

    def test_new_ticket_id(self):
        """Ticket IDs are unique and sequential."""
        state = BankDealerState(bank_id="TestBank", current_day=3)

        id1 = state.new_ticket_id()
        id2 = state.new_ticket_id()

        assert id1 == "TestBank_D3_T1"
        assert id2 == "TestBank_D3_T2"
        assert state.ticket_count == 2

    def test_withdrawals_remaining(self):
        """Remaining withdrawals = forecast - realized."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)
        state.withdrawal_forecast = 10000
        state.withdrawals_realized = 3000

        assert state.withdrawals_remaining == 7000

    def test_withdrawals_remaining_non_negative(self):
        """Remaining withdrawals cannot be negative."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)
        state.withdrawal_forecast = 5000
        state.withdrawals_realized = 8000  # More than forecast

        assert state.withdrawals_remaining == 0


class TestCentralBankParams:
    """Tests for CentralBankParams."""

    def test_corridor_width(self):
        """Corridor width = ceiling - floor."""
        params = CentralBankParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
        )

        assert params.corridor_width == Decimal("0.02")

    def test_default_rates(self):
        """Default CB params have 1% floor, 3% ceiling."""
        params = CentralBankParams()

        assert params.reserve_remuneration_rate == Decimal("0.01")
        assert params.cb_borrowing_rate == Decimal("0.03")


class TestBalanceSheetSummary:
    """Tests for balance sheet diagnostic output."""

    def test_summary_includes_all_items(self):
        """Summary includes all balance sheet items."""
        state = BankDealerState(bank_id="TestBank", reserves=100000)
        state.add_deposit_cohort(day=0, origin="payment", amount=50000, rate=Decimal("0.01"))
        state.add_loan_cohort(day=0, amount=30000, rate=Decimal("0.03"))

        summary = state.balance_sheet_summary()

        assert summary["bank_id"] == "TestBank"
        assert summary["assets"]["reserves"] == 100000
        assert summary["assets"]["loans"] == 30000
        assert summary["liabilities"]["deposits_total"] == 50000
        assert summary["equity"] == 80000  # 130000 - 50000
