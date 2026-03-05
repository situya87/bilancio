"""
Unit tests for the BankDealer day runner.

Tests verify the intraday structure:
- Part A: Open (carry-over from yesterday)
- Part B: Scheduled settlements (CB repay, loan repay)
- Part C: Ticket-by-ticket client flow
- Part D: Intraday scheduled items (deposit interest)
- Part E: Pre-close CB top-up

References:
- "Banks-as-Dealers with deposits on demand" specification Section 6
"""

from decimal import Decimal

from bilancio.banking.day_runner import DayRunner, MultiBankDayRunner
from bilancio.banking.state import BankDealerState
from bilancio.banking.ticket_processor import TicketProcessor
from bilancio.banking.types import Ticket, TicketType
from tests.banking.conftest import create_standard_params


def create_day_runner(
    reserves: int = 100000,
    deposits: int = 50000,
) -> DayRunner:
    """Create a DayRunner with initial state."""
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

    processor = TicketProcessor(
        state=state,
        cb_params=cb_params,
        pricing_params=pricing_params,
    )

    return DayRunner(processor=processor)


class TestDayRunnerBasic:
    """Basic tests for DayRunner functionality."""

    def test_run_empty_day(self):
        """Running day with no tickets works."""
        runner = create_day_runner()

        result = runner.run_day(
            day=1,
            tickets=[],
            withdrawal_forecast=0,
        )

        assert result.day == 1
        assert result.tickets_processed == 0
        assert result.bank_id == "TestBank"

    def test_day_result_structure(self):
        """DayResult contains all expected fields."""
        runner = create_day_runner()

        result = runner.run_day(day=1, tickets=[])

        assert hasattr(result, "opening_reserves")
        assert hasattr(result, "closing_reserves")
        assert hasattr(result, "opening_deposits")
        assert hasattr(result, "closing_deposits")
        assert hasattr(result, "opening_quote")
        assert hasattr(result, "closing_quote")
        assert hasattr(result, "events")


class TestPartAOpen:
    """Tests for Part A: Day opening."""

    def test_day_advances_correctly(self):
        """Day advances to specified day number."""
        runner = create_day_runner()

        result = runner.run_day(day=5, tickets=[])

        assert runner.processor.state.current_day == 5
        assert result.day == 5

    def test_withdrawal_forecast_set(self):
        """Withdrawal forecast is set correctly."""
        runner = create_day_runner()

        runner.run_day(day=1, tickets=[], withdrawal_forecast=25000)

        assert runner.processor.state.withdrawal_forecast == 25000

    def test_opening_quote_computed(self):
        """Opening quote is computed on day open."""
        runner = create_day_runner()

        result = runner.run_day(day=1, tickets=[])

        assert result.opening_quote is not None
        assert result.opening_quote.day == 1

    def test_opening_state_captured(self):
        """Opening state is recorded in result."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        result = runner.run_day(day=1, tickets=[])

        assert result.opening_reserves == 100000
        assert result.opening_deposits == 50000


class TestPartBScheduledSettlements:
    """Tests for Part B: Scheduled settlements."""

    def test_loan_repayment_processed(self):
        """Loan repayments are processed on maturity day."""
        runner = create_day_runner(reserves=100000)
        state = runner.processor.state

        # Add loan that matures on day 10
        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        # Run day 10 (loan maturity)
        result = runner.run_day(day=10, tickets=[])

        # Loan repayment = 51500, should increase reserves
        assert result.loan_repayments == 51500
        assert state.total_loans == 0

    def test_cb_repayment_processed(self):
        """CB repayments are processed on maturity day."""
        runner = create_day_runner(reserves=100000)
        state = runner.processor.state

        # Add CB borrowing that matures on day 2
        state.add_cb_borrowing(day=0, amount=30000, cb_rate=Decimal("0.03"))

        # Run day 2 (CB maturity)
        result = runner.run_day(day=2, tickets=[])

        # CB repayment = 30900, should decrease reserves
        assert result.cb_repayment == 30900
        assert state.total_cb_borrowing == 0

    def test_multiple_settlements_same_day(self):
        """Multiple settlements can occur on same day."""
        runner = create_day_runner(reserves=200000)
        state = runner.processor.state

        # Add CB borrowing that matures on day 10
        state.add_cb_borrowing(day=8, amount=20000, cb_rate=Decimal("0.03"))

        # Add loan that matures on day 10
        state.add_loan_cohort(day=0, amount=50000, rate=Decimal("0.03"))

        result = runner.run_day(day=10, tickets=[])

        assert result.cb_repayment == 20600  # 20000 × 1.03
        assert result.loan_repayments == 51500  # 50000 × 1.03


class TestPartCTicketProcessing:
    """Tests for Part C: Ticket processing."""

    def test_tickets_processed_sequentially(self):
        """All tickets are processed."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        tickets = [
            Ticket(
                id=f"T{i}",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=1000,
                client_id=f"client_{i}",
                created_day=1,
                counterparty_bank_id="BankB",
            )
            for i in range(5)
        ]

        result = runner.run_day(day=1, tickets=tickets)

        assert result.tickets_processed == 5
        assert len(result.ticket_results) == 5

    def test_mixed_ticket_types(self):
        """Different ticket types are handled correctly."""
        runner = create_day_runner(reserves=100000, deposits=80000)

        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=10000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
            Ticket(
                id="T2",
                ticket_type=TicketType.LOAN,
                amount=20000,
                client_id="client_b",
                created_day=1,
            ),
            Ticket(
                id="T3",
                ticket_type=TicketType.WITHDRAWAL,
                amount=15000,
                client_id="client_c",
                created_day=1,
                counterparty_bank_id="BankC",
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        assert result.tickets_processed == 3
        assert all(r.success for r in result.ticket_results)


class TestPartDDepositInterest:
    """Tests for Part D: Deposit interest crediting."""

    def test_interest_credited_on_schedule(self):
        """Interest is credited on interest days."""
        runner = create_day_runner()
        state = runner.processor.state

        # Add deposit on day 0 with 1% rate
        state.deposit_cohorts.clear()
        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=100000,
            rate=Decimal("0.01"),
        )

        # Run day 2 (first interest day)
        result = runner.run_day(day=2, tickets=[])

        # Interest = 1% × 100000 = 1000
        assert result.deposit_interest_credited == 1000

    def test_no_interest_before_schedule(self):
        """No interest before next interest day."""
        runner = create_day_runner()
        state = runner.processor.state

        state.deposit_cohorts.clear()
        state.add_deposit_cohort(
            day=0,
            origin="payment",
            amount=100000,
            rate=Decimal("0.01"),
        )

        # Run day 1 (interest due on day 2)
        result = runner.run_day(day=1, tickets=[])

        assert result.deposit_interest_credited == 0


class TestPartECBTopUp:
    """Tests for Part E: CB top-up (overdraft conversion)."""

    def test_overdraft_converted_to_cb_borrowing(self):
        """Negative reserves are converted to CB borrowing."""
        runner = create_day_runner(reserves=30000, deposits=100000)

        # Withdrawal that causes overdraft
        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.WITHDRAWAL,
                amount=50000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        # Overdraft = 20000 (30000 - 50000 = -20000)
        # Should be converted to CB borrowing
        assert result.cb_topup_amount == 20000
        assert runner.processor.state.reserves == 0
        assert runner.processor.state.total_cb_borrowing == 20000

    def test_no_topup_if_positive_reserves(self):
        """No CB top-up if reserves stay positive."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.WITHDRAWAL,
                amount=20000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        assert result.cb_topup_amount == 0
        assert runner.processor.state.total_cb_borrowing == 0


class TestEventLogging:
    """Tests for event logging during day execution."""

    def test_events_logged(self):
        """Events are logged for each part."""
        runner = create_day_runner()

        result = runner.run_day(day=1, tickets=[])

        # Should have at least the open event
        assert len(result.events) >= 1
        assert any(e.part == "A" for e in result.events)

    def test_ticket_events_logged(self):
        """Ticket processing events are logged."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.LOAN,
                amount=10000,
                client_id="client_a",
                created_day=1,
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        ticket_events = [e for e in result.events if e.part == "C"]
        assert len(ticket_events) == 1

    def test_cb_topup_event_logged(self):
        """CB top-up event is logged."""
        runner = create_day_runner(reserves=10000, deposits=100000)

        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.WITHDRAWAL,
                amount=30000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        topup_events = [e for e in result.events if e.part == "E"]
        assert len(topup_events) == 1


class TestClosingState:
    """Tests for closing state capture."""

    def test_closing_state_captured(self):
        """Closing state is captured in result."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        tickets = [
            Ticket(
                id="T1",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=20000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
        ]

        result = runner.run_day(day=1, tickets=tickets)

        assert result.closing_reserves == 120000  # 100000 + 20000
        assert result.closing_deposits == 70000  # 50000 + 20000

    def test_closing_quote_captured(self):
        """Closing quote is captured in result."""
        runner = create_day_runner()

        result = runner.run_day(day=1, tickets=[])

        assert result.closing_quote is not None


class TestMultiBankDayRunner:
    """Tests for multi-bank day coordination."""

    def test_multi_bank_run_day(self):
        """Can run day for multiple banks."""
        cb_params, pricing_params = create_standard_params()

        # Create two banks
        runners = {}
        for bank_id in ["BankA", "BankB"]:
            state = BankDealerState(bank_id=bank_id, reserves=100000)
            state.add_deposit_cohort(day=0, origin="payment", amount=50000, rate=Decimal("0.015"))
            processor = TicketProcessor(state, cb_params, pricing_params)
            runners[bank_id] = DayRunner(processor=processor)

        multi_runner = MultiBankDayRunner(runners=runners)

        results = multi_runner.run_day(
            day=1,
            tickets_by_bank={"BankA": [], "BankB": []},
        )

        assert "BankA" in results
        assert "BankB" in results
        assert results["BankA"].day == 1
        assert results["BankB"].day == 1

    def test_multi_bank_system_state(self):
        """Can get aggregate system state."""
        cb_params, pricing_params = create_standard_params()

        runners = {}
        for bank_id, reserves in [("BankA", 100000), ("BankB", 80000)]:
            state = BankDealerState(bank_id=bank_id, reserves=reserves)
            state.add_deposit_cohort(day=0, origin="payment", amount=50000, rate=Decimal("0.015"))
            processor = TicketProcessor(state, cb_params, pricing_params)
            runners[bank_id] = DayRunner(processor=processor)

        multi_runner = MultiBankDayRunner(runners=runners)
        multi_runner.run_day(day=1, tickets_by_bank={})

        system_state = multi_runner.get_system_state()

        assert system_state["total_reserves"] == 180000
        assert system_state["total_deposits"] == 100000
        assert "BankA" in system_state["banks"]
        assert "BankB" in system_state["banks"]


class TestMultiDaySequence:
    """Tests for running multiple days in sequence."""

    def test_multi_day_sequence(self):
        """State carries over between days."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        # Day 1: Add some deposits
        tickets_d1 = [
            Ticket(
                id="T1",
                ticket_type=TicketType.PAYMENT_CREDIT,
                amount=20000,
                client_id="client_a",
                created_day=1,
                counterparty_bank_id="BankB",
            ),
        ]
        runner.run_day(day=1, tickets=tickets_d1)

        # Day 2: State should have the extra deposits
        result_d2 = runner.run_day(day=2, tickets=[])

        assert result_d2.opening_reserves == 120000
        assert result_d2.opening_deposits == 70000

    def test_loan_lifecycle(self):
        """Loan lifecycle: issuance to repayment."""
        runner = create_day_runner(reserves=100000, deposits=50000)

        # Day 0: Issue loan
        tickets_d0 = [
            Ticket(
                id="L1",
                ticket_type=TicketType.LOAN,
                amount=30000,
                client_id="borrower",
                created_day=0,
            ),
        ]
        runner.run_day(day=0, tickets=tickets_d0)

        state = runner.processor.state
        assert state.total_loans == 30000

        # Run through to day 10 (loan maturity)
        for day in range(1, 10):
            runner.run_day(day=day, tickets=[])

        result_d10 = runner.run_day(day=10, tickets=[])

        # Loan should be repaid (30000 × 1.03 = 30900)
        assert result_d10.loan_repayments > 0
        assert state.total_loans == 0


class TestWithdrawalForecastCallback:
    """Tests for withdrawal forecast callback."""

    def test_forecast_callback(self):
        """Withdrawal forecast can come from callback."""
        runner = create_day_runner()

        # Set up callback
        forecast_values = {1: 10000, 2: 15000, 3: 8000}
        runner.withdrawal_forecast_fn = lambda day: forecast_values.get(day, 0)

        runner.run_day(day=1, tickets=[])
        assert runner.processor.state.withdrawal_forecast == 10000

        runner.run_day(day=2, tickets=[])
        assert runner.processor.state.withdrawal_forecast == 15000

    def test_explicit_forecast_overrides_callback(self):
        """Explicit forecast parameter overrides callback."""
        runner = create_day_runner()
        runner.withdrawal_forecast_fn = lambda day: 99999

        runner.run_day(day=1, tickets=[], withdrawal_forecast=5000)

        assert runner.processor.state.withdrawal_forecast == 5000
