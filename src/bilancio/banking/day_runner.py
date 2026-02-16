"""
Day runner for Banks-as-Dealers.

Implements the intraday structure from Section 6 of the specification:
- Part A: Open (carry-over from yesterday)
- Part B: Scheduled settlements (CB repay, loan repay)
- Part C: Ticket-by-ticket client flow
- Part D: Intraday scheduled items (deposit interest)
- Part E: Pre-close CB top-up
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bilancio.banking.ticket_processor import TicketProcessor, TicketResult
from bilancio.banking.types import Quote, Ticket


@dataclass
class DayEvent:
    """Record of an event during the day."""

    day: int
    part: str  # "A", "B", "C", "D", "E"
    event_type: str
    description: str
    amount: int = 0
    details: dict[str, Any] | None = None


@dataclass
class DayResult:
    """Result of running a complete day."""

    day: int
    bank_id: str

    # Opening state
    opening_reserves: int = 0
    opening_deposits: int = 0
    opening_loans: int = 0

    # Closing state
    closing_reserves: int = 0
    closing_deposits: int = 0
    closing_loans: int = 0
    closing_cb_borrowing: int = 0

    # Part B: Scheduled settlements
    cb_repayment: int = 0
    loan_repayments: int = 0

    # Part C: Tickets processed
    tickets_processed: int = 0
    ticket_results: list[TicketResult] = field(default_factory=list)

    # Part D: Interest
    deposit_interest_credited: int = 0

    # Part E: CB top-up
    cb_topup_amount: int = 0

    # Quotes
    opening_quote: Quote | None = None
    closing_quote: Quote | None = None

    # Event log
    events: list[DayEvent] = field(default_factory=list)


@dataclass
class DayRunner:
    """
    Runs a complete day for a BankDealer.

    Coordinates the Parts A-E sequence and maintains the event log.
    """

    processor: TicketProcessor

    # Optional callback for withdrawal forecast
    # Returns expected withdrawals for the day
    withdrawal_forecast_fn: Callable[[int], int] | None = None

    def run_day(
        self,
        day: int,
        tickets: list[Ticket],
        withdrawal_forecast: int | None = None,
    ) -> DayResult:
        """
        Run a complete day sequence.

        Args:
            day: The day number
            tickets: List of client tickets to process (Part C)
            withdrawal_forecast: Expected withdrawals for today
                               (if None, uses forecast_fn or 0)

        Returns:
            DayResult with full day summary
        """
        state = self.processor.state

        result = DayResult(day=day, bank_id=state.bank_id)

        # Determine withdrawal forecast
        if withdrawal_forecast is None:
            if self.withdrawal_forecast_fn is not None:
                withdrawal_forecast = self.withdrawal_forecast_fn(day)
            else:
                withdrawal_forecast = 0

        # =====================================================================
        # Part A: Open
        # =====================================================================
        self._part_a_open(day, withdrawal_forecast, result)

        # =====================================================================
        # Part B: Scheduled settlements
        # =====================================================================
        self._part_b_scheduled(day, result)

        # =====================================================================
        # Part C: Ticket-by-ticket flow
        # =====================================================================
        self._part_c_tickets(tickets, result)

        # =====================================================================
        # Part D: Intraday scheduled items
        # =====================================================================
        self._part_d_calendar(day, result)

        # =====================================================================
        # Part E: Pre-close CB top-up
        # =====================================================================
        self._part_e_topup(day, result)

        # Record closing state
        result.closing_reserves = state.reserves
        result.closing_deposits = state.total_deposits
        result.closing_loans = state.total_loans
        result.closing_cb_borrowing = state.total_cb_borrowing
        result.closing_quote = state.current_quote

        return result

    def _part_a_open(
        self,
        day: int,
        withdrawal_forecast: int,
        result: DayResult,
    ) -> None:
        """
        Part A: Open the day.

        - Advance state to new day
        - Set withdrawal forecast
        - Compute opening quote
        """
        state = self.processor.state

        # Record opening state (before advancing)
        if state.current_day == day - 1:
            result.opening_reserves = state.reserves
            result.opening_deposits = state.total_deposits
            result.opening_loans = state.total_loans

        # Initialize the day
        opening_quote = self.processor.initialize_day(day, withdrawal_forecast)
        result.opening_quote = opening_quote

        # Update opening state after initialization
        result.opening_reserves = state.reserves
        result.opening_deposits = state.total_deposits
        result.opening_loans = state.total_loans

        result.events.append(
            DayEvent(
                day=day,
                part="A",
                event_type="open",
                description=f"Day {day} opened. Withdrawal forecast: {withdrawal_forecast}",
                details={
                    "reserves": state.reserves,
                    "deposits": state.total_deposits,
                    "loans": state.total_loans,
                    "deposit_rate": str(opening_quote.deposit_rate),
                    "loan_rate": str(opening_quote.loan_rate),
                },
            )
        )

    def _part_b_scheduled(self, day: int, result: DayResult) -> None:
        """
        Part B: Process scheduled settlements.

        - CB borrowing repayments (cohorts maturing today)
        - Loan repayments (cohorts maturing today)
        """
        state = self.processor.state

        # CB repayments (outflow)
        cb_repaid = state.process_cb_repayment(day)
        if cb_repaid > 0:
            state.reserves -= cb_repaid
            result.cb_repayment = cb_repaid
            result.events.append(
                DayEvent(
                    day=day,
                    part="B",
                    event_type="cb_repay",
                    description=f"CB borrowing repayment: {cb_repaid}",
                    amount=-cb_repaid,
                )
            )

        # Loan repayments (inflow)
        loan_received = state.process_loan_repayment(day)
        if loan_received > 0:
            state.reserves += loan_received
            result.loan_repayments = loan_received
            result.events.append(
                DayEvent(
                    day=day,
                    part="B",
                    event_type="loan_repay",
                    description=f"Loan repayments received: {loan_received}",
                    amount=loan_received,
                )
            )

        # Refresh projection after scheduled settlements
        if cb_repaid > 0 or loan_received > 0:
            self.processor.refresh_projection()
            self.processor.refresh_quotes()

    def _part_c_tickets(
        self,
        tickets: list[Ticket],
        result: DayResult,
    ) -> None:
        """
        Part C: Process client tickets one by one.

        After each ticket:
        - Update projection
        - Refresh quotes
        """
        for ticket in tickets:
            ticket_result = self.processor.process_ticket(ticket)
            result.ticket_results.append(ticket_result)
            result.tickets_processed += 1

            status = "SUCCESS" if ticket_result.success else "FAILED"
            result.events.append(
                DayEvent(
                    day=result.day,
                    part="C",
                    event_type=f"ticket_{ticket.ticket_type.value}",
                    description=f"[{status}] {ticket_result.message}",
                    amount=ticket.amount,
                    details={
                        "ticket_id": ticket.id,
                        "client_id": ticket.client_id,
                        "success": ticket_result.success,
                        "reserve_delta": ticket_result.reserve_delta,
                        "deposit_delta": ticket_result.deposit_delta,
                    },
                )
            )

    def _part_d_calendar(self, day: int, result: DayResult) -> None:
        """
        Part D: Process intraday calendar items.

        - Credit deposit interest (deposit-only leg, no R movement)
        """
        state = self.processor.state

        interest_credited = state.credit_deposit_interest(day)
        if interest_credited > 0:
            result.deposit_interest_credited = interest_credited
            result.events.append(
                DayEvent(
                    day=day,
                    part="D",
                    event_type="deposit_interest",
                    description=f"Deposit interest credited: {interest_credited}",
                    amount=interest_credited,
                )
            )

    def _part_e_topup(self, day: int, result: DayResult) -> None:
        """
        Part E: Pre-close CB top-up.

        If reserves are negative (intraday overdraft), convert to CB borrowing.
        """
        state = self.processor.state
        cb_params = self.processor.cb_params

        if state.reserves < 0:
            # Negative reserves = intraday overdraft
            # Convert to formal CB borrowing
            shortfall = -state.reserves

            state.add_cb_borrowing(
                day=day,
                amount=shortfall,
                cb_rate=cb_params.cb_borrowing_rate,
            )

            # Reserves topped up to zero
            state.reserves = 0

            result.cb_topup_amount = shortfall
            result.events.append(
                DayEvent(
                    day=day,
                    part="E",
                    event_type="cb_topup",
                    description=f"CB top-up (overdraft conversion): {shortfall}",
                    amount=shortfall,
                    details={
                        "cb_rate": str(cb_params.cb_borrowing_rate),
                        "maturity_day": day + 2,
                    },
                )
            )

            # Refresh projection after CB borrowing
            self.processor.refresh_projection()
            self.processor.refresh_quotes()


# =============================================================================
# Multi-bank day coordination
# =============================================================================


@dataclass
class MultiBankDayRunner:
    """
    Coordinates a day across multiple BankDealers.

    Handles:
    - Synchronized day start
    - Inter-bank payment routing
    - End-of-day settlement
    """

    runners: dict[str, DayRunner]  # bank_id -> DayRunner

    def run_day(
        self,
        day: int,
        tickets_by_bank: dict[str, list[Ticket]],
        withdrawal_forecasts: dict[str, int] | None = None,
    ) -> dict[str, DayResult]:
        """
        Run a complete day for all banks.

        For simplicity, this runs each bank's Parts A-E sequentially.
        In a more sophisticated implementation, Part C tickets could
        interleave across banks.

        Args:
            day: The day number
            tickets_by_bank: Dict of bank_id -> list of tickets
            withdrawal_forecasts: Optional dict of bank_id -> forecast

        Returns:
            Dict of bank_id -> DayResult
        """
        results = {}

        for bank_id, runner in self.runners.items():
            tickets = tickets_by_bank.get(bank_id, [])
            forecast = withdrawal_forecasts.get(bank_id, 0) if withdrawal_forecasts else 0

            results[bank_id] = runner.run_day(
                day=day,
                tickets=tickets,
                withdrawal_forecast=forecast,
            )

        return results

    def get_system_state(self) -> dict[str, Any]:
        """Get aggregate state across all banks."""
        total_reserves = 0
        total_deposits = 0
        total_loans = 0
        total_cb_borrowing = 0

        bank_states = {}

        for bank_id, runner in self.runners.items():
            state = runner.processor.state
            total_reserves += state.reserves
            total_deposits += state.total_deposits
            total_loans += state.total_loans
            total_cb_borrowing += state.total_cb_borrowing

            bank_states[bank_id] = {
                "reserves": state.reserves,
                "deposits": state.total_deposits,
                "loans": state.total_loans,
                "cb_borrowing": state.total_cb_borrowing,
                "equity": state.equity,
            }

        return {
            "total_reserves": total_reserves,
            "total_deposits": total_deposits,
            "total_loans": total_loans,
            "total_cb_borrowing": total_cb_borrowing,
            "banks": bank_states,
        }
