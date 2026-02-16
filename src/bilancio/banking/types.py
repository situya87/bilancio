"""
Core types for the banking kernel.

Based on "Banks-as-Dealers with deposits on demand" specification.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class TicketType(Enum):
    """Types of client tickets processed by the bank."""

    # A client receives a payment from another bank's client
    # → Creates payment-origin deposit, reserves flow in via RTGS
    PAYMENT_CREDIT = "payment_credit"

    # A client takes out a loan
    # → Creates loan-origin deposit AND loan asset, no reserve movement
    LOAN = "loan"

    # A client withdraws/pays out to another bank's client
    # → Cancels deposits, reserves flow out via RTGS
    WITHDRAWAL = "withdrawal"


@dataclass
class Ticket:
    """
    A single client event processed by the bank.

    Each ticket is rate-stamped at the time of booking with the quotes
    in force (r_D, r_L), and later generates interest, withdrawals,
    and repayments according to timing rules.
    """

    id: str
    ticket_type: TicketType
    amount: int  # In minor units (e.g., cents)
    client_id: str  # Which client initiated this
    created_day: int

    # For inter-bank transfers: which bank is the counterparty?
    # None for intra-bank transfers or loans
    counterparty_bank_id: str | None = None

    # Rate stamps at time of booking (2-day rates)
    stamped_deposit_rate: Decimal | None = None
    stamped_loan_rate: Decimal | None = None

    # For withdrawals: which deposit cohort(s) to draw from
    # If None, use FIFO or bank's allocation rule
    source_cohort_key: tuple[int, str] | None = None


@dataclass
class DepositCohort:
    """
    Deposits created on a specific day, tracked for interest accrual.

    Deposits are payable ON DEMAND - principal leaves only when the
    depositor initiates a withdrawal/outgoing payment ticket.

    Interest is credited every 2 days (at τ+2, τ+4, ...) while the
    deposit remains outstanding, as a deposit-only leg (no R movement).
    """

    issuance_day: int
    origin: str  # "payment" or "loan"
    principal: int  # Remaining principal (decreases with withdrawals)
    stamped_rate: Decimal  # 2-day deposit rate at issuance

    # Accumulated interest credits (added to deposit liability)
    interest_accrued: int = 0

    # Track which interest periods have been credited
    last_interest_day: int | None = None

    @property
    def total_balance(self) -> int:
        """Principal plus accrued interest."""
        return self.principal + self.interest_accrued

    @property
    def cohort_key(self) -> tuple[int, str]:
        """Unique key for this cohort: (issuance_day, origin)."""
        return (self.issuance_day, self.origin)

    def next_interest_day(self) -> int:
        """Next day when interest should be credited."""
        if self.last_interest_day is None:
            return self.issuance_day + 2
        return self.last_interest_day + 2

    def compute_interest(self) -> int:
        """
        Compute interest for one 2-day period.
        Interest = stamped_rate × principal (simple interest per period).
        """
        # Interest on principal only (not compound)
        interest = int(self.stamped_rate * self.principal)
        return interest


@dataclass
class LoanCohort:
    """
    Loans created on a specific day.

    All loans repay at t+10 (single bullet repayment).
    Repayment = (1 + r_L) × principal, paid in reserves.
    """

    issuance_day: int
    principal: int
    stamped_rate: Decimal  # 2-day loan rate at issuance

    @property
    def maturity_day(self) -> int:
        """Loans mature at issuance + 10."""
        return self.issuance_day + 10

    @property
    def repayment_amount(self) -> int:
        """Amount due at maturity: (1 + r_L) × principal."""
        return int((1 + self.stamped_rate) * self.principal)


@dataclass
class CBBorrowingCohort:
    """
    Central bank borrowing created on a specific day.

    CB borrowing is for 2 days at rate i_B.
    Repayment at t+2: (1 + i_B) × principal.
    """

    issuance_day: int
    principal: int
    cb_rate: Decimal  # i_B at time of borrowing

    @property
    def maturity_day(self) -> int:
        """CB borrowing matures at issuance + 2."""
        return self.issuance_day + 2

    @property
    def repayment_amount(self) -> int:
        """Amount due at maturity: (1 + i_B) × principal."""
        return int((1 + self.cb_rate) * self.principal)


@dataclass
class ScheduledLeg:
    """
    A scheduled cash flow (reserve or deposit movement) at a future date.

    Used for building the 10-day reserve projection.
    """

    day: int
    amount: int  # Positive = inflow, negative = outflow
    leg_type: str  # "cb_repay", "cb_remuneration", "loan_repay", "deposit_interest"
    source_cohort: str | None = None  # Description of originating cohort


@dataclass
class ClientAccount:
    """
    A client's account at a bank.

    Tracks the client's deposits and loans at this specific bank.
    In a full system this would link to the client agent.
    """

    client_id: str
    bank_id: str

    # Client's view of their deposits (sum of all cohorts they own)
    # In practice, the bank tracks cohorts; this is a convenience view
    deposit_balance: int = 0

    # Client's outstanding loans from this bank
    loan_balance: int = 0


@dataclass
class Quote:
    """
    The bank's current quoted rates.

    These are 2-day effective rates on the funding plane.
    """

    deposit_rate: Decimal  # r_D - what the bank pays on deposits (bid)
    loan_rate: Decimal  # r_L - what the bank charges on loans (ask)
    day: int
    ticket_number: int = 0  # Which ticket these quotes are for

    # Diagnostic info
    inventory: int | None = None  # x = R_{t+2} - R^tar
    cash_tightness: Decimal | None = None  # L*
    risk_index: Decimal | None = None  # ρ
    midline: Decimal | None = None  # m^(2)_bank
