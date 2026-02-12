"""
BankDealer state management.

Implements the cohortized balance sheet from Section 5 of the specification.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from bilancio.banking.types import (
    DepositCohort,
    LoanCohort,
    CBBorrowingCohort,
    ScheduledLeg,
    Quote,
)


@dataclass
class CentralBankParams:
    """
    Central bank policy parameters.

    These define the corridor within which the BankDealer operates.
    """
    # 2-day reserve remuneration rate (floor)
    reserve_remuneration_rate: Decimal = Decimal("0.01")  # i_R^(2)

    # 2-day CB borrowing rate (ceiling)
    cb_borrowing_rate: Decimal = Decimal("0.03")  # i_B

    @property
    def corridor_width(self) -> Decimal:
        """Ω^(2) = ceiling - floor."""
        return self.cb_borrowing_rate - self.reserve_remuneration_rate


@dataclass
class BankDealerState:
    """
    Complete state of a BankDealer at any point in time.

    This is a cohortized balance sheet where every line is tagged by:
    - Issuance date (born at t-a)
    - Next cash-flow/maturity date
    - Means of settlement (Reserve vs Deposit)
    - Counterparty

    Balance sheet identity:
    Assets = Liabilities + Equity
    (Reserves + Loans) = (Deposits + CB Borrowing) + Equity
    """
    bank_id: str
    current_day: int = 0

    # === Assets ===

    # Reserves held at Central Bank (R_t)
    # Can go negative intraday (interpreted as CB overdraft)
    reserves: int = 0

    # Loans by cohort, keyed by issuance_day
    # Each cohort matures at issuance_day + 10
    loan_cohorts: Dict[int, LoanCohort] = field(default_factory=dict)

    # === Liabilities ===

    # Deposits by cohort, keyed by (issuance_day, origin)
    # origin is "payment" or "loan"
    # Deposits are ON DEMAND - no fixed maturity
    deposit_cohorts: Dict[tuple[int, str], DepositCohort] = field(default_factory=dict)

    # CB borrowing by cohort, keyed by issuance_day
    # Each cohort matures at issuance_day + 2
    cb_borrowing_cohorts: Dict[int, CBBorrowingCohort] = field(default_factory=dict)

    # === Intraday tracking ===

    # Withdrawal forecast for current day (from CMRE/simulator)
    withdrawal_forecast: int = 0  # W^c_t

    # Cumulative realized withdrawals within current day
    withdrawals_realized: int = 0  # W^real_t

    # Ticket counter for current day
    ticket_count: int = 0

    # Current quotes (updated after each ticket)
    current_quote: Optional[Quote] = None

    # === History (for diagnostics) ===
    quote_history: List[Quote] = field(default_factory=list)

    # =========================================================================
    # Balance Sheet Properties
    # =========================================================================

    @property
    def total_loans(self) -> int:
        """L_t: Total outstanding loans."""
        return sum(c.principal for c in self.loan_cohorts.values())

    @property
    def total_deposits(self) -> int:
        """D_t: Total deposits (principal + accrued interest)."""
        return sum(c.total_balance for c in self.deposit_cohorts.values())

    @property
    def deposits_payment_origin(self) -> int:
        """D^pay_t: Payment-origin deposits."""
        return sum(
            c.total_balance
            for k, c in self.deposit_cohorts.items()
            if k[1] == "payment"
        )

    @property
    def deposits_loan_origin(self) -> int:
        """D^loan_t: Loan-origin deposits."""
        return sum(
            c.total_balance
            for k, c in self.deposit_cohorts.items()
            if k[1] == "loan"
        )

    @property
    def total_cb_borrowing(self) -> int:
        """B_t: Total CB borrowing outstanding."""
        return sum(c.principal for c in self.cb_borrowing_cohorts.values())

    @property
    def total_assets(self) -> int:
        """Total assets = Reserves + Loans."""
        return self.reserves + self.total_loans

    @property
    def total_liabilities(self) -> int:
        """Total liabilities = Deposits + CB Borrowing."""
        return self.total_deposits + self.total_cb_borrowing

    @property
    def equity(self) -> int:
        """E_t: Book equity (balancing item)."""
        return self.total_assets - self.total_liabilities

    @property
    def withdrawals_remaining(self) -> int:
        """W^rem_t: Remaining expected withdrawals for today."""
        return max(0, self.withdrawal_forecast - self.withdrawals_realized)

    # =========================================================================
    # Scheduled Legs (for reserve projection)
    # =========================================================================

    def get_scheduled_legs(self, from_day: int, to_day: int) -> List[ScheduledLeg]:
        """
        Get all scheduled reserve legs in the range [from_day, to_day].

        This includes:
        - CB borrowing repayments (t+2 from issuance)
        - Loan repayments (t+10 from issuance)

        Does NOT include:
        - On-demand deposit withdrawals (not scheduled)
        - Deposit interest (deposit-only, no R movement)
        """
        legs = []

        # CB borrowing repayments
        for issuance_day, cohort in self.cb_borrowing_cohorts.items():
            if from_day <= cohort.maturity_day <= to_day:
                legs.append(ScheduledLeg(
                    day=cohort.maturity_day,
                    amount=-cohort.repayment_amount,  # Outflow
                    leg_type="cb_repay",
                    source_cohort=f"CB_borrow_{issuance_day}",
                ))

        # Loan repayments
        for issuance_day, loan_cohort in self.loan_cohorts.items():
            if from_day <= loan_cohort.maturity_day <= to_day:
                legs.append(ScheduledLeg(
                    day=loan_cohort.maturity_day,
                    amount=loan_cohort.repayment_amount,  # Inflow
                    leg_type="loan_repay",
                    source_cohort=f"Loan_{issuance_day}",
                ))

        return sorted(legs, key=lambda x: x.day)

    def get_deposit_interest_legs(self, from_day: int, to_day: int) -> List[ScheduledLeg]:
        """
        Get scheduled deposit interest legs (deposit-only, no R movement).

        Interest is credited every 2 days for each cohort still outstanding.
        """
        legs = []

        for cohort_key, cohort in self.deposit_cohorts.items():
            if cohort.principal <= 0:
                continue

            # Find interest days in range
            next_int_day = cohort.next_interest_day()
            while next_int_day <= to_day:
                if next_int_day >= from_day:
                    interest = cohort.compute_interest()
                    legs.append(ScheduledLeg(
                        day=next_int_day,
                        amount=interest,  # Deposit credit (not reserve)
                        leg_type="deposit_interest",
                        source_cohort=f"Dep_{cohort_key[0]}_{cohort_key[1]}",
                    ))
                next_int_day += 2

        return sorted(legs, key=lambda x: x.day)

    # =========================================================================
    # Cohort Operations
    # =========================================================================

    def add_deposit_cohort(
        self,
        day: int,
        origin: str,
        amount: int,
        rate: Decimal,
    ) -> tuple[int, str]:
        """
        Add a new deposit cohort or increase existing one.

        Returns the cohort key.
        """
        key = (day, origin)

        if key in self.deposit_cohorts:
            # Add to existing cohort (same day, same origin)
            self.deposit_cohorts[key].principal += amount
        else:
            # Create new cohort
            self.deposit_cohorts[key] = DepositCohort(
                issuance_day=day,
                origin=origin,
                principal=amount,
                stamped_rate=rate,
            )

        return key

    def add_loan_cohort(
        self,
        day: int,
        amount: int,
        rate: Decimal,
    ) -> int:
        """
        Add a new loan cohort or increase existing one.

        Returns the issuance day (key).
        """
        if day in self.loan_cohorts:
            # Add to existing cohort
            self.loan_cohorts[day].principal += amount
        else:
            # Create new cohort
            self.loan_cohorts[day] = LoanCohort(
                issuance_day=day,
                principal=amount,
                stamped_rate=rate,
            )

        return day

    def add_cb_borrowing(
        self,
        day: int,
        amount: int,
        cb_rate: Decimal,
    ) -> int:
        """
        Add CB borrowing (top-up at end of day or intraday overdraft conversion).

        Returns the issuance day (key).
        """
        if day in self.cb_borrowing_cohorts:
            # Add to existing cohort
            self.cb_borrowing_cohorts[day].principal += amount
        else:
            # Create new cohort
            self.cb_borrowing_cohorts[day] = CBBorrowingCohort(
                issuance_day=day,
                principal=amount,
                cb_rate=cb_rate,
            )

        return day

    def withdraw_from_deposits(
        self,
        amount: int,
        allocation_rule: str = "fifo",
    ) -> int:
        """
        Withdraw from deposits using specified allocation rule.

        Returns the actual amount withdrawn (may be less if insufficient).

        Allocation rules:
        - "fifo": First-in-first-out (oldest cohorts first)
        - "lifo": Last-in-first-out (newest cohorts first)
        - "proportional": Pro-rata from all cohorts
        """
        remaining = amount
        withdrawn = 0

        if allocation_rule == "fifo":
            # Sort cohorts by issuance day (oldest first)
            sorted_keys = sorted(
                self.deposit_cohorts.keys(),
                key=lambda k: k[0],
            )
        elif allocation_rule == "lifo":
            # Sort cohorts by issuance day (newest first)
            sorted_keys = sorted(
                self.deposit_cohorts.keys(),
                key=lambda k: -k[0],
            )
        else:
            # Default to FIFO
            sorted_keys = sorted(
                self.deposit_cohorts.keys(),
                key=lambda k: k[0],
            )

        for key in sorted_keys:
            if remaining <= 0:
                break

            cohort = self.deposit_cohorts[key]
            available = cohort.total_balance

            if available <= 0:
                continue

            take = min(remaining, available)

            # First take from accrued interest, then principal
            if cohort.interest_accrued >= take:
                cohort.interest_accrued -= take
            else:
                take_from_interest = cohort.interest_accrued
                take_from_principal = take - take_from_interest
                cohort.interest_accrued = 0
                cohort.principal -= take_from_principal

            withdrawn += take
            remaining -= take

        return withdrawn

    def process_cb_repayment(self, day: int) -> int:
        """
        Process CB borrowing repayment for cohorts maturing on this day.

        Returns the total amount repaid (outflow from reserves).
        """
        total_repaid = 0
        to_remove = []

        for issuance_day, cohort in self.cb_borrowing_cohorts.items():
            if cohort.maturity_day == day:
                total_repaid += cohort.repayment_amount
                to_remove.append(issuance_day)

        for key in to_remove:
            del self.cb_borrowing_cohorts[key]

        return total_repaid

    def process_loan_repayment(self, day: int) -> int:
        """
        Process loan repayments for cohorts maturing on this day.

        Returns the total amount received (inflow to reserves).
        """
        total_received = 0
        to_remove = []

        for issuance_day, cohort in self.loan_cohorts.items():
            if cohort.maturity_day == day:
                total_received += cohort.repayment_amount
                to_remove.append(issuance_day)

        for key in to_remove:
            del self.loan_cohorts[key]

        return total_received

    def credit_deposit_interest(self, day: int) -> int:
        """
        Credit deposit interest for all cohorts due on this day.

        Interest is credited every 2 days (at τ+2, τ+4, ...).
        This is a deposit-only leg - no reserve movement.

        Returns total interest credited.
        """
        total_interest = 0

        for cohort in self.deposit_cohorts.values():
            if cohort.principal <= 0:
                continue

            if cohort.next_interest_day() == day:
                interest = cohort.compute_interest()
                cohort.interest_accrued += interest
                cohort.last_interest_day = day
                total_interest += interest

        return total_interest

    # =========================================================================
    # Day Management
    # =========================================================================

    def advance_to_day(self, day: int) -> None:
        """Reset intraday counters for a new day."""
        self.current_day = day
        self.withdrawals_realized = 0
        self.ticket_count = 0
        self.current_quote = None

    def new_ticket_id(self) -> str:
        """Generate a unique ticket ID for this bank/day."""
        self.ticket_count += 1
        return f"{self.bank_id}_D{self.current_day}_T{self.ticket_count}"

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def balance_sheet_summary(self) -> Dict[str, object]:
        """Return a summary of the balance sheet for diagnostics."""
        return {
            "bank_id": self.bank_id,
            "day": self.current_day,
            "assets": {
                "reserves": self.reserves,
                "loans": self.total_loans,
                "total": self.total_assets,
            },
            "liabilities": {
                "deposits_payment": self.deposits_payment_origin,
                "deposits_loan": self.deposits_loan_origin,
                "deposits_total": self.total_deposits,
                "cb_borrowing": self.total_cb_borrowing,
                "total": self.total_liabilities,
            },
            "equity": self.equity,
            "cohort_counts": {
                "deposit_cohorts": len(self.deposit_cohorts),
                "loan_cohorts": len(self.loan_cohorts),
                "cb_borrowing_cohorts": len(self.cb_borrowing_cohorts),
            },
        }

    def __str__(self) -> str:
        """Pretty-print balance sheet."""
        summary = self.balance_sheet_summary()
        lines = [
            f"=== BankDealer {self.bank_id} (Day {self.current_day}) ===",
            "Assets:",
            f"  Reserves:      {self.reserves:>12,}",
            f"  Loans:         {self.total_loans:>12,}",
            f"  Total Assets:  {self.total_assets:>12,}",
            "Liabilities:",
            f"  Deposits (pay):{self.deposits_payment_origin:>12,}",
            f"  Deposits (ln): {self.deposits_loan_origin:>12,}",
            f"  CB Borrowing:  {self.total_cb_borrowing:>12,}",
            f"  Total Liab:    {self.total_liabilities:>12,}",
            f"Equity:          {self.equity:>12,}",
            f"---",
            f"Cohorts: {len(self.deposit_cohorts)} dep, {len(self.loan_cohorts)} loan, {len(self.cb_borrowing_cohorts)} CB",
        ]
        return "\n".join(lines)
