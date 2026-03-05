"""
Interbank Loan instrument.

An InterbankLoanContract is an asset of the lending bank and a liability of the borrowing bank.
It represents an overnight reserve loan from one bank to another, created by the interbank
call auction.

Key properties:
- Lending bank is the asset holder (creditor)
- Borrowing bank is the liability issuer
- Overnight maturity (issuance_day + 1)
- Repayment includes interest at the clearing rate
"""

from dataclasses import dataclass, field
from decimal import Decimal

from .base import Instrument, InstrumentKind


@dataclass
class InterbankLoanContract(Instrument):
    """
    Interbank loan between two commercial banks.

    The lending bank holds this as an asset (claim on borrower).
    The borrowing bank holds this as a liability.
    Repayment = principal * (1 + rate) at maturity (overnight = day+1).
    """

    rate: Decimal = field(default=Decimal("0"))
    issuance_day: int = 0

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.INTERBANK_LOAN
        # Set due_day for contracts_by_due_day index (overnight = day+1)
        if self.due_day is None:
            self.due_day = self.issuance_day + 1

    @property
    def maturity_day(self) -> int:
        """Interbank loans mature overnight (day + 1)."""
        return self.issuance_day + 1

    @property
    def repayment_amount(self) -> int:
        """Amount due at maturity: principal * (1 + rate)."""
        return int(self.amount * (1 + self.rate))

    @property
    def interest_amount(self) -> int:
        """Interest portion of repayment."""
        return self.repayment_amount - self.amount

    @property
    def principal(self) -> int:
        """Original loan principal (alias for amount)."""
        return self.amount

    def is_due(self, current_day: int) -> bool:
        """Check if this loan is due for repayment."""
        return current_day >= self.maturity_day
