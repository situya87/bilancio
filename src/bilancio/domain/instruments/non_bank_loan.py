"""
Non-Bank Loan instrument.

A NonBankLoan is an asset of a non-bank lender and a liability of a borrower.
Unlike CB loans which create new reserves, these transfer existing cash.

Key properties:
- Issued by any agent (as liability issuer / borrower)
- NonBankLender is the asset holder (lender)
- Configurable maturity (default 2 days)
- Repayment includes interest at the loan rate
"""

from dataclasses import dataclass, field
from decimal import Decimal

from .base import Instrument, InstrumentKind


@dataclass
class NonBankLoan(Instrument):
    """
    Loan from a non-bank lender to a borrower.

    The lender holds this as an asset (claim on borrower).
    The borrower holds this as a liability.
    Repayment = principal × (1 + rate) at maturity.
    """
    rate: Decimal = field(default=Decimal("0.05"))
    issuance_day: int = 0
    maturity_days: int = 2

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.NON_BANK_LOAN

    @property
    def maturity_day(self) -> int:
        """Loan matures at issuance_day + maturity_days."""
        return self.issuance_day + self.maturity_days

    @property
    def repayment_amount(self) -> int:
        """Amount due at maturity: principal × (1 + rate)."""
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
