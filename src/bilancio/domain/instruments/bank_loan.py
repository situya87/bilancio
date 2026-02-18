"""
Bank Loan instrument.

A BankLoan is an asset of a commercial bank and a liability of a borrower (trader).
Unlike CB loans, bank loans create deposits (money creation) rather than reserves.

Key properties:
- Bank is the asset holder (creditor)
- Borrower (Household/Firm) is the liability issuer
- Configurable maturity (default: maturity_days // 2)
- Repayment includes interest at the loan rate
"""

from dataclasses import dataclass, field
from decimal import Decimal

from .base import Instrument, InstrumentKind


@dataclass
class BankLoan(Instrument):
    """
    Loan from a commercial bank to a trader.

    The bank holds this as an asset (claim on borrower).
    The borrower holds this as a liability.
    Repayment = principal * (1 + rate) at maturity.
    """

    rate: Decimal = field(default=Decimal("0.02"))
    issuance_day: int = 0
    maturity_days: int = 5

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.BANK_LOAN

    @property
    def maturity_day(self) -> int:
        """Loan matures at issuance_day + maturity_days."""
        return self.issuance_day + self.maturity_days

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
