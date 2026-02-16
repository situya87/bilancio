"""
Central Bank Loan instrument.

A CBLoan is an asset of the Central Bank and a liability of a commercial bank.
It represents reserves lent by the CB to a bank at the CB lending rate (ceiling).

Key properties:
- Issued by CentralBank (as asset holder)
- Bank is the liability issuer (borrower)
- Matures at issuance_day + 2 (2-day term)
- Repayment includes interest at cb_rate
"""

from dataclasses import dataclass, field
from decimal import Decimal

from .base import Instrument, InstrumentKind


@dataclass
class CBLoan(Instrument):
    """
    Central Bank loan to a commercial bank.

    This is the CB's asset (claim on the bank) and the bank's liability.
    The loan matures at issuance_day + 2 with repayment = principal × (1 + cb_rate).
    """

    # Interest rate on the loan (2-day rate, the CB ceiling rate)
    cb_rate: Decimal = field(default=Decimal("0.03"))

    # Day the loan was issued
    issuance_day: int = 0

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.CB_LOAN

    @property
    def maturity_day(self) -> int:
        """CB loans mature 2 days after issuance."""
        return self.issuance_day + 2

    @property
    def repayment_amount(self) -> int:
        """Amount due at maturity: principal × (1 + cb_rate)."""
        return int(self.amount * (1 + self.cb_rate))

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
