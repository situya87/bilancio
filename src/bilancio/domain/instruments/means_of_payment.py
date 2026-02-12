from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .base import Instrument, InstrumentKind


@dataclass
class Cash(Instrument):
    """
    Bearer CB liability; issuer is CB; holder can be anyone per policy.

    Cash does not accrue interest.
    """
    def __post_init__(self) -> None:
        self.kind = InstrumentKind.CASH


@dataclass
class BankDeposit(Instrument):
    """
    Liability of a commercial bank; holder is typically household/firm.

    Interest accrual is handled by the banking kernel's cohort tracking,
    not on this base instrument.
    """
    def __post_init__(self) -> None:
        self.kind = InstrumentKind.BANK_DEPOSIT


@dataclass
class ReserveDeposit(Instrument):
    """
    Liability of the central bank; holders are banks/treasury per policy.

    Reserves accrue interest at the reserve remuneration rate (floor rate).
    Interest is credited every 2 days as new reserves.

    The interest tracking fields are optional - if not provided, this
    behaves as a simple non-interest-bearing reserve (backwards compatible).
    """
    # Interest rate on reserves (2-day rate, the CB floor rate)
    # If None, this reserve does not accrue interest
    remuneration_rate: Optional[Decimal] = field(default=None)

    # Day this reserve was created (for interest calculation)
    issuance_day: int = field(default=0)

    # Last day interest was credited on this reserve
    last_interest_day: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.RESERVE_DEPOSIT

    def next_interest_day(self) -> Optional[int]:
        """
        Next day when interest should be credited.

        Returns None if this reserve doesn't accrue interest.
        """
        if self.remuneration_rate is None:
            return None

        if self.last_interest_day is None:
            return self.issuance_day + 2
        return self.last_interest_day + 2

    def compute_interest(self) -> int:
        """
        Compute interest for one 2-day period.

        Returns 0 if this reserve doesn't accrue interest.
        """
        if self.remuneration_rate is None:
            return 0

        # Interest on principal (simple interest per period)
        return int(self.remuneration_rate * self.amount)

    def is_interest_due(self, current_day: int) -> bool:
        """Check if interest is due on this reserve."""
        next_day = self.next_interest_day()
        if next_day is None:
            return False
        return current_day >= next_day
