"""Collateral pledge record for secured NBFI lending (Plan 059).

A CollateralPledge records the relationship between a NonBankLoan and
the Payable(s) pledged as collateral. It tracks the pledge lifecycle:
active → released (on repayment) or seized (on borrower default) or
impaired (on issuer default).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class CollateralPledge:
    """A pledge of a specific payable as collateral for a non-bank loan.

    Attributes:
        pledge_id: Unique identifier for this pledge
        loan_id: The NonBankLoan this secures
        payable_id: The pledged Payable contract ID
        borrower_id: Agent who pledged the payable
        lender_id: Agent who holds the lien
        pledged_day: Simulation day when pledge was created
        face_value: Payable face value at time of pledge (minor units)
        haircut: Haircut applied (0–1), determines collateral value
        collateral_value: face_value × (1 - haircut), in minor units
        status: Lifecycle state — "active", "released", "seized", "impaired"
    """

    pledge_id: str
    loan_id: str
    payable_id: str
    borrower_id: str
    lender_id: str
    pledged_day: int
    face_value: int
    haircut: Decimal
    collateral_value: int
    status: str = "active"

    def __post_init__(self) -> None:
        if self.status not in ("active", "released", "seized", "impaired"):
            raise ValueError(f"Invalid pledge status: {self.status}")
        if not (Decimal("0") <= self.haircut < Decimal("1")):
            raise ValueError(f"Haircut must be in [0, 1): {self.haircut}")
