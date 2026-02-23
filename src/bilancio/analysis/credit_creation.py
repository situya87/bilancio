"""Credit creation and destruction tracking.

Track endogenous money creation through bank lending, dealer financing,
and NBFI loans.  Measure credit velocity and the net credit impulse
(creation minus destruction) over the simulation horizon.

All functions consume the standard event log (list of dicts) produced by
the simulation engine.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

Event = dict[str, Any]


def credit_created_by_type(
    events: list[Event],
) -> dict[str, Decimal]:
    """Total credit created, broken down by instrument type.

    Credit creation events:
    - BankLoanIssued -> "bank_loan"
    - CBLoanCreated -> "cb_loan"
    - InterbankOvernightCreated -> "interbank_overnight"
    - NonBankLoanCreated -> "nbfi_loan" (non-bank lender)
    - PayableCreated -> "payable" (trade credit)

    Returns:
        {"bank_loan": Decimal, "cb_loan": Decimal, ...}
    """
    created: dict[str, Decimal] = defaultdict(lambda: Decimal(0))

    kind_map = {
        "BankLoanIssued": "bank_loan",
        "CBLoanCreated": "cb_loan",
        "InterbankOvernightCreated": "interbank_overnight",
        "NonBankLoanCreated": "nbfi_loan",
        "PayableCreated": "payable",
    }

    for e in events:
        kind = e.get("kind", "")
        if kind in kind_map:
            amt = Decimal(str(e.get("amount", 0)))
            created[kind_map[kind]] += amt

    return dict(created)


def credit_destroyed_by_type(
    events: list[Event],
) -> dict[str, Decimal]:
    """Total credit destroyed (defaults/writeoffs), by instrument type.

    Destruction events:
    - ObligationWrittenOff with contract_kind -> maps to type
    - AgentDefaulted -> counts the shortfall amount

    Returns:
        {"payable": Decimal, "bank_deposit": Decimal, "interbank_loan": Decimal, ...}
    """
    destroyed: dict[str, Decimal] = defaultdict(lambda: Decimal(0))

    for e in events:
        kind = e.get("kind", "")
        amt = Decimal(str(e.get("amount", 0)))

        if kind == "ObligationWrittenOff":
            contract_kind = str(e.get("contract_kind", "unknown"))
            destroyed[contract_kind] += amt

        elif kind == "CBFinalSettlementWrittenOff":
            destroyed["cb_loan"] += amt

        elif kind == "CBLoanFreezeWrittenOff":
            destroyed["cb_loan_freeze"] += amt

    return dict(destroyed)


def net_credit_impulse(
    events: list[Event],
) -> Decimal:
    """Net credit impulse = total created - total destroyed.

    Positive means the system expanded credit on net; negative means
    contraction (more defaults/writeoffs than new lending).
    """
    created = sum(credit_created_by_type(events).values(), Decimal(0))
    destroyed = sum(credit_destroyed_by_type(events).values(), Decimal(0))
    return created - destroyed


def credit_creation_by_day(
    events: list[Event],
) -> dict[int, dict[str, Decimal]]:
    """Credit created per day, broken down by type.

    Returns:
        {day: {"bank_loan": Decimal, "payable": Decimal, ...}}
    """
    by_day: dict[int, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: Decimal(0))
    )

    kind_map = {
        "BankLoanIssued": "bank_loan",
        "CBLoanCreated": "cb_loan",
        "InterbankOvernightCreated": "interbank_overnight",
        "NonBankLoanCreated": "nbfi_loan",
        "PayableCreated": "payable",
    }

    for e in events:
        kind = e.get("kind", "")
        if kind in kind_map:
            day = int(e.get("day", 0))
            amt = Decimal(str(e.get("amount", 0)))
            by_day[day][kind_map[kind]] += amt

    return {day: dict(v) for day, v in sorted(by_day.items())}


def credit_destruction_by_day(
    events: list[Event],
) -> dict[int, dict[str, Decimal]]:
    """Credit destroyed per day, broken down by type.

    Returns:
        {day: {"payable": Decimal, "bank_deposit": Decimal, ...}}
    """
    by_day: dict[int, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: Decimal(0))
    )

    for e in events:
        kind = e.get("kind", "")
        amt = Decimal(str(e.get("amount", 0)))

        if kind == "ObligationWrittenOff":
            day = int(e.get("day", 0))
            contract_kind = str(e.get("contract_kind", "unknown"))
            by_day[day][contract_kind] += amt

        elif kind in ("CBFinalSettlementWrittenOff", "CBLoanFreezeWrittenOff"):
            day = int(e.get("day", 0))
            by_day[day]["cb_loan"] += amt

    return {day: dict(v) for day, v in sorted(by_day.items())}
