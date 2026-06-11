"""Rollover helpers for settled clean-core payables."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from bilancio.engines.clean_core_types import ZERO, CleanPayable, CleanState

DepositPayment = Callable[[CleanState, str, str, Decimal], Decimal]
CashTransfer = Callable[[CleanState, str, str, Decimal], Decimal]


def rollover_settled_payables(
    state: CleanState,
    settled_payables: list[tuple[str, str, Decimal, int]],
    *,
    pay_with_deposit: DepositPayment,
    transfer_cash_with_events: CashTransfer,
) -> list[str]:
    max_due_day = state.day
    for payable in state.payables:
        if payable.settled:
            continue
        if payable.due_day > max_due_day:
            max_due_day = payable.due_day

    new_payable_ids: list[str] = []
    for debtor_id, creditor_id, amount, maturity_distance in settled_payables:
        new_due_day = max_due_day + maturity_distance
        payable_id = rollover_single_payable(
            state,
            debtor_id,
            creditor_id,
            amount,
            maturity_distance,
            new_due_day,
            pay_with_deposit=pay_with_deposit,
            transfer_cash_with_events=transfer_cash_with_events,
        )
        if payable_id is not None:
            new_payable_ids.append(payable_id)
    return new_payable_ids


def rollover_single_payable(
    state: CleanState,
    debtor_id: str,
    creditor_id: str,
    amount: Decimal,
    maturity_distance: int,
    new_due_day: int,
    *,
    pay_with_deposit: DepositPayment,
    transfer_cash_with_events: CashTransfer,
) -> str | None:
    if debtor_id not in state.agents or debtor_id in state.defaulted_agent_ids:
        return None
    if creditor_id not in state.agents or creditor_id in state.defaulted_agent_ids:
        return None

    new_payable = CleanPayable(
        id=f"PAY_rollover_{len(state.payables)}",
        debtor=debtor_id,
        creditor=creditor_id,
        amount=amount,
        due_day=new_due_day,
        maturity_distance=maturity_distance,
    )
    state.payables.append(new_payable)

    cash_transferred = pay_with_deposit(state, creditor_id, debtor_id, amount)
    remaining = amount - cash_transferred
    if remaining > ZERO:
        cash_paid = min(state.cash[creditor_id], remaining)
        if cash_paid > ZERO:
            transfer_cash_with_events(state, creditor_id, debtor_id, cash_paid)
            cash_transferred += cash_paid

    if cash_transferred != amount:
        state.log(
            "RolloverPartial",
            debtor=debtor_id,
            creditor=creditor_id,
            amount=amount,
            cash_transferred=cash_transferred,
            new_due_day=new_due_day,
            payable_id=new_payable.id,
            cash_transfer=True,
        )
    else:
        state.log(
            "PayableRolledOver",
            debtor=debtor_id,
            creditor=creditor_id,
            amount=amount,
            new_due_day=new_due_day,
            maturity_distance=maturity_distance,
            payable_id=new_payable.id,
            cash_transfer=True,
        )
    return new_payable.id
