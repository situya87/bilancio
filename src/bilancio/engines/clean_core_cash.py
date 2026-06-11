"""Cash lot operations for clean-core scenario state."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.engines.clean_core_types import ZERO, CleanState


def add_cash_lot(state: CleanState, agent_id: str, amount: Decimal) -> None:
    if amount > ZERO:
        state.cash_lots[agent_id].append(amount)


def take_cash_lots(state: CleanState, agent_id: str, amount: Decimal) -> list[Decimal]:
    remaining = amount
    pieces: list[Decimal] = []
    lots = state.cash_lots[agent_id]
    if not lots and state.cash[agent_id] > ZERO:
        lots.append(state.cash[agent_id])

    while remaining > ZERO and lots:
        lot = lots.pop(0)
        take = min(lot, remaining)
        pieces.append(take)
        leftover = lot - take
        if leftover > ZERO:
            lots.insert(0, leftover)
        remaining -= take

    if remaining != ZERO:
        raise ValueError(f"insufficient {agent_id} cash lots: short by {remaining}")
    return pieces


def merge_cash_lots(state: CleanState, agent_id: str, *, log: Any) -> None:
    lots = state.cash_lots[agent_id]
    if len(lots) <= 1:
        return
    for index in range(len(lots) - 1):
        log(
            "InstrumentMerged",
            keep=f"cash:{agent_id}",
            removed=f"cash:{agent_id}:merged:{state.day}:{index}",
        )
    total = sum(lots, ZERO)
    state.cash_lots[agent_id] = [total] if total > ZERO else []


def transfer_cash_with_events(
    state: CleanState,
    from_agent: str,
    to_agent: str,
    amount: Decimal,
    *,
    log: Any | None = None,
) -> Decimal:
    if amount <= ZERO:
        return ZERO
    require_at_least(state.cash[from_agent], amount, f"{from_agent} cash")
    pieces = take_cash_lots(state, from_agent, amount)
    state.cash[from_agent] -= amount
    state.cash[to_agent] += amount
    event_log = log or state.log
    for piece in pieces:
        add_cash_lot(state, to_agent, piece)
        event_log("CashTransferred", frm=from_agent, to=to_agent, amount=piece)
    merge_cash_lots(state, to_agent, log=event_log)
    return amount


def require_at_least(actual: Decimal, required: Decimal, label: str) -> None:
    if actual < required:
        raise ValueError(f"insufficient {label}: required {required}, available {actual}")
