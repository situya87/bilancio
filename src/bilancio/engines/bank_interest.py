"""Deposit interest accrual for active banking.

Credits interest on bank deposits every N days (default: 2).
Interest is a book entry — increases deposit (bank liability),
no reserve movement. The bank earns its margin from loans (r_L)
exceeding deposit costs (r_D).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.domain.instruments.base import InstrumentKind

if TYPE_CHECKING:
    from bilancio.engines.banking_subsystem import BankingSubsystem
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)


def accrue_deposit_interest(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Credit interest on deposits every N days.

    Interest = deposit_amount * r_D (from the bank's current quote).
    This is a deposit-only book entry (no reserve movement).

    Args:
        system: Main system state.
        current_day: Current simulation day.
        banking: Active banking subsystem.

    Returns:
        List of interest event dicts.
    """
    events: list[dict] = []
    period = banking.interest_period

    if current_day % period != 0 or current_day == 0:
        return events  # Only accrue on period boundaries, skip day 0

    for agent_id, agent in system.state.agents.items():
        for cid in agent.asset_ids:
            contract = system.state.contracts.get(cid)
            if contract is None or contract.kind != InstrumentKind.BANK_DEPOSIT:
                continue

            bank_id = contract.liability_issuer_id
            bank_state = banking.banks.get(bank_id)
            if bank_state is None:
                continue

            # Use current r_D from this bank
            r_D = Decimal("0")
            if bank_state.current_quote:
                r_D = bank_state.current_quote.deposit_rate

            if r_D <= Decimal("0"):
                continue

            interest = int(contract.amount * r_D)
            if interest > 0:
                contract.amount += interest
                events.append({
                    "kind": "DepositInterest",
                    "day": current_day,
                    "agent": agent_id,
                    "bank": bank_id,
                    "interest": interest,
                    "rate": str(r_D),
                    "deposit_balance": contract.amount,
                })

    return events
