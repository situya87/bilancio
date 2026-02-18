"""Interbank lending for active banking.

After client payment clearing (Phase C), surplus banks lend reserves
to deficit banks. Remaining deficits are covered by CB borrowing.

Interbank rate is negotiated between borrower and lender midlines.
Loans mature in 2 days (matching CB borrowing tenor).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.domain.instruments.base import InstrumentKind

if TYPE_CHECKING:
    from bilancio.engines.banking_subsystem import BankingSubsystem, InterbankLoan
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)


def run_interbank_lending(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Redistribute reserves between banks via bilateral interbank lending.

    After client payment netting (Phase C), banks with excess reserves
    lend to banks with deficits. Rate is negotiated as midpoint of
    borrower's and lender's midlines.

    Returns list of interbank loan event dicts.
    """
    from bilancio.engines.banking_subsystem import (
        InterbankLoan,
        _get_bank_deposits_total,
        _get_bank_reserves,
    )

    events: list[dict] = []

    # 1. Compute each bank's reserve surplus/deficit
    positions: dict[str, int] = {}
    for bank_id, bank_state in banking.banks.items():
        reserves = _get_bank_reserves(system, bank_id)
        target = bank_state.pricing_params.reserve_target
        positions[bank_id] = reserves - target

    # 2. Sort: surplus banks (lenders) and deficit banks (borrowers)
    surplus_banks = [(bid, pos) for bid, pos in positions.items() if pos > 0]
    deficit_banks = [(bid, pos) for bid, pos in positions.items() if pos < 0]

    surplus_banks.sort(key=lambda x: -x[1])  # Largest surplus first
    deficit_banks.sort(key=lambda x: x[1])   # Largest deficit first

    # 3. Match: deficit banks borrow from surplus banks
    for def_bank_id, deficit in deficit_banks:
        remaining_need = abs(deficit)
        def_state = banking.banks[def_bank_id]

        for i, (sur_bank_id, surplus) in enumerate(surplus_banks):
            if remaining_need <= 0 or surplus <= 0:
                break

            # Rate negotiation
            borrower_mid = Decimal("0")
            if def_state.current_quote and def_state.current_quote.midline:
                borrower_mid = def_state.current_quote.midline

            lender_rate = Decimal("0")
            sur_state = banking.banks[sur_bank_id]
            if sur_state.current_quote:
                lender_rate = sur_state.current_quote.deposit_rate

            # Lender accepts if offered rate > their deposit rate
            if borrower_mid <= lender_rate:
                continue

            interbank_rate = (borrower_mid + lender_rate) / 2

            transfer = min(remaining_need, surplus)

            # Transfer reserves
            try:
                system.transfer_reserves(sur_bank_id, def_bank_id, transfer)
            except Exception:
                logger.warning(
                    "Interbank transfer failed: %s -> %s, amount=%d",
                    sur_bank_id, def_bank_id, transfer,
                )
                continue

            # Record interbank loan
            ib_loan = InterbankLoan(
                lender_bank=sur_bank_id,
                borrower_bank=def_bank_id,
                amount=transfer,
                rate=interbank_rate,
                issuance_day=current_day,
                maturity_day=current_day + 2,
            )
            banking.interbank_loans.append(ib_loan)

            surplus_banks[i] = (sur_bank_id, surplus - transfer)
            remaining_need -= transfer

            events.append({
                "kind": "InterbankLoan",
                "day": current_day,
                "lender": sur_bank_id,
                "borrower": def_bank_id,
                "amount": transfer,
                "rate": str(interbank_rate),
                "maturity_day": current_day + 2,
            })

    return events


def run_interbank_repayments(
    system: System,
    current_day: int,
    banking: BankingSubsystem,
) -> list[dict]:
    """Process interbank loan repayments due today.

    Transfers reserves from borrower to lender (principal + interest).
    """
    events: list[dict] = []
    remaining = []

    for loan in banking.interbank_loans:
        if loan.maturity_day == current_day:
            repayment = loan.repayment_amount

            try:
                system.transfer_reserves(loan.borrower_bank, loan.lender_bank, repayment)
                events.append({
                    "kind": "InterbankRepaid",
                    "day": current_day,
                    "lender": loan.lender_bank,
                    "borrower": loan.borrower_bank,
                    "principal": loan.amount,
                    "repayment": repayment,
                    "interest": repayment - loan.amount,
                })
            except Exception:
                # Borrower lacks reserves — refinance via CB
                from bilancio.engines.banking_subsystem import _get_bank_reserves

                available = _get_bank_reserves(system, loan.borrower_bank)
                shortfall = repayment - available
                if shortfall < 0:
                    shortfall = 0

                try:
                    if shortfall > 0:
                        cb_loan_id = system.cb_lend_reserves(
                            loan.borrower_bank, shortfall, current_day,
                        )
                        events.append({
                            "kind": "CBRefinance",
                            "day": current_day,
                            "borrower": loan.borrower_bank,
                            "amount": shortfall,
                            "cb_loan_id": cb_loan_id,
                            "reason": "interbank_repayment",
                        })

                    # Retry the transfer after CB refinancing
                    system.transfer_reserves(
                        loan.borrower_bank, loan.lender_bank, repayment,
                    )
                    events.append({
                        "kind": "InterbankRepaid",
                        "day": current_day,
                        "lender": loan.lender_bank,
                        "borrower": loan.borrower_bank,
                        "principal": loan.amount,
                        "repayment": repayment,
                        "interest": repayment - loan.amount,
                        "cb_refinanced": True,
                    })
                except Exception:
                    logger.warning(
                        "Interbank repayment failed even after CB refinancing: "
                        "%s -> %s, amount=%d",
                        loan.borrower_bank, loan.lender_bank, repayment,
                    )
                    remaining.append(loan)
                    continue
        else:
            remaining.append(loan)

    banking.interbank_loans = remaining
    return events
