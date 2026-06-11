"""Central-bank loan settlement and bank-resolution helpers for clean-core."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.engines.clean_core_banking import run_clean_bank_loan_winddown
from bilancio.engines.clean_core_cash import add_cash_lot, require_at_least
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanBankingConfig,
    CleanCBLoan,
    CleanState,
)


def finalize_banking_marker_events(
    state: CleanState,
    *,
    final_day: int,
    reached_stable: bool,
    banking_config: CleanBankingConfig | None,
) -> None:
    """Append legacy-compatible no-op banking shutdown events for marker-mode runs."""
    if banking_config is None:
        return

    state.day = final_day
    if reached_stable and not state.cb_lending_frozen:
        state.cb_lending_frozen = True
        state.log("CBLendingFreezeStability")

    run_clean_bank_loan_winddown(state)

    outstanding = state.cb_loans_outstanding
    state.log("CBFinalSettlementStart", cb_loans_outstanding=outstanding)
    settlement = run_clean_final_cb_settlement(state)
    state.log(
        "CBFinalSettlementEnd",
        loans_attempted=settlement["loans_attempted"],
        loans_repaid=settlement["loans_repaid"],
        loans_written_off=settlement["loans_written_off"],
        bank_defaults=settlement["bank_defaults"],
        total_written_off_amount=settlement["total_written_off_amount"],
        cb_loans_outstanding_pre_final=outstanding,
        cb_loans_outstanding_post_final=state.cb_loans_outstanding,
        cb_reserves_initial=state.cb_reserves_initial,
        cb_reserves_final=state.cb_reserves_outstanding,
        cb_interest_total_paid=state.cb_interest_total_paid,
        cb_loans_created_count=state.cb_loans_created_count,
    )


def run_clean_final_cb_settlement(state: CleanState) -> dict[str, Decimal | int]:
    loans_attempted = 0
    loans_repaid = 0
    loans_written_off = 0
    bank_defaults = 0
    total_written_off_amount = ZERO
    defaulted_banks: set[str] = set()

    for loan in list(state.cb_loans):
        if loan.settled:
            continue
        if loan.bank in defaulted_banks:
            loans_attempted += 1
            loans_written_off += 1
            total_written_off_amount += loan.amount
            write_off_clean_cb_loan(
                state,
                loan,
                reason="bank_already_defaulted",
            )
            continue

        loans_attempted += 1
        repayment = loan.amount + loan.interest_amount
        if state.reserves[loan.bank] >= repayment:
            repay_clean_cb_loan(state, loan)
            loans_repaid += 1
            state.log(
                "CBFinalSettlementRepaid",
                loan_id=loan.id,
                bank_id=loan.bank,
            )
            continue

        loans_written_off += 1
        total_written_off_amount += loan.amount
        defaulted_banks.add(loan.bank)
        if loan.bank in state.agents and loan.bank not in state.defaulted_agent_ids:
            state.defaulted_agent_ids.add(loan.bank)
            bank_defaults += 1
            state.log(
                "CBFinalSettlementBankDefault",
                bank_id=loan.bank,
                loan_id=loan.id,
                shortfall=loan.amount,
            )

        write_off_clean_cb_loan(state, loan, reason="insufficient_reserves")
        for other_loan in list(state.cb_loans):
            if other_loan.settled or other_loan.bank != loan.bank:
                continue
            loans_attempted += 1
            loans_written_off += 1
            total_written_off_amount += other_loan.amount
            write_off_clean_cb_loan(
                state,
                other_loan,
                reason="bank_already_defaulted",
            )
        resolve_clean_failed_bank(state, loan.bank)
        write_off_clean_bank_liabilities(state, loan.bank)

    return {
        "loans_attempted": loans_attempted,
        "loans_repaid": loans_repaid,
        "loans_written_off": loans_written_off,
        "bank_defaults": bank_defaults,
        "total_written_off_amount": total_written_off_amount,
    }


def repay_due_cb_loans(state: CleanState) -> None:
    for loan in list(state.cb_loans):
        if loan.settled or state.day < loan.issuance_day + 2:
            continue
        if state.reserves[loan.bank] < loan.repayment_amount:
            if state.cb_lending_frozen:
                write_off_frozen_cb_loan(state, loan)
                continue
            refinance_cb_loan_repayment(state, loan.bank, loan.repayment_amount)
        require_at_least(
            state.reserves[loan.bank],
            loan.repayment_amount,
            f"{loan.bank} reserves to repay CB loan",
        )
        repay_clean_cb_loan(state, loan)


def refinance_cb_loan_repayment(
    state: CleanState,
    bank_id: str,
    repayment: Decimal,
) -> None:
    if state.central_bank_id is None:
        raise ValueError("No central bank found for CB refinancing")
    rate = Decimal("0.03")
    loan_id = f"L_{len(state.cb_loans)}"
    reserve_id = f"R_{state.day}_{len(state.cb_loans)}"
    state.reserves[bank_id] += repayment
    state.cb_reserves_outstanding += repayment
    state.cb_loans_outstanding += repayment
    state.cb_loans_created_count += 1
    state.cb_loans.append(
        CleanCBLoan(
            id=loan_id,
            bank=bank_id,
            central_bank=state.central_bank_id,
            amount=repayment,
            rate=rate,
            issuance_day=state.day,
        )
    )
    state.log(
        "CBLoanCreated",
        bank_id=bank_id,
        amount=repayment,
        loan_id=loan_id,
        reserve_id=reserve_id,
        cb_rate=str(rate),
        maturity_day=state.day + 2,
    )


def repay_clean_cb_loan(state: CleanState, loan: CleanCBLoan) -> None:
    repayment = loan.repayment_amount
    state.reserves[loan.bank] -= repayment
    loan.settled = True
    state.cb_reserves_outstanding -= repayment
    state.cb_loans_outstanding -= loan.amount
    state.cb_interest_total_paid += loan.interest_amount
    state.log(
        "CBLoanRepaid",
        bank_id=loan.bank,
        loan_id=loan.id,
        principal=loan.amount,
        interest=loan.interest_amount,
        total_repaid=repayment,
    )


def write_off_frozen_cb_loan(state: CleanState, loan: CleanCBLoan) -> None:
    state.log("CBLendingFrozen", bank_id=loan.bank, amount=loan.repayment_amount, day=state.day)
    if loan.bank in state.agents and loan.bank not in state.defaulted_agent_ids:
        state.defaulted_agent_ids.add(loan.bank)
        state.log(
            "BankDefaultCBFreeze",
            bank_id=loan.bank,
            loan_id=loan.id,
            amount=loan.amount,
        )
    loan.settled = True
    state.cb_loans_outstanding -= loan.amount
    state.log(
        "CBLoanFreezeWrittenOff",
        bank_id=loan.bank,
        loan_id=loan.id,
        amount=loan.amount,
    )


def write_off_clean_cb_loan(
    state: CleanState,
    loan: CleanCBLoan,
    *,
    reason: str,
) -> None:
    loan.settled = True
    state.cb_loans_outstanding -= loan.amount
    state.log(
        "CBFinalSettlementWrittenOff",
        loan_id=loan.id,
        bank_id=loan.bank,
        amount=loan.amount,
        reason=reason,
    )


def resolve_clean_failed_bank(state: CleanState, bank_id: str) -> None:
    total_reserves = state.reserves[bank_id]
    if total_reserves <= ZERO:
        state.log(
            "BankResolutionCompleted",
            bank_id=bank_id,
            total_reserves=0,
            cb_claims_cancelled=0,
            depositor_distributions=0,
        )
        return

    remaining_reserves = total_reserves
    depositor_distributions: list[dict[str, Any]] = []
    deposit_claims = [
        (customer_id, amount)
        for (customer_id, candidate_bank), amount in state.deposits.items()
        if candidate_bank == bank_id and amount > ZERO
    ]
    total_deposit_claims = sum((amount for _, amount in deposit_claims), ZERO)

    if total_deposit_claims > ZERO:
        total_distributed = ZERO
        for depositor_id, claim_amount in deposit_claims:
            share = Decimal(round((claim_amount / total_deposit_claims) * remaining_reserves))
            share = min(share, remaining_reserves - total_distributed)
            if share <= ZERO:
                continue
            surviving_bank_id = find_clean_surviving_bank(state, depositor_id, bank_id)
            state.reserves[bank_id] -= share
            state.cb_reserves_outstanding -= share
            total_distributed += share
            if surviving_bank_id is not None:
                state.reserves[surviving_bank_id] += share
                state.cb_reserves_outstanding += share
                state.deposits[(depositor_id, surviving_bank_id)] += share
                state.log(
                    "ReservesTransferred",
                    frm=bank_id,
                    to=surviving_bank_id,
                    amount=share,
                    instr_id=f"resolution:{bank_id}:{surviving_bank_id}:{depositor_id}",
                )
                depositor_distributions.append(
                    {
                        "depositor": depositor_id,
                        "amount": share,
                        "method": "deposit_at_surviving_bank",
                        "bank": surviving_bank_id,
                    }
                )
            else:
                instr_id = f"resolution:{bank_id}:{depositor_id}"
                state.log(
                    "ReservesToCash",
                    bank_id=bank_id,
                    amount=share,
                    instr_id=instr_id,
                )
                state.cash[depositor_id] += share
                add_cash_lot(state, depositor_id, share)
                state.log(
                    "CashTransferred",
                    frm=bank_id,
                    to=depositor_id,
                    amount=share,
                    instr_id=instr_id,
                )
                depositor_distributions.append(
                    {
                        "depositor": depositor_id,
                        "amount": share,
                        "method": "cash_no_surviving_bank",
                    }
                )

    state.log(
        "BankResolutionCompleted",
        bank_id=bank_id,
        total_reserves=total_reserves,
        remaining_reserves=remaining_reserves,
        cb_claims_cancelled=0,
        depositor_distributions=depositor_distributions,
        num_depositors=len(depositor_distributions),
    )


def find_clean_surviving_bank(
    state: CleanState,
    depositor_id: str,
    failed_bank_id: str,
) -> str | None:
    for customer_id, candidate_bank in state.deposits:
        if customer_id != depositor_id or candidate_bank == failed_bank_id:
            continue
        bank = state.agents.get(candidate_bank)
        if bank is not None and candidate_bank not in state.defaulted_agent_ids:
            return candidate_bank
    return None


def write_off_clean_bank_liabilities(state: CleanState, bank_id: str) -> None:
    for key, amount in list(state.deposits.items()):
        customer_id, candidate_bank = key
        if candidate_bank != bank_id or amount <= ZERO:
            continue
        state.log(
            "ObligationWrittenOff",
            contract_id=f"deposit:{customer_id}:{bank_id}",
            alias=None,
            debtor=bank_id,
            creditor=customer_id,
            contract_kind="bank_deposit",
            amount=amount,
        )
        del state.deposits[key]
