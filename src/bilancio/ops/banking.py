from __future__ import annotations

from typing import TYPE_CHECKING

from bilancio.core.atomic_tx import atomic
from bilancio.core.errors import ValidationError
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.ops.primitives import coalesce_deposits, split

if TYPE_CHECKING:
    from bilancio.engines.system import System


def deposit_cash(system: System, customer_id: str, bank_id: str, amount: int) -> str:
    if amount <= 0:
        raise ValidationError("amount must be positive")
    # collect payer cash, splitting as needed
    with atomic(system):
        remaining = amount
        cash_ids = [
            cid
            for cid in list(system.state.agents[customer_id].asset_ids)
            if system.state.contracts[cid].kind == InstrumentKind.CASH
        ]
        moved_piece_ids = []
        for cid in cash_ids:
            instr = system.state.contracts[cid]
            if instr.amount > remaining:
                cid = split(system, cid, remaining)
                instr = system.state.contracts[cid]
            # move holder to bank (issuer CB unchanged)
            system.state.agents[customer_id].asset_ids.remove(cid)
            system.state.agents[bank_id].asset_ids.append(cid)
            instr.asset_holder_id = bank_id
            moved_piece_ids.append(cid)
            remaining -= instr.amount
            if remaining == 0:
                break
        if remaining != 0:
            raise ValidationError("insufficient cash for deposit")

        # credit/ensure deposit
        dep_id = coalesce_deposits(system, customer_id, bank_id)
        system.state.contracts[dep_id].amount += amount
        system.log(
            "CashDeposited",
            customer=customer_id,
            bank=bank_id,
            amount=amount,
            cash_piece_ids=moved_piece_ids,
            deposit_id=dep_id,
        )
        return dep_id


def withdraw_cash(system: System, customer_id: str, bank_id: str, amount: int) -> str:
    if amount <= 0:
        raise ValidationError("amount must be positive")
    with atomic(system):
        # 1) debit deposit
        dep_ids = system.deposit_ids(customer_id, bank_id)
        if not dep_ids:
            raise ValidationError("no deposit at this bank")
        remaining = amount
        for dep_id in dep_ids:
            dep = system.state.contracts[dep_id]
            take = min(dep.amount, remaining)
            dep.amount -= take
            remaining -= take
            if dep.amount == 0 and take > 0:
                # remove empty instrument
                holder = system.state.agents[customer_id]
                issuer = system.state.agents[bank_id]
                holder.asset_ids.remove(dep_id)
                issuer.liability_ids.remove(dep_id)
                del system.state.contracts[dep_id]
            if remaining == 0:
                break
        if remaining != 0:
            raise ValidationError("insufficient deposit balance")

        # 2) move cash from bank vault → customer (require sufficient cash on hand)
        bank_cash_ids = [
            cid
            for cid in list(system.state.agents[bank_id].asset_ids)
            if system.state.contracts[cid].kind == InstrumentKind.CASH
        ]
        remaining = amount
        moved_piece_ids = []
        for cid in bank_cash_ids:
            instr = system.state.contracts[cid]
            if instr.amount > remaining:
                cid = split(system, cid, remaining)
                instr = system.state.contracts[cid]
            system.state.agents[bank_id].asset_ids.remove(cid)
            system.state.agents[customer_id].asset_ids.append(cid)
            instr.asset_holder_id = customer_id
            moved_piece_ids.append(cid)
            remaining -= instr.amount
            if remaining == 0:
                break
        if remaining != 0:
            raise ValidationError(
                "bank has insufficient cash on hand (MVP: no auto conversion from reserves)"
            )

        system.log(
            "CashWithdrawn",
            customer=customer_id,
            bank=bank_id,
            amount=amount,
            cash_piece_ids=moved_piece_ids,
        )
        # 3) coalesce customer cash if you want tidy balances (optional)
        return "ok"


def client_payment(
    system: System,
    payer_id: str,
    payer_bank: str,
    payee_id: str,
    payee_bank: str,
    amount: int,
    allow_cash_fallback: bool = False,
) -> str:
    if amount <= 0:
        raise ValidationError("amount must be positive")
    with atomic(system):
        # 1) debit payer's deposit at payer_bank
        dep_ids = system.deposit_ids(payer_id, payer_bank)
        remaining = amount
        deposit_paid = 0
        for dep_id in dep_ids:
            dep = system.state.contracts[dep_id]
            take = min(dep.amount, remaining)
            dep.amount -= take
            remaining -= take
            deposit_paid += take
            if dep.amount == 0 and take > 0:
                holder = system.state.agents[payer_id]
                issuer = system.state.agents[payer_bank]
                holder.asset_ids.remove(dep_id)
                issuer.liability_ids.remove(dep_id)
                del system.state.contracts[dep_id]
            if remaining == 0:
                break

        # Optional fallback: use payer's cash for the remainder (real-world "pay cash")
        cash_paid = 0
        if remaining and allow_cash_fallback:
            cash_ids = [
                cid
                for cid in list(system.state.agents[payer_id].asset_ids)
                if system.state.contracts[cid].kind == InstrumentKind.CASH
            ]
            for cid in cash_ids:
                instr = system.state.contracts[cid]
                if instr.amount > remaining:
                    cid = split(system, cid, remaining)
                    instr = system.state.contracts[cid]
                # move payer cash → payee (physical cash handover)
                system.state.agents[payer_id].asset_ids.remove(cid)
                system.state.agents[payee_id].asset_ids.append(cid)
                instr.asset_holder_id = payee_id
                cash_paid += instr.amount
                remaining -= instr.amount
                if remaining == 0:
                    break

        if remaining != 0:
            raise ValidationError("insufficient funds for payment")

        # 2) credit payee's deposit at payee_bank (only the deposit portion, not cash)
        dep_rx = coalesce_deposits(system, payee_id, payee_bank)
        system.state.contracts[dep_rx].amount += deposit_paid

        # 3) Log payment events with proper classification
        if deposit_paid > 0:
            if payer_bank == payee_bank:
                # Intra-bank payment (same bank)
                system.log(
                    "IntraBankPayment",
                    payer=payer_id,
                    payee=payee_id,
                    bank=payer_bank,
                    amount=deposit_paid,
                )
            else:
                # Inter-bank payment (cross-bank, will need clearing)
                system.log(
                    "ClientPayment",
                    payer=payer_id,
                    payer_bank=payer_bank,
                    payee=payee_id,
                    payee_bank=payee_bank,
                    amount=deposit_paid,
                )

        if cash_paid > 0:
            # Cash payment was used as fallback
            system.log("CashPayment", payer=payer_id, payee=payee_id, amount=cash_paid)

        return dep_rx


def burn_bank_cash(system: System, bank_id: str) -> int:
    """Burn all CASH instruments held by a bank.

    After deposit_cash, banks hold CASH instruments they never use.
    The paper says bank assets = Reserves + Loans only (Section 5).
    This removes CASH from bank balance sheets.

    Args:
        system: System instance
        bank_id: Bank agent ID

    Returns:
        Total amount of cash burned
    """
    bank = system.state.agents.get(bank_id)
    if bank is None:
        return 0

    total_burned = 0
    cash_ids = [
        cid for cid in list(bank.asset_ids)
        if system.state.contracts.get(cid) is not None
        and system.state.contracts[cid].kind == InstrumentKind.CASH
    ]

    for cid in cash_ids:
        contract = system.state.contracts[cid]
        total_burned += contract.amount
        # Remove from bank's assets
        bank.asset_ids.remove(cid)
        # Remove from CB's liabilities (CASH issuer is always CB)
        cb_id = contract.liability_issuer_id
        cb_agent = system.state.agents.get(cb_id)
        if cb_agent and cid in cb_agent.liability_ids:
            cb_agent.liability_ids.remove(cid)
        # Delete the contract
        del system.state.contracts[cid]

    if total_burned > 0:
        system.log("BankCashBurned", bank_id=bank_id, amount=total_burned)

    return total_burned
