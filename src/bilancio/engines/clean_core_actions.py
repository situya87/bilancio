"""Scenario action application helpers for the clean-core engine."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.engines.clean_core_cash import add_cash_lot as _add_cash_lot
from bilancio.engines.clean_core_cash import require_at_least as _require_at_least
from bilancio.engines.clean_core_cash import take_cash_lots as _take_cash_lots
from bilancio.engines.clean_core_cash import transfer_cash_with_events as _transfer_cash_with_events
from bilancio.engines.clean_core_config import select_action_payload as _select_action_payload
from bilancio.engines.clean_core_contracts import contract_id_for_alias as _contract_id_for_alias
from bilancio.engines.clean_core_contracts import transfer_delivery_claim as _transfer_delivery_claim
from bilancio.engines.clean_core_contracts import transfer_payable_claim as _transfer_payable_claim
from bilancio.engines.clean_core_interbank import primary_bank_for_customer as _primary_bank_for_customer
from bilancio.engines.clean_core_inventory import first_stock_lot_by_sku as _first_stock_lot_by_sku
from bilancio.engines.clean_core_inventory import move_stock_lot as _move_stock_lot
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanCBLoan,
    CleanDeliveryObligation,
    CleanPayable,
    CleanState,
    CleanStockLot,
)


def _as_decimal(amount: Any) -> Decimal:
    # Legacy config application casts financial action amounts with int(...).
    # Preserve that public scenario behavior while keeping Decimal internally.
    return Decimal(int(Decimal(str(amount))))


def _unique_clean_id(state: CleanState, prefix: str, preferred_index: int) -> str:
    used = {
        *(state.stocks.keys()),
        *(payable.id for payable in state.payables),
        *(obligation.id for obligation in state.delivery_obligations),
        *(loan.id for loan in state.cb_loans),
        *(loan.id for loan in state.non_bank_loans),
        *(loan.id for loan in state.bank_loans),
    }
    candidate = f"{prefix}_{preferred_index}"
    if candidate not in used:
        return candidate

    next_index = 0
    while True:
        candidate = f"{prefix}_{next_index}"
        if candidate not in used:
            return candidate
        next_index += 1


def _apply_action(state: CleanState, action: dict[str, Any], *, index: int, setup: bool) -> None:
    log = state.log_setup if setup else state.log
    action_name, payload = _select_action_payload(action)
    if action_name is None or not isinstance(payload, dict):
        unsupported = next(iter(action.keys()), "<empty>") if isinstance(action, dict) else "<invalid>"
        raise NotImplementedError(f"clean core does not support initial action: {unsupported}")
    action = {action_name: payload}

    if "mint_reserves" in action:
        payload = action["mint_reserves"]
        amount = _as_decimal(payload["amount"])
        bank_id = payload["to"]
        state.reserves[bank_id] += amount
        state.cb_reserves_outstanding += amount
        event = {"to": bank_id, "amount": amount}
        if payload.get("alias") is not None:
            event["alias"] = payload["alias"]
        log("ReservesMinted", **event)
        return

    if "mint_cash" in action:
        payload = action["mint_cash"]
        amount = _as_decimal(payload["amount"])
        agent_id = payload["to"]
        state.cash[agent_id] += amount
        _add_cash_lot(state, agent_id, amount)
        event = {"to": agent_id, "amount": amount}
        if payload.get("alias") is not None:
            event["alias"] = payload["alias"]
        log("CashMinted", **event)
        return

    if "transfer_reserves" in action:
        payload = action["transfer_reserves"]
        amount = _as_decimal(payload["amount"])
        from_bank = payload["from_bank"]
        to_bank = payload["to_bank"]
        if from_bank == to_bank:
            raise ValueError("no-op transfer")
        receiver_reserves_before = state.reserves[to_bank]
        _require_at_least(state.reserves[from_bank], amount, f"{from_bank} reserves")
        state.reserves[from_bank] -= amount
        state.reserves[to_bank] += amount
        log("ReservesTransferred", frm=from_bank, to=to_bank, amount=amount)
        if receiver_reserves_before:
            log("InstrumentMerged", keep=f"reserve:{to_bank}", removed=f"transfer:{from_bank}:{to_bank}:{state.day}")
        return

    if "transfer_cash" in action:
        payload = action["transfer_cash"]
        amount = _as_decimal(payload["amount"])
        from_agent = payload["from_agent"]
        to_agent = payload["to_agent"]
        if from_agent == to_agent:
            raise ValueError("no-op transfer")
        _transfer_cash_with_events(state, from_agent, to_agent, amount, log=log)
        return

    if "deposit_cash" in action:
        payload = action["deposit_cash"]
        amount = _as_decimal(payload["amount"])
        customer_id = payload["customer"]
        bank_id = payload["bank"]
        _require_at_least(state.cash[customer_id], amount, f"{customer_id} cash")
        _take_cash_lots(state, customer_id, amount)
        state.cash[customer_id] -= amount
        state.cash[bank_id] += amount
        _add_cash_lot(state, bank_id, amount)
        state.deposits[(customer_id, bank_id)] += amount
        log("CashDeposited", customer=customer_id, bank=bank_id, amount=amount)
        return

    if "withdraw_cash" in action:
        payload = action["withdraw_cash"]
        amount = _as_decimal(payload["amount"])
        customer_id = payload["customer"]
        bank_id = payload["bank"]
        _require_at_least(state.deposits[(customer_id, bank_id)], amount, f"{customer_id} deposit at {bank_id}")
        _require_at_least(state.cash[bank_id], amount, f"{bank_id} cash")
        _take_cash_lots(state, bank_id, amount)
        state.deposits[(customer_id, bank_id)] -= amount
        state.cash[bank_id] -= amount
        state.cash[customer_id] += amount
        _add_cash_lot(state, customer_id, amount)
        log("CashWithdrawn", customer=customer_id, bank=bank_id, amount=amount)
        return

    if "client_payment" in action:
        payload = action["client_payment"]
        amount = _as_decimal(payload["amount"])
        payer_id = payload["payer"]
        payee_id = payload["payee"]
        if payer_id not in state.agents or payee_id not in state.agents:
            raise ValueError(f"Unknown agent in client_payment: {payer_id} or {payee_id}")
        payer_bank = _primary_bank_for_customer(state, payer_id)
        payee_bank = _primary_bank_for_customer(state, payee_id)
        if payer_bank is None or payee_bank is None:
            raise ValueError(f"Cannot determine banks for client_payment from {payer_id} to {payee_id}")
        _require_at_least(state.deposits[(payer_id, payer_bank)], amount, "payer deposit")
        state.deposits[(payer_id, payer_bank)] -= amount
        state.deposits[(payee_id, payee_bank)] += amount
        if payer_bank == payee_bank:
            log("IntraBankPayment", payer=payer_id, payee=payee_id, bank=payer_bank, amount=amount)
        else:
            log(
                "ClientPayment",
                payer=payer_id,
                payer_bank=payer_bank,
                payee=payee_id,
                payee_bank=payee_bank,
                amount=amount,
            )
        return

    if "burn_bank_cash" in action:
        payload = action["burn_bank_cash"]
        bank_id = payload["bank"]
        burned = state.cash[bank_id]
        if burned:
            state.cash[bank_id] = ZERO
            state.cash_lots[bank_id].clear()
            log("BankCashBurned", bank_id=bank_id, amount=burned)
        return

    if "create_cb_loan" in action:
        payload = action["create_cb_loan"]
        if state.central_bank_id is None:
            raise ValueError("No central bank found for create_cb_loan")
        loan = CleanCBLoan(
            id=_unique_clean_id(state, "CBL", index),
            bank=payload["bank"],
            central_bank=state.central_bank_id,
            amount=_as_decimal(payload["amount"]),
            rate=Decimal(str(payload.get("rate", Decimal("0.03")))),
            issuance_day=int(payload.get("issuance_day", 0)),
            alias=payload.get("alias"),
        )
        state.cb_loans.append(loan)
        event = {
            "bank": loan.bank,
            "amount": loan.amount,
            "rate": str(loan.rate),
            "issuance_day": loan.issuance_day,
            "loan_id": loan.id,
            "alias": loan.alias,
        }
        log("CBLoanCreated", **event)
        return

    if "create_payable" in action:
        payload = action["create_payable"]
        amount = _as_decimal(payload["amount"])
        payable = CleanPayable(
            id=_unique_clean_id(state, "PAY", index),
            debtor=payload["from"],
            creditor=payload["to"],
            amount=amount,
            due_day=int(payload["due_day"]),
            maturity_distance=int(payload.get("maturity_distance") or payload["due_day"]),
            alias=payload.get("alias"),
        )
        state.payables.append(payable)
        event = {
            "debtor": payable.debtor,
            "creditor": payable.creditor,
            "amount": amount,
            "due_day": payable.due_day,
            "maturity_distance": int(payload.get("maturity_distance") or payable.due_day),
            "payable_id": payable.id,
            "alias": payable.alias,
        }
        log(
            "PayableCreated",
            **event,
        )
        return

    if "create_stock" in action:
        payload = action["create_stock"]
        stock_id = _unique_clean_id(state, "S", index)
        lot = CleanStockLot(
            id=stock_id,
            owner=payload["owner"],
            sku=payload["sku"],
            quantity=int(payload["quantity"]),
            unit_price=Decimal(str(payload["unit_price"])),
        )
        state.stocks[stock_id] = lot
        log(
            "StockCreated",
            owner=lot.owner,
            sku=lot.sku,
            qty=lot.quantity,
            unit_price=lot.unit_price,
            stock_id=stock_id,
        )
        return

    if "transfer_stock" in action:
        payload = action["transfer_stock"]
        from_agent = payload["from_agent"]
        to_agent = payload["to_agent"]
        sku = payload["sku"]
        quantity = int(payload["quantity"])
        stock = _first_stock_lot_by_sku(state, from_agent, sku)
        if stock is None:
            raise ValueError(f"No stock with SKU {sku} owned by {from_agent}")
        if stock.quantity < quantity:
            raise ValueError(f"Insufficient stock: {stock.quantity} < {quantity}")
        moving_id = _move_stock_lot(state, stock, from_agent, to_agent, quantity, log=log)
        moving_stock = state.stocks[moving_id]
        log(
            "StockTransferred",
            frm=from_agent,
            to=to_agent,
            stock_id=moving_id,
            sku=moving_stock.sku,
            qty=moving_stock.quantity,
        )
        return

    if "create_delivery_obligation" in action:
        payload = action["create_delivery_obligation"]
        obligation = CleanDeliveryObligation(
            id=_unique_clean_id(state, "D", index),
            debtor=payload["from"],
            creditor=payload["to"],
            sku=payload["sku"],
            quantity=int(payload["quantity"]),
            unit_price=Decimal(str(payload["unit_price"])),
            due_day=int(payload["due_day"]),
            alias=payload.get("alias"),
        )
        state.delivery_obligations.append(obligation)
        event = {
            "id": obligation.id,
            "frm": obligation.debtor,
            "to": obligation.creditor,
            "sku": obligation.sku,
            "qty": obligation.quantity,
            "due_day": obligation.due_day,
            "unit_price": obligation.unit_price,
        }
        if obligation.alias is not None:
            event["alias"] = obligation.alias
        log("DeliveryObligationCreated", **event)
        return

    if "transfer_claim" in action:
        payload = action["transfer_claim"]
        alias = payload.get("contract_alias")
        explicit_id = payload.get("contract_id")
        id_from_alias = _contract_id_for_alias(state, alias) if alias is not None else None
        if alias is not None and id_from_alias is None:
            raise ValueError(f"Unknown alias: {alias}")
        if alias is not None and explicit_id is not None and id_from_alias != explicit_id:
            raise ValueError(f"Alias {alias} and contract_id {explicit_id} refer to different contracts")
        resolved_id = explicit_id or id_from_alias
        if not resolved_id:
            raise ValueError("transfer_claim requires contract_alias or contract_id to resolve a contract")
        if _transfer_payable_claim(state, resolved_id, payload["to_agent"], alias=alias, log=log):
            return
        if _transfer_delivery_claim(state, resolved_id, payload["to_agent"], alias=alias, log=log):
            return
        raise ValueError(f"Contract not found: {resolved_id}")

    action_name = next(iter(action.keys()), "<empty>")
    raise NotImplementedError(f"clean core does not support initial action: {action_name}")


