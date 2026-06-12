"""Scenario action execution for the v2 kernel.

Actions are the public scenario vocabulary (``mint_cash``, ``create_payable``,
…). Each one translates into ledger operations; the ledger records the
events. Semantics match the existing engine exactly, including the legacy
``int(...)`` cast of financial amounts and the action dispatch precedence.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio_v2.ledger import Ledger

ACTION_DISPATCH_ORDER = (
    "mint_reserves",
    "mint_cash",
    "transfer_reserves",
    "transfer_cash",
    "deposit_cash",
    "withdraw_cash",
    "client_payment",
    "create_stock",
    "transfer_stock",
    "create_delivery_obligation",
    "create_payable",
    "create_cb_loan",
    "burn_bank_cash",
    "transfer_claim",
)


class UnsupportedActionError(NotImplementedError):
    pass


def as_amount(value: Any) -> Decimal:
    # The public scenario contract truncates financial amounts to whole
    # minor units (legacy behavior); keep Decimal internally.
    return Decimal(int(Decimal(str(value))))


def select_action_payload(action: Any) -> tuple[str | None, Any]:
    if not isinstance(action, dict):
        return None, None
    for action_name in ACTION_DISPATCH_ORDER:
        if action_name in action:
            return action_name, action[action_name]
    return None, None


def action_references_agent(action: dict[str, Any], agent_id: str) -> bool:
    if not isinstance(action, dict) or len(action) != 1:
        return False
    action_name, payload = next(iter(action.items()))
    if not isinstance(payload, dict):
        return False
    fields_by_action = {
        "mint_reserves": ("to",),
        "mint_cash": ("to",),
        "transfer_reserves": ("from_bank", "to_bank"),
        "transfer_cash": ("from_agent", "to_agent"),
        "deposit_cash": ("customer", "bank"),
        "withdraw_cash": ("customer", "bank"),
        "client_payment": ("payer", "payee"),
        "create_stock": ("owner",),
        "transfer_stock": ("from_agent", "to_agent"),
        "create_delivery_obligation": ("from", "from_agent", "to", "to_agent"),
        "create_payable": ("from", "from_agent", "to", "to_agent"),
    }
    return any(payload.get(field) == agent_id for field in fields_by_action.get(action_name, ()))


def apply_action(ledger: Ledger, action: dict[str, Any], *, index: int, setup: bool) -> None:
    action_name, payload = select_action_payload(action)
    if action_name is None or not isinstance(payload, dict):
        unsupported = next(iter(action.keys()), "<empty>") if isinstance(action, dict) else "<invalid>"
        raise UnsupportedActionError(f"v2 kernel does not support initial action: {unsupported}")

    if action_name == "mint_reserves":
        ledger.mint_reserves(payload["to"], as_amount(payload["amount"]), alias=payload.get("alias"), setup=setup)
    elif action_name == "mint_cash":
        ledger.mint_cash(payload["to"], as_amount(payload["amount"]), alias=payload.get("alias"), setup=setup)
    elif action_name == "transfer_reserves":
        ledger.transfer_reserves(payload["from_bank"], payload["to_bank"], as_amount(payload["amount"]), setup=setup)
    elif action_name == "transfer_cash":
        from_agent = payload["from_agent"]
        to_agent = payload["to_agent"]
        if from_agent == to_agent:
            raise ValueError("no-op transfer")
        ledger.transfer_cash(from_agent, to_agent, as_amount(payload["amount"]), setup=setup)
    elif action_name == "deposit_cash":
        ledger.deposit_cash(payload["customer"], payload["bank"], as_amount(payload["amount"]), setup=setup)
    elif action_name == "withdraw_cash":
        ledger.withdraw_cash(payload["customer"], payload["bank"], as_amount(payload["amount"]), setup=setup)
    elif action_name == "client_payment":
        _apply_client_payment(ledger, payload, setup=setup)
    elif action_name == "burn_bank_cash":
        ledger.burn_bank_cash(payload["bank"], setup=setup)
    elif action_name == "create_cb_loan":
        ledger.create_cb_loan(
            loan_id=ledger.unique_contract_id("CBL", index),
            bank=payload["bank"],
            amount=as_amount(payload["amount"]),
            rate=Decimal(str(payload.get("rate", Decimal("0.03")))),
            issuance_day=int(payload.get("issuance_day", 0)),
            alias=payload.get("alias"),
            setup=setup,
        )
    elif action_name == "create_payable":
        due_day = int(payload["due_day"])
        debtor = payload["from"]
        creditor = payload["to"]
        amount = as_amount(payload["amount"])
        maturity_distance = int(payload.get("maturity_distance") or payload["due_day"])
        if ledger.clearing_config is not None and getattr(ledger.clearing_config, "mode", None) == "ccp":
            # Novation chokepoint (Plan 061): member↔member payables are
            # rewritten into A→CCP1 + CCP1→B legs at creation time. Lazy
            # import: the plugin imports the ledger, never actions.
            from bilancio_v2.plugins.clearinghouse_ccp import is_member, novate_payable_creation

            if is_member(ledger, debtor) and is_member(ledger, creditor):
                novate_payable_creation(
                    ledger,
                    debtor=debtor,
                    creditor=creditor,
                    amount=amount,
                    due_day=due_day,
                    maturity_distance=maturity_distance,
                    alias=payload.get("alias"),
                    setup=setup,
                    index=index,
                )
                return
            if is_member(ledger, debtor) != is_member(ledger, creditor):
                from bilancio.core.errors import ConfigurationError

                # A member with a non-member payable counterparty breaks the
                # star: on expulsion the mutualized fund would leak to the
                # non-member via reassignment (PR #173 review, stage 1 scope).
                raise ConfigurationError(
                    "payables linking a clearing member and a non-member are not supported in ccp mode (stage 1 scope)"
                )
        ledger.create_payable(
            payable_id=ledger.unique_contract_id("PAY", index),
            debtor=debtor,
            creditor=creditor,
            amount=amount,
            due_day=due_day,
            maturity_distance=maturity_distance,
            alias=payload.get("alias"),
            setup=setup,
        )
    elif action_name == "create_stock":
        ledger.create_stock(
            stock_id=ledger.unique_contract_id("S", index),
            owner=payload["owner"],
            sku=payload["sku"],
            quantity=int(payload["quantity"]),
            unit_price=Decimal(str(payload["unit_price"])),
            setup=setup,
        )
    elif action_name == "transfer_stock":
        _apply_transfer_stock(ledger, payload, setup=setup)
    elif action_name == "create_delivery_obligation":
        ledger.create_delivery_obligation(
            obligation_id=ledger.unique_contract_id("D", index),
            debtor=payload["from"],
            creditor=payload["to"],
            sku=payload["sku"],
            quantity=int(payload["quantity"]),
            unit_price=Decimal(str(payload["unit_price"])),
            due_day=int(payload["due_day"]),
            alias=payload.get("alias"),
            setup=setup,
        )
    elif action_name == "transfer_claim":
        _apply_transfer_claim(ledger, payload, setup=setup)
    else:  # pragma: no cover - dispatch order and branches are exhaustive
        raise UnsupportedActionError(f"v2 kernel does not support initial action: {action_name}")


def _apply_client_payment(ledger: Ledger, payload: dict[str, Any], *, setup: bool) -> None:
    amount = as_amount(payload["amount"])
    payer_id = payload["payer"]
    payee_id = payload["payee"]
    if payer_id not in ledger.agents or payee_id not in ledger.agents:
        raise ValueError(f"Unknown agent in client_payment: {payer_id} or {payee_id}")
    payer_bank = ledger.primary_bank_for_customer(payer_id)
    payee_bank = ledger.primary_bank_for_customer(payee_id)
    if payer_bank is None or payee_bank is None:
        raise ValueError(f"Cannot determine banks for client_payment from {payer_id} to {payee_id}")
    if ledger.deposits[(payer_id, payer_bank)] < amount:
        raise ValueError(f"insufficient payer deposit: required {amount}, available {ledger.deposits[(payer_id, payer_bank)]}")
    ledger.move_deposit(payer_id, payer_bank, payee_id, payee_bank, amount, setup=setup)


def _apply_transfer_stock(ledger: Ledger, payload: dict[str, Any], *, setup: bool) -> None:
    from_agent = payload["from_agent"]
    to_agent = payload["to_agent"]
    sku = payload["sku"]
    quantity = int(payload["quantity"])
    stock = ledger.first_stock_lot_by_sku(from_agent, sku)
    if stock is None:
        raise ValueError(f"No stock with SKU {sku} owned by {from_agent}")
    if stock.quantity < quantity:
        raise ValueError(f"Insufficient stock: {stock.quantity} < {quantity}")
    moving_id = ledger.move_stock_lot(stock, from_agent, to_agent, quantity, setup=setup)
    moving_stock = ledger.stocks[moving_id]
    ledger.record(
        "StockTransferred",
        setup=setup,
        frm=from_agent,
        to=to_agent,
        stock_id=moving_id,
        sku=moving_stock.sku,
        qty=moving_stock.quantity,
    )


def _apply_transfer_claim(ledger: Ledger, payload: dict[str, Any], *, setup: bool) -> None:
    alias = payload.get("contract_alias")
    explicit_id = payload.get("contract_id")
    id_from_alias = ledger.contract_id_for_alias(alias) if alias is not None else None
    if alias is not None and id_from_alias is None:
        raise ValueError(f"Unknown alias: {alias}")
    if alias is not None and explicit_id is not None and id_from_alias != explicit_id:
        raise ValueError(f"Alias {alias} and contract_id {explicit_id} refer to different contracts")
    resolved_id = explicit_id or id_from_alias
    if not resolved_id:
        raise ValueError("transfer_claim requires contract_alias or contract_id to resolve a contract")
    to_agent = payload["to_agent"]

    for payable in ledger.payables:
        if payable.id != resolved_id:
            continue
        if ledger.clearing_config is not None and getattr(ledger.clearing_config, "mode", None) == "ccp":
            from bilancio.core.errors import ConfigurationError
            from bilancio_v2.plugins.clearinghouse_ccp import is_member

            # Novation escape hatch (PR #173 review): transferring a novated
            # leg re-links the star, and transferring any claim to a member
            # whose debtor is also a member recreates a member<->member edge.
            if payable.origin_debtor is not None:
                raise ConfigurationError(
                    "transfer_claim cannot move a novated CCP leg (stage 1 ccp scope)"
                )
            if is_member(ledger, payable.debtor) and is_member(ledger, to_agent):
                raise ConfigurationError(
                    "transfer_claim would create a member-to-member payable in ccp mode (stage 1 scope)"
                )
        old_holder = payable.creditor
        payable.creditor = to_agent
        ledger.record(
            "ClaimTransferred",
            setup=setup,
            contract_id=payable.id,
            frm=old_holder,
            to=to_agent,
            contract_kind="payable",
            amount=payable.amount,
            due_day=payable.due_day,
            sku=None,
            alias=alias,
        )
        return
    for obligation in ledger.delivery_obligations:
        if obligation.id != resolved_id:
            continue
        old_holder = obligation.creditor
        obligation.creditor = to_agent
        ledger.record(
            "ClaimTransferred",
            setup=setup,
            contract_id=obligation.id,
            frm=old_holder,
            to=to_agent,
            contract_kind="delivery_obligation",
            amount=obligation.quantity,
            due_day=obligation.due_day,
            sku=obligation.sku,
            alias=alias,
        )
        return
    raise ValueError(f"Contract not found: {resolved_id}")
