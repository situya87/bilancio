"""Contract helper functions for the clean-core scenario engine."""

from __future__ import annotations

from typing import Any

from bilancio.engines.clean_core_types import CleanState


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


def contract_id_for_alias(state: CleanState, alias: str | None) -> str | None:
    if alias is None:
        return None
    for payable in state.payables:
        if payable.alias == alias:
            return payable.id
    for obligation in state.delivery_obligations:
        if obligation.alias == alias:
            return obligation.id
    return None


def transfer_payable_claim(
    state: CleanState,
    contract_id: str,
    to_agent: str,
    *,
    alias: str | None,
    log: Any,
) -> bool:
    for payable in state.payables:
        if payable.id != contract_id:
            continue
        old_holder = payable.creditor
        payable.creditor = to_agent
        log(
            "ClaimTransferred",
            contract_id=payable.id,
            frm=old_holder,
            to=to_agent,
            contract_kind="payable",
            amount=payable.amount,
            due_day=payable.due_day,
            sku=None,
            alias=alias,
        )
        return True
    return False


def transfer_delivery_claim(
    state: CleanState,
    contract_id: str,
    to_agent: str,
    *,
    alias: str | None,
    log: Any,
) -> bool:
    for obligation in state.delivery_obligations:
        if obligation.id != contract_id:
            continue
        old_holder = obligation.creditor
        obligation.creditor = to_agent
        log(
            "ClaimTransferred",
            contract_id=obligation.id,
            frm=old_holder,
            to=to_agent,
            contract_kind="delivery_obligation",
            amount=obligation.quantity,
            due_day=obligation.due_day,
            sku=obligation.sku,
            alias=alias,
        )
        return True
    return False
