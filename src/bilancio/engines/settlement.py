"""Settlement engine (Phase B) for settling payables due today."""

from __future__ import annotations

from typing import List, Tuple

from bilancio.core.atomic_tx import atomic
from bilancio.core.errors import DefaultError, ValidationError
from bilancio.ops.banking import client_payment
from bilancio.ops.aliases import get_alias_for_id

DEFAULT_MODE_FAIL_FAST = "fail-fast"
DEFAULT_MODE_EXPEL = "expel-agent"

# Plan 024: Track settled payables for rollover
_settled_payables_for_rollover: List[Tuple[str, str, int, int, int]] = []
# List of (debtor_id, creditor_id, amount, maturity_distance, current_day)

_ACTION_AGENT_FIELDS = {
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
    "transfer_claim": ("to_agent",),
}

_ACTION_CONTRACT_FIELDS = {
    "transfer_claim": ("contract_id", "contract_alias"),
    "mint_cash": ("alias",),
    "mint_reserves": ("alias",),
    "create_delivery_obligation": ("alias",),
    "create_payable": ("alias",),
}


def _get_default_mode(system) -> str:
    """Return the configured default-handling mode for the system."""
    return getattr(system, "default_mode", DEFAULT_MODE_FAIL_FAST)


def _get_risk_assessor(system):
    subsystem = getattr(system.state, "dealer_subsystem", None)
    return getattr(subsystem, "risk_assessor", None)


def due_payables(system, day: int):
    """Scan contracts for payables with due_day == day."""
    for c in system.state.contracts.values():
        if c.kind == "payable" and getattr(c, "due_day", None) == day:
            yield c


def due_delivery_obligations(system, day: int):
    """Scan contracts for delivery obligations with due_day == day."""
    for c in system.state.contracts.values():
        if c.kind == "delivery_obligation" and getattr(c, "due_day", None) == day:
            yield c


def _pay_with_deposits(system, debtor_id, creditor_id, amount) -> int:
    """Pay using bank deposits. Returns amount actually paid."""
    debtor_deposit_ids = []
    for cid in system.state.agents[debtor_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references
        if contract.kind == "bank_deposit":
            debtor_deposit_ids.append(cid)

    if not debtor_deposit_ids:
        return 0

    available = sum(system.state.contracts[cid].amount for cid in debtor_deposit_ids)
    if available == 0:
        return 0

    pay_amount = min(amount, available)

    debtor_bank_id = None
    creditor_bank_id = None

    if debtor_deposit_ids:
        debtor_bank_id = system.state.contracts[debtor_deposit_ids[0]].liability_issuer_id

    creditor_deposit_ids = []
    for cid in system.state.agents[creditor_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references
        if contract.kind == "bank_deposit":
            creditor_deposit_ids.append(cid)

    if creditor_deposit_ids:
        creditor_bank_id = system.state.contracts[creditor_deposit_ids[0]].liability_issuer_id
    else:
        creditor_bank_id = debtor_bank_id

    if not debtor_bank_id or not creditor_bank_id:
        return 0

    try:
        client_payment(system, debtor_id, debtor_bank_id, creditor_id, creditor_bank_id, pay_amount)
        return pay_amount
    except ValidationError:
        return 0


def _pay_with_cash(system, debtor_id, creditor_id, amount) -> int:
    """Pay using cash. Returns amount actually paid."""
    debtor_cash_ids = []
    for cid in system.state.agents[debtor_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references (contract may have been removed)
        if contract.kind == "cash":
            debtor_cash_ids.append(cid)

    if not debtor_cash_ids:
        return 0

    available = sum(system.state.contracts[cid].amount for cid in debtor_cash_ids)
    if available == 0:
        return 0

    pay_amount = min(amount, available)

    try:
        system.transfer_cash(debtor_id, creditor_id, pay_amount)
        return pay_amount
    except ValidationError:
        return 0


def _pay_bank_to_bank_with_reserves(system, debtor_bank_id, creditor_bank_id, amount) -> int:
    """Pay using reserves between banks. Returns amount actually paid."""
    if debtor_bank_id == creditor_bank_id:
        return 0

    debtor_reserve_ids = []
    for cid in system.state.agents[debtor_bank_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references
        if contract.kind == "reserve_deposit":
            debtor_reserve_ids.append(cid)

    if not debtor_reserve_ids:
        return 0

    available = sum(system.state.contracts[cid].amount for cid in debtor_reserve_ids)
    if available == 0:
        return 0

    pay_amount = min(amount, available)

    try:
        system.transfer_reserves(debtor_bank_id, creditor_bank_id, pay_amount)
        return pay_amount
    except ValidationError:
        return 0


def _deliver_stock(system, debtor_id, creditor_id, sku: str, required_quantity: int) -> int:
    """Transfer stock lots from debtor to creditor by SKU using FIFO allocation."""
    available_stocks = []
    for stock_id in system.state.agents[debtor_id].stock_ids:
        stock = system.state.stocks[stock_id]
        if stock.sku == sku:
            available_stocks.append((stock_id, stock.quantity))

    if not available_stocks:
        return 0

    total_available = sum(quantity for _, quantity in available_stocks)
    if total_available == 0:
        return 0

    deliver_quantity = min(required_quantity, total_available)
    remaining_to_deliver = deliver_quantity

    available_stocks.sort(key=lambda x: x[0])

    try:
        for stock_id, stock_quantity in available_stocks:
            if remaining_to_deliver == 0:
                break

            transfer_qty = min(remaining_to_deliver, stock_quantity)
            system._transfer_stock_internal(stock_id, debtor_id, creditor_id, transfer_qty)
            remaining_to_deliver -= transfer_qty

        return deliver_quantity
    except ValidationError:
        return 0


def _remove_contract(system, contract_id):
    """Remove contract from system and update agent registries."""
    contract = system.state.contracts.get(contract_id)
    if contract is None:
        return  # Already removed
    contract_kind = contract.kind
    contract_amount = getattr(contract, "amount", 0)

    # For secondary market transfers (e.g., payables sold to dealers),
    # remove from the effective holder, not the original asset_holder_id
    effective_holder_id = getattr(contract, 'effective_creditor', None) or contract.asset_holder_id
    effective_holder = system.state.agents.get(effective_holder_id)
    if effective_holder and contract_id in effective_holder.asset_ids:
        effective_holder.asset_ids.remove(contract_id)

    # Also check original asset_holder in case it wasn't transferred properly
    if effective_holder_id != contract.asset_holder_id:
        original_holder = system.state.agents.get(contract.asset_holder_id)
        if original_holder and contract_id in original_holder.asset_ids:
            original_holder.asset_ids.remove(contract_id)

    liability_issuer = system.state.agents[contract.liability_issuer_id]
    if contract_id in liability_issuer.liability_ids:
        liability_issuer.liability_ids.remove(contract_id)

    del system.state.contracts[contract_id]

    if contract_kind == "cash":
        system.state.cb_cash_outstanding -= contract_amount
    elif contract_kind == "reserve_deposit":
        system.state.cb_reserves_outstanding -= contract_amount


def _action_references_agent(action_dict, agent_id: str) -> bool:
    """Return True if the scheduled action references the given agent."""
    if not isinstance(action_dict, dict) or len(action_dict) != 1:
        return False

    action_name, payload = next(iter(action_dict.items()))
    if not isinstance(payload, dict):
        return False

    for field in _ACTION_AGENT_FIELDS.get(action_name, ()): 
        value = payload.get(field)
        if isinstance(value, str) and value == agent_id:
            return True
        if isinstance(value, list) and agent_id in value:
            return True
    return False


def _cancel_scheduled_actions_for_agent(
    system,
    agent_id: str,
    cancelled_contract_ids: set[str] | None = None,
    cancelled_aliases: set[str] | None = None,
) -> None:
    """Remove and log scheduled actions that involve a defaulted agent or cancelled contracts."""
    cancelled_contract_ids = cancelled_contract_ids or set()
    cancelled_aliases = cancelled_aliases or set()
    if not system.state.scheduled_actions_by_day:
        return

    for day, actions in list(system.state.scheduled_actions_by_day.items()):
        remaining = []
        for action_dict in actions:
            if _action_references_agent(action_dict, agent_id) or _action_references_contract(action_dict, cancelled_contract_ids, cancelled_aliases):
                action_name = next(iter(action_dict.keys()), "unknown") if isinstance(action_dict, dict) else "unknown"
                system.log(
                    "ScheduledActionCancelled",
                    agent=agent_id,
                    scheduled_day=day,
                    action=action_name,
                    mode=_get_default_mode(system),
                )
                continue
            remaining.append(action_dict)
        if remaining:
            system.state.scheduled_actions_by_day[day] = remaining
        else:
            del system.state.scheduled_actions_by_day[day]


def _action_references_contract(action_dict, contract_ids: set[str], aliases: set[str]) -> bool:
    if not isinstance(action_dict, dict) or len(action_dict) != 1:
        return False
    if not contract_ids and not aliases:
        return False

    action_name, payload = next(iter(action_dict.items()))
    if not isinstance(payload, dict):
        return False

    for field in _ACTION_CONTRACT_FIELDS.get(action_name, ("contract_id", "contract_alias", "alias")):
        value = payload.get(field)
        if isinstance(value, str) and (value in contract_ids or value in aliases):
            return True

    # common fallbacks
    for key in ("contract_id", "contract_alias", "alias"):
        value = payload.get(key)
        if isinstance(value, str) and (value in contract_ids or value in aliases):
            return True

    return False


def _reconnect_ring(system, defaulted_agent_id: str, successor_id: str, day: int) -> dict | None:
    """Reconnect the ring after an agent defaults.

    When agent X defaults, find the predecessor P (who owed X) and create a new
    payable from P to S (the successor, who X owed). The ring shrinks from N to N-1.

    Args:
        system: The System instance
        defaulted_agent_id: ID of the agent that just defaulted
        successor_id: ID of the agent the defaulted agent owed (ring successor)
        day: Current simulation day

    Returns:
        Dict with reconnection info, or None if reconnection not possible
    """
    from bilancio.domain.instruments.credit import Payable

    # Find predecessor: an active ring payable where asset_holder_id == defaulted_agent_id
    # (i.e., someone owes the defaulted agent)
    predecessor_payable = None
    for c in list(system.state.contracts.values()):
        if (c.kind == "payable"
            and c.asset_holder_id == defaulted_agent_id
            and c.liability_issuer_id != defaulted_agent_id
            and c.liability_issuer_id not in system.state.defaulted_agent_ids):
            predecessor_payable = c
            break

    if predecessor_payable is None:
        # No predecessor found (already defaulted or ring fully collapsed)
        return None

    predecessor_id = predecessor_payable.liability_issuer_id

    # Check if predecessor == successor (ring down to 2, one defaults → only 1 left)
    if predecessor_id == successor_id:
        # Ring collapsed to a single agent - remove orphaned payable, log collapse
        system.log(
            "RingCollapsed",
            defaulted_agent=defaulted_agent_id,
            remaining_agent=predecessor_id,
            removed_payable=predecessor_payable.id,
        )
        _remove_contract(system, predecessor_payable.id)
        return None

    # Get parameters from predecessor's payable
    amount = predecessor_payable.amount
    maturity_distance = getattr(predecessor_payable, 'maturity_distance', None)
    if maturity_distance is None:
        # Fallback: use due_day - current_day if available
        if predecessor_payable.due_day is not None:
            maturity_distance = max(1, predecessor_payable.due_day - day)
        else:
            maturity_distance = 1

    new_due_day = day + maturity_distance

    # Remove old payable: predecessor → defaulted agent
    old_payable_id = predecessor_payable.id
    _remove_contract(system, old_payable_id)

    # Create new payable: predecessor → successor
    new_payable = Payable(
        id=system.new_contract_id("PAY"),
        kind="payable",
        amount=amount,
        denom="X",
        asset_holder_id=successor_id,
        liability_issuer_id=predecessor_id,
        due_day=new_due_day,
        maturity_distance=maturity_distance,
    )
    system.add_contract(new_payable)

    # Log events
    system.log(
        "RingReconnected",
        defaulted_agent=defaulted_agent_id,
        predecessor=predecessor_id,
        successor=successor_id,
        old_payable=old_payable_id,
        new_payable=new_payable.id,
        amount=amount,
        maturity_distance=maturity_distance,
        new_due_day=new_due_day,
    )

    system.log(
        "PayableCreated",
        contract_id=new_payable.id,
        debtor=predecessor_id,
        creditor=successor_id,
        amount=amount,
        due_day=new_due_day,
        maturity_distance=maturity_distance,
        reason="ring_reconnection",
    )

    return {
        "predecessor": predecessor_id,
        "successor": successor_id,
        "old_payable": old_payable_id,
        "new_payable": new_payable.id,
        "amount": amount,
        "maturity_distance": maturity_distance,
        "new_due_day": new_due_day,
    }


def _expel_agent(
    system,
    agent_id: str,
    *,
    trigger_contract_id: str | None = None,
    trigger_kind: str | None = None,
    trigger_shortfall: int | None = None,
    cancelled_contract_ids: set[str] | None = None,
    cancelled_aliases: set[str] | None = None,
) -> None:
    """Mark an agent as defaulted, write off obligations, and cancel future actions."""
    if _get_default_mode(system) != DEFAULT_MODE_EXPEL:
        return

    if agent_id in system.state.defaulted_agent_ids:
        return

    agent = system.state.agents.get(agent_id)
    if agent is None:
        return

    if agent.kind == "central_bank":
        raise DefaultError("Central bank cannot default")

    agent.defaulted = True
    system.state.defaulted_agent_ids.add(agent_id)

    system.log(
        "AgentDefaulted",
        agent=agent_id,
        frm=agent_id,
        trigger_contract=trigger_contract_id,
        contract_kind=trigger_kind,
        shortfall=trigger_shortfall,
        mode=_get_default_mode(system),
    )

    cancelled_contract_ids = set(cancelled_contract_ids or [])
    cancelled_aliases = set(cancelled_aliases or [])

    # Remove any aliases provided for already-cancelled contracts
    for alias in list(cancelled_aliases):
        system.state.aliases.pop(alias, None)

    for cid, contract in list(system.state.contracts.items()):
        if contract.liability_issuer_id != agent_id:
            continue
        if trigger_contract_id and cid == trigger_contract_id:
            continue

        alias = get_alias_for_id(system, cid)
        if alias:
            cancelled_aliases.add(alias)
        payload = {
            "contract_id": cid,
            "alias": alias,
            "debtor": contract.liability_issuer_id,
            "creditor": contract.asset_holder_id,
            "contract_kind": contract.kind,
            "amount": getattr(contract, "amount", None),
            "due_day": getattr(contract, "due_day", None),
        }
        if hasattr(contract, "sku"):
            payload["sku"] = getattr(contract, "sku")
        if payload.get("due_day") is None:
            payload.pop("due_day", None)

        system.log("ObligationWrittenOff", **payload)
        _remove_contract(system, cid)
        cancelled_contract_ids.add(cid)
        if alias:
            system.state.aliases.pop(alias, None)

    _cancel_scheduled_actions_for_agent(system, agent_id, cancelled_contract_ids, cancelled_aliases)

    # If every non-central-bank agent has defaulted, halt the simulation with a DefaultError.
    if all(
        (ag.kind == "central_bank") or getattr(ag, "defaulted", False)
        for ag in system.state.agents.values()
    ):
        raise DefaultError("All non-central-bank agents have defaulted")


def settle_due_delivery_obligations(system, day: int):
    """Settle all delivery obligations due today using stock operations."""
    for obligation in list(due_delivery_obligations(system, day)):
        if obligation.id not in system.state.contracts:
            continue

        debtor = system.state.agents[obligation.liability_issuer_id]
        if getattr(debtor, "defaulted", False):
            continue

        creditor = system.state.agents[obligation.asset_holder_id]
        required_sku = obligation.sku
        required_quantity = obligation.amount

        with atomic(system):
            delivered_quantity = _deliver_stock(system, debtor.id, creditor.id, required_sku, required_quantity)

            if delivered_quantity != required_quantity:
                shortage = required_quantity - delivered_quantity
                if _get_default_mode(system) == DEFAULT_MODE_FAIL_FAST:
                    raise DefaultError(
                        f"Insufficient stock to settle delivery obligation {obligation.id}: {shortage} units of {required_sku} still owed"
                    )

                alias = get_alias_for_id(system, obligation.id)
                cancelled_contract_ids = {obligation.id}
                cancelled_aliases = {alias} if alias else set()
                if delivered_quantity > 0:
                    system.log(
                        "PartialSettlement",
                        contract_id=obligation.id,
                        alias=alias,
                        debtor=debtor.id,
                        creditor=creditor.id,
                        contract_kind=obligation.kind,
                        settlement_kind="delivery",
                        delivered_quantity=delivered_quantity,
                        required_quantity=required_quantity,
                        shortfall=shortage,
                        sku=required_sku,
                    )

                system.log(
                    "ObligationDefaulted",
                    contract_id=obligation.id,
                    alias=alias,
                    debtor=debtor.id,
                    creditor=creditor.id,
                    contract_kind=obligation.kind,
                    shortfall=shortage,
                    delivered_quantity=delivered_quantity,
                    required_quantity=required_quantity,
                    sku=required_sku,
                    qty=shortage,
                )

                _remove_contract(system, obligation.id)
                _expel_agent(
                    system,
                    debtor.id,
                    trigger_contract_id=obligation.id,
                    trigger_kind=obligation.kind,
                    trigger_shortfall=shortage,
                    cancelled_contract_ids=cancelled_contract_ids,
                    cancelled_aliases=cancelled_aliases,
                )
                continue

            system._cancel_delivery_obligation_internal(obligation.id)
            alias = get_alias_for_id(system, obligation.id)
            system.log(
                "DeliveryObligationSettled",
                obligation_id=obligation.id,
                contract_id=obligation.id,
                alias=alias,
                debtor=debtor.id,
                creditor=creditor.id,
                sku=required_sku,
                qty=required_quantity,
            )


def settle_due(system, day: int, *, rollover_enabled: bool = False):
    """Settle all obligations due today (payables and delivery obligations).

    Args:
        system: The system to settle
        day: Current simulation day
        rollover_enabled: If True, create new payables for successfully settled ones (Plan 024)

    Returns:
        List of settled payable info for rollover: [(debtor_id, creditor_id, amount, maturity_distance)]
    """
    settled_for_rollover = []
    risk_assessor = _get_risk_assessor(system)

    for payable in list(due_payables(system, day)):
        if payable.id not in system.state.contracts:
            continue

        debtor = system.state.agents[payable.liability_issuer_id]
        if getattr(debtor, "defaulted", False):
            continue

        # Use effective_creditor to handle secondary market transfers
        # (holder_id if transferred, otherwise original asset_holder_id)
        creditor_id = payable.effective_creditor
        creditor = system.state.agents[creditor_id]
        order = system.policy.settlement_order(debtor)

        remaining = payable.amount
        payments_summary: list[dict] = []

        # Save payable info before potential removal
        payable_amount = payable.amount
        payable_maturity_distance = getattr(payable, 'maturity_distance', None)
        original_creditor = payable.asset_holder_id  # Original creditor for rollover

        with atomic(system):
            for method in order:
                if remaining == 0:
                    break

                if method == "bank_deposit":
                    paid_now = _pay_with_deposits(system, debtor.id, creditor.id, remaining)
                elif method == "cash":
                    paid_now = _pay_with_cash(system, debtor.id, creditor.id, remaining)
                elif method == "reserve_deposit":
                    paid_now = _pay_bank_to_bank_with_reserves(system, debtor.id, creditor.id, remaining)
                else:
                    raise ValidationError(f"unknown payment method {method}")

                remaining -= paid_now
                if paid_now > 0:
                    payments_summary.append({"method": method, "amount": paid_now})

            if remaining != 0:
                if _get_default_mode(system) == DEFAULT_MODE_FAIL_FAST:
                    raise DefaultError(f"Insufficient funds to settle payable {payable.id}: {remaining} still owed")

                alias = get_alias_for_id(system, payable.id)
                cancelled_contract_ids = {payable.id}
                cancelled_aliases = {alias} if alias else set()
                amount_paid = payable.amount - remaining

                if amount_paid > 0:
                    payload = {
                        "contract_id": payable.id,
                        "alias": alias,
                        "debtor": debtor.id,
                        "creditor": creditor.id,
                        "contract_kind": payable.kind,
                        "settlement_kind": "payable",
                        "amount_paid": amount_paid,
                        "shortfall": remaining,
                        "original_amount": payable.amount,
                    }
                    if payments_summary:
                        payload["distribution"] = payments_summary
                    system.log("PartialSettlement", **payload)

                system.log(
                    "ObligationDefaulted",
                    contract_id=payable.id,
                    alias=alias,
                    debtor=debtor.id,
                    creditor=creditor.id,
                    contract_kind=payable.kind,
                    shortfall=remaining,
                    amount_paid=amount_paid,
                    original_amount=payable.amount,
                    amount=remaining,
                )

                # Capture ring successor before payable removal (for reconnection)
                ring_successor_id = payable.asset_holder_id

                _remove_contract(system, payable.id)
                _expel_agent(
                    system,
                    debtor.id,
                    trigger_contract_id=payable.id,
                    trigger_kind=payable.kind,
                    trigger_shortfall=remaining,
                    cancelled_contract_ids=cancelled_contract_ids,
                    cancelled_aliases=cancelled_aliases,
                )
                if risk_assessor:
                    risk_assessor.update_history(
                        day=day,
                        issuer_id=debtor.id,
                        defaulted=True,
                    )
                # Reconnect ring if rollover is enabled
                if rollover_enabled:
                    _reconnect_ring(system, debtor.id, ring_successor_id, day)
                continue

            _remove_contract(system, payable.id)
            alias = get_alias_for_id(system, payable.id)
            system.log(
                "PayableSettled",
                pid=payable.id,
                contract_id=payable.id,
                alias=alias,
                debtor=debtor.id,
                creditor=creditor.id,
                amount=payable_amount,
            )
            if risk_assessor:
                risk_assessor.update_history(
                    day=day,
                    issuer_id=debtor.id,
                    defaulted=False,
                )

            # Plan 024: Track for rollover (only if successfully settled AND rollover enabled)
            if rollover_enabled and payable_maturity_distance is not None:
                # Use original creditor for rollover, not secondary market holder
                settled_for_rollover.append((
                    debtor.id,
                    original_creditor,
                    payable_amount,
                    payable_maturity_distance,
                ))

    settle_due_delivery_obligations(system, day)

    return settled_for_rollover


def rollover_settled_payables(system, day: int, settled_payables: list, dealer_active: bool = False):
    """Create new payables for successfully settled ones (continuous issuance via rollover).

    Per PDF specification (Plan 024):
    - Each liability records its original maturity distance ΔT
    - When a claim is repaid, borrower immediately issues new claim of same size/ΔT
    - Due date = max(all current due_days in system) + ΔT
      This queues the new payable after the last existing maturity,
      preventing synchronized settlement waves.

    Model C (when dealer_active=True):
    - Trader payables: No cash transfer (debt represents ongoing real economic relationships)
    - VBT/Dealer payables: Cash transfer maintained (financial intermediaries who lend money)

    Args:
        system: The system
        day: Current simulation day
        settled_payables: List of (debtor_id, creditor_id, amount, maturity_distance)
        dealer_active: If True, use Model C (no cash transfer for trader payables)

    Returns:
        List of new payable IDs created by rollover
    """
    from bilancio.domain.instruments.credit import Payable

    # Compute max due_day across all current payables in the system.
    # Rolled-over payables queue after this, preventing synchronized waves.
    max_due_day = day  # Floor: at minimum, use current day
    for c in system.state.contracts.values():
        if c.kind == "payable":
            due = getattr(c, "due_day", None)
            if due is not None and due > max_due_day:
                max_due_day = due

    new_payable_ids = []
    for debtor_id, creditor_id, amount, maturity_distance in settled_payables:
        # Skip if either party has defaulted
        debtor = system.state.agents.get(debtor_id)
        creditor = system.state.agents.get(creditor_id)

        if debtor is None or getattr(debtor, "defaulted", False):
            continue
        if creditor is None or getattr(creditor, "defaulted", False):
            continue

        new_due_day = max_due_day + maturity_distance

        with atomic(system):
            # 1. Create new payable with same amount and ΔT
            new_payable = Payable(
                id=system.new_contract_id("PAY"),
                kind="payable",
                amount=amount,
                denom="X",
                asset_holder_id=creditor_id,  # creditor holds the asset
                liability_issuer_id=debtor_id,  # debtor issues the liability
                due_day=new_due_day,
                maturity_distance=maturity_distance,
            )
            system.add_contract(new_payable)
            new_payable_ids.append(new_payable.id)

            # Model C: no cash transfer for real-economy relationships
            # Model A: cash transfer for financial intermediaries (VBT/Dealer)
            is_financial_creditor = creditor_id.startswith("vbt_") or creditor_id.startswith("dealer_")
            if not dealer_active or is_financial_creditor:
                # 2. Transfer cash from creditor to debtor (new issuance)
                cash_transferred = _pay_with_cash(system, creditor_id, debtor_id, amount)
            else:
                # Model C: skip cash transfer for trader payables
                cash_transferred = 0

            if cash_transferred != amount and (not dealer_active or is_financial_creditor):
                # Creditor doesn't have enough cash - this shouldn't happen in normal rollover
                # but handle gracefully
                system.log(
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
                system.log(
                    "PayableRolledOver",
                    debtor=debtor_id,
                    creditor=creditor_id,
                    amount=amount,
                    new_due_day=new_due_day,
                    maturity_distance=maturity_distance,
                    payable_id=new_payable.id,
                    cash_transfer=(not dealer_active or is_financial_creditor),
                )

    return new_payable_ids
