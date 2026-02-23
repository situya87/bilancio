"""Settlement engine (Phase B) for settling payables due today."""

from __future__ import annotations

import logging
from collections.abc import Generator
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bilancio.core.atomic_tx import atomic
from bilancio.core.errors import DefaultError, SimulationHalt, ValidationError
from bilancio.core.events import EventKind
from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.instruments.base import Instrument, InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.domain.instruments.non_bank_loan import NonBankLoan
from bilancio.ops.aliases import get_alias_for_id
from bilancio.ops.banking import client_payment

if TYPE_CHECKING:
    from bilancio.engines.banking_subsystem import BankingSubsystem
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)

DEFAULT_MODE_FAIL_FAST = "fail-fast"
DEFAULT_MODE_EXPEL = "expel-agent"

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


def _get_default_mode(system: System) -> str:
    """Return the configured default-handling mode for the system."""
    return system.default_mode


def _get_risk_assessor(system: System) -> Any:
    subsystem = system.state.dealer_subsystem
    return subsystem.risk_assessor if subsystem is not None else None


def _update_risk_history(system: System, day: int, issuer_id: str, defaulted: bool) -> None:
    """Update risk assessor history, including per-trader assessors.

    Propagates the observation to both the shared risk_assessor and all
    per-trader assessors.  Per-trader assessors receive the same raw history;
    information friction is applied via ``default_observability`` inside
    ``estimate_default_prob``, not by filtering ``update_history`` calls.
    """
    risk_assessor = _get_risk_assessor(system)
    if risk_assessor:
        risk_assessor.update_history(day=day, issuer_id=issuer_id, defaulted=defaulted)
    # Propagate to per-trader assessors
    dealer_sub = system.state.dealer_subsystem
    if dealer_sub and dealer_sub.trader_assessors:
        for ta in dealer_sub.trader_assessors.values():
            ta.update_history(day=day, issuer_id=issuer_id, defaulted=defaulted)


def due_payables(system: System, day: int) -> Generator[Instrument, None, None]:
    """Look up payables with due_day == day using the due-day index."""
    for cid in list(system.state.contracts_by_due_day.get(day, ())):
        c = system.state.contracts.get(cid)
        if c is not None and c.kind == InstrumentKind.PAYABLE:
            yield c


def due_delivery_obligations(system: System, day: int) -> Generator[Instrument, None, None]:
    """Look up delivery obligations with due_day == day using the due-day index."""
    for cid in list(system.state.contracts_by_due_day.get(day, ())):
        c = system.state.contracts.get(cid)
        if c is not None and c.kind == InstrumentKind.DELIVERY_OBLIGATION:
            yield c


def _pay_with_deposits(system: System, debtor_id: str, creditor_id: str, amount: int) -> int:
    """Pay using bank deposits. Returns amount actually paid.

    When a banking subsystem is active, routes payments:
    - Debtor pays from the bank with the lowest r_D (min opportunity cost)
    - Creditor receives at the bank with the highest r_D
    """
    logger.debug(
        "_pay_with_deposits: debtor=%s creditor=%s amount=%d", debtor_id, creditor_id, amount
    )
    debtor_deposit_ids = []
    for cid in system.state.agents[debtor_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references
        if contract.kind == InstrumentKind.BANK_DEPOSIT:
            debtor_deposit_ids.append(cid)

    if not debtor_deposit_ids:
        return 0

    available = sum(system.state.contracts[cid].amount for cid in debtor_deposit_ids)
    if available == 0:
        return 0

    pay_amount = min(amount, available)

    # Determine banking subsystem for routing
    banking_sub = getattr(system.state, "banking_subsystem", None)

    if banking_sub is not None:
        # Multi-bank split payment: pay from banks sorted by ascending r_D
        creditor_bank_id = _select_receive_bank(system, banking_sub, creditor_id)
        if not creditor_bank_id:
            return 0

        # Group deposits by bank with balances
        bank_balances: dict[str, int] = {}
        for cid in debtor_deposit_ids:
            contract = system.state.contracts.get(cid)
            if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
                bid = contract.liability_issuer_id
                bank_balances[bid] = bank_balances.get(bid, 0) + contract.amount

        # Sort by ascending r_D (lowest opportunity cost first)
        sorted_banks = []
        for bid, balance in bank_balances.items():
            if balance <= 0:
                continue
            bank_state = banking_sub.banks.get(bid)
            r_d = Decimal("0")
            if bank_state and bank_state.current_quote:
                r_d = bank_state.current_quote.deposit_rate
            sorted_banks.append((r_d, bid, balance))
        sorted_banks.sort(key=lambda x: x[0])

        # Pay from each bank until full amount covered
        total_paid = 0
        remaining = pay_amount
        for _, bid, balance in sorted_banks:
            if remaining <= 0:
                break
            chunk = min(balance, remaining)
            try:
                client_payment(system, debtor_id, bid, creditor_id, creditor_bank_id, chunk)
                total_paid += chunk
                remaining -= chunk
            except ValidationError:
                continue
        return total_paid
    else:
        # Original behavior: use first available deposit's bank
        debtor_bank_id = None
        creditor_bank_id = None

        if debtor_deposit_ids:
            debtor_bank_id = system.state.contracts[debtor_deposit_ids[0]].liability_issuer_id

        creditor_deposit_ids = []
        for cid in system.state.agents[creditor_id].asset_ids:
            contract = system.state.contracts.get(cid)
            if contract is None:
                continue
            if contract.kind == InstrumentKind.BANK_DEPOSIT:
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


def _select_pay_bank(
    system: System,
    banking_sub: BankingSubsystem,
    debtor_id: str,
    debtor_deposit_ids: list[str],
    amount: int,
) -> str | None:
    """Select the debtor's bank with the lowest deposit rate (r_D) for payment.

    Minimises the opportunity cost of drawing down deposits: the payer
    withdraws from the bank where deposits earn the *least*.

    Algorithm:
        1. Group the debtor's deposit contracts by issuing bank.
        2. Among banks whose balance covers *amount*, pick the one
           with the lowest r_D.
        3. If no single bank suffices, fall back to the first bank
           with any positive balance (partial payment will follow).

    Args:
        system: Main system state.
        banking_sub: Active banking subsystem (provides r_D quotes).
        debtor_id: Agent paying.
        debtor_deposit_ids: Pre-filtered list of the debtor's BANK_DEPOSIT
            contract IDs.
        amount: Target payment amount.

    Returns:
        The chosen bank_id, or ``None`` if the debtor has no deposits.
    """
    from decimal import Decimal

    # Group deposits by bank
    bank_balances: dict[str, int] = {}
    for cid in debtor_deposit_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
            bid = contract.liability_issuer_id
            bank_balances[bid] = bank_balances.get(bid, 0) + contract.amount

    candidates = []
    for bid, balance in bank_balances.items():
        if balance >= amount:
            bank_state = banking_sub.banks.get(bid)
            r_d = Decimal("0")
            if bank_state and bank_state.current_quote:
                r_d = bank_state.current_quote.deposit_rate
            candidates.append((r_d, balance, bid))

    if candidates:
        candidates.sort(key=lambda x: x[0])  # Lowest r_D first
        return candidates[0][2]

    # No single bank has enough — fall back to first bank with any balance
    for bid, balance in bank_balances.items():
        if balance > 0:
            return bid
    return None


def _select_receive_bank(
    system: System,
    banking_sub: BankingSubsystem,
    creditor_id: str,
) -> str | None:
    """Select the creditor's bank with the highest deposit rate (r_D).

    Maximises the return on the received deposit: the payee's incoming
    payment is credited to the bank that pays the *most* on deposits.

    Algorithm:
        1. Scan the creditor's existing BANK_DEPOSIT contracts.
        2. For each issuing bank, look up its current r_D quote.
        3. Return the bank with the highest r_D.
        4. Fall back to any deposit-issuing bank, or the subsystem's
           ``best_deposit_bank`` routing.

    Args:
        system: Main system state.
        banking_sub: Active banking subsystem (provides r_D quotes).
        creditor_id: Agent receiving payment.

    Returns:
        The chosen bank_id, or ``None`` if no bank can be determined.
    """
    from decimal import Decimal

    agent = system.state.agents.get(creditor_id)
    if agent is None:
        return None

    best_rate = Decimal("-1")
    best_bank = None

    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
            bid = contract.liability_issuer_id
            bank_state = banking_sub.banks.get(bid)
            r_d = Decimal("0")
            if bank_state and bank_state.current_quote:
                r_d = bank_state.current_quote.deposit_rate
            if r_d > best_rate:
                best_rate = r_d
                best_bank = bid

    if best_bank is not None:
        return best_bank

    # Creditor has no deposits — use first available bank from any deposit
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
            return contract.liability_issuer_id

    # No deposits at all — try to get bank from banking subsystem
    return banking_sub.best_deposit_bank(creditor_id)


def _pay_with_cash(system: System, debtor_id: str, creditor_id: str, amount: int) -> int:
    """Pay using cash. Returns amount actually paid."""
    logger.debug("_pay_with_cash: debtor=%s creditor=%s amount=%d", debtor_id, creditor_id, amount)
    debtor_cash_ids = []
    for cid in system.state.agents[debtor_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references (contract may have been removed)
        if contract.kind == InstrumentKind.CASH:
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


def _pay_bank_to_bank_with_reserves(
    system: System, debtor_bank_id: str, creditor_bank_id: str, amount: int
) -> int:
    """Pay using reserves between banks. Returns amount actually paid."""
    logger.debug(
        "_pay_bank_to_bank_with_reserves: debtor=%s creditor=%s amount=%d",
        debtor_bank_id,
        creditor_bank_id,
        amount,
    )
    if debtor_bank_id == creditor_bank_id:
        return 0

    debtor_reserve_ids = []
    for cid in system.state.agents[debtor_bank_id].asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue  # Skip stale references
        if contract.kind == InstrumentKind.RESERVE_DEPOSIT:
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


def _deliver_stock(
    system: System, debtor_id: str, creditor_id: str, sku: str, required_quantity: int
) -> int:
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


def _remove_contract(system: System, contract_id: str) -> None:
    """Remove contract from system and update agent registries."""
    contract = system.state.contracts.get(contract_id)
    if contract is None:
        return  # Already removed
    contract_kind = contract.kind
    contract_amount = contract.amount
    logger.debug("removing contract %s (kind=%s)", contract_id, contract_kind)

    # Maintain due_day index
    due_day_val = contract.due_day
    if due_day_val is not None:
        bucket = system.state.contracts_by_due_day.get(due_day_val)
        if bucket:
            try:
                bucket.remove(contract_id)
            except ValueError:
                pass
            if not bucket:
                del system.state.contracts_by_due_day[due_day_val]

    # For secondary market transfers (e.g., payables sold to dealers),
    # remove from the effective holder, not the original asset_holder_id
    effective_holder_id = (
        contract.effective_creditor if isinstance(contract, Payable) else contract.asset_holder_id
    )
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

    if contract_kind == InstrumentKind.CASH:
        system.state.cb_cash_outstanding -= contract_amount
    elif contract_kind == InstrumentKind.RESERVE_DEPOSIT:
        system.state.cb_reserves_outstanding -= contract_amount
    elif contract_kind == InstrumentKind.CB_LOAN:
        system.state.cb_loans_outstanding -= contract_amount


def _action_references_agent(action_dict: object, agent_id: str) -> bool:
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
    system: System,
    agent_id: str,
    cancelled_contract_ids: set[str] | None = None,
    cancelled_aliases: set[str] | None = None,
) -> None:
    """Remove and log scheduled actions that involve a defaulted agent or cancelled contracts."""
    cancelled_contract_ids = cancelled_contract_ids or set()
    cancelled_aliases = cancelled_aliases or set()
    if not system.state.scheduled_actions_by_day:
        return
    logger.debug("_cancel_scheduled_actions_for_agent: agent=%s", agent_id)

    for day, actions in list(system.state.scheduled_actions_by_day.items()):
        remaining = []
        for action_dict in actions:
            if _action_references_agent(action_dict, agent_id) or _action_references_contract(
                action_dict, cancelled_contract_ids, cancelled_aliases
            ):
                action_name = (
                    next(iter(action_dict.keys()), "unknown")
                    if isinstance(action_dict, dict)
                    else "unknown"
                )
                system.log(
                    EventKind.SCHEDULED_ACTION_CANCELLED,
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


def _action_references_contract(
    action_dict: object, contract_ids: set[str], aliases: set[str]
) -> bool:
    if not isinstance(action_dict, dict) or len(action_dict) != 1:
        return False
    if not contract_ids and not aliases:
        return False

    action_name, payload = next(iter(action_dict.items()))
    if not isinstance(payload, dict):
        return False

    for field in _ACTION_CONTRACT_FIELDS.get(
        action_name, ("contract_id", "contract_alias", "alias")
    ):
        value = payload.get(field)
        if isinstance(value, str) and (value in contract_ids or value in aliases):
            return True

    # common fallbacks
    for key in ("contract_id", "contract_alias", "alias"):
        value = payload.get(key)
        if isinstance(value, str) and (value in contract_ids or value in aliases):
            return True

    return False


def _reconnect_ring(
    system: System, defaulted_agent_id: str, successor_id: str, day: int
) -> dict[str, object] | None:
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
    # Find predecessor: an active ring payable where asset_holder_id == defaulted_agent_id
    # (i.e., someone owes the defaulted agent).
    # NOTE: We intentionally use asset_holder_id (original creditor), NOT effective_creditor,
    # because the ring topology is defined by original issuance. If the payable was traded to
    # a dealer in secondary market, effective_creditor would be the dealer — but the ring
    # reconnection must still link the original predecessor to the successor.
    predecessor_payable: Payable | None = None
    for c in list(system.state.contracts.values()):
        if (
            c.kind == InstrumentKind.PAYABLE
            and isinstance(c, Payable)
            and c.asset_holder_id == defaulted_agent_id
            and c.liability_issuer_id != defaulted_agent_id
            and c.liability_issuer_id not in system.state.defaulted_agent_ids
        ):
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
            EventKind.RING_COLLAPSED,
            defaulted_agent=defaulted_agent_id,
            remaining_agent=predecessor_id,
            removed_payable=predecessor_payable.id,
        )
        _remove_contract(system, predecessor_payable.id)
        return None

    # Get parameters from predecessor's payable
    amount = predecessor_payable.amount
    maturity_distance = predecessor_payable.maturity_distance
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
        kind=InstrumentKind.PAYABLE,
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
        EventKind.RING_RECONNECTED,
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
        EventKind.PAYABLE_CREATED,
        contract_id=new_payable.id,
        debtor=predecessor_id,
        creditor=successor_id,
        amount=amount,
        due_day=new_due_day,
        maturity_distance=maturity_distance,
        reason="ring_reconnection",
    )

    logger.info(
        "ring reconnected: %s -> %s (was %s)", predecessor_id, successor_id, defaulted_agent_id
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


def _distribute_pro_rata_recovery(system: System, agent_id: str) -> None:
    """Distribute a defaulting agent's remaining liquid assets to creditors proportionally.

    Before liabilities are written off, any cash and bank deposits the agent
    still holds are divided among all creditors (payable holders and non-bank
    loan lenders) in proportion to each creditor's outstanding claim.
    Deposits are used first, then cash for the remainder.

    Args:
        system: The System instance
        agent_id: ID of the defaulting agent whose liquid assets will be distributed
    """
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return

    # Sum all CASH and BANK_DEPOSIT instruments held by the defaulting agent
    total_liquid = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT):
            total_liquid += contract.amount

    if total_liquid <= 0:
        return

    # Collect all outstanding liabilities (payables + NonBankLoans)
    claims: list[tuple[str, int]] = []  # (creditor_id, claim_amount)
    for contract in system.state.contracts.values():
        if contract.liability_issuer_id != agent_id:
            continue
        if contract.kind == InstrumentKind.PAYABLE and isinstance(contract, Payable):
            creditor_id = contract.effective_creditor
            claims.append((creditor_id, contract.amount))
        elif contract.kind == InstrumentKind.NON_BANK_LOAN and isinstance(contract, NonBankLoan):
            creditor_id = contract.asset_holder_id
            claims.append((creditor_id, contract.repayment_amount))

    total_claims = sum(amount for _, amount in claims)
    if total_claims <= 0:
        return

    # Distribute liquid assets pro-rata to each creditor (round to avoid truncation loss)
    recovery_details: list[dict[str, object]] = []
    total_distributed = 0
    for creditor_id, claim_amount in claims:
        share = round((claim_amount / total_claims) * total_liquid)
        if share <= 0:
            continue
        transferred = _pay_with_deposits(system, agent_id, creditor_id, share)
        if transferred < share:
            try:
                system.transfer_cash(
                    from_agent_id=agent_id, to_agent_id=creditor_id, amount=share - transferred
                )
                transferred = share
            except ValidationError:
                pass  # agent may not have enough cash either
        if transferred > 0:
            total_distributed += transferred
            recovery_details.append(
                {
                    "creditor": creditor_id,
                    "claim": claim_amount,
                    "recovery": transferred,
                }
            )

    if total_distributed > 0:
        system.log(
            EventKind.PRO_RATA_RECOVERY,
            agent=agent_id,
            total_liquid=total_liquid,
            total_claims=total_claims,
            total_distributed=total_distributed,
            num_creditors=len(recovery_details),
            details=recovery_details,
        )
        logger.info(
            "pro-rata recovery: agent=%s distributed=%d/%d to %d creditors",
            agent_id,
            total_distributed,
            total_liquid,
            len(recovery_details),
        )


def _consume_reserves_from_bank(system: System, bank_id: str, amount: int) -> None:
    """Consume reserve instruments from a bank and update CB counters.

    Unlike _remove_contract, consume() does NOT update cb_reserves_outstanding,
    so we must do it explicitly here.
    """
    from bilancio.ops.primitives import consume

    remaining = amount
    reserve_ids = [
        cid
        for cid in system.state.agents[bank_id].asset_ids
        if cid in system.state.contracts
        and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    ]

    for cid in list(reserve_ids):
        if remaining <= 0:
            break
        instr = system.state.contracts.get(cid)
        if instr is None:
            continue
        take = min(instr.amount, remaining)
        consume(system, cid, take)
        remaining -= take

    # Update CB reserves counter (consume doesn't do this)
    system.state.cb_reserves_outstanding -= (amount - remaining)


def _find_surviving_bank(system: System, depositor_id: str, failed_bank_id: str) -> str | None:
    """Find a non-defaulted bank where the depositor has an existing deposit."""
    agent = system.state.agents.get(depositor_id)
    if agent is None:
        return None

    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (
            contract
            and contract.kind == InstrumentKind.BANK_DEPOSIT
            and contract.liability_issuer_id != failed_bank_id
        ):
            # Check that this bank hasn't defaulted
            bank_agent = system.state.agents.get(contract.liability_issuer_id)
            if bank_agent and not bank_agent.defaulted:
                return contract.liability_issuer_id

    return None


def _create_resolution_deposit(system: System, depositor_id: str, bank_id: str, amount: int) -> None:
    """Create or increase a deposit for a depositor at a receiving bank.

    Mirrors bank_lending.py:_increase_deposit.
    """
    from bilancio.engines.bank_lending import _increase_deposit
    _increase_deposit(system, depositor_id, bank_id, amount)


def _resolve_to_cash(system: System, bank_id: str, depositor_id: str, amount: int) -> None:
    """Convert bank reserves to cash and transfer to depositor."""
    system.convert_reserves_to_cash(bank_id, amount)
    system.transfer_cash(bank_id, depositor_id, amount)


def _resolve_failed_bank(system: System, bank_id: str) -> None:
    """Resolve a failed bank by distributing reserves to claimants.

    Resolution procedure:
    1. Collect bank's total reserves (RESERVE_DEPOSIT assets)
    2. Cancel CB loans against reserves (CB claims have priority)
    3. Distribute remaining reserves to depositors pro-rata:
       - If depositor has a surviving bank: transfer reserves + create deposit
       - If all depositor's banks failed: convert reserves to cash + transfer
    4. Clean up banking subsystem state
    """
    from bilancio.engines.banking_subsystem import _get_bank_reserves

    agent = system.state.agents.get(bank_id)
    if agent is None:
        return

    # 1. Calculate total reserves
    total_reserves = _get_bank_reserves(system, bank_id)
    if total_reserves <= 0:
        system.log(
            "BankResolutionCompleted",
            bank_id=bank_id,
            total_reserves=0,
            cb_claims_cancelled=0,
            depositor_distributions=0,
        )
        return

    remaining_reserves = total_reserves
    cb_claims_cancelled = 0

    # 2. CB loan claims first - cancel reserves against outstanding CB loans
    cb_loans = [
        (cid, contract)
        for cid, contract in list(system.state.contracts.items())
        if contract.kind == InstrumentKind.CB_LOAN
        and isinstance(contract, CBLoan)
        and contract.liability_issuer_id == bank_id
    ]

    for loan_id, loan in cb_loans:
        if remaining_reserves <= 0:
            break
        if loan_id not in system.state.contracts:
            continue

        cancel_amount = min(loan.amount, remaining_reserves)

        # Consume reserves from bank
        _consume_reserves_from_bank(system, bank_id, cancel_amount)
        remaining_reserves -= cancel_amount

        # Remove the CB loan
        _remove_contract(system, loan_id)
        cb_claims_cancelled += cancel_amount

        system.log(
            "BankResolutionCBClaimCancelled",
            bank_id=bank_id,
            loan_id=loan_id,
            loan_amount=loan.amount,
            cancelled_amount=cancel_amount,
        )

    # 3. Distribute remaining reserves to depositors pro-rata
    depositor_distributions = []

    if remaining_reserves > 0:
        # Collect depositor claims (BankDeposit where bank is liability issuer)
        deposit_claims: list[tuple[str, int]] = []  # (depositor_id, amount)
        for cid, contract in system.state.contracts.items():
            if (
                contract.kind == InstrumentKind.BANK_DEPOSIT
                and contract.liability_issuer_id == bank_id
            ):
                deposit_claims.append((contract.asset_holder_id, contract.amount))

        total_deposit_claims = sum(amount for _, amount in deposit_claims)

        if total_deposit_claims > 0:
            total_distributed = 0
            for depositor_id, claim_amount in deposit_claims:
                # Pro-rata share, with rounding guard
                share = round((claim_amount / total_deposit_claims) * remaining_reserves)
                share = min(share, remaining_reserves - total_distributed)
                if share <= 0:
                    continue

                # Find a surviving bank for this depositor
                surviving_bank_id = _find_surviving_bank(system, depositor_id, bank_id)

                if surviving_bank_id is not None:
                    # Transfer reserves to surviving bank + create deposit
                    try:
                        system.transfer_reserves(bank_id, surviving_bank_id, share)
                        _create_resolution_deposit(system, depositor_id, surviving_bank_id, share)
                        total_distributed += share
                        depositor_distributions.append({
                            "depositor": depositor_id,
                            "amount": share,
                            "method": "deposit_at_surviving_bank",
                            "bank": surviving_bank_id,
                        })
                    except (ValidationError, Exception):
                        # Fallback to cash if transfer fails
                        try:
                            _resolve_to_cash(system, bank_id, depositor_id, share)
                            total_distributed += share
                            depositor_distributions.append({
                                "depositor": depositor_id,
                                "amount": share,
                                "method": "cash_fallback",
                            })
                        except (ValidationError, Exception):
                            logger.warning(
                                "Failed to distribute reserves to depositor %s: share=%d",
                                depositor_id, share,
                            )
                else:
                    # No surviving bank — convert to cash
                    try:
                        _resolve_to_cash(system, bank_id, depositor_id, share)
                        total_distributed += share
                        depositor_distributions.append({
                            "depositor": depositor_id,
                            "amount": share,
                            "method": "cash_no_surviving_bank",
                        })
                    except (ValidationError, Exception):
                        logger.warning(
                            "Failed to convert reserves to cash for depositor %s: share=%d",
                            depositor_id, share,
                        )

    # 4. Clean up banking subsystem
    banking_sub = getattr(system.state, "banking_subsystem", None)
    if banking_sub is not None:
        bank_state = banking_sub.banks.get(bank_id)
        if bank_state:
            bank_state.outstanding_loans.clear()
            bank_state.total_loan_principal = 0

    system.log(
        "BankResolutionCompleted",
        bank_id=bank_id,
        total_reserves=total_reserves,
        remaining_reserves=remaining_reserves,
        cb_claims_cancelled=cb_claims_cancelled,
        depositor_distributions=depositor_distributions,
        num_depositors=len(depositor_distributions),
    )
    logger.info(
        "Bank resolution completed: bank=%s reserves=%d cb_cancelled=%d depositors=%d",
        bank_id, total_reserves, cb_claims_cancelled, len(depositor_distributions),
    )


def _write_off_liabilities(
    system: System,
    agent_id: str,
    *,
    skip_contract_id: str | None = None,
    cancelled_contract_ids: set[str] | None = None,
    cancelled_aliases: set[str] | None = None,
) -> tuple[int, int]:
    """Write off all outstanding liabilities issued by *agent_id*.

    Logs ``ObligationWrittenOff`` for every cancelled contract and removes
    contracts/aliases from the system.

    Returns:
        (n_contracts_cancelled, n_aliases_cancelled)
    """
    if cancelled_contract_ids is None:
        cancelled_contract_ids = set()
    if cancelled_aliases is None:
        cancelled_aliases = set()

    # Remove any aliases provided for already-cancelled contracts
    for alias in list(cancelled_aliases):
        system.state.aliases.pop(alias, None)

    for cid, contract in list(system.state.contracts.items()):
        if contract.liability_issuer_id != agent_id:
            continue
        if skip_contract_id and cid == skip_contract_id:
            continue

        contract_alias = get_alias_for_id(system, cid)
        if contract_alias:
            cancelled_aliases.add(contract_alias)
        payload = {
            "contract_id": cid,
            "alias": contract_alias,
            "debtor": contract.liability_issuer_id,
            "creditor": contract.asset_holder_id,
            "contract_kind": contract.kind,
            "amount": contract.amount,
            "due_day": contract.due_day,
        }
        if isinstance(contract, DeliveryObligation):
            payload["sku"] = contract.sku
        if payload.get("due_day") is None:
            payload.pop("due_day", None)

        system.log(EventKind.OBLIGATION_WRITTEN_OFF, **payload)
        _remove_contract(system, cid)
        cancelled_contract_ids.add(cid)
        if contract_alias:
            system.state.aliases.pop(contract_alias, None)

    return len(cancelled_contract_ids), len(cancelled_aliases)


def _expel_agent(
    system: System,
    agent_id: str,
    *,
    trigger_contract_id: str | None = None,
    trigger_kind: str | None = None,
    trigger_shortfall: int | None = None,
    cancelled_contract_ids: set[str] | None = None,
    cancelled_aliases: set[str] | None = None,
) -> None:
    """Expel a defaulted agent from the simulation.

    Sequence:
    1. Guard: no-op in fail-fast mode or if already defaulted.
    2. Raises SimulationHalt if the agent is the central bank.
    3. Marks the agent as defaulted and adds to defaulted_agent_ids.
    4. Distributes remaining cash to creditors pro-rata (recovery).
    5. Writes off all outstanding liabilities issued by the agent.
    6. Cancels scheduled actions referencing the agent or its contracts.
    7. If all non-CB agents have defaulted, raises SimulationHalt
       (system collapse).
    """
    if _get_default_mode(system) != DEFAULT_MODE_EXPEL:
        return

    if agent_id in system.state.defaulted_agent_ids:
        return

    agent = system.state.agents.get(agent_id)
    if agent is None:
        return

    if agent.kind == AgentKind.CENTRAL_BANK:
        raise SimulationHalt("Central bank cannot default", halt_kind="cb_default")

    agent.defaulted = True
    system.state.defaulted_agent_ids.add(agent_id)
    logger.info("agent %s defaulted (trigger=%s)", agent_id, trigger_contract_id)

    system.log(
        EventKind.AGENT_DEFAULTED,
        agent=agent_id,
        frm=agent_id,
        trigger_contract=trigger_contract_id,
        contract_kind=trigger_kind,
        shortfall=trigger_shortfall,
        mode=_get_default_mode(system),
    )

    # Bank resolution: distribute reserves to depositors
    # Non-bank recovery: distribute cash to creditors pro-rata
    if agent.kind == AgentKind.BANK:
        _resolve_failed_bank(system, agent_id)
    else:
        _distribute_pro_rata_recovery(system, agent_id)

    cancelled_contract_ids = set(cancelled_contract_ids or [])
    cancelled_aliases = set(cancelled_aliases or [])

    n_contracts, n_aliases = _write_off_liabilities(
        system, agent_id,
        skip_contract_id=trigger_contract_id,
        cancelled_contract_ids=cancelled_contract_ids,
        cancelled_aliases=cancelled_aliases,
    )

    _cancel_scheduled_actions_for_agent(system, agent_id, cancelled_contract_ids, cancelled_aliases)
    logger.warning(
        "agent %s expelled: cancelled %d contracts, %d aliases",
        agent_id,
        n_contracts,
        n_aliases,
    )

    # If every non-central-bank agent has defaulted, halt the simulation with a DefaultError.
    if all(
        (ag.kind == AgentKind.CENTRAL_BANK) or ag.defaulted for ag in system.state.agents.values()
    ):
        raise SimulationHalt(
            "All non-central-bank agents have defaulted", halt_kind="system_collapse"
        )


def settle_due_delivery_obligations(system: System, day: int) -> None:
    """Settle all delivery obligations due today using stock operations."""
    due_list = list(due_delivery_obligations(system, day))
    logger.info("settle_due_delivery_obligations: day %d, %d obligation(s) due", day, len(due_list))
    for obligation_instr in due_list:
        if obligation_instr.id not in system.state.contracts:
            continue
        assert isinstance(obligation_instr, DeliveryObligation)
        obligation = obligation_instr

        debtor = system.state.agents[obligation.liability_issuer_id]
        if debtor.defaulted:
            continue

        creditor = system.state.agents[obligation.asset_holder_id]
        required_sku = obligation.sku
        required_quantity = obligation.amount

        with atomic(system):
            delivered_quantity = _deliver_stock(
                system, debtor.id, creditor.id, required_sku, required_quantity
            )

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
                        EventKind.PARTIAL_SETTLEMENT,
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
                    EventKind.OBLIGATION_DEFAULTED,
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
                EventKind.DELIVERY_OBLIGATION_SETTLED,
                obligation_id=obligation.id,
                contract_id=obligation.id,
                alias=alias,
                debtor=debtor.id,
                creditor=creditor.id,
                sku=required_sku,
                qty=required_quantity,
            )


def _settle_single_payable(
    system: System,
    payable: Payable,
    day: int,
    *,
    risk_assessor: Any,
    rollover_enabled: bool,
) -> tuple[bool, tuple[str, str, int, int] | None]:
    """Settle a single payable using the settlement waterfall.

    Attempts payment in the order defined by policy.settlement_order():
    deposits → cash → reserves.  Each method is tried in sequence until
    the full amount is covered.  All mutations are wrapped in an atomic
    block so partial payments roll back on failure.

    Returns:
        (True, rollover_info) on full settlement, where rollover_info is
        (debtor_id, creditor_id, amount, maturity_distance) or None.
        (False, None) on default — the payable is written off and the
        debtor is expelled.
    """
    debtor = system.state.agents[payable.liability_issuer_id]
    if debtor.defaulted:
        return True, None  # skip silently

    creditor_id = payable.effective_creditor
    creditor = system.state.agents[creditor_id]
    order = system.policy.settlement_order(debtor)
    logger.debug(
        "_settle_single_payable: payable=%s debtor=%s creditor=%s amount=%d order=%s",
        payable.id,
        debtor.id,
        creditor_id,
        payable.amount,
        [str(m) for m in order],
    )

    remaining = payable.amount
    payments_summary: list[dict[str, object]] = []
    payable_amount = payable.amount
    payable_maturity_distance = payable.maturity_distance
    original_creditor = payable.asset_holder_id

    with atomic(system):
        for method in order:
            if remaining == 0:
                break
            if method == InstrumentKind.BANK_DEPOSIT:
                paid_now = _pay_with_deposits(system, debtor.id, creditor.id, remaining)
            elif method == InstrumentKind.CASH:
                paid_now = _pay_with_cash(system, debtor.id, creditor.id, remaining)
            elif method == InstrumentKind.RESERVE_DEPOSIT:
                paid_now = _pay_bank_to_bank_with_reserves(
                    system, debtor.id, creditor.id, remaining
                )
            else:
                raise ValidationError(f"unknown payment method {method}")
            remaining -= paid_now
            if paid_now > 0:
                payments_summary.append({"method": method, "amount": paid_now})

        if remaining != 0:
            return _handle_payable_default(
                system,
                payable,
                debtor,
                creditor,
                day,
                remaining=remaining,
                payments_summary=payments_summary,
                risk_assessor=risk_assessor,
                rollover_enabled=rollover_enabled,
            ), None

        _remove_contract(system, payable.id)
        alias = get_alias_for_id(system, payable.id)
        logger.debug(
            "payable settled: %s, debtor=%s, creditor=%s, amount=%d",
            payable.id,
            debtor.id,
            creditor.id,
            payable_amount,
        )
        system.log(
            EventKind.PAYABLE_SETTLED,
            pid=payable.id,
            contract_id=payable.id,
            alias=alias,
            debtor=debtor.id,
            creditor=creditor.id,
            amount=payable_amount,
        )
        _update_risk_history(system, day=day, issuer_id=debtor.id, defaulted=False)

        rollover_info = None
        if rollover_enabled and payable_maturity_distance is not None:
            rollover_info = (
                debtor.id,
                original_creditor,
                payable_amount,
                payable_maturity_distance,
            )
        return True, rollover_info


def _handle_payable_default(
    system: System,
    payable: Payable,
    debtor: Agent,
    creditor: Agent,
    day: int,
    *,
    remaining: int,
    payments_summary: list[dict[str, object]],
    risk_assessor: Any,
    rollover_enabled: bool,
) -> bool:
    """Handle a payable that could not be fully settled (default cascade).

    Sequence:
    1. In fail-fast mode, raises DefaultError immediately.
    2. Logs PARTIAL_SETTLEMENT if any payment was made.
    3. Logs OBLIGATION_DEFAULTED for the shortfall.
    4. Removes the payable contract.
    5. Expels the debtor (_expel_agent), which writes off all remaining
       obligations and cancels scheduled actions.
    6. Updates the risk assessor with the default observation.
    7. If rollover is enabled, reconnects the ring around the expelled agent.

    Always returns False (settlement failed).
    """
    logger.warning("payable default: %s, debtor=%s, shortfall=%d", payable.id, debtor.id, remaining)
    if _get_default_mode(system) == DEFAULT_MODE_FAIL_FAST:
        raise DefaultError(
            f"Insufficient funds to settle payable {payable.id}: {remaining} still owed"
        )

    alias = get_alias_for_id(system, payable.id)
    cancelled_contract_ids = {payable.id}
    cancelled_aliases = {alias} if alias else set()
    amount_paid = payable.amount - remaining

    if amount_paid > 0:
        payload: dict[str, object] = {
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
        system.log(EventKind.PARTIAL_SETTLEMENT, **payload)

    system.log(
        EventKind.OBLIGATION_DEFAULTED,
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
    _update_risk_history(system, day=day, issuer_id=debtor.id, defaulted=True)
    if rollover_enabled:
        _reconnect_ring(system, debtor.id, ring_successor_id, day)
    return False


def settle_due(
    system: System, day: int, *, rollover_enabled: bool = False
) -> list[tuple[str, str, int, int]]:
    """Settle all obligations due today (payables and delivery obligations).

    Args:
        system: The system to settle
        day: Current simulation day
        rollover_enabled: If True, create new payables for successfully settled ones (Plan 024)

    Returns:
        List of settled payable info for rollover: [(debtor_id, creditor_id, amount, maturity_distance)]
    """
    due_list = list(due_payables(system, day))
    logger.info("settle_due: day %d, %d payable(s) due", day, len(due_list))
    settled_for_rollover = []
    risk_assessor = _get_risk_assessor(system)

    for payable_instr in due_list:
        if payable_instr.id not in system.state.contracts:
            continue
        assert isinstance(payable_instr, Payable)
        settled, rollover_info = _settle_single_payable(
            system,
            payable_instr,
            day,
            risk_assessor=risk_assessor,
            rollover_enabled=rollover_enabled,
        )
        if rollover_info is not None:
            settled_for_rollover.append(rollover_info)

    settle_due_delivery_obligations(system, day)
    return settled_for_rollover


def _rollover_single_payable(
    system: System,
    debtor_id: str,
    creditor_id: str,
    amount: int,
    maturity_distance: int,
    new_due_day: int,
    dealer_active: bool,
) -> str | None:
    """Create a single rolled-over payable and optionally transfer cash.

    Returns the new payable ID on success, or None if either party
    has defaulted or is missing.
    """
    debtor = system.state.agents.get(debtor_id)
    creditor = system.state.agents.get(creditor_id)

    if debtor is None or debtor.defaulted:
        logger.debug("rollover: skipping debtor=%s (defaulted or missing)", debtor_id)
        return None
    if creditor is None or creditor.defaulted:
        logger.debug("rollover: skipping creditor=%s (defaulted or missing)", creditor_id)
        return None

    with atomic(system):
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=amount,
            denom="X",
            asset_holder_id=creditor_id,
            liability_issuer_id=debtor_id,
            due_day=new_due_day,
            maturity_distance=maturity_distance,
        )
        system.add_contract(new_payable)

        is_financial_creditor = creditor_id.startswith("vbt_") or creditor_id.startswith("dealer_")
        if not dealer_active or is_financial_creditor:
            cash_transferred = _pay_with_deposits(system, creditor_id, debtor_id, amount)
            if cash_transferred < amount:
                cash_transferred += _pay_with_cash(
                    system, creditor_id, debtor_id, amount - cash_transferred
                )
        else:
            cash_transferred = 0

        if cash_transferred != amount and (not dealer_active or is_financial_creditor):
            system.log(
                EventKind.ROLLOVER_PARTIAL,
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
                EventKind.PAYABLE_ROLLED_OVER,
                debtor=debtor_id,
                creditor=creditor_id,
                amount=amount,
                new_due_day=new_due_day,
                maturity_distance=maturity_distance,
                payable_id=new_payable.id,
                cash_transfer=(not dealer_active or is_financial_creditor),
            )

    return new_payable.id


def rollover_settled_payables(
    system: System,
    day: int,
    settled_payables: list[tuple[str, str, int, int]],
    dealer_active: bool = False,
) -> list[str]:
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
    max_due_day = day
    for c in system.state.contracts.values():
        if c.kind == InstrumentKind.PAYABLE and isinstance(c, Payable):
            if c.due_day is not None and c.due_day > max_due_day:
                max_due_day = c.due_day

    logger.debug(
        "rollover: day %d, %d payable(s) to roll, max_due_day=%d",
        day,
        len(settled_payables),
        max_due_day,
    )
    new_payable_ids: list[str] = []
    for debtor_id, creditor_id, amount, maturity_distance in settled_payables:
        new_due_day = max_due_day + maturity_distance
        pid = _rollover_single_payable(
            system,
            debtor_id,
            creditor_id,
            amount,
            maturity_distance,
            new_due_day,
            dealer_active,
        )
        if pid is not None:
            new_payable_ids.append(pid)

    return new_payable_ids
