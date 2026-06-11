"""Payable, default, recovery, and delivery settlement helpers for clean-core."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from decimal import Decimal

from bilancio.core.errors import DefaultError
from bilancio.engines.clean_core_banking import clean_agent_banks as _clean_agent_banks
from bilancio.engines.clean_core_banking import clean_bank_profile as _clean_bank_profile
from bilancio.engines.clean_core_banking import clean_bank_quote as _clean_bank_quote
from bilancio.engines.clean_core_cash import transfer_cash_with_events as _transfer_cash_with_events
from bilancio.engines.clean_core_contracts import action_references_agent as _action_references_agent
from bilancio.engines.clean_core_dealer import _clean_update_dealer_risk_history
from bilancio.engines.clean_core_interbank import primary_bank_for_customer as _primary_bank_for_customer
from bilancio.engines.clean_core_inventory import deliver_stock_for_obligation as _deliver_stock_for_obligation
from bilancio.engines.clean_core_lender import agent_liquid_assets as _agent_liquid_assets
from bilancio.engines.clean_core_rollover import rollover_settled_payables as _rollover_settled_payables_impl
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanBankingConfig,
    CleanDeliveryObligation,
    CleanPayable,
    CleanState,
)


def _settle_payable(
    state: CleanState,
    payable: CleanPayable,
    policy_order: dict[str, list[str]],
    *,
    banking_config: CleanBankingConfig | None = None,
) -> tuple[bool, tuple[str, str, Decimal, int] | None]:
    rollback_snapshot = None
    if state.default_mode == "fail-fast":
        rollback_snapshot = (
            state.cash.copy(),
            defaultdict(list, {agent_id: list(lots) for agent_id, lots in state.cash_lots.items()}),
            state.deposits.copy(),
            len(state.events),
        )

    try:
        remaining = payable.amount
        debtor = state.agents[payable.debtor]
        for means in policy_order.get(debtor.kind, []):
            if remaining <= 0:
                break
            if means == "cash":
                paid = min(state.cash[payable.debtor], remaining)
                if paid:
                    _transfer_cash_with_events(state, payable.debtor, payable.creditor, paid)
                    remaining -= paid
            elif means == "bank_deposit":
                paid = _pay_with_deposit(
                    state,
                    payable.debtor,
                    payable.creditor,
                    remaining,
                    banking_config=banking_config,
                )
                remaining -= paid
    except DefaultError:
        if rollback_snapshot is not None:
            cash, cash_lots, deposits, event_count = rollback_snapshot
            state.cash = cash
            state.cash_lots = cash_lots
            state.deposits = deposits
            del state.events[event_count:]
        raise

    if remaining:
        try:
            return _handle_payable_default(state, payable, remaining), None
        except DefaultError:
            if rollback_snapshot is not None:
                cash, cash_lots, deposits, event_count = rollback_snapshot
                state.cash = cash
                state.cash_lots = cash_lots
                state.deposits = deposits
                del state.events[event_count:]
            raise
    payable.settled = True
    state.log(
        "PayableSettled",
        pid=payable.id,
        contract_id=payable.id,
        alias=payable.alias,
        debtor=payable.debtor,
        creditor=payable.creditor,
        amount=payable.amount,
    )
    _clean_update_dealer_risk_history(
        state,
        issuer_id=payable.debtor,
        defaulted=False,
    )
    rollover_info = None
    if state.rollover_enabled:
        rollover_info = (
            payable.debtor,
            payable.creditor,
            payable.amount,
            payable.maturity_distance,
        )
    return True, rollover_info


def _pay_with_deposit(
    state: CleanState,
    payer: str,
    payee: str,
    amount: Decimal,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> Decimal:
    if banking_config is not None:
        return _pay_with_routed_deposits(state, payer, payee, amount, banking_config)

    payer_bank = _primary_bank_for_customer(state, payer)
    payee_bank = _primary_bank_for_customer(state, payee)
    if payer_bank is None:
        return ZERO
    if payee_bank is None:
        payee_bank = payer_bank

    paid = min(state.deposits[(payer, payer_bank)], amount)
    if not paid:
        return ZERO

    state.deposits[(payer, payer_bank)] -= paid
    state.deposits[(payee, payee_bank)] += paid

    if payer_bank == payee_bank:
        state.log("IntraBankPayment", payer=payer, payee=payee, bank=payer_bank, amount=paid)
    else:
        state.log(
            "ClientPayment",
            payer=payer,
            payer_bank=payer_bank,
            payee=payee,
            payee_bank=payee_bank,
            amount=paid,
        )
    return paid


def _pay_with_routed_deposits(
    state: CleanState,
    payer: str,
    payee: str,
    amount: Decimal,
    banking_config: CleanBankingConfig,
) -> Decimal:
    payer_balances = {
        bank_id: balance
        for (customer_id, bank_id), balance in state.deposits.items()
        if customer_id == payer and balance > ZERO
    }
    if not payer_balances:
        return ZERO

    pay_amount = min(amount, sum(payer_balances.values(), ZERO))
    payee_bank = _select_clean_receive_bank(state, payee, banking_config)
    if payee_bank is None:
        return ZERO

    profile = _clean_bank_profile(banking_config)
    sorted_banks: list[tuple[Decimal, str, Decimal]] = []
    for bank_id, balance in payer_balances.items():
        quote, _params = _clean_bank_quote(state, bank_id, banking_config, profile)
        sorted_banks.append((quote.deposit_rate, bank_id, balance))
    sorted_banks.sort(key=lambda item: item[0])

    paid_total = ZERO
    remaining = pay_amount
    for _rate, payer_bank, balance in sorted_banks:
        if remaining <= ZERO:
            break
        paid = min(balance, remaining)
        state.deposits[(payer, payer_bank)] -= paid
        state.deposits[(payee, payee_bank)] += paid
        if payer_bank == payee_bank:
            state.log("IntraBankPayment", payer=payer, payee=payee, bank=payer_bank, amount=paid)
        else:
            state.log(
                "ClientPayment",
                payer=payer,
                payer_bank=payer_bank,
                payee=payee,
                payee_bank=payee_bank,
                amount=paid,
            )
        paid_total += paid
        remaining -= paid
    return paid_total


def _select_clean_receive_bank(
    state: CleanState,
    payee: str,
    banking_config: CleanBankingConfig,
) -> str | None:
    profile = _clean_bank_profile(banking_config)
    candidates: list[tuple[Decimal, str]] = []
    for customer_id, bank_id in state.deposits:
        if customer_id != payee:
            continue
        bank = state.agents.get(bank_id)
        if bank is None or bank.kind != "bank" or bank_id in state.defaulted_agent_ids:
            continue
        quote, _params = _clean_bank_quote(state, bank_id, banking_config, profile)
        candidates.append((quote.deposit_rate, bank_id))
    if not candidates:
        for bank_id in _clean_agent_banks(state, payee, banking_config):
            bank = state.agents.get(bank_id)
            if bank is None or bank.kind != "bank" or bank_id in state.defaulted_agent_ids:
                continue
            quote, _params = _clean_bank_quote(state, bank_id, banking_config, profile)
            candidates.append((quote.deposit_rate, bank_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _handle_payable_default(state: CleanState, payable: CleanPayable, remaining: Decimal) -> bool:
    if state.default_mode != "expel-agent":
        raise DefaultError(
            f"Insufficient funds to settle payable {payable.id}: {remaining} still owed"
        )

    amount_paid = payable.amount - remaining
    if amount_paid > 0:
        state.log(
            "PartialSettlement",
            contract_id=payable.id,
            alias=payable.alias,
            debtor=payable.debtor,
            creditor=payable.creditor,
            contract_kind="payable",
            settlement_kind="payable",
            amount_paid=amount_paid,
            shortfall=remaining,
            original_amount=payable.amount,
            distribution=[{"method": "cash", "amount": amount_paid}],
        )

    state.log(
        "ObligationDefaulted",
        contract_id=payable.id,
        alias=payable.alias,
        debtor=payable.debtor,
        creditor=payable.creditor,
        contract_kind="payable",
        shortfall=remaining,
        amount_paid=amount_paid,
        original_amount=payable.amount,
        amount=remaining,
    )
    creditor_weights = _collect_creditor_weights(state, payable.debtor)
    payable.settled = True
    _expel_agent(state, payable.debtor, trigger_contract_id=payable.id, trigger_shortfall=remaining)
    _clean_update_dealer_risk_history(
        state,
        issuer_id=payable.debtor,
        defaulted=True,
    )
    _reassign_receivables(state, payable.debtor, creditor_weights)
    return False


def _rollover_settled_payables(
    state: CleanState,
    settled_payables: list[tuple[str, str, Decimal, int]],
) -> list[str]:
    return _rollover_settled_payables_impl(
        state,
        settled_payables,
        pay_with_deposit=_pay_with_deposit,
        transfer_cash_with_events=_transfer_cash_with_events,
    )


def _expel_agent(
    state: CleanState,
    agent_id: str,
    *,
    trigger_contract_id: str,
    trigger_shortfall: Decimal,
    trigger_kind: str = "payable",
) -> None:
    state.defaulted_agent_ids.add(agent_id)
    state.log(
        "AgentDefaulted",
        agent=agent_id,
        frm=agent_id,
        trigger_contract=trigger_contract_id,
        contract_kind=trigger_kind,
        shortfall=trigger_shortfall,
        mode=state.default_mode,
    )

    _distribute_pro_rata_recovery(state, agent_id)
    _write_off_liabilities(state, agent_id, skip_contract_id=trigger_contract_id)
    _cancel_scheduled_actions_for_agent(state, agent_id)


def _collect_creditor_weights(state: CleanState, agent_id: str) -> dict[str, Decimal]:
    claims: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
    for payable in state.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        claims[payable.creditor] += payable.amount
    for loan in state.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        claims[loan.lender] += loan.amount
    for loan in state.bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        claims[loan.bank] += loan.amount
    for loan in state.cb_loans:
        if loan.settled or loan.bank != agent_id:
            continue
        claims[loan.central_bank] += loan.amount
    for obligation in state.delivery_obligations:
        if obligation.settled or obligation.debtor != agent_id:
            continue
        claims[obligation.creditor] += Decimal(obligation.quantity)

    total = sum(claims.values(), ZERO)
    if total <= ZERO:
        return {}
    return {creditor: amount / total for creditor, amount in claims.items()}


def _distribute_pro_rata_recovery(state: CleanState, agent_id: str) -> None:
    total_liquid = _agent_liquid_assets(state, agent_id)
    if total_liquid <= ZERO:
        return

    claims: list[tuple[str, Decimal]] = []
    for payable in state.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        claims.append((payable.creditor, payable.amount))
    for loan in state.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        claims.append((loan.lender, loan.repayment_amount))

    total_claims = sum((amount for _, amount in claims), ZERO)
    if total_claims <= ZERO:
        return

    details = []
    total_distributed = ZERO
    for creditor_id, claim_amount in claims:
        share = Decimal(round(float((claim_amount / total_claims) * total_liquid)))
        if share <= ZERO:
            continue
        transferred = _pay_with_deposit(state, agent_id, creditor_id, share)
        remainder = share - transferred
        if remainder > ZERO:
            paid_cash = min(state.cash[agent_id], remainder)
            if paid_cash > ZERO:
                _transfer_cash_with_events(state, agent_id, creditor_id, paid_cash)
                transferred += paid_cash
        if transferred > ZERO:
            total_distributed += transferred
            details.append(
                {
                    "creditor": creditor_id,
                    "claim": claim_amount,
                    "recovery": transferred,
                }
            )

    if total_distributed > ZERO:
        state.log(
            "ProRataRecovery",
            agent=agent_id,
            total_liquid=total_liquid,
            total_claims=total_claims,
            total_distributed=total_distributed,
            num_creditors=len(details),
            details=details,
        )


def _write_off_liabilities(
    state: CleanState,
    agent_id: str,
    *,
    skip_contract_id: str | None,
) -> None:
    for payable in state.payables:
        if payable.settled or payable.debtor != agent_id or payable.id == skip_contract_id:
            continue
        payable.settled = True
        state.log(
            "ObligationWrittenOff",
            contract_id=payable.id,
            alias=payable.alias,
            debtor=payable.debtor,
            creditor=payable.creditor,
            contract_kind="payable",
            amount=payable.amount,
            due_day=payable.due_day,
        )

    for loan in state.non_bank_loans:
        if loan.settled or loan.borrower != agent_id or loan.id == skip_contract_id:
            continue
        loan.settled = True
        state.log(
            "ObligationWrittenOff",
            contract_id=loan.id,
            alias=None,
            debtor=loan.borrower,
            creditor=loan.lender,
            contract_kind="non_bank_loan",
            amount=loan.amount,
        )

    for loan in state.bank_loans:
        if loan.settled or loan.borrower != agent_id or loan.id == skip_contract_id:
            continue
        loan.settled = True
        state.log(
            "ObligationWrittenOff",
            contract_id=loan.id,
            alias=None,
            debtor=loan.borrower,
            creditor=loan.bank,
            contract_kind="bank_loan",
            amount=loan.amount,
        )

    for obligation in state.delivery_obligations:
        if obligation.settled or obligation.debtor != agent_id or obligation.id == skip_contract_id:
            continue
        obligation.settled = True
        state.log(
            "ObligationWrittenOff",
            contract_id=obligation.id,
            alias=obligation.alias,
            debtor=obligation.debtor,
            creditor=obligation.creditor,
            contract_kind="delivery_obligation",
            amount=obligation.quantity,
            due_day=obligation.due_day,
            sku=obligation.sku,
        )


def _reassign_receivables(
    state: CleanState,
    defaulted_agent_id: str,
    creditor_weights: dict[str, Decimal],
) -> None:
    if not creditor_weights:
        for receivable in state.payables:
            if not receivable.settled and receivable.creditor == defaulted_agent_id:
                receivable.settled = True
        return

    for receivable in list(state.payables):
        if receivable.settled or receivable.creditor != defaulted_agent_id:
            continue
        if receivable.debtor == defaulted_agent_id:
            continue
        if receivable.debtor in state.defaulted_agent_ids:
            continue

        old_payable_id = receivable.id
        original_amount = receivable.amount
        maturity_distance = receivable.maturity_distance
        new_due_day = state.day + maturity_distance
        receivable.settled = True

        for creditor_id, weight in creditor_weights.items():
            if creditor_id == receivable.debtor:
                continue
            new_amount = Decimal(int(original_amount * weight))
            if new_amount < Decimal("1"):
                continue
            new_payable = CleanPayable(
                id=f"PAY_reassigned_{len(state.payables)}",
                debtor=receivable.debtor,
                creditor=creditor_id,
                amount=new_amount,
                due_day=new_due_day,
                maturity_distance=maturity_distance,
            )
            state.payables.append(new_payable)
            state.log(
                "ReceivableReassigned",
                defaulted_agent=defaulted_agent_id,
                debtor=receivable.debtor,
                new_creditor=creditor_id,
                old_payable=old_payable_id,
                new_payable=new_payable.id,
                amount=new_amount,
                weight=float(weight),
                maturity_distance=maturity_distance,
                new_due_day=new_due_day,
            )
            state.log(
                "PayableCreated",
                contract_id=new_payable.id,
                debtor=receivable.debtor,
                creditor=creditor_id,
                amount=new_amount,
                due_day=new_due_day,
                maturity_distance=maturity_distance,
                reason="receivable_reassignment",
            )


def _cancel_scheduled_actions_for_agent(state: CleanState, agent_id: str) -> None:
    for scheduled_day, actions in list(state.scheduled_actions_by_day.items()):
        remaining_actions = []
        for action in actions:
            if _action_references_agent(action, agent_id):
                action_name = next(iter(action.keys()), "unknown")
                state.log(
                    "ScheduledActionCancelled",
                    agent=agent_id,
                    scheduled_day=scheduled_day,
                    action=action_name,
                    mode=state.default_mode,
                )
            else:
                remaining_actions.append(action)
        if remaining_actions:
            state.scheduled_actions_by_day[scheduled_day] = remaining_actions
        else:
            del state.scheduled_actions_by_day[scheduled_day]


def _settle_delivery_obligation(state: CleanState, obligation: CleanDeliveryObligation) -> bool:
    rollback_snapshot = (
        {stock_id: replace(stock) for stock_id, stock in state.stocks.items()},
        len(state.events),
    )
    delivered_quantity = _deliver_stock_for_obligation(
        state,
        obligation.debtor,
        obligation.creditor,
        obligation.sku,
        obligation.quantity,
    )
    if delivered_quantity != obligation.quantity:
        shortage = obligation.quantity - delivered_quantity
        if state.default_mode != "expel-agent":
            stocks, event_count = rollback_snapshot
            state.stocks = stocks
            del state.events[event_count:]
            raise DefaultError(
                f"Insufficient stock to settle delivery obligation {obligation.id}: "
                f"{shortage} units of {obligation.sku} still owed"
            )
        _handle_delivery_default(state, obligation, delivered_quantity, shortage)
        return False

    obligation.settled = True
    state.log(
        "DeliveryObligationCancelled",
        obligation_id=obligation.id,
        contract_id=obligation.id,
        alias=obligation.alias,
        debtor=obligation.debtor,
        creditor=obligation.creditor,
        sku=obligation.sku,
        qty=obligation.quantity,
    )
    state.log(
        "DeliveryObligationSettled",
        obligation_id=obligation.id,
        contract_id=obligation.id,
        alias=obligation.alias,
        debtor=obligation.debtor,
        creditor=obligation.creditor,
        sku=obligation.sku,
        qty=obligation.quantity,
    )
    return True


def _handle_delivery_default(
    state: CleanState,
    obligation: CleanDeliveryObligation,
    delivered_quantity: int,
    shortage: int,
) -> None:
    if delivered_quantity > 0:
        state.log(
            "PartialSettlement",
            contract_id=obligation.id,
            alias=obligation.alias,
            debtor=obligation.debtor,
            creditor=obligation.creditor,
            contract_kind="delivery_obligation",
            settlement_kind="delivery",
            delivered_quantity=delivered_quantity,
            required_quantity=obligation.quantity,
            shortfall=shortage,
            sku=obligation.sku,
        )

    state.log(
        "ObligationDefaulted",
        contract_id=obligation.id,
        alias=obligation.alias,
        debtor=obligation.debtor,
        creditor=obligation.creditor,
        contract_kind="delivery_obligation",
        shortfall=shortage,
        delivered_quantity=delivered_quantity,
        required_quantity=obligation.quantity,
        sku=obligation.sku,
        qty=shortage,
    )
    obligation.settled = True
    _expel_agent(
        state,
        obligation.debtor,
        trigger_contract_id=obligation.id,
        trigger_kind="delivery_obligation",
        trigger_shortfall=Decimal(shortage),
    )
