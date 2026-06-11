"""Settlement phase (Subphase B2): due payables and delivery obligations.

Ports the clean-core settlement semantics exactly:

* Payables settle by the debtor kind's means-of-payment priority
  (capability matrix), partially if needed.
* ``fail-fast``: a shortfall raises :class:`DefaultError` and the whole
  settlement attempt is rolled back atomically.
* ``expel-agent``: a shortfall defaults the debtor — partial settlement is
  recorded, the debtor is expelled (pro-rata recovery of its liquid assets,
  liability write-offs, scheduled-action cancellation) and its receivables
  are reassigned pro-rata to its creditors.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from bilancio.core.errors import DefaultError
from bilancio_v2.actions import action_references_agent
from bilancio_v2.ledger import ZERO, DeliveryObligation, Ledger, Payable
from bilancio_v2.plugins.base import RunContext


def update_dealer_risk_history(ledger: Ledger, *, issuer_id: str, defaulted: bool) -> None:
    from bilancio_v2.plugins.dealer import update_dealer_risk_history as _update

    _update(ledger, issuer_id=issuer_id, defaulted=defaulted)


class SettlementPhase:
    name = "SubphaseB2"

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        impactful = False
        settled_for_rollover: list[tuple[str, str, Decimal, int, Decimal]] = []
        if ledger.netted_rollover_queue:
            if ctx.rollover_enabled:
                # Fully-netted payables settled without cash ("net-settle,
                # gross-roll", Plan 059): roll at full face, zero cash return.
                settled_for_rollover.extend(
                    (debtor, creditor, face, maturity_distance, ZERO)
                    for debtor, creditor, face, maturity_distance in ledger.netted_rollover_queue
                )
            ledger.netted_rollover_queue.clear()
        for payable in list(ledger.payables):
            if payable.settled or payable.due_day != ledger.day:
                continue
            settled, rollover_info = settle_payable(ledger, payable, ctx)
            impactful = settled or impactful
            if rollover_info is not None:
                settled_for_rollover.append(rollover_info)
        for obligation in ledger.delivery_obligations:
            if obligation.settled or obligation.due_day != ledger.day:
                continue
            impactful = settle_delivery_obligation(ledger, obligation, ctx) or impactful
        if ctx.rollover_enabled and settled_for_rollover:
            ledger.log("SubphaseB_Rollover")
            rollover_settled_payables(ledger, settled_for_rollover)
        return impactful


def settle_payable(ledger: Ledger, payable: Payable, ctx: RunContext) -> tuple[bool, tuple[str, str, Decimal, int, Decimal] | None]:
    checkpoint = ledger.checkpoint() if ctx.default_mode == "fail-fast" else None

    remaining = payable.amount
    debtor = ledger.agents[payable.debtor]
    for means in ctx.policy.mop_order(debtor.kind):
        if remaining <= 0:
            break
        if means == "cash":
            paid = min(ledger.cash[payable.debtor], remaining)
            if paid:
                ledger.transfer_cash(payable.debtor, payable.creditor, paid)
                remaining -= paid
        elif means == "bank_deposit":
            paid = pay_with_deposit(
                ledger,
                payable.debtor,
                payable.creditor,
                remaining,
                banking_config=ctx.banking_config,
            )
            remaining -= paid
        # Other means (e.g. reserve_deposit) have no payable-settlement
        # channel in this slice, matching the existing engine.

    if remaining:
        try:
            return handle_payable_default(ledger, payable, remaining, ctx), None
        except DefaultError:
            if checkpoint is not None:
                ledger.restore(checkpoint)
            raise

    payable.settled = True
    ledger.log(
        "PayableSettled",
        pid=payable.id,
        contract_id=payable.id,
        alias=payable.alias,
        debtor=payable.debtor,
        creditor=payable.creditor,
        amount=payable.amount,
    )
    update_dealer_risk_history(ledger, issuer_id=payable.debtor, defaulted=False)
    rollover_info = None
    if ctx.rollover_enabled:
        # Partially netted payables roll at full original face; only the
        # cash-settled residual generates a cash return-flow.
        rollover_info = (
            payable.debtor,
            payable.creditor,
            payable.amount + payable.netted_amount,
            payable.maturity_distance,
            payable.amount,
        )
    return True, rollover_info


def pay_with_deposit(
    ledger: Ledger,
    payer: str,
    payee: str,
    amount: Decimal,
    *,
    banking_config: Any | None = None,
) -> Decimal:
    if banking_config is not None:
        return pay_with_routed_deposits(ledger, payer, payee, amount, banking_config)

    payer_bank = ledger.primary_bank_for_customer(payer)
    payee_bank = ledger.primary_bank_for_customer(payee)
    if payer_bank is None:
        return ZERO
    if payee_bank is None:
        payee_bank = payer_bank
    paid = min(ledger.deposits[(payer, payer_bank)], amount)
    if not paid:
        return ZERO
    ledger.move_deposit(payer, payer_bank, payee, payee_bank, paid)
    return paid


def pay_with_routed_deposits(ledger: Ledger, payer: str, payee: str, amount: Decimal, banking_config: Any) -> Decimal:
    """Banking-mode deposit payment: drain the payer's cheapest-deposit-rate
    banks first, crediting the payee at its highest-deposit-rate bank."""
    from bilancio_v2.plugins.banking import bank_profile, bank_quote

    payer_balances = {
        bank_id: balance for (customer_id, bank_id), balance in ledger.deposits.items() if customer_id == payer and balance > ZERO
    }
    if not payer_balances:
        return ZERO

    pay_amount = min(amount, sum(payer_balances.values(), ZERO))
    payee_bank = select_receive_bank(ledger, payee, banking_config)
    if payee_bank is None:
        return ZERO

    profile = bank_profile(banking_config)
    sorted_banks: list[tuple[Decimal, str, Decimal]] = []
    for bank_id, balance in payer_balances.items():
        quote, _params = bank_quote(ledger, bank_id, banking_config, profile)
        sorted_banks.append((quote.deposit_rate, bank_id, balance))
    sorted_banks.sort(key=lambda item: item[0])

    paid_total = ZERO
    remaining = pay_amount
    for _rate, payer_bank, balance in sorted_banks:
        if remaining <= ZERO:
            break
        paid = min(balance, remaining)
        ledger.move_deposit(payer, payer_bank, payee, payee_bank, paid)
        paid_total += paid
        remaining -= paid
    return paid_total


def select_receive_bank(ledger: Ledger, payee: str, banking_config: Any) -> str | None:
    from bilancio_v2.plugins.banking import agent_banks, bank_profile, bank_quote

    profile = bank_profile(banking_config)
    candidates: list[tuple[Decimal, str]] = []
    for customer_id, bank_id in ledger.deposits:
        if customer_id != payee:
            continue
        bank = ledger.agents.get(bank_id)
        if bank is None or bank.kind != "bank" or bank_id in ledger.defaulted_agent_ids:
            continue
        quote, _params = bank_quote(ledger, bank_id, banking_config, profile)
        candidates.append((quote.deposit_rate, bank_id))
    if not candidates:
        for bank_id in agent_banks(ledger, payee, banking_config):
            bank = ledger.agents.get(bank_id)
            if bank is None or bank.kind != "bank" or bank_id in ledger.defaulted_agent_ids:
                continue
            quote, _params = bank_quote(ledger, bank_id, banking_config, profile)
            candidates.append((quote.deposit_rate, bank_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def handle_payable_default(ledger: Ledger, payable: Payable, remaining: Decimal, ctx: RunContext) -> bool:
    if ctx.default_mode != "expel-agent":
        raise DefaultError(f"Insufficient funds to settle payable {payable.id}: {remaining} still owed")

    amount_paid = payable.amount - remaining
    if amount_paid > 0:
        ledger.log(
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

    ledger.log(
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
    creditor_weights = collect_creditor_weights(ledger, payable.debtor)
    payable.settled = True
    expel_agent(
        ledger,
        payable.debtor,
        ctx,
        trigger_contract_id=payable.id,
        trigger_shortfall=remaining,
    )
    update_dealer_risk_history(ledger, issuer_id=payable.debtor, defaulted=True)
    reassign_receivables(ledger, payable.debtor, creditor_weights)
    return False


def expel_agent(
    ledger: Ledger,
    agent_id: str,
    ctx: RunContext,
    *,
    trigger_contract_id: str,
    trigger_shortfall: Decimal,
    trigger_kind: str = "payable",
) -> None:
    ledger.defaulted_agent_ids.add(agent_id)
    ledger.log(
        "AgentDefaulted",
        agent=agent_id,
        frm=agent_id,
        trigger_contract=trigger_contract_id,
        contract_kind=trigger_kind,
        shortfall=trigger_shortfall,
        mode=ctx.default_mode,
    )
    distribute_pro_rata_recovery(ledger, agent_id)
    write_off_liabilities(ledger, agent_id, skip_contract_id=trigger_contract_id)
    cancel_scheduled_actions_for_agent(ledger, agent_id, ctx)


def collect_creditor_weights(ledger: Ledger, agent_id: str) -> dict[str, Decimal]:
    claims: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        claims[payable.creditor] += payable.amount
    for nb_loan in ledger.non_bank_loans:
        if nb_loan.settled or nb_loan.borrower != agent_id:
            continue
        claims[nb_loan.lender] += nb_loan.amount
    for bank_loan in ledger.bank_loans:
        if bank_loan.settled or bank_loan.borrower != agent_id:
            continue
        claims[bank_loan.bank] += bank_loan.amount
    for cb_loan in ledger.cb_loans:
        if cb_loan.settled or cb_loan.bank != agent_id:
            continue
        claims[cb_loan.central_bank] += cb_loan.amount
    for obligation in ledger.delivery_obligations:
        if obligation.settled or obligation.debtor != agent_id:
            continue
        claims[obligation.creditor] += Decimal(obligation.quantity)

    total = sum(claims.values(), ZERO)
    if total <= ZERO:
        return {}
    return {creditor: amount / total for creditor, amount in claims.items()}


def distribute_pro_rata_recovery(ledger: Ledger, agent_id: str) -> None:
    total_liquid = ledger.agent_liquid_assets(agent_id)
    if total_liquid <= ZERO:
        return

    claims: list[tuple[str, Decimal]] = []
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        claims.append((payable.creditor, payable.amount))
    for loan in ledger.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        claims.append((loan.lender, loan.repayment_amount))

    total_claims = sum((amount for _, amount in claims), ZERO)
    if total_claims <= ZERO:
        return

    details = []
    total_distributed = ZERO
    for creditor_id, claim_amount in claims:
        # float round preserved from the existing engine: the recovered share
        # is the float-rounded pro-rata fraction of liquid assets.
        share = Decimal(round(float((claim_amount / total_claims) * total_liquid)))
        if share <= ZERO:
            continue
        transferred = pay_with_deposit(ledger, agent_id, creditor_id, share)
        remainder = share - transferred
        if remainder > ZERO:
            paid_cash = min(ledger.cash[agent_id], remainder)
            if paid_cash > ZERO:
                ledger.transfer_cash(agent_id, creditor_id, paid_cash)
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
        ledger.log(
            "ProRataRecovery",
            agent=agent_id,
            total_liquid=total_liquid,
            total_claims=total_claims,
            total_distributed=total_distributed,
            num_creditors=len(details),
            details=details,
        )


def write_off_liabilities(ledger: Ledger, agent_id: str, *, skip_contract_id: str | None) -> None:
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id or payable.id == skip_contract_id:
            continue
        payable.settled = True
        ledger.log(
            "ObligationWrittenOff",
            contract_id=payable.id,
            alias=payable.alias,
            debtor=payable.debtor,
            creditor=payable.creditor,
            contract_kind="payable",
            amount=payable.amount,
            due_day=payable.due_day,
        )

    for loan in ledger.non_bank_loans:
        if loan.settled or loan.borrower != agent_id or loan.id == skip_contract_id:
            continue
        loan.settled = True
        ledger.log(
            "ObligationWrittenOff",
            contract_id=loan.id,
            alias=None,
            debtor=loan.borrower,
            creditor=loan.lender,
            contract_kind="non_bank_loan",
            amount=loan.amount,
        )

    for bank_loan in ledger.bank_loans:
        if bank_loan.settled or bank_loan.borrower != agent_id or bank_loan.id == skip_contract_id:
            continue
        bank_loan.settled = True
        ledger.log(
            "ObligationWrittenOff",
            contract_id=bank_loan.id,
            alias=None,
            debtor=bank_loan.borrower,
            creditor=bank_loan.bank,
            contract_kind="bank_loan",
            amount=bank_loan.amount,
        )

    for obligation in ledger.delivery_obligations:
        if obligation.settled or obligation.debtor != agent_id or obligation.id == skip_contract_id:
            continue
        obligation.settled = True
        ledger.log(
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


def reassign_receivables(
    ledger: Ledger,
    defaulted_agent_id: str,
    creditor_weights: dict[str, Decimal],
) -> None:
    if not creditor_weights:
        for receivable in ledger.payables:
            if not receivable.settled and receivable.creditor == defaulted_agent_id:
                receivable.settled = True
        return

    for receivable in list(ledger.payables):
        if receivable.settled or receivable.creditor != defaulted_agent_id:
            continue
        if receivable.debtor == defaulted_agent_id:
            continue
        if receivable.debtor in ledger.defaulted_agent_ids:
            continue

        old_payable_id = receivable.id
        original_amount = receivable.amount
        maturity_distance = receivable.maturity_distance
        new_due_day = ledger.day + maturity_distance
        receivable.settled = True

        for creditor_id, weight in creditor_weights.items():
            if creditor_id == receivable.debtor:
                continue
            new_amount = Decimal(int(original_amount * weight))
            if new_amount < Decimal("1"):
                continue
            new_payable_id = f"PAY_reassigned_{len(ledger.payables)}"
            ledger.log(
                "ReceivableReassigned",
                defaulted_agent=defaulted_agent_id,
                debtor=receivable.debtor,
                new_creditor=creditor_id,
                old_payable=old_payable_id,
                new_payable=new_payable_id,
                amount=new_amount,
                weight=float(weight),
                maturity_distance=maturity_distance,
                new_due_day=new_due_day,
            )
            ledger.create_payable(
                payable_id=new_payable_id,
                debtor=receivable.debtor,
                creditor=creditor_id,
                amount=new_amount,
                due_day=new_due_day,
                maturity_distance=maturity_distance,
                reason="receivable_reassignment",
            )


def cancel_scheduled_actions_for_agent(ledger: Ledger, agent_id: str, ctx: RunContext) -> None:
    for scheduled_day, actions in list(ledger.scheduled_actions_by_day.items()):
        remaining_actions = []
        for action in actions:
            if action_references_agent(action, agent_id):
                action_name = next(iter(action.keys()), "unknown")
                ledger.log(
                    "ScheduledActionCancelled",
                    agent=agent_id,
                    scheduled_day=scheduled_day,
                    action=action_name,
                    mode=ctx.default_mode,
                )
            else:
                remaining_actions.append(action)
        if remaining_actions:
            ledger.scheduled_actions_by_day[scheduled_day] = remaining_actions
        else:
            del ledger.scheduled_actions_by_day[scheduled_day]


def settle_delivery_obligation(ledger: Ledger, obligation: DeliveryObligation, ctx: RunContext) -> bool:
    checkpoint = ledger.checkpoint()
    delivered_quantity = deliver_stock_for_obligation(ledger, obligation.debtor, obligation.creditor, obligation.sku, obligation.quantity)
    if delivered_quantity != obligation.quantity:
        shortage = obligation.quantity - delivered_quantity
        if ctx.default_mode != "expel-agent":
            ledger.restore(checkpoint, restore_stocks=True)
            raise DefaultError(
                f"Insufficient stock to settle delivery obligation {obligation.id}: {shortage} units of {obligation.sku} still owed"
            )
        handle_delivery_default(ledger, obligation, delivered_quantity, shortage, ctx)
        return False

    obligation.settled = True
    ledger.log(
        "DeliveryObligationCancelled",
        obligation_id=obligation.id,
        contract_id=obligation.id,
        alias=obligation.alias,
        debtor=obligation.debtor,
        creditor=obligation.creditor,
        sku=obligation.sku,
        qty=obligation.quantity,
    )
    ledger.log(
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


def deliver_stock_for_obligation(ledger: Ledger, debtor: str, creditor: str, sku: str, quantity: int) -> int:
    available = sorted(
        (stock for stock in ledger.stocks.values() if stock.owner == debtor and stock.sku == sku and stock.quantity > 0),
        key=lambda stock: stock.id,
    )
    if not available:
        return 0

    total_available = sum(stock.quantity for stock in available)
    deliver_quantity = min(quantity, total_available)
    remaining = deliver_quantity
    for stock in available:
        if remaining == 0:
            break
        transfer_qty = min(remaining, stock.quantity)
        moving_id = ledger.move_stock_lot(stock, debtor, creditor, transfer_qty)
        moving_stock = ledger.stocks[moving_id]
        ledger.log(
            "StockTransferred",
            frm=debtor,
            to=creditor,
            stock_id=moving_id,
            sku=moving_stock.sku,
            qty=moving_stock.quantity,
        )
        remaining -= transfer_qty
    return deliver_quantity


def rollover_settled_payables(ledger: Ledger, settled_payables: list[tuple[str, str, Decimal, int, Decimal]]) -> list[str]:
    """Refinance settled payables past the latest open maturity (Plan 024).

    The creditor returns the settlement cash to the debtor (deposit first,
    then cash) and a new payable is created at ``max open due day +
    maturity distance``, keeping the debt ring rolling indefinitely.
    """
    max_due_day = ledger.day
    for payable in ledger.payables:
        if payable.settled:
            continue
        if payable.due_day > max_due_day:
            max_due_day = payable.due_day

    new_payable_ids: list[str] = []
    for debtor_id, creditor_id, amount, maturity_distance, cash_return in settled_payables:
        payable_id = rollover_single_payable(
            ledger,
            debtor_id,
            creditor_id,
            amount,
            maturity_distance,
            max_due_day + maturity_distance,
            cash_return=cash_return,
        )
        if payable_id is not None:
            new_payable_ids.append(payable_id)
    return new_payable_ids


def rollover_single_payable(
    ledger: Ledger,
    debtor_id: str,
    creditor_id: str,
    amount: Decimal,
    maturity_distance: int,
    new_due_day: int,
    cash_return: Decimal | None = None,
) -> str | None:
    if cash_return is None:
        cash_return = amount
    if debtor_id not in ledger.agents or debtor_id in ledger.defaulted_agent_ids:
        return None
    if creditor_id not in ledger.agents or creditor_id in ledger.defaulted_agent_ids:
        return None

    new_payable = ledger.add_rollover_payable(
        debtor=debtor_id,
        creditor=creditor_id,
        amount=amount,
        due_day=new_due_day,
        maturity_distance=maturity_distance,
    )

    cash_transferred = ZERO
    if cash_return > ZERO:
        cash_transferred = pay_with_deposit(ledger, creditor_id, debtor_id, cash_return)
        remaining = cash_return - cash_transferred
        if remaining > ZERO:
            cash_paid = min(ledger.cash[creditor_id], remaining)
            if cash_paid > ZERO:
                ledger.transfer_cash(creditor_id, debtor_id, cash_paid)
                cash_transferred += cash_paid

    if cash_return == ZERO:
        # Fully-netted face: the gross roll happens with no cash return-flow.
        ledger.log(
            "PayableRolledOver",
            debtor=debtor_id,
            creditor=creditor_id,
            amount=amount,
            new_due_day=new_due_day,
            maturity_distance=maturity_distance,
            payable_id=new_payable.id,
            cash_transfer=False,
        )
    elif cash_transferred != amount:
        ledger.log(
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
        ledger.log(
            "PayableRolledOver",
            debtor=debtor_id,
            creditor=creditor_id,
            amount=amount,
            new_due_day=new_due_day,
            maturity_distance=maturity_distance,
            payable_id=new_payable.id,
            cash_transfer=True,
        )
    return new_payable.id


def handle_delivery_default(
    ledger: Ledger,
    obligation: DeliveryObligation,
    delivered_quantity: int,
    shortage: int,
    ctx: RunContext,
) -> None:
    if delivered_quantity > 0:
        ledger.log(
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

    ledger.log(
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
    expel_agent(
        ledger,
        obligation.debtor,
        ctx,
        trigger_contract_id=obligation.id,
        trigger_kind="delivery_obligation",
        trigger_shortfall=Decimal(shortage),
    )
