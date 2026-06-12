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
from bilancio_v2.ledger import ZERO, DeliveryObligation, InvariantViolation, Ledger, Payable
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.plugins.certificates import (
    apply_certificate_writedown,
    close_pledges_for_defaulted_debtor,
    is_clearinghouse,
)
from bilancio_v2.plugins.clearinghouse_ccp import (
    active_ccp_id,
    allocate_pro_rata,
    apply_ccp_waterfall,
    draw_fund_for_payout_gap,
    is_member,
)
from bilancio_v2.policy import MOP_CLEARINGHOUSE_CERT


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
        ccp = active_ccp_id(ledger)
        if ccp is not None:
            # Two-leg settlement day (Plan 061 §2.2): pay-ins (creditor ==
            # CCP1) settle first via the normal mop path (shortfalls run the
            # waterfall inside the default machinery), then the CCP payout
            # leg (with VMGH haircut if the pool is short), then all other
            # payables exactly as today. CCP legs are atomic per day: the
            # CCP never pays before collecting, so cash[CCP1] ≥ 0 holds by
            # construction.
            for payable in list(ledger.payables):
                if payable.settled or payable.due_day != ledger.day or payable.creditor != ccp:
                    continue
                settled, rollover_info = settle_payable(ledger, payable, ctx)
                impactful = settled or impactful
                if rollover_info is not None:
                    settled_for_rollover.append(rollover_info)
            impactful = settle_ccp_payouts(ledger, ccp, ctx) or impactful
        for payable in list(ledger.payables):
            if payable.settled or payable.due_day != ledger.day:
                continue
            if ccp is not None and (payable.creditor == ccp or payable.debtor == ccp):
                continue  # CCP legs already handled atomically above
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
    # Pledged collateral (Plan 060): settlement proceeds are redirected to
    # the clearinghouse instead of the creditor of record.
    receiver = payable.pledged_to or payable.creditor
    for means in ctx.policy.mop_order(debtor.kind):
        if remaining <= 0:
            break
        if means == "cash":
            paid = min(ledger.cash[payable.debtor], remaining)
            if paid:
                ledger.transfer_cash(payable.debtor, receiver, paid)
                remaining -= paid
        elif means == "bank_deposit":
            paid = pay_with_deposit(
                ledger,
                payable.debtor,
                receiver,
                remaining,
                banking_config=ctx.banking_config,
            )
            remaining -= paid
        elif means == MOP_CLEARINGHOUSE_CERT:
            paid = min(ledger.certificates[payable.debtor], remaining)
            if paid:
                ledger.transfer_certificates(payable.debtor, receiver, paid)
                if is_clearinghouse(ledger, receiver):
                    # The clearinghouse receiving its own liability retires it.
                    ledger.retire_certificates(receiver, paid, reason="payment_to_clearinghouse")
                remaining -= paid
        # Other means (e.g. reserve_deposit) have no payable-settlement
        # channel in this slice, matching the existing engine.

    if payable.pledged_to is not None:
        record_pledge_proceeds(ledger, payable, payable.amount - remaining, fully_settled=remaining == ZERO)

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
    if (
        ctx.rollover_enabled
        and payable.pledged_to is None
        and payable.id not in ledger.certificate_recourse_ids
        and not is_clearinghouse(ledger, payable.creditor)
    ):
        # Partially netted payables roll at full original face; only the
        # cash-settled residual generates a cash return-flow. Pledged
        # collateral never rolls (its proceeds belong to the clearinghouse),
        # neither do certificate recourse payables (the clearinghouse would
        # otherwise return the recovered cash to the member), and neither do
        # claims reassigned to the clearinghouse itself (rolling one would
        # bounce the recovery back to the payer forever).
        ccp = active_ccp_id(ledger)
        if ccp is not None and payable.debtor == ccp:
            # CCP1→B out-legs NEVER enqueue rollover (Plan 061 §2.3): they
            # have no independent economic life — the origin pair rolls
            # once, recorded from the A→CCP1 in-leg.
            rollover_info = None
        elif ccp is not None and payable.creditor == ccp:
            if payable.origin_debtor is None or payable.origin_creditor is None:
                raise InvariantViolation(f"ccp in-leg {payable.id} is missing its origin pair")
            # Roll the ORIGIN pair at gross face; the cash return-flow is
            # the cash-settled portion of this in-leg and runs B→A directly
            # (refinancing is a new bilateral credit decision — only the
            # resulting payable is re-novated).
            rollover_info = (
                payable.origin_debtor,
                payable.origin_creditor,
                payable.amount + payable.netted_amount,
                payable.maturity_distance,
                payable.amount,
            )
        else:
            rollover_info = (
                payable.debtor,
                payable.creditor,
                payable.amount + payable.netted_amount,
                payable.maturity_distance,
                payable.amount,
            )
    return True, rollover_info


def record_pledge_proceeds(ledger: Ledger, payable: Payable, paid: Decimal, *, fully_settled: bool) -> None:
    """Track settlement proceeds routed to the clearinghouse for a pledged payable."""
    for pledge in ledger.certificate_pledges:
        if pledge.payable_id != payable.id or pledge.closed:
            continue
        if paid > ZERO:
            pledge.proceeds += paid
        if fully_settled:
            pledge.settlement_day = ledger.day
        return


def settle_ccp_payouts(ledger: Ledger, ccp: str, ctx: RunContext) -> bool:
    """The CCP payout leg (Plan 061 §2.2 step 2), after all pay-ins.

    The payout pool is the CCP's free cash (cash minus the fund) — i.e.
    today's collections, recoveries, and any prior surplus. If the pool is
    short, the remaining fund is drawn pro-rata (structural gaps from past
    defaults); if it is still short, VMGH applies haircut factor
    ``h = pool/required`` pro-rata across every payout (largest-remainder in
    whole units, conserving the pool exactly), each leg is marked settled,
    and the haircut residue is a final loss to the receiving member.
    """
    due_out = [
        payable
        for payable in ledger.payables
        if not payable.settled and payable.due_day == ledger.day and payable.debtor == ccp
    ]
    if not due_out:
        return False
    required = sum((payable.amount for payable in due_out), ZERO)
    available = ledger.cash[ccp] - ledger.ccp_fund_total
    if available < required:
        available += draw_fund_for_payout_gap(ledger, required - available)

    if available >= required:
        impactful = False
        for payable in due_out:
            settled, rollover_info = settle_payable(ledger, payable, ctx)
            if rollover_info is not None:
                raise InvariantViolation(f"ccp out-leg {payable.id} produced a rollover entry")
            impactful = settled or impactful
        return impactful

    # VMGH haircut day: pay h·amount per leg, conserving the pool exactly.
    pool = available
    haircut_factor = float(pool / required)
    shares = allocate_pro_rata([(payable.id, payable.amount) for payable in due_out], pool)
    for payable in due_out:
        paid = shares.get(payable.id, ZERO)
        if paid > ZERO:
            ledger.transfer_cash(ccp, payable.creditor, paid)
        payable.settled = True
        ledger.log(
            "VMGHHaircutApplied",
            contract_id=payable.id,
            creditor=payable.creditor,
            face=payable.amount,
            paid=paid,
            haircut=payable.amount - paid,
            haircut_factor=haircut_factor,
        )
    return True


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
    ccp = active_ccp_id(ledger)
    ccp_pay_in_shortfall = ccp is not None and payable.creditor == ccp
    ccp_liquid_before = ledger.agent_liquid_assets(ccp) if ccp_pay_in_shortfall else ZERO
    expel_agent(
        ledger,
        payable.debtor,
        ctx,
        trigger_contract_id=payable.id,
        trigger_shortfall=remaining,
    )
    update_dealer_risk_history(ledger, issuer_id=payable.debtor, defaulted=True)
    reassign_receivables(ledger, payable.debtor, creditor_weights)
    if ccp_pay_in_shortfall:
        # Loss waterfall (Plan 061 §2.5): the member's missed pay-in is
        # absorbed by recovery (already routed to CCP1 by the expel
        # machinery), then its own fund contribution, then the mutualized
        # tranche; any residue shrinks the same day's payout pool (VMGH).
        recovery = ledger.agent_liquid_assets(ccp) - ccp_liquid_before
        apply_ccp_waterfall(ledger, member=payable.debtor, shortfall=remaining, recovery=recovery)
    if payable.id in ledger.certificate_recourse_ids:
        # Member defaulted on a certificate recourse payable: the unpaid
        # remainder routes through the clearinghouse loss waterfall
        # (interest-margin equity first, then a pro-rata holder haircut).
        apply_certificate_writedown(ledger, payable.debtor, remaining, trigger=payable.id)
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
    recoveries = distribute_pro_rata_recovery(ledger, agent_id)
    write_off_liabilities(ledger, agent_id, skip_contract_id=trigger_contract_id, recoveries=recoveries)
    # Pledged collateral whose debtor just defaulted: close the pledge at
    # recovered value and bill the member's deficiency (Plan 060 recourse).
    close_pledges_for_defaulted_debtor(ledger, agent_id, recoveries)
    cancel_scheduled_actions_for_agent(ledger, agent_id, ctx)


def collect_creditor_weights(ledger: Ledger, agent_id: str) -> dict[str, Decimal]:
    claims: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        # Pledged claims belong to the clearinghouse, not the creditor of record.
        claims[payable.pledged_to or payable.creditor] += payable.amount
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


def distribute_pro_rata_recovery(ledger: Ledger, agent_id: str) -> dict[str, Decimal]:
    """Distribute the defaulted agent's liquid assets pro-rata to its creditors.

    Returns the recovered amount per claim contract id (consumed by the
    certificate recourse path; empty outside certificates mode behavior).
    """
    recovered_by_contract: dict[str, Decimal] = {}
    total_liquid = ledger.agent_liquid_assets(agent_id)
    if total_liquid <= ZERO:
        return recovered_by_contract

    claims: list[tuple[str, Decimal, str]] = []
    for payable in ledger.payables:
        if payable.settled or payable.debtor != agent_id:
            continue
        # Pledged claims (Plan 060): recovery routes to the clearinghouse.
        claims.append((payable.pledged_to or payable.creditor, payable.amount, payable.id))
    for loan in ledger.non_bank_loans:
        if loan.settled or loan.borrower != agent_id:
            continue
        claims.append((loan.lender, loan.repayment_amount, loan.id))

    total_claims = sum((amount for _, amount, _ in claims), ZERO)
    if total_claims <= ZERO:
        return recovered_by_contract

    details = []
    total_distributed = ZERO
    for creditor_id, claim_amount, contract_id in claims:
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
        remainder = share - transferred
        if remainder > ZERO:
            # Certificates transfer last (after deposits and cash, Plan 060).
            paid_certificates = min(ledger.certificates[agent_id], remainder)
            if paid_certificates > ZERO:
                ledger.transfer_certificates(agent_id, creditor_id, paid_certificates)
                if is_clearinghouse(ledger, creditor_id):
                    ledger.retire_certificates(creditor_id, paid_certificates, reason="payment_to_clearinghouse")
                transferred += paid_certificates
        if transferred > ZERO:
            total_distributed += transferred
            recovered_by_contract[contract_id] = recovered_by_contract.get(contract_id, ZERO) + transferred
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
    return recovered_by_contract


def write_off_liabilities(
    ledger: Ledger,
    agent_id: str,
    *,
    skip_contract_id: str | None,
    recoveries: dict[str, Decimal] | None = None,
) -> None:
    recoveries = recoveries or {}
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
        if payable.id in ledger.certificate_recourse_ids:
            # An open recourse payable dies with the member: the unrecovered
            # face routes through the clearinghouse loss waterfall.
            loss = payable.amount - recoveries.get(payable.id, ZERO)
            apply_certificate_writedown(ledger, agent_id, loss, trigger=payable.id)

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
            if not receivable.settled and receivable.creditor == defaulted_agent_id and receivable.pledged_to is None:
                receivable.settled = True
        return

    for receivable in list(ledger.payables):
        if receivable.settled or receivable.creditor != defaulted_agent_id:
            continue
        if receivable.pledged_to is not None:
            # Pledged collateral stays alive and routed to the clearinghouse;
            # it is never reassigned to the defaulted member's creditors.
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
            if (
                active_ccp_id(ledger) is not None
                and is_member(ledger, receivable.debtor)
                and is_member(ledger, creditor_id)
            ):
                # Star guard (Plan 061 §3): under ccp mode a defaulted
                # member's receivables are all CCP1→m legs, and CCP1 is the
                # sole creditor weight, so the skip rule above offsets them.
                # Reaching this branch would re-link two members and break
                # the novation invariant — it must be structurally
                # unreachable.
                raise InvariantViolation(
                    f"reassignment of {defaulted_agent_id}'s receivable {old_payable_id} would create "
                    f"a member↔member payable ({receivable.debtor} → {creditor_id}) in ccp mode"
                )
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
    ccp = active_ccp_id(ledger)
    for debtor_id, creditor_id, amount, maturity_distance, cash_return in settled_payables:
        if ccp is not None and ccp in (debtor_id, creditor_id):
            # Rollover entries in ccp mode are origin pairs (Plan 061 §2.3);
            # a CCP leg in the queue means an out-leg enqueued itself — a bug.
            raise InvariantViolation(f"ccp leg ({debtor_id} → {creditor_id}) reached the rollover queue")
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

    ccp = active_ccp_id(ledger)
    if ccp is not None and is_member(ledger, debtor_id) and is_member(ledger, creditor_id):
        # Re-novation at roll time (Plan 061 §2.3): rollover entries in ccp
        # mode are origin pairs, so the rolled payable is recreated as fresh
        # A→CCP1 + CCP1→B legs. Like all rollover payables they emit no
        # PayableCreated event (observable through PayableRolledOver, whose
        # payable_id is the in-leg); the cash return-flow below runs B→A
        # directly, unchanged.
        new_payable = ledger.add_rollover_payable(
            debtor=debtor_id,
            creditor=ccp,
            amount=amount,
            due_day=new_due_day,
            maturity_distance=maturity_distance,
            origin_debtor=debtor_id,
            origin_creditor=creditor_id,
        )
        ledger.add_rollover_payable(
            debtor=ccp,
            creditor=creditor_id,
            amount=amount,
            due_day=new_due_day,
            maturity_distance=maturity_distance,
            origin_debtor=debtor_id,
            origin_creditor=creditor_id,
        )
    else:
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
                remaining -= cash_paid
        if remaining > ZERO:
            # Certificates return last (Plan 060): rollover re-lends what was
            # repaid, and face settled in certificates must flow back the same
            # way or the debtor would owe the rolled face twice over.
            cert_balance = ledger.certificates.get(creditor_id, ZERO)
            cert_paid = min(cert_balance, remaining)
            if cert_paid > ZERO:
                ledger.transfer_certificates(creditor_id, debtor_id, cert_paid)
                cash_transferred += cert_paid

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
    elif cash_transferred != cash_return:
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
