"""Clearinghouse loan-certificate facility (Subphase B_Certificates, Plan 060 stage 2).

Each day, between :class:`ClearingPhase` (netting) and :class:`SettlementPhase`:

1. **Redemptions** — pledges whose collateral fully settled on a prior day
   are closed: per-diem interest is charged, the member's own certificate
   holdings are burned first (up to the issued amount), the clearinghouse
   keeps cash covering the remainder owed (redemption pool for circulating
   paper), and the true excess returns to the member in cash.
2. **Issuance** — every non-defaulted household/firm with a post-netting
   cash shortfall on its dues today pledges whole, later-due receivables
   (earliest due first) and receives ``int(face × (1 − cert_haircut))``
   bearer certificates, capped per member per day at
   ``max_issuance_per_member × gross dues today``.
3. **Redemption window** — while the clearinghouse holds cash, holders
   redeem certificates for cash at par in sorted-holder order.

Recourse (pledged collateral whose debtor defaults) and the loss waterfall
(clearinghouse interest-margin equity first, then a pro-rata holder haircut;
the clearinghouse itself cannot default) are implemented here and invoked
from the settlement plugin's default machinery.

Everything is deterministic: members, holders, and pledges are visited in
sorted/creation order. The phase returns ``False`` for the stability-impact
signal (like ratings/lending, credit decisions are not impactful settlement
activity).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bilancio_v2.ledger import ZERO, CertificatePledge, Ledger, Payable
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.subsystem_config import CleanClearingConfig

ONE = Decimal("1")
DAYS_PER_YEAR = Decimal("360")

CLEARINGHOUSE_AGENT_ID = "CH1"
CLEARINGHOUSE_AGENT_KIND = "clearinghouse"
CLEARINGHOUSE_AGENT_NAME = "Clearinghouse"


@dataclass(frozen=True)
class CertificateFacilityPhase:
    name = "SubphaseB_Certificates"
    config: CleanClearingConfig

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        process_redemptions(ledger, self.config)
        issue_against_collateral(ledger, self.config)
        run_redemption_window(ledger)
        return False  # credit decisions are not impactful settlement activity


def clearinghouse_id(ledger: Ledger) -> str | None:
    for agent_id in sorted(ledger.agents):
        if ledger.agents[agent_id].kind == CLEARINGHOUSE_AGENT_KIND:
            return agent_id
    return None


def is_clearinghouse(ledger: Ledger, agent_id: str) -> bool:
    agent = ledger.agents.get(agent_id)
    return agent is not None and agent.kind == CLEARINGHOUSE_AGENT_KIND


def per_diem_interest(config: CleanClearingConfig, certificates_issued: Decimal, days_outstanding: int) -> Decimal:
    """Per-diem financing charge: ``int(issued × cert_rate/360 × days)``."""
    if days_outstanding <= 0:
        return ZERO
    return Decimal(int(certificates_issued * config.cert_rate / DAYS_PER_YEAR * Decimal(days_outstanding)))


# ---------------------------------------------------------------------------
# Redemption (collateral matured on a prior day)
# ---------------------------------------------------------------------------


def process_redemptions(ledger: Ledger, config: CleanClearingConfig) -> None:
    """Close pledges whose collateral fully settled on a prior day.

    Pledges are visited in creation order (deterministic). Pledges whose
    collateral debtor defaulted are closed by the settlement machinery
    (``close_pledges_for_defaulted_debtor``), never here.
    """
    for pledge in list(ledger.certificate_pledges):
        if pledge.closed or pledge.settlement_day is None:
            continue
        redeem_pledge(ledger, config, pledge)


def redeem_pledge(ledger: Ledger, config: CleanClearingConfig, pledge: CertificatePledge) -> None:
    ch_id = clearinghouse_id(ledger)
    if ch_id is None:
        return
    member = pledge.member
    days_outstanding = (pledge.settlement_day or ledger.day) - pledge.issuance_day
    interest = per_diem_interest(config, pledge.certificates_issued, days_outstanding)
    ledger.charge_certificate_interest(member, pledge.id, interest, days_outstanding=days_outstanding)

    # Burn the member's own holdings first (decided open question 3): the
    # member pays down its certificate debt at par with whatever it still holds.
    burned = min(ledger.certificates[member], pledge.certificates_issued)
    if burned > ZERO:
        ledger.retire_certificates(member, burned, reason="redemption_burn")

    pledge.closed = True
    owed_after_burn = pledge.certificates_issued + interest - burned
    deficiency = owed_after_burn - pledge.proceeds
    if deficiency > ZERO:
        # Proceeds did not cover the remainder owed (possible only when
        # interest exceeds the haircut buffer): member owes the difference.
        create_certificate_recourse(
            ledger,
            pledge,
            deficiency=deficiency,
            recovered=pledge.proceeds,
            interest=interest,
        )
        return

    excess = -deficiency
    if excess > ZERO and member not in ledger.defaulted_agent_ids:
        # The clearinghouse keeps cash covering owed-after-burn (the
        # redemption pool for circulating paper plus its interest margin) and
        # returns the true excess. Capped at available cash: proceeds paid in
        # certificates were burned at par and carry no cash.
        paid = min(excess, ledger.cash[ch_id])
        if paid > ZERO:
            ledger.transfer_cash(ch_id, member, paid)
            ledger.log("CertificateExcessReturned", member=member, pledge_id=pledge.id, amount=paid)
    # If the member defaulted, the excess stays in the redemption pool
    # (the defaulted member's estate backs still-circulating paper).


# ---------------------------------------------------------------------------
# Issuance (need-based, post-netting shortfall)
# ---------------------------------------------------------------------------


def issue_against_collateral(ledger: Ledger, config: CleanClearingConfig) -> None:
    ch_id = clearinghouse_id(ledger)
    if ch_id is None:
        return
    for member_id in sorted(ledger.agents):
        agent = ledger.agents[member_id]
        if agent.kind not in ("household", "firm") or member_id in ledger.defaulted_agent_ids:
            continue
        # Post-netting gross dues today (ClearingPhase already reduced faces).
        dues_today = sum(
            (
                payable.amount
                for payable in ledger.payables
                if not payable.settled and payable.due_day == ledger.day and payable.debtor == member_id
            ),
            ZERO,
        )
        if dues_today <= ZERO:
            continue
        shortfall = dues_today - ledger.cash[member_id] - ledger.certificates[member_id]
        if shortfall <= ZERO:
            continue
        issuance_cap = Decimal(int(config.max_issuance_per_member * dues_today))
        issued_today = ZERO
        for receivable in eligible_collateral(ledger, config, member_id):
            if shortfall <= ZERO:
                break
            if issued_today >= issuance_cap:
                break  # daily cap reached (whole receivables only, so the
                # final pledge may overshoot by at most one face value)
            certificates = Decimal(int(receivable.amount * (ONE - config.cert_haircut)))
            if certificates <= ZERO:
                continue
            pledge_id = f"CP_{len(ledger.certificate_pledges)}"
            receivable.pledged_to = ch_id
            pledge = CertificatePledge(
                id=pledge_id,
                member=member_id,
                payable_id=receivable.id,
                pledged_face=receivable.amount,
                certificates_issued=certificates,
                issuance_day=ledger.day,
            )
            ledger.certificate_pledges.append(pledge)
            ledger.log(
                "CertificatePledgeCreated",
                pledge_id=pledge_id,
                member=member_id,
                payable_id=receivable.id,
                pledged_face=receivable.amount,
                certificates_issued=certificates,
                issuance_day=ledger.day,
            )
            ledger.issue_certificates(member_id, certificates, pledge_id=pledge_id)
            issued_today += certificates
            shortfall -= certificates


def eligible_collateral(ledger: Ledger, config: CleanClearingConfig, member_id: str) -> list[Payable]:
    """The member's pledgeable receivables, earliest due day first.

    Eligible: unsettled, due strictly after today, within ``cert_max_tenor``
    (no tenor filter when null), not own-issued, unpledged, and the debtor
    has not defaulted.
    """
    eligible = [
        payable
        for payable in ledger.payables
        if not payable.settled
        and payable.creditor == member_id
        and payable.pledged_to is None
        and payable.debtor != member_id
        and payable.due_day > ledger.day
        and payable.debtor not in ledger.defaulted_agent_ids
        and (config.cert_max_tenor is None or payable.due_day - ledger.day <= config.cert_max_tenor)
    ]
    eligible.sort(key=lambda payable: (payable.due_day, payable.id))
    return eligible


# ---------------------------------------------------------------------------
# Redemption window
# ---------------------------------------------------------------------------


def run_redemption_window(ledger: Ledger) -> None:
    """While the clearinghouse holds cash, holders redeem certificates at par.

    Deterministic holder order (sorted agent id); defaulted holders are
    expelled from the ring and do not present for redemption.
    """
    ch_id = clearinghouse_id(ledger)
    if ch_id is None:
        return
    for holder in sorted(ledger.agents):
        if ledger.cash[ch_id] <= ZERO:
            break
        if holder == ch_id or holder in ledger.defaulted_agent_ids:
            continue
        amount = min(ledger.certificates.get(holder, ZERO), ledger.cash[ch_id])
        if amount <= ZERO:
            continue
        ledger.transfer_cash(ch_id, holder, amount)
        ledger.retire_certificates(holder, amount, reason="redemption_window")
        ledger.log("CertificateRedemptionWindow", holder=holder, amount=amount)


def finalize_certificates(ledger: Ledger, *, final_day: int) -> None:
    """End-of-run unconditional redemption (mirrors ``finalize_banking``).

    Pledges that matured on the final simulated day are redeemed, open
    recourse payables are resolved (the member's own certificates settle at
    par; the remainder routes through the loss waterfall), then any remaining
    clearinghouse cash redeems circulating certificates.
    """
    config = ledger.clearing_config
    if config is None or config.mode != "certificates":
        return
    ledger.day = final_day
    process_redemptions(ledger, config)
    resolve_open_recourse(ledger)
    run_redemption_window(ledger)


def resolve_open_recourse(ledger: Ledger) -> None:
    """Resolve recourse payables still open at simulation end.

    A recourse payable created on the final day (due ``final_day + 1``) never
    reaches settlement; without this hook its deficiency would silently
    vanish — no haircut, ``cert_default_losses`` understated. The member's
    own certificate holdings settle at par (burned); the remainder is written
    down through the waterfall.
    """
    for payable in ledger.payables:
        if payable.settled or payable.id not in ledger.certificate_recourse_ids:
            continue
        remaining = payable.amount
        burned = min(ledger.certificates.get(payable.debtor, ZERO), remaining)
        if burned > ZERO:
            ledger.retire_certificates(payable.debtor, burned, reason="recourse_finalize_burn")
            remaining -= burned
        payable.settled = True
        if remaining > ZERO:
            apply_certificate_writedown(ledger, payable.debtor, remaining, trigger=payable.id)


# ---------------------------------------------------------------------------
# Recourse and the loss waterfall (invoked from the settlement plugin)
# ---------------------------------------------------------------------------


def close_pledges_for_defaulted_debtor(ledger: Ledger, debtor_id: str, recoveries: dict[str, Decimal]) -> None:
    """Close every open pledge whose collateral debtor just defaulted.

    The pledge closes at recovered value (partial settlement proceeds already
    routed to the clearinghouse, plus the pro-rata recovery share for the
    pledged claim); the member's deficiency (issued + per-diem interest −
    recovered) becomes a member→CH payable due next day.
    """
    if not ledger.certificate_pledges:
        return
    config = ledger.clearing_config
    if config is None:
        return
    for pledge in ledger.certificate_pledges:
        if pledge.closed:
            continue
        payable = next((candidate for candidate in ledger.payables if candidate.id == pledge.payable_id), None)
        if payable is None or payable.debtor != debtor_id:
            continue
        pledge.proceeds += recoveries.get(pledge.payable_id, ZERO)
        pledge.closed = True
        days_outstanding = ledger.day - pledge.issuance_day
        interest = per_diem_interest(config, pledge.certificates_issued, days_outstanding)
        ledger.charge_certificate_interest(pledge.member, pledge.id, interest, days_outstanding=days_outstanding)
        deficiency = pledge.certificates_issued + interest - pledge.proceeds
        if deficiency > ZERO:
            create_certificate_recourse(
                ledger,
                pledge,
                deficiency=deficiency,
                recovered=pledge.proceeds,
                interest=interest,
            )
        elif -deficiency > ZERO and pledge.member not in ledger.defaulted_agent_ids:
            ch_id = clearinghouse_id(ledger)
            if ch_id is not None:
                paid = min(-deficiency, ledger.cash[ch_id])
                if paid > ZERO:
                    ledger.transfer_cash(ch_id, pledge.member, paid)
                    ledger.log("CertificateExcessReturned", member=pledge.member, pledge_id=pledge.id, amount=paid)


def create_certificate_recourse(
    ledger: Ledger,
    pledge: CertificatePledge,
    *,
    deficiency: Decimal,
    recovered: Decimal,
    interest: Decimal,
) -> None:
    """Create the next-day member→CH recourse payable for a pledge deficiency.

    If the member has already defaulted (it cannot be billed), the deficiency
    goes straight to the loss waterfall.
    """
    ch_id = clearinghouse_id(ledger)
    if ch_id is None:
        return
    if pledge.member in ledger.defaulted_agent_ids:
        apply_certificate_writedown(ledger, pledge.member, deficiency, trigger=pledge.id)
        return
    payable_id = ledger.unique_contract_id("PAY_recourse", len(ledger.payables))
    ledger.create_payable(
        payable_id=payable_id,
        debtor=pledge.member,
        creditor=ch_id,
        amount=deficiency,
        due_day=ledger.day + 1,
        maturity_distance=1,
        reason="certificate_recourse",
    )
    ledger.certificate_recourse_ids.add(payable_id)
    ledger.log(
        "CertificateRecourseCreated",
        member=pledge.member,
        pledge_id=pledge.id,
        payable_id=payable_id,
        deficiency=deficiency,
        recovered=recovered,
        interest=interest,
    )


def apply_certificate_writedown(ledger: Ledger, member: str, loss: Decimal, *, trigger: str) -> None:
    """Loss waterfall (decided open question 6): clearinghouse interest-margin
    equity first, then a pro-rata haircut on all certificate holders.

    The clearinghouse cannot default; any loss beyond margin plus outstanding
    certificates is reported as uncovered (terminal, conservation unaffected).
    """
    if loss <= ZERO:
        return
    absorbed = min(ledger.cert_interest_margin, loss)
    ledger.cert_interest_margin -= absorbed
    remainder = loss - absorbed

    haircut_total = ZERO
    details: list[dict[str, object]] = []
    if remainder > ZERO:
        holders = [
            (holder, ledger.certificates[holder])
            for holder in sorted(ledger.certificates)
            if ledger.certificates[holder] > ZERO and not is_clearinghouse(ledger, holder)
        ]
        outstanding = sum((balance for _, balance in holders), ZERO)
        haircut_total = min(remainder, outstanding)
        if haircut_total > ZERO:
            shares: dict[str, Decimal] = {}
            allocated = ZERO
            for holder, balance in holders:
                share = Decimal(int(haircut_total * balance / outstanding))
                shares[holder] = share
                allocated += share
            leftover = haircut_total - allocated
            for holder, balance in holders:
                if leftover <= ZERO:
                    break
                take = min(balance - shares[holder], leftover)
                if take > ZERO:
                    shares[holder] += take
                    leftover -= take
            for holder, _balance in holders:
                amount = shares[holder]
                if amount > ZERO:
                    ledger.certificates[holder] -= amount
                    details.append({"holder": holder, "amount": amount})
            ledger.certificates_retired_total += haircut_total

    ledger.log(
        "CertificateHaircutApplied",
        member=member,
        trigger=trigger,
        loss=loss,
        absorbed_by_margin=absorbed,
        haircut_total=haircut_total,
        uncovered=remainder - haircut_total,
        details=details,
    )
