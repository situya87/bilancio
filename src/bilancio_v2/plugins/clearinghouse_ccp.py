"""Central counterparty via novation (Plan 061 stage 3).

When the clearinghouse block runs with ``mode: ccp``, every member↔member
payable is rewritten *at creation time* into two legs against the CCP
(A→B becomes A→CCP1 + CCP1→B, both carrying the origin pair), turning the
debt ring into a star. This module hosts the pieces of that institution
that are not settlement-order logic:

* **Novation** — :func:`novate_payable_creation` is the single rewrite used
  by the ``create_payable`` action chokepoint; rollover re-novation lives in
  the settlement plugin but uses the same membership predicates from here.
* **Default fund** — day-0 cash-proportional contributions and next-day
  pro-rata replenishment (capped at member cash, shortfall carried), run at
  the top of :class:`~bilancio_v2.plugins.clearing.ClearingPhase`.
* **Loss waterfall** — own contribution → mutualized pro-rata tranche
  (largest-remainder in whole units) → VMGH residual; recoveries from the
  expelled member credit back against the day's drawdown.
* **Structural fund draw** — when the payout leg is short because a past
  default left CCP1→B legs open without matching in-legs, the remaining
  fund is drawn pro-rata before any VMGH haircut.

Fund draws move **no cash**: the fund physically sits in CCP1's single cash
balance, so a drawdown only re-labels cash the CCP already holds (the
end-of-day ledger invariant keeps ``fund_total ≤ cash[CCP1]``). Everything
is deterministic: members are visited in sorted-id order and all
largest-remainder ties break by sorted id.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

from bilancio_v2.ledger import (
    CCP_AGENT_KIND,
    CCP_MEMBER_KINDS,
    ZERO,
    InvariantViolation,
    Ledger,
    Payable,
)
from bilancio_v2.subsystem_config import CleanClearingConfig

ONE = Decimal("1")

CCP_AGENT_ID = "CCP1"
CCP_AGENT_NAME = "Central Counterparty"


# ---------------------------------------------------------------------------
# Mode / membership predicates
# ---------------------------------------------------------------------------


def ccp_mode_enabled(ledger: Ledger) -> bool:
    config = ledger.clearing_config
    return config is not None and getattr(config, "mode", None) == "ccp"


def ccp_agent_id(ledger: Ledger) -> str | None:
    for agent_id in sorted(ledger.agents):
        if ledger.agents[agent_id].kind == CCP_AGENT_KIND:
            return agent_id
    return None


def active_ccp_id(ledger: Ledger) -> str | None:
    """The CCP agent id when ccp mode is on (else ``None``)."""
    if not ccp_mode_enabled(ledger):
        return None
    return ccp_agent_id(ledger)


def is_member(ledger: Ledger, agent_id: str) -> bool:
    """Clearing members are the ring agents (households/firms); CCP1, the
    central bank, banks, lenders, and rating agencies are never members."""
    agent = ledger.agents.get(agent_id)
    return agent is not None and agent.kind in CCP_MEMBER_KINDS


# ---------------------------------------------------------------------------
# Novation (Design §1)
# ---------------------------------------------------------------------------


def novate_payable_creation(
    ledger: Ledger,
    *,
    debtor: str,
    creditor: str,
    amount: Decimal,
    due_day: int,
    maturity_distance: int,
    alias: str | None,
    setup: bool,
    index: int,
) -> tuple[Payable, Payable]:
    """Rewrite a member↔member payable creation into A→CCP1 + CCP1→B legs.

    Both legs carry identical amount/due_day/maturity_distance and the
    origin pair; the member↔member payable is never instantiated, so the
    novation invariant holds by construction. The alias (used for contract
    lookup) goes to the in-leg; the out-leg id is the in-leg id + ``_out``
    (unique because ``unique_contract_id`` only ever mints ``PREFIX_<int>``).
    """
    ccp = ccp_agent_id(ledger)
    if ccp is None:
        raise InvariantViolation("ccp mode is on but no central_counterparty agent is registered")
    in_id = ledger.unique_contract_id("PAY", index)
    in_leg = ledger.create_payable(
        payable_id=in_id,
        debtor=debtor,
        creditor=ccp,
        amount=amount,
        due_day=due_day,
        maturity_distance=maturity_distance,
        alias=alias,
        setup=setup,
        origin_debtor=debtor,
        origin_creditor=creditor,
    )
    out_leg = ledger.create_payable(
        payable_id=f"{in_id}_out",
        debtor=ccp,
        creditor=creditor,
        amount=amount,
        due_day=due_day,
        maturity_distance=maturity_distance,
        alias=None,
        setup=setup,
        origin_debtor=debtor,
        origin_creditor=creditor,
    )
    return in_leg, out_leg


# ---------------------------------------------------------------------------
# Default fund: collection and replenishment (Design §2.5 / §2.6)
# ---------------------------------------------------------------------------


def run_ccp_fund_step(ledger: Ledger, config: CleanClearingConfig) -> None:
    """The CCP step hosted at the top of ``ClearingPhase`` in ccp mode.

    First invocation (targets unset) collects the initial contributions;
    every later day replenishes toward the original targets.
    """
    ccp = ccp_agent_id(ledger)
    if ccp is None:
        return
    if not ledger.ccp_fund_target:
        collect_initial_contributions(ledger, config, ccp)
        return
    if config.replenishment_enabled:
        replenish_fund(ledger, ccp)


def collect_initial_contributions(ledger: Ledger, config: CleanClearingConfig, ccp: str) -> None:
    """Each member contributes ``int(ccp_fund_share × cash)`` to CCP1.

    Cash moves via ordinary ``transfer_cash`` (conservation holds for free);
    the contribution is additionally recorded in the fund ledger fields and
    pinned as the member's replenishment target.
    """
    for member_id in sorted(ledger.agents):
        if not is_member(ledger, member_id) or member_id in ledger.defaulted_agent_ids:
            continue
        contribution = Decimal(int(config.ccp_fund_share * ledger.cash[member_id]))
        ledger.ccp_fund_target[member_id] = contribution
        if contribution <= ZERO:
            continue
        ledger.transfer_cash(member_id, ccp, contribution)
        ledger.ccp_fund_contribution[member_id] = ledger.ccp_fund_contribution.get(member_id, ZERO) + contribution
        ledger.ccp_fund_total += contribution
        ledger.log(
            "CCPFundContribution",
            member=member_id,
            amount=contribution,
            fund_total=ledger.ccp_fund_total,
        )


def replenish_fund(ledger: Ledger, ccp: str) -> None:
    """Surviving members top up toward their original contribution.

    Capped at available member cash NET of the member's pay-ins due today
    (an uncapped top-up — or one that grabs cash a member needs for today's
    settlement — would manufacture defaults inside a bookkeeping step); the
    shortfall carries forward implicitly as the remaining target gap.
    """
    for member_id in sorted(ledger.ccp_fund_target):
        if member_id in ledger.defaulted_agent_ids:
            continue
        gap = ledger.ccp_fund_target[member_id] - ledger.ccp_fund_contribution.get(member_id, ZERO)
        if gap <= ZERO:
            continue
        dues_today = sum(
            (
                payable.amount
                for payable in ledger.payables
                if not payable.settled and payable.due_day == ledger.day and payable.debtor == member_id
            ),
            ZERO,
        )
        top_up = min(gap, ledger.cash[member_id] - dues_today)
        if top_up <= ZERO:
            continue
        ledger.transfer_cash(member_id, ccp, top_up)
        ledger.ccp_fund_contribution[member_id] = ledger.ccp_fund_contribution.get(member_id, ZERO) + top_up
        ledger.ccp_fund_total += top_up
        ledger.log(
            "CCPFundReplenished",
            member=member_id,
            amount=top_up,
            gap_remaining=gap - top_up,
            fund_total=ledger.ccp_fund_total,
        )


# ---------------------------------------------------------------------------
# Loss waterfall (Design §2.5) and structural payout-gap draw
# ---------------------------------------------------------------------------


def apply_ccp_waterfall(ledger: Ledger, *, member: str, shortfall: Decimal, recovery: Decimal) -> None:
    """Three-tier waterfall for member ``member`` missing its pay-in by ``shortfall``.

    The recovery already routed to CCP1 by the expel machinery credits back
    against the day's drawdown (it is real cash at the CCP, so it directly
    covers part of the missed pay-in). The remainder draws the member's own
    contribution, then the surviving members' contributions pro-rata
    (largest-remainder in whole units, ties by sorted id); any residue is the
    VMGH exposure that shrinks the same day's payout pool.
    """
    recovery_credit = min(max(recovery, ZERO), shortfall)
    residual = shortfall - recovery_credit

    own = min(residual, ledger.ccp_fund_contribution.get(member, ZERO))
    if own > ZERO:
        ledger.ccp_fund_contribution[member] -= own
        ledger.ccp_fund_total -= own
        residual -= own

    mutualized = ZERO
    if residual > ZERO:
        survivors = [
            (member_id, ledger.ccp_fund_contribution[member_id])
            for member_id in sorted(ledger.ccp_fund_contribution)
            if member_id != member
            and member_id not in ledger.defaulted_agent_ids
            and ledger.ccp_fund_contribution[member_id] > ZERO
        ]
        pool = sum((balance for _, balance in survivors), ZERO)
        mutualized = min(residual, pool)
        if mutualized > ZERO:
            shares = allocate_pro_rata(survivors, mutualized)
            for member_id in sorted(shares):
                ledger.ccp_fund_contribution[member_id] -= shares[member_id]
            ledger.ccp_fund_total -= mutualized
            residual -= mutualized

    ledger.log(
        "CCPFundDrawdown",
        member=member,
        own_tranche=own,
        mutualized_tranche=mutualized,
        vmgh_residual=residual,
        recovery_credit=recovery_credit,
        fund_total=ledger.ccp_fund_total,
    )


def draw_fund_for_payout_gap(ledger: Ledger, gap: Decimal) -> Decimal:
    """Draw the remaining fund pro-rata to cover a payout-leg gap.

    A gap with no same-day member default is structural: a past default
    wrote off the member's future in-legs while its CCP1→B out-legs stayed
    open (the mutualization), so they are funded at maturity by fund/VMGH.
    Emitted as a ``CCPFundDrawdown`` with ``member=None``.
    """
    if gap <= ZERO:
        return ZERO
    holders = [
        (member_id, ledger.ccp_fund_contribution[member_id])
        for member_id in sorted(ledger.ccp_fund_contribution)
        if ledger.ccp_fund_contribution[member_id] > ZERO
    ]
    pool = sum((balance for _, balance in holders), ZERO)
    draw = min(gap, pool)
    if draw <= ZERO:
        return ZERO
    shares = allocate_pro_rata(holders, draw)
    for member_id in sorted(shares):
        ledger.ccp_fund_contribution[member_id] -= shares[member_id]
    ledger.ccp_fund_total -= draw
    ledger.log(
        "CCPFundDrawdown",
        member=None,
        own_tranche=ZERO,
        mutualized_tranche=draw,
        vmgh_residual=gap - draw,
        recovery_credit=ZERO,
        fund_total=ledger.ccp_fund_total,
    )
    return draw


# ---------------------------------------------------------------------------
# Largest-remainder allocation (shared by waterfall and VMGH payout)
# ---------------------------------------------------------------------------


def allocate_pro_rata(weights: list[tuple[str, Decimal]], total: Decimal) -> dict[str, Decimal]:
    """Split ``total`` across keys pro-rata to weights, conserving it exactly.

    Largest-remainder rounding in whole Decimal units (ties broken by sorted
    key), each share capped at its weight. ``total`` must not exceed the
    weight sum. Mirrors ``clearing.allocate_single_edge``.
    """
    by_key = dict(weights)
    pool = sum(by_key.values(), ZERO)
    if total > pool:
        raise InvariantViolation(f"pro-rata allocation total {total} exceeds weight sum {pool}")
    if total == pool:
        return dict(by_key)

    shares: dict[str, Decimal] = {}
    remainders: list[tuple[Decimal, str]] = []
    allocated = ZERO
    for key in sorted(by_key):
        exact = total * by_key[key] / pool
        base = exact.to_integral_value(rounding=ROUND_FLOOR)
        shares[key] = base
        allocated += base
        remainders.append((exact - base, key))

    leftover = total - allocated
    order = sorted(remainders, key=lambda item: (-item[0], item[1]))
    for _remainder, key in order:
        if leftover < ONE:
            break
        if shares[key] + ONE <= by_key[key]:
            shares[key] += ONE
            leftover -= ONE
    if leftover > ZERO:
        # Non-integer residue (non-integer weights): spread in tie order,
        # bounded by each key's remaining capacity.
        for _remainder, key in order:
            if leftover <= ZERO:
                break
            take = min(leftover, by_key[key] - shares[key])
            if take > ZERO:
                shares[key] += take
                leftover -= take
    if leftover != ZERO:
        raise InvariantViolation(f"pro-rata allocation left {leftover} unallocated")
    return shares
