"""Default-fund and VMGH tests for the central counterparty (Plan 061).

Covers the waterfall arithmetic (own/mutualized split with largest-remainder
rounding, recovery credit, VMGH residual), the structural payout-gap draw,
VMGH pro-rata payout conserving the pool exactly, fund collection and the
end-of-day fund invariant, replenishment capped at member cash with
carry-forward, and checkpoint/restore + fail-fast rollback of all new fund
state. All amounts are whole integers.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.config.models import ScenarioConfig
from bilancio.core.errors import DefaultError
from bilancio_v2 import prepare_scenario, run_day, run_scenario, run_until_stable
from bilancio_v2.ledger import CCP_AGENT_KIND, ZERO, InvariantViolation, Ledger, Payable
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.plugins.clearinghouse_ccp import (
    allocate_pro_rata,
    apply_ccp_waterfall,
    draw_fund_for_payout_gap,
)
from bilancio_v2.plugins.settlement import settle_ccp_payouts
from bilancio_v2.policy import CapabilityMatrix
from bilancio_v2.subsystem_config import CleanClearingConfig

D = Decimal


def ccp_scenario(
    payables: list[tuple[str, str, int, int]],
    *,
    n_firms: int = 4,
    cash: dict[str, int] | None = None,
    clearinghouse: dict | None = None,
    default_handling: str = "expel-agent",
    rollover: bool = False,
    max_days: int = 8,
    quiet_days: int = 2,
) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"F{i + 1}", "kind": "firm", "name": f"Firm {i + 1}"} for i in range(n_firms)
    ]
    initial_actions: list[dict] = []
    for agent_id, amount in (cash or {}).items():
        initial_actions.append({"mint_cash": {"to": agent_id, "amount": amount}})
    for debtor, creditor, amount, due_day in payables:
        initial_actions.append({"create_payable": {"from": debtor, "to": creditor, "amount": amount, "due_day": due_day}})
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "ccp-waterfall-test",
            "agents": agents,
            "initial_actions": initial_actions,
            "clearinghouse": {"enabled": True, "mode": "ccp", **(clearinghouse or {})},
            "run": {
                "max_days": max_days,
                "quiet_days": quiet_days,
                "default_handling": default_handling,
                "rollover_enabled": rollover,
            },
        }
    )


def hand_ledger(contributions: dict[str, int], *, ccp_cash: int | None = None) -> Ledger:
    """A bare ccp-mode ledger with the fund pre-seeded (cash backs it)."""
    ledger = Ledger()
    ledger.clearing_config = CleanClearingConfig(mode="ccp")
    ledger.register_agent("CCP1", CCP_AGENT_KIND, "Central Counterparty")
    for member in contributions:
        ledger.register_agent(member, "firm", member)
    total = sum(contributions.values())
    ledger.mint_cash("CCP1", D(ccp_cash if ccp_cash is not None else total))
    ledger.ccp_fund_contribution = {member: D(amount) for member, amount in contributions.items()}
    ledger.ccp_fund_target = {member: D(amount) for member, amount in contributions.items()}
    ledger.ccp_fund_total = D(total)
    return ledger


def events_of(result, kind: str) -> list[dict]:
    return [event for event in result.events if event["kind"] == kind]


# -- largest-remainder allocation ----------------------------------------------


def test_allocate_pro_rata_worked_example() -> None:
    # The plan's worked example: 35 drawn from three contributions of 25
    # each → 12, 12, 11 (largest remainder, ties by sorted id).
    shares = allocate_pro_rata([("B", D(25)), ("C", D(25)), ("D", D(25))], D(35))
    assert shares == {"B": D(12), "C": D(12), "D": D(11)}
    assert sum(shares.values()) == D(35)


def test_allocate_pro_rata_conserves_and_caps() -> None:
    shares = allocate_pro_rata([("A", D(7)), ("B", D(2)), ("C", D(1))], D(10))
    assert shares == {"A": D(7), "B": D(2), "C": D(1)}
    shares = allocate_pro_rata([("A", D(100)), ("B", D(1))], D(50))
    assert sum(shares.values()) == D(50)
    assert all(shares[key] <= cap for key, cap in (("A", D(100)), ("B", D(1))))
    with pytest.raises(InvariantViolation):
        allocate_pro_rata([("A", D(5))], D(6))


# -- waterfall arithmetic (Design §2.5) -----------------------------------------


def test_waterfall_own_then_mutualized_then_residual() -> None:
    ledger = hand_ledger({"A": 25, "B": 25, "C": 25, "D": 25})
    ledger.defaulted_agent_ids.add("A")
    apply_ccp_waterfall(ledger, member="A", shortfall=D(60), recovery=ZERO)
    assert ledger.ccp_fund_contribution == {"A": ZERO, "B": D(13), "C": D(13), "D": D(14)}
    assert ledger.ccp_fund_total == D(40)
    event = ledger.journal.as_dicts()[-1]
    assert event["kind"] == "CCPFundDrawdown"
    assert event["member"] == "A"
    assert event["own_tranche"] == D(25)
    assert event["mutualized_tranche"] == D(35)
    assert event["vmgh_residual"] == ZERO
    assert event["recovery_credit"] == ZERO


def test_waterfall_vmgh_residual_when_fund_exhausted() -> None:
    ledger = hand_ledger({"A": 10, "B": 5, "C": 5})
    ledger.defaulted_agent_ids.add("A")
    apply_ccp_waterfall(ledger, member="A", shortfall=D(30), recovery=ZERO)
    assert ledger.ccp_fund_total == ZERO
    event = ledger.journal.as_dicts()[-1]
    assert event["own_tranche"] == D(10)
    assert event["mutualized_tranche"] == D(10)
    assert event["vmgh_residual"] == D(10)


def test_waterfall_recovery_credits_against_the_drawdown() -> None:
    ledger = hand_ledger({"A": 25, "B": 25, "C": 25, "D": 25})
    ledger.defaulted_agent_ids.add("A")
    apply_ccp_waterfall(ledger, member="A", shortfall=D(60), recovery=D(40))
    # Recovery 40 covers the shortfall first: only 20 is drawn (all own).
    assert ledger.ccp_fund_contribution == {"A": D(5), "B": D(25), "C": D(25), "D": D(25)}
    assert ledger.ccp_fund_total == D(80)
    event = ledger.journal.as_dicts()[-1]
    assert event["recovery_credit"] == D(40)
    assert event["own_tranche"] == D(20)
    assert event["mutualized_tranche"] == ZERO
    assert event["vmgh_residual"] == ZERO


def test_waterfall_skips_defaulted_members_in_mutualized_tranche() -> None:
    ledger = hand_ledger({"A": 10, "B": 20, "C": 20})
    ledger.defaulted_agent_ids.update({"A", "B"})
    apply_ccp_waterfall(ledger, member="A", shortfall=D(40), recovery=ZERO)
    # B is dead: only C's contribution mutualizes.
    assert ledger.ccp_fund_contribution == {"A": ZERO, "B": D(20), "C": ZERO}
    event = ledger.journal.as_dicts()[-1]
    assert event["mutualized_tranche"] == D(20)
    assert event["vmgh_residual"] == D(10)


def test_structural_payout_gap_draw() -> None:
    ledger = hand_ledger({"A": 25, "B": 25, "C": 25})
    drawn = draw_fund_for_payout_gap(ledger, D(35))
    assert drawn == D(35)
    assert ledger.ccp_fund_total == D(40)
    event = ledger.journal.as_dicts()[-1]
    assert event["kind"] == "CCPFundDrawdown"
    assert event["member"] is None
    assert event["mutualized_tranche"] == D(35)
    assert event["vmgh_residual"] == ZERO


# -- VMGH payout leg (Design §2.2) ----------------------------------------------


def test_vmgh_payout_conserves_the_pool_exactly() -> None:
    ledger = hand_ledger({}, ccp_cash=120)  # no fund: 120 free cash
    ledger.register_agent("X", "firm", "X")
    ledger.register_agent("Y", "firm", "Y")
    ledger.day = 1
    ledger.payables.append(
        Payable(
            id="P_out_1", debtor="CCP1", creditor="X", amount=D(120), due_day=1,
            maturity_distance=1, origin_debtor="Z", origin_creditor="X",
        )
    )
    ledger.payables.append(
        Payable(
            id="P_out_2", debtor="CCP1", creditor="Y", amount=D(30), due_day=1,
            maturity_distance=1, origin_debtor="Z", origin_creditor="Y",
        )
    )
    ctx = RunContext(policy=CapabilityMatrix.default(), default_mode="expel-agent")
    assert settle_ccp_payouts(ledger, "CCP1", ctx) is True
    haircuts = [event for event in ledger.journal.as_dicts() if event["kind"] == "VMGHHaircutApplied"]
    assert len(haircuts) == 2
    assert sum(event["paid"] for event in haircuts) == D(120)
    for event in haircuts:
        assert event["paid"] + event["haircut"] == event["face"]
        assert event["haircut_factor"] == pytest.approx(0.8)
    assert all(payable.settled for payable in ledger.payables)
    assert ledger.cash["CCP1"] == ZERO
    assert ledger.cash["X"] == D(96)
    assert ledger.cash["Y"] == D(24)


def test_payout_leg_full_payment_emits_no_haircut() -> None:
    ledger = hand_ledger({}, ccp_cash=50)
    ledger.register_agent("X", "firm", "X")
    ledger.day = 1
    ledger.payables.append(
        Payable(
            id="P_out", debtor="CCP1", creditor="X", amount=D(50), due_day=1,
            maturity_distance=1, origin_debtor="Z", origin_creditor="X",
        )
    )
    ctx = RunContext(policy=CapabilityMatrix.default(), default_mode="expel-agent")
    assert settle_ccp_payouts(ledger, "CCP1", ctx) is True
    events = ledger.journal.as_dicts()
    assert not [event for event in events if event["kind"] == "VMGHHaircutApplied"]
    assert [event for event in events if event["kind"] == "PayableSettled"]
    assert ledger.cash["X"] == D(50)


# -- integrated scenario: worked example from the plan/example yaml --------------


def test_integrated_default_runs_contribution_waterfall_and_haircut() -> None:
    result = run_scenario(
        ccp_scenario(
            [("F1", "F2", 100, 1), ("F2", "F3", 100, 2), ("F3", "F4", 100, 3), ("F4", "F1", 100, 4)],
            cash={"F1": 80, "F2": 100, "F3": 100, "F4": 100},
        )
    )
    ledger = result.ledger
    assert sorted(ledger.defaulted_agent_ids) == ["F1"]
    contributions = events_of(result, "CCPFundContribution")
    assert [(event["member"], event["amount"]) for event in contributions] == [
        ("F1", D(4)),
        ("F2", D(5)),
        ("F3", D(5)),
        ("F4", D(5)),
    ]
    drawdowns = events_of(result, "CCPFundDrawdown")
    assert len(drawdowns) == 1
    assert drawdowns[0]["member"] == "F1"
    assert drawdowns[0]["own_tranche"] == D(4)
    assert drawdowns[0]["mutualized_tranche"] == D(15)
    assert drawdowns[0]["vmgh_residual"] == D(5)
    haircuts = events_of(result, "VMGHHaircutApplied")
    assert len(haircuts) == 1
    assert haircuts[0]["creditor"] == "F2"
    assert haircuts[0]["paid"] == D(95)
    assert haircuts[0]["haircut"] == D(5)
    # Fund invariant held at every day-end (check_invariants ran daily);
    # final state: replenished by survivors, conserved, cash-backed.
    assert ledger.ccp_fund_total == sum(ledger.ccp_fund_contribution.values(), ZERO)
    assert ledger.ccp_fund_total <= ledger.cash["CCP1"]
    assert ledger.ccp_fund_total == D(15)


# -- replenishment (Design §2.6) -------------------------------------------------


def test_replenishment_capped_at_member_cash_with_carry_forward() -> None:
    # Day 1: F1 defaults (cash 0 vs 30 due); waterfall drains F2/F3's
    # contributions. F2 is left with 1 cash so its day-2 top-up is capped,
    # and the residual gap carries to day 3 after F2 is paid by the CCP.
    result = run_scenario(
        ccp_scenario(
            [("F1", "F2", 30, 1), ("F3", "F2", 40, 2)],
            n_firms=3,
            cash={"F1": 0, "F2": 120, "F3": 100},
            max_days=6,
        )
    )
    replenished = events_of(result, "CCPFundReplenished")
    assert replenished, "no replenishment happened"
    by_member: dict[str, list[dict]] = {}
    for event in replenished:
        by_member.setdefault(event["member"], []).append(event)
    # Every top-up respects the cash cap (cash can never go negative — the
    # daily invariant would catch it) and gaps shrink monotonically to zero.
    for member_events in by_member.values():
        gaps = [event["gap_remaining"] for event in member_events]
        assert all(gaps[i] > gaps[i + 1] or gaps[i + 1] == ZERO for i in range(len(gaps) - 1))
    ledger = result.ledger
    for member, target in ledger.ccp_fund_target.items():
        if member not in ledger.defaulted_agent_ids:
            assert ledger.ccp_fund_contribution.get(member, ZERO) == target


def test_replenishment_partial_when_cash_short() -> None:
    ledger = hand_ledger({"A": 10, "B": 10})
    ledger.ccp_fund_contribution["A"] = ZERO
    ledger.ccp_fund_total = D(10)
    ledger.mint_cash("A", D(4))
    from bilancio_v2.plugins.clearinghouse_ccp import replenish_fund

    replenish_fund(ledger, "CCP1")
    event = ledger.journal.as_dicts()[-1]
    assert event["kind"] == "CCPFundReplenished"
    assert event["member"] == "A"
    assert event["amount"] == D(4)
    assert event["gap_remaining"] == D(6)
    assert ledger.ccp_fund_contribution["A"] == D(4)
    assert ledger.cash["A"] == ZERO


def test_replenishment_disabled_skips_top_ups() -> None:
    result = run_scenario(
        ccp_scenario(
            [("F1", "F2", 100, 1), ("F2", "F3", 50, 2)],
            n_firms=3,
            cash={"F1": 40, "F2": 100, "F3": 100},
            clearinghouse={"replenishment_enabled": False},
        )
    )
    assert events_of(result, "CCPFundDrawdown")
    assert not events_of(result, "CCPFundReplenished")


# -- checkpoint / restore and fail-fast rollback ----------------------------------


def test_checkpoint_restores_all_fund_state() -> None:
    ledger = hand_ledger({"A": 25, "B": 25})
    checkpoint = ledger.checkpoint()
    ledger.defaulted_agent_ids.add("A")
    apply_ccp_waterfall(ledger, member="A", shortfall=D(40), recovery=ZERO)
    ledger.ccp_fund_target["C"] = D(9)
    assert ledger.ccp_fund_total == D(10)
    ledger.restore(checkpoint)
    assert ledger.ccp_fund_contribution == {"A": D(25), "B": D(25)}
    assert ledger.ccp_fund_target == {"A": D(25), "B": D(25)}
    assert ledger.ccp_fund_total == D(50)
    assert not [event for event in ledger.journal.as_dicts() if event["kind"] == "CCPFundDrawdown"]


def test_fail_fast_shortfall_rolls_back_atomically() -> None:
    config = ccp_scenario(
        [("F1", "F2", 100, 1)],
        n_firms=2,
        cash={"F1": 40, "F2": 50},
        default_handling="fail-fast",
        max_days=2,
    )
    runtime = prepare_scenario(config)
    run_day(runtime, 0)  # contributions collected, nothing due
    fund_before = dict(runtime.ledger.ccp_fund_contribution)
    total_before = runtime.ledger.ccp_fund_total
    cash_before = dict(runtime.ledger.cash)
    with pytest.raises(DefaultError):
        run_day(runtime, 1)
    ledger = runtime.ledger
    # The settlement attempt was rolled back atomically: cash and the whole
    # fund state are exactly as they were before the failing day.
    assert dict(ledger.ccp_fund_contribution) == fund_before
    assert ledger.ccp_fund_total == total_before
    assert {k: v for k, v in ledger.cash.items() if v} == {k: v for k, v in cash_before.items() if v}
    assert not [event for event in ledger.journal.as_dicts() if event["kind"] == "CCPFundDrawdown"]
    ledger.check_invariants()


def test_run_until_stable_holds_fund_invariant_daily() -> None:
    config = ccp_scenario(
        [("F1", "F2", 100, 1), ("F2", "F3", 80, 2), ("F3", "F1", 90, 3)],
        n_firms=3,
        cash={"F1": 60, "F2": 70, "F3": 80},
        max_days=6,
    )
    runtime = prepare_scenario(config)

    def check_day(rt, day):
        ledger = rt.ledger
        assert ledger.ccp_fund_total == sum(ledger.ccp_fund_contribution.values(), ZERO)
        assert ledger.ccp_fund_total <= ledger.cash["CCP1"]
        assert ledger.cash["CCP1"] >= ZERO

    run_until_stable(runtime, max_days=6, quiet_days=2, day_callback=check_day)
