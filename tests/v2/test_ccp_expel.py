"""Expel-under-the-star tests for the central counterparty (Plan 061 §3).

A defaulting member goes through the EXISTING default pipeline
(``handle_payable_default → expel_agent → reassign_receivables``); under
the novated star its receivables (all CCP1→m legs) must offset rather than
re-link members, its future in-legs are written off, and the CCP1→B
out-legs with ``origin_debtor == m`` stay OPEN — that openness *is* the
mutualization. Includes the explicit regression test on the
``reassign_receivables`` star guard (acceptance criterion 4).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.config.models import ScenarioConfig
from bilancio_v2 import prepare_scenario, run_scenario, run_until_stable
from bilancio_v2.ledger import CCP_AGENT_KIND, CCP_MEMBER_KINDS, ZERO, InvariantViolation, Ledger, Payable
from bilancio_v2.plugins.settlement import reassign_receivables
from bilancio_v2.subsystem_config import CleanClearingConfig

D = Decimal


def ccp_scenario(
    payables: list[tuple[str, str, int, int]],
    *,
    n_firms: int = 4,
    cash: dict[str, int] | None = None,
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
            "name": "ccp-expel-test",
            "agents": agents,
            "initial_actions": initial_actions,
            "clearinghouse": {"enabled": True, "mode": "ccp"},
            "run": {
                "max_days": max_days,
                "quiet_days": quiet_days,
                "default_handling": "expel-agent",
                "rollover_enabled": False,
            },
        }
    )


def events_of(result, kind: str) -> list[dict]:
    return [event for event in result.events if event["kind"] == kind]


def member_to_member(ledger) -> list[Payable]:
    return [
        payable
        for payable in ledger.payables
        if not payable.settled
        and ledger.agents.get(payable.debtor) is not None
        and ledger.agents.get(payable.creditor) is not None
        and ledger.agents[payable.debtor].kind in CCP_MEMBER_KINDS
        and ledger.agents[payable.creditor].kind in CCP_MEMBER_KINDS
    ]


# -- largest-member default (acceptance criterion 4) ----------------------------


def test_largest_member_default_leaves_a_clean_star() -> None:
    # F1 is the largest debtor (100 due day 1, 80 due day 3) with a day-2
    # receivable from F4. It defaults on day 1.
    config = ccp_scenario(
        [("F1", "F2", 100, 1), ("F1", "F3", 80, 3), ("F4", "F1", 90, 2), ("F2", "F4", 50, 2)],
        cash={"F1": 30, "F2": 100, "F3": 60, "F4": 100},
        max_days=6,
    )
    runtime = prepare_scenario(config)
    open_after_default: dict[int, list[Payable]] = {}

    def snapshot(rt, day):
        assert not member_to_member(rt.ledger), f"member↔member payable on day {day}"
        open_after_default[day] = [p for p in rt.ledger.payables if not p.settled]

    result = run_until_stable(runtime, max_days=6, quiet_days=2, day_callback=snapshot)
    ledger = result.ledger
    assert "F1" in ledger.defaulted_agent_ids

    # (i) Zero member↔member payables on every day (checked in the callback).
    # (ii) All CCP1→F1 legs settled (offset) the day F1 died: F1's day-2
    # receivable never pays F1 and is never re-linked to another member.
    day1_open = open_after_default[1]
    assert not [p for p in day1_open if p.creditor == "F1"]
    # (iii) The CCP1→F3 out-leg with origin_debtor == F1 is still OPEN after
    # the default — that openness is the mutualization.
    orphans = [p for p in day1_open if p.debtor == "CCP1" and p.origin_debtor == "F1"]
    assert [(p.creditor, p.amount, p.due_day) for p in orphans] == [("F3", D(80), 3)]
    # (iv) F1's future in-leg was written off at expulsion.
    write_offs = events_of(result, "ObligationWrittenOff")
    assert [(e["debtor"], e["creditor"], e["amount"]) for e in write_offs] == [("F1", "CCP1", D(80))]
    # (v) No receivable reassignment ever created a payable.
    assert not [
        event for event in events_of(result, "PayableCreated") if event.get("reason") == "receivable_reassignment"
    ]
    assert not events_of(result, "ReceivableReassigned")


def test_creditor_of_the_defaulter_is_paid_not_starved() -> None:
    # Baseline contrast: F2's claim survives F1's death. On day 1 the CCP
    # pays F2 from collections + the fund, possibly haircut — F2 is never
    # starved by F1 directly.
    config = ccp_scenario(
        [("F1", "F2", 100, 1)],
        n_firms=2,
        cash={"F1": 20, "F2": 50},
        max_days=4,
    )
    result = run_scenario(config)
    ledger = result.ledger
    assert "F1" in ledger.defaulted_agent_ids
    haircuts = events_of(result, "VMGHHaircutApplied")
    assert len(haircuts) == 1
    event = haircuts[0]
    assert event["creditor"] == "F2"
    assert event["paid"] + event["haircut"] == event["face"] == D(100)
    assert event["paid"] > ZERO  # paid (possibly haircut) rather than starved
    # Arithmetic: contributions F1=1, F2=2 (5% of cash); F1 pays 19 of 100,
    # waterfall draws own 1 + mutualized 2, pool = 19 + 3 = 22 → F2 paid 22;
    # next day F2 replenishes its 2 back into the fund.
    assert event["paid"] == D(22)
    assert ledger.cash["F2"] == D(50) - D(2) + D(22) - D(2)
    # The out-leg is settled either way and the CCP holds no negative cash.
    assert all(p.settled for p in ledger.payables if p.debtor == "CCP1" and p.due_day == 1)
    assert ledger.cash["CCP1"] >= ZERO


def test_orphan_out_leg_is_funded_at_maturity_by_fund_and_vmgh() -> None:
    # F1 dies on day 1; its day-3 in-leg is written off, but the CCP1→F3
    # out-leg survives to maturity, funded by a structural fund draw (the
    # CCPFundDrawdown with member=None) and/or VMGH.
    config = ccp_scenario(
        [("F1", "F2", 100, 1), ("F1", "F3", 80, 3)],
        n_firms=3,
        cash={"F1": 30, "F2": 200, "F3": 60},
        max_days=6,
    )
    result = run_scenario(config)
    ledger = result.ledger
    assert "F1" in ledger.defaulted_agent_ids
    # The orphan out-leg was resolved at maturity, not left hanging.
    orphan = [p for p in ledger.payables if p.debtor == "CCP1" and p.origin_debtor == "F1" and p.due_day == 3]
    assert orphan and all(p.settled for p in orphan)
    day3 = [event for event in result.events if event["day"] == 3]
    structural = [
        event for event in day3 if event["kind"] == "CCPFundDrawdown" and event["member"] is None
    ]
    vmgh = [event for event in day3 if event["kind"] == "VMGHHaircutApplied" and event["creditor"] == "F3"]
    paid_cash = [
        event for event in day3 if event["kind"] == "CashTransferred" and event["frm"] == "CCP1" and event["to"] == "F3"
    ]
    assert structural or vmgh or paid_cash, "orphan out-leg neither funded nor haircut at maturity"
    assert paid_cash, "F3 received nothing at maturity"


# -- the reassign star guard (regression) ---------------------------------------


def make_ccp_ledger() -> Ledger:
    ledger = Ledger()
    ledger.clearing_config = CleanClearingConfig(mode="ccp")
    ledger.register_agent("CCP1", CCP_AGENT_KIND, "Central Counterparty")
    for member in ("F1", "F2", "F3"):
        ledger.register_agent(member, "firm", member)
    return ledger


def test_reassign_guard_raises_on_member_to_member_relink() -> None:
    # Craft the buggy situation the guard exists for: a defaulted member
    # holding a receivable on another MEMBER (impossible under novation)
    # with member creditor weights — reassignment would re-link members.
    ledger = make_ccp_ledger()
    ledger.day = 1
    ledger.defaulted_agent_ids.add("F1")
    ledger.payables.append(
        Payable(id="P_bug", debtor="F2", creditor="F1", amount=D(50), due_day=2, maturity_distance=1)
    )
    with pytest.raises(InvariantViolation, match="member↔member"):
        reassign_receivables(ledger, "F1", {"F3": D(1)})


def test_reassign_offsets_ccp_legs_without_creating_payables() -> None:
    # The natural star case: the dead member's receivables are CCP1→m legs
    # and CCP1 is the sole creditor weight — the skip rule offsets them.
    ledger = make_ccp_ledger()
    ledger.day = 1
    ledger.defaulted_agent_ids.add("F1")
    receivable = Payable(
        id="P_out", debtor="CCP1", creditor="F1", amount=D(90), due_day=2, maturity_distance=1,
        origin_debtor="F2", origin_creditor="F1",
    )
    ledger.payables.append(receivable)
    payables_before = len(ledger.payables)
    reassign_receivables(ledger, "F1", {"CCP1": D(1)})
    assert receivable.settled
    assert len(ledger.payables) == payables_before
    assert not [event for event in ledger.journal.as_dicts() if event["kind"] == "ReceivableReassigned"]
