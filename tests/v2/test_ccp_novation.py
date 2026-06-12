"""Novation tests for the central counterparty (Plan 061, Design §1).

Covers the rewrite at every payable-creation site (setup actions, scheduled
actions, rollover incl. the netted-rollover queue), the novation invariant
at every day-end, the leg-face reconciliation (acceptance criterion 1),
mode gating (ccp rejects dealer/lender/banking arms), and inertness when
the block is absent or in netting mode. All amounts are whole integers.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.config.models import ClearinghouseConfig, ScenarioConfig
from bilancio.core.errors import ConfigurationError
from bilancio_v2 import prepare_scenario, run_scenario, run_until_stable
from bilancio_v2.ledger import CCP_AGENT_KIND, CCP_MEMBER_KINDS, ZERO
from bilancio_v2.plugins.clearinghouse_ccp import ccp_agent_id
from bilancio_v2.scenario_gates import build_clearing_config

D = Decimal


def ccp_scenario(
    payables: list[tuple[str, str, int, int]],
    *,
    n_firms: int = 4,
    cash: dict[str, int] | None = None,
    clearinghouse: dict | None | bool = None,
    scheduled: list[tuple[int, dict]] | None = None,
    extra_agents: list[dict] | None = None,
    default_handling: str = "expel-agent",
    rollover: bool = False,
    max_days: int = 8,
    quiet_days: int = 2,
    **scenario_extras,
) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"F{i + 1}", "kind": "firm", "name": f"Firm {i + 1}"} for i in range(n_firms)
    ]
    agents.extend(extra_agents or [])
    initial_actions: list[dict] = []
    for agent_id, amount in (cash or {}).items():
        initial_actions.append({"mint_cash": {"to": agent_id, "amount": amount}})
    for debtor, creditor, amount, due_day in payables:
        initial_actions.append({"create_payable": {"from": debtor, "to": creditor, "amount": amount, "due_day": due_day}})
    data: dict = {
        "version": 1,
        "name": "ccp-test",
        "agents": agents,
        "initial_actions": initial_actions,
        "scheduled_actions": [{"day": day, "action": action} for day, action in (scheduled or [])],
        "run": {
            "max_days": max_days,
            "quiet_days": quiet_days,
            "default_handling": default_handling,
            "rollover_enabled": rollover,
        },
        **scenario_extras,
    }
    if clearinghouse is not False:
        data["clearinghouse"] = {"enabled": True, "mode": "ccp", **(clearinghouse or {})}
    return ScenarioConfig.model_validate(data)


def events_of(result, kind: str) -> list[dict]:
    return [event for event in result.events if event["kind"] == kind]


def unsettled(ledger):
    return [payable for payable in ledger.payables if not payable.settled]


def assert_novation_invariant(ledger) -> None:
    for payable in unsettled(ledger):
        debtor = ledger.agents.get(payable.debtor)
        creditor = ledger.agents.get(payable.creditor)
        assert not (
            debtor is not None
            and creditor is not None
            and debtor.kind in CCP_MEMBER_KINDS
            and creditor.kind in CCP_MEMBER_KINDS
        ), f"member↔member payable {payable.id} live in ccp mode"


# -- schema and gating --------------------------------------------------------


def test_ccp_mode_accepted_by_schema_and_gate() -> None:
    config = ccp_scenario([("F1", "F2", 100, 1)])
    clearing = build_clearing_config(config)
    assert clearing is not None and clearing.mode == "ccp"
    assert clearing.ccp_fund_share == D("0.05")
    assert clearing.vmgh_enabled and clearing.replenishment_enabled


def test_vmgh_disabled_rejected_at_schema() -> None:
    with pytest.raises(ValueError, match="vmgh_enabled=false is not supported"):
        ClearinghouseConfig(enabled=True, mode="ccp", vmgh_enabled=False)


def test_ccp_can_fail_rejected_at_schema() -> None:
    with pytest.raises(ValueError, match="ccp_can_fail=true is not supported"):
        ClearinghouseConfig(enabled=True, mode="ccp", ccp_can_fail=True)


def test_ccp_rejects_lender_arm() -> None:
    config = ccp_scenario([("F1", "F2", 100, 1)], lender={"enabled": True})
    with pytest.raises(ConfigurationError, match="cannot be combined with a lender"):
        build_clearing_config(config)


def test_ccp_rejects_dealer_arm() -> None:
    config = ccp_scenario([("F1", "F2", 100, 1)], dealer={"enabled": True})
    with pytest.raises(ConfigurationError, match="cannot be combined with a dealer"):
        build_clearing_config(config)


def test_ccp_rejects_active_balanced_dealer() -> None:
    config = ccp_scenario(
        [("F1", "F2", 100, 1)],
        balanced_dealer={"enabled": True, "mode": "active"},
    )
    with pytest.raises(ConfigurationError, match="active balanced dealer"):
        build_clearing_config(config)


def test_ccp_rejects_banking_config() -> None:
    from bilancio_v2.subsystem_config import CleanBankingConfig

    config = ccp_scenario([("F1", "F2", 100, 1)], cash={"F1": 100})
    with pytest.raises(ConfigurationError, match="cannot be combined with banking"):
        prepare_scenario(config, banking_config=CleanBankingConfig())


# -- inertness when off -------------------------------------------------------


def test_absent_block_emits_no_ccp_state_or_events() -> None:
    config = ccp_scenario([("F1", "F2", 100, 1)], cash={"F1": 100}, clearinghouse=False)
    result = run_scenario(config)
    assert not any(event["kind"].startswith(("CCP", "VMGH")) for event in result.events)
    assert not any(agent.kind == CCP_AGENT_KIND for agent in result.ledger.agents.values())
    assert not any("origin_debtor" in event for event in result.events)
    assert all(payable.origin_debtor is None for payable in result.ledger.payables)
    assert result.ledger.ccp_fund_total == ZERO and not result.ledger.ccp_fund_contribution


def test_netting_mode_unchanged_by_ccp_code() -> None:
    config = ccp_scenario([("F1", "F2", 100, 1)], cash={"F1": 100}, clearinghouse={"mode": "netting"})
    result = run_scenario(config)
    assert not any(event["kind"].startswith(("CCP", "VMGH")) for event in result.events)
    assert not any(agent.kind == CCP_AGENT_KIND for agent in result.ledger.agents.values())
    assert all(payable.origin_debtor is None for payable in result.ledger.payables)


# -- agent registration and phase order ---------------------------------------


def test_ccp_mode_registers_agent_but_no_new_phase_slot() -> None:
    runtime = prepare_scenario(ccp_scenario([("F1", "F2", 100, 1)], cash={"F1": 100}))
    assert [phase.name for phase in runtime.phases] == ["SubphaseB1", "SubphaseB_Clearing", "SubphaseB2", "PhaseC"]
    ccp = ccp_agent_id(runtime.ledger)
    assert ccp == "CCP1"
    assert runtime.ledger.agents[ccp].kind == CCP_AGENT_KIND
    assert runtime.ledger.agents[ccp].name == "Central Counterparty"
    assert runtime.ctx.policy.mop_order(CCP_AGENT_KIND) == ("cash",)
    assert runtime.ctx.policy.rows[CCP_AGENT_KIND].can_default is False


# -- novation at the creation sites --------------------------------------------


def test_setup_action_is_novated_into_two_legs() -> None:
    runtime = prepare_scenario(ccp_scenario([("F1", "F2", 100, 3)], cash={"F1": 100}))
    ledger = runtime.ledger
    assert len(ledger.payables) == 2
    in_leg, out_leg = ledger.payables
    assert (in_leg.debtor, in_leg.creditor) == ("F1", "CCP1")
    assert (out_leg.debtor, out_leg.creditor) == ("CCP1", "F2")
    for leg in (in_leg, out_leg):
        assert leg.amount == D(100)
        assert leg.due_day == 3
        assert leg.maturity_distance == 3
        assert (leg.origin_debtor, leg.origin_creditor) == ("F1", "F2")
    assert out_leg.id == f"{in_leg.id}_out"
    created = [event for event in ledger.journal.as_dicts() if event["kind"] == "PayableCreated"]
    assert len(created) == 2
    for event in created:
        assert event["origin_debtor"] == "F1"
        assert event["origin_creditor"] == "F2"


def test_alias_goes_to_the_in_leg() -> None:
    config = ccp_scenario([], cash={"F1": 100})
    config.initial_actions.append(
        {"create_payable": {"from": "F1", "to": "F2", "amount": 50, "due_day": 2, "alias": "ring_leg"}}
    )
    runtime = prepare_scenario(config)
    in_leg, out_leg = runtime.ledger.payables
    assert in_leg.alias == "ring_leg"
    assert out_leg.alias is None
    assert runtime.ledger.contract_id_for_alias("ring_leg") == in_leg.id


def test_scheduled_action_is_novated() -> None:
    config = ccp_scenario(
        [],
        cash={"F1": 100},
        scheduled=[(1, {"create_payable": {"from": "F1", "to": "F2", "amount": 60, "due_day": 3}})],
        max_days=2,
        quiet_days=5,
    )
    runtime = prepare_scenario(config)
    run_until_stable(runtime, max_days=2, quiet_days=5)
    legs = [payable for payable in runtime.ledger.payables if payable.origin_debtor == "F1"]
    assert {(leg.debtor, leg.creditor) for leg in legs} == {("F1", "CCP1"), ("CCP1", "F2")}
    assert_novation_invariant(runtime.ledger)


def test_member_to_nonmember_payable_is_not_novated() -> None:
    config = ccp_scenario(
        [],
        cash={"F1": 100},
        extra_agents=[{"id": "L1", "kind": "non_bank_lender", "name": "Lender"}],
    )
    config.initial_actions.append({"create_payable": {"from": "F1", "to": "L1", "amount": 40, "due_day": 2}})
    runtime = prepare_scenario(config)
    assert len(runtime.ledger.payables) == 1
    payable = runtime.ledger.payables[0]
    assert (payable.debtor, payable.creditor) == ("F1", "L1")
    assert payable.origin_debtor is None


# -- novation invariant and leg-face reconciliation (acceptance criterion 1) ---


def test_novation_invariant_and_leg_reconciliation_every_day() -> None:
    # F1 defaults on day 1 (cash 40 vs 100 due); its day-3 in-leg is written
    # off while the matching out-leg stays open, so the leg-face difference
    # must reconcile exactly to the written-off in-legs of defaulted members.
    config = ccp_scenario(
        [("F1", "F2", 100, 1), ("F1", "F3", 80, 3), ("F2", "F4", 50, 2), ("F4", "F1", 90, 4)],
        cash={"F1": 40, "F2": 120, "F3": 60, "F4": 100},
        max_days=6,
        quiet_days=2,
    )
    runtime = prepare_scenario(config)

    def check_day(rt, day):
        ledger = rt.ledger
        assert_novation_invariant(ledger)
        in_face = sum((p.amount for p in unsettled(ledger) if p.creditor == "CCP1"), ZERO)
        out_face = sum((p.amount for p in unsettled(ledger) if p.debtor == "CCP1"), ZERO)
        # Reconciliation: out-legs may exceed in-legs by exactly the open
        # out-face whose origin debtor has defaulted (the mutualization),
        # while offset receivables of dead members shrink the out side.
        orphan_out = sum(
            (p.amount for p in unsettled(ledger) if p.debtor == "CCP1" and p.origin_debtor in ledger.defaulted_agent_ids),
            ZERO,
        )
        orphan_in = sum(
            (p.amount for p in unsettled(ledger) if p.creditor == "CCP1" and p.origin_creditor in ledger.defaulted_agent_ids),
            ZERO,
        )
        assert in_face - orphan_in == out_face - orphan_out, f"leg faces diverged on day {day}"
        if not ledger.defaulted_agent_ids:
            assert in_face == out_face

    result = run_until_stable(runtime, max_days=6, quiet_days=2, day_callback=check_day)
    ledger = result.ledger
    assert "F1" in ledger.defaulted_agent_ids
    # F1's day-3 in-leg was written off; the CCP1→F3 out-leg survived and
    # was funded at maturity (possibly haircut) rather than starving F3.
    write_offs = [
        event
        for event in result.events
        if event["kind"] == "ObligationWrittenOff" and event["debtor"] == "F1" and event["creditor"] == "CCP1"
    ]
    assert sum(event["amount"] for event in write_offs) == D(80)
    assert not [p for p in unsettled(ledger) if p.origin_debtor == "F1"]


# -- rollover re-novation (Design §2.3) ----------------------------------------


def test_rollover_re_novates_the_origin_pair() -> None:
    config = ccp_scenario(
        [("F1", "F2", 100, 1)],
        cash={"F1": 120, "F2": 50},
        rollover=True,
        max_days=3,
        quiet_days=10,
    )
    runtime = prepare_scenario(config)
    run_until_stable(runtime, max_days=3, quiet_days=10)
    ledger = runtime.ledger
    assert_novation_invariant(ledger)
    rolled = [payable for payable in ledger.payables if payable.id.startswith("PAY_rollover_")]
    assert rolled, "rollover produced no payables"
    assert len(rolled) % 2 == 0  # always novated pairs
    for payable in rolled:
        assert "CCP1" in (payable.debtor, payable.creditor)
        assert payable.origin_debtor == "F1" and payable.origin_creditor == "F2"
    # The rollover event shape is unchanged: it names the ORIGIN pair and
    # the in-leg id, with the cash return-flow running B→A directly.
    events = [event for event in runtime.ledger.journal.as_dicts() if event["kind"] in ("PayableRolledOver", "RolloverPartial")]
    assert events
    for event in events:
        assert event["debtor"] == "F1" and event["creditor"] == "F2"


def test_fully_netted_legs_roll_the_origin_pair_without_cash() -> None:
    # F1↔F2 owe each other 100 due day 1: on the star both in-legs net
    # against both out-legs, and the netted-rollover queue must carry the
    # ORIGIN pairs (gross-roll, zero cash return) — never a CCP leg.
    config = ccp_scenario(
        [("F1", "F2", 100, 1), ("F2", "F1", 100, 1)],
        cash={"F1": 50, "F2": 50},
        rollover=True,
        max_days=2,
        quiet_days=10,
    )
    runtime = prepare_scenario(config)
    run_until_stable(runtime, max_days=2, quiet_days=10)
    ledger = runtime.ledger
    assert_novation_invariant(ledger)
    rolled_events = [event for event in ledger.journal.as_dicts() if event["kind"] == "PayableRolledOver"]
    no_cash = [event for event in rolled_events if event["cash_transfer"] is False]
    assert {(event["debtor"], event["creditor"]) for event in no_cash} == {("F1", "F2"), ("F2", "F1")}
    for payable in ledger.payables:
        if payable.id.startswith("PAY_rollover_"):
            assert "CCP1" in (payable.debtor, payable.creditor)
            assert payable.origin_debtor is not None


def test_golden_example_matches_capture() -> None:
    # ring_ccp.yaml is also pinned by the golden suite; this is a cheap
    # guard that the example still runs to a stable star with one default.
    from bilancio_v2 import load_scenario
    from tests.v2.golden_io import SCENARIO_DIR

    result = run_scenario(load_scenario(SCENARIO_DIR / "ring_ccp.yaml"))
    assert result.reached_stable
    assert sorted(result.ledger.defaulted_agent_ids) == ["F1"]
    assert_novation_invariant(result.ledger)
