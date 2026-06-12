"""Unit tests for the clearinghouse netting phase (Plan 059): gating,
conservation, and the netting algorithm (cycles, pro-rata allocation,
determinism)."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import pytest

from bilancio.config.models import ClearinghouseConfig, ScenarioConfig
from bilancio.core.errors import ConfigurationError
from bilancio_v2 import prepare_scenario, run_scenario
from bilancio_v2.ledger import ZERO, Ledger, Payable
from bilancio_v2.plugins.clearing import (
    allocate_single_edge,
    cancel_bilateral,
    cancel_cycles,
    collect_due_payables,
    find_cycle,
)
from bilancio_v2.scenario_gates import build_clearing_config

D = Decimal


def ring_scenario(
    faces: list[int],
    *,
    cash: list[int] | None = None,
    clearing: bool = True,
    default_handling: str = "expel-agent",
    rollover: bool = False,
    max_days: int = 3,
) -> ScenarioConfig:
    n = len(faces)
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"F{i + 1}", "kind": "firm", "name": f"Firm {i + 1}"} for i in range(n)
    ]
    initial_actions: list[dict] = []
    for i, amount in enumerate(cash or []):
        if amount:
            initial_actions.append({"mint_cash": {"to": f"F{i + 1}", "amount": amount}})
    for i, face in enumerate(faces):
        initial_actions.append(
            {
                "create_payable": {
                    "from": f"F{i + 1}",
                    "to": f"F{(i + 1) % n + 1}",
                    "amount": face,
                    "due_day": 1,
                }
            }
        )
    data: dict = {
        "version": 1,
        "name": "clearing-test-ring",
        "agents": agents,
        "initial_actions": initial_actions,
        "run": {
            "max_days": max_days,
            "quiet_days": 10,
            "default_handling": default_handling,
            "rollover_enabled": rollover,
        },
    }
    if clearing:
        data["clearinghouse"] = {"enabled": True, "mode": "netting"}
    return ScenarioConfig.model_validate(data)


def make_payable(payable_id: str, debtor: str, creditor: str, amount: int, *, due_day: int = 1) -> Payable:
    return Payable(
        id=payable_id,
        debtor=debtor,
        creditor=creditor,
        amount=D(amount),
        due_day=due_day,
        maturity_distance=1,
    )


CLEARING_EVENT_KINDS = {"SubphaseB_Clearing", "PayableNetted", "ClearingExecuted"}


# -- gating (acceptance criterion 1) ---------------------------------------


def test_absent_block_registers_no_phase_and_emits_no_events() -> None:
    config = ring_scenario([100] * 5, clearing=False)
    runtime = prepare_scenario(config)
    assert [phase.name for phase in runtime.phases] == ["SubphaseB1", "SubphaseB2", "PhaseC"]

    result = run_scenario(config)
    assert not CLEARING_EVENT_KINDS & {event["kind"] for event in result.events}


def test_enabled_false_registers_no_phase() -> None:
    config = ring_scenario([100] * 5, clearing=False)
    config = config.model_copy(update={"clearinghouse": ClearinghouseConfig(enabled=False)})
    assert build_clearing_config(config) is None
    runtime = prepare_scenario(config)
    assert [phase.name for phase in runtime.phases] == ["SubphaseB1", "SubphaseB2", "PhaseC"]


def test_enabled_registers_phase_immediately_before_settlement() -> None:
    runtime = prepare_scenario(ring_scenario([100] * 5))
    names = [phase.name for phase in runtime.phases]
    assert names == ["SubphaseB1", "SubphaseB_Clearing", "SubphaseB2", "PhaseC"]


def test_unknown_mode_raises_configuration_error() -> None:
    # "ccp" became a supported mode in Plan 061, so the unknown-mode probe
    # (bypassing schema validation via model_construct) uses a made-up one.
    config = ring_scenario([100] * 5, clearing=False)
    config = config.model_copy(update={"clearinghouse": ClearinghouseConfig.model_construct(enabled=True, mode="swaps")})
    with pytest.raises(ConfigurationError):
        build_clearing_config(config)


# -- conservation (acceptance criterion 3) ----------------------------------


def test_clearing_executed_matches_netted_events_and_preserves_positions() -> None:
    result = run_scenario(ring_scenario([100, 60, 140], cash=[40, 0, 80]))
    netted = [event for event in result.events if event["kind"] == "PayableNetted"]
    executed = [event for event in result.events if event["kind"] == "ClearingExecuted"]
    assert executed

    for summary in executed:
        day = summary["day"]
        day_netted = [event for event in netted if event["day"] == day]
        assert summary["face_extinguished"] == sum((event["netted_amount"] for event in day_netted), ZERO)
        assert summary["residual_face"] == summary["gross_face_due"] - summary["face_extinguished"]

        as_debtor: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
        as_creditor: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
        for event in day_netted:
            as_debtor[event["debtor"]] += event["netted_amount"]
            as_creditor[event["creditor"]] += event["netted_amount"]
        for agent_id in set(as_debtor) | set(as_creditor):
            assert as_debtor[agent_id] == as_creditor[agent_id]

    result.ledger.check_invariants()


# -- algorithm units (acceptance criterion 5) --------------------------------


def test_equal_face_ring_fully_nets() -> None:
    result = run_scenario(ring_scenario([100] * 5))
    executed = next(event for event in result.events if event["kind"] == "ClearingExecuted")
    assert executed["gross_face_due"] == D(500)
    assert executed["face_extinguished"] == D(500)
    assert executed["residual_face"] == ZERO
    assert executed["n_fully_netted"] == 5
    assert all(payable.settled for payable in result.ledger.payables)


def test_heterogeneous_residual_is_acyclic_and_net_positions_preserved() -> None:
    weights = {("A", "B"): D(5), ("B", "C"): D(3), ("C", "A"): D(7), ("A", "C"): D(2)}

    def net_positions(edge_weights: dict[tuple[str, str], Decimal]) -> dict[str, Decimal]:
        positions: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
        for (debtor, creditor), weight in edge_weights.items():
            positions[debtor] += weight
            positions[creditor] -= weight
        return {"A": positions["A"], "B": positions["B"], "C": positions["C"]}

    before = net_positions(weights)
    bilateral = cancel_bilateral(weights)
    cyclic = cancel_cycles(weights)

    assert bilateral == {("A", "C"): D(2), ("C", "A"): D(2)}
    assert cyclic == {("A", "B"): D(3), ("B", "C"): D(3), ("C", "A"): D(3)}
    assert find_cycle({edge for edge, weight in weights.items() if weight > ZERO}) is None
    assert net_positions(weights) == before
    assert all(weight >= ZERO for weight in weights.values())


def test_pro_rata_largest_remainder_conserves_edge_total() -> None:
    payables = [make_payable("PAY_0", "A", "B", 3), make_payable("PAY_1", "A", "B", 7)]
    allocations = allocate_single_edge(payables, D(5))
    assert allocations == [(payables[0], D(2)), (payables[1], D(3))]
    assert sum((reduction for _, reduction in allocations), ZERO) == D(5)


def test_pro_rata_ties_break_by_payable_id() -> None:
    payables = [make_payable(f"PAY_{i}", "A", "B", 1) for i in range(3)]
    allocations = allocate_single_edge(payables, D(2))
    assert allocations == [(payables[0], D(1)), (payables[1], D(1))]


def test_full_edge_reduction_allocates_full_faces() -> None:
    payables = [make_payable("PAY_0", "A", "B", 3), make_payable("PAY_1", "A", "B", 7)]
    allocations = allocate_single_edge(payables, D(10))
    assert allocations == [(payables[0], D(3)), (payables[1], D(7))]


def test_deterministic_netted_sequence_across_runs() -> None:
    config = ring_scenario([100, 60, 140, 90, 110], cash=[10, 0, 5, 0, 0])
    first = [event for event in run_scenario(config).events if event["kind"] == "PayableNetted"]
    second = [event for event in run_scenario(config).events if event["kind"] == "PayableNetted"]
    assert first == second
    assert first


def test_collect_due_payables_skips_settled_other_days_and_defaulted() -> None:
    ledger = Ledger()
    ledger.day = 1
    ledger.defaulted_agent_ids.add("F9")
    due = make_payable("PAY_0", "F1", "F2", 10)
    settled = make_payable("PAY_1", "F2", "F3", 10)
    settled.settled = True
    later = make_payable("PAY_2", "F3", "F1", 10, due_day=2)
    defaulted_debtor = make_payable("PAY_3", "F9", "F1", 10)
    defaulted_creditor = make_payable("PAY_4", "F1", "F9", 10)
    ledger.payables.extend([due, settled, later, defaulted_debtor, defaulted_creditor])
    assert collect_due_payables(ledger) == [due]
