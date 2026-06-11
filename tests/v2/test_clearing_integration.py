"""Integration tests for clearing + settlement + rollover + default handling
(Plan 059, acceptance criteria 4, 6, 7)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.core.errors import DefaultError
from bilancio_v2 import prepare_scenario, run_day, run_scenario
from bilancio_v2.ledger import ZERO, Ledger

from tests.v2.test_clearing import ring_scenario

D = Decimal


def simulation_cash_transfers(events: list[dict]) -> list[dict]:
    return [event for event in events if event["kind"] == "CashTransferred" and event.get("phase") == "simulation"]


def open_face(ledger: Ledger) -> Decimal:
    return sum((payable.amount for payable in ledger.payables if not payable.settled), ZERO)


# -- full netting at kappa ~ 0 (acceptance criterion 4) ----------------------


def test_full_netting_clears_balanced_ring_without_cash() -> None:
    result = run_scenario(ring_scenario([100] * 5, cash=None))
    kinds = [event["kind"] for event in result.events]
    assert "ClearingExecuted" in kinds
    assert "ObligationDefaulted" not in kinds
    assert "AgentDefaulted" not in kinds
    assert not simulation_cash_transfers(result.events)

    executed = next(event for event in result.events if event["kind"] == "ClearingExecuted")
    assert executed["face_extinguished"] == executed["gross_face_due"] == D(500)
    assert executed["residual_face"] == ZERO
    assert not result.ledger.defaulted_agent_ids


def test_identical_ring_without_clearing_defaults() -> None:
    result = run_scenario(ring_scenario([100] * 5, cash=None, clearing=False))
    kinds = [event["kind"] for event in result.events]
    assert "ObligationDefaulted" in kinds
    assert "AgentDefaulted" in kinds
    assert result.ledger.defaulted_agent_ids


# -- "net-settle, gross-roll" rollover (acceptance criterion 6) ---------------


def test_fully_netted_payables_roll_at_full_face_without_cash() -> None:
    config = ring_scenario([100] * 5, rollover=True, max_days=3)
    result = run_scenario(config)
    ledger = result.ledger

    rolled = [event for event in result.events if event["kind"] == "PayableRolledOver"]
    # Days 1 and 2 each fully net the ring and gross-roll all five payables.
    assert len(rolled) == 10
    assert all(event["amount"] == D(100) for event in rolled)
    assert all(event["cash_transfer"] is False for event in rolled)
    assert not simulation_cash_transfers(result.events)
    assert "ObligationDefaulted" not in {event["kind"] for event in result.events}

    # The open debt stock is stationary at the original gross face.
    assert open_face(ledger) == D(500)
    assert not ledger.netted_rollover_queue
    ledger.check_invariants()


def test_partially_netted_payables_roll_at_original_face_with_residual_cash_return() -> None:
    # Cycle cancellation extinguishes 60 on every leg; residuals (40, 0, 80)
    # settle in cash and the cash return-flow covers only those residuals.
    config = ring_scenario([100, 60, 140], cash=[40, 0, 80], rollover=True, max_days=2)
    result = run_scenario(config)
    ledger = result.ledger

    rolled_full = [event for event in result.events if event["kind"] == "PayableRolledOver"]
    rolled_partial = [event for event in result.events if event["kind"] == "RolloverPartial"]

    assert [(event["amount"], event["cash_transfer"]) for event in rolled_full] == [(D(60), False)]
    assert sorted((event["amount"], event["cash_transferred"]) for event in rolled_partial) == [
        (D(100), D(40)),
        (D(140), D(80)),
    ]

    # Gross face is restored in full and cash is conserved.
    assert open_face(ledger) == D(300)
    assert sum(ledger.cash.values(), ZERO) == D(120)
    assert "ObligationDefaulted" not in {event["kind"] for event in result.events}
    ledger.check_invariants()


# -- default interaction (acceptance criterion 7) -----------------------------


def test_expel_agent_defaults_on_residual_after_netting() -> None:
    config = ring_scenario([100, 60, 140], cash=None, default_handling="expel-agent")
    result = run_scenario(config)
    events = result.events

    defaulted = next(event for event in events if event["kind"] == "ObligationDefaulted")
    assert defaulted["debtor"] == "F1"
    assert defaulted["shortfall"] == D(40)  # the post-netting residual, not the gross 100
    assert defaulted["original_amount"] == D(40)

    kinds = [event["kind"] for event in events]
    assert "AgentDefaulted" in kinds
    assert "ReceivableReassigned" in kinds
    assert "F1" in result.ledger.defaulted_agent_ids

    # Netting on later days never touches the defaulted agent.
    for event in events:
        if event["kind"] == "PayableNetted" and event["day"] > 1:
            assert "F1" not in (event["debtor"], event["creditor"])
    result.ledger.check_invariants()


def test_fail_fast_raises_but_retains_prior_netting() -> None:
    config = ring_scenario([100, 60, 140], cash=None, default_handling="fail-fast")
    runtime = prepare_scenario(config)
    run_day(runtime, 0)
    with pytest.raises(DefaultError):
        run_day(runtime, 1)

    events = runtime.ledger.journal.as_dicts()
    netted = [event for event in events if event["kind"] == "PayableNetted"]
    assert len(netted) == 3
    assert sum((event["netted_amount"] for event in netted), ZERO) == D(180)
    assert "ClearingExecuted" in {event["kind"] for event in events}
