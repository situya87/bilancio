"""Tests for the simulator snapshot module."""

from __future__ import annotations

import pytest

from bilancio.analysis.balances import AgentBalance
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.system import System
from dashboards.simulator.lib.bridge import (
    add_agent,
    create_payable,
    mint_cash,
)
from dashboards.simulator.lib.snapshot import (
    AgentDiff,
    InstrumentDelta,
    capture_snapshot,
    compute_diff,
)


@pytest.fixture()
def system_with_agents() -> System:
    """System with CB + 2 firms, each with 100 cash."""
    system = System(policy=PolicyEngine.default())
    add_agent(system, "central_bank", "CB", "Central Bank")
    add_agent(system, "firm", "F1", "Firm Alpha")
    add_agent(system, "firm", "F2", "Firm Beta")
    mint_cash(system, "F1", 100)
    mint_cash(system, "F2", 100)
    return system


class TestCaptureSnapshot:
    def test_captures_all_agents(self, system_with_agents: System) -> None:
        snap = capture_snapshot(system_with_agents)
        assert "CB" in snap
        assert "F1" in snap
        assert "F2" in snap

    def test_captures_balance_data(self, system_with_agents: System) -> None:
        snap = capture_snapshot(system_with_agents)
        assert snap["F1"].assets_by_kind.get("cash", 0) == 100

    def test_empty_system(self) -> None:
        system = System(policy=PolicyEngine.default())
        snap = capture_snapshot(system)
        assert snap == {}


class TestComputeDiff:
    def test_no_change(self, system_with_agents: System) -> None:
        snap1 = capture_snapshot(system_with_agents)
        snap2 = capture_snapshot(system_with_agents)
        diffs = compute_diff(snap1, snap2)
        for agent_diff in diffs.values():
            assert not agent_diff.has_changes

    def test_cash_increase(self, system_with_agents: System) -> None:
        snap_before = capture_snapshot(system_with_agents)
        mint_cash(system_with_agents, "F1", 50)
        snap_after = capture_snapshot(system_with_agents)
        diffs = compute_diff(snap_before, snap_after)

        f1_diff = diffs["F1"]
        assert f1_diff.has_changes
        cash_delta = next(d for d in f1_diff.asset_deltas if d.kind == "cash")
        assert cash_delta.previous == 100
        assert cash_delta.current == 150
        assert cash_delta.delta == 50

    def test_payable_creates_diffs(self, system_with_agents: System) -> None:
        snap_before = capture_snapshot(system_with_agents)
        create_payable(system_with_agents, "F1", "F2", 30, 5)
        snap_after = capture_snapshot(system_with_agents)
        diffs = compute_diff(snap_before, snap_after)

        # F1 gets a liability
        f1_diff = diffs["F1"]
        payable_delta = next(d for d in f1_diff.liability_deltas if d.kind == "payable")
        assert payable_delta.delta == 30

        # F2 gets an asset
        f2_diff = diffs["F2"]
        payable_delta = next(d for d in f2_diff.asset_deltas if d.kind == "payable")
        assert payable_delta.delta == 30

    def test_new_agent_in_after(self) -> None:
        """Agent appearing in 'after' but not 'before' is handled."""
        before: dict[str, AgentBalance] = {}
        after: dict[str, AgentBalance] = {
            "NEW": AgentBalance(
                agent_id="NEW",
                assets_by_kind={"cash": 100},
                liabilities_by_kind={},
                total_financial_assets=100,
                total_financial_liabilities=0,
                net_financial=100,
                nonfinancial_assets_by_kind={},
                total_nonfinancial_value=0,
                nonfinancial_liabilities_by_kind={},
                total_nonfinancial_liability_value=0,
                inventory_by_sku={},
                total_inventory_value=0,
            ),
        }
        diffs = compute_diff(before, after)
        assert "NEW" in diffs
        assert diffs["NEW"].has_changes


class TestInstrumentDelta:
    def test_positive_delta(self) -> None:
        d = InstrumentDelta(kind="cash", previous=100, current=150, delta=50)
        assert d.delta == 50

    def test_negative_delta(self) -> None:
        d = InstrumentDelta(kind="cash", previous=100, current=70, delta=-30)
        assert d.delta == -30

    def test_zero_delta(self) -> None:
        d = InstrumentDelta(kind="cash", previous=100, current=100, delta=0)
        assert d.delta == 0


class TestAgentDiff:
    def test_has_changes_true(self) -> None:
        diff = AgentDiff(
            agent_id="X",
            asset_deltas=[InstrumentDelta("cash", 0, 10, 10)],
        )
        assert diff.has_changes

    def test_has_changes_false(self) -> None:
        diff = AgentDiff(
            agent_id="X",
            asset_deltas=[InstrumentDelta("cash", 10, 10, 0)],
            liability_deltas=[InstrumentDelta("payable", 5, 5, 0)],
        )
        assert not diff.has_changes
