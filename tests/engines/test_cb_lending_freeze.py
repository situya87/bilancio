"""Tests for CB lending freeze mechanism (Plan 038)."""

from __future__ import annotations

import pytest

from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.system import State, System


def _make_banking_system(reserves: int = 0) -> System:
    """Create a minimal system with CB + bank for testing."""
    system = System(policy=PolicyEngine.default(), default_mode="expel-agent")
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.add_agent(cb)
    bank = Bank(id="bank_1", name="Bank 1", kind="bank")
    system.add_agent(bank)
    if reserves > 0:
        system.mint_reserves("bank_1", reserves)
    return system


class TestFreezeFlag:
    def test_freeze_flag_starts_false(self) -> None:
        state = State()
        assert state.cb_lending_frozen is False

    def test_freeze_flag_can_be_set(self) -> None:
        state = State()
        state.cb_lending_frozen = True
        assert state.cb_lending_frozen is True


class TestFreezeBlocksCBLending:
    def test_freeze_blocks_cb_lending(self) -> None:
        system = _make_banking_system()
        system.state.cb_lending_frozen = True

        with pytest.raises(ValueError, match="CB lending is frozen"):
            system.cb_lend_reserves("bank_1", 100, day=5)

    def test_freeze_logs_event(self) -> None:
        system = _make_banking_system()
        system.state.cb_lending_frozen = True

        with pytest.raises(ValueError):
            system.cb_lend_reserves("bank_1", 100, day=5)

        frozen_events = [e for e in system.state.events if e["kind"] == "CBLendingFrozen"]
        assert len(frozen_events) == 1
        assert frozen_events[0]["bank_id"] == "bank_1"
        assert frozen_events[0]["amount"] == 100
        assert frozen_events[0]["day"] == 5

    def test_lending_works_before_freeze(self) -> None:
        system = _make_banking_system()
        assert system.state.cb_lending_frozen is False

        loan_id = system.cb_lend_reserves("bank_1", 100, day=0)
        assert loan_id in system.state.contracts
        assert system.state.contracts[loan_id].kind == InstrumentKind.CB_LOAN


class TestFreezeActivatesAtCutoffDay:
    def test_freeze_activates_at_cutoff_day(self) -> None:
        """run_until_stable activates freeze at cutoff day."""
        from bilancio.engines.simulation import run_until_stable

        system = _make_banking_system()
        # Enable rollover so that the simulation runs longer (stability = no defaults)
        system.state.rollover_enabled = True

        # Add a household with a payable so the simulation has activity
        h = Agent(id="H1", kind=AgentKind.HOUSEHOLD, name="Agent 1")
        system.add_agent(h)
        system.mint_cash("H1", 1000)

        # Create payable due on day 1 so there's at least some activity
        from bilancio.domain.instruments.credit import Payable

        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="CB",
            liability_issuer_id="H1",
            due_day=1,
        )
        system.add_contract(payable)

        run_until_stable(
            system,
            max_days=10,
            quiet_days=8,  # Large quiet_days so simulation doesn't stop before cutoff
            cb_lending_cutoff_day=3,
        )

        assert system.state.cb_lending_frozen is True

        activation_events = [
            e for e in system.state.events if e["kind"] == "CBLendingFreezeActivated"
        ]
        assert len(activation_events) == 1
        assert activation_events[0]["cutoff_day"] == 3

    def test_no_freeze_when_cutoff_none(self) -> None:
        from bilancio.engines.simulation import run_until_stable

        system = _make_banking_system()
        h = Agent(id="H1", kind=AgentKind.HOUSEHOLD, name="Agent 1")
        system.add_agent(h)
        system.mint_cash("H1", 1000)

        run_until_stable(
            system,
            max_days=5,
            quiet_days=2,
            cb_lending_cutoff_day=None,
        )

        assert system.state.cb_lending_frozen is False

    def test_freeze_does_not_activate_before_cutoff(self) -> None:
        """If cutoff_day=100 and max_days=5, freeze should not activate."""
        from bilancio.engines.simulation import run_until_stable

        system = _make_banking_system()
        h = Agent(id="H1", kind=AgentKind.HOUSEHOLD, name="Agent 1")
        system.add_agent(h)
        system.mint_cash("H1", 1000)

        run_until_stable(
            system,
            max_days=5,
            quiet_days=2,
            cb_lending_cutoff_day=100,
        )

        assert system.state.cb_lending_frozen is False


class TestBankDefaultWhenFrozen:
    def test_bank_defaults_when_frozen_and_cant_repay(self) -> None:
        """Bank with due CB loan and no reserves defaults when frozen."""
        from bilancio.engines.simulation import run_day
        from bilancio.ops.primitives import consume

        system = _make_banking_system(reserves=200)

        # Create a CB loan that will be due on day 2
        loan_id = system.cb_lend_reserves("bank_1", 100, day=0)
        loan = system.state.contracts[loan_id]
        assert loan.maturity_day == 2

        # Consume all bank reserves so it can't repay
        bank = system.state.agents["bank_1"]
        reserve_ids = [
            cid for cid in list(bank.asset_ids)
            if cid in system.state.contracts
            and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        ]
        for rid in reserve_ids:
            r = system.state.contracts[rid]
            consume(system, rid, r.amount)
            system.state.cb_reserves_outstanding -= r.amount

        # Freeze CB lending
        system.state.cb_lending_frozen = True

        # Advance to day 2 when loan is due
        system.state.day = 2
        run_day(system)

        # Bank should be defaulted
        assert system.state.agents["bank_1"].defaulted is True
        assert "bank_1" in system.state.defaulted_agent_ids

        # Check event
        default_events = [
            e for e in system.state.events if e["kind"] == "BankDefaultCBFreeze"
        ]
        assert len(default_events) == 1
        assert default_events[0]["bank_id"] == "bank_1"

    def test_bank_repays_normally_before_freeze(self) -> None:
        """Bank with enough reserves repays CB loan normally."""
        from bilancio.engines.simulation import run_day

        system = _make_banking_system(reserves=500)

        # Create a CB loan due day 2
        system.cb_lend_reserves("bank_1", 100, day=0)

        # Don't freeze
        assert system.state.cb_lending_frozen is False

        # Advance to day 2
        system.state.day = 2
        run_day(system)

        # Bank should NOT be defaulted
        assert system.state.agents["bank_1"].defaulted is False

    def test_bank_refinances_normally_when_not_frozen(self) -> None:
        """Bank without enough reserves can refinance when not frozen."""
        from bilancio.engines.simulation import run_day
        from bilancio.ops.primitives import consume

        system = _make_banking_system(reserves=200)

        # Create CB loan due day 2
        system.cb_lend_reserves("bank_1", 100, day=0)

        # Consume all reserves so bank can't repay directly
        bank = system.state.agents["bank_1"]
        reserve_ids = [
            cid for cid in list(bank.asset_ids)
            if cid in system.state.contracts
            and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        ]
        for rid in reserve_ids:
            r = system.state.contracts[rid]
            consume(system, rid, r.amount)
            system.state.cb_reserves_outstanding -= r.amount

        # NOT frozen — should refinance
        assert system.state.cb_lending_frozen is False

        system.state.day = 2
        run_day(system)

        # Bank should NOT be defaulted (refinancing worked)
        assert system.state.agents["bank_1"].defaulted is False


class TestFreezeDefaultResetsStability:
    def test_bank_default_cb_freeze_counted_as_default(self) -> None:
        """BankDefaultCBFreeze is in DEFAULT_EVENTS so stability counter resets."""
        from bilancio.engines.simulation import DEFAULT_EVENTS

        assert "BankDefaultCBFreeze" in DEFAULT_EVENTS

    def test_freeze_default_resets_quiet_counter(self) -> None:
        """run_until_stable does not stop early when a bank defaults from freeze."""
        from bilancio.engines.simulation import _defaults_today, run_day
        from bilancio.ops.primitives import consume

        system = _make_banking_system(reserves=200)

        # Create CB loan due on day 2 (issuance_day=0 + 2)
        loan_id = system.cb_lend_reserves("bank_1", 100, day=0)

        # Consume all reserves so bank can't repay
        bank = system.state.agents["bank_1"]
        reserve_ids = [
            cid for cid in list(bank.asset_ids)
            if cid in system.state.contracts
            and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        ]
        for rid in reserve_ids:
            r = system.state.contracts[rid]
            consume(system, rid, r.amount)
            system.state.cb_reserves_outstanding -= r.amount

        # Freeze and run day 2
        system.state.cb_lending_frozen = True
        system.state.day = 2
        run_day(system)

        # _defaults_today should count the BankDefaultCBFreeze event
        defaults = _defaults_today(system, 2)
        assert defaults >= 1


class TestFinalSettlementStillRuns:
    def test_final_settlement_still_runs_after_freeze(self) -> None:
        """Final CB settlement catches remaining loans after freeze."""
        from bilancio.engines.simulation import run_final_cb_settlement

        system = _make_banking_system(reserves=500)

        # Create a CB loan
        system.cb_lend_reserves("bank_1", 100, day=0)

        # Freeze
        system.state.cb_lending_frozen = True

        # Run final settlement (should still work — it doesn't need CB lending)
        result = run_final_cb_settlement(system)
        assert result["loans_attempted"] >= 1
        # Bank has enough reserves, so it should repay
        assert result["loans_repaid"] >= 1
