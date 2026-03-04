"""Tests for the simulator bridge module."""

from __future__ import annotations

import pytest

from bilancio.analysis.balances import agent_balance
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.system import System
from dashboards.simulator.lib.bridge import (
    _find_central_bank,
    add_agent,
    create_cb_loan,
    create_payable,
    mint_cash,
    mint_reserves,
    replay_action,
)


@pytest.fixture()
def system() -> System:
    """Create a fresh system with default policy."""
    return System(policy=PolicyEngine.default())


@pytest.fixture()
def system_with_cb(system: System) -> System:
    """System with a central bank already added."""
    add_agent(system, "central_bank", "CB", "Central Bank")
    return system


@pytest.fixture()
def system_with_agents(system_with_cb: System) -> System:
    """System with CB + 2 firms + 1 bank."""
    add_agent(system_with_cb, "firm", "F1", "Firm Alpha")
    add_agent(system_with_cb, "firm", "F2", "Firm Beta")
    add_agent(system_with_cb, "bank", "B1", "Bank One")
    return system_with_cb


class TestAddAgent:
    def test_returns_action_dict(self, system: System) -> None:
        action = add_agent(system, "central_bank", "CB", "Central Bank")
        assert action["type"] == "add_agent"
        assert action["kind"] == "central_bank"
        assert action["agent_id"] == "CB"
        assert action["name"] == "Central Bank"

    def test_agent_is_registered(self, system: System) -> None:
        add_agent(system, "central_bank", "CB", "Central Bank")
        assert "CB" in system.state.agents

    def test_add_firm(self, system_with_cb: System) -> None:
        action = add_agent(system_with_cb, "firm", "F1", "Firm Alpha")
        assert action["kind"] == "firm"
        assert "F1" in system_with_cb.state.agents


class TestMintCash:
    def test_returns_action_dict(self, system_with_agents: System) -> None:
        action = mint_cash(system_with_agents, "F1", 100)
        assert action["type"] == "mint_cash"
        assert action["to_agent_id"] == "F1"
        assert action["amount"] == 100

    def test_cash_appears_in_balance(self, system_with_agents: System) -> None:
        mint_cash(system_with_agents, "F1", 100)
        bal = agent_balance(system_with_agents, "F1")
        assert bal.assets_by_kind.get("cash", 0) == 100

    def test_multiple_mints_accumulate(self, system_with_agents: System) -> None:
        mint_cash(system_with_agents, "F1", 100)
        mint_cash(system_with_agents, "F1", 50)
        bal = agent_balance(system_with_agents, "F1")
        assert bal.assets_by_kind.get("cash", 0) == 150


class TestMintReserves:
    def test_returns_action_dict(self, system_with_agents: System) -> None:
        action = mint_reserves(system_with_agents, "B1", 200)
        assert action["type"] == "mint_reserves"
        assert action["to_bank_id"] == "B1"
        assert action["amount"] == 200

    def test_reserves_appear_in_balance(self, system_with_agents: System) -> None:
        mint_reserves(system_with_agents, "B1", 200)
        bal = agent_balance(system_with_agents, "B1")
        assert bal.assets_by_kind.get("reserve_deposit", 0) == 200


class TestCreatePayable:
    def test_returns_action_dict(self, system_with_agents: System) -> None:
        action = create_payable(system_with_agents, "F1", "F2", 50, 2)
        assert action["type"] == "create_payable"
        assert action["debtor_id"] == "F1"
        assert action["creditor_id"] == "F2"
        assert action["amount"] == 50
        assert action["due_day"] == 2

    def test_payable_appears_in_balances(self, system_with_agents: System) -> None:
        create_payable(system_with_agents, "F1", "F2", 50, 2)
        # F1 has a liability
        bal_f1 = agent_balance(system_with_agents, "F1")
        assert bal_f1.liabilities_by_kind.get("payable", 0) == 50
        # F2 has an asset
        bal_f2 = agent_balance(system_with_agents, "F2")
        assert bal_f2.assets_by_kind.get("payable", 0) == 50


class TestCreateCBLoan:
    def test_returns_action_dict(self, system_with_agents: System) -> None:
        action = create_cb_loan(system_with_agents, "B1", 100, "0.05")
        assert action["type"] == "create_cb_loan"
        assert action["bank_id"] == "B1"
        assert action["amount"] == 100
        assert action["rate"] == "0.05"

    def test_loan_appears_in_balances(self, system_with_agents: System) -> None:
        create_cb_loan(system_with_agents, "B1", 100, "0.05")
        bal_b1 = agent_balance(system_with_agents, "B1")
        assert bal_b1.liabilities_by_kind.get("cb_loan", 0) == 100
        bal_cb = agent_balance(system_with_agents, "CB")
        assert bal_cb.assets_by_kind.get("cb_loan", 0) == 100


class TestReplayAction:
    def test_replay_add_agent(self, system: System) -> None:
        action = {"type": "add_agent", "kind": "central_bank", "agent_id": "CB", "name": "CB"}
        replay_action(system, action)
        assert "CB" in system.state.agents

    def test_replay_mint_cash(self, system_with_agents: System) -> None:
        action = {"type": "mint_cash", "to_agent_id": "F1", "amount": 75}
        replay_action(system_with_agents, action)
        bal = agent_balance(system_with_agents, "F1")
        assert bal.assets_by_kind.get("cash", 0) == 75

    def test_replay_create_payable(self, system_with_agents: System) -> None:
        action = {
            "type": "create_payable",
            "debtor_id": "F1",
            "creditor_id": "F2",
            "amount": 30,
            "due_day": 3,
            "instr_id": "pay-test123",
        }
        replay_action(system_with_agents, action)
        assert "pay-test123" in system_with_agents.state.contracts

    def test_replay_unknown_type_raises(self, system: System) -> None:
        with pytest.raises(ValueError, match="Unknown action type"):
            replay_action(system, {"type": "unknown_op"})


class TestFindCentralBank:
    def test_finds_cb(self, system_with_cb: System) -> None:
        assert _find_central_bank(system_with_cb) == "CB"

    def test_raises_when_no_cb(self, system: System) -> None:
        with pytest.raises(ValueError, match="No central bank"):
            _find_central_bank(system)
