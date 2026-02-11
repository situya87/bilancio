"""Tests that no entity ever holds negative cash during dealer trading.

These tests verify the hard invariant: no agent should ever hold negative
amounts of any asset. This covers three root causes that were fixed:

1. Buy-trade reversal bug: reversing a rejected buy subtracted price*face
   instead of just price (unit price)
2. Missing pre-trade solvency checks: trades could execute when the payer
   couldn't afford the scaled price (bid * face)
3. Sync burn overflow: sync_dealer_to_system could burn more cash than
   an agent actually held
"""

import pytest
from decimal import Decimal

from bilancio.config.models import ScenarioConfig
from bilancio.config.apply import apply_to_system
from bilancio.engines.system import System
from bilancio.engines.simulation import run_day
from bilancio.engines.dealer_integration import (
    _get_agent_cash,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_scenario_config(
    *,
    n_traders: int = 10,
    face_value: str = "20",
    outside_mid_ratio: str = "0.75",
    kappa: str = "0.5",
    mode: str = "active",
):
    """Build a ScenarioConfig dict that triggers dealer trading.

    Creates n_traders in a ring, each owing `face_value` to the next,
    with VBT and Dealer agents holding claims on some traders.
    """
    agents = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
    ]
    for i in range(1, n_traders + 1):
        agents.append({"id": f"H{i}", "kind": "household", "name": f"H{i}"})

    # Add VBT and Dealer per bucket
    for bucket in ("short", "mid", "long"):
        agents.append({"id": f"vbt_{bucket}", "kind": "household", "name": f"VBT ({bucket})"})
        agents.append({"id": f"dealer_{bucket}", "kind": "household", "name": f"Dealer ({bucket})"})

    initial_actions = []

    # Mint cash to traders based on kappa
    fv = Decimal(face_value)
    k = Decimal(kappa)
    cash_per_trader = round(fv * k)
    for i in range(1, n_traders + 1):
        initial_actions.append({"mint_cash": {"to": f"H{i}", "amount": cash_per_trader}})

    # Create ring payables: H1->H2, H2->H3, ..., Hn->H1
    # Stagger due days across buckets: short(2), mid(5), long(10)
    due_days_cycle = [2, 5, 10]
    for i in range(1, n_traders + 1):
        to_agent = f"H{i % n_traders + 1}"
        due_day = due_days_cycle[(i - 1) % 3]
        initial_actions.append({
            "create_payable": {
                "from": f"H{i}",
                "to": to_agent,
                "amount": fv,
                "due_day": due_day,
                "maturity_distance": due_day,
            }
        })

    # Create payables from some traders to VBT/Dealer (simulating balanced setup)
    bucket_map = {2: "short", 5: "mid", 10: "long"}
    for i in range(1, min(n_traders + 1, 7)):  # First 6 traders
        due_day = due_days_cycle[(i - 1) % 3]
        bucket = bucket_map[due_day]
        # Payable to VBT
        initial_actions.append({
            "create_payable": {
                "from": f"H{i}",
                "to": f"vbt_{bucket}",
                "amount": round(fv * Decimal("0.25")),
                "due_day": due_day,
                "maturity_distance": due_day,
            }
        })
        # Payable to Dealer
        initial_actions.append({
            "create_payable": {
                "from": f"H{i}",
                "to": f"dealer_{bucket}",
                "amount": round(fv * Decimal("0.125")),
                "due_day": due_day,
                "maturity_distance": due_day,
            }
        })

    # Mint cash to VBT and Dealer (market value of holdings)
    omr = Decimal(outside_mid_ratio)
    for bucket in ("short", "mid", "long"):
        vbt_cash = round(fv * Decimal("0.25") * 2 * omr)  # ~2 claims per bucket
        dealer_cash = round(fv * Decimal("0.125") * 2 * omr)
        initial_actions.append({"mint_cash": {"to": f"vbt_{bucket}", "amount": max(1, vbt_cash)}})
        initial_actions.append({"mint_cash": {"to": f"dealer_{bucket}", "amount": max(1, dealer_cash)}})

    return {
        "name": "negative-cash-test",
        "agents": agents,
        "initial_actions": initial_actions,
        "run": {
            "mode": "until_stable",
            "max_days": 30,
            "quiet_days": 2,
            "rollover_enabled": True,
        },
        "dealer": {
            "enabled": True,
            "ticket_size": 1,
            "dealer_share": "0.25",
            "vbt_share": "0.50",
        },
        "balanced_dealer": {
            "enabled": True,
            "face_value": face_value,
            "outside_mid_ratio": outside_mid_ratio,
            "mode": mode,
            "rollover_enabled": True,
        },
    }


def _assert_no_negative_system_cash(system: System, context: str = ""):
    """Assert that no contract in the system has a negative amount."""
    for contract_id, contract in system.state.contracts.items():
        assert contract.amount >= 0, (
            f"Negative contract amount: {contract.kind} {contract_id} "
            f"amount={contract.amount}, holder={contract.asset_holder_id} "
            f"[{context}]"
        )


def _assert_no_negative_dealer_cash(system: System, context: str = ""):
    """Assert that no dealer/VBT entity has negative cash in the dealer subsystem."""
    subsystem = getattr(system.state, "dealer_subsystem", None)
    if subsystem is None:
        return

    for bucket_id, dealer in subsystem.dealers.items():
        assert dealer.cash >= 0, (
            f"Negative dealer cash: dealer_{bucket_id} cash={dealer.cash} [{context}]"
        )

    for bucket_id, vbt in subsystem.vbts.items():
        assert vbt.cash >= 0, (
            f"Negative VBT cash: vbt_{bucket_id} cash={vbt.cash} [{context}]"
        )

    for trader_id, trader in subsystem.traders.items():
        assert trader.cash >= 0, (
            f"Negative trader cash in subsystem: {trader_id} cash={trader.cash} [{context}]"
        )


def _assert_no_negative_agent_cash(system: System, context: str = ""):
    """Assert that no agent has negative total cash via _get_agent_cash."""
    for agent_id in system.state.agents:
        cash = _get_agent_cash(system, agent_id)
        assert cash >= 0, (
            f"Negative agent cash: {agent_id} cash={cash} [{context}]"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoNegativeCashDuringTrading:
    """Verify that dealer trading never produces negative cash positions."""

    def test_active_trading_no_negative_cash(self):
        """Run active dealer trading for multiple days, check invariants each day."""
        data = _balanced_scenario_config(n_traders=10, kappa="1.0", mode="active")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(15):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            _assert_no_negative_system_cash(system, ctx)
            _assert_no_negative_agent_cash(system, ctx)
            _assert_no_negative_dealer_cash(system, ctx)

    def test_stressed_system_no_negative_cash(self):
        """Low kappa (stressed) system - more likely to trigger edge cases."""
        data = _balanced_scenario_config(n_traders=10, kappa="0.3", mode="active")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(15):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            _assert_no_negative_system_cash(system, ctx)
            _assert_no_negative_agent_cash(system, ctx)
            _assert_no_negative_dealer_cash(system, ctx)

    def test_high_face_value_no_negative_cash(self):
        """Large face values amplify the price*face scaling bug."""
        data = _balanced_scenario_config(
            n_traders=8, face_value="100", kappa="0.5", mode="active"
        )
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(15):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            _assert_no_negative_system_cash(system, ctx)
            _assert_no_negative_agent_cash(system, ctx)
            _assert_no_negative_dealer_cash(system, ctx)

    def test_low_outside_mid_no_negative_cash(self):
        """Low outside_mid_ratio means wider VBT spreads and more pressure."""
        data = _balanced_scenario_config(
            n_traders=8, outside_mid_ratio="0.5", kappa="0.5", mode="active"
        )
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(15):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            _assert_no_negative_system_cash(system, ctx)
            _assert_no_negative_agent_cash(system, ctx)
            _assert_no_negative_dealer_cash(system, ctx)


class TestSyncDoesNotCreateNegatives:
    """Verify that sync_dealer_to_system caps burns at available cash."""

    def test_sync_caps_burn_at_available(self):
        """If trader.cash in subsystem < 0 vs main system, sync must not go negative."""
        data = _balanced_scenario_config(n_traders=6, kappa="1.0", mode="active")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        if subsystem is None:
            pytest.skip("No dealer subsystem attached")

        # Run a few trading phases to build up trades
        for day in range(5):
            run_day(system, enable_dealer=True)

        # After sync, no agent should have negative cash
        _assert_no_negative_system_cash(system, "after-sync")
        _assert_no_negative_agent_cash(system, "after-sync")


class TestBuyTradeReversalCorrectness:
    """Verify that buy-trade reversals undo exactly the unit price, not price*face."""

    def test_reversal_preserves_dealer_cash(self):
        """After a rejected buy trade, dealer cash should not go wildly negative."""
        data = _balanced_scenario_config(n_traders=10, kappa="1.0", mode="active")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        if subsystem is None:
            pytest.skip("No dealer subsystem attached")

        # Snapshot initial cash for all dealers
        initial_cash = {}
        for bucket_id, dealer in subsystem.dealers.items():
            initial_cash[f"dealer_{bucket_id}"] = dealer.cash
        for bucket_id, vbt in subsystem.vbts.items():
            initial_cash[f"vbt_{bucket_id}"] = vbt.cash

        # Run trading for several days
        for day in range(10):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            # Dealers/VBTs should never have cash drop catastrophically
            # (the old bug caused cash to drop by ~25x per rejected trade)
            for bucket_id, dealer in subsystem.dealers.items():
                assert dealer.cash >= Decimal("-1"), (
                    f"dealer_{bucket_id} cash={dealer.cash} is catastrophically negative [{ctx}]"
                )
            for bucket_id, vbt in subsystem.vbts.items():
                assert vbt.cash >= Decimal("-1"), (
                    f"vbt_{bucket_id} cash={vbt.cash} is catastrophically negative [{ctx}]"
                )


class TestSystemInvariantsHoldWithDealer:
    """System.assert_invariants() must pass after every day with dealer trading."""

    def test_invariants_hold_active_mode(self):
        """Active dealer mode: invariants hold every day."""
        data = _balanced_scenario_config(n_traders=8, kappa="0.8", mode="active")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(12):
            run_day(system, enable_dealer=True)
            system.assert_invariants()

    def test_passive_mode_no_negative_cash(self):
        """Passive dealer mode: no negative cash positions (baseline check)."""
        data = _balanced_scenario_config(n_traders=8, kappa="0.8", mode="passive")
        config = ScenarioConfig(**data)
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        apply_to_system(config, system)

        for day in range(12):
            ctx = f"day={system.state.day}"
            run_day(system, enable_dealer=True)

            _assert_no_negative_system_cash(system, ctx)
            _assert_no_negative_agent_cash(system, ctx)
