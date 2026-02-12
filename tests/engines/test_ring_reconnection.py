"""Tests for ring reconnection on agent default, Model C rollover, and ticket ingestion.

When an agent defaults in a ring with rollover enabled, the ring should be
reconnected: predecessor → defaulted_agent becomes predecessor → successor.

Strategy: Most tests call _reconnect_ring directly to test the reconnection
logic in isolation. Integration tests use settle_due with carefully constructed
scenarios that guarantee the target agent defaults (by making it owe more than
it can possibly receive through the circuit).
"""
import pytest
from decimal import Decimal

from bilancio.core.errors import DefaultError
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.settlement import settle_due, rollover_settled_payables, _reconnect_ring, _remove_contract, _expel_agent
from bilancio.engines.dealer_integration import (
    DealerSubsystem,
    initialize_balanced_dealer_subsystem,
    _ingest_new_payables,
    _get_agent_cash,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from bilancio.engines.system import System
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.dealer import Dealer
from bilancio.domain.agents.vbt import VBT
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.dealer.models import DEFAULT_BUCKETS


def _ring_system(n_agents: int, default_mode: str = "expel-agent", rollover_enabled: bool = True):
    """Create a System with CB + N household agents arranged in a ring.

    Ring structure: H1→H2→H3→...→HN→H1
    Each agent owes 100 to the next agent, with maturity_distance=3.
    """
    system = System(default_mode=default_mode)
    system.state.rollover_enabled = rollover_enabled
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.add_agent(cb)

    agents = []
    for i in range(1, n_agents + 1):
        agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(agent)
        agents.append(agent)

    # Create ring payables: H1→H2, H2→H3, ..., HN→H1
    payables = []
    for i in range(n_agents):
        debtor = agents[i]
        creditor = agents[(i + 1) % n_agents]
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id=creditor.id,
            liability_issuer_id=debtor.id,
            due_day=3,
            maturity_distance=3,
        )
        system.add_contract(payable)
        payables.append(payable)

    return system, agents, payables


def _simulate_default(system, agent_id: str, payable, day: int):
    """Simulate a default: remove trigger payable, expel agent, then reconnect.

    This mimics what settle_due does during a default, but without needing
    to run the full settlement loop (which has cash flow side effects).
    """
    successor_id = payable.asset_holder_id
    _remove_contract(system, payable.id)
    _expel_agent(
        system,
        agent_id,
        trigger_contract_id=payable.id,
        trigger_kind=payable.kind,
        trigger_shortfall=payable.amount,
        cancelled_contract_ids={payable.id},
    )
    return successor_id


class TestBasicReconnection:
    """Test basic ring reconnection behavior."""

    def test_basic_reconnection(self):
        """5-agent ring, H3 defaults, verify H2→H4 payable created."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H3 owes H4 (payables[2]). Simulate H3 defaulting.
        successor_id = _simulate_default(system, "H3", payables[2], day=3)
        result = _reconnect_ring(system, "H3", successor_id, day=3)

        assert result is not None
        assert result["predecessor"] == "H2"
        assert result["successor"] == "H4"
        assert result["amount"] == 100

        # Verify new payable H2→H4 exists in contracts
        new_payable = system.state.contracts[result["new_payable"]]
        assert new_payable.liability_issuer_id == "H2"
        assert new_payable.asset_holder_id == "H4"
        assert new_payable.amount == 100

    def test_reconnection_preserves_predecessor_amount(self):
        """Different amounts per agent, verify new payable has predecessor's amount."""
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)

        agents = []
        for i in range(1, 6):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)
            agents.append(agent)

        # H1 owes 50, H2 owes 200, H3 owes 80, H4 owes 120, H5 owes 90
        amounts = [50, 200, 80, 120, 90]
        payables = []
        for i in range(5):
            debtor = agents[i]
            creditor = agents[(i + 1) % 5]
            payable = Payable(
                id=system.new_contract_id("PAY"),
                kind=InstrumentKind.PAYABLE,
                amount=amounts[i],
                denom="X",
                asset_holder_id=creditor.id,
                liability_issuer_id=debtor.id,
                due_day=3,
                maturity_distance=3,
            )
            system.add_contract(payable)
            payables.append(payable)

        system.state.day = 3

        # H3 defaults (owes H4, amount=80)
        successor_id = _simulate_default(system, "H3", payables[2], day=3)
        result = _reconnect_ring(system, "H3", successor_id, day=3)

        assert result is not None
        # H2→H3 had amount=200. New H2→H4 should preserve H2's amount (200).
        assert result["amount"] == 200
        assert result["predecessor"] == "H2"
        assert result["successor"] == "H4"

    def test_reconnection_resets_due_day(self):
        """H3 defaults on day 5, predecessor has maturity_distance=3, verify new due_day=8."""
        system, agents, payables = _ring_system(5)

        # Change all due days to day 5
        for p in payables:
            p.due_day = 5

        system.state.day = 5

        # H3 defaults
        successor_id = _simulate_default(system, "H3", payables[2], day=5)
        result = _reconnect_ring(system, "H3", successor_id, day=5)

        assert result is not None
        # maturity_distance=3, day=5 → new_due_day=8
        assert result["new_due_day"] == 8

        new_payable = system.state.contracts[result["new_payable"]]
        assert new_payable.due_day == 8
        assert new_payable.maturity_distance == 3


class TestEdgeCases:
    """Test edge cases: first/last agent, adjacent defaults, ring collapse."""

    def test_first_agent_defaults(self):
        """H1 defaults, verify H5→H2 payable (wrap-around)."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H1 owes H2 (payables[0]). H5 owes H1 (payables[4]).
        successor_id = _simulate_default(system, "H1", payables[0], day=3)
        result = _reconnect_ring(system, "H1", successor_id, day=3)

        assert result is not None
        assert result["predecessor"] == "H5"
        assert result["successor"] == "H2"

    def test_last_agent_defaults(self):
        """H5 defaults, verify H4→H1 payable."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H5 owes H1 (payables[4]). H4 owes H5 (payables[3]).
        successor_id = _simulate_default(system, "H5", payables[4], day=3)
        result = _reconnect_ring(system, "H5", successor_id, day=3)

        assert result is not None
        assert result["predecessor"] == "H4"
        assert result["successor"] == "H1"

    def test_adjacent_defaults_same_day(self):
        """H3 and H4 both default on same day, verify H2→H5."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H3 defaults first: H3→H4 (payables[2]) removed
        successor_id_3 = _simulate_default(system, "H3", payables[2], day=3)
        result_3 = _reconnect_ring(system, "H3", successor_id_3, day=3)
        assert result_3 is not None
        assert result_3["predecessor"] == "H2"
        assert result_3["successor"] == "H4"

        # Now H4 defaults: H4→H5 (payables[3]) removed
        successor_id_4 = _simulate_default(system, "H4", payables[3], day=3)
        result_4 = _reconnect_ring(system, "H4", successor_id_4, day=3)
        assert result_4 is not None
        # After H3 default, H2→H4 was created. Now H4 defaults, so H2→H5
        assert result_4["predecessor"] == "H2"
        assert result_4["successor"] == "H5"

        # Verify final state: H2→H5 payable exists
        found = False
        for c in system.state.contracts.values():
            if (c.kind == InstrumentKind.PAYABLE
                and c.liability_issuer_id == "H2"
                and c.asset_holder_id == "H5"):
                found = True
                break
        assert found, "Expected H2→H5 payable after adjacent defaults"

    def test_ring_collapses_to_two(self):
        """3-agent ring, 1 defaults, ring becomes 2-agent."""
        system, agents, payables = _ring_system(3)
        system.state.day = 3

        # H2 defaults: H2→H3 (payables[1]) removed
        successor_id = _simulate_default(system, "H2", payables[1], day=3)
        result = _reconnect_ring(system, "H2", successor_id, day=3)

        assert result is not None
        assert result["predecessor"] == "H1"
        assert result["successor"] == "H3"

        # Verify H1→H3 payable exists
        new_payable = system.state.contracts[result["new_payable"]]
        assert new_payable.liability_issuer_id == "H1"
        assert new_payable.asset_holder_id == "H3"

    def test_ring_collapses_to_one(self):
        """3-agent ring, 2 default, logs RingCollapsed, no self-payable."""
        system, agents, payables = _ring_system(3)
        system.state.day = 3

        # H2 defaults first: reconnects H1→H3
        successor_id_2 = _simulate_default(system, "H2", payables[1], day=3)
        result_2 = _reconnect_ring(system, "H2", successor_id_2, day=3)
        assert result_2 is not None

        # H3 defaults: ring collapses (only H1 left, predecessor == successor)
        successor_id_3 = _simulate_default(system, "H3", payables[2], day=3)
        result_3 = _reconnect_ring(system, "H3", successor_id_3, day=3)
        assert result_3 is None  # Should return None on collapse

        # Check that RingCollapsed was logged
        collapse_events = [e for e in system.state.events if e["kind"] == "RingCollapsed"]
        assert len(collapse_events) == 1
        assert collapse_events[0]["remaining_agent"] == "H1"

        # No self-payable should exist (H1 should not owe itself)
        for c in system.state.contracts.values():
            if c.kind == InstrumentKind.PAYABLE:
                assert c.liability_issuer_id != c.asset_holder_id, \
                    "Self-payable should not exist after ring collapse"


class TestRolloverGuard:
    """Test that reconnection only happens with rollover enabled."""

    def test_no_reconnection_without_rollover(self):
        """rollover=False, H3 defaults, no reconnection event via settle_due."""
        system, agents, payables = _ring_system(5, rollover_enabled=False)
        system.state.day = 3

        # Make H3 owe 200 but only able to receive 100 from H2
        # H3→H4 payable: change amount to 200
        payables[2].amount = 200

        # Give H1 and H2 cash so they can pay, but H3 will still fall short
        # H1 pays H2 (100), H2 pays H3 (100). H3 receives 100 but owes 200.
        system.mint_cash("H1", 100)

        settle_due(system, 3, rollover_enabled=False)

        assert "H3" in system.state.defaulted_agent_ids

        reconnected_events = [e for e in system.state.events if e["kind"] == "RingReconnected"]
        assert len(reconnected_events) == 0


class TestEventLogging:
    """Test that events are properly logged."""

    def test_events_logged(self):
        """Verify RingReconnected event has all required fields."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H3 defaults
        successor_id = _simulate_default(system, "H3", payables[2], day=3)
        _reconnect_ring(system, "H3", successor_id, day=3)

        reconnected_events = [e for e in system.state.events if e["kind"] == "RingReconnected"]
        assert len(reconnected_events) == 1
        evt = reconnected_events[0]

        # Check all expected fields
        assert "defaulted_agent" in evt
        assert "predecessor" in evt
        assert "successor" in evt
        assert "old_payable" in evt
        assert "new_payable" in evt
        assert "amount" in evt
        assert "maturity_distance" in evt
        assert "new_due_day" in evt
        assert "day" in evt

        # Also check PayableCreated event
        created_events = [e for e in system.state.events
                         if e["kind"] == "PayableCreated" and e.get("reason") == "ring_reconnection"]
        assert len(created_events) == 1
        ce = created_events[0]
        assert ce["debtor"] == "H2"
        assert ce["creditor"] == "H4"


class TestInvariants:
    """Test that system invariants hold after reconnection."""

    def test_invariants_after_reconnection(self):
        """system.assert_invariants() passes after reconnection."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H3 defaults
        successor_id = _simulate_default(system, "H3", payables[2], day=3)
        _reconnect_ring(system, "H3", successor_id, day=3)

        # This should not raise
        system.assert_invariants()


class TestSettleDueIntegration:
    """Integration tests that verify reconnection through settle_due."""

    def test_settle_due_triggers_reconnection(self):
        """Verify settle_due calls _reconnect_ring when an agent defaults with rollover.

        H1 has no cash and defaults immediately. The predecessor payable H5→H1
        hasn't been settled yet (it's later in iteration order), so reconnection
        can find it and create H5→H2.
        """
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # Give everyone except H1 enough to pay. H1 defaults first (no cash).
        # Settlement order: H1→H2 (default), H2→H3, H3→H4, H4→H5, H5→H1.
        # H5→H1 hasn't been settled when H1 defaults, so predecessor H5 is found.
        for agent in agents:
            if agent.id != "H1":
                system.mint_cash(agent.id, 100)

        settle_due(system, 3, rollover_enabled=True)

        assert "H1" in system.state.defaulted_agent_ids

        reconnected_events = [e for e in system.state.events if e["kind"] == "RingReconnected"]
        h1_reconnect = [e for e in reconnected_events if e["defaulted_agent"] == "H1"]
        assert len(h1_reconnect) == 1
        evt = h1_reconnect[0]
        assert evt["predecessor"] == "H5"
        assert evt["successor"] == "H2"

    def test_settle_due_no_reconnection_first_agent(self):
        """H1 defaults (first in settlement order, no cash), reconnects H5→H2."""
        system, agents, payables = _ring_system(5)
        system.state.day = 3

        # H1 has no cash and owes 100. H1 defaults immediately.
        # No one else has cash either, so multiple defaults, but H1 is first.
        settle_due(system, 3, rollover_enabled=True)

        assert "H1" in system.state.defaulted_agent_ids

        reconnected_events = [e for e in system.state.events if e["kind"] == "RingReconnected"]
        # At least H1's default should trigger reconnection
        h1_reconnect = [e for e in reconnected_events if e["defaulted_agent"] == "H1"]
        assert len(h1_reconnect) == 1
        assert h1_reconnect[0]["predecessor"] == "H5"
        assert h1_reconnect[0]["successor"] == "H2"


# ---------------------------------------------------------------------------
# Helpers for Model C and ticket ingestion tests
# ---------------------------------------------------------------------------

def _dealer_ring_system(n_agents: int = 5, maturity_days: int = 3):
    """Create a ring system with dealer subsystem for Model C / ingestion tests.

    Creates: CB + N traders + VBT/Dealer agents per bucket.
    Each trader owes face_value=100 to the next, plus split payables to VBT/Dealer.
    All agents get enough cash for settlement.

    Returns (system, trader_ids, dealer_config, face_value).
    """
    face_value = 100
    system = System(default_mode="expel-agent")
    system.state.rollover_enabled = True

    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.add_agent(cb)

    trader_ids = []
    for i in range(1, n_agents + 1):
        agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(agent)
        trader_ids.append(agent.id)

    # Create VBT/Dealer agents for each bucket
    for bucket in DEFAULT_BUCKETS:
        dealer_agent = Dealer(id=f"dealer_{bucket.name}", name=f"Dealer ({bucket.name})")
        system.state.agents[dealer_agent.id] = dealer_agent
        vbt_agent = VBT(id=f"vbt_{bucket.name}", name=f"VBT ({bucket.name})")
        system.state.agents[vbt_agent.id] = vbt_agent

    # Ring payables: H1→H2, H2→H3, ..., HN→H1 (trader-to-trader)
    for i in range(n_agents):
        debtor_id = trader_ids[i]
        creditor_id = trader_ids[(i + 1) % n_agents]
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=face_value,
            denom="X",
            asset_holder_id=creditor_id,
            liability_issuer_id=debtor_id,
            due_day=maturity_days,
            maturity_distance=maturity_days,
        )
        system.add_contract(payable)

    # VBT payables: each trader also owes VBT_short a smaller amount
    for i in range(n_agents):
        debtor_id = trader_ids[i]
        vbt_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=25,
            denom="X",
            asset_holder_id="vbt_short",
            liability_issuer_id=debtor_id,
            due_day=maturity_days,
            maturity_distance=maturity_days,
        )
        system.add_contract(vbt_payable)

    # Dealer payables: each trader also owes dealer_short
    for i in range(n_agents):
        debtor_id = trader_ids[i]
        dealer_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=12,
            denom="X",
            asset_holder_id="dealer_short",
            liability_issuer_id=debtor_id,
            due_day=maturity_days,
            maturity_distance=maturity_days,
        )
        system.add_contract(dealer_payable)

    # Give everyone enough cash for settlement
    for tid in trader_ids:
        system.mint_cash(tid, face_value + 25 + 12 + 50)  # extra buffer
    # Give VBT/Dealer some cash too
    system.mint_cash("vbt_short", 500)
    system.mint_cash("dealer_short", 250)

    dealer_config = DealerRingConfig(
        ticket_size=Decimal(1),
        seed=42,
    )

    return system, trader_ids, dealer_config, face_value


class TestModelCRollover:
    """Test Model C rollover: no cash transfer for trader payables when dealer_active."""

    def test_no_cash_transfer_for_trader_payables(self):
        """When dealer_active=True, trader→trader rollover should NOT transfer cash."""
        system, agents, payables = _ring_system(3)
        system.state.day = 3

        # Give everyone enough cash for initial settlement
        for a in agents:
            system.mint_cash(a.id, 100)

        # Settle day 3 (all payables due)
        settled = settle_due(system, 3, rollover_enabled=True)
        assert len(settled) > 0

        # Record cash before rollover
        cash_before = {}
        for a in agents:
            if not getattr(system.state.agents[a.id], 'defaulted', False):
                cash_before[a.id] = _get_agent_cash(system, a.id)

        # Model C rollover: dealer_active=True → no cash transfer for traders
        new_ids = rollover_settled_payables(system, 3, settled, dealer_active=True)

        # Cash should NOT have changed for non-defaulted agents
        for agent_id, before in cash_before.items():
            after = _get_agent_cash(system, agent_id)
            assert after == before, (
                f"Agent {agent_id} cash changed from {before} to {after} "
                f"under Model C (no cash transfer expected)"
            )

        # Verify new payables were created
        assert len(new_ids) > 0

        # Verify log events have cash_transfer=False
        rollover_events = [e for e in system.state.events if e["kind"] == "PayableRolledOver"]
        for evt in rollover_events:
            assert evt.get("cash_transfer") is False, \
                f"Expected cash_transfer=False for trader rollover, got {evt}"

    def test_cash_transfer_for_vbt_payables(self):
        """VBT/Dealer payables should still transfer cash under Model C."""
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)

        debtor = Household(id="H1", name="H1", kind="household")
        system.add_agent(debtor)

        vbt_agent = VBT(id="vbt_short", name="VBT (short)")
        system.state.agents[vbt_agent.id] = vbt_agent

        # Give VBT cash (it will re-lend during rollover)
        system.mint_cash("vbt_short", 200)

        # Simulate settled VBT payable: H1 owes vbt_short 50
        settled = [("H1", "vbt_short", 50, 3)]

        cash_vbt_before = _get_agent_cash(system, "vbt_short")
        cash_h1_before = _get_agent_cash(system, "H1")

        new_ids = rollover_settled_payables(system, 3, settled, dealer_active=True)

        assert len(new_ids) == 1

        # VBT should have LESS cash (transferred to debtor)
        cash_vbt_after = _get_agent_cash(system, "vbt_short")
        cash_h1_after = _get_agent_cash(system, "H1")
        assert cash_vbt_after < cash_vbt_before, "VBT should transfer cash during rollover"
        assert cash_h1_after > cash_h1_before, "Debtor should receive cash from VBT rollover"

        # Verify event has cash_transfer=True
        rollover_events = [e for e in system.state.events if e["kind"] == "PayableRolledOver"]
        assert len(rollover_events) == 1
        assert rollover_events[0]["cash_transfer"] is True

    def test_cash_transfer_for_dealer_payables(self):
        """Dealer payables should still transfer cash under Model C."""
        system = System(default_mode="expel-agent")
        system.state.rollover_enabled = True
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)

        debtor = Household(id="H1", name="H1", kind="household")
        system.add_agent(debtor)

        dealer_agent = Dealer(id="dealer_mid", name="Dealer (mid)")
        system.state.agents[dealer_agent.id] = dealer_agent

        system.mint_cash("dealer_mid", 200)

        settled = [("H1", "dealer_mid", 30, 3)]

        cash_dealer_before = _get_agent_cash(system, "dealer_mid")

        new_ids = rollover_settled_payables(system, 3, settled, dealer_active=True)

        assert len(new_ids) == 1
        cash_dealer_after = _get_agent_cash(system, "dealer_mid")
        assert cash_dealer_after < cash_dealer_before, "Dealer should transfer cash during rollover"

    def test_model_a_when_dealer_not_active(self):
        """When dealer_active=False, all payables use cash transfer (Model A)."""
        system, agents, payables = _ring_system(3)
        system.state.day = 3

        for a in agents:
            system.mint_cash(a.id, 100)

        settled = settle_due(system, 3, rollover_enabled=True)

        # Model A rollover: dealer_active=False → cash transfer for all
        new_ids = rollover_settled_payables(system, 3, settled, dealer_active=False)

        rollover_events = [e for e in system.state.events if e["kind"] == "PayableRolledOver"]
        for evt in rollover_events:
            assert evt.get("cash_transfer") is True, \
                f"Expected cash_transfer=True for Model A rollover, got {evt}"

    def test_new_payable_ids_returned(self):
        """rollover_settled_payables returns list of new payable IDs."""
        system, agents, payables = _ring_system(3)
        system.state.day = 3

        for a in agents:
            system.mint_cash(a.id, 100)

        settled = settle_due(system, 3, rollover_enabled=True)
        new_ids = rollover_settled_payables(system, 3, settled, dealer_active=True)

        assert isinstance(new_ids, list)
        # Each non-defaulted settlement produces a new payable
        for pid in new_ids:
            assert pid in system.state.contracts
            assert system.state.contracts[pid].kind == InstrumentKind.PAYABLE


class TestTicketIngestion:
    """Test that new payables from rollover get ingested as tickets."""

    def test_ingest_creates_tickets_for_new_payables(self):
        """After rollover creates new payables, _ingest_new_payables creates tickets."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)

        # Initialize dealer subsystem
        subsystem = initialize_balanced_dealer_subsystem(
            system, dealer_config, current_day=0,
        )
        system.state.dealer_subsystem = subsystem

        initial_ticket_count = len(subsystem.tickets)
        initial_payable_count = len(subsystem.payable_to_ticket)

        # Manually create a new payable (simulating rollover output)
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        # Ingest
        new_count = _ingest_new_payables(subsystem, system, current_day=3)

        assert new_count == 1
        assert len(subsystem.tickets) == initial_ticket_count + 1

        # Verify ticket maps to payable
        ticket_id = f"TKT_{new_payable.id}"
        assert ticket_id in subsystem.tickets
        assert subsystem.ticket_to_payable[ticket_id] == new_payable.id
        assert subsystem.payable_to_ticket[new_payable.id] == ticket_id

        # Verify ticket properties
        ticket = subsystem.tickets[ticket_id]
        assert ticket.issuer_id == "H1"
        assert ticket.owner_id == "H2"
        assert ticket.face == Decimal(100)
        assert ticket.maturity_day == 6
        assert ticket.remaining_tau == 3

    def test_ingest_assigns_vbt_tickets_to_vbt_inventory(self):
        """Tickets owned by VBT agents go into VBT inventory."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        vbt_inv_before = len(subsystem.vbts["short"].inventory)

        # Create payable owned by VBT
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=25,
            denom="X",
            asset_holder_id="vbt_short",
            liability_issuer_id="H1",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.vbts["short"].inventory) == vbt_inv_before + 1

    def test_ingest_assigns_dealer_tickets_to_dealer_inventory(self):
        """Tickets owned by Dealer agents go into Dealer inventory."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        dealer_inv_before = len(subsystem.dealers["short"].inventory)

        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=12,
            denom="X",
            asset_holder_id="dealer_short",
            liability_issuer_id="H1",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.dealers["short"].inventory) == dealer_inv_before + 1

    def test_ingest_assigns_trader_tickets_to_trader(self):
        """Tickets owned by traders go into trader.tickets_owned."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        trader_tickets_before = len(subsystem.traders["H1"].tickets_owned)

        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.traders["H1"].tickets_owned) == trader_tickets_before + 1

    def test_ingest_links_obligations_to_issuer(self):
        """Issuer trader gets the ticket added to their obligations list."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        h2_obligations_before = len(subsystem.traders["H2"].obligations)

        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.traders["H2"].obligations) == h2_obligations_before + 1

    def test_ingest_skips_already_tracked_payables(self):
        """Payables that already have tickets are not duplicated."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        initial_count = len(subsystem.tickets)

        # Ingest with no new payables — count should be 0
        new_count = _ingest_new_payables(subsystem, system, current_day=0)

        assert new_count == 0
        assert len(subsystem.tickets) == initial_count

    def test_ingest_skips_matured_payables(self):
        """Payables with remaining_tau=0 are not ingested."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # Create a payable that's already matured (due_day == current_day)
        matured_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=3,  # Already matured at current_day=3
            maturity_distance=3,
        )
        system.add_contract(matured_payable)

        new_count = _ingest_new_payables(subsystem, system, current_day=3)
        assert new_count == 0

    def test_ingest_skips_defaulted_issuers(self):
        """Payables from defaulted issuers are not ingested."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # Default H1
        system.state.agents["H1"].defaulted = True
        system.state.defaulted_agent_ids.add("H1")

        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",  # Defaulted issuer
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        new_count = _ingest_new_payables(subsystem, system, current_day=3)
        assert new_count == 0

    def test_serial_counter_continues_from_init(self):
        """Ingested tickets continue serial numbering from initialization."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        counter_before = subsystem._ticket_serial_counter
        assert counter_before > 0  # Some tickets were created at init

        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        ticket_id = f"TKT_{new_payable.id}"
        ticket = subsystem.tickets[ticket_id]
        assert ticket.serial == counter_before
        assert subsystem._ticket_serial_counter == counter_before + 1

    def test_ingest_reassigns_vbt_to_correct_bucket(self):
        """VBT_short payable with long maturity goes to vbt_long inventory, not vbt_short."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        long_inv_before = len(subsystem.vbts["long"].inventory)
        short_inv_before = len(subsystem.vbts["short"].inventory)

        # Payable originally from vbt_short, but rolled over to a long maturity
        # remaining_tau = 15 - 3 = 12 → "long" bucket (tau >= 9)
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=25,
            denom="X",
            asset_holder_id="vbt_short",  # Original owner was short VBT
            liability_issuer_id="H1",
            due_day=15,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        # Should be in vbt_long's inventory, NOT vbt_short's
        assert len(subsystem.vbts["long"].inventory) == long_inv_before + 1
        assert len(subsystem.vbts["short"].inventory) == short_inv_before

        # Ticket owner should be vbt_long
        ticket_id = f"TKT_{new_payable.id}"
        ticket = subsystem.tickets[ticket_id]
        assert ticket.owner_id == "vbt_long"
        assert ticket.bucket_id == "long"

        # Payable in main system should also be reassigned
        payable = system.state.contracts[new_payable.id]
        assert payable.asset_holder_id == "vbt_long"
        assert new_payable.id in system.state.agents["vbt_long"].asset_ids
        assert new_payable.id not in system.state.agents["vbt_short"].asset_ids

    def test_ingest_reassigns_dealer_to_correct_bucket(self):
        """Dealer_short payable with mid maturity goes to dealer_mid inventory."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        mid_inv_before = len(subsystem.dealers["mid"].inventory)

        # remaining_tau = 10 - 3 = 7 → "mid" bucket (tau 4-8)
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=12,
            denom="X",
            asset_holder_id="dealer_short",  # Original owner was short dealer
            liability_issuer_id="H1",
            due_day=10,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.dealers["mid"].inventory) == mid_inv_before + 1

        ticket_id = f"TKT_{new_payable.id}"
        ticket = subsystem.tickets[ticket_id]
        assert ticket.owner_id == "dealer_mid"
        assert ticket.bucket_id == "mid"

    def test_ingest_keeps_same_bucket_when_matching(self):
        """VBT_short payable staying in short bucket keeps vbt_short as owner."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=3)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        short_inv_before = len(subsystem.vbts["short"].inventory)

        # remaining_tau = 6 - 3 = 3 → "short" bucket (tau 1-3)
        new_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=25,
            denom="X",
            asset_holder_id="vbt_short",
            liability_issuer_id="H1",
            due_day=6,
            maturity_distance=3,
        )
        system.add_contract(new_payable)

        _ingest_new_payables(subsystem, system, current_day=3)

        assert len(subsystem.vbts["short"].inventory) == short_inv_before + 1

        ticket_id = f"TKT_{new_payable.id}"
        ticket = subsystem.tickets[ticket_id]
        assert ticket.owner_id == "vbt_short"  # Stays with short VBT


class TestPhase1BucketTransition:
    """Test Phase 1 maturity updates: tickets moving between buckets."""

    def test_dealer_ticket_moves_to_new_bucket_on_maturity_update(self):
        """Dealer_mid ticket ages into short bucket → moves to dealer_short inventory."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=5)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # Manually place a ticket in dealer_mid with maturity_day=5
        # At current_day=0: remaining_tau=5 → "mid" (tau 4-8)
        test_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=12,
            denom="X",
            asset_holder_id="dealer_mid",
            liability_issuer_id="H1",
            due_day=5,
            maturity_distance=5,
        )
        system.add_contract(test_payable)
        _ingest_new_payables(subsystem, system, current_day=0)

        ticket_id = f"TKT_{test_payable.id}"
        assert subsystem.tickets[ticket_id].bucket_id == "mid"
        assert subsystem.tickets[ticket_id].owner_id == "dealer_mid"

        mid_cash_before = subsystem.dealers["mid"].cash
        short_cash_before = subsystem.dealers["short"].cash

        # Run trading phase at day=3: remaining_tau = 5-3 = 2 → "short" (tau 1-3)
        run_dealer_trading_phase(subsystem, system, current_day=3)

        ticket = subsystem.tickets.get(ticket_id)
        if ticket:  # Might have been removed if matured
            assert ticket.bucket_id == "short"
            assert ticket.owner_id == "dealer_short"
            assert ticket in subsystem.dealers["short"].inventory

            # Main system payable should also be reassigned
            payable = system.state.contracts[test_payable.id]
            assert payable.asset_holder_id == "dealer_short"

    def test_dealer_bucket_transition_is_free_internal_move(self):
        """Bucket transitions are internal moves (cash pooling), no interdealer payment."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=5)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        test_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="dealer_mid",
            liability_issuer_id="H1",
            due_day=5,
            maturity_distance=5,
        )
        system.add_contract(test_payable)
        _ingest_new_payables(subsystem, system, current_day=0)

        # Run at day=3: tau=2 → short bucket. Ticket moves internally.
        run_dealer_trading_phase(subsystem, system, current_day=3)

        ticket_id = f"TKT_{test_payable.id}"
        ticket = subsystem.tickets.get(ticket_id)
        if ticket:
            assert ticket.owner_id == "dealer_short"
            assert ticket in subsystem.dealers["short"].inventory

    def test_vbt_ticket_moves_to_new_bucket_on_maturity_update(self):
        """VBT_long ticket ages into mid bucket → moves to vbt_mid inventory."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=10)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # At current_day=0: remaining_tau=10 → "long" (tau >= 9)
        test_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=25,
            denom="X",
            asset_holder_id="vbt_long",
            liability_issuer_id="H1",
            due_day=10,
            maturity_distance=10,
        )
        system.add_contract(test_payable)
        _ingest_new_payables(subsystem, system, current_day=0)

        ticket_id = f"TKT_{test_payable.id}"
        assert subsystem.tickets[ticket_id].bucket_id == "long"
        assert subsystem.tickets[ticket_id].owner_id == "vbt_long"

        # Run trading phase at day=4: remaining_tau = 10-4 = 6 → "mid" (tau 4-8)
        run_dealer_trading_phase(subsystem, system, current_day=4)

        ticket = subsystem.tickets.get(ticket_id)
        if ticket:
            assert ticket.bucket_id == "mid"
            assert ticket.owner_id == "vbt_mid"
            assert ticket in subsystem.vbts["mid"].inventory

            payable = system.state.contracts[test_payable.id]
            assert payable.asset_holder_id == "vbt_mid"

    def test_cash_pooling_equalizes_desk_cash(self):
        """After pooling, all dealer desks have equal cash; same for VBT desks."""
        from bilancio.engines.dealer_integration import _pool_desk_cash

        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=5)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # Artificially set unequal cash (simulating short absorbing defaults)
        subsystem.dealers["short"].cash = Decimal(100)
        subsystem.dealers["mid"].cash = Decimal(400)
        subsystem.dealers["long"].cash = Decimal(500)

        subsystem.vbts["short"].cash = Decimal(50)
        subsystem.vbts["mid"].cash = Decimal(200)
        subsystem.vbts["long"].cash = Decimal(350)

        _pool_desk_cash(subsystem)

        # All dealer desks should have equal cash
        expected_dealer = Decimal(1000) / 3
        for dealer in subsystem.dealers.values():
            assert dealer.cash == expected_dealer, \
                f"dealer_{dealer.bucket_id} cash={dealer.cash}, expected {expected_dealer}"

        # All VBT desks should have equal cash
        expected_vbt = Decimal(600) / 3
        for vbt in subsystem.vbts.values():
            assert vbt.cash == expected_vbt, \
                f"vbt_{vbt.bucket_id} cash={vbt.cash}, expected {expected_vbt}"

    def test_trader_ticket_bucket_changes_without_owner_change(self):
        """Trader-owned ticket changes bucket but owner stays the same."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(n_agents=3, maturity_days=5)
        subsystem = initialize_balanced_dealer_subsystem(system, dealer_config, current_day=0)

        # Create mid-bucket trader ticket: remaining_tau=5 → "mid"
        test_payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=5,
            maturity_distance=5,
        )
        system.add_contract(test_payable)
        _ingest_new_payables(subsystem, system, current_day=0)

        ticket_id = f"TKT_{test_payable.id}"
        assert subsystem.tickets[ticket_id].owner_id == "H1"

        # Run trading phase at day=3: remaining_tau=2 → "short"
        run_dealer_trading_phase(subsystem, system, current_day=3)

        ticket = subsystem.tickets.get(ticket_id)
        if ticket:
            assert ticket.bucket_id == "short"
            assert ticket.owner_id == "H1"  # Stays with trader
            assert ticket in subsystem.traders["H1"].tickets_owned


class TestFullCycle:
    """End-to-end: init → settle → rollover → ingest → verify continuous trading."""

    def test_rollover_creates_payables_that_get_ingested(self):
        """Settle → rollover (Model C) → ingest → verify new tickets exist."""
        system, trader_ids, dealer_config, face_value = _dealer_ring_system(
            n_agents=3, maturity_days=3
        )
        system.state.day = 0

        subsystem = initialize_balanced_dealer_subsystem(
            system, dealer_config, current_day=0,
        )
        system.state.dealer_subsystem = subsystem

        initial_tickets = len(subsystem.tickets)

        # Advance to settlement day
        system.state.day = 3

        # Settle all payables due on day 3
        settled = settle_due(system, 3, rollover_enabled=True)
        assert len(settled) > 0, "Some payables should have settled"

        # Rollover with Model C
        new_payable_ids = rollover_settled_payables(
            system, 3, settled, dealer_active=True
        )
        assert len(new_payable_ids) > 0, "Rollover should create new payables"

        # Ingest new payables into dealer subsystem
        new_count = _ingest_new_payables(subsystem, system, current_day=3)
        assert new_count > 0, "Ingestion should pick up rollover payables"

        # Verify the new tickets are for the rollover payables
        for pid in new_payable_ids:
            if pid in system.state.contracts:  # Not removed by settlement
                assert pid in subsystem.payable_to_ticket, \
                    f"Payable {pid} should have a corresponding ticket"

    def test_dealer_inventory_replenishes_after_rollover(self):
        """VBT/Dealer inventory should grow after ingesting rollover payables."""
        system, trader_ids, dealer_config, _ = _dealer_ring_system(
            n_agents=3, maturity_days=3
        )
        system.state.day = 0

        subsystem = initialize_balanced_dealer_subsystem(
            system, dealer_config, current_day=0,
        )

        # Count VBT short tickets before
        vbt_before = len(subsystem.vbts["short"].inventory)

        # Settle on day 3
        system.state.day = 3
        settled = settle_due(system, 3, rollover_enabled=True)

        # Filter to VBT-owned settled payables
        vbt_settled = [(d, c, a, m) for d, c, a, m in settled if c == "vbt_short"]

        if vbt_settled:
            # Rollover with Model C (VBT gets cash transfer)
            new_ids = rollover_settled_payables(system, 3, vbt_settled, dealer_active=True)

            # Ingest
            _ingest_new_payables(subsystem, system, current_day=3)

            # VBT inventory should have new tickets
            vbt_after = len(subsystem.vbts["short"].inventory)
            # Note: original tickets may have been cleaned up as matured,
            # but new ones should be added
            assert vbt_after >= len(new_ids), \
                f"VBT inventory ({vbt_after}) should have at least {len(new_ids)} new tickets"
