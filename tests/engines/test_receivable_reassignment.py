"""Tests for pro-rata receivable reassignment on agent default (Plan 055).

When an agent defaults, its receivables (payables where it is the creditor)
are reassigned pro-rata to its own creditors, based on what the defaulting
agent owed each of them.

For the ring topology (one debtor, one creditor per agent), this produces
identical behavior to the old _reconnect_ring() function.
"""

from decimal import Decimal

from bilancio.core.events import EventKind
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.means_of_payment import Cash
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.settlement import (
    _collect_creditor_weights,
    _expel_agent,
    _reassign_receivables,
    _remove_contract,
    settle_due,
)
from bilancio.engines.system import System


def _ring_system(n_agents: int, default_mode: str = "expel-agent"):
    """Create a System with CB + N household agents arranged in a ring.

    Ring structure: H1->H2->H3->...->HN->H1
    Each agent owes 100 to the next agent, with maturity_distance=3.
    """
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.add_agent(cb)

    agents = []
    for i in range(1, n_agents + 1):
        agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(agent)
        agents.append(agent)

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


def _multi_creditor_system():
    """Create a system where H2 owes H3 (60) and H4 (40), and H1 owes H2 (100).

    H1 ->100-> H2 ->60-> H3
                    ->40-> H4

    When H2 defaults, H1's receivable (H1->H2) should be reassigned:
    - H1->H3: 60 (60% weight)
    - H1->H4: 40 (40% weight)
    """
    system = System(default_mode="expel-agent")
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.add_agent(cb)

    for i in range(1, 5):
        agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(agent)

    # H1 owes H2: 100 (this is H2's receivable)
    p1 = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=3,
        maturity_distance=3,
    )
    system.add_contract(p1)

    # H2 owes H3: 60 (H2's liability to H3)
    p2 = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=60,
        denom="X",
        asset_holder_id="H3",
        liability_issuer_id="H2",
        due_day=3,
        maturity_distance=3,
    )
    system.add_contract(p2)

    # H2 owes H4: 40 (H2's liability to H4)
    p3 = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=40,
        denom="X",
        asset_holder_id="H4",
        liability_issuer_id="H2",
        due_day=3,
        maturity_distance=3,
    )
    system.add_contract(p3)

    return system, p1, p2, p3


class TestCollectCreditorWeights:
    """Test _collect_creditor_weights() function."""

    def test_ring_single_creditor(self):
        """In a ring, each agent has exactly one creditor."""
        system, agents, payables = _ring_system(3)
        # H2 owes H3 (payable H2->H3), so H3 is H2's only creditor
        weights = _collect_creditor_weights(system, "H2")
        assert "H3" in weights
        assert weights["H3"] == Decimal(1)

    def test_multi_creditor(self):
        """Agent with two creditors gets proportional weights."""
        system, p1, p2, p3 = _multi_creditor_system()
        weights = _collect_creditor_weights(system, "H2")
        assert len(weights) == 2
        assert weights["H3"] == Decimal("60") / Decimal("100")
        assert weights["H4"] == Decimal("40") / Decimal("100")

    def test_exclude_trigger(self):
        """Trigger payable is excluded from weight calculation."""
        system, p1, p2, p3 = _multi_creditor_system()
        # Exclude p2 (H2->H3, amount=60), leaving only p3 (H2->H4, amount=40)
        weights = _collect_creditor_weights(system, "H2", exclude_contract_id=p2.id)
        assert len(weights) == 1
        assert weights["H4"] == Decimal(1)

    def test_no_creditors(self):
        """Agent with no liabilities returns empty dict."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        agent = Household(id="H1", name="Household 1", kind="household")
        system.add_agent(agent)
        weights = _collect_creditor_weights(system, "H1")
        assert weights == {}


class TestReassignReceivables:
    """Test _reassign_receivables() function."""

    def test_ring_parity(self):
        """In a 3-agent ring, reassignment produces same result as _reconnect_ring().

        Ring: H1->H2->H3->H1
        H2 defaults. H2's creditor is H3 (weight=1.0).
        H2's receivable: H1->H2 (amount=100).
        Result: H1->H3 (amount=100) -- same as ring reconnection.

        NOTE: In the actual settle_due flow, the trigger is removed before
        collecting weights, leaving zero remaining liabilities for a pure ring.
        To achieve ring-parity behavior, we collect weights BEFORE removing
        the trigger (i.e., the full liability set including the trigger).
        """
        system, agents, payables = _ring_system(3)
        # payables[0] = H1->H2, payables[1] = H2->H3, payables[2] = H3->H1

        # Collect weights BEFORE removing trigger to include H2->H3
        weights = _collect_creditor_weights(system, "H2")
        assert weights["H3"] == Decimal(1)

        # Simulate H2 defaulting: trigger is payable[1] (H2->H3)
        trigger = payables[1]  # H2->H3
        _remove_contract(system, trigger.id)

        # Expel H2
        _expel_agent(
            system, "H2",
            trigger_contract_id=trigger.id,
            trigger_kind=trigger.kind,
            trigger_shortfall=100,
            cancelled_contract_ids={trigger.id},
        )

        # Reassign
        result = _reassign_receivables(system, "H2", weights, day=1)

        # Should have created one new payable: H1->H3
        assert len(result) == 1
        assert result[0]["debtor"] == "H1"
        assert result[0]["new_creditor"] == "H3"
        assert result[0]["amount"] == 100

    def test_multi_creditor_pro_rata(self):
        """Agent with 2 creditors (60/40 split), verify amounts."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 5):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 100 (receivable for H2)
        r1 = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        )
        system.add_contract(r1)

        # H2 owes H3: 60
        l1 = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=60, denom="X",
            asset_holder_id="H3", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        )
        system.add_contract(l1)

        # H2 owes H4: 40
        l2 = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=40, denom="X",
            asset_holder_id="H4", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        )
        system.add_contract(l2)

        # Collect weights (both liabilities, no exclusion)
        weights = _collect_creditor_weights(system, "H2")
        assert weights["H3"] == Decimal("60") / Decimal("100")
        assert weights["H4"] == Decimal("40") / Decimal("100")

        # Expel H2
        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)

        # Reassign receivables
        result = _reassign_receivables(system, "H2", weights, day=1)

        # H1->H2(100) should become H1->H3(60) and H1->H4(40)
        assert len(result) == 2
        amounts = {r["new_creditor"]: r["amount"] for r in result}
        assert amounts["H3"] == 60
        assert amounts["H4"] == 40

    def test_multi_debtor(self):
        """Agent with 2 debtors: both receivables reassigned."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 5):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 50
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=50, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        ))

        # H3 owes H2: 80
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=80, denom="X",
            asset_holder_id="H2", liability_issuer_id="H3",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H4: 100 (sole creditor)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H4", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        weights = _collect_creditor_weights(system, "H2")
        assert weights["H4"] == Decimal(1)

        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)
        result = _reassign_receivables(system, "H2", weights, day=1)

        # Both receivables should be reassigned to H4
        assert len(result) == 2
        amounts = sorted([r["amount"] for r in result])
        assert amounts == [50, 80]
        assert all(r["new_creditor"] == "H4" for r in result)

    def test_no_creditors_debt_relief(self):
        """No creditors -> receivables deleted (debt relief)."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 3):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 100 (receivable for H2)
        p = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        )
        system.add_contract(p)

        # H2 has no liabilities (no creditors)
        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)

        # Empty weights = no creditors
        result = _reassign_receivables(system, "H2", {}, day=1)
        assert result == []  # Nothing to reassign to
        # Original receivable should be removed
        assert p.id not in system.state.contracts

    def test_no_receivables(self):
        """No receivables -> no-op."""
        system, agents, payables = _ring_system(3)
        # Give H2 creditors but no receivables won't happen naturally in ring,
        # so we manually remove H2's receivable (H1->H2)
        h1_to_h2 = payables[0]  # H1->H2
        _remove_contract(system, h1_to_h2.id)

        weights = _collect_creditor_weights(system, "H2")
        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)

        result = _reassign_receivables(system, "H2", weights, day=1)
        assert result == []

    def test_defaulted_debtor_skipped(self):
        """Don't reassign receivables from already-defaulted agents."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 4):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 100
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H3: 100
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H3", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        # Default H1 first
        system.state.agents["H1"].defaulted = True
        system.state.defaulted_agent_ids.add("H1")

        weights = _collect_creditor_weights(system, "H2")
        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)
        result = _reassign_receivables(system, "H2", weights, day=1)

        # H1 is defaulted, so the H1->H2 receivable should be skipped
        assert result == []

    def test_minimum_amount_threshold(self):
        """Tiny reassigned fractions (< 1) are skipped."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 5):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 1 (tiny receivable)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=1, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H3: 90
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=90, denom="X",
            asset_holder_id="H3", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H4: 10
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=10, denom="X",
            asset_holder_id="H4", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        weights = _collect_creditor_weights(system, "H2")
        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)
        result = _reassign_receivables(system, "H2", weights, day=1)

        # H3 gets 90% of 1 = 0.9 -> rounds to 0 -> skipped
        # H4 gets 10% of 1 = 0.1 -> rounds to 0 -> skipped
        # Both are below threshold, so no reassignments
        assert len(result) == 0

    def test_self_loop_skipped(self):
        """Self-loops (debtor == creditor) are skipped in reassignment."""
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 4):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 100 (receivable for H2 from H1)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H1: 50 (creditor is H1 -- same as debtor of receivable!)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=50, denom="X",
            asset_holder_id="H1", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        # H2 owes H3: 50
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=50, denom="X",
            asset_holder_id="H3", liability_issuer_id="H2",
            due_day=3, maturity_distance=3,
        ))

        weights = _collect_creditor_weights(system, "H2")
        assert weights["H1"] == Decimal("0.5")
        assert weights["H3"] == Decimal("0.5")

        _expel_agent(system, "H2", trigger_contract_id=None, trigger_kind="payable", trigger_shortfall=0)
        result = _reassign_receivables(system, "H2", weights, day=1)

        # H1->H2 receivable: H1 is both debtor AND one of the creditors
        # H1->H1 self-loop should be SKIPPED
        # H1->H3 should be created (50% of 100 = 50)
        assert len(result) == 1
        assert result[0]["debtor"] == "H1"
        assert result[0]["new_creditor"] == "H3"
        assert result[0]["amount"] == 50

    def test_event_logging(self):
        """Verify RECEIVABLE_REASSIGNED events are logged."""
        system, agents, payables = _ring_system(3)
        # Collect weights BEFORE removing trigger to include H2->H3
        weights = _collect_creditor_weights(system, "H2")
        trigger = payables[1]  # H2->H3
        _remove_contract(system, trigger.id)
        _expel_agent(
            system, "H2",
            trigger_contract_id=trigger.id,
            trigger_kind=trigger.kind,
            trigger_shortfall=100,
            cancelled_contract_ids={trigger.id},
        )
        _reassign_receivables(system, "H2", weights, day=1)

        reassign_events = [
            e for e in system.state.events
            if e.get("kind") == EventKind.RECEIVABLE_REASSIGNED
        ]
        assert len(reassign_events) == 1
        evt = reassign_events[0]
        assert evt["defaulted_agent"] == "H2"
        assert evt["debtor"] == "H1"
        assert evt["new_creditor"] == "H3"
        assert evt["amount"] == 100


class TestSettleDueWithReassignment:
    """Integration tests: verify settle_due uses reassignment path."""

    def test_settle_due_reassigns_multi_creditor(self):
        """In a multi-creditor topology, receivables are reassigned on default.

        H1 ->100-> H2 ->60-> H3
                        ->40-> H4

        H2 has no cash and defaults on H2->H3 (due day 1). H1->H2 is not
        due yet (day 5). After H2 defaults:
        - Trigger: H2->H3 (removed during default)
        - Remaining liability: H2->H4 (weight=1.0)
        - H2's receivable: H1->H2 -> reassigned to H1->H4
        """
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
        system.add_agent(cb)
        for i in range(1, 5):
            agent = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            system.add_agent(agent)

        # H1 owes H2: 100 (H2's receivable, not due yet)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=100, denom="X",
            asset_holder_id="H2", liability_issuer_id="H1",
            due_day=5, maturity_distance=5,
        ))

        # H2 owes H3: 60 (due day 1, H2 will default on this)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=60, denom="X",
            asset_holder_id="H3", liability_issuer_id="H2",
            due_day=1, maturity_distance=3,
        ))

        # H2 owes H4: 40 (due day 5, not due yet -- remains as liability)
        system.add_contract(Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=40, denom="X",
            asset_holder_id="H4", liability_issuer_id="H2",
            due_day=5, maturity_distance=3,
        ))

        # H2 has NO cash -> will default on H2->H3

        # Run settlement for day 1 (only H2->H3 is due)
        settle_due(system, day=1, rollover_enabled=False)

        # H2 should have defaulted
        assert "H2" in system.state.defaulted_agent_ids

        # Check for RECEIVABLE_REASSIGNED events
        reassign_events = [
            e for e in system.state.events
            if e.get("kind") == EventKind.RECEIVABLE_REASSIGNED
        ]
        # Creditor weights are collected BEFORE trigger removal, so both
        # H3 (60) and H4 (40) are included in weights:
        # H1->H2(100) → H1->H3(60) + H1->H4(40)
        assert len(reassign_events) == 2
        amounts = {evt["new_creditor"]: evt["amount"] for evt in reassign_events}
        assert amounts["H3"] == 60
        assert amounts["H4"] == 40
        assert all(evt["defaulted_agent"] == "H2" for evt in reassign_events)
        assert all(evt["debtor"] == "H1" for evt in reassign_events)
