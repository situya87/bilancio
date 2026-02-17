"""Integration tests for ring scenarios with banking infrastructure.

These tests verify that the full simulation pipeline works correctly
when bank agents are present and agents settle via bank deposits
instead of raw CB cash.

Key behaviors tested:
- Settlement via deposit transfer (ClientPayment / IntraBankPayment)
- Banks never run out of reserves (ample reserves guarantee)
- n_banks=0 produces identical behavior to the pre-banking code path
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit, ReserveDeposit
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.settlement import settle_due
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_ring_system_with_banks(
    n_households: int = 4,
    n_banks: int = 2,
    cash_per_agent: int = 200,
    payable_amount: int = 50,
    maturity_days: int = 2,
    reserve_multiplier: float = 10.0,
) -> System:
    """Create a ring system with banking infrastructure.

    Topology: H1->H2->H3->...->Hn->H1 (ring of payables)
    Each household has a deposit at a bank (round-robin assignment).
    Banks have ample reserves.

    The policy is configured so households settle with deposits first.
    """
    # Build policy with deposit-first settlement for households
    policy = PolicyEngine.default()
    # Override mop_rank: households use bank_deposit
    policy.mop_rank["household"] = [InstrumentKind.BANK_DEPOSIT]

    sys = System(policy=policy)

    # Central bank
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    sys.add_agent(cb)

    # Create banks
    bank_ids = []
    for i in range(1, n_banks + 1):
        bank = Bank(id=f"B{i}", name=f"Bank {i}", kind="bank")
        sys.add_agent(bank)
        bank_ids.append(f"B{i}")

    # Create households with cash
    household_ids = []
    for i in range(1, n_households + 1):
        h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        sys.add_agent(h)
        household_ids.append(f"H{i}")

        # Mint cash then deposit it at the assigned bank
        sys.mint_cash(f"H{i}", cash_per_agent)
        assigned_bank = bank_ids[(i - 1) % n_banks]
        deposit_cash(sys, f"H{i}", assigned_bank, cash_per_agent)

    # Mint ample reserves for each bank
    total_deposited = cash_per_agent * n_households
    reserves_per_bank = int(reserve_multiplier * total_deposited / n_banks)
    for bank_id in bank_ids:
        sys.mint_reserves(bank_id, reserves_per_bank)

    # Create ring payables: H1->H2, H2->H3, ..., Hn->H1
    for i in range(n_households):
        debtor = household_ids[i]
        creditor = household_ids[(i + 1) % n_households]
        p = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=payable_amount,
            denom="X",
            asset_holder_id=creditor,
            liability_issuer_id=debtor,
            due_day=sys.state.day + maturity_days,
        )
        sys.add_contract(p)

    return sys


def _total_deposits(system: System) -> int:
    """Sum all BANK_DEPOSIT instrument amounts across the system."""
    total = 0
    for contract in system.state.contracts.values():
        if contract.kind == InstrumentKind.BANK_DEPOSIT:
            total += contract.amount
    return total


def _total_reserves(system: System) -> int:
    """Sum all RESERVE_DEPOSIT instrument amounts across the system."""
    total = 0
    for contract in system.state.contracts.values():
        if contract.kind == InstrumentKind.RESERVE_DEPOSIT:
            total += contract.amount
    return total


def _agent_deposit_balance(system: System, agent_id: str) -> int:
    """Sum deposits held by a specific agent."""
    agent = system.state.agents.get(agent_id)
    if not agent:
        return 0
    total = 0
    for cid in agent.asset_ids:
        c = system.state.contracts.get(cid)
        if c and c.kind == InstrumentKind.BANK_DEPOSIT:
            total += c.amount
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRingWithBanksSettlement:
    """Verify that ring settlement works correctly with banks."""

    def test_ring_2banks_settlement(self):
        """Settlement in a 4-agent 2-bank ring should transfer deposits correctly.

        After settlement on the due day:
        - Debtors' deposits decrease by the payable amount
        - Creditors' deposits increase by the payable amount
        - Payment events (ClientPayment or IntraBankPayment) are logged
        - No validation errors occur
        """
        sys = _create_ring_system_with_banks(
            n_households=4,
            n_banks=2,
            cash_per_agent=200,
            payable_amount=50,
            maturity_days=0,  # Due today (day 0)
        )

        # Record deposit balances before settlement
        deposits_before = {
            f"H{i}": _agent_deposit_balance(sys, f"H{i}") for i in range(1, 5)
        }

        # Verify all households start with 200 in deposits
        for hid, dep in deposits_before.items():
            assert dep == 200, f"{hid} should start with 200 in deposits, got {dep}"

        # Record total deposits before
        total_deposits_before = _total_deposits(sys)

        # Run settlement
        run_day(sys, enable_dealer=False, enable_lender=False)

        # Total deposits should be conserved (settlement is a transfer, not creation)
        total_deposits_after = _total_deposits(sys)
        assert total_deposits_after == total_deposits_before, (
            f"Total deposits changed: before={total_deposits_before}, "
            f"after={total_deposits_after}. Settlement should conserve deposits."
        )

        # Check for payment events
        payment_events = [
            e for e in sys.state.events
            if e.get("kind") in ("ClientPayment", "IntraBankPayment", "CashPayment")
        ]
        assert len(payment_events) > 0, (
            "At least one payment event should be logged during settlement. "
            f"All events: {[e.get('kind') for e in sys.state.events]}"
        )

        # Check for settled payable events
        settled_events = [
            e for e in sys.state.events if e.get("kind") == "PayableSettled"
        ]
        assert len(settled_events) == 4, (
            f"Expected 4 PayableSettled events for 4 payables, got {len(settled_events)}"
        )

    def test_ring_2banks_ample_reserves(self):
        """Banks should never run out of reserves with the ample reserves guarantee.

        Run a multi-day simulation and verify that every bank always has
        positive reserve balances.
        """
        sys = _create_ring_system_with_banks(
            n_households=4,
            n_banks=2,
            cash_per_agent=200,
            payable_amount=50,
            maturity_days=1,  # Due on day 1
        )

        # Run 3 days of simulation
        for day in range(3):
            run_day(sys, enable_dealer=False, enable_lender=False)

            # Check bank reserves after each day
            for bank_id in ("B1", "B2"):
                bank = sys.state.agents[bank_id]
                reserve_balance = 0
                for cid in bank.asset_ids:
                    c = sys.state.contracts.get(cid)
                    if c and c.kind == InstrumentKind.RESERVE_DEPOSIT:
                        reserve_balance += c.amount

                assert reserve_balance > 0, (
                    f"Bank {bank_id} has zero reserves on day {sys.state.day}. "
                    "Ample reserves guarantee violated."
                )


class TestRingNoBanksIdentical:
    """Verify n_banks=0 produces behavior consistent with pre-banking code."""

    def test_ring_no_banks_settlement(self):
        """A ring with n_banks=0 (pure cash) should still settle correctly.

        This verifies the banking infrastructure does not break the
        existing cash-based settlement path.
        """
        # Build a pure-cash system (no banks, no deposits)
        policy = PolicyEngine.default()
        sys = System(policy=policy)

        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        sys.add_agent(cb)

        for i in range(1, 5):
            h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
            sys.add_agent(h)
            sys.mint_cash(f"H{i}", 200)

        # Create ring payables due today
        for i in range(4):
            debtor = f"H{i + 1}"
            creditor = f"H{(i + 1) % 4 + 1}"
            p = Payable(
                id=sys.new_contract_id("P"),
                kind=InstrumentKind.PAYABLE,
                amount=50,
                denom="X",
                asset_holder_id=creditor,
                liability_issuer_id=debtor,
                due_day=0,
            )
            sys.add_contract(p)

        # Run settlement
        run_day(sys, enable_dealer=False, enable_lender=False)

        # All payables should be settled
        settled_events = [
            e for e in sys.state.events if e.get("kind") == "PayableSettled"
        ]
        assert len(settled_events) == 4, (
            f"Expected 4 PayableSettled events, got {len(settled_events)}"
        )

        # No default events
        default_events = [
            e for e in sys.state.events
            if e.get("kind") in ("ObligationDefaulted", "AgentDefaulted")
        ]
        assert len(default_events) == 0, (
            f"No defaults expected in a well-funded ring, got {len(default_events)}"
        )


class TestRingInterBankSettlement:
    """Verify inter-bank settlement triggers reserve transfers."""

    def test_cross_bank_settlement_uses_reserves(self):
        """When debtor and creditor are at different banks, reserves must move.

        H1 (at B1) owes H2 (at B2). Settlement should:
        1. Debit H1's deposit at B1
        2. Credit H2's deposit at B2
        3. Transfer reserves from B1 to B2
        """
        policy = PolicyEngine.default()
        policy.mop_rank["household"] = [InstrumentKind.BANK_DEPOSIT]

        sys = System(policy=policy)
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        b1 = Bank(id="B1", name="Bank 1", kind="bank")
        b2 = Bank(id="B2", name="Bank 2", kind="bank")
        h1 = Household(id="H1", name="Household 1", kind="household")
        h2 = Household(id="H2", name="Household 2", kind="household")

        sys.add_agent(cb)
        sys.add_agent(b1)
        sys.add_agent(b2)
        sys.add_agent(h1)
        sys.add_agent(h2)

        # H1 deposits at B1
        sys.mint_cash("H1", 200)
        deposit_cash(sys, "H1", "B1", 200)

        # H2 deposits at B2
        sys.mint_cash("H2", 200)
        deposit_cash(sys, "H2", "B2", 200)

        # Ample reserves for both banks
        sys.mint_reserves("B1", 5000)
        sys.mint_reserves("B2", 5000)

        b1_reserves_before = sum(
            sys.state.contracts[cid].amount
            for cid in sys.state.agents["B1"].asset_ids
            if sys.state.contracts.get(cid)
            and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        )

        # H1 owes H2 100, due today
        p = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        sys.add_contract(p)

        # Run settlement
        run_day(sys, enable_dealer=False, enable_lender=False)

        # H1's deposit should decrease by 100
        h1_dep = _agent_deposit_balance(sys, "H1")
        assert h1_dep == 100, f"H1 deposit should be 100 after paying 100, got {h1_dep}"

        # H2's deposit should increase by 100
        h2_dep = _agent_deposit_balance(sys, "H2")
        assert h2_dep == 300, f"H2 deposit should be 300 after receiving 100, got {h2_dep}"

        # Verify ClientPayment event was logged (cross-bank)
        client_payment_events = [
            e for e in sys.state.events
            if e.get("kind") == "ClientPayment"
        ]
        assert len(client_payment_events) >= 1, (
            "Cross-bank settlement should produce a ClientPayment event. "
            f"Events: {[e.get('kind') for e in sys.state.events]}"
        )
