"""Cross-phase interaction tests for banking mode and dealer sync.

These tests verify that the dealer subsystem correctly handles agents who
hold bank deposits instead of (or in addition to) raw CB cash.

Key behaviors tested:
- Deposits survive a dealer sync round-trip (from-system -> to-system)
- Sync adjusts deposits (not cash) when in banking mode
- Dealer/VBT sync also adjusts deposits in banking mode
"""

from decimal import Decimal

from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.engines.dealer_integration import (
    _get_agent_cash,
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from bilancio.engines.dealer_sync import (
    _sync_trader_cash_from_system,
    _sync_trader_cash_to_system,
)
from bilancio.engines.system import System

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRADER_IDS = ["H1", "H2", "H3"]


def create_dealer_config() -> DealerRingConfig:
    """Standard dealer configuration for testing."""
    return DealerRingConfig(
        ticket_size=Decimal(1),
        buckets=list(DEFAULT_BUCKETS),
        dealer_share=Decimal("0.25"),
        vbt_share=Decimal("0.50"),
        vbt_anchors={
            "short": (Decimal("1.0"), Decimal("0.20")),
            "mid": (Decimal("1.0"), Decimal("0.30")),
            "long": (Decimal("1.0"), Decimal("0.40")),
        },
        phi_M=Decimal("0.1"),
        phi_O=Decimal("0.1"),
        clip_nonneg_B=True,
        seed=42,
    )


def create_system_with_deposits() -> System:
    """Create a system where traders have bank deposits AND payables.

    H1, H2, H3 each have a deposit at B1 (no raw cash).
    Payables form a ring: H1->H2, H2->H3, H3->H1.
    """
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(bank)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)

    # Give each household a bank deposit (no raw cash)
    for hid, amount in [("H1", 100), ("H2", 100), ("H3", 100)]:
        dep = BankDeposit(
            id=f"DEP_{hid}",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=amount,
            denom="X",
            asset_holder_id=hid,
            liability_issuer_id="B1",
        )
        sys.add_contract(dep)

    # Ring payables
    p1 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=50,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day + 2,
    )
    sys.add_contract(p1)

    p2 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=30,
        denom="X",
        asset_holder_id="H3",
        liability_issuer_id="H2",
        due_day=sys.state.day + 5,
    )
    sys.add_contract(p2)

    p3 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=20,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="H3",
        due_day=sys.state.day + 10,
    )
    sys.add_contract(p3)

    return sys


def create_system_with_cash_and_deposits() -> System:
    """Create a system where traders have both cash AND deposits.

    H1 has 50 cash + 100 deposit = 150 total
    H2 has 100 deposit only
    H3 has 100 deposit only
    """
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(bank)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)

    # H1 has both cash and deposit
    sys.mint_cash("H1", 50)
    dep1 = BankDeposit(
        id="DEP_H1",
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=100,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="B1",
    )
    sys.add_contract(dep1)

    # H2, H3 only have deposits
    for hid, amount in [("H2", 100), ("H3", 100)]:
        dep = BankDeposit(
            id=f"DEP_{hid}",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=amount,
            denom="X",
            asset_holder_id=hid,
            liability_issuer_id="B1",
        )
        sys.add_contract(dep)

    # Ring payables
    p1 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=50,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day + 2,
    )
    sys.add_contract(p1)

    p2 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=30,
        denom="X",
        asset_holder_id="H3",
        liability_issuer_id="H2",
        due_day=sys.state.day + 5,
    )
    sys.add_contract(p2)

    p3 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=20,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="H3",
        due_day=sys.state.day + 10,
    )
    sys.add_contract(p3)

    return sys


def _get_deposit_amount(sys: System, agent_id: str) -> int:
    """Get the total bank deposit amount for an agent."""
    agent = sys.state.agents.get(agent_id)
    if not agent:
        return 0
    total = 0
    for cid in agent.asset_ids:
        c = sys.state.contracts.get(cid)
        if c and c.kind == InstrumentKind.BANK_DEPOSIT:
            total += c.amount
    return total


def _get_cash_amount(sys: System, agent_id: str) -> int:
    """Get the total CB cash amount for an agent."""
    agent = sys.state.agents.get(agent_id)
    if not agent:
        return 0
    total = 0
    for cid in agent.asset_ids:
        c = sys.state.contracts.get(cid)
        if c and c.kind == InstrumentKind.CASH:
            total += c.amount
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDepositSurvivesDealerSync:
    """Verify deposits survive dealer sync round-trips."""

    def test_deposit_survives_dealer_sync_roundtrip(self):
        """Sync from-system then to-system with no trades must leave deposits unchanged.

        When traders have bank deposits, the dealer sync must recognize the
        deposits as part of the trader's cash and not create phantom mints/burns.
        """
        sys = create_system_with_deposits()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Record deposits before round-trip
        deposits_before = {tid: _get_deposit_amount(sys, tid) for tid in TRADER_IDS}

        # Record total "cash" (cash + deposits) before
        total_before = {tid: _get_agent_cash(sys, tid) for tid in TRADER_IDS}

        # Round-trip: sync from system, then sync back
        _sync_trader_cash_from_system(subsystem, sys)
        _sync_trader_cash_to_system(subsystem, sys)

        # Deposits must be unchanged
        for tid in TRADER_IDS:
            dep_after = _get_deposit_amount(sys, tid)
            assert dep_after == deposits_before[tid], (
                f"Agent {tid} deposit changed from {deposits_before[tid]} to {dep_after} "
                f"during no-op sync round-trip"
            )

        # Total "cash" (cash + deposits) must be unchanged
        for tid in TRADER_IDS:
            total_after = _get_agent_cash(sys, tid)
            assert total_after == total_before[tid], (
                f"Agent {tid} total cash changed from {total_before[tid]} to {total_after} "
                f"during no-op sync round-trip"
            )

    def test_deposit_survives_full_dealer_cycle(self):
        """Full trading phase + sync must not alter deposits when no trades occur.

        With a disabled subsystem, the entire dealer cycle should be a no-op
        for agents holding deposits.
        """
        sys = create_system_with_deposits()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.enabled = False

        # Record deposits before
        deposits_before = {tid: _get_deposit_amount(sys, tid) for tid in TRADER_IDS}

        # Run full cycle
        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        sync_dealer_to_system(subsystem, sys)

        assert events == [], "Disabled subsystem should produce no trade events"

        # Deposits must be unchanged
        for tid in TRADER_IDS:
            dep_after = _get_deposit_amount(sys, tid)
            assert dep_after == deposits_before[tid], (
                f"Agent {tid} deposit changed from {deposits_before[tid]} to {dep_after} "
                f"after disabled-subsystem dealer phase + sync"
            )


class TestDealerSyncAdjustsDeposit:
    """Verify that sync adjusts deposits (not cash) in banking mode."""

    def test_dealer_sync_adjusts_deposit_not_cash(self):
        """When trader has deposits and cash, sync should adjust the deposit.

        Setup: H1 has 50 cash + 100 deposit = 150 total.
        After a simulated trade that increases trader.cash by 20 in the subsystem,
        the sync-to-system should increase H1's deposit by 20 (not mint new CB cash).
        """
        sys = create_system_with_cash_and_deposits()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Sync from system so trader.cash is up to date
        _sync_trader_cash_from_system(subsystem, sys)

        # Verify trader H1 sees 150 (50 cash + 100 deposit)
        trader_h1 = subsystem.traders.get("H1")
        assert trader_h1 is not None, "H1 should be a trader in the subsystem"
        assert trader_h1.cash == Decimal(150), (
            f"H1 trader.cash should be 150 (50 cash + 100 deposit), got {trader_h1.cash}"
        )

        # Simulate a trade: H1 gains 20 in the subsystem
        trader_h1.cash += Decimal(20)

        # Record state before sync-to
        cash_before = _get_cash_amount(sys, "H1")
        deposit_before = _get_deposit_amount(sys, "H1")

        # Sync back to system
        _sync_trader_cash_to_system(subsystem, sys)

        # Check: the deposit should have increased, not the CB cash
        cash_after = _get_cash_amount(sys, "H1")
        deposit_after = _get_deposit_amount(sys, "H1")

        # Deposit should increase by 20
        assert deposit_after == deposit_before + 20, (
            f"H1 deposit should increase by 20: before={deposit_before}, after={deposit_after}. "
            f"The sync should adjust deposits in banking mode."
        )

        # CB cash should remain unchanged (sync should NOT mint/retire cash)
        assert cash_after == cash_before, (
            f"H1 CB cash should be unchanged: before={cash_before}, after={cash_after}. "
            f"In banking mode, sync should adjust deposits, not mint/retire cash."
        )

    def test_dealer_sync_decreases_deposit(self):
        """When trader.cash decreases (sells at loss), deposit should decrease.

        Setup: H2 has 100 deposit.
        Simulate trader.cash decreasing by 15 (sold a claim at discount).
        Sync should decrease H2's deposit by 15.
        """
        sys = create_system_with_deposits()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Sync from system
        _sync_trader_cash_from_system(subsystem, sys)

        trader_h2 = subsystem.traders.get("H2")
        assert trader_h2 is not None, "H2 should be a trader"
        assert trader_h2.cash == Decimal(100)

        # Simulate trade: H2 loses 15
        trader_h2.cash -= Decimal(15)

        deposit_before = _get_deposit_amount(sys, "H2")

        # Sync back
        _sync_trader_cash_to_system(subsystem, sys)

        deposit_after = _get_deposit_amount(sys, "H2")
        assert deposit_after == deposit_before - 15, (
            f"H2 deposit should decrease by 15: before={deposit_before}, after={deposit_after}"
        )

    def test_no_deposit_falls_back_to_cash(self):
        """Without deposits, sync should fall back to mint/retire CB cash.

        This verifies the pure-cash path still works for agents that have
        no bank deposits.
        """
        # Create system with cash only (no deposits)
        sys = System()
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        h3 = Household(id="H3", name="H3", kind="household")
        sys.add_agent(cb)
        sys.add_agent(bank)
        sys.add_agent(h1)
        sys.add_agent(h2)
        sys.add_agent(h3)
        sys.mint_cash("H1", 100)
        sys.mint_cash("H2", 100)
        sys.mint_cash("H3", 100)

        # Add payables so dealer subsystem has traders
        p1 = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=2,
        )
        sys.add_contract(p1)

        p2 = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=30,
            denom="X",
            asset_holder_id="H3",
            liability_issuer_id="H2",
            due_day=5,
        )
        sys.add_contract(p2)

        p3 = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=20,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H3",
            due_day=10,
        )
        sys.add_contract(p3)

        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Sync from system
        _sync_trader_cash_from_system(subsystem, sys)

        # Simulate trade: H1 gains 30
        trader_h1 = subsystem.traders.get("H1")
        assert trader_h1 is not None
        trader_h1.cash += Decimal(30)

        cash_before = _get_cash_amount(sys, "H1")

        # Sync back: should mint 30 CB cash (no deposits to adjust)
        _sync_trader_cash_to_system(subsystem, sys)

        cash_after = _get_cash_amount(sys, "H1")
        assert cash_after == cash_before + 30, (
            f"H1 cash should increase by 30 via mint: before={cash_before}, after={cash_after}"
        )

        # Verify no deposits were created
        deposit_after = _get_deposit_amount(sys, "H1")
        assert deposit_after == 0, (
            f"H1 should have no deposits in pure-cash mode, got {deposit_after}"
        )


class TestExternalCashChangePreservedInBankingMode:
    """Verify that externally-added deposits survive the dealer sync cycle.

    This is the banking-mode equivalent of test_disabled_subsystem_preserves_cash
    from test_dealer_sync_cash.py. Instead of minting cash externally, we
    increase a deposit amount externally (simulating lending proceeds).
    """

    def test_external_deposit_increase_preserved(self):
        """Externally increased deposits must not be burned by dealer sync.

        1. Initialize dealer subsystem (trader.cash set from system).
        2. Externally increase H1's deposit by 200 (simulating banking-mode lending).
        3. Disable subsystem.
        4. Run trading phase + sync.
        5. H1's deposit must still reflect the 200 increase.
        """
        sys = create_system_with_deposits()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Externally increase H1's deposit (simulating a loan disbursement)
        dep_h1 = sys.state.contracts.get("DEP_H1")
        assert dep_h1 is not None and dep_h1.kind == InstrumentKind.BANK_DEPOSIT
        dep_h1.amount += 200  # H1 now has 300 in deposit

        # Verify H1 total is now 300
        assert _get_agent_cash(sys, "H1") == Decimal(300)

        # Disable subsystem
        subsystem.enabled = False

        # Record state before cycle
        total_before = {tid: _get_agent_cash(sys, tid) for tid in TRADER_IDS}

        # Run full cycle
        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        sync_dealer_to_system(subsystem, sys)

        assert events == []

        # All balances must be preserved
        for tid in TRADER_IDS:
            total_after = _get_agent_cash(sys, tid)
            assert total_after == total_before[tid], (
                f"Agent {tid} balance changed from {total_before[tid]} to {total_after} "
                f"after disabled-subsystem cycle. External deposit change was likely lost."
            )
