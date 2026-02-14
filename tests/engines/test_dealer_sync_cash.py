"""
Tests for dealer cash sync round-trip correctness.

These tests prevent regression of the bug where _sync_trader_cash_to_system()
without a prior _sync_trader_cash_from_system() call erroneously burns or mints
cash. The root cause was that run_dealer_trading_phase() returned early when
subsystem.enabled=False WITHOUT calling _sync_trader_cash_from_system(), but
sync_dealer_to_system() still ran _sync_trader_cash_to_system() which found
stale trader.cash values and burned lending-phase cash increases.

The fix moved the _sync_trader_cash_from_system() call before the enabled check
in run_dealer_trading_phase(). These tests verify:
  1. A sync round-trip with no trades is a no-op.
  2. Skipping sync_from before sync_to produces incorrect results (documents bug).
  3. A disabled subsystem preserves cash after the full trading+sync cycle.
"""

import pytest
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import (
    DealerSubsystem,
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
    _get_agent_cash,
)
from bilancio.engines.dealer_sync import (
    _sync_trader_cash_from_system,
    _sync_trader_cash_to_system,
)
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.dealer.models import DEFAULT_BUCKETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_system_with_payables() -> System:
    """Create a minimal test system with agents, cash, and payables."""
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
    sys.mint_cash("H1", 100)
    sys.mint_cash("H2", 100)
    sys.mint_cash("H3", 100)

    # H1 owes H2, due in 2 days
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

    # H2 owes H3, due in 5 days
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

    # H3 owes H1, due in 10 days
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


def create_dealer_config() -> DealerRingConfig:
    """Create a standard dealer configuration for testing."""
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


TRADER_IDS = ["H1", "H2", "H3"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_round_trip_no_op():
    """Sync from-system then to-system with no trades must leave cash unchanged.

    After initialize_dealer_subsystem sets up trader.cash values, calling
    _sync_trader_cash_from_system (which refreshes trader.cash from the system)
    followed by _sync_trader_cash_to_system (which computes deltas and mints/burns)
    should be a perfect no-op: every delta is zero, so no minting or burning occurs.
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Record each trader's cash before the round-trip
    cash_before = {tid: _get_agent_cash(sys, tid) for tid in TRADER_IDS}

    # Round-trip: sync from system, then sync back
    _sync_trader_cash_from_system(subsystem, sys)
    _sync_trader_cash_to_system(subsystem, sys)

    # Each agent's cash must be exactly the same
    for tid in TRADER_IDS:
        cash_after = _get_agent_cash(sys, tid)
        assert cash_after == cash_before[tid], (
            f"Agent {tid} cash changed from {cash_before[tid]} to {cash_after} "
            f"during no-op sync round-trip"
        )


def test_sync_to_without_sync_from_detects_stale():
    """Skipping _sync_trader_cash_from_system causes incorrect cash changes.

    This test documents the exact bug pattern:
    1. Initialize subsystem (trader.cash is set from system cash).
    2. Externally add cash to an agent (simulating lending proceeds).
    3. Call _sync_trader_cash_to_system WITHOUT refreshing trader.cash first.
    4. The stale trader.cash is lower than the actual system cash, so the
       function computes a negative delta and burns the lending cash.

    The test asserts that total system cash CHANGED (decreased), proving the
    burn happened. This is the wrong behavior that the production fix prevents.
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Sync once so trader.cash matches system (baseline)
    _sync_trader_cash_from_system(subsystem, sys)

    # Simulate lending: H1 receives 200 additional cash from a loan
    sys.mint_cash("H1", 200)

    # Verify H1 now has 300 (100 original + 200 from lending)
    assert _get_agent_cash(sys, "H1") == Decimal(300)

    # Record total cash across all traders before the buggy sync
    total_before = sum(_get_agent_cash(sys, tid) for tid in TRADER_IDS)

    # BUG PATTERN: call sync_to WITHOUT calling sync_from first.
    # trader.cash for H1 is still 100 (stale), but system cash is 300.
    # Delta = 100 - 300 = -200, so the function burns 200 from H1.
    _sync_trader_cash_to_system(subsystem, sys)

    total_after = sum(_get_agent_cash(sys, tid) for tid in TRADER_IDS)

    # Total cash must have decreased — the 200 was erroneously burned
    assert total_after < total_before, (
        f"Expected total cash to decrease due to stale sync bug, "
        f"but total_before={total_before}, total_after={total_after}"
    )
    # Specifically, H1 should have lost the 200 that was minted
    assert _get_agent_cash(sys, "H1") == Decimal(100), (
        f"H1 should be back to 100 after the stale sync burned the 200, "
        f"got {_get_agent_cash(sys, 'H1')}"
    )


def test_disabled_subsystem_preserves_cash():
    """A disabled subsystem must not alter any agent's cash.

    The production fix ensures _sync_trader_cash_from_system() is called
    even when subsystem.enabled=False, so that the subsequent
    sync_dealer_to_system() -> _sync_trader_cash_to_system() round-trip
    sees current cash and computes delta=0.

    This test simulates the exact scenario: mint extra cash (lending),
    disable the subsystem, then run the full trading-phase + sync cycle.
    All agent cash balances must be preserved.
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Simulate lending: H1 receives 200 additional cash
    sys.mint_cash("H1", 200)

    # Disable the subsystem (NBFI-only mode, no dealer trading)
    subsystem.enabled = False

    # Record each agent's cash before the trading phase
    cash_before = {tid: _get_agent_cash(sys, tid) for tid in TRADER_IDS}

    # Verify H1 has 300 (100 original + 200 lending)
    assert cash_before["H1"] == Decimal(300)

    # Run the full cycle: trading phase + sync back to system
    events = run_dealer_trading_phase(subsystem, sys, current_day=0)
    sync_dealer_to_system(subsystem, sys)

    # No trades should have happened (subsystem was disabled)
    assert events == [], "Disabled subsystem should produce no trade events"

    # Every agent's cash must be exactly preserved
    for tid in TRADER_IDS:
        cash_after = _get_agent_cash(sys, tid)
        assert cash_after == cash_before[tid], (
            f"Agent {tid} cash changed from {cash_before[tid]} to {cash_after} "
            f"after disabled-subsystem trading phase + sync. "
            f"The lending cash was likely burned by stale sync."
        )
