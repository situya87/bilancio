"""
Tests for dealer subsystem integration with main simulation engine.

This module tests the bridge between the main bilancio system (Payables)
and the dealer module (Tickets) through the DealerSubsystem wrapper.
"""

from decimal import Decimal

from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import (
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from bilancio.engines.system import System


def test_dealer_subsystem_initialization():
    """Test that dealer subsystem initializes correctly from system state."""
    # Setup system with payables
    system = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    hh1 = Household(id="HH1", name="Household 1", kind="household")
    hh2 = Household(id="HH2", name="Household 2", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Create payables
    p1 = Payable(
        id="P1",
        kind="payable",
        amount=100,
        denom="USD",
        asset_holder_id="HH1",
        liability_issuer_id="HH2",
        due_day=5,
    )
    p2 = Payable(
        id="P2",
        kind="payable",
        amount=200,
        denom="USD",
        asset_holder_id="HH2",
        liability_issuer_id="HH1",
        due_day=10,
    )
    system.add_contract(p1)
    system.add_contract(p2)

    # Initialize dealer subsystem
    config = DealerRingConfig(
        ticket_size=Decimal(1), dealer_share=Decimal("0.25"), vbt_share=Decimal("0.50"), seed=42
    )
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Verify subsystem state
    assert len(subsystem.tickets) == 2, "Should have 2 tickets"
    assert len(subsystem.dealers) == 3, "Should have 3 dealer buckets (short/mid/long)"
    assert len(subsystem.vbts) == 3, "Should have 3 VBT buckets"
    assert len(subsystem.traders) == 2, "Should have 2 traders (HH1, HH2)"

    # Verify ticket-payable mapping
    assert len(subsystem.ticket_to_payable) == 2
    assert len(subsystem.payable_to_ticket) == 2
    assert "P1" in subsystem.payable_to_ticket
    assert "P2" in subsystem.payable_to_ticket

    # Verify ticket properties
    for ticket in subsystem.tickets.values():
        assert ticket.face > 0
        assert ticket.bucket_id in ["short", "mid", "long"]
        assert ticket.remaining_tau >= 0


def test_ticket_bucket_assignment():
    """Test that tickets are assigned to correct maturity buckets."""
    system = System()
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    hh1 = Household(id="HH1", name="HH1", kind="household")
    hh2 = Household(id="HH2", name="HH2", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Create payables with different maturities
    # short: tau in {1,2,3}
    # mid: tau in {4,...,8}
    # long: tau >= 9
    payables = [
        ("P1", 2, "short"),  # tau=2 -> short
        ("P2", 5, "mid"),  # tau=5 -> mid
        ("P3", 10, "long"),  # tau=10 -> long
    ]

    for pid, due_day, _expected_bucket in payables:
        p = Payable(
            id=pid,
            kind="payable",
            amount=100,
            denom="USD",
            asset_holder_id="HH1",
            liability_issuer_id="HH2",
            due_day=due_day,
        )
        system.add_contract(p)

    # Initialize subsystem
    config = DealerRingConfig(seed=42)
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Verify bucket assignments
    for pid, due_day, expected_bucket in payables:
        ticket_id = subsystem.payable_to_ticket[pid]
        ticket = subsystem.tickets[ticket_id]
        assert ticket.bucket_id == expected_bucket, (
            f"Payable {pid} with due_day={due_day} should be in {expected_bucket} bucket"
        )


def test_run_dealer_trading_phase():
    """Test that trading phase executes without errors."""
    system = System()
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    hh1 = Household(id="HH1", name="HH1", kind="household")
    hh2 = Household(id="HH2", name="HH2", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Create payables
    p1 = Payable(
        id="P1",
        kind="payable",
        amount=100,
        denom="USD",
        asset_holder_id="HH1",
        liability_issuer_id="HH2",
        due_day=5,
    )
    system.add_contract(p1)

    # Initialize subsystem
    config = DealerRingConfig(seed=42)
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Run trading phase
    events = run_dealer_trading_phase(subsystem, system, current_day=0)

    # Should execute without errors (may have 0 events if no eligible traders)
    assert isinstance(events, list)
    for event in events:
        assert "kind" in event
        assert event["kind"] == "dealer_trade"
        assert "trader" in event
        assert "side" in event
        assert "price" in event


def test_subsystem_enabled_flag():
    """Test that subsystem respects enabled flag."""
    system = System()
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    hh1 = Household(id="HH1", name="HH1", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)

    # Initialize subsystem
    config = DealerRingConfig(seed=42)
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Disable subsystem
    subsystem.enabled = False

    # Run trading phase - should return empty list
    events = run_dealer_trading_phase(subsystem, system, current_day=0)
    assert events == [], "Disabled subsystem should generate no events"

    # Re-enable
    subsystem.enabled = True
    events = run_dealer_trading_phase(subsystem, system, current_day=0)
    # Should work normally (may still be empty if no eligible trades)
    assert isinstance(events, list)


def test_sync_dealer_to_system():
    """Test that sync_dealer_to_system updates payable ownership."""
    system = System()
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    hh1 = Household(id="HH1", name="HH1", kind="household")
    hh2 = Household(id="HH2", name="HH2", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Create payable
    p1 = Payable(
        id="P1",
        kind="payable",
        amount=100,
        denom="USD",
        asset_holder_id="HH1",
        liability_issuer_id="HH2",
        due_day=5,
    )
    system.add_contract(p1)

    # Initialize subsystem
    config = DealerRingConfig(seed=42)
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Manually simulate a trade (change ticket ownership)
    ticket_id = subsystem.payable_to_ticket["P1"]
    ticket = subsystem.tickets[ticket_id]
    ticket.owner_id = "HH2"  # Simulate transfer

    # Sync back to system
    sync_dealer_to_system(subsystem, system)

    # Verify payable holder updated
    payable = system.state.contracts["P1"]
    assert payable.holder_id == "HH2", "Payable holder should be updated after sync"


def test_multiple_trading_phases():
    """Test running multiple trading phases in sequence."""
    system = System()
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    hh1 = Household(id="HH1", name="HH1", kind="household")
    hh2 = Household(id="HH2", name="HH2", kind="household")

    system.add_agent(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Create payables at different maturities
    p1 = Payable(
        id="P1",
        kind="payable",
        amount=100,
        denom="USD",
        asset_holder_id="HH1",
        liability_issuer_id="HH2",
        due_day=5,
    )
    p2 = Payable(
        id="P2",
        kind="payable",
        amount=200,
        denom="USD",
        asset_holder_id="HH2",
        liability_issuer_id="HH1",
        due_day=10,
    )
    system.add_contract(p1)
    system.add_contract(p2)

    # Initialize subsystem
    config = DealerRingConfig(seed=42)
    subsystem = initialize_dealer_subsystem(system, config, current_day=0)

    # Run multiple trading phases
    all_events = []
    for day in range(5):
        events = run_dealer_trading_phase(subsystem, system, current_day=day)
        all_events.extend(events)

    # Should execute without errors
    assert isinstance(all_events, list)

    # Verify tickets updated their remaining_tau
    for ticket in subsystem.tickets.values():
        # After processing day 4, tickets should have tau reduced
        assert ticket.remaining_tau == max(0, ticket.maturity_day - 4)
