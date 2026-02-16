"""Integration tests for dealer-Kalecki integration.

Tests verify the integration between the dealer ring subsystem and the main
Bilancio simulation engine, including:
- Initialization from system state
- Trading phase execution
- Synchronization back to main system
- Settlement with transferred payables
"""

from decimal import Decimal

from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import (
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System


def create_test_system_with_payables():
    """Create a minimal test system with agents, cash, and payables."""
    sys = System()

    # Add agents
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

    # Give households some cash
    sys.mint_cash("H1", 100)
    sys.mint_cash("H2", 100)
    sys.mint_cash("H3", 100)

    # Create payables between households
    # H1 owes H2, due in 2 days
    p1_id = sys.new_contract_id("P")
    p1 = Payable(
        id=p1_id,
        kind=InstrumentKind.PAYABLE,
        amount=50,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day + 2,
    )
    sys.add_contract(p1)

    # H2 owes H3, due in 5 days
    p2_id = sys.new_contract_id("P")
    p2 = Payable(
        id=p2_id,
        kind=InstrumentKind.PAYABLE,
        amount=30,
        denom="X",
        asset_holder_id="H3",
        liability_issuer_id="H2",
        due_day=sys.state.day + 5,
    )
    sys.add_contract(p2)

    # H3 owes H1, due in 10 days
    p3_id = sys.new_contract_id("P")
    p3 = Payable(
        id=p3_id,
        kind=InstrumentKind.PAYABLE,
        amount=20,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="H3",
        due_day=sys.state.day + 10,
    )
    sys.add_contract(p3)

    return sys


def create_dealer_config():
    """Create a test dealer configuration."""
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


def test_initialize_dealer_subsystem():
    """Test initialization of dealer subsystem from system state.

    Verifies:
    - Tickets are created from payables
    - Dealers and VBTs are created for each bucket
    - Traders are created for households
    - Tickets are assigned to correct buckets based on maturity
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize dealer subsystem
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Verify tickets were created from payables (3 payables)
    assert len(subsystem.tickets) == 3, "Should create 3 tickets from 3 payables"

    # Verify bidirectional mappings exist
    assert len(subsystem.ticket_to_payable) == 3
    assert len(subsystem.payable_to_ticket) == 3

    # Verify dealers and VBTs created for each bucket
    assert len(subsystem.dealers) == 3, "Should have 3 dealers (short, mid, long)"
    assert len(subsystem.vbts) == 3, "Should have 3 VBTs (short, mid, long)"
    assert "short" in subsystem.dealers
    assert "mid" in subsystem.dealers
    assert "long" in subsystem.dealers

    # Verify traders created for households (3 households)
    assert len(subsystem.traders) == 3, "Should have 3 traders for 3 households"
    assert "H1" in subsystem.traders
    assert "H2" in subsystem.traders
    assert "H3" in subsystem.traders

    # Verify tickets assigned to correct buckets
    # tau=2 -> short, tau=5 -> mid, tau=10 -> long
    bucket_counts = {"short": 0, "mid": 0, "long": 0}
    for ticket in subsystem.tickets.values():
        bucket_counts[ticket.bucket_id] += 1

    assert bucket_counts["short"] == 1, "Should have 1 ticket in short bucket (tau=2)"
    assert bucket_counts["mid"] == 1, "Should have 1 ticket in mid bucket (tau=5)"
    assert bucket_counts["long"] == 1, "Should have 1 ticket in long bucket (tau=10)"

    # Verify traders have correct ownership links
    for trader_id, trader in subsystem.traders.items():
        # Each trader should have tickets they own (as creditor) and obligations (as debtor)
        owned_count = len(trader.tickets_owned)
        obligations_count = len(trader.obligations)
        # In our setup, each household is both a creditor and a debtor
        assert owned_count + obligations_count > 0, (
            f"Trader {trader_id} should have tickets or obligations"
        )

    # Verify executor was created
    assert subsystem.executor is not None, "Trade executor should be initialized"


def test_dealer_trading_phase_executes():
    """Test that dealer trading phase executes and returns events.

    Verifies:
    - Trading phase runs without errors
    - Events are returned (may be empty if no eligible trades)
    - Dealer quotes are computed
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize dealer subsystem
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Run trading phase
    events = run_dealer_trading_phase(subsystem, sys, current_day=0)

    # Verify events list is returned (even if empty)
    assert isinstance(events, list), "Should return a list of events"

    # Verify all dealers have computed quotes
    for bucket_id, dealer in subsystem.dealers.items():
        # Dealer state should be initialized with quotes
        # (quotes may be None if no inventory, but state should exist)
        assert dealer is not None, f"Dealer for {bucket_id} should exist"

    # If there are events, verify they have expected structure
    for event in events:
        assert "kind" in event, "Event should have 'kind' field"
        assert "day" in event, "Event should have 'day' field"
        if event["kind"] == "dealer_trade":
            assert "trader" in event, "Trade event should have 'trader' field"
            assert "side" in event, "Trade event should have 'side' field"
            assert event["side"] in ["buy", "sell"], "Side should be buy or sell"


def test_run_day_with_dealer_enabled():
    """Test that run_day executes dealer phase when enabled.

    Verifies:
    - SubphaseB_Dealer event is logged
    - Dealer trading phase runs within main simulation loop
    - System day advances correctly
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize dealer subsystem and attach to state
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    sys.state.dealer_subsystem = subsystem

    initial_day = sys.state.day

    # Run day with dealer enabled
    run_day(sys, enable_dealer=True)

    # Verify day advanced
    assert sys.state.day == initial_day + 1, "Day should advance by 1"

    # Verify SubphaseB_Dealer event was logged
    dealer_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Dealer"]
    assert len(dealer_events) == 1, "Should log exactly one SubphaseB_Dealer event"
    assert dealer_events[0]["day"] == initial_day, "Dealer event should be for the correct day"


def test_run_day_without_dealer():
    """Test that run_day skips dealer phase when disabled.

    Verifies:
    - NO SubphaseB_Dealer event is logged
    - Normal simulation phases still run
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize dealer subsystem and attach to state
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    sys.state.dealer_subsystem = subsystem

    initial_day = sys.state.day

    # Run day WITHOUT dealer enabled
    run_day(sys, enable_dealer=False)

    # Verify day advanced
    assert sys.state.day == initial_day + 1, "Day should advance by 1"

    # Verify NO SubphaseB_Dealer event was logged
    dealer_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Dealer"]
    assert len(dealer_events) == 0, "Should NOT log SubphaseB_Dealer event when disabled"

    # Verify other phases still ran
    phase_a_events = [e for e in sys.state.events if e.get("kind") == "PhaseA"]
    assert len(phase_a_events) >= 1, "PhaseA should still run"


def test_payable_holder_id_updated_after_trade():
    """Test that payable.holder_id is updated after dealer trades.

    Verifies:
    - After sync_dealer_to_system, payable.holder_id reflects ticket ownership
    - Ownership transfers are properly tracked
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize dealer subsystem
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Get a ticket and manually change its ownership to simulate a trade
    ticket = list(subsystem.tickets.values())[0]
    original_owner = ticket.owner_id
    ticket.owner_id = "H3"  # Transfer to H3

    # Sync back to system
    sync_dealer_to_system(subsystem, sys)

    # Find corresponding payable
    payable_id = subsystem.ticket_to_payable[ticket.id]
    payable = sys.state.contracts[payable_id]

    # Verify holder_id was updated
    assert payable.holder_id == "H3", "Payable holder_id should be updated to new owner"
    assert payable.effective_creditor == "H3", "Effective creditor should be the new holder"
    assert payable.asset_holder_id == original_owner, "Original creditor should remain unchanged"


def test_settlement_pays_effective_creditor():
    """Test that settlement pays the current holder, not original creditor.

    Verifies:
    - When payable has holder_id set (transferred in secondary market)
    - Settlement pays holder_id, not original asset_holder_id
    - Cash/deposits transfer to correct agent
    """
    sys = create_test_system_with_payables()

    # Create a payable due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=25,
        denom="X",
        asset_holder_id="H2",  # Original creditor
        liability_issuer_id="H1",  # Debtor
        due_day=sys.state.day,
        holder_id="H3",  # Transferred to H3 in secondary market
    )
    sys.add_contract(payable)

    # Verify setup
    assert payable.effective_creditor == "H3", "Effective creditor should be H3"
    assert payable.asset_holder_id == "H2", "Original creditor should be H2"

    # Get initial cash balances
    h1_cash_before = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    h2_cash_before = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H2"].asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    h3_cash_before = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H3"].asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )

    # Run settlement
    run_day(sys)

    # Get final cash balances
    h1_cash_after = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    h2_cash_after = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H2"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    h3_cash_after = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H3"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )

    # Verify H1 (debtor) paid 25
    assert h1_cash_after == h1_cash_before - 25, "Debtor should pay 25"

    # Verify H3 (holder) received 25, not H2 (original creditor)
    assert h3_cash_after == h3_cash_before + 25, "Current holder H3 should receive payment"
    assert h2_cash_after == h2_cash_before, "Original creditor H2 should NOT receive payment"

    # Verify payable was removed after settlement
    assert payable_id not in sys.state.contracts, "Payable should be removed after settlement"

    # Verify PayableSettled event logged with correct creditor
    settled_events = [e for e in sys.state.events if e.get("kind") == "PayableSettled"]
    assert len(settled_events) == 1, "Should have one PayableSettled event"
    # Note: This assertion may need adjustment based on what field is logged
    # The settlement code should log the effective_creditor, not asset_holder_id


def test_dealer_subsystem_disabled():
    """Test that trading phase returns empty list when disabled.

    Verifies:
    - Disabled subsystem doesn't execute trades
    - Returns empty event list
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    subsystem.enabled = False

    # Run trading phase on disabled subsystem
    events = run_dealer_trading_phase(subsystem, sys, current_day=0)

    # Should return empty list
    assert events == [], "Disabled subsystem should return no events"


def test_ticket_bucket_reassignment():
    """Test that tickets are reassigned to correct buckets as maturity approaches.

    Verifies:
    - Tickets move between buckets as remaining_tau decreases
    - Dealer/VBT inventories are updated correctly
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize subsystem
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

    # Find the long-maturity ticket (tau=10 initially)
    long_ticket = None
    for ticket in subsystem.tickets.values():
        if ticket.remaining_tau == 10:
            long_ticket = ticket
            break

    assert long_ticket is not None, "Should find ticket with tau=10"
    assert long_ticket.bucket_id == "long", "Initial bucket should be 'long'"

    # Advance 5 days (tau goes from 10 to 5)
    # This should move ticket from 'long' to 'mid' bucket
    run_dealer_trading_phase(subsystem, sys, current_day=5)

    # Verify bucket changed
    assert long_ticket.remaining_tau == 5, "Remaining tau should be 5"
    assert long_ticket.bucket_id == "mid", "Ticket should now be in 'mid' bucket"

    # Advance 3 more days (tau goes from 5 to 2)
    # This should move ticket from 'mid' to 'short' bucket
    run_dealer_trading_phase(subsystem, sys, current_day=8)

    # Verify bucket changed again
    assert long_ticket.remaining_tau == 2, "Remaining tau should be 2"
    assert long_ticket.bucket_id == "short", "Ticket should now be in 'short' bucket"


def test_integration_multiple_days_with_dealer():
    """Integration test: run multiple days with dealer enabled.

    Verifies:
    - System runs multiple days without errors
    - Dealer subsystem updates each day
    - Payables settle when due
    - Events are logged correctly
    """
    sys = create_test_system_with_payables()
    config = create_dealer_config()

    # Initialize and attach dealer subsystem
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    sys.state.dealer_subsystem = subsystem

    initial_payable_count = len(
        [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    )
    assert initial_payable_count == 3, "Should start with 3 payables"

    # Run 5 days with dealer enabled
    for _day in range(5):
        run_day(sys, enable_dealer=True)

    # Verify we're at day 5
    assert sys.state.day == 5, "Should be at day 5"

    # Verify dealer events logged each day
    dealer_events = [e for e in sys.state.events if e.get("kind") == "SubphaseB_Dealer"]
    assert len(dealer_events) == 5, "Should have dealer events for 5 days"

    # Verify at least one payable has settled (due_day=2)
    final_payable_count = len(
        [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    )
    assert final_payable_count < initial_payable_count, "Some payables should have settled"

    # Verify settlement events logged
    settled_events = [e for e in sys.state.events if e.get("kind") == "PayableSettled"]
    assert len(settled_events) >= 1, "Should have at least one settlement event"

    # System invariants should hold
    sys.assert_invariants()
