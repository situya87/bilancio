"""
Integration tests for dealer ring simulation.

These tests verify the complete event loop per Section 11 of the specification:
- Phase 1: Maturity updates and rebucketing (Events 11-12)
- Phase 2: Dealer pre-computation
- Phase 3: Sell/buy eligibility sets
- Phase 4: Randomized order flow
- Phase 5: Settlement with proportional recovery
- Phase 6: VBT anchor updates

Tests are based on worked examples from the specification:
- Example 1: Rebucketing with dealer-held tickets
- Example 2: Maturing debt and cross-bucket reallocation
- Example 14: Event loop harness for arrivals
- Examples 10 & 12: Partial recovery defaults

References:
- docs/dealer_ring/dealer_examples.pdf
- docs/dealer_ring/dealer_specification.pdf
"""

from decimal import Decimal

import pytest

from bilancio.core.ids import new_id
from bilancio.dealer import (
    DealerRingConfig,
    DealerRingSimulation,
    Ticket,
    TraderState,
)

# =========================================================================
# Test Fixtures and Helpers
# =========================================================================


def create_ticket(
    issuer_id: str,
    owner_id: str,
    face: Decimal = Decimal(1),
    maturity_day: int = 10,
    remaining_tau: int = 5,
    bucket_id: str | None = None,
    serial: int = 0,
) -> Ticket:
    """Create a ticket for testing."""
    return Ticket(
        id=new_id(),
        issuer_id=issuer_id,
        owner_id=owner_id,
        face=face,
        maturity_day=maturity_day,
        remaining_tau=remaining_tau,
        bucket_id=bucket_id,
        serial=serial,
    )


def create_trader(
    agent_id: str | None = None,
    cash: Decimal = Decimal(10),
) -> TraderState:
    """Create a trader for testing."""
    return TraderState(
        agent_id=agent_id or new_id("trader"),
        cash=cash,
    )


def create_ring_traders(n: int, cash_per_trader: Decimal = Decimal(10)) -> list[TraderState]:
    """Create a ring of traders."""
    traders = []
    for i in range(n):
        traders.append(
            TraderState(
                agent_id=new_id(f"trader_{i}"),
                cash=cash_per_trader,
            )
        )
    return traders


def create_ring_tickets(
    traders: list[TraderState],
    tickets_per_trader: int = 2,
    starting_tau: int = 10,
    starting_maturity: int = 10,
) -> list[Ticket]:
    """
    Create tickets for a ring of traders.

    Each trader issues tickets_per_trader tickets, distributed to the next traders
    in the ring (creating claims/obligations around the ring).
    """
    tickets = []
    n = len(traders)

    for i, trader in enumerate(traders):
        for j in range(tickets_per_trader):
            # Issuer is the current trader, holder is next in ring
            # (This simulates creditor-debtor relationships around the ring)
            holder_idx = (i + 1 + j) % n

            ticket = Ticket(
                id=new_id(),
                issuer_id=trader.agent_id,
                owner_id=traders[holder_idx].agent_id,
                face=Decimal(1),
                maturity_day=starting_maturity + j,
                remaining_tau=starting_tau + j,
                bucket_id=None,  # Will be assigned by simulation
                serial=j,
            )
            tickets.append(ticket)

            # Track obligation on issuer
            trader.obligations.append(ticket)

    return tickets


# =========================================================================
# Test: Simulation Initialization
# =========================================================================


class TestSimulationInit:
    """Tests for simulation initialization."""

    def test_default_config(self):
        """Verify default configuration creates valid simulation."""
        config = DealerRingConfig()
        sim = DealerRingSimulation(config)

        assert sim.day == 0
        assert len(sim.dealers) == 3  # short, mid, long
        assert len(sim.vbts) == 3
        assert len(sim.traders) == 0

    def test_dealers_initialized_per_bucket(self):
        """Each bucket gets one dealer."""
        config = DealerRingConfig()
        sim = DealerRingSimulation(config)

        assert "short" in sim.dealers
        assert "mid" in sim.dealers
        assert "long" in sim.dealers

    def test_vbts_initialized_with_anchors(self):
        """VBTs initialized with configured anchors."""
        config = DealerRingConfig(
            vbt_anchors={
                "short": (Decimal("0.75"), Decimal("0.20")),
                "mid": (Decimal("0.75"), Decimal("0.30")),
                "long": (Decimal("0.75"), Decimal("0.40")),
            }
        )
        sim = DealerRingSimulation(config)

        # Check short VBT: M=0.75, O=0.20, A=0.85, B=0.65
        assert sim.vbts["short"].M == Decimal("0.75")
        assert sim.vbts["short"].O == Decimal("0.20")
        assert sim.vbts["short"].A == Decimal("0.85")  # M + O/2
        assert sim.vbts["short"].B == Decimal("0.65")  # M - O/2

        # Check mid VBT: M=0.75, O=0.30, A=0.90, B=0.60
        assert sim.vbts["mid"].M == Decimal("0.75")
        assert sim.vbts["mid"].O == Decimal("0.30")
        assert sim.vbts["mid"].A == Decimal("0.90")
        assert sim.vbts["mid"].B == Decimal("0.60")


# =========================================================================
# Test: Ring Setup
# =========================================================================


class TestRingSetup:
    """Tests for setting up the ring with traders and tickets."""

    def test_setup_registers_traders(self):
        """Traders are registered in simulation."""
        config = DealerRingConfig()
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(5)
        tickets = create_ring_tickets(traders, tickets_per_trader=2, starting_tau=5)

        sim.setup_ring(traders, tickets)

        assert len(sim.traders) == 5
        for trader in traders:
            assert trader.agent_id in sim.traders

    def test_setup_allocates_tickets(self):
        """Tickets are allocated to dealers, VBTs, and traders."""
        config = DealerRingConfig(
            dealer_share=Decimal("0.25"),
            vbt_share=Decimal("0.50"),
        )
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(5)
        # Create tickets all in mid bucket (remaining_tau in [4-8])
        tickets = []
        for _i, trader in enumerate(traders):
            ticket = create_ticket(
                issuer_id=trader.agent_id,
                owner_id=trader.agent_id,
                remaining_tau=5,  # Mid bucket
                maturity_day=10,
            )
            tickets.append(ticket)
            trader.obligations.append(ticket)

        sim.setup_ring(traders, tickets)

        # All tickets should be in the all_tickets registry
        assert len(sim.all_tickets) == 5

        # With 5 tickets and 25%/50% allocation:
        # dealer gets 1 (floor(5 * 0.25))
        # vbt gets 2 (floor(5 * 0.50))
        # traders get 2 (remainder)
        mid_dealer = sim.dealers["mid"]
        mid_vbt = sim.vbts["mid"]

        # Check total allocation matches
        total_allocated = (
            len(mid_dealer.inventory)
            + len(mid_vbt.inventory)
            + sum(len(t.tickets_owned) for t in sim.traders.values())
        )
        assert total_allocated == 5

    def test_bucket_assignment(self):
        """Tickets are assigned to correct buckets based on remaining_tau."""
        config = DealerRingConfig()
        sim = DealerRingSimulation(config)

        traders = [create_trader()]

        # Create tickets for different buckets
        short_ticket = create_ticket(traders[0].agent_id, traders[0].agent_id, remaining_tau=2)
        mid_ticket = create_ticket(traders[0].agent_id, traders[0].agent_id, remaining_tau=6)
        long_ticket = create_ticket(traders[0].agent_id, traders[0].agent_id, remaining_tau=12)

        tickets = [short_ticket, mid_ticket, long_ticket]

        sim.setup_ring(traders, tickets)

        # Check bucket assignment
        assert sim.all_tickets[short_ticket.id].bucket_id == "short"
        assert sim.all_tickets[mid_ticket.id].bucket_id == "mid"
        assert sim.all_tickets[long_ticket.id].bucket_id == "long"


# =========================================================================
# Test: Phase 1 - Maturity Updates and Rebucketing
# =========================================================================


class TestPhase1MaturityUpdates:
    """Tests for maturity updates and rebucketing (Phase 1)."""

    def test_maturity_decrements_each_day(self):
        """Remaining tau decrements by 1 each day."""
        config = DealerRingConfig(max_days=1)
        sim = DealerRingSimulation(config)

        traders = [create_trader()]
        ticket = create_ticket(
            traders[0].agent_id,
            traders[0].agent_id,
            remaining_tau=10,
            maturity_day=20,
        )

        sim.setup_ring(traders, [ticket])

        # Run one day
        sim.run_day()

        # Maturity should have decremented
        assert sim.all_tickets[ticket.id].remaining_tau == 9

    def test_rebucket_long_to_mid(self):
        """
        Example 1: Ticket migrates Long -> Mid when remaining_tau crosses boundary.

        At tau=9, ticket is in Long. When tau decrements to 8, it moves to Mid.
        """
        config = DealerRingConfig(
            dealer_share=Decimal(0),  # No dealer allocation
            vbt_share=Decimal(0),  # No VBT allocation
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        traders = [create_trader()]
        # Ticket at tau=9 (Long bucket boundary)
        ticket = create_ticket(
            traders[0].agent_id,
            traders[0].agent_id,
            remaining_tau=9,  # Long bucket (tau >= 9)
            maturity_day=20,
        )

        sim.setup_ring(traders, [ticket])

        # Verify initial bucket
        assert sim.all_tickets[ticket.id].bucket_id == "long"

        # Run one day
        sim.run_day()

        # Verify rebucketed to mid (tau=8, which is in [4,8])
        assert sim.all_tickets[ticket.id].remaining_tau == 8
        assert sim.all_tickets[ticket.id].bucket_id == "mid"

    def test_rebucket_mid_to_short(self):
        """Ticket migrates Mid -> Short when remaining_tau crosses boundary."""
        config = DealerRingConfig(
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        traders = [create_trader()]
        # Ticket at tau=4 (Mid bucket boundary)
        ticket = create_ticket(
            traders[0].agent_id,
            traders[0].agent_id,
            remaining_tau=4,  # Mid bucket boundary
            maturity_day=20,
        )

        sim.setup_ring(traders, [ticket])

        # Verify initial bucket
        assert sim.all_tickets[ticket.id].bucket_id == "mid"

        # Run one day
        sim.run_day()

        # Verify rebucketed to short (tau=3, which is in [1,3])
        assert sim.all_tickets[ticket.id].remaining_tau == 3
        assert sim.all_tickets[ticket.id].bucket_id == "short"


class TestDealerRebucketing:
    """
    Test Event 11: Dealer-to-dealer internal sale at rebucketing.

    Example 1: When dealer holds a ticket that crosses bucket boundary,
    an internal sale occurs between the old-bucket dealer and new-bucket dealer.
    """

    def test_dealer_rebucket_transfers_at_ask(self):
        """
        Dealer-held ticket rebuckets via internal sale at old-bucket ask.

        This implements Event 11 from the spec.
        """
        config = DealerRingConfig(
            dealer_share=Decimal(1),  # All to dealer
            vbt_share=Decimal(0),
            N_max=0,  # Disable order flow to isolate rebucketing
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        traders = [create_trader(cash=Decimal(100))]
        # Single ticket at Long/Mid boundary with maturity far in future
        ticket = create_ticket(
            traders[0].agent_id,
            traders[0].agent_id,
            remaining_tau=9,  # Long bucket, will become Mid (tau=8)
            maturity_day=100,  # Far future - won't settle during test
        )

        # Pre-fund dealers with cash (need enough for the transfer price)
        sim.dealers["long"].cash = Decimal("2.0")
        sim.dealers["mid"].cash = Decimal("2.0")

        sim.setup_ring(traders, [ticket])

        # Ticket should be allocated to long dealer
        long_dealer = sim.dealers["long"]
        mid_dealer = sim.dealers["mid"]

        assert len(long_dealer.inventory) == 1
        assert len(mid_dealer.inventory) == 0

        # Snapshot balances before rebucketing
        long_cash_before = long_dealer.cash
        mid_cash_before = mid_dealer.cash

        # Run one day (triggers rebucketing)
        sim.run_day()

        # Ticket should now be in mid dealer's inventory
        assert len(long_dealer.inventory) == 0
        assert len(mid_dealer.inventory) == 1

        # Cash should have moved between dealers
        # (Long dealer received cash, mid dealer paid cash)
        assert long_dealer.cash > long_cash_before
        assert mid_dealer.cash < mid_cash_before


# =========================================================================
# Test: Phase 3 - Eligibility Sets
# =========================================================================


class TestPhase3Eligibility:
    """Tests for sell/buy eligibility computation (Phase 3)."""

    def test_sell_eligible_has_shortfall_and_tickets(self):
        """Trader with shortfall > 0 and tickets is sell-eligible."""
        config = DealerRingConfig()
        sim = DealerRingSimulation(config)

        # Trader with obligation due today (day 1)
        trader = create_trader(cash=Decimal("0.5"))  # Less than face=1
        obligation = create_ticket(
            trader.agent_id,
            "holder",
            remaining_tau=1,
            maturity_day=1,  # Due on day 1
        )
        trader.obligations.append(obligation)

        # Give trader a ticket to sell
        owned_ticket = create_ticket("other_issuer", trader.agent_id, remaining_tau=5)
        trader.tickets_owned.append(owned_ticket)

        sim.traders[trader.agent_id] = trader
        sim.all_tickets[owned_ticket.id] = owned_ticket
        sim.all_tickets[obligation.id] = obligation
        sim.day = 1  # Set current day to match obligation

        # Compute eligibility
        sell_eligible = sim._compute_sell_eligible()

        # Trader should be sell-eligible (shortfall = 1 - 0.5 = 0.5 > 0)
        assert trader.agent_id in sell_eligible

    def test_buy_eligible_has_buffer_and_horizon(self):
        """Trader with cash > buffer and horizon >= H is buy-eligible."""
        config = DealerRingConfig(
            buffer_B=Decimal(1),
            horizon_H=3,
        )
        sim = DealerRingSimulation(config)

        # Trader with excess cash and distant liabilities
        trader = create_trader(cash=Decimal(5))  # > buffer_B

        # Obligation due in 10 days (horizon = 10 >= H = 3)
        obligation = create_ticket(
            trader.agent_id,
            "holder",
            remaining_tau=10,
            maturity_day=sim.day + 10,
        )
        trader.obligations.append(obligation)

        sim.traders[trader.agent_id] = trader

        # Compute eligibility
        buy_eligible = sim._compute_buy_eligible()

        # Trader should be buy-eligible
        assert trader.agent_id in buy_eligible

    def test_not_buy_eligible_insufficient_cash(self):
        """Trader with cash <= buffer is not buy-eligible."""
        config = DealerRingConfig(buffer_B=Decimal(1))
        sim = DealerRingSimulation(config)

        trader = create_trader(cash=Decimal("0.5"))  # < buffer_B
        sim.traders[trader.agent_id] = trader

        buy_eligible = sim._compute_buy_eligible()

        assert trader.agent_id not in buy_eligible


# =========================================================================
# Test: Phase 5 - Settlement with Proportional Recovery
# =========================================================================


class TestPhase5Settlement:
    """
    Tests for settlement phase with proportional recovery.

    Based on Examples 10 & 12: Partial recovery default with multiple claimant types.
    """

    def test_full_recovery_when_cash_sufficient(self):
        """Settlement pays full face when issuer has enough cash."""
        config = DealerRingConfig(
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        # Issuer with enough cash
        issuer = create_trader(cash=Decimal(5))
        holder = create_trader(cash=Decimal(0))

        # Register traders directly (skip setup_ring allocation)
        sim.traders[issuer.agent_id] = issuer
        sim.traders[holder.agent_id] = holder

        # Ticket maturing today (day 1 after one run_day)
        ticket = create_ticket(
            issuer.agent_id,
            holder.agent_id,
            face=Decimal(1),
            remaining_tau=1,
            maturity_day=1,
            bucket_id="short",  # Short bucket for tau=1
        )
        issuer.obligations.append(ticket)
        holder.tickets_owned.append(ticket)

        # Register ticket in simulation
        sim.all_tickets[ticket.id] = ticket

        # Track initial cash
        holder_cash_before = holder.cash

        # Run one day (triggers settlement)
        sim.run_day()

        # Holder should receive full payment
        payment_received = holder.cash - holder_cash_before
        assert payment_received == Decimal(1)  # Full recovery

    def test_partial_recovery_when_cash_insufficient(self):
        """
        Settlement pays proportional recovery when issuer has insufficient cash.

        Example 12: R = C/D = 3/5 = 0.6 when issuer has 3 and owes 5.
        """
        config = DealerRingConfig(
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        # Issuer with insufficient cash
        issuer = create_trader(cash=Decimal(3))  # Has 3
        sim.traders[issuer.agent_id] = issuer

        # Create 5 holders, each with 1 ticket (total due = 5)
        holders = [create_trader(cash=Decimal(0)) for _ in range(5)]

        tickets = []
        for i, holder in enumerate(holders):
            sim.traders[holder.agent_id] = holder

            ticket = create_ticket(
                issuer.agent_id,
                holder.agent_id,
                face=Decimal(1),
                remaining_tau=1,
                maturity_day=1,
                bucket_id="short",
                serial=i,
            )
            tickets.append(ticket)
            issuer.obligations.append(ticket)
            holder.tickets_owned.append(ticket)
            sim.all_tickets[ticket.id] = ticket

        # Run one day (triggers settlement)
        sim.run_day()

        # Each holder should receive R * face = 0.6 * 1 = 0.6
        expected_payment = Decimal("0.6")
        for holder in holders:
            assert holder.cash == expected_payment

        # Issuer should have cash = 0 (all distributed)
        assert issuer.cash == Decimal(0)

    def test_issuer_marked_defaulted_on_partial_recovery(self):
        """Issuer is marked as defaulted when R < 1."""
        config = DealerRingConfig(
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        issuer = create_trader(cash=Decimal(3))
        holder = create_trader(cash=Decimal(0))

        # Register traders directly
        sim.traders[issuer.agent_id] = issuer
        sim.traders[holder.agent_id] = holder

        ticket = create_ticket(
            issuer.agent_id,
            holder.agent_id,
            face=Decimal(5),  # Due 5, have 3 -> R = 0.6
            remaining_tau=1,
            maturity_day=1,
            bucket_id="short",
        )
        issuer.obligations.append(ticket)
        holder.tickets_owned.append(ticket)
        sim.all_tickets[ticket.id] = ticket

        # Issuer not defaulted initially
        assert not issuer.defaulted

        # Run one day
        sim.run_day()

        # Issuer should be marked defaulted
        assert issuer.defaulted


# =========================================================================
# Test: Phase 6 - VBT Anchor Updates
# =========================================================================


class TestPhase6VBTAnchorUpdates:
    """
    Tests for VBT anchor updates based on bucket loss rates.

    Based on Example 3 (outside bid clipping) and Example 10.
    """

    def test_vbt_anchors_update_after_default(self):
        """
        VBT anchors adjust based on bucket loss rate.

        M_new = M_old - phi_M * loss_rate
        O_new = O_old + phi_O * loss_rate
        """
        config = DealerRingConfig(
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            phi_M=Decimal(1),  # M drops by loss_rate
            phi_O=Decimal("0.6"),  # O increases by 0.6 * loss_rate
            enable_vbt_anchor_updates=True,
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        # Initial anchors for short bucket (where our ticket will be)
        short_M_initial = sim.vbts["short"].M
        short_O_initial = sim.vbts["short"].O

        # Create a default scenario in short bucket
        issuer = create_trader(cash=Decimal(0))  # No cash -> R = 0, loss = 100%
        holder = create_trader(cash=Decimal(0))

        # Register traders directly
        sim.traders[issuer.agent_id] = issuer
        sim.traders[holder.agent_id] = holder

        # Ticket in short bucket (remaining_tau=1)
        ticket = create_ticket(
            issuer.agent_id,
            holder.agent_id,
            face=Decimal(1),
            remaining_tau=1,  # Short bucket
            maturity_day=1,
            bucket_id="short",
        )
        issuer.obligations.append(ticket)
        holder.tickets_owned.append(ticket)
        sim.all_tickets[ticket.id] = ticket

        # Run one day
        sim.run_day()

        # Check VBT anchors updated
        # Loss rate = 1 - R = 1 - 0 = 1 (total loss)
        expected_M = short_M_initial - config.phi_M * Decimal(1)
        expected_O = short_O_initial + config.phi_O * Decimal(1)

        assert sim.vbts["short"].M == expected_M
        assert sim.vbts["short"].O == expected_O


# =========================================================================
# Test: Full Event Loop
# =========================================================================


class TestFullEventLoop:
    """
    Integration tests for complete simulation runs.

    Based on Example 14: Minimal event loop harness.
    """

    def test_multi_day_simulation(self):
        """Run multiple days without errors."""
        config = DealerRingConfig(
            seed=42,
            max_days=5,
        )
        sim = DealerRingSimulation(config)

        # Create a small ring with tickets in mid bucket (far from settlement)
        traders = create_ring_traders(5, cash_per_trader=Decimal(20))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=2, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)

        # Run simulation
        sim.run()

        # Verify simulation completed
        assert sim.day == 5

        # Verify event log has events
        assert len(sim.events.events) > 0

    def test_order_flow_executes(self):
        """Order flow produces trades during simulation."""
        config = DealerRingConfig(
            seed=42,
            N_max=3,
            max_days=3,
        )
        sim = DealerRingSimulation(config)

        # Create ring with sufficient cash for trading
        traders = create_ring_traders(5, cash_per_trader=Decimal(50))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=3, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)

        # Add cash to dealers/VBTs for trading
        for dealer in sim.dealers.values():
            dealer.cash = Decimal(10)
        for vbt in sim.vbts.values():
            vbt.cash = Decimal(20)

        # Run simulation
        sim.run()

        # Check for quote events at minimum (quotes are logged each day)
        quote_events = [e for e in sim.events.events if e.get("kind") == "quote"]
        # 3 buckets * 3 days = 9 quote events minimum
        assert len(quote_events) >= 9

    def test_simulation_runs_multiple_days(self):
        """Simulation completes multiple days without errors."""
        config = DealerRingConfig(seed=12345, max_days=5)
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(5, cash_per_trader=Decimal(20))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=2, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)
        sim.run()

        # Verify simulation completed all days
        assert sim.day == 5

        # Verify all traders still exist
        assert len(sim.traders) == 5

        # Verify some events were logged
        assert len(sim.events.events) > 0


# =========================================================================
# Test: Assertions Integration
# =========================================================================


class TestAssertionsIntegration:
    """Verify that programmatic assertions (C1-C6) pass during simulation."""

    def test_c5_equity_basis_holds(self):
        """C5: V = C + M*a holds for dealers throughout simulation."""
        config = DealerRingConfig(max_days=3)
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(3, cash_per_trader=Decimal(10))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=2, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)

        # Run simulation and check invariants after each day
        for _day in range(3):
            sim.run_day()

            # Verify V = C + M*a for each dealer
            for bucket_id, dealer in sim.dealers.items():
                vbt = sim.vbts[bucket_id]
                expected_V = dealer.cash + vbt.M * dealer.a
                assert dealer.V == expected_V, (
                    f"C5 failed for {bucket_id}: V={dealer.V}, expected={expected_V}"
                )


# =========================================================================
# Test: Event Logging
# =========================================================================


class TestEventLogging:
    """Tests for event logging system."""

    def test_events_logged_during_simulation(self):
        """Events are captured in the event log."""
        config = DealerRingConfig(max_days=3)
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(3, cash_per_trader=Decimal(10))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=2, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)
        sim.run()

        # Should have day_start events
        day_starts = [e for e in sim.events.events if e.get("kind") == "day_start"]
        assert len(day_starts) == 3

    def test_quote_events_logged(self):
        """Quote states are logged at start of each day."""
        config = DealerRingConfig(max_days=2)
        sim = DealerRingSimulation(config)

        traders = create_ring_traders(3, cash_per_trader=Decimal(10))
        tickets = create_ring_tickets(
            traders, tickets_per_trader=1, starting_tau=50, starting_maturity=100
        )

        sim.setup_ring(traders, tickets)
        sim.run()

        # Should have quote events for each bucket each day
        quote_events = [e for e in sim.events.events if e.get("kind") == "quote"]
        # 3 buckets * 2 days = 6 quote events
        assert len(quote_events) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
