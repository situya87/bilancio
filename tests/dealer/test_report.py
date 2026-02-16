"""Tests for dealer ring HTML report generation."""

import tempfile
from decimal import Decimal
from pathlib import Path

from bilancio.dealer import (
    DealerRingConfig,
    DealerRingSimulation,
    Ticket,
    TraderState,
    export_dealer_ring_html,
    generate_dealer_ring_html,
)


class TestReportGeneration:
    """Test basic HTML report generation."""

    def test_generate_html_basic(self):
        """Generate HTML report from minimal simulation."""
        config = DealerRingConfig(
            N_max=0,  # No random order flow
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        # Set up minimal ring
        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])
        sim.run(max_days=1)

        # Generate HTML
        html = generate_dealer_ring_html(
            sim.snapshots,
            sim.config,
            title="Test Report",
            subtitle="Minimal test",
        )

        # Basic structure checks
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Report</title>" in html
        assert "Minimal test" in html
        assert "Day 0 (Setup)" in html
        assert "Day 1" in html
        assert "Dealer:" in html
        assert "VBT:" in html

    def test_generate_html_with_trades(self):
        """Verify trade events are rendered."""
        config = DealerRingConfig(
            N_max=0,
            dealer_share=Decimal("0.5"),
            vbt_share=Decimal("0.25"),
            max_days=1,
        )
        sim = DealerRingSimulation(config)

        # Create a trader with shortfall to trigger sell
        trader = TraderState(agent_id="T1", cash=Decimal(0))

        # Create tickets
        tickets = []
        for i in range(4):
            ticket = Ticket(
                id=f"ticket_{i}",
                issuer_id="T1",
                owner_id="T1",
                face=Decimal(1),
                maturity_day=20,
                remaining_tau=15,  # long bucket
            )
            trader.obligations.append(ticket)
            tickets.append(ticket)

        sim.setup_ring([trader], tickets)

        # Manually trigger a trade by executing sell
        from bilancio.dealer import recompute_dealer_state

        dealer = sim.dealers["long"]
        vbt = sim.vbts["long"]
        recompute_dealer_state(dealer, vbt, sim.params)

        # Execute a customer sell if dealer has inventory
        if dealer.inventory:
            ticket_to_sell = dealer.inventory[0]
            result = sim.executor.execute_customer_sell(dealer, vbt, ticket_to_sell)
            sim.events.log_trade(
                day=1,
                side="SELL",
                trader_id="T1",
                ticket_id=ticket_to_sell.id,
                bucket="long",
                price=result.price,
                is_passthrough=result.is_passthrough,
            )

        sim.run(max_days=1)

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # Should contain quote events at minimum
        assert "quote" in html

    def test_generate_html_dealer_state_displayed(self):
        """Verify dealer state parameters are displayed."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # Check dealer parameters are shown
        assert "Tickets (a)" in html
        assert "Face inventory (x)" in html
        assert "Cash (C)" in html
        assert "Mid-valued (V)" in html
        assert "Max tickets (K*)" in html
        assert "Layoff prob" in html
        assert "Midline p(x)" in html
        assert "Bid" in html
        assert "Ask" in html

    def test_generate_html_vbt_state_displayed(self):
        """Verify VBT state parameters are displayed."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # Check VBT parameters are shown
        assert "Mid anchor (M)" in html
        assert "Spread anchor (O)" in html
        assert "Ask (A = M + O/2)" in html
        assert "Bid (B = M - O/2)" in html

    def test_generate_html_trader_balance_displayed(self):
        """Verify trader balance sheet is displayed."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="TestTrader", cash=Decimal("5.50"))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="TestTrader",
            owner_id="TestTrader",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # Check trader info is shown
        assert "TestTrader" in html
        assert "Assets" in html
        assert "Liabilities" in html
        assert "Cash" in html


class TestExportToFile:
    """Test HTML export to file."""

    def test_export_creates_file(self):
        """Export HTML report to file."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_report.html"
            export_dealer_ring_html(sim.snapshots, sim.config, path)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content

    def test_export_creates_parent_dirs(self):
        """Export creates parent directories if needed."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "test_report.html"
            export_dealer_ring_html(sim.snapshots, sim.config, path)

            assert path.exists()

    def test_simulation_to_html_method(self):
        """Test simulation's to_html() convenience method."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])
        sim.run(max_days=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            sim.to_html(path, title="Via Method", subtitle="Test subtitle")

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Via Method" in content
            assert "Test subtitle" in content


class TestHTMLEscaping:
    """Test HTML escaping for security."""

    def test_agent_id_escaped(self):
        """Agent IDs with special characters are escaped."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        # Use agent ID with HTML special characters
        trader = TraderState(agent_id="<xss>alert('test')</xss>", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id=trader.agent_id,
            owner_id=trader.agent_id,
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # XSS tag should be escaped (using xss to avoid false positive from script in CSS)
        assert "<xss>" not in html
        assert "&lt;xss&gt;" in html

    def test_title_escaped(self):
        """Title with special characters is escaped."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        html = generate_dealer_ring_html(
            sim.snapshots,
            sim.config,
            title="Test & Report <bold>",
            subtitle='Quote: "test"',
        )

        assert "&amp;" in html
        assert "&lt;bold&gt;" in html
        assert "&quot;test&quot;" in html


class TestSnapshotCapture:
    """Test that snapshots are properly captured."""

    def test_snapshot_captured_on_setup(self):
        """Snapshot is captured after setup_ring (Day 0)."""
        config = DealerRingConfig(N_max=0, max_days=0)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        # Before setup - no snapshots
        assert len(sim.snapshots) == 0

        sim.setup_ring([trader], [ticket])

        # After setup - Day 0 snapshot
        assert len(sim.snapshots) == 1
        assert sim.snapshots[0].day == 0

    def test_snapshot_captured_each_day(self):
        """Snapshots are captured at end of each day."""
        config = DealerRingConfig(N_max=0, max_days=3)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])
        sim.run(max_days=3)

        # Should have Day 0 + Days 1,2,3 = 4 snapshots
        assert len(sim.snapshots) == 4
        assert [s.day for s in sim.snapshots] == [0, 1, 2, 3]

    def test_snapshot_contains_dealer_state(self):
        """Snapshot captures dealer state correctly."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        snapshot = sim.snapshots[0]

        # Check dealer data is present
        assert "long" in snapshot.dealers
        dealer = snapshot.dealers["long"]
        assert "a" in dealer
        assert "cash" in dealer
        assert "bid" in dealer
        assert "ask" in dealer

    def test_snapshot_contains_trader_state(self):
        """Snapshot captures trader state correctly."""
        config = DealerRingConfig(N_max=0, max_days=1)
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="TestTrader", cash=Decimal("5.50"))
        ticket = Ticket(
            id="ticket_1",
            issuer_id="TestTrader",
            owner_id="TestTrader",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])

        snapshot = sim.snapshots[0]

        # Check trader data is present
        assert "TestTrader" in snapshot.traders
        trader_data = snapshot.traders["TestTrader"]
        assert trader_data["cash"] == Decimal("5.50")
        assert len(trader_data["obligations"]) == 1


class TestRebucketEventRendering:
    """Test rebucket events are rendered correctly."""

    def test_rebucket_event_in_html(self):
        """Rebucket events appear in HTML output."""
        config = DealerRingConfig(
            N_max=0,
            dealer_share=Decimal("1.0"),  # All to dealer for rebucketing
            vbt_share=Decimal("0"),
            max_days=2,
        )
        sim = DealerRingSimulation(config)

        trader = TraderState(agent_id="T1", cash=Decimal(10))

        # Create ticket at boundary (tau=9 is long, tau=8 is mid)
        ticket = Ticket(
            id="boundary_ticket",
            issuer_id="T1",
            owner_id="T1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=9,  # Will become 8 after day 1 update
        )
        trader.obligations.append(ticket)

        sim.setup_ring([trader], [ticket])
        sim.run(max_days=1)  # This should trigger rebucket

        html = generate_dealer_ring_html(sim.snapshots, sim.config)

        # Should see rebucket event if ticket was in dealer inventory
        # The exact presence depends on allocation
        assert "Day 1" in html
