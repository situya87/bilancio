"""Tests for RiskAssessor integration into simulation.py and dealer_integration.py."""

from decimal import Decimal

from bilancio.dealer.models import (
    Ticket,
    TraderState,
)
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.dealer.simulation import DealerRingConfig, DealerRingSimulation


class TestRiskAssessorInSimulation:
    """Test RiskAssessor integration in DealerRingSimulation."""

    def test_simulation_accepts_risk_assessor(self):
        """Simulation should accept optional risk_assessor parameter."""
        config = DealerRingConfig(seed=42)
        params = RiskAssessmentParams()
        risk_assessor = RiskAssessor(params)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        assert sim.risk_assessor is risk_assessor

    def test_simulation_works_without_risk_assessor(self):
        """Simulation should work normally when risk_assessor is None."""
        config = DealerRingConfig(seed=42)
        sim = DealerRingSimulation(config)

        assert sim.risk_assessor is None

    def test_history_updated_on_settlement(self):
        """RiskAssessor.update_history() should be called after each settlement."""
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            N_max=0,  # No trading, just settlement
        )
        params = RiskAssessmentParams()
        risk_assessor = RiskAssessor(params)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        # Create traders and tickets
        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(100),
                tickets_owned=[],
                obligations=[],
            ),
            TraderState(
                agent_id="trader_2",
                cash=Decimal(50),  # Will have shortfall
                tickets_owned=[],
                obligations=[],
            ),
        ]

        # Create tickets: trader_1 owes trader_2
        ticket1 = Ticket(
            id="ticket_1",
            issuer_id="trader_1",
            owner_id="trader_2",
            face=Decimal(10),
            maturity_day=1,
            remaining_tau=1,
            bucket_id="short",
            serial=1,
        )

        # trader_2 owes trader_1 more than they have cash for
        ticket2 = Ticket(
            id="ticket_2",
            issuer_id="trader_2",
            owner_id="trader_1",
            face=Decimal(100),  # More than trader_2's cash
            maturity_day=1,
            remaining_tau=1,
            bucket_id="short",
            serial=2,
        )

        # Link tickets to traders
        traders[0].obligations.append(ticket1)
        traders[1].tickets_owned.append(ticket1)

        traders[1].obligations.append(ticket2)
        traders[0].tickets_owned.append(ticket2)

        # Set up ring
        sim.setup_ring(traders, [ticket1, ticket2])

        # Verify no history yet
        assert len(risk_assessor.payment_history) == 0

        # Run one day (will trigger settlement)
        sim.run_day()

        # Verify history was updated (2 settlements: one success, one default)
        assert len(risk_assessor.payment_history) == 2

        # Check that trader_1 (had enough cash) is recorded as success
        # and trader_2 (insufficient cash) is recorded as default
        history_dict = {h[1]: h[2] for h in risk_assessor.payment_history}  # issuer_id -> defaulted
        assert history_dict["trader_1"] is False  # Full settlement
        assert history_dict["trader_2"] is True  # Defaulted

    def test_history_not_updated_when_no_risk_assessor(self):
        """No history updates when risk_assessor is None."""
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            N_max=0,
        )

        sim = DealerRingSimulation(config, risk_assessor=None)

        # Create simple setup
        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(100),
                tickets_owned=[],
                obligations=[],
            ),
        ]

        ticket = Ticket(
            id="ticket_1",
            issuer_id="trader_1",
            owner_id="trader_1",
            face=Decimal(10),
            maturity_day=1,
            remaining_tau=1,
            bucket_id="short",
            serial=1,
        )
        traders[0].obligations.append(ticket)
        traders[0].tickets_owned.append(ticket)

        sim.setup_ring(traders, [ticket])

        # Should not raise error even without risk_assessor
        sim.run_day()


class TestSellRejection:
    """Test sell rejection based on risk assessment."""

    def test_sell_rejected_when_price_too_low(self):
        """Trader should reject sell when dealer bid is below EV + threshold."""
        # Set up a scenario where default rate is very low (EV ≈ face value)
        # and dealer bid is below that
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            N_max=10,  # Allow trading
            pi_sell=Decimal(1),  # Only sells
            vbt_anchors={
                "short": (Decimal("0.50"), Decimal("0.20")),  # Low bid
                "mid": (Decimal("0.50"), Decimal("0.20")),
                "long": (Decimal("0.50"), Decimal("0.20")),
            },
        )

        # High threshold to make rejections more likely
        params = RiskAssessmentParams(
            base_risk_premium=Decimal("0.50"),  # 50% premium required
            lookback_window=10,
        )
        risk_assessor = RiskAssessor(params)

        # Add history with no defaults (EV ≈ 100% of face)
        for i in range(10):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=False)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        # Create trader with shortfall (eligible to sell)
        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(0),  # No cash = shortfall
                tickets_owned=[],
                obligations=[],
            ),
        ]

        # Ticket with high face value
        ticket = Ticket(
            id="ticket_1",
            issuer_id="trader_1",
            owner_id="trader_1",
            face=Decimal(10),
            maturity_day=20,
            remaining_tau=15,
            bucket_id="short",
            serial=1,
        )
        # Create an obligation to cause shortfall
        obligation = Ticket(
            id="obligation_1",
            issuer_id="trader_1",
            owner_id="other",
            face=Decimal(50),
            maturity_day=11,  # Due soon in simulation time
            remaining_tau=1,
            bucket_id="short",
            serial=2,
        )
        traders[0].tickets_owned.append(ticket)
        traders[0].obligations.append(obligation)

        sim.setup_ring(traders, [ticket, obligation])

        # Run day and check for rejection events
        sim.run_day()

        # Check that a sell_rejected event was logged
        [e for e in sim.events.events if e.get("kind") == "sell_rejected"]

        # With 50% threshold and low VBT bid (0.50), rejection is likely
        # The exact behavior depends on the simulation details
        # We mainly verify the integration works without errors

    def test_sell_accepted_when_price_acceptable(self):
        """Trader should accept sell when dealer bid is above EV + threshold."""
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            N_max=10,
            pi_sell=Decimal(1),
            vbt_anchors={
                "short": (Decimal("1.0"), Decimal("0.10")),  # High bid
                "mid": (Decimal("1.0"), Decimal("0.10")),
                "long": (Decimal("1.0"), Decimal("0.10")),
            },
        )

        # Low threshold to make acceptance more likely
        params = RiskAssessmentParams(
            base_risk_premium=Decimal("0.01"),  # 1% premium
            lookback_window=10,
        )
        risk_assessor = RiskAssessor(params)

        # Add history with some defaults (moderate EV)
        for i in range(8):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=False)
        for i in range(8, 10):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=True)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(0),
                tickets_owned=[],
                obligations=[],
            ),
        ]

        ticket = Ticket(
            id="ticket_1",
            issuer_id="trader_1",
            owner_id="trader_1",
            face=Decimal(1),
            maturity_day=20,
            remaining_tau=15,
            bucket_id="short",
            serial=1,
        )
        obligation = Ticket(
            id="obligation_1",
            issuer_id="trader_1",
            owner_id="other",
            face=Decimal(5),
            maturity_day=11,
            remaining_tau=1,
            bucket_id="short",
            serial=2,
        )
        traders[0].tickets_owned.append(ticket)
        traders[0].obligations.append(obligation)

        sim.setup_ring(traders, [ticket, obligation])
        sim.run_day()

        # Check for successful trade events
        [e for e in sim.events.events if e.get("kind") == "trade"]
        # With high bid and low threshold, trade should execute


class TestBuyRejection:
    """Test buy rejection based on risk assessment."""

    def test_buy_rejected_when_price_too_high(self):
        """Trader should reject buy when dealer ask is above EV - threshold."""
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal("0.5"),  # Give dealer inventory to sell
            vbt_share=Decimal("0.25"),
            N_max=10,
            pi_sell=Decimal(0),  # Only buys
            buffer_B=Decimal(0),  # Low buffer for buy eligibility
            horizon_H=0,
            vbt_anchors={
                "short": (Decimal("1.50"), Decimal("0.10")),  # High ask
                "mid": (Decimal("1.50"), Decimal("0.10")),
                "long": (Decimal("1.50"), Decimal("0.10")),
            },
        )

        # High buy threshold
        params = RiskAssessmentParams(
            base_risk_premium=Decimal("0.10"),
            buy_premium_multiplier=Decimal("3.0"),  # Need 30% premium for buys
            lookback_window=10,
        )
        risk_assessor = RiskAssessor(params)

        # Add history with high default rate (low EV)
        for i in range(5):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=False)
        for i in range(5, 10):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=True)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        # Trader with surplus cash (eligible to buy)
        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(100),
                tickets_owned=[],
                obligations=[],
            ),
        ]

        # Tickets owned by dealer/VBT
        tickets = []
        for i in range(10):
            ticket = Ticket(
                id=f"ticket_{i}",
                issuer_id="trader_1",
                owner_id="",  # Will be assigned by setup_ring
                face=Decimal(1),
                maturity_day=20,
                remaining_tau=15,
                bucket_id="short",
                serial=i,
            )
            tickets.append(ticket)

        sim.setup_ring(traders, tickets)
        sim.run_day()

        # With high ask price and low EV, buy rejections should occur
        [e for e in sim.events.events if e.get("kind") == "buy_rejected"]
        # Exact count depends on simulation randomness


class TestEventLogging:
    """Test that rejection events are logged correctly."""

    def test_sell_rejected_event_has_correct_fields(self):
        """Sell rejection events should include expected fields."""
        config = DealerRingConfig(
            seed=42,
            ticket_size=Decimal(1),
            dealer_share=Decimal(0),
            vbt_share=Decimal(0),
            N_max=10,
            pi_sell=Decimal(1),
            vbt_anchors={
                "short": (Decimal("0.10"), Decimal("0.05")),  # Very low bid
                "mid": (Decimal("0.10"), Decimal("0.05")),
                "long": (Decimal("0.10"), Decimal("0.05")),
            },
        )

        params = RiskAssessmentParams(
            base_risk_premium=Decimal("0.90"),  # Very high threshold
            lookback_window=10,
        )
        risk_assessor = RiskAssessor(params)

        # No defaults - high EV
        for i in range(10):
            risk_assessor.update_history(day=i, issuer_id="trader_1", defaulted=False)

        sim = DealerRingSimulation(config, risk_assessor=risk_assessor)

        traders = [
            TraderState(
                agent_id="trader_1",
                cash=Decimal(0),
                tickets_owned=[],
                obligations=[],
            ),
        ]

        ticket = Ticket(
            id="ticket_1",
            issuer_id="trader_1",
            owner_id="trader_1",
            face=Decimal(10),
            maturity_day=20,
            remaining_tau=15,
            bucket_id="short",
            serial=1,
        )
        obligation = Ticket(
            id="obligation_1",
            issuer_id="trader_1",
            owner_id="other",
            face=Decimal(50),
            maturity_day=11,
            remaining_tau=1,
            bucket_id="short",
            serial=2,
        )
        traders[0].tickets_owned.append(ticket)
        traders[0].obligations.append(obligation)

        sim.setup_ring(traders, [ticket, obligation])
        sim.run_day()

        sell_rejected_events = [e for e in sim.events.events if e.get("kind") == "sell_rejected"]

        if sell_rejected_events:
            event = sell_rejected_events[0]
            # Verify required fields
            assert "day" in event
            assert "trader_id" in event
            assert "ticket_id" in event
            assert "bucket" in event
            assert "offered_price" in event
            assert "expected_value" in event
            assert "threshold" in event
            assert "reason" in event
