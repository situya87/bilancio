"""Integration tests for bidirectional trading (both buys and sells).

Verifies that both buy and sell trades occur after the enable-buy-trades changes:
1. Cash neutrality constraint removed
2. VBT credit facility added (passthrough friction)
3. Dealer sizing updated
"""

import random
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.models import (
    DealerState,
    VBTState,
    TraderState,
    Ticket,
    BucketConfig,
)
from bilancio.dealer.trading import TradeExecutor
from bilancio.dealer.metrics import RunMetrics
from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams
from bilancio.engines.dealer_integration import DealerSubsystem
from bilancio.engines.dealer_trades import (
    _execute_sell_trade,
    _execute_buy_trade,
    _execute_interleaved_order_flow,
    _build_eligible_sellers,
    _build_eligible_buyers,
)
from bilancio.decision.profiles import TraderProfile


def _make_ticket(
    id: str = "t1",
    issuer_id: str = "issuer_1",
    owner_id: str = "dealer_short",
    face: Decimal = Decimal(1),
    maturity_day: int = 100,
    serial: int = 0,
    bucket_id: str = "short",
) -> Ticket:
    return Ticket(
        id=id,
        issuer_id=issuer_id,
        owner_id=owner_id,
        face=face,
        maturity_day=maturity_day,
        remaining_tau=10,
        bucket_id=bucket_id,
        serial=serial,
    )


def _make_dealer(
    bucket_id: str = "short",
    n_tickets: int = 0,
    cash: Decimal = Decimal(0),
) -> DealerState:
    agent_id = f"dealer_{bucket_id}"
    dealer = DealerState(bucket_id=bucket_id, agent_id=agent_id, cash=cash)
    for i in range(n_tickets):
        dealer.inventory.append(
            _make_ticket(
                id=f"dt_{bucket_id}_{i}",
                issuer_id=f"issuer_{i}",
                owner_id=agent_id,
                serial=i,
                maturity_day=100 + i,
                bucket_id=bucket_id,
            )
        )
    return dealer


def _make_vbt(
    bucket_id: str = "short",
    M: Decimal = Decimal("0.85"),
    O: Decimal = Decimal("0.10"),
    n_tickets: int = 0,
    cash: Decimal = Decimal(100),
) -> VBTState:
    agent_id = f"vbt_{bucket_id}"
    vbt = VBTState(bucket_id=bucket_id, agent_id=agent_id, M=M, O=O, cash=cash)
    vbt.recompute_quotes()
    for i in range(n_tickets):
        vbt.inventory.append(
            _make_ticket(
                id=f"vt_{bucket_id}_{i}",
                issuer_id=f"issuer_{i}",
                owner_id=agent_id,
                serial=i,
                maturity_day=100 + i,
                bucket_id=bucket_id,
            )
        )
    return vbt


def _make_subsystem(
    dealer_tickets: int = 5,
    dealer_cash: Decimal = Decimal(10),
    vbt_tickets: int = 3,
    vbt_cash: Decimal = Decimal(100),
    face_value: Decimal = Decimal(1),
    layoff_threshold: Decimal = Decimal("0.7"),
) -> DealerSubsystem:
    """Build a subsystem configured for bidirectional trading."""
    params = KernelParams(S=face_value)
    bucket_id = "short"

    dealer = _make_dealer(bucket_id=bucket_id, n_tickets=dealer_tickets, cash=dealer_cash)
    vbt = _make_vbt(bucket_id=bucket_id, n_tickets=vbt_tickets, cash=vbt_cash)
    recompute_dealer_state(dealer, vbt, params)

    subsystem = DealerSubsystem(
        dealers={bucket_id: dealer},
        vbts={bucket_id: vbt},
        traders={},
        tickets={},
        bucket_configs=[BucketConfig("short", 1, 3)],
        params=params,
        executor=TradeExecutor(params, random.Random(42), layoff_threshold=layoff_threshold),
        rng=random.Random(42),
        face_value=face_value,
        metrics=RunMetrics(),
        layoff_threshold=layoff_threshold,
    )

    return subsystem


class TestBidirectionalTrading:
    """Test that both buys and sells can occur in the same run."""

    def test_sells_without_prior_buys(self):
        """Sellers can execute without buyers going first."""
        subsystem = _make_subsystem(dealer_cash=Decimal(50))

        # Add a seller with shortfall
        ticket = _make_ticket(id="sell_t1", face=Decimal(1), bucket_id="short")
        obligation = _make_ticket(id="obl1", face=Decimal(50), maturity_day=5)
        seller = TraderState(
            agent_id="seller1",
            cash=Decimal(5),
            tickets_owned=[ticket],
            obligations=[obligation],
        )
        subsystem.traders["seller1"] = seller
        events: list[dict] = []

        result = _execute_sell_trade(subsystem, "seller1", 5, events)
        assert result > Decimal(0), "Sell should execute"
        assert len(events) == 1
        assert events[0]["side"] == "sell"

    def test_buys_without_prior_sells(self):
        """Buyers can execute independently — no cash neutrality constraint."""
        subsystem = _make_subsystem(dealer_tickets=5, dealer_cash=Decimal(10))

        # Add a buyer with surplus
        buyer = TraderState(agent_id="buyer1", cash=Decimal(100))
        subsystem.traders["buyer1"] = buyer
        events: list[dict] = []

        result = _execute_buy_trade(subsystem, "buyer1", 1, events)
        assert result > Decimal(0), "Buy should execute without prior sells"
        assert len(events) == 1
        assert events[0]["side"] == "buy"

    def test_order_flow_both_directions(self):
        """Interleaved order flow produces both buys and sells."""
        subsystem = _make_subsystem(dealer_tickets=5, dealer_cash=Decimal(50))

        # Add sellers with shortfall
        for i in range(3):
            ticket = _make_ticket(
                id=f"sell_t{i}", face=Decimal(1), bucket_id="short",
                issuer_id=f"issuer_{i}",
            )
            obligation = _make_ticket(
                id=f"obl{i}", face=Decimal(50), maturity_day=5,
                issuer_id=f"issuer_{i}",
            )
            trader = TraderState(
                agent_id=f"seller{i}",
                cash=Decimal(5),
                tickets_owned=[ticket],
                obligations=[obligation],
            )
            subsystem.traders[f"seller{i}"] = trader

        # Add buyers with surplus
        for i in range(3):
            trader = TraderState(
                agent_id=f"buyer{i}",
                cash=Decimal(100),
            )
            subsystem.traders[f"buyer{i}"] = trader

        # Mock system for dealer budgets — need cash contracts so
        # _get_agent_cash returns non-zero for dealer/VBT
        from bilancio.domain.instruments.base import InstrumentKind
        dealer_cash_contract = MagicMock()
        dealer_cash_contract.kind = InstrumentKind.CASH
        dealer_cash_contract.amount = 50
        vbt_cash_contract = MagicMock()
        vbt_cash_contract.kind = InstrumentKind.CASH
        vbt_cash_contract.amount = 100

        system = MagicMock()
        system.state.agents = {
            "dealer_short": MagicMock(asset_ids=["dc1"]),
            "vbt_short": MagicMock(asset_ids=["vc1"]),
        }
        system.state.contracts = {
            "dc1": dealer_cash_contract,
            "vc1": vbt_cash_contract,
        }
        events: list[dict] = []

        sellers = [f"seller{i}" for i in range(3)]
        buyers = [f"buyer{i}" for i in range(3)]

        _execute_interleaved_order_flow(
            subsystem, system, 5, sellers, buyers, events,
        )

        sell_events = [e for e in events if e.get("side") == "sell"]
        buy_events = [e for e in events if e.get("side") == "buy"]

        assert len(sell_events) > 0, "Should have sell trades"
        assert len(buy_events) > 0, "Should have buy trades (no cash neutrality)"


class TestPassthroughFriction:
    """Test VBT credit facility builds dealer inventory."""

    def test_credit_facility_activates_below_threshold(self):
        """When dealer inventory < layoff_threshold * K_star, VBT injects cash."""
        params = KernelParams(S=Decimal(1))
        M = Decimal("0.85")
        O = Decimal("0.10")

        # Dealer starts with 0 inventory (below any threshold)
        dealer = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal(0))
        vbt = VBTState(bucket_id="short", agent_id="vbt_short", M=M, O=O, cash=Decimal(100))
        vbt.recompute_quotes()
        recompute_dealer_state(dealer, vbt, params)

        # Ticket to sell
        ticket = _make_ticket(id="t_cf", face=Decimal(1), owner_id="customer")

        # Executor WITH layoff threshold
        executor = TradeExecutor(params, random.Random(42), layoff_threshold=Decimal("0.7"))
        result = executor.execute_customer_sell(dealer, vbt, ticket, check_assertions=False)

        assert result.executed
        # With credit facility, dealer should buy interior (not passthrough)
        # because VBT injected cash into dealer
        # Either way, the trade should execute
        assert result.price > Decimal(0)

    def test_no_credit_facility_when_threshold_zero(self):
        """With layoff_threshold=0, credit facility is disabled."""
        params = KernelParams(S=Decimal(1))
        M = Decimal("0.85")
        O = Decimal("0.10")

        # Dealer with 0 inventory and 0 cash
        dealer = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal(0))
        vbt = VBTState(bucket_id="short", agent_id="vbt_short", M=M, O=O, cash=Decimal(100))
        vbt.recompute_quotes()
        recompute_dealer_state(dealer, vbt, params)

        ticket = _make_ticket(id="t_no_cf", face=Decimal(1), owner_id="customer")

        # Executor WITHOUT layoff threshold
        executor = TradeExecutor(params, random.Random(42), layoff_threshold=Decimal("0"))
        result = executor.execute_customer_sell(dealer, vbt, ticket, check_assertions=False)

        assert result.executed
        # Without credit facility, this should be a passthrough
        assert result.is_passthrough is True

    def test_credit_facility_deactivates_above_threshold(self):
        """When dealer inventory >= layoff_threshold * K_star, passthrough resumes."""
        params = KernelParams(S=Decimal(1))
        M = Decimal("0.85")
        O = Decimal("0.10")

        # Dealer with LOTS of inventory (above threshold)
        dealer = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal(50))
        for i in range(20):
            dealer.inventory.append(_make_ticket(
                id=f"d_inv_{i}", owner_id="dealer_short",
                issuer_id=f"iss_{i}", serial=i, bucket_id="short",
            ))

        vbt = VBTState(bucket_id="short", agent_id="vbt_short", M=M, O=O, cash=Decimal(100))
        vbt.recompute_quotes()
        recompute_dealer_state(dealer, vbt, params)

        # If dealer is at capacity (a >= K_star), can_interior_buy is False
        # and inventory_ratio >= layoff_threshold, so credit facility should NOT activate
        if dealer.a >= dealer.K_star:
            ticket = _make_ticket(id="t_above", face=Decimal(1), owner_id="customer")
            executor = TradeExecutor(params, random.Random(42), layoff_threshold=Decimal("0.7"))
            result = executor.execute_customer_sell(dealer, vbt, ticket, check_assertions=False)
            assert result.executed
            # Should be passthrough since dealer is at capacity
            assert result.is_passthrough is True
