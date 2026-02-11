"""
Comprehensive unit tests for bilancio.dealer.metrics module.

Focuses on maximizing code coverage by testing all dataclasses,
computed properties, methods, helper functions, and CSV/JSON export
with synthetic data -- no full system initialization required.
"""

import csv
import json
import pytest
from decimal import Decimal
from pathlib import Path

from bilancio.dealer.metrics import (
    TradeRecord,
    DealerSnapshot,
    TraderSnapshot,
    TicketOutcome,
    SystemStateSnapshot,
    RepaymentEvent,
    RunMetrics,
    LiabilityInfo,
    classify_trading_strategy,
    compute_safety_margin,
    compute_saleable_value,
    build_liability_map,
    compute_trading_stats_by_trader,
    get_trades_before_day,
    build_repayment_events,
)


# ---------------------------------------------------------------------------
# Helpers to build realistic test fixtures
# ---------------------------------------------------------------------------

def _trade(
    day=1,
    bucket="short",
    side="SELL",
    trader_id="h0",
    ticket_id="T1",
    issuer_id="h1",
    maturity_day=5,
    face_value=Decimal(1),
    price=Decimal("0.90"),
    unit_price=Decimal("0.90"),
    is_passthrough=False,
    is_liquidity_driven=False,
    reduces_margin_below_zero=False,
    dealer_bid_before=Decimal("0.88"),
    dealer_ask_before=Decimal("0.98"),
    dealer_cash_before=Decimal(100),
    dealer_inventory_before=5,
    trader_cash_before=Decimal(50),
    trader_safety_margin_before=Decimal(10),
    trader_safety_margin_after=Decimal(20),
    hit_inventory_limit=False,
    run_id="run1",
    regime="active",
    **kw,
) -> TradeRecord:
    return TradeRecord(
        day=day,
        bucket=bucket,
        side=side,
        trader_id=trader_id,
        ticket_id=ticket_id,
        issuer_id=issuer_id,
        maturity_day=maturity_day,
        face_value=face_value,
        price=price,
        unit_price=unit_price,
        is_passthrough=is_passthrough,
        is_liquidity_driven=is_liquidity_driven,
        reduces_margin_below_zero=reduces_margin_below_zero,
        dealer_bid_before=dealer_bid_before,
        dealer_ask_before=dealer_ask_before,
        dealer_cash_before=dealer_cash_before,
        dealer_inventory_before=dealer_inventory_before,
        trader_cash_before=trader_cash_before,
        trader_safety_margin_before=trader_safety_margin_before,
        trader_safety_margin_after=trader_safety_margin_after,
        hit_inventory_limit=hit_inventory_limit,
        run_id=run_id,
        regime=regime,
        **kw,
    )


def _snapshot(
    day=1,
    bucket="short",
    inventory=5,
    cash=Decimal(100),
    bid=Decimal("0.88"),
    ask=Decimal("0.98"),
    midline=Decimal("0.93"),
    vbt_mid=Decimal("0.95"),
    vbt_spread=Decimal("0.10"),
    ticket_size=Decimal(1),
    max_capacity=10,
    run_id="run1",
    regime="active",
    is_at_zero=False,
    hit_vbt_this_step=False,
    total_system_face=Decimal(500),
    dealer_share_pct=Decimal("2.0"),
) -> DealerSnapshot:
    return DealerSnapshot(
        day=day,
        bucket=bucket,
        inventory=inventory,
        cash=cash,
        bid=bid,
        ask=ask,
        midline=midline,
        vbt_mid=vbt_mid,
        vbt_spread=vbt_spread,
        ticket_size=ticket_size,
        max_capacity=max_capacity,
        run_id=run_id,
        regime=regime,
        is_at_zero=is_at_zero,
        hit_vbt_this_step=hit_vbt_this_step,
        total_system_face=total_system_face,
        dealer_share_pct=dealer_share_pct,
    )


def _trader_snapshot(
    day=1,
    trader_id="h0",
    cash=Decimal(100),
    tickets_held_count=2,
    tickets_held_ids=None,
    total_face_held=Decimal(2),
    obligations_remaining=Decimal(50),
    saleable_value=Decimal("1.76"),
    safety_margin=Decimal("51.76"),
    defaulted=False,
) -> TraderSnapshot:
    return TraderSnapshot(
        day=day,
        trader_id=trader_id,
        cash=cash,
        tickets_held_count=tickets_held_count,
        tickets_held_ids=tickets_held_ids or ["T1", "T2"],
        total_face_held=total_face_held,
        obligations_remaining=obligations_remaining,
        saleable_value=saleable_value,
        safety_margin=safety_margin,
        defaulted=defaulted,
    )


def _ticket_outcome(
    ticket_id="T1",
    issuer_id="h1",
    maturity_day=5,
    face_value=Decimal(1),
    purchased_from_dealer=False,
    purchase_day=None,
    purchase_price=None,
    purchaser_id=None,
    sold_to_dealer=False,
    sale_day=None,
    sale_price=None,
    seller_id=None,
    settled=False,
    settlement_day=None,
    recovery_rate=None,
    settlement_amount=None,
) -> TicketOutcome:
    return TicketOutcome(
        ticket_id=ticket_id,
        issuer_id=issuer_id,
        maturity_day=maturity_day,
        face_value=face_value,
        purchased_from_dealer=purchased_from_dealer,
        purchase_day=purchase_day,
        purchase_price=purchase_price,
        purchaser_id=purchaser_id,
        sold_to_dealer=sold_to_dealer,
        sale_day=sale_day,
        sale_price=sale_price,
        seller_id=seller_id,
        settled=settled,
        settlement_day=settlement_day,
        recovery_rate=recovery_rate,
        settlement_amount=settlement_amount,
    )


# ===================================================================
# TradeRecord
# ===================================================================

class TestTradeRecord:
    def test_to_dict_keys(self):
        t = _trade()
        d = t.to_dict()
        expected_keys = {
            "run_id", "regime",
            "day", "bucket", "side", "trader_id", "ticket_id",
            "issuer_id", "maturity_day", "face_value", "price", "unit_price",
            "is_passthrough",
            "dealer_inventory_before", "dealer_cash_before",
            "dealer_bid_before", "dealer_ask_before", "vbt_mid_before",
            "trader_cash_before", "trader_safety_margin_before",
            "dealer_inventory_after", "dealer_cash_after",
            "dealer_bid_after", "dealer_ask_after",
            "trader_cash_after", "trader_safety_margin_after",
            "is_liquidity_driven", "reduces_margin_below_zero",
            "hit_inventory_limit",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_decimal_serialization(self):
        t = _trade(face_value=Decimal("1.5"), price=Decimal("1.35"))
        d = t.to_dict()
        assert d["face_value"] == "1.5"
        assert d["price"] == "1.35"

    def test_defaults(self):
        t = _trade()
        assert t.dealer_inventory_after == 0
        assert t.dealer_cash_after == Decimal(0)
        assert t.vbt_mid_before == Decimal(0)


# ===================================================================
# DealerSnapshot
# ===================================================================

class TestDealerSnapshot:
    def test_mark_to_mid_equity(self):
        s = _snapshot(cash=Decimal(100), vbt_mid=Decimal("0.95"),
                      inventory=10, ticket_size=Decimal(1))
        # E = 100 + 0.95 * 10 * 1 = 109.5
        assert s.mark_to_mid_equity == Decimal("109.50")

    def test_mark_to_mid_equity_zero_inventory(self):
        s = _snapshot(inventory=0)
        assert s.mark_to_mid_equity == s.cash

    def test_capacity_pct(self):
        s = _snapshot(inventory=5, max_capacity=10)
        assert s.capacity_pct == Decimal(50)

    def test_capacity_pct_zero_max(self):
        s = _snapshot(inventory=5, max_capacity=0)
        assert s.capacity_pct == Decimal(0)

    def test_spread(self):
        s = _snapshot(bid=Decimal("0.88"), ask=Decimal("0.98"))
        assert s.spread == Decimal("0.10")

    def test_dealer_premium_vs_face(self):
        s = _snapshot(midline=Decimal("1.05"))
        assert s.dealer_premium_vs_face == Decimal("0.05")

    def test_dealer_premium_pct(self):
        s = _snapshot(midline=Decimal("0.85"))
        assert s.dealer_premium_pct == Decimal("-15")

    def test_vbt_premium_vs_face(self):
        s = _snapshot(vbt_mid=Decimal("1.02"))
        assert s.vbt_premium_vs_face == Decimal("0.02")

    def test_vbt_premium_pct(self):
        s = _snapshot(vbt_mid=Decimal("0.90"))
        assert s.vbt_premium_pct == Decimal("-10")

    def test_to_dict_keys_and_computed(self):
        s = _snapshot()
        d = s.to_dict()
        assert "mark_to_mid_equity" in d
        assert "spread" in d
        assert "dealer_premium_pct" in d
        assert "vbt_premium_pct" in d
        assert "capacity_pct" in d
        assert "is_at_zero" in d
        assert "hit_vbt_this_step" in d
        assert "total_system_face" in d
        assert "dealer_share_pct" in d
        assert "run_id" in d
        assert "regime" in d


# ===================================================================
# TraderSnapshot
# ===================================================================

class TestTraderSnapshot:
    def test_to_dict(self):
        ts = _trader_snapshot()
        d = ts.to_dict()
        assert d["trader_id"] == "h0"
        assert d["cash"] == "100"
        assert d["defaulted"] is False
        assert d["tickets_held_ids"] == ["T1", "T2"]

    def test_defaulted_flag(self):
        ts = _trader_snapshot(defaulted=True)
        assert ts.defaulted is True
        assert ts.to_dict()["defaulted"] is True


# ===================================================================
# TicketOutcome
# ===================================================================

class TestTicketOutcome:
    def test_realized_return_not_purchased(self):
        o = _ticket_outcome(purchased_from_dealer=False)
        assert o.realized_return() is None

    def test_realized_return_no_purchase_price(self):
        o = _ticket_outcome(purchased_from_dealer=True, purchase_price=None)
        assert o.realized_return() is None

    def test_realized_return_held_to_maturity(self):
        o = _ticket_outcome(
            purchased_from_dealer=True,
            purchase_price=Decimal("0.90"),
            purchaser_id="h0",
            settled=True,
            settlement_amount=Decimal("1.00"),
        )
        # R = (1.00 - 0.90) / 0.90 = 0.111...
        r = o.realized_return()
        assert r is not None
        assert abs(r - Decimal("0.1") / Decimal("0.9")) < Decimal("0.0001")

    def test_realized_return_resold_to_dealer(self):
        o = _ticket_outcome(
            purchased_from_dealer=True,
            purchase_price=Decimal("0.80"),
            purchaser_id="h0",
            sold_to_dealer=True,
            sale_price=Decimal("0.85"),
        )
        # R = (0.85 - 0.80) / 0.80 = 0.0625
        r = o.realized_return()
        assert r is not None
        assert r == Decimal("0.05") / Decimal("0.80")

    def test_realized_return_still_held(self):
        """Not sold, not settled -> None."""
        o = _ticket_outcome(
            purchased_from_dealer=True,
            purchase_price=Decimal("0.90"),
            purchaser_id="h0",
        )
        assert o.realized_return() is None

    def test_realized_return_zero_purchase_price(self):
        o = _ticket_outcome(
            purchased_from_dealer=True,
            purchase_price=Decimal(0),
            purchaser_id="h0",
            settled=True,
            settlement_amount=Decimal(1),
        )
        assert o.realized_return() is None

    def test_to_dict_with_return(self):
        o = _ticket_outcome(
            purchased_from_dealer=True,
            purchase_price=Decimal("0.90"),
            purchaser_id="h0",
            settled=True,
            settlement_amount=Decimal("1.00"),
        )
        d = o.to_dict()
        assert d["realized_return"] is not None
        assert d["purchased_from_dealer"] is True

    def test_to_dict_no_return(self):
        o = _ticket_outcome()
        d = o.to_dict()
        assert d["realized_return"] is None
        assert d["purchase_price"] is None
        assert d["sale_price"] is None
        assert d["recovery_rate"] is None
        assert d["settlement_amount"] is None

    def test_to_dict_with_sale_price(self):
        o = _ticket_outcome(
            sold_to_dealer=True,
            sale_price=Decimal("0.85"),
            seller_id="h0",
        )
        d = o.to_dict()
        assert d["sale_price"] == "0.85"

    def test_to_dict_with_recovery_rate(self):
        o = _ticket_outcome(
            settled=True,
            recovery_rate=Decimal("0.5"),
            settlement_amount=Decimal("0.5"),
        )
        d = o.to_dict()
        assert d["recovery_rate"] == "0.5"
        assert d["settlement_amount"] == "0.5"


# ===================================================================
# SystemStateSnapshot
# ===================================================================

class TestSystemStateSnapshot:
    def test_debt_to_money(self):
        s = SystemStateSnapshot(
            run_id="r1", regime="active", day=1,
            total_face_value=Decimal(1000),
            total_cash=Decimal(500),
        )
        assert s.debt_to_money == Decimal(2)

    def test_debt_to_money_zero_cash(self):
        s = SystemStateSnapshot(
            run_id="r1", regime="active", day=1,
            total_face_value=Decimal(1000),
            total_cash=Decimal(0),
        )
        assert s.debt_to_money == Decimal(0)

    def test_to_dict(self):
        s = SystemStateSnapshot(
            run_id="r1", regime="active", day=3,
            total_face_value=Decimal(800),
            face_bucket_short=Decimal(200),
            face_bucket_mid=Decimal(300),
            face_bucket_long=Decimal(300),
            total_cash=Decimal(400),
        )
        d = s.to_dict()
        assert d["run_id"] == "r1"
        assert d["regime"] == "active"
        assert d["day"] == 3
        assert d["total_face_value"] == "800"
        assert d["debt_to_money"] == "2"


# ===================================================================
# RepaymentEvent
# ===================================================================

class TestRepaymentEvent:
    def test_to_dict(self):
        e = RepaymentEvent(
            run_id="r1", regime="active",
            trader_id="h0", liability_id="PAY_1",
            maturity_day=5, face_value=Decimal(50),
            outcome="repaid",
            buy_count=2, sell_count=1,
            net_cash_pnl=Decimal("10.5"),
            strategy="round_trip",
        )
        d = e.to_dict()
        assert d["outcome"] == "repaid"
        assert d["strategy"] == "round_trip"
        assert d["net_cash_pnl"] == "10.5"
        assert d["buy_count"] == 2
        assert d["sell_count"] == 1


# ===================================================================
# classify_trading_strategy
# ===================================================================

class TestClassifyTradingStrategy:
    def test_no_trade(self):
        assert classify_trading_strategy(0, 0) == "no_trade"

    def test_hold_to_maturity(self):
        assert classify_trading_strategy(3, 0) == "hold_to_maturity"

    def test_sell_before(self):
        assert classify_trading_strategy(0, 2) == "sell_before"

    def test_round_trip(self):
        assert classify_trading_strategy(1, 1) == "round_trip"

    def test_round_trip_many(self):
        assert classify_trading_strategy(5, 3) == "round_trip"


# ===================================================================
# RunMetrics -- Dealer Profitability (Section 8.2)
# ===================================================================

class TestRunMetricsDealerProfitability:
    def _metrics_with_snapshots(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {
            "short": Decimal(100),
            "mid": Decimal(200),
        }
        # Day-1 snapshots
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", cash=Decimal(110), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
            _snapshot(day=1, bucket="mid", cash=Decimal(190), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
            # Day-3 snapshots (final)
            _snapshot(day=3, bucket="short", cash=Decimal(120), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
            _snapshot(day=3, bucket="mid", cash=Decimal(180), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
        ]
        return m

    def test_dealer_pnl_by_bucket(self):
        m = self._metrics_with_snapshots()
        pnl = m.dealer_pnl_by_bucket()
        # short: 120 - 100 = 20, mid: 180 - 200 = -20
        assert pnl["short"] == Decimal(20)
        assert pnl["mid"] == Decimal(-20)

    def test_dealer_return_by_bucket(self):
        m = self._metrics_with_snapshots()
        returns = m.dealer_return_by_bucket()
        assert returns["short"] == Decimal(20) / Decimal(100)
        assert returns["mid"] == Decimal(-20) / Decimal(200)

    def test_dealer_return_zero_initial_equity(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(0)}
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", cash=Decimal(10), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
        ]
        returns = m.dealer_return_by_bucket()
        assert returns["short"] == Decimal(0)

    def test_total_dealer_pnl(self):
        m = self._metrics_with_snapshots()
        assert m.total_dealer_pnl() == Decimal(0)  # 20 + (-20)

    def test_total_dealer_return(self):
        m = self._metrics_with_snapshots()
        # total_pnl=0 / total_initial=300
        assert m.total_dealer_return() == Decimal(0)

    def test_total_dealer_return_zero_initial(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {}
        assert m.total_dealer_return() == Decimal(0)

    def test_is_dealer_profitable_true(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(100)}
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", cash=Decimal(110), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
        ]
        assert m.is_dealer_profitable() is True

    def test_is_dealer_profitable_false(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(100)}
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", cash=Decimal(80), vbt_mid=Decimal(1),
                      inventory=0, ticket_size=Decimal(1)),
        ]
        assert m.is_dealer_profitable() is False

    def test_dealer_pnl_no_snapshots_for_bucket(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"long": Decimal(50)}
        m.dealer_snapshots = []
        pnl = m.dealer_pnl_by_bucket()
        assert pnl["long"] == Decimal(0)

    def test_spread_income_interior_sell(self):
        m = RunMetrics()
        t = _trade(side="SELL", is_passthrough=False,
                   dealer_bid_before=Decimal("0.88"),
                   dealer_ask_before=Decimal("0.98"),
                   face_value=Decimal(10))
        m.trades = [t]
        # midline = (0.88 + 0.98)/2 = 0.93
        # income = (0.93 - 0.88) * 10 = 0.50
        assert m.spread_income_total() == Decimal("0.50")

    def test_spread_income_interior_buy(self):
        m = RunMetrics()
        t = _trade(side="BUY", is_passthrough=False,
                   dealer_bid_before=Decimal("0.88"),
                   dealer_ask_before=Decimal("0.98"),
                   face_value=Decimal(10))
        m.trades = [t]
        # midline = 0.93 , income = (0.98 - 0.93) * 10 = 0.50
        assert m.spread_income_total() == Decimal("0.50")

    def test_spread_income_passthrough_excluded(self):
        m = RunMetrics()
        m.trades = [_trade(is_passthrough=True, face_value=Decimal(10))]
        assert m.spread_income_total() == Decimal(0)

    def test_passthrough_count(self):
        m = RunMetrics()
        m.trades = [
            _trade(is_passthrough=True),
            _trade(is_passthrough=False),
            _trade(is_passthrough=True),
        ]
        assert m.passthrough_count() == 2

    def test_interior_count(self):
        m = RunMetrics()
        m.trades = [
            _trade(is_passthrough=True),
            _trade(is_passthrough=False),
            _trade(is_passthrough=False),
        ]
        assert m.interior_count() == 2


# ===================================================================
# RunMetrics -- Trader Returns (Section 8.3)
# ===================================================================

class TestRunMetricsTraderReturns:
    def test_trader_returns_single(self):
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(
                purchased_from_dealer=True,
                purchase_price=Decimal("0.80"),
                purchaser_id="h0",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
        }
        returns = m.trader_returns()
        # R = (1.00 - 0.80) / 0.80 = 0.25
        assert "h0" in returns
        assert returns["h0"] == Decimal("0.25")

    def test_trader_returns_multiple_tickets(self):
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(
                ticket_id="T1",
                purchased_from_dealer=True,
                purchase_price=Decimal("0.80"),
                purchaser_id="h0",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
            "T2": _ticket_outcome(
                ticket_id="T2",
                purchased_from_dealer=True,
                purchase_price=Decimal("1.00"),
                purchaser_id="h0",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
        }
        returns = m.trader_returns()
        # T1: R=0.25, T2: R=0.00, mean = 0.125
        assert returns["h0"] == Decimal("0.125")

    def test_trader_returns_empty(self):
        m = RunMetrics()
        assert m.trader_returns() == {}

    def test_trader_returns_no_realized(self):
        """Ticket purchased but not yet settled or sold."""
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(
                purchased_from_dealer=True,
                purchase_price=Decimal("0.80"),
                purchaser_id="h0",
            ),
        }
        returns = m.trader_returns()
        assert returns == {}

    def test_mean_trader_return(self):
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(
                ticket_id="T1",
                purchased_from_dealer=True,
                purchase_price=Decimal("0.80"),
                purchaser_id="h0",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
            "T2": _ticket_outcome(
                ticket_id="T2",
                purchased_from_dealer=True,
                purchase_price=Decimal("0.50"),
                purchaser_id="h1",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
        }
        # h0: R=0.25, h1: R=1.0 => mean = 0.625
        assert m.mean_trader_return() == Decimal("0.625")

    def test_mean_trader_return_empty(self):
        m = RunMetrics()
        assert m.mean_trader_return() == Decimal(0)

    def test_fraction_profitable_traders(self):
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(
                ticket_id="T1",
                purchased_from_dealer=True,
                purchase_price=Decimal("0.80"),
                purchaser_id="h0",
                settled=True,
                settlement_amount=Decimal("1.00"),
            ),
            "T2": _ticket_outcome(
                ticket_id="T2",
                purchased_from_dealer=True,
                purchase_price=Decimal("1.00"),
                purchaser_id="h1",
                settled=True,
                settlement_amount=Decimal("0.80"),
            ),
        }
        # h0 profitable (R=0.25), h1 not (R=-0.20)
        assert m.fraction_profitable_traders() == Decimal("0.5")

    def test_fraction_profitable_traders_empty(self):
        m = RunMetrics()
        assert m.fraction_profitable_traders() == Decimal(0)


# ===================================================================
# RunMetrics -- Liquidity / Rescue (Section 8.3)
# ===================================================================

class TestRunMetricsLiquidity:
    def test_liquidity_driven_sales(self):
        m = RunMetrics()
        m.trades = [
            _trade(side="SELL", is_liquidity_driven=True),
            _trade(side="SELL", is_liquidity_driven=False),
            _trade(side="BUY", is_liquidity_driven=True),  # not a SELL
        ]
        assert m.liquidity_driven_sales() == 1

    def test_rescue_events(self):
        m = RunMetrics()
        m.trades = [
            _trade(side="SELL", is_liquidity_driven=True,
                   trader_safety_margin_before=Decimal(-10),
                   trader_safety_margin_after=Decimal(5)),
            _trade(side="SELL", is_liquidity_driven=True,
                   trader_safety_margin_before=Decimal(-10),
                   trader_safety_margin_after=Decimal(-2)),  # still negative
            _trade(side="BUY", is_liquidity_driven=True,
                   trader_safety_margin_before=Decimal(-10),
                   trader_safety_margin_after=Decimal(5)),  # not a SELL
        ]
        assert m.rescue_events() == 1

    def test_rescue_exact_zero_margin(self):
        """Margin going from negative to exactly zero counts as rescue."""
        m = RunMetrics()
        m.trades = [
            _trade(side="SELL", is_liquidity_driven=True,
                   trader_safety_margin_before=Decimal(-5),
                   trader_safety_margin_after=Decimal(0)),
        ]
        assert m.rescue_events() == 1


# ===================================================================
# RunMetrics -- Repayment-Priority (Section 8.4)
# ===================================================================

class TestRunMetricsRepaymentPriority:
    def test_unsafe_buy_count(self):
        m = RunMetrics()
        m.trades = [
            _trade(side="BUY", reduces_margin_below_zero=True),
            _trade(side="BUY", reduces_margin_below_zero=False),
            _trade(side="SELL", reduces_margin_below_zero=True),
        ]
        assert m.unsafe_buy_count() == 2  # counts all trades with flag

    def test_fraction_unsafe_buys(self):
        m = RunMetrics()
        m.trades = [
            _trade(side="BUY", reduces_margin_below_zero=True),
            _trade(side="BUY", reduces_margin_below_zero=False),
        ]
        assert m.fraction_unsafe_buys() == Decimal("0.5")

    def test_fraction_unsafe_buys_no_buys(self):
        m = RunMetrics()
        m.trades = [_trade(side="SELL")]
        assert m.fraction_unsafe_buys() == Decimal(0)

    def test_margin_at_default_distribution(self):
        m = RunMetrics()
        m.trader_snapshots = [
            _trader_snapshot(defaulted=True, safety_margin=Decimal(-20)),
            _trader_snapshot(defaulted=True, safety_margin=Decimal(-30)),
            _trader_snapshot(defaulted=False, safety_margin=Decimal(50)),
        ]
        margins = m.margin_at_default_distribution()
        assert len(margins) == 2
        assert Decimal(-20) in margins
        assert Decimal(-30) in margins

    def test_mean_margin_at_default(self):
        m = RunMetrics()
        m.trader_snapshots = [
            _trader_snapshot(defaulted=True, safety_margin=Decimal(-20)),
            _trader_snapshot(defaulted=True, safety_margin=Decimal(-30)),
        ]
        assert m.mean_margin_at_default() == Decimal(-25)

    def test_mean_margin_at_default_none(self):
        m = RunMetrics()
        m.trader_snapshots = [
            _trader_snapshot(defaulted=False),
        ]
        assert m.mean_margin_at_default() is None


# ===================================================================
# RunMetrics -- Debt to money ratio
# ===================================================================

class TestRunMetricsDebtToMoney:
    def test_ratio_normal(self):
        m = RunMetrics()
        m.initial_total_debt = Decimal(250)
        m.initial_total_money = Decimal(500)
        assert m.debt_to_money_ratio == Decimal("0.5")

    def test_ratio_zero_money(self):
        m = RunMetrics()
        m.initial_total_debt = Decimal(250)
        m.initial_total_money = Decimal(0)
        assert m.debt_to_money_ratio == Decimal(0)


# ===================================================================
# RunMetrics -- Mid Price Timeseries
# ===================================================================

class TestRunMetricsTimeseries:
    def _metrics_with_multi_day(self):
        m = RunMetrics()
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", midline=Decimal("0.93"), vbt_mid=Decimal("0.95")),
            _snapshot(day=1, bucket="mid", midline=Decimal("0.90"), vbt_mid=Decimal("0.92")),
            _snapshot(day=2, bucket="short", midline=Decimal("0.94"), vbt_mid=Decimal("0.96")),
            _snapshot(day=2, bucket="mid", midline=Decimal("0.91"), vbt_mid=Decimal("0.93")),
        ]
        return m

    def test_dealer_mid_timeseries(self):
        m = self._metrics_with_multi_day()
        ts = m.dealer_mid_timeseries()
        assert ts[1]["short"] == Decimal("0.93")
        assert ts[2]["mid"] == Decimal("0.91")

    def test_vbt_mid_timeseries(self):
        m = self._metrics_with_multi_day()
        ts = m.vbt_mid_timeseries()
        assert ts[1]["short"] == Decimal("0.95")
        assert ts[2]["short"] == Decimal("0.96")

    def test_dealer_premium_timeseries(self):
        m = self._metrics_with_multi_day()
        ts = m.dealer_premium_timeseries()
        # midline=0.93, premium_pct = (0.93-1)*100 = -7
        assert ts[1]["short"] == Decimal("-7")

    def test_vbt_premium_timeseries(self):
        m = self._metrics_with_multi_day()
        ts = m.vbt_premium_timeseries()
        # vbt_mid=0.95, premium_pct = (0.95-1)*100 = -5
        assert ts[1]["short"] == Decimal("-5")

    def test_final_dealer_mids(self):
        m = self._metrics_with_multi_day()
        finals = m._final_dealer_mids()
        assert finals["short"] == float(Decimal("0.94"))
        assert finals["mid"] == float(Decimal("0.91"))

    def test_final_vbt_mids(self):
        m = self._metrics_with_multi_day()
        finals = m._final_vbt_mids()
        assert finals["short"] == float(Decimal("0.96"))
        assert finals["mid"] == float(Decimal("0.93"))

    def test_final_dealer_premiums(self):
        m = self._metrics_with_multi_day()
        finals = m._final_dealer_premiums()
        # day-2 short midline=0.94 => premium_pct = -6
        assert finals["short"] == float(Decimal("-6"))

    def test_final_vbt_premiums(self):
        m = self._metrics_with_multi_day()
        finals = m._final_vbt_premiums()
        # day-2 short vbt_mid=0.96 => premium_pct = -4
        assert finals["short"] == float(Decimal("-4"))

    def test_final_methods_empty_snapshots(self):
        m = RunMetrics()
        assert m._final_dealer_mids() == {}
        assert m._final_vbt_mids() == {}
        assert m._final_dealer_premiums() == {}
        assert m._final_vbt_premiums() == {}


# ===================================================================
# RunMetrics -- Summary
# ===================================================================

class TestRunMetricsSummary:
    def test_summary_keys(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(100)}
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", cash=Decimal(100),
                      vbt_mid=Decimal(1), inventory=0, ticket_size=Decimal(1)),
        ]
        s = m.summary()
        expected = {
            "dealer_total_pnl", "dealer_total_return", "dealer_profitable",
            "dealer_pnl_by_bucket", "dealer_return_by_bucket",
            "spread_income_total", "interior_trades", "passthrough_trades",
            "mean_trader_return", "fraction_profitable_traders",
            "liquidity_driven_sales", "rescue_events",
            "unsafe_buy_count", "fraction_unsafe_buys",
            "mean_margin_at_default",
            "total_trades", "total_sell_trades", "total_buy_trades",
            "initial_total_debt", "initial_total_money", "debt_to_money_ratio",
            "dealer_mid_final", "vbt_mid_final",
            "dealer_premium_final_pct", "vbt_premium_final_pct",
        }
        assert set(s.keys()) == expected

    def test_summary_json_serializable(self):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(100)}
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short"),
        ]
        s = m.summary()
        # Must not raise
        json.dumps(s)

    def test_summary_mean_margin_at_default_none(self):
        m = RunMetrics()
        s = m.summary()
        assert s["mean_margin_at_default"] is None


# ===================================================================
# RunMetrics -- CSV / JSON Exports
# ===================================================================

class TestRunMetricsExports:
    def test_to_trade_log_csv(self, tmp_path):
        m = RunMetrics()
        m.trades = [_trade(), _trade(day=2, trader_id="h1")]
        path = str(tmp_path / "trades.csv")
        m.to_trade_log_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_to_trade_log_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "trades.csv")
        m.to_trade_log_csv(path)
        assert not Path(path).exists()

    def test_to_dealer_snapshots_csv(self, tmp_path):
        m = RunMetrics()
        m.dealer_snapshots = [_snapshot(day=1), _snapshot(day=2)]
        path = str(tmp_path / "dealer.csv")
        m.to_dealer_snapshots_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_to_dealer_snapshots_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "dealer.csv")
        m.to_dealer_snapshots_csv(path)
        assert not Path(path).exists()

    def test_to_trader_snapshots_csv(self, tmp_path):
        m = RunMetrics()
        m.trader_snapshots = [_trader_snapshot(), _trader_snapshot(trader_id="h1")]
        path = str(tmp_path / "traders.csv")
        m.to_trader_snapshots_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        # tickets_held_ids should be excluded from CSV
        assert "tickets_held_ids" not in rows[0]

    def test_to_trader_snapshots_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "traders.csv")
        m.to_trader_snapshots_csv(path)
        assert not Path(path).exists()

    def test_to_ticket_outcomes_csv(self, tmp_path):
        m = RunMetrics()
        m.ticket_outcomes = {
            "T1": _ticket_outcome(ticket_id="T1"),
            "T2": _ticket_outcome(ticket_id="T2"),
        }
        path = str(tmp_path / "outcomes.csv")
        m.to_ticket_outcomes_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_to_ticket_outcomes_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "outcomes.csv")
        m.to_ticket_outcomes_csv(path)
        assert not Path(path).exists()

    def test_to_summary_json(self, tmp_path):
        m = RunMetrics()
        m.initial_equity_by_bucket = {"short": Decimal(100)}
        m.dealer_snapshots = [_snapshot(day=1, bucket="short")]
        path = str(tmp_path / "summary.json")
        m.to_summary_json(path)

        with open(path) as f:
            data = json.load(f)
        assert "dealer_total_pnl" in data

    def test_to_mid_timeseries_csv(self, tmp_path):
        m = RunMetrics()
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short"),
            _snapshot(day=2, bucket="mid"),
        ]
        path = str(tmp_path / "mid.csv")
        m.to_mid_timeseries_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "dealer_midline" in rows[0]
        assert "vbt_mid" in rows[0]

    def test_to_mid_timeseries_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "mid.csv")
        m.to_mid_timeseries_csv(path)
        assert not Path(path).exists()

    def test_to_inventory_timeseries_csv(self, tmp_path):
        m = RunMetrics()
        m.dealer_snapshots = [
            _snapshot(day=1, bucket="short", run_id="r1", regime="active",
                      max_capacity=10, is_at_zero=False, hit_vbt_this_step=False,
                      total_system_face=Decimal(500), dealer_share_pct=Decimal("2.0")),
            _snapshot(day=2, bucket="short"),
        ]
        path = str(tmp_path / "inventory.csv")
        m.to_inventory_timeseries_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "dealer_inventory" in rows[0]
        assert "capacity_pct" in rows[0]

    def test_to_inventory_timeseries_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "inventory.csv")
        m.to_inventory_timeseries_csv(path)
        assert not Path(path).exists()

    def test_to_system_state_csv(self, tmp_path):
        m = RunMetrics()
        m.system_state_snapshots = [
            SystemStateSnapshot(
                run_id="r1", regime="active", day=1,
                total_face_value=Decimal(1000), total_cash=Decimal(500),
            ),
            SystemStateSnapshot(
                run_id="r1", regime="active", day=2,
                total_face_value=Decimal(900), total_cash=Decimal(500),
            ),
        ]
        path = str(tmp_path / "system_state.csv")
        m.to_system_state_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["day"] == "1"

    def test_to_system_state_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "system_state.csv")
        m.to_system_state_csv(path)
        assert not Path(path).exists()

    def test_to_repayment_events_csv(self, tmp_path):
        m = RunMetrics()
        m.repayment_events = [
            RepaymentEvent(
                run_id="r1", regime="active",
                trader_id="h0", liability_id="PAY_1",
                maturity_day=5, face_value=Decimal(50),
                outcome="repaid", strategy="no_trade",
            ),
            RepaymentEvent(
                run_id="r1", regime="active",
                trader_id="h1", liability_id="PAY_2",
                maturity_day=3, face_value=Decimal(30),
                outcome="defaulted", strategy="sell_before",
            ),
        ]
        path = str(tmp_path / "repayment.csv")
        m.to_repayment_events_csv(path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        # Sorted by maturity_day then trader_id: day3/h1 first, day5/h0 second
        assert len(rows) == 2
        assert rows[0]["maturity_day"] == "3"

    def test_to_repayment_events_csv_empty(self, tmp_path):
        m = RunMetrics()
        path = str(tmp_path / "repayment.csv")
        m.to_repayment_events_csv(path)
        assert not Path(path).exists()

    def test_export_creates_parent_dirs(self, tmp_path):
        m = RunMetrics()
        m.trades = [_trade()]
        path = str(tmp_path / "sub" / "dir" / "trades.csv")
        m.to_trade_log_csv(path)
        assert Path(path).exists()


# ===================================================================
# Helper: compute_safety_margin
# ===================================================================

class _FakeTicket:
    """Minimal ticket-like object for safety margin tests."""
    def __init__(self, bucket_id, face):
        self.bucket_id = bucket_id
        self.face = face


class _FakeDealer:
    """Minimal dealer-like object for safety margin tests."""
    def __init__(self, bid):
        self.bid = bid


class TestComputeSafetyMargin:
    def test_basic(self):
        cash = Decimal(100)
        tickets = [_FakeTicket("short", Decimal(10))]
        obligations = [_FakeTicket("short", Decimal(50))]
        dealers = {"short": _FakeDealer(Decimal("0.90"))}
        margin = compute_safety_margin(cash, tickets, obligations, dealers, Decimal(1))
        # A = 100 + 0.90 * 10 = 109
        # D = 50
        # m = 109 - 50 = 59
        assert margin == Decimal(59)

    def test_no_dealer_for_bucket(self):
        """Falls back to face value if no dealer for bucket."""
        cash = Decimal(100)
        tickets = [_FakeTicket("long", Decimal(10))]
        obligations = []
        dealers = {}
        margin = compute_safety_margin(cash, tickets, obligations, dealers, Decimal(1))
        # A = 100 + 10 (face fallback)
        assert margin == Decimal(110)

    def test_empty(self):
        margin = compute_safety_margin(Decimal(50), [], [], {}, Decimal(1))
        assert margin == Decimal(50)


class TestComputeSaleableValue:
    def test_with_dealer(self):
        tickets = [
            _FakeTicket("short", Decimal(10)),
            _FakeTicket("mid", Decimal(20)),
        ]
        dealers = {
            "short": _FakeDealer(Decimal("0.90")),
            "mid": _FakeDealer(Decimal("0.85")),
        }
        value = compute_saleable_value(tickets, dealers)
        # 0.90*10 + 0.85*20 = 9 + 17 = 26
        assert value == Decimal(26)

    def test_no_dealer_fallback(self):
        tickets = [_FakeTicket("long", Decimal(10))]
        value = compute_saleable_value(tickets, {})
        assert value == Decimal(10)

    def test_empty(self):
        assert compute_saleable_value([], {}) == Decimal(0)


# ===================================================================
# LiabilityInfo
# ===================================================================

class TestLiabilityInfo:
    def test_defaults(self):
        li = LiabilityInfo(
            trader_id="h0", liability_id="PAY_1",
            maturity_day=5, face_value=Decimal(50),
        )
        assert li.settled is False
        assert li.settled_day is None


# ===================================================================
# build_liability_map
# ===================================================================

class TestBuildLiabilityMap:
    def test_basic(self):
        events = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 5, "amount": 50},
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_2",
             "debtor": "h1", "due_day": 7, "amount": 30},
            {"kind": "PayableSettled", "day": 5, "payable_id": "PAY_1"},
        ]
        liabilities, final_day = build_liability_map(events)
        assert len(liabilities) == 2
        assert liabilities["PAY_1"].settled is True
        assert liabilities["PAY_1"].settled_day == 5
        assert liabilities["PAY_2"].settled is False
        assert final_day == 5

    def test_settled_via_pid(self):
        events = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 5, "amount": 50},
            {"kind": "PayableSettled", "day": 5, "pid": "PAY_1"},
        ]
        liabilities, _ = build_liability_map(events)
        assert liabilities["PAY_1"].settled is True

    def test_settled_via_contract_id(self):
        events = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 5, "amount": 50},
            {"kind": "PayableSettled", "day": 5, "contract_id": "PAY_1"},
        ]
        liabilities, _ = build_liability_map(events)
        assert liabilities["PAY_1"].settled is True

    def test_empty_events(self):
        liabilities, final_day = build_liability_map([])
        assert len(liabilities) == 0
        assert final_day == 0

    def test_event_field_alias(self):
        """Tests the 'event' field name fallback (used in some logs)."""
        events = [
            {"event": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 5, "amount": 50},
        ]
        liabilities, _ = build_liability_map(events)
        assert "PAY_1" in liabilities

    def test_settle_unknown_id_ignored(self):
        events = [
            {"kind": "PayableSettled", "day": 5, "payable_id": "UNKNOWN"},
        ]
        liabilities, _ = build_liability_map(events)
        assert len(liabilities) == 0

    def test_missing_payable_id_skipped(self):
        events = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "",
             "debtor": "h0", "due_day": 5, "amount": 50},
        ]
        liabilities, _ = build_liability_map(events)
        assert len(liabilities) == 0


# ===================================================================
# compute_trading_stats_by_trader
# ===================================================================

class TestComputeTradingStatsByTrader:
    def test_basic(self):
        trades = [
            _trade(day=1, trader_id="h0", side="SELL", price=Decimal("0.90")),
            _trade(day=2, trader_id="h0", side="BUY", price=Decimal("0.95")),
            _trade(day=1, trader_id="h1", side="SELL", price=Decimal("0.85")),
        ]
        stats = compute_trading_stats_by_trader(trades)
        assert len(stats["h0"]) == 2
        assert len(stats["h1"]) == 1
        assert stats["h0"][0]["side"] == "SELL"

    def test_empty(self):
        assert compute_trading_stats_by_trader([]) == {}


# ===================================================================
# get_trades_before_day
# ===================================================================

class TestGetTradesBeforeDay:
    def test_filters_by_day(self):
        trader_trades = [
            {"day": 1, "side": "BUY", "price": Decimal("0.95")},
            {"day": 3, "side": "SELL", "price": Decimal("0.90")},
            {"day": 5, "side": "BUY", "price": Decimal("0.96")},
        ]
        stats = get_trades_before_day(trader_trades, 4)
        assert stats["buy_count"] == 1
        assert stats["sell_count"] == 1
        assert stats["net_cash_pnl"] == Decimal("0.90") - Decimal("0.95")

    def test_none_before(self):
        stats = get_trades_before_day(
            [{"day": 5, "side": "BUY", "price": Decimal("0.95")}],
            3,
        )
        assert stats["buy_count"] == 0
        assert stats["sell_count"] == 0
        assert stats["net_cash_pnl"] == Decimal(0)

    def test_empty(self):
        stats = get_trades_before_day([], 10)
        assert stats["buy_count"] == 0


# ===================================================================
# build_repayment_events
# ===================================================================

class TestBuildRepaymentEvents:
    def test_basic_repaid_and_defaulted(self):
        event_log = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 5, "amount": 50},
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_2",
             "debtor": "h1", "due_day": 4, "amount": 30},
            {"kind": "PayableSettled", "day": 5, "payable_id": "PAY_1"},
            # PAY_2 not settled => defaulted
            {"kind": "day_end", "day": 6},  # ensure final_day >= maturity
        ]
        trades = [
            _trade(day=2, trader_id="h0", side="SELL"),
            _trade(day=3, trader_id="h1", side="BUY"),
        ]
        events = build_repayment_events(event_log, trades, run_id="r1", regime="active")
        assert len(events) == 2
        by_id = {e.liability_id: e for e in events}
        assert by_id["PAY_1"].outcome == "repaid"
        assert by_id["PAY_2"].outcome == "defaulted"
        # h0 sold before day 5
        assert by_id["PAY_1"].sell_count == 1
        # h1 bought before day 4
        assert by_id["PAY_2"].buy_count == 1
        assert by_id["PAY_2"].strategy == "hold_to_maturity"

    def test_future_maturity_excluded(self):
        """Liabilities with maturity > final_day are not included."""
        event_log = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 100, "amount": 50},
            {"kind": "day_end", "day": 5},
        ]
        events = build_repayment_events(event_log, [])
        assert len(events) == 0

    def test_empty_trader_id_skipped(self):
        event_log = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "", "due_day": 3, "amount": 50},
            {"kind": "day_end", "day": 5},
        ]
        events = build_repayment_events(event_log, [])
        assert len(events) == 0

    def test_no_events(self):
        events = build_repayment_events([], [])
        assert events == []

    def test_run_id_and_regime_propagated(self):
        event_log = [
            {"kind": "PayableCreated", "day": 1, "payable_id": "PAY_1",
             "debtor": "h0", "due_day": 3, "amount": 50},
            {"kind": "day_end", "day": 5},
        ]
        events = build_repayment_events(event_log, [], run_id="test_run", regime="passive")
        assert events[0].run_id == "test_run"
        assert events[0].regime == "passive"


# ===================================================================
# RunMetrics -- Empty state (edge cases)
# ===================================================================

class TestRunMetricsEmpty:
    def test_empty_metrics(self):
        m = RunMetrics()
        assert m.total_dealer_pnl() == Decimal(0)
        assert m.total_dealer_return() == Decimal(0)
        assert m.is_dealer_profitable() is True
        assert m.spread_income_total() == Decimal(0)
        assert m.passthrough_count() == 0
        assert m.interior_count() == 0
        assert m.trader_returns() == {}
        assert m.mean_trader_return() == Decimal(0)
        assert m.fraction_profitable_traders() == Decimal(0)
        assert m.liquidity_driven_sales() == 0
        assert m.rescue_events() == 0
        assert m.unsafe_buy_count() == 0
        assert m.fraction_unsafe_buys() == Decimal(0)
        assert m.margin_at_default_distribution() == []
        assert m.mean_margin_at_default() is None
        assert m.dealer_mid_timeseries() == {}
        assert m.vbt_mid_timeseries() == {}
        assert m.dealer_premium_timeseries() == {}
        assert m.vbt_premium_timeseries() == {}

    def test_summary_on_empty(self):
        m = RunMetrics()
        s = m.summary()
        assert s["total_trades"] == 0
        assert s["mean_margin_at_default"] is None
        # Must be JSON serializable
        json.dumps(s)


# ===================================================================
# RunMetrics -- run_id / regime
# ===================================================================

class TestRunMetricsContext:
    def test_run_context(self):
        m = RunMetrics(run_id="test_run", regime="passive")
        assert m.run_id == "test_run"
        assert m.regime == "passive"
