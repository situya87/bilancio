"""
Tests for dealer.models and dealer.events modules.

Covers:
- Ticket, BucketConfig, DealerState, VBTState, TraderState dataclasses
- EventLog recording, indexing, serialization, and query methods
"""

import json
from decimal import Decimal

import pytest

from bilancio.dealer.events import EventLog
from bilancio.dealer.models import (
    DEFAULT_BUCKETS,
    BucketConfig,
    DealerState,
    Ticket,
    TraderState,
    VBTState,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ticket(
    tid: str = "t1",
    issuer: str = "issuer_a",
    owner: str = "owner_b",
    face: Decimal = Decimal(1),
    maturity_day: int = 5,
    remaining_tau: int = 3,
    bucket_id: str = "short",
    serial: int = 0,
) -> Ticket:
    return Ticket(
        id=tid,
        issuer_id=issuer,
        owner_id=owner,
        face=face,
        maturity_day=maturity_day,
        remaining_tau=remaining_tau,
        bucket_id=bucket_id,
        serial=serial,
    )


# ===================================================================
# BucketConfig
# ===================================================================


class TestBucketConfig:
    def test_creation(self):
        bc = BucketConfig(name="short", tau_min=1, tau_max=3)
        assert bc.name == "short"
        assert bc.tau_min == 1
        assert bc.tau_max == 3

    def test_unbounded(self):
        bc = BucketConfig(name="long", tau_min=9, tau_max=None)
        assert bc.tau_max is None

    def test_default_buckets_structure(self):
        assert len(DEFAULT_BUCKETS) == 3
        names = [b.name for b in DEFAULT_BUCKETS]
        assert names == ["short", "mid", "long"]
        # short covers 1-3
        assert DEFAULT_BUCKETS[0].tau_min == 1
        assert DEFAULT_BUCKETS[0].tau_max == 3
        # mid covers 4-8
        assert DEFAULT_BUCKETS[1].tau_min == 4
        assert DEFAULT_BUCKETS[1].tau_max == 8
        # long is unbounded from 9
        assert DEFAULT_BUCKETS[2].tau_min == 9
        assert DEFAULT_BUCKETS[2].tau_max is None


# ===================================================================
# Ticket
# ===================================================================


class TestTicket:
    def test_creation_with_all_fields(self):
        t = _ticket()
        assert t.id == "t1"
        assert t.issuer_id == "issuer_a"
        assert t.owner_id == "owner_b"
        assert t.face == Decimal(1)
        assert t.maturity_day == 5
        assert t.remaining_tau == 3
        assert t.bucket_id == "short"
        assert t.serial == 0

    def test_defaults(self):
        t = Ticket(id="x", issuer_id="i", owner_id="o", face=Decimal(10), maturity_day=7)
        assert t.remaining_tau == 0
        assert t.bucket_id == ""
        assert t.serial == 0

    def test_face_is_decimal(self):
        t = _ticket(face=Decimal("1.50"))
        assert isinstance(t.face, Decimal)
        assert t.face == Decimal("1.50")


# ===================================================================
# DealerState
# ===================================================================


class TestDealerState:
    def test_defaults(self):
        ds = DealerState(bucket_id="short")
        assert ds.bucket_id == "short"
        assert ds.agent_id == ""
        assert ds.inventory == []
        assert ds.cash == Decimal(0)
        assert ds.a == 0
        assert ds.x == Decimal(0)
        assert ds.V == Decimal(0)
        assert ds.K_star == 0
        assert ds.X_star == Decimal(0)
        assert ds.N == 1
        assert ds.lambda_ == Decimal(1)
        assert ds.I == Decimal(0)
        assert ds.bid == Decimal(0)
        assert ds.ask == Decimal(0)
        assert ds.midline == Decimal(0)
        assert ds.is_pinned_bid is False
        assert ds.is_pinned_ask is False

    def test_ticket_ids_by_issuer_empty(self):
        ds = DealerState(bucket_id="short")
        assert ds.ticket_ids_by_issuer() == {}

    def test_ticket_ids_by_issuer_single_issuer(self):
        t1 = _ticket(tid="t1", issuer="A")
        t2 = _ticket(tid="t2", issuer="A")
        ds = DealerState(bucket_id="short", inventory=[t1, t2])
        result = ds.ticket_ids_by_issuer()
        assert result == {"A": ["t1", "t2"]}

    def test_ticket_ids_by_issuer_multiple_issuers(self):
        t1 = _ticket(tid="t1", issuer="A")
        t2 = _ticket(tid="t2", issuer="B")
        t3 = _ticket(tid="t3", issuer="A")
        ds = DealerState(bucket_id="mid", inventory=[t1, t2, t3])
        result = ds.ticket_ids_by_issuer()
        assert set(result.keys()) == {"A", "B"}
        assert result["A"] == ["t1", "t3"]
        assert result["B"] == ["t2"]

    def test_inventory_independence(self):
        """Each DealerState should have its own inventory list."""
        ds1 = DealerState(bucket_id="short")
        ds2 = DealerState(bucket_id="mid")
        ds1.inventory.append(_ticket())
        assert len(ds2.inventory) == 0

    def test_custom_values(self):
        ds = DealerState(
            bucket_id="long",
            agent_id="dealer_1",
            cash=Decimal("500"),
            a=3,
            x=Decimal(3),
            bid=Decimal("0.85"),
            ask=Decimal("0.95"),
            midline=Decimal("0.90"),
            is_pinned_bid=True,
            is_pinned_ask=True,
        )
        assert ds.agent_id == "dealer_1"
        assert ds.cash == Decimal("500")
        assert ds.a == 3
        assert ds.bid == Decimal("0.85")
        assert ds.ask == Decimal("0.95")
        assert ds.midline == Decimal("0.90")
        assert ds.is_pinned_bid is True
        assert ds.is_pinned_ask is True


# ===================================================================
# VBTState
# ===================================================================


class TestVBTState:
    def test_defaults(self):
        v = VBTState(bucket_id="short")
        assert v.bucket_id == "short"
        assert v.agent_id == ""
        assert v.M == Decimal(1)
        assert v.O == Decimal("0.30")
        assert v.A == Decimal(0)
        assert v.B == Decimal(0)
        assert v.phi_M == Decimal(1)
        assert v.phi_O == Decimal("0.6")
        assert v.O_min == Decimal(0)
        assert v.clip_nonneg_B is True
        assert v.inventory == []
        assert v.cash == Decimal(0)

    def test_recompute_quotes_basic(self):
        v = VBTState(bucket_id="short", M=Decimal("0.80"), O=Decimal("0.30"))
        v.recompute_quotes()
        assert v.A == Decimal("0.80") + Decimal("0.15")  # 0.95
        assert v.B == Decimal("0.80") - Decimal("0.15")  # 0.65

    def test_recompute_quotes_minimum_spread(self):
        """O_min enforces a floor on the effective spread."""
        v = VBTState(bucket_id="short", M=Decimal("0.80"), O=Decimal("0.01"), O_min=Decimal("0.10"))
        v.recompute_quotes()
        # O_min = 0.10 takes effect since O < O_min
        assert v.A == Decimal("0.80") + Decimal("0.05")  # 0.85
        assert v.B == Decimal("0.80") - Decimal("0.05")  # 0.75

    def test_recompute_quotes_nonneg_bid_clipping(self):
        """B is clipped to 0 when M is very low."""
        v = VBTState(bucket_id="short", M=Decimal("0.05"), O=Decimal("0.30"))
        v.recompute_quotes()
        assert v.B == Decimal(0)  # 0.05 - 0.15 = -0.10 -> clipped to 0
        assert v.A == Decimal("0.05") + Decimal("0.15")

    def test_recompute_quotes_nonneg_clipping_disabled(self):
        """With clip_nonneg_B=False, B can go negative."""
        v = VBTState(
            bucket_id="short",
            M=Decimal("0.05"),
            O=Decimal("0.30"),
            clip_nonneg_B=False,
        )
        v.recompute_quotes()
        assert v.B == Decimal("0.05") - Decimal("0.15")  # -0.10
        assert v.B < Decimal(0)

    def test_update_from_loss(self):
        v = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.30"),
            phi_M=Decimal(1),
            phi_O=Decimal("0.6"),
        )
        loss = Decimal("0.10")
        v.update_from_loss(loss)
        # M_new = 0.80 - 1 * 0.10 = 0.70
        assert v.M == Decimal("0.70")
        # O_new = max(0, 0.30 + 0.6 * 0.10) = 0.36
        assert v.O == Decimal("0.36")
        # Quotes are recomputed
        assert v.A == Decimal("0.70") + Decimal("0.36") / 2  # 0.88
        assert v.B == Decimal("0.70") - Decimal("0.36") / 2  # 0.52

    def test_update_from_loss_respects_o_min(self):
        v = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.05"),
            phi_M=Decimal(1),
            phi_O=Decimal("0.6"),
            O_min=Decimal("0.10"),
        )
        v.update_from_loss(Decimal(0))
        # O_new = max(O_min=0.10, 0.05 + 0.6*0) = max(0.10, 0.05) = 0.10
        assert v.O == Decimal("0.10")
        # M unchanged: 0.80 - 1*0 = 0.80
        assert v.M == Decimal("0.80")
        # Quotes: A = 0.80 + 0.10/2 = 0.85, B = 0.80 - 0.10/2 = 0.75
        assert v.A == Decimal("0.85")
        assert v.B == Decimal("0.75")

    def test_update_from_loss_large_loss_clips_bid_to_zero(self):
        v = VBTState(bucket_id="short", M=Decimal("0.50"), O=Decimal("0.30"))
        # Large loss drives M down so B goes negative
        v.update_from_loss(Decimal("0.50"))
        # M_new = 0.50 - 0.50 = 0
        assert v.M == Decimal(0)
        # O_new = max(0, 0.30 + 0.6*0.50) = 0.60
        assert v.O == Decimal("0.60")
        # B = 0 - 0.30 = -0.30 -> clipped to 0
        assert v.B == Decimal(0)

    def test_multiple_losses_accumulate(self):
        v = VBTState(bucket_id="short", M=Decimal(1), O=Decimal("0.30"))
        v.update_from_loss(Decimal("0.10"))
        v.update_from_loss(Decimal("0.10"))
        # M = 1 - 0.10 - 0.10 = 0.80
        assert v.M == Decimal("0.80")
        # O = 0.30 + 0.06 + 0.06 = 0.42
        assert v.O == Decimal("0.42")

    def test_inventory_independence(self):
        v1 = VBTState(bucket_id="short")
        v2 = VBTState(bucket_id="mid")
        v1.inventory.append(_ticket())
        assert len(v2.inventory) == 0


# ===================================================================
# TraderState
# ===================================================================


class TestTraderState:
    def test_defaults(self):
        ts = TraderState(agent_id="trader_1")
        assert ts.agent_id == "trader_1"
        assert ts.cash == Decimal(0)
        assert ts.tickets_owned == []
        assert ts.obligations == []
        assert ts.asset_issuer_id is None
        assert ts.defaulted is False

    def test_payment_due_no_obligations(self):
        ts = TraderState(agent_id="t1")
        assert ts.payment_due(day=5) == Decimal(0)

    def test_payment_due_single_obligation(self):
        ob = _ticket(tid="ob1", issuer="t1", face=Decimal(100), maturity_day=5)
        ts = TraderState(agent_id="t1", obligations=[ob])
        assert ts.payment_due(day=5) == Decimal(100)
        assert ts.payment_due(day=4) == Decimal(0)

    def test_payment_due_multiple_obligations_same_day(self):
        ob1 = _ticket(tid="ob1", face=Decimal(100), maturity_day=5)
        ob2 = _ticket(tid="ob2", face=Decimal(50), maturity_day=5)
        ob3 = _ticket(tid="ob3", face=Decimal(25), maturity_day=7)
        ts = TraderState(agent_id="t1", obligations=[ob1, ob2, ob3])
        assert ts.payment_due(day=5) == Decimal(150)
        assert ts.payment_due(day=7) == Decimal(25)

    def test_shortfall_zero_when_enough_cash(self):
        ob = _ticket(tid="ob1", face=Decimal(50), maturity_day=3)
        ts = TraderState(agent_id="t1", cash=Decimal(100), obligations=[ob])
        assert ts.shortfall(day=3) == Decimal(0)

    def test_shortfall_positive_when_insufficient_cash(self):
        ob = _ticket(tid="ob1", face=Decimal(100), maturity_day=3)
        ts = TraderState(agent_id="t1", cash=Decimal(40), obligations=[ob])
        assert ts.shortfall(day=3) == Decimal(60)

    def test_shortfall_zero_when_no_obligations(self):
        ts = TraderState(agent_id="t1", cash=Decimal(100))
        assert ts.shortfall(day=5) == Decimal(0)

    def test_shortfall_exact_cash(self):
        ob = _ticket(tid="ob1", face=Decimal(100), maturity_day=3)
        ts = TraderState(agent_id="t1", cash=Decimal(100), obligations=[ob])
        assert ts.shortfall(day=3) == Decimal(0)

    def test_earliest_liability_day(self):
        ob1 = _ticket(tid="ob1", maturity_day=5)
        ob2 = _ticket(tid="ob2", maturity_day=10)
        ob3 = _ticket(tid="ob3", maturity_day=3)
        ts = TraderState(agent_id="t1", obligations=[ob1, ob2, ob3])
        assert ts.earliest_liability_day(after_day=0) == 3
        assert ts.earliest_liability_day(after_day=3) == 5
        assert ts.earliest_liability_day(after_day=5) == 10
        assert ts.earliest_liability_day(after_day=10) is None

    def test_earliest_liability_day_no_obligations(self):
        ts = TraderState(agent_id="t1")
        assert ts.earliest_liability_day(after_day=0) is None

    def test_defaulted_flag(self):
        ts = TraderState(agent_id="t1", defaulted=True)
        assert ts.defaulted is True

    def test_list_independence(self):
        ts1 = TraderState(agent_id="t1")
        ts2 = TraderState(agent_id="t2")
        ts1.tickets_owned.append(_ticket())
        ts1.obligations.append(_ticket(tid="ob"))
        assert len(ts2.tickets_owned) == 0
        assert len(ts2.obligations) == 0


# ===================================================================
# EventLog
# ===================================================================


class TestEventLog:
    # --- initialization -------------------------------------------------

    def test_empty_log(self):
        log = EventLog()
        assert log.events == []
        assert log.defaults_by_day == {}
        assert log.trades_by_day == {}
        assert log.settlements_by_day == {}

    # --- generic log method ---------------------------------------------

    def test_log_basic(self):
        log = EventLog()
        log.log("custom_event", day=1, foo="bar", count=42)
        assert len(log.events) == 1
        e = log.events[0]
        assert e["kind"] == "custom_event"
        assert e["day"] == 1
        assert e["foo"] == "bar"
        assert e["count"] == 42

    def test_log_serializes_decimals(self):
        log = EventLog()
        log.log("x", day=1, price=Decimal("1.50"), nested={"val": Decimal("0.5")})
        e = log.events[0]
        assert e["price"] == "1.50"
        assert e["nested"]["val"] == "0.5"

    def test_log_serializes_lists_of_decimals(self):
        log = EventLog()
        log.log("x", day=1, prices=[Decimal(1), Decimal(2)])
        e = log.events[0]
        assert e["prices"] == ["1", "2"]

    def test_log_serializes_tuples(self):
        log = EventLog()
        log.log("x", day=1, pair=(Decimal("0.5"), "text"))
        e = log.events[0]
        assert e["pair"] == ["0.5", "text"]

    # --- day_start ------------------------------------------------------

    def test_log_day_start(self):
        log = EventLog()
        log.log_day_start(day=1)
        assert len(log.events) == 1
        assert log.events[0]["kind"] == "day_start"
        assert log.events[0]["day"] == 1

    # --- trade ----------------------------------------------------------

    def test_log_trade(self):
        log = EventLog()
        log.log_trade(
            day=2,
            side="BUY",
            trader_id="tr_1",
            ticket_id="tk_1",
            bucket="short",
            price=Decimal("0.90"),
            is_passthrough=False,
        )
        assert len(log.events) == 1
        e = log.events[0]
        assert e["kind"] == "trade"
        assert e["day"] == 2
        assert e["side"] == "BUY"
        assert e["trader_id"] == "tr_1"
        assert e["ticket_id"] == "tk_1"
        assert e["bucket"] == "short"
        assert e["price"] == "0.90"
        assert e["is_passthrough"] is False

    def test_log_trade_indexes_by_day(self):
        log = EventLog()
        log.log_trade(
            day=3,
            side="SELL",
            trader_id="t",
            ticket_id="tk",
            bucket="mid",
            price=Decimal(1),
            is_passthrough=True,
        )
        assert 3 in log.trades_by_day
        assert len(log.trades_by_day[3]) == 1

    def test_log_trade_passthrough(self):
        log = EventLog()
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk",
            bucket="short",
            price=Decimal("0.80"),
            is_passthrough=True,
        )
        assert log.events[0]["is_passthrough"] is True

    # --- sell_rejected / buy_rejected -----------------------------------

    def test_log_sell_rejected(self):
        log = EventLog()
        log.log_sell_rejected(
            day=1,
            trader_id="tr",
            ticket_id="tk",
            bucket="mid",
            offered_price=Decimal("0.70"),
            expected_value=Decimal("0.85"),
            threshold=Decimal("0.02"),
            reason="price_below_ev_plus_threshold",
        )
        e = log.events[0]
        assert e["kind"] == "sell_rejected"
        assert e["offered_price"] == "0.70"
        assert e["expected_value"] == "0.85"
        assert e["threshold"] == "0.02"
        assert e["reason"] == "price_below_ev_plus_threshold"

    def test_log_buy_rejected(self):
        log = EventLog()
        log.log_buy_rejected(
            day=1,
            trader_id="tr",
            ticket_id="tk",
            bucket="short",
            offered_price=Decimal("0.95"),
            expected_value=Decimal("0.80"),
            threshold=Decimal("0.03"),
            reason="ev_below_price_plus_threshold",
        )
        e = log.events[0]
        assert e["kind"] == "buy_rejected"
        assert e["offered_price"] == "0.95"
        assert e["expected_value"] == "0.80"
        assert e["reason"] == "ev_below_price_plus_threshold"

    # --- quote ----------------------------------------------------------

    def test_log_quote(self):
        log = EventLog()
        log.log_quote(
            day=5,
            bucket="long",
            dealer_bid=Decimal("0.80"),
            dealer_ask=Decimal("0.95"),
            vbt_bid=Decimal("0.75"),
            vbt_ask=Decimal("1.00"),
            inventory=3,
            capacity=Decimal(10),
            is_pinned_bid=True,
            is_pinned_ask=False,
        )
        e = log.events[0]
        assert e["kind"] == "quote"
        assert e["bucket"] == "long"
        assert e["dealer_bid"] == "0.80"
        assert e["dealer_ask"] == "0.95"
        assert e["vbt_bid"] == "0.75"
        assert e["vbt_ask"] == "1.00"
        assert e["inventory"] == 3
        assert e["capacity"] == "10"
        assert e["is_pinned_bid"] is True
        assert e["is_pinned_ask"] is False

    # --- settlement -----------------------------------------------------

    def test_log_settlement(self):
        log = EventLog()
        log.log_settlement(day=4, issuer_id="iss_1", total_paid=Decimal(200), n_tickets=2)
        e = log.events[0]
        assert e["kind"] == "settlement"
        assert e["issuer_id"] == "iss_1"
        assert e["total_paid"] == "200"
        assert e["n_tickets"] == 2
        assert e["recovery_rate"] == "1"

    def test_log_settlement_indexes_by_day(self):
        log = EventLog()
        log.log_settlement(day=4, issuer_id="iss", total_paid=Decimal(100), n_tickets=1)
        assert 4 in log.settlements_by_day
        assert len(log.settlements_by_day[4]) == 1

    # --- default --------------------------------------------------------

    def test_log_default(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss_2",
            recovery_rate=Decimal("0.40"),
            total_due=Decimal(100),
            total_paid=Decimal(40),
            n_tickets=5,
            bucket="short",
        )
        e = log.events[0]
        assert e["kind"] == "default"
        assert e["issuer_id"] == "iss_2"
        assert e["recovery_rate"] == "0.40"
        assert e["total_due"] == "100"
        assert e["total_paid"] == "40"
        assert e["n_tickets"] == 5
        assert e["bucket"] == "short"

    def test_log_default_indexes_by_day(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(100),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert 3 in log.defaults_by_day
        assert len(log.defaults_by_day[3]) == 1

    # --- rebucket -------------------------------------------------------

    def test_log_rebucket(self):
        log = EventLog()
        log.log_rebucket(
            day=6,
            ticket_id="tk_5",
            old_bucket="mid",
            new_bucket="short",
            price=Decimal("0.92"),
            holder_type="dealer",
        )
        e = log.events[0]
        assert e["kind"] == "rebucket"
        assert e["ticket_id"] == "tk_5"
        assert e["old_bucket"] == "mid"
        assert e["new_bucket"] == "short"
        assert e["price"] == "0.92"
        assert e["holder_type"] == "dealer"

    # --- vbt anchor update ----------------------------------------------

    def test_log_vbt_anchor_update(self):
        log = EventLog()
        log.log_vbt_anchor_update(
            day=4,
            bucket="short",
            M_old=Decimal(1),
            M_new=Decimal("0.90"),
            O_old=Decimal("0.30"),
            O_new=Decimal("0.36"),
            loss_rate=Decimal("0.10"),
        )
        e = log.events[0]
        assert e["kind"] == "vbt_anchor_update"
        assert e["M_old"] == "1"
        assert e["M_new"] == "0.90"
        assert e["O_old"] == "0.30"
        assert e["O_new"] == "0.36"
        assert e["loss_rate"] == "0.10"

    # --- get_bucket_loss_rate -------------------------------------------

    def test_get_bucket_loss_rate_no_defaults(self):
        log = EventLog()
        assert log.get_bucket_loss_rate(day=1, bucket_id="short") == Decimal(0)

    def test_get_bucket_loss_rate_full_default(self):
        """Recovery 0 => loss rate 1."""
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(100),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert log.get_bucket_loss_rate(day=3, bucket_id="short") == Decimal(1)

    def test_get_bucket_loss_rate_partial_recovery(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal("0.40"),
            total_due=Decimal(100),
            total_paid=Decimal(40),
            n_tickets=1,
            bucket="short",
        )
        # loss = 100 - 40 = 60, face = 100, rate = 0.60
        assert log.get_bucket_loss_rate(day=3, bucket_id="short") == Decimal("0.6")

    def test_get_bucket_loss_rate_wrong_bucket(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(100),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert log.get_bucket_loss_rate(day=3, bucket_id="mid") == Decimal(0)

    def test_get_bucket_loss_rate_wrong_day(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(100),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert log.get_bucket_loss_rate(day=4, bucket_id="short") == Decimal(0)

    def test_get_bucket_loss_rate_multiple_defaults(self):
        log = EventLog()
        # default 1: due 100, paid 60 => loss 40
        log.log_default(
            day=3,
            issuer_id="A",
            recovery_rate=Decimal("0.60"),
            total_due=Decimal(100),
            total_paid=Decimal(60),
            n_tickets=1,
            bucket="short",
        )
        # default 2: due 200, paid 100 => loss 100
        log.log_default(
            day=3,
            issuer_id="B",
            recovery_rate=Decimal("0.50"),
            total_due=Decimal(200),
            total_paid=Decimal(100),
            n_tickets=2,
            bucket="short",
        )
        # total_loss = 40 + 100 = 140, total_face = 100 + 200 = 300
        # loss_rate = 140/300
        expected = Decimal(140) / Decimal(300)
        assert log.get_bucket_loss_rate(day=3, bucket_id="short") == expected

    # --- day query methods ----------------------------------------------

    def test_get_trades_for_day(self):
        log = EventLog()
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk1",
            bucket="short",
            price=Decimal(1),
            is_passthrough=False,
        )
        log.log_trade(
            day=2,
            side="SELL",
            trader_id="t",
            ticket_id="tk2",
            bucket="short",
            price=Decimal(1),
            is_passthrough=False,
        )
        assert len(log.get_trades_for_day(1)) == 1
        assert len(log.get_trades_for_day(2)) == 1
        assert len(log.get_trades_for_day(3)) == 0

    def test_get_defaults_for_day(self):
        log = EventLog()
        log.log_default(
            day=3,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(50),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert len(log.get_defaults_for_day(3)) == 1
        assert len(log.get_defaults_for_day(4)) == 0

    def test_get_settlements_for_day(self):
        log = EventLog()
        log.log_settlement(day=5, issuer_id="iss", total_paid=Decimal(100), n_tickets=2)
        assert len(log.get_settlements_for_day(5)) == 1
        assert len(log.get_settlements_for_day(6)) == 0

    def test_get_events_for_day(self):
        log = EventLog()
        log.log_day_start(day=1)
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk",
            bucket="short",
            price=Decimal(1),
            is_passthrough=False,
        )
        log.log_day_start(day=2)
        assert len(log.get_events_for_day(1)) == 2
        assert len(log.get_events_for_day(2)) == 1
        assert len(log.get_events_for_day(99)) == 0

    def test_get_all_events(self):
        log = EventLog()
        log.log_day_start(day=1)
        log.log_day_start(day=2)
        all_events = log.get_all_events()
        assert len(all_events) == 2
        # Must be a copy
        all_events.append({"kind": "fake"})
        assert len(log.events) == 2

    # --- to_jsonl -------------------------------------------------------

    def test_to_jsonl(self, tmp_path):
        log = EventLog()
        log.log_day_start(day=1)
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk",
            bucket="short",
            price=Decimal("0.90"),
            is_passthrough=False,
        )
        out = tmp_path / "events.jsonl"
        log.to_jsonl(str(out))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        parsed_0 = json.loads(lines[0])
        assert parsed_0["kind"] == "day_start"
        parsed_1 = json.loads(lines[1])
        assert parsed_1["kind"] == "trade"
        assert parsed_1["price"] == "0.90"

    def test_to_jsonl_creates_parent_directories(self, tmp_path):
        log = EventLog()
        log.log_day_start(day=1)
        out = tmp_path / "sub" / "dir" / "events.jsonl"
        log.to_jsonl(str(out))
        assert out.exists()

    def test_to_jsonl_empty_log(self, tmp_path):
        log = EventLog()
        out = tmp_path / "empty.jsonl"
        log.to_jsonl(str(out))
        assert out.read_text() == ""

    # --- to_dataframe ---------------------------------------------------

    def test_to_dataframe(self):
        pytest.importorskip("pandas")
        log = EventLog()
        log.log_day_start(day=1)
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk",
            bucket="short",
            price=Decimal(1),
            is_passthrough=False,
        )
        df = log.to_dataframe()
        assert len(df) == 2
        assert list(df.columns[:2]) == ["kind", "day"]
        assert df.iloc[0]["kind"] == "day_start"
        assert df.iloc[1]["kind"] == "trade"

    def test_to_dataframe_empty(self):
        pytest.importorskip("pandas")
        log = EventLog()
        df = log.to_dataframe()
        assert len(df) == 0

    # --- index independence ---------------------------------------------

    def test_index_independence_across_instances(self):
        log1 = EventLog()
        log2 = EventLog()
        log1.log_default(
            day=1,
            issuer_id="iss",
            recovery_rate=Decimal(0),
            total_due=Decimal(100),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="short",
        )
        assert len(log2.defaults_by_day) == 0
        assert len(log2.trades_by_day) == 0
        assert len(log2.settlements_by_day) == 0

    # --- mixed events ordering ------------------------------------------

    def test_chronological_ordering(self):
        log = EventLog()
        log.log_day_start(day=1)
        log.log_trade(
            day=1,
            side="BUY",
            trader_id="t",
            ticket_id="tk",
            bucket="short",
            price=Decimal(1),
            is_passthrough=False,
        )
        log.log_settlement(day=1, issuer_id="iss", total_paid=Decimal(100), n_tickets=1)
        log.log_day_start(day=2)
        log.log_default(
            day=2,
            issuer_id="iss2",
            recovery_rate=Decimal(0),
            total_due=Decimal(50),
            total_paid=Decimal(0),
            n_tickets=1,
            bucket="mid",
        )
        all_events = log.get_all_events()
        assert len(all_events) == 5
        kinds = [e["kind"] for e in all_events]
        assert kinds == ["day_start", "trade", "settlement", "day_start", "default"]

    # --- multiple trades same day ---------------------------------------

    def test_multiple_trades_indexed_same_day(self):
        log = EventLog()
        for i in range(5):
            log.log_trade(
                day=3,
                side="BUY",
                trader_id=f"t{i}",
                ticket_id=f"tk{i}",
                bucket="short",
                price=Decimal(1),
                is_passthrough=False,
            )
        assert len(log.trades_by_day[3]) == 5
        assert len(log.get_trades_for_day(3)) == 5

    # --- serialize nested structure -------------------------------------

    def test_serialize_deeply_nested(self):
        log = EventLog()
        log.log(
            "complex",
            day=1,
            data={"a": [Decimal(1), {"b": Decimal(2)}]},
        )
        e = log.events[0]
        assert e["data"]["a"][0] == "1"
        assert e["data"]["a"][1]["b"] == "2"

    def test_serialize_non_decimal_passthrough(self):
        log = EventLog()
        log.log("x", day=1, text="hello", number=42, flag=True, none_val=None)
        e = log.events[0]
        assert e["text"] == "hello"
        assert e["number"] == 42
        assert e["flag"] is True
        assert e["none_val"] is None
