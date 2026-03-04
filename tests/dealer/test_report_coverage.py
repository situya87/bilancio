"""Additional coverage tests for dealer/report.py.

Targets uncovered rendering functions: _fmt_decimal edge cases,
_render_events_table with various event kinds, _render_trader_balance
with defaulted trader, _render_dealer_card with inventory.
"""

from __future__ import annotations

from decimal import Decimal

from bilancio.dealer.report import (
    _fmt_decimal,
    _html_escape,
    _render_dealer_card,
    _render_events_table,
    _render_trader_balance,
    _render_vbt_card,
)


# ============================================================================
# _fmt_decimal edge cases
# ============================================================================


class TestFmtDecimal:
    """Cover _fmt_decimal edge cases."""

    def test_none_returns_dash(self):
        assert _fmt_decimal(None) == "—"

    def test_string_valid_decimal(self):
        result = _fmt_decimal("3.14")
        assert "3.14" in result

    def test_string_invalid(self):
        result = _fmt_decimal("not-a-number")
        assert result == "not-a-number"

    def test_int_value(self):
        result = _fmt_decimal(42)
        assert "42" in result

    def test_float_value(self):
        result = _fmt_decimal(0.5)
        assert "0.5" in result

    def test_decimal_strips_trailing_zeros(self):
        result = _fmt_decimal(Decimal("1.2000"))
        assert result == "1.2"


# ============================================================================
# _render_events_table with different event kinds
# ============================================================================


class TestRenderEventsTable:
    """Cover _render_events_table with all event kind branches."""

    def test_empty_events(self):
        html = _render_events_table([], "Test Events")
        assert "No events for this day" in html

    def test_day_start_filtered_out(self):
        events = [{"kind": "day_start"}]
        html = _render_events_table(events, "Test")
        assert "No events for this day" in html

    def test_trade_event(self):
        events = [
            {
                "kind": "trade",
                "side": "BUY",
                "price": Decimal("0.95"),
                "bucket": "short",
                "trader_id": "T1",
                "ticket_id": "tick_001",
            }
        ]
        html = _render_events_table(events, "Trades")
        assert "BUY" in html
        assert "short" in html

    def test_trade_passthrough(self):
        events = [
            {
                "kind": "trade",
                "side": "SELL",
                "price": Decimal("0.90"),
                "bucket": "mid",
                "is_passthrough": True,
                "trader_id": "T2",
                "ticket_id": "tick_002",
            }
        ]
        html = _render_events_table(events, "Passthrough")
        assert "passthrough" in html.lower()

    def test_rebucket_event(self):
        events = [
            {
                "kind": "rebucket",
                "old_bucket": "long",
                "new_bucket": "mid",
                "price": Decimal("0.88"),
                "holder_type": "dealer",
                "ticket_id": "tick_003",
            }
        ]
        html = _render_events_table(events, "Rebucket")
        assert "long" in html
        assert "mid" in html

    def test_settlement_event(self):
        events = [
            {
                "kind": "settlement",
                "issuer_id": "T1",
                "total_paid": Decimal("10.5"),
                "n_tickets": 3,
            }
        ]
        html = _render_events_table(events, "Settlements")
        assert "Paid" in html
        assert "3 tickets" in html

    def test_default_event(self):
        events = [
            {
                "kind": "default",
                "issuer_id": "T3",
                "recovery_rate": 0.5,
                "total_due": Decimal("100"),
                "total_paid": Decimal("50"),
                "bucket": "short",
            }
        ]
        html = _render_events_table(events, "Defaults")
        assert "Recovery" in html
        assert "T3" in html

    def test_quote_event(self):
        events = [
            {
                "kind": "quote",
                "bucket": "long",
                "dealer_bid": Decimal("0.85"),
                "dealer_ask": Decimal("0.92"),
                "vbt_bid": Decimal("0.80"),
                "vbt_ask": Decimal("0.95"),
                "inventory": 5,
                "capacity": Decimal("10"),
            }
        ]
        html = _render_events_table(events, "Quotes")
        assert "long" in html
        assert "inv=5" in html

    def test_vbt_anchor_update_event(self):
        events = [
            {
                "kind": "vbt_anchor_update",
                "bucket": "short",
                "M_old": Decimal("1.0"),
                "M_new": Decimal("0.95"),
                "O_old": Decimal("0.20"),
                "O_new": Decimal("0.25"),
                "loss_rate": Decimal("0.05"),
            }
        ]
        html = _render_events_table(events, "VBT Updates")
        assert "short" in html
        assert "loss rate" in html

    def test_unknown_event_kind(self):
        events = [{"kind": "unknown_event", "data": "some_value"}]
        html = _render_events_table(events, "Unknown")
        assert "unknown_event" in html


# ============================================================================
# _render_trader_balance
# ============================================================================


class TestRenderTraderBalance:
    """Cover _render_trader_balance with various trader states."""

    def test_basic_trader(self):
        trader_dict = {
            "agent_id": "T1",
            "cash": Decimal("10.5"),
            "defaulted": False,
            "tickets_owned": [],
            "obligations": [],
        }
        html = _render_trader_balance(trader_dict)
        assert "T1" in html
        assert "No tickets owned" in html
        assert "No obligations" in html

    def test_defaulted_trader(self):
        trader_dict = {
            "agent_id": "T2",
            "cash": Decimal("0"),
            "defaulted": True,
            "tickets_owned": [],
            "obligations": [],
        }
        html = _render_trader_balance(trader_dict)
        assert "DEFAULTED" in html

    def test_trader_with_tickets(self):
        trader_dict = {
            "agent_id": "T3",
            "cash": Decimal("5"),
            "defaulted": False,
            "tickets_owned": [
                {"id": "tick_001", "face": "2.0", "issuer_id": "T1", "bucket_id": "short", "remaining_tau": 3},
                {"id": "tick_002", "face": Decimal("3.0"), "issuer_id": "T2", "bucket_id": "mid", "remaining_tau": 7},
            ],
            "obligations": [
                {"id": "tick_003", "face": "4.0", "owner_id": "T4", "maturity_day": 10},
            ],
        }
        html = _render_trader_balance(trader_dict)
        assert "tick_001" in html
        assert "tick_002" in html
        assert "tick_003" in html

    def test_trader_with_string_cash(self):
        trader_dict = {
            "agent_id": "T4",
            "cash": "15.0",
            "defaulted": False,
            "tickets_owned": [],
            "obligations": [],
        }
        html = _render_trader_balance(trader_dict)
        assert "T4" in html


# ============================================================================
# _render_dealer_card with inventory
# ============================================================================


class TestRenderDealerCard:
    """Cover _render_dealer_card with inventory items."""

    def test_dealer_with_inventory(self):
        dealer_dict = {
            "bucket_id": "short",
            "a": 2,
            "x": Decimal("3.5"),
            "cash": Decimal("10"),
            "V": Decimal("4.0"),
            "K_star": 5,
            "X_star": Decimal("6"),
            "N": 3,
            "lambda_": Decimal("0.1"),
            "I": Decimal("0.05"),
            "midline": Decimal("0.95"),
            "bid": Decimal("0.92"),
            "ask": Decimal("0.97"),
            "is_pinned_bid": True,
            "is_pinned_ask": False,
            "inventory": [
                {"id": "t1", "issuer_id": "T1", "face": Decimal("1.0"), "remaining_tau": 3},
                {"id": "t2", "issuer_id": "T2", "face": Decimal("2.5"), "remaining_tau": 5},
            ],
        }
        vbt_dict = {}
        html = _render_dealer_card(dealer_dict, vbt_dict)
        assert "SHORT" in html
        assert "Inventory (2 tickets)" in html
        assert "t1" in html
        assert "t2" in html
        assert "pinned" in html.lower()

    def test_dealer_no_inventory(self):
        dealer_dict = {
            "bucket_id": "mid",
            "a": 0,
            "x": Decimal("0"),
            "cash": Decimal("5"),
            "V": Decimal("0"),
            "K_star": 3,
            "X_star": Decimal("4"),
            "N": 2,
            "lambda_": Decimal("0"),
            "I": Decimal("0.10"),
            "midline": Decimal("0.90"),
            "bid": Decimal("0.85"),
            "ask": Decimal("0.95"),
            "is_pinned_bid": False,
            "is_pinned_ask": False,
            "inventory": [],
        }
        vbt_dict = {}
        html = _render_dealer_card(dealer_dict, vbt_dict)
        assert "MID" in html
        # Empty inventory: the inventory-list section is not rendered
        assert "inventory-list" not in html


# ============================================================================
# _render_vbt_card
# ============================================================================


class TestRenderVbtCard:
    """Cover _render_vbt_card."""

    def test_basic(self):
        vbt_dict = {
            "bucket_id": "long",
            "M": Decimal("0.90"),
            "O": Decimal("0.30"),
            "A": Decimal("1.05"),
            "B": Decimal("0.75"),
            "cash": Decimal("50"),
            "inventory": [{"id": "t1"}, {"id": "t2"}],
        }
        html = _render_vbt_card(vbt_dict)
        assert "LONG" in html
        assert "2 tickets" in html
