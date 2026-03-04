"""Additional coverage tests for html_export.py.

Targets uncovered branches: event type mapping, safe_int_conversion edge cases,
balance rendering from AgentBalance objects, phase splitting, convergence status,
network JSON building, and CSS fallback.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bilancio.ui.html_export import (
    _build_rows_from_balance,
    _format_amount,
    _html_escape,
    _map_event_fields,
    _render_events_table,
    _render_t_account_from_rows,
    _safe_int_conversion,
    _split_by_phases,
    _split_phase_b_into_subphases,
    export_pretty_html,
)


# ============================================================================
# _safe_int_conversion edge cases
# ============================================================================


class TestSafeIntConversion:
    """Cover edge cases in _safe_int_conversion."""

    def test_none_returns_none(self):
        assert _safe_int_conversion(None) is None

    def test_int_passthrough(self):
        assert _safe_int_conversion(42) == 42

    def test_float_truncation(self):
        assert _safe_int_conversion(3.7) == 3

    def test_decimal_conversion(self):
        assert _safe_int_conversion(Decimal("123")) == 123

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            _safe_int_conversion(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            _safe_int_conversion(float("inf"))

    def test_string_with_commas(self):
        assert _safe_int_conversion("1,000") == 1000

    def test_empty_string_returns_none(self):
        assert _safe_int_conversion("") is None

    def test_whitespace_string_returns_none(self):
        assert _safe_int_conversion("   ") is None

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid numeric"):
            _safe_int_conversion("abc")

    def test_object_with_dunder_int(self):
        class CustomInt:
            def __int__(self):
                return 99

        assert _safe_int_conversion(CustomInt()) == 99

    def test_object_with_failing_dunder_int(self):
        class BadInt:
            def __int__(self):
                raise ArithmeticError("nope")

        with pytest.raises(ValueError, match="Invalid numeric"):
            _safe_int_conversion(BadInt())

    def test_unconvertible_type_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _safe_int_conversion([1, 2, 3])


# ============================================================================
# _format_amount
# ============================================================================


class TestFormatAmount:
    """Cover _format_amount branches."""

    def test_none_returns_dash(self):
        assert _format_amount(None) == "\u2014"

    def test_dash_string(self):
        assert _format_amount("\u2014") == "\u2014"
        assert _format_amount("-") == "\u2014"

    def test_integer(self):
        assert _format_amount(1234) == "1,234"

    def test_string_number(self):
        assert _format_amount("5000") == "5,000"

    def test_fallback_float_string(self):
        # Something that _safe_int_conversion fails on but float() succeeds
        assert _format_amount("3.14") == "3"

    def test_totally_invalid_returns_escaped(self):
        result = _format_amount("not-a-number")
        assert "not-a-number" in result


# ============================================================================
# _html_escape
# ============================================================================


class TestHtmlEscape:
    """Cover _html_escape edge cases."""

    def test_none_returns_empty(self):
        assert _html_escape(None) == ""

    def test_special_chars(self):
        result = _html_escape("<script>alert('xss')</script>")
        assert "<" not in result
        assert "&#x27;" in result

    def test_quotes(self):
        result = _html_escape('He said "hello"')
        assert "&quot;" in result


# ============================================================================
# _map_event_fields for various event kinds
# ============================================================================


class TestMapEventFields:
    """Cover different event kind branches in _map_event_fields."""

    def test_cash_deposited(self):
        e = {"kind": "CashDeposited", "customer": "H1", "bank": "B1", "amount": 100}
        m = _map_event_fields(e)
        assert m["from"] == "H1"
        assert m["to"] == "B1"

    def test_cash_withdrawn(self):
        e = {"kind": "CashWithdrawn", "customer": "H1", "bank": "B1", "amount": 50}
        m = _map_event_fields(e)
        assert m["from"] == "B1"
        assert m["to"] == "H1"

    def test_client_payment(self):
        e = {
            "kind": "ClientPayment",
            "payer": "H1",
            "payee": "H2",
            "payer_bank": "B1",
            "payee_bank": "B2",
            "amount": 200,
        }
        m = _map_event_fields(e)
        assert m["from"] == "H1"
        assert m["to"] == "H2"
        assert "B1" in m["notes"]
        assert "B2" in m["notes"]

    def test_intra_bank_payment(self):
        e = {"kind": "IntraBankPayment", "payer": "H1", "payee": "H2", "amount": 100}
        m = _map_event_fields(e)
        assert m["from"] == "H1"
        assert m["to"] == "H2"

    def test_cash_payment(self):
        e = {"kind": "CashPayment", "payer": "H1", "payee": "H2", "amount": 50}
        m = _map_event_fields(e)
        assert m["from"] == "H1"
        assert m["to"] == "H2"

    def test_interbank_cleared(self):
        e = {
            "kind": "InterbankCleared",
            "debtor_bank": "B1",
            "creditor_bank": "B2",
            "amount": 500,
            "due_day": 3,
        }
        m = _map_event_fields(e)
        assert m["from"] == "B1"
        assert m["to"] == "B2"
        assert "due 3" in m["notes"]

    def test_interbank_overnight_created(self):
        e = {
            "kind": "InterbankOvernightCreated",
            "debtor_bank": "B1",
            "creditor_bank": "B2",
            "amount": 300,
        }
        m = _map_event_fields(e)
        assert m["from"] == "B1"
        assert m["to"] == "B2"

    def test_stock_created(self):
        e = {"kind": "StockCreated", "owner": "F1", "stock_id": "s001"}
        m = _map_event_fields(e)
        assert m["from"] == "F1"
        assert m["to"] == "\u2014"
        assert "s001" in m["id_or_alias"]

    def test_dealer_trade_sell_passthrough(self):
        e = {
            "kind": "dealer_trade",
            "trader": "T1",
            "is_passthrough": True,
            "side": "sell",
            "unit_price": 0.85,
            "bucket": "short",
            "price": 17,
            "ticket_id": "tk01",
        }
        m = _map_event_fields(e)
        assert m["from"] == "T1"
        assert m["to"] == "VBT"
        assert "sell" in m["notes"]
        assert "0.85" in m["notes"]
        assert "passthrough" in m["notes"]

    def test_dealer_trade_buy_interior_no_price(self):
        e = {
            "kind": "dealer_trade",
            "trader": "T2",
            "is_passthrough": False,
            "side": "buy",
            "bucket": "mid",
        }
        m = _map_event_fields(e)
        assert m["from"] == "T2"
        assert m["to"] == "Dealer"
        assert "buy" in m["notes"]
        assert "interior" in m["notes"]

    def test_claim_transferred_dealer(self):
        e = {
            "kind": "ClaimTransferredDealer",
            "from_holder": "D1",
            "to_holder": "T1",
            "due_day": 5,
        }
        m = _map_event_fields(e)
        assert m["from"] == "D1"
        assert m["to"] == "T1"
        assert "due day 5" in m["notes"]

    def test_claim_transferred_dealer_no_due_day(self):
        e = {"kind": "ClaimTransferredDealer", "from_holder": "D1", "to_holder": "T1"}
        m = _map_event_fields(e)
        assert m["notes"] == ""

    def test_agent_defaulted_with_details(self):
        e = {
            "kind": "AgentDefaulted",
            "agent": "F1",
            "shortfall": 500,
            "trigger_contract": "c001",
        }
        m = _map_event_fields(e)
        assert "shortfall 500" in m["notes"]
        assert "trigger c001" in m["notes"]

    def test_agent_defaulted_no_details(self):
        e = {"kind": "AgentDefaulted", "agent": "F1"}
        m = _map_event_fields(e)
        assert m["notes"] == "default"

    def test_generic_event_fallback(self):
        e = {"kind": "SomeUnknownEvent", "frm": "A", "to": "B", "amount": 10}
        m = _map_event_fields(e)
        assert m["from"] == "A"
        assert m["to"] == "B"

    def test_id_or_alias_preference(self):
        """alias takes precedence over contract_id."""
        e = {"kind": "test", "alias": "myalias", "contract_id": "c123"}
        m = _map_event_fields(e)
        assert m["id_or_alias"] == "myalias"

    def test_id_or_alias_contract_id(self):
        e = {"kind": "test", "contract_id": "c123"}
        m = _map_event_fields(e)
        assert m["id_or_alias"] == "c123"

    def test_id_or_alias_payable_id(self):
        e = {"kind": "test", "payable_id": "p456"}
        m = _map_event_fields(e)
        assert m["id_or_alias"] == "p456"


# ============================================================================
# _split_by_phases and _split_phase_b_into_subphases
# ============================================================================


class TestPhaseSplitting:
    """Cover phase splitting logic."""

    def test_split_by_phases_basic(self):
        events = [
            {"kind": "PhaseA"},
            {"kind": "e1"},
            {"kind": "PhaseB"},
            {"kind": "e2"},
            {"kind": "PhaseC"},
            {"kind": "e3"},
        ]
        buckets = _split_by_phases(events)
        assert len(buckets["A"]) == 1
        assert buckets["A"][0]["kind"] == "e1"
        assert len(buckets["B"]) == 1
        assert len(buckets["C"]) == 1

    def test_split_by_phases_empty(self):
        buckets = _split_by_phases([])
        assert buckets == {"A": [], "B": [], "C": []}

    def test_split_phase_b_subphases(self):
        events = [
            {"kind": "SubphaseB1"},
            {"kind": "scheduled_action"},
            {"kind": "SubphaseB_Dealer"},
            {"kind": "dealer_trade"},
            {"kind": "SubphaseB2"},
            {"kind": "settlement"},
        ]
        b1, bd, b2 = _split_phase_b_into_subphases(events)
        assert len(b1) == 1
        assert b1[0]["kind"] == "scheduled_action"
        assert len(bd) == 1
        assert bd[0]["kind"] == "dealer_trade"
        assert len(b2) == 1
        assert b2[0]["kind"] == "settlement"

    def test_split_phase_b_no_markers(self):
        """Events without subphase markers all go to B1 (current_phase=0)."""
        events = [{"kind": "e1"}, {"kind": "e2"}]
        b1, bd, b2 = _split_phase_b_into_subphases(events)
        assert len(b1) == 2
        assert len(bd) == 0
        assert len(b2) == 0


# ============================================================================
# _render_t_account_from_rows
# ============================================================================


class TestRenderTAccountFromRows:
    """Cover T-account rendering."""

    def test_empty_rows(self):
        html = _render_t_account_from_rows("TestAgent", [], [])
        assert "TestAgent" in html
        assert "empty" in html

    def test_with_data(self):
        assets = [
            {
                "name": "Cash",
                "quantity": 10,
                "value_minor": 1000,
                "counterparty_name": "CB",
                "maturity": "on-demand",
                "id_or_alias": "c001",
            }
        ]
        liabs = [
            {
                "name": "Deposit",
                "quantity": None,
                "value_minor": 500,
                "counterparty_name": "H1",
                "maturity": None,
            }
        ]
        html = _render_t_account_from_rows("Bank 1", assets, liabs)
        assert "Cash" in html
        assert "1,000" in html
        assert "Deposit" in html
        assert "500" in html

    def test_non_numeric_values(self):
        """Rows with invalid numeric values should still render."""
        assets = [
            {
                "name": "Bad",
                "quantity": "abc",
                "value_minor": "xyz",
                "counterparty_name": None,
                "maturity": None,
            }
        ]
        html = _render_t_account_from_rows("Test", assets, [])
        assert "Bad" in html
        assert "\u2014" in html  # em-dash for invalid values


# ============================================================================
# _render_events_table
# ============================================================================


class TestRenderEventsTable:
    """Cover events table rendering."""

    def test_empty_events(self):
        html = _render_events_table("TestPhase", [])
        assert "No events" in html

    def test_filters_phase_markers(self):
        events = [
            {"kind": "PhaseA"},
            {"kind": "PhaseB"},
            {"kind": "PhaseC"},
            {"kind": "SubphaseB1"},
            {"kind": "SubphaseB2"},
            {"kind": "SubphaseB_Dealer"},
            {"kind": "RealEvent", "amount": 100},
        ]
        html = _render_events_table("Test", events)
        # PhaseA etc are excluded, only RealEvent remains
        assert "RealEvent" in html
        assert "PhaseA" not in html


# ============================================================================
# export_pretty_html convergence branches
# ============================================================================


class TestExportPrettyHtmlConvergence:
    """Cover convergence and non-convergence status display."""

    def _make_system(self):
        from bilancio.config.apply import create_agent
        from bilancio.engines.system import System

        sys = System()
        cb = create_agent(
            type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})
        )
        sys.add_agent(cb)
        return sys

    def test_converged_status(self, tmp_path):
        sys = self._make_system()
        out = tmp_path / "report.html"
        # Provide 3 quiet days with quiet=True
        days_data = [
            {"day": 1, "events": [], "quiet": True},
            {"day": 2, "events": [], "quiet": True},
            {"day": 3, "events": [], "quiet": True},
        ]
        export_pretty_html(
            system=sys,
            out_path=out,
            scenario_name="Conv Test",
            description="Testing convergence",
            agent_ids=["CB"],
            initial_balances={},
            days_data=days_data,
            max_days=10,
            quiet_days=3,
        )
        html = out.read_text()
        assert "Converged on day 3" in html

    def test_not_converged_with_max_days(self, tmp_path):
        sys = self._make_system()
        out = tmp_path / "report.html"
        days_data = [
            {"day": 1, "events": [], "quiet": False},
        ]
        export_pretty_html(
            system=sys,
            out_path=out,
            scenario_name="No Conv Test",
            description=None,
            agent_ids=["CB"],
            initial_balances={},
            days_data=days_data,
            max_days=5,
            quiet_days=3,
        )
        html = out.read_text()
        assert "Stopped without convergence" in html
        assert "max_days = 5" in html


# ============================================================================
# _load_css fallback
# ============================================================================


class TestLoadCssFallback:
    """Cover CSS loading with fallback."""

    def test_css_loads_successfully(self):
        from bilancio.ui.html_export import _load_css

        css = _load_css()
        assert len(css) > 0

    def test_css_fallback_on_missing_file(self):
        from bilancio.ui.html_export import _load_css

        with patch("bilancio.ui.html_export.Path.read_text", side_effect=FileNotFoundError):
            css = _load_css()
            assert "font-family" in css
            assert "border-collapse" in css


# ============================================================================
# export_pretty_html with day rows and initial_rows
# ============================================================================


class TestExportWithRows:
    """Cover the initial_rows and day_rows branches."""

    def test_with_initial_rows(self, tmp_path):
        from bilancio.config.apply import create_agent
        from bilancio.engines.system import System

        sys = System()
        cb = create_agent(
            type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})
        )
        h1 = create_agent(
            type("S", (), {"id": "H1", "kind": "household", "name": "Alice"})
        )
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 5000)

        out = tmp_path / "report.html"
        initial_rows = {
            "H1": {
                "assets": [
                    {
                        "name": "Cash",
                        "quantity": None,
                        "value_minor": 5000,
                        "counterparty_name": None,
                        "maturity": "on-demand",
                    }
                ],
                "liabs": [],
            }
        }

        from bilancio.analysis.balances import agent_balance

        export_pretty_html(
            system=sys,
            out_path=out,
            scenario_name="Rows Test",
            description=None,
            agent_ids=["H1"],
            initial_balances={"H1": agent_balance(sys, "H1")},
            days_data=[],
            initial_rows=initial_rows,
        )
        html = out.read_text()
        assert "Cash" in html
        assert "5,000" in html

    def test_with_day_rows(self, tmp_path):
        from bilancio.config.apply import create_agent
        from bilancio.engines.system import System

        sys = System()
        cb = create_agent(
            type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})
        )
        sys.add_agent(cb)

        out = tmp_path / "report.html"
        days_data = [
            {
                "day": 1,
                "events": [],
                "rows": {
                    "CB": {
                        "assets": [
                            {
                                "name": "Reserves",
                                "quantity": None,
                                "value_minor": 3000,
                                "counterparty_name": None,
                                "maturity": None,
                            }
                        ],
                        "liabs": [],
                    }
                },
            }
        ]

        export_pretty_html(
            system=sys,
            out_path=out,
            scenario_name="DayRows Test",
            description=None,
            agent_ids=["CB"],
            initial_balances={},
            days_data=days_data,
        )
        html = out.read_text()
        assert "Reserves" in html
        assert "3,000" in html

    def test_day_events_with_dealer_trading(self, tmp_path):
        """Cover dealer trading events in day rendering."""
        from bilancio.config.apply import create_agent
        from bilancio.engines.system import System

        sys = System()
        cb = create_agent(
            type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})
        )
        sys.add_agent(cb)

        out = tmp_path / "report.html"
        days_data = [
            {
                "day": 1,
                "events": [
                    {"kind": "PhaseB"},
                    {"kind": "SubphaseB_Dealer"},
                    {
                        "kind": "dealer_trade",
                        "trader": "T1",
                        "side": "sell",
                        "is_passthrough": False,
                        "bucket": "short",
                        "unit_price": 0.90,
                        "price": 18,
                        "ticket_id": "tk01",
                    },
                    {"kind": "SubphaseB2"},
                ],
            }
        ]

        export_pretty_html(
            system=sys,
            out_path=out,
            scenario_name="Dealer Test",
            description=None,
            agent_ids=["CB"],
            initial_balances={},
            days_data=days_data,
        )
        html = out.read_text()
        assert "Dealer Trading" in html
        assert "dealer_trade" in html
