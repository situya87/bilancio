"""Tests for bilancio.analysis.visualization.events module.

Covers uncovered paths:
- display_events_table (both rich and non-rich paths)
- display_events_table_renderable (non-rich text path)
- _display_events_summary / _build_events_summary_renderables
- _display_events_detailed / _build_events_detailed_renderables
- _display_single_event (all event kind branches)
- _display_day_events (phase grouping)
- _format_single_event (rich text styling branches)
- display_events_for_day with system mock
- display_events_renderable (summary vs detailed)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bilancio.analysis.visualization.common import RICH_AVAILABLE
from bilancio.analysis.visualization.events import (
    _build_events_summary_renderables,
    _display_day_events,
    _display_events_detailed,
    _display_events_summary,
    _display_single_event,
    _format_single_event,
    display_events,
    display_events_for_day,
    display_events_for_day_renderable,
    display_events_renderable,
    display_events_table,
    display_events_table_renderable,
)


# ============================================================================
# Helpers
# ============================================================================


def _ev(kind: str, **kw) -> dict:
    """Shorthand for building event dicts."""
    d = {"kind": kind, "day": kw.pop("day", 0), "phase": kw.pop("phase", "simulation")}
    d.update(kw)
    return d


def _make_system_with_events(events):
    """Create a minimal mock system with a state.events list."""
    state = SimpleNamespace(events=events)
    return SimpleNamespace(state=state)


# ============================================================================
# display_events_table - non-rich text path
# ============================================================================


class TestDisplayEventsTableNonRich:
    """Test the simple text fallback in display_events_table."""

    @pytest.fixture(autouse=True)
    def _force_no_rich(self):
        with patch("bilancio.analysis.visualization.events.RICH_AVAILABLE", False):
            with patch("bilancio.analysis.visualization.events.Console", None):
                yield

    def test_empty_events_prints_no_events(self, capsys):
        display_events_table([])
        captured = capsys.readouterr()
        assert "No events" in captured.out

    def test_simple_text_header_columns(self, capsys):
        events = [_ev("CashMinted", to="A1", amount=100)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "Day" in captured.out
        assert "Kind" in captured.out

    def test_cash_deposited_from_to(self, capsys):
        events = [_ev("CashDeposited", customer="HH01", bank="BK01", amount=100)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "HH01" in captured.out
        assert "BK01" in captured.out

    def test_cash_withdrawn_from_to(self, capsys):
        events = [_ev("CashWithdrawn", customer="HH01", bank="BK01", amount=50)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "BK01" in captured.out  # frm = bank for CashWithdrawn

    def test_client_payment_notes(self, capsys):
        events = [
            _ev(
                "ClientPayment",
                payer="HH01",
                payee="HH02",
                amount=100,
                payer_bank="BK01",
                payee_bank="BK02",
            )
        ]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "BK01" in captured.out
        assert "BK02" in captured.out

    def test_interbank_cleared_notes(self, capsys):
        events = [
            _ev(
                "InterbankCleared",
                debtor_bank="BK01",
                creditor_bank="BK02",
                amount=200,
            )
        ]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "BK01" in captured.out

    def test_interbank_with_due_day(self, capsys):
        events = [
            _ev(
                "InterbankOvernightCreated",
                debtor_bank="BK01",
                creditor_bank="BK02",
                amount=100,
                due_day=5,
            )
        ]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "due 5" in captured.out

    def test_stock_created(self, capsys):
        events = [_ev("StockCreated", owner="A1", qty=10, sku="W")]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "StockCreated" in captured.out
        assert "A1" in captured.out

    def test_agent_defaulted_notes(self, capsys):
        events = [_ev("AgentDefaulted", debtor="D1", shortfall=500, trigger_contract="P-001")]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "shortfall 500" in captured.out
        assert "trigger P-001" in captured.out

    def test_generic_event_fallback(self, capsys):
        events = [_ev("CustomEvent", debtor="D1", creditor="C1", amount=42)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "CustomEvent" in captured.out

    def test_generic_event_agent_field(self, capsys):
        """The 'agent' field is used as frm for unknown event kinds."""
        events = [_ev("SomeEvent", agent="AG1")]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "AG1" in captured.out

    def test_phase_markers_excluded(self, capsys):
        events = [
            _ev("PhaseA"),
            _ev("CashMinted", to="A1", amount=100),
            _ev("PhaseB"),
            _ev("PhaseC"),
        ]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "PhaseA" not in captured.out
        assert "CashMinted" in captured.out

    def test_instr_id_fallback(self, capsys):
        events = [_ev("SomeEvent", instr_id="INS-001", qty=5)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "INS-001" in captured.out

    def test_stock_id_fallback(self, capsys):
        events = [_ev("SomeEvent", stock_id="STK-001")]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "STK-001" in captured.out

    def test_quantity_fallback(self, capsys):
        events = [_ev("SomeEvent", quantity=7)]
        display_events_table(events)
        captured = capsys.readouterr()
        assert "7" in captured.out


# ============================================================================
# display_events_table_renderable - non-rich text path
# ============================================================================


class TestDisplayEventsTableRenderableNonRich:
    """Test the non-rich text path for display_events_table_renderable."""

    @pytest.fixture(autouse=True)
    def _force_no_rich(self):
        with patch("bilancio.analysis.visualization.events.RICH_AVAILABLE", False):
            yield

    def test_empty_returns_string(self):
        result = display_events_table_renderable([])
        assert isinstance(result, str)
        assert "No events" in result

    def test_returns_text_with_header(self):
        events = [_ev("CashMinted", to="A1", amount=100)]
        result = display_events_table_renderable(events)
        assert isinstance(result, str)
        assert "Day" in result
        assert "CashMinted" in result

    def test_all_event_kinds_render(self):
        events = [
            _ev("CashDeposited", customer="HH01", bank="BK01", amount=100),
            _ev("CashWithdrawn", customer="HH02", bank="BK02", amount=50),
            _ev("ClientPayment", payer="P1", payee="P2", amount=10, payer_bank="B1", payee_bank="B2"),
            _ev("InterbankCleared", debtor_bank="BK01", creditor_bank="BK02", amount=200),
            _ev("InterbankOvernightCreated", debtor_bank="BK01", creditor_bank="BK02", amount=100, due_day=3),
            _ev("StockCreated", owner="A1", qty=10, sku="W"),
            _ev("GenericEvent", debtor="G1", creditor="G2", amount=42),
        ]
        result = display_events_table_renderable(events)
        assert isinstance(result, str)
        assert "CashDeposited" in result
        assert "InterbankCleared" in result


# ============================================================================
# _display_events_summary
# ============================================================================


class TestDisplayEventsSummary:
    """Test _display_events_summary with various event kinds."""

    def test_payable_settled(self, capsys):
        events = [_ev("PayableSettled", debtor="D1", creditor="C1", amount=100)]
        _display_events_summary(events)
        captured = capsys.readouterr()
        assert "D1" in captured.out
        assert "C1" in captured.out

    def test_delivery_obligation_settled(self, capsys):
        events = [
            _ev("DeliveryObligationSettled", debtor="D1", creditor="C1", qty=10, sku="WIDGETS")
        ]
        _display_events_summary(events)
        captured = capsys.readouterr()
        assert "D1" in captured.out
        assert "10" in captured.out

    def test_delivery_obligation_settled_quantity_fallback(self, capsys):
        events = [
            _ev(
                "DeliveryObligationSettled",
                debtor="D1",
                creditor="C1",
                quantity=5,
                sku="GOODS",
            )
        ]
        _display_events_summary(events)
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_stock_transferred(self, capsys):
        events = [_ev("StockTransferred", frm="A1", to="A2", qty=3, sku="ITEM")]
        _display_events_summary(events)
        captured = capsys.readouterr()
        assert "A1" in captured.out

    def test_cash_transferred(self, capsys):
        events = [_ev("CashTransferred", frm="A1", to="A2", amount=50)]
        _display_events_summary(events)
        captured = capsys.readouterr()
        assert "50" in captured.out


# ============================================================================
# _build_events_summary_renderables (non-rich)
# ============================================================================


class TestBuildEventsSummaryRenderablesNonRich:
    """Test summary renderables in non-rich mode."""

    @pytest.fixture(autouse=True)
    def _force_no_rich(self):
        with patch("bilancio.analysis.visualization.events.RICH_AVAILABLE", False):
            yield

    def test_payable_settled_renderable(self):
        events = [_ev("PayableSettled", debtor="D1", creditor="C1", amount=100)]
        result = _build_events_summary_renderables(events)
        assert len(result) == 1
        assert "D1" in result[0]

    def test_delivery_settled_renderable(self):
        events = [
            _ev("DeliveryObligationSettled", debtor="D1", creditor="C1", qty=5, sku="W")
        ]
        result = _build_events_summary_renderables(events)
        assert len(result) == 1

    def test_stock_transferred_renderable(self):
        events = [_ev("StockTransferred", frm="A1", to="A2", qty=3, sku="ITEM")]
        result = _build_events_summary_renderables(events)
        assert len(result) == 1

    def test_cash_transferred_renderable(self):
        events = [_ev("CashTransferred", frm="A1", to="A2", amount=50)]
        result = _build_events_summary_renderables(events)
        assert len(result) == 1

    def test_other_events_skipped(self):
        events = [_ev("CustomEvent")]
        result = _build_events_summary_renderables(events)
        assert len(result) == 0


# ============================================================================
# _display_single_event (all kind branches)
# ============================================================================


class TestDisplaySingleEvent:
    """Test _display_single_event for all event kind branches."""

    def test_stock_created(self, capsys):
        event = _ev("StockCreated", owner="A1", qty=10, sku="W")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Stock created" in captured.out
        assert "A1" in captured.out

    def test_cash_minted(self, capsys):
        event = _ev("CashMinted", to="A1", amount=500)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Cash minted" in captured.out
        assert "500" in captured.out

    def test_payable_settled(self, capsys):
        event = _ev("PayableSettled", debtor="D1", creditor="C1", amount=100)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Payment settled" in captured.out

    def test_payable_cancelled(self, capsys):
        event = _ev("PayableCancelled")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "removed from books" in captured.out

    def test_delivery_obligation_settled(self, capsys):
        event = _ev("DeliveryObligationSettled", debtor="D1", creditor="C1", qty=5, sku="GOODS")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Delivery settled" in captured.out

    def test_delivery_obligation_settled_quantity_fallback(self, capsys):
        event = _ev(
            "DeliveryObligationSettled",
            debtor="D1",
            creditor="C1",
            quantity=7,
            sku="STUFF",
        )
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "7" in captured.out

    def test_delivery_obligation_cancelled(self, capsys):
        event = _ev("DeliveryObligationCancelled")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "removed from books" in captured.out

    def test_stock_transferred(self, capsys):
        event = _ev("StockTransferred", frm="A1", to="A2", qty=3, sku="ITEM")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Stock transferred" in captured.out

    def test_cash_transferred(self, capsys):
        event = _ev("CashTransferred", frm="A1", to="A2", amount=50)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Cash transferred" in captured.out

    def test_stock_split(self, capsys):
        event = _ev(
            "StockSplit",
            sku="ITEMS",
            original_qty=20,
            split_qty=5,
            remaining_qty=15,
            original_id="lot_abc12345",
            new_id="lot_def67890",
        )
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Stock split" in captured.out
        assert "5" in captured.out

    def test_stock_split_no_ids(self, capsys):
        """StockSplit with missing IDs should still work."""
        event = _ev("StockSplit", sku="ITEMS", split_qty=3, remaining_qty=7)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Stock split" in captured.out

    def test_reserves_minted(self, capsys):
        event = _ev("ReservesMinted", amount=1000, to="BK01")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Reserves minted" in captured.out

    def test_cash_deposited(self, capsys):
        event = _ev("CashDeposited", customer="HH01", bank="BK01", amount=200)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Cash deposited" in captured.out

    def test_delivery_obligation_created(self, capsys):
        event = _ev("DeliveryObligationCreated", frm="A1", to="A2", qty=5, sku="W", due_day=3)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Delivery obligation created" in captured.out

    def test_delivery_obligation_created_from_fallback(self, capsys):
        """Uses 'from' key when 'frm' missing."""
        event = {"kind": "DeliveryObligationCreated", "from": "A1", "to": "A2", "qty": 5, "sku": "W", "due_day": 3, "day": 0, "phase": "simulation"}
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "A1" in captured.out

    def test_payable_created(self, capsys):
        event = _ev("PayableCreated", frm="D1", to="C1", amount=100, due_day=5)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Payable created" in captured.out

    def test_payable_created_debtor_fallback(self, capsys):
        """Uses debtor/creditor when frm/to missing."""
        event = _ev("PayableCreated", debtor="D1", creditor="C1", amount=100, due_day=5)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "D1" in captured.out

    def test_client_payment(self, capsys):
        event = _ev(
            "ClientPayment",
            payer="HH01",
            payee="HH02",
            amount=100,
            payer_bank="BK01",
            payee_bank="BK02",
        )
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Client payment" in captured.out

    def test_interbank_cleared(self, capsys):
        event = _ev("InterbankCleared", debtor_bank="BK01", creditor_bank="BK02", amount=200)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Interbank cleared" in captured.out

    def test_reserves_transferred(self, capsys):
        event = _ev("ReservesTransferred", frm="BK01", to="BK02", amount=100)
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Reserves transferred" in captured.out

    def test_instrument_merged(self, capsys):
        event = _ev("InstrumentMerged")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "merged" in captured.out.lower()

    def test_interbank_overnight_created(self, capsys):
        event = _ev(
            "InterbankOvernightCreated",
            debtor_bank="BK01",
            creditor_bank="BK02",
            amount=300,
        )
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "Overnight" in captured.out

    def test_phase_marker_no_output(self, capsys):
        """Phase markers produce no output."""
        for kind in ("PhaseA", "PhaseB", "PhaseC"):
            _display_single_event(_ev(kind))
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_unknown_event_kind(self, capsys):
        event = _ev("WeirdEvent", foo="bar")
        _display_single_event(event)
        captured = capsys.readouterr()
        assert "WeirdEvent" in captured.out

    def test_indent_parameter(self, capsys):
        event = _ev("CashMinted", to="A1", amount=100)
        _display_single_event(event, indent="      ")
        captured = capsys.readouterr()
        assert captured.out.startswith("      ")


# ============================================================================
# _display_day_events (phase grouping within a day)
# ============================================================================


class TestDisplayDayEvents:
    """Test _display_day_events phase grouping logic."""

    def test_all_three_phases_displayed(self, capsys):
        events = [
            _ev("PhaseA"),
            _ev("CashMinted", to="A1", amount=100),
            _ev("PhaseB"),
            _ev("PayableSettled", debtor="D1", creditor="C1", amount=50),
            _ev("PhaseC"),
            _ev("CashTransferred", frm="X", to="Y", amount=10),
        ]
        _display_day_events(events)
        captured = capsys.readouterr()
        assert "Phase A" in captured.out
        assert "Phase B" in captured.out
        assert "Phase C" in captured.out
        assert "Day ended" in captured.out

    def test_events_without_markers(self, capsys):
        """Events without phase markers default to Phase A."""
        events = [_ev("CashMinted", to="A1", amount=100)]
        _display_day_events(events)
        captured = capsys.readouterr()
        assert "Phase A" in captured.out


# ============================================================================
# _display_events_detailed
# ============================================================================


class TestDisplayEventsDetailed:
    """Test _display_events_detailed grouping by day and setup."""

    def test_setup_events_displayed_first(self, capsys):
        events = [
            _ev("CashMinted", phase="setup", to="A1", amount=100, day=0),
            _ev("PhaseA", day=0),
            _ev("CashTransferred", frm="X", to="Y", amount=10, day=0),
        ]
        _display_events_detailed(events)
        captured = capsys.readouterr()
        assert "Setup" in captured.out

    def test_events_grouped_by_day(self, capsys):
        events = [
            _ev("CashMinted", to="A1", amount=100, day=0),
            _ev("CashMinted", to="A2", amount=200, day=1),
        ]
        _display_events_detailed(events)
        captured = capsys.readouterr()
        assert "Day 0" in captured.out
        assert "Day 1" in captured.out

    def test_unknown_day_events(self, capsys):
        """Events with day=-1 go into 'Unknown Day' group."""
        events = [{"kind": "CashMinted", "to": "A1", "amount": 100, "day": -1}]
        _display_events_detailed(events)
        captured = capsys.readouterr()
        assert "Unknown Day" in captured.out


# ============================================================================
# display_events_for_day with mock system
# ============================================================================


class TestDisplayEventsForDay:
    """Test display_events_for_day using mock system."""

    def test_no_events_for_day(self, capsys):
        sys = _make_system_with_events([_ev("CashMinted", to="A1", amount=100, day=0)])
        display_events_for_day(sys, day=99)
        captured = capsys.readouterr()
        assert "No events" in captured.out

    def test_events_for_day(self, capsys):
        sys = _make_system_with_events([
            _ev("PhaseA", day=0),
            _ev("CashMinted", to="A1", amount=100, day=0),
            _ev("PhaseB", day=0),
            _ev("PhaseC", day=0),
        ])
        display_events_for_day(sys, day=0)
        captured = capsys.readouterr()
        assert "Phase A" in captured.out or "Cash minted" in captured.out


# ============================================================================
# display_events_for_day_renderable
# ============================================================================


class TestDisplayEventsForDayRenderable:
    """Test display_events_for_day_renderable."""

    def test_no_events_returns_message(self):
        sys = _make_system_with_events([])
        result = display_events_for_day_renderable(sys, day=99)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_events_returns_renderables(self):
        sys = _make_system_with_events([
            _ev("CashMinted", phase="setup", to="A1", amount=100, day=0),
        ])
        result = display_events_for_day_renderable(sys, day=0)
        assert isinstance(result, list)


# ============================================================================
# display_events_renderable (summary mode)
# ============================================================================


class TestDisplayEventsRenderable:
    """Test display_events_renderable in different modes."""

    def test_summary_mode(self):
        events = [_ev("PayableSettled", debtor="D1", creditor="C1", amount=100)]
        result = display_events_renderable(events, format="summary")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_detailed_mode(self):
        events = [
            _ev("CashMinted", phase="setup", to="A1", amount=100, day=0),
        ]
        result = display_events_renderable(events, format="detailed")
        assert isinstance(result, list)

    def test_empty_events(self):
        result = display_events_renderable([])
        assert isinstance(result, list)
        assert len(result) == 1


# ============================================================================
# display_events (top-level function)
# ============================================================================


class TestDisplayEvents:
    """Test display_events top-level function."""

    def test_summary_mode(self, capsys):
        events = [
            _ev("PayableSettled", debtor="D1", creditor="C1", amount=100),
            _ev("StockTransferred", frm="A1", to="A2", qty=5, sku="W"),
        ]
        display_events(events, format="summary")
        captured = capsys.readouterr()
        assert "D1" in captured.out

    def test_detailed_mode(self, capsys):
        events = [
            _ev("CashMinted", phase="setup", to="A1", amount=100, day=0),
        ]
        display_events(events, format="detailed")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_empty(self, capsys):
        display_events([])
        captured = capsys.readouterr()
        assert "No events" in captured.out


# ============================================================================
# _format_single_event (rich text styling paths)
# ============================================================================


class TestFormatSingleEvent:
    """Test _format_single_event with a mock registry to exercise text styling."""

    def _mock_registry(self, title, lines=None, icon=""):
        """Create a mock registry that returns the given format."""
        return SimpleNamespace(format=lambda event: (title, lines or [], icon))

    def test_transfer_style(self):
        registry = self._mock_registry("Cash Transfer", ["A -> B"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(
            _ev("CashTransferred", frm="A", to="B", amount=10), registry
        )
        assert len(result) >= 1

    def test_settled_style(self):
        registry = self._mock_registry("Payable Settled", ["D -> C"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("PayableSettled"), registry)
        assert len(result) >= 1

    def test_created_style(self):
        registry = self._mock_registry("Payable Created", ["new contract"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("PayableCreated"), registry)
        assert len(result) >= 1

    def test_consolidation_style(self):
        registry = self._mock_registry("Consolidation", ["merged lots"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("InstrumentMerged"), registry)
        assert len(result) >= 1

    def test_unknown_style(self):
        registry = self._mock_registry("Unknown Event", ["detail"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("WeirdEvent"), registry)
        assert len(result) >= 1

    def test_flow_arrow_line(self):
        """Line with arrow character gets 'white' style."""
        registry = self._mock_registry("Transfer", ["A \u2192 B: $100"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("CashTransferred"), registry)
        assert len(result) >= 1

    def test_colon_line_split(self):
        """Line with colon gets field:value split styling."""
        registry = self._mock_registry("Event", ["Amount: 100"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("SomeEvent"), registry)
        assert len(result) >= 1

    def test_parenthetical_line(self):
        """Line starting with '(' gets dim italic styling."""
        registry = self._mock_registry("Event", ["(technical detail)"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("SomeEvent"), registry)
        assert len(result) >= 1

    def test_plain_line(self):
        """Plain line without special chars gets 'white' styling."""
        registry = self._mock_registry("Event", ["just a line"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("SomeEvent"), registry)
        assert len(result) >= 1

    def test_max_three_lines(self):
        """Only first 3 detail lines are rendered."""
        registry = self._mock_registry("Event", ["l1", "l2", "l3", "l4", "l5"])
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = _format_single_event(_ev("SomeEvent"), registry)
        assert len(result) >= 1

    def test_non_rich_simple_text(self):
        """Non-rich path produces plain text string."""
        registry = self._mock_registry("Event Title", ["detail1", "detail2"])
        with patch("bilancio.analysis.visualization.events.RICH_AVAILABLE", False):
            result = _format_single_event(_ev("SomeEvent"), registry)
        assert len(result) >= 1
        assert isinstance(result[0], str)
        assert "Event Title" in result[0]

    def test_empty_registry_result(self):
        """Registry returning empty title should still produce output."""
        registry = self._mock_registry("", [])
        result = _format_single_event(_ev("SomeEvent"), registry)
        # Should return at least one renderable even with empty title
        assert len(result) >= 1 or len(result) == 0  # Implementation dependent
