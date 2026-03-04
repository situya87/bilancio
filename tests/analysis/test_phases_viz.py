"""Tests for bilancio.analysis.visualization.phases module.

Covers:
- Phase bucketing logic (PhaseA, PhaseB, PhaseC markers)
- Setup table building
- Rich and non-rich fallback paths
- Various event kinds (CashDeposited, CashWithdrawn, ClientPayment, etc.)
- Edge cases: empty events, missing fields, AgentDefaulted notes
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bilancio.analysis.visualization.phases import (
    _build_single_setup_table,
    display_events_tables_by_phase_renderables,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_event(kind: str, **kwargs) -> dict:
    """Build a minimal event dict."""
    e = {"kind": kind}
    e.update(kwargs)
    return e


def _phase_stream(*groups):
    """Build a list of events with phase markers interleaved.

    Each group is (phase_kind, [events]).  Phase marker is inserted first.
    """
    evs = []
    for marker, events in groups:
        evs.append({"kind": marker})
        evs.extend(events)
    return evs


# ============================================================================
# Phase bucketing
# ============================================================================


class TestPhaseBucketing:
    """Test that events are correctly bucketed by phase markers."""

    def test_events_split_into_phases(self):
        """Events after PhaseB go into B, after PhaseC into C."""
        events = _phase_stream(
            ("PhaseA", [_make_event("CashMinted", to="A1", amount=100)]),
            (
                "PhaseB",
                [_make_event("PayableSettled", debtor="D", creditor="C", amount=50)],
            ),
            ("PhaseC", [_make_event("CashTransferred", frm="X", to="Y", amount=10)]),
        )
        result = display_events_tables_by_phase_renderables(events, day=1)
        # Should have Phase A (has rows), Phase B, Phase C = 3 tables
        assert len(result) == 3

    def test_empty_phase_a_omitted(self):
        """When Phase A has no events, its table is omitted."""
        events = _phase_stream(
            ("PhaseA", []),
            ("PhaseB", [_make_event("CashTransferred", frm="X", to="Y", amount=5)]),
            ("PhaseC", []),
        )
        result = display_events_tables_by_phase_renderables(events, day=2)
        # Phase A empty -> omitted.  B and C always present.
        assert len(result) == 2

    def test_all_phases_empty(self):
        """With only phase markers and no data events, B+C tables still returned."""
        events = [{"kind": "PhaseA"}, {"kind": "PhaseB"}, {"kind": "PhaseC"}]
        result = display_events_tables_by_phase_renderables(events, day=0)
        # B and C are always returned even when empty
        assert len(result) >= 2

    def test_no_phase_markers_defaults_to_a(self):
        """Events without any phase marker default to bucket A."""
        events = [_make_event("CashMinted", to="A1", amount=100)]
        result = display_events_tables_by_phase_renderables(events)
        # Should have Phase A (has rows) + B + C
        assert len(result) == 3


# ============================================================================
# Setup table path
# ============================================================================


class TestSetupTable:
    """Test _build_single_setup_table and the setup detection path."""

    def test_setup_events_detected(self):
        """When events contain phase='setup', a single setup table is returned."""
        events = [
            _make_event("CashMinted", phase="setup", to="A1", amount=100),
            _make_event("StockCreated", phase="setup", owner="A1", qty=10, sku="W"),
        ]
        result = display_events_tables_by_phase_renderables(events, day=0)
        assert len(result) == 1  # Single setup table

    def test_setup_table_with_day(self):
        """Setup table title includes day number."""
        events = [_make_event("CashMinted", phase="setup", to="A1", amount=100)]
        result = _build_single_setup_table(events, day=5)
        assert len(result) == 1

    def test_setup_table_without_day(self):
        """Setup table works without day parameter."""
        events = [_make_event("CashMinted", phase="setup", to="A1", amount=100)]
        result = _build_single_setup_table(events, day=None)
        assert len(result) == 1

    def test_setup_excludes_phase_markers(self):
        """Phase markers (PhaseA/B/C) are excluded from setup table rows."""
        events = [
            _make_event("CashMinted", phase="setup", to="A1", amount=100),
            _make_event("PhaseA", phase="setup"),
        ]
        result = _build_single_setup_table(events, day=0)
        assert len(result) == 1


# ============================================================================
# Event kind from/to mapping (non-rich fallback exercises the text path)
# ============================================================================


class TestEventKindMapping:
    """Test from/to extraction for various event kinds in the non-rich fallback."""

    @pytest.fixture(autouse=True)
    def _force_no_rich(self):
        """Force RICH_AVAILABLE=False so the simple text path is exercised."""
        with patch("bilancio.analysis.visualization.phases.RICH_AVAILABLE", False):
            yield

    def _get_text(self, events, day=0):
        """Return the concatenated text output of all renderables."""
        result = display_events_tables_by_phase_renderables(events, day=day)
        return "\n".join(str(r) for r in result)

    def test_cash_deposited_mapping(self):
        events = _phase_stream(
            ("PhaseB", [_make_event("CashDeposited", customer="HH01", bank="BK01", amount=100)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "CashDeposited" in text
        assert "HH01" in text

    def test_cash_withdrawn_mapping(self):
        events = _phase_stream(
            ("PhaseB", [_make_event("CashWithdrawn", customer="HH01", bank="BK01", amount=50)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "CashWithdrawn" in text
        assert "BK01" in text  # frm = bank for CashWithdrawn

    def test_client_payment_notes(self):
        events = _phase_stream(
            (
                "PhaseB",
                [
                    _make_event(
                        "ClientPayment",
                        payer="HH01",
                        payee="HH02",
                        amount=100,
                        payer_bank="BK01",
                        payee_bank="BK02",
                    )
                ],
            ),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "BK01" in text
        assert "BK02" in text

    def test_intra_bank_payment(self):
        events = _phase_stream(
            (
                "PhaseB",
                [_make_event("IntraBankPayment", payer="HH01", payee="HH02", amount=50)],
            ),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "IntraBankPayment" in text
        assert "HH01" in text

    def test_cash_payment(self):
        events = _phase_stream(
            ("PhaseB", [_make_event("CashPayment", payer="A1", payee="A2", amount=25)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "CashPayment" in text

    def test_interbank_cleared_notes(self):
        events = _phase_stream(
            (
                "PhaseC",
                [
                    _make_event(
                        "InterbankCleared",
                        debtor_bank="BK01",
                        creditor_bank="BK02",
                        amount=200,
                    )
                ],
            ),
        )
        text = self._get_text(events)
        assert "InterbankCleared" in text
        assert "BK01" in text

    def test_interbank_overnight_with_due_day(self):
        events = _phase_stream(
            (
                "PhaseC",
                [
                    _make_event(
                        "InterbankOvernightCreated",
                        debtor_bank="BK01",
                        creditor_bank="BK02",
                        amount=300,
                        due_day=5,
                    )
                ],
            ),
        )
        text = self._get_text(events)
        assert "due 5" in text

    def test_stock_created_mapping(self):
        events = _phase_stream(
            ("PhaseB", [_make_event("StockCreated", owner="A1", qty=10, sku="W")]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "StockCreated" in text
        assert "A1" in text

    def test_agent_defaulted_notes(self):
        events = _phase_stream(
            (
                "PhaseB",
                [
                    _make_event(
                        "AgentDefaulted",
                        debtor="D1",
                        shortfall=500,
                        trigger_contract="P-001",
                    )
                ],
            ),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "shortfall 500" in text
        assert "trigger P-001" in text

    def test_agent_defaulted_shortfall_only(self):
        events = _phase_stream(
            ("PhaseB", [_make_event("AgentDefaulted", debtor="D1", shortfall=100)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "shortfall 100" in text

    def test_agent_defaulted_no_extra_info(self):
        """AgentDefaulted with no shortfall or trigger should produce empty notes."""
        events = _phase_stream(
            ("PhaseB", [_make_event("AgentDefaulted", debtor="D1")]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "AgentDefaulted" in text

    def test_generic_event_fallback(self):
        """Unknown event kind falls back to frm/from/debtor/payer."""
        events = _phase_stream(
            (
                "PhaseB",
                [_make_event("CustomEvent", debtor="D1", creditor="C1", amount=42)],
            ),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "CustomEvent" in text
        assert "D1" in text

    def test_sku_instr_stock_fallback(self):
        """SKU field tries sku, then instr_id, then stock_id."""
        events = _phase_stream(
            ("PhaseB", [_make_event("SomeEvent", instr_id="INS-001", qty=5)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "INS-001" in text

    def test_qty_quantity_fallback(self):
        """Quantity field tries qty first, then quantity."""
        events = _phase_stream(
            ("PhaseB", [_make_event("SomeEvent", quantity=7)]),
            ("PhaseC", []),
        )
        text = self._get_text(events)
        assert "7" in text

    def test_empty_events_list(self):
        """Empty event list still returns at least B and C tables."""
        result = display_events_tables_by_phase_renderables([])
        # B + C always returned
        assert len(result) >= 2


# ============================================================================
# Rich path (with actual Rich objects)
# ============================================================================


class TestRichPath:
    """Test the Rich rendering path produces proper table objects."""

    @pytest.fixture(autouse=True)
    def _check_rich(self):
        from bilancio.analysis.visualization.common import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich library not available")

    def test_rich_phase_tables_type(self):
        """Rich path returns Rich Table objects."""
        from rich.table import Table as RichTable

        events = _phase_stream(
            ("PhaseA", [_make_event("CashMinted", to="A1", amount=100)]),
            ("PhaseB", [_make_event("PayableSettled", debtor="D", creditor="C", amount=50)]),
            ("PhaseC", []),
        )
        result = display_events_tables_by_phase_renderables(events, day=1)
        # Each renderable should be a RichTable
        for r in result:
            assert isinstance(r, RichTable)

    def test_rich_setup_table_type(self):
        """Rich setup table returns Rich Table."""
        from rich.table import Table as RichTable

        events = [
            _make_event("CashMinted", phase="setup", to="A1", amount=100),
        ]
        result = display_events_tables_by_phase_renderables(events, day=0)
        assert len(result) == 1
        assert isinstance(result[0], RichTable)

    def test_rich_phase_c_alternating_row_style(self):
        """Phase C tables should get different alternating row colors."""
        events = _phase_stream(
            ("PhaseC", [
                _make_event("CashTransferred", frm="X", to="Y", amount=10),
                _make_event("CashTransferred", frm="A", to="B", amount=20),
            ]),
        )
        result = display_events_tables_by_phase_renderables(events, day=0)
        # Phase B (empty) + Phase C (with rows)
        assert len(result) == 2

    def test_rich_all_event_kinds_in_phase(self):
        """All event kinds are rendered without error in rich path."""
        events = _phase_stream(
            (
                "PhaseB",
                [
                    _make_event("CashDeposited", customer="HH01", bank="BK01", amount=100),
                    _make_event("CashWithdrawn", customer="HH02", bank="BK02", amount=50),
                    _make_event("ClientPayment", payer="HH01", payee="HH02", amount=25, payer_bank="BK01", payee_bank="BK02"),
                    _make_event("InterbankCleared", debtor_bank="BK01", creditor_bank="BK02", amount=200),
                    _make_event("InterbankOvernightCreated", debtor_bank="BK01", creditor_bank="BK02", amount=100, due_day=3),
                    _make_event("StockCreated", owner="A1", qty=10, sku="W"),
                    _make_event("AgentDefaulted", debtor="D1", shortfall=500, trigger_contract="P-001"),
                    _make_event("GenericEvent", debtor="G1", creditor="G2", amount=42),
                ],
            ),
            ("PhaseC", []),
        )
        result = display_events_tables_by_phase_renderables(events, day=1)
        assert len(result) >= 2

    def test_rich_day_included_in_title(self):
        """Day number is included in the table title when provided."""
        from rich.table import Table as RichTable

        events = _phase_stream(
            ("PhaseB", [_make_event("CashMinted", to="A1", amount=10)]),
            ("PhaseC", []),
        )
        result = display_events_tables_by_phase_renderables(events, day=7)
        for r in result:
            if isinstance(r, RichTable) and r.title:
                assert "Day 7" in str(r.title)

    def test_rich_no_day_in_title(self):
        """When day is None, title does not include 'Day'."""
        events = _phase_stream(
            ("PhaseB", [_make_event("CashMinted", to="A1", amount=10)]),
            ("PhaseC", []),
        )
        result = display_events_tables_by_phase_renderables(events, day=None)
        assert len(result) >= 2

    def test_rich_setup_table_with_varied_events(self):
        """Rich setup table handles all event kinds."""
        events = [
            _make_event("CashDeposited", phase="setup", customer="HH01", bank="BK01", amount=100),
            _make_event("CashWithdrawn", phase="setup", customer="HH02", bank="BK02", amount=50),
            _make_event("ClientPayment", phase="setup", payer="P1", payee="P2", amount=10, payer_bank="B1", payee_bank="B2"),
            _make_event("IntraBankPayment", phase="setup", payer="P3", payee="P4", amount=15),
            _make_event("InterbankCleared", phase="setup", debtor_bank="BK01", creditor_bank="BK02", amount=200),
            _make_event("InterbankOvernightCreated", phase="setup", debtor_bank="BK01", creditor_bank="BK02", amount=100, due_day=3),
            _make_event("StockCreated", phase="setup", owner="A1", qty=10, sku="W"),
            _make_event("GenericEvent", phase="setup", debtor="G1", creditor="G2", amount=42),
        ]
        result = display_events_tables_by_phase_renderables(events, day=0)
        assert len(result) == 1  # Single setup table


# ============================================================================
# Non-rich path for setup table
# ============================================================================


class TestSetupTableNonRich:
    """Test the non-rich fallback path in _build_single_setup_table."""

    @pytest.fixture(autouse=True)
    def _force_no_rich(self):
        with patch("bilancio.analysis.visualization.phases.RICH_AVAILABLE", False):
            yield

    def test_non_rich_setup_with_day(self):
        events = [_make_event("CashMinted", phase="setup", to="A1", amount=100)]
        result = _build_single_setup_table(events, day=3)
        assert len(result) == 1
        text = str(result[0])
        assert "Setup (Day 3)" in text

    def test_non_rich_setup_without_day(self):
        events = [_make_event("CashMinted", phase="setup", to="A1", amount=100)]
        result = _build_single_setup_table(events, day=None)
        text = str(result[0])
        assert "Setup" in text

    def test_non_rich_setup_all_event_kinds(self):
        """Non-rich setup handles all event from/to mapping kinds."""
        events = [
            _make_event("CashDeposited", phase="setup", customer="HH01", bank="BK01", amount=100),
            _make_event("CashWithdrawn", phase="setup", customer="HH02", bank="BK02", amount=50),
            _make_event("ClientPayment", phase="setup", payer="P1", payee="P2", amount=10),
            _make_event("InterbankCleared", phase="setup", debtor_bank="BK01", creditor_bank="BK02", amount=200),
            _make_event("StockCreated", phase="setup", owner="A1", qty=10, sku="W"),
            _make_event("GenericEvent", phase="setup", debtor="G1", creditor="G2", amount=42),
        ]
        result = _build_single_setup_table(events, day=0)
        text = str(result[0])
        assert "CashDeposited" in text
        assert "HH01" in text
        assert "StockCreated" in text
