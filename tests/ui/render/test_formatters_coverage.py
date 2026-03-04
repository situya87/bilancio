"""Coverage tests for ui/render/formatters.py.

Targets all uncovered formatter functions: DeliveryObligationCreated,
DeliveryObligationSettled, CashWithdrawn, IntraBankPayment, CashPayment,
InstrumentMerged, StockSplit, DeliveryObligationCancelled.
"""

from __future__ import annotations

from bilancio.ui.render.formatters import registry


# ============================================================================
# Uncovered formatter functions
# ============================================================================


class TestDeliveryObligationCreated:
    """Cover format_delivery_obligation_created."""

    def test_basic(self):
        event = {
            "kind": "DeliveryObligationCreated",
            "sku": "WIDGET",
            "quantity": 10,
            "frm": "F1",
            "to": "F2",
            "due_day": 5,
        }
        title, lines, icon = registry.format(event)
        assert "Delivery Obligation" in title
        assert "WIDGET" in title
        assert "10" in title
        assert "F1" in lines[0] and "F2" in lines[0]
        assert "Day 5" in lines[1]

    def test_no_due_day(self):
        event = {
            "kind": "DeliveryObligationCreated",
            "sku": "BOLT",
            "qty": 3,
            "frm": "F1",
            "to": "F2",
        }
        title, lines, icon = registry.format(event)
        assert "BOLT" in title
        assert len(lines) == 1  # No due_day line


class TestDeliveryObligationSettled:
    """Cover format_delivery_obligation_settled."""

    def test_basic(self):
        event = {
            "kind": "DeliveryObligationSettled",
            "sku": "WIDGET",
            "quantity": 10,
            "debtor": "F1",
            "creditor": "F2",
        }
        title, lines, icon = registry.format(event)
        assert "Delivery Settled" in title
        assert "10" in title
        assert "WIDGET" in title
        assert "F1" in lines[0] and "F2" in lines[0]

    def test_uses_qty_fallback(self):
        event = {
            "kind": "DeliveryObligationSettled",
            "sku": "NUT",
            "qty": 5,
            "debtor": "A",
            "creditor": "B",
        }
        title, lines, icon = registry.format(event)
        assert "5" in title


class TestCashWithdrawn:
    """Cover format_cash_withdrawn."""

    def test_basic(self):
        event = {
            "kind": "CashWithdrawn",
            "customer": "H1",
            "bank": "B1",
            "amount": 200,
        }
        title, lines, icon = registry.format(event)
        assert "Cash Withdrawal" in title
        assert "200" in title
        assert "H1" in lines[0]
        assert "B1" in lines[0]


class TestIntraBankPayment:
    """Cover format_intra_bank_payment."""

    def test_basic(self):
        event = {
            "kind": "IntraBankPayment",
            "payer": "H1",
            "payee": "H2",
            "amount": 100,
            "bank": "B1",
        }
        title, lines, icon = registry.format(event)
        assert "Intra-Bank Payment" in title
        assert "100" in title
        assert "H1" in lines[0] and "H2" in lines[0]
        assert "B1" in lines[1]


class TestCashPayment:
    """Cover format_cash_payment."""

    def test_basic(self):
        event = {
            "kind": "CashPayment",
            "payer": "H1",
            "payee": "H2",
            "amount": 50,
        }
        title, lines, icon = registry.format(event)
        assert "Cash Payment" in title
        assert "50" in title
        assert "H1" in lines[0] and "H2" in lines[0]


class TestInstrumentMerged:
    """Cover format_instrument_merged."""

    def test_basic(self):
        event = {
            "kind": "InstrumentMerged",
            "keep": "C_abc12345",
            "removed": "C_def67890",
        }
        title, lines, icon = registry.format(event)
        assert "Cash Consolidation" in title
        assert "Merged" in lines[0]
        assert "Reduces fragmentation" in lines[1]

    def test_unknown_values(self):
        event = {
            "kind": "InstrumentMerged",
        }
        title, lines, icon = registry.format(event)
        assert "Cash Consolidation" in title
        assert "Unknown" in lines[0]


class TestStockSplit:
    """Cover format_stock_split."""

    def test_basic(self):
        event = {
            "kind": "StockSplit",
            "sku": "WIDGET",
            "original_qty": 10,
            "split_qty": 3,
            "remaining_qty": 7,
        }
        title, lines, icon = registry.format(event)
        assert "Stock Split" in title
        assert "3" in title
        assert "WIDGET" in title
        assert "10" in lines[0]
        assert "7" in lines[0]
        assert "Preparing transfer" in lines[1]


class TestDeliveryObligationCancelled:
    """Cover format_delivery_cancelled."""

    def test_basic(self):
        event = {
            "kind": "DeliveryObligationCancelled",
            "obligation_id": "OBL_abc12345",
            "debtor": "F1",
        }
        title, lines, icon = registry.format(event)
        assert "Obligation Cleared" in title
        assert "F1" in lines[0]
        assert "abc12345" in lines[1]

    def test_unknown_obligation_id(self):
        event = {
            "kind": "DeliveryObligationCancelled",
        }
        title, lines, icon = registry.format(event)
        assert "Obligation Cleared" in title
        assert "Unknown" in lines[0]
        assert "Unknown" in lines[1]


# ============================================================================
# Generic formatter fallback
# ============================================================================


class TestGenericFormatter:
    """Cover the generic formatter for unknown event kinds."""

    def test_unknown_kind(self):
        event = {
            "kind": "SomeNewEventKind",
            "custom_field": "value1",
            "another": 42,
        }
        title, lines, icon = registry.format(event)
        assert "SomeNewEventKind Event" in title
        assert any("custom_field" in line for line in lines)

    def test_unknown_kind_limits_to_3_lines(self):
        event = {
            "kind": "BigEvent",
            "field1": "a",
            "field2": "b",
            "field3": "c",
            "field4": "d",
            "field5": "e",
        }
        title, lines, icon = registry.format(event)
        assert len(lines) <= 3

    def test_skips_meta_fields(self):
        event = {
            "kind": "TestKind",
            "day": 1,
            "phase": "B",
            "type": "test",
            "real_data": "important",
        }
        title, lines, icon = registry.format(event)
        # day, phase, type should be skipped
        assert not any("day" in line for line in lines)
        assert any("real_data" in line for line in lines)


# ============================================================================
# Already-covered formatters - verify they still work
# ============================================================================


class TestCoveredFormattersStillWork:
    """Quick smoke tests for already-covered formatters."""

    def test_cash_transferred(self):
        event = {"kind": "CashTransferred", "amount": 1000, "frm": "A", "to": "B"}
        title, lines, icon = registry.format(event)
        assert "1,000" in title

    def test_reserves_transferred(self):
        event = {"kind": "ReservesTransferred", "amount": 500, "frm": "B1", "to": "B2"}
        title, lines, icon = registry.format(event)
        assert "Reserves Transfer" in title

    def test_stock_transferred(self):
        event = {
            "kind": "StockTransferred",
            "sku": "GEAR",
            "qty": 5,
            "frm": "F1",
            "to": "H1",
            "unit_price": 100,
        }
        title, lines, icon = registry.format(event)
        assert "GEAR" in title
        assert len(lines) == 2  # from->to + unit price

    def test_stock_transferred_no_price(self):
        event = {
            "kind": "StockTransferred",
            "sku": "GEAR",
            "qty": 5,
            "frm": "F1",
            "to": "H1",
        }
        title, lines, icon = registry.format(event)
        assert len(lines) == 1  # just from->to, no price

    def test_payable_created_with_due_day(self):
        event = {"kind": "PayableCreated", "amount": 200, "debtor": "H1", "creditor": "H2", "due_day": 3}
        title, lines, icon = registry.format(event)
        assert "Day 3" in lines[1]

    def test_payable_created_no_due_day(self):
        event = {"kind": "PayableCreated", "amount": 200, "frm": "H1", "to": "H2"}
        title, lines, icon = registry.format(event)
        assert len(lines) == 1

    def test_stock_created_with_unit_price(self):
        event = {"kind": "StockCreated", "owner": "F1", "sku": "W", "qty": 10, "unit_price": 50}
        title, lines, icon = registry.format(event)
        assert "Value" in lines[1]
        assert "total" in lines[1]

    def test_stock_created_with_string_price(self):
        event = {"kind": "StockCreated", "owner": "F1", "sku": "W", "qty": "ten", "unit_price": "fifty"}
        title, lines, icon = registry.format(event)
        # Non-numeric qty/price fall through to else branch
        assert "Value" in lines[1]

    def test_stock_created_no_price(self):
        event = {"kind": "StockCreated", "owner": "F1", "sku": "W", "qty": 10}
        title, lines, icon = registry.format(event)
        assert len(lines) == 1  # just owner

    def test_phase_markers(self):
        title_a, _, _ = registry.format({"kind": "PhaseA", "day": 5})
        assert "Day 5" in title_a

        title_b, _, _ = registry.format({"kind": "PhaseB"})
        assert "Business hours" in title_b

        title_c, _, _ = registry.format({"kind": "PhaseC"})
        assert "End of day" in title_c
