"""Coverage tests for bilancio.ui.render.formatters.

Focuses on uncovered formatter paths:
- ReservesTransferred, DeliveryObligationCreated, DeliveryObligationSettled
- PayableCreated with due_day, PayableSettled
- CashDeposited, CashWithdrawn, ClientPayment, IntraBankPayment, CashPayment
- InstrumentMerged, InterbankCleared, ReservesMinted, StockSplit
- DeliveryObligationCancelled
- StockCreated with unit_price and total value calculation
- StockTransferred with unit_price
- Generic fallback with many fields (truncation to 3 lines)
"""

from bilancio.ui.render.formatters import registry


class TestReservesTransferred:
    def test_basic(self):
        event = {"kind": "ReservesTransferred", "frm": "bank_1", "to": "bank_2", "amount": 500}
        title, lines, icon = registry.format(event)
        assert "[BANK]" in title
        assert "Reserves Transfer" in title
        assert "$500" in title
        assert "bank_1 → bank_2" in lines[0]
        assert icon == "[BANK]"

    def test_missing_fields(self):
        event = {"kind": "ReservesTransferred"}
        title, lines, icon = registry.format(event)
        assert "Unknown → Unknown" in lines[0]


class TestDeliveryObligationCreated:
    def test_with_due_day(self):
        event = {
            "kind": "DeliveryObligationCreated",
            "sku": "WHEAT",
            "quantity": 50,
            "frm": "farm",
            "to": "mill",
            "due_day": 3,
        }
        title, lines, icon = registry.format(event)
        assert "[DOC]" in title
        assert "WHEAT" in title
        assert "farm → mill" in lines[0]
        assert "Due: Day 3" in lines[1]
        assert icon == "[DOC]"

    def test_without_due_day(self):
        event = {"kind": "DeliveryObligationCreated", "sku": "OIL", "qty": 10, "frm": "a", "to": "b"}
        title, lines, icon = registry.format(event)
        assert "OIL" in title
        assert len(lines) == 1  # no due_day line

    def test_uses_qty_fallback(self):
        event = {"kind": "DeliveryObligationCreated", "qty": 7, "frm": "a", "to": "b"}
        title, lines, _ = registry.format(event)
        assert "7" in title


class TestDeliveryObligationSettled:
    def test_basic(self):
        event = {
            "kind": "DeliveryObligationSettled",
            "sku": "STEEL",
            "quantity": 20,
            "debtor": "factory",
            "creditor": "supplier",
        }
        title, lines, icon = registry.format(event)
        assert "[OK]" in title
        assert "STEEL" in title
        assert "factory → supplier" in lines[0]


class TestPayableCreated:
    def test_with_due_day(self):
        event = {
            "kind": "PayableCreated",
            "amount": 1000,
            "debtor": "H1",
            "creditor": "H2",
            "due_day": 5,
        }
        title, lines, icon = registry.format(event)
        assert "[PAY]" in title
        assert "$1,000" in title
        assert "H1 owes H2" in lines[0]
        assert "Due: Day 5" in lines[1]

    def test_without_due_day(self):
        event = {"kind": "PayableCreated", "amount": 200, "frm": "A", "to": "B"}
        title, lines, _ = registry.format(event)
        assert "A owes B" in lines[0]
        assert len(lines) == 1

    def test_due_day_zero(self):
        event = {"kind": "PayableCreated", "amount": 100, "debtor": "X", "creditor": "Y", "due_day": 0}
        title, lines, _ = registry.format(event)
        assert "Due: Day 0" in lines[1]


class TestPayableSettled:
    def test_basic(self):
        event = {"kind": "PayableSettled", "amount": 750, "debtor": "H1", "creditor": "H2"}
        title, lines, icon = registry.format(event)
        assert "Payable Settled" in title
        assert "$750" in title
        assert "H1 → H2" in lines[0]
        assert icon == "$"


class TestCashDeposited:
    def test_basic(self):
        event = {"kind": "CashDeposited", "customer": "H1", "bank": "bank_1", "amount": 500}
        title, lines, icon = registry.format(event)
        assert "[ATM]" in title
        assert "$500" in title
        assert "H1 → bank_1" in lines[0]


class TestCashWithdrawn:
    def test_basic(self):
        event = {"kind": "CashWithdrawn", "customer": "H2", "bank": "bank_2", "amount": 300}
        title, lines, icon = registry.format(event)
        assert "[PAY]" in title
        assert "$300" in title
        assert "H2 <- bank_2" in lines[0]


class TestClientPayment:
    def test_basic(self):
        event = {
            "kind": "ClientPayment",
            "payer": "H1",
            "payee": "H2",
            "amount": 200,
            "payer_bank": "bank_1",
            "payee_bank": "bank_2",
        }
        title, lines, icon = registry.format(event)
        assert "[CARD]" in title
        assert "$200" in title
        assert "H1 → H2" in lines[0]
        assert "via bank_1 → bank_2" in lines[1]


class TestIntraBankPayment:
    def test_basic(self):
        event = {
            "kind": "IntraBankPayment",
            "payer": "H1",
            "payee": "H2",
            "amount": 150,
            "bank": "bank_1",
        }
        title, lines, icon = registry.format(event)
        assert "[BANK]" in title
        assert "$150" in title
        assert "H1 → H2" in lines[0]
        assert "at bank_1" in lines[1]


class TestCashPayment:
    def test_basic(self):
        event = {"kind": "CashPayment", "payer": "H1", "payee": "H2", "amount": 100}
        title, lines, icon = registry.format(event)
        assert "[CASH]" in title
        assert "$100" in title
        assert "H1 → H2" in lines[0]


class TestInstrumentMerged:
    def test_basic(self):
        event = {"kind": "InstrumentMerged", "keep": "cash_abc12345", "removed": "cash_def67890"}
        title, lines, icon = registry.format(event)
        assert "[MRG]" in title
        assert "Cash Consolidation" in title
        assert "Merged:" in lines[0]
        assert "(Reduces fragmentation)" in lines[1]

    def test_unknown_ids(self):
        event = {"kind": "InstrumentMerged", "keep": "Unknown", "removed": "Unknown"}
        title, lines, _ = registry.format(event)
        assert "Unknown" in lines[0]


class TestInterbankCleared:
    def test_basic(self):
        event = {"kind": "InterbankCleared", "debtor_bank": "bank_1", "creditor_bank": "bank_2", "amount": 1000}
        title, lines, icon = registry.format(event)
        assert "[CLR]" in title
        assert "$1,000" in title
        assert "bank_1 → bank_2" in lines[0]


class TestReservesMinted:
    def test_basic(self):
        event = {"kind": "ReservesMinted", "to": "bank_1", "amount": 5000}
        title, lines, icon = registry.format(event)
        assert "[RSV]" in title
        assert "$5,000" in title
        assert "Bank: bank_1" in lines[0]


class TestStockSplit:
    def test_basic(self):
        event = {
            "kind": "StockSplit",
            "sku": "BREAD",
            "original_qty": 100,
            "split_qty": 30,
            "remaining_qty": 70,
        }
        title, lines, icon = registry.format(event)
        assert "[SPLIT]" in title
        assert "30 BREAD" in title
        assert "100" in lines[0]
        assert "70" in lines[0]


class TestDeliveryObligationCancelled:
    def test_basic(self):
        event = {
            "kind": "DeliveryObligationCancelled",
            "obligation_id": "DOB_abc12345",
            "debtor": "H1",
        }
        title, lines, icon = registry.format(event)
        assert "[OK]" in title
        assert "Obligation Cleared" in title
        assert "By: H1" in lines[0]
        assert "abc12345" in lines[1]

    def test_unknown_obligation_id(self):
        event = {"kind": "DeliveryObligationCancelled", "obligation_id": "Unknown", "debtor": "H1"}
        title, lines, _ = registry.format(event)
        assert "Unknown" in lines[1]


class TestStockCreated:
    def test_with_unit_price_numeric(self):
        event = {"kind": "StockCreated", "owner": "farm", "sku": "CORN", "qty": 100, "unit_price": 5}
        title, lines, icon = registry.format(event)
        assert "100 CORN" in title
        assert "Owner: farm" in lines[0]
        assert "$5" in lines[1]
        assert "$500" in lines[1]  # total value

    def test_with_unit_price_non_numeric(self):
        event = {"kind": "StockCreated", "owner": "farm", "sku": "CORN", "qty": "many", "unit_price": "N/A"}
        title, lines, _ = registry.format(event)
        assert "Owner: farm" in lines[0]
        # Non-numeric goes to else branch
        assert "N/A" in lines[1]

    def test_without_unit_price(self):
        event = {"kind": "StockCreated", "owner": "farm", "sku": "OIL", "quantity": 10}
        title, lines, _ = registry.format(event)
        assert "10 OIL" in title
        assert len(lines) == 1  # only owner, no price line

    def test_uses_quantity_fallback(self):
        event = {"kind": "StockCreated", "owner": "w", "sku": "X", "quantity": 42}
        title, _, _ = registry.format(event)
        assert "42" in title


class TestStockTransferredWithPrice:
    def test_with_unit_price(self):
        event = {
            "kind": "StockTransferred",
            "sku": "WHEAT",
            "qty": 50,
            "frm": "farm",
            "to": "mill",
            "unit_price": 10,
        }
        title, lines, icon = registry.format(event)
        assert "50 WHEAT" in title
        assert "farm → mill" in lines[0]
        assert "@ $10" in lines[1]


class TestGenericFormatter:
    def test_many_fields_truncated(self):
        """Generic formatter truncates to 3 lines."""
        event = {
            "kind": "BrandNewEvent",
            "field_a": "aaa",
            "field_b": "bbb",
            "field_c": "ccc",
            "field_d": "ddd",
            "field_e": "eee",
        }
        title, lines, icon = registry.format(event)
        assert "BrandNewEvent Event" in title
        assert len(lines) == 3
        assert icon == "❓"

    def test_skips_meta_fields(self):
        """Generic formatter skips kind, day, phase, type."""
        event = {"kind": "Exotic", "day": 5, "phase": "A", "type": "x", "data": "important"}
        title, lines, _ = registry.format(event)
        assert "data: important" in lines[0]
        assert len(lines) == 1  # Only the data field, meta skipped

    def test_no_kind(self):
        """Event without kind uses 'Unknown'."""
        event = {"data": "x"}
        title, lines, _ = registry.format(event)
        assert "Unknown Event" in title


class TestCashMinted:
    def test_basic(self):
        event = {"kind": "CashMinted", "to": "H5", "amount": 999}
        title, lines, icon = registry.format(event)
        assert "[MINT]" in title
        assert "$999" in title
        assert "To: H5" in lines[0]


class TestPhaseFormatters:
    def test_phase_a(self):
        event = {"kind": "PhaseA", "day": 3}
        title, lines, icon = registry.format(event)
        assert "Day 3" in title
        assert icon == "[TIME]"

    def test_phase_b(self):
        event = {"kind": "PhaseB"}
        title, lines, icon = registry.format(event)
        assert "Business hours" in title
        assert icon == "[MORN]"

    def test_phase_c(self):
        event = {"kind": "PhaseC"}
        title, lines, icon = registry.format(event)
        assert "End of day" in title
        assert icon == "[NIGHT]"
