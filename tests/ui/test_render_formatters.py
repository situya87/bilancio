"""Test event formatters and registry."""

from bilancio.ui.render.formatters import EventFormatterRegistry, registry


def test_event_formatter_registry():
    """Test that event formatter registry works correctly."""
    registry = EventFormatterRegistry()

    # Test registration
    @registry.register("TestEvent")
    def test_formatter(event):
        return "Test Event", ["Test line"], "🧪"

    # Test formatting with registered event
    event = {"kind": "TestEvent", "data": "test"}
    title, lines, icon = registry.format(event)
    assert title == "Test Event"
    assert lines == ["Test line"]
    assert icon == "🧪"


def test_format_event_with_known_kind():
    """Test formatting known event kinds."""
    # Test CashTransferred
    event = {"kind": "CashTransferred", "frm": "bank1", "to": "household1", "amount": 1000}
    title, lines, icon = registry.format(event)
    assert "Cash Transfer" in title
    assert lines[0] == "bank1 → household1"
    assert icon == "💰"

    # Test StockTransferred
    event = {
        "kind": "StockTransferred",
        "frm": "firm1",
        "to": "household1",
        "qty": 10,
        "sku": "BREAD",
    }
    title, lines, icon = registry.format(event)
    assert "Stock Transfer" in title
    assert lines[0] == "firm1 → household1"
    assert icon == "📦"


def test_format_event_with_unknown_kind():
    """Test formatting unknown event kinds."""
    event = {"kind": "UnknownEventType", "some_data": "value"}
    title, lines, icon = registry.format(event)
    assert "UnknownEventType Event" in title
    assert len(lines) > 0
    assert icon == "❓"


def test_format_event_missing_fields():
    """Test formatting events with missing expected fields."""
    # CashTransferred with missing amount
    event = {"kind": "CashTransferred", "frm": "bank1", "to": "household1"}
    title, lines, icon = registry.format(event)
    assert "Cash Transfer" in title
    assert lines[0] == "bank1 → household1"  # Should handle missing amount gracefully

    # StockTransferred with missing qty
    event = {"kind": "StockTransferred", "frm": "firm1", "to": "household1", "sku": "BREAD"}
    title, lines, icon = registry.format(event)
    assert "Stock Transfer" in title
    assert lines[0] == "firm1 → household1"  # Should handle missing qty gracefully


def test_all_event_kinds_have_formatters():
    """Test that common event kinds have formatters."""
    common_kinds = [
        "CashTransferred",
        "StockTransferred",
        "PayableSettled",
        "DeliveryObligationSettled",
        "CashMinted",
        "StockCreated",
        "PhaseA",
        "PhaseB",
        "PhaseC",
    ]

    for kind in common_kinds:
        event = {"kind": kind}
        title, lines, icon = registry.format(event)
        # Should not fall back to generic formatter
        assert "Event:" not in title or kind in ["PhaseA", "PhaseB", "PhaseC"]
