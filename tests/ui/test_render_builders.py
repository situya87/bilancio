"""Test Rich builders for rendering."""

from decimal import Decimal

from bilancio.ui.render.models import AgentBalanceView, BalanceItemView, DayEventsView, EventView
from bilancio.ui.render.rich_builders import (
    build_agent_balance_table,
    build_events_panel,
    build_multiple_agent_balances,
    convert_raw_event_to_view,
)


def test_build_agent_balance_table():
    """Test building a Rich table for agent balance."""
    # Create a sample balance view
    balance_view = AgentBalanceView(
        agent_id="bank1",
        agent_name="Central Bank",
        agent_kind="bank",
        items=[
            BalanceItemView(
                category="asset", instrument="reserves", amount=10000, value=Decimal("10000")
            ),
            BalanceItemView(
                category="liability", instrument="deposits", amount=5000, value=Decimal("5000")
            ),
        ],
    )

    # Build table
    table = build_agent_balance_table(balance_view)

    # Verify it's a Table (or string if Rich not available)
    assert table is not None
    if not isinstance(table, str):
        # Rich is available, check it's a Table
        from rich.table import Table

        assert isinstance(table, Table)


def test_build_multiple_agent_balances():
    """Test building columns for multiple agent balances."""
    # Create sample balance views
    balances = [
        AgentBalanceView(
            agent_id="bank1",
            agent_name="Bank One",
            agent_kind="bank",
            items=[
                BalanceItemView(
                    category="asset", instrument="cash", amount=1000, value=Decimal("1000")
                )
            ],
        ),
        AgentBalanceView(
            agent_id="household1",
            agent_name="Household One",
            agent_kind="household",
            items=[
                BalanceItemView(
                    category="asset", instrument="deposits", amount=500, value=Decimal("500")
                )
            ],
        ),
    ]

    # Build columns
    columns = build_multiple_agent_balances(balances)

    # Verify result
    assert columns is not None
    if not isinstance(columns, str):
        # Rich is available, check it's Columns
        from rich.columns import Columns

        assert isinstance(columns, Columns)


def test_build_events_panel():
    """Test building a panel for day events."""
    # Create sample day events view
    day_view = DayEventsView(
        day=1,
        phases={
            "A": [
                EventView(
                    kind="PhaseA",
                    title="Day begins",
                    lines=["Starting day 1"],
                    icon="⏰",
                    raw_event={"kind": "PhaseA", "day": 1},
                )
            ],
            "B": [
                EventView(
                    kind="CashTransferred",
                    title="Cash transferred",
                    lines=["bank1 → household1: $100"],
                    icon="💵",
                    raw_event={"kind": "CashTransferred", "day": 1},
                )
            ],
            "C": [],
        },
    )

    # Build panel
    panel = build_events_panel(day_view)

    # Verify result
    assert panel is not None
    if not isinstance(panel, str):
        # Rich is available, check it's a Panel
        from rich.panel import Panel

        assert isinstance(panel, Panel)


def test_convert_raw_event_to_view():
    """Test converting raw event dict to EventView."""
    # Test with a CashTransferred event
    raw_event = {
        "kind": "CashTransferred",
        "frm": "bank1",
        "to": "household1",
        "amount": 1000,
        "day": 1,
    }

    event_view = convert_raw_event_to_view(raw_event)

    assert isinstance(event_view, EventView)
    assert "Cash Transfer" in event_view.title
    assert event_view.lines[0] == "bank1 → household1"
    assert event_view.icon == "💰"
    assert event_view.raw_event == raw_event


def test_convert_unknown_event_to_view():
    """Test converting unknown event type to EventView."""
    raw_event = {"kind": "CustomEvent", "data": "some value"}

    event_view = convert_raw_event_to_view(raw_event)

    assert isinstance(event_view, EventView)
    assert "CustomEvent" in event_view.title
    assert event_view.icon == "❓"
