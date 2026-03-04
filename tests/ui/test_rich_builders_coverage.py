"""Additional coverage tests for rich_builders.py.

Targets uncovered branches: build_events_list, build_day_summary,
convert_raw_events_to_day_view, _format_currency, and edge cases.
"""

from __future__ import annotations

from decimal import Decimal

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bilancio.ui.render.models import (
    AgentBalanceView,
    BalanceItemView,
    DayEventsView,
    DaySummaryView,
    EventView,
)
from bilancio.ui.render.rich_builders import (
    _format_currency,
    build_agent_balance_table,
    build_day_summary,
    build_events_list,
    build_events_panel,
    build_multiple_agent_balances,
    convert_raw_event_to_view,
    convert_raw_events_to_day_view,
)


# ============================================================================
# _format_currency
# ============================================================================


class TestFormatCurrency:
    """Cover _format_currency branches."""

    def test_basic_format(self):
        assert _format_currency(1234) == "1,234"

    def test_show_sign_positive(self):
        assert _format_currency(500, show_sign=True) == "+500"

    def test_show_sign_zero(self):
        assert _format_currency(0, show_sign=True) == "0"

    def test_show_sign_negative(self):
        assert _format_currency(-100, show_sign=True) == "-100"

    def test_decimal_value(self):
        result = _format_currency(Decimal("9999"))
        assert result == "9,999"


# ============================================================================
# build_agent_balance_table edge cases
# ============================================================================


class TestBuildAgentBalanceTableEdgeCases:
    """Cover edge cases in build_agent_balance_table."""

    def test_agent_name_same_as_id(self):
        """When name == id, title should not repeat the name."""
        view = AgentBalanceView(
            agent_id="B1",
            agent_name="B1",
            agent_kind="bank",
            items=[
                BalanceItemView(
                    category="Assets",
                    instrument="Cash",
                    amount=100,
                    value=Decimal("100"),
                )
            ],
        )
        table = build_agent_balance_table(view)
        assert isinstance(table, Table)
        # Title should be "B1 (bank)" not "B1 [B1] (bank)"
        assert table.title == "B1 (bank)"

    def test_no_assets_only_liabilities(self):
        """Table renders even with zero assets."""
        view = AgentBalanceView(
            agent_id="H1",
            agent_name="Alice",
            agent_kind="household",
            items=[
                BalanceItemView(
                    category="Liabilities",
                    instrument="Payable",
                    amount=500,
                    value=Decimal("500"),
                ),
            ],
        )
        table = build_agent_balance_table(view)
        assert isinstance(table, Table)

    def test_negative_net_worth(self):
        """Negative net worth should use red style."""
        view = AgentBalanceView(
            agent_id="F1",
            agent_name="Firm1",
            agent_kind="firm",
            items=[
                BalanceItemView(
                    category="Assets",
                    instrument="Cash",
                    amount=100,
                    value=Decimal("100"),
                ),
                BalanceItemView(
                    category="Liabilities",
                    instrument="Debt",
                    amount=500,
                    value=Decimal("500"),
                ),
            ],
        )
        table = build_agent_balance_table(view)
        assert isinstance(table, Table)

    def test_empty_items(self):
        """Table renders with no items at all."""
        view = AgentBalanceView(
            agent_id="X1", agent_name="Empty", agent_kind="firm", items=[]
        )
        table = build_agent_balance_table(view)
        assert isinstance(table, Table)


# ============================================================================
# build_events_panel edge cases
# ============================================================================


class TestBuildEventsPanelEdgeCases:
    """Cover edge cases in build_events_panel."""

    def test_empty_phases(self):
        view = DayEventsView(day=5, phases={})
        panel = build_events_panel(view)
        assert isinstance(panel, Panel)

    def test_phase_with_empty_events(self):
        """Phase exists but has no events; should be skipped."""
        view = DayEventsView(
            day=2,
            phases={
                "A": [],
                "B": [
                    EventView(
                        kind="Test",
                        title="Something",
                        lines=["detail"],
                        icon="T",
                        raw_event={},
                    )
                ],
            },
        )
        panel = build_events_panel(view)
        assert isinstance(panel, Panel)


# ============================================================================
# build_events_list
# ============================================================================


class TestBuildEventsList:
    """Cover build_events_list."""

    def test_empty_phases_returns_no_events_text(self):
        view = DayEventsView(day=1, phases={})
        renderables = build_events_list(view)
        assert len(renderables) == 1
        assert "No events" in str(renderables[0])

    def test_with_events(self):
        view = DayEventsView(
            day=1,
            phases={
                "A": [
                    EventView(
                        kind="CashMinted",
                        title="Cash Minted",
                        lines=["To: CB"],
                        icon="M",
                        raw_event={},
                    )
                ],
            },
        )
        renderables = build_events_list(view)
        assert len(renderables) > 0
        # Should have phase header, event text, detail, and spacing
        texts = [str(r) for r in renderables]
        assert any("A" in t for t in texts)
        assert any("Cash Minted" in t for t in texts)

    def test_skips_empty_phases(self):
        view = DayEventsView(
            day=1,
            phases={
                "A": [],
                "B": [
                    EventView(
                        kind="X",
                        title="Event",
                        lines=[],
                        icon="!",
                        raw_event={},
                    )
                ],
                "C": [],
            },
        )
        renderables = build_events_list(view)
        # Should only have B phase header + event + spacing
        texts = [str(r) for r in renderables]
        assert any("B" in t for t in texts)


# ============================================================================
# build_day_summary
# ============================================================================


class TestBuildDaySummary:
    """Cover build_day_summary."""

    def test_with_events_and_balances(self):
        events_view = DayEventsView(
            day=3,
            phases={
                "B": [
                    EventView(
                        kind="Test",
                        title="Event",
                        lines=[],
                        icon="!",
                        raw_event={},
                    )
                ]
            },
        )
        balances = [
            AgentBalanceView(
                agent_id="B1",
                agent_name="Bank",
                agent_kind="bank",
                items=[
                    BalanceItemView(
                        category="Assets",
                        instrument="Cash",
                        amount=1000,
                        value=1000,
                    )
                ],
            )
        ]
        view = DaySummaryView(
            day=3, events_view=events_view, agent_balances=balances
        )
        renderables = build_day_summary(view)
        assert len(renderables) >= 2
        # First should be day header panel
        assert isinstance(renderables[0], Panel)

    def test_no_events(self):
        events_view = DayEventsView(day=1, phases={})
        view = DaySummaryView(day=1, events_view=events_view, agent_balances=[])
        renderables = build_day_summary(view)
        assert len(renderables) >= 2
        # Should include "No events" text
        texts = [str(r) for r in renderables]
        assert any("No events" in t for t in texts)


# ============================================================================
# convert_raw_events_to_day_view
# ============================================================================


class TestConvertRawEventsToDayView:
    """Cover convert_raw_events_to_day_view."""

    def test_basic_conversion(self):
        raw = [
            {"kind": "CashMinted", "to": "CB", "amount": 1000, "phase": "setup"},
            {
                "kind": "CashTransferred",
                "frm": "CB",
                "to": "H1",
                "amount": 500,
                "phase": "B",
            },
        ]
        view = convert_raw_events_to_day_view(raw, day=0)
        assert isinstance(view, DayEventsView)
        assert view.day == 0
        assert "setup" in view.phases
        assert "B" in view.phases
        assert len(view.phases["setup"]) == 1
        assert len(view.phases["B"]) == 1

    def test_empty_events(self):
        view = convert_raw_events_to_day_view([], day=5)
        assert view.day == 5
        assert len(view.phases) == 0

    def test_unknown_phase(self):
        raw = [{"kind": "Custom", "phase": "unknown", "data": "val"}]
        view = convert_raw_events_to_day_view(raw, day=1)
        assert "unknown" in view.phases
        assert len(view.phases["unknown"]) == 1
        assert view.phases["unknown"][0].kind == "Custom"
