"""Tests for the analysis/visualization module.

Tests cover:
- Balance sheet display functions (rich and simple formats)
- Event display functions
- Build helper functions (build_t_account_rows, etc.)
"""

from decimal import Decimal

import pytest

from bilancio.analysis.balances import agent_balance
from bilancio.analysis.visualization import (
    # Common utilities
    RICH_AVAILABLE,
    BalanceRow,
    TAccount,
    build_t_account_rows,
    display_agent_balance_from_balance,
    # Balance functions
    display_agent_balance_table,
    display_agent_balance_table_renderable,
    # Event functions
    display_events,
    display_events_for_day,
    display_events_for_day_renderable,
    display_events_renderable,
    display_events_table,
    display_events_table_renderable,
    # Phase functions
    display_events_tables_by_phase_renderables,
    display_multiple_agent_balances,
    display_multiple_agent_balances_renderable,
    parse_day_from_maturity,
)
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.ops.banking import client_payment, deposit_cash

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_system():
    """Create a simple system with bank and household for basic tests.

    Setup:
    - Central Bank (CB01)
    - Bank (BK01) with 5000 in reserves
    - Household (HH01) with 1000 cash, 600 deposited at bank
    """
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="BK01", name="Test Bank", kind="bank")
    household = Household(id="HH01", name="Test Household", kind="household")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(household)

    # Mint reserves to bank
    system.mint_reserves("BK01", 5000)

    # Mint cash to household and deposit some
    system.mint_cash("HH01", 1000)
    deposit_cash(system, "HH01", "BK01", 600)

    return system


@pytest.fixture
def multi_agent_system():
    """Create a system with multiple agents for comparison tests.

    Setup:
    - Central Bank (CB01)
    - Two Banks (BK01, BK02) with reserves
    - Two Households (HH01, HH02) with deposits at different banks
    """
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank1 = Bank(id="BK01", name="Bank One", kind="bank")
    bank2 = Bank(id="BK02", name="Bank Two", kind="bank")
    hh1 = Household(id="HH01", name="Household One", kind="household")
    hh2 = Household(id="HH02", name="Household Two", kind="household")

    system.bootstrap_cb(cb)
    system.add_agent(bank1)
    system.add_agent(bank2)
    system.add_agent(hh1)
    system.add_agent(hh2)

    # Setup reserves
    system.mint_reserves("BK01", 5000)
    system.mint_reserves("BK02", 3000)

    # Setup household deposits
    system.mint_cash("HH01", 2000)
    system.mint_cash("HH02", 1500)
    deposit_cash(system, "HH01", "BK01", 1500)
    deposit_cash(system, "HH02", "BK02", 1000)

    return system


@pytest.fixture
def system_with_events():
    """Create a system that has run a day simulation to generate events.

    This includes payables, payments, and interbank clearing.
    """
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank1 = Bank(id="BK01", name="Bank One", kind="bank")
    bank2 = Bank(id="BK02", name="Bank Two", kind="bank")
    hh1 = Household(id="HH01", name="Alice", kind="household")
    hh2 = Household(id="HH02", name="Bob", kind="household")
    hh3 = Household(id="HH03", name="Charlie", kind="household")

    system.bootstrap_cb(cb)
    system.add_agent(bank1)
    system.add_agent(bank2)
    system.add_agent(hh1)
    system.add_agent(hh2)
    system.add_agent(hh3)

    # Banks get reserves
    system.mint_reserves("BK01", 500)
    system.mint_reserves("BK02", 500)

    # Households get cash and deposit
    system.mint_cash("HH01", 300)
    system.mint_cash("HH02", 200)
    system.mint_cash("HH03", 100)
    deposit_cash(system, "HH01", "BK01", 300)
    deposit_cash(system, "HH02", "BK01", 200)
    deposit_cash(system, "HH03", "BK02", 100)

    # Create a payable due today
    payable_id = system.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="X",
        asset_holder_id="HH02",
        liability_issuer_id="HH01",
        due_day=system.state.day,
    )
    system.add_contract(payable)

    # Create cross-bank payment
    client_payment(system, "HH02", "BK01", "HH03", "BK02", 50)

    # Run a day to generate phase events
    run_day(system)

    return system


# ============================================================================
# Tests for Balance Display Functions
# ============================================================================


class TestDisplayAgentBalanceTable:
    """Tests for display_agent_balance_table function."""

    def test_display_rich_format_runs_without_error(self, simple_system, capsys):
        """Test that rich format display executes without errors."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich library not available")

        display_agent_balance_table(simple_system, "HH01", format="rich")
        # Just verify it runs - output goes to console
        capsys.readouterr()
        # Rich output may or may not be captured depending on console setup

    def test_display_simple_format(self, simple_system, capsys):
        """Test that simple format produces text output."""
        display_agent_balance_table(simple_system, "HH01", format="simple")
        captured = capsys.readouterr()

        # Verify key elements are in output
        assert "HH01" in captured.out or "Test Household" in captured.out
        assert "ASSETS" in captured.out
        assert "LIABILITIES" in captured.out

    def test_display_with_custom_title(self, simple_system, capsys):
        """Test display with custom title."""
        custom_title = "Custom Balance Title"
        display_agent_balance_table(simple_system, "HH01", format="simple", title=custom_title)
        captured = capsys.readouterr()

        assert custom_title in captured.out

    def test_display_bank_balance(self, simple_system, capsys):
        """Test displaying bank balance (has both assets and liabilities)."""
        display_agent_balance_table(simple_system, "BK01", format="simple")
        captured = capsys.readouterr()

        # Bank should have reserves as assets and deposits as liabilities
        assert "ASSETS" in captured.out
        assert "LIABILITIES" in captured.out


class TestDisplayAgentBalanceFromBalance:
    """Tests for display_agent_balance_from_balance function."""

    def test_display_from_balance_object(self, simple_system, capsys):
        """Test displaying from a pre-computed AgentBalance object."""
        balance = agent_balance(simple_system, "HH01")
        display_agent_balance_from_balance(balance, format="simple")
        captured = capsys.readouterr()

        assert "ASSETS" in captured.out
        assert "LIABILITIES" in captured.out

    def test_display_from_balance_with_title(self, simple_system, capsys):
        """Test displaying from balance with custom title."""
        balance = agent_balance(simple_system, "BK01")
        display_agent_balance_from_balance(balance, format="simple", title="Bank Balance Sheet")
        captured = capsys.readouterr()

        assert "Bank Balance Sheet" in captured.out


class TestDisplayMultipleAgentBalances:
    """Tests for display_multiple_agent_balances function."""

    def test_display_multiple_by_id(self, multi_agent_system, capsys):
        """Test displaying multiple agents by ID."""
        display_multiple_agent_balances(multi_agent_system, ["HH01", "HH02"], format="simple")
        captured = capsys.readouterr()

        # Both agents should appear in output
        assert "HH01" in captured.out or "Household One" in captured.out
        assert "HH02" in captured.out or "Household Two" in captured.out

    def test_display_multiple_by_balance_objects(self, multi_agent_system, capsys):
        """Test displaying multiple agents using AgentBalance objects."""
        balances = [
            agent_balance(multi_agent_system, "BK01"),
            agent_balance(multi_agent_system, "BK02"),
        ]
        display_multiple_agent_balances(multi_agent_system, balances, format="simple")
        captured = capsys.readouterr()

        # Both banks should appear
        assert "BK01" in captured.out or "Bank One" in captured.out
        assert "BK02" in captured.out or "Bank Two" in captured.out

    def test_display_mixed_items(self, multi_agent_system, capsys):
        """Test displaying with mix of IDs and balance objects."""
        balance_hh1 = agent_balance(multi_agent_system, "HH01")
        display_multiple_agent_balances(multi_agent_system, [balance_hh1, "HH02"], format="simple")
        captured = capsys.readouterr()

        # Both should appear
        assert "HH01" in captured.out or "Household One" in captured.out


# ============================================================================
# Tests for Balance Renderable Functions
# ============================================================================


class TestBalanceRenderables:
    """Tests for balance renderable functions."""

    def test_balance_table_renderable_rich(self, simple_system):
        """Test that renderable returns appropriate type for rich format."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich library not available")

        result = display_agent_balance_table_renderable(simple_system, "HH01", format="rich")
        # Should return a Rich Table
        assert result is not None
        # Rich Table has specific attributes
        assert hasattr(result, "columns") or isinstance(result, str)

    def test_balance_table_renderable_simple(self, simple_system):
        """Test that renderable returns string for simple format."""
        result = display_agent_balance_table_renderable(simple_system, "HH01", format="simple")
        assert isinstance(result, str)
        assert "ASSETS" in result

    def test_multiple_balances_renderable_simple(self, multi_agent_system):
        """Test multiple balances renderable in simple format."""
        result = display_multiple_agent_balances_renderable(
            multi_agent_system, ["HH01", "HH02"], format="simple"
        )
        assert isinstance(result, str)
        assert "HH01" in result or "Household One" in result


# ============================================================================
# Tests for Build Functions
# ============================================================================


class TestBuildTAccountRows:
    """Tests for build_t_account_rows function."""

    def test_returns_taccount_structure(self, simple_system):
        """Test that build_t_account_rows returns proper TAccount structure."""
        taccount = build_t_account_rows(simple_system, "HH01")

        assert isinstance(taccount, TAccount)
        assert isinstance(taccount.assets, list)
        assert isinstance(taccount.liabilities, list)

    def test_asset_rows_have_expected_fields(self, simple_system):
        """Test that asset rows contain expected BalanceRow fields."""
        taccount = build_t_account_rows(simple_system, "HH01")

        # Household should have assets (cash, deposit)
        assert len(taccount.assets) > 0

        for row in taccount.assets:
            assert isinstance(row, BalanceRow)
            assert row.name is not None
            # value_minor should be set for financial assets
            assert row.value_minor is not None or row.quantity is not None

    def test_liability_rows_for_bank(self, simple_system):
        """Test that bank liability rows are built correctly."""
        taccount = build_t_account_rows(simple_system, "BK01")

        # Bank should have liabilities (deposits from household)
        assert len(taccount.liabilities) > 0

        deposit_found = False
        for row in taccount.liabilities:
            if "bank_deposit" in row.name or "deposit" in row.name.lower():
                deposit_found = True
                assert row.value_minor == 600  # HH01 deposited 600

        assert deposit_found, "Bank should have deposit liability"

    def test_counterparty_info_present(self, simple_system):
        """Test that counterparty information is included in rows."""
        taccount = build_t_account_rows(simple_system, "HH01")

        # Find the bank deposit asset
        for row in taccount.assets:
            if "bank_deposit" in row.name:
                # Counterparty should be the bank
                assert row.counterparty_name is not None
                assert "BK01" in row.counterparty_name or "Test Bank" in row.counterparty_name

    def test_delivery_obligations_as_receivables(self, simple_system):
        """Test that delivery obligations appear as receivables on asset side."""
        # Add delivery obligation
        simple_system.create_delivery_obligation(
            "BK01", "HH01", sku="GOODS", quantity=10, unit_price=Decimal("5"), due_day=1
        )

        taccount = build_t_account_rows(simple_system, "HH01")

        # HH01 should now have a receivable (right to receive goods)
        receivable_found = any("receivable" in row.name.lower() for row in taccount.assets)
        assert receivable_found, "Should have receivable for delivery obligation"

    def test_delivery_obligations_as_liabilities(self, simple_system):
        """Test that delivery obligations appear on liability side for issuer."""
        simple_system.create_delivery_obligation(
            "HH01", "BK01", sku="WIDGETS", quantity=5, unit_price=Decimal("10"), due_day=2
        )

        taccount = build_t_account_rows(simple_system, "HH01")

        # HH01 should have an obligation liability
        obligation_found = any("obligation" in row.name.lower() for row in taccount.liabilities)
        assert obligation_found, "Should have obligation liability"


class TestParseDayFromMaturity:
    """Tests for parse_day_from_maturity helper function."""

    def test_parses_valid_day_string(self):
        """Test parsing valid 'Day N' format."""
        assert parse_day_from_maturity("Day 1") == 1
        assert parse_day_from_maturity("Day 10") == 10
        assert parse_day_from_maturity("Day 100") == 100

    def test_returns_infinity_for_invalid(self):
        """Test that invalid formats return infinity."""
        import math

        assert parse_day_from_maturity(None) == math.inf
        assert parse_day_from_maturity("") > 10**9
        assert parse_day_from_maturity("on-demand") > 10**9
        assert parse_day_from_maturity("Day X") > 10**9
        assert parse_day_from_maturity("Tomorrow") > 10**9


# ============================================================================
# Tests for Event Display Functions
# ============================================================================


class TestDisplayEvents:
    """Tests for display_events function."""

    def test_display_empty_events(self, capsys):
        """Test displaying empty event list."""
        display_events([], format="detailed")
        captured = capsys.readouterr()

        assert "No events" in captured.out

    def test_display_detailed_format(self, system_with_events, capsys):
        """Test detailed format includes phase information."""
        events = system_with_events.state.events
        display_events(events, format="detailed")
        captured = capsys.readouterr()

        # Should include phase labels
        assert "Phase" in captured.out or "Day" in captured.out

    def test_display_summary_format(self, system_with_events, capsys):
        """Test summary format shows condensed events."""
        events = system_with_events.state.events
        display_events(events, format="summary")
        # Should run without error - summary format may produce less output


class TestDisplayEventsTable:
    """Tests for display_events_table function."""

    def test_display_events_table_has_columns(self, system_with_events, capsys):
        """Test that events table includes expected columns."""
        events = system_with_events.state.events
        display_events_table(events)
        # Should run without error


class TestDisplayEventsTableRenderable:
    """Tests for display_events_table_renderable function."""

    def test_renderable_empty_events(self):
        """Test renderable with empty events returns appropriate message."""
        result = display_events_table_renderable([])

        if RICH_AVAILABLE:
            # Returns Rich Text or similar
            assert result is not None
        else:
            assert "No events" in str(result)

    def test_renderable_with_events(self, system_with_events):
        """Test renderable with actual events returns table structure."""
        events = system_with_events.state.events
        result = display_events_table_renderable(events)

        assert result is not None
        if RICH_AVAILABLE:
            # Should be a Rich Table with columns
            assert hasattr(result, "columns") or isinstance(result, str)
        else:
            assert isinstance(result, str)


class TestDisplayEventsForDay:
    """Tests for display_events_for_day function."""

    def test_display_events_for_specific_day(self, system_with_events, capsys):
        """Test displaying events for a specific day."""
        # Day 0 should have events (run_day increments day)
        display_events_for_day(system_with_events, 0)
        captured = capsys.readouterr()

        # Should show day events or "no events" message
        assert len(captured.out) > 0

    def test_display_events_for_future_day(self, system_with_events, capsys):
        """Test displaying events for day with no events."""
        display_events_for_day(system_with_events, 999)
        captured = capsys.readouterr()

        assert "No events" in captured.out


class TestDisplayEventsRenderable:
    """Tests for display_events_renderable function."""

    def test_renderable_returns_list(self, system_with_events):
        """Test that renderable returns list of renderables."""
        events = system_with_events.state.events
        result = display_events_renderable(events, format="detailed")

        assert isinstance(result, list)

    def test_renderable_empty_events(self):
        """Test renderable with empty events."""
        result = display_events_renderable([])

        assert isinstance(result, list)
        assert len(result) > 0  # Should have "no events" message


class TestDisplayEventsForDayRenderable:
    """Tests for display_events_for_day_renderable function."""

    def test_renderable_for_day_with_events(self, system_with_events):
        """Test getting renderables for a day with events."""
        result = display_events_for_day_renderable(system_with_events, 0)

        assert isinstance(result, list)

    def test_renderable_for_empty_day(self, system_with_events):
        """Test getting renderables for a day without events."""
        result = display_events_for_day_renderable(system_with_events, 999)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have "no events" message


# ============================================================================
# Tests for Phase Display Functions
# ============================================================================


class TestDisplayEventsTablesByPhase:
    """Tests for display_events_tables_by_phase_renderables function."""

    def test_phase_tables_returns_list(self, system_with_events):
        """Test that phase tables returns list of renderables."""
        # Get events for day 0
        day_events = [e for e in system_with_events.state.events if e.get("day") == 0]
        result = display_events_tables_by_phase_renderables(day_events, day=0)

        assert isinstance(result, list)

    def test_phase_tables_include_phases_b_and_c(self, system_with_events):
        """Test that result includes Phase B and C tables."""
        day_events = [e for e in system_with_events.state.events if e.get("day") == 0]
        result = display_events_tables_by_phase_renderables(day_events, day=0)

        # Should have at least 2 tables (Phase B and C, possibly A)
        assert len(result) >= 2

    def test_phase_tables_with_empty_events(self):
        """Test phase tables with empty events."""
        result = display_events_tables_by_phase_renderables([])

        assert isinstance(result, list)
        # Should still return phase table structures even if empty


# ============================================================================
# Tests for Common Types and Constants
# ============================================================================


class TestCommonTypes:
    """Tests for common types exported from the module."""

    def test_balance_row_dataclass(self):
        """Test BalanceRow dataclass creation."""
        row = BalanceRow(
            name="cash",
            quantity=None,
            value_minor=1000,
            counterparty_name="CB01",
            maturity="on-demand",
        )

        assert row.name == "cash"
        assert row.value_minor == 1000
        assert row.counterparty_name == "CB01"

    def test_balance_row_with_quantity(self):
        """Test BalanceRow with quantity for non-financial items."""
        row = BalanceRow(
            name="WIDGETS receivable",
            quantity=50,
            value_minor=5000,
            counterparty_name="Firm A",
            maturity="Day 5",
        )

        assert row.quantity == 50
        assert row.maturity == "Day 5"

    def test_taccount_dataclass(self):
        """Test TAccount dataclass creation."""
        assets = [BalanceRow("cash", None, 1000, "CB", "on-demand")]
        liabilities = [BalanceRow("deposit", None, 500, "HH01", "on-demand")]

        taccount = TAccount(assets=assets, liabilities=liabilities)

        assert len(taccount.assets) == 1
        assert len(taccount.liabilities) == 1
        assert taccount.assets[0].name == "cash"

    def test_rich_available_constant(self):
        """Test that RICH_AVAILABLE is a boolean."""
        assert isinstance(RICH_AVAILABLE, bool)


# ============================================================================
# Integration Tests
# ============================================================================


class TestVisualizationIntegration:
    """Integration tests combining multiple visualization functions."""

    def test_full_visualization_workflow(self, system_with_events, capsys):
        """Test a complete visualization workflow."""
        # Display balances for all households
        display_multiple_agent_balances(
            system_with_events, ["HH01", "HH02", "HH03"], format="simple"
        )

        # Display events for day 0
        display_events_for_day(system_with_events, 0)

        # Build T-accounts for bank
        taccount = build_t_account_rows(system_with_events, "BK01")

        captured = capsys.readouterr()

        # Verify all components ran
        assert "HH01" in captured.out or "Alice" in captured.out
        assert isinstance(taccount, TAccount)

    def test_display_before_and_after_adding_obligation(self, simple_system, capsys):
        """Test displaying state before and after adding delivery obligation."""
        # Display initial state
        display_agent_balance_table(simple_system, "HH01", format="simple")
        initial_output = capsys.readouterr().out

        # Add a delivery obligation (doesn't require complex settlement)
        simple_system.create_delivery_obligation(
            from_agent="HH01",
            to_agent="BK01",
            sku="WIDGETS",
            quantity=10,
            unit_price=Decimal("5"),
            due_day=1,
        )

        # Display after state
        display_agent_balance_table(simple_system, "HH01", format="simple")
        after_output = capsys.readouterr().out

        # Verify both outputs were generated
        assert len(initial_output) > 0
        assert len(after_output) > 0
