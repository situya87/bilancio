"""Tests for bilancio.analysis.visualization.balances module.

Covers uncovered paths:
- display_agent_t_account / display_agent_t_account_renderable (simple fallback + rich)
- _build_simple_agent_balance_string (title generation paths)
- _build_simple_multiple_agent_balances_string
- _display_simple_multiple_agent_balances (non-financial assets/liabilities, truncation)
- _display_rich_multiple_agent_balances (non-financial paths)
- _create_rich_agent_balance_table / _create_compact_rich_balance_table
- _fmt_qty, _fmt_val, _cells helper functions
- build_t_account_rows edge cases (missing contracts, aliases)
- display_agent_balance_table_renderable (simple path)
- display_multiple_agent_balances_renderable (simple + rich paths)
- Non-financial asset/liability display paths
- Net financial negative styling
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from bilancio.analysis.balances import AgentBalance, agent_balance
from bilancio.analysis.visualization.balances import (
    _build_simple_agent_balance_string,
    _build_simple_multiple_agent_balances_string,
    _cells,
    _fmt_qty,
    _fmt_val,
    build_t_account_rows,
    display_agent_balance_table,
    display_agent_balance_table_renderable,
    display_agent_t_account,
    display_agent_t_account_renderable,
    display_multiple_agent_balances,
    display_multiple_agent_balances_renderable,
)
from bilancio.analysis.visualization.common import RICH_AVAILABLE, BalanceRow, TAccount
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_system():
    """Create a simple system with bank and household."""
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="BK01", name="Test Bank", kind="bank")
    household = Household(id="HH01", name="Test Household", kind="household")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(household)

    system.mint_reserves("BK01", 5000)
    system.mint_cash("HH01", 1000)
    deposit_cash(system, "HH01", "BK01", 600)

    return system


@pytest.fixture
def multi_agent_system():
    """Create a system with multiple agents."""
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

    system.mint_reserves("BK01", 5000)
    system.mint_reserves("BK02", 3000)

    system.mint_cash("HH01", 2000)
    system.mint_cash("HH02", 1500)
    deposit_cash(system, "HH01", "BK01", 1500)
    deposit_cash(system, "HH02", "BK02", 1000)

    return system


@pytest.fixture
def system_with_delivery_obligations():
    """Create a system with delivery obligations for non-financial paths."""
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    hh1 = Household(id="HH01", name="Alice", kind="household")
    hh2 = Household(id="HH02", name="Bob", kind="household")

    system.bootstrap_cb(cb)
    system.add_agent(hh1)
    system.add_agent(hh2)

    system.mint_cash("HH01", 1000)
    system.mint_cash("HH02", 500)

    # Create stock for HH01
    system.create_stock("HH01", "INVENTORY", 50, Decimal("100"))

    # Create delivery obligation: HH01 owes HH02 goods
    system.create_delivery_obligation(
        "HH01", "HH02", "WIDGETS", 25, Decimal("10"), due_day=3
    )

    return system


def _make_balance(
    agent_id="HH01",
    assets_by_kind=None,
    liabilities_by_kind=None,
    total_financial_assets=0,
    total_financial_liabilities=0,
    net_financial=0,
    nonfinancial_assets_by_kind=None,
    nonfinancial_liabilities_by_kind=None,
    total_nonfinancial_value=Decimal(0),
    total_nonfinancial_liability_value=Decimal(0),
    inventory_by_sku=None,
    total_inventory_value=Decimal(0),
):
    """Helper to create an AgentBalance with defaults."""
    return AgentBalance(
        agent_id=agent_id,
        assets_by_kind=assets_by_kind or {},
        liabilities_by_kind=liabilities_by_kind or {},
        total_financial_assets=total_financial_assets,
        total_financial_liabilities=total_financial_liabilities,
        net_financial=net_financial,
        nonfinancial_assets_by_kind=nonfinancial_assets_by_kind or {},
        nonfinancial_liabilities_by_kind=nonfinancial_liabilities_by_kind or {},
        total_nonfinancial_value=total_nonfinancial_value,
        total_nonfinancial_liability_value=total_nonfinancial_liability_value,
        inventory_by_sku=inventory_by_sku or {},
        total_inventory_value=total_inventory_value,
    )


# ============================================================================
# _fmt_qty, _fmt_val, _cells helpers
# ============================================================================


class TestFormatHelpers:
    """Test the _fmt_qty, _fmt_val, _cells helper functions."""

    def test_fmt_qty_with_value(self):
        row = BalanceRow("test", quantity=42, value_minor=100, counterparty_name=None, maturity=None)
        assert _fmt_qty(row) == "42"

    def test_fmt_qty_none_quantity(self):
        row = BalanceRow("test", quantity=None, value_minor=100, counterparty_name=None, maturity=None)
        assert _fmt_qty(row) == "\u2014"  # em dash

    def test_fmt_qty_none_row(self):
        assert _fmt_qty(None) == "\u2014"

    def test_fmt_val_with_value(self):
        row = BalanceRow("test", quantity=None, value_minor=1000, counterparty_name=None, maturity=None)
        assert _fmt_val(row) == "1,000"

    def test_fmt_val_none_value(self):
        row = BalanceRow("test", quantity=None, value_minor=None, counterparty_name=None, maturity=None)
        assert _fmt_val(row) == "\u2014"

    def test_fmt_val_none_row(self):
        assert _fmt_val(None) == "\u2014"

    def test_cells_with_row(self):
        row = BalanceRow("cash", quantity=None, value_minor=500, counterparty_name="CB01", maturity="on-demand")
        result = _cells(row)
        assert result == ("cash", "\u2014", "500", "CB01", "on-demand")

    def test_cells_none_row(self):
        result = _cells(None)
        assert result == ("", "", "", "", "")

    def test_cells_none_counterparty_and_maturity(self):
        row = BalanceRow("item", quantity=10, value_minor=100, counterparty_name=None, maturity=None)
        result = _cells(row)
        assert result[3] == "\u2014"
        assert result[4] == "\u2014"

    def test_fmt_qty_large_number(self):
        row = BalanceRow("test", quantity=1000000, value_minor=100, counterparty_name=None, maturity=None)
        assert _fmt_qty(row) == "1,000,000"


# ============================================================================
# display_agent_t_account (simple path)
# ============================================================================


class TestDisplayAgentTAccount:
    """Test display_agent_t_account."""

    def test_simple_format(self, simple_system, capsys):
        display_agent_t_account(simple_system, "HH01", format="simple")
        captured = capsys.readouterr()
        assert "Assets:" in captured.out
        assert "Liabilities:" in captured.out

    def test_rich_format(self, simple_system, capsys):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_agent_t_account(simple_system, "HH01", format="rich")
        # Should run without error; output goes to console
        capsys.readouterr()

    def test_fallback_when_rich_unavailable(self, simple_system, capsys):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            display_agent_t_account(simple_system, "HH01", format="rich")
        captured = capsys.readouterr()
        assert "Assets:" in captured.out


# ============================================================================
# display_agent_t_account_renderable
# ============================================================================


class TestDisplayAgentTAccountRenderable:
    """Test display_agent_t_account_renderable."""

    def test_non_rich_returns_string(self, simple_system):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            result = display_agent_t_account_renderable(simple_system, "HH01")
        assert isinstance(result, str)
        assert "Assets:" in result

    def test_rich_returns_table(self, simple_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_t_account_renderable(simple_system, "HH01")
        assert hasattr(result, "columns")  # Rich Table has .columns

    def test_bank_t_account_has_liabilities(self, simple_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_t_account_renderable(simple_system, "BK01")
        assert hasattr(result, "columns")  # Rich Table has columns

    def test_agent_name_in_title(self, simple_system):
        """When agent has a name different from ID, title shows both."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_t_account_renderable(simple_system, "HH01")
        title = str(result.title)
        assert "Test Household" in title
        assert "HH01" in title


# ============================================================================
# build_t_account_rows edge cases
# ============================================================================


class TestBuildTAccountRowsEdgeCases:
    """Test build_t_account_rows with edge cases."""

    def test_with_delivery_obligations(self, system_with_delivery_obligations):
        """Delivery obligations show as receivables/obligations."""
        # HH02 has a receivable (delivery obligation where HH02 is creditor)
        acct = build_t_account_rows(system_with_delivery_obligations, "HH02")
        receivable_names = [r.name for r in acct.assets]
        assert any("receivable" in n for n in receivable_names)

    def test_with_delivery_obligation_liabilities(self, system_with_delivery_obligations):
        """HH01 has an obligation liability."""
        acct = build_t_account_rows(system_with_delivery_obligations, "HH01")
        liability_names = [r.name for r in acct.liabilities]
        assert any("obligation" in n for n in liability_names)

    def test_with_stocks(self, system_with_delivery_obligations):
        """HH01 has inventory stocks as assets."""
        acct = build_t_account_rows(system_with_delivery_obligations, "HH01")
        asset_names = [r.name for r in acct.assets]
        assert any("INVENTORY" in n for n in asset_names)

    def test_missing_contract(self, simple_system):
        """When a contract in asset_ids doesn't exist, it is skipped."""
        agent = simple_system.state.agents["HH01"]
        agent.asset_ids.add("NONEXISTENT_CONTRACT")
        acct = build_t_account_rows(simple_system, "HH01")
        # Should succeed without error, nonexistent skipped
        assert isinstance(acct, TAccount)

    def test_missing_contract_in_liabilities(self, simple_system):
        """When a contract in liability_ids doesn't exist, it is skipped."""
        agent = simple_system.state.agents["BK01"]
        agent.liability_ids.add("NONEXISTENT_CONTRACT")
        acct = build_t_account_rows(simple_system, "BK01")
        assert isinstance(acct, TAccount)

    def test_sort_order_assets(self, system_with_delivery_obligations):
        """Assets are sorted: inventory first, then receivables, then financial."""
        acct = build_t_account_rows(system_with_delivery_obligations, "HH01")
        # First item should be stock (INVENTORY), not a financial item
        if acct.assets:
            assert acct.assets[0].name == "INVENTORY" or "receivable" in acct.assets[0].name or acct.assets[0].counterparty_name == "\u2014"


# ============================================================================
# _build_simple_agent_balance_string
# ============================================================================


class TestBuildSimpleAgentBalanceString:
    """Test _build_simple_agent_balance_string title generation."""

    def test_with_system_and_agent_id(self, simple_system):
        balance = agent_balance(simple_system, "HH01")
        result = _build_simple_agent_balance_string(balance, simple_system, "HH01")
        assert "Test Household" in result
        assert "HH01" in result

    def test_with_agent_id_matching_name(self, simple_system):
        """When agent name equals ID, only ID is shown."""
        # Temporarily set agent name to match ID
        agent = simple_system.state.agents["HH01"]
        original_name = agent.name
        agent.name = "HH01"
        balance = agent_balance(simple_system, "HH01")
        result = _build_simple_agent_balance_string(balance, simple_system, "HH01")
        assert "HH01" in result
        agent.name = original_name

    def test_without_system(self):
        balance = _make_balance(agent_id="TEST01")
        result = _build_simple_agent_balance_string(balance, system=None, agent_id=None)
        assert "Agent TEST01" in result

    def test_with_custom_title(self, simple_system):
        balance = agent_balance(simple_system, "HH01")
        result = _build_simple_agent_balance_string(balance, title="Custom Title")
        assert "Custom Title" in result

    def test_content_includes_totals(self, simple_system):
        balance = agent_balance(simple_system, "HH01")
        result = _build_simple_agent_balance_string(balance, simple_system, "HH01")
        assert "ASSETS" in result
        assert "LIABILITIES" in result
        assert "NET FINANCIAL" in result


# ============================================================================
# _build_simple_multiple_agent_balances_string
# ============================================================================


class TestBuildSimpleMultipleAgentBalancesString:
    """Test _build_simple_multiple_agent_balances_string."""

    def test_with_system(self, multi_agent_system):
        balances = [
            agent_balance(multi_agent_system, "HH01"),
            agent_balance(multi_agent_system, "HH02"),
        ]
        result = _build_simple_multiple_agent_balances_string(balances, multi_agent_system)
        assert "Household One" in result or "HH01" in result
        assert "Household Two" in result or "HH02" in result

    def test_without_system(self):
        balances = [_make_balance("A1", net_financial=100), _make_balance("A2", net_financial=-50)]
        result = _build_simple_multiple_agent_balances_string(balances, system=None)
        assert "A1" in result
        assert "A2" in result
        assert "Net Financial" in result


# ============================================================================
# display_agent_balance_table_renderable
# ============================================================================


class TestDisplayAgentBalanceTableRenderable:
    """Test display_agent_balance_table_renderable."""

    def test_simple_format(self, simple_system):
        result = display_agent_balance_table_renderable(simple_system, "HH01", format="simple")
        assert isinstance(result, str)
        assert "ASSETS" in result

    def test_simple_with_custom_title(self, simple_system):
        result = display_agent_balance_table_renderable(
            simple_system, "HH01", format="simple", title="My Title"
        )
        assert isinstance(result, str)
        assert "My Title" in result

    def test_rich_format(self, simple_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_balance_table_renderable(simple_system, "HH01", format="rich")
        assert hasattr(result, "columns")  # Rich Table

    def test_rich_agent_name_in_title(self, simple_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_balance_table_renderable(simple_system, "HH01", format="rich")
        assert "Test Household" in str(result.title)

    def test_rich_agent_id_equals_name(self, simple_system):
        """When agent name equals ID, only ID is in title."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        agent = simple_system.state.agents["HH01"]
        original = agent.name
        agent.name = "HH01"
        result = display_agent_balance_table_renderable(simple_system, "HH01", format="rich")
        assert "HH01" in str(result.title)
        agent.name = original

    def test_rich_with_custom_title(self, simple_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_balance_table_renderable(
            simple_system, "HH01", format="rich", title="Custom"
        )
        assert "Custom" in str(result.title)

    def test_fallback_to_simple_when_no_rich(self, simple_system):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            result = display_agent_balance_table_renderable(simple_system, "HH01", format="rich")
        assert isinstance(result, str)


# ============================================================================
# display_multiple_agent_balances_renderable
# ============================================================================


class TestDisplayMultipleAgentBalancesRenderable:
    """Test display_multiple_agent_balances_renderable."""

    def test_simple_format(self, multi_agent_system):
        result = display_multiple_agent_balances_renderable(
            multi_agent_system, ["HH01", "HH02"], format="simple"
        )
        assert isinstance(result, str)

    def test_rich_format(self, multi_agent_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_multiple_agent_balances_renderable(
            multi_agent_system, ["HH01", "HH02"], format="rich"
        )
        # Should be a Columns renderable
        assert result is not None

    def test_rich_with_balance_objects(self, multi_agent_system):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        balances = [
            agent_balance(multi_agent_system, "HH01"),
            agent_balance(multi_agent_system, "HH02"),
        ]
        result = display_multiple_agent_balances_renderable(
            multi_agent_system, balances, format="rich"
        )
        assert result is not None

    def test_rich_agent_name_equals_id(self, multi_agent_system):
        """When agent name equals ID, title just uses ID."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        agent = multi_agent_system.state.agents["HH01"]
        original = agent.name
        agent.name = "HH01"
        result = display_multiple_agent_balances_renderable(
            multi_agent_system, ["HH01"], format="rich"
        )
        assert result is not None
        agent.name = original

    def test_fallback_to_simple_when_no_rich(self, multi_agent_system):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            result = display_multiple_agent_balances_renderable(
                multi_agent_system, ["HH01", "HH02"], format="rich"
            )
        assert isinstance(result, str)


# ============================================================================
# display_agent_balance_table with Rich format
# ============================================================================


class TestDisplayAgentBalanceTableRich:
    """Test display_agent_balance_table in rich format."""

    def test_rich_format_runs(self, simple_system, capsys):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_agent_balance_table(simple_system, "HH01", format="rich")
        capsys.readouterr()

    def test_fallback_when_no_rich(self, simple_system, capsys):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            display_agent_balance_table(simple_system, "HH01", format="rich")
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_agent_id_as_name(self, simple_system, capsys):
        """Agent with name matching ID shows just ID in title."""
        agent = simple_system.state.agents["HH01"]
        original = agent.name
        agent.name = "HH01"
        display_agent_balance_table(simple_system, "HH01", format="simple")
        captured = capsys.readouterr()
        assert "HH01" in captured.out
        agent.name = original


# ============================================================================
# display_multiple_agent_balances (rich and simple)
# ============================================================================


class TestDisplayMultipleAgentBalancesDisplay:
    """Test display_multiple_agent_balances printing."""

    def test_rich_format(self, multi_agent_system, capsys):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_multiple_agent_balances(
            multi_agent_system, ["HH01", "HH02"], format="rich"
        )
        capsys.readouterr()

    def test_simple_format(self, multi_agent_system, capsys):
        display_multiple_agent_balances(
            multi_agent_system, ["HH01", "HH02"], format="simple"
        )
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out or "HH01" in captured.out

    def test_fallback_when_no_rich(self, multi_agent_system, capsys):
        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            display_multiple_agent_balances(
                multi_agent_system, ["HH01", "HH02"], format="rich"
            )
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ============================================================================
# Non-financial display paths (delivery obligations)
# ============================================================================


class TestNonFinancialDisplayPaths:
    """Test paths for displaying non-financial assets and liabilities."""

    def test_simple_with_delivery_obligations(self, system_with_delivery_obligations, capsys):
        """Simple format shows delivery obligation details."""
        display_agent_balance_table(system_with_delivery_obligations, "HH01", format="simple")
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out

    def test_rich_with_delivery_obligations(self, system_with_delivery_obligations, capsys):
        """Rich format shows delivery obligation details."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_agent_balance_table(system_with_delivery_obligations, "HH01", format="rich")
        captured = capsys.readouterr()
        assert captured.out

    def test_simple_multiple_with_delivery_obligations(
        self, system_with_delivery_obligations, capsys
    ):
        """Simple format for multiple agents with delivery obligations."""
        display_multiple_agent_balances(
            system_with_delivery_obligations, ["HH01", "HH02"], format="simple"
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_rich_multiple_with_delivery_obligations(
        self, system_with_delivery_obligations, capsys
    ):
        """Rich format for multiple agents with delivery obligations."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_multiple_agent_balances(
            system_with_delivery_obligations, ["HH01", "HH02"], format="rich"
        )
        captured = capsys.readouterr()
        assert captured.out

    def test_renderable_with_delivery_obligations(self, system_with_delivery_obligations):
        """Renderable functions handle delivery obligations."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_balance_table_renderable(
            system_with_delivery_obligations, "HH01", format="rich"
        )
        assert hasattr(result, "columns")

    def test_t_account_with_delivery_obligations(self, system_with_delivery_obligations):
        """T-account renderable handles delivery obligations."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        result = display_agent_t_account_renderable(system_with_delivery_obligations, "HH01")
        assert hasattr(result, "columns")


# ============================================================================
# Negative net financial styling
# ============================================================================


class TestNegativeNetFinancial:
    """Test that negative net financial is styled correctly."""

    def test_simple_negative_net(self, capsys):
        """Negative net financial displays in simple format."""
        balance = _make_balance(
            agent_id="TEST01",
            total_financial_assets=100,
            total_financial_liabilities=500,
            net_financial=-400,
        )
        result = _build_simple_agent_balance_string(balance)
        assert "-400" in result

    def test_rich_negative_net(self, simple_system):
        """Rich format handles negative net financial (red styling)."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        # CB has negative net financial (liabilities > assets)
        result = display_agent_balance_table_renderable(simple_system, "CB01", format="rich")
        assert hasattr(result, "columns")


# ============================================================================
# Non-financial value totals
# ============================================================================


class TestNonfinancialValueTotals:
    """Test display of non-financial value totals in balance sheets."""

    def test_simple_with_nonfinancial_value(self, system_with_delivery_obligations, capsys):
        """Simple balance sheet shows valued delivery totals."""
        display_agent_balance_table(system_with_delivery_obligations, "HH01", format="simple")
        captured = capsys.readouterr()
        # Should show total or at least run without error
        assert len(captured.out) > 0

    def test_rich_with_nonfinancial_value(self, system_with_delivery_obligations, capsys):
        """Rich balance sheet shows valued delivery totals."""
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        display_agent_balance_table(system_with_delivery_obligations, "HH01", format="rich")
        capsys.readouterr()  # Just ensure no error

    def test_renderable_simple_with_nonfinancial(self, system_with_delivery_obligations):
        """Simple renderable includes non-financial totals."""
        result = display_agent_balance_table_renderable(
            system_with_delivery_obligations, "HH01", format="simple"
        )
        assert isinstance(result, str)

    def test_multiple_simple_with_nonfinancial(self, system_with_delivery_obligations, capsys):
        """Multiple agent simple display with non-financial items."""
        display_multiple_agent_balances(
            system_with_delivery_obligations, ["HH01", "HH02"], format="simple"
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ============================================================================
# Agent with no name
# ============================================================================


class TestAgentWithNoName:
    """Test display when agent has no name or name equals ID."""

    def test_display_agent_with_none_name(self, simple_system, capsys):
        """Agent with name=None shows just the ID."""
        agent = simple_system.state.agents["HH01"]
        original = agent.name
        agent.name = None
        display_agent_balance_table(simple_system, "HH01", format="simple")
        captured = capsys.readouterr()
        assert "HH01" in captured.out
        agent.name = original

    def test_multiple_agent_with_none_name(self, multi_agent_system, capsys):
        """Multiple agent display when some agents have no names."""
        agent = multi_agent_system.state.agents["HH01"]
        original = agent.name
        agent.name = None
        display_multiple_agent_balances(
            multi_agent_system, ["HH01", "HH02"], format="simple"
        )
        captured = capsys.readouterr()
        assert "HH01" in captured.out
        agent.name = original


# ============================================================================
# display_agent_balance_from_balance (rich path)
# ============================================================================


class TestDisplayAgentBalanceFromBalance:
    """Test display_agent_balance_from_balance with rich format."""

    def test_rich_format(self, simple_system, capsys):
        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")
        balance = agent_balance(simple_system, "HH01")
        from bilancio.analysis.visualization.balances import display_agent_balance_from_balance

        display_agent_balance_from_balance(balance, format="rich")
        capsys.readouterr()

    def test_rich_fallback(self, simple_system, capsys):
        balance = agent_balance(simple_system, "HH01")
        from bilancio.analysis.visualization.balances import display_agent_balance_from_balance

        with patch("bilancio.analysis.visualization.balances.RICH_AVAILABLE", False):
            display_agent_balance_from_balance(balance, format="rich")
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_default_title(self, simple_system, capsys):
        balance = agent_balance(simple_system, "HH01")
        from bilancio.analysis.visualization.balances import display_agent_balance_from_balance

        display_agent_balance_from_balance(balance, format="simple")
        captured = capsys.readouterr()
        assert "Agent HH01" in captured.out
