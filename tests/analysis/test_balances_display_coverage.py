"""Coverage tests for analysis/visualization/balances.py.

Targets uncovered display functions: display_agent_balance_table (simple format),
display_agent_balance_from_balance, display_multiple_agent_balances (simple),
build_t_account_rows, display_agent_t_account (simple + rich).
"""

from __future__ import annotations

from bilancio.analysis.balances import agent_balance
from bilancio.analysis.visualization.balances import (
    _cells,
    _display_simple_multiple_agent_balances,
    _fmt_qty,
    _fmt_val,
    build_t_account_rows,
    display_agent_balance_from_balance,
    display_agent_balance_table,
    display_agent_t_account,
    display_agent_t_account_renderable,
    display_multiple_agent_balances,
)
from bilancio.analysis.visualization.common import BalanceRow
from bilancio.config.apply import create_agent
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.system import System

# ============================================================================
# Helpers
# ============================================================================


def _make_system():
    """Create a system with CB, two households, and some instruments."""
    sys = System()
    cb = create_agent(type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})())
    h1 = create_agent(type("S", (), {"id": "H1", "kind": "household", "name": "Alice"})())
    h2 = create_agent(type("S", (), {"id": "H2", "kind": "household", "name": "Bob"})())
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.mint_cash("H1", 1000)
    sys.mint_cash("H2", 500)
    return sys


def _make_system_with_payable():
    """Create a system with a payable."""
    sys = _make_system()
    payable = Payable(
        id="P1",
        kind=InstrumentKind.PAYABLE,
        amount=200,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=5,
    )
    sys.add_contract(payable)
    return sys


# ============================================================================
# display_agent_balance_table
# ============================================================================


class TestDisplayAgentBalanceTable:
    """Cover display_agent_balance_table with different formats."""

    def test_rich_format(self, capsys):
        sys = _make_system()
        display_agent_balance_table(sys, "H1", format="rich")
        captured = capsys.readouterr()
        assert "Alice" in captured.out or "H1" in captured.out

    def test_simple_format(self, capsys):
        sys = _make_system()
        display_agent_balance_table(sys, "H1", format="simple")
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out
        assert "LIABILITIES" in captured.out
        assert "TOTAL FINANCIAL" in captured.out

    def test_custom_title(self, capsys):
        sys = _make_system()
        display_agent_balance_table(sys, "H1", format="simple", title="Custom Title")
        captured = capsys.readouterr()
        assert "Custom Title" in captured.out

    def test_agent_name_same_as_id(self, capsys):
        sys = System()
        cb = create_agent(type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})())
        h1 = create_agent(type("S", (), {"id": "H1", "kind": "household", "name": "H1"})())
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)
        display_agent_balance_table(sys, "H1", format="simple")
        captured = capsys.readouterr()
        assert "H1" in captured.out


# ============================================================================
# display_agent_balance_from_balance
# ============================================================================


class TestDisplayAgentBalanceFromBalance:
    """Cover display_agent_balance_from_balance."""

    def test_rich_format(self, capsys):
        sys = _make_system()
        balance = agent_balance(sys, "H1")
        display_agent_balance_from_balance(balance, format="rich")
        captured = capsys.readouterr()
        assert "H1" in captured.out

    def test_simple_format(self, capsys):
        sys = _make_system()
        balance = agent_balance(sys, "H1")
        display_agent_balance_from_balance(balance, format="simple")
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out

    def test_custom_title(self, capsys):
        sys = _make_system()
        balance = agent_balance(sys, "H1")
        display_agent_balance_from_balance(balance, format="simple", title="My Title")
        captured = capsys.readouterr()
        assert "My Title" in captured.out


# ============================================================================
# display_multiple_agent_balances
# ============================================================================


class TestDisplayMultipleAgentBalances:
    """Cover display_multiple_agent_balances."""

    def test_rich_format_with_ids(self, capsys):
        sys = _make_system()
        display_multiple_agent_balances(sys, ["H1", "H2"], format="rich")
        captured = capsys.readouterr()
        # Should render something
        assert len(captured.out) > 0

    def test_simple_format_with_ids(self, capsys):
        sys = _make_system()
        display_multiple_agent_balances(sys, ["H1", "H2"], format="simple")
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out
        assert "LIABILITIES" in captured.out

    def test_simple_format_with_balance_objects(self, capsys):
        sys = _make_system()
        b1 = agent_balance(sys, "H1")
        b2 = agent_balance(sys, "H2")
        display_multiple_agent_balances(sys, [b1, b2], format="simple")
        captured = capsys.readouterr()
        assert "ASSETS" in captured.out

    def test_simple_format_without_system_agents(self, capsys):
        """When agent not in system, uses agent_id as header."""
        sys = _make_system()
        b1 = agent_balance(sys, "H1")
        # Create a new empty system (no agents)
        sys2 = System()
        _display_simple_multiple_agent_balances([b1], sys2)
        captured = capsys.readouterr()
        assert "H1" in captured.out


# ============================================================================
# build_t_account_rows
# ============================================================================


class TestBuildTAccountRows:
    """Cover build_t_account_rows."""

    def test_basic(self):
        sys = _make_system()
        acct = build_t_account_rows(sys, "H1")
        assert len(acct.assets) > 0
        # H1 has cash as asset
        assert any("cash" in a.name.lower() for a in acct.assets)

    def test_with_payable(self):
        sys = _make_system_with_payable()
        acct_h1 = build_t_account_rows(sys, "H1")
        acct_h2 = build_t_account_rows(sys, "H2")
        # H1 has payable as liability
        assert len(acct_h1.liabilities) > 0
        # H2 has payable as asset
        assert len(acct_h2.assets) > 0

    def test_cb_has_liabilities(self):
        sys = _make_system()
        acct = build_t_account_rows(sys, "CB")
        # CB has cash liabilities (issued cash to H1, H2)
        assert len(acct.liabilities) > 0


# ============================================================================
# display_agent_t_account
# ============================================================================


class TestDisplayAgentTAccount:
    """Cover display_agent_t_account."""

    def test_rich_format(self, capsys):
        sys = _make_system()
        display_agent_t_account(sys, "H1", format="rich")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_simple_format(self, capsys):
        sys = _make_system()
        display_agent_t_account(sys, "H1", format="simple")
        captured = capsys.readouterr()
        assert "Assets" in captured.out

    def test_with_payable_rich(self, capsys):
        sys = _make_system_with_payable()
        display_agent_t_account(sys, "H1", format="rich")
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ============================================================================
# display_agent_t_account_renderable
# ============================================================================


class TestDisplayAgentTAccountRenderable:
    """Cover display_agent_t_account_renderable."""

    def test_returns_renderable(self):
        sys = _make_system()
        result = display_agent_t_account_renderable(sys, "H1")
        assert hasattr(result, "columns")  # Rich Table has columns

    def test_with_payable(self):
        sys = _make_system_with_payable()
        result = display_agent_t_account_renderable(sys, "H1")
        assert hasattr(result, "columns")


# ============================================================================
# Helper functions
# ============================================================================


class TestHelperFunctions:
    """Cover _fmt_qty, _fmt_val, _cells."""

    def test_fmt_qty_with_value(self):
        row = BalanceRow("Cash", quantity=42, value_minor=100, counterparty_name=None, maturity=None)
        assert _fmt_qty(row) == "42"

    def test_fmt_qty_none_row(self):
        assert _fmt_qty(None) == "—"

    def test_fmt_qty_none_quantity(self):
        row = BalanceRow("Cash", quantity=None, value_minor=100, counterparty_name=None, maturity=None)
        assert _fmt_qty(row) == "—"

    def test_fmt_val_with_value(self):
        row = BalanceRow("Cash", quantity=None, value_minor=1000, counterparty_name=None, maturity=None)
        assert _fmt_val(row) == "1,000"

    def test_fmt_val_none_row(self):
        assert _fmt_val(None) == "—"

    def test_fmt_val_none_value(self):
        row = BalanceRow("Cash", quantity=None, value_minor=None, counterparty_name=None, maturity=None)
        assert _fmt_val(row) == "—"

    def test_cells_with_row(self):
        row = BalanceRow(
            "Cash", quantity=10, value_minor=500, counterparty_name="CB", maturity="on-demand"
        )
        cells = _cells(row)
        assert cells[0] == "Cash"
        assert cells[1] == "10"
        assert cells[2] == "500"
        assert cells[3] == "CB"
        assert cells[4] == "on-demand"

    def test_cells_none_row(self):
        cells = _cells(None)
        assert cells == ("", "", "", "", "")

    def test_cells_with_none_counterparty(self):
        row = BalanceRow("Cash", quantity=None, value_minor=100, counterparty_name=None, maturity=None)
        cells = _cells(row)
        assert cells[3] == "—"
        assert cells[4] == "—"
