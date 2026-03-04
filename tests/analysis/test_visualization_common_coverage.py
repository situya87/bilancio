"""Coverage tests for analysis/visualization/common.py.

Targets: _format_currency, _print, _format_agent, parse_day_from_maturity,
BalanceRow, TAccount dataclasses.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from bilancio.analysis.visualization.common import (
    BalanceRow,
    TAccount,
    _format_agent,
    _format_currency,
    _print,
    parse_day_from_maturity,
)


# ============================================================================
# _format_currency
# ============================================================================


class TestFormatCurrency:
    """Cover _format_currency."""

    def test_basic(self):
        assert _format_currency(1000) == "1,000"

    def test_negative(self):
        assert _format_currency(-500) == "-500"

    def test_zero(self):
        assert _format_currency(0) == "0"

    def test_show_sign_positive(self):
        assert _format_currency(100, show_sign=True) == "+100"

    def test_show_sign_negative(self):
        assert _format_currency(-50, show_sign=True) == "-50"

    def test_show_sign_zero(self):
        assert _format_currency(0, show_sign=True) == "0"

    def test_large_number(self):
        assert _format_currency(1000000) == "1,000,000"


# ============================================================================
# _print
# ============================================================================


class TestPrint:
    """Cover _print with and without console."""

    def test_with_console(self):
        console = MagicMock()
        _print("hello", console)
        console.print.assert_called_once_with("hello")

    def test_without_console(self, capsys):
        _print("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out


# ============================================================================
# _format_agent
# ============================================================================


class TestFormatAgent:
    """Cover _format_agent."""

    def test_agent_not_found(self):
        system = MagicMock()
        system.state.agents.get.return_value = None
        result = _format_agent("X1", system)
        assert result == "X1"

    def test_agent_with_name(self):
        agent = MagicMock()
        agent.name = "Alice"
        system = MagicMock()
        system.state.agents.get.return_value = agent
        result = _format_agent("H1", system)
        assert result == "Alice [H1]"

    def test_agent_name_same_as_id(self):
        agent = MagicMock()
        agent.name = "H1"
        system = MagicMock()
        system.state.agents.get.return_value = agent
        result = _format_agent("H1", system)
        assert result == "H1"

    def test_agent_no_name(self):
        agent = MagicMock()
        agent.name = None
        system = MagicMock()
        system.state.agents.get.return_value = agent
        result = _format_agent("H1", system)
        assert result == "H1"


# ============================================================================
# parse_day_from_maturity
# ============================================================================


class TestParseDayFromMaturity:
    """Cover parse_day_from_maturity."""

    def test_valid_day_string(self):
        assert parse_day_from_maturity("Day 42") == 42

    def test_day_with_spaces(self):
        assert parse_day_from_maturity("Day  10 ") == 10

    def test_none_returns_inf(self):
        result = parse_day_from_maturity(None)
        assert result == math.inf

    def test_non_string_returns_inf(self):
        result = parse_day_from_maturity(123)  # type: ignore[arg-type]
        assert result == math.inf

    def test_no_day_prefix(self):
        result = parse_day_from_maturity("Maturity 5")
        assert result == math.inf

    def test_invalid_number(self):
        result = parse_day_from_maturity("Day abc")
        assert result == math.inf

    def test_empty_string(self):
        result = parse_day_from_maturity("")
        assert result == math.inf


# ============================================================================
# Dataclasses
# ============================================================================


class TestDataclasses:
    """Cover BalanceRow and TAccount dataclasses."""

    def test_balance_row_creation(self):
        row = BalanceRow(
            name="Cash",
            quantity=10,
            value_minor=1000,
            counterparty_name="CB",
            maturity="on-demand",
            id_or_alias="c001",
        )
        assert row.name == "Cash"
        assert row.quantity == 10
        assert row.value_minor == 1000
        assert row.counterparty_name == "CB"
        assert row.maturity == "on-demand"
        assert row.id_or_alias == "c001"

    def test_balance_row_defaults(self):
        row = BalanceRow(
            name="Test", quantity=None, value_minor=None, counterparty_name=None, maturity=None
        )
        assert row.id_or_alias is None

    def test_t_account_creation(self):
        assets = [
            BalanceRow("Cash", 1, 100, None, None),
        ]
        liabs = [
            BalanceRow("Debt", 1, 50, None, None),
        ]
        acct = TAccount(assets=assets, liabilities=liabs)
        assert len(acct.assets) == 1
        assert len(acct.liabilities) == 1
        assert acct.assets[0].name == "Cash"
        assert acct.liabilities[0].name == "Debt"
