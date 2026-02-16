"""
Tests for bilancio.export.writers — targeting uncovered lines.

Uncovered:
  - Lines 28-29: decimal_default else branch (non-integer Decimal)
  - Lines 124-183: write_balances_snapshot (entire function)
"""

import csv
import json
from decimal import Decimal

import pytest

from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.system import System
from bilancio.export.writers import (
    decimal_default,
    write_balances_csv,
    write_balances_snapshot,
    write_events_jsonl,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_system_with_payable():
    """Build a minimal System with a CB, two households, and one payable."""
    sys = System()
    cb = CentralBank(id="CB", name="CentralBank")
    h1 = Household(id="H1", name="Alice", kind="household")
    h2 = Household(id="H2", name="Bob", kind="household")
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)

    with sys.setup():
        # Give H1 some cash so the system has financial instruments
        sys.mint_cash("H1", 500)

        # Create a payable: H2 owes H1 100
        payable = Payable(
            id="PAY1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=5,
        )
        sys.add_contract(payable)

    return sys


def _make_system_with_stock():
    """Build a System with a stock lot for snapshot testing."""
    sys = System()
    cb = CentralBank(id="CB", name="CentralBank")
    h1 = Household(id="H1", name="Alice", kind="household")
    h2 = Household(id="H2", name="Bob", kind="household")
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)

    with sys.setup():
        sys.mint_cash("H1", 200)
        sys.create_stock("H1", sku="CHAIR", quantity=10, unit_price=Decimal("25"))

    return sys


# =============================================================================
# decimal_default tests
# =============================================================================


class TestDecimalDefault:
    """Tests for the JSON encoder helper."""

    def test_integer_decimal_returns_int(self):
        """Decimal('42.00') normalizes to integer 42."""
        assert decimal_default(Decimal("42.00")) == 42

    def test_non_integer_decimal_returns_str(self):
        """Decimal('3.14') returns '3.14' as string (line 28-29)."""
        result = decimal_default(Decimal("3.14"))
        assert result == "3.14"
        assert isinstance(result, str)

    def test_non_integer_decimal_trailing_zeros_stripped(self):
        """Decimal('1.500') normalizes to '1.5' string."""
        result = decimal_default(Decimal("1.500"))
        assert result == "1.5"

    def test_non_serializable_raises(self):
        """Non-Decimal types raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            decimal_default(object())


# =============================================================================
# write_balances_csv tests
# =============================================================================


class TestWriteBalancesCsv:
    """Tests for CSV balance export."""

    def test_writes_csv_with_system_rows(self, tmp_path):
        """CSV contains agent rows plus SYSTEM summary rows."""
        sys = _make_system_with_payable()
        out = tmp_path / "balances.csv"
        write_balances_csv(sys, out)

        assert out.exists()
        with open(out) as f:
            reader = list(csv.DictReader(f))

        # Should have at least the agent rows + SYSTEM rows.
        # as_rows() adds 1 SYSTEM row, write_balances_csv adds 3 more = 4 total.
        system_rows = [r for r in reader if r.get("agent_id") == "SYSTEM"]
        assert len(system_rows) >= 3
        # The 3 explicitly written SYSTEM rows have item_name
        named_system_rows = [r for r in system_rows if r.get("item_name")]
        names = {r["item_name"] for r in named_system_rows}
        assert "Total Assets" in names
        assert "Total Liabilities" in names
        assert "Total Equity" in names

    def test_decimal_converted_to_float(self, tmp_path):
        """Decimal values in rows are converted to floats for CSV output."""
        sys = _make_system_with_stock()
        out = tmp_path / "balances.csv"
        write_balances_csv(sys, out)

        # Just verify it doesn't crash and file is valid CSV
        with open(out) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) > 0


# =============================================================================
# write_events_jsonl tests
# =============================================================================


class TestWriteEventsJsonl:
    """Tests for JSONL event export."""

    def test_writes_events_as_jsonl(self, tmp_path):
        """Each event becomes one JSON line."""
        sys = _make_system_with_payable()
        out = tmp_path / "events.jsonl"
        write_events_jsonl(sys, out)

        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == len(sys.state.events)

        # Each line must be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "kind" in parsed

    def test_empty_events(self, tmp_path):
        """System with no events writes an empty file."""
        sys = System()
        out = tmp_path / "events.jsonl"
        write_events_jsonl(sys, out)

        assert out.exists()
        assert out.read_text() == ""


# =============================================================================
# write_balances_snapshot tests  (lines 124-183)
# =============================================================================


class TestWriteBalancesSnapshot:
    """Tests for the JSON balance snapshot export."""

    def test_snapshot_contains_all_agents(self, tmp_path):
        """When agent_ids=None, snapshot includes every agent."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0)

        data = json.loads(out.read_text())
        assert data["day"] == 0
        # All three agents should be present
        assert set(data["agents"].keys()) == {"CB", "H1", "H2"}

    def test_snapshot_filters_agent_ids(self, tmp_path):
        """When agent_ids is provided, only those agents appear."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=1, agent_ids=["H1"])

        data = json.loads(out.read_text())
        assert set(data["agents"].keys()) == {"H1"}

    def test_snapshot_skips_unknown_agent(self, tmp_path):
        """Unknown agent IDs are silently skipped."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0, agent_ids=["H1", "NONEXISTENT"])

        data = json.loads(out.read_text())
        assert "NONEXISTENT" not in data["agents"]
        assert "H1" in data["agents"]

    def test_snapshot_records_assets_and_liabilities(self, tmp_path):
        """Agent H1 has assets (cash + payable), H2 has liabilities (payable)."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0)

        data = json.loads(out.read_text())

        h1 = data["agents"]["H1"]
        assert h1["name"] == "Alice"
        # H1 holds at least 2 assets: cash and payable
        assert len(h1["assets"]) >= 2

        h2 = data["agents"]["H2"]
        assert h2["name"] == "Bob"
        # H2 issued the payable
        assert len(h2["liabilities"]) >= 1

    def test_snapshot_records_stocks(self, tmp_path):
        """Stock lots appear in the snapshot with sku, quantity, unit_price, total_value."""
        sys = _make_system_with_stock()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0)

        data = json.loads(out.read_text())

        h1 = data["agents"]["H1"]
        assert len(h1["stocks"]) == 1
        stock = h1["stocks"][0]
        assert stock["sku"] == "CHAIR"
        assert stock["quantity"] == 10
        # unit_price and total_value are Decimals -> go through decimal_default
        assert stock["total_value"] == 250  # 10 * 25

    def test_snapshot_asset_details(self, tmp_path):
        """Each asset record contains id, type, and amount."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0, agent_ids=["H1"])

        data = json.loads(out.read_text())
        assets = data["agents"]["H1"]["assets"]
        for asset in assets:
            assert "id" in asset
            assert "type" in asset
            assert "amount" in asset

    def test_snapshot_liability_details(self, tmp_path):
        """Each liability record contains id, type, and amount."""
        sys = _make_system_with_payable()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0, agent_ids=["H2"])

        data = json.loads(out.read_text())
        liabilities = data["agents"]["H2"]["liabilities"]
        for liability in liabilities:
            assert "id" in liability
            assert "type" in liability
            assert "amount" in liability

    def test_snapshot_no_stocks_is_empty_list(self, tmp_path):
        """Agent without stocks has an empty stocks list."""
        sys = _make_system_with_payable()  # no stocks created
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0, agent_ids=["H1"])

        data = json.loads(out.read_text())
        assert data["agents"]["H1"]["stocks"] == []

    def test_snapshot_decimal_unit_price_serialized(self, tmp_path):
        """Decimal unit_price goes through decimal_default in JSON output."""
        sys = _make_system_with_stock()
        out = tmp_path / "snap.json"
        write_balances_snapshot(sys, out, day=0, agent_ids=["H1"])

        data = json.loads(out.read_text())
        stock = data["agents"]["H1"]["stocks"][0]
        # unit_price=25 normalizes to int 25
        assert stock["unit_price"] == 25
