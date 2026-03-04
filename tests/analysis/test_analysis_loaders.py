"""Tests for bilancio.analysis.loaders module."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from bilancio.analysis.loaders import (
    _find_csv,
    _to_decimal,
    load_bank_snapshots,
    load_dealer_snapshots,
    read_balances_csv,
    read_events_jsonl,
)


# ---------------------------------------------------------------------------
# _to_decimal
# ---------------------------------------------------------------------------


class TestToDecimal:
    def test_valid_int(self):
        assert _to_decimal(42) == Decimal("42")

    def test_valid_float(self):
        assert _to_decimal(3.14) == Decimal("3.14")

    def test_valid_string(self):
        assert _to_decimal("100.50") == Decimal("100.50")

    def test_none_returns_zero(self):
        assert _to_decimal(None) == Decimal("0")

    def test_invalid_string_returns_zero(self):
        assert _to_decimal("not_a_number") == Decimal("0")

    def test_already_decimal(self):
        d = Decimal("99.99")
        assert _to_decimal(d) is d

    def test_bool_true(self):
        assert _to_decimal(True) == Decimal("1")

    def test_bool_false(self):
        assert _to_decimal(False) == Decimal("0")


# ---------------------------------------------------------------------------
# read_events_jsonl
# ---------------------------------------------------------------------------


class TestReadEventsJsonl:
    def test_reads_valid_events(self, tmp_path: Path):
        events = [
            {"type": "setup", "day": 0, "agent": "a1", "amount": "100.00"},
            {"type": "payment", "day": 1, "agent": "a2", "amount": 50},
            {"type": "default", "day": 2, "agent": "a3", "amount": "25.5", "due_day": 5},
        ]
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        result = list(read_events_jsonl(p))

        assert len(result) == 3
        assert result[0]["amount"] == Decimal("100.00")
        assert result[0]["day"] == 0
        assert result[1]["amount"] == Decimal("50")
        assert result[1]["day"] == 1
        assert result[2]["due_day"] == 5

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")

        result = list(read_events_jsonl(p))
        assert result == []

    def test_skips_blank_lines(self, tmp_path: Path):
        content = (
            '{"type": "a", "day": 0}\n'
            "\n"
            '{"type": "b", "day": 1}\n'
            "   \n"
            '{"type": "c", "day": 2}\n'
        )
        p = tmp_path / "gaps.jsonl"
        p.write_text(content)

        result = list(read_events_jsonl(p))
        assert len(result) == 3
        assert [e["type"] for e in result] == ["a", "b", "c"]

    def test_malformed_line_raises(self, tmp_path: Path):
        """Malformed JSON raises json.JSONDecodeError (the loader does not silently skip)."""
        content = '{"type": "ok", "day": 0}\n' "NOT VALID JSON\n"
        p = tmp_path / "bad.jsonl"
        p.write_text(content)

        gen = read_events_jsonl(p)
        first = next(gen)
        assert first["type"] == "ok"
        with pytest.raises(json.JSONDecodeError):
            next(gen)


# ---------------------------------------------------------------------------
# read_balances_csv
# ---------------------------------------------------------------------------


class TestReadBalancesCsv:
    def test_reads_valid_csv(self, tmp_path: Path):
        csv_content = (
            "day,agent,item_type,item,amount\n"
            "0,firm_1,asset,cash,1000\n"
            "0,firm_1,liability,payable,500\n"
            "1,firm_2,asset,cash,200\n"
        )
        p = tmp_path / "balances.csv"
        p.write_text(csv_content)

        rows = read_balances_csv(p)

        assert len(rows) == 3
        assert rows[0]["day"] == "0"
        assert rows[0]["agent"] == "firm_1"
        assert rows[0]["amount"] == "1000"
        assert rows[1]["item_type"] == "liability"
        assert rows[2]["agent"] == "firm_2"

    def test_empty_csv_has_no_rows(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("day,agent,item_type,item,amount\n")

        rows = read_balances_csv(p)
        assert rows == []


# ---------------------------------------------------------------------------
# _find_csv
# ---------------------------------------------------------------------------


class TestFindCsv:
    def test_finds_in_direct_path(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("col\n1\n")

        result = _find_csv(tmp_path, "data.csv")
        assert result == tmp_path / "data.csv"

    def test_finds_in_out_subdir(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "data.csv").write_text("col\n1\n")

        result = _find_csv(tmp_path, "data.csv")
        assert result == out / "data.csv"

    def test_prefers_direct_over_out(self, tmp_path: Path):
        """When the file exists in both locations, the direct path wins."""
        (tmp_path / "data.csv").write_text("direct\n")
        out = tmp_path / "out"
        out.mkdir()
        (out / "data.csv").write_text("nested\n")

        result = _find_csv(tmp_path, "data.csv")
        assert result == tmp_path / "data.csv"

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert _find_csv(tmp_path, "nonexistent.csv") is None


# ---------------------------------------------------------------------------
# load_dealer_snapshots
# ---------------------------------------------------------------------------


class TestLoadDealerSnapshots:
    def test_loads_valid_csv(self, tmp_path: Path):
        csv_content = (
            "day,dealer_cash,inventory_count,inventory_value\n"
            "0,1000.0,0,0.0\n"
            "1,800.0,2,180.0\n"
        )
        (tmp_path / "dealer_state.csv").write_text(csv_content)

        df = load_dealer_snapshots(tmp_path)

        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["day", "dealer_cash", "inventory_count", "inventory_value"]
        assert df["dealer_cash"].iloc[0] == 1000.0

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert load_dealer_snapshots(tmp_path) is None

    def test_loads_from_out_subdir(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "dealer_state.csv").write_text("day,cash\n0,100\n")

        df = load_dealer_snapshots(tmp_path)
        assert df is not None
        assert len(df) == 1


# ---------------------------------------------------------------------------
# load_bank_snapshots
# ---------------------------------------------------------------------------


class TestLoadBankSnapshots:
    def test_loads_valid_csv(self, tmp_path: Path):
        csv_content = (
            "day,bank_reserves,total_loans,total_deposits\n"
            "0,5000.0,0.0,3000.0\n"
            "1,4500.0,500.0,3000.0\n"
        )
        (tmp_path / "bank_state.csv").write_text(csv_content)

        df = load_bank_snapshots(tmp_path)

        assert df is not None
        assert len(df) == 2
        assert "bank_reserves" in df.columns
        assert df["total_loans"].iloc[1] == 500.0

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert load_bank_snapshots(tmp_path) is None

    def test_loads_from_out_subdir(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "bank_state.csv").write_text("day,reserves\n0,5000\n")

        df = load_bank_snapshots(tmp_path)
        assert df is not None
        assert len(df) == 1
