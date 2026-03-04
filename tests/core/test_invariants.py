"""Tests for bilancio.core.invariants module.

Covers the invariant assertion functions, including error paths
(duplicate refs, mismatched CB totals, negative balances, stock checks).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.core.invariants import (
    assert_all_stock_ids_owned,
    assert_cb_cash_matches_outstanding,
    assert_cb_reserves_match,
    assert_double_entry_numeric,
    assert_no_duplicate_refs,
    assert_no_duplicate_stock_refs,
    assert_no_negative_balances,
    assert_no_negative_stocks,
)
from bilancio.domain.agent import Agent
from bilancio.domain.goods import StockLot
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.means_of_payment import BankDeposit, Cash, ReserveDeposit
from bilancio.engines.system import System


def _make_system_with_cash(amount: int, cb_outstanding: int) -> System:
    """Create a system with a single Cash contract and set cb_cash_outstanding."""
    system = System()
    cash = Cash(
        id="C1",
        kind=InstrumentKind.CASH,
        amount=amount,
        denom="X",
        asset_holder_id="A1",
        liability_issuer_id="CB",
    )
    system.state.contracts["C1"] = cash
    system.state.cb_cash_outstanding = cb_outstanding
    return system


class TestAssertCbCashMatchesOutstanding:
    """Tests for assert_cb_cash_matches_outstanding."""

    def test_matching_passes(self):
        system = _make_system_with_cash(100, 100)
        assert_cb_cash_matches_outstanding(system)  # should not raise

    def test_mismatch_raises(self):
        system = _make_system_with_cash(100, 200)
        with pytest.raises(AssertionError, match="CB cash mismatch"):
            assert_cb_cash_matches_outstanding(system)

    def test_empty_contracts_zero_outstanding(self):
        system = System()
        system.state.cb_cash_outstanding = 0
        assert_cb_cash_matches_outstanding(system)  # should not raise

    def test_empty_contracts_nonzero_outstanding(self):
        system = System()
        system.state.cb_cash_outstanding = 50
        with pytest.raises(AssertionError, match="CB cash mismatch"):
            assert_cb_cash_matches_outstanding(system)


class TestAssertNoNegativeBalances:
    """Tests for assert_no_negative_balances."""

    def test_positive_cash_passes(self):
        system = _make_system_with_cash(100, 100)
        assert_no_negative_balances(system)  # should not raise

    def test_negative_cash_raises(self):
        system = _make_system_with_cash(-10, -10)
        with pytest.raises(AssertionError, match="negative balance detected"):
            assert_no_negative_balances(system)

    def test_negative_bank_deposit_raises(self):
        system = System()
        dep = BankDeposit(
            id="BD1",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=-5,
            denom="X",
            asset_holder_id="A1",
            liability_issuer_id="B1",
        )
        system.state.contracts["BD1"] = dep
        with pytest.raises(AssertionError, match="negative balance detected"):
            assert_no_negative_balances(system)

    def test_negative_reserve_deposit_raises(self):
        system = System()
        res = ReserveDeposit(
            id="RD1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=-1,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
        )
        system.state.contracts["RD1"] = res
        with pytest.raises(AssertionError, match="negative balance detected"):
            assert_no_negative_balances(system)

    def test_zero_balance_passes(self):
        system = _make_system_with_cash(0, 0)
        assert_no_negative_balances(system)  # should not raise


class TestAssertCbReservesMatch:
    """Tests for assert_cb_reserves_match."""

    def test_matching_reserves_passes(self):
        system = System()
        res = ReserveDeposit(
            id="RD1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=500,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
        )
        system.state.contracts["RD1"] = res
        system.state.cb_reserves_outstanding = 500
        assert_cb_reserves_match(system)  # should not raise

    def test_mismatch_reserves_raises(self):
        system = System()
        res = ReserveDeposit(
            id="RD1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=500,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
        )
        system.state.contracts["RD1"] = res
        system.state.cb_reserves_outstanding = 999
        with pytest.raises(AssertionError, match="CB reserves mismatch"):
            assert_cb_reserves_match(system)


class TestAssertDoubleEntryNumeric:
    """Tests for assert_double_entry_numeric."""

    def test_positive_amounts_pass(self):
        system = _make_system_with_cash(100, 100)
        assert_double_entry_numeric(system)  # should not raise

    def test_negative_amount_raises(self):
        system = _make_system_with_cash(-1, -1)
        with pytest.raises(AssertionError, match="negative amount detected"):
            assert_double_entry_numeric(system)


class TestAssertNoDuplicateRefs:
    """Tests for assert_no_duplicate_refs."""

    def test_no_duplicates_passes(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm", asset_ids={"C1", "C2"})
        system.state.agents["A1"] = agent
        assert_no_duplicate_refs(system)  # should not raise

    def test_duplicate_asset_ref_raises(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm")
        # Manually create a list-like with duplicates (sets normally prevent this,
        # but we test the iteration logic by converting to list)
        agent.asset_ids = ["C1", "C1"]  # type: ignore[assignment]
        system.state.agents["A1"] = agent
        with pytest.raises(AssertionError, match="duplicate asset ref"):
            assert_no_duplicate_refs(system)

    def test_duplicate_liability_ref_raises(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm")
        agent.liability_ids = ["L1", "L1"]  # type: ignore[assignment]
        system.state.agents["A1"] = agent
        with pytest.raises(AssertionError, match="duplicate liability ref"):
            assert_no_duplicate_refs(system)

    def test_empty_agents_passes(self):
        system = System()
        assert_no_duplicate_refs(system)  # should not raise


class TestAssertAllStockIdsOwned:
    """Tests for assert_all_stock_ids_owned."""

    def test_valid_ownership_passes(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm", stock_ids={"S1"})
        system.state.agents["A1"] = agent
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=10,
            unit_price=Decimal("5"), owner_id="A1",
        )
        system.state.stocks["S1"] = stock
        assert_all_stock_ids_owned(system)  # should not raise

    def test_stock_owned_by_multiple_agents_raises(self):
        system = System()
        a1 = Agent(id="A1", name="Agent 1", kind="firm")
        a2 = Agent(id="A2", name="Agent 2", kind="firm")
        a1.stock_ids = ["S1"]  # type: ignore[assignment]
        a2.stock_ids = ["S1"]  # type: ignore[assignment]
        system.state.agents["A1"] = a1
        system.state.agents["A2"] = a2
        with pytest.raises(AssertionError, match="owned by multiple agents"):
            assert_all_stock_ids_owned(system)

    def test_stock_in_registry_without_owner_raises(self):
        system = System()
        # No agents own S1, but it's in the registry
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=5,
            unit_price=Decimal("2"), owner_id="A1",
        )
        system.state.stocks["S1"] = stock
        with pytest.raises(AssertionError, match="no agent owns it"):
            assert_all_stock_ids_owned(system)

    def test_stock_owner_id_mismatch_raises(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm", stock_ids={"S1"})
        system.state.agents["A1"] = agent
        # Stock says owner_id="A2" but A1 holds it
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=5,
            unit_price=Decimal("2"), owner_id="A2",
        )
        system.state.stocks["S1"] = stock
        with pytest.raises(AssertionError, match="doesn't match owning agent"):
            assert_all_stock_ids_owned(system)


class TestAssertNoNegativeStocks:
    """Tests for assert_no_negative_stocks."""

    def test_positive_stock_passes(self):
        system = System()
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=10,
            unit_price=Decimal("5"), owner_id="A1",
        )
        system.state.stocks["S1"] = stock
        assert_no_negative_stocks(system)  # should not raise

    def test_negative_stock_raises(self):
        system = System()
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=-1,
            unit_price=Decimal("5"), owner_id="A1",
        )
        system.state.stocks["S1"] = stock
        with pytest.raises(AssertionError, match="negative quantity"):
            assert_no_negative_stocks(system)

    def test_zero_stock_passes(self):
        system = System()
        stock = StockLot(
            id="S1", kind="stock_lot", sku="widget", quantity=0,
            unit_price=Decimal("5"), owner_id="A1",
        )
        system.state.stocks["S1"] = stock
        assert_no_negative_stocks(system)  # should not raise


class TestAssertNoDuplicateStockRefs:
    """Tests for assert_no_duplicate_stock_refs."""

    def test_no_duplicates_passes(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm", stock_ids={"S1", "S2"})
        system.state.agents["A1"] = agent
        assert_no_duplicate_stock_refs(system)  # should not raise

    def test_duplicate_stock_ref_raises(self):
        system = System()
        agent = Agent(id="A1", name="Agent 1", kind="firm")
        agent.stock_ids = ["S1", "S1"]  # type: ignore[assignment]
        system.state.agents["A1"] = agent
        with pytest.raises(AssertionError, match="duplicate stock ref"):
            assert_no_duplicate_stock_refs(system)

    def test_empty_agents_passes(self):
        system = System()
        assert_no_duplicate_stock_refs(system)  # should not raise
