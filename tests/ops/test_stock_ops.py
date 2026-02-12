"""Tests for stock operation primitives."""

import pytest
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.domain.agents.firm import Firm
from bilancio.domain.goods import StockLot
from bilancio.ops.primitives_stock import (
    stock_fungible_key,
    split_stock,
    merge_stock,
    consume_stock,
)
from bilancio.core.errors import ValidationError


def make_system_with_stock(sku="WIDGET", quantity=100, unit_price=Decimal("10.00"), divisible=True):
    """Create a test system with a single stock lot."""
    system = System()
    with system.setup():
        f1 = Firm(id="F1", name="Firm1", kind="firm")
        system.add_agent(f1)
    stock_id = system.create_stock("F1", sku=sku, quantity=quantity, unit_price=unit_price, divisible=divisible)
    return system, stock_id


def make_system_with_two_stocks(sku="WIDGET", qty1=100, qty2=50, price=Decimal("10.00")):
    """Create a test system with two stock lots."""
    system = System()
    with system.setup():
        f1 = Firm(id="F1", name="Firm1", kind="firm")
        system.add_agent(f1)
    stock1_id = system.create_stock("F1", sku=sku, quantity=qty1, unit_price=price)
    stock2_id = system.create_stock("F1", sku=sku, quantity=qty2, unit_price=price)
    return system, stock1_id, stock2_id


class TestStockFungibleKey:
    """Tests for stock_fungible_key function."""

    def test_basic_key_generation(self):
        """Test that fungible key includes all relevant fields."""
        lot = StockLot(
            id="S1",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1",
            divisible=True
        )
        key = stock_fungible_key(lot)
        assert key == ("stock_lot", "WIDGET", "F1", Decimal("10.00"))

    def test_same_lots_same_key(self):
        """Test that identical lots produce the same key."""
        lot1 = StockLot(
            id="S1",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        lot2 = StockLot(
            id="S2",
            kind="stock_lot",
            sku="WIDGET",
            quantity=50,  # Different quantity should not matter
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        assert stock_fungible_key(lot1) == stock_fungible_key(lot2)

    def test_different_sku_different_key(self):
        """Test that different SKUs produce different keys."""
        lot1 = StockLot(
            id="S1",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        lot2 = StockLot(
            id="S2",
            kind="stock_lot",
            sku="GADGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        assert stock_fungible_key(lot1) != stock_fungible_key(lot2)

    def test_different_price_different_key(self):
        """Test that different unit prices produce different keys."""
        lot1 = StockLot(
            id="S1",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        lot2 = StockLot(
            id="S2",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("15.00"),
            owner_id="F1"
        )
        assert stock_fungible_key(lot1) != stock_fungible_key(lot2)

    def test_different_owner_different_key(self):
        """Test that different owners produce different keys."""
        lot1 = StockLot(
            id="S1",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F1"
        )
        lot2 = StockLot(
            id="S2",
            kind="stock_lot",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("10.00"),
            owner_id="F2"
        )
        assert stock_fungible_key(lot1) != stock_fungible_key(lot2)


class TestSplitStock:
    """Tests for split_stock function."""

    def test_normal_split(self):
        """Test splitting a stock lot into two pieces."""
        system, stock_id = make_system_with_stock(quantity=100)

        # Split off 30 units
        new_id = split_stock(system, stock_id, 30)

        # Check original stock reduced
        original = system.state.stocks[stock_id]
        assert original.quantity == 70

        # Check new stock created
        new_stock = system.state.stocks[new_id]
        assert new_stock.quantity == 30
        assert new_stock.sku == original.sku
        assert new_stock.unit_price == original.unit_price
        assert new_stock.owner_id == original.owner_id

        # Check agent has both stocks
        agent = system.state.agents["F1"]
        assert stock_id in agent.stock_ids
        assert new_id in agent.stock_ids

        # Check event logged
        events = [e for e in system.state.events if e.get("kind") == "StockSplit"]
        assert len(events) == 1
        assert events[0]["original_id"] == stock_id
        assert events[0]["new_id"] == new_id
        assert events[0]["split_qty"] == 30
        assert events[0]["remaining_qty"] == 70

    def test_split_preserves_properties(self):
        """Test that split preserves all stock properties."""
        system, stock_id = make_system_with_stock(
            sku="GADGET",
            quantity=100,
            unit_price=Decimal("25.50"),
            divisible=True
        )

        new_id = split_stock(system, stock_id, 40)

        original = system.state.stocks[stock_id]
        new_stock = system.state.stocks[new_id]

        assert new_stock.sku == "GADGET"
        assert new_stock.unit_price == Decimal("25.50")
        assert new_stock.divisible == True
        assert new_stock.owner_id == "F1"

    def test_split_indivisible_lot_raises_error(self):
        """Test that splitting an indivisible lot raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100, divisible=False)

        with pytest.raises(ValidationError, match="not divisible"):
            split_stock(system, stock_id, 30)

    def test_split_zero_quantity_raises_error(self):
        """Test that splitting zero quantity raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid split quantity"):
            split_stock(system, stock_id, 0)

    def test_split_negative_quantity_raises_error(self):
        """Test that splitting negative quantity raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid split quantity"):
            split_stock(system, stock_id, -10)

    def test_split_full_quantity_raises_error(self):
        """Test that splitting the full quantity raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid split quantity"):
            split_stock(system, stock_id, 100)

    def test_split_exceeding_quantity_raises_error(self):
        """Test that splitting more than available raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid split quantity"):
            split_stock(system, stock_id, 150)


class TestMergeStock:
    """Tests for merge_stock function."""

    def test_normal_merge(self):
        """Test merging two fungible stock lots."""
        system, stock1_id, stock2_id = make_system_with_two_stocks(qty1=100, qty2=50)

        # Merge stock2 into stock1
        result_id = merge_stock(system, stock1_id, stock2_id)

        # Check result is the keep_id
        assert result_id == stock1_id

        # Check merged quantity
        merged_stock = system.state.stocks[stock1_id]
        assert merged_stock.quantity == 150

        # Check removed stock is gone
        assert stock2_id not in system.state.stocks

        # Check agent's stock_ids updated
        agent = system.state.agents["F1"]
        assert stock1_id in agent.stock_ids
        assert stock2_id not in agent.stock_ids

        # Check event logged
        events = [e for e in system.state.events if e.get("kind") == "StockMerged"]
        assert len(events) == 1
        assert events[0]["keep_id"] == stock1_id
        assert events[0]["remove_id"] == stock2_id
        assert events[0]["final_qty"] == 150
        assert events[0]["keep_qty"] == 100
        assert events[0]["merged_qty"] == 50

    def test_merge_different_sku_raises_error(self):
        """Test that merging different SKUs raises ValidationError."""
        system = System()
        with system.setup():
            f1 = Firm(id="F1", name="Firm1", kind="firm")
            system.add_agent(f1)

        stock1_id = system.create_stock("F1", sku="WIDGET", quantity=100, unit_price=Decimal("10.00"))
        stock2_id = system.create_stock("F1", sku="GADGET", quantity=50, unit_price=Decimal("10.00"))

        with pytest.raises(ValidationError, match="not fungible"):
            merge_stock(system, stock1_id, stock2_id)

    def test_merge_different_price_raises_error(self):
        """Test that merging different unit prices raises ValidationError."""
        system = System()
        with system.setup():
            f1 = Firm(id="F1", name="Firm1", kind="firm")
            system.add_agent(f1)

        stock1_id = system.create_stock("F1", sku="WIDGET", quantity=100, unit_price=Decimal("10.00"))
        stock2_id = system.create_stock("F1", sku="WIDGET", quantity=50, unit_price=Decimal("15.00"))

        with pytest.raises(ValidationError, match="not fungible"):
            merge_stock(system, stock1_id, stock2_id)

    def test_merge_different_owner_raises_error(self):
        """Test that merging different owners raises ValidationError."""
        system = System()
        with system.setup():
            f1 = Firm(id="F1", name="Firm1", kind="firm")
            f2 = Firm(id="F2", name="Firm2", kind="firm")
            system.add_agent(f1)
            system.add_agent(f2)

        stock1_id = system.create_stock("F1", sku="WIDGET", quantity=100, unit_price=Decimal("10.00"))
        stock2_id = system.create_stock("F2", sku="WIDGET", quantity=50, unit_price=Decimal("10.00"))

        with pytest.raises(ValidationError, match="not fungible"):
            merge_stock(system, stock1_id, stock2_id)

    def test_merge_preserves_keep_stock_properties(self):
        """Test that merge preserves the keep stock's properties."""
        system, stock1_id, stock2_id = make_system_with_two_stocks(
            sku="GADGET",
            qty1=100,
            qty2=50,
            price=Decimal("25.50")
        )

        original_stock1 = system.state.stocks[stock1_id]
        original_sku = original_stock1.sku
        original_price = original_stock1.unit_price
        original_owner = original_stock1.owner_id

        merge_stock(system, stock1_id, stock2_id)

        merged_stock = system.state.stocks[stock1_id]
        assert merged_stock.sku == original_sku
        assert merged_stock.unit_price == original_price
        assert merged_stock.owner_id == original_owner


class TestConsumeStock:
    """Tests for consume_stock function."""

    def test_partial_consumption(self):
        """Test partially consuming a stock lot."""
        system, stock_id = make_system_with_stock(quantity=100)

        # Consume 30 units
        consume_stock(system, stock_id, 30)

        # Check quantity reduced
        stock = system.state.stocks[stock_id]
        assert stock.quantity == 70

        # Check stock still exists
        assert stock_id in system.state.stocks
        assert stock_id in system.state.agents["F1"].stock_ids

        # Check event logged
        events = [e for e in system.state.events if e.get("kind") == "StockConsumed"]
        assert len(events) == 1
        assert events[0]["stock_id"] == stock_id
        assert events[0]["qty"] == 30
        assert events[0]["remaining"] == 70
        assert events[0]["complete"] == False

    def test_complete_consumption(self):
        """Test completely consuming a stock lot."""
        system, stock_id = make_system_with_stock(quantity=100)

        # Consume all units
        consume_stock(system, stock_id, 100)

        # Check stock removed
        assert stock_id not in system.state.stocks
        assert stock_id not in system.state.agents["F1"].stock_ids

        # Check event logged
        events = [e for e in system.state.events if e.get("kind") == "StockConsumed"]
        assert len(events) == 1
        assert events[0]["stock_id"] == stock_id
        assert events[0]["qty"] == 100
        assert events[0]["complete"] == True

    def test_consume_zero_quantity_raises_error(self):
        """Test that consuming zero quantity raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid consumption quantity"):
            consume_stock(system, stock_id, 0)

    def test_consume_negative_quantity_raises_error(self):
        """Test that consuming negative quantity raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid consumption quantity"):
            consume_stock(system, stock_id, -10)

    def test_consume_exceeding_quantity_raises_error(self):
        """Test that consuming more than available raises ValidationError."""
        system, stock_id = make_system_with_stock(quantity=100)

        with pytest.raises(ValidationError, match="Invalid consumption quantity"):
            consume_stock(system, stock_id, 150)

    def test_multiple_partial_consumptions(self):
        """Test multiple partial consumptions in sequence."""
        system, stock_id = make_system_with_stock(quantity=100)

        # First consumption
        consume_stock(system, stock_id, 20)
        assert system.state.stocks[stock_id].quantity == 80

        # Second consumption
        consume_stock(system, stock_id, 30)
        assert system.state.stocks[stock_id].quantity == 50

        # Third consumption
        consume_stock(system, stock_id, 50)
        assert stock_id not in system.state.stocks

        # Check all events logged
        events = [e for e in system.state.events if e.get("kind") == "StockConsumed"]
        assert len(events) == 3
