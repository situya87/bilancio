"""Stock operations primitives - separate from financial instrument operations."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.core.errors import ValidationError
from bilancio.core.ids import InstrId, new_id
from bilancio.domain.goods import StockLot

if TYPE_CHECKING:
    from bilancio.engines.system import System


def stock_fungible_key(lot: StockLot) -> tuple[str, str, str, Decimal]:
    """Return fungible key for stock lots. Include price so lots with different valuation don't merge."""
    return (lot.kind, lot.sku, lot.owner_id, lot.unit_price)


def split_stock(system: "System", stock_id: InstrId, quantity: int) -> InstrId:
    """Split a stock lot into two pieces. Returns ID of the new split piece."""
    stock = system.state.stocks[stock_id]

    if not stock.divisible:
        raise ValidationError("Stock lot is not divisible")
    if quantity <= 0 or quantity >= stock.quantity:
        raise ValidationError("Invalid split quantity")

    # Create new stock lot with split quantity
    new_id_val = new_id("S")
    new_stock = StockLot(
        id=new_id_val,
        kind="stock_lot",
        sku=stock.sku,
        quantity=quantity,
        unit_price=stock.unit_price,
        owner_id=stock.owner_id,
        divisible=stock.divisible,
    )

    # Update original stock quantity
    stock.quantity -= quantity

    # Register new stock
    system.state.stocks[new_id_val] = new_stock
    system.state.agents[stock.owner_id].stock_ids.add(new_id_val)

    system.log(
        "StockSplit",
        original_id=stock_id,
        new_id=new_id_val,
        sku=stock.sku,
        original_qty=stock.quantity + quantity,
        split_qty=quantity,
        remaining_qty=stock.quantity,
    )

    return new_id_val


def merge_stock(system: "System", keep_id: InstrId, remove_id: InstrId) -> InstrId:
    """Merge two stock lots. Returns the ID of the kept lot."""
    keep_stock = system.state.stocks[keep_id]
    remove_stock = system.state.stocks[remove_id]

    # Check if stocks are fungible
    if stock_fungible_key(keep_stock) != stock_fungible_key(remove_stock):
        raise ValidationError("Stock lots are not fungible - cannot merge")

    # Merge quantities
    original_qty = keep_stock.quantity
    keep_stock.quantity += remove_stock.quantity

    # Remove the merged stock
    del system.state.stocks[remove_id]
    system.state.agents[remove_stock.owner_id].stock_ids.remove(remove_id)

    system.log(
        "StockMerged",
        keep_id=keep_id,
        remove_id=remove_id,
        sku=keep_stock.sku,
        final_qty=keep_stock.quantity,
        keep_qty=original_qty,
        merged_qty=remove_stock.quantity,
    )

    return keep_id


def consume_stock(system: "System", stock_id: InstrId, quantity: int) -> None:
    """Consume (destroy) a portion of stock. For future use if needed."""
    stock = system.state.stocks[stock_id]

    if quantity <= 0 or quantity > stock.quantity:
        raise ValidationError("Invalid consumption quantity")

    if quantity == stock.quantity:
        # Complete consumption - remove the stock
        del system.state.stocks[stock_id]
        system.state.agents[stock.owner_id].stock_ids.remove(stock_id)
        system.log("StockConsumed", stock_id=stock_id, sku=stock.sku, qty=quantity, complete=True)
    else:
        # Partial consumption
        stock.quantity -= quantity
        system.log(
            "StockConsumed",
            stock_id=stock_id,
            sku=stock.sku,
            qty=quantity,
            remaining=stock.quantity,
            complete=False,
        )
