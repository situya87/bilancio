"""Inventory lot operations for clean-core scenario state."""

from __future__ import annotations

from typing import Any

from bilancio.engines.clean_core_types import CleanState, CleanStockLot


def move_stock_lot(
    state: CleanState,
    stock: CleanStockLot,
    from_agent: str,
    to_agent: str,
    quantity: int,
    *,
    log: Any,
) -> str:
    moving_id = stock.id
    if quantity < stock.quantity:
        original_quantity = stock.quantity
        moving_id = f"S_split_{stock.id}_{state.day}_{len(state.stocks)}"
        state.stocks[moving_id] = CleanStockLot(
            id=moving_id,
            owner=stock.owner,
            sku=stock.sku,
            quantity=quantity,
            unit_price=stock.unit_price,
        )
        stock.quantity -= quantity
        log(
            "StockSplit",
            original_id=stock.id,
            new_id=moving_id,
            sku=stock.sku,
            original_qty=original_quantity,
            split_qty=quantity,
            remaining_qty=stock.quantity,
        )

    moving_stock = state.stocks[moving_id]
    if moving_stock.owner != from_agent:
        raise ValueError("Stock owner mismatch")
    moving_stock.owner = to_agent
    return moving_id


def first_stock_lot_by_sku(
    state: CleanState,
    owner: str,
    sku: str,
) -> CleanStockLot | None:
    for stock in state.stocks.values():
        if stock.owner == owner and stock.sku == sku:
            return stock
    return None


def deliver_stock_for_obligation(
    state: CleanState,
    debtor: str,
    creditor: str,
    sku: str,
    quantity: int,
) -> int:
    available = sorted(
        (
            stock
            for stock in state.stocks.values()
            if stock.owner == debtor and stock.sku == sku and stock.quantity > 0
        ),
        key=lambda stock: stock.id,
    )
    if not available:
        return 0

    total_available = sum(stock.quantity for stock in available)
    deliver_quantity = min(quantity, total_available)
    remaining = deliver_quantity
    for stock in available:
        if remaining == 0:
            break
        transfer_qty = min(remaining, stock.quantity)
        moving_id = move_stock_lot(state, stock, debtor, creditor, transfer_qty, log=state.log)
        moving_stock = state.stocks[moving_id]
        state.log(
            "StockTransferred",
            frm=debtor,
            to=creditor,
            stock_id=moving_id,
            sku=moving_stock.sku,
            qty=moving_stock.quantity,
        )
        remaining -= transfer_qty
    return deliver_quantity
