# non-financial; not a liability of anyone
from dataclasses import dataclass
from decimal import Decimal

from bilancio.core.ids import AgentId, InstrId

StockId = InstrId  # reuse ID machinery


@dataclass
class StockLot:
    id: StockId
    kind: str  # fixed: "stock_lot"
    sku: str
    quantity: int
    unit_price: Decimal
    owner_id: AgentId
    divisible: bool = True

    def __post_init__(self) -> None:
        self.kind = "stock_lot"
        # Ensure unit_price is a Decimal (callers may pass float/int at runtime)
        if not isinstance(self.unit_price, Decimal):
            self.unit_price = Decimal(str(self.unit_price))  # type: ignore[unreachable]

    @property
    def value(self) -> Decimal:
        return Decimal(str(self.quantity)) * self.unit_price
