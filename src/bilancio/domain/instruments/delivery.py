from dataclasses import dataclass
from decimal import Decimal
from bilancio.domain.instruments.base import Instrument, InstrumentKind

@dataclass
class DeliveryObligation(Instrument):
    # amount = quantity promised (rename for clarity at call sites)
    sku: str
    unit_price: Decimal
    due_day: int

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.DELIVERY_OBLIGATION
        # Ensure unit_price is a Decimal (callers may pass float/int at runtime)
        if not isinstance(self.unit_price, Decimal):
            self.unit_price = Decimal(str(self.unit_price))  # type: ignore[unreachable]

    def is_financial(self) -> bool:
        # Shows up in balance analysis as non-financial (valued) obligation
        return False

    @property
    def valued_amount(self) -> Decimal:
        from decimal import Decimal as D
        return D(str(self.amount)) * self.unit_price
        
    def validate_type_invariants(self) -> None:
        # Standard bilateral instrument validation (holder != issuer)
        super().validate_type_invariants()
        assert self.unit_price >= 0, "unit_price must be non-negative"
        assert self.due_day >= 0, "due_day must be non-negative"