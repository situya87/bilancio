from dataclasses import dataclass

from .base import Instrument, InstrumentKind


@dataclass
class Payable(Instrument):
    """A payable instrument representing a credit obligation.

    Fields:
        due_day: The day when the payable is due for settlement
        holder_id: Optional secondary market holder. If set, this agent currently
                   holds the payable and should receive settlement payment.
                   If None, the original creditor (asset_holder_id) receives payment.
        maturity_distance: Original maturity distance (ΔT) in days. Used for rollover
                          to create new payables with same ΔT after settlement.

    Properties:
        effective_creditor: Returns the agent ID who should receive settlement
                           payment - either the secondary market holder_id or the
                           original asset_holder_id.
    """
    due_day: int | None = None
    holder_id: str | None = None
    maturity_distance: int | None = None  # Plan 024: for continuous rollover

    @property
    def effective_creditor(self) -> str:
        """Return the agent who should receive settlement payment.

        Returns holder_id if transferred in secondary market, otherwise
        returns the original asset_holder_id.
        """
        return self.holder_id if self.holder_id else self.asset_holder_id

    def __post_init__(self) -> None:
        self.kind = InstrumentKind.PAYABLE

    def validate_type_invariants(self) -> None:
        super().validate_type_invariants()
        assert self.due_day is not None and self.due_day >= 0, "payable must have due_day"
