from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bilancio.core.ids import AgentId, InstrId


class InstrumentKind(str, Enum):
    """Enumeration of instrument/contract types.

    Using str mixin ensures InstrumentKind values work as dict keys,
    compare equal to their string values, and are JSON-serializable.
    """
    CASH = "cash"
    BANK_DEPOSIT = "bank_deposit"
    RESERVE_DEPOSIT = "reserve_deposit"
    PAYABLE = "payable"
    CB_LOAN = "cb_loan"
    NON_BANK_LOAN = "non_bank_loan"
    DELIVERY_OBLIGATION = "delivery_obligation"

    def __str__(self) -> str:
        return self.value


@dataclass
class Instrument:
    id: InstrId
    kind: InstrumentKind
    amount: int                    # minor units
    denom: str
    asset_holder_id: AgentId
    liability_issuer_id: AgentId
    due_day: int | None = None

    @property
    def effective_creditor(self) -> AgentId:
        return self.asset_holder_id

    def is_financial(self) -> bool:  # override if needed
        return True

    def validate_type_invariants(self) -> None:
        assert self.amount >= 0, "amount must be non-negative"
        assert self.asset_holder_id != self.liability_issuer_id, "self-counterparty forbidden"
