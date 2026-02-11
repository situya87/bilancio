from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bilancio.core.ids import AgentId, InstrId


class AgentKind(str, Enum):
    """Enumeration of agent types in the financial system.

    Using str mixin ensures AgentKind values work as dict keys,
    compare equal to their string values, and are JSON-serializable.
    """
    CENTRAL_BANK = "central_bank"
    BANK = "bank"
    HOUSEHOLD = "household"
    TREASURY = "treasury"
    FIRM = "firm"
    INVESTMENT_FUND = "investment_fund"
    INSURANCE_COMPANY = "insurance_company"
    DEALER = "dealer"
    VBT = "vbt"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Agent:
    id: AgentId
    name: str
    kind: str  # Still accepts str for backward compatibility
    asset_ids: list[InstrId] = field(default_factory=list)
    liability_ids: list[InstrId] = field(default_factory=list)
    stock_ids: list[InstrId] = field(default_factory=list)
    defaulted: bool = False
