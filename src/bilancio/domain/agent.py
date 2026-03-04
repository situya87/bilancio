"""Agent domain model.

New Agent Type Checklist
========================
When adding a new AgentKind, define these aspects in the plan:

1. Instruments — What it holds (assets) and issues (liabilities).
2. Means of Payment — What it uses to settle (mop_rank in policy.py).
3. Decision-Making Model — Behavioral model with tunable parameters
   (profile dataclass like TraderProfile), not a hard-coded formula.
4. Information Model — What it can observe, how it updates beliefs,
   whether it has its own risk assessor, observability friction, learning.
5. Capitalization — How it gets initial resources, what share of system.
6. Timing / Phase — When it acts in the daily simulation cycle.
7. Failure Mode — What happens when it defaults, cascade risk.
8. Interactions — Which agent types it transacts with, bilateral constraints.
"""

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
    NON_BANK_LENDER = "non_bank_lender"
    RATING_AGENCY = "rating_agency"

    def __str__(self) -> str:
        return self.value


@dataclass
class Agent:
    id: AgentId
    name: str
    kind: str  # Still accepts str for backward compatibility
    asset_ids: set[InstrId] = field(default_factory=set)
    liability_ids: set[InstrId] = field(default_factory=set)
    stock_ids: set[InstrId] = field(default_factory=set)
    defaulted: bool = False
    jurisdiction_id: str | None = None
