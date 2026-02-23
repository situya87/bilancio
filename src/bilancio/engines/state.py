"""Simulation state container.

The ``State`` dataclass holds the full mutable state of a bilancio
simulation: agents, contracts, stocks, events, indices, and configuration
flags.  It is a pure data container with no behaviour — all mutations go
through :class:`bilancio.engines.system.System`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bilancio.core.ids import AgentId, InstrId
from bilancio.domain.agent import Agent
from bilancio.domain.goods import StockLot
from bilancio.domain.instruments.base import Instrument


@dataclass
class State:
    agents: dict[AgentId, Agent] = field(default_factory=dict)
    contracts: dict[InstrId, Instrument] = field(default_factory=dict)
    stocks: dict[InstrId, StockLot] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    events_by_day: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    day: int = 0
    cb_cash_outstanding: int = 0
    cb_reserves_outstanding: int = 0
    cb_loans_outstanding: int = 0  # Total CB loans to banks (principal)
    cb_loans_created_count: int = 0  # Total CB loans issued (count)
    cb_interest_total_paid: int = 0  # Cumulative interest paid to CB (reserves destroyed)
    cb_reserves_initial: int = 0  # Reserves at simulation start (set by run_until_stable)
    phase: str = "simulation"
    # Aliases for created contracts (alias -> contract_id)
    aliases: dict[str, str] = field(default_factory=dict)
    # Scheduled actions to run at Phase B1 by day (day -> list of action dicts)
    scheduled_actions_by_day: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    # Track agents that have defaulted and been expelled from future activity
    defaulted_agent_ids: set[AgentId] = field(default_factory=set)
    # Plan 024: Enable continuous rollover of settled payables
    rollover_enabled: bool = False
    # Index contracts by due_day for fast lookup (preserves insertion order)
    contracts_by_due_day: dict[int, list[str]] = field(default_factory=dict)
    dealer_subsystem: Any = None
    banking_subsystem: Any = None
    jurisdictions: dict[str, Any] = field(default_factory=dict)  # str -> Jurisdiction
    fx_market: Any = None  # FXMarket instance
    lender_config: Any = None
    rating_config: Any = None
    rating_registry: dict[str, Any] = field(default_factory=dict)
    estimate_log: list[Any] = field(default_factory=list)
    estimate_logging_enabled: bool = False
    cb_lending_frozen: bool = False
