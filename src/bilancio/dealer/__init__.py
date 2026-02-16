"""
Dealer module for the Kalecki ring simulation.

This module implements a secondary market dealer system that provides liquidity
to ring agents through market-making. The dealer quotes bid/ask prices and
absorbs customer flow, routing to VBTs (value-based traders) when capacity
is exhausted.

Components:
- models: Data models (Ticket, DealerState, VBTState, TraderState)
- kernel: L1 dealer pricing kernel
- assertions: C1-C6 programmatic invariant checks
- events: Event logging system
- trading: Trade execution logic (Events 1-2, 9-10)
- simulation: Dealer ring event loop orchestrator

References:
- Specification of the Dealer Module for the Kalecki Ring (Bilancio Simulation)
- Examples Document with programmatic assertions
"""

from .assertions import (
    EPSILON_CASH,
    EPSILON_QTY,
    assert_c1_double_entry,
    assert_c2_quote_bounds,
    assert_c3_feasibility,
    assert_c4_passthrough_invariant,
    assert_c5_equity_basis,
    assert_c6_anchor_timing,
    run_all_assertions,
)
from .events import EventLog
from .kernel import (
    M_MIN,
    ExecutionResult,
    KernelParams,
    can_interior_buy,
    can_interior_sell,
    recompute_dealer_state,
)
from .models import (
    DEFAULT_BUCKETS,
    BucketConfig,
    DealerState,
    Ticket,
    TicketId,
    TraderState,
    VBTState,
)
from .report import (
    export_dealer_ring_html,
    generate_dealer_ring_html,
)
from .simulation import (
    DaySnapshot,
    DealerRingConfig,
    DealerRingSimulation,
)
from .trading import TradeExecutor

__all__ = [
    # Models
    "TicketId",
    "BucketConfig",
    "DEFAULT_BUCKETS",
    "Ticket",
    "DealerState",
    "VBTState",
    "TraderState",
    # Kernel
    "M_MIN",
    "KernelParams",
    "ExecutionResult",
    "recompute_dealer_state",
    "can_interior_buy",
    "can_interior_sell",
    # Assertions
    "EPSILON_CASH",
    "EPSILON_QTY",
    "assert_c1_double_entry",
    "assert_c2_quote_bounds",
    "assert_c3_feasibility",
    "assert_c4_passthrough_invariant",
    "assert_c5_equity_basis",
    "assert_c6_anchor_timing",
    "run_all_assertions",
    # Events
    "EventLog",
    # Trading
    "TradeExecutor",
    # Simulation
    "DaySnapshot",
    "DealerRingConfig",
    "DealerRingSimulation",
    # Report
    "generate_dealer_ring_html",
    "export_dealer_ring_html",
]
