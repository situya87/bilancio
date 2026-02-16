"""Balance sheet visualization utilities for the bilancio system.

This module has been split into submodules for better organization:
- balances: Balance sheet and T-account display
- events: Event table formatting
- phases: Phase summary visualization
- common: Shared utilities and constants
"""

from __future__ import annotations

# Balance sheet functions
from bilancio.analysis.visualization.balances import (
    build_t_account_rows,
    display_agent_balance_from_balance,
    display_agent_balance_table,
    display_agent_balance_table_renderable,
    display_agent_t_account,
    display_agent_t_account_renderable,
    display_multiple_agent_balances,
    display_multiple_agent_balances_renderable,
)

# Re-export public API from submodules to maintain backward compatibility
# Common utilities and types
from bilancio.analysis.visualization.common import (
    RICH_AVAILABLE,
    BalanceRow,
    RenderableType,
    TAccount,
    parse_day_from_maturity,
)

# Event display functions
from bilancio.analysis.visualization.events import (
    display_events,
    display_events_for_day,
    display_events_for_day_renderable,
    display_events_renderable,
    display_events_table,
    display_events_table_renderable,
)

# Phase summary functions
from bilancio.analysis.visualization.phases import (
    display_events_tables_by_phase_renderables,
)

# Run comparison visualizations
from bilancio.analysis.visualization.run_comparison import (
    RunComparison,
    comparisons_to_dataframe,
    generate_comparison_html,
    load_job_comparison_data,
    quick_visualize,
)

__all__ = [
    # Common
    "RICH_AVAILABLE",
    "RenderableType",
    "BalanceRow",
    "TAccount",
    "parse_day_from_maturity",
    # Balance sheets
    "display_agent_balance_table",
    "display_agent_balance_from_balance",
    "display_multiple_agent_balances",
    "build_t_account_rows",
    "display_agent_t_account",
    "display_agent_t_account_renderable",
    "display_agent_balance_table_renderable",
    "display_multiple_agent_balances_renderable",
    # Events
    "display_events",
    "display_events_table",
    "display_events_table_renderable",
    "display_events_for_day",
    "display_events_renderable",
    "display_events_for_day_renderable",
    # Phases
    "display_events_tables_by_phase_renderables",
    # Run comparison
    "RunComparison",
    "load_job_comparison_data",
    "comparisons_to_dataframe",
    "generate_comparison_html",
    "quick_visualize",
]
