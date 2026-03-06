"""Visualization utilities for bilancio analysis outputs."""

from __future__ import annotations

from importlib import import_module

_MODULE_EXPORTS = {
    "bilancio.analysis.visualization.common": [
        "RICH_AVAILABLE",
        "BalanceRow",
        "RenderableType",
        "TAccount",
        "parse_day_from_maturity",
    ],
    "bilancio.analysis.visualization.balances": [
        "build_t_account_rows",
        "display_agent_balance_from_balance",
        "display_agent_balance_table",
        "display_agent_balance_table_renderable",
        "display_agent_t_account",
        "display_agent_t_account_renderable",
        "display_multiple_agent_balances",
        "display_multiple_agent_balances_renderable",
    ],
    "bilancio.analysis.visualization.events": [
        "display_events",
        "display_events_for_day",
        "display_events_for_day_renderable",
        "display_events_renderable",
        "display_events_table",
        "display_events_table_renderable",
    ],
    "bilancio.analysis.visualization.phases": [
        "display_events_tables_by_phase_renderables",
    ],
    "bilancio.analysis.visualization.run_comparison": [
        "RunComparison",
        "comparisons_to_dataframe",
        "generate_comparison_html",
        "load_job_comparison_data",
        "quick_visualize",
    ],
}

_ATTRIBUTE_TO_MODULE = {
    attribute: module_name
    for module_name, attributes in _MODULE_EXPORTS.items()
    for attribute in attributes
}

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


def __getattr__(name: str) -> object:
    """Load optional visualization exports on first access."""
    module_name = _ATTRIBUTE_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
