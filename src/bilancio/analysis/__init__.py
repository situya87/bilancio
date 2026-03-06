"""Analysis package for bilancio."""

from __future__ import annotations

from importlib import import_module

_MODULE_EXPORTS = {
    "bilancio.analysis.beliefs": [
        "BeliefPoint",
        "CalibrationBucket",
        "EstimateSummary",
        "belief_trajectory",
        "belief_vs_reality",
        "estimate_summary",
        "export_estimates_jsonl",
    ],
    "bilancio.analysis.contagion": [
        "DefaultRecord",
        "classify_defaults",
        "contagion_by_day",
        "default_counts_by_type",
        "default_dependency_graph",
        "time_to_contagion",
    ],
    "bilancio.analysis.credit_creation": [
        "credit_created_by_type",
        "credit_creation_by_day",
        "credit_destroyed_by_type",
        "credit_destruction_by_day",
        "net_credit_impulse",
    ],
    "bilancio.analysis.dealer_usage_summary": [
        "build_dealer_usage_by_run",
        "run_dealer_usage_analysis",
    ],
    "bilancio.analysis.funding_chains": [
        "cash_inflows_by_source",
        "cash_outflows_by_type",
        "funding_mix",
        "liquidity_providers",
    ],
    "bilancio.analysis.intermediary_frontier": [
        "FrontierArtifact",
        "FrontierPair",
        "RunOutcome",
        "build_frontier_artifact",
        "build_frontier_pairs",
        "compute_loss_floor_table",
        "discover_run_dirs",
        "load_run_outcomes",
        "summarize_frontier_pairs",
        "write_frontier_artifact",
    ],
    "bilancio.analysis.mechanism_activity": [
        "run_mechanism_activity_analysis",
    ],
    "bilancio.analysis.metrics_computer": [
        "MetricsBundle",
        "MetricsComputer",
    ],
    "bilancio.analysis.network_analysis": [
        "betweenness_centrality",
        "build_obligation_adjacency",
        "node_degree",
        "systemic_importance",
        "weighted_degree",
    ],
    "bilancio.analysis.post_sweep": [
        "run_post_sweep_analysis",
    ],
    "bilancio.analysis.pricing_analysis": [
        "average_price_ratio_by_day",
        "bid_ask_spread_by_day",
        "fire_sale_indicator",
        "price_discovery_speed",
        "trade_prices_by_day",
        "trade_volume_by_day",
    ],
    "bilancio.analysis.strategy_outcomes": [
        "build_strategy_outcomes_by_run",
        "build_strategy_outcomes_overall",
        "run_strategy_analysis",
    ],
}

_ATTRIBUTE_TO_MODULE = {
    attribute: module_name
    for module_name, attributes in _MODULE_EXPORTS.items()
    for attribute in attributes
}

__all__ = [
    # beliefs
    "BeliefPoint",
    "CalibrationBucket",
    "EstimateSummary",
    "belief_trajectory",
    "belief_vs_reality",
    "estimate_summary",
    "export_estimates_jsonl",
    # contagion
    "DefaultRecord",
    "classify_defaults",
    "contagion_by_day",
    "default_counts_by_type",
    "default_dependency_graph",
    "time_to_contagion",
    # credit creation
    "credit_created_by_type",
    "credit_creation_by_day",
    "credit_destroyed_by_type",
    "credit_destruction_by_day",
    "net_credit_impulse",
    # dealer usage
    "build_dealer_usage_by_run",
    "run_dealer_usage_analysis",
    # intermediary frontier
    "FrontierArtifact",
    "FrontierPair",
    "RunOutcome",
    "build_frontier_artifact",
    "build_frontier_pairs",
    "compute_loss_floor_table",
    "discover_run_dirs",
    "load_run_outcomes",
    "summarize_frontier_pairs",
    "write_frontier_artifact",
    # mechanism activity
    "run_mechanism_activity_analysis",
    # funding chains
    "cash_inflows_by_source",
    "cash_outflows_by_type",
    "funding_mix",
    "liquidity_providers",
    # metrics
    "MetricsBundle",
    "MetricsComputer",
    # network analysis
    "betweenness_centrality",
    "build_obligation_adjacency",
    "node_degree",
    "systemic_importance",
    "weighted_degree",
    # pricing analysis
    "average_price_ratio_by_day",
    "bid_ask_spread_by_day",
    "fire_sale_indicator",
    "price_discovery_speed",
    "trade_prices_by_day",
    "trade_volume_by_day",
    # post-sweep analysis
    "run_post_sweep_analysis",
    # strategy outcomes
    "build_strategy_outcomes_by_run",
    "build_strategy_outcomes_overall",
    "run_strategy_analysis",
]


def __getattr__(name: str) -> object:
    """Load optional analysis exports on first access."""
    module_name = _ATTRIBUTE_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
