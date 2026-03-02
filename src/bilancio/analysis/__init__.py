"""Analysis package for bilancio."""

from bilancio.analysis.beliefs import (
    BeliefPoint,
    CalibrationBucket,
    EstimateSummary,
    belief_trajectory,
    belief_vs_reality,
    estimate_summary,
    export_estimates_jsonl,
)
from bilancio.analysis.contagion import (
    DefaultRecord,
    classify_defaults,
    contagion_by_day,
    default_counts_by_type,
    default_dependency_graph,
    time_to_contagion,
)
from bilancio.analysis.credit_creation import (
    credit_created_by_type,
    credit_creation_by_day,
    credit_destroyed_by_type,
    credit_destruction_by_day,
    net_credit_impulse,
)
from bilancio.analysis.dealer_usage_summary import (
    build_dealer_usage_by_run,
    run_dealer_usage_analysis,
)
from bilancio.analysis.mechanism_activity import run_mechanism_activity_analysis
from bilancio.analysis.funding_chains import (
    cash_inflows_by_source,
    cash_outflows_by_type,
    funding_mix,
    liquidity_providers,
)
from bilancio.analysis.metrics_computer import (
    MetricsBundle,
    MetricsComputer,
)
from bilancio.analysis.network_analysis import (
    betweenness_centrality,
    build_obligation_adjacency,
    node_degree,
    systemic_importance,
    weighted_degree,
)
from bilancio.analysis.pricing_analysis import (
    average_price_ratio_by_day,
    bid_ask_spread_by_day,
    fire_sale_indicator,
    price_discovery_speed,
    trade_prices_by_day,
    trade_volume_by_day,
)
from bilancio.analysis.strategy_outcomes import (
    build_strategy_outcomes_by_run,
    build_strategy_outcomes_overall,
    run_strategy_analysis,
)

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
    # strategy outcomes
    "build_strategy_outcomes_by_run",
    "build_strategy_outcomes_overall",
    "run_strategy_analysis",
]
