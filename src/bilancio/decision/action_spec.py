"""Action specifications for declarative agent behavior configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionDef:
    """One action this agent kind can perform."""
    action: str          # "settle", "sell_ticket", "buy_ticket", "borrow", "lend", "rate"
    phase: str           # "B2_Settlement", "B_Dealer", "B_Lending", "B_Rating"
    strategy: str | None = None  # strategy name, e.g. "liquidity_driven_seller"
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec:
    """Complete behavioral spec for one agent kind (or specific agents)."""
    kind: str                              # "household", "non_bank_lender", etc.
    actions: tuple[ActionDef, ...]         # what it can do (tuple for frozen)
    profile_type: str | None = None        # "trader", "lender", "vbt", "rating"
    profile_params: dict[str, Any] = field(default_factory=dict)
    information: str = "omniscient"        # preset name: "omniscient", "realistic", "blind"
    information_overrides: dict[str, str] = field(default_factory=dict)
    agent_ids: tuple[str, ...] | None = None  # None = all agents of this kind


# Valid action names
VALID_ACTIONS = frozenset({
    "settle", "sell_ticket", "buy_ticket", "borrow", "lend", "rate",
})

# Valid phase names
VALID_PHASES = frozenset({
    "B2_Settlement", "B_Dealer", "B_Lending", "B_Rating",
})

# Strategy registry: maps strategy name → class
STRATEGY_REGISTRY: dict[str, type] = {}


def _populate_strategy_registry() -> None:
    """Lazily populate the strategy registry to avoid circular imports."""
    if STRATEGY_REGISTRY:
        return
    from bilancio.decision.intentions import LiquidityDrivenSeller, SurplusBuyer
    from bilancio.decision.protocols import LinearPricer
    STRATEGY_REGISTRY.update({
        "liquidity_driven_seller": LiquidityDrivenSeller,
        "surplus_buyer": SurplusBuyer,
        "linear_pricer": LinearPricer,
    })


def resolve_strategy(name: str | None, params: dict[str, Any] | None = None) -> Any:
    """Resolve a strategy name to an instance.

    Args:
        name: Strategy name from the registry, or None for no strategy.
        params: Optional constructor parameters.

    Returns:
        Strategy instance, or None if name is None.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    if name is None:
        return None
    _populate_strategy_registry()
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    return cls(**(params or {}))
