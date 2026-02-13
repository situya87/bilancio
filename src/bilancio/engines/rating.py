"""Rating engine: rating agency methodology and execution.

Implements the rating phase where a rating agency observes system
participants, estimates default probabilities, and publishes them to
a system-wide registry consumed by lenders and traders.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System
    from bilancio.decision.profiles import RatingProfile
    from bilancio.information.profile import InformationProfile

logger = logging.getLogger(__name__)


@dataclass
class RatingConfig:
    """Configuration for the rating agency phase."""
    rating_profile: Optional["RatingProfile"] = None
    information_profile: Optional["InformationProfile"] = None


def run_rating_phase(
    system: "System",
    current_day: int,
    rating_config: Optional[RatingConfig] = None,
) -> List[Dict[str, Any]]:
    """Run the rating agency phase.

    The rating agency:
    1. Selects a subset of eligible agents (HOUSEHOLD/FIRM) based on coverage_fraction
    2. Computes a default probability for each selected agent
    3. Merges results into the rating_registry (carry-forward for unrated agents)
    4. Returns rating events

    Args:
        system: The simulation system
        current_day: Current simulation day
        rating_config: Rating configuration (uses defaults if None)

    Returns:
        List of rating event dicts
    """
    from bilancio.domain.agent import AgentKind
    from bilancio.decision.profiles import RatingProfile

    config = rating_config or RatingConfig()
    profile = config.rating_profile or RatingProfile()
    events: List[Dict[str, Any]] = []

    # Find the rating agency
    agency_id = None
    for agent_id, agent in system.state.agents.items():
        if agent.kind == AgentKind.RATING_AGENCY and not agent.defaulted:
            agency_id = agent_id
            break

    if agency_id is None:
        return events

    # Build information service if profile configured
    info = None
    if config.information_profile is not None:
        from bilancio.information.service import InformationService
        info = InformationService(
            system, config.information_profile, observer_id=agency_id,
        )

    # Collect eligible agents (HOUSEHOLD and FIRM, non-defaulted)
    eligible: List[str] = []
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            continue
        if agent.kind in (AgentKind.HOUSEHOLD, AgentKind.FIRM):
            eligible.append(agent_id)

    if not eligible:
        return events

    # Select subset based on coverage_fraction using deterministic RNG
    n_to_rate = max(1, int(len(eligible) * float(profile.coverage_fraction)))
    rng = random.Random(current_day * 31337 + len(eligible))
    selected = rng.sample(eligible, min(n_to_rate, len(eligible)))

    # Ensure rating_registry exists
    registry = getattr(system.state, 'rating_registry', None)
    if registry is None:
        system.state.rating_registry = {}  # type: ignore[attr-defined]
        registry = system.state.rating_registry  # type: ignore[attr-defined]

    # Rate each selected agent
    ratings_published: Dict[str, str] = {}
    for agent_id in selected:
        agent = system.state.agents[agent_id]
        if agent.defaulted:
            p_default = Decimal("1.0")
        else:
            p_default = _compute_rating(info, agent_id, current_day, profile, system)

        registry[agent_id] = p_default
        ratings_published[agent_id] = str(p_default)

    # Mark defaulted agents as 1.0 in registry
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            registry[agent_id] = Decimal("1.0")

    events.append({
        "kind": "RatingsPublished",
        "day": current_day,
        "agency_id": agency_id,
        "n_rated": len(ratings_published),
        "n_eligible": len(eligible),
        "ratings": ratings_published,
    })

    logger.debug(
        "Rating agency %s rated %d/%d agents on day %d",
        agency_id, len(ratings_published), len(eligible), current_day,
    )

    return events


def _compute_rating(
    info: Any,
    agent_id: str,
    current_day: int,
    profile: "RatingProfile",
    system: "System",
) -> Decimal:
    """Compute a default probability rating for a single agent.

    Combines:
    - Balance sheet component: maps coverage ratio to default probability
    - History component: observed/estimated default probability
    - Weighted combination + conservatism_bias, clamped to [0.01, 0.99]

    Args:
        info: InformationService or None (None = omniscient)
        agent_id: Agent to rate
        current_day: Current day
        profile: Rating methodology parameters
        system: The simulation system

    Returns:
        Default probability estimate as Decimal
    """
    from bilancio.decision.profiles import RatingProfile

    # ── Balance sheet component ──
    if info is not None:
        net_worth = info.get_counterparty_net_worth(agent_id, current_day)
        obligations = info.get_counterparty_obligations(agent_id, current_day, profile.lookback_window)
    else:
        net_worth = _raw_net_worth(system, agent_id)
        obligations = _raw_total_liabilities(system, agent_id)

    if net_worth is None or obligations is None:
        bs_score = profile.no_data_prior
    elif obligations == 0:
        bs_score = Decimal("0.02")
    else:
        coverage = Decimal(str(net_worth)) / Decimal(str(max(obligations, 1)))
        if coverage >= 2:
            bs_score = Decimal("0.02")
        elif coverage >= Decimal("1.5"):
            bs_score = Decimal("0.05")
        elif coverage >= 1:
            # Linear interpolation: 1.0 -> 0.15, 1.5 -> 0.05
            t = (coverage - Decimal("1")) / Decimal("0.5")
            bs_score = Decimal("0.15") - t * Decimal("0.10")
        elif coverage > 0:
            # Linear interpolation: 0 -> 0.55, 1.0 -> 0.15
            t = coverage
            bs_score = Decimal("0.55") - t * Decimal("0.40")
        else:
            bs_score = Decimal("0.60")

    # ── History component ──
    if info is not None:
        hist_prob = info.get_default_probability(agent_id, current_day)
        hist_score = hist_prob if hist_prob is not None else profile.no_data_prior
    else:
        # Omniscient: use raw default probs
        hist_score = _raw_default_prob_for_agent(system, agent_id)

    # ── Weighted combination ──
    total_weight = profile.balance_sheet_weight + profile.history_weight
    if total_weight > 0:
        combined = (
            profile.balance_sheet_weight * bs_score
            + profile.history_weight * hist_score
        ) / total_weight
    else:
        combined = profile.no_data_prior

    # Add conservatism bias and clamp
    result = combined + profile.conservatism_bias
    return max(Decimal("0.01"), min(Decimal("0.99"), result))


def _raw_net_worth(system: "System", agent_id: str) -> int:
    """Get raw net worth for an agent."""
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    total_assets = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None:
            total_assets += contract.amount
    total_liabilities = 0
    for cid in agent.liability_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None:
            total_liabilities += contract.amount
    return total_assets - total_liabilities


def _raw_total_liabilities(system: "System", agent_id: str) -> int:
    """Get total liabilities for an agent."""
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    total = 0
    for cid in agent.liability_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None:
            total += contract.amount
    return total


def _raw_default_prob_for_agent(system: "System", agent_id: str) -> Decimal:
    """Get raw default probability for a single agent.

    Uses dealer risk assessor if available, otherwise falls back to
    system-wide heuristic.
    """
    agent = system.state.agents.get(agent_id)
    if agent is None or agent.defaulted:
        return Decimal("1.0")

    # Try dealer risk assessor
    dealer_sub = system.state.dealer_subsystem
    if (
        dealer_sub is not None
        and hasattr(dealer_sub, "risk_assessor")
        and dealer_sub.risk_assessor is not None
    ):
        p = dealer_sub.risk_assessor.estimate_default_prob(agent_id)
        return Decimal(str(p)) if p is not None else Decimal("0.15")

    # Fallback: system-wide heuristic
    n_agents = len(system.state.agents)
    n_defaulted = len(system.state.defaulted_agent_ids)
    base_rate = Decimal(str(n_defaulted / max(n_agents, 1)))
    return max(Decimal("0.01"), min(Decimal("0.99"), base_rate + Decimal("0.05")))
