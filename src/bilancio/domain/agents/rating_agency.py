"""
Rating agency agent for information production.

A RatingAgency observes system participants, estimates default probabilities,
and publishes ratings for other agents to consume. It holds no financial
instruments and has no means of payment — it is a pure information utility.
"""
from dataclasses import dataclass, field

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class RatingAgency(Agent):
    """
    Rating agency that produces default probability estimates.

    The agency observes system participants through its InformationProfile,
    computes credit ratings (continuous default probabilities), and publishes
    them to a system-wide rating registry consumed by lenders and traders.

    Attributes inherited from Agent:
        id: Unique identifier (e.g., "rating_agency")
        name: Display name (e.g., "Rating Agency")
        kind: Always "rating_agency"
        asset_ids: Empty — no instruments held
        liability_ids: Empty — no instruments issued
        stock_ids: Empty — no physical goods
        defaulted: Always False (cannot default)
    """

    kind: str = field(default=AgentKind.RATING_AGENCY, init=False)
