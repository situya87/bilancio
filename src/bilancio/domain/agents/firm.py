"""Firm agent representing companies, manufacturers, and other business entities."""

from dataclasses import dataclass

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class Firm(Agent):
    """
    A firm/company agent that can:
    - Hold and transfer cash
    - Issue and receive payables
    - Own and transfer stock inventory
    - Create and settle delivery obligations
    - Participate in economic transactions

    This represents any business entity that isn't a bank or financial institution.
    Examples: manufacturers, trading companies, service providers.
    """

    def __post_init__(self) -> None:
        """Ensure the agent kind is set to 'firm'."""
        if self.kind != AgentKind.FIRM:
            self.kind = AgentKind.FIRM
