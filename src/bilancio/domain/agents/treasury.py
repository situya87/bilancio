from dataclasses import dataclass

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class Treasury(Agent):
    def __post_init__(self) -> None:
        self.kind = AgentKind.TREASURY
