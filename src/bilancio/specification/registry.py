"""
Registry for tracking all agent and instrument specifications.

The registry ensures that when a new agent or instrument is added,
all required relationships are defined.
"""

from dataclasses import dataclass, field
from typing import Any

from .models import AgentSpec, InstrumentSpec


@dataclass
class SpecificationRegistry:
    """
    Central registry for all agent and instrument specifications.

    Provides methods to:
    - Register new agents and instruments
    - Check for missing relationships
    - Validate completeness of the system
    """

    agents: dict[str, AgentSpec] = field(default_factory=dict)
    instruments: dict[str, InstrumentSpec] = field(default_factory=dict)

    def register_agent(self, spec: AgentSpec) -> list[str]:
        """
        Register an agent specification.

        Returns list of warnings about missing instrument relations.
        """
        warnings = []

        # Check for missing instrument relations
        for instrument_name in self.instruments:
            if not spec.has_relation(instrument_name):
                warnings.append(
                    f"Agent '{spec.name}' missing relation to instrument '{instrument_name}'"
                )

        self.agents[spec.name] = spec
        return warnings

    def register_instrument(self, spec: InstrumentSpec) -> list[str]:
        """
        Register an instrument specification.

        Returns list of warnings about missing agent relations.
        """
        warnings = []

        # Check for missing agent relations
        for agent_name in self.agents:
            if not spec.has_relation(agent_name):
                warnings.append(
                    f"Instrument '{spec.name}' missing relation to agent '{agent_name}'"
                )

        # Check for missing instrument interactions
        for other_instrument in self.instruments:
            if other_instrument not in spec.instrument_interactions:
                warnings.append(
                    f"Instrument '{spec.name}' missing interaction with instrument '{other_instrument}'"
                )

        self.instruments[spec.name] = spec
        return warnings

    def get_agent(self, name: str) -> AgentSpec | None:
        """Get an agent specification by name."""
        return self.agents.get(name)

    def get_instrument(self, name: str) -> InstrumentSpec | None:
        """Get an instrument specification by name."""
        return self.instruments.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def list_instruments(self) -> list[str]:
        """List all registered instrument names."""
        return list(self.instruments.keys())

    def get_missing_agent_relations(self, agent_name: str) -> list[str]:
        """Get list of instruments missing relations for an agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return []

        missing = []
        for instrument_name in self.instruments:
            if not agent.has_relation(instrument_name):
                missing.append(instrument_name)
        return missing

    def get_missing_instrument_relations(self, instrument_name: str) -> list[str]:
        """Get list of agents missing relations for an instrument."""
        instrument = self.instruments.get(instrument_name)
        if not instrument:
            return []

        missing = []
        for agent_name in self.agents:
            if not instrument.has_relation(agent_name):
                missing.append(agent_name)
        return missing

    def get_relationship_matrix(self) -> dict[str, dict[str, str]]:
        """
        Generate a relationship matrix showing all agent-instrument pairs.

        Returns:
            Dict mapping agent_name -> instrument_name -> status
            where status is "defined", "missing", or the position type
        """
        matrix: dict[str, dict[str, str]] = {}

        for agent_name, agent in self.agents.items():
            matrix[agent_name] = {}
            for instrument_name in self.instruments:
                relation = agent.get_relation(instrument_name)
                if relation:
                    matrix[agent_name][instrument_name] = relation.position.value
                else:
                    matrix[agent_name][instrument_name] = "MISSING"

        return matrix

    def get_completeness_summary(self) -> dict[str, Any]:
        """
        Get a summary of specification completeness.

        Returns dict with:
        - total_agents: number of agents
        - total_instruments: number of instruments
        - total_pairs: total agent-instrument pairs needed
        - defined_pairs: number of pairs with relations defined
        - missing_pairs: list of (agent, instrument) pairs without relations
        """
        total_agents = len(self.agents)
        total_instruments = len(self.instruments)
        total_pairs = total_agents * total_instruments

        defined_pairs = 0
        missing_pairs = []

        for agent_name, agent in self.agents.items():
            for instrument_name in self.instruments:
                if agent.has_relation(instrument_name):
                    defined_pairs += 1
                else:
                    missing_pairs.append((agent_name, instrument_name))

        return {
            "total_agents": total_agents,
            "total_instruments": total_instruments,
            "total_pairs": total_pairs,
            "defined_pairs": defined_pairs,
            "missing_pairs": missing_pairs,
            "completeness_ratio": defined_pairs / total_pairs if total_pairs > 0 else 1.0,
        }
