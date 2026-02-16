"""
Validators for checking specification completeness.

These validators ensure that all required relationships and
specifications are properly defined before code is written.
"""

from dataclasses import dataclass, field
from typing import Any

from .models import (
    AgentRelation,
    AgentSpec,
    BalanceSheetPosition,
    InstrumentRelation,
    InstrumentSpec,
)
from .registry import SpecificationRegistry


@dataclass
class ValidationError:
    """A single validation error."""

    category: str  # e.g., "missing_relation", "incomplete_spec", "inconsistency"
    entity: str  # Agent or instrument name
    field: str  # Which field has the error
    message: str  # Human-readable description


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, category: str, entity: str, field: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(category, entity, field, message))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


def validate_agent_completeness(
    agent: AgentSpec,
    registry: SpecificationRegistry,
) -> ValidationResult:
    """
    Validate that an agent specification is complete.

    Checks:
    1. All required fields are present
    2. Relations exist for all instruments in the registry
    3. Each relation is internally consistent
    4. Decisions reference valid instruments
    5. Lifecycle is specified if agent can hold instruments
    """
    result = ValidationResult(is_valid=True)

    # 1. Check required fields
    if not agent.name:
        result.add_error("missing_field", agent.name or "unknown", "name", "Agent name is required")

    if not agent.description:
        result.add_error(
            "missing_field", agent.name, "description", "Agent description is required"
        )

    # 2. Check relations for all instruments
    for instrument_name in registry.list_instruments():
        if not agent.has_relation(instrument_name):
            result.add_error(
                "missing_relation",
                agent.name,
                f"instrument_relations[{instrument_name}]",
                f"Missing relation to instrument '{instrument_name}'",
            )
        else:
            # 3. Check relation is internally consistent
            relation = agent.get_relation(instrument_name)
            assert relation is not None
            is_complete, errors = relation.is_complete()
            for error in errors:
                result.add_error(
                    "incomplete_relation",
                    agent.name,
                    f"instrument_relations[{instrument_name}]",
                    error,
                )

    # 4. Check decisions reference valid instruments
    for decision in agent.decisions:
        is_complete, errors = decision.is_complete()
        for error in errors:
            result.add_error(
                "incomplete_decision", agent.name, f"decisions[{decision.name}]", error
            )

        # Check that instruments_involved exist
        for instrument_name in decision.instruments_involved:
            if instrument_name not in registry.instruments:
                result.add_warning(
                    f"Decision '{decision.name}' references unknown instrument '{instrument_name}'"
                )

    # 5. Check lifecycle if agent can hold any instruments
    can_hold_any = any(rel.can_hold for rel in agent.instrument_relations.values())
    if can_hold_any and not agent.lifecycle:
        result.add_warning(
            f"Agent '{agent.name}' can hold instruments but has no lifecycle specification"
        )

    # 6. Check banking consistency
    if agent.has_bank_account and not agent.bank_assignment_rule:
        result.add_warning(
            f"Agent '{agent.name}' has bank account but no assignment rule specified"
        )

    return result


def validate_instrument_completeness(
    instrument: InstrumentSpec,
    registry: SpecificationRegistry,
) -> ValidationResult:
    """
    Validate that an instrument specification is complete.

    Checks:
    1. All required fields are present
    2. Relations exist for all agents in the registry
    3. Each relation is internally consistent
    4. Lifecycle is fully specified
    5. Interactions with other instruments are defined
    """
    result = ValidationResult(is_valid=True)

    # 1. Check required fields
    if not instrument.name:
        result.add_error(
            "missing_field", instrument.name or "unknown", "name", "Instrument name is required"
        )

    if not instrument.description:
        result.add_error(
            "missing_field", instrument.name, "description", "Instrument description is required"
        )

    # 2. Check relations for all agents
    for agent_name in registry.list_agents():
        if not instrument.has_relation(agent_name):
            result.add_error(
                "missing_relation",
                instrument.name,
                f"agent_relations[{agent_name}]",
                f"Missing relation to agent '{agent_name}'",
            )
        else:
            # 3. Check relation is internally consistent
            relation = instrument.get_relation(agent_name)
            assert relation is not None
            is_complete, errors = relation.is_complete()
            for error in errors:
                result.add_error(
                    "incomplete_relation", instrument.name, f"agent_relations[{agent_name}]", error
                )

    # 4. Check lifecycle
    if not instrument.lifecycle:
        result.add_error(
            "missing_field", instrument.name, "lifecycle", "Instrument lifecycle is required"
        )
    else:
        is_complete, errors = instrument.lifecycle.is_complete()
        for error in errors:
            result.add_error("incomplete_lifecycle", instrument.name, "lifecycle", error)

    # 5. Check interactions with other instruments
    for other_name in registry.list_instruments():
        if other_name == instrument.name:
            continue  # Don't need self-interaction

        if other_name not in instrument.instrument_interactions:
            result.add_warning(
                f"Instrument '{instrument.name}' missing interaction with '{other_name}'"
            )
        else:
            interaction = instrument.instrument_interactions[other_name]
            is_complete, errors = interaction.is_complete()
            for error in errors:
                result.add_error(
                    "incomplete_interaction",
                    instrument.name,
                    f"instrument_interactions[{other_name}]",
                    error,
                )

    return result


def validate_all_relationships(registry: SpecificationRegistry) -> ValidationResult:
    """
    Validate that all agent-instrument relationships are consistent.

    Checks:
    1. Every agent has relations to every instrument
    2. Every instrument has relations to every agent
    3. Relations are symmetric (if agent can hold, instrument says agent can hold)
    4. All lifecycle events are accounted for
    """
    result = ValidationResult(is_valid=True)

    # 1 & 2: Check completeness (already done by individual validators)
    # Just verify the matrix is complete
    summary = registry.get_completeness_summary()

    for agent_name, instrument_name in summary["missing_pairs"]:
        result.add_error(
            "missing_relation",
            f"{agent_name}<->{instrument_name}",
            "relationship",
            f"No relationship defined between agent '{agent_name}' and instrument '{instrument_name}'",
        )

    # 3. Check symmetry
    for agent_name, agent in registry.agents.items():
        for instrument_name, instrument in registry.instruments.items():
            agent_rel = agent.get_relation(instrument_name)
            instrument_rel = instrument.get_relation(agent_name)

            if agent_rel and instrument_rel:
                # Check can_hold is consistent
                if agent_rel.can_hold != instrument_rel.can_hold:
                    result.add_error(
                        "inconsistency",
                        f"{agent_name}<->{instrument_name}",
                        "can_hold",
                        f"Agent says can_hold={agent_rel.can_hold}, instrument says can_hold={instrument_rel.can_hold}",
                    )

                # Check can_create/can_issue is consistent
                if agent_rel.can_create != instrument_rel.can_issue:
                    result.add_error(
                        "inconsistency",
                        f"{agent_name}<->{instrument_name}",
                        "can_create/can_issue",
                        f"Agent says can_create={agent_rel.can_create}, instrument says can_issue={instrument_rel.can_issue}",
                    )

                # Check position is consistent
                if agent_rel.position != instrument_rel.position:
                    result.add_error(
                        "inconsistency",
                        f"{agent_name}<->{instrument_name}",
                        "position",
                        f"Agent says position={agent_rel.position.value}, instrument says position={instrument_rel.position.value}",
                    )

    # 4. Validate individual specs
    for _agent_name, agent in registry.agents.items():
        agent_result = validate_agent_completeness(agent, registry)
        result.merge(agent_result)

    for _instrument_name, instrument in registry.instruments.items():
        instrument_result = validate_instrument_completeness(instrument, registry)
        result.merge(instrument_result)

    return result


def generate_stub_relations(
    registry: SpecificationRegistry,
) -> dict[str, list[Any]]:
    """
    Generate stub relations for missing agent-instrument pairs.

    Returns a dict with:
    - 'agent_stubs': List of (agent_name, instrument_name, InstrumentRelation)
    - 'instrument_stubs': List of (instrument_name, agent_name, AgentRelation)

    These can be used to quickly add the missing relations.
    """

    agent_stubs = []
    instrument_stubs = []

    for agent_name, agent in registry.agents.items():
        for instrument_name in registry.list_instruments():
            if not agent.has_relation(instrument_name):
                stub = InstrumentRelation(
                    instrument_name=instrument_name,
                    position=BalanceSheetPosition.NOT_APPLICABLE,
                    can_create=False,
                    can_hold=False,
                    can_transfer=False,
                )
                agent_stubs.append((agent_name, instrument_name, stub))

    for instrument_name, instrument in registry.instruments.items():
        for agent_name in registry.list_agents():
            if not instrument.has_relation(agent_name):
                agent_stub = AgentRelation(
                    agent_name=agent_name,
                    position=BalanceSheetPosition.NOT_APPLICABLE,
                    can_issue=False,
                    can_hold=False,
                )
                instrument_stubs.append((instrument_name, agent_name, agent_stub))

    return {
        "agent_stubs": agent_stubs,
        "instrument_stubs": instrument_stubs,
    }
