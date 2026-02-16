"""
Data models for agent and instrument specifications.

These models define the structure of complete specifications,
ensuring all relationships are explicitly declared.
"""

from dataclasses import dataclass, field
from enum import Enum


class BalanceSheetPosition(Enum):
    """Position of an instrument on an agent's balance sheet."""

    ASSET = "asset"
    LIABILITY = "liability"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class InstrumentRelation:
    """
    Specification of how an agent relates to a specific instrument.

    Every (Agent, Instrument) pair must have one of these defined.
    """

    instrument_name: str

    # Balance sheet
    position: BalanceSheetPosition
    balance_sheet_entry: str | None = None  # e.g., "deposits", "loans", "inventory"

    # Capabilities
    can_create: bool = False  # Can this agent create/issue this instrument?
    can_hold: bool = False  # Can this agent hold this instrument?
    can_transfer: bool = False  # Can this agent transfer this instrument to others?

    # Settlement
    settlement_role: str | None = None  # e.g., "issuer", "holder", "intermediary"
    settlement_action: str | None = None  # What happens at settlement?

    # Decision relevance
    affects_decisions: bool = False
    decision_description: str | None = None  # How does it affect decisions?

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if this relation is fully specified."""
        errors = []

        # If can hold or create, must have balance sheet entry
        if (
            self.can_hold or self.can_create
        ) and self.position == BalanceSheetPosition.NOT_APPLICABLE:
            errors.append("Can hold/create but position is NOT_APPLICABLE")

        if (self.can_hold or self.can_create) and not self.balance_sheet_entry:
            errors.append("Can hold/create but no balance_sheet_entry specified")

        # If can hold, should have settlement action
        if self.can_hold and not self.settlement_action:
            errors.append("Can hold but no settlement_action specified")

        # If affects decisions, should have description
        if self.affects_decisions and not self.decision_description:
            errors.append("Affects decisions but no decision_description")

        return len(errors) == 0, errors


@dataclass
class AgentRelation:
    """
    Specification of how an instrument relates to a specific agent.

    Every (Instrument, Agent) pair must have one of these defined.
    """

    agent_name: str

    # Balance sheet
    position: BalanceSheetPosition
    balance_sheet_entry: str | None = None

    # Capabilities
    can_issue: bool = False  # Can this agent issue this instrument?
    can_hold: bool = False  # Can this agent hold this instrument?

    # Settlement
    settlement_action: str | None = None

    # Creation
    creation_trigger: str | None = None  # What causes creation for this agent?
    creation_effect: str | None = None  # What happens when created?

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if this relation is fully specified."""
        errors = []

        if (
            self.can_hold or self.can_issue
        ) and self.position == BalanceSheetPosition.NOT_APPLICABLE:
            errors.append("Can hold/issue but position is NOT_APPLICABLE")

        if (self.can_hold or self.can_issue) and not self.balance_sheet_entry:
            errors.append("Can hold/issue but no balance_sheet_entry specified")

        if self.can_issue and not self.creation_trigger:
            errors.append("Can issue but no creation_trigger specified")

        if self.can_hold and not self.settlement_action:
            errors.append("Can hold but no settlement_action specified")

        return len(errors) == 0, errors


@dataclass
class DecisionSpec:
    """Specification of a decision an agent makes."""

    name: str
    trigger: str  # When is this decision made?
    inputs: list[str]  # What state variables are considered?
    instruments_involved: list[str]  # Which instruments are relevant?
    outputs: list[str]  # What actions result?
    logic_description: str  # How is the decision made?

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if decision is fully specified."""
        errors = []

        if not self.trigger:
            errors.append("No trigger specified")
        if not self.inputs:
            errors.append("No inputs specified")
        if not self.outputs:
            errors.append("No outputs specified")
        if not self.logic_description:
            errors.append("No logic_description specified")

        return len(errors) == 0, errors


@dataclass
class LifecycleSpec:
    """Specification of lifecycle events for an agent or instrument."""

    # Creation
    creation_trigger: str | None = None
    creation_effect: str | None = None

    # Holding period
    interest_accrual: str | None = None  # Does interest accrue? How?
    transferable: bool | None = None
    transfer_mechanism: str | None = None

    # Maturity/Settlement
    maturity_trigger: str | None = None
    full_settlement_action: str | None = None
    partial_settlement_action: str | None = None

    # Rollover
    can_rollover: bool = False
    rollover_trigger: str | None = None
    rollover_terms: str | None = None

    # Default
    default_trigger: str | None = None
    default_priority: str | None = None
    recovery_mechanism: str | None = None

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if lifecycle is fully specified."""
        errors = []

        if not self.creation_trigger:
            errors.append("No creation_trigger specified")
        if not self.maturity_trigger:
            errors.append("No maturity_trigger specified")
        if not self.full_settlement_action:
            errors.append("No full_settlement_action specified")

        if self.can_rollover and not self.rollover_trigger:
            errors.append("Can rollover but no rollover_trigger specified")

        return len(errors) == 0, errors


@dataclass
class AgentSpec:
    """
    Complete specification of an agent type.

    Includes all relationships to instruments and decision-making logic.
    """

    name: str
    description: str

    # State
    state_fields: dict[str, str] = field(default_factory=dict)  # field_name -> type/description

    # Banking
    has_bank_account: bool = False
    bank_assignment_rule: str | None = None

    # Instrument relationships (one per instrument in the system)
    instrument_relations: dict[str, InstrumentRelation] = field(default_factory=dict)

    # Decision-making
    decisions: list[DecisionSpec] = field(default_factory=list)

    # Lifecycle
    lifecycle: LifecycleSpec | None = None

    def get_relation(self, instrument_name: str) -> InstrumentRelation | None:
        """Get the relation specification for a given instrument."""
        return self.instrument_relations.get(instrument_name)

    def has_relation(self, instrument_name: str) -> bool:
        """Check if a relation exists for the given instrument."""
        return instrument_name in self.instrument_relations


@dataclass
class InstrumentInteraction:
    """Specification of how one instrument interacts with another."""

    other_instrument: str
    relationship: str  # "substitutes", "complements", "independent", "created_together"
    decision_tradeoff: str  # When would agent choose one over the other?

    def is_complete(self) -> tuple[bool, list[str]]:
        errors = []
        if not self.relationship:
            errors.append("No relationship specified")
        if not self.decision_tradeoff:
            errors.append("No decision_tradeoff specified")
        return len(errors) == 0, errors


@dataclass
class InstrumentSpec:
    """
    Complete specification of an instrument type.

    Includes all relationships to agents and lifecycle.
    """

    name: str
    description: str

    # Attributes
    attributes: dict[str, str] = field(default_factory=dict)  # attribute_name -> type/description

    # Classification
    tradeable: bool = False
    interest_bearing: bool = False
    secured: bool = False

    # Agent relationships (one per agent in the system)
    agent_relations: dict[str, AgentRelation] = field(default_factory=dict)

    # Lifecycle
    lifecycle: LifecycleSpec | None = None

    # Interactions with other instruments
    instrument_interactions: dict[str, InstrumentInteraction] = field(default_factory=dict)

    def get_relation(self, agent_name: str) -> AgentRelation | None:
        """Get the relation specification for a given agent."""
        return self.agent_relations.get(agent_name)

    def has_relation(self, agent_name: str) -> bool:
        """Check if a relation exists for the given agent."""
        return agent_name in self.agent_relations
