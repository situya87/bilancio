"""
Additional tests for bilancio.specification.validators — targeting uncovered lines.

Uncovered lines:
  - 45: ValidationResult.merge sets is_valid=False
  - 66: agent missing name
  - 69: agent missing description
  - 85, 96: incomplete relation / incomplete decision errors
  - 106: decision references unknown instrument (warning)
  - 122: has_bank_account but no bank_assignment_rule (warning)
  - 147: instrument missing name
  - 150: instrument missing description
  - 166: incomplete instrument-side relation
  - 184: incomplete lifecycle
  - 204: incomplete instrument interaction
  - 231: missing pairs in validate_all_relationships
  - 256: inconsistent can_create/can_issue
  - 265: inconsistent position
  - 296-324: generate_stub_relations
"""

import pytest

from bilancio.specification import (
    AgentRelation,
    AgentSpec,
    BalanceSheetPosition,
    DecisionSpec,
    InstrumentRelation,
    InstrumentSpec,
    LifecycleSpec,
    SpecificationRegistry,
    ValidationResult,
)
from bilancio.specification.models import InstrumentInteraction
from bilancio.specification.validators import (
    generate_stub_relations,
    validate_agent_completeness,
    validate_all_relationships,
    validate_instrument_completeness,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_registry() -> SpecificationRegistry:
    return SpecificationRegistry()


def _make_complete_lifecycle() -> LifecycleSpec:
    """Return a lifecycle spec that passes all completeness checks."""
    return LifecycleSpec(
        creation_trigger="On setup",
        maturity_trigger="tau==0",
        full_settlement_action="Pay face",
    )


# =============================================================================
# ValidationResult.merge  (line 45)
# =============================================================================


class TestValidationResultMerge:
    """Tests for merging validation results."""

    def test_merge_invalid_into_valid_sets_invalid(self):
        """Merging an invalid result into a valid one makes it invalid (line 45)."""
        valid = ValidationResult(is_valid=True)
        invalid = ValidationResult(is_valid=True)
        invalid.add_error("test", "entity", "field", "something is wrong")
        assert not invalid.is_valid

        valid.merge(invalid)
        assert not valid.is_valid
        assert len(valid.errors) == 1

    def test_merge_valid_into_valid_stays_valid(self):
        """Merging two valid results keeps the target valid."""
        a = ValidationResult(is_valid=True)
        b = ValidationResult(is_valid=True)
        b.add_warning("minor issue")

        a.merge(b)
        assert a.is_valid
        assert len(a.warnings) == 1

    def test_merge_accumulates_errors_and_warnings(self):
        """Errors and warnings from both sides accumulate."""
        a = ValidationResult(is_valid=True)
        a.add_warning("w1")
        b = ValidationResult(is_valid=True)
        b.add_error("cat", "e", "f", "err1")
        b.add_warning("w2")

        a.merge(b)
        assert len(a.warnings) == 2
        assert len(a.errors) == 1


# =============================================================================
# validate_agent_completeness — missing fields (lines 66, 69)
# =============================================================================


class TestAgentMissingFields:
    """Tests for agent validation when required fields are empty."""

    def test_empty_name_is_error(self, empty_registry):
        """Agent with empty name triggers missing_field error (line 66)."""
        agent = AgentSpec(name="", description="some desc")
        empty_registry.register_agent(agent)
        result = validate_agent_completeness(agent, empty_registry)

        assert not result.is_valid
        name_errors = [e for e in result.errors if e.field == "name"]
        assert len(name_errors) >= 1

    def test_empty_description_is_error(self, empty_registry):
        """Agent with empty description triggers error (line 69)."""
        agent = AgentSpec(name="TestAgent", description="")
        empty_registry.register_agent(agent)
        result = validate_agent_completeness(agent, empty_registry)

        assert not result.is_valid
        desc_errors = [e for e in result.errors if e.field == "description"]
        assert len(desc_errors) >= 1


# =============================================================================
# validate_agent_completeness — incomplete relations (line 85)
# =============================================================================


class TestAgentIncompleteRelation:
    """Tests for agent validation when an instrument relation is incomplete."""

    def test_incomplete_instrument_relation_is_error(self, empty_registry):
        """A relation that fails is_complete() adds an error (line 85)."""
        # Register an instrument first
        instr = InstrumentSpec(
            name="Bond",
            description="A bond",
            lifecycle=_make_complete_lifecycle(),
        )
        empty_registry.register_instrument(instr)

        # Agent has a relation to Bond but the relation is incomplete
        agent = AgentSpec(
            name="Investor",
            description="Invests",
            instrument_relations={
                "Bond": InstrumentRelation(
                    instrument_name="Bond",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=True,
                    # Missing balance_sheet_entry and settlement_action
                ),
            },
        )
        empty_registry.register_agent(agent)

        result = validate_agent_completeness(agent, empty_registry)
        assert not result.is_valid
        incomplete_errors = [e for e in result.errors if e.category == "incomplete_relation"]
        assert len(incomplete_errors) >= 1


# =============================================================================
# validate_agent_completeness — incomplete decisions (line 96)
# =============================================================================


class TestAgentIncompleteDecision:
    """Tests for incomplete decision specs on an agent."""

    def test_incomplete_decision_is_error(self, empty_registry):
        """Decision missing trigger/inputs/outputs triggers error (line 96)."""
        agent = AgentSpec(
            name="Decider",
            description="Makes decisions",
            decisions=[
                DecisionSpec(
                    name="bad_decision",
                    trigger="",  # Empty => incomplete
                    inputs=[],  # Empty => incomplete
                    instruments_involved=[],
                    outputs=[],  # Empty => incomplete
                    logic_description="",  # Empty => incomplete
                ),
            ],
        )
        empty_registry.register_agent(agent)

        result = validate_agent_completeness(agent, empty_registry)
        assert not result.is_valid
        decision_errors = [e for e in result.errors if e.category == "incomplete_decision"]
        assert len(decision_errors) >= 1


# =============================================================================
# validate_agent_completeness — unknown instrument in decision (line 106)
# =============================================================================


class TestAgentDecisionUnknownInstrument:
    """Tests for decision referencing unknown instrument."""

    def test_unknown_instrument_in_decision_is_warning(self, empty_registry):
        """Decision instruments_involved referencing non-existent instrument (line 106)."""
        agent = AgentSpec(
            name="Speculator",
            description="Speculates",
            decisions=[
                DecisionSpec(
                    name="trade",
                    trigger="price signal",
                    inputs=["market_price"],
                    instruments_involved=["FictionalDerivative"],
                    outputs=["trade_order"],
                    logic_description="Buy if cheap",
                ),
            ],
        )
        empty_registry.register_agent(agent)

        result = validate_agent_completeness(agent, empty_registry)
        assert any("FictionalDerivative" in w for w in result.warnings)


# =============================================================================
# validate_agent_completeness — bank account without assignment rule (line 122)
# =============================================================================


class TestAgentBankAccountWarning:
    """Test warning when agent has bank_account but no assignment rule."""

    def test_bank_account_without_rule_warns(self, empty_registry):
        """has_bank_account=True without bank_assignment_rule (line 122)."""
        agent = AgentSpec(
            name="Orphan",
            description="Has account, no rule",
            has_bank_account=True,
            bank_assignment_rule=None,
        )
        empty_registry.register_agent(agent)

        result = validate_agent_completeness(agent, empty_registry)
        assert any("assignment rule" in w for w in result.warnings)


# =============================================================================
# validate_instrument_completeness — missing fields (lines 147, 150)
# =============================================================================


class TestInstrumentMissingFields:
    """Tests for instrument validation when required fields are empty."""

    def test_empty_name_is_error(self, empty_registry):
        """Instrument with empty name (line 147)."""
        instr = InstrumentSpec(name="", description="desc")
        empty_registry.register_instrument(instr)
        result = validate_instrument_completeness(instr, empty_registry)

        assert not result.is_valid
        name_errors = [e for e in result.errors if e.field == "name"]
        assert len(name_errors) >= 1

    def test_empty_description_is_error(self, empty_registry):
        """Instrument with empty description (line 150)."""
        instr = InstrumentSpec(name="TestInstr", description="")
        empty_registry.register_instrument(instr)
        result = validate_instrument_completeness(instr, empty_registry)

        assert not result.is_valid
        desc_errors = [e for e in result.errors if e.field == "description"]
        assert len(desc_errors) >= 1


# =============================================================================
# validate_instrument_completeness — incomplete agent relation (line 166)
# =============================================================================


class TestInstrumentIncompleteRelation:
    """Tests for incomplete agent relations on an instrument."""

    def test_incomplete_agent_relation_is_error(self, empty_registry):
        """Agent relation that fails is_complete() triggers error (line 166)."""
        # Register an agent first
        agent = AgentSpec(name="Bank", description="A bank")
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Loan",
            description="A loan",
            lifecycle=_make_complete_lifecycle(),
            agent_relations={
                "Bank": AgentRelation(
                    agent_name="Bank",
                    position=BalanceSheetPosition.ASSET,
                    can_issue=True,
                    # Missing creation_trigger and balance_sheet_entry => incomplete
                ),
            },
        )
        empty_registry.register_instrument(instr)

        result = validate_instrument_completeness(instr, empty_registry)
        assert not result.is_valid
        incomplete_errors = [e for e in result.errors if e.category == "incomplete_relation"]
        assert len(incomplete_errors) >= 1


# =============================================================================
# validate_instrument_completeness — incomplete lifecycle (line 184)
# =============================================================================


class TestInstrumentIncompleteLifecycle:
    """Tests for instrument with incomplete lifecycle."""

    def test_lifecycle_missing_creation_trigger(self, empty_registry):
        """Lifecycle without creation_trigger triggers error (line 184)."""
        instr = InstrumentSpec(
            name="Bond",
            description="A bond",
            lifecycle=LifecycleSpec(
                # creation_trigger missing
                maturity_trigger="tau==0",
                full_settlement_action="Pay face",
            ),
        )
        empty_registry.register_instrument(instr)

        result = validate_instrument_completeness(instr, empty_registry)
        assert not result.is_valid
        lifecycle_errors = [e for e in result.errors if e.category == "incomplete_lifecycle"]
        assert len(lifecycle_errors) >= 1


# =============================================================================
# validate_instrument_completeness — incomplete interaction (line 204)
# =============================================================================


class TestInstrumentIncompleteInteraction:
    """Tests for instrument with incomplete instrument interaction."""

    def test_incomplete_interaction_is_error(self, empty_registry):
        """Interaction missing relationship or decision_tradeoff (line 204)."""
        # Register a second instrument so interaction check fires
        instr_a = InstrumentSpec(
            name="Alpha",
            description="First instrument",
            lifecycle=_make_complete_lifecycle(),
            instrument_interactions={
                "Beta": InstrumentInteraction(
                    other_instrument="Beta",
                    relationship="",  # Empty => incomplete
                    decision_tradeoff="",  # Empty => incomplete
                ),
            },
        )
        empty_registry.register_instrument(instr_a)

        instr_b = InstrumentSpec(
            name="Beta",
            description="Second instrument",
            lifecycle=_make_complete_lifecycle(),
        )
        empty_registry.register_instrument(instr_b)

        result = validate_instrument_completeness(instr_a, empty_registry)
        assert not result.is_valid
        interaction_errors = [e for e in result.errors if e.category == "incomplete_interaction"]
        assert len(interaction_errors) >= 1


# =============================================================================
# validate_all_relationships — missing pairs (line 231)
# =============================================================================


class TestAllRelationshipsMissingPairs:
    """Tests for the top-level validate_all_relationships function."""

    def test_missing_pairs_reported(self, empty_registry):
        """Missing agent-instrument pairs from the registry (line 231)."""
        agent = AgentSpec(name="Trader", description="A trader")
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Payable",
            description="A payable",
            lifecycle=_make_complete_lifecycle(),
        )
        empty_registry.register_instrument(instr)

        result = validate_all_relationships(empty_registry)
        assert not result.is_valid
        missing_errors = [
            e
            for e in result.errors
            if e.category == "missing_relation" and "Trader" in e.entity and "Payable" in e.entity
        ]
        assert len(missing_errors) >= 1


# =============================================================================
# validate_all_relationships — inconsistent can_create/can_issue (line 256)
# =============================================================================


class TestAllRelationshipsInconsistentCreateIssue:
    """Tests for can_create vs can_issue inconsistency."""

    def test_inconsistent_can_create_can_issue(self, empty_registry):
        """Agent.can_create != instrument.can_issue is an error (line 256)."""
        agent = AgentSpec(
            name="Bank",
            description="A bank",
            instrument_relations={
                "Loan": InstrumentRelation(
                    instrument_name="Loan",
                    position=BalanceSheetPosition.ASSET,
                    can_create=True,
                    can_hold=True,
                    balance_sheet_entry="loans",
                    settlement_action="collect",
                ),
            },
        )
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Loan",
            description="A loan",
            lifecycle=_make_complete_lifecycle(),
            agent_relations={
                "Bank": AgentRelation(
                    agent_name="Bank",
                    position=BalanceSheetPosition.ASSET,
                    can_issue=False,  # Inconsistent with agent's can_create=True
                    can_hold=True,
                    balance_sheet_entry="loans",
                    settlement_action="collect",
                    creation_trigger="on request",
                ),
            },
        )
        empty_registry.register_instrument(instr)

        result = validate_all_relationships(empty_registry)
        inconsistency_errors = [
            e for e in result.errors if e.category == "inconsistency" and "can_create" in e.field
        ]
        assert len(inconsistency_errors) >= 1


# =============================================================================
# validate_all_relationships — inconsistent position (line 265)
# =============================================================================


class TestAllRelationshipsInconsistentPosition:
    """Tests for position inconsistency between agent and instrument."""

    def test_inconsistent_position(self, empty_registry):
        """Agent.position != instrument.position triggers error (line 265)."""
        agent = AgentSpec(
            name="Holder",
            description="A holder",
            instrument_relations={
                "Note": InstrumentRelation(
                    instrument_name="Note",
                    position=BalanceSheetPosition.ASSET,
                    can_create=False,
                    can_hold=True,
                    balance_sheet_entry="notes",
                    settlement_action="receive",
                ),
            },
        )
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Note",
            description="A note",
            lifecycle=_make_complete_lifecycle(),
            agent_relations={
                "Holder": AgentRelation(
                    agent_name="Holder",
                    position=BalanceSheetPosition.LIABILITY,  # Inconsistent
                    can_issue=False,
                    can_hold=True,
                    balance_sheet_entry="notes",
                    settlement_action="receive",
                ),
            },
        )
        empty_registry.register_instrument(instr)

        result = validate_all_relationships(empty_registry)
        position_errors = [
            e for e in result.errors if e.category == "inconsistency" and "position" in e.field
        ]
        assert len(position_errors) >= 1


# =============================================================================
# generate_stub_relations (lines 296-324)
# =============================================================================


class TestGenerateStubRelations:
    """Tests for the stub generation utility."""

    def test_generates_agent_stubs_for_missing(self, empty_registry):
        """Generates InstrumentRelation stubs for agents missing relations."""
        agent = AgentSpec(name="Trader", description="A trader")
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Payable",
            description="A payable",
            lifecycle=_make_complete_lifecycle(),
        )
        empty_registry.register_instrument(instr)

        stubs = generate_stub_relations(empty_registry)

        assert len(stubs["agent_stubs"]) == 1
        agent_name, instrument_name, relation = stubs["agent_stubs"][0]
        assert agent_name == "Trader"
        assert instrument_name == "Payable"
        assert relation.position == BalanceSheetPosition.NOT_APPLICABLE
        assert relation.can_create is False
        assert relation.can_hold is False
        assert relation.can_transfer is False

    def test_generates_instrument_stubs_for_missing(self, empty_registry):
        """Generates AgentRelation stubs for instruments missing relations."""
        agent = AgentSpec(name="Bank", description="A bank")
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Deposit",
            description="A deposit",
            lifecycle=_make_complete_lifecycle(),
        )
        empty_registry.register_instrument(instr)

        stubs = generate_stub_relations(empty_registry)

        assert len(stubs["instrument_stubs"]) == 1
        instrument_name, agent_name, relation = stubs["instrument_stubs"][0]
        assert instrument_name == "Deposit"
        assert agent_name == "Bank"
        assert relation.position == BalanceSheetPosition.NOT_APPLICABLE
        assert relation.can_issue is False
        assert relation.can_hold is False

    def test_no_stubs_when_all_defined(self, empty_registry):
        """No stubs generated when all relations are defined."""
        agent = AgentSpec(
            name="Trader",
            description="A trader",
            instrument_relations={
                "Bond": InstrumentRelation(
                    instrument_name="Bond",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=True,
                    balance_sheet_entry="bonds",
                    settlement_action="receive coupon",
                ),
            },
        )
        empty_registry.register_agent(agent)

        instr = InstrumentSpec(
            name="Bond",
            description="A bond",
            lifecycle=_make_complete_lifecycle(),
            agent_relations={
                "Trader": AgentRelation(
                    agent_name="Trader",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=True,
                    balance_sheet_entry="bonds",
                    settlement_action="receive coupon",
                ),
            },
        )
        empty_registry.register_instrument(instr)

        stubs = generate_stub_relations(empty_registry)
        assert len(stubs["agent_stubs"]) == 0
        assert len(stubs["instrument_stubs"]) == 0

    def test_stubs_for_multiple_missing_pairs(self, empty_registry):
        """Stubs for 2 agents x 2 instruments with no relations."""
        for name in ("A1", "A2"):
            empty_registry.register_agent(AgentSpec(name=name, description=f"Agent {name}"))
        for name in ("I1", "I2"):
            empty_registry.register_instrument(
                InstrumentSpec(
                    name=name,
                    description=f"Instrument {name}",
                    lifecycle=_make_complete_lifecycle(),
                )
            )

        stubs = generate_stub_relations(empty_registry)
        # 2 agents * 2 instruments = 4 agent stubs
        assert len(stubs["agent_stubs"]) == 4
        # 2 instruments * 2 agents = 4 instrument stubs
        assert len(stubs["instrument_stubs"]) == 4
