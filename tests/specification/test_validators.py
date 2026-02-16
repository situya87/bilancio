"""
Tests for specification validators.

These tests ensure the validation framework correctly identifies
missing or incomplete specifications.
"""

import pytest

from bilancio.specification import (
    AgentRelation,
    AgentSpec,
    BalanceSheetPosition,
    InstrumentRelation,
    InstrumentSpec,
    LifecycleSpec,
    SpecificationRegistry,
    validate_agent_completeness,
    validate_all_relationships,
    validate_instrument_completeness,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_registry() -> SpecificationRegistry:
    """Empty registry for testing."""
    return SpecificationRegistry()


@pytest.fixture
def basic_registry() -> SpecificationRegistry:
    """Registry with basic agents and instruments."""
    registry = SpecificationRegistry()

    # Add a simple agent
    agent = AgentSpec(
        name="Trader",
        description="Ring participant",
        has_bank_account=True,
        bank_assignment_rule="round-robin across trader banks",
    )
    registry.register_agent(agent)

    # Add a simple instrument
    instrument = InstrumentSpec(
        name="Payable",
        description="Promise to pay at maturity",
        tradeable=True,
        interest_bearing=False,
        lifecycle=LifecycleSpec(
            creation_trigger="Initial setup or rollover",
            maturity_trigger="remaining_tau == 0",
            full_settlement_action="Issuer pays face value to holder",
        ),
    )
    registry.register_instrument(instrument)

    return registry


# =============================================================================
# InstrumentRelation Tests
# =============================================================================


class TestInstrumentRelation:
    """Tests for InstrumentRelation completeness checks."""

    def test_not_applicable_is_complete(self):
        """NA relation with no capabilities is complete."""
        rel = InstrumentRelation(
            instrument_name="Test",
            position=BalanceSheetPosition.NOT_APPLICABLE,
            can_create=False,
            can_hold=False,
        )
        is_complete, errors = rel.is_complete()
        assert is_complete
        assert len(errors) == 0

    def test_can_hold_requires_balance_sheet_entry(self):
        """If can_hold, must specify balance_sheet_entry."""
        rel = InstrumentRelation(
            instrument_name="Test",
            position=BalanceSheetPosition.ASSET,
            can_hold=True,
            # Missing balance_sheet_entry
        )
        is_complete, errors = rel.is_complete()
        assert not is_complete
        assert any("balance_sheet_entry" in e for e in errors)

    def test_can_hold_requires_settlement_action(self):
        """If can_hold, must specify settlement_action."""
        rel = InstrumentRelation(
            instrument_name="Test",
            position=BalanceSheetPosition.ASSET,
            can_hold=True,
            balance_sheet_entry="tickets_owned",
            # Missing settlement_action
        )
        is_complete, errors = rel.is_complete()
        assert not is_complete
        assert any("settlement_action" in e for e in errors)

    def test_complete_holding_relation(self):
        """Complete relation for holding an asset."""
        rel = InstrumentRelation(
            instrument_name="Payable",
            position=BalanceSheetPosition.ASSET,
            can_hold=True,
            balance_sheet_entry="tickets_owned",
            settlement_action="Receive payment at maturity",
        )
        is_complete, errors = rel.is_complete()
        assert is_complete
        assert len(errors) == 0


# =============================================================================
# AgentRelation Tests
# =============================================================================


class TestAgentRelation:
    """Tests for AgentRelation completeness checks."""

    def test_can_issue_requires_creation_trigger(self):
        """If can_issue, must specify creation_trigger."""
        rel = AgentRelation(
            agent_name="Bank",
            position=BalanceSheetPosition.ASSET,
            can_issue=True,
            balance_sheet_entry="loans_outstanding",
            # Missing creation_trigger
        )
        is_complete, errors = rel.is_complete()
        assert not is_complete
        assert any("creation_trigger" in e for e in errors)


# =============================================================================
# Registry Tests
# =============================================================================


class TestSpecificationRegistry:
    """Tests for the specification registry."""

    def test_register_agent_warns_about_missing_relations(self, basic_registry):
        """Registering agent without instrument relations generates warnings."""
        new_agent = AgentSpec(
            name="NewAgent",
            description="Test agent",
        )
        warnings = basic_registry.register_agent(new_agent)
        assert len(warnings) > 0
        assert any("Payable" in w for w in warnings)

    def test_register_instrument_warns_about_missing_relations(self, basic_registry):
        """Registering instrument without agent relations generates warnings."""
        new_instrument = InstrumentSpec(
            name="NewInstrument",
            description="Test instrument",
        )
        warnings = basic_registry.register_instrument(new_instrument)
        assert len(warnings) > 0
        assert any("Trader" in w for w in warnings)

    def test_completeness_summary(self, basic_registry):
        """Completeness summary correctly identifies missing pairs."""
        summary = basic_registry.get_completeness_summary()

        assert summary["total_agents"] == 1
        assert summary["total_instruments"] == 1
        assert summary["total_pairs"] == 1
        # Trader doesn't have Payable relation defined
        assert len(summary["missing_pairs"]) == 1
        assert ("Trader", "Payable") in summary["missing_pairs"]

    def test_relationship_matrix(self, basic_registry):
        """Relationship matrix shows missing relations."""
        matrix = basic_registry.get_relationship_matrix()

        assert "Trader" in matrix
        assert "Payable" in matrix["Trader"]
        assert matrix["Trader"]["Payable"] == "MISSING"


# =============================================================================
# Agent Validation Tests
# =============================================================================


class TestAgentValidation:
    """Tests for agent completeness validation."""

    def test_missing_instrument_relation_is_error(self, basic_registry):
        """Missing instrument relation generates error."""
        agent = basic_registry.get_agent("Trader")
        result = validate_agent_completeness(agent, basic_registry)

        assert not result.is_valid
        assert any(
            e.category == "missing_relation" and "Payable" in e.message for e in result.errors
        )

    def test_complete_agent_is_valid(self, empty_registry):
        """Agent with all relations defined is valid."""
        # Register instrument first
        instrument = InstrumentSpec(
            name="Payable",
            description="Test",
            lifecycle=LifecycleSpec(
                creation_trigger="test",
                maturity_trigger="test",
                full_settlement_action="test",
            ),
        )
        empty_registry.register_instrument(instrument)

        # Now create agent with complete relation
        agent = AgentSpec(
            name="Trader",
            description="Test trader",
            instrument_relations={
                "Payable": InstrumentRelation(
                    instrument_name="Payable",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=True,
                    balance_sheet_entry="tickets_owned",
                    settlement_action="Receive payment",
                ),
            },
        )
        empty_registry.register_agent(agent)

        result = validate_agent_completeness(agent, empty_registry)
        # Should have no errors about missing relations
        relation_errors = [e for e in result.errors if e.category == "missing_relation"]
        assert len(relation_errors) == 0


# =============================================================================
# Instrument Validation Tests
# =============================================================================


class TestInstrumentValidation:
    """Tests for instrument completeness validation."""

    def test_missing_lifecycle_is_error(self, empty_registry):
        """Instrument without lifecycle generates error."""
        instrument = InstrumentSpec(
            name="Test",
            description="Test instrument",
            # No lifecycle
        )
        empty_registry.register_instrument(instrument)

        result = validate_instrument_completeness(instrument, empty_registry)
        assert not result.is_valid
        assert any(e.field == "lifecycle" for e in result.errors)

    def test_missing_agent_relation_is_error(self, basic_registry):
        """Missing agent relation generates error."""
        instrument = basic_registry.get_instrument("Payable")
        result = validate_instrument_completeness(instrument, basic_registry)

        assert not result.is_valid
        assert any(
            e.category == "missing_relation" and "Trader" in e.message for e in result.errors
        )


# =============================================================================
# System-Wide Validation Tests
# =============================================================================


class TestSystemValidation:
    """Tests for system-wide relationship validation."""

    def test_inconsistent_can_hold_is_error(self, empty_registry):
        """Inconsistent can_hold between agent and instrument is error."""
        # Create agent that says it can hold
        agent = AgentSpec(
            name="Trader",
            description="Test",
            instrument_relations={
                "Payable": InstrumentRelation(
                    instrument_name="Payable",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=True,
                    balance_sheet_entry="tickets",
                    settlement_action="receive",
                ),
            },
        )
        empty_registry.register_agent(agent)

        # Create instrument that says agent cannot hold
        instrument = InstrumentSpec(
            name="Payable",
            description="Test",
            lifecycle=LifecycleSpec(
                creation_trigger="test",
                maturity_trigger="test",
                full_settlement_action="test",
            ),
            agent_relations={
                "Trader": AgentRelation(
                    agent_name="Trader",
                    position=BalanceSheetPosition.ASSET,
                    can_hold=False,  # Inconsistent!
                ),
            },
        )
        empty_registry.register_instrument(instrument)

        result = validate_all_relationships(empty_registry)
        assert not result.is_valid
        assert any(e.category == "inconsistency" and "can_hold" in e.field for e in result.errors)

    def test_consistent_system_is_valid(self, empty_registry):
        """Fully consistent system passes validation."""
        # Create agent
        agent = AgentSpec(
            name="Trader",
            description="Ring participant",
            instrument_relations={
                "Payable": InstrumentRelation(
                    instrument_name="Payable",
                    position=BalanceSheetPosition.ASSET,
                    can_create=True,
                    can_hold=True,
                    balance_sheet_entry="tickets_owned",
                    settlement_action="Receive payment at maturity",
                ),
            },
        )
        empty_registry.register_agent(agent)

        # Create instrument with consistent relation
        instrument = InstrumentSpec(
            name="Payable",
            description="Promise to pay",
            lifecycle=LifecycleSpec(
                creation_trigger="Initial setup",
                maturity_trigger="remaining_tau == 0",
                full_settlement_action="Pay face value",
            ),
            agent_relations={
                "Trader": AgentRelation(
                    agent_name="Trader",
                    position=BalanceSheetPosition.ASSET,
                    can_issue=True,
                    can_hold=True,
                    balance_sheet_entry="tickets_owned",
                    settlement_action="Receive payment at maturity",
                    creation_trigger="Initial setup or purchase",
                ),
            },
        )
        empty_registry.register_instrument(instrument)

        result = validate_all_relationships(empty_registry)
        # Should have no inconsistency errors
        inconsistency_errors = [e for e in result.errors if e.category == "inconsistency"]
        assert len(inconsistency_errors) == 0


# =============================================================================
# Current System Validation Tests
# =============================================================================


class TestCurrentSystemValidation:
    """Tests for the current bilancio system specification."""

    def test_current_system_has_all_agents(self):
        """Current system has all expected agents."""
        from bilancio.specification.current_system import get_current_registry

        registry = get_current_registry()
        agents = registry.list_agents()

        expected_agents = {"Trader", "Bank", "Dealer", "VBT", "CentralBank"}
        assert set(agents) == expected_agents

    def test_current_system_has_all_instruments(self):
        """Current system has all expected instruments."""
        from bilancio.specification.current_system import get_current_registry

        registry = get_current_registry()
        instruments = registry.list_instruments()

        expected_instruments = {"Payable", "Deposit", "Loan", "Reserve", "CBLoan"}
        assert set(instruments) == expected_instruments

    def test_current_system_is_valid(self):
        """Current system specification passes all validation checks."""
        from bilancio.specification.current_system import validate_current_system

        result = validate_current_system()
        assert result.is_valid, f"Current system has errors: {[e.message for e in result.errors]}"

    def test_all_agent_instrument_pairs_defined(self):
        """Every (agent, instrument) pair has a relation defined."""
        from bilancio.specification.current_system import get_current_registry

        registry = get_current_registry()
        summary = registry.get_completeness_summary()

        assert len(summary["missing_pairs"]) == 0, f"Missing pairs: {summary['missing_pairs']}"
        assert summary["completeness_ratio"] == 1.0

    def test_trader_has_correct_instrument_relations(self):
        """Trader has relations to all instruments."""
        from bilancio.specification.current_system import get_current_registry

        registry = get_current_registry()
        trader = registry.get_agent("Trader")

        # Trader should hold Payable, Deposit, Loan and NOT Reserve/CBLoan
        payable_rel = trader.get_relation("Payable")
        assert payable_rel.can_hold is True
        assert payable_rel.can_create is True

        deposit_rel = trader.get_relation("Deposit")
        assert deposit_rel.can_hold is True
        assert deposit_rel.can_create is False

        loan_rel = trader.get_relation("Loan")
        assert loan_rel.can_hold is True  # As borrower (liability)
        assert loan_rel.can_create is False

        reserve_rel = trader.get_relation("Reserve")
        assert reserve_rel.can_hold is False

    def test_bank_has_correct_instrument_relations(self):
        """Bank has correct relations to all instruments."""
        from bilancio.specification.current_system import get_current_registry

        registry = get_current_registry()
        bank = registry.get_agent("Bank")

        # Bank creates deposits and loans
        deposit_rel = bank.get_relation("Deposit")
        assert deposit_rel.can_create is True
        assert deposit_rel.can_hold is False  # Bank is issuer, not holder

        loan_rel = bank.get_relation("Loan")
        assert loan_rel.can_create is True
        assert loan_rel.can_hold is True  # As lender (asset)

        # Bank holds reserves
        reserve_rel = bank.get_relation("Reserve")
        assert reserve_rel.can_hold is True

    def test_relationship_symmetry(self):
        """Agent-instrument relations are symmetric (agent.can_hold == instrument.can_hold)."""
        from bilancio.specification.current_system import (
            validate_current_system,
        )

        result = validate_current_system()

        # Check no inconsistency errors (which indicate symmetry violations)
        inconsistency_errors = [e for e in result.errors if e.category == "inconsistency"]
        assert len(inconsistency_errors) == 0, (
            f"Symmetry violations: {[e.message for e in inconsistency_errors]}"
        )
