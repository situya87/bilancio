"""Coverage tests for specification/registry.py.

Targets uncovered methods: get_missing_agent_relations, get_missing_instrument_relations,
get_relationship_matrix, get_completeness_summary.
"""

from __future__ import annotations

from bilancio.specification.models import AgentSpec, InstrumentSpec
from bilancio.specification.registry import SpecificationRegistry


def _make_agent_spec(name: str) -> AgentSpec:
    """Create a simple AgentSpec for testing."""
    spec = AgentSpec(
        name=name,
        description=f"Test agent {name}",
    )
    return spec


def _make_instrument_spec(
    name: str, agent_relations: dict | None = None, interactions: dict | None = None
) -> InstrumentSpec:
    """Create a simple InstrumentSpec for testing."""
    spec = InstrumentSpec(
        name=name,
        description=f"Test instrument {name}",
        agent_relations=agent_relations or {},
        instrument_interactions=interactions or {},
    )
    return spec


class TestGetMissingAgentRelations:
    """Cover get_missing_agent_relations."""

    def test_missing_agent_returns_empty(self):
        reg = SpecificationRegistry()
        result = reg.get_missing_agent_relations("nonexistent")
        assert result == []

    def test_no_instruments_returns_empty(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        reg.register_agent(agent)
        result = reg.get_missing_agent_relations("Bank")
        assert result == []

    def test_reports_missing_instrument_relation(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        instrument = _make_instrument_spec("Cash")
        reg.register_agent(agent)
        reg.register_instrument(instrument)
        result = reg.get_missing_agent_relations("Bank")
        assert "Cash" in result


class TestGetMissingInstrumentRelations:
    """Cover get_missing_instrument_relations."""

    def test_missing_instrument_returns_empty(self):
        reg = SpecificationRegistry()
        result = reg.get_missing_instrument_relations("nonexistent")
        assert result == []

    def test_no_agents_returns_empty(self):
        reg = SpecificationRegistry()
        instrument = _make_instrument_spec("Cash")
        reg.register_instrument(instrument)
        result = reg.get_missing_instrument_relations("Cash")
        assert result == []

    def test_reports_missing_agent_relation(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        instrument = _make_instrument_spec("Cash")
        reg.register_agent(agent)
        reg.register_instrument(instrument)
        result = reg.get_missing_instrument_relations("Cash")
        assert "Bank" in result


class TestGetRelationshipMatrix:
    """Cover get_relationship_matrix."""

    def test_empty_registry(self):
        reg = SpecificationRegistry()
        matrix = reg.get_relationship_matrix()
        assert matrix == {}

    def test_with_missing_relations(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        instrument = _make_instrument_spec("Cash")
        reg.register_agent(agent)
        reg.register_instrument(instrument)
        matrix = reg.get_relationship_matrix()
        assert "Bank" in matrix
        assert matrix["Bank"]["Cash"] == "MISSING"


class TestGetCompletenessSummary:
    """Cover get_completeness_summary."""

    def test_empty_registry(self):
        reg = SpecificationRegistry()
        summary = reg.get_completeness_summary()
        assert summary["total_agents"] == 0
        assert summary["total_instruments"] == 0
        assert summary["total_pairs"] == 0
        assert summary["defined_pairs"] == 0
        assert summary["missing_pairs"] == []
        assert summary["completeness_ratio"] == 1.0

    def test_all_missing(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        instrument = _make_instrument_spec("Cash")
        reg.register_agent(agent)
        reg.register_instrument(instrument)
        summary = reg.get_completeness_summary()
        assert summary["total_agents"] == 1
        assert summary["total_instruments"] == 1
        assert summary["total_pairs"] == 1
        assert summary["defined_pairs"] == 0
        assert len(summary["missing_pairs"]) == 1
        assert summary["completeness_ratio"] == 0.0

    def test_multiple_agents_and_instruments(self):
        reg = SpecificationRegistry()
        for name in ["Bank", "Household"]:
            reg.register_agent(_make_agent_spec(name))
        for name in ["Cash", "Deposit"]:
            reg.register_instrument(_make_instrument_spec(name))
        summary = reg.get_completeness_summary()
        assert summary["total_agents"] == 2
        assert summary["total_instruments"] == 2
        assert summary["total_pairs"] == 4
        assert len(summary["missing_pairs"]) == 4


class TestRegisterWithWarnings:
    """Cover register_agent/register_instrument warning paths."""

    def test_register_agent_warns_about_missing_relations(self):
        reg = SpecificationRegistry()
        instrument = _make_instrument_spec("Cash")
        reg.register_instrument(instrument)
        agent = _make_agent_spec("Bank")
        warnings = reg.register_agent(agent)
        assert len(warnings) == 1
        assert "Cash" in warnings[0]

    def test_register_instrument_warns_about_missing_agent_relations(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        reg.register_agent(agent)
        instrument = _make_instrument_spec("Cash")
        warnings = reg.register_instrument(instrument)
        assert any("Bank" in w for w in warnings)

    def test_register_instrument_warns_about_missing_instrument_interactions(self):
        reg = SpecificationRegistry()
        inst1 = _make_instrument_spec("Cash")
        reg.register_instrument(inst1)
        inst2 = _make_instrument_spec("Deposit")
        warnings = reg.register_instrument(inst2)
        assert any("Cash" in w for w in warnings)


class TestListMethods:
    """Cover list_agents and list_instruments."""

    def test_list_agents(self):
        reg = SpecificationRegistry()
        reg.register_agent(_make_agent_spec("Bank"))
        reg.register_agent(_make_agent_spec("Household"))
        names = reg.list_agents()
        assert "Bank" in names
        assert "Household" in names

    def test_list_instruments(self):
        reg = SpecificationRegistry()
        reg.register_instrument(_make_instrument_spec("Cash"))
        names = reg.list_instruments()
        assert "Cash" in names

    def test_get_agent(self):
        reg = SpecificationRegistry()
        agent = _make_agent_spec("Bank")
        reg.register_agent(agent)
        assert reg.get_agent("Bank") is agent
        assert reg.get_agent("nonexistent") is None

    def test_get_instrument(self):
        reg = SpecificationRegistry()
        inst = _make_instrument_spec("Cash")
        reg.register_instrument(inst)
        assert reg.get_instrument("Cash") is inst
        assert reg.get_instrument("nonexistent") is None
