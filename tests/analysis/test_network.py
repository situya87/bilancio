"""Tests for network graph data extraction."""

import pytest

from bilancio.analysis.network import (
    NetworkEdge,
    NetworkNode,
    _snapshot_to_dict,
    build_network_data,
    build_network_time_series,
)
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit, Cash, ReserveDeposit
from bilancio.engines.system import System


def test_network_node_creation():
    """Test NetworkNode dataclass creation."""
    node = NetworkNode(id="A1", name="Agent 1", kind="household")
    assert node.id == "A1"
    assert node.name == "Agent 1"
    assert node.kind == "household"


def test_network_edge_creation():
    """Test NetworkEdge dataclass creation."""
    edge = NetworkEdge(
        source="A1", target="A2", amount=1000, instrument_type="payable", contract_id="C1"
    )
    assert edge.source == "A1"
    assert edge.target == "A2"
    assert edge.amount == 1000
    assert edge.instrument_type == "payable"
    assert edge.contract_id == "C1"


def test_build_network_data_simple(system_with_simple_contracts):
    """Test building network data with simple contracts."""
    system = system_with_simple_contracts

    snapshot = build_network_data(system, day=0)

    assert snapshot.day == 0
    assert len(snapshot.nodes) == 3  # CB, H1, H2
    assert len(snapshot.edges) == 2  # 2 cash contracts

    # Check nodes
    node_ids = {node.id for node in snapshot.nodes}
    assert "CB" in node_ids
    assert "H1" in node_ids
    assert "H2" in node_ids

    # Check edges
    edge_types = {edge.instrument_type for edge in snapshot.edges}
    assert "cash" in edge_types


def test_build_network_data_multiple_contracts(system_with_multiple_instruments):
    """Test building network data with multiple instrument types."""
    system = system_with_multiple_instruments

    snapshot = build_network_data(system, day=0)

    assert snapshot.day == 0
    assert len(snapshot.nodes) == 4  # CB, B1, H1, H2

    # Should have various instrument types
    edge_types = {edge.instrument_type for edge in snapshot.edges}
    assert "cash" in edge_types
    assert "bank_deposit" in edge_types
    assert "reserve_deposit" in edge_types


def test_build_network_data_filter_instruments(system_with_multiple_instruments):
    """Test filtering network edges by instrument type."""
    system = system_with_multiple_instruments

    # Filter for only cash
    snapshot = build_network_data(system, day=0, instrument_types=["cash"])

    assert all(edge.instrument_type == "cash" for edge in snapshot.edges)

    # Filter for multiple types
    snapshot = build_network_data(system, day=0, instrument_types=["cash", "bank_deposit"])

    edge_types = {edge.instrument_type for edge in snapshot.edges}
    assert edge_types.issubset({"cash", "bank_deposit"})


def test_build_network_time_series(system_with_simple_contracts):
    """Test building network snapshots for multiple days."""
    system = system_with_simple_contracts

    days = [0, 1, 2]
    snapshots = build_network_time_series(system, days)

    assert len(snapshots) == 3
    assert snapshots[0].day == 0
    assert snapshots[1].day == 1
    assert snapshots[2].day == 2

    # Each snapshot should have the same nodes
    for snapshot in snapshots:
        assert len(snapshot.nodes) == 3


def test_empty_network(empty_system):
    """Test building network with no contracts."""
    system = empty_system

    snapshot = build_network_data(system, day=0)

    assert snapshot.day == 0
    assert len(snapshot.edges) == 0
    # Should still have nodes (agents)
    assert len(snapshot.nodes) > 0


def test_network_json_serialization(system_with_simple_contracts):
    """Test converting NetworkSnapshot to JSON-serializable dict."""
    system = system_with_simple_contracts

    snapshot = build_network_data(system, day=0)
    snapshot_dict = _snapshot_to_dict(snapshot)

    assert isinstance(snapshot_dict, dict)
    assert "day" in snapshot_dict
    assert "nodes" in snapshot_dict
    assert "edges" in snapshot_dict
    assert snapshot_dict["day"] == 0

    # Check that nodes are dicts
    assert isinstance(snapshot_dict["nodes"], list)
    if len(snapshot_dict["nodes"]) > 0:
        assert isinstance(snapshot_dict["nodes"][0], dict)
        assert "id" in snapshot_dict["nodes"][0]

    # Check that edges are dicts
    assert isinstance(snapshot_dict["edges"], list)
    if len(snapshot_dict["edges"]) > 0:
        assert isinstance(snapshot_dict["edges"][0], dict)
        assert "source" in snapshot_dict["edges"][0]
        assert "target" in snapshot_dict["edges"][0]


def test_edge_source_target_mapping(system_with_simple_contracts):
    """Test that edges correctly map asset_holder_id to source and liability_issuer_id to target."""
    system = system_with_simple_contracts

    # Add a specific contract we can verify
    payable_id = system.new_contract_id()
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=500,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="H2",
        due_day=5,
    )
    system.add_contract(payable)

    snapshot = build_network_data(system, day=0)

    # Find the payable edge
    payable_edge = next((e for e in snapshot.edges if e.contract_id == payable_id), None)
    assert payable_edge is not None
    assert payable_edge.source == "H1"  # asset holder
    assert payable_edge.target == "H2"  # liability issuer
    assert payable_edge.amount == 500
    assert payable_edge.instrument_type == "payable"


# Fixtures


@pytest.fixture
def empty_system():
    """Create a system with agents but no contracts."""
    system = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    system.add_agent(cb)
    system.add_agent(h1)
    return system


@pytest.fixture
def system_with_simple_contracts():
    """Create a system with a few agents and simple contracts."""
    system = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    system.add_agent(cb)
    system.add_agent(h1)
    system.add_agent(h2)

    # Add some cash contracts
    cash1_id = system.new_contract_id()
    cash1 = Cash(
        id=cash1_id,
        kind=InstrumentKind.CASH,
        amount=100,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="CB",
    )
    system.add_contract(cash1)

    cash2_id = system.new_contract_id()
    cash2 = Cash(
        id=cash2_id,
        kind=InstrumentKind.CASH,
        amount=200,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="CB",
    )
    system.add_contract(cash2)

    return system


@pytest.fixture
def system_with_multiple_instruments():
    """Create a system with various instrument types."""
    system = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    system.add_agent(cb)
    system.add_agent(b1)
    system.add_agent(h1)
    system.add_agent(h2)

    # Cash
    cash_id = system.new_contract_id()
    cash = Cash(
        id=cash_id,
        kind=InstrumentKind.CASH,
        amount=100,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="CB",
    )
    system.add_contract(cash)

    # Bank deposit
    deposit_id = system.new_contract_id()
    deposit = BankDeposit(
        id=deposit_id,
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=500,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="B1",
    )
    system.add_contract(deposit)

    # Reserve deposit
    reserve_id = system.new_contract_id()
    reserve = ReserveDeposit(
        id=reserve_id,
        kind=InstrumentKind.RESERVE_DEPOSIT,
        amount=1000,
        denom="X",
        asset_holder_id="B1",
        liability_issuer_id="CB",
    )
    system.add_contract(reserve)

    # Payable
    payable_id = system.new_contract_id()
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=200,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=5,
    )
    system.add_contract(payable)

    return system
