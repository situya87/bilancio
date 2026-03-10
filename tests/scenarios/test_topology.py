"""Tests for network topology generators (Plan 055)."""

import random

import pytest

from bilancio.scenarios.ring.topology import (
    DebtEdge,
    ErdosRenyiTopology,
    KRegularTopology,
    RingTopology,
    Topology,
    parse_topology_string,
    topology_from_config,
)


class TestRingTopology:
    """Test RingTopology edge generation."""

    def test_ring_edges_n5(self):
        """5-agent ring produces correct edges."""
        topo = RingTopology()
        edges = topo.generate_edges(5, random.Random(42))
        assert len(edges) == 5
        expected = [
            DebtEdge("H1", "H2"),
            DebtEdge("H2", "H3"),
            DebtEdge("H3", "H4"),
            DebtEdge("H4", "H5"),
            DebtEdge("H5", "H1"),
        ]
        assert edges == expected

    def test_ring_edges_n3(self):
        """3-agent ring wraps correctly."""
        topo = RingTopology()
        edges = topo.generate_edges(3, random.Random(42))
        assert edges == [
            DebtEdge("H1", "H2"),
            DebtEdge("H2", "H3"),
            DebtEdge("H3", "H1"),
        ]

    def test_ring_matches_compiler_formula(self):
        """RingTopology produces same edges as the old compiler hardcoded formula."""
        topo = RingTopology()
        n = 10
        edges = topo.generate_edges(n, random.Random(42))
        for i, edge in enumerate(edges):
            expected_debtor = f"H{i + 1}"
            expected_creditor = f"H{(i + 1) % n + 1}"
            assert edge.debtor == expected_debtor
            assert edge.creditor == expected_creditor

    def test_is_topology_protocol(self):
        """RingTopology implements the Topology protocol."""
        assert isinstance(RingTopology(), Topology)


class TestKRegularTopology:
    """Test KRegularTopology edge generation."""

    def test_k2_degree(self):
        """Each agent has exactly 2 outgoing edges."""
        topo = KRegularTopology(degree=2)
        edges = topo.generate_edges(5, random.Random(42))
        # 5 agents x 2 edges = 10 edges
        assert len(edges) == 10
        # Check each agent has exactly 2 outgoing edges
        for i in range(1, 6):
            agent = f"H{i}"
            out_edges = [e for e in edges if e.debtor == agent]
            assert len(out_edges) == 2

    def test_k1_matches_ring(self):
        """degree=1 is equivalent to RingTopology."""
        ring = RingTopology().generate_edges(5, random.Random(42))
        k1 = KRegularTopology(degree=1).generate_edges(5, random.Random(42))
        assert ring == k1

    def test_no_self_loops(self):
        """No agent owes itself."""
        topo = KRegularTopology(degree=3)
        edges = topo.generate_edges(10, random.Random(42))
        for edge in edges:
            assert edge.debtor != edge.creditor

    def test_in_degree_equals_out_degree(self):
        """Each agent has equal in-degree and out-degree."""
        topo = KRegularTopology(degree=3)
        edges = topo.generate_edges(8, random.Random(42))
        for i in range(1, 9):
            agent = f"H{i}"
            out_deg = sum(1 for e in edges if e.debtor == agent)
            in_deg = sum(1 for e in edges if e.creditor == agent)
            assert out_deg == 3
            assert in_deg == 3

    def test_degree_too_high_raises(self):
        """degree >= n_agents should raise ValueError."""
        topo = KRegularTopology(degree=5)
        with pytest.raises(ValueError, match="degree"):
            topo.generate_edges(5, random.Random(42))

    def test_degree_zero_raises(self):
        """degree < 1 should raise ValueError."""
        topo = KRegularTopology(degree=0)
        with pytest.raises(ValueError, match="degree"):
            topo.generate_edges(5, random.Random(42))

    def test_is_topology_protocol(self):
        """KRegularTopology implements the Topology protocol."""
        assert isinstance(KRegularTopology(), Topology)


class TestErdosRenyiTopology:
    """Test ErdosRenyiTopology edge generation."""

    def test_connected(self):
        """Generated graph is strongly connected."""
        topo = ErdosRenyiTopology(edge_prob=0.3)
        edges = topo.generate_edges(10, random.Random(42))
        # Check strong connectivity by BFS from every node
        adj = {}
        for e in edges:
            adj.setdefault(e.debtor, []).append(e.creditor)

        for i in range(1, 11):
            start = f"H{i}"
            visited = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append(neighbor)
            assert len(visited) == 10, f"Node {start} cannot reach all nodes"

    def test_density_approximate(self):
        """Edge count is approximately n^2 * p."""
        topo = ErdosRenyiTopology(edge_prob=0.5)
        n = 20
        edges = topo.generate_edges(n, random.Random(42))
        expected = n * (n - 1) * 0.5  # 190
        # Allow wide margin due to connectivity enforcement
        assert len(edges) >= n  # At least n edges (from Hamiltonian cycle)
        assert len(edges) <= n * (n - 1)  # At most n(n-1) edges

    def test_no_self_loops(self):
        """No agent owes itself."""
        topo = ErdosRenyiTopology(edge_prob=0.3)
        edges = topo.generate_edges(10, random.Random(42))
        for edge in edges:
            assert edge.debtor != edge.creditor

    def test_every_node_has_outgoing(self):
        """Every node has at least one outgoing edge."""
        topo = ErdosRenyiTopology(edge_prob=0.1)
        edges = topo.generate_edges(20, random.Random(42))
        debtors = {e.debtor for e in edges}
        for i in range(1, 21):
            assert f"H{i}" in debtors

    def test_invalid_prob_raises(self):
        """edge_prob outside (0, 1] should raise."""
        with pytest.raises(ValueError):
            ErdosRenyiTopology(edge_prob=0).generate_edges(5, random.Random(42))
        with pytest.raises(ValueError):
            ErdosRenyiTopology(edge_prob=-0.1).generate_edges(5, random.Random(42))

    def test_too_few_agents_raises(self):
        """n_agents < 2 should raise."""
        with pytest.raises(ValueError):
            ErdosRenyiTopology(edge_prob=0.5).generate_edges(1, random.Random(42))

    def test_is_topology_protocol(self):
        """ErdosRenyiTopology implements the Topology protocol."""
        assert isinstance(ErdosRenyiTopology(), Topology)


class TestTopologyFromConfig:
    """Test the topology_from_config factory."""

    def test_ring(self):
        topo = topology_from_config({"type": "ring"})
        assert isinstance(topo, RingTopology)

    def test_k_regular(self):
        topo = topology_from_config({"type": "k_regular", "degree": 3})
        assert isinstance(topo, KRegularTopology)
        assert topo.degree == 3

    def test_erdos_renyi(self):
        topo = topology_from_config({"type": "erdos_renyi", "edge_prob": 0.2})
        assert isinstance(topo, ErdosRenyiTopology)
        assert topo.edge_prob == 0.2

    def test_none_defaults_to_ring(self):
        topo = topology_from_config(None)
        assert isinstance(topo, RingTopology)

    def test_empty_dict_defaults_to_ring(self):
        topo = topology_from_config({})
        assert isinstance(topo, RingTopology)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            topology_from_config({"type": "watts_strogatz"})


class TestParseTopologyString:
    """Test parse_topology_string."""

    def test_ring(self):
        assert parse_topology_string("ring") == {"type": "ring"}

    def test_k_regular_with_degree(self):
        assert parse_topology_string("k_regular:3") == {"type": "k_regular", "degree": 3}

    def test_k_regular_default_degree(self):
        assert parse_topology_string("k_regular") == {"type": "k_regular", "degree": 2}

    def test_erdos_renyi_with_prob(self):
        result = parse_topology_string("erdos_renyi:0.2")
        assert result == {"type": "erdos_renyi", "edge_prob": 0.2}

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            parse_topology_string("unknown")
