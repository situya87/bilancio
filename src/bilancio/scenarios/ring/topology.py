"""Network topology generators for debt graph construction.

Provides a Topology protocol and concrete implementations for generating
directed debt edges. The ring topology is the default; k-regular and
Erdos-Renyi topologies enable richer network structures.

Topology generators produce only the graph structure (debtor->creditor edges).
Amounts and maturities are assigned separately by the compiler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DebtEdge:
    """A directed edge in the debt network: debtor owes creditor."""

    debtor: str  # e.g. "H3"
    creditor: str  # e.g. "H7"


@runtime_checkable
class Topology(Protocol):
    """Protocol for debt network topology generators."""

    def generate_edges(self, n_agents: int, rng: random.Random) -> list[DebtEdge]:
        """Generate directed debt edges for n agents.

        Agent IDs follow the convention H1, H2, ..., Hn.
        Amounts and maturities are NOT assigned here.

        Args:
            n_agents: Number of agents in the network
            rng: Random number generator for reproducibility

        Returns:
            List of DebtEdge instances defining the debt graph
        """
        ...


@dataclass(frozen=True)
class RingTopology:
    """Each agent owes the next: H{i} -> H{(i+1) % n + 1}. Degree=1.

    This reproduces the exact formula used in compiler.py line 146:
        from_agent = f"H{idx + 1}"
        to_agent = f"H{(idx + 1) % params.n_agents + 1}"
    """

    def generate_edges(self, n_agents: int, rng: random.Random) -> list[DebtEdge]:
        return [
            DebtEdge(
                debtor=f"H{i + 1}",
                creditor=f"H{(i + 1) % n_agents + 1}",
            )
            for i in range(n_agents)
        ]


@dataclass(frozen=True)
class KRegularTopology:
    """Each agent has exactly k outgoing edges. Uniform out-degree.

    Edges connect each agent to the next k agents in ring order,
    ensuring every agent has out-degree=k and in-degree=k.
    No self-loops are created.

    For degree=1, this is equivalent to RingTopology.
    """

    degree: int = 2

    def generate_edges(self, n_agents: int, rng: random.Random) -> list[DebtEdge]:
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        if self.degree >= n_agents:
            raise ValueError(
                f"degree ({self.degree}) must be < n_agents ({n_agents})"
            )

        edges: list[DebtEdge] = []
        for i in range(n_agents):
            debtor = f"H{i + 1}"
            for k in range(1, self.degree + 1):
                creditor_idx = (i + k) % n_agents
                creditor = f"H{creditor_idx + 1}"
                edges.append(DebtEdge(debtor=debtor, creditor=creditor))
        return edges


@dataclass(frozen=True)
class ErdosRenyiTopology:
    """Each directed pair connected with probability p.

    Generates a random directed graph where each possible directed edge
    (i -> j, i != j) exists independently with probability edge_prob.
    Enforces strong connectivity by adding bridge edges if needed.
    """

    edge_prob: float = 0.1

    def generate_edges(self, n_agents: int, rng: random.Random) -> list[DebtEdge]:
        if not 0 < self.edge_prob <= 1:
            raise ValueError(
                f"edge_prob must be in (0, 1], got {self.edge_prob}"
            )
        if n_agents < 2:
            raise ValueError(f"n_agents must be >= 2, got {n_agents}")

        # Generate random directed edges
        edge_set: set[tuple[int, int]] = set()
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and rng.random() < self.edge_prob:
                    edge_set.add((i, j))

        # Ensure strong connectivity by checking reachability
        # and adding bridge edges along a Hamiltonian cycle if needed
        edge_set = self._ensure_strongly_connected(edge_set, n_agents, rng)

        # Ensure every node has at least one outgoing edge
        for i in range(n_agents):
            has_outgoing = any(e[0] == i for e in edge_set)
            if not has_outgoing:
                # Connect to next agent in ring order
                j = (i + 1) % n_agents
                edge_set.add((i, j))

        return [
            DebtEdge(debtor=f"H{i + 1}", creditor=f"H{j + 1}")
            for i, j in sorted(edge_set)
        ]

    @staticmethod
    def _ensure_strongly_connected(
        edge_set: set[tuple[int, int]], n: int, rng: random.Random
    ) -> set[tuple[int, int]]:
        """Add minimum edges to ensure strong connectivity."""

        def _reachable(start: int, edges: set[tuple[int, int]]) -> set[int]:
            visited = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for i, j in edges:
                    if i == node and j not in visited:
                        visited.add(j)
                        frontier.append(j)
            return visited

        # Check if already strongly connected
        all_nodes = set(range(n))
        if n <= 1:
            return edge_set

        # Check forward reachability from node 0
        forward = _reachable(0, edge_set)
        if forward == all_nodes:
            # Check backward reachability (reverse edges)
            reverse_edges = {(j, i) for i, j in edge_set}
            backward = _reachable(0, reverse_edges)
            if backward == all_nodes:
                return edge_set

        # Not strongly connected: add a random Hamiltonian cycle
        nodes = list(range(n))
        rng.shuffle(nodes)
        for idx in range(n):
            i = nodes[idx]
            j = nodes[(idx + 1) % n]
            edge_set.add((i, j))

        return edge_set


def topology_from_config(config: dict[str, Any] | None = None) -> Topology:
    """Factory: create a Topology from a config dict.

    Examples:
        topology_from_config({"type": "ring"})  -> RingTopology()
        topology_from_config({"type": "k_regular", "degree": 3})  -> KRegularTopology(degree=3)
        topology_from_config({"type": "erdos_renyi", "edge_prob": 0.2})  -> ErdosRenyiTopology(edge_prob=0.2)
        topology_from_config(None)  -> RingTopology()  (default)
        topology_from_config({})  -> RingTopology()  (default)

    Args:
        config: Dict with 'type' key and optional type-specific params.
            If None or empty, defaults to RingTopology.

    Returns:
        A Topology instance
    """
    if not config:
        return RingTopology()

    topo_type = config.get("type", "ring")

    if topo_type == "ring":
        return RingTopology()
    elif topo_type == "k_regular":
        degree = config.get("degree", 2)
        return KRegularTopology(degree=int(degree))
    elif topo_type == "erdos_renyi":
        edge_prob = config.get("edge_prob", 0.1)
        return ErdosRenyiTopology(edge_prob=float(edge_prob))
    else:
        raise ValueError(f"Unknown topology type: {topo_type!r}")


def parse_topology_string(spec: str) -> dict[str, Any]:
    """Parse a topology spec string like 'ring', 'k_regular:3', 'erdos_renyi:0.1'.

    Args:
        spec: Topology specification string

    Returns:
        Config dict suitable for topology_from_config()
    """
    parts = spec.strip().split(":")
    topo_type = parts[0]

    if topo_type == "ring":
        return {"type": "ring"}
    elif topo_type == "k_regular":
        degree = int(parts[1]) if len(parts) > 1 else 2
        return {"type": "k_regular", "degree": degree}
    elif topo_type == "erdos_renyi":
        edge_prob = float(parts[1]) if len(parts) > 1 else 0.1
        return {"type": "erdos_renyi", "edge_prob": edge_prob}
    else:
        raise ValueError(f"Unknown topology spec: {spec!r}")


__all__ = [
    "DebtEdge",
    "Topology",
    "RingTopology",
    "KRegularTopology",
    "ErdosRenyiTopology",
    "topology_from_config",
    "parse_topology_string",
]
