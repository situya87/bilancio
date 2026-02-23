"""Network analysis — centrality, systemic importance, and structure.

Extends ``network.py`` (which extracts graph data) with analytical metrics:
node centrality, clustering, and systemic importance scores.

Operates on ``NetworkSnapshot`` objects from ``bilancio.analysis.network``
or directly on adjacency representations built from the event log.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

Event = dict[str, Any]
AgentId = str


def build_obligation_adjacency(
    events: list[Event],
) -> dict[AgentId, dict[AgentId, Decimal]]:
    """Build weighted directed adjacency from PayableCreated events.

    Returns:
        {debtor: {creditor: total_amount_owed}}
    """
    adj: dict[AgentId, dict[AgentId, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: Decimal(0))
    )
    for e in events:
        if e.get("kind") != "PayableCreated":
            continue
        debtor = e.get("debtor") or e.get("from")
        creditor = e.get("creditor") or e.get("to")
        amt = Decimal(str(e.get("amount", 0)))
        if debtor and creditor:
            adj[str(debtor)][str(creditor)] += amt
    return dict(adj)


def node_degree(
    events: list[Event],
) -> dict[AgentId, dict[str, int]]:
    """Compute in-degree and out-degree for each node.

    In the obligation graph: out-degree = number of creditors (debts owed),
    in-degree = number of debtors (debts receivable).

    Returns:
        {agent_id: {"in_degree": int, "out_degree": int}}
    """
    adj = build_obligation_adjacency(events)
    all_agents: set[str] = set()
    out_deg: dict[str, int] = defaultdict(int)
    in_deg: dict[str, int] = defaultdict(int)

    for debtor, creditors in adj.items():
        all_agents.add(debtor)
        out_deg[debtor] = len(creditors)
        for creditor in creditors:
            all_agents.add(creditor)
            in_deg[creditor] += 1

    return {
        a: {"in_degree": in_deg.get(a, 0), "out_degree": out_deg.get(a, 0)}
        for a in all_agents
    }


def weighted_degree(
    events: list[Event],
) -> dict[AgentId, dict[str, Decimal]]:
    """Compute weighted in/out degree (total amount owed/receivable).

    Returns:
        {agent_id: {"total_owed": Decimal, "total_receivable": Decimal}}
    """
    adj = build_obligation_adjacency(events)
    all_agents: set[str] = set()
    owed: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    receivable: dict[str, Decimal] = defaultdict(lambda: Decimal(0))

    for debtor, creditors in adj.items():
        all_agents.add(debtor)
        for creditor, amount in creditors.items():
            all_agents.add(creditor)
            owed[debtor] += amount
            receivable[creditor] += amount

    return {
        a: {
            "total_owed": owed.get(a, Decimal(0)),
            "total_receivable": receivable.get(a, Decimal(0)),
        }
        for a in all_agents
    }


def betweenness_centrality(
    events: list[Event],
) -> dict[AgentId, float]:
    """Approximate betweenness centrality using unweighted shortest paths.

    Uses Brandes' algorithm on the obligation graph (ignoring weights).
    Normalized by (n-1)(n-2) for directed graphs.

    Returns:
        {agent_id: centrality_score} where 0 <= score <= 1
    """
    adj = build_obligation_adjacency(events)
    # Build unweighted adjacency for BFS
    neighbors: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for debtor, creditors in adj.items():
        all_nodes.add(debtor)
        for creditor in creditors:
            all_nodes.add(creditor)
            neighbors[debtor].add(creditor)

    n = len(all_nodes)
    if n <= 2:
        return {a: 0.0 for a in all_nodes}

    cb: dict[str, float] = {a: 0.0 for a in all_nodes}

    for s in all_nodes:
        # BFS from s
        stack: list[str] = []
        pred: dict[str, list[str]] = {a: [] for a in all_nodes}
        sigma: dict[str, int] = {a: 0 for a in all_nodes}
        sigma[s] = 1
        dist: dict[str, int] = {a: -1 for a in all_nodes}
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in neighbors.get(v, set()):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta: dict[str, float] = {a: 0.0 for a in all_nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # Normalize
    norm = (n - 1) * (n - 2)
    if norm > 0:
        for a in cb:
            cb[a] /= norm

    return cb


def systemic_importance(
    events: list[Event],
) -> list[dict[str, Any]]:
    """Rank agents by systemic importance.

    Combines weighted degree (total obligations) with betweenness
    centrality.  The score is a simple composite:
        score = 0.5 * normalized_total_obligations + 0.5 * betweenness

    Returns list sorted by score descending:
        [{"agent_id": str, "total_obligations": Decimal,
          "betweenness": float, "score": float}]
    """
    w_deg = weighted_degree(events)
    bc = betweenness_centrality(events)

    # Normalize total obligations
    max_obl = max(
        (v["total_owed"] + v["total_receivable"] for v in w_deg.values()),
        default=Decimal(0),
    )

    results: list[dict[str, Any]] = []
    for agent_id in w_deg:
        total_obl = w_deg[agent_id]["total_owed"] + w_deg[agent_id]["total_receivable"]
        norm_obl = float(total_obl / max_obl) if max_obl > 0 else 0.0
        b = bc.get(agent_id, 0.0)
        score = 0.5 * norm_obl + 0.5 * b
        results.append({
            "agent_id": agent_id,
            "total_obligations": total_obl,
            "betweenness": b,
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
