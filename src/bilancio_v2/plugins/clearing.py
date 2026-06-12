"""Clearinghouse netting phase (Subphase B_Clearing, Plan 059 stage 1).

Each day, before cash settlement, the clearinghouse collects the payables
due that day, aggregates them into a debtor→creditor edge graph, and
extinguishes offsetting obligations without any cash movement — first
bilaterally, then along directed cycles (the maximal in-place netting that
preserves every agent's net position exactly). Per-edge cancellations are
allocated back to individual payables pro-rata with largest-remainder
rounding. Only residual net obligations reach :class:`SettlementPhase`.

Everything is deterministic: agents are visited in sorted-id order, edges
in sorted (debtor, creditor) order, and payables in sorted payable-id
order, so the same input always produces the identical event sequence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from bilancio_v2.ledger import ZERO, InvariantViolation, Ledger, Payable
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.subsystem_config import CleanClearingConfig

Edge = tuple[str, str]


@dataclass(frozen=True)
class ClearingPhase:
    name = "SubphaseB_Clearing"
    config: CleanClearingConfig

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        due = collect_due_payables(ledger)
        if not due:
            return False

        balances_before = _balance_totals(ledger)
        gross_face_due = sum((payable.amount for payable in due), ZERO)
        edge_payables = group_payables_by_edge(due)
        weights = {edge: sum((payable.amount for payable in payables), ZERO) for edge, payables in edge_payables.items()}

        reductions: defaultdict[Edge, Decimal] = defaultdict(lambda: ZERO)
        for edge, cancelled in cancel_bilateral(weights).items():
            reductions[edge] += cancelled
        for edge, cancelled in cancel_cycles(weights).items():
            reductions[edge] += cancelled

        allocations = allocate_edge_reductions(edge_payables, dict(reductions))

        face_extinguished = ZERO
        n_fully_netted = 0
        extinguished_as_debtor: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
        extinguished_as_creditor: defaultdict[str, Decimal] = defaultdict(lambda: ZERO)
        for payable, reduction in allocations:
            ledger.net_payable(payable, reduction)
            face_extinguished += reduction
            extinguished_as_debtor[payable.debtor] += reduction
            extinguished_as_creditor[payable.creditor] += reduction
            if payable.settled:
                n_fully_netted += 1
                if ctx.rollover_enabled:
                    ledger.netted_rollover_queue.append(
                        (payable.debtor, payable.creditor, payable.netted_amount, payable.maturity_distance)
                    )

        edge_total = sum(reductions.values(), ZERO)
        if face_extinguished != edge_total:
            raise InvariantViolation(f"netting allocation mismatch: payable reductions {face_extinguished} != edge cancellations {edge_total}")
        for agent_id in sorted(set(extinguished_as_debtor) | set(extinguished_as_creditor)):
            if extinguished_as_debtor[agent_id] != extinguished_as_creditor[agent_id]:
                raise InvariantViolation(
                    f"netting changed the net position of {agent_id}: "
                    f"debtor-side {extinguished_as_debtor[agent_id]} != creditor-side {extinguished_as_creditor[agent_id]}"
                )
        if _balance_totals(ledger) != balances_before:
            raise InvariantViolation("netting moved cash, reserves, or deposits")

        if face_extinguished > ZERO:
            ledger.log(
                "ClearingExecuted",
                gross_face_due=gross_face_due,
                face_extinguished=face_extinguished,
                residual_face=gross_face_due - face_extinguished,
                n_payables_affected=len(allocations),
                n_fully_netted=n_fully_netted,
            )
        return face_extinguished > ZERO


def collect_due_payables(ledger: Ledger) -> list[Payable]:
    return [
        payable
        for payable in ledger.payables
        if not payable.settled
        and payable.due_day == ledger.day
        and payable.debtor not in ledger.defaulted_agent_ids
        and payable.creditor not in ledger.defaulted_agent_ids
        # Pledged collateral (Plan 060) is fenced off from netting: its
        # proceeds belong to the clearinghouse, not the creditor of record.
        and payable.pledged_to is None
    ]


def group_payables_by_edge(due: list[Payable]) -> dict[Edge, list[Payable]]:
    grouped: defaultdict[Edge, list[Payable]] = defaultdict(list)
    for payable in due:
        grouped[(payable.debtor, payable.creditor)].append(payable)
    return {edge: sorted(payables, key=lambda payable: payable.id) for edge, payables in grouped.items()}


def cancel_bilateral(weights: dict[Edge, Decimal]) -> dict[Edge, Decimal]:
    """Cancel offsetting flow on each unordered agent pair, mutating ``weights``."""
    reductions: dict[Edge, Decimal] = {}
    for edge in sorted(weights):
        debtor, creditor = edge
        if debtor >= creditor:
            continue
        reverse = (creditor, debtor)
        cancelled = min(weights.get(edge, ZERO), weights.get(reverse, ZERO))
        if cancelled <= ZERO:
            continue
        weights[edge] -= cancelled
        weights[reverse] -= cancelled
        reductions[edge] = cancelled
        reductions[reverse] = cancelled
    return reductions


def cancel_cycles(weights: dict[Edge, Decimal]) -> dict[Edge, Decimal]:
    """Cancel directed cycles until the positive-weight graph is acyclic, mutating ``weights``."""
    reductions: defaultdict[Edge, Decimal] = defaultdict(lambda: ZERO)
    while True:
        cycle = find_cycle({edge for edge, weight in weights.items() if weight > ZERO})
        if cycle is None:
            return dict(reductions)
        cycle_edges = [(cycle[index], cycle[(index + 1) % len(cycle)]) for index in range(len(cycle))]
        cancelled = min(weights[edge] for edge in cycle_edges)
        for edge in cycle_edges:
            weights[edge] -= cancelled
            reductions[edge] += cancelled


def find_cycle(edges: set[Edge]) -> list[str] | None:
    """Return the first directed cycle found by DFS (sorted-agent-id order), as a node list."""
    adjacency: defaultdict[str, list[str]] = defaultdict(list)
    for debtor, creditor in edges:
        adjacency[debtor].append(creditor)
    for neighbors in adjacency.values():
        neighbors.sort()

    finished: set[str] = set()
    for start in sorted(adjacency):
        if start in finished:
            continue
        path = [start]
        path_index = {start: 0}
        stack = [iter(adjacency[start])]
        while stack:
            advanced = False
            for neighbor in stack[-1]:
                if neighbor in path_index:
                    return path[path_index[neighbor] :]
                if neighbor in finished or neighbor not in adjacency:
                    continue
                path.append(neighbor)
                path_index[neighbor] = len(path) - 1
                stack.append(iter(adjacency[neighbor]))
                advanced = True
                break
            if not advanced:
                node = path.pop()
                del path_index[node]
                finished.add(node)
                stack.pop()
    return None


def allocate_edge_reductions(
    edge_payables: dict[Edge, list[Payable]],
    reductions: dict[Edge, Decimal],
) -> list[tuple[Payable, Decimal]]:
    allocations: list[tuple[Payable, Decimal]] = []
    for edge in sorted(reductions):
        reduction = reductions[edge]
        if reduction <= ZERO:
            continue
        allocations.extend(allocate_single_edge(edge_payables[edge], reduction))
    return allocations


def allocate_single_edge(payables: list[Payable], reduction: Decimal) -> list[tuple[Payable, Decimal]]:
    """Split an edge cancellation across its payables pro-rata by amount.

    Largest-remainder rounding in whole Decimal units (ties broken by sorted
    payable id) conserves the per-edge total exactly.
    """
    total = sum((payable.amount for payable in payables), ZERO)
    if reduction > total:
        raise InvariantViolation(f"edge reduction {reduction} exceeds edge face {total}")
    if reduction == total:
        return [(payable, payable.amount) for payable in payables]

    by_id = {payable.id: payable for payable in payables}
    shares: dict[str, Decimal] = {}
    remainders: list[tuple[Decimal, str]] = []
    allocated = ZERO
    for payable in payables:
        exact = reduction * payable.amount / total
        base = exact.to_integral_value(rounding=ROUND_FLOOR)
        shares[payable.id] = base
        allocated += base
        remainders.append((exact - base, payable.id))

    leftover = reduction - allocated
    order = sorted(remainders, key=lambda item: (-item[0], item[1]))
    for _remainder, payable_id in order:
        if leftover < Decimal("1"):
            break
        if shares[payable_id] + Decimal("1") <= by_id[payable_id].amount:
            shares[payable_id] += Decimal("1")
            leftover -= Decimal("1")
    if leftover > ZERO:
        # Non-integer residue (non-integer faces): spread across payables
        # in tie order, bounded by each payable's remaining capacity.
        for _remainder, payable_id in order:
            if leftover <= ZERO:
                break
            take = min(leftover, by_id[payable_id].amount - shares[payable_id])
            if take > ZERO:
                shares[payable_id] += take
                leftover -= take
    if leftover != ZERO:
        raise InvariantViolation(f"edge reduction left {leftover} unallocated")

    return [(payable, shares[payable.id]) for payable in payables if shares[payable.id] > ZERO]


def _balance_totals(ledger: Ledger) -> tuple[Decimal, Decimal, Decimal]:
    return (
        sum(ledger.cash.values(), ZERO),
        sum(ledger.reserves.values(), ZERO),
        sum(ledger.deposits.values(), ZERO),
    )
