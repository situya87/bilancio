"""Balance sheet snapshot capture and diff computation."""

from __future__ import annotations

from dataclasses import dataclass, field

from bilancio.analysis.balances import AgentBalance, agent_balance
from bilancio.engines.system import System


@dataclass
class InstrumentDelta:
    """Change in a single instrument kind for one agent."""

    kind: str
    previous: int
    current: int
    delta: int  # current - previous


@dataclass
class AgentDiff:
    """Balance sheet diff for one agent between two time points."""

    agent_id: str
    asset_deltas: list[InstrumentDelta] = field(default_factory=list)
    liability_deltas: list[InstrumentDelta] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if any delta is non-zero."""
        return any(d.delta != 0 for d in self.asset_deltas) or any(
            d.delta != 0 for d in self.liability_deltas
        )


def capture_snapshot(system: System) -> dict[str, AgentBalance]:
    """Capture balance for every agent in the system.

    Returns a dict mapping agent_id to AgentBalance.
    """
    snapshot: dict[str, AgentBalance] = {}
    for agent_id in system.state.agents:
        snapshot[agent_id] = agent_balance(system, agent_id)
    return snapshot


def compute_diff(
    before: dict[str, AgentBalance],
    after: dict[str, AgentBalance],
) -> dict[str, AgentDiff]:
    """Compare two snapshots and return per-agent diffs.

    Handles agents added between snapshots (all values treated as new).
    """
    all_agent_ids = set(before.keys()) | set(after.keys())
    diffs: dict[str, AgentDiff] = {}

    for agent_id in all_agent_ids:
        before_bal = before.get(agent_id)
        after_bal = after.get(agent_id)

        asset_deltas = _compute_side_deltas(
            before_bal.assets_by_kind if before_bal else {},
            after_bal.assets_by_kind if after_bal else {},
        )
        liability_deltas = _compute_side_deltas(
            before_bal.liabilities_by_kind if before_bal else {},
            after_bal.liabilities_by_kind if after_bal else {},
        )

        diffs[agent_id] = AgentDiff(
            agent_id=agent_id,
            asset_deltas=asset_deltas,
            liability_deltas=liability_deltas,
        )

    return diffs


def _compute_side_deltas(
    before: dict[str, int],
    after: dict[str, int],
) -> list[InstrumentDelta]:
    """Compute deltas for one side (assets or liabilities)."""
    all_kinds = set(before.keys()) | set(after.keys())
    deltas: list[InstrumentDelta] = []

    for kind in sorted(all_kinds):
        prev = before.get(kind, 0)
        curr = after.get(kind, 0)
        deltas.append(
            InstrumentDelta(kind=kind, previous=prev, current=curr, delta=curr - prev)
        )

    return deltas
