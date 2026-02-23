"""Contagion analysis — classify defaults and trace cascade paths.

Extends the basic ``cascade_fraction`` in ``metrics.py`` with richer
default classification (primary vs secondary), contagion timing, and
default dependency graphs.

All functions consume the standard event log (list of dicts) produced by
the simulation engine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

Event = dict[str, Any]
AgentId = str


@dataclass
class DefaultRecord:
    """Record of a single agent default with classification."""

    agent_id: str
    day: int
    shortfall: Decimal = Decimal(0)
    is_primary: bool = True
    upstream_defaulters: list[str] = field(default_factory=list)


def _agent_from_event(e: Event) -> str | None:
    """Extract agent ID from an AgentDefaulted event."""
    for key in ("agent", "frm"):
        val = e.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return None


def classify_defaults(events: list[Event]) -> list[DefaultRecord]:
    """Classify each default as primary or secondary (contagion).

    A default is **secondary** if any of the agent's debtors defaulted
    before it, meaning the agent lost expected inflows.  Otherwise it is
    **primary** (the agent failed on its own).

    Returns list of DefaultRecord in chronological order.
    """
    # Build obligation graph: creditor -> set of debtors
    creditor_to_debtors: dict[str, set[str]] = defaultdict(set)
    for e in events:
        if e.get("kind") == "PayableCreated":
            debtor = e.get("debtor") or e.get("from")
            creditor = e.get("creditor") or e.get("to")
            if debtor and creditor:
                creditor_to_debtors[str(creditor)].add(str(debtor))

    # Process defaults in order
    records: list[DefaultRecord] = []
    defaulted_so_far: set[str] = set()

    for e in events:
        if e.get("kind") != "AgentDefaulted":
            continue
        agent = _agent_from_event(e)
        if not agent or agent in defaulted_so_far:
            continue

        day = int(e.get("day", 0))
        shortfall = Decimal(str(e.get("shortfall", 0)))
        debtors = creditor_to_debtors.get(agent, set())
        upstream = sorted(debtors & defaulted_so_far)
        is_primary = len(upstream) == 0

        records.append(DefaultRecord(
            agent_id=agent,
            day=day,
            shortfall=shortfall,
            is_primary=is_primary,
            upstream_defaulters=upstream,
        ))
        defaulted_so_far.add(agent)

    return records


def default_counts_by_type(events: list[Event]) -> dict[str, int]:
    """Count primary vs secondary defaults.

    Returns:
        {"primary": int, "secondary": int, "total": int}
    """
    records = classify_defaults(events)
    primary = sum(1 for r in records if r.is_primary)
    secondary = sum(1 for r in records if not r.is_primary)
    return {"primary": primary, "secondary": secondary, "total": len(records)}


def contagion_by_day(events: list[Event]) -> dict[int, dict[str, int]]:
    """Defaults per day, split by primary/secondary.

    Returns:
        {day: {"primary": int, "secondary": int}}
    """
    records = classify_defaults(events)
    by_day: dict[int, dict[str, int]] = defaultdict(
        lambda: {"primary": 0, "secondary": 0}
    )
    for r in records:
        key = "primary" if r.is_primary else "secondary"
        by_day[r.day][key] += 1
    return dict(by_day)


def time_to_contagion(events: list[Event]) -> int | None:
    """Days between first primary default and first secondary default.

    Returns None if there are no secondary defaults.
    """
    records = classify_defaults(events)
    first_primary_day: int | None = None
    first_secondary_day: int | None = None

    for r in records:
        if r.is_primary and first_primary_day is None:
            first_primary_day = r.day
        if not r.is_primary and first_secondary_day is None:
            first_secondary_day = r.day

    if first_primary_day is None or first_secondary_day is None:
        return None
    return first_secondary_day - first_primary_day


def default_dependency_graph(
    events: list[Event],
) -> dict[str, list[str]]:
    """Build a graph of default dependencies.

    Returns mapping: defaulted_agent -> list of upstream defaulters
    that contributed to this agent's default.  Primary defaults map
    to an empty list.
    """
    records = classify_defaults(events)
    return {r.agent_id: r.upstream_defaulters for r in records}
