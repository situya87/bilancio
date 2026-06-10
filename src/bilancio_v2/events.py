"""Typed event journal for the v2 kernel.

Every state change in the v2 ledger is recorded as an :class:`Event`.
Events are immutable once appended; the journal is append-only except for
explicit checkpoint/rollback used by fail-fast settlement semantics.

``Event.to_dict()`` produces the exact dict shape emitted by the existing
engines (``{"kind", "day", "phase", **payload}``) so all downstream
analysis/export tooling — and the parity harness — work unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str
    day: int
    # "setup" | "simulation" | None — informational subsystem events
    # (ratings, lending decisions) carry no phase, matching the existing engine.
    phase: str | None
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        if self.phase is None:
            return {"kind": self.kind, "day": self.day, **self.data}
        return {"kind": self.kind, "day": self.day, "phase": self.phase, **self.data}


@dataclass
class EventJournal:
    """Append-only event log with explicit truncation for atomic rollback."""

    _events: list[Event] = field(default_factory=list)

    def append(self, kind: str, *, day: int, phase: str | None, **data: Any) -> Event:
        event = Event(kind=kind, day=day, phase=phase, data=MappingProxyType(dict(data)))
        self._events.append(event)
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def truncate(self, length: int) -> None:
        """Roll the journal back to ``length`` events (fail-fast rollback)."""
        del self._events[length:]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def on_day(self, day: int, kind: str | None = None) -> list[Event]:
        return [e for e in self._events if e.day == day and (kind is None or e.kind == kind)]
