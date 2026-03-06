"""Structured result returned by run_until_stable().

Plan 051: Replaces the bare list[DayReport] return value with a rich
dataclass that carries stop reason, stability snapshots, and post-loop
metadata while preserving backward compatibility via the `reports` field.

Backward compatibility: ``len(result)``, ``result[i]``, and iteration
all forward to ``result.reports`` so that existing code treating the
return value as ``list[DayReport]`` keeps working.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, overload

from bilancio.engines.termination import StabilitySnapshot, StopReason


@dataclass
class SimulationResult:
    """Structured output of a simulation run.

    Attributes:
        reports: Per-day DayReport list (backward compatible).
        stop_reason: Why the simulation loop terminated.
        stop_day: The day number when the loop stopped.
        stability_snapshots: Per-day stability snapshots collected during the run.
        winddown_days: Extra days used for bank loan wind-down (0 if none).
        final_cb_settlement: Result dict from final CB settlement, or None.
    """

    reports: list[Any]  # list[DayReport] — Any to avoid circular import
    stop_reason: StopReason
    stop_day: int
    stability_snapshots: list[StabilitySnapshot] = field(default_factory=list)
    winddown_days: int = 0
    final_cb_settlement: dict[str, Any] | None = None

    # -- list-like forwarding for backward compatibility --

    def __len__(self) -> int:
        return len(self.reports)

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> list[Any]: ...
    def __getitem__(self, index: int | slice) -> Any:
        return self.reports[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.reports)
