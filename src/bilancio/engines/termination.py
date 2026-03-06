"""Termination policy framework for simulation stop decisions.

Plan 051: Centralizes stop-logic that was previously duplicated in
simulation.py (run_until_stable) and ui/run.py (run_until_stable_mode).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bilancio.domain.instruments.base import InstrumentKind

if TYPE_CHECKING:
    from bilancio.engines.system import System

# ---------------------------------------------------------------------------
# Event sets used to classify day activity
# ---------------------------------------------------------------------------

IMPACT_EVENTS = {
    "PayableSettled",
    "DeliveryObligationSettled",
    "InterbankCleared",
    "InterbankOvernightCreated",
    "InterbankAuctionTrade",
}

DEFAULT_EVENTS = {
    "ObligationDefaulted",
    "ObligationWrittenOff",
    "AgentDefaulted",
    "BankDefaultCBFreeze",
}


# ---------------------------------------------------------------------------
# Day-level helpers
# ---------------------------------------------------------------------------


def _impacted_today(system: System, day: int) -> int:
    """Count impactful events that occurred on *day*."""
    return sum(
        1 for e in system.state.events if e.get("day") == day and e.get("kind") in IMPACT_EVENTS
    )


def _defaults_today(system: System, day: int) -> int:
    """Count default events that occurred on *day*."""
    return sum(
        1 for e in system.state.events if e.get("day") == day and e.get("kind") in DEFAULT_EVENTS
    )


def _has_open_obligations(system: System) -> bool:
    """Return True if any payable or delivery obligation contracts remain."""
    for c in system.state.contracts.values():
        if c.kind in (InstrumentKind.PAYABLE, InstrumentKind.DELIVERY_OBLIGATION):
            return True
    return False


# ---------------------------------------------------------------------------
# StopReason enum
# ---------------------------------------------------------------------------


class StopReason(enum.Enum):
    """Why the simulation loop terminated."""

    STABILITY_REACHED = "stability_reached"
    MAX_DAYS_REACHED = "max_days_reached"
    FATAL_ERROR = "fatal_error"
    USER_STOP = "user_stop"


# ---------------------------------------------------------------------------
# StabilitySnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilitySnapshot:
    """Per-day snapshot of stability-related counters."""

    day: int
    consecutive_quiet: int
    consecutive_no_defaults: int
    has_open_obligations: bool
    impacted_count: int
    default_count: int


def compute_stability_snapshot(
    system: System,
    day: int,
    consecutive_quiet: int,
    consecutive_no_defaults: int,
) -> StabilitySnapshot:
    """Build a StabilitySnapshot from current system state."""
    return StabilitySnapshot(
        day=day,
        consecutive_quiet=consecutive_quiet,
        consecutive_no_defaults=consecutive_no_defaults,
        has_open_obligations=_has_open_obligations(system),
        impacted_count=_impacted_today(system, day),
        default_count=_defaults_today(system, day),
    )


# ---------------------------------------------------------------------------
# TerminationPolicy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TerminationPolicy(Protocol):
    """Decides whether the simulation loop should stop."""

    def evaluate(
        self,
        snapshot: StabilitySnapshot,
        quiet_days: int,
        rollover_enabled: bool,
    ) -> StopReason | None:
        """Return a StopReason to stop, or None to continue."""
        ...


# ---------------------------------------------------------------------------
# LegacyTerminationPolicy — exact reproduction of inline logic
# ---------------------------------------------------------------------------


@dataclass
class LegacyTerminationPolicy:
    """Reproduces the pre-Plan-051 inline stop logic exactly.

    Non-rollover: stop when consecutive_quiet >= quiet_days AND no open obligations.
    Rollover:     stop when consecutive_no_defaults >= quiet_days.
    """

    def evaluate(
        self,
        snapshot: StabilitySnapshot,
        quiet_days: int,
        rollover_enabled: bool,
    ) -> StopReason | None:
        if rollover_enabled:
            if snapshot.consecutive_no_defaults >= quiet_days:
                return StopReason.STABILITY_REACHED
        else:
            if snapshot.consecutive_quiet >= quiet_days and not snapshot.has_open_obligations:
                return StopReason.STABILITY_REACHED
        return None
