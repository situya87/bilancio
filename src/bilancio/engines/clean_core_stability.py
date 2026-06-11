"""Stability and termination bookkeeping for clean-core scenario runs."""

from __future__ import annotations

from bilancio.engines.clean_core_types import (
    CleanScenarioRuntime,
    CleanStabilityTracker,
    CleanState,
)
from bilancio.engines.termination import IMPACT_EVENTS, StabilitySnapshot

CLEAN_CORE_DEFAULT_EVENTS = {
    "ObligationDefaulted",
    "ObligationWrittenOff",
    "AgentDefaulted",
    "BankDefaultCBFreeze",
}


def has_pending_future_obligations(state: CleanState) -> bool:
    """Return whether any unsettled obligation remains after the current day."""
    return (
        any(not payable.settled and payable.due_day > state.day for payable in state.payables)
        or any(
            not obligation.settled and obligation.due_day > state.day
            for obligation in state.delivery_obligations
        )
    )


def update_runtime_stability(
    runtime: CleanScenarioRuntime,
    *,
    day: int,
    impactful: bool,
    quiet_days: int,
    tracker: CleanStabilityTracker,
) -> bool:
    state = runtime.state
    defaults = defaults_on_day(state, day)
    if not impactful:
        tracker.consecutive_quiet += 1
    else:
        tracker.consecutive_quiet = 0
    if defaults == 0:
        tracker.consecutive_no_defaults += 1
    else:
        tracker.consecutive_no_defaults = 0

    if state.rollover_enabled:
        state.quiet_days = tracker.consecutive_no_defaults
        stable_today = state.quiet_days >= quiet_days
    elif day == 0:
        if not impactful:
            state.quiet_days += 1
        stable_today = False
    else:
        pending_future = has_pending_future_obligations(state)
        state.quiet_days = state.quiet_days + 1 if not impactful else 0
        stable_today = state.quiet_days >= quiet_days and not pending_future

    tracker.snapshots.append(
        clean_stability_snapshot(
            state,
            day,
            consecutive_quiet=(
                tracker.consecutive_quiet if state.rollover_enabled else state.quiet_days
            ),
            consecutive_no_defaults=tracker.consecutive_no_defaults,
        )
    )
    return stable_today


def clean_stability_snapshot(
    state: CleanState,
    day: int,
    *,
    consecutive_quiet: int,
    consecutive_no_defaults: int,
) -> StabilitySnapshot:
    return StabilitySnapshot(
        day=day,
        consecutive_quiet=consecutive_quiet,
        consecutive_no_defaults=consecutive_no_defaults,
        has_open_obligations=has_pending_future_obligations(state),
        impacted_count=impacted_on_day(state, day),
        default_count=defaults_on_day(state, day),
    )


def defaults_on_day(state: CleanState, day: int) -> int:
    return sum(
        1
        for event in state.events
        if event.get("day") == day and event.get("kind") in CLEAN_CORE_DEFAULT_EVENTS
    )


def impacted_on_day(state: CleanState, day: int) -> int:
    return sum(
        1
        for event in state.events
        if event.get("day") == day and event.get("kind") in IMPACT_EVENTS
    )
