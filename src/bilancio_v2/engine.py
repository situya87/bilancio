"""The v2 simulation engine: scenario preparation and the daily cycle.

The engine is deliberately small. It owns the day loop, the phase order,
the stability stop rule, and the daily invariant check — everything else
(scheduled actions, settlement, interbank clearing) is a phase plugin.

Scenarios are loaded through the existing ``bilancio.config`` schema, so
every YAML file that runs on the existing engines runs here unchanged.
Features not yet rebuilt (dealer, jurisdictions, action specs) are
rejected explicitly at preparation time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bilancio.config.models import ScenarioConfig
from bilancio.engines.clean_core_config import build_lender_config, build_rating_config
from bilancio.engines.termination import (
    DEFAULT_EVENTS,
    IMPACT_EVENTS,
    StabilitySnapshot,
    StopReason,
)
from bilancio_v2.actions import apply_action
from bilancio_v2.ledger import Ledger
from bilancio_v2.plugins.base import PhasePlugin, RunContext
from bilancio_v2.plugins.interbank import InterbankPhase
from bilancio_v2.plugins.lending import LendingPhase
from bilancio_v2.plugins.rating import RatingPhase
from bilancio_v2.plugins.settlement import SettlementPhase
from bilancio_v2.policy import CapabilityMatrix


class UnsupportedScenarioError(NotImplementedError):
    """The scenario uses a subsystem the v2 kernel has not rebuilt yet."""


class ScheduledActionsPhase:
    """Subphase B1: user-scheduled actions for the current day."""

    name = "SubphaseB1"

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        for index, action in enumerate(ledger.scheduled_actions_by_day.get(ledger.day, [])):
            apply_action(ledger, action, index=index, setup=False)
        return False


@dataclass(frozen=True)
class Runtime:
    ledger: Ledger
    ctx: RunContext
    phases: tuple[PhasePlugin, ...]


@dataclass
class StabilityTracker:
    consecutive_quiet: int = 0
    consecutive_no_defaults: int = 0
    snapshots: list[StabilitySnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class RunResult:
    ledger: Ledger
    final_day: int
    reached_stable: bool
    stop_reason: StopReason
    stability_snapshots: tuple[StabilitySnapshot, ...] = ()

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.ledger.journal.as_dicts()

    @property
    def stop_day(self) -> int:
        return self.final_day


def _unsupported_reason(config: ScenarioConfig) -> str | None:
    if config.dealer is not None:
        return "dealer subsystem"
    if config.balanced_dealer is not None:
        return "balanced dealer subsystem"
    if config.action_specs:
        return "action specs"
    if config.jurisdictions:
        return "jurisdictions"
    if config.fx_rates:
        return "fx rates"
    return None


def prepare_scenario(config: ScenarioConfig) -> Runtime:
    """Apply scenario setup and return a runtime that can be stepped day by day."""
    reason = _unsupported_reason(config)
    if reason is not None:
        raise UnsupportedScenarioError(f"v2 kernel does not support {reason} yet")

    policy = CapabilityMatrix.default()
    if config.policy_overrides is not None:
        policy = policy.with_mop_overrides(config.policy_overrides.mop_rank)

    ledger = Ledger()
    for agent_spec in config.agents:
        ledger.register_agent(agent_spec.id, str(agent_spec.kind), agent_spec.name)

    for scheduled in config.scheduled_actions:
        ledger.scheduled_actions_by_day.setdefault(scheduled.day, []).append(scheduled.action)

    for index, action in enumerate(config.initial_actions):
        apply_action(ledger, action, index=index, setup=True)
    ledger.cb_reserves_initial = ledger.cb_reserves_outstanding
    ledger.estimate_logging_enabled = config.run.estimate_logging
    ledger.check_invariants()

    # The YAML→subsystem-config mapping is shared with the existing engine
    # so both kernels always read a scenario identically.
    rating_config = build_rating_config(config)
    lender_config = build_lender_config(config)

    ctx = RunContext(
        policy=policy,
        default_mode=config.run.default_handling,
        rollover_enabled=config.run.rollover_enabled,
    )
    phases: list[PhasePlugin] = [ScheduledActionsPhase()]
    if rating_config is not None:
        phases.append(RatingPhase(config=rating_config))
    if lender_config is not None:
        phases.append(LendingPhase(config=lender_config))
    phases.append(SettlementPhase())
    phases.append(InterbankPhase())
    return Runtime(ledger=ledger, ctx=ctx, phases=tuple(phases))


def run_day(runtime: Runtime, day: int) -> bool:
    """Run a single day and return whether it had impactful settlement events."""
    ledger = runtime.ledger
    ledger.day = day
    ledger.log("PhaseA")
    ledger.log("PhaseB")
    impactful = False
    for phase in runtime.phases:
        ledger.log(phase.name)
        impactful = phase.run(ledger, runtime.ctx) or impactful
    ledger.check_invariants()
    return impactful


def has_pending_future_obligations(ledger: Ledger) -> bool:
    return any(not payable.settled and payable.due_day > ledger.day for payable in ledger.payables) or any(
        not obligation.settled and obligation.due_day > ledger.day for obligation in ledger.delivery_obligations
    )


def defaults_on_day(ledger: Ledger, day: int) -> int:
    return sum(1 for event in ledger.journal if event.day == day and event.kind in DEFAULT_EVENTS)


def impacted_on_day(ledger: Ledger, day: int) -> int:
    return sum(1 for event in ledger.journal if event.day == day and event.kind in IMPACT_EVENTS)


def update_stability(
    runtime: Runtime,
    *,
    day: int,
    impactful: bool,
    quiet_days: int,
    tracker: StabilityTracker,
) -> bool:
    ledger = runtime.ledger
    defaults = defaults_on_day(ledger, day)
    if not impactful:
        tracker.consecutive_quiet += 1
    else:
        tracker.consecutive_quiet = 0
    if defaults == 0:
        tracker.consecutive_no_defaults += 1
    else:
        tracker.consecutive_no_defaults = 0

    if runtime.ctx.rollover_enabled:
        # Rollover keeps obligations open forever; stability is measured by
        # consecutive default-free days instead of quiet days.
        ledger.quiet_days = tracker.consecutive_no_defaults
        stable_today = ledger.quiet_days >= quiet_days
    elif day == 0:
        if not impactful:
            ledger.quiet_days += 1
        stable_today = False
    else:
        pending_future = has_pending_future_obligations(ledger)
        ledger.quiet_days = ledger.quiet_days + 1 if not impactful else 0
        stable_today = ledger.quiet_days >= quiet_days and not pending_future

    tracker.snapshots.append(
        StabilitySnapshot(
            day=day,
            consecutive_quiet=(tracker.consecutive_quiet if runtime.ctx.rollover_enabled else ledger.quiet_days),
            consecutive_no_defaults=tracker.consecutive_no_defaults,
            has_open_obligations=has_pending_future_obligations(ledger),
            impacted_count=impacted_on_day(ledger, day),
            default_count=defaults,
        )
    )
    return stable_today


def run_until_stable(
    runtime: Runtime,
    *,
    max_days: int,
    quiet_days: int,
    day_callback: Callable[[Runtime, int], None] | None = None,
) -> RunResult:
    reached_stable = False
    final_day = 0
    tracker = StabilityTracker()
    for day in range(max_days):
        impactful = run_day(runtime, day)
        stable_today = update_stability(runtime, day=day, impactful=impactful, quiet_days=quiet_days, tracker=tracker)
        if day_callback:
            day_callback(runtime, day)
        final_day = day + 1
        if stable_today:
            reached_stable = True
            break

    return RunResult(
        ledger=runtime.ledger,
        final_day=final_day,
        reached_stable=reached_stable,
        stop_reason=(StopReason.STABILITY_REACHED if reached_stable else StopReason.MAX_DAYS_REACHED),
        stability_snapshots=tuple(tracker.snapshots),
    )


def run_scenario(
    config: ScenarioConfig,
    *,
    max_days: int | None = None,
    quiet_days: int | None = None,
) -> RunResult:
    """Prepare and run a scenario with the legacy-compatible stop rule."""
    runtime = prepare_scenario(config)
    return run_until_stable(
        runtime,
        max_days=max_days if max_days is not None else config.run.max_days,
        quiet_days=quiet_days if quiet_days is not None else config.run.quiet_days,
    )
