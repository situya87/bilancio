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
from dataclasses import dataclass, field, replace
from typing import Any

from bilancio.config.models import ScenarioConfig
from bilancio.engines.termination import (
    DEFAULT_EVENTS,
    IMPACT_EVENTS,
    StabilitySnapshot,
    StopReason,
)
from bilancio_v2.actions import apply_action
from bilancio_v2.ledger import Ledger
from bilancio_v2.plugins.banking import (
    BankLendingPhase,
    BankQuotesPhase,
    finalize_banking,
    initial_banking_reserve_targets,
)
from bilancio_v2.plugins.base import PhasePlugin, RunContext
from bilancio_v2.plugins.clearing import ClearingPhase
from bilancio_v2.plugins.dealer import (
    DealerPhase,
    initialize_active_dealer_subsystem,
    initialize_dealer_marker,
)
from bilancio_v2.plugins.interbank import InterbankPhase
from bilancio_v2.plugins.lending import LendingPhase
from bilancio_v2.plugins.rating import RatingPhase
from bilancio_v2.plugins.settlement import SettlementPhase
from bilancio_v2.policy import CapabilityMatrix
from bilancio_v2.scenario_gates import (
    build_clearing_config,
    build_dealer_config,
    build_lender_config,
    build_rating_config,
    clean_core_configuration_error_reason,
    clean_core_unsupported_reason,
)
from bilancio_v2.subsystem_config import CleanBankingConfig


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
    def state(self) -> Ledger:
        """Alias for the ledger, satisfying the views/display state protocol."""
        return self.ledger

    @property
    def stop_day(self) -> int:
        return self.final_day


def _unsupported_reason(config: ScenarioConfig) -> str | None:
    # The clean-core gate functions define the supported domain; v2 matches
    # it exactly so both engines accept and reject the same scenarios.
    # (Jurisdiction/FX metadata is carried but, like clean-core, not acted
    # on — multi-currency semantics live only in the legacy v1 engine.)
    return clean_core_unsupported_reason(config)


def prepare_scenario(
    config: ScenarioConfig,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> Runtime:
    """Apply scenario setup and return a runtime that can be stepped day by day."""
    configuration_error = clean_core_configuration_error_reason(config)
    if configuration_error is not None:
        from bilancio.core.errors import ConfigurationError

        raise ConfigurationError(configuration_error)
    reason = _unsupported_reason(config)
    if reason is not None:
        raise UnsupportedScenarioError(reason)

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

    dealer_config = build_dealer_config(config)
    ledger.dealer_config = dealer_config
    if dealer_config is not None:
        if dealer_config.balanced_active:
            initialize_active_dealer_subsystem(ledger, dealer_config)
        else:
            initialize_dealer_marker(ledger, dealer_config)

    if banking_config is not None and not banking_config.reserve_targets:
        banking_config = replace(
            banking_config,
            reserve_targets=initial_banking_reserve_targets(ledger, banking_config),
        )

    # The YAML→subsystem-config mapping is shared with the existing engine
    # so both kernels always read a scenario identically.
    rating_config = build_rating_config(config)
    lender_config = build_lender_config(config)
    clearing_config = build_clearing_config(config)

    ctx = RunContext(
        policy=policy,
        default_mode=config.run.default_handling,
        rollover_enabled=config.run.rollover_enabled,
        banking_config=banking_config,
    )
    phases: list[PhasePlugin] = [ScheduledActionsPhase()]
    if rating_config is not None:
        phases.append(RatingPhase(config=rating_config))
    if banking_config is not None:
        phases.append(BankQuotesPhase())
    if lender_config is not None:
        phases.append(LendingPhase(config=lender_config))
    if banking_config is not None and banking_config.enable_bank_lending:
        phases.append(BankLendingPhase(config=banking_config))
    if dealer_config is not None:
        phases.append(DealerPhase(config=dealer_config))
    if clearing_config is not None:
        phases.append(ClearingPhase(config=clearing_config))
    phases.append(SettlementPhase())
    phases.append(InterbankPhase())
    return Runtime(ledger=ledger, ctx=ctx, phases=tuple(phases))


def run_day(runtime: Runtime, day: int) -> bool:
    """Run a single day and return whether it had impactful settlement events."""
    ledger = runtime.ledger
    ledger.day = day
    banking_config = runtime.ctx.banking_config
    if (
        banking_config is not None
        and banking_config.cb_lending_cutoff_day is not None
        and day >= banking_config.cb_lending_cutoff_day
        and not ledger.cb_lending_frozen
    ):
        ledger.cb_lending_frozen = True
        ledger.log("CBLendingFreezeActivated", cutoff_day=banking_config.cb_lending_cutoff_day)
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
    progress_callback: Callable[[int, int], None] | None = None,
    day_callback: Callable[[Runtime, int], None] | None = None,
) -> RunResult:
    reached_stable = False
    final_day = 0
    tracker = StabilityTracker()
    for day in range(max_days):
        impactful = run_day(runtime, day)
        if progress_callback:
            progress_callback(day + 1, max_days)
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
    banking_config: CleanBankingConfig | None = None,
) -> RunResult:
    """Prepare and run a scenario with the legacy-compatible stop rule.

    In banking mode the end-of-run shutdown (bank loan winddown + final CB
    settlement) runs after the stop rule fires, matching the existing CLI.
    """
    runtime = prepare_scenario(config, banking_config=banking_config)
    result = run_until_stable(
        runtime,
        max_days=max_days if max_days is not None else config.run.max_days,
        quiet_days=quiet_days if quiet_days is not None else config.run.quiet_days,
    )
    finalize_banking(
        result.ledger,
        final_day=result.final_day,
        reached_stable=result.reached_stable,
        banking_config=banking_config,
    )
    return result
