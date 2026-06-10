"""Parity harness: v2 kernel vs the existing clean-core engine.

Runs the same scenario on both engines and compares the full event stream
and final balances. This is the migration oracle: a scenario is considered
ported only when it produces an identical observable run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bilancio.config.models import ScenarioConfig
from bilancio.engines import clean_core
from bilancio_v2.engine import RunResult, run_scenario


@dataclass
class ParityReport:
    scenario_name: str
    events_equal: bool
    balances_equal: bool
    final_day_equal: bool
    reached_stable_equal: bool
    diffs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.events_equal and self.balances_equal and self.final_day_equal and self.reached_stable_equal


def _nonzero(counter: Mapping[Any, Any]) -> dict[Any, Any]:
    return {key: value for key, value in counter.items() if value}


def _legacy_balances(state: Any) -> dict[str, Any]:
    return {
        "cash": _nonzero(state.cash),
        "reserves": _nonzero(state.reserves),
        "deposits": _nonzero(state.deposits),
        "defaulted": set(state.defaulted_agent_ids),
        "cb_reserves_outstanding": state.cb_reserves_outstanding,
        "cb_loans_outstanding": state.cb_loans_outstanding,
        "cb_interest_total_paid": state.cb_interest_total_paid,
    }


def _v2_balances(result: RunResult) -> dict[str, Any]:
    ledger = result.ledger
    return {
        "cash": _nonzero(ledger.cash),
        "reserves": _nonzero(ledger.reserves),
        "deposits": _nonzero(ledger.deposits),
        "defaulted": set(ledger.defaulted_agent_ids),
        "cb_reserves_outstanding": ledger.cb_reserves_outstanding,
        "cb_loans_outstanding": ledger.cb_loans_outstanding,
        "cb_interest_total_paid": ledger.cb_interest_total_paid,
    }


def compare_runs(
    config: ScenarioConfig,
    *,
    max_days: int | None = None,
    quiet_days: int | None = None,
    banking_config: Any | None = None,
    max_event_diffs: int = 5,
) -> ParityReport:
    if banking_config is not None:
        # Banking mode mirrors the CLI sequence: prepare with the banking
        # config, run until stable, then run the end-of-run banking shutdown.
        legacy_runtime = clean_core.prepare_scenario(config, banking_config=banking_config)
        legacy_result = clean_core.run_runtime_until_stable(
            legacy_runtime,
            max_days=max_days if max_days is not None else config.run.max_days,
            quiet_days=quiet_days if quiet_days is not None else config.run.quiet_days,
        )
        clean_core.finalize_banking_marker_events(
            legacy_result.state,
            final_day=legacy_result.final_day,
            reached_stable=legacy_result.reached_stable,
            banking_config=banking_config,
        )
    else:
        legacy_result = clean_core.run_basic_scenario(config, max_days=max_days, quiet_days=quiet_days)
    v2_result = run_scenario(config, max_days=max_days, quiet_days=quiet_days, banking_config=banking_config)

    diffs: list[str] = []

    legacy_events = legacy_result.events
    v2_events = v2_result.events
    events_equal = legacy_events == v2_events
    if not events_equal:
        if len(legacy_events) != len(v2_events):
            diffs.append(f"event count: legacy={len(legacy_events)} v2={len(v2_events)}")
        shown = 0
        for index, (legacy_event, v2_event) in enumerate(zip(legacy_events, v2_events, strict=False)):
            if legacy_event != v2_event:
                diffs.append(f"event[{index}]: legacy={legacy_event!r} v2={v2_event!r}")
                shown += 1
                if shown >= max_event_diffs:
                    break
        if shown == 0 and len(legacy_events) != len(v2_events):
            longer, source = (legacy_events, "legacy") if len(legacy_events) > len(v2_events) else (v2_events, "v2")
            extra_start = min(len(legacy_events), len(v2_events))
            for event in longer[extra_start : extra_start + max_event_diffs]:
                diffs.append(f"extra {source} event: {event!r}")

    legacy_balances = _legacy_balances(legacy_result.state)
    v2_balances = _v2_balances(v2_result)
    balances_equal = legacy_balances == v2_balances
    if not balances_equal:
        for key in legacy_balances:
            if legacy_balances[key] != v2_balances[key]:
                diffs.append(f"balances[{key}]: legacy={legacy_balances[key]!r} v2={v2_balances[key]!r}")

    final_day_equal = legacy_result.final_day == v2_result.final_day
    if not final_day_equal:
        diffs.append(f"final_day: legacy={legacy_result.final_day} v2={v2_result.final_day}")
    reached_stable_equal = legacy_result.reached_stable == v2_result.reached_stable
    if not reached_stable_equal:
        diffs.append(f"reached_stable: legacy={legacy_result.reached_stable} v2={v2_result.reached_stable}")

    return ParityReport(
        scenario_name=config.name,
        events_equal=events_equal,
        balances_equal=balances_equal,
        final_day_equal=final_day_equal,
        reached_stable_equal=reached_stable_equal,
        diffs=diffs,
    )
