"""Golden-snapshot harness for the v2 kernel.

Historically this module compared v2 against the clean-core engine live;
after the clean-core engine's deletion, the golden snapshots captured while
live parity held (``tests/v2/golden*``) are the oracle. A run snapshot is
the full event stream plus final balances, JSON-normalized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from bilancio.config.models import ScenarioConfig
from bilancio_v2.engine import RunResult, run_scenario


@dataclass
class ParityReport:
    scenario_name: str
    events_equal: bool
    balances_equal: bool
    diffs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.events_equal and self.balances_equal


def _jsonable(value: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = json.loads(json.dumps(value, default=str))
    return normalized


def _v2_balances(result: RunResult) -> dict[str, Any]:
    ledger = result.ledger
    return {
        "cash": {k: v for k, v in ledger.cash.items() if v},
        "reserves": {k: v for k, v in ledger.reserves.items() if v},
        "deposits": {k: v for k, v in ledger.deposits.items() if v},
        "defaulted": set(ledger.defaulted_agent_ids),
        "cb_reserves_outstanding": ledger.cb_reserves_outstanding,
        "cb_loans_outstanding": ledger.cb_loans_outstanding,
        "cb_interest_total_paid": ledger.cb_interest_total_paid,
    }


def run_snapshot(events: list[dict[str, Any]], balances: dict[str, Any]) -> dict[str, Any]:
    """JSON-normalized snapshot of a run: full event stream + final balances."""
    return _jsonable(
        {
            "events": events,
            "balances": {
                "cash": {k: str(v) for k, v in sorted(balances["cash"].items())},
                "reserves": {k: str(v) for k, v in sorted(balances["reserves"].items())},
                "deposits": {f"{customer}@{bank}": str(amount) for (customer, bank), amount in sorted(balances["deposits"].items())},
                "defaulted": sorted(balances["defaulted"]),
            },
        }
    )


def snapshot_run(
    config: ScenarioConfig,
    *,
    max_days: int | None = None,
    quiet_days: int | None = None,
    banking_config: Any | None = None,
) -> dict[str, Any]:
    """Run a scenario on the v2 kernel and return its snapshot."""
    result = run_scenario(config, max_days=max_days, quiet_days=quiet_days, banking_config=banking_config)
    return run_snapshot(result.events, _v2_balances(result))


def compare_to_golden(
    config: ScenarioConfig,
    golden: dict[str, Any],
    *,
    max_days: int | None = None,
    quiet_days: int | None = None,
    banking_config: Any | None = None,
    max_event_diffs: int = 5,
) -> ParityReport:
    """Run on v2 and diff against a stored golden snapshot."""
    actual = snapshot_run(config, max_days=max_days, quiet_days=quiet_days, banking_config=banking_config)
    diffs: list[str] = []

    golden_events = golden["events"]
    actual_events = actual["events"]
    events_equal = actual_events == golden_events
    if not events_equal:
        if len(golden_events) != len(actual_events):
            diffs.append(f"event count: golden={len(golden_events)} v2={len(actual_events)}")
        shown = 0
        for index, (golden_event, actual_event) in enumerate(zip(golden_events, actual_events, strict=False)):
            if golden_event != actual_event:
                diffs.append(f"event[{index}]: golden={golden_event!r} v2={actual_event!r}")
                shown += 1
                if shown >= max_event_diffs:
                    break
        if shown == 0 and len(golden_events) != len(actual_events):
            longer, source = (golden_events, "golden") if len(golden_events) > len(actual_events) else (actual_events, "v2")
            extra_start = min(len(golden_events), len(actual_events))
            for event in longer[extra_start : extra_start + max_event_diffs]:
                diffs.append(f"extra {source} event: {event!r}")

    balances_equal = actual["balances"] == golden["balances"]
    if not balances_equal:
        for key in golden["balances"]:
            if golden["balances"][key] != actual["balances"].get(key):
                diffs.append(f"balances[{key}]: golden={golden['balances'][key]!r} v2={actual['balances'].get(key)!r}")

    return ParityReport(
        scenario_name=config.name,
        events_equal=events_equal,
        balances_equal=balances_equal,
        diffs=diffs,
    )
