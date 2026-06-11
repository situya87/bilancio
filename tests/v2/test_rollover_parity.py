"""Deterministic parity scenarios for payable rollover (Plan 024).

Rollover refinances every settled payable past the latest open maturity,
returning the settlement cash from creditor to debtor. Stability switches
to consecutive default-free days. Covers the full-rollover path, the
partial-rollover path (creditor cannot return all cash), and rollover
interrupted by a default cascade.
"""

from __future__ import annotations

import pytest

from bilancio.config.models import ScenarioConfig
from tests.v2.golden_cases import load_golden_case, v2_case_snapshot


def rollover_ring(cash_per_agent: int, default_handling: str) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"H{i}", "kind": "household", "name": f"H{i}"} for i in range(5)
    ]
    actions: list[dict] = [{"mint_cash": {"to": f"H{i}", "amount": cash_per_agent}} for i in range(5)]
    for i in range(5):
        actions.append(
            {
                "create_payable": {
                    "from": f"H{i}",
                    "to": f"H{(i + 1) % 5}",
                    "amount": 200,
                    "due_day": 1 + i % 3,
                    "maturity_distance": 2,
                }
            }
        )
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": f"rollover-{default_handling}-{cash_per_agent}",
            "agents": agents,
            "initial_actions": actions,
            "run": {
                "max_days": 20,
                "quiet_days": 3,
                "default_handling": default_handling,
                "rollover_enabled": True,
            },
        }
    )


@pytest.mark.parametrize(
    ("cash", "default_handling"),
    [
        (300, "fail-fast"),  # everyone liquid: clean perpetual rollover
        (300, "expel-agent"),
        (100, "expel-agent"),  # stressed ring: defaults interrupt rollover
        (150, "expel-agent"),  # partial rollovers (creditor can't return all)
    ],
)
def test_rollover_matches_golden(cash: int, default_handling: str) -> None:
    golden = load_golden_case(f"rollover_{default_handling}_{cash}")
    snapshot = v2_case_snapshot(rollover_ring(cash, default_handling), None)
    assert snapshot["balances"] == golden["balances"]
    assert snapshot["events"] == golden["events"]
