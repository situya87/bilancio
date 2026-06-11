"""Property-based parity: random scenarios must run identically on both engines.

Hypothesis generates random payment networks (banks, households, firms,
deposits, payables — including ones designed to default) and asserts the v2
kernel reproduces the clean-core engine event-for-event, including default
cascades, pro-rata recovery, and receivable reassignment.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bilancio.config.models import ScenarioConfig
from tests.v2.test_banking_parity import assert_v2_self_consistent


@st.composite
def scenario_configs(draw: st.DrawFn) -> ScenarioConfig:
    n_customers = draw(st.integers(min_value=2, max_value=6))
    customers = [f"A{i}" for i in range(n_customers)]
    kinds = [draw(st.sampled_from(["household", "firm"])) for _ in range(n_customers)]

    agents = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "B1", "kind": "bank", "name": "Bank One"},
        {"id": "B2", "kind": "bank", "name": "Bank Two"},
    ] + [{"id": customer, "kind": kind, "name": customer} for customer, kind in zip(customers, kinds, strict=False)]

    actions: list[dict] = [
        {"mint_reserves": {"to": "B1", "amount": 10_000}},
        {"mint_reserves": {"to": "B2", "amount": 10_000}},
    ]
    for customer in customers:
        cash = draw(st.integers(min_value=0, max_value=1_000))
        if cash:
            actions.append({"mint_cash": {"to": customer, "amount": cash}})
            deposit = draw(st.integers(min_value=0, max_value=cash))
            if deposit:
                bank = draw(st.sampled_from(["B1", "B2"]))
                actions.append({"deposit_cash": {"customer": customer, "bank": bank, "amount": deposit}})

    n_payables = draw(st.integers(min_value=1, max_value=8))
    for _ in range(n_payables):
        debtor, creditor = draw(st.lists(st.sampled_from(customers), min_size=2, max_size=2, unique=True))
        # Amounts above the endowment ceiling force defaults and exercise
        # expulsion, pro-rata recovery, and receivable reassignment.
        amount = draw(st.integers(min_value=1, max_value=1_500))
        due_day = draw(st.integers(min_value=1, max_value=5))
        actions.append(
            {
                "create_payable": {
                    "from": debtor,
                    "to": creditor,
                    "amount": amount,
                    "due_day": due_day,
                }
            }
        )

    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "property-parity",
            "agents": agents,
            "initial_actions": actions,
            "run": {
                "max_days": 15,
                "quiet_days": 2,
                "default_handling": "expel-agent",
                "rollover_enabled": draw(st.booleans()),
            },
        }
    )


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(config=scenario_configs())
def test_random_scenarios_self_consistent(config: ScenarioConfig) -> None:
    assert_v2_self_consistent(config)
