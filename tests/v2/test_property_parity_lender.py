"""Property-based parity for the lender/rating slice.

Random payment networks with a non-bank lender (kappa-aware or signal-based
pricing, optional preventive lending, varied information visibility) and an
optional rating agency must run identically on both engines — including
seeded-noise observations, loan decision events, loan servicing, and default
cascades through outstanding loans.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bilancio.config.models import ScenarioConfig
from bilancio_v2.parity import compare_runs


@st.composite
def lender_scenario_configs(draw: st.DrawFn) -> ScenarioConfig:
    n_customers = draw(st.integers(min_value=3, max_value=6))
    customers = [f"A{i}" for i in range(n_customers)]

    agents = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "LENDER", "kind": "non_bank_lender", "name": "Lender"},
    ] + [
        {
            "id": customer,
            "kind": draw(st.sampled_from(["household", "firm"])),
            "name": customer,
        }
        for customer in customers
    ]
    if draw(st.booleans()):
        agents.append({"id": "RATER", "kind": "rating_agency", "name": "Rating Agency"})

    actions: list[dict] = [{"mint_cash": {"to": "LENDER", "amount": draw(st.integers(100, 1_000))}}]
    for customer in customers:
        cash = draw(st.integers(min_value=0, max_value=300))
        if cash:
            actions.append({"mint_cash": {"to": customer, "amount": cash}})

    # Ring-ish payables, sized to create genuine shortfalls the lender can fill.
    n_payables = draw(st.integers(min_value=2, max_value=8))
    for _ in range(n_payables):
        debtor, creditor = draw(st.lists(st.sampled_from(customers), min_size=2, max_size=2, unique=True))
        actions.append(
            {
                "create_payable": {
                    "from": debtor,
                    "to": creditor,
                    "amount": draw(st.integers(min_value=50, max_value=500)),
                    "due_day": draw(st.integers(min_value=1, max_value=4)),
                }
            }
        )

    lender: dict = {
        "enabled": True,
        "maturity_days": draw(st.integers(min_value=1, max_value=4)),
        "horizon": draw(st.integers(min_value=1, max_value=5)),
        "max_single_exposure": "0.4",
        "max_total_exposure": "0.9",
        "ranking_mode": draw(st.sampled_from(["profit", "cascade", "blended"])),
        "info_cash_visibility": draw(st.sampled_from(["perfect", "noisy", "none"])),
        "info_history_visibility": draw(st.sampled_from(["perfect", "noisy", "none"])),
        "max_loans_per_borrower_per_day": draw(st.sampled_from([0, 1])),
        "marginal_relief_min_ratio": draw(st.sampled_from(["0", "2.0"])),
        "daily_expected_loss_budget_ratio": draw(st.sampled_from(["0", "0.02"])),
        "min_coverage_ratio": draw(st.sampled_from(["0", "0.5"])),
        "coverage_mode": draw(st.sampled_from(["gate", "graduated"])),
    }
    if draw(st.booleans()):
        lender["kappa"] = "0.5"
        lender["preventive_lending"] = draw(st.booleans())
        lender["maturity_matching"] = draw(st.booleans())

    config: dict = {
        "version": 1,
        "name": "lender-property-parity",
        "agents": agents,
        "initial_actions": actions,
        "lender": lender,
        "run": {
            "max_days": 12,
            "quiet_days": 2,
            "default_handling": "expel-agent",
        },
    }
    if any(agent["kind"] == "rating_agency" for agent in agents):
        config["rating_agency"] = {
            "enabled": True,
            "info_profile": draw(st.sampled_from(["omniscient", "realistic"])),
            "coverage_fraction": draw(st.sampled_from(["0.5", "0.8", "1.0"])),
        }

    return ScenarioConfig.model_validate(config)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(config=lender_scenario_configs())
def test_lender_scenarios_run_identically(config: ScenarioConfig) -> None:
    report = compare_runs(config)
    assert report.ok, "\n".join(report.diffs)
