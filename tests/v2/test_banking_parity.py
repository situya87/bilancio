"""Parity for the banking subsystem (slice 4).

Banking is configured via ``CleanBankingConfig`` (built by sweep runners /
generator metadata, not the YAML schema), so the oracle runs both engines
in-process with the same config object, mirroring the CLI sequence:
prepare → run until stable → end-of-run banking shutdown.

Covers bank quotes/lending, routed deposits, CB refinance and lending
freeze (bank defaults + resolution + deposit write-offs), loan winddown,
final CB settlement, and the interbank auction record. Scenarios where the
engines refuse the run (insolvent interbank transfer) must refuse with the
same error.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bilancio.config.models import ScenarioConfig
from bilancio.engines import clean_core
from bilancio.engines.clean_core_types import CleanBankingConfig
from bilancio_v2 import run_scenario
from bilancio_v2.parity import compare_runs


def base_scenario() -> ScenarioConfig:
    agents = [
        {"id": "CB", "kind": "central_bank", "name": "CB"},
        {"id": "B1", "kind": "bank", "name": "B1"},
        {"id": "B2", "kind": "bank", "name": "B2"},
    ] + [{"id": f"H{i}", "kind": "household", "name": f"H{i}"} for i in range(4)]
    actions: list[dict] = [
        {"mint_reserves": {"to": "B1", "amount": 2000}},
        {"mint_reserves": {"to": "B2", "amount": 2000}},
    ]
    for i in range(4):
        actions += [
            {"mint_cash": {"to": f"H{i}", "amount": 500}},
            {
                "deposit_cash": {
                    "customer": f"H{i}",
                    "bank": "B1" if i % 2 == 0 else "B2",
                    "amount": 400,
                }
            },
        ]
    actions += [
        {"create_payable": {"from": "H0", "to": "H1", "amount": 350, "due_day": 1}},
        {"create_payable": {"from": "H1", "to": "H2", "amount": 600, "due_day": 2}},
        {"create_payable": {"from": "H2", "to": "H3", "amount": 450, "due_day": 2}},
        {"create_payable": {"from": "H3", "to": "H0", "amount": 700, "due_day": 3}},
    ]
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "banking-parity",
            "agents": agents,
            "initial_actions": actions,
            "run": {"max_days": 15, "quiet_days": 2, "default_handling": "expel-agent"},
        }
    )


BANKING_CONFIGS = {
    "plain": CleanBankingConfig(kappa=Decimal("0.5")),
    "lending": CleanBankingConfig(kappa=Decimal("0.5"), enable_bank_lending=True, maturity_days=4),
    "coverage_gate": CleanBankingConfig(
        kappa=Decimal("1"),
        enable_bank_lending=True,
        maturity_days=3,
        min_coverage_ratio=Decimal("0.5"),
    ),
    "cb_freeze": CleanBankingConfig(
        kappa=Decimal("0.5"),
        enable_bank_lending=True,
        maturity_days=4,
        cb_lending_cutoff_day=3,
    ),
    "adaptive_corridor": CleanBankingConfig(
        kappa=Decimal("0.8"),
        adaptive_corridor=True,
        enable_bank_lending=True,
        maturity_days=5,
        credit_risk_loading=Decimal("0.1"),
    ),
}


@pytest.mark.parametrize("name", sorted(BANKING_CONFIGS))
def test_banking_modes_run_identically(name: str) -> None:
    report = compare_runs(base_scenario(), banking_config=BANKING_CONFIGS[name])
    assert report.ok, f"banking parity broken ({name}):\n" + "\n".join(report.diffs)


@st.composite
def banking_cases(draw: st.DrawFn) -> tuple[ScenarioConfig, CleanBankingConfig]:
    n_banks = draw(st.integers(min_value=1, max_value=3))
    n_customers = draw(st.integers(min_value=3, max_value=6))
    banks = [f"B{j}" for j in range(n_banks)]
    customers = [f"H{i}" for i in range(n_customers)]
    agents = (
        [{"id": "CB", "kind": "central_bank", "name": "CB"}]
        + [{"id": bank, "kind": "bank", "name": bank} for bank in banks]
        + [{"id": c, "kind": draw(st.sampled_from(["household", "firm"])), "name": c} for c in customers]
    )
    has_lender = draw(st.booleans())
    if has_lender:
        agents.append({"id": "LENDER", "kind": "non_bank_lender", "name": "Lender"})

    actions: list[dict] = []
    for bank in banks:
        # Sometimes under-reserve banks to force CB refinance / freeze paths.
        actions.append({"mint_reserves": {"to": bank, "amount": draw(st.sampled_from([100, 400, 2000]))}})
        if draw(st.booleans()):
            actions.append(
                {
                    "create_cb_loan": {
                        "bank": bank,
                        "amount": draw(st.integers(100, 600)),
                        "rate": "0.03",
                    }
                }
            )
    if has_lender:
        actions.append({"mint_cash": {"to": "LENDER", "amount": draw(st.integers(100, 600))}})
    for customer in customers:
        cash = draw(st.integers(min_value=0, max_value=400))
        if cash:
            actions.append({"mint_cash": {"to": customer, "amount": cash}})
            deposit = draw(st.integers(min_value=0, max_value=cash))
            if deposit:
                actions.append(
                    {
                        "deposit_cash": {
                            "customer": customer,
                            "bank": draw(st.sampled_from(banks)),
                            "amount": deposit,
                        }
                    }
                )
    for _ in range(draw(st.integers(min_value=2, max_value=8))):
        debtor, creditor = draw(st.lists(st.sampled_from(customers), min_size=2, max_size=2, unique=True))
        actions.append(
            {
                "create_payable": {
                    "from": debtor,
                    "to": creditor,
                    "amount": draw(st.integers(50, 700)),
                    "due_day": draw(st.integers(1, 5)),
                }
            }
        )

    config_dict: dict = {
        "version": 1,
        "name": "banking-property",
        "agents": agents,
        "initial_actions": actions,
        "run": {"max_days": 14, "quiet_days": 2, "default_handling": "expel-agent"},
    }
    if has_lender:
        lender: dict = {"enabled": True, "maturity_days": draw(st.integers(1, 3)), "horizon": 3}
        if draw(st.booleans()):
            lender["kappa"] = "0.5"
        config_dict["lender"] = lender

    banking = CleanBankingConfig(
        kappa=Decimal(draw(st.sampled_from(["0.3", "0.5", "1"]))),
        adaptive_corridor=draw(st.booleans()),
        enable_bank_lending=draw(st.booleans()),
        maturity_days=draw(st.integers(2, 6)),
        credit_risk_loading=Decimal(draw(st.sampled_from(["0", "0.1"]))),
        min_coverage_ratio=Decimal(draw(st.sampled_from(["0", "0.5"]))),
        cb_lending_cutoff_day=draw(st.sampled_from([None, 2, 4])),
    )
    return ScenarioConfig.model_validate(config_dict), banking


def run_legacy_banking(config: ScenarioConfig, banking: CleanBankingConfig) -> None:
    runtime = clean_core.prepare_scenario(config, banking_config=banking)
    result = clean_core.run_runtime_until_stable(runtime, max_days=config.run.max_days, quiet_days=config.run.quiet_days)
    clean_core.finalize_banking_marker_events(
        result.state,
        final_day=result.final_day,
        reached_stable=result.reached_stable,
        banking_config=banking,
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=banking_cases())
def test_banking_scenarios_run_identically(
    case: tuple[ScenarioConfig, CleanBankingConfig],
) -> None:
    config, banking = case
    legacy_err = v2_err = None
    try:
        run_legacy_banking(config, banking)
    except Exception as exc:  # noqa: BLE001 — exception parity is the assertion
        legacy_err = str(exc)
    try:
        run_scenario(config, banking_config=banking)
    except Exception as exc:  # noqa: BLE001
        v2_err = str(exc)
    if legacy_err is not None or v2_err is not None:
        assert legacy_err == v2_err
        return
    report = compare_runs(config, banking_config=banking)
    assert report.ok, "\n".join(report.diffs)
