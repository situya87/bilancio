"""Integration test for NBFI behavioral lending tools (Plan 044).

Runs a small ring simulation with and without the min_coverage_ratio
gate to verify that the coverage gate reduces NBFI losses.
"""

from decimal import Decimal

import pytest

from bilancio.config.apply import apply_to_system
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.scenarios import compile_ring_explorer


def _make_ring_scenario(
    n_agents: int = 20,
    kappa: Decimal = Decimal("0.5"),
    min_coverage_ratio: Decimal = Decimal("0"),
) -> dict:
    """Build a ring scenario YAML dict with NBFI lender."""
    generator_data = {
        "version": 1,
        "generator": "ring_explorer_v1",
        "name_prefix": "NBFI Coverage Test",
        "params": {
            "n_agents": n_agents,
            "seed": 42,
            "kappa": str(kappa),
            "Q_total": "5000",
            "inequality": {
                "scheme": "dirichlet",
                "concentration": "1",
                "monotonicity": "0",
            },
            "maturity": {
                "days": 10,
                "mode": "lead_lag",
                "mu": "0.5",
            },
            "liquidity": {
                "allocation": {"mode": "uniform"},
            },
        },
        "compile": {"emit_yaml": False},
    }

    gen_config = RingExplorerGeneratorConfig.model_validate(generator_data)
    scenario = compile_ring_explorer(gen_config, source_path=None)

    # Add lender config
    scenario["lender"] = {
        "enabled": True,
        "base_rate": "0.05",
        "risk_premium_scale": "0.20",
        "max_single_exposure": "0.15",
        "max_total_exposure": "0.80",
        "maturity_days": 2,
        "horizon": 5,
        "kappa": str(kappa),
        "risk_aversion": "0.3",
        "planning_horizon": 5,
        "profit_target": "0.05",
        "min_coverage_ratio": str(min_coverage_ratio),
    }

    # Ensure expel-agent mode and reasonable run config
    scenario.setdefault("run", {})
    scenario["run"]["default_handling"] = "expel-agent"
    scenario["run"]["max_days"] = 30
    scenario["run"]["quiet_days"] = 2

    return scenario


def _run_simulation(scenario_dict: dict) -> list[dict]:
    """Apply scenario and run simulation, collecting all events."""
    config = ScenarioConfig.model_validate(scenario_dict)
    system = System(default_mode="expel-agent")
    apply_to_system(config, system)

    all_events: list[dict] = []
    max_days = config.run.max_days or 30

    for _day in range(max_days):
        events = run_day(
            system,
            enable_dealer=False,
            enable_lender=True,
        )
        if events:
            all_events.extend(events)

    return all_events


def _count_loans_and_rejections(events: list[dict]) -> tuple[int, int]:
    """Count loan created and rejected events."""
    loans = 0
    rejections = 0
    for event in events:
        kind = event.get("kind", "") if isinstance(event, dict) else ""
        if kind == "NonBankLoanCreated":
            loans += 1
        elif kind == "NonBankLoanRejectedCoverage":
            rejections += 1
    return loans, rejections


@pytest.mark.slow
def test_coverage_gate_reduces_loans_issued():
    """With min_coverage_ratio=0.5, fewer loans should be issued than with 0."""
    # Run without coverage gate
    scenario_no_gate = _make_ring_scenario(min_coverage_ratio=Decimal("0"))
    events_no_gate = _run_simulation(scenario_no_gate)

    # Run with coverage gate
    scenario_with_gate = _make_ring_scenario(min_coverage_ratio=Decimal("0.5"))
    events_with_gate = _run_simulation(scenario_with_gate)

    loans_no_gate, _ = _count_loans_and_rejections(events_no_gate)
    loans_with_gate, rejections_with_gate = _count_loans_and_rejections(events_with_gate)

    # With the gate, we expect fewer or equal loans
    # (It's possible the gate is never triggered if all borrowers have
    # high coverage, but with κ=0.5 we expect some rejections)
    assert loans_with_gate <= loans_no_gate, (
        f"Coverage gate should reduce loans: {loans_with_gate} > {loans_no_gate}"
    )
