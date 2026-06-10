"""Parity oracle: the v2 kernel must reproduce the clean-core engine exactly.

Every supported example scenario is run on both engines and compared
event-for-event and balance-for-balance. Scenarios both engines reject must
fail with the identical error. Scenarios using subsystems the v2 kernel has
not rebuilt yet must be rejected explicitly (never silently mis-simulated).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bilancio.core.errors import DefaultError
from bilancio.engines import clean_core
from bilancio_v2 import UnsupportedScenarioError, load_scenario, run_scenario
from bilancio_v2.parity import compare_runs

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "examples" / "scenarios"

# Full-run parity: both engines complete and must match exactly.
SUPPORTED = [
    "default_handling_demo",
    "firm_delivery",
    "interbank_netting",
    "intraday_netting",
    "payment_demo",
    "rich_simulation",
    "sasa_scenario",
    "simple_bank",
    "two_banks_interbank",
]

# Failure parity: both engines reject the run (fail-fast shortfall without a
# liquidity-providing subsystem); the error must be identical.
FAIL_IDENTICALLY = [
    "kalecki_with_dealer",
    "simple_dealer",
    "simple_dealer_demo_n_3_kappa_0_5_c_1_mu_0",
]

# Explicitly out of the rebuilt slice: v2 must reject, not mis-simulate.
UNSUPPORTED = [
    "ring_with_action_specs",
    "simple_nbfi",
    "two_jurisdictions",
]


def _scenario(name: str):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


def test_example_scenarios_are_classified() -> None:
    known = {*SUPPORTED, *FAIL_IDENTICALLY, *UNSUPPORTED}
    on_disk = {path.stem for path in SCENARIO_DIR.glob("*.yaml")}
    assert on_disk == known, (
        f"examples/scenarios changed — classify new scenarios into SUPPORTED/FAIL_IDENTICALLY/UNSUPPORTED: {sorted(on_disk ^ known)}"
    )


@pytest.mark.parametrize("name", SUPPORTED)
def test_full_run_parity(name: str) -> None:
    report = compare_runs(_scenario(name))
    assert report.ok, f"parity broken for {name}:\n" + "\n".join(report.diffs)


@pytest.mark.parametrize("name", FAIL_IDENTICALLY)
def test_failure_parity(name: str) -> None:
    config = _scenario(name)
    with pytest.raises(DefaultError) as legacy_exc:
        clean_core.run_basic_scenario(config)
    with pytest.raises(DefaultError) as v2_exc:
        run_scenario(config)
    assert str(v2_exc.value) == str(legacy_exc.value)


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_scenarios_are_rejected(name: str) -> None:
    with pytest.raises(UnsupportedScenarioError):
        run_scenario(_scenario(name))
