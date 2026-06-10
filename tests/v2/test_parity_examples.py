"""Parity oracle: the v2 kernel must reproduce the clean-core engine exactly.

Every example scenario is run on both engines and compared event-for-event
and balance-for-balance; scenarios both engines reject must fail with the
identical error. The v2 kernel's supported domain equals clean-core's by
construction (shared gate functions), so every example is covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bilancio.core.errors import DefaultError
from bilancio.engines import clean_core
from bilancio_v2 import load_scenario, run_scenario
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
    "ring_with_action_specs",
    "sasa_scenario",
    "simple_bank",
    "simple_nbfi",
    "two_banks_interbank",
    "two_jurisdictions",
]

# Failure parity: both engines reject the run (fail-fast shortfall in a
# liquidity-stressed ring whose dealer config the generator drops); the
# error must be identical.
FAIL_IDENTICALLY = [
    "kalecki_with_dealer",
    "simple_dealer",
    "simple_dealer_demo_n_3_kappa_0_5_c_1_mu_0",
]


def _scenario(name: str):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


def test_example_scenarios_are_classified() -> None:
    known = {*SUPPORTED, *FAIL_IDENTICALLY}
    on_disk = {path.stem for path in SCENARIO_DIR.glob("*.yaml")}
    assert on_disk == known, (
        f"examples/scenarios changed — classify new scenarios into SUPPORTED/FAIL_IDENTICALLY: {sorted(on_disk ^ known)}"
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
