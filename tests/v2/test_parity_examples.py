"""Example-scenario oracle: the v2 kernel must match the golden snapshots.

The goldens were captured while live parity against the clean-core engine
held (event-for-event, balance-for-balance), so they carry cross-engine
authority. Scenarios the engine refuses must refuse with the pinned error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bilancio.core.errors import DefaultError
from bilancio_v2 import load_scenario, run_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "examples" / "scenarios"

# Full-run parity against goldens (see test_golden.py for the comparison).
SUPPORTED = [
    "clearing_ring",
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

# Failure parity: fail-fast shortfalls in liquidity-stressed rings whose
# dealer config the generator drops. Error messages pinned while the
# clean-core engine raised the identical errors.
FAIL_IDENTICALLY = {
    "kalecki_with_dealer": "Insufficient funds to settle payable PAY_9: 251 still owed",
    "simple_dealer": "Insufficient funds to settle payable PAY_5: 144 still owed",
    "simple_dealer_demo_n_3_kappa_0_5_c_1_mu_0": ("Insufficient funds to settle payable PAY_5: 144 still owed"),
}


def _scenario(name: str):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


def test_example_scenarios_are_classified() -> None:
    known = {*SUPPORTED, *FAIL_IDENTICALLY}
    on_disk = {path.stem for path in SCENARIO_DIR.glob("*.yaml")}
    assert on_disk == known, (
        f"examples/scenarios changed — classify new scenarios into SUPPORTED/FAIL_IDENTICALLY: {sorted(on_disk ^ known)}"
    )


@pytest.mark.parametrize("name", sorted(FAIL_IDENTICALLY))
def test_failure_parity(name: str) -> None:
    with pytest.raises(DefaultError) as exc:
        run_scenario(_scenario(name))
    assert str(exc.value) == FAIL_IDENTICALLY[name]
