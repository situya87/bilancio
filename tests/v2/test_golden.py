"""The v2 kernel must match golden runs captured from the existing engine.

Unlike the live parity tests (which compare both engines at HEAD), the
goldens pin the contract at capture time — they catch the failure mode
where a change breaks *both* engines in the same way.
"""

from __future__ import annotations

import pytest

from bilancio_v2 import load_scenario, run_scenario
from bilancio_v2.parity import _v2_balances
from tests.v2.golden_io import SCENARIO_DIR, golden_path, load_golden, run_snapshot
from tests.v2.test_parity_examples import SUPPORTED


@pytest.mark.parametrize("name", SUPPORTED)
def test_v2_matches_golden(name: str) -> None:
    if not golden_path(name).exists():
        pytest.fail(f"missing golden file for {name}; regenerate with `uv run python -m tests.v2.golden_io`")
    config = load_scenario(SCENARIO_DIR / f"{name}.yaml")
    result = run_scenario(config)
    snapshot = run_snapshot(result.events, _v2_balances(result))
    golden = load_golden(name)
    assert snapshot["balances"] == golden["balances"], f"balances diverged for {name}"
    assert snapshot["events"] == golden["events"], f"event stream diverged for {name}"
