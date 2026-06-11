"""Golden snapshots for the deterministic parity cases.

Captured immediately before the clean-core engine's deletion, while the
live parity suite proved v2 == clean-core on every one of these cases —
so the snapshots carry cross-engine authority. They pin the observable
contract (full event stream + final balances) for the rollover, rating,
banking, and dealer test scenarios. Regenerating later snapshots v2
itself (regression pinning only).

Regenerate with:  uv run python -m tests.v2.golden_cases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN_CASE_DIR = Path(__file__).resolve().parent / "golden_cases"


def case_specs() -> dict[str, tuple[Any, Any]]:
    """All deterministic parity cases: name -> (ScenarioConfig, banking_config)."""
    from tests.v2.test_banking_parity import BANKING_CONFIGS, base_scenario
    from tests.v2.test_dealer_parity import dealer_scenario
    from tests.v2.test_rating_parity import rating_scenario
    from tests.v2.test_rollover_parity import rollover_ring

    cases: dict[str, tuple[Any, Any]] = {}
    for cash, default_handling in [
        (300, "fail-fast"),
        (300, "expel-agent"),
        (100, "expel-agent"),
        (150, "expel-agent"),
    ]:
        cases[f"rollover_{default_handling}_{cash}"] = (
            rollover_ring(cash, default_handling),
            None,
        )
    for profile in ["omniscient", "realistic"]:
        cases[f"rating_{profile}"] = (rating_scenario(profile), None)
    for name, banking in BANKING_CONFIGS.items():
        cases[f"banking_{name}"] = (base_scenario(), banking)
    cases["dealer_marker"] = (dealer_scenario(None), None)
    cases["dealer_marker_risk"] = (dealer_scenario(None, risk_enabled=True), None)
    cases["dealer_passive"] = (dealer_scenario("passive"), None)
    cases["dealer_active"] = (dealer_scenario("active", capitalize_market_makers=True), None)
    cases["dealer_active_risk"] = (
        dealer_scenario("active", capitalize_market_makers=True, risk_enabled=True, cash=60),
        None,
    )
    return cases


def golden_case_path(name: str) -> Path:
    return GOLDEN_CASE_DIR / f"{name}.json"


def load_golden_case(name: str) -> dict[str, Any]:
    return json.loads(golden_case_path(name).read_text())


def v2_case_snapshot(config: Any, banking_config: Any) -> dict[str, Any]:
    from bilancio_v2 import run_scenario
    from bilancio_v2.parity import _v2_balances
    from tests.v2.golden_io import run_snapshot

    result = run_scenario(config, banking_config=banking_config)
    return run_snapshot(result.events, _v2_balances(result))


def capture_all() -> None:
    GOLDEN_CASE_DIR.mkdir(exist_ok=True)
    for name, (config, banking_config) in case_specs().items():
        snapshot = v2_case_snapshot(config, banking_config)
        golden_case_path(name).write_text(json.dumps(snapshot, indent=1) + "\n")
        print(f"captured {name}: {len(snapshot['events'])} events")


if __name__ == "__main__":
    capture_all()
