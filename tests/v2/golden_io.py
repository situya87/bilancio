"""Golden-file capture and comparison for the v2 parity oracle.

Golden files are captured from the *existing* clean-core engine — they pin
the observable contract (full event stream + final balances) that the v2
kernel must reproduce. Regenerate with::

    uv run python -m tests.v2.golden_io

Regeneration should only ever happen from a commit where the existing
engine is trusted; the goldens are the migration safety net.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SCENARIO_DIR = Path(__file__).resolve().parents[2] / "examples" / "scenarios"


def _jsonable(value: Any) -> Any:
    """Round-trip helper: Decimals and tuples become JSON-stable strings/lists."""
    return json.loads(json.dumps(value, default=str))


def run_snapshot(events: list[dict[str, Any]], balances: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(
        {
            "events": events,
            "balances": {
                "cash": {k: str(v) for k, v in sorted(balances["cash"].items())},
                "reserves": {k: str(v) for k, v in sorted(balances["reserves"].items())},
                "deposits": {f"{customer}@{bank}": str(amount) for (customer, bank), amount in sorted(balances["deposits"].items())},
                "defaulted": sorted(balances["defaulted"]),
            },
        }
    )


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict[str, Any]:
    return json.loads(golden_path(name).read_text())


def capture_all() -> None:
    from bilancio.config.loaders import load_yaml
    from bilancio.engines import clean_core
    from bilancio_v2.parity import _legacy_balances
    from tests.v2.test_parity_examples import SUPPORTED

    GOLDEN_DIR.mkdir(exist_ok=True)
    for name in SUPPORTED:
        config = load_yaml(SCENARIO_DIR / f"{name}.yaml")
        result = clean_core.run_basic_scenario(config)
        snapshot = run_snapshot(result.events, _legacy_balances(result.state))
        golden_path(name).write_text(json.dumps(snapshot, indent=1) + "\n")
        print(f"captured {name}: {len(snapshot['events'])} events")


if __name__ == "__main__":
    capture_all()
