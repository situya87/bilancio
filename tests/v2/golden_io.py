"""Golden-file capture and loading for the example-scenario oracle.

The golden files were originally captured from the clean-core engine while
the live parity suite proved v2 == clean-core, so they carry cross-engine
authority. Regenerating (``uv run python -m tests.v2.golden_io``) snapshots
the current v2 kernel — regression pinning only — so regenerate only from a
commit whose behavior you trust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bilancio_v2.parity import run_snapshot  # noqa: F401  (re-exported for tests)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SCENARIO_DIR = Path(__file__).resolve().parents[2] / "examples" / "scenarios"


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict[str, Any]:
    return json.loads(golden_path(name).read_text())


def capture_all() -> None:
    from bilancio_v2 import load_scenario
    from bilancio_v2.parity import snapshot_run
    from tests.v2.test_parity_examples import SUPPORTED

    GOLDEN_DIR.mkdir(exist_ok=True)
    for name in SUPPORTED:
        config = load_scenario(SCENARIO_DIR / f"{name}.yaml")
        snapshot = snapshot_run(config)
        golden_path(name).write_text(json.dumps(snapshot, indent=1) + "\n")
        print(f"captured {name}: {len(snapshot['events'])} events")


if __name__ == "__main__":
    capture_all()
