#!/usr/bin/env python3
"""Build intermediary frontier and structural loss-floor artifacts.

Usage:
  uv run python scripts/build_intermediary_frontier.py <experiment_dir>
  uv run python scripts/build_intermediary_frontier.py <experiment_dir> --out-dir out/custom/frontier
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bilancio.analysis.intermediary_frontier import (
    build_frontier_artifact,
    write_frontier_artifact,
)


def _parse_relief_bands(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        return (0.0, 0.01, 0.03, 0.05)
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute default-relief/intermediary-loss/system-loss frontier artifacts.",
    )
    parser.add_argument("experiment_dir", type=Path, help="Sweep/experiment root directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiment_dir>/aggregate/intermediary_frontier)",
    )
    parser.add_argument(
        "--relief-bands",
        type=str,
        default="0.0,0.01,0.03,0.05",
        help="Comma-separated lower bounds for relief bands (default: 0.0,0.01,0.03,0.05)",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        raise SystemExit(f"Experiment directory does not exist: {experiment_dir}")

    out_dir = args.out_dir or (experiment_dir / "aggregate" / "intermediary_frontier")
    artifact = build_frontier_artifact(
        experiment_dir,
        relief_bands=_parse_relief_bands(args.relief_bands),
    )
    files = write_frontier_artifact(artifact, out_dir)

    print(f"Pairs: {len(artifact.pairs)}")
    print(f"Summary rows (arm): {len(artifact.summary_by_arm)}")
    print(f"Loss-floor rows (arm): {len(artifact.loss_floor_by_arm)}")
    print(f"Wrote artifacts to: {out_dir}")
    for key, path in files.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

