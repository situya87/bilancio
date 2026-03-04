#!/usr/bin/env python3
"""Run all extended benchmarks and aggregate status into one report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class BenchmarkRun:
    name: str
    script: str
    returncode: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full extended benchmark suite.")
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/extended_benchmark_suite_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/extended_benchmark_suite_report.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        ("Metamorphic Behavior Benchmark", "scripts/run_metamorphic_behavior_benchmark.py"),
        ("Long-Horizon Drift Benchmark", "scripts/run_long_horizon_drift_benchmark.py"),
        ("Stochastic Robustness Benchmark", "scripts/run_stochastic_robustness_benchmark.py"),
        ("Scientific Comparison Benchmark", "scripts/run_scientific_comparison_benchmark.py"),
        ("Calibration / Stylized-Facts Benchmark", "scripts/run_stylized_facts_benchmark.py"),
        ("Scenario Plugin Contract Benchmark", "scripts/run_plugin_contract_benchmark.py"),
        (
            "Scenario Compile-to-Apply Equivalence Benchmark",
            "scripts/run_compile_apply_equivalence_benchmark.py",
        ),
        ("Local-vs-Cloud Parity Benchmark", "scripts/run_local_cloud_parity_benchmark.py"),
        (
            "Failure-Injection Integration Benchmark",
            "scripts/run_failure_injection_integration_benchmark.py",
        ),
    ]

    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    runs: list[BenchmarkRun] = []
    for name, script in benchmarks:
        print(f"Running: {name}")
        completed = subprocess.run([sys.executable, script], cwd=cwd, check=False, env=env)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        runs.append(BenchmarkRun(name=name, script=script, returncode=completed.returncode, status=status))

    failed = [r for r in runs if r.returncode != 0]
    suite_status = "PASS" if not failed else "FAIL"

    report = {
        "benchmark": "Extended Benchmark Suite",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": suite_status,
        "total": len(runs),
        "passed": len(runs) - len(failed),
        "failed": len(failed),
        "runs": [asdict(r) for r in runs],
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Extended Benchmark Suite")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at_utc']}`")
    lines.append(f"Status: **{suite_status}**")
    lines.append("")
    lines.append("| Benchmark | Status | Return code |")
    lines.append("|---|---|---:|")
    for run in runs:
        lines.append(f"| {run.name} | {run.status} | {run.returncode} |")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Suite status: {suite_status}")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")

    return 0 if suite_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
