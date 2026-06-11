"""Entrypoint contract tests for benchmark scripts.

Test intent:
- Ensure every benchmark script used as a quality gate can run from its public
  CLI entrypoint.
- Pin the JSON, Markdown, critical-gate, and provenance sidecar contracts.
- Catch report schema drift before benchmark artifacts are used for governance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


BENCHMARK_ENTRYPOINTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "scientific_comparison",
        "scripts/run_scientific_comparison_benchmark.py",
        (
            "--replicates",
            "2",
            "--n-bootstrap",
            "20",
            "--max-days",
            "8",
            "--artifacts-dir",
            "{tmp}/scientific_artifacts",
        ),
    ),
    ("regression", "scripts/run_regression_benchmark.py", ()),
    ("failure_mode", "scripts/run_failure_mode_benchmark.py", ()),
    (
        "metamorphic_behavior",
        "scripts/run_metamorphic_behavior_benchmark.py",
        (),
    ),
    (
        "long_horizon_drift",
        "scripts/run_long_horizon_drift_benchmark.py",
        ("--days", "20", "--window", "5"),
    ),
    (
        "local_cloud_parity",
        "scripts/run_local_cloud_parity_benchmark.py",
        (),
    ),
    (
        "failure_injection_integration",
        "scripts/run_failure_injection_integration_benchmark.py",
        (),
    ),
    (
        "compile_apply_equivalence",
        "scripts/run_compile_apply_equivalence_benchmark.py",
        (),
    ),
    ("plugin_contract", "scripts/run_plugin_contract_benchmark.py", ()),
)


def _format_args(args: Sequence[str], tmp_path: Path) -> list[str]:
    return [arg.format(tmp=tmp_path) for arg in args]


def _run_benchmark_entrypoint(
    script: str,
    *,
    extra_args: Sequence[str],
    out_json: Path,
    out_md: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        [
            sys.executable,
            script,
            *extra_args,
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _assert_report_schema(report: dict[str, Any], *, out_json: Path) -> None:
    required = {
        "benchmark",
        "target_score",
        "total_score",
        "status",
        "grade",
        "categories",
        "critical_checks",
        "critical_failures",
        "generated_at_utc",
        "provenance_manifest_path",
    }
    assert required.issubset(report), sorted(required - set(report))
    assert isinstance(report["benchmark"], str) and report["benchmark"]
    assert isinstance(report["total_score"], int | float)
    assert 0 <= float(report["total_score"]) <= 100
    assert report["status"] == "PASS"
    assert isinstance(report["categories"], list) and report["categories"]
    assert isinstance(report["critical_checks"], list) and report["critical_checks"]
    assert isinstance(report["critical_failures"], list)

    for category in report["categories"]:
        assert {"name", "max_points", "earned_points", "details"}.issubset(category)
        assert isinstance(category["name"], str) and category["name"]
        assert isinstance(category["max_points"], int | float)
        assert isinstance(category["earned_points"], int | float)
        assert isinstance(category["details"], dict)

    for check in report["critical_checks"]:
        assert {"code", "passed", "message"}.issubset(check)
        assert isinstance(check["code"], str) and check["code"]
        assert isinstance(check["passed"], bool)
        assert isinstance(check["message"], str)

    provenance_path = Path(report["provenance_manifest_path"])
    if not provenance_path.is_absolute():
        provenance_path = out_json.parent / provenance_path
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["benchmark"] == report["benchmark"]
    assert "config_hash" in provenance
    assert "runtime" in provenance


@pytest.mark.parametrize(
    ("name", "script", "extra_args"),
    BENCHMARK_ENTRYPOINTS,
    ids=[case[0] for case in BENCHMARK_ENTRYPOINTS],
)
def test_benchmark_entrypoint_report_contract(
    tmp_path: Path,
    name: str,
    script: str,
    extra_args: tuple[str, ...],
) -> None:
    out_json = tmp_path / f"{name}.json"
    out_md = tmp_path / f"{name}.md"
    result = _run_benchmark_entrypoint(
        script,
        extra_args=_format_args(extra_args, tmp_path),
        out_json=out_json,
        out_md=out_md,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert out_json.exists()
    assert out_md.exists()

    report = json.loads(out_json.read_text(encoding="utf-8"))
    _assert_report_schema(report, out_json=out_json)

    markdown = out_md.read_text(encoding="utf-8")
    assert report["benchmark"] in markdown
    assert "Critical Gates" in markdown
