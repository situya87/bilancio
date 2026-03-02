#!/usr/bin/env python3
"""Run simulation efficiency benchmark with Modal-safety focus.

This benchmark is designed to reduce regressions that can break cloud sweeps:
- Runtime blowups that risk function timeouts
- Memory growth that risks container OOM
- Artifact growth that inflates volume I/O and transfer overhead
- Configuration drift between cloud defaults and Modal function limits

Outputs:
- JSON report: temp/simulation_efficiency_benchmark_report.json
- Markdown report: temp/simulation_efficiency_benchmark_report.md
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from bilancio.cloud.config import CloudConfig
from bilancio.ui.run import run_scenario


@dataclass
class ModalFunctionLimits:
    timeout_seconds: int
    memory_mb: int


@dataclass
class ModalLimits:
    run_simulation: ModalFunctionLimits
    compute_aggregate_metrics: ModalFunctionLimits


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    output: str
    passed: int
    failed: int


@dataclass
class WorkloadSample:
    runtime_seconds: float
    peak_tracemalloc_mb: float
    artifact_total_mb: float
    events_count: int
    events_bytes: int
    balances_bytes: int
    html_bytes: int
    day_final: int
    status: str
    error: str | None = None


@dataclass
class WorkloadAggregate:
    name: str
    repeats: int
    samples: list[WorkloadSample]
    runtime_median_seconds: float
    runtime_worst_seconds: float
    peak_tracemalloc_mb_worst: float
    artifact_total_mb_worst: float
    events_count_worst: int
    passed: bool


@dataclass
class CriticalCheck:
    code: str
    passed: bool
    message: str


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def parse_pytest_counts(output: str, returncode: int) -> tuple[int, int]:
    passed_match = re.search(r"(\d+)\s+passed", output)
    failed_match = re.search(r"(\d+)\s+failed", output)
    error_match = re.search(r"(\d+)\s+error", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = 0
    if failed_match:
        failed += int(failed_match.group(1))
    if error_match:
        failed += int(error_match.group(1))
    if returncode != 0 and passed == 0 and failed == 0:
        failed = 1
    return passed, failed


def run_command(command: list[str], cwd: Path, name: str) -> CommandResult:
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    passed, failed = parse_pytest_counts(output, completed.returncode)
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        output=output.strip(),
        passed=passed,
        failed=failed,
    )


def extract_modal_limits(modal_app_path: Path) -> ModalLimits:
    tree = ast.parse(modal_app_path.read_text(encoding="utf-8"), filename=str(modal_app_path))

    def find_limits(function_name: str) -> ModalFunctionLimits:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        fn = decorator.func
                        if isinstance(fn, ast.Attribute) and fn.attr == "function":
                            timeout = 0
                            memory = 0
                            for kw in decorator.keywords:
                                if kw.arg == "timeout":
                                    timeout = int(ast.literal_eval(kw.value))
                                elif kw.arg == "memory":
                                    memory = int(ast.literal_eval(kw.value))
                            if timeout > 0 and memory > 0:
                                return ModalFunctionLimits(timeout_seconds=timeout, memory_mb=memory)
        raise ValueError(f"Could not extract Modal limits for {function_name}")

    return ModalLimits(
        run_simulation=find_limits("run_simulation"),
        compute_aggregate_metrics=find_limits("compute_aggregate_metrics"),
    )


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _scenario_yaml_for_ring(n_agents: int, seed: int = 42) -> dict[str, Any]:
    return {
        "version": 1,
        "generator": "ring_explorer_v1",
        "name_prefix": f"effbench-n{n_agents}",
        "params": {
            "n_agents": n_agents,
            "seed": seed,
            "kappa": "1",
            "Q_total": str(100 * n_agents),
            "liquidity": {"allocation": {"mode": "uniform"}},
            "inequality": {"scheme": "dirichlet", "concentration": "1", "monotonicity": "0"},
            "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
        },
        "compile": {"emit_yaml": False},
    }


def run_scenario_workload(
    *,
    scenario_yaml: dict[str, Any] | None,
    scenario_path: Path | None,
    run_dir: Path,
    max_days: int,
) -> WorkloadSample:
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = run_dir / "scenario.yaml"
    if scenario_yaml is not None:
        scenario_file.write_text(yaml.dump(scenario_yaml, sort_keys=False), encoding="utf-8")
    elif scenario_path is not None:
        scenario_file.write_text(scenario_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise ValueError("Either scenario_yaml or scenario_path must be provided")

    balances_csv = run_dir / "balances.csv"
    events_jsonl = run_dir / "events.jsonl"
    run_html = run_dir / "run.html"

    # Suppress simulation console output; benchmark should emit structured report only.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        tracemalloc.start()
        start = perf_counter()
        try:
            run_scenario(
                path=scenario_file,
                mode="until_stable",
                max_days=max_days,
                quiet_days=2,
                show="none",
                check_invariants="daily",
                default_handling="expel-agent",
                export={
                    "balances_csv": str(balances_csv),
                    "events_jsonl": str(events_jsonl),
                },
                html_output=run_html,
            )
            status = "passed"
            error = None
        except SystemExit as exc:  # run_scenario may sys.exit(1) on fatal setup/apply errors
            status = "failed"
            error = f"SystemExit({exc.code})"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        runtime_seconds = perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    events_bytes = events_jsonl.stat().st_size if events_jsonl.exists() else 0
    balances_bytes = balances_csv.stat().st_size if balances_csv.exists() else 0
    html_bytes = run_html.stat().st_size if run_html.exists() else 0
    artifact_total_mb = (events_bytes + balances_bytes + html_bytes) / (1024 * 1024)
    events_count = _count_lines(events_jsonl)

    day_final = 0
    if events_jsonl.exists():
        # Extract last day from events stream (safe fallback to 0).
        with events_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                day = event.get("day")
                if isinstance(day, int) and day > day_final:
                    day_final = day

    return WorkloadSample(
        runtime_seconds=runtime_seconds,
        peak_tracemalloc_mb=peak / (1024 * 1024),
        artifact_total_mb=artifact_total_mb,
        events_count=events_count,
        events_bytes=events_bytes,
        balances_bytes=balances_bytes,
        html_bytes=html_bytes,
        day_final=day_final,
        status=status,
        error=error,
    )


def aggregate_workload(name: str, samples: list[WorkloadSample]) -> WorkloadAggregate:
    runtime_values = [s.runtime_seconds for s in samples]
    peak_values = [s.peak_tracemalloc_mb for s in samples]
    artifact_values = [s.artifact_total_mb for s in samples]
    events_values = [s.events_count for s in samples]
    passed = all(s.status == "passed" for s in samples)
    return WorkloadAggregate(
        name=name,
        repeats=len(samples),
        samples=samples,
        runtime_median_seconds=statistics.median(runtime_values),
        runtime_worst_seconds=max(runtime_values),
        peak_tracemalloc_mb_worst=max(peak_values),
        artifact_total_mb_worst=max(artifact_values),
        events_count_worst=max(events_values),
        passed=passed,
    )


def workload_runtime_score(value: float, budget_seconds: float) -> float:
    # Strict: full score at budget; zero by 1.5x budget.
    if value <= budget_seconds:
        return 1.0
    if budget_seconds <= 0:
        return 0.0
    return bounded((1.5 * budget_seconds - value) / (0.5 * budget_seconds))


def workload_memory_score(projected_mb: float, budget_mb: float) -> float:
    if projected_mb <= budget_mb:
        return 1.0
    if budget_mb <= 0:
        return 0.0
    return bounded((1.35 * budget_mb - projected_mb) / (0.35 * budget_mb))


def workload_artifact_score(artifact_mb: float, budget_mb: float) -> float:
    if artifact_mb <= budget_mb:
        return 1.0
    if budget_mb <= 0:
        return 0.0
    return bounded((1.30 * budget_mb - artifact_mb) / (0.30 * budget_mb))


def grade_for_score(total_score: float) -> str:
    if total_score >= 90:
        return "A"
    if total_score >= 80:
        return "B"
    if total_score >= 70:
        return "C"
    if total_score >= 60:
        return "D"
    return "F"


def worse_grade(left: str, right: str) -> str:
    order = ["A", "B", "C", "D", "F"]
    left_idx = order.index(left)
    right_idx = order.index(right)
    return left if left_idx >= right_idx else right


def cap_grade_for_critical_failures(base_grade: str, failure_count: int) -> str:
    if failure_count <= 0:
        return base_grade
    if failure_count >= 4:
        cap = "F"
    elif failure_count >= 2:
        cap = "D"
    else:
        cap = "C"
    return worse_grade(base_grade, cap)


def evaluate_critical_checks(
    *,
    perf_cmd: CommandResult,
    workloads: list[WorkloadAggregate],
    workload_specs: list[tuple[str, dict[str, Any] | None, Path | None, int, float, float, float]],
    modal_limits: ModalLimits,
    cloud_cfg: CloudConfig,
    projected_peak_mb_worst: float,
    recommended_max_parallel: int,
    critical_memory_headroom_ratio: float,
    critical_timeout_headroom_ratio: float,
    critical_parallel_alignment: float,
) -> tuple[list[CriticalCheck], dict[str, float]]:
    checks: list[CriticalCheck] = []

    throughput_green = perf_cmd.returncode == 0 and perf_cmd.failed == 0
    checks.append(
        CriticalCheck(
            code="throughput_guard::green",
            passed=throughput_green,
            message=(
                "Performance guard tests must be green "
                f"(returncode={perf_cmd.returncode}, failed={perf_cmd.failed})."
            ),
        )
    )

    all_workloads_passed = all(wl.passed for wl in workloads)
    failing_workloads = [wl.name for wl in workloads if not wl.passed]
    checks.append(
        CriticalCheck(
            code="workloads::all_passed",
            passed=all_workloads_passed,
            message=(
                "All benchmark workloads must execute successfully."
                if all_workloads_passed
                else f"Failed workloads: {', '.join(failing_workloads)}."
            ),
        )
    )

    runtime_violations: list[str] = []
    artifact_violations: list[str] = []
    for wl, spec in zip(workloads, workload_specs, strict=True):
        _name, _yaml, _path, _days, runtime_budget, _memory_budget, artifact_budget = spec
        if wl.runtime_worst_seconds > runtime_budget:
            runtime_violations.append(
                f"{wl.name}: {wl.runtime_worst_seconds:.3f}s > budget {runtime_budget:.3f}s"
            )
        if wl.artifact_total_mb_worst > artifact_budget:
            artifact_violations.append(
                f"{wl.name}: {wl.artifact_total_mb_worst:.3f}MB > budget {artifact_budget:.3f}MB"
            )
    checks.append(
        CriticalCheck(
            code="runtime::within_budget",
            passed=not runtime_violations,
            message=(
                "All workloads within runtime budgets."
                if not runtime_violations
                else f"Runtime budget violations: {'; '.join(runtime_violations)}."
            ),
        )
    )
    checks.append(
        CriticalCheck(
            code="artifacts::within_budget",
            passed=not artifact_violations,
            message=(
                "All workloads within artifact budgets."
                if not artifact_violations
                else f"Artifact budget violations: {'; '.join(artifact_violations)}."
            ),
        )
    )

    worst_runtime_seconds = max((wl.runtime_worst_seconds for wl in workloads), default=0.0)
    timeout_seconds = float(max(modal_limits.run_simulation.timeout_seconds, 1))
    timeout_headroom_ratio = 1.0 - (worst_runtime_seconds / timeout_seconds)
    checks.append(
        CriticalCheck(
            code="modal::timeout_headroom",
            passed=timeout_headroom_ratio >= critical_timeout_headroom_ratio,
            message=(
                f"Timeout headroom must be >= {critical_timeout_headroom_ratio:.2f} "
                f"(got {timeout_headroom_ratio:.3f})."
            ),
        )
    )

    hard_mem_mb = float(max(modal_limits.run_simulation.memory_mb, 1))
    memory_headroom_ratio = 1.0 - (projected_peak_mb_worst / hard_mem_mb)
    checks.append(
        CriticalCheck(
            code="modal::memory_headroom",
            passed=memory_headroom_ratio >= critical_memory_headroom_ratio,
            message=(
                f"Projected memory headroom must be >= {critical_memory_headroom_ratio:.2f} "
                f"(got {memory_headroom_ratio:.3f})."
            ),
        )
    )

    parallel_alignment_ratio = recommended_max_parallel / float(max(cloud_cfg.max_parallel, 1))
    checks.append(
        CriticalCheck(
            code="config::parallel_alignment",
            passed=parallel_alignment_ratio >= critical_parallel_alignment,
            message=(
                f"`recommended_max_parallel / cloud.max_parallel` must be >= "
                f"{critical_parallel_alignment:.2f} (got {parallel_alignment_ratio:.3f})."
            ),
        )
    )

    checks.append(
        CriticalCheck(
            code="config::memory_limit_match",
            passed=cloud_cfg.memory_mb == modal_limits.run_simulation.memory_mb,
            message=(
                "Cloud memory_mb must match Modal run_simulation memory "
                f"(cloud={cloud_cfg.memory_mb}, modal={modal_limits.run_simulation.memory_mb})."
            ),
        )
    )
    checks.append(
        CriticalCheck(
            code="config::timeout_within_modal",
            passed=0 < cloud_cfg.timeout_seconds <= modal_limits.run_simulation.timeout_seconds,
            message=(
                "Cloud timeout_seconds must be >0 and <= Modal run_simulation timeout "
                f"(cloud={cloud_cfg.timeout_seconds}, modal={modal_limits.run_simulation.timeout_seconds})."
            ),
        )
    )

    metrics = {
        "timeout_headroom_ratio": timeout_headroom_ratio,
        "memory_headroom_ratio": memory_headroom_ratio,
        "parallel_alignment_ratio": parallel_alignment_ratio,
    }
    return checks, metrics


def build_markdown_report(
    *,
    generated_at: str,
    total_score: float,
    status: str,
    grade: str,
    base_grade: str,
    target_score: float,
    meets_target: bool,
    modal_limits: ModalLimits,
    cloud_cfg: CloudConfig,
    workloads: list[WorkloadAggregate],
    projected_peak_mb_worst: float,
    recommended_max_parallel: int,
    safety_metrics: dict[str, float],
    critical_checks: list[CriticalCheck],
    category_scores: dict[str, float],
    perf_cmd: CommandResult,
) -> str:
    lines: list[str] = []
    lines.append("# Simulation Efficiency Benchmark")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append(f"Target score: **{target_score:.1f}/100**")
    lines.append("")
    lines.append("## Modal Envelope")
    lines.append("")
    lines.append(f"- `run_simulation` timeout: **{modal_limits.run_simulation.timeout_seconds}s**")
    lines.append(f"- `run_simulation` memory: **{modal_limits.run_simulation.memory_mb} MB**")
    lines.append(f"- `compute_aggregate_metrics` timeout: **{modal_limits.compute_aggregate_metrics.timeout_seconds}s**")
    lines.append(f"- `compute_aggregate_metrics` memory: **{modal_limits.compute_aggregate_metrics.memory_mb} MB**")
    lines.append(f"- Cloud config timeout: **{cloud_cfg.timeout_seconds}s**")
    lines.append(f"- Cloud config memory: **{cloud_cfg.memory_mb} MB**")
    lines.append(f"- Cloud config max_parallel: **{cloud_cfg.max_parallel}**")
    lines.append("")
    lines.append("## Score")
    lines.append("")
    lines.append(f"- Status: **{status}**")
    lines.append(f"- Total: **{total_score:.2f}/100** ({grade})")
    lines.append(f"- Base score grade: **{base_grade}**")
    lines.append(f"- Target met: **{'yes' if meets_target else 'no'}**")
    lines.append(f"- Gap to target: **{max(0.0, target_score - total_score):.2f}**")
    lines.append(f"- Projected worst-case peak memory (proxy): **{projected_peak_mb_worst:.2f} MB**")
    lines.append(f"- Recommended safe max_parallel (memory-based cap): **{recommended_max_parallel}**")
    lines.append(f"- Timeout headroom ratio: **{safety_metrics.get('timeout_headroom_ratio', 0.0):.3f}**")
    lines.append(f"- Memory headroom ratio: **{safety_metrics.get('memory_headroom_ratio', 0.0):.3f}**")
    lines.append(f"- Parallel alignment ratio: **{safety_metrics.get('parallel_alignment_ratio', 0.0):.3f}**")
    lines.append("")
    lines.append("### Category Scores")
    lines.append("")
    for name, score in category_scores.items():
        lines.append(f"- {name}: **{score:.2f}**")
    lines.append("")
    lines.append("## Critical Gates")
    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("|---|---|---|")
    for check in critical_checks:
        gate_status = "PASS" if check.passed else "FAIL"
        lines.append(f"| `{check.code}` | {gate_status} | {check.message} |")
    lines.append("")
    lines.append("## Workloads")
    lines.append("")
    lines.append("| Workload | Runtime median (s) | Runtime worst (s) | Peak trace mem (MB) | Artifacts (MB) | Events | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for wl in workloads:
        wl_status = "PASS" if wl.passed else "FAIL"
        lines.append(
            f"| {wl.name} | {wl.runtime_median_seconds:.3f} | {wl.runtime_worst_seconds:.3f} | "
            f"{wl.peak_tracemalloc_mb_worst:.3f} | {wl.artifact_total_mb_worst:.3f} | "
            f"{wl.events_count_worst} | {wl_status} |"
        )
    lines.append("")
    lines.append("## Throughput Guard")
    lines.append("")
    lines.append(
        f"- `{perf_cmd.name}`: returncode={perf_cmd.returncode}, passed={perf_cmd.passed}, failed={perf_cmd.failed}"
    )
    lines.append(f"- Command: `{' '.join(perf_cmd.command)}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simulation efficiency benchmark.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per workload (default: 1)")
    parser.add_argument("--target-score", type=float, default=85.0, help="Target score (default: 85)")
    parser.add_argument(
        "--memory-multiplier",
        type=float,
        default=8.0,
        help="Multiplier from tracemalloc peak MB to projected RSS MB (default: 8.0)",
    )
    parser.add_argument(
        "--critical-memory-headroom",
        type=float,
        default=0.20,
        help="Critical minimum memory headroom ratio vs Modal memory limit (default: 0.20).",
    )
    parser.add_argument(
        "--critical-timeout-headroom",
        type=float,
        default=0.30,
        help="Critical minimum timeout headroom ratio vs Modal timeout (default: 0.30).",
    )
    parser.add_argument(
        "--critical-parallel-alignment",
        type=float,
        default=0.70,
        help=(
            "Critical minimum ratio for recommended_max_parallel / cloud.max_parallel "
            "(default: 0.70)."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/simulation_efficiency_benchmark_report.json",
        help="Path for JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/simulation_efficiency_benchmark_report.md",
        help="Path for Markdown report.",
    )
    args = parser.parse_args()
    for name, value in (
        ("--critical-memory-headroom", args.critical_memory_headroom),
        ("--critical-timeout-headroom", args.critical_timeout_headroom),
        ("--critical-parallel-alignment", args.critical_parallel_alignment),
    ):
        if value < 0.0 or value > 1.0:
            parser.error(f"{name} must be between 0 and 1 (got {value}).")

    cwd = Path(__file__).resolve().parents[1]
    modal_limits = extract_modal_limits(cwd / "src/bilancio/cloud/modal_app.py")
    cloud_cfg = CloudConfig()

    # Quick throughput guard from existing benchmark tests.
    perf_cmd = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-cov",
            "tests/benchmark/test_performance.py",
            "-k",
            "settlement_throughput or run_day_throughput or ring_creation_scaling",
        ],
        cwd=cwd,
        name="throughput_guard",
    )

    workload_specs: list[tuple[str, dict[str, Any] | None, Path | None, int, float, float, float]] = [
        # name, scenario_yaml, scenario_path, max_days, runtime_budget_s, memory_budget_mb, artifact_budget_mb
        ("generated_ring_n100", _scenario_yaml_for_ring(100), None, 30, 8.0, 768.0, 20.0),
        ("generated_ring_n200", _scenario_yaml_for_ring(200), None, 30, 20.0, 1024.0, 40.0),
        ("rich_simulation_example", None, cwd / "examples/scenarios/rich_simulation.yaml", 30, 5.0, 512.0, 10.0),
    ]

    root_run_dir = cwd / "temp" / "simulation_efficiency_benchmark"
    workloads: list[WorkloadAggregate] = []

    for name, scenario_yaml, scenario_path, max_days, _rt_budget, _mem_budget, _art_budget in workload_specs:
        samples: list[WorkloadSample] = []
        for i in range(args.repeats):
            run_dir = root_run_dir / name / f"rep{i+1}"
            sample = run_scenario_workload(
                scenario_yaml=scenario_yaml,
                scenario_path=scenario_path,
                run_dir=run_dir,
                max_days=max_days,
            )
            samples.append(sample)
        workloads.append(aggregate_workload(name, samples))

    # Score category 1: Runtime & throughput (40)
    runtime_components: list[float] = []
    for wl, spec in zip(workloads, workload_specs, strict=True):
        _name, _yaml, _path, _days, runtime_budget, _mem_budget, _artifact_budget = spec
        runtime_components.append(workload_runtime_score(wl.runtime_worst_seconds, runtime_budget))
    runtime_workload_score = 25.0 * (sum(runtime_components) / len(runtime_components))
    throughput_total = perf_cmd.passed + perf_cmd.failed
    throughput_pass_rate = (perf_cmd.passed / throughput_total) if throughput_total else 0.0
    throughput_guard_score = 15.0 * throughput_pass_rate
    runtime_total_score = runtime_workload_score + throughput_guard_score

    # Score category 2: Memory headroom (30)
    memory_components: list[float] = []
    projected_peaks: list[float] = []
    for wl, spec in zip(workloads, workload_specs, strict=True):
        _name, _yaml, _path, _days, _rt_budget, memory_budget, _artifact_budget = spec
        projected_mb = wl.peak_tracemalloc_mb_worst * args.memory_multiplier
        projected_peaks.append(projected_mb)
        memory_components.append(workload_memory_score(projected_mb, memory_budget))
    memory_workload_score = 20.0 * (sum(memory_components) / len(memory_components))
    projected_peak_mb_worst = max(projected_peaks) if projected_peaks else 0.0
    hard_mem = float(modal_limits.run_simulation.memory_mb)
    hard_headroom_ratio = 1.0 - (projected_peak_mb_worst / hard_mem if hard_mem > 0 else 1.0)
    modal_headroom_score = 10.0 * bounded(hard_headroom_ratio / 0.5)  # full points at >=50% headroom
    memory_total_score = memory_workload_score + modal_headroom_score

    # Score category 3: Artifact footprint (15)
    artifact_components: list[float] = []
    event_count_components: list[float] = []
    for wl, spec in zip(workloads, workload_specs, strict=True):
        _name, _yaml, _path, _days, _rt_budget, _mem_budget, artifact_budget = spec
        artifact_components.append(workload_artifact_score(wl.artifact_total_mb_worst, artifact_budget))
        # Soft event budget for payload growth risk in events_jsonl.
        event_count_components.append(workload_artifact_score(float(wl.events_count_worst), 50_000.0))
    artifact_size_score = 10.0 * (sum(artifact_components) / len(artifact_components))
    event_volume_score = 5.0 * (sum(event_count_components) / len(event_count_components))
    artifact_total_score = artifact_size_score + event_volume_score

    # Recommended max_parallel by memory (bounded by configured cap).
    projected_peak = max(projected_peak_mb_worst, 1.0)
    safe_memory_pool = modal_limits.run_simulation.memory_mb * 0.70
    max_parallel_by_memory = max(1, int(safe_memory_pool // projected_peak))
    recommended_max_parallel = max(1, min(cloud_cfg.max_parallel, max_parallel_by_memory))

    # Score category 4: Modal config safety & consistency (15)
    memory_match = int(cloud_cfg.memory_mb == modal_limits.run_simulation.memory_mb)
    timeout_consistent = int(0 < cloud_cfg.timeout_seconds <= modal_limits.run_simulation.timeout_seconds)
    aggregate_ratio_ok = int(
        0 < modal_limits.compute_aggregate_metrics.timeout_seconds < modal_limits.run_simulation.timeout_seconds
    )
    parallel_alignment = bounded(
        recommended_max_parallel / float(max(cloud_cfg.max_parallel, 1))
    )
    parallel_alignment_score = 6.0 * bounded((parallel_alignment - 0.40) / 0.60)
    config_total_score = (
        4.0 * memory_match
        + 4.0 * timeout_consistent
        + 1.0 * aggregate_ratio_ok
        + parallel_alignment_score
    )

    total_score = runtime_total_score + memory_total_score + artifact_total_score + config_total_score
    base_grade = grade_for_score(total_score)
    critical_checks, safety_metrics = evaluate_critical_checks(
        perf_cmd=perf_cmd,
        workloads=workloads,
        workload_specs=workload_specs,
        modal_limits=modal_limits,
        cloud_cfg=cloud_cfg,
        projected_peak_mb_worst=projected_peak_mb_worst,
        recommended_max_parallel=recommended_max_parallel,
        critical_memory_headroom_ratio=args.critical_memory_headroom,
        critical_timeout_headroom_ratio=args.critical_timeout_headroom,
        critical_parallel_alignment=args.critical_parallel_alignment,
    )
    critical_failures = [check for check in critical_checks if not check.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"
    gap_to_target = max(0.0, args.target_score - total_score)

    generated_at = datetime.now(timezone.utc).isoformat()
    category_scores = {
        "Runtime & Throughput (40)": runtime_total_score,
        "Memory Headroom (30)": memory_total_score,
        "Artifact Footprint (15)": artifact_total_score,
        "Modal Config Safety (15)": config_total_score,
    }

    report = {
        "generated_at_utc": generated_at,
        "target_score": args.target_score,
        "total_score": total_score,
        "status": status,
        "meets_target": meets_target,
        "base_grade": base_grade,
        "grade": grade,
        "gap_to_target": gap_to_target,
        "modal_limits": asdict(modal_limits),
        "cloud_config": asdict(cloud_cfg),
        "category_scores": category_scores,
        "throughput_guard": asdict(perf_cmd),
        "workloads": [asdict(wl) for wl in workloads],
        "memory_multiplier": args.memory_multiplier,
        "critical_thresholds": {
            "memory_headroom_ratio": args.critical_memory_headroom,
            "timeout_headroom_ratio": args.critical_timeout_headroom,
            "parallel_alignment_ratio": args.critical_parallel_alignment,
        },
        "critical_checks": [asdict(check) for check in critical_checks],
        "critical_failures": [asdict(check) for check in critical_failures],
        "safety_metrics": safety_metrics,
        "projected_peak_mb_worst": projected_peak_mb_worst,
        "recommended_max_parallel": recommended_max_parallel,
        "config_checks": {
            "memory_match": bool(memory_match),
            "timeout_consistent": bool(timeout_consistent),
            "aggregate_timeout_ratio_ok": bool(aggregate_ratio_ok),
            "parallel_alignment": parallel_alignment,
            "parallel_alignment_score": parallel_alignment_score,
        },
    }

    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(
        build_markdown_report(
            generated_at=generated_at,
            total_score=total_score,
            status=status,
            grade=grade,
            base_grade=base_grade,
            target_score=args.target_score,
            meets_target=meets_target,
            modal_limits=modal_limits,
            cloud_cfg=cloud_cfg,
            workloads=workloads,
            projected_peak_mb_worst=projected_peak_mb_worst,
            recommended_max_parallel=recommended_max_parallel,
            safety_metrics=safety_metrics,
            critical_checks=critical_checks,
            category_scores=category_scores,
            perf_cmd=perf_cmd,
        ),
        encoding="utf-8",
    )

    print(f"Simulation efficiency score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"Target score: {args.target_score:.1f} (gap {gap_to_target:.2f})")
    print(f"Projected worst peak memory: {projected_peak_mb_worst:.2f} MB")
    print(f"Recommended safe max_parallel: {recommended_max_parallel}")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
