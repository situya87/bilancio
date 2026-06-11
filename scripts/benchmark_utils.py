#!/usr/bin/env python3
"""Shared helpers for benchmark scripts.

Provides consistent scoring, grade capping, and report generation so
all benchmark runners emit compatible JSON/Markdown outputs.
"""

import hashlib
import json
import platform
import resource
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CriticalCheck:
    code: str
    passed: bool
    message: str


@dataclass
class CategoryResult:
    name: str
    max_points: float
    earned_points: float
    details: dict[str, Any]


@dataclass(frozen=True)
class OperationalBudget:
    wall_time_seconds: float
    memory_mb: float
    cloud_cost_usd: float | None = None


OPERATIONAL_BUDGETS: dict[str, OperationalBudget] = {
    "Metamorphic Behavior Benchmark": OperationalBudget(300.0, 2048.0),
    "Long-Horizon Drift Benchmark": OperationalBudget(300.0, 2048.0),
    "Stochastic Robustness Benchmark": OperationalBudget(300.0, 2048.0),
    "Scientific Comparison Benchmark": OperationalBudget(600.0, 2048.0),
    "Calibration / Stylized-Facts Benchmark": OperationalBudget(300.0, 2048.0),
    "Regression Benchmark": OperationalBudget(120.0, 1024.0),
    "Failure-Mode Benchmark": OperationalBudget(120.0, 1024.0),
    "Scenario Plugin Contract Benchmark": OperationalBudget(60.0, 512.0),
    "Scenario Compile-to-Apply Equivalence Benchmark": OperationalBudget(120.0, 1024.0),
    "Local-vs-Cloud Parity Benchmark": OperationalBudget(120.0, 1024.0, 0.001),
    "Failure-Injection Integration Benchmark": OperationalBudget(120.0, 1024.0),
}


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def lerp_score(value: float, full_at: float, zero_at: float, max_points: float) -> float:
    """Linear interpolation score helper.

    - If ``full_at <= zero_at``: lower values are better.
    - If ``full_at > zero_at``: higher values are better.
    """
    if full_at <= zero_at:
        if value <= full_at:
            return max_points
        if value >= zero_at:
            return 0.0
        return max_points * (zero_at - value) / (zero_at - full_at)

    if value >= full_at:
        return max_points
    if value <= zero_at:
        return 0.0
    return max_points * (value - zero_at) / (full_at - zero_at)


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


def cap_grade_for_critical_failures(base_grade: str, failure_count: int) -> str:
    """Apply conservative grade cap based on number of critical gate failures."""
    if failure_count <= 0:
        return base_grade
    if failure_count >= 3:
        cap = "F"
    elif failure_count >= 2:
        cap = "D"
    else:
        cap = "C"

    order = ["A", "B", "C", "D", "F"]
    return order[max(order.index(base_grade), order.index(cap))]


def generated_at_utc() -> str:
    return datetime.now(UTC).isoformat()


def current_peak_memory_mb() -> float | None:
    """Return process peak RSS in MB where the platform exposes it."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError, AttributeError):
        return None

    if peak <= 0:
        return None
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_lock_fingerprints(cwd: Path | None = None) -> dict[str, str]:
    """Return SHA-256 fingerprints for dependency lock/config files."""
    root = cwd or _repo_root()
    candidates = [
        "uv.lock",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "rust/bilancio_kernel/Cargo.lock",
        "rust/bilancio_kernel/uv.lock",
        "rust/bilancio_kernel/pyproject.toml",
    ]
    return {
        rel_path: _file_sha256(root / rel_path)
        for rel_path in candidates
        if (root / rel_path).exists()
    }


def _benchmark_config_from_report(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("benchmark_config"), dict):
        return report["benchmark_config"]

    details = report.get("details") if isinstance(report.get("details"), dict) else {}
    config: dict[str, Any] = {
        "benchmark": report.get("benchmark"),
        "target_score": report.get("target_score"),
    }
    if "grid" in details:
        config["grid"] = details["grid"]
    if "scenarios" in details:
        config["scenarios"] = details["scenarios"]
    if "tiers" in details:
        config["tiers"] = details["tiers"]
    return config


def _seed_map_from_report(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("seed_map"), dict):
        return report["seed_map"]
    details = report.get("details") if isinstance(report.get("details"), dict) else {}
    if isinstance(details.get("seed_map"), dict):
        return details["seed_map"]
    return {}


def operational_budget_for(benchmark_name: str) -> OperationalBudget | None:
    """Return the configured operational budget for a benchmark."""
    return OPERATIONAL_BUDGETS.get(benchmark_name)


def build_benchmark_provenance(
    *,
    benchmark_name: str,
    config: dict[str, Any] | None = None,
    seed_map: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Build a benchmark provenance manifest.

    The manifest is intentionally independent from the result score so a
    report can be traced back to exact code, dependency locks, runtime
    metadata, benchmark configuration, and seed assignments.
    """
    from bilancio.provenance import collect_provenance

    base = collect_provenance()
    config_payload = config or {}
    seed_payload = seed_map or {}
    root = cwd or _repo_root()

    return {
        "schema_version": 1,
        "benchmark": benchmark_name,
        "created_at_utc": base.get("timestamp_utc"),
        "git": {
            "sha": base.get("git_sha"),
            "dirty": base.get("git_dirty"),
        },
        "dependencies": {
            "installed_fingerprint": base.get("dep_fingerprint"),
            "lockfiles": dependency_lock_fingerprints(root),
        },
        "runtime": {
            "python_version": base.get("python_version"),
            "platform": base.get("platform"),
            "cpu_count": base.get("cpu_count"),
            "bilancio_version": base.get("bilancio_version"),
        },
        "config_hash": _json_fingerprint(config_payload),
        "config": config_payload,
        "seed_map": seed_payload,
    }


def report_dict(
    *,
    benchmark_name: str,
    target_score: float,
    total_score: float,
    status: str,
    meets_target: bool,
    base_grade: str,
    grade: str,
    elapsed_seconds: float,
    categories: list[CategoryResult],
    critical_checks: list[CriticalCheck],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget = operational_budget_for(benchmark_name)
    budget_check: dict[str, Any] | None = None
    checks = list(critical_checks)
    if budget is not None:
        peak_memory_mb = current_peak_memory_mb()
        cloud_cost_usd = None
        if extra and isinstance(extra.get("cloud_cost_usd"), int | float):
            cloud_cost_usd = float(extra["cloud_cost_usd"])
        budget_check = check_operational_budget(
            elapsed_seconds,
            peak_memory_mb,
            wall_time_budget_seconds=budget.wall_time_seconds,
            memory_budget_mb=budget.memory_mb,
            cloud_cost_usd=cloud_cost_usd,
            cloud_cost_budget_usd=budget.cloud_cost_usd,
        )
        checks.append(
            CriticalCheck(
                code="operational::within_budget",
                passed=bool(budget_check["all_ok"]),
                message=(
                    f"wall_time={budget_check['wall_time_seconds']}s/"
                    f"{budget_check['wall_time_budget_seconds']}s, "
                    f"memory={budget_check['peak_memory_mb']}MB/"
                    f"{budget_check['memory_budget_mb']}MB, "
                    f"cloud_cost={budget_check['cloud_cost_usd']}/"
                    f"{budget_check['cloud_cost_budget_usd']}"
                ),
            )
        )

    critical_failures = [c for c in checks if not c.passed]
    final_status = "FAIL" if critical_failures else status
    final_grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    out: dict[str, Any] = {
        "benchmark": benchmark_name,
        "generated_at_utc": generated_at_utc(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "target_score": float(target_score),
        "total_score": round(float(total_score), 3),
        "status": final_status,
        "meets_target": bool(meets_target),
        "base_grade": base_grade,
        "grade": final_grade,
        "gap_to_target": round(max(0.0, float(target_score) - float(total_score)), 3),
        "categories": [asdict(c) for c in categories],
        "critical_checks": [asdict(c) for c in checks],
        "critical_failures": [asdict(c) for c in critical_failures],
    }
    if budget_check is not None:
        out["operational_budget"] = budget_check
    if extra:
        out.update(extra)
    return out


def build_markdown_report(
    *,
    title: str,
    generated_at: str,
    target_score: float,
    total_score: float,
    status: str,
    grade: str,
    base_grade: str,
    meets_target: bool,
    categories: list[CategoryResult],
    critical_checks: list[CriticalCheck],
    summary_lines: list[str] | None = None,
    detail_sections: list[tuple[str, list[str]]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append(f"Target score: **{target_score:.1f}/100**")
    lines.append("")

    lines.append("## Scorecard")
    lines.append("")
    lines.append(f"- Status: **{status}**")
    lines.append(f"- Total score: **{total_score:.2f}/100**")
    lines.append(f"- Grade: **{grade}** (base: {base_grade})")
    lines.append(f"- Target met: **{'yes' if meets_target else 'no'}**")
    lines.append(f"- Gap to target: **{max(0.0, target_score - total_score):.2f}**")
    lines.append("")

    if summary_lines:
        lines.append("## Summary")
        lines.append("")
        for ln in summary_lines:
            lines.append(f"- {ln}")
        lines.append("")

    lines.append("## Category Scores")
    lines.append("")
    lines.append("| Category | Earned | Max |")
    lines.append("|---|---:|---:|")
    for cat in categories:
        lines.append(f"| {cat.name} | {cat.earned_points:.2f} | {cat.max_points:.2f} |")
    lines.append(f"| **Total** | **{total_score:.2f}** | **100.00** |")
    lines.append("")

    lines.append("## Critical Gates")
    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("|---|---|---|")
    for check in critical_checks:
        st = "PASS" if check.passed else "FAIL"
        lines.append(f"| `{check.code}` | {st} | {check.message} |")
    lines.append("")

    if detail_sections:
        lines.append("## Details")
        lines.append("")
        for section_title, section_lines in detail_sections:
            lines.append(f"### {section_title}")
            lines.append("")
            for ln in section_lines:
                lines.append(f"- {ln}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def check_operational_budget(
    elapsed_seconds: float,
    peak_memory_mb: float | None = None,
    *,
    wall_time_budget_seconds: float = 300.0,
    memory_budget_mb: float = 2048.0,
    cloud_cost_usd: float | None = None,
    cloud_cost_budget_usd: float | None = None,
) -> dict[str, Any]:
    """Check if operational budgets are met.

    Args:
        elapsed_seconds: Wall time in seconds.
        peak_memory_mb: Peak memory usage in MB (None if not measured).
        wall_time_budget_seconds: Maximum allowed wall time.
        memory_budget_mb: Maximum allowed peak memory.
        cloud_cost_usd: Optional measured/estimated cloud cost.
        cloud_cost_budget_usd: Optional allowed cloud cost.

    Returns:
        Dict with budget check results.
    """
    wall_ok = elapsed_seconds <= wall_time_budget_seconds
    mem_ok = peak_memory_mb is None or peak_memory_mb <= memory_budget_mb
    cost_ok = (
        cloud_cost_usd is None
        or cloud_cost_budget_usd is None
        or cloud_cost_usd <= cloud_cost_budget_usd
    )

    return {
        "wall_time_seconds": round(elapsed_seconds, 3),
        "wall_time_budget_seconds": wall_time_budget_seconds,
        "wall_time_ok": wall_ok,
        "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb is not None else None,
        "memory_budget_mb": memory_budget_mb,
        "memory_ok": mem_ok,
        "cloud_cost_usd": cloud_cost_usd,
        "cloud_cost_budget_usd": cloud_cost_budget_usd,
        "cloud_cost_ok": cost_ok,
        "all_ok": wall_ok and mem_ok and cost_ok,
    }


def write_reports(report: dict[str, Any], markdown: str, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = out_json.with_name(f"{out_json.stem}_provenance.json")
    provenance = build_benchmark_provenance(
        benchmark_name=str(report.get("benchmark", "unknown")),
        config=_benchmark_config_from_report(report),
        seed_map=_seed_map_from_report(report),
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, default=str),
        encoding="utf-8",
    )
    report["provenance_manifest_path"] = str(provenance_path)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")
