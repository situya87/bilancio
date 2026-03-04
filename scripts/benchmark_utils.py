#!/usr/bin/env python3
"""Shared helpers for benchmark scripts.

Provides consistent scoring, grade capping, and report generation so
all benchmark runners emit compatible JSON/Markdown outputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    return datetime.now(timezone.utc).isoformat()


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
    critical_failures = [c for c in critical_checks if not c.passed]
    out: dict[str, Any] = {
        "benchmark": benchmark_name,
        "generated_at_utc": generated_at_utc(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "target_score": float(target_score),
        "total_score": round(float(total_score), 3),
        "status": status,
        "meets_target": bool(meets_target),
        "base_grade": base_grade,
        "grade": grade,
        "gap_to_target": round(max(0.0, float(target_score) - float(total_score)), 3),
        "categories": [asdict(c) for c in categories],
        "critical_checks": [asdict(c) for c in critical_checks],
        "critical_failures": [asdict(c) for c in critical_failures],
    }
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


def write_reports(report: dict[str, Any], markdown: str, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")
