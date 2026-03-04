#!/usr/bin/env python3
"""Metamorphic Behavior Benchmark.

Checks simulator invariants under transformed inputs:
- Scale all nominal values
- Relabel agents
- Reorder setup actions

Outputs:
- JSON: temp/metamorphic_behavior_benchmark_report.json
- Markdown: temp/metamorphic_behavior_benchmark_report.md
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from benchmark_sim_utils import (
    compile_ring_scenario,
    run_scenario_dict,
    transform_relabel_households,
    transform_reorder_initial_actions,
    transform_scale_nominal,
)
from benchmark_utils import (
    CategoryResult,
    CriticalCheck,
    bounded,
    build_markdown_report,
    cap_grade_for_critical_failures,
    generated_at_utc,
    grade_for_score,
    lerp_score,
    report_dict,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metamorphic behavior benchmark.")
    parser.add_argument("--target-score", type=float, default=85.0)
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.03,
        help="Tolerance for invariant ratios (default: 0.03)",
    )
    parser.add_argument(
        "--scale-tolerance",
        type=float,
        default=0.08,
        help="Relative tolerance for scale covariance checks (default: 0.08)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/metamorphic_behavior_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/metamorphic_behavior_benchmark_report.md",
    )
    return parser.parse_args()


def _ratio_delta(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    return abs(a - b)


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    baseline_scenario = compile_ring_scenario(
        n_agents=24,
        kappa=Decimal("0.8"),
        concentration=Decimal("1.0"),
        mu=Decimal("0.5"),
        seed=4242,
        maturity_days=6,
        name_prefix="metamorphic-baseline",
    )

    baseline = run_scenario_dict(baseline_scenario, max_days=40, quiet_days=2)

    scale_factor = 10
    scaled = run_scenario_dict(transform_scale_nominal(baseline_scenario, scale_factor), max_days=40)
    relabeled = run_scenario_dict(transform_relabel_households(baseline_scenario), max_days=40)
    reordered = run_scenario_dict(transform_reorder_initial_actions(baseline_scenario), max_days=40)

    scale_default_delta = abs(scaled.default_ratio - baseline.default_ratio)
    scale_loss_ratio_delta = _ratio_delta(scaled.total_loss_ratio, baseline.total_loss_ratio)
    relabel_default_delta = abs(relabeled.default_ratio - baseline.default_ratio)
    relabel_loss_ratio_delta = _ratio_delta(relabeled.total_loss_ratio, baseline.total_loss_ratio)
    reorder_default_delta = abs(reordered.default_ratio - baseline.default_ratio)
    reorder_loss_ratio_delta = _ratio_delta(reordered.total_loss_ratio, baseline.total_loss_ratio)

    scaled_loss_factor = (
        (scaled.total_loss / baseline.total_loss) if baseline.total_loss > 0 else float(scale_factor)
    )
    scaled_loss_factor_rel_error = abs(scaled_loss_factor - scale_factor) / max(scale_factor, 1.0)

    relabel_event_rel_error = abs(relabeled.events_count - baseline.events_count) / max(
        1.0, float(baseline.events_count)
    )
    reorder_event_rel_error = abs(reordered.events_count - baseline.events_count) / max(
        1.0, float(baseline.events_count)
    )

    # Category 1: invariant ratios under relabel/reorder/scale (40)
    invariant_ratio_errors = [
        max(scale_default_delta, scale_loss_ratio_delta),
        max(relabel_default_delta, relabel_loss_ratio_delta),
        max(reorder_default_delta, reorder_loss_ratio_delta),
    ]
    invariant_ratio_score = 40.0 * (
        sum(lerp_score(v, 0.0, args.ratio_tolerance * 2.0, 1.0) for v in invariant_ratio_errors)
        / 3.0
    )

    # Category 2: structure invariance on relabel/reorder (30)
    structure_errors = [relabel_event_rel_error, reorder_event_rel_error]
    structure_score = 30.0 * (
        sum(lerp_score(v, 0.0, 0.12, 1.0) for v in structure_errors) / len(structure_errors)
    )

    # Category 3: scale covariance (30)
    scale_cov_score = 20.0 * lerp_score(scaled_loss_factor_rel_error, 0.0, args.scale_tolerance, 1.0)
    scale_cov_score += 10.0 * lerp_score(scale_default_delta, 0.0, args.ratio_tolerance, 1.0)

    categories = [
        CategoryResult(
            name="Invariant Ratios",
            max_points=40.0,
            earned_points=round(invariant_ratio_score, 3),
            details={
                "errors": invariant_ratio_errors,
                "ratio_tolerance": args.ratio_tolerance,
            },
        ),
        CategoryResult(
            name="Structure Invariance",
            max_points=30.0,
            earned_points=round(structure_score, 3),
            details={
                "event_rel_errors": structure_errors,
            },
        ),
        CategoryResult(
            name="Scale Covariance",
            max_points=30.0,
            earned_points=round(scale_cov_score, 3),
            details={
                "expected_scale_factor": scale_factor,
                "observed_scale_factor": scaled_loss_factor,
                "relative_error": scaled_loss_factor_rel_error,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="metamorphic::scale_ratio_invariant",
            passed=(
                scale_default_delta <= args.ratio_tolerance
                and scale_loss_ratio_delta <= args.ratio_tolerance
            ),
            message=(
                f"scale defaults/loss ratios stable (d_default={scale_default_delta:.4f}, "
                f"d_loss_ratio={scale_loss_ratio_delta:.4f})"
            ),
        ),
        CriticalCheck(
            code="metamorphic::relabel_invariant",
            passed=(
                relabel_default_delta <= args.ratio_tolerance
                and relabel_loss_ratio_delta <= args.ratio_tolerance
            ),
            message=(
                f"relabel stable (d_default={relabel_default_delta:.4f}, "
                f"d_loss_ratio={relabel_loss_ratio_delta:.4f})"
            ),
        ),
        CriticalCheck(
            code="metamorphic::reorder_invariant",
            passed=(
                reorder_default_delta <= args.ratio_tolerance
                and reorder_loss_ratio_delta <= args.ratio_tolerance
            ),
            message=(
                f"reorder stable (d_default={reorder_default_delta:.4f}, "
                f"d_loss_ratio={reorder_loss_ratio_delta:.4f})"
            ),
        ),
        CriticalCheck(
            code="metamorphic::scale_covariance",
            passed=(scaled_loss_factor_rel_error <= args.scale_tolerance),
            message=(
                f"scaled loss factor={scaled_loss_factor:.4f} "
                f"(rel_error={scaled_loss_factor_rel_error:.4f})"
            ),
        ),
    ]

    total_score = bounded(sum(c.earned_points for c in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    critical_failures = [c for c in checks if not c.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"

    elapsed = perf_counter() - t0
    generated_at = generated_at_utc()

    details = {
        "baseline": {
            "defaults": baseline.defaults_count,
            "default_ratio": baseline.default_ratio,
            "events": baseline.events_count,
            "total_loss": baseline.total_loss,
            "total_loss_ratio": baseline.total_loss_ratio,
        },
        "scaled": {
            "defaults": scaled.defaults_count,
            "default_ratio": scaled.default_ratio,
            "events": scaled.events_count,
            "total_loss": scaled.total_loss,
            "total_loss_ratio": scaled.total_loss_ratio,
        },
        "relabeled": {
            "defaults": relabeled.defaults_count,
            "default_ratio": relabeled.default_ratio,
            "events": relabeled.events_count,
            "total_loss": relabeled.total_loss,
            "total_loss_ratio": relabeled.total_loss_ratio,
        },
        "reordered": {
            "defaults": reordered.defaults_count,
            "default_ratio": reordered.default_ratio,
            "events": reordered.events_count,
            "total_loss": reordered.total_loss,
            "total_loss_ratio": reordered.total_loss_ratio,
        },
        "deltas": {
            "scale_default_delta": scale_default_delta,
            "scale_loss_ratio_delta": scale_loss_ratio_delta,
            "relabel_default_delta": relabel_default_delta,
            "relabel_loss_ratio_delta": relabel_loss_ratio_delta,
            "reorder_default_delta": reorder_default_delta,
            "reorder_loss_ratio_delta": reorder_loss_ratio_delta,
            "scaled_loss_factor": scaled_loss_factor,
            "scaled_loss_factor_rel_error": scaled_loss_factor_rel_error,
        },
    }

    report = report_dict(
        benchmark_name="Metamorphic Behavior Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={"details": details, "generated_at_utc": generated_at},
    )

    md = build_markdown_report(
        title="Metamorphic Behavior Benchmark",
        generated_at=generated_at,
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        grade=grade,
        base_grade=base_grade,
        meets_target=meets_target,
        categories=categories,
        critical_checks=checks,
        summary_lines=[
            f"baseline defaults={baseline.defaults_count}/{baseline.household_count}",
            f"scaled_loss_factor={scaled_loss_factor:.4f} (target={scale_factor})",
            f"critical_failures={len(critical_failures)}",
        ],
        detail_sections=[
            (
                "Baseline",
                [
                    f"defaults={baseline.defaults_count}",
                    f"default_ratio={baseline.default_ratio:.6f}",
                    f"events={baseline.events_count}",
                    f"total_loss={baseline.total_loss:.6f}",
                    f"total_loss_ratio={baseline.total_loss_ratio}",
                ],
            ),
            (
                "Transform Deltas",
                [
                    f"scale_default_delta={scale_default_delta:.6f}",
                    f"scale_loss_ratio_delta={scale_loss_ratio_delta:.6f}",
                    f"relabel_default_delta={relabel_default_delta:.6f}",
                    f"relabel_loss_ratio_delta={relabel_loss_ratio_delta:.6f}",
                    f"reorder_default_delta={reorder_default_delta:.6f}",
                    f"reorder_loss_ratio_delta={reorder_loss_ratio_delta:.6f}",
                ],
            ),
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Metamorphic benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
