#!/usr/bin/env python3
"""Long-Horizon Drift Benchmark.

Runs long simulations (default 300 days) to detect slow drift/leaks in:
- invariants
- events/day growth
- open obligations
- traced memory growth
"""

from __future__ import annotations

import argparse
import tracemalloc
from decimal import Decimal
from pathlib import Path
from statistics import mean
from time import perf_counter

from benchmark_sim_utils import compile_ring_scenario, count_open_payables, scenario_to_system
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
from bilancio.engines.simulation import run_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-horizon drift benchmark.")
    parser.add_argument("--target-score", type=float, default=85.0)
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument(
        "--window",
        type=int,
        default=40,
        help="Window size for start/end drift comparisons (default: 40)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/long_horizon_drift_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/long_horizon_drift_benchmark_report.md",
    )
    return parser.parse_args()


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 1.0 if num <= 0 else float("inf")
    return num / den


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    scenario = compile_ring_scenario(
        n_agents=30,
        kappa=Decimal("1.0"),
        concentration=Decimal("1.0"),
        mu=Decimal("0.5"),
        seed=2026,
        maturity_days=8,
        name_prefix="long-horizon-drift",
    )
    system = scenario_to_system(scenario)
    # Keep the system active across horizon (continuous rollover).
    system.state.rollover_enabled = True

    events_per_day: list[int] = []
    open_payables: list[int] = []
    memory_current_mb: list[float] = []
    cumulative_events: list[int] = []
    invariant_failures: list[str] = []
    runtime_error: str | None = None

    tracemalloc.start()
    try:
        for day_idx in range(args.days):
            events_before = len(system.state.events)
            run_day(system)
            events_today = len(system.state.events) - events_before
            events_per_day.append(events_today)
            open_payables.append(count_open_payables(system))
            cumulative_events.append(len(system.state.events))

            try:
                system.assert_invariants()
            except Exception as exc:  # noqa: BLE001 - benchmark records failure details
                invariant_failures.append(f"day={day_idx}: {type(exc).__name__}: {exc}")
                break

            current, _peak = tracemalloc.get_traced_memory()
            memory_current_mb.append(current / (1024 * 1024))
    except Exception as exc:  # noqa: BLE001
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        tracemalloc.stop()

    completed_days = len(events_per_day)
    w = min(max(10, args.window), max(1, completed_days // 2))

    start_events_mean = mean(events_per_day[:w]) if events_per_day else 0.0
    end_events_mean = mean(events_per_day[-w:]) if events_per_day else 0.0
    event_growth_ratio = _safe_ratio(end_events_mean, start_events_mean)
    max_events_day = max(events_per_day) if events_per_day else 0

    start_open_mean = mean(open_payables[:w]) if open_payables else 0.0
    end_open_mean = mean(open_payables[-w:]) if open_payables else 0.0
    open_growth_ratio = _safe_ratio(end_open_mean, start_open_mean)

    memory_intensity = [
        m / max(1.0, float(ev)) for m, ev in zip(memory_current_mb, cumulative_events, strict=False)
    ]
    start_mem = mean(memory_current_mb[:w]) if memory_current_mb else 0.0
    end_mem = mean(memory_current_mb[-w:]) if memory_current_mb else 0.0
    start_mem_intensity = mean(memory_intensity[:w]) if memory_intensity else 0.0
    end_mem_intensity = mean(memory_intensity[-w:]) if memory_intensity else 0.0
    mem_growth_ratio = _safe_ratio(end_mem_intensity, max(1e-6, start_mem_intensity))

    # Category 1: horizon completion + invariant integrity (40)
    completion_ratio = completed_days / max(1, args.days)
    completion_score = 20.0 * bounded(completion_ratio)
    invariant_score = 20.0 if not invariant_failures and runtime_error is None else 0.0
    cat1 = completion_score + invariant_score

    # Category 2: event growth boundedness (25)
    event_growth_score = 15.0 * lerp_score(event_growth_ratio, 1.0, 2.2, 1.0)
    event_peak_score = 10.0 * lerp_score(float(max_events_day), 300.0, 1200.0, 1.0)
    cat2 = event_growth_score + event_peak_score

    # Category 3: obligations drift boundedness (20)
    cat3 = 20.0 * lerp_score(open_growth_ratio, 1.0, 1.8, 1.0)

    # Category 4: memory growth boundedness (15)
    cat4 = 15.0 * lerp_score(mem_growth_ratio, 1.0, 2.5, 1.0)

    categories = [
        CategoryResult(
            name="Completion & Invariants",
            max_points=40.0,
            earned_points=round(cat1, 3),
            details={
                "completed_days": completed_days,
                "target_days": args.days,
                "invariant_failures": invariant_failures,
                "runtime_error": runtime_error,
            },
        ),
        CategoryResult(
            name="Event Drift",
            max_points=25.0,
            earned_points=round(cat2, 3),
            details={
                "start_events_mean": start_events_mean,
                "end_events_mean": end_events_mean,
                "event_growth_ratio": event_growth_ratio,
                "max_events_day": max_events_day,
            },
        ),
        CategoryResult(
            name="Obligation Drift",
            max_points=20.0,
            earned_points=round(cat3, 3),
            details={
                "start_open_mean": start_open_mean,
                "end_open_mean": end_open_mean,
                "open_growth_ratio": open_growth_ratio,
            },
        ),
        CategoryResult(
            name="Memory Drift",
            max_points=15.0,
            earned_points=round(cat4, 3),
            details={
                "start_current_mb": start_mem,
                "end_current_mb": end_mem,
                "start_mem_per_event_mb": start_mem_intensity,
                "end_mem_per_event_mb": end_mem_intensity,
                "growth_ratio": mem_growth_ratio,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="drift::completed_horizon",
            passed=(completed_days >= args.days and runtime_error is None),
            message=f"completed_days={completed_days}, target={args.days}, runtime_error={runtime_error}",
        ),
        CriticalCheck(
            code="drift::no_invariant_violations",
            passed=(len(invariant_failures) == 0),
            message=(
                "no invariant violations"
                if not invariant_failures
                else f"first_violation={invariant_failures[0]}"
            ),
        ),
        CriticalCheck(
            code="drift::event_growth_bounded",
            passed=(event_growth_ratio <= 1.8 and max_events_day <= 1000),
            message=(
                f"event_growth_ratio={event_growth_ratio:.4f}, max_events_day={max_events_day}"
            ),
        ),
        CriticalCheck(
            code="drift::open_obligations_bounded",
            passed=(open_growth_ratio <= 1.5),
            message=f"open_growth_ratio={open_growth_ratio:.4f}",
        ),
        CriticalCheck(
            code="drift::memory_growth_bounded",
            passed=(mem_growth_ratio <= 2.2),
            message=f"mem_growth_ratio={mem_growth_ratio:.4f}",
        ),
    ]

    total_score = bounded(sum(c.earned_points for c in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    failures = [c for c in checks if not c.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not failures else "FAIL"

    elapsed = perf_counter() - t0
    generated_at = generated_at_utc()

    report = report_dict(
        benchmark_name="Long-Horizon Drift Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={
            "details": {
                "completed_days": completed_days,
                "event_growth_ratio": event_growth_ratio,
                "open_growth_ratio": open_growth_ratio,
                "mem_growth_ratio": mem_growth_ratio,
                "max_events_day": max_events_day,
                "window": w,
            },
            "generated_at_utc": generated_at,
        },
    )

    md = build_markdown_report(
        title="Long-Horizon Drift Benchmark",
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
            f"completed_days={completed_days}/{args.days}",
            f"event_growth_ratio={event_growth_ratio:.4f}",
            f"open_growth_ratio={open_growth_ratio:.4f}",
            f"mem_growth_ratio={mem_growth_ratio:.4f}",
        ],
        detail_sections=[
            (
                "Drift Metrics",
                [
                    f"start_events_mean={start_events_mean:.4f}",
                    f"end_events_mean={end_events_mean:.4f}",
                    f"max_events_day={max_events_day}",
                    f"start_open_mean={start_open_mean:.4f}",
                    f"end_open_mean={end_open_mean:.4f}",
                    f"start_current_mb={start_mem:.4f}",
                    f"end_current_mb={end_mem:.4f}",
                    f"start_mem_per_event_mb={start_mem_intensity:.8f}",
                    f"end_mem_per_event_mb={end_mem_intensity:.8f}",
                ],
            ),
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Long-horizon drift benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
