#!/usr/bin/env python3
"""Failure-Mode Benchmark.

Verifies the simulation handles extreme/failure scenarios correctly:
- Liquidity shock (kappa=0.1)
- Default cascade (kappa=0.3, c=0.2)
- Abundant liquidity (kappa=3, should have low defaults)

Each scenario must complete without crashing and produce expected
failure behavior (high defaults, cascades) while preserving invariants.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark_sim_utils import compile_ring_scenario, run_scenario_dict
from benchmark_utils import (
    CategoryResult,
    CriticalCheck,
    bounded,
    build_markdown_report,
    cap_grade_for_critical_failures,
    generated_at_utc,
    grade_for_score,
    report_dict,
    write_reports,
)


SCENARIOS: dict[str, dict[str, Any]] = {
    "liquidity_shock": {
        "n_agents": 15,
        "kappa": Decimal("0.1"),
        "concentration": Decimal("1"),
        "mu": Decimal("0"),
        "seed": 42,
        "maturity_days": 5,
        "max_days": 15,
        "expect_high_defaults": True,
        "expect_delta_above": 0.3,
    },
    "default_cascade": {
        "n_agents": 15,
        "kappa": Decimal("0.3"),
        "concentration": Decimal("0.2"),
        "mu": Decimal("0"),
        "seed": 42,
        "maturity_days": 6,
        "max_days": 20,
        "expect_high_defaults": True,
        "expect_delta_above": 0.1,
    },
    "abundant_liquidity": {
        "n_agents": 15,
        "kappa": Decimal("3"),
        "concentration": Decimal("1"),
        "mu": Decimal("0.5"),
        "seed": 42,
        "maturity_days": 5,
        "max_days": 15,
        "expect_high_defaults": False,
        "expect_delta_below": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run failure-mode benchmark.")
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/failure_mode_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/failure_mode_benchmark_report.md",
    )
    return parser.parse_args()


def _run_scenario_safe(config: dict[str, Any]) -> dict[str, Any]:
    """Run scenario, return result dict with 'completed' flag."""
    try:
        scenario = compile_ring_scenario(
            n_agents=config["n_agents"],
            kappa=config["kappa"],
            concentration=config["concentration"],
            mu=config["mu"],
            seed=config["seed"],
            maturity_days=config["maturity_days"],
        )
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            result = run_scenario_dict(scenario, max_days=config["max_days"])

        delta = float(result.default_ratio) if result.default_ratio is not None else None
        return {
            "completed": True,
            "delta_total": delta,
            "defaults_count": result.defaults_count,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "completed": False,
            "delta_total": None,
            "defaults_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    results: dict[str, dict[str, Any]] = {}
    for name, config in SCENARIOS.items():
        results[name] = _run_scenario_safe(config)

    # Scoring
    completions = sum(1 for r in results.values() if r["completed"])
    completion_rate = completions / len(SCENARIOS)

    behavior_checks = 0
    behavior_passes = 0
    behavior_details: list[str] = []

    for name, config in SCENARIOS.items():
        result = results[name]
        if not result["completed"]:
            continue

        delta = result["delta_total"]
        if delta is None:
            continue

        if config.get("expect_high_defaults"):
            behavior_checks += 1
            threshold = config.get("expect_delta_above", 0.1)
            if delta >= threshold:
                behavior_passes += 1
            else:
                behavior_details.append(
                    f"{name}: expected delta >= {threshold}, got {delta:.4f}"
                )

        if not config.get("expect_high_defaults") and "expect_delta_below" in config:
            behavior_checks += 1
            threshold = config["expect_delta_below"]
            if delta <= threshold:
                behavior_passes += 1
            else:
                behavior_details.append(
                    f"{name}: expected delta <= {threshold}, got {delta:.4f}"
                )

    behavior_rate = behavior_passes / max(1, behavior_checks)

    cat1_score = 50.0 * completion_rate
    cat2_score = 50.0 * behavior_rate

    categories = [
        CategoryResult(
            name="Scenario Completion",
            max_points=50.0,
            earned_points=round(cat1_score, 3),
            details={
                "completed": completions,
                "total": len(SCENARIOS),
                "errors": {
                    n: r["error"] for n, r in results.items() if r["error"]
                },
            },
        ),
        CategoryResult(
            name="Expected Failure Behavior",
            max_points=50.0,
            earned_points=round(cat2_score, 3),
            details={
                "behavior_checks": behavior_checks,
                "behavior_passes": behavior_passes,
                "behavior_rate": behavior_rate,
                "failures": behavior_details,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="failure_mode::all_scenarios_complete",
            passed=(completions == len(SCENARIOS)),
            message=f"completed={completions}, expected={len(SCENARIOS)}",
        ),
        CriticalCheck(
            code="failure_mode::expected_behavior_observed",
            passed=(behavior_rate >= 0.8),
            message=f"behavior_rate={behavior_rate:.2f}, threshold=0.80",
        ),
    ]

    total_score = bounded(sum(c.earned_points for c in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    critical_failures = [c for c in checks if not c.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"

    elapsed = perf_counter() - t0

    report = report_dict(
        benchmark_name="Failure-Mode Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={"scenario_results": results, "generated_at_utc": generated_at_utc()},
    )

    markdown = build_markdown_report(
        title="Failure-Mode Benchmark",
        generated_at=generated_at_utc(),
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        grade=grade,
        base_grade=base_grade,
        meets_target=meets_target,
        categories=categories,
        critical_checks=checks,
        summary_lines=[
            f"completed={completions}/{len(SCENARIOS)}",
            f"behavior_rate={behavior_rate:.2f}",
        ],
    )

    write_reports(report, markdown, out_json, out_md)

    print(f"Failure-mode benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
