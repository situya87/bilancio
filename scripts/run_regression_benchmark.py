#!/usr/bin/env python3
"""Regression Benchmark.

Runs small ring simulations at three tiers (small, medium, stress) and
compares critical metrics against stored fingerprints to detect drift.

Usage:
    python scripts/run_regression_benchmark.py
    python scripts/run_regression_benchmark.py --update-fingerprints
"""

from __future__ import annotations

import argparse
import io
import json
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


TIERS = {
    "small": {
        "n_agents": 8,
        "kappa": Decimal("1"),
        "concentration": Decimal("1"),
        "mu": Decimal("0"),
        "seed": 42,
        "maturity_days": 5,
        "max_days": 15,
    },
    "medium": {
        "n_agents": 20,
        "kappa": Decimal("0.5"),
        "concentration": Decimal("1"),
        "mu": Decimal("0.25"),
        "seed": 42,
        "maturity_days": 8,
        "max_days": 20,
    },
    "stress": {
        "n_agents": 20,
        "kappa": Decimal("0.3"),
        "concentration": Decimal("0.5"),
        "mu": Decimal("0"),
        "seed": 42,
        "maturity_days": 6,
        "max_days": 25,
    },
}

METRIC_KEYS = ["default_ratio", "defaults_count", "total_loss_ratio", "max_day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression benchmark.")
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument("--update-fingerprints", action="store_true")
    parser.add_argument(
        "--fingerprints",
        type=str,
        default="scripts/regression_fingerprints.json",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/regression_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/regression_benchmark_report.md",
    )
    return parser.parse_args()


def _run_tier(tier_config: dict[str, Any]) -> dict[str, Any]:
    """Run a single tier and return metrics."""
    scenario = compile_ring_scenario(
        n_agents=tier_config["n_agents"],
        kappa=tier_config["kappa"],
        concentration=tier_config["concentration"],
        mu=tier_config["mu"],
        seed=tier_config["seed"],
        maturity_days=tier_config["maturity_days"],
    )
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        result = run_scenario_dict(scenario, max_days=tier_config["max_days"])

    return {
        "default_ratio": float(result.default_ratio) if result.default_ratio is not None else None,
        "defaults_count": result.defaults_count,
        "total_loss_ratio": float(result.total_loss_ratio) if result.total_loss_ratio is not None else None,
        "max_day": result.system.state.day,
    }


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    fp_path = cwd / args.fingerprints

    t0 = perf_counter()

    # Run all tiers
    tier_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for tier_name, tier_config in TIERS.items():
        try:
            tier_results[tier_name] = _run_tier(tier_config)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{tier_name}: {type(exc).__name__}: {exc}")
            tier_results[tier_name] = {}

    # Load or create fingerprints
    fingerprints: dict[str, Any] = {}
    if fp_path.exists() and not args.update_fingerprints:
        fingerprints = json.loads(fp_path.read_text())

    if args.update_fingerprints:
        # Write current results as fingerprints
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(json.dumps(tier_results, indent=2, default=str))
        print(f"Updated fingerprints at {fp_path}")
        return 0

    # Compare against fingerprints
    tiers_completed = len(TIERS) - len(failures)
    completion_rate = tiers_completed / len(TIERS)

    matches = 0
    total_checks = 0
    drift_details: list[str] = []

    for tier_name, result in tier_results.items():
        if tier_name not in fingerprints:
            drift_details.append(f"{tier_name}: no fingerprint (run --update-fingerprints)")
            continue

        fp = fingerprints[tier_name]
        for key in METRIC_KEYS:
            if key not in result or key not in fp:
                continue
            total_checks += 1
            actual = result[key]
            expected = fp[key]
            # Use tolerant float comparison for all numeric values
            try:
                if actual is not None and expected is not None and abs(float(actual) - float(expected)) < 1e-10:
                    matches += 1
                elif actual == expected:
                    matches += 1
                else:
                    drift_details.append(
                        f"{tier_name}.{key}: expected={expected}, got={actual}"
                    )
            except (TypeError, ValueError):
                if actual == expected:
                    matches += 1
                else:
                    drift_details.append(
                        f"{tier_name}.{key}: expected={expected}, got={actual}"
                    )

    match_rate = matches / max(1, total_checks)

    # Scoring
    cat1_score = 40.0 * completion_rate
    cat2_score = 60.0 * match_rate

    categories = [
        CategoryResult(
            name="Tier Completion",
            max_points=40.0,
            earned_points=round(cat1_score, 3),
            details={
                "tiers_completed": tiers_completed,
                "total_tiers": len(TIERS),
                "failures": failures,
            },
        ),
        CategoryResult(
            name="Metric Stability",
            max_points=60.0,
            earned_points=round(cat2_score, 3),
            details={
                "matches": matches,
                "total_checks": total_checks,
                "match_rate": match_rate,
                "drift": drift_details,
            },
        ),
    ]

    has_fingerprints = bool(fingerprints)
    checks = [
        CriticalCheck(
            code="regression::all_tiers_complete",
            passed=(tiers_completed == len(TIERS)),
            message=f"completed={tiers_completed}, expected={len(TIERS)}",
        ),
        CriticalCheck(
            code="regression::fingerprints_present",
            passed=has_fingerprints,
            message=f"fingerprints_path={fp_path}, loaded={has_fingerprints}",
        ),
        CriticalCheck(
            code="regression::no_metric_drift",
            passed=(match_rate == 1.0 or not has_fingerprints),
            message=f"match_rate={match_rate}, drift_count={len(drift_details)}",
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
        benchmark_name="Regression Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={"tier_results": tier_results, "generated_at_utc": generated_at_utc()},
    )

    markdown = build_markdown_report(
        title="Regression Benchmark",
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
            f"tiers_completed={tiers_completed}/{len(TIERS)}",
            f"metric_match_rate={match_rate:.4f}",
            f"drift_count={len(drift_details)}",
        ],
    )

    write_reports(report, markdown, out_json, out_md)

    print(f"Regression benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
