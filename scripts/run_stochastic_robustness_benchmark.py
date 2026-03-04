#!/usr/bin/env python3
"""Stochastic Robustness Benchmark.

Runs canonical scenarios across many seeds and checks distribution stability.
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

from benchmark_sim_utils import compile_ring_scenario, run_scenario_dict
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
    parser = argparse.ArgumentParser(description="Run stochastic robustness benchmark.")
    parser.add_argument("--target-score", type=float, default=85.0)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/stochastic_robustness_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/stochastic_robustness_benchmark_report.md",
    )
    return parser.parse_args()


def _ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    m = mean(values)
    if len(values) == 1:
        return m, m, m
    sd = pstdev(values)
    margin = 1.96 * sd / math.sqrt(len(values))
    return m, m - margin, m + margin


def _outlier_fraction(values: list[float]) -> float:
    if not values:
        return 0.0
    med = median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = median(abs_dev)
    if mad <= 1e-9:
        return 0.0
    outliers = [v for v in values if abs(v - med) > 3.0 * mad]
    return len(outliers) / len(values)


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    seed_values = [1001 + i for i in range(args.seeds)]
    paired_default_effects: list[float] = []
    low_defaults: list[float] = []
    high_defaults: list[float] = []
    low_loss: list[float] = []
    high_loss: list[float] = []
    failures: list[str] = []

    for seed in seed_values:
        try:
            low = run_scenario_dict(
                compile_ring_scenario(
                    n_agents=24,
                    kappa=Decimal("0.6"),
                    concentration=Decimal("1.0"),
                    mu=Decimal("0.5"),
                    seed=seed,
                    maturity_days=6,
                    name_prefix=f"stoch-low-{seed}",
                ),
                max_days=35,
            )
            high = run_scenario_dict(
                compile_ring_scenario(
                    n_agents=24,
                    kappa=Decimal("1.6"),
                    concentration=Decimal("1.0"),
                    mu=Decimal("0.5"),
                    seed=seed,
                    maturity_days=6,
                    name_prefix=f"stoch-high-{seed}",
                ),
                max_days=35,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"seed={seed}: {type(exc).__name__}: {exc}")
            continue

        low_defaults.append(low.default_ratio)
        high_defaults.append(high.default_ratio)
        low_loss.append(low.total_loss_ratio or 0.0)
        high_loss.append(high.total_loss_ratio or 0.0)
        paired_default_effects.append(low.default_ratio - high.default_ratio)

    run_successes = len(paired_default_effects)
    run_total = len(seed_values)
    run_success_rate = run_successes / max(1, run_total)

    monotonic_hits = sum(1 for eff in paired_default_effects if eff >= 0.0)
    monotonic_rate = monotonic_hits / max(1, run_successes)

    effect_mean, effect_ci_low, effect_ci_high = _ci95(paired_default_effects)
    effect_sd = pstdev(paired_default_effects) if len(paired_default_effects) > 1 else 0.0

    outlier_frac_low = _outlier_fraction(low_defaults)
    outlier_frac_high = _outlier_fraction(high_defaults)
    outlier_frac_effect = _outlier_fraction(paired_default_effects)

    low_cv = (pstdev(low_defaults) / max(1e-9, mean(low_defaults))) if low_defaults else 0.0
    high_cv = (pstdev(high_defaults) / max(1e-9, mean(high_defaults))) if high_defaults else 0.0

    # Category 1: reliability (20)
    cat1 = 20.0 * run_success_rate

    # Category 2: monotonic consistency (35)
    cat2 = 35.0 * lerp_score(monotonic_rate, 1.0, 0.70, 1.0)

    # Category 3: effect size stability (25)
    ci_penalty = max(0.0, 0.0 - effect_ci_low)
    cat3 = 15.0 * lerp_score(ci_penalty, 0.0, 0.08, 1.0)
    cat3 += 10.0 * lerp_score(effect_sd, 0.0, 0.20, 1.0)

    # Category 4: distribution discipline (20)
    cat4 = 8.0 * lerp_score(low_cv, 0.0, 1.0, 1.0)
    cat4 += 8.0 * lerp_score(high_cv, 0.0, 1.0, 1.0)
    cat4 += 4.0 * lerp_score(outlier_frac_effect, 0.0, 0.25, 1.0)

    categories = [
        CategoryResult(
            name="Run Reliability",
            max_points=20.0,
            earned_points=round(cat1, 3),
            details={"successes": run_successes, "total": run_total, "failures": failures},
        ),
        CategoryResult(
            name="Monotonic Consistency",
            max_points=35.0,
            earned_points=round(cat2, 3),
            details={"monotonic_hits": monotonic_hits, "successes": run_successes, "rate": monotonic_rate},
        ),
        CategoryResult(
            name="Effect Stability",
            max_points=25.0,
            earned_points=round(cat3, 3),
            details={
                "effect_mean": effect_mean,
                "effect_ci_low": effect_ci_low,
                "effect_ci_high": effect_ci_high,
                "effect_sd": effect_sd,
            },
        ),
        CategoryResult(
            name="Distribution Discipline",
            max_points=20.0,
            earned_points=round(cat4, 3),
            details={
                "low_cv": low_cv,
                "high_cv": high_cv,
                "outlier_frac_low": outlier_frac_low,
                "outlier_frac_high": outlier_frac_high,
                "outlier_frac_effect": outlier_frac_effect,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="stochastic::all_runs_completed",
            passed=(run_successes == run_total),
            message=f"completed {run_successes}/{run_total} paired runs",
        ),
        CriticalCheck(
            code="stochastic::monotonicity_rate",
            passed=(monotonic_rate >= 0.90),
            message=f"monotonic_rate={monotonic_rate:.4f}",
        ),
        CriticalCheck(
            code="stochastic::ci_not_crossing_zero",
            passed=(effect_ci_low >= -0.02),
            message=f"effect_ci=[{effect_ci_low:.4f}, {effect_ci_high:.4f}]",
        ),
        CriticalCheck(
            code="stochastic::effect_variance_bounded",
            passed=(effect_sd <= 0.18),
            message=f"effect_sd={effect_sd:.4f}",
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

    report = report_dict(
        benchmark_name="Stochastic Robustness Benchmark",
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
                "seeds": seed_values,
                "paired_default_effects": paired_default_effects,
                "low_defaults": low_defaults,
                "high_defaults": high_defaults,
                "low_loss_ratio": low_loss,
                "high_loss_ratio": high_loss,
            },
            "generated_at_utc": generated_at,
        },
    )

    md = build_markdown_report(
        title="Stochastic Robustness Benchmark",
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
            f"run_success_rate={run_success_rate:.4f}",
            f"monotonic_rate={monotonic_rate:.4f}",
            f"effect_mean={effect_mean:.4f}",
            f"effect_ci=[{effect_ci_low:.4f}, {effect_ci_high:.4f}]",
            f"effect_sd={effect_sd:.4f}",
        ],
        detail_sections=[
            (
                "Distribution Metrics",
                [
                    f"low_cv={low_cv:.4f}",
                    f"high_cv={high_cv:.4f}",
                    f"outlier_frac_low={outlier_frac_low:.4f}",
                    f"outlier_frac_high={outlier_frac_high:.4f}",
                    f"outlier_frac_effect={outlier_frac_effect:.4f}",
                ],
            ),
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Stochastic robustness benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
