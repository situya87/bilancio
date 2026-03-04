#!/usr/bin/env python3
"""Scenario Compile-to-Apply Equivalence Benchmark.

Verifies that plugin output, once compiled/applied, preserves intended
agent counts, debt totals, and maturity distributions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark_sim_utils import scenario_to_system
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
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import ScenarioConfig
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.scenarios.registry import get_plugin


@dataclass
class EquivalenceResult:
    name: str
    passed: bool
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compile-to-apply equivalence benchmark.")
    parser.add_argument("--target-score", type=float, default=95.0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/compile_apply_equivalence_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/compile_apply_equivalence_benchmark_report.md",
    )
    return parser.parse_args()


def _hash_scenario(scenario: dict[str, Any]) -> str:
    return json.dumps(scenario, sort_keys=True, separators=(",", ":"), default=str)


def _hist_from_scenario(scenario: dict[str, Any]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for action in scenario.get("initial_actions", []):
        payload = action.get("create_payable")
        if not isinstance(payload, dict):
            continue
        due_day = int(payload.get("due_day", 0))
        hist[due_day] = hist.get(due_day, 0) + 1
    return hist


def _hist_from_system(system) -> dict[int, int]:
    hist: dict[int, int] = {}
    for c in system.state.contracts.values():
        if c.kind != InstrumentKind.PAYABLE:
            continue
        due_day = int(c.due_day or 0)
        hist[due_day] = hist.get(due_day, 0) + 1
    return hist


def _sum_payables_scenario(scenario: dict[str, Any]) -> int:
    total = 0
    for action in scenario.get("initial_actions", []):
        payload = action.get("create_payable")
        if isinstance(payload, dict) and payload.get("amount") is not None:
            total += int(Decimal(str(payload["amount"])))
    return total


def _sum_payables_system(system) -> int:
    return sum(c.amount for c in system.state.contracts.values() if c.kind == InstrumentKind.PAYABLE)


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    plugin = get_plugin("kalecki_ring")

    fixtures = [
        {
            "name": "ring_basic",
            "params": {
                "kappa": Decimal("0.8"),
                "concentration": Decimal("1.0"),
                "mu": Decimal("0.5"),
                "monotonicity": Decimal("0"),
            },
            "base": {"n_agents": 20, "maturity_days": 6, "Q_total": Decimal("2000")},
            "seed": 42,
        },
        {
            "name": "ring_stress",
            "params": {
                "kappa": Decimal("0.35"),
                "concentration": Decimal("0.7"),
                "mu": Decimal("0.25"),
                "monotonicity": Decimal("-1"),
            },
            "base": {"n_agents": 25, "maturity_days": 8, "Q_total": Decimal("2500")},
            "seed": 99,
        },
        {
            "name": "balanced_active",
            "params": {
                "kappa": Decimal("0.9"),
                "concentration": Decimal("1.2"),
                "mu": Decimal("0.5"),
                "monotonicity": Decimal("0"),
            },
            "base": {
                "n_agents": 18,
                "maturity_days": 7,
                "Q_total": Decimal("1800"),
                "mode": "active",
            },
            "seed": 11,
        },
    ]

    results: list[EquivalenceResult] = []
    deterministic_same_seed = True
    deterministic_diff_seed = False

    for fx in fixtures:
        name = fx["name"]
        try:
            scenario = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
            ScenarioConfig(**preprocess_config(scenario))
            system = scenario_to_system(scenario)

            scenario_agents = len(scenario.get("agents", []))
            system_agents = len(system.state.agents)
            scenario_payables = _sum_payables_scenario(scenario)
            system_payables = _sum_payables_system(system)
            scenario_hist = _hist_from_scenario(scenario)
            system_hist = _hist_from_system(system)

            structural_match = scenario_agents == system_agents
            payable_match = scenario_payables == system_payables
            maturity_match = scenario_hist == system_hist

            passed = structural_match and payable_match and maturity_match
            results.append(
                EquivalenceResult(
                    name=name,
                    passed=passed,
                    details={
                        "scenario_agents": scenario_agents,
                        "system_agents": system_agents,
                        "scenario_payables": scenario_payables,
                        "system_payables": system_payables,
                        "scenario_maturity_hist": scenario_hist,
                        "system_maturity_hist": system_hist,
                        "structural_match": structural_match,
                        "payable_match": payable_match,
                        "maturity_match": maturity_match,
                    },
                )
            )

            if name == "ring_basic":
                s1 = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
                s2 = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
                deterministic_same_seed = _hash_scenario(s1) == _hash_scenario(s2)
                s3 = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"] + 1)
                deterministic_diff_seed = _hash_scenario(s1) != _hash_scenario(s3)

        except Exception as exc:  # noqa: BLE001
            results.append(
                EquivalenceResult(
                    name=name,
                    passed=False,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / max(1, total)

    structural_pass = sum(1 for r in results if r.details.get("structural_match") is True)
    payable_pass = sum(1 for r in results if r.details.get("payable_match") is True)
    maturity_pass = sum(1 for r in results if r.details.get("maturity_match") is True)

    cat1 = 35.0 * (structural_pass / max(1, total))
    cat2 = 30.0 * (payable_pass / max(1, total))
    cat3 = 20.0 * (maturity_pass / max(1, total))
    cat4 = 10.0 * (1.0 if deterministic_same_seed else 0.0) + 5.0 * (
        1.0 if deterministic_diff_seed else 0.0
    )

    categories = [
        CategoryResult(
            name="Structural Equivalence",
            max_points=35.0,
            earned_points=round(cat1, 3),
            details={"matches": structural_pass, "total": total},
        ),
        CategoryResult(
            name="Debt Total Equivalence",
            max_points=30.0,
            earned_points=round(cat2, 3),
            details={"matches": payable_pass, "total": total},
        ),
        CategoryResult(
            name="Maturity Distribution Equivalence",
            max_points=20.0,
            earned_points=round(cat3, 3),
            details={"matches": maturity_pass, "total": total},
        ),
        CategoryResult(
            name="Seeded Reproducibility",
            max_points=15.0,
            earned_points=round(cat4, 3),
            details={
                "deterministic_same_seed": deterministic_same_seed,
                "deterministic_diff_seed": deterministic_diff_seed,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="compile_apply::all_fixtures_match",
            passed=(passed == total),
            message=f"fixtures passed {passed}/{total}",
        ),
        CriticalCheck(
            code="compile_apply::debt_totals_exact",
            passed=(payable_pass == total),
            message=f"payable totals matched {payable_pass}/{total}",
        ),
        CriticalCheck(
            code="compile_apply::maturity_hist_exact",
            passed=(maturity_pass == total),
            message=f"maturity hist matched {maturity_pass}/{total}",
        ),
        CriticalCheck(
            code="compile_apply::seed_reproducible",
            passed=deterministic_same_seed,
            message=f"same-seed deterministic={deterministic_same_seed}",
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
        benchmark_name="Scenario Compile-to-Apply Equivalence Benchmark",
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
                "results": [
                    {"name": r.name, "passed": r.passed, "details": r.details} for r in results
                ],
                "pass_rate": pass_rate,
            },
            "generated_at_utc": generated_at,
        },
    )

    md = build_markdown_report(
        title="Scenario Compile-to-Apply Equivalence Benchmark",
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
            f"fixture_pass_rate={pass_rate:.4f}",
            f"structural_match={structural_pass}/{total}",
            f"payable_match={payable_pass}/{total}",
            f"maturity_match={maturity_pass}/{total}",
            f"deterministic_same_seed={deterministic_same_seed}",
            f"deterministic_diff_seed={deterministic_diff_seed}",
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Compile/apply equivalence benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
