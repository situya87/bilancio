#!/usr/bin/env python3
"""Scenario Plugin Contract Benchmark.

Validates plugin protocol/schema/default/error contract against a fixture corpus.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

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
from bilancio.scenarios.protocol import ScenarioPlugin
from bilancio.scenarios.registry import get_plugin


@dataclass
class FixtureResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario plugin contract benchmark.")
    parser.add_argument("--target-score", type=float, default=95.0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/plugin_contract_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/plugin_contract_benchmark_report.md",
    )
    return parser.parse_args()


def _hash_scenario(scenario: dict[str, Any]) -> str:
    return json.dumps(scenario, sort_keys=True, default=str, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    t0 = perf_counter()

    plugin = get_plugin("kalecki_ring")

    # Category 1: protocol + metadata
    protocol_ok = isinstance(plugin, ScenarioPlugin)
    meta = plugin.metadata
    meta_checks = {
        "name": bool(meta.name),
        "display_name": bool(meta.display_name),
        "description": bool(meta.description),
        "version_positive": meta.version >= 1,
        "supports_dealer_flag": isinstance(meta.supports_dealer, bool),
        "supports_lender_flag": isinstance(meta.supports_lender, bool),
    }
    meta_match = sum(1 for ok in meta_checks.values() if ok)
    cat1 = 5.0 * (1.0 if protocol_ok else 0.0) + 15.0 * (meta_match / len(meta_checks))

    # Category 2: dimensions + config_model contract
    dims = plugin.parameter_dimensions()
    names = [d.name for d in dims]
    unique_names = len(names) == len(set(names)) and len(names) > 0

    defaults_in_range = 0
    defaults_total = 0
    for dim in dims:
        low, high = dim.valid_range
        for dv in dim.default_values:
            defaults_total += 1
            in_low = low is None or dv >= low
            in_high = high is None or dv <= high
            if in_low and in_high:
                defaults_in_range += 1

    model_cls = plugin.config_model()
    config_model_ok = isinstance(model_cls, type) and issubclass(model_cls, BaseModel)

    dim_score_unique = 8.0 if unique_names else 0.0
    dim_score_defaults = 12.0 * (
        (defaults_in_range / defaults_total) if defaults_total > 0 else 0.0
    )
    dim_score_model = 5.0 if config_model_ok else 0.0
    cat2 = dim_score_unique + dim_score_defaults + dim_score_model

    # Category 3: valid fixture corpus contract compliance
    valid_fixtures = [
        {
            "name": "basic_mid",
            "params": {
                "kappa": Decimal("1.0"),
                "concentration": Decimal("1.0"),
                "mu": Decimal("0.5"),
                "monotonicity": Decimal("0"),
            },
            "base": {"n_agents": 10, "maturity_days": 5, "Q_total": Decimal("1000")},
            "seed": 42,
        },
        {
            "name": "stress_low_kappa",
            "params": {
                "kappa": Decimal("0.3"),
                "concentration": Decimal("0.5"),
                "mu": Decimal("0.2"),
                "monotonicity": Decimal("-1"),
            },
            "base": {"n_agents": 20, "maturity_days": 8, "Q_total": Decimal("2000")},
            "seed": 7,
        },
        {
            "name": "high_liquidity",
            "params": {
                "kappa": Decimal("2.0"),
                "concentration": Decimal("2.0"),
                "mu": Decimal("0.8"),
                "monotonicity": Decimal("1"),
            },
            "base": {"n_agents": 15, "maturity_days": 6, "Q_total": Decimal("1500")},
            "seed": 99,
        },
        {
            "name": "balanced_mode_active",
            "params": {
                "kappa": Decimal("0.8"),
                "concentration": Decimal("1.0"),
                "mu": Decimal("0.5"),
                "monotonicity": Decimal("0"),
            },
            "base": {
                "n_agents": 12,
                "maturity_days": 6,
                "Q_total": Decimal("1200"),
                "mode": "active",
            },
            "seed": 123,
        },
    ]

    valid_results: list[FixtureResult] = []
    deterministic_same_seed = True
    deterministic_different_seed = False

    for fx in valid_fixtures:
        try:
            scenario = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
            scenario_copy = copy.deepcopy(scenario)
            ScenarioConfig(**preprocess_config(scenario_copy))
            valid_results.append(FixtureResult(fx["name"], True, "compiled and validated"))

            if fx["name"] == "basic_mid":
                same_1 = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
                same_2 = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
                deterministic_same_seed = _hash_scenario(same_1) == _hash_scenario(same_2)

                diff = plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"] + 1)
                deterministic_different_seed = _hash_scenario(same_1) != _hash_scenario(diff)
        except Exception as exc:  # noqa: BLE001
            valid_results.append(FixtureResult(fx["name"], False, f"{type(exc).__name__}: {exc}"))

    valid_passed = sum(1 for r in valid_results if r.passed)
    valid_total = len(valid_results)
    valid_pass_rate = valid_passed / max(1, valid_total)
    cat3 = 30.0 * valid_pass_rate + 5.0 * (1.0 if deterministic_different_seed else 0.0)

    # Category 4: invalid fixture corpus (error semantics)
    invalid_fixtures = [
        {
            "name": "missing_kappa",
            "params": {"concentration": Decimal("1"), "mu": Decimal("0.5")},
            "base": {"n_agents": 10, "maturity_days": 5},
            "seed": 1,
        },
        {
            "name": "negative_kappa",
            "params": {"kappa": Decimal("-0.1"), "concentration": Decimal("1"), "mu": Decimal("0.5")},
            "base": {"n_agents": 10, "maturity_days": 5},
            "seed": 1,
        },
        {
            "name": "bad_monotonicity",
            "params": {"kappa": Decimal("1"), "concentration": Decimal("1"), "mu": Decimal("0.5"), "monotonicity": Decimal("2")},
            "base": {"n_agents": 10, "maturity_days": 5},
            "seed": 1,
        },
    ]

    invalid_results: list[FixtureResult] = []
    for fx in invalid_fixtures:
        try:
            plugin.compile(fx["params"], base_config=fx["base"], seed=fx["seed"])
            invalid_results.append(FixtureResult(fx["name"], False, "unexpectedly compiled"))
        except (ValueError, KeyError, ValidationError, TypeError) as exc:
            invalid_results.append(FixtureResult(fx["name"], True, f"raised {type(exc).__name__}"))
        except Exception as exc:  # noqa: BLE001
            invalid_results.append(FixtureResult(fx["name"], True, f"raised {type(exc).__name__}"))

    invalid_passed = sum(1 for r in invalid_results if r.passed)
    invalid_total = len(invalid_results)
    invalid_pass_rate = invalid_passed / max(1, invalid_total)
    cat4 = 20.0 * invalid_pass_rate

    categories = [
        CategoryResult(
            name="Protocol & Metadata",
            max_points=20.0,
            earned_points=round(cat1, 3),
            details={"protocol_ok": protocol_ok, "metadata_checks": meta_checks},
        ),
        CategoryResult(
            name="Dimensions & Schema",
            max_points=25.0,
            earned_points=round(cat2, 3),
            details={
                "unique_dimension_names": unique_names,
                "dimension_names": names,
                "defaults_in_range": defaults_in_range,
                "defaults_total": defaults_total,
                "config_model_ok": config_model_ok,
            },
        ),
        CategoryResult(
            name="Valid Fixture Compliance",
            max_points=35.0,
            earned_points=round(cat3, 3),
            details={
                "pass_rate": valid_pass_rate,
                "results": [r.__dict__ for r in valid_results],
                "deterministic_same_seed": deterministic_same_seed,
                "deterministic_different_seed": deterministic_different_seed,
            },
        ),
        CategoryResult(
            name="Invalid Fixture Rejection",
            max_points=20.0,
            earned_points=round(cat4, 3),
            details={
                "pass_rate": invalid_pass_rate,
                "results": [r.__dict__ for r in invalid_results],
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="plugin_contract::protocol_conformance",
            passed=protocol_ok,
            message=f"isinstance(plugin, ScenarioPlugin)={protocol_ok}",
        ),
        CriticalCheck(
            code="plugin_contract::valid_fixture_pass_rate",
            passed=(valid_passed == valid_total),
            message=f"valid fixtures passed {valid_passed}/{valid_total}",
        ),
        CriticalCheck(
            code="plugin_contract::invalid_fixture_fail_rate",
            passed=(invalid_passed == invalid_total),
            message=f"invalid fixtures rejected {invalid_passed}/{invalid_total}",
        ),
        CriticalCheck(
            code="plugin_contract::deterministic_same_seed",
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

    generated_at = generated_at_utc()

    report = report_dict(
        benchmark_name="Scenario Plugin Contract Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=(perf_counter() - t0),
        categories=categories,
        critical_checks=checks,
        extra={"generated_at_utc": generated_at},
    )

    md = build_markdown_report(
        title="Scenario Plugin Contract Benchmark",
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
            f"plugin={meta.name} (version={meta.version})",
            f"valid_fixture_pass_rate={valid_pass_rate:.4f}",
            f"invalid_fixture_pass_rate={invalid_pass_rate:.4f}",
            f"deterministic_same_seed={deterministic_same_seed}",
            f"deterministic_different_seed={deterministic_different_seed}",
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Plugin contract benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
