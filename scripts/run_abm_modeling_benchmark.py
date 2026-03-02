#!/usr/bin/env python3
"""Run an ABM modeling standards benchmark for Bilancio.

This benchmark is complementary to engineering-quality benchmarks.
It focuses on modeling standards commonly expected in agent-based
modeling workflows:

1. Reproducibility and stochastic discipline
2. Verification and accounting invariants
3. Experiment design and statistical inference
4. Validation and behavioral plausibility
5. Transparency and model documentation
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    output: str


@dataclass
class SuiteScore:
    name: str
    max_points: float
    score: float
    passed: int
    failed: int
    pass_rate: float
    command: list[str]
    returncode: int


@dataclass
class CategoryScore:
    name: str
    max_points: float
    score: float
    components: dict[str, float]
    evidence: dict[str, Any]


@dataclass
class CriticalCheck:
    code: str
    passed: bool
    message: str


def run_command(command: list[str], cwd: Path, name: str) -> CommandResult:
    env = os.environ.copy()
    src_path = str(cwd / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        output=output.strip(),
    )


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


def run_pytest_suite(
    *,
    cwd: Path,
    name: str,
    paths: list[str],
    kexpr: str,
    max_points: float,
) -> tuple[SuiteScore, CommandResult]:
    command = [sys.executable, "-m", "pytest", "-q", "--no-cov", *paths, "-k", kexpr]
    result = run_command(command, cwd, name)
    passed, failed = parse_pytest_counts(result.output, result.returncode)
    total = passed + failed
    pass_rate = (passed / total) if total else 0.0
    score = max_points * pass_rate
    suite = SuiteScore(
        name=name,
        max_points=max_points,
        score=score,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        command=command,
        returncode=result.returncode,
    )
    return suite, result


def file_contains(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return bool(re.search(pattern, text, flags=re.MULTILINE))


def count_substring(paths: list[Path], needle: str) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        total += path.read_text(encoding="utf-8").count(needle)
    return total


def list_py_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def count_test_defs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), flags=re.MULTILINE))


def count_files(root: Path, pattern: str = "*") -> int:
    if not root.exists():
        return 0
    return len(list(root.glob(pattern)))


def public_docstring_coverage(roots: list[Path]) -> tuple[float, int, int]:
    total_public = 0
    documented_public = 0
    for root in roots:
        for file_path in list_py_files(root):
            text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(file_path))

            # Module docstring counts as one public artifact.
            total_public += 1
            if ast.get_docstring(tree):
                documented_public += 1

            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    total_public += 1
                    if ast.get_docstring(node):
                        documented_public += 1

    coverage = (documented_public / total_public) if total_public else 0.0
    return coverage, documented_public, total_public


def score_reproducibility(cwd: Path) -> tuple[CategoryScore, list[SuiteScore], list[CommandResult]]:
    suite, cmd = run_pytest_suite(
        cwd=cwd,
        name="reproducibility_suite",
        paths=[
            "tests/experiments/test_sweep_runners.py",
            "tests/unit/test_generic_sampling.py",
            "tests/engines/test_simulation_coverage.py",
        ],
        kexpr=(
            "reproducibility_with_same_seed or different_seeds_produce_different_samples "
            "or seed_increments or reproducible or different_seeds_differ "
            "or random_seed_sets_deterministic"
        ),
        max_points=12.0,
    )

    checks = {
        "balanced_base_seed": file_contains(
            cwd / "src/bilancio/experiments/balanced_comparison.py", r"base_seed"
        ),
        "bank_base_seed": file_contains(
            cwd / "src/bilancio/experiments/bank_comparison.py", r"base_seed"
        ),
        "nbfi_base_seed": file_contains(
            cwd / "src/bilancio/experiments/nbfi_comparison.py", r"base_seed"
        ),
        "balanced_next_seed": file_contains(
            cwd / "src/bilancio/experiments/balanced_comparison.py", r"def _next_seed"
        ),
        "bank_next_seed": file_contains(
            cwd / "src/bilancio/experiments/bank_comparison.py", r"def _next_seed"
        ),
        "nbfi_next_seed": file_contains(
            cwd / "src/bilancio/experiments/nbfi_comparison.py", r"def _next_seed"
        ),
        "paired_same_seed_design": file_contains(
            cwd / "src/bilancio/experiments/comparison.py", r"same seed"
        ),
        "stats_seed_controls": file_contains(
            cwd / "src/bilancio/stats/analyzer.py", r"seed: int \| None = 42"
        ),
    }
    matched = sum(1 for ok in checks.values() if ok)
    static_score = 8.0 * (matched / len(checks))
    total_score = suite.score + static_score
    category = CategoryScore(
        name="Reproducibility & Stochastic Discipline",
        max_points=20.0,
        score=total_score,
        components={
            "test_suite": suite.score,
            "seed_plumbing_static": static_score,
        },
        evidence={
            "suite_passed": suite.passed,
            "suite_failed": suite.failed,
            "seed_checks_matched": matched,
            "seed_checks_total": len(checks),
            "checks": checks,
        },
    )
    return category, [suite], [cmd]


def score_verification(cwd: Path) -> tuple[CategoryScore, list[SuiteScore], list[CommandResult]]:
    suite, cmd = run_pytest_suite(
        cwd=cwd,
        name="verification_suite",
        paths=[
            "tests/property/test_invariants_property.py",
            "tests/regression/test_simulation_properties.py",
            "tests/engines/test_no_negative_cash.py",
        ],
        kexpr="preserves_invariants or system_invariants_after_simulation or invariants_hold_active_mode",
        max_points=14.0,
    )

    invariants_module_exists = (cwd / "src/bilancio/core/invariants.py").exists()
    property_suite_exists = (cwd / "tests/property/test_invariants_property.py").exists()
    invariant_calls = count_substring(list_py_files(cwd / "tests"), "assert_invariants(")
    runtime_invariant_modes = count_substring(
        [
            cwd / "src/bilancio/experiments/ring.py",
            cwd / "src/bilancio/ui/run.py",
            cwd / "src/bilancio/runners/models.py",
        ],
        "check_invariants",
    )

    static_score = 0.0
    static_score += 2.0 if invariants_module_exists else 0.0
    static_score += 2.0 if property_suite_exists else 0.0
    static_score += 1.5 * bounded(invariant_calls / 40.0)
    static_score += 0.5 * bounded(runtime_invariant_modes / 5.0)

    category = CategoryScore(
        name="Verification & Accounting Invariants",
        max_points=20.0,
        score=suite.score + static_score,
        components={
            "test_suite": suite.score,
            "invariant_signal_static": static_score,
        },
        evidence={
            "suite_passed": suite.passed,
            "suite_failed": suite.failed,
            "invariants_module_exists": invariants_module_exists,
            "property_suite_exists": property_suite_exists,
            "assert_invariants_calls_in_tests": invariant_calls,
            "runtime_invariant_signal_count": runtime_invariant_modes,
        },
    )
    return category, [suite], [cmd]


def score_experiment_design(cwd: Path) -> tuple[CategoryScore, list[SuiteScore], list[CommandResult]]:
    suite, cmd = run_pytest_suite(
        cwd=cwd,
        name="stats_design_suite",
        paths=[
            "tests/unit/test_stats.py",
            "tests/unit/test_sweep_analysis.py",
        ],
        kexpr=(
            "reproducibility or paired_effect or effect_table_summary "
            "or sensitivity_ranking or significance_with_enough_replicates "
            "or skips_cells_with_one_replicate or basic_sensitivity"
        ),
        max_points=12.0,
    )

    module_checks = {
        "bootstrap_module": (cwd / "src/bilancio/stats/bootstrap.py").exists(),
        "effect_size_module": (cwd / "src/bilancio/stats/effect_size.py").exists(),
        "significance_module": (cwd / "src/bilancio/stats/significance.py").exists(),
        "sensitivity_module": (cwd / "src/bilancio/stats/sensitivity.py").exists(),
        "cell_module": (cwd / "src/bilancio/stats/cell.py").exists(),
        "analyzer_module": (cwd / "src/bilancio/stats/analyzer.py").exists(),
    }
    api_checks = {
        "summarize_paired_cell": file_contains(cwd / "src/bilancio/stats/cell.py", r"def summarize_paired_cell"),
        "paired_t_test": file_contains(cwd / "src/bilancio/stats/significance.py", r"def paired_t_test"),
        "cohens_d": file_contains(cwd / "src/bilancio/stats/effect_size.py", r"def cohens_d"),
        "morris_screening": file_contains(cwd / "src/bilancio/stats/sensitivity.py", r"def morris_screening"),
        "treatment_effect_table": file_contains(cwd / "src/bilancio/stats/analyzer.py", r"def treatment_effect_table"),
    }

    matched = sum(1 for ok in module_checks.values() if ok) + sum(1 for ok in api_checks.values() if ok)
    total_checks = len(module_checks) + len(api_checks)
    static_score = 8.0 * (matched / total_checks)

    category = CategoryScore(
        name="Experiment Design & Statistical Inference",
        max_points=20.0,
        score=suite.score + static_score,
        components={
            "test_suite": suite.score,
            "statistical_tooling_static": static_score,
        },
        evidence={
            "suite_passed": suite.passed,
            "suite_failed": suite.failed,
            "matched_checks": matched,
            "total_checks": total_checks,
            "module_checks": module_checks,
            "api_checks": api_checks,
        },
    )
    return category, [suite], [cmd]


def score_validation(cwd: Path) -> tuple[CategoryScore, list[SuiteScore], list[CommandResult]]:
    core_suite, core_cmd = run_pytest_suite(
        cwd=cwd,
        name="behavioral_validation_suite",
        paths=["tests/regression/test_simulation_properties.py"],
        kexpr="lending_effect_is_nonzero or trading_effect_is_reasonable or system_invariants_after_simulation",
        max_points=6.0,
    )
    comparative_suite, comparative_cmd = run_pytest_suite(
        cwd=cwd,
        name="behavioral_comparative_statics_suite",
        paths=["tests/regression/test_simulation_properties.py"],
        kexpr=(
            "defaults_increase_with_lower_kappa or nbfi_creates_loans_when_shortfalls_exist "
            "or bank_lending_restricted_at_low_kappa or cb_backstop_bounded_at_low_kappa "
            "or settlement_forecast_nonzero_with_cross_bank_payables or banking_differs_from_passive"
        ),
        max_points=6.0,
    )

    regression_tests = count_test_defs(cwd / "tests/regression/test_simulation_properties.py")
    integration_test_files = count_files(cwd / "tests/integration", "test_*.py")
    modeling_docs = (
        count_files(cwd / "docs/analysis", "*")
        + count_files(cwd / "docs/research", "*")
        + count_files(cwd / "docs/dealer_ring", "*")
    )
    stylized_signal_tests = count_substring(list_py_files(cwd / "tests"), "stylized")

    static_score = 0.0
    static_score += 2.5 * bounded(regression_tests / 12.0)
    static_score += 1.5 * bounded(integration_test_files / 12.0)
    static_score += 2.5 * bounded(modeling_docs / 24.0)
    static_score += 1.5 * bounded(stylized_signal_tests / 8.0)

    test_suite_score = core_suite.score + comparative_suite.score

    category = CategoryScore(
        name="Validation & Behavioral Plausibility",
        max_points=20.0,
        score=test_suite_score + static_score,
        components={
            "test_suite_core": core_suite.score,
            "test_suite_comparative_statics": comparative_suite.score,
            "validation_signal_static": static_score,
        },
        evidence={
            "core_suite_passed": core_suite.passed,
            "core_suite_failed": core_suite.failed,
            "comparative_suite_passed": comparative_suite.passed,
            "comparative_suite_failed": comparative_suite.failed,
            "regression_test_count": regression_tests,
            "integration_test_file_count": integration_test_files,
            "modeling_docs_count": modeling_docs,
            "stylized_signal_test_mentions": stylized_signal_tests,
        },
    )
    return category, [core_suite, comparative_suite], [core_cmd, comparative_cmd]


def score_transparency(cwd: Path) -> CategoryScore:
    doc_roots = [
        cwd / "src/bilancio/engines",
        cwd / "src/bilancio/experiments",
        cwd / "src/bilancio/scenarios",
        cwd / "src/bilancio/stats",
    ]
    doc_cov, documented, public_items = public_docstring_coverage(doc_roots)

    scenario_examples = count_files(cwd / "examples/scenarios", "*.yaml")
    protocol_docs = [
        cwd / "docs/concepts.md",
        cwd / "docs/architecture.md",
        cwd / "docs/quickstart.md",
        cwd / "docs/cli.md",
        cwd / "docs/exercises_scenarios.md",
    ]
    protocol_docs_present = sum(1 for p in protocol_docs if p.exists())

    # Deliberately strict ABM-standard artifacts (ODD-style and validation protocol docs).
    standards_artifacts = [
        cwd / "docs/spec/odd_protocol.md",
        cwd / "docs/spec/calibration.md",
        cwd / "docs/spec/validation_matrix.md",
        cwd / "docs/spec/experiment_protocol.md",
        cwd / "docs/spec/parameter_table.md",
        cwd / "docs/spec/stylized_facts.md",
    ]
    standards_artifacts_present = sum(1 for p in standards_artifacts if p.exists())

    doc_cov_score = 8.0 * bounded(doc_cov / 0.80)
    scenario_score = 4.0 * bounded(scenario_examples / 20.0)
    protocol_score = 4.0 * bounded(protocol_docs_present / len(protocol_docs))
    standards_score = 4.0 * bounded(standards_artifacts_present / len(standards_artifacts))

    return CategoryScore(
        name="Transparency & Model Documentation",
        max_points=20.0,
        score=doc_cov_score + scenario_score + protocol_score + standards_score,
        components={
            "public_docstring_coverage": doc_cov_score,
            "scenario_examples": scenario_score,
            "protocol_docs_baseline": protocol_score,
            "abm_standards_artifacts": standards_score,
        },
        evidence={
            "docstring_coverage_ratio": doc_cov,
            "documented_public_items": documented,
            "public_items": public_items,
            "scenario_examples": scenario_examples,
            "protocol_docs_present": protocol_docs_present,
            "protocol_docs_total": len(protocol_docs),
            "standards_artifacts_present": standards_artifacts_present,
            "standards_artifacts_total": len(standards_artifacts),
        },
    )


def grade_for_score(total: float) -> str:
    if total >= 95.0:
        return "A+"
    if total >= 90.0:
        return "A"
    if total >= 80.0:
        return "B"
    if total >= 70.0:
        return "C"
    if total >= 60.0:
        return "D"
    return "F"


def worse_grade(left: str, right: str) -> str:
    order = ["A+", "A", "B", "C", "D", "F"]
    left_idx = order.index(left)
    right_idx = order.index(right)
    return left if left_idx >= right_idx else right


def cap_grade_for_critical_failures(base_grade: str, failure_count: int) -> str:
    if failure_count <= 0:
        return base_grade
    if failure_count >= 5:
        cap = "F"
    elif failure_count >= 3:
        cap = "D"
    else:
        cap = "C"
    return worse_grade(base_grade, cap)


def evaluate_critical_checks(
    *, cwd: Path, categories: list[CategoryScore], suites: list[SuiteScore]
) -> list[CriticalCheck]:
    suite_by_name = {suite.name: suite for suite in suites}
    checks: list[CriticalCheck] = []

    required_suites: list[tuple[str, int]] = [
        ("reproducibility_suite", 5),
        ("verification_suite", 5),
        ("stats_design_suite", 5),
        ("behavioral_validation_suite", 2),
        ("behavioral_comparative_statics_suite", 4),
    ]
    for suite_name, min_passed in required_suites:
        suite = suite_by_name.get(suite_name)
        passed = (
            suite is not None
            and suite.returncode == 0
            and suite.failed == 0
            and suite.passed >= min_passed
        )
        if suite is None:
            message = f"Missing suite result for `{suite_name}`."
        else:
            message = (
                f"`{suite_name}` must be fully green and include >= {min_passed} tests "
                f"(got passed={suite.passed}, failed={suite.failed}, returncode={suite.returncode})."
            )
        checks.append(
            CriticalCheck(
                code=f"suite::{suite_name}",
                passed=passed,
                message=message,
            )
        )

    validation = next(
        (cat for cat in categories if cat.name == "Validation & Behavioral Plausibility"),
        None,
    )
    validation_evidence = validation.evidence if validation else {}
    stylized_mentions = int(validation_evidence.get("stylized_signal_test_mentions", 0))
    regression_tests = int(validation_evidence.get("regression_test_count", 0))
    checks.append(
        CriticalCheck(
            code="validation::stylized_signal_tests",
            passed=stylized_mentions >= 2,
            message=f"Need >=2 stylized-signal test mentions (got {stylized_mentions}).",
        )
    )
    checks.append(
        CriticalCheck(
            code="validation::regression_depth",
            passed=regression_tests >= 8,
            message=f"Need >=8 behavioral regression tests in core suite file (got {regression_tests}).",
        )
    )

    transparency = next(
        (cat for cat in categories if cat.name == "Transparency & Model Documentation"),
        None,
    )
    transparency_evidence = transparency.evidence if transparency else {}
    doc_cov_ratio = float(transparency_evidence.get("docstring_coverage_ratio", 0.0))
    standards_present = int(transparency_evidence.get("standards_artifacts_present", 0))
    checks.append(
        CriticalCheck(
            code="transparency::docstring_floor",
            passed=doc_cov_ratio >= 0.65,
            message=f"Public docstring coverage must be >=0.65 (got {doc_cov_ratio:.3f}).",
        )
    )
    checks.append(
        CriticalCheck(
            code="transparency::standards_artifact_count",
            passed=standards_present >= 4,
            message=f"Need >=4 ABM standards artifacts under docs/spec (got {standards_present}).",
        )
    )

    mandatory_docs = [
        cwd / "docs/spec/odd_protocol.md",
        cwd / "docs/spec/validation_matrix.md",
        cwd / "docs/spec/experiment_protocol.md",
    ]
    missing = [str(path.relative_to(cwd)) for path in mandatory_docs if not path.exists()]
    checks.append(
        CriticalCheck(
            code="transparency::mandatory_abm_docs",
            passed=not missing,
            message=(
                "Mandatory ABM docs present: ODD protocol, validation matrix, experiment protocol."
                if not missing
                else f"Missing mandatory ABM docs: {', '.join(missing)}."
            ),
        )
    )

    return checks


def recommendation_for_category(name: str) -> list[str]:
    recommendations = {
        "Reproducibility & Stochastic Discipline": [
            "Expand deterministic-seed tests to cloud and runner resume paths.",
            "Standardize seed provenance in every exported artifact schema.",
        ],
        "Verification & Accounting Invariants": [
            "Add invariant checks to more cross-subsystem integration tests.",
            "Increase property-based tests for settlement and lending interactions.",
        ],
        "Experiment Design & Statistical Inference": [
            "Enforce minimum replicate counts in all sweep runners before effect reporting.",
            "Add CI checks that block reports when paired-cell assumptions are violated.",
        ],
        "Validation & Behavioral Plausibility": [
            "Codify stylized-fact tests (named explicitly) for key macro outcomes.",
            "Add validation tests against known comparative statics across kappa/mu/c.",
        ],
        "Transparency & Model Documentation": [
            "Add ABM standards artifacts (ODD protocol, calibration note, validation matrix).",
            "Raise public docstring coverage in engines/experiments/scenarios/stats.",
        ],
    }
    return recommendations.get(name, [])


def build_markdown_report(
    *,
    generated_at: str,
    target_score: float,
    categories: list[CategoryScore],
    suites: list[SuiteScore],
    total_score: float,
    status: str,
    grade: str,
    base_grade: str,
    meets_target: bool,
    critical_checks: list[CriticalCheck],
) -> str:
    lines: list[str] = []
    lines.append("# ABM Modeling Benchmark")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append(f"Target score: **{target_score:.1f}/100**")
    lines.append("")
    lines.append("## Score Summary")
    lines.append("")
    lines.append("| Category | Score | Max |")
    lines.append("|---|---:|---:|")
    for cat in categories:
        lines.append(f"| {cat.name} | {cat.score:.2f} | {cat.max_points:.1f} |")
    lines.append(f"| **Total** | **{total_score:.2f}** | **100.0** |")
    lines.append("")
    lines.append(f"Status: **{status}**")
    lines.append(f"Grade: **{grade}** (base score grade: {base_grade})")
    lines.append(f"Target met: **{'yes' if meets_target else 'no'}**")
    lines.append(f"Gap to target: **{max(0.0, target_score - total_score):.2f}**")
    lines.append("")
    lines.append("## Critical Gates")
    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("|---|---|---|")
    for check in critical_checks:
        gate_status = "PASS" if check.passed else "FAIL"
        lines.append(f"| `{check.code}` | {gate_status} | {check.message} |")
    lines.append("")
    lines.append("## Suite Results")
    lines.append("")
    lines.append("| Suite | Passed | Failed | Pass rate | Score |")
    lines.append("|---|---:|---:|---:|---:|")
    for suite in suites:
        lines.append(
            f"| {suite.name} | {suite.passed} | {suite.failed} | "
            f"{suite.pass_rate*100:.1f}% | {suite.score:.2f}/{suite.max_points:.1f} |"
        )
    lines.append("")
    lines.append("## Priority Improvements")
    lines.append("")
    for cat in sorted(categories, key=lambda c: c.score / c.max_points)[:3]:
        ratio = cat.score / cat.max_points
        if ratio >= 0.90:
            continue
        lines.append(f"### {cat.name}")
        lines.append(f"Current: {cat.score:.2f}/{cat.max_points:.1f} ({ratio*100:.1f}%)")
        for rec in recommendation_for_category(cat.name):
            lines.append(f"- {rec}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ABM modeling standards benchmark.")
    parser.add_argument(
        "--target-score",
        type=float,
        default=90.0,
        help="Target score to reach (default: 90.0).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/abm_modeling_benchmark_report.json",
        help="Path for JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/abm_modeling_benchmark_report.md",
        help="Path for Markdown report.",
    )
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parents[1]

    categories: list[CategoryScore] = []
    suites: list[SuiteScore] = []
    commands: list[CommandResult] = []

    for scorer in (
        score_reproducibility,
        score_verification,
        score_experiment_design,
        score_validation,
    ):
        category, category_suites, category_commands = scorer(cwd)
        categories.append(category)
        suites.extend(category_suites)
        commands.extend(category_commands)

    categories.append(score_transparency(cwd))

    total_score = sum(cat.score for cat in categories)
    base_grade = grade_for_score(total_score)
    critical_checks = evaluate_critical_checks(cwd=cwd, categories=categories, suites=suites)
    critical_failures = [check for check in critical_checks if not check.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"
    generated_at = datetime.now(timezone.utc).isoformat()

    report = {
        "generated_at_utc": generated_at,
        "target_score": args.target_score,
        "total_score": total_score,
        "status": status,
        "meets_target": meets_target,
        "base_grade": base_grade,
        "grade": grade,
        "gap_to_target": max(0.0, args.target_score - total_score),
        "critical_checks": [asdict(check) for check in critical_checks],
        "critical_failures": [asdict(check) for check in critical_failures],
        "categories": [asdict(c) for c in categories],
        "suites": [asdict(s) for s in suites],
        "commands": [asdict(c) for c in commands],
    }

    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(
        build_markdown_report(
            generated_at=generated_at,
            target_score=args.target_score,
            categories=categories,
            suites=suites,
            total_score=total_score,
            status=status,
            grade=grade,
            base_grade=base_grade,
            meets_target=meets_target,
            critical_checks=critical_checks,
        ),
        encoding="utf-8",
    )

    print(f"ABM benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"Target score: {args.target_score:.1f} (gap {max(0.0, args.target_score - total_score):.2f})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
