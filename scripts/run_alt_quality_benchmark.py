#!/usr/bin/env python3
"""Run an alternative, reproducible quality benchmark for Bilancio."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    output: str


@dataclass
class StaticMetrics:
    source_files: int
    total_functions: int
    typed_functions: int
    long_functions_over_80: int
    long_files_over_800: int
    broad_exception_handlers: int
    public_doc_items: int
    public_doc_items_with_docstring: int


@dataclass
class BenchmarkScores:
    verification_strength: float
    static_correctness: float
    maintainability_risk: float
    api_type_clarity: float
    documentation_signal: float
    total: float
    grade: str


def run_command(command: list[str], cwd: Path, name: str) -> CommandResult:
    env = os.environ.copy()
    src_path = str(cwd / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"
    )
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            name=name,
            command=command,
            returncode=127,
            output=str(exc),
        )
    output = (completed.stdout or "") + (completed.stderr or "")
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        output=output.strip(),
    )


def missing_python_module(result: CommandResult, module_name: str) -> bool:
    text = result.output.lower()
    return (
        result.returncode != 0
        and f"no module named {module_name.lower()}" in text
    )


def parse_pytest_counts(output: str) -> tuple[int, int]:
    passed_match = re.search(r"(\d+)\s+passed", output)
    failed_match = re.search(r"(\d+)\s+failed", output)
    error_match = re.search(r"(\d+)\s+error", output)

    passed = int(passed_match.group(1)) if passed_match else 0
    failures = 0
    if failed_match:
        failures += int(failed_match.group(1))
    if error_match:
        failures += int(error_match.group(1))

    return passed, failures


def parse_mypy_errors(output: str, returncode: int) -> int:
    if returncode == 0:
        return 0

    found_match = re.search(r"Found (\d+) errors?", output)
    if found_match:
        return int(found_match.group(1))

    return len(re.findall(r": error:", output))


def parse_ruff_issues(output: str, returncode: int) -> int:
    if returncode == 0:
        return 0

    found_match = re.search(r"Found (\d+) errors?", output)
    if found_match:
        return int(found_match.group(1))

    issue_lines = re.findall(r"^[^:\n]+:\d+:\d+:\s+[A-Z]+\d+", output, flags=re.MULTILINE)
    return len(issue_lines)


def annotation_coverage_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args_to_check: list[ast.arg] = []
    args_to_check.extend(node.args.posonlyargs)
    args_to_check.extend(node.args.args)
    args_to_check.extend(node.args.kwonlyargs)
    if node.args.vararg is not None:
        args_to_check.append(node.args.vararg)
    if node.args.kwarg is not None:
        args_to_check.append(node.args.kwarg)

    for index, arg in enumerate(args_to_check):
        if index == 0 and arg.arg in {"self", "cls"}:
            continue
        if arg.annotation is None:
            return False

    return node.returns is not None


def is_broad_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True

    if isinstance(handler.type, ast.Name):
        return handler.type.id == "Exception"

    if isinstance(handler.type, ast.Tuple):
        for element in handler.type.elts:
            if isinstance(element, ast.Name) and element.id == "Exception":
                return True

    return False


def collect_static_metrics(src_dir: Path) -> StaticMetrics:
    source_files = sorted(src_dir.rglob("*.py"))

    total_functions = 0
    typed_functions = 0
    long_functions_over_80 = 0
    long_files_over_800 = 0
    broad_exception_handlers = 0
    public_doc_items = 0
    public_doc_items_with_docstring = 0

    for file_path in source_files:
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > 800:
            long_files_over_800 += 1

        tree = ast.parse(text, filename=str(file_path))
        module_docstring = ast.get_docstring(tree)
        public_doc_items += 1
        if module_docstring:
            public_doc_items_with_docstring += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                total_functions += 1
                if annotation_coverage_for_function(node):
                    typed_functions += 1

                if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno:
                    function_length = node.end_lineno - node.lineno + 1
                    if function_length > 80:
                        long_functions_over_80 += 1

            if isinstance(node, ast.ExceptHandler) and is_broad_exception(node):
                broad_exception_handlers += 1

        for top_level in tree.body:
            if isinstance(top_level, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if top_level.name.startswith("_"):
                    continue
                public_doc_items += 1
                if ast.get_docstring(top_level):
                    public_doc_items_with_docstring += 1

    return StaticMetrics(
        source_files=len(source_files),
        total_functions=total_functions,
        typed_functions=typed_functions,
        long_functions_over_80=long_functions_over_80,
        long_files_over_800=long_files_over_800,
        broad_exception_handlers=broad_exception_handlers,
        public_doc_items=public_doc_items,
        public_doc_items_with_docstring=public_doc_items_with_docstring,
    )


def bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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


def compute_scores(
    *,
    tests_passed: int,
    tests_failed: int,
    coverage_percent: float,
    mypy_errors: int,
    ruff_issues: int,
    metrics: StaticMetrics,
) -> BenchmarkScores:
    test_total = tests_passed + tests_failed
    pass_rate = (tests_passed / test_total) if test_total else 0.0

    test_execution = 20.0 * pass_rate
    coverage_component = bounded((coverage_percent / 85.0) * 15.0, 0.0, 15.0)
    verification_strength = test_execution + coverage_component

    mypy_component = max(0.0, 15.0 - (mypy_errors * 0.2))
    ruff_component = max(0.0, 10.0 - (ruff_issues * 0.1))
    static_correctness = mypy_component + ruff_component

    long_file_ratio = (
        (metrics.long_files_over_800 / metrics.source_files) if metrics.source_files else 0.0
    )
    long_function_ratio = (
        (metrics.long_functions_over_80 / metrics.total_functions) if metrics.total_functions else 0.0
    )
    broad_exception_ratio = (
        (metrics.broad_exception_handlers / metrics.total_functions) if metrics.total_functions else 0.0
    )
    # Maintainability model is ratio-based so larger codebases are not over-penalized
    # solely due to absolute size. Ratios above 18% for long files/functions drop to 0.
    file_size_component = bounded(((0.18 - long_file_ratio) / 0.18) * 7.0, 0.0, 7.0)
    function_size_component = bounded(((0.18 - long_function_ratio) / 0.18) * 7.0, 0.0, 7.0)
    # Broad handlers above 5% of functions drop to 0.
    exception_component = bounded(((0.05 - broad_exception_ratio) / 0.05) * 6.0, 0.0, 6.0)
    maintainability_risk = file_size_component + function_size_component + exception_component

    typed_ratio = (
        (metrics.typed_functions / metrics.total_functions)
        if metrics.total_functions
        else 0.0
    )
    api_type_clarity = bounded((typed_ratio / 0.95) * 10.0, 0.0, 10.0)

    doc_ratio = (
        (metrics.public_doc_items_with_docstring / metrics.public_doc_items)
        if metrics.public_doc_items
        else 0.0
    )
    documentation_signal = bounded((doc_ratio / 0.8) * 10.0, 0.0, 10.0)

    total = (
        verification_strength
        + static_correctness
        + maintainability_risk
        + api_type_clarity
        + documentation_signal
    )
    total = bounded(total, 0.0, 100.0)

    return BenchmarkScores(
        verification_strength=round(verification_strength, 2),
        static_correctness=round(static_correctness, 2),
        maintainability_risk=round(maintainability_risk, 2),
        api_type_clarity=round(api_type_clarity, 2),
        documentation_signal=round(documentation_signal, 2),
        total=round(total, 2),
        grade=grade_for_score(total),
    )


def extract_coverage_percent(coverage_json_path: Path) -> float:
    if not coverage_json_path.exists():
        return 0.0
    data: dict[str, Any] = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    percent = totals.get("percent_covered")
    if isinstance(percent, int | float):
        return float(percent)
    percent_display = totals.get("percent_covered_display")
    try:
        return float(percent_display)
    except (TypeError, ValueError):
        return 0.0


def generate_markdown_report(
    *,
    scores: BenchmarkScores,
    tests_passed: int,
    tests_failed: int,
    coverage_percent: float,
    mypy_errors: int,
    ruff_issues: int,
    metrics: StaticMetrics,
) -> str:
    typed_ratio = (
        (metrics.typed_functions / metrics.total_functions) if metrics.total_functions else 0.0
    )
    long_file_ratio = (metrics.long_files_over_800 / metrics.source_files) if metrics.source_files else 0.0
    long_function_ratio = (
        (metrics.long_functions_over_80 / metrics.total_functions) if metrics.total_functions else 0.0
    )
    broad_exception_ratio = (
        (metrics.broad_exception_handlers / metrics.total_functions) if metrics.total_functions else 0.0
    )
    doc_ratio = (
        (metrics.public_doc_items_with_docstring / metrics.public_doc_items)
        if metrics.public_doc_items
        else 0.0
    )
    lines = [
        "# Alternative Quality Benchmark Report",
        "",
        "## Scorecard",
        "",
        f"- Overall score: **{scores.total}/100**",
        f"- Grade: **{scores.grade}**",
        "",
        "## Category Scores",
        "",
        f"- Verification Strength (35): {scores.verification_strength}",
        f"- Static Correctness (25): {scores.static_correctness}",
        f"- Maintainability Risk (20): {scores.maintainability_risk}",
        f"- API Type Clarity (10): {scores.api_type_clarity}",
        f"- Documentation Signal (10): {scores.documentation_signal}",
        "",
        "## Raw Signals",
        "",
        f"- Pytest: {tests_passed} passed, {tests_failed} failed/errors",
        f"- Coverage: {coverage_percent:.2f}%",
        f"- mypy errors: {mypy_errors}",
        f"- ruff issues: {ruff_issues}",
        f"- Source files scanned: {metrics.source_files}",
        (
            "- Typed functions: "
            f"{metrics.typed_functions}/{metrics.total_functions} ({typed_ratio:.2%})"
        ),
        (
            "- Public docstring coverage: "
            f"{metrics.public_doc_items_with_docstring}/{metrics.public_doc_items} "
            f"({doc_ratio:.2%})"
        ),
        f"- Files > 800 LOC: {metrics.long_files_over_800}",
        f"- Functions > 80 LOC: {metrics.long_functions_over_80}",
        f"- Broad exception handlers: {metrics.broad_exception_handlers}",
        f"- Long-file ratio: {long_file_ratio:.2%}",
        f"- Long-function ratio: {long_function_ratio:.2%}",
        f"- Broad-exception ratio: {broad_exception_ratio:.2%}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing src/ and tests/.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("temp/alt_quality_benchmark_report.json"),
        help="Path for JSON report output.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("temp/alt_quality_benchmark_report.md"),
        help="Path for Markdown report output.",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip pytest/mypy/ruff execution and only run static AST metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    report_json = (project_root / args.report_json).resolve()
    report_md = (project_root / args.report_md).resolve()
    coverage_json_path = (project_root / "temp/alt_quality_coverage.json").resolve()

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    coverage_json_path.parent.mkdir(parents=True, exist_ok=True)

    tests_passed = 0
    tests_failed = 0
    mypy_errors = 0
    ruff_issues = 0
    command_results: list[CommandResult] = []

    if not args.skip_execution:
        pytest_result = run_command(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=project_root,
            name="pytest",
        )
        command_results.append(pytest_result)
        tests_passed, tests_failed = parse_pytest_counts(pytest_result.output)

        coverage_result = run_command(
            [sys.executable, "-m", "coverage", "json", "-o", str(coverage_json_path)],
            cwd=project_root,
            name="coverage-json",
        )
        command_results.append(coverage_result)

        mypy_result = run_command(
            [sys.executable, "-m", "mypy", "src/bilancio"],
            cwd=project_root,
            name="mypy",
        )
        command_results.append(mypy_result)
        if missing_python_module(mypy_result, "mypy"):
            mypy_result = run_command(
                ["uv", "run", "--extra", "dev", "mypy", "src/bilancio"],
                cwd=project_root,
                name="mypy-uv-fallback",
            )
            command_results.append(mypy_result)
        mypy_errors = parse_mypy_errors(mypy_result.output, mypy_result.returncode)

        ruff_result = run_command(
            [sys.executable, "-m", "ruff", "check", "src", "tests"],
            cwd=project_root,
            name="ruff",
        )
        command_results.append(ruff_result)
        if missing_python_module(ruff_result, "ruff"):
            ruff_result = run_command(
                ["uv", "run", "--extra", "dev", "ruff", "check", "src", "tests"],
                cwd=project_root,
                name="ruff-uv-fallback",
            )
            command_results.append(ruff_result)
        ruff_issues = parse_ruff_issues(ruff_result.output, ruff_result.returncode)

    coverage_percent = extract_coverage_percent(coverage_json_path)
    metrics = collect_static_metrics(project_root / "src" / "bilancio")
    scores = compute_scores(
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        coverage_percent=coverage_percent,
        mypy_errors=mypy_errors,
        ruff_issues=ruff_issues,
        metrics=metrics,
    )

    report = {
        "benchmark": "Alternative Quality Benchmark v2",
        "scores": asdict(scores),
        "signals": {
            "tests_passed": tests_passed,
            "tests_failed_or_error": tests_failed,
            "coverage_percent": round(coverage_percent, 2),
            "mypy_errors": mypy_errors,
            "ruff_issues": ruff_issues,
            **asdict(metrics),
        },
        "commands": [asdict(result) for result in command_results],
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(
        generate_markdown_report(
            scores=scores,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            coverage_percent=coverage_percent,
            mypy_errors=mypy_errors,
            ruff_issues=ruff_issues,
            metrics=metrics,
        ),
        encoding="utf-8",
    )

    print(f"Alternative Quality Benchmark score: {scores.total}/100 (grade {scores.grade})")
    print(f"JSON report: {report_json}")
    print(f"Markdown report: {report_md}")

    # Non-zero exit when foundational checks fail.
    if tests_failed > 0 or mypy_errors > 0 or ruff_issues > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
