# Alternative Quality Benchmark (AQB)

This benchmark is an alternative to the existing engineering benchmark and is intentionally more machine-checkable.

## Goal

Produce a reproducible quality score from command outputs and static code signals, with minimal subjective judgement.

## Score Model (100 points)

1. Verification Strength (35 points)
- Test execution quality (20): based on pytest pass rate.
- Coverage quality (15): scales linearly, with full points at 85% line coverage.

2. Static Correctness (25 points)
- mypy hygiene (15): full points at 0 errors; linear penalty per error.
- Ruff hygiene (10): full points at 0 issues; linear penalty per issue.

3. Maintainability Risk (20 points)
- Large files (7): ratio-based score from share of source files over 800 LOC.
- Long functions (7): ratio-based score from share of functions over 80 LOC.
- Broad exception usage (6): ratio-based score from share of broad handlers.

Maintainability thresholds (v2):
- Long file/function ratios score linearly, dropping to 0 at 18% incidence.
- Broad exception ratio drops to 0 at 5% incidence.
- This keeps scoring fair across different repository sizes.

4. API Type Clarity (10 points)
- Typed function coverage from AST analysis, with full points at 95%+ annotated functions.

5. Documentation Signal (10 points)
- Public docstring coverage across modules, public functions, and public classes, with full points at 80%+.

## Grades

- A: 90-100
- B: 80-89.99
- C: 70-79.99
- D: 60-69.99
- F: <60

## Reproducible Runner

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_alt_quality_benchmark.py
```

Outputs:

- JSON report: `temp/alt_quality_benchmark_report.json`
- Markdown report: `temp/alt_quality_benchmark_report.md`

The runner also stores command outputs for auditability and exits non-zero if tests, mypy, or ruff fail.
