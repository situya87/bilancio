# ABM Modeling Benchmark

This benchmark evaluates Bilancio against practical ABM modeling standards, not just software engineering quality.

It complements:
- `docs/benchmark/ENGINEERING_BENCHMARK.md`
- `docs/benchmark/ALTERNATIVE_QUALITY_BENCHMARK.md`

## Goal

Create a repeatable score that tracks progress toward stronger ABM practice:
- reproducibility
- formal verification
- statistical rigor in experiments
- behavioral validation
- model transparency/documentation

## Scoring Model (100 points)

1. Reproducibility & Stochastic Discipline (20)
- Test suite pass rate for seed/reproducibility checks (12)
- Static seed plumbing signals across experiment runners and stats APIs (8)

2. Verification & Accounting Invariants (20)
- Invariant/property regression suite pass rate (14)
- Static verification signals (`core/invariants.py`, invariant checks in tests/runtime) (6)

3. Experiment Design & Statistical Inference (20)
- Statistical test suite pass rate (12)
- Static checks for bootstrap/effect-size/significance/sensitivity tooling (8)

4. Validation & Behavioral Plausibility (20)
- Behavioral regression suite pass rate (12)
- Static validation signals (regression/integration coverage, modeling docs, stylized-fact signal) (8)

5. Transparency & Model Documentation (20)
- Public docstring coverage for engines/experiments/scenarios/stats (8)
- Scenario example coverage (`examples/scenarios/*.yaml`) (4)
- Core protocol docs presence (4)
- ABM standards artifacts (ODD/calibration/validation protocol docs) (4)

## Critical Gates (Blocking)

This benchmark now enforces both:
- a numeric score (`0-100`)
- a hard `PASS`/`FAIL` status

A run is `PASS` only if:
- total score meets `--target-score`
- all critical gates pass

Critical gates currently include:
- Reproducibility suite fully green
- Verification/invariant suite fully green
- Statistical inference suite fully green
- Behavioral core and comparative-statics suites fully green
- Minimum behavioral depth signal (regression and stylized-fact coverage)
- Documentation floor:
  - public docstring coverage minimum
  - minimum ABM standards artifact count in `docs/spec/`
  - mandatory files:
    - `docs/spec/odd_protocol.md`
    - `docs/spec/validation_matrix.md`
    - `docs/spec/experiment_protocol.md`

When critical gates fail:
- runner exits non-zero
- grade is capped downward
- report includes `critical_failures` with exact causes

## Grade Bands

- `A+`: >=95
- `A`: >=90
- `B`: >=80
- `C`: >=70
- `D`: >=60
- `F`: <60

## Target Ladder

- Minimum lab-grade target: `80`
- Strong internal research target: `90`
- Publication-ready target: `95`

Recommended team goal for this repo: **90** first, then push to **95**.

For release gating, rely on `status == PASS`, not score alone.

## Runner

```bash
PYTHONPATH=src .venv/bin/python scripts/run_abm_modeling_benchmark.py
```

Outputs:
- `temp/abm_modeling_benchmark_report.json`
- `temp/abm_modeling_benchmark_report.md`

## Improvement Benchmark To Reach

Use this progression:

1. Phase 1 (Reach 90)
- Keep all ABM benchmark suites green.
- Raise weakest category above 80%.
- Add at least 3 ABM standards artifacts under `docs/spec/`:
  - `odd_protocol.md`
  - `validation_matrix.md`
  - `experiment_protocol.md`

2. Phase 2 (Reach 95)
- Raise transparency category >=90%.
- Add explicit stylized-fact tests (named with `stylized`/`validation` intent).
- Expand behavioral validation suites across banking/dealer/lender comparison arms.

3. Sustainment
- Run ABM benchmark in CI on PRs touching `engines/`, `experiments/`, or `stats/`.
- Block merges when benchmark `status` is `FAIL`.
- Alert when score drops by >2 points from main baseline, even when status remains `PASS`.
