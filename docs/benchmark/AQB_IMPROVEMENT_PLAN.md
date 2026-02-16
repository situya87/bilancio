# AQB Improvement Plan (Path to 90)

This plan is based on the latest Alternative Quality Benchmark (AQB) run.

## Current Score

- AQB score: **84.65 / 100** (Grade B)
- Tests: **2766 passed, 0 failed**
- Coverage: **77.36%**
- mypy errors: **0**
- ruff issues: **0**
- Maintainability risk: **6.0 / 20**
- Broad exception handlers: **0**
- Files over 800 LOC: **16**
- Functions over 80 LOC: **106**

## Gap to 90

We need **+5.35 points**.

Given AQB scoring weights, this gap is now almost entirely maintainability-driven.

## What Will Move the Score

1. Reduce files over 800 LOC from 16 to <=4.
- Estimated gain: up to **+4.0** (file-size component)

2. Reduce functions over 80 LOC from 106 to <=35.
- Estimated gain: up to **+7.0** (function-size component)

3. Raise coverage from 77.37% to >=85%.
- Estimated gain: up to **+1.35**

A combination of file/function decomposition plus higher coverage is required to pass 90.

## Priority Backlog

1. P0: Split oversized modules
- Target first:
  - `src/bilancio/ui/html_export.py`
  - `src/bilancio/ui/cli/sweep.py`
  - `src/bilancio/specification/current_system.py`
  - `src/bilancio/dealer/bank_integration.py`
  - `src/bilancio/scenarios/ring_explorer.py`
- Goal: move rendering/spec helper blocks into dedicated modules to get each file under 800 LOC.

2. P1: Continue file decomposition for largest modules
- `src/bilancio/dealer/simulation.py`
- `src/bilancio/dealer/bank_dealer_simulation.py`
- `src/bilancio/dealer/metrics.py`
- `src/bilancio/experiments/balanced_comparison.py`
- `src/bilancio/experiments/ring.py`

3. P1: Coverage increase
- Add focused tests for low-coverage modules:
  - `src/bilancio/cloud/modal_app.py`
  - `src/bilancio/jobs/supabase_store.py`
  - `src/bilancio/storage/supabase_registry.py`
  - `src/bilancio/ui/cli/sweep.py`

## Verification Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m mypy src/bilancio
uv run --extra dev ruff check src tests
PYTHONPATH=src .venv/bin/python scripts/run_alt_quality_benchmark.py
```

## Exit Criteria

- AQB score >= 90
- mypy: 0 errors
- ruff: 0 issues
- files >800 LOC: <=4
- functions >80 LOC: <=35
- coverage >=85%
