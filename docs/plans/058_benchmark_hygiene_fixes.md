# Plan 058 — Benchmark Hygiene Fixes

**Branch**: `plan/058-benchmark-hygiene-fixes`
**Date**: 2026-03-10
**Status**: Draft

## Goal

Bring the test suite to 0 failures and clear all critical-gate failures across the 17-benchmark suite. This is a housekeeping plan — no new features, no behavioral changes.

## Current State (baseline, 2026-03-10)

| Metric | Value |
|---|---|
| Tests passing | 6,337 / 6,346 (99.86%) |
| Failures | 7 |
| Coverage | 84.31% (threshold 75%) |
| Benchmarks passing | 15 / 17 |
| Benchmarks failing | Regression (C), Simulation Efficiency (C) |

## Scope

Four independent fixes, ordered by priority:

---

### Fix 1: Delete duplicate `(1)` files

**Problem**: macOS-created duplicate files (e.g. `test_performance_parity (1).py`) are picked up by pytest, causing 4 failures from stale/broken copies. Additional duplicates clutter the repo.

**Files to delete** (all untracked):
```
tests/cloud/test_sweep_trigger_unit (1).py
tests/regression/test_performance_parity (1).py
tests/core/test_performance_flags (1).py
tests/experiments/test_deprecation_warnings (1).py
tests/jobs/test_supabase_store (1).py
src/bilancio/scenarios/ring_explorer (1).py
src/bilancio/ui/cli/sweep (1).py
docs/analysis/bank_idle_vs_bank_lend_specification (1).md
docs/plans/052_integration_discipline_hardening_plan (1).md
.coverage (1)
.coverage (2)
.coverage (3)
```

**Also clean up** cached bytecode from duplicates:
```
tests/unit/__pycache__/test_buy_controls (1).cpython-312-pytest-8.4.1.pyc
tests/core/__pycache__/test_performance_flags (1).cpython-312-pytest-8.4.1.pyc
tests/experiments/__pycache__/test_deprecation_warnings (1).cpython-312-pytest-8.4.1.pyc
tests/cloud/__pycache__/test_sweep_trigger_unit (1).cpython-312-pytest-8.4.1.pyc
tests/jobs/__pycache__/test_supabase_store (1).cpython-312-pytest-8.4.1.pyc
tests/regression/__pycache__/test_performance_parity (1).cpython-312-pytest-8.4.1.pyc
```

**Preventive measure**: Add a `.gitignore` pattern to block future `(1)`, `(2)`, `(3)` duplicates:
```gitignore
# macOS duplicate files
*\ (1)*
*\ (2)*
*\ (3)*
```

**Acceptance criteria**:
- [ ] All listed files deleted
- [ ] `find . -name "*\ (1)*" -o -name "*\ (2)*" -o -name "*\ (3)*"` returns 0 results (excluding `.mypy_cache`)
- [ ] `.gitignore` updated with pattern
- [ ] 4 test failures eliminated

---

### Fix 2: Update `VIZ_MENU` test expectations

**Problem**: Commit `5ddc7bf1` added a `"notebook"` entry to `VIZ_MENU` in `sweep_setup.py` but didn't update the 3 tests that hardcode the menu size.

**File**: `tests/ui/test_sweep_setup.py`

**Changes**:

1. `test_viz_menu_has_expected_entries` (~line 477): Add `"notebook"` to the expected set of VIZ_MENU keys.

2. `test_dealer_gets_all_visualizations` (~line 525): Change `assert len(available) == 7` to `assert len(available) == 8`.

3. `test_legacy_available_analyses_combines` (~line 554): Change `assert len(available) == 17` to `assert len(available) == 18` and update comment from `# 10 data + 7 viz` to `# 10 data + 8 viz`.

**Acceptance criteria**:
- [ ] All 3 tests pass
- [ ] No other test_sweep_setup tests break
- [ ] Total test failures reduced from 7 to 0

---

### Fix 3: Investigate and resolve regression fingerprint drift

**Problem**: The Regression Benchmark fails its `regression::no_metric_drift` critical gate. Two `max_day` values have drifted:

| Tier | Fingerprint | Actual | Delta |
|---|---|---|---|
| small | max_day=4 | max_day=5 | +1 |
| medium | max_day=10 | max_day=13 | +3 |

All other metrics (default_ratio, defaults_count, total_loss_ratio) are **unchanged**, meaning the simulation reaches the same economic outcomes but takes longer to stabilize.

**Investigation steps**:
1. Read `scripts/run_regression_benchmark.py` to understand how tiers are configured and how fingerprints are compared.
2. Check `git log --oneline -20` for recent changes that could affect settlement timing (Plan 055 topology changes, VBT pricing, settlement ordering).
3. Run the `small` tier scenario standalone and trace which day the last settlement occurs vs the fingerprint expectation.
4. Determine: is the new timing **correct** (i.e., a deliberate behavioral change from a recent feature) or a **bug**?

**Resolution** (one of):
- **A) Update fingerprints**: If the timing change is an expected consequence of a recent feature (e.g., Plan 055 shared seed, VBT initial pricing), regenerate `scripts/regression_fingerprints.json` with current values. Add a comment in the commit explaining which change caused the drift.
- **B) Fix the regression**: If the timing change is unintentional, identify and fix the root cause.

**Acceptance criteria**:
- [ ] Root cause identified and documented in commit message
- [ ] `regression::no_metric_drift` critical gate passes
- [ ] Regression Benchmark grade rises from C to A
- [ ] No other benchmark scores degrade

---

### Fix 4: Align `max_parallel` cloud config with memory estimate

**Problem**: The Simulation Efficiency Benchmark fails `config::parallel_alignment`. The benchmark projects worst-case peak memory of ~250 MB per worker, recommending `max_parallel=5`. The cloud config specifies `max_parallel=10`. The gate requires `recommended / configured >= 0.70` — current ratio is 0.50.

**Investigation steps**:
1. Read the efficiency benchmark to understand how `recommended_max_parallel` is computed (memory model).
2. Check where `max_parallel=10` is configured (likely `cloud_executor.py` or `modal_app.py`).
3. Determine if the memory estimate is conservative (it's a proxy, not real measurement) or if 10 is genuinely risky.

**Resolution** (one of):
- **A) Lower `max_parallel` to 7**: Satisfies 5/7 >= 0.70. Conservative and safe.
- **B) Lower `max_parallel` to 5**: Matches recommendation exactly. Slower sweeps but safest.
- **C) Tune the benchmark's memory model**: If the 250 MB projection is overly conservative (actual peaks are much lower), adjust the scaling factor.

**Acceptance criteria**:
- [ ] `config::parallel_alignment` critical gate passes
- [ ] Simulation Efficiency Benchmark grade rises from C to A
- [ ] Cloud sweep functionality is not degraded (verify with a dry-run or check Modal memory dashboards)

---

## Sweep Surface

This plan touches **no simulation logic**, so no sweep is needed. Verification is purely via the test suite and benchmark scripts.

## Verification Plan

After all 4 fixes:

```bash
# 1. Full test suite — expect 0 failures
uv run pytest tests/ -v

# 2. Regression benchmark — expect grade A
uv run python scripts/run_regression_benchmark.py

# 3. Simulation efficiency benchmark — expect grade A
uv run python scripts/run_simulation_efficiency_benchmark.py

# 4. Extended benchmark suite — expect all PASS
uv run python scripts/run_extended_benchmark_suite.py

# 5. ABM benchmark — expect no degradation from 95.88
uv run python scripts/run_abm_modeling_benchmark.py
```

**Target end state**:
- 0 test failures (currently 7)
- 17/17 benchmarks passing (currently 15/17)
- No benchmark score regression
- Coverage >= 84% (unchanged)

## Risks

- **Fix 3 (fingerprints)**: If the `max_day` drift is a genuine bug rather than an expected side-effect, the fix could be more involved than a fingerprint update. Capped at 1 hour of investigation before escalating.
- **Fix 4 (max_parallel)**: Lowering parallelism will slow cloud sweeps proportionally. A reduction from 10 to 7 adds ~43% more wall time to large sweeps.
