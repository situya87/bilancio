# Extended Benchmark Suite

This document defines the additional benchmark families added on top of the existing quality benchmarks.

Each benchmark emits:
- JSON report (`temp/...json`)
- Markdown report (`temp/...md`)
- exit code `0` only when both the target score and all critical gates pass

## Benchmarks

### 1) Metamorphic Behavior Benchmark
- Script: `scripts/run_metamorphic_behavior_benchmark.py`
- Focus: invariance under nominal scaling, agent relabeling, and action reordering.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_metamorphic_behavior_benchmark.py
```
- Outputs:
  - `temp/metamorphic_behavior_benchmark_report.json`
  - `temp/metamorphic_behavior_benchmark_report.md`

### 2) Long-Horizon Drift Benchmark
- Script: `scripts/run_long_horizon_drift_benchmark.py`
- Focus: slow drift/leaks across long runs (default 300 days), with daily invariant checks.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_long_horizon_drift_benchmark.py
```
- Outputs:
  - `temp/long_horizon_drift_benchmark_report.json`
  - `temp/long_horizon_drift_benchmark_report.md`

### 3) Stochastic Robustness Benchmark
- Script: `scripts/run_stochastic_robustness_benchmark.py`
- Focus: seed-to-seed distribution stability and monotonic effect consistency.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_stochastic_robustness_benchmark.py
```
- Outputs:
  - `temp/stochastic_robustness_benchmark_report.json`
  - `temp/stochastic_robustness_benchmark_report.md`

### 4) Calibration / Stylized-Facts Benchmark
- Script: `scripts/run_stylized_facts_benchmark.py`
- Focus: required stylized facts (liquidity monotonicity, NBFI activation, banking differentiation, invariants).
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_stylized_facts_benchmark.py
```
- Outputs:
  - `temp/stylized_facts_benchmark_report.json`
  - `temp/stylized_facts_benchmark_report.md`

### 5) Scenario Plugin Contract Benchmark
- Script: `scripts/run_plugin_contract_benchmark.py`
- Focus: plugin protocol/schema/default/error semantics with valid/invalid fixture corpus.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_plugin_contract_benchmark.py
```
- Outputs:
  - `temp/plugin_contract_benchmark_report.json`
  - `temp/plugin_contract_benchmark_report.md`

### 6) Compile-to-Apply Equivalence Benchmark
- Script: `scripts/run_compile_apply_equivalence_benchmark.py`
- Focus: compile/apply preservation of counts, debt totals, and maturity distribution.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_compile_apply_equivalence_benchmark.py
```
- Outputs:
  - `temp/compile_apply_equivalence_benchmark_report.json`
  - `temp/compile_apply_equivalence_benchmark_report.md`

### 7) Local-vs-Cloud Parity Benchmark
- Script: `scripts/run_local_cloud_parity_benchmark.py`
- Focus: local run artifacts and metric parity vs cloud metrics computation path.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_cloud_parity_benchmark.py
```
- Outputs:
  - `temp/local_cloud_parity_benchmark_report.json`
  - `temp/local_cloud_parity_benchmark_report.md`

### 8) Failure-Injection Integration Benchmark
- Script: `scripts/run_failure_injection_integration_benchmark.py`
- Focus: invalid config handling, mixed status accounting, malformed artifact rejection, retry behavior.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_failure_injection_integration_benchmark.py
```
- Outputs:
  - `temp/failure_injection_integration_benchmark_report.json`
  - `temp/failure_injection_integration_benchmark_report.md`

## Optional knobs

Most scripts accept:
- `--target-score`
- `--out-json`
- `--out-md`

Additional script-specific knobs include:
- `run_long_horizon_drift_benchmark.py`: `--days`, `--window`
- `run_stochastic_robustness_benchmark.py`: `--seeds`
- `run_local_cloud_parity_benchmark.py`: `--tolerance`

## Run All

Run the full extended suite with one command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_extended_benchmark_suite.py
```

Suite outputs:
- `temp/extended_benchmark_suite_report.json`
- `temp/extended_benchmark_suite_report.md`

## GitHub Action

Manual workflow is available at `.github/workflows/extended-benchmarks.yml`:
- Trigger via **Actions -> Extended Benchmarks -> Run workflow**.
- Runs each benchmark in a matrix and uploads JSON/Markdown reports as artifacts.

## Suggested CI rollout

1. Start as non-blocking for 1 week to collect baseline reports.
2. Promote to blocking with explicit required checks in branch protection.
3. Keep critical gates strict; tune only score thresholds if needed.
