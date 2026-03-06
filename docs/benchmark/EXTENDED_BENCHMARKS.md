# Extended Benchmark Suite

This document defines the heavier benchmark families that sit above everyday PR CI.

Use them for scheduled health checks, major engine/cloud changes, and release readiness. They are
valuable, but they are intentionally not the default merge gate for ordinary pull requests.

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

### 4) Scientific Comparison Benchmark
- Script: `scripts/run_scientific_comparison_benchmark.py`
- Focus: scientific quality of comparison experiments and analysis outputs:
  - paired seed-matched control/treatment design
  - replication discipline per parameter cell
  - inferential completeness (CIs, p-values, effect sizes)
  - multiplicity handling (Benjamini-Hochberg FDR)
  - reproducible, schema-complete analysis artifacts
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_scientific_comparison_benchmark.py
```
- Outputs:
  - `temp/scientific_comparison_benchmark_report.json`
  - `temp/scientific_comparison_benchmark_report.md`

### 5) Calibration / Stylized-Facts Benchmark
- Script: `scripts/run_stylized_facts_benchmark.py`
- Focus: required stylized facts (liquidity monotonicity, NBFI activation, banking differentiation, invariants).
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_stylized_facts_benchmark.py
```
- Outputs:
  - `temp/stylized_facts_benchmark_report.json`
  - `temp/stylized_facts_benchmark_report.md`

### 6) Scenario Plugin Contract Benchmark
- Script: `scripts/run_plugin_contract_benchmark.py`
- Focus: plugin protocol/schema/default/error semantics with valid/invalid fixture corpus.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_plugin_contract_benchmark.py
```
- Outputs:
  - `temp/plugin_contract_benchmark_report.json`
  - `temp/plugin_contract_benchmark_report.md`

### 7) Compile-to-Apply Equivalence Benchmark
- Script: `scripts/run_compile_apply_equivalence_benchmark.py`
- Focus: compile/apply preservation of counts, debt totals, and maturity distribution.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_compile_apply_equivalence_benchmark.py
```
- Outputs:
  - `temp/compile_apply_equivalence_benchmark_report.json`
  - `temp/compile_apply_equivalence_benchmark_report.md`

### 8) Local-vs-Cloud Parity Benchmark
- Script: `scripts/run_local_cloud_parity_benchmark.py`
- Focus: local run artifacts and metric parity vs cloud metrics computation path.
- Run:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_cloud_parity_benchmark.py
```
- Outputs:
  - `temp/local_cloud_parity_benchmark_report.json`
  - `temp/local_cloud_parity_benchmark_report.md`

### 9) Failure-Injection Integration Benchmark
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
- Trigger via **Actions -> Extended Benchmarks -> Run workflow** for release / major-feature gates.
- A scheduled nightly run also keeps baseline health visible without blocking every PR.
- The workflow can also be called from other workflows when a release process needs a strict gate.

## Suggested CI rollout

1. Keep PR CI focused on linting, typing, and the normal non-benchmark test suite.
2. Run this suite nightly (or other scheduled cadence) as a non-blocking health audit.
3. Run it again before releases or major engine/cloud changes as an explicit gate.
