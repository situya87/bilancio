# Simulation Efficiency Benchmark (Modal Safety)

This benchmark tracks whether simulation performance stays safely inside cloud execution limits.

It is specifically aimed at preventing Modal sweep breakage from:
- run-level timeout overruns
- run-level memory pressure and OOM risk
- oversized artifacts that increase I/O and transfer cost
- drift between cloud defaults and Modal function resource limits

## Runner

```bash
PYTHONPATH=src .venv/bin/python scripts/run_simulation_efficiency_benchmark.py
```

Outputs:
- `temp/simulation_efficiency_benchmark_report.json`
- `temp/simulation_efficiency_benchmark_report.md`

## Score Model (100 points)

1. Runtime & Throughput (40)
- Workload runtime budgets for representative scenarios/rings (25)
- Existing performance benchmark guard tests (`tests/benchmark/test_performance.py`) (15)

2. Memory Headroom (30)
- Per-workload projected memory vs budgets (20)
- Hard headroom against Modal `run_simulation` memory limit (10)

3. Artifact Footprint (15)
- Output artifact size budgets (10)
- Event volume growth budget (5)

4. Modal Config Safety (15)
- `CloudConfig.memory_mb` matches Modal `run_simulation` memory (8)
- `CloudConfig.timeout_seconds` does not exceed Modal timeout (5)
- Aggregate timeout shorter than run timeout (2)

## Critical Gates (Blocking)

This benchmark now uses both score and hard gates.

A run is `PASS` only if:
- total score meets `--target-score`
- all critical gates pass

Critical gates currently include:
- Throughput guard tests fully green (`tests/benchmark/test_performance.py`)
- All benchmark workloads run successfully
- Every workload stays within runtime and artifact budgets
- Minimum timeout headroom vs Modal timeout
- Minimum memory headroom vs Modal memory limit
- Minimum `recommended_max_parallel / cloud.max_parallel` alignment
- Cloud memory and timeout settings remain consistent with Modal limits

When a critical gate fails:
- runner exits non-zero
- grade is capped downward
- report includes `critical_failures` and safety ratios

## Default Target

- Team target: **>=85**
- Production comfort target: **>=90**

## Operational Use

Run this benchmark:
- before large sweeps
- after changes in `engines/`, `experiments/`, `cloud/`, or `ui/run.py`
- before increasing `max_parallel`

Use the report field `recommended_max_parallel` as a safe upper bound for memory-constrained runs.

Treat benchmark `status` as the merge gate for cloud/sweep-related PRs.

## Notes

- Memory is measured with `tracemalloc` and scaled by a configurable multiplier (`--memory-multiplier`, default `8.0`) to approximate total process RSS.
- This is a conservative proxy, not an exact container RSS measurement.
- Keep the multiplier stable across runs if you want comparable trend lines.
- Critical gate thresholds are configurable:
  - `--critical-memory-headroom`
  - `--critical-timeout-headroom`
  - `--critical-parallel-alignment`
