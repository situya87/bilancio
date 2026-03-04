# Operational Performance Budgets

This document defines runtime and memory thresholds for benchmark
suites to ensure CI remains fast and resource-efficient.

## Budget Thresholds

| Benchmark | Wall Time Budget | Memory Budget | Notes |
|-----------|-----------------|--------------|-------|
| Metamorphic Behavior | 300s (5 min) | 2 GB | Runs ~20 scenarios |
| Long-Horizon Drift | 300s (5 min) | 2 GB | Single long simulation |
| Stochastic Robustness | 300s (5 min) | 2 GB | Multiple seeded runs |
| Scientific Comparison | 600s (10 min) | 2 GB | Full grid with replicates |
| Regression | 120s (2 min) | 1 GB | 3 small tiers |
| Failure-Mode | 120s (2 min) | 1 GB | 3 extreme scenarios |
| Plugin Contract | 60s (1 min) | 512 MB | Schema validation only |
| Compile-Apply Equiv | 120s (2 min) | 1 GB | Paired equivalence |
| Local-Cloud Parity | 120s (2 min) | 1 GB | Mock cloud comparison |
| Failure Injection | 120s (2 min) | 1 GB | Error path testing |

## Monitoring

Each benchmark reports wall time in its JSON output under `elapsed_seconds`.
The `check_operational_budget()` helper in `benchmark_utils.py` can be
used to validate budgets programmatically.

## Cost Budgets (Cloud)

For Modal cloud execution:

| Operation | Cost Budget | Notes |
|-----------|------------|-------|
| Single simulation | $0.001 | ~30s CPU |
| Small sweep (10 runs) | $0.01 | |
| Medium sweep (50 runs) | $0.05 | |
| Large sweep (250 runs) | $0.25 | |

## When Budgets Are Exceeded

If a benchmark exceeds its budget:

1. Check for regression in simulation performance
2. Review recent changes to the simulation engine
3. Consider if the benchmark parameters need adjustment
4. File an issue if the regression is real

## CI Timeout

The GitHub Actions workflow has a 60-minute timeout per benchmark job.
Individual benchmarks should complete well within this limit.
