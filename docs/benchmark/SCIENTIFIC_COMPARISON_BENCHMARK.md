# Scientific Comparison Benchmark

This benchmark evaluates whether comparison experiments are run and analyzed
in a scientifically defensible way.

It complements existing ABM, robustness, and stylized-facts benchmarks by
focusing specifically on **comparison design + statistical output quality**.

## Runner

```bash
PYTHONPATH=src .venv/bin/python scripts/run_scientific_comparison_benchmark.py
```

Outputs:
- `temp/scientific_comparison_benchmark_report.json`
- `temp/scientific_comparison_benchmark_report.md`

## What It Checks

1. Design & Pairing Discipline (30)
- seed-matched paired control/treatment runs
- complete execution of the planned comparison grid
- balanced replication across parameter cells
- no duplicate cell/seed rows

2. Inference Completeness (30)
- per-cell treatment effects are analyzable
- each row has valid CI, p-value, and Cohen's d
- replicate counts meet the configured floor
- uncertainty intervals are properly reported

3. Multiplicity & Robustness (20)
- Benjamini-Hochberg FDR correction is computed and valid
- FDR-significant count never exceeds raw-significant count
- effect-direction coherence across cells is measured

4. Output Schema & Reproducibility (20)
- analysis artifacts are emitted (`stats_effects.csv`, `stats_cells.csv`,
  `stats_summary.json`, `stats_sensitivity.json`)
- artifact schemas include required scientific fields
- re-running analysis with the same seed produces identical outputs

## Critical Gates

A run is `PASS` only if:
- score meets `--target-score` (default `90`)
- all gates pass

Current gates:
- `scientific::paired_runs_complete`
- `scientific::replication_floor`
- `scientific::effect_rows_valid`
- `scientific::fdr_report_valid`
- `scientific::analysis_artifacts_schema`
- `scientific::deterministic_reanalysis`

## Why This Is Separate From ABM Benchmark

The ABM benchmark already checks reproducibility/invariants/statistical
tooling availability, but it does not directly enforce that comparison
outputs themselves satisfy end-to-end scientific reporting standards
(paired design quality, multiplicity treatment, and schema-complete
analysis artifacts).
