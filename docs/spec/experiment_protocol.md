# Experiment Protocol

## Purpose

Define a minimum protocol for reproducible, comparable Bilancio experiments
across local and cloud execution.

## Pre-Run Checklist

1. Validate scenario config and parameter bounds.
2. Record code revision and benchmark status:
   - ABM benchmark status must be `PASS`
   - simulation efficiency benchmark status must be `PASS`
3. Fix run seeds and replicate policy.
4. Confirm output directories and artifact naming.

## Design Requirements

- Use paired-arm designs where possible (same seed per arm) for causal contrast.
- Declare sweep dimensions and fixed controls explicitly:
  - `kappa`, `concentration`, `mu`, `monotonicity`, `outside_mid_ratio`
- Set minimum replicate count before inferential reporting.
- Define stopping rules and exclusion policy before running.

## Execution Requirements

For each run, capture:
- run id
- scenario parameters
- seed
- run options (`max_days`, `quiet_days`, invariant mode)
- artifact paths and completion status

Execution environments:
- Local: `PYTHONPATH=src .venv/bin/python ...`
- Cloud: Modal runner with explicit cloud config snapshot

## Post-Run Validation

1. Confirm invariant checks and run status are clean.
2. Reject cells below minimum replicate threshold for significance reporting.
3. Report both raw and normalized loss metrics.
4. Include intermediary-loss components in system-loss accounting.

## Reporting Standard

Every experiment summary must include:
- denominator definition for percentage metrics
- replicate counts by cell
- effect-size and confidence interval fields
- explicit missing-data policy
- benchmark status at execution time

## Reproducibility Metadata

Store or log:
- random seed(s)
- executable command or runner config
- generated artifact index
- model version and key dependency versions

