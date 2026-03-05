# Plan 049: Intermediary Loss Efficiency and Behavioral Policy Redesign

## Purpose

Define a scientific, implementation-ready plan to reduce intermediary loss transfer while preserving (or improving) default prevention and total system outcomes.

This plan focuses on the observed pattern where some treatments reduce trader defaults by shifting losses to intermediaries, and sets a structured path to identify when this is necessary versus avoidable.

## Baseline Snapshot (March 5, 2026)

From latest sweep recomputations using:

- `default_relief = delta_control - delta_treatment`
- `inst_loss_shift = intermediary_loss_treatment - intermediary_loss_control`
- `net_system_relief = system_loss_control - system_loss_treatment`

Observed:

- Bank (three-way, `idle -> lend`): default relief in `94.4%`, but `0.0%` of those help runs avoid extra intermediary loss.
- Bank (double/full sweeps): some help-without-extra-loss cases exist (`~10-15%` of all runs), but minority.
- Dealer (three-way, `passive -> active`): default relief in `40.5%`; `84.3%` of help runs occur without extra intermediary loss.
- NBFI (three-way, `idle -> lend`): mixed regime; `60.0%` of help runs avoid extra intermediary loss, usually with smaller relief.

Interpretation baseline:

- Some intermediary loss is structurally expected in mutually indebted networks.
- Current bank behavior likely leaves avoidable loss transfer on the table.

## Status Update After PR143 Review (March 5, 2026)

Historical PR143 review findings were used to reprioritize this plan.

Since that review, part of the adaptive wiring has been improved in code. However, not all adaptive roadmap items are complete, and end-to-end preset guarantees are still insufficient for scientific conclusions.

Implication:

- We should not treat current adaptive sweep outcomes as fully settled evidence yet.
- The first phase remains wiring integrity + benchmark integrity before new behavioral policy tuning.

## Consolidated Backlog From Plan 050

This document is now the single active execution plan. Remaining adaptive-profile items from Plan 050 are tracked here.

Pending items to close before policy conclusions:

- [x] Implement or explicitly de-scope `adaptive_per_bucket_tracking` — **de-scoped**: removed from profiles, config models, and preset builder (no engine consumer).
- [x] Implement or explicitly de-scope `adaptive_issuer_pricing` — **de-scoped**: removed from profiles, config models, and preset builder (no engine consumer).
- [x] Implement runtime behavior for `adaptive_profit_target` or remove — **removed**: flag was set but never consumed by any engine; removed from LenderProfile, config, and preset builder.
- [x] Propagate and consume `stress_horizon` from adaptive presets — **fixed**: added `stress_horizon` field to `BalancedDealerConfig` and wired through both `apply.py` VBTProfile construction paths.
- [x] Implement or explicitly de-scope `adaptive_stress_horizon` as a behavioral switch — **de-scoped**: boolean flag removed (was dead code). The `stress_horizon` int value IS consumed by `dealer_sync.py` and is now properly wired.
- [x] Either integrate `scenario_informed_prior(kappa, mu, c)` into active prior construction or formally retire it — **deprecated**: function kept (tested, correct) but marked deprecated; only `kappa_informed_prior` used in production.
- [x] Implement or remove `adaptive_escalation` from active roadmap language — **confirmed absent**: never existed in code, only in docs. No action needed.
- [x] Add end-to-end regression tests for `--adapt` presets at run level — **implemented**: `tests/integration/test_adaptive_preset_e2e.py` (29 tests) verifies all 4 presets through full pipeline with regression guards.

## Scope

In scope:

- behavioral policy redesign for banks/dealers/NBFI
- scientific benchmark and decision frontier analysis
- guardrails that preserve simulation integrity
- experimental validation across seeds and stress regimes

Out of scope:

- changing accounting identities or default mechanics
- tuning by hand to one seed or one scenario only

## Scientific Benchmark (Primary Evaluation Standard)

Every policy variant must be evaluated on a 3-metric frontier:

1. Default relief (`default_relief`, higher is better)
2. Intermediary loss shift (`inst_loss_shift`, lower is better; negative preferred)
3. Net system relief (`net_system_relief`, higher is better)

A variant is considered scientifically better only if it is Pareto-improving on this frontier for a meaningful share of runs (not one-off cases).

Secondary diagnostics:

- share of runs with `default_relief > 0 and inst_loss_shift <= 0`
- share of runs with `default_relief > 0 and net_system_relief > 0`
- intermediary loss/capital ratio
- tail-risk metrics (p95/p99 intermediary drawdown)

Metric definition guardrail:

- `net_system_relief` must be computed from **total system loss** in treatment vs control (including trader/household/firm losses plus intermediary losses), so we do not mistake “extra institutional loss” for overall welfare change.

## Workstreams

### W0. Adaptive Wiring and Benchmark Integrity (Prerequisite) — COMPLETE

- [x] Ensure adaptive overrides survive scenario validation — fixed `stress_horizon` pipeline gap (was silently dropped by Pydantic).
- [x] Merge all override buckets — all 6 buckets now wire through; `stress_horizon` added to `BalancedDealerConfig`.
- [x] Resolve declared-but-unused adaptive flags — removed 4 dead flags (`adaptive_profit_target`, `adaptive_stress_horizon`, `adaptive_per_bucket_tracking`, `adaptive_issuer_pricing`); deprecated `scenario_informed_prior`.
- [x] Add end-to-end tests — `tests/integration/test_adaptive_preset_e2e.py` (29 tests covering all 4 presets through full pipeline).
- [x] Add a regression check — `TestRegressionGuard` verifies all override keys are accepted by Pydantic models and have matching profile fields.

Acceptance:

- `--adapt` presets produce observable and test-verified runtime differences where expected. ✓
- Unsupported adaptive options fail loudly or are absent from public preset output (no silent no-op behavior). ✓

### W1. Frontier Instrumentation and Reporting

- [ ] Add an analysis utility that computes the three primary metrics and Pareto labels per run.
- [ ] Add regime-level summaries (bank, dealer, NBFI, combined) and stress-slice summaries by `kappa`, concentration, and seed.
- [ ] Export machine-readable outputs for reproducible comparisons.
- [ ] Recompute baseline/treatment comparisons only after W0 is complete and verified.

Acceptance:

- For any sweep CSV, we can reproduce frontier stats with one command.
- Reports clearly separate loss-shifting improvements from genuine system improvements.

### W2. Behavioral Levers (Bank/NBFI/Dealer)

- [ ] Implement a marginal-benefit trigger: intervene only when expected default relief per expected intermediary loss exceeds threshold.
- [ ] Tighten state-contingent risk pricing and terms: steeper risk premium, stricter limits at high estimated default probability, and stress-aware maturity control.
- [ ] Add capital-preservation guardrails: per-day/per-run risk budget and stop-loss style caps.
- [ ] Add claim-protection options where model-consistent (e.g., haircut/collateralized terms) without violating accounting.

Acceptance:

- At least one policy variant increases the share of `help + no extra intermediary loss` in bank regimes.
- No regression in accounting invariants or deterministic reproducibility.

### W3. Structural-Loss Floor Identification

- [ ] Estimate empirical loss floor: minimum intermediary loss shift needed to achieve target default relief bands.
- [ ] Separate unavoidable-loss regions (structural) from avoidable-loss regions (policy design).
- [ ] Document where intervention is not efficient and should rationally be withheld.

Acceptance:

- For each regime, publish a table of relief bands vs minimum observed intermediary cost.
- Decision guidance exists for “do not intervene” zones.

### W4. Validation and Robustness

- [ ] Run evaluation on held-out seeds and parameter regions not used for tuning.
- [ ] Perform sensitivity checks across `kappa`, concentration, `mu`, and monotonicity.
- [ ] Add regression tests for benchmark outputs and metric definitions.

Acceptance:

- Candidate policy effect persists on held-out data.
- No dependence on one narrow parameter slice.

## Simulation-Quality Preservation Rules

Any implementation in this plan must satisfy all of the following:

1. Accounting consistency:
   - No hidden subsidy or implicit money creation.
   - Cash/reserve/claim conservation remains valid under existing invariants.
2. Causal consistency:
   - Decision rules use only information available at decision time.
   - No look-ahead leakage from future events.
3. Behavioral coherence:
   - Intermediaries do not accept systematically negative expected value unless explicitly configured as policy support with bounded budget.
4. Reproducibility:
   - Same seed + same config produces same outputs.
5. Stress robustness:
   - Improvements are not limited to benign states; tails must be checked.

## Experiment Sequence

1. Adaptive wiring + end-to-end preset verification (W0).
2. Baseline freeze and metric verification using total-loss-consistent definitions.
3. Single-lever ablations (one policy change at a time).
4. Combined-lever tests on top-performing single levers.
5. Held-out validation and stress-tail review.
6. Promote only Pareto-improving variants, with explicit “minimum necessary intermediary loss” reporting.

## Deliverables

By completion:

- Updated behavioral policy configs and implementation notes.
- A reproducible frontier analysis artifact for each major sweep.
- A short decision memo: where intermediary loss is unavoidable vs avoidable.
- Regression tests protecting metric correctness and policy behavior.
