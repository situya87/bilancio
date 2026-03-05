# Plan 050: Adaptive Decision Profiles (Implementation Status)

## Purpose

This document now records implementation status only.

All remaining work is tracked in a single active plan:

- `docs/plans/049_intermediary_loss_efficiency_plan.md`

The original full Plan 050 specification has been archived at:

- `docs/plans/archive/050_adaptive_decision_profiles_legacy.md`

## Status Snapshot (March 5, 2026)

### Active in code today

- Adaptive preset framework (`static`, `calibrated`, `responsive`, `full`) and sweep wiring.
- Trader within-run adaptation:
  - `adaptive_risk_aversion`
  - `adaptive_reserves`
- Risk/valuation adaptation:
  - `adaptive_ev_term_structure`
- VBT adaptation:
  - `adaptive_term_structure`
  - `adaptive_base_spreads`
  - `adaptive_convex_spreads`
- Lender within-run adaptation:
  - `adaptive_rates`
  - `adaptive_capital_conservation`
  - `adaptive_prevention`
- Banking/CB adaptation path in sweep runs:
  - `adaptive_corridor`
  - `adaptive_betas`
  - `adaptive_early_warning`

### Implemented as parameter overrides (functionally active, flag itself not directly consumed)

- `adaptive_planning_horizon` via `planning_horizon` override.
- `adaptive_lookback` via `lookback_window` override.
- `adaptive_issuer_specific` via `use_issuer_specific` override.
- Lender pre-run calibration flags via value overrides (`risk_aversion`, `max_loan_maturity`).

### Deferred or still incomplete

- `adaptive_per_bucket_tracking` (deferred).
- `adaptive_issuer_pricing` (deferred).
- `adaptive_profit_target` runtime behavior not implemented.
- Preset `stress_horizon` propagation incomplete in scenario path.
- `adaptive_stress_horizon` not enforced as an active behavioral switch.
- `scenario_informed_prior(kappa, mu, c)` exists but is not used.
- `adaptive_escalation` remains roadmap language, not active code.

## Execution Rule

- Do not add new pending checklists to this file.
- Add or update pending execution items only in:
  - `docs/plans/049_intermediary_loss_efficiency_plan.md`
