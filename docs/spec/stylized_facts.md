# Stylized Facts and Testable Expectations

This document captures model-level expectations that should remain stable
across refactors unless intentionally changed.

## SF-1: Liquidity Stress Increases Default Risk

- Statement: holding topology and obligation size fixed, lower kappa should
  weakly increase defaults.
- Why: lower initial liquidity reduces settlement feasibility and amplifies
  rollover pressure.
- Test anchor:
  - `tests/regression/test_simulation_properties.py::test_defaults_increase_with_lower_kappa`
  - `tests/regression/test_simulation_properties.py::test_stylized_fact_liquidity_shortfall_produces_defaults`

## SF-2: Credit Supply Activates Under Shortfalls

- Statement: with a funded NBFI and borrowers in shortfall, loan issuance
  should be non-zero under permissive risk limits.
- Why: the lender policy is intended to bridge temporary liquidity gaps.
- Test anchor:
  - `tests/regression/test_simulation_properties.py::test_lending_effect_is_nonzero`
  - `tests/regression/test_simulation_properties.py::test_stylized_fact_nbfi_credit_supply_emits_loans`

## SF-3: Banking Regime Changes Outcomes Relative to Passive

- Statement: enabling banking/lending should produce measurably different
  system trajectories than passive clearing.
- Why: reserve constraints and bank credit channels alter payment dynamics.
- Test anchor:
  - `tests/regression/test_simulation_properties.py::test_banking_differs_from_passive`

## SF-4: Invariants Are Non-Negotiable

- Statement: invariant checks must hold regardless of policy mode.
- Why: accounting consistency is a model validity requirement, not a tuning goal.
- Test anchor:
  - `tests/property/test_invariants_property.py`
  - `tests/regression/test_simulation_properties.py::test_system_invariants_after_simulation`

## Change Control

If a stylized fact fails:
1. classify as model bug vs intended mechanism change
2. update tests and this document together
3. record rationale in PR with before/after metrics

