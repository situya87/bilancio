# Validation Matrix

This matrix maps key behavioral claims to executable tests and diagnostics.

## Structural and Accounting Validity

| Claim | Validation Mechanism | Current Evidence |
|---|---|---|
| Double-entry and accounting invariants hold during simulation | Property and regression tests invoking invariant assertions | `tests/property/test_invariants_property.py`, `tests/regression/test_simulation_properties.py` |
| No negative-cash regressions in active trading mode | Engine-level regression tests | `tests/engines/test_no_negative_cash.py` |
| Settlement clearing behavior remains coherent under edge cases | Clearing and settlement coverage tests | `tests/engines/test_clearing_coverage.py`, `tests/engines/test_settlement_coverage.py` |

## Behavioral / Stylized Validation

| Claim | Validation Mechanism | Current Evidence |
|---|---|---|
| Lower liquidity (`kappa`) increases default pressure | Comparative regression runs across stress levels | `tests/regression/test_simulation_properties.py::test_defaults_increase_with_lower_kappa` |
| NBFI supply activates under shortfall stress | Event-based regression tests for loan creation | `tests/regression/test_simulation_properties.py::test_lending_effect_is_nonzero` |
| Banking regime differs from passive baseline | A/B regression comparison with equal seed | `tests/regression/test_simulation_properties.py::test_banking_differs_from_passive` |
| Stylized low-kappa stress generates defaults | Dedicated stylized regression test | `tests/regression/test_simulation_properties.py::test_stylized_fact_liquidity_shortfall_produces_defaults` |

## Statistical and Experiment Integrity

| Claim | Validation Mechanism | Current Evidence |
|---|---|---|
| Paired effects and inferential metrics are computed consistently | Unit tests for stats modules and sweep analysis | `tests/unit/test_stats.py`, `tests/unit/test_sweep_analysis.py` |
| Seed handling provides reproducible sweeps | Reproducibility tests with same/different seeds | `tests/experiments/test_sweep_runners.py` |
| Performance guardrails catch gross regressions | Throughput benchmarks in test suite | `tests/benchmark/test_performance.py` |

## Known Gaps To Address

- Calibration against external empirical targets is still limited.
- Automated sensitivity stress suites should be expanded across more
  high-dimensional parameter combinations.
- Additional stylized-fact coverage is needed for intermediary-loss behavior.

