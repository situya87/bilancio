# RNG Governance

This document describes how random number generators (RNGs) are managed
in bilancio simulations to ensure reproducibility.

## Seed Derivation

The simulation uses a hierarchical seed structure:

1. **Base seed** (`base_seed`): Set in the scenario or sweep configuration.
   Each run in a sweep derives its seed sequentially from the base.

2. **Dealer subsystem RNG** (`dealer_config.seed`): The dealer subsystem
   creates its own `random.Random` instance seeded from the dealer config.

3. **Per-trader RNGs**: Each trader gets an independent RNG via
   `rng.randint(0, 2**31)` from the subsystem RNG. This ensures trader
   decisions are independent and reproducible.

4. **Rating RNG**: The rating/risk assessment system uses a separate
   RNG stream for credit evaluations.

## Independent Streams

| Stream | Scope | Seeded From |
|--------|-------|-------------|
| Scenario RNG | Ring setup, debt allocation | `base_seed` |
| Dealer subsystem RNG | Dealer matching, bucket selection | `dealer_config.seed` |
| Per-trader RNGs | Buy/sell decisions, trade selection | subsystem RNG |
| Rating RNG | Credit risk assessment | scenario seed |

## Semantics-Preserving Rule

**Critical invariant**: Semantics-preserving optimizations (those in
`SEMANTICS_PRESERVING` in `core/performance.py`) must NEVER call an RNG.

This means:
- Caching dealer quotes: OK (avoids recomputation, no RNG call)
- Fast atomic: OK (skips deepcopy, no RNG call)
- Prune ineligible: OK (skips agents, but doesn't change RNG sequence)
- Preview buy: OK (different code path, same RNG sequence)

The `matching_order` flag is classified as `SEMANTICS_CHANGING` because
sorting sellers/buyers by urgency changes the order in which RNG calls
happen during trade matching, potentially producing different outcomes.

## Verification

The regression test suite (`tests/regression/test_performance_parity.py`)
verifies that compatible, fast, and aggressive presets produce
bit-identical `delta_total` and `phi_total` for the same seed.
