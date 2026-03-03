# Alternative ABM Modeling Benchmark (AABM)

## Purpose

The AABM benchmark actually **runs simulations** to verify that the ABM produces correct economic behavior. Unlike the original ABM Modeling Benchmark (which checks whether test suites pass and documentation exists), AABM measures:

- **Seed reproducibility** — deterministic results from deterministic inputs
- **Accounting conservation** — system invariants hold under diverse parameters
- **Comparative statics** — economic monotonicity properties hold (e.g., lower liquidity → more defaults)
- **Cross-seed convergence** — results are stable across different random seeds
- **Dealer effect direction** — secondary market trading weakly reduces defaults
- **Boundary behavior** — extreme parameters produce expected extreme outcomes

## Running

```bash
uv run python scripts/run_alt_abm_modeling_benchmark.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target-score` | 80 | Minimum score to pass |
| `--out-json` | `temp/alt_abm_modeling_benchmark_report.json` | JSON report path |
| `--out-md` | `temp/alt_abm_modeling_benchmark_report.md` | Markdown report path |

### Expected runtime

~1-2 seconds locally. The benchmark uses small ring sizes (n=20-25) for speed.

## Score Model (100 points)

| # | Category | Points | What it measures |
|---|----------|--------|------------------|
| 1 | Seed Reproducibility | 15 | Same seed → identical results (10), different seed → different results (5) |
| 2 | Accounting Conservation | 20 | `system.assert_invariants()` + no negative cash across 6 parameter combos |
| 3 | Comparative Statics | 25 | Kappa monotonicity (15) + concentration/inequality effect (10) |
| 4 | Cross-Seed Convergence | 15 | CV of default counts across 5 seeds |
| 5 | Dealer Effect Direction | 15 | Mean active defaults ≤ mean passive (10) + strict improvement in ≥1 seed (5) |
| 6 | Boundary Behavior | 10 | κ=10 → 0 defaults (5) + κ=0.1 → ≥50% agents default (5) |

## Critical Gates

These gates cap the grade regardless of score. Any failure caps grade to C or worse.

| Gate | Condition | What it catches |
|------|-----------|-----------------|
| `reproducibility::same_seed` | Same seed produces identical default + event counts | Non-determinism bugs |
| `conservation::all_pass` | All 6 parameter configs pass `assert_invariants()` | Double-entry or balance bugs |
| `statics::kappa_monotonic` | All 3 kappa pairs satisfy defaults(low κ) ≥ defaults(high κ) | Broken settlement logic |
| `boundary::high_kappa_no_defaults` | κ=10 produces exactly 0 defaults | False defaults under abundant liquidity |

## Category Details

### 1. Seed Reproducibility (15 pts)

**Same seed (10 pts):** Two runs with n=20, κ=0.5, Dirichlet c=0.5, maturity=5, seed=42 must produce identical default counts and total event counts. Uses explorer-generated rings with Dirichlet-distributed debts so the seed controls meaningful randomness (unlike uniform rings where all agents are identical).

**Different seed (5 pts):** seed=42 vs seed=99 must produce at least one difference (default count or event count). With Dirichlet concentration c=0.5, different seeds produce different debt distributions, so results should differ.

### 2. Accounting Conservation (20 pts)

Six parameter configurations spanning different stress levels:

| Config | n | κ | maturity | seed | Stress level |
|--------|---|---|----------|------|-------------|
| small_ring | 10 | 0.5 | 3 | 42 | Moderate, small |
| stressed | 20 | 0.3 | 5 | 42 | High stress |
| kappa_1 | 20 | 1.0 | 5 | 42 | Balanced |
| abundant | 20 | 2.0 | 5 | 42 | Low stress |
| medium_alt_seed | 15 | 0.8 | 7 | 99 | Medium, alt seed |
| large_long_maturity | 25 | 0.6 | 10 | 7 | Long maturity |

Each config: build ring, run until stable, check `system.assert_invariants()` + no negative cash. Score = 20 × (passed / 6).

### 3. Comparative Statics (25 pts)

**Kappa monotonicity (15 pts):** With n=20, maturity=5, seed=42, and κ ∈ {0.3, 0.5, 1.0, 2.0}:
- defaults(κ=0.3) ≥ defaults(κ=0.5) ≥ defaults(κ=1.0) ≥ defaults(κ=2.0)
- 3 pairwise comparisons, score = 15 × (pairs_met / 3)

**Concentration effect (10 pts):** Using `compile_ring_explorer` with Dirichlet concentration parameter c ∈ {0.5, 5.0} at κ=0.5:
- With uniform cash allocation, more equal debts (high c) means ALL agents are equally underwater → more defaults
- More unequal debts (low c) means many agents have tiny debts they can pay → fewer defaults
- defaults(c=5.0) ≥ defaults(c=0.5) → 10 pts if met

Note: This direction is specific to uniform cash allocation. With proportional cash allocation the effect would reverse.

### 4. Cross-Seed Convergence (15 pts)

n=25, κ=0.5, maturity=5. Seeds {42, 43, 44, 45, 46}. Coefficient of variation (CV) of default counts across seeds:

| CV | Score |
|----|-------|
| ≤ 0.2 | 15 (full) |
| 0.2 - 0.8 | Linear interpolation |
| ≥ 0.8 | 0 |

Low CV indicates stable, predictable behavior across random seeds.

### 5. Dealer Effect Direction (15 pts)

n=25, κ=0.5, Dirichlet c=0.5, maturity=5. Seeds {42, 43, 44}. Each seed: passive run vs active run (with dealer subsystem using `DealerRingConfig` with standard parameters). Uses explorer-generated rings with heterogeneous debts to give the dealer secondary market trading opportunities.

- **Mean condition (10 pts):** mean(active_defaults) ≤ mean(passive_defaults)
- **Strict improvement (5 pts):** At least one seed where active_defaults < passive_defaults

The dealer subsystem should weakly reduce defaults by providing liquidity through secondary market trading. Note: at small ring sizes with standard VBT parameters, the dealer may not produce any trades, limiting its impact.

### 6. Boundary Behavior (10 pts)

**High κ (5 pts):** κ=10 (cash=1000, payable=100), n=20, maturity=5 → exactly 0 defaults. With 10× liquidity, no agent should ever default.

**Low κ (5 pts):** κ=0.1 (cash=10, payable=100), n=20, maturity=5 → ≥50% of agents default. With only 10% liquidity coverage, widespread default is expected.

## Grading

| Score | Grade |
|-------|-------|
| ≥ 90 | A |
| ≥ 80 | B |
| ≥ 70 | C |
| ≥ 60 | D |
| < 60 | F |

Critical gate failures cap the grade: 1 failure → cap at C, 2 → D, 3+ → F.

## Output

### JSON Report

```json
{
  "benchmark": "Alternative ABM Modeling Benchmark (AABM)",
  "total_score": 80.0,
  "grade": "B",
  "status": "PASS",
  "categories": [...],
  "critical_checks": [...],
  "critical_failures": [...]
}
```

### Markdown Report

Human-readable summary with scorecard, category table, critical gates table, and per-category details.
