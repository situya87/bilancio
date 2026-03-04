# Calibration Guide

This guide provides recommended parameter envelopes and known-good
combinations for bilancio simulations.

## Parameter Envelopes by Regime

### Low Stress (exploratory)

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| kappa (κ) | 1.5 - 4.0 | Liquidity abundant |
| concentration (c) | 1.0 - 3.0 | Moderate to equal debt distribution |
| mu (μ) | 0.3 - 0.7 | Evenly distributed maturity timing |
| outside_mid_ratio (ρ) | 0.85 - 0.95 | Moderate outside money |
| n_agents | 50 - 200 | Larger samples reduce noise |
| maturity_days | 5 - 15 | Standard horizons |

### Moderate Stress (default regime for sweeps)

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| kappa (κ) | 0.5 - 2.0 | Balanced liquidity |
| concentration (c) | 0.5 - 2.0 | Some inequality |
| mu (μ) | 0.0 - 0.5 | Front- to mid-loaded |
| outside_mid_ratio (ρ) | 0.85 - 0.95 | Standard |
| n_agents | 50 - 100 | |
| maturity_days | 8 - 12 | |

### High Stress (crisis scenarios)

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| kappa (κ) | 0.1 - 0.5 | Severely constrained |
| concentration (c) | 0.2 - 1.0 | Unequal debt |
| mu (μ) | 0.0 - 0.25 | Front-loaded stress |
| outside_mid_ratio (ρ) | 0.80 - 0.90 | Lower outside money |
| n_agents | 20 - 50 | Smaller rings for speed |
| maturity_days | 5 - 8 | Shorter horizons |

## Known-Good Combinations

These parameter sets have been validated through extensive testing:

### Dealer Effect Benchmark

```yaml
n_agents: 100
maturity_days: 10
kappas: [0.25, 0.5, 1, 2, 4]
concentrations: [1]
mus: [0]
outside_mid_ratios: [0.90]
risk_aversion: 0
planning_horizon: 10
```

Expected behavior: trading effect positive for κ < 1, diminishing for κ > 2.

### Bank Lending Benchmark

```yaml
n_agents: 100
maturity_days: 10
kappas: [0.3, 0.5, 1, 2]
concentrations: [1]
mus: [0]
credit_risk_loading: 0.5
max_borrower_risk: 0.5
```

Expected behavior: lending reduces defaults for κ < 1.

### Scientific Comparison (Minimal)

```yaml
n_agents: 24
maturity_days: 6
kappas: [0.4, 1.0]
concentrations: [0.5, 2.0]
mus: [0.25, 0.75]
replicates: 4
```

Expected behavior: sufficient power for detecting MDE = 0.05 with 4 replicates.

## Sweep Pre-Flight Protocol

Before running any sweep, the CLI enforces a pre-flight check (see
CLAUDE.md, "Sweep Pre-Flight: Interactive Parameter Review"). This
protocol includes:

1. Parameter sanity checks (valid ranges, warnings for extremes)
2. Economic viability checks (sell/buy trade viability, dealer capacity)
3. Scale and cost estimation
4. User confirmation before execution

Refer to the pre-flight section in CLAUDE.md for the complete protocol.

## Anti-Patterns

Avoid these parameter combinations:

| Anti-Pattern | Why | Fix |
|-------------|-----|-----|
| κ > 10 | No stress, no trading activity | Use κ ≤ 4 |
| κ < 0.05 | Nearly everyone defaults immediately | Use κ ≥ 0.1 |
| c < 0.1 | One agent holds almost all debt | Use c ≥ 0.2 |
| ρ = 1.0 | Buy trades impossible (no discount) | Use ρ ≤ 0.95 |
| maturity_days = 1 | No temporal spread for trading | Use ≥ 5 |
| n_agents < 10 | Too noisy for meaningful statistics | Use ≥ 20 |
| n_agents > 500 | Very slow without clear benefit | Use ≤ 200 |
