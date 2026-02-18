# 4-Arm Balanced Comparison Sweep — February 2025

## Configuration

| Parameter | Value |
|-----------|-------|
| Arms | passive, active (dealer), lender (NBFI), dealer+lender |
| n_agents | 50 |
| maturity_days | 10 |
| face_value | 20 |
| kappas | 0.25, 0.5, 1, 2, 4 |
| concentrations | 0.5, 1, 2 |
| mus | 0, 0.5, 1 |
| outside_mid_ratio (ρ) | 0.90 |
| risk_aversion | 0 |
| planning_horizon | 10 |
| rollover | enabled |
| Total pairs | 45 (5×3×3) |
| Total runs | 180 (45 pairs × 4 arms) |
| Duration | 1h 49m (local) |

## Results by Kappa (averaged over c and μ)

| κ | δ_passive | δ_active | δ_lender | δ_dealer+lender | trading_effect | lending_effect | combined_effect |
|---|-----------|----------|----------|-----------------|----------------|----------------|-----------------|
| 0.25 | 0.801 | 0.737 | 0.736 | 0.774 | 0.065 | 0.065 | 0.027 |
| 0.50 | 0.738 | 0.647 | 0.651 | 0.676 | 0.091 | 0.087 | 0.061 |
| 1.00 | 0.579 | 0.472 | 0.460 | 0.508 | 0.107 | 0.119 | 0.071 |
| 2.00 | 0.316 | 0.237 | 0.233 | 0.274 | 0.079 | 0.084 | 0.042 |
| 4.00 | 0.113 | 0.075 | 0.090 | 0.097 | 0.038 | 0.023 | 0.016 |

- **Trading improved defaults in 32/45 pairs (71%)**
- Trading effect peaks at κ=1 (10.7pp reduction)
- Lending effect is comparable to trading at all κ levels
- Combined dealer+lender effect is smaller than each individual effect — diminishing returns when both operate

## Key Findings

1. **Both dealers and lenders reduce defaults**, with similar magnitude at moderate stress (κ=0.5–2).
2. **Diminishing returns**: The combined arm (dealer+lender) does not achieve the sum of individual effects, suggesting they compete for the same marginal agents.
3. **Dealer effect peaks at κ=1**: At extreme stress (κ=0.25), defaults are so high that trading can only marginally help. At low stress (κ=4), few agents need help.
4. **Lending slightly outperforms trading at κ=1–2**, but trading is better at extreme stress (κ=0.25).

## Banking Arms (Not Included)

The three banking arms (bank-passive, bank-dealer, bank-dealer-NBFI) were excluded from this sweep due to a convergence issue: at low κ, cascading bank failures generate excessive reserve transfer retries, causing individual runs to take 30+ minutes. This is a performance issue in the banking subsystem (Plan 039) that needs optimization before full sweeps are feasible.

## Files

- `comparison.csv` — Full results with all metrics per pair
- `summary.json` — Aggregate summary statistics
