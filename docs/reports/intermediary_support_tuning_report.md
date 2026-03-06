# Intermediary Support Tuning Report

Generated: `2026-03-06T08:18:22.101909+00:00`

This report tunes support behavior sequentially for dealer-only, bank-only, and NBFI-only comparisons.
The ranking objective is lexicographic: maximize safe help, maximize help with non-negative net system relief, minimize loss per unit of help, then maximize safe default relief.

> **Note**: This report was generated with small-scale parameters (`n_agents=36`, `discovery_reps=1`) for quick iteration. Results are directional only and should be confirmed with production-scale parameters before drawing firm conclusions.

## Run Envelope

- `n_agents`: `36`
- `discovery_reps`: `1`
- `validation_reps`: `2`
- `discovery_base_seed`: `4100`
- `validation_base_seed`: `9100`
- `kappas`: `0.5,2`
- `concentrations`: `0.5,1.5`
- `mus`: `0,0.5`
- `outside_mid_ratios`: `0.90`

## Dealer-Only

### Candidate Set

| Candidate | Rationale |
| --- | --- |
| baseline | Current dealer-side comparison defaults. |
| flow_buffered_vbt | Slightly wider VBT response to stress and order-flow pressure. |
| prudent_traders | Moderately longer planning and lower urgency from traders. |
| coordinated_buffering | Combine modest trader prudence with modest VBT stress buffering. |

### Discovery Sweep

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8/8 | 0.375 | 0.375 | 0.000 | 0.051 | -324.125 | `discovery/dealer/baseline` |
| flow_buffered_vbt | 8/8 | 0.375 | 0.375 | 20550.210 | 0.060 | -123.935 | `discovery/dealer/flow_buffered_vbt` |
| prudent_traders | 8/8 | 0.375 | 0.375 | 2527.121 | 0.059 | -123.768 | `discovery/dealer/prudent_traders` |
| coordinated_buffering | 8/8 | 0.375 | 0.375 | 20550.210 | 0.060 | 33.065 | `discovery/dealer/coordinated_buffering` |

### Held-Out Validation

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 16/16 | 0.312 | 0.312 | 63303.834 | 0.090 | 94.936 | `validation/dealer/baseline` |

### Final Recommendation: `baseline`

Current dealer-side comparison defaults.

Recommended overrides:
No override beat the baseline.

Frontier artifacts: `validation/dealer/baseline/aggregate/intermediary_frontier`

## Bank-Only

### Candidate Set

| Candidate | Rationale |
| --- | --- |
| baseline | Current bank comparison defaults. |
| cb_liquidity_buffer | Softer CB escalation and slightly deeper backstop capacity. |
| trader_buffer | Moderately more conservative trader purchasing under stress. |
| cb_plus_trader_buffer | Combine a softer CB backstop with modest trader prudence. |

### Discovery Sweep

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8/8 | 0.000 | 0.250 | 10885.741 | - | -1506.125 | `discovery/bank/baseline` |
| cb_liquidity_buffer | 8/8 | 0.000 | 0.125 | 12035.550 | - | -923.125 | `discovery/bank/cb_liquidity_buffer` |
| trader_buffer | 8/8 | 0.000 | 0.250 | 10537.271 | - | -1999.000 | `discovery/bank/trader_buffer` |
| cb_plus_trader_buffer | 8/8 | 0.000 | 0.125 | 11815.789 | - | -2465.000 | `discovery/bank/cb_plus_trader_buffer` |

### Held-Out Validation

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 16/16 | 0.000 | 0.188 | 9475.850 | - | -1198.938 | `validation/bank/baseline` |
| trader_buffer | 16/16 | 0.000 | 0.188 | 9748.615 | - | -1753.062 | `validation/bank/trader_buffer` |

### Final Recommendation: `baseline`

Current bank comparison defaults.

Recommended overrides:
No override beat the baseline.

Frontier artifacts: `validation/bank/baseline/aggregate/intermediary_frontier`

## NBFI-Only

### Candidate Set

| Candidate | Rationale |
| --- | --- |
| baseline | Current NBFI comparison defaults. |
| measured_guardrails | Light-touch NBFI loss discipline that aims to preserve support capacity. |
| discipline_buffer | Tighter underwriting, shorter risky terms, and collateral discipline. |
| preventive_guardrails | Allow selective preventive support, but only with explicit loss guardrails. |
| guardrails_plus_prudent_flow | Pair NBFI guardrails with modest trader and VBT stress-damping. |

### Discovery Sweep

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8/8 | 0.125 | 0.125 | 0.000 | 0.023 | -38.500 | `discovery/nbfi/baseline` |
| measured_guardrails | 8/8 | 0.125 | 0.125 | 0.000 | 0.023 | 40.375 | `discovery/nbfi/measured_guardrails` |
| discipline_buffer | 8/8 | 0.000 | 0.000 | - | - | -4.500 | `discovery/nbfi/discipline_buffer` |
| preventive_guardrails | 8/8 | 0.000 | 0.000 | 0.000 | - | -14.000 | `discovery/nbfi/preventive_guardrails` |
| guardrails_plus_prudent_flow | 8/8 | 0.000 | 0.000 | - | - | -4.375 | `discovery/nbfi/guardrails_plus_prudent_flow` |

### Held-Out Validation

| Candidate | Complete | Safe Help Share | Net+ Help Share | Loss/Help | Mean Safe Relief | Mean Net Relief | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 16/16 | 0.250 | 0.250 | 2387.458 | 0.085 | 45.313 | `validation/nbfi/baseline` |
| measured_guardrails | 16/16 | 0.188 | 0.188 | 3295.198 | 0.093 | 42.812 | `validation/nbfi/measured_guardrails` |

### Final Recommendation: `baseline`

Current NBFI comparison defaults.

Recommended overrides:
No override beat the baseline.

Frontier artifacts: `validation/nbfi/baseline/aggregate/intermediary_frontier`
