# Plan 041: κ-Informed Pricing and Deviation-from-Expectation Rate Adjustment

## Motivation

The current sweep results show that banking arms produce 100-1000x more system loss than passive runs, and NBFI lending mostly redistributes loss rather than reducing it. The root cause: intermediaries price as if operating in a normal economy regardless of system stress.

- **CB corridor**: hardcoded at floor=1%, ceiling=3% for all κ
- **NBFI**: starts at `base_rate=0.05`, initial `p_default=0.15` regardless of κ
- **Bank**: Treynor pricing pinned to the static CB corridor
- **VBT**: already κ-informed (uses `kappa_informed_prior` for credit-adjusted mid)

When κ=0.3, the system will see 60-80% default rates, but the CB charges 3% and the NBFI charges ~8%. Intermediaries lend freely into a failing system, absorbing catastrophic losses.

## Design Principle: Deviation from Expectation

All intermediaries should:

1. **Initialize** rates using `P_0 = kappa_informed_prior(κ)` — embedding the system's structural risk into day-zero pricing
2. **Adjust** rates based on how realized defaults deviate from that expectation — only tightening when things are *worse* than κ implied

The key formula:

```
surprise_t = max(0, P_realized_t - P_0)
```

If the system behaves as badly as κ predicted, surprise = 0 and rates stay put. Rates only move when the system is worse than expected.

## Component 1: CB Corridor — κ-Informed Initialization + Dynamic Adjustment

### Current state

```python
# domain/agents/central_bank.py
reserve_remuneration_rate: Decimal = Decimal("0.01")  # floor, static
cb_lending_rate: Decimal = Decimal("0.03")             # ceiling, static
```

Rate escalation exists (`effective_lending_rate`) but is turned off (`rate_escalation_slope=0`).

### New design

#### 1a. κ-informed initial corridor

The corridor encodes the CB's view of system risk at day zero.

```
P_0 = kappa_informed_prior(κ)

mid_0   = base_mid   + α_mid   × P_0
width_0 = base_width + α_width × P_0

r_floor_0   = mid_0 - width_0 / 2
r_ceiling_0 = mid_0 + width_0 / 2
```

Parameters (suggested defaults):
- `base_mid = 0.02` — corridor midpoint at zero default risk
- `α_mid = 0.10` — mid sensitivity to default probability
- `base_width = 0.02` — corridor width at zero default risk
- `α_width = 0.10` — width sensitivity to default probability

Example at κ=0.3 (P_0 ≈ 0.15):
- `mid_0 = 0.02 + 0.10 × 0.15 = 0.035`
- `width_0 = 0.02 + 0.10 × 0.15 = 0.035`
- `r_floor = 0.035 - 0.0175 = 0.0175` (1.75%)
- `r_ceiling = 0.035 + 0.0175 = 0.0525` (5.25%)

Example at κ=2.0 (P_0 = 0.05):
- `mid_0 = 0.02 + 0.10 × 0.05 = 0.025`
- `width_0 = 0.02 + 0.10 × 0.05 = 0.025`
- `r_floor = 0.025 - 0.0125 = 0.0125` (1.25%)
- `r_ceiling = 0.025 + 0.0125 = 0.0375` (3.75%)

#### 1b. Dynamic corridor adjustment (deviation from expectation)

Each day, the CB observes the realized default rate and adjusts:

```
P_realized_t = n_defaulted_agents_t / n_total_agents

surprise_t = max(0, P_realized_t - P_0)

mid_t   = mid_0   + β_mid   × surprise_t
width_t = width_0 + β_width × surprise_t

r_floor_t   = mid_t - width_t / 2
r_ceiling_t = mid_t + width_t / 2
```

Parameters:
- `β_mid = 0.50` — how aggressively mid rises per unit of surprise
- `β_width = 0.30` — how aggressively spread widens per unit of surprise

When `P_realized = P_0`, surprise = 0 and the corridor stays at initial values. When defaults exceed expectations, mid rises and spread widens — the CB tightens.

### Implementation

**Where**: `CentralBank` dataclass in `domain/agents/central_bank.py`

New fields:
```python
# κ-informed initial corridor
kappa_prior: Decimal = Decimal("0.15")  # P_0, set at init from kappa_informed_prior
base_mid: Decimal = Decimal("0.02")
alpha_mid: Decimal = Decimal("0.10")
base_width: Decimal = Decimal("0.02")
alpha_width: Decimal = Decimal("0.10")

# Dynamic adjustment
beta_mid: Decimal = Decimal("0.50")
beta_width: Decimal = Decimal("0.30")
```

New method:
```python
def compute_corridor(self, n_defaulted: int, n_total: int) -> tuple[Decimal, Decimal]:
    """Compute current floor and ceiling based on deviation from expectation.

    Returns (r_floor, r_ceiling).
    """
    P_0 = self.kappa_prior
    P_realized = Decimal(n_defaulted) / Decimal(max(n_total, 1))
    surprise = max(Decimal(0), P_realized - P_0)

    mid = self.base_mid + self.alpha_mid * P_0 + self.beta_mid * surprise
    width = self.base_width + self.alpha_width * P_0 + self.beta_width * surprise

    r_floor = max(Decimal(0), mid - width / 2)
    r_ceiling = mid + width / 2
    return r_floor, r_ceiling
```

**Call site**: The banking day runner (`banking/day_runner.py`) should call `compute_corridor()` at the start of each day, and update the `PricingParams` that the bank's Treynor kernel uses. This flows through automatically — the bank's inside spread, midline, and quotes all derive from the corridor parameters.

**Wiring**: In the compiler (`scenarios/ring/compiler.py`), when creating the CentralBank agent for banking arms, set `kappa_prior = kappa_informed_prior(κ)` and pass the α/β parameters.

## Component 2: NBFI — κ-Informed Initial Prior

### Current state

```python
# engines/lending.py, line 186
p_default = default_probs.get(agent_id, Decimal("0.15"))  # flat prior
```

```python
# LendingConfig
base_rate: Decimal = Decimal("0.05")  # κ-unaware
```

The NBFI already has Bayesian updating via `RiskAssessor.update_history()` and blends with a coverage-based heuristic. The gap is the initial prior.

### New design

#### 2a. κ-informed initial prior

Replace the flat 0.15 prior with `P_0(κ)`:

```python
# In LendingConfig or at wiring time
initial_prior: Decimal  # = kappa_informed_prior(κ), set at scenario compilation
```

This flows into:
- `_estimate_default_probs()` fallback: use `P_0` instead of `0.15`
- `RiskAssessor` initialization: set the no-history prior to `P_0`
- `LenderProfile.base_default_estimate`: set to `P_0`

#### 2b. κ-informed base rate

The NBFI's base rate should also reflect system risk:

```
base_rate = profit_target + α_nbfi × P_0
```

Where `profit_target ≈ 0.02` (desired margin) and `α_nbfi ≈ 0.20` (risk loading).

At κ=0.3 (P_0=0.15): `rate ≈ 0.02 + 0.20 × 0.15 = 0.05` (5%)
At κ=2.0 (P_0=0.05): `rate ≈ 0.02 + 0.20 × 0.05 = 0.03` (3%)

The per-borrower `risk_premium_scale × p_default` is then added on top, using the Bayesian posterior. So the NBFI starts expensive when κ is low and cheap when κ is high — with per-borrower discrimination on top.

#### 2c. Dynamic adjustment (already exists)

The Bayesian updating already adjusts `p_default` per borrower as defaults are observed. The RiskAssessor's `update_history()` is called on every loan repayment/default (line 373). No new mechanism needed — just ensure the prior is correctly set to P_0(κ).

### Implementation

**Where**:
- `scenarios/ring/compiler.py`: pass `kappa_informed_prior(κ)` into lending config
- `engines/lending.py`: use `config.initial_prior` instead of hardcoded `0.15`
- `decision/risk_assessment.py`: accept κ-informed prior for no-history default

Minimal code change — mostly wiring the prior through existing config paths.

## Component 3: VBT — Already κ-Informed (Verification)

### Current state

The VBT is already the best-calibrated intermediary:

1. **Initial cash**: `cash_ratio = 1 - kappa_informed_prior(κ)` (compiler line 428)
2. **Credit-adjusted mid**: `M = outside_mid_ratio × (1 - P_default)` updated daily
3. **Daily update**: `_update_vbt_credit_mids` recomputes mids using RiskAssessor posteriors
4. **Sensitivity**: `vbt_mid_sensitivity` (default 1.0) controls reaction to defaults

### What to verify

- That `vbt_mid_sensitivity=1.0` is flowing through correctly (it is)
- That the VBT's `RiskAssessor` starts from `P_0(κ)` not a flat prior
- Confirm the VBT spread also widens with defaults (controlled by `vbt_spread_sensitivity`, default 0.0 — should this be non-zero?)

### Decision: `vbt_spread_sensitivity`

Currently `vbt_spread_sensitivity=0.0` means VBT spread is fixed. For consistency with the CB corridor widening with surprise, we should consider setting a default > 0. However, this is a secondary concern — the mid adjustment is the primary channel. Leave at 0.0 for now, note as future work.

## Component 4: Bank — Inherits from CB Corridor

### Current state

The bank's Treynor pricing kernel takes `PricingParams` which includes:
- `reserve_remuneration_rate` (= CB floor)
- `cb_borrowing_rate` (= CB ceiling)

These are set once at initialization and never updated.

### New design

When the CB corridor adjusts daily (Component 1b), the bank's `PricingParams` must be updated to match. The bank's inside spread, midline, and quotes then automatically reflect the new corridor.

### Implementation

In `banking/day_runner.py`, at the start of each day:
1. Call `central_bank.compute_corridor(n_defaulted, n_total)` to get current floor/ceiling
2. Update the bank's `PricingParams.reserve_remuneration_rate` and `PricingParams.cb_borrowing_rate`
3. The Treynor kernel then produces correctly-calibrated quotes for the rest of the day

The `credit_risk_loading` parameter (0.5) adds per-borrower credit risk on top of the Treynor rate — this already works and doesn't need changing. The fix is in the corridor it operates within.

## Implementation Order

### Phase 1: κ-Informed Initialization (no dynamic adjustment yet)

1. **CB corridor**: Add `kappa_prior` field to `CentralBank`. Add `corridor_mid_for_kappa()` and `corridor_width_for_kappa()` class methods to `BankProfile` (or similar). Wire in compiler to set initial floor/ceiling from κ.
2. **NBFI prior**: Wire `kappa_informed_prior(κ)` through `LendingConfig.initial_prior`. Replace hardcoded `0.15` fallbacks.
3. **Bank PricingParams**: Inherits from CB change automatically.
4. **VBT**: Verify already correct. Ensure RiskAssessor prior = P_0(κ).

**Test**: Re-run the 7-arm sweep. Banking arm losses should be substantially lower since initial rates now embed system risk.

### Phase 2: Dynamic Corridor Adjustment

5. **CB `compute_corridor()`**: Implement the deviation-from-expectation method.
6. **Banking day runner**: Call `compute_corridor()` each day, update PricingParams.
7. **NBFI corridor anchor**: The lending.py corridor anchor (lines 238-248) already reads from the banking subsystem — verify it picks up the updated corridor.

**Test**: Re-run sweep. Banking arm losses should decrease further, especially at low κ where defaults exceed initial expectations the most.

### Phase 3: Parameter Sweep

8. **Expose α/β parameters**: Add `--cb-alpha-mid`, `--cb-beta-mid`, etc. to CLI (or bundle into a `--cb-corridor-mode` preset).
9. **Sweep α/β**: Run sweeps varying the sensitivity parameters to find values that produce reasonable loss levels.

## Files to Modify

| File | Change |
|------|--------|
| `domain/agents/central_bank.py` | Add `kappa_prior`, α/β fields, `compute_corridor()` method |
| `scenarios/ring/compiler.py` | Set CB `kappa_prior` from `kappa_informed_prior(κ)`, compute initial corridor |
| `engines/lending.py` | Use `config.initial_prior` instead of hardcoded `0.15` |
| `engines/lending.py:LendingConfig` | Add `initial_prior` field |
| `banking/day_runner.py` | Update PricingParams from CB corridor each day |
| `ui/run.py` | Wire κ-informed CB params at simulation startup |
| `experiments/balanced_comparison.py` | Pass α/β params through config |
| `ui/cli/sweep.py` | Expose new CLI options |
| `dealer/priors.py` | No change (already provides `kappa_informed_prior`) |

## Expected Impact

| Arm | Before (current) | After Phase 1 | After Phase 2 |
|-----|------------------|---------------|---------------|
| Passive | baseline | unchanged | unchanged |
| Active | sometimes better | unchanged | unchanged |
| NBFI | mostly worse | less worse (higher initial rates = less reckless lending) | similar |
| Bank arms | catastrophically worse | much less worse (corridor embeds risk) | further improved (corridor tightens with surprise defaults) |

The key prediction: banking arms should go from 100-1000x worse to being in the same order of magnitude as other arms, because the CB corridor will be wide enough to make bank lending properly expensive in stressed systems.

## Open Questions

1. **Should `kappa_informed_prior` be recalibrated?** Current range is [0.05, 0.20] but realized defaults reach 0.60-0.80. The prior may need to be more aggressive: e.g., `P_0 = 1 / (1 + κ)` which gives 0.77 at κ=0.3 and 0.33 at κ=2. However, P_0 isn't meant to predict the *final* default rate — it's the *initial* assessment before any cascade. The cascade itself is emergent.

2. **Should the NBFI also widen its risk premium with surprise?** Currently `risk_premium_scale` is static. Could make it `risk_premium_scale × (1 + β_nbfi × surprise)`.

3. **Should the CB corridor have a floor on width?** To prevent it from collapsing to zero when κ is very high.

4. **How should `max_borrower_risk` interact with the κ-informed prior?** If P_0(κ=0.3) = 0.15 and `max_borrower_risk=0.4`, the bank will lend to almost everyone initially. But if P_0 were closer to the true default rate, many borrowers would be rationed immediately.
