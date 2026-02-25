# Plan 040: Decision Intelligence Improvements

## Motivation

The 7-arm sweep (job: `exemplify-tapioca-flashing-clamor`) revealed three
architectural gaps in agent decision-making:

| Gap | Description | Impact |
|-----|-------------|--------|
| **G2** | NBFI lender uses static heuristic (`1/(1+κ)/coverage`), no memory | Lending decisions ignore past loan outcomes |
| **G3** | VBT prices reactively from historical defaults only | No forward-looking stress signals → mispriced when stress is imminent |
| **G5** | Buy-side controls insufficient at high κ | Dealer extracts 15-23% of system cash via one-directional buy flow |

All three changes are **backward-compatible**: new parameters default to
values that reproduce current behavior exactly.

---

## Phase 1: NBFI Bayesian Belief Tracker

### Problem

The NBFI lender estimates default probability via:

```python
p_base = 1 / (1 + kappa)
coverage = (cash + receivables) / obligations
p_default = clamp(0.01, 0.95, p_base / coverage)
```

This is recalculated from scratch each day. The lender has no memory of
which borrowers repaid and which defaulted. A borrower who defaulted
yesterday gets the same rate as one who always repaid (if their coverage
ratios are equal).

### Solution

Give the NBFI its own `RiskAssessor` (from `decision/risk_assessment.py`).
The assessor's `BeliefTracker` maintains a sliding window of payment
outcomes, providing a Bayesian posterior that improves with data.

**Blending rule:**

```
n = number of observations for this borrower
w_bayes = min(1, n / warmup_observations)

p_final = w_bayes × p_bayesian + (1 - w_bayes) × p_coverage
```

Early in the simulation (few observations), coverage-ratio dominates.
As loan outcomes accumulate, the Bayesian posterior takes over.

### Changes

1. **`LenderProfile`** — add `risk_assessment_params`:
   ```python
   risk_assessment_params: RiskAssessmentParams | None = None
   warmup_observations: int = 10  # data points before Bayesian dominates
   ```

2. **`LendingConfig`** — add mutable `risk_assessor` field:
   ```python
   risk_assessor: RiskAssessor | None = None  # persists across days
   ```

3. **`run_lending_phase()`** — replace waterfall with assessor:
   - If `config.risk_assessor` exists, use it as primary p_default source
   - Blend with coverage-ratio when warmup incomplete
   - If no assessor and no profile, fall back to existing waterfall (backward compat)

4. **`run_loan_repayments()`** — update assessor with outcomes:
   - After each loan repays/defaults, call `assessor.update_history(day, borrower_id, defaulted)`

5. **Ring setup** (`config/apply.py`) — create assessor when LenderProfile has params:
   - Use `create_assessor()` factory from `decision/factories.py`

### Backward Compatibility

- `risk_assessment_params = None` (default) → existing waterfall behavior
- `warmup_observations = 10` → coverage ratio dominates for first ~10 data points

---

## Phase 2: VBT Forward-Looking Stress Signal

### Problem

VBT computes `M = ρ × (1 - P_default)` where `P_default` is a Bayesian
posterior over PAST payment history. If a large wave of obligations is about
to mature and agents lack cash to pay them, the VBT doesn't know and
continues pricing as if the future looks like the past.

### Solution

Add a forward stress estimate that scans upcoming obligation maturities
against available system cash.

**Forward stress formula:**

```
For horizon h days:
  total_due = Σ obligations maturing in [t, t+h]
  total_cash = Σ agent cash balances (non-defaulted agents)
  stress = max(0, 1 - total_cash / total_due)
```

When `total_cash ≥ total_due`, stress = 0 (system has enough cash).
When `total_cash < total_due`, stress > 0 (defaults likely).

**Blending into VBT pricing:**

```
p_blend = (1 - forward_weight) × p_historical + forward_weight × stress_forward
M = ρ × (1 - p_blend)
```

### Changes

1. **`VBTProfile`** — add forward-looking fields:
   ```python
   forward_weight: Decimal = Decimal("0.0")   # 0 = disabled
   stress_horizon: int = 5                     # days to look ahead
   ```

2. **`CreditAdjustedVBTPricing`** — add `compute_mid_blended()`:
   ```python
   def compute_mid_blended(
       self, p_default: Decimal, p_forward: Decimal,
       initial_prior: Decimal, forward_weight: Decimal,
   ) -> Decimal:
       p_blend = (1 - forward_weight) * p_default + forward_weight * p_forward
       return self._compute_mid_from_p(p_blend, initial_prior)
   ```

3. **New function** `estimate_forward_stress()` in `engines/dealer_sync.py`:
   - Scans `system.state.contracts` for obligations due within horizon
   - Computes aggregate cash vs obligations ratio
   - Returns Decimal stress signal ∈ [0, 1]

4. **`_update_vbt_credit_mids()`** — use blended estimate:
   - If `vbt_profile.forward_weight > 0`, compute forward stress and blend
   - Otherwise, use existing p_historical only (backward compat)

### Backward Compatibility

- `forward_weight = 0.0` (default) → `p_blend = p_historical` → existing behavior exactly

---

## Phase 3: System-Aware Buy Controls

### Problem

At high κ (kappa ≥ 1), agents accumulate surplus cash and use the dealer
to buy claims from the VBT. The VBT provides unlimited supply at nearly
fixed prices, so buying continues until the dealer has drained 15-23% of
system liquidity.

The root cause is two-fold:
1. The VBT has no concept of inventory cost or flow imbalance
2. The TradeGate's buy premium is too low when the earning opportunity
   is objectively small (low default probability → tiny EV gain)

### Solution A: VBT Flow-Aware Ask Pricing

The VBT should track its cumulative net outflow and widen the ask when
it is being drained. This is economically natural: a market maker facing
persistent one-directional flow raises prices on the depleted side.

**Formula:**

```
net_outflow = cumulative units sold to customers - cumulative units bought from customers
outflow_ratio = net_outflow / initial_inventory_value
ask_premium = flow_sensitivity × max(0, outflow_ratio)
A_effective = A + ask_premium
```

When flow is balanced (net_outflow ≈ 0), no adjustment. When the VBT has
been net-selling heavily, the ask rises, naturally discouraging more buys.

### Solution B: Earning-Motive Premium in TradeGate

When an agent buys for EARNING (no shortfall), add an additional premium
that reflects the low marginal value of the trade:

```
In should_buy():
  if trader_shortfall == 0:
    buy_threshold += earning_motive_premium
```

This creates a higher bar for speculative buys while leaving
liquidity-motivated buys unaffected.

### Changes

1. **`VBTProfile`** — add flow control:
   ```python
   flow_sensitivity: Decimal = Decimal("0.0")  # 0 = disabled
   ```

2. **`VBTState`** — add flow tracking:
   ```python
   cumulative_outflow: Decimal = Decimal("0")   # face value sold to customers
   cumulative_inflow: Decimal = Decimal("0")     # face value bought from customers
   ```

3. **Passthrough execution** — track flows:
   - After VBT sells a ticket (passthrough buy): increment outflow
   - After VBT buys a ticket (passthrough sell): increment inflow

4. **VBT ask computation** — apply flow premium:
   - `A_eff = A + flow_sensitivity × max(0, outflow_ratio)`
   - Applied in `recompute_quotes()` or at trade execution time

5. **`RiskAssessmentParams`** — add earning premium:
   ```python
   earning_motive_premium: Decimal = Decimal("0.0")  # 0 = disabled
   ```

6. **`TradeGate.should_buy()`** — apply earning premium:
   - When `trader_shortfall == 0`, add `earning_motive_premium` to threshold

### Backward Compatibility

- `flow_sensitivity = 0.0` → no VBT ask adjustment
- `earning_motive_premium = 0.0` → no additional buy premium

---

## Testing Strategy

### Phase 1 Tests
- NBFI assessor creates and persists across days
- Bayesian posterior converges with loan outcomes
- Blending: coverage dominates early, posterior dominates late
- Backward compat: no assessor → existing waterfall behavior
- Regression: existing NBFI tests still pass

### Phase 2 Tests
- Forward stress = 0 when system is liquid
- Forward stress > 0 when obligations exceed cash
- Blended mid is lower when forward stress is high
- Backward compat: forward_weight=0 → identical pricing
- Regression: existing VBT/dealer tests still pass

### Phase 3 Tests
- VBT outflow tracking increments correctly
- VBT ask widens with outflow
- Earning premium raises buy threshold when shortfall = 0
- Earning premium has no effect when shortfall > 0
- Backward compat: both parameters = 0 → existing behavior
- Integration: high-κ scenario shows reduced cash extraction

---

## Execution Order

Phase 1 (NBFI BeliefTracker) → Phase 2 (VBT Forward Stress) → Phase 3 (Buy Controls)

Each phase is independently valuable and independently testable.
