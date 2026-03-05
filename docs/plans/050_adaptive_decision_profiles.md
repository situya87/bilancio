# Plan 050: Adaptive Decision Profiles

## Motivation

The bilancio simulation has six agent types, each with a decision profile that governs pricing, risk assessment, and trading behavior. Currently, most profile parameters are **static defaults** that neither calibrate to scenario parameters (κ, μ, c) nor adapt to observed conditions during the run. This means:

- A VBT in a κ=0.3 world uses the same base spreads as one in κ=3
- A trader looking at a 1-day-to-maturity claim assigns the same default probability as a 15-day claim
- The CB corridor doesn't incorporate μ (maturity timing) or c (debt concentration)
- The NBFI lender charges the same base rate throughout the run even as 30% of borrowers default

This plan makes profiles **adaptive** — calibrating to scenario parameters before the run, and responding to observed conditions during the run.

## Architectural Principle: Who Adapts vs Who Pipes

```
VBT (intelligence) ──→ M, O quotes ──→ Dealer (pure Treynor) ──→ secondary market prices
CB  (intelligence) ──→ r_floor, r_ceiling ──→ Bank (pure Treynor) ──→ lending/deposit rates
```

- **Dealer** and **Bank** are Treynor market-makers. They adapt ONLY through (a) order flow changing their inventory position and (b) the outside quotes/rates they receive. They do NOT independently read scenario parameters or observed defaults. Their intelligence is *inherited* from the VBT and CB respectively.
- **VBT (Value-Based Trader)** is the primary intelligence for the secondary market. It observes the system, incorporates scenario parameters, and sets reference quotes (M, O) that flow through to the dealer via the Treynor kernel.
- **CB (Central Bank)** is the primary intelligence for the banking system. It observes default rates and bank-level indicators, and sets corridor rates (r_floor, r_ceiling) that flow through to banks via the Treynor kernel.
- **Trader (Firm/Household)** is the decision-making agent — fully adaptive in both valuation and behavior.
- **NBFI Lender** is an investor — independently considers scenario parameters, portfolio performance, and counterparty risk when making lending decisions.

**Consequence:** Improving VBT pricing automatically improves dealer pricing. Improving CB corridor logic automatically improves bank pricing. We do not add independent adaptation to dealers or banks.

## Design Principles

1. **All enhancements are FLAGS** — each new behavior has an on/off toggle, defaulting to off for backward compatibility.
2. **"Full adaptive" preset** — a single flag (`adaptive=True` or `preset="adaptive"`) turns everything on. Runs display which adaptive features are enabled in the pre-flight summary.
3. **No silent changes** — the pre-flight sweep display (CLAUDE.md §Step 6) shows all active adaptive features before any run.
4. **Profiles remain frozen dataclasses** — within-run adaptation happens in the engine layer, not by mutating profile fields. Profiles define the *capacity* for adaptation (sensitivity parameters); engines execute it.
5. **Testable in isolation** — each adaptive feature has a unit test that verifies the expected change in behavior. A/B comparison: adaptive-off vs adaptive-on.

---

## WI-1: Trader Pre-Run Calibration

**Goal:** Trader profile parameters calibrate to scenario parameters (κ, μ, c, maturity_days) before the run starts.

### 1a. Planning horizon scales with maturity_days

**Current:** `TraderProfile.planning_horizon = 10` (fixed).

**Problem:** In a 5-day scenario, looking 10 days ahead is pointless (no obligations exist). In a 20-day scenario, looking 10 days ahead misses half the obligations.

**Change:** When `adaptive_planning_horizon=True`:
```
planning_horizon = max(3, min(20, maturity_days))
```
This ensures the trader's look-ahead covers the full scenario time horizon without exceeding the 20-day cap.

**File:** `src/bilancio/decision/profiles.py` — add `adaptive_planning_horizon: bool = False` to `TraderProfile`.
**File:** Scenario compilation (wherever TraderProfile is instantiated) — apply formula when flag is on.

### 1b. Lookback window scales with maturity_days

**Current:** `RiskAssessmentParams.lookback_window = 5` (fixed).

**Problem:** In a 20-day scenario, a 5-day lookback forgets 75% of history. In a 3-day scenario, a 5-day lookback never fills up.

**Change:** When `adaptive_lookback=True`:
```
lookback_window = max(3, min(maturity_days, 15))
```

**File:** `src/bilancio/decision/risk_assessment.py` — add `adaptive_lookback: bool = False` to `RiskAssessmentParams`.

### 1c. Initial prior incorporates μ and c

**Current:** `kappa_informed_prior(κ) = 0.05 + 0.15 × max(0, 1-κ)/(1+κ)`. Range: [0.05, 0.20]. Only κ matters.

**Problem:** μ=0 (front-loaded stress) means early defaults are likely — the prior should be higher at run start. Low c (concentrated debt) means some issuers carry disproportionate risk — the prior should reflect systemic fragility, not just average liquidity.

**Change:** When `adaptive_prior=True`, extend `kappa_informed_prior` to a `scenario_informed_prior(κ, μ, c)`:
```
stress_kappa = max(0, 1 - κ) / (1 + κ)              # existing [0, 1]
stress_mu    = (1 - μ)^2                              # front-loading penalty [0, 1]
stress_c     = 1 / (1 + c)                            # concentration penalty [0, 1)

combined_stress = stress_kappa + 0.05 × stress_mu + 0.05 × stress_c
P_prior = 0.05 + 0.15 × min(combined_stress, 1.0)
```

The μ and c terms are deliberately small (0.05 weight each) — κ remains the dominant driver, but front-loading and concentration add meaningful adjustments at the margins.

**File:** `src/bilancio/dealer/priors.py` — add `scenario_informed_prior(κ, μ, c)` alongside existing `kappa_informed_prior(κ)`.
**File:** Scenario compilation — use the new function when `adaptive_prior=True`.

---

## WI-2: Trader Within-Run Adaptation

**Goal:** Trader behavior responds to observed conditions during the run.

### 2a. Risk aversion responds to observed stress

**Current:** `TraderProfile.risk_aversion = 0` (fixed for entire run).

**Problem:** A trader who has observed 30% of counterparties defaulting should become more cautious. Currently, the only response channel is through `p_default` updating, but the *behavioral* parameters (buy_premium, surplus threshold) stay constant.

**Change:** When `adaptive_risk_aversion=True`, the engine computes an effective risk aversion each day:
```
observed_default_rate = n_defaulted / n_total
base_ra = profile.risk_aversion
effective_ra = base_ra + (1 - base_ra) × min(1, observed_default_rate / 0.3)
```

At 30%+ default rate, effective_ra approaches 1.0 regardless of the base setting. This feeds into:
- `buy_risk_premium = 0.01 + 0.02 × effective_ra` (pickier buying)
- `buy_premium_multiplier = 1.0 + effective_ra` (higher hurdle)

**File:** `src/bilancio/engines/dealer_sync.py` or the trading phase — compute `effective_ra` daily, pass to trade decisions.
**Important:** The `TraderProfile` dataclass stays frozen. The engine computes `effective_ra` and passes it as a runtime override.

### 2b. Buy reserve fraction responds to stress

**Current:** `TraderProfile.buy_reserve_fraction = 1.0` (fixed).

**Problem:** Under crisis, agents should hoard more cash (higher reserve fraction = fewer eligible buyers). Under abundance, they should loosen.

**Change:** When `adaptive_reserves=True`:
```
stress_signal = system_default_rate / 0.3            # normalized, >1 means severe stress
effective_reserve = base_reserve + (1 - base_reserve) × min(1, stress_signal × 0.5)
```

At 30% default rate, reserves increase by 50% of the remaining headroom.

**File:** Trading phase — apply effective reserve fraction when deciding buy eligibility.

### 2c. EV incorporates term structure

**Current:** `EV = (1 - p_default) × face` — same p_default for all maturities.

**Problem:** A claim due tomorrow that hasn't defaulted yet has almost no residual risk. A claim due in 15 days has substantial risk. Using the system-wide default rate for both misprices both.

**Change:** When `adaptive_ev_term_structure=True`, the `EVValuer` uses maturity-dependent default probability.

**Important:** The RiskAssessor's `p_default` is a **lookback-window payment-failure frequency** — `(alpha + defaults) / (2*alpha + total)` over the last N days (risk_assessment.py:186-192). It is NOT a cumulative default probability over a specific time horizon. To convert this to a term-structure-aware price, we interpret it as a daily hazard rate proxy:

```
# Interpret p_default as a per-period (per-day) failure rate
# This is valid because the RiskAssessor counts payment failures per
# observation, and each observation corresponds to roughly one payment
# event per day per issuer.
h = p_default                                        # daily hazard rate proxy

# Survival probability to remaining maturity τ (in days)
p_survive(τ) = (1 - h)^τ                            # discrete compounding

# Term-adjusted EV
EV(τ) = p_survive(τ) × face
```

Using `(1-h)^τ` (discrete) instead of `exp(-h×τ)` (continuous) is more appropriate here because the underlying data is discrete daily observations, not a continuous process.

**Calibration note:** At p_default=0.15, τ=1: EV = 0.85×face (modest discount). At τ=10: EV = 0.197×face (steep discount). If this proves too aggressive, a dampening parameter `term_strength ∈ [0,1]` can blend: `h_effective = term_strength × p_default`. Default `term_strength=0.5` halves the hazard rate, making the term effect less extreme while still meaningful.

This means short-maturity claims get higher EV (more of the risk has been "survived") and long-maturity claims get lower EV.

**File:** `src/bilancio/decision/risk_assessment.py` (`EVValuer.expected_value`) — add `remaining_maturity` and `term_strength` parameters, compute term-adjusted EV when flag is on.

### 2d. Issuer-specific assessment on by default

**Current:** `RiskAssessmentParams.use_issuer_specific = False`.

**Problem:** All issuers get the same default probability. An agent with 10× cash and zero upcoming obligations is treated identically to one on the brink of default.

**Change:** When `adaptive_issuer_specific=True`, set `use_issuer_specific=True`. This is the existing mechanism — it already works, it's just off by default.

**File:** `src/bilancio/decision/risk_assessment.py` — `RiskAssessmentParams`, add `adaptive_issuer_specific: bool = False` that overrides `use_issuer_specific`.

---

## WI-3: VBT Pre-Run Calibration

**Goal:** VBT reference quotes (M, O) calibrate to scenario parameters before the run starts. This is the highest-leverage change — VBT quotes flow through to dealer pricing via Treynor.

### 3a. Term-structure-aware VBT mid pricing

**Current:** `M = ρ × (1 - p_default)` — identical for all buckets (short/mid/long). One mid price for the entire market.

**Problem:** This is the single largest pricing gap. A 1-day claim should be priced much closer to par than a 15-day claim. The VBT, as the Value-Based Trader, should be the entity that expresses this view.

**Change:** When `adaptive_term_structure=True`, compute per-bucket M:

**Hazard rate interpretation:** The RiskAssessor's `p_default` is a lookback-window payment-failure frequency (risk_assessment.py:186-192), not a horizon probability. We interpret it as a daily hazard rate proxy (see WI-2c for rationale):

```
# Daily hazard rate proxy from observed default frequency
h = p_default                                        # per-day failure rate

# Per-bucket midpoint τ (remaining days to maturity)
τ_short = 2      # midpoint of bucket [1, 3]
τ_mid   = 6      # midpoint of bucket [4, 8]
τ_long  = 12     # midpoint of bucket [9, ∞)

# Per-bucket survival and mid (discrete compounding)
p_survive(τ) = (1 - h)^τ
M(τ) = ρ × p_survive(τ)
```

**Dampening:** If term differentiation proves too aggressive (e.g., at p_default=0.15, long-bucket M drops to ρ×0.14 which may kill all long-maturity trading), apply the same `term_strength` parameter as WI-2c: `h_effective = term_strength × p_default`. Default `term_strength=0.5`.

**Numerical examples (with term_strength=0.5, ρ=0.90):**
| p_default | h_eff | M_short (τ=2) | M_mid (τ=6) | M_long (τ=12) |
|-----------|-------|---------------|-------------|----------------|
| 0.05 | 0.025 | 0.856 | 0.772 | 0.663 |
| 0.15 | 0.075 | 0.770 | 0.569 | 0.360 |
| 0.30 | 0.150 | 0.650 | 0.338 | 0.127 |

**Effect on dealer:**
- Short-bucket dealer gets a higher M → narrower spread → more liquid
- Long-bucket dealer gets a lower M → wider spread → less liquid
- Dealer doesn't need to "know" anything — VBT M flows through Treynor automatically

**Dependencies:** This change requires updating `_update_vbt_credit_mids()` in `dealer_sync.py` (line 331: `vbt.M = new_M`) to compute per-bucket M instead of a single value.

**Interaction with bucket transitions:** Currently `_move_ticket_to_new_bucket()` transfers tickets for free. With maturity-differentiated M, a ticket moving from long to mid gets a higher M, which looks like value creation. Two options:
- (a) Accept this — the value increase reflects genuine risk reduction as the claim approaches maturity. The VBT is correctly re-marking the claim.
- (b) Add a transition cost — the receiving desk pays the sending desk at the old M. This is more complex and probably unnecessary since both desks belong to the same firm.

**Recommendation:** Option (a). The free transition is correct — the claim genuinely became less risky by aging.

**File:** `src/bilancio/engines/dealer_sync.py` — modify `_update_vbt_credit_mids()`.
**File:** `src/bilancio/decision/valuers.py` — add `compute_mid_term_structure(p_default, initial_prior, remaining_tau)` to `CreditAdjustedVBTPricing`.
**File:** `src/bilancio/decision/profiles.py` — add `adaptive_term_structure: bool = False` to `VBTProfile`.

### 3b. Base spreads calibrate to scenario stress

**Current:** Base spreads are hardcoded per bucket: short=0.20, mid=0.30, long=0.40.

**Problem:** In a κ=0.3 scenario, these spreads may be too tight (VBT is underpricing risk). In a κ=3 scenario, they're too wide (VBT is overpricing risk, killing trading volume).

**Change:** When `adaptive_base_spreads=True`:
```
stress = max(0, 1 - κ) / (1 + κ)                    # same formula as corridor

O_short = 0.10 + 0.20 × stress                      # [0.10, 0.30]
O_mid   = 0.15 + 0.30 × stress                      # [0.15, 0.45]
O_long  = 0.20 + 0.40 × stress                      # [0.20, 0.60]
```

At κ=1 (balanced): O = {0.10, 0.15, 0.20} — tighter than current.
At κ=0 (maximum stress): O = {0.30, 0.45, 0.60} — wider than current.
At κ=3 (abundant): O = {0.10, 0.15, 0.20} — same as balanced (stress floor).

**File:** Scenario compilation — compute base spreads from κ when flag is on, pass to VBT initialization.
**File:** `src/bilancio/decision/profiles.py` — add `adaptive_base_spreads: bool = False` to `VBTProfile`.

### 3c. Activate spread_sensitivity

**Current:** `VBTProfile.spread_sensitivity = 0.0` — spreads never widen during the run.

**Change:** In the adaptive preset, set `spread_sensitivity = 0.6`. This is already implemented in `CreditAdjustedVBTPricing.compute_spread()` (valuers.py:155-160):
```
O = base_O + spread_sensitivity × p_default
```

This is not a code change — just a default value change in the adaptive preset.

### 3d. Activate flow_sensitivity

**Current:** `VBTProfile.flow_sensitivity = 0.0` — VBT provides unlimited liquidity at fixed ask.

**Change:** In the adaptive preset, set `flow_sensitivity = 0.5`. This is already implemented in `VBTState.recompute_quotes()` (models.py:247-258):
```
net_outflow = cumulative_outflow - cumulative_inflow
if net_outflow > 0:
    outflow_ratio = net_outflow / initial_face
    ask_premium = flow_sensitivity × outflow_ratio
    A = A + ask_premium
```

This is not a code change — just a default value change in the adaptive preset.

### 3e. Activate forward_weight

**Current:** `VBTProfile.forward_weight = 0.0` — VBT is purely backward-looking.

**Change:** In the adaptive preset, set `forward_weight = 0.3`. This is already implemented in `_update_vbt_credit_mids()` (dealer_sync.py:286-294):
```
p_forward = estimate_forward_stress(system, current_day, stress_horizon)
p_blend = (1 - forward_weight) × p_default + forward_weight × p_forward
```

This is not a code change — just a default value change in the adaptive preset. Also ensure `stress_horizon` scales with `maturity_days` (see 3f).

### 3f. Stress horizon scales with maturity_days

**Current:** `VBTProfile.stress_horizon = 5` (fixed).

**Problem:** In a 20-day scenario, looking only 5 days ahead misses upcoming payment clusters. In a 3-day scenario, 5 days overshoots.

**Change:** When `adaptive_stress_horizon=True`:
```
stress_horizon = max(3, min(maturity_days, 15))
```

**File:** `src/bilancio/decision/profiles.py` — add `adaptive_stress_horizon: bool = False` to `VBTProfile`.

---

## WI-4: VBT Within-Run Adaptation

**Goal:** VBT quotes respond intelligently to observed conditions during the run.

### 4a. Per-bucket default tracking

**Current:** VBT uses one system-wide `p_default` for all buckets (`_update_vbt_credit_mids`, dealer_sync.py:264).

**Problem:** Short-maturity claims that are close to settlement have less residual risk than long-maturity claims. If the system default rate is 15%, a claim due tomorrow has already survived most of its risk window.

**Change:** When `adaptive_per_bucket_tracking=True`, track defaults per maturity bucket.

**Smoothing and fallback:** Per-bucket counts are sparse early in the run (and always sparse for long-maturity claims). Use Laplace smoothing with fallback to the system-wide rate:

```
# Track: payment outcomes per bucket (accumulated over lookback_window)
# Each settlement event is tagged with the bucket the claim was in when it settled.

MIN_OBSERVATIONS = 5                                 # minimum sample before trusting bucket rate

For each bucket b ∈ {short, mid, long}:
    defaults_b = count of defaults in bucket b within lookback window
    total_b    = count of all settlements in bucket b within lookback window

    if total_b >= MIN_OBSERVATIONS:
        # Laplace-smoothed per-bucket rate (same smoothing as RiskAssessor)
        alpha = smoothing_alpha                      # default 1.0
        p_bucket_b = (alpha + defaults_b) / (2 × alpha + total_b)

        # Blend with system rate — per-bucket data dominates as sample grows
        w_bucket = min(1, total_b / (3 × MIN_OBSERVATIONS))
        p_default_b = w_bucket × p_bucket_b + (1 - w_bucket) × p_default_system
    else:
        # Insufficient data: fall back to system-wide rate
        p_default_b = p_default_system
```

**Interaction with WI-3a:** When both per-bucket tracking AND term-structure M are active, use per-bucket p_default in the hazard-rate formula:
```
h_b = term_strength × p_default_b
M(b) = ρ × (1 - h_b)^τ_b
```

This gives the best of both: empirically grounded per-bucket rates with hazard-based maturity discounting.

**File:** `src/bilancio/engines/dealer_sync.py` — extend `_update_vbt_credit_mids()` to maintain per-bucket settlement counters and compute blended rates.

**File:** `src/bilancio/engines/dealer_sync.py` — extend `_update_vbt_credit_mids()` to track per-bucket default/survival counts from payment history.

### 4b. Per-issuer VBT pricing

**Current:** Even when `issuer_specific_pricing=True` is configured, the VBT mid `M` is still computed from the system-wide default rate (dealer_sync.py:331). Per-issuer default probabilities only affect dealer spread adjustments, not the VBT reference price.

**Change:** When `adaptive_issuer_pricing=True`, VBT quotes become issuer-aware:
```
# For each ticket in VBT inventory, compute issuer-specific M
M_issuer = ρ × (1 - p_default_issuer)

# VBT still posts a single bid/ask per bucket, but uses the
# inventory-weighted average issuer quality to set M
avg_p = weighted_average(p_default_issuer for ticket in inventory)
M_bucket = ρ × (1 - avg_p)
```

**Alternative approach:** Instead of averaging, the VBT could post different quotes for different issuers. This is more complex (the dealer kernel currently assumes one VBT quote per bucket) and may not be worth the implementation cost in this phase.

**Recommendation:** Start with inventory-weighted average. Revisit per-issuer quoting in a future plan if needed.

**File:** `src/bilancio/engines/dealer_sync.py` — modify `_update_vbt_credit_mids()`.

### 4c. Convex spread widening

**Current:** `O = base_O + spread_sensitivity × p_default` — linear in p_default.

**Problem:** Real outside markets widen spreads nonlinearly as stress increases. At 5% default rate, spreads should be slightly wider. At 30% default rate, spreads should be dramatically wider (liquidity dries up).

**Change:** When `adaptive_convex_spreads=True`, use a convex formula:
```
O = base_O + spread_sensitivity × p_default^2 / (1 - p_default)
```

At p=0.05: widening = 0.6 × 0.0025/0.95 = 0.0016 (minimal)
At p=0.15: widening = 0.6 × 0.0225/0.85 = 0.0159 (moderate)
At p=0.30: widening = 0.6 × 0.09/0.70 = 0.0771 (significant)
At p=0.50: widening = 0.6 × 0.25/0.50 = 0.30 (dramatic)

**File:** `src/bilancio/decision/valuers.py` — add convex variant to `compute_spread()`.
**File:** `src/bilancio/decision/profiles.py` — add `adaptive_convex_spreads: bool = False` to `VBTProfile`.

---

## WI-5: Central Bank Adaptation

**Goal:** CB corridor rates calibrate to scenario parameters and respond to observed conditions. This is the primary intelligence for the banking system — improvements here flow through to all banks via Treynor.

### 5a. Corridor incorporates μ and c

**Current:** Corridor mid and width depend only on κ:
```
stress = max(0, 1-κ) / (1+κ)
corridor_mid = r_base + r_stress × stress
corridor_width = omega_base + omega_stress × stress
```

**Problem:** μ=0 (front-loaded stress) means banks face early deposit drains — corridor should start wider. Low c (concentrated debt) means systemic risk is higher — corridor should start wider.

**Change:** When `adaptive_corridor=True`:
```
stress_kappa = max(0, 1-κ) / (1+κ)
stress_mu    = (1 - μ)^2                              # front-loading [0, 1]
stress_c     = 1 / (1 + c)                            # concentration [0, 1)

combined_stress = stress_kappa + 0.3 × stress_mu + 0.2 × stress_c

corridor_mid   = r_base + r_stress × combined_stress
corridor_width = omega_base + omega_stress × combined_stress
```

The weights (0.3 for μ, 0.2 for c) make these secondary to κ but meaningful.

**File:** `src/bilancio/decision/profiles.py` — add `adaptive_corridor: bool = False` to `BankProfile`.
**File:** `BankProfile.corridor_mid()` and `corridor_width()` — extend to accept μ, c when flag is on.

### 5b. Beta parameters scale with κ

**Current:** `CentralBank.beta_mid = 0.50`, `beta_width = 0.30` — constant regardless of scenario.

**Problem:** In a κ=0.3 scenario, defaults are expected — the CB should be less reactive to "surprises" (most of them are anticipated). In a κ=3 scenario, any default is genuinely surprising and should trigger a larger corridor adjustment.

**Change:** When `adaptive_betas=True`:
```
expected_default_rate = kappa_informed_prior(κ)       # what the CB expects

# Scale betas inversely with expected rate — more reactive when defaults are rare
beta_scale = 1 / (1 + 5 × expected_default_rate)     # [0.5, 0.95] approx

beta_mid   = 0.50 × beta_scale
beta_width = 0.30 × beta_scale
```

At κ=0.3 (expected ~15%): beta_mid ≈ 0.29, beta_width ≈ 0.17 — CB is calmer.
At κ=3 (expected ~5%): beta_mid ≈ 0.40, beta_width ≈ 0.24 — CB is more reactive.

**File:** `src/bilancio/domain/agents/central_bank.py` — add `adaptive_betas: bool = False`. Apply scaling in `compute_corridor()`.

### 5c. Rate escalation scales with system size

**Current:** `rate_escalation_slope = 0.05` (fixed). Auto-activated when banking is enabled.

**Problem:** A 200-agent system generates much more aggregate CB demand than a 10-agent system. The escalation slope should be flatter for larger systems (each individual loan is a smaller fraction of total) and steeper for smaller ones.

**Change:** When `adaptive_escalation=True`:
```
base_slope = 0.05
escalation_slope = base_slope × (50 / n_agents)       # normalized to n=50 reference

# Clamp to prevent extreme values
escalation_slope = max(0.01, min(0.20, escalation_slope))
```

At n=10: slope = 0.05 × 50/10 = 0.25 → clamped to 0.20 (steep — each loan matters a lot)
At n=50: slope = 0.05 × 50/50 = 0.05 (reference)
At n=200: slope = 0.05 × 50/200 = 0.0125 (gentle — large diversified system)

**File:** Scenario compilation — compute escalation_slope from n_agents when flag is on.

### 5d. CB observes bank-level indicators

**Current:** `compute_corridor()` adjusts only based on realized defaults (`n_defaulted / n_total`).

**Problem:** The CB could observe early warning signals from banks: falling reserve ratios, rising CB utilization, increasing lending volumes. These signal stress before defaults materialize.

**Change:** When `adaptive_early_warning=True`, add bank-stress signal to corridor computation:
```
# Aggregate bank stress signal
avg_reserve_ratio = mean(reserves / deposits for each bank)
target_ratio = bank_profile.reserve_target_ratio
bank_stress = max(0, 1 - avg_reserve_ratio / target_ratio)   # [0, 1]

# Blend with default surprise
surprise = max(0, p_realized - kappa_prior)
combined_signal = surprise + 0.3 × bank_stress

# Apply to corridor
mid = base_mid + beta_mid × combined_signal
width = base_width + beta_width × combined_signal
```

**File:** `src/bilancio/domain/agents/central_bank.py` — extend `compute_corridor()` to accept bank stress signal.
**File:** `src/bilancio/engines/banking_subsystem.py` — pass aggregate bank stress in `update_cb_corridor()`.

---

## WI-6: NBFI Lender Pre-Run Calibration

**Goal:** NBFI parameters calibrate to scenario parameters and market rates.

### 6a. Activate all Plan 046 features in adaptive preset

**Current:** All Plan 046 features are off by default: `maturity_matching=False`, `ranking_mode="profit"`, `coverage_mode="gate"`, `preventive_lending=False`.

**Change:** In the adaptive preset:
```python
LenderProfile(
    maturity_matching=True,
    ranking_mode="blended",
    cascade_weight=Decimal("0.5"),
    coverage_mode="graduated",
    coverage_penalty_scale=Decimal("0.10"),
    preventive_lending=True,
    prevention_threshold=Decimal("0.3"),
    max_loans_per_borrower_per_day=2,
)
```

This is not a code change — just the adaptive preset wiring.

### 6b. Risk aversion calibrates to κ

**Current:** `LenderProfile.risk_aversion = 0.3` (fixed).

**Change:** When `adaptive_risk_aversion=True`:
```
stress = max(0, 1 - κ) / (1 + κ)                    # [0, 1]
risk_aversion = 0.2 + 0.5 × stress                  # [0.2, 0.7]
```

At κ=0.3: risk_aversion ≈ 0.53 (cautious)
At κ=1: risk_aversion = 0.20 (aggressive)
At κ=3: risk_aversion = 0.20 (same, floor)

**File:** `src/bilancio/decision/profiles.py` — add `adaptive_risk_aversion: bool = False` to `LenderProfile`.

### 6c. Profit target anchors to market rates

**Current:** `LenderProfile.profit_target = 0.05` (fixed, regardless of corridor rates).

**Problem:** If the bank corridor ceiling is at 0.08, the NBFI charging 0.05 is undercutting banks, which may not be realistic or intended. If the corridor ceiling is at 0.02, the NBFI at 0.05 is so expensive nobody will borrow.

**Change:** When `adaptive_profit_target=True`:
```
# Anchor to CB corridor
r_ceiling = bank_profile.r_ceiling(κ)
profit_target = r_ceiling × 1.2                      # NBFI charges 20% premium over CB ceiling

# Clamp
profit_target = max(0.02, min(0.15, profit_target))
```

This positions the NBFI as a more expensive but more accessible alternative to bank lending (since NBFI has no reserve requirement constraint).

**File:** Scenario compilation — compute profit_target from corridor when flag is on.

### 6d. Max loan maturity scales with maturity_days

**Current:** `LenderProfile.max_loan_maturity = 10` (fixed).

**Problem:** In a 5-day scenario, 10-day loans extend beyond the scenario. In a 30-day scenario, 10-day loans may be too short to help.

**Change:** When `adaptive_loan_maturity=True`:
```
max_loan_maturity = max(2, min(maturity_days - 1, 15))
```

**File:** `src/bilancio/decision/profiles.py` — add `adaptive_loan_maturity: bool = False` to `LenderProfile`.

---

## WI-7: NBFI Lender Within-Run Adaptation

**Goal:** NBFI responds to portfolio performance and observed conditions during the run.

### 7a. Rate adaptation to portfolio performance

**Current:** Base rate (`profit_target`) is fixed for the entire run.

**Problem:** If 30% of the NBFI's borrowers have defaulted, it should raise rates to compensate for losses. If all borrowers are performing, it should maintain or lower rates to attract more business.

**Change:** When `adaptive_rates=True`, the engine computes a daily rate multiplier:
```
portfolio_default_rate = n_borrower_defaults / max(n_loans_issued, 1)
expected_default_rate = base_default_estimate         # = 1/(1+κ) from LenderProfile

# Rate multiplier: >1 when losses exceed expectations, =1 when on target
loss_ratio = portfolio_default_rate / max(expected_default_rate, 0.01)
rate_multiplier = max(1.0, min(3.0, loss_ratio))

effective_profit_target = profit_target × rate_multiplier
```

The formula is a single multiplication — `rate_multiplier` directly scales `profit_target`. When portfolio losses match expectations (`loss_ratio ≈ 1`), rates stay at base. When losses are 2× expected, rates double. Clamped at 3× to prevent runaway pricing.

**Numerical examples (profit_target=0.05, κ=1 → expected_default=0.50):**
| Portfolio default rate | loss_ratio | multiplier | effective rate |
|----------------------|------------|------------|----------------|
| 0.10 | 0.20 | 1.0 (floor) | 0.05 |
| 0.50 | 1.00 | 1.0 | 0.05 |
| 1.00 | 2.00 | 2.0 | 0.10 |
| 1.50+ | 3.00+ | 3.0 (cap) | 0.15 |

**File:** `src/bilancio/engines/lending.py` — add daily rate adjustment logic in `run_lending_phase()`.

### 7b. Capital conservation mode

**Current:** NBFI lends until exposure limits bind, regardless of remaining capital adequacy.

**Problem:** As the NBFI's capital erodes from defaults, it should become more selective — tightening exposure limits rather than lending until it's empty.

**Change:** When `adaptive_capital_conservation=True`:
```
capital_utilization = total_exposure / initial_capital
conservation_factor = max(0.2, 1 - capital_utilization)

# Scale down effective limits
effective_max_total_exposure = max_total_exposure × conservation_factor
effective_max_single_exposure = max_single_exposure × conservation_factor
```

At 80% utilization: limits shrink to 20% of original (highly conservative).
At 20% utilization: limits at 80% of original (still lending freely).

**File:** `src/bilancio/engines/lending.py` — apply conservation scaling in `_execute_ranked_opportunities()`.

### 7c. Dynamic prevention threshold

**Current:** `prevention_threshold = 0.3` (fixed). NBFI preventively lends when an issuer's p_default exceeds this threshold.

**Problem:** During a cascade, 0.3 may be too high — by the time an issuer hits 30%, the damage has already started. In calm periods, 0.3 may be too low — many issuers temporarily show elevated risk.

**Change:** When `adaptive_prevention=True`:
```
# Track system default acceleration
if current_day > 1:
    delta_defaults = n_defaults_today - n_defaults_yesterday
    acceleration = delta_defaults / max(n_total, 1)
else:
    acceleration = 0

# Lower threshold when cascading, raise when stable
effective_threshold = prevention_threshold - 0.5 × acceleration
effective_threshold = max(0.10, min(0.50, effective_threshold))
```

During a cascade (acceleration > 0), the threshold drops, making the NBFI more proactive. During stable periods, the threshold rises, making it more selective.

**File:** `src/bilancio/engines/lending.py` — apply in `_collect_preventive_opportunities()`.

---

## WI-8: Dealer and Bank Structural Improvements

These are **not** adaptive intelligence changes — they are structural/mechanical fixes that improve the Treynor pipe.

### 8a. Deprecate alpha_vbt / alpha_trader (compatibility migration)

**Current:** `alpha_vbt` and `alpha_trader` fields are stored in config models but never used for computation. A comment at dealer_integration.py:487 says "replaces alpha blending." However, these fields are wired through the full stack:

| Layer | Files | References |
|-------|-------|------------|
| Config model | `config/models.py:731-740` | Field definitions in `BalancedDealerConfig` |
| Config apply | `config/apply.py:709-710, 1016-1017` | Passed to `initialize_dealer_subsystem()` |
| Dealer integration | `dealer_integration.py:167-168, 448-449, 519-520` | `DealerSubsystemState` fields + init |
| Experiment runners | `ring.py:299-300, 363-364, 849-850, 1262-1263` | Function params + manifest output |
| Balanced comparison | `balanced_comparison.py:87-88, 572-577, 782-783, 1118-1119` + 8 more arm call sites, CSV output | Config + pass-through + output columns |
| NBFI comparison | `nbfi_comparison.py:179-180, 375-376, 420-421` | Config + pass-through |
| Bank comparison | `bank_comparison.py:460-461` | Hardcoded as `Decimal("0")` |
| CLI | `ui/cli/sweep.py:776-780, 1019-1020, 1171-1172, 1199-1200` | Click options + display |
| Tests | 3 test files, ~7 lines | Assertions on defaults, serialization |

**This is a compatibility migration, not just cleanup.** Removing these fields touches ~36 source lines and ~7 test lines across 10+ files, and may break existing YAML configs or saved experiment manifests that reference these fields.

**Change (two-phase):**

**Phase A (this plan):** Deprecate — keep the fields but add deprecation warnings:
- Add `# DEPRECATED: unused, will be removed in v0.X` comments to all field definitions
- Log a `DeprecationWarning` in `config/apply.py` if either value is non-zero
- Do NOT remove from CLI options yet (would break existing scripts)

**Phase B (future plan):** Remove — after one release cycle:
- Remove fields from config models, CLI options, experiment runners
- Remove from CSV output columns (add migration note)
- Update tests

**Files:** `config/models.py`, `config/apply.py` (Phase A only).

### 8b. Volume-weighted cash pooling (optional)

**Current:** `_pool_desk_cash()` splits cash equally across desks.

**Change:** When `volume_weighted_pooling=True`:
```
# Weight by recent trading volume (face value traded in last 3 days)
weights = {desk_id: max(1, desk.recent_volume) for desk_id in desks}
total_weight = sum(weights.values())

for desk_id, desk in desks.items():
    desk.cash = total_cash × weights[desk_id] / total_weight
```

This ensures active desks get more capital. The `max(1, ...)` prevents a desk with zero volume from being starved completely.

**File:** `src/bilancio/engines/dealer_sync.py` — modify `_pool_desk_cash()`.
**Note:** This requires tracking per-desk trading volume, which isn't currently stored. Need to add a `recent_volume` field to `DealerState`/`VBTState`.

### 8c. "Fool me once" rehabilitation (Bank)

**Current:** Borrowers who defaulted on a bank loan are blacklisted permanently.

**Change:** When `bank_rehabilitation=True`:
```
# Check if borrower has been performing since default
days_since_default = current_day - default_day
if days_since_default > maturity_days:
    # Probationary: allow borrowing but at higher rate
    rate = base_rate + credit_risk_loading × 0.5     # 50% surcharge
```

**File:** `src/bilancio/engines/bank_lending.py` — modify the blacklist check.

---

## WI-9: Adaptive Presets and Experimental Design

### 9a. Two-axis preset system

All 23 adaptive flags fall into exactly one of two categories:

| Category | When it acts | What it does |
|----------|-------------|-------------|
| **Pre-run calibration** | Before day 0 | Adjusts initial parameters to scenario (κ, μ, c, maturity_days, n_agents) |
| **Within-run adaptation** | Each day during the run | Adjusts behavior in response to observed conditions (defaults, stress, portfolio performance) |

This gives a **2×2 experimental design** with four presets:

| Preset | Pre-run | Within-run | CLI flag | What it tests |
|--------|---------|-----------|----------|---------------|
| `static` | OFF | OFF | `--adapt=static` | Current behavior — all defaults, no adjustment. Baseline. |
| `calibrated` | **ON** | OFF | `--adapt=calibrated` | Parameters tuned to scenario, but frozen during run. Tests whether better initial conditions alone improve outcomes. |
| `responsive` | OFF | **ON** | `--adapt=responsive` | Default initial parameters, but agents react to what happens. Tests whether in-run learning alone helps. |
| `full` | **ON** | **ON** | `--adapt=full` | Everything on. Best-effort adaptive behavior. |

**The key experimental question:** Is the benefit from adaptation primarily in getting the initial setup right (calibrated), in reacting to events (responsive), or in both? Running all four presets across a κ sweep answers this.

### 9b. Flag classification

Each flag belongs to exactly one category:

**Pre-run calibration flags** (set once before day 0, frozen during run):

| # | Flag | Profile | What it calibrates to |
|---|------|---------|----------------------|
| 1 | `adaptive_planning_horizon` | TraderProfile | maturity_days |
| 2 | `adaptive_lookback` | RiskAssessmentParams | maturity_days |
| 3 | `adaptive_prior` | RiskAssessmentParams | κ, μ, c |
| 7 | `adaptive_issuer_specific` | RiskAssessmentParams | (activates existing mechanism) |
| 8 | `adaptive_term_structure` | VBTProfile | hazard rate from p_default + τ |
| 9 | `adaptive_base_spreads` | VBTProfile | κ |
| 11 | `adaptive_stress_horizon` | VBTProfile | maturity_days |
| 14 | `adaptive_corridor` | BankProfile | κ, μ, c |
| 15 | `adaptive_betas` | CentralBank | κ |
| 16 | `adaptive_escalation` | CentralBank | n_agents |
| 18 | `adaptive_risk_aversion` | LenderProfile | κ |
| 19 | `adaptive_profit_target` | LenderProfile | CB corridor rates |
| 20 | `adaptive_loan_maturity` | LenderProfile | maturity_days |

**Within-run adaptation flags** (active each day, respond to observed state):

| # | Flag | Profile | What it responds to |
|---|------|---------|---------------------|
| 4 | `adaptive_risk_aversion` | TraderProfile | observed default rate |
| 5 | `adaptive_reserves` | TraderProfile | observed default rate |
| 6 | `adaptive_ev_term_structure` | TraderProfile | remaining τ + observed p_default |
| 10 | `adaptive_convex_spreads` | VBTProfile | observed p_default (nonlinear) |
| 12 | `adaptive_per_bucket_tracking` | VBTProfile | per-bucket settlement outcomes |
| 13 | `adaptive_issuer_pricing` | VBTProfile | per-issuer default rates |
| 17 | `adaptive_early_warning` | CentralBank | bank reserve ratios |
| 21 | `adaptive_rates` | LenderProfile | portfolio default rate |
| 22 | `adaptive_capital_conservation` | LenderProfile | capital utilization |
| 23 | `adaptive_prevention` | LenderProfile | default acceleration |

**Plus existing value-parameters activated by all adaptive presets** (these are not boolean flags but value changes in the adaptive presets):
- `spread_sensitivity = 0.6` (VBTProfile, currently 0.0) — within-run
- `flow_sensitivity = 0.5` (VBTProfile, currently 0.0) — within-run
- `forward_weight = 0.3` (VBTProfile, currently 0.0) — within-run
- `use_issuer_specific = True` (RiskAssessmentParams, currently False) — pre-run
- All Plan 046 features ON (LenderProfile) — pre-run

**Note:** `adaptive_risk_aversion` appears in both TraderProfile (within-run, #4) and LenderProfile (pre-run, #18). These are different flags on different profiles — the Trader's risk aversion adapts daily to observed defaults, while the Lender's risk aversion is calibrated once to κ before the run.

### 9c. Preset implementation

```python
from enum import Enum

class AdaptPreset(str, Enum):
    STATIC = "static"           # both OFF — current behavior
    CALIBRATED = "calibrated"   # pre-run ON, within-run OFF
    RESPONSIVE = "responsive"   # pre-run OFF, within-run ON
    FULL = "full"               # both ON

def build_adaptive_profiles(
    preset: AdaptPreset,
    kappa: Decimal,
    mu: Decimal,
    c: Decimal,
    maturity_days: int,
    n_agents: int,
) -> dict:
    """Return profiles configured for the given adaptation preset."""

    pre_run  = preset in (AdaptPreset.CALIBRATED, AdaptPreset.FULL)
    in_run   = preset in (AdaptPreset.RESPONSIVE, AdaptPreset.FULL)

    return {
        "trader_profile": TraderProfile(
            # Pre-run calibration
            planning_horizon=(
                max(3, min(20, maturity_days)) if pre_run else 10
            ),
            adaptive_planning_horizon=pre_run,
            adaptive_issuer_specific=pre_run,
            # Within-run adaptation
            adaptive_risk_aversion=in_run,
            adaptive_reserves=in_run,
            adaptive_ev_term_structure=in_run,
            # Values activated by any adaptive preset
            risk_aversion=Decimal("0.3"),
            buy_reserve_fraction=Decimal("0.5"),
            trading_motive="liquidity_then_earning",
        ),
        "risk_params": RiskAssessmentParams(
            # Pre-run
            lookback_window=(
                max(3, min(maturity_days, 15)) if pre_run else 5
            ),
            initial_prior=(
                scenario_informed_prior(kappa, mu, c) if pre_run
                else kappa_informed_prior(kappa)
            ),
            use_issuer_specific=pre_run,
            adaptive_lookback=pre_run,
        ),
        "vbt_profile": VBTProfile(
            # Pre-run
            adaptive_term_structure=pre_run,
            adaptive_base_spreads=pre_run,
            adaptive_stress_horizon=pre_run,
            stress_horizon=(
                max(3, min(maturity_days, 15)) if pre_run else 5
            ),
            # Within-run
            adaptive_convex_spreads=in_run,
            adaptive_per_bucket_tracking=in_run,
            adaptive_issuer_pricing=in_run,
            # Values activated by any adaptive preset
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.6"),
            flow_sensitivity=Decimal("0.5"),
            forward_weight=Decimal("0.3"),
        ),
        "bank_profile": BankProfile(
            # Pre-run
            adaptive_corridor=pre_run,
        ),
        "lender_profile": LenderProfile(
            # Pre-run
            adaptive_risk_aversion=pre_run,
            adaptive_profit_target=pre_run,
            adaptive_loan_maturity=pre_run,
            # Within-run
            adaptive_rates=in_run,
            adaptive_capital_conservation=in_run,
            adaptive_prevention=in_run,
            # Plan 046 features (activated by any adaptive preset)
            maturity_matching=True,
            ranking_mode="blended",
            coverage_mode="graduated",
            preventive_lending=True,
            max_loans_per_borrower_per_day=2,
        ),
        "cb_params": {
            # Pre-run
            "adaptive_betas": pre_run,
            "adaptive_escalation": pre_run,
            # Within-run
            "adaptive_early_warning": in_run,
        },
    }
```

**File:** New file `src/bilancio/decision/adaptive.py`.

### 9d. CLI integration

```bash
# Four presets via --adapt flag
uv run bilancio sweep balanced --adapt=static   ...   # baseline (current behavior)
uv run bilancio sweep balanced --adapt=calibrated ...  # pre-run only
uv run bilancio sweep balanced --adapt=responsive ...  # within-run only
uv run bilancio sweep balanced --adapt=full ...        # both on

# Default: --adapt=static (backward compatible)
# Can also pass individual flags to override:
uv run bilancio sweep balanced --adapt=calibrated --adaptive-convex-spreads ...
```

### 9e. Pre-flight display

Extend the sweep pre-flight summary (CLAUDE.md §Step 6) with an adaptive features section:

```
─── G. Adaptive Features ─────────────────────────────
  Preset: full (pre-run ON, within-run ON)

  Trader:                                         Category
    ✓ planning_horizon=15 (maturity_days=15)      [PRE]
    ✓ lookback_window=10 (maturity_days)           [PRE]
    ✓ initial_prior=0.12 (κ=0.5, μ=0.3, c=1)     [PRE]
    ✓ issuer-specific assessment                   [PRE]
    ✓ risk_aversion responds to defaults           [RUN]
    ✓ buy_reserve_fraction responds to stress      [RUN]
    ✓ EV term structure (maturity-dependent)       [RUN]

  VBT:
    ✓ term-structure M (per-bucket pricing)        [PRE]
    ✓ base spreads: O={0.15, 0.23, 0.30} (κ)      [PRE]
    ✓ stress_horizon=10 (maturity_days)            [PRE]
    ✓ spread_sensitivity=0.6                       [RUN]
    ✓ flow_sensitivity=0.5                         [RUN]
    ✓ forward_weight=0.3                           [RUN]
    ✓ convex spread widening                       [RUN]
    ✓ per-bucket default tracking                  [RUN]
    ✓ issuer-weighted mid pricing                  [RUN]

  CB:
    ✓ corridor: μ,c-aware (combined_stress=0.35)   [PRE]
    ✓ adaptive betas (κ-scaled)                    [PRE]
    ✓ escalation slope=0.04 (n-scaled)             [PRE]
    ✓ bank stress early warning                    [RUN]

  NBFI:
    ✓ all Plan 046 features ON                     [PRE]
    ✓ risk_aversion=0.35 (κ-calibrated)            [PRE]
    ✓ profit_target=0.036 (corridor-anchored)      [PRE]
    ✓ max_loan_maturity=9 (maturity_days)          [PRE]
    ✓ adaptive rates (portfolio-responsive)        [RUN]
    ✓ capital conservation                         [RUN]
    ✓ dynamic prevention threshold                 [RUN]

  [PRE] = set before run from scenario params
  [RUN] = adapts each day to observed conditions
```

For intermediate presets (`calibrated` or `responsive`), only the relevant category shows ✓ marks; the other shows `–` (off).

**File:** Pre-flight logic in CLAUDE.md or sweep runner.

---

## Implementation Phases

### Phase 1: VBT Intelligence (Highest Leverage)

| Work Item | Description | New Code | Tests |
|-----------|-------------|----------|-------|
| WI-3a | Term-structure VBT mid | ~50 lines | ~8 |
| WI-3b | κ-calibrated base spreads | ~20 lines | ~4 |
| WI-3c | Activate spread_sensitivity | Preset only | ~2 |
| WI-3d | Activate flow_sensitivity | Preset only | ~2 |
| WI-3e | Activate forward_weight | Preset only | ~2 |
| WI-3f | Stress horizon scaling | ~5 lines | ~2 |
| WI-4c | Convex spread widening | ~15 lines | ~4 |

**Why first:** VBT quotes flow through to dealer via Treynor. Fixing VBT pricing fixes the entire secondary market without touching dealer code.

### Phase 2: Trader Adaptation + CB Intelligence

| Work Item | Description | New Code | Tests |
|-----------|-------------|----------|-------|
| WI-1a-c | Trader pre-run calibration | ~30 lines | ~6 |
| WI-2a-d | Trader within-run adaptation | ~60 lines | ~10 |
| WI-5a-d | CB corridor improvements | ~50 lines | ~10 |

**Why second:** Trader and CB are the demand side and the banking backstop. With VBT fixed (Phase 1), these two complete the core adaptive loop.

### Phase 3: NBFI + Structural

| Work Item | Description | New Code | Tests |
|-----------|-------------|----------|-------|
| WI-6a-d | NBFI pre-run calibration | ~30 lines | ~6 |
| WI-7a-c | NBFI within-run adaptation | ~40 lines | ~8 |
| WI-8a-c | Dead code cleanup, structural | ~30 lines | ~4 |
| WI-9a-b | Adaptive preset + pre-flight | ~80 lines | ~6 |

### Phase 4: Validation

| Work Item | Description |
|-----------|-------------|
| 2×2 sweep | Run all four presets across κ ∈ {0.25, 0.5, 1, 2, 4} |
| Regression | Verify `--adapt=static` produces bit-identical results to current code |
| Sensitivity | Verify each flag individually changes behavior in expected direction |
| Decomposition | Compare calibrated vs responsive to determine which axis matters more |

---

## Verification Strategy

### Per-feature tests

1. **Unit test** — flag off produces old behavior; flag on produces different (correct) behavior
2. **Monotonicity test** — increasing stress produces expected directional change (e.g., higher κ → lower prior)
3. **Regression test** — `--adapt=static` produces bit-identical results to current code (no behavioral change when all flags are off)

### 2×2 experimental design

The four presets form a factorial experiment that decomposes the total adaptive effect:

```
                    Within-run OFF          Within-run ON
                ┌───────────────────┬───────────────────┐
Pre-run OFF     │  static           │  responsive       │
                │  (baseline)       │  (learning only)  │
                ├───────────────────┼───────────────────┤
Pre-run ON      │  calibrated       │  full             │
                │  (setup only)     │  (both)           │
                └───────────────────┴───────────────────┘
```

**Run all four across κ ∈ {0.25, 0.5, 1, 2, 4}** with arms = passive + active (dealer comparison). This produces:

- **Total adaptive effect** = `δ(static) - δ(full)` — how much adaptation helps overall
- **Calibration effect** = `δ(static) - δ(calibrated)` — how much getting the initial setup right helps
- **Learning effect** = `δ(static) - δ(responsive)` — how much in-run adaptation alone helps
- **Interaction effect** = `(calibrated effect) + (learning effect) - (total effect)` — whether pre-run and within-run are complementary or redundant

**Expected outcomes:**
- At low κ (stressed): both calibration and learning should help (large total effect)
- At high κ (abundant): calibration matters less (system is already stable), learning may not activate (few defaults to react to)
- If `calibration effect >> learning effect`: the main issue is bad initial parameters, not lack of adaptation
- If `learning effect >> calibration effect`: the main issue is rigidity during the run, not initial miscalibration

---

## Flag Inventory

| # | Flag | Profile | Category | `static` | `calibrated` | `responsive` | `full` |
|---|------|---------|----------|----------|-------------|-------------|--------|
| 1 | `adaptive_planning_horizon` | TraderProfile | PRE | – | ✓ | – | ✓ |
| 2 | `adaptive_lookback` | RiskAssessmentParams | PRE | – | ✓ | – | ✓ |
| 3 | `adaptive_prior` | RiskAssessmentParams | PRE | – | ✓ | – | ✓ |
| 4 | `adaptive_risk_aversion` | TraderProfile | RUN | – | – | ✓ | ✓ |
| 5 | `adaptive_reserves` | TraderProfile | RUN | – | – | ✓ | ✓ |
| 6 | `adaptive_ev_term_structure` | TraderProfile | RUN | – | – | ✓ | ✓ |
| 7 | `adaptive_issuer_specific` | RiskAssessmentParams | PRE | – | ✓ | – | ✓ |
| 8 | `adaptive_term_structure` | VBTProfile | PRE | – | ✓ | – | ✓ |
| 9 | `adaptive_base_spreads` | VBTProfile | PRE | – | ✓ | – | ✓ |
| 10 | `adaptive_convex_spreads` | VBTProfile | RUN | – | – | ✓ | ✓ |
| 11 | `adaptive_stress_horizon` | VBTProfile | PRE | – | ✓ | – | ✓ |
| 12 | `adaptive_per_bucket_tracking` | VBTProfile | RUN | – | – | ✓ | ✓ |
| 13 | `adaptive_issuer_pricing` | VBTProfile | RUN | – | – | ✓ | ✓ |
| 14 | `adaptive_corridor` | BankProfile | PRE | – | ✓ | – | ✓ |
| 15 | `adaptive_betas` | CentralBank | PRE | – | ✓ | – | ✓ |
| 16 | `adaptive_escalation` | CentralBank | PRE | – | ✓ | – | ✓ |
| 17 | `adaptive_early_warning` | CentralBank | RUN | – | – | ✓ | ✓ |
| 18 | `adaptive_risk_aversion` | LenderProfile | PRE | – | ✓ | – | ✓ |
| 19 | `adaptive_profit_target` | LenderProfile | PRE | – | ✓ | – | ✓ |
| 20 | `adaptive_loan_maturity` | LenderProfile | PRE | – | ✓ | – | ✓ |
| 21 | `adaptive_rates` | LenderProfile | RUN | – | – | ✓ | ✓ |
| 22 | `adaptive_capital_conservation` | LenderProfile | RUN | – | – | ✓ | ✓ |
| 23 | `adaptive_prevention` | LenderProfile | RUN | – | – | ✓ | ✓ |

**Count:** 13 PRE-run flags, 10 within-RUN flags. All off in `static`, all on in `full`.

**Value-parameters activated by all non-static presets** (not boolean flags — these set continuous values):
- `spread_sensitivity = 0.6` (VBTProfile, currently 0.0)
- `flow_sensitivity = 0.5` (VBTProfile, currently 0.0)
- `forward_weight = 0.3` (VBTProfile, currently 0.0)
- `use_issuer_specific = True` (RiskAssessmentParams, currently False)
- All Plan 046 features ON (LenderProfile)
- `risk_aversion = 0.3` (TraderProfile, currently 0)
- `buy_reserve_fraction = 0.5` (TraderProfile, currently 1.0)
- `trading_motive = "liquidity_then_earning"` (TraderProfile, currently "liquidity_only")
