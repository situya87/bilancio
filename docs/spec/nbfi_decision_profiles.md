# Decision Profile Specification: Passive → Dealer → NBFI Pipeline

This document specifies all decision profiles in the balanced comparison pipeline:
traders, dealers, VBTs, and the NBFI lender. For each profile, it distinguishes
**tunable parameters** (wired to CLI/sweep) from **structural defaults** (fixed
unless code is changed).

## Architecture Overview

The NBFI comparison sweep runs **two arms only**: passive (baseline) and lender (NBFI).
The active dealer arm is excluded because dealer trading dominates the lending effect
(+16.9pp vs +5.8pp at κ=0.3), making it impossible to isolate and measure NBFI impact
when both are present. The combined dealer+lender arm showed no additive benefit over
dealer alone at κ≥0.5.

| Arm | Intermediaries | What it measures | Included |
|-----|---------------|------------------|----------|
| **Passive** | None | Pure settlement baseline | Yes |
| **Active** | Dealer + VBT | Secondary market impact | **No** — use dealer spec instead |
| **Lender** | NBFI | Credit provision impact | Yes |
| **Dealer+Lender** | Dealer + VBT + NBFI | Combined effect | **No** — dealer dominates |

All arms share the same ring topology (agents, payables, cash endowments) seeded
identically. The only difference is which intermediaries are active.

**To measure dealer impact**: use the dealer specification at `docs/specs/dealer_passive_vs_active.md`.
**To measure NBFI impact**: use this specification (passive vs lender only).

---

## 1. Trader Profile

**File:** `src/bilancio/decision/profiles.py`
**Applies to:** All firms/households in the ring (when dealer arm is active).

Traders decide whether to **sell** receivables to the dealer (for immediate cash)
or **buy** receivables from the dealer (to earn the face-par spread).

### Tunable Parameters (CLI-wired)

| Field | Type | Default | CLI Flag | Effect |
|-------|------|---------|----------|--------|
| `risk_aversion` | Decimal | 0 | `--risk-aversion` | Higher → pickier buyers. Buy premium = `0.01 + 0.02 × RA` |
| `planning_horizon` | int | 10 | `--planning-horizon` | Days to look ahead for obligations. Longer → more cash reserved |
| `aggressiveness` | Decimal | 1.0 | `--aggressiveness` | Lower → higher surplus needed to buy. Threshold = `face × (1 - agg)` |
| `default_observability` | Decimal | 1.0 | `--default-observability` | Lower → agents slower to update beliefs from observed defaults |
| `buy_reserve_fraction` | Decimal | 0.5 | `--buy-reserve-fraction` | Fraction of upcoming dues reserved before buying |
| `trading_motive` | str | "liquidity_then_earning" | `--trading-motive` | `liquidity_only` / `liquidity_then_earning` / `unrestricted` |

### Computed (not independently tunable)

| Property | Formula | Notes |
|----------|---------|-------|
| `buy_risk_premium` | `0.01 + 0.02 × risk_aversion` | Min spread a buyer demands |
| `surplus_threshold_factor` | `1 - aggressiveness` | Multiplied by face value for buyer eligibility |
| `sell_horizon` | `planning_horizon` | Seller look-ahead = planning horizon |
| `buy_horizon` | `planning_horizon` | Buyer look-ahead = planning horizon |

### Sell Decision

A trader sells ticket `t` to the dealer at bid `B_d` if:

```
EV_hold = (1 - p_default) × face
threshold = base_premium - urgency × min(1, shortfall / wealth)
B_d ≥ EV_hold + threshold
```

Where `base_premium = 0` (selling converts uncertainty to cash — no premium
needed), and `urgency` lowers the threshold when the trader has cash shortfalls.

### Buy Decision

A trader buys ticket `t` from the dealer at ask `A_d` if:

1. **Eligibility**: `cash - upcoming_dues ≥ face × (1 - aggressiveness)`
2. **Risk gate**: `EV_hold ≥ A_d × face + threshold`

Where threshold includes:
- `buy_risk_premium` (from risk aversion)
- `earning_motive_premium` (if no liquidity shortfall, buying is speculative)
- Liquidity adjustment: `p_blended × max(0.75, 1 - cash_ratio)`

---

## 2. VBT Profile (Value-Based Trader / Outside Liquidity)

**File:** `src/bilancio/decision/profiles.py`
**Role:** Provides outside reference prices (mid, bid, ask) that anchor dealer quotes.

The VBT is the "outside market" — it sets the price floor (bid) and ceiling (ask)
within which the dealer operates. VBT quotes are credit-adjusted based on observed
default rates.

### Tunable Parameters (CLI-wired)

| Field | Type | Default | CLI Flag | Effect |
|-------|------|---------|----------|--------|
| `mid_sensitivity` | Decimal | 1.0 | `--vbt-mid-sensitivity` | 0=fixed mid, 1=fully tracks defaults |
| `spread_sensitivity` | Decimal | 0.0 | `--vbt-spread-sensitivity` | 0=fixed spread, 1=spreads widen with defaults |
| `spread_scale` | Decimal | 1.0 | `--spread-scale` | Global multiplier on all bucket spreads |
| `flow_sensitivity` | Decimal | 0.0 | `--flow-sensitivity` | Ask premium when VBT is net seller (draining) |

### Structural (not CLI-wired)

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `forward_weight` | Decimal | 0.0 | Blend forward stress into default prob |
| `stress_horizon` | int | 5 | Days ahead for forward stress estimation |

### VBT Pricing Update (daily)

```
p_default = RiskAssessor.estimate_default_prob("_system_", day)
M_raw = outside_mid_ratio × (1 - p_default)
M_init = outside_mid_ratio × (1 - initial_prior)
M = M_init + mid_sensitivity × (M_raw - M_init)

O = O_base × (1 + spread_sensitivity × p_default)

A = M + O/2          # VBT ask (ceiling)
B = M - O/2          # VBT bid (floor)
```

When `mid_sensitivity = 0`: M stays at its initial value (static pricing).
When `mid_sensitivity = 1`: M fully tracks the observed default rate.

### Bucket Spreads

Per-bucket outside spreads (`O`) are set at initialization:

| Bucket | Base Spread | Effective at P=0.15 |
|--------|-------------|---------------------|
| short (1-3 days) | 0.04 | ~0.13 |
| mid (4-6 days) | 0.08 | ~0.17 |
| long (7+ days) | 0.12 | ~0.21 |

These are multiplied by `spread_scale` and then adjusted by `spread_sensitivity`.
Effective spreads at a given default probability P are: `O_base + 0.6 × P` (the
credit-sensitive corridor formula).

---

## 3. Dealer (Treynor Kernel)

**File:** `src/bilancio/dealer/kernel.py`, `src/bilancio/engines/dealer_integration.py`
**Role:** Market-maker between sellers (liquidity-stressed agents) and buyers (cash-surplus agents).

The dealer uses a Treynor-style inventory-sensitive pricing kernel. It holds
an inventory of tickets and quotes bid/ask around a midline that moves with
inventory position.

### Tunable Parameters (CLI-wired)

| Field | Default | CLI Flag | Effect |
|-------|---------|----------|--------|
| `dealer_share_per_bucket` | 0.05 | `--dealer-share-per-bucket` | Dealer capital as fraction of bucket face value |
| `vbt_share_per_bucket` | 0.20 | `--vbt-share-per-bucket` | VBT capital as fraction of bucket |
| `alpha_vbt` | 0 | `--alpha-vbt` | VBT informedness (0=naive, 1=κ-informed prior) |
| `alpha_trader` | 0 | `--alpha-trader` | Trader informedness (0=naive, 1=κ-informed prior) |
| `outside_mid_ratio` (ρ) | 0.90 | `--outside-mid-ratios` | Outside money discount (sweep grid param) |
| `issuer_specific_pricing` | False | `--issuer-specific-pricing` | Per-issuer bid/ask adjustments |
| `dealer_concentration_limit` | 0 | `--dealer-concentration-limit` | Max fraction of bucket from single issuer |
| `trading_rounds` | 1 | `--trading-rounds` | Sub-rounds per day (more = more trades) |

> **Note:** The `VBTProfile` dataclass declares `vbt_share_per_bucket` with a default of
> `Decimal("0.25")`, but the NBFI comparison config (`BalancedComparisonConfig`) overrides
> this to **0.20** at sweep construction time. The values in the table above (0.05 dealer,
> 0.20 VBT) reflect the effective NBFI sweep defaults, matching the scenario model defaults.

### Structural (not CLI-wired)

| Derived Quantity | Formula | Meaning |
|-----------------|---------|---------|
| `K*` | `floor(V / (M × S))` | Max fundable tickets |
| `X*` | `S × K*` | One-sided capacity (face) |
| `λ` | `S / (X* + S)` | Layoff probability |
| `I` | `λ × O` | Inside width |
| `midline` | `M - O/(X*+2S) × (x - X*/2)` | Inventory-adjusted mid |
| `bid` | `max(B, midline - I/2)` | Clipped to VBT floor |
| `ask` | `min(A, midline + I/2)` | Clipped to VBT ceiling |

### Dealer Capacity

Dealer capacity is determined by its cash relative to ticket prices:

```
dealer_cash ≈ dealer_share_per_bucket × bucket_face_value
K* = floor(dealer_cash / (M × S))
```

- `K* < 3`: Capacity binds immediately → dealer is a passthrough
- `K* > n_agents/2`: Dealer never constrained → tight spread
- Sweet spot: 5 to n_agents/4

### Guard Regime

When VBT mid `M ≤ 0.02` (extreme stress), the dealer collapses to outside quotes
(`bid = B`, `ask = A`) — no interior market-making.

### Issuer-Specific Pricing (Feature 109)

When enabled, per-issuer dealer quotes are adjusted:

```
multiplier = max(0, 1 - (P_issuer - P_system))
bid_adjusted = bid × multiplier
ask_adjusted = ask × multiplier
```

Riskier issuers get lower bids (dealer won't buy) and lower asks (dealer sells
cheaper to clear inventory).

---

## 4. Risk Assessment (shared by Traders and Dealer)

**File:** `src/bilancio/decision/risk_assessment.py`
**Used by:** `TradeGate.should_sell()`, `TradeGate.should_buy()`, VBT mid updates.

### RiskAssessmentParams

| Field | Type | Default | Source | Effect |
|-------|------|---------|--------|--------|
| `lookback_window` | int | 5 | Hardcoded | Days of history considered |
| `smoothing_alpha` | Decimal | 1.0 | Hardcoded | Laplace smoothing (higher = more conservative) |
| `urgency_sensitivity` | Decimal | 0.30 | Hardcoded | How much cash stress lowers sell threshold |
| `initial_prior` | Decimal | 0.15 | κ-aware or 0.15 | Default probability before any observations |
| `default_observability` | Decimal | 1.0 | From TraderProfile | Friction in updating beliefs |
| `buy_risk_premium` | Decimal | 0.01 | From TraderProfile | `0.01 + 0.02 × risk_aversion` |
| `use_issuer_specific` | bool | False | Hardcoded | Per-issuer vs system-wide tracking |

### Belief Update (BeliefTracker)

```
p_empirical = (α + defaults) / (2α + total)     # Laplace smoothed
p = initial_prior + observability × (p_empirical - initial_prior)
```

When `observability = 0`: beliefs stay at `initial_prior` forever.
When `observability = 1`: beliefs fully track observed default rates.

### κ-Informed Prior

When alpha > 0 (VBT or trader is "informed"), the initial prior is computed from κ:

```
initial_prior = 1 / (1 + κ)
```

This gives the VBT/trader structural knowledge of system liquidity stress.

---

## 5. NBFI Lender Profile

**File:** `src/bilancio/decision/profiles.py`
**Phase:** Runs after settlement, before dealer trading.
**Role:** Provides unsecured short-term loans to agents with shortfalls.

### Core Parameters (CLI-wired)

| Field | Type | Default | CLI Flag | Effect |
|-------|------|---------|----------|--------|
| `lender_share` | Decimal | 0.10 | `--lender-share` | Capital as fraction of system cash |
| `base_rate` | Decimal | 0.05 | (scenario config) | Floor interest rate |
| `risk_premium_scale` | Decimal | — | Computed: `0.1 + 0.4 × risk_aversion` | Risk premium multiplier |
| `risk_aversion` | Decimal | 0.3 | (scenario config) | Higher → wider risk premium |
| `max_single_exposure` | Decimal | 0.15 | (scenario config) | Max to single borrower |
| `max_total_exposure` | Decimal | 0.80 | (scenario config) | Max total lending |
| `min_coverage_ratio` | Decimal | 0.5 | `--lender-min-coverage` | Balance-sheet gate threshold |

### Plan 046 Parameters (CLI-wired)

| Field | Type | Default | CLI Flag | Effect |
|-------|------|---------|----------|--------|
| `maturity_matching` | bool | False | `--lender-maturity-matching` | Match loan maturity to nearest receivable |
| `min_loan_maturity` | int | 2 | `--lender-min-loan-maturity` | Floor when maturity matching |
| `max_loans_per_borrower_per_day` | int | 0 | `--lender-max-loans-per-borrower-per-day` | Max outstanding loans per borrower (0=unlimited) |
| `ranking_mode` | str | "profit" | `--lender-ranking-mode` | `profit` / `cascade` / `blended` |
| `cascade_weight` | Decimal | 0.5 | `--lender-cascade-weight` | Weight in blended mode |
| `coverage_mode` | str | "gate" | `--lender-coverage-mode` | `gate` (reject) / `graduated` (penalize rate) |
| `coverage_penalty_scale` | Decimal | 0.10 | `--lender-coverage-penalty-scale` | Rate premium per unit below threshold |
| `preventive_lending` | bool | False | `--lender-preventive-lending` | Proactive lending to at-risk agents |
| `prevention_threshold` | Decimal | 0.3 | `--lender-prevention-threshold` | Issuer default prob trigger |

### Lending Decision Pipeline

```
1. Identify borrowers with shortfalls (upcoming_dues - cash > 0)
2. Coverage gate: reject if coverage < min_coverage_ratio
   - "gate" mode: binary reject
   - "graduated" mode: add rate premium instead
3. Risk screen: reject if p_default > max_default_prob (0.50)
4. Price: rate = base_rate + risk_premium_scale × p_default
5. Rank opportunities:
   - "profit": by expected_profit = rate × (1 - p_default)
   - "cascade": by coverage × downstream_damage × (1 - p_default)
   - "blended": weighted combination
6. Execute loans in rank order, checking:
   - Concentration limit (outstanding loans to this borrower)
   - Exposure limits (single + total)
   - Maturity matching (if enabled)
7. Preventive pass (if enabled): lend to agents without shortfalls
   whose receivables are from high-risk counterparties
```

### Maturity Matching (Phase 1A)

```
nearest_receivable_day = earliest receivable due day for this borrower
effective_maturity = max(min_loan_maturity, min(nearest_day - today + 1, max_loan_maturity))
term_premium = rate × (1 + 0.01 × (maturity - 2))  # if maturity > 2
```

### Concentration Limit (Phase 1B)

Counts **outstanding** (non-repaid) loans from lender to borrower via
`_count_existing_loans()`. If count ≥ limit, loan is rejected.

### Cascade-Aware Ranking (Phase 2)

Downstream damage = sum of payables the borrower owes to others. If the borrower
defaults, these creditors lose their receivables.

```
cascade_score = coverage_ratio × normalized_downstream × (1 - p_default)
blended_score = cascade_weight × cascade_score + (1 - cascade_weight) × profit_score
```

### Graduated Coverage (Phase 3)

Instead of rejecting sub-threshold borrowers, add a rate premium:

```
penalty = coverage_penalty_scale × max(0, min_coverage_ratio - actual_coverage)
adjusted_rate = base_rate + risk_premium_scale × p_default + penalty
```

Still rejects deeply insolvent borrowers (coverage < -1.0).

### Preventive Lending (Phase 4)

Second pass after main lending. For agents WITHOUT shortfalls:

```
at_risk = sum(receivable.amount for each receivable
              where issuer's p_default > prevention_threshold)
```

If `at_risk > 0`, offer a preemptive loan sized to cover the at-risk receivables.
Uses InformationService per-issuer probabilities when available (review fix #3).

---

## 6. What Is Sensitive to Run Parameters

### Sweep Grid Parameters (affect all arms equally)

| Parameter | Symbol | Effect | Sensitive Profiles |
|-----------|--------|--------|-------------------|
| `kappa` (κ) | L₀/S₁ | System liquidity stress | All (via initial_prior, coverage ratios) |
| `concentration` (c) | Dirichlet | Debt inequality | None directly (topology only) |
| `mu` (μ) | — | Payment timing skew | All (via shortfall timing) |
| `outside_mid_ratio` (ρ) | — | Outside money discount | VBT, Dealer (via M), Trader (via EV) |
| `n_agents` | — | Ring size | Capacity ratios, statistical noise |
| `maturity_days` | — | Payment horizon | All (via temporal spread) |
| `face_value` | — | Ticket size | Dealer capacity, loan amounts |
| `seed` | — | Randomness | All (different default cascades) |

### Profile Parameters (affect specific arm behavior)

| Parameter | Affects | NOT affected |
|-----------|---------|-------------|
| `risk_aversion` | Trader buy decisions | Sell decisions, VBT, Dealer kernel, NBFI |
| `planning_horizon` | Trader sell/buy horizons | VBT, Dealer kernel, NBFI |
| `vbt_mid_sensitivity` | VBT mid pricing → Dealer quotes → Trade prices | Trader decisions directly |
| `vbt_spread_sensitivity` | VBT spread → Dealer inside width | Trader decisions directly |
| `dealer_share_per_bucket` | Dealer capacity K* | VBT, Trader decisions |
| `alpha_vbt` | VBT initial prior | Trader, Dealer kernel |
| `alpha_trader` | Trader initial prior | VBT, Dealer kernel |
| `lender_share` | NBFI capital | Dealer, VBT, Trader |
| `lender_min_coverage` | NBFI borrower screening | Dealer, VBT, Trader |
| `ranking_mode` | NBFI loan allocation order | Dealer, VBT, Trader |
| `maturity_matching` | NBFI loan terms | Dealer, VBT, Trader |

### Cross-Profile Interactions

1. **VBT → Dealer → Trader**: VBT sets anchors, dealer quotes within them, traders decide against dealer quotes. VBT sensitivity parameters cascade through the chain.

2. **Defaults → Beliefs → Prices**: Default events update `BeliefTracker`, which changes `p_default`, which moves VBT mid, which shifts dealer quotes, which changes trade viability.

3. **NBFI → Settlement → Trading**: NBFI loans change agent cash → affects which agents have shortfalls → changes who sells to dealer → changes dealer inventory → changes prices.

4. **κ → Everything**: Lower κ means more defaults, higher initial priors (if α > 0), lower VBT mids, wider dealer spreads, more lending demand.

---

## 7. Backward Compatibility

All Plan 046 NBFI features default to **off** (pre-046 behavior):

| Feature | Default | Pre-046 Behavior |
|---------|---------|-----------------|
| Maturity matching | `False` | Fixed 2-day loans |
| Concentration limit | `0` (unlimited) | No limit |
| Ranking mode | `"profit"` | Rank by expected profit |
| Coverage mode | `"gate"` | Binary reject/accept |
| Preventive lending | `False` | Only reactive lending |

To reproduce pre-046 results: use default parameters (no `--lender-*` flags).
