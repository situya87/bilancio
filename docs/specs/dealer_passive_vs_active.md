# Specification: Dealer Passive vs Active Comparison

This document specifies the balanced comparison experiment that measures the impact of dealer market-making on a Kalecki ring economy. It fully describes the decision profiles of all three agent types (traders, dealers, VBTs), what each parameter does, and which parameters respond to run-level inputs (κ, ρ, etc.) versus remaining static.

## 1. Experiment Structure

The balanced comparison runs two simulations from the **same initial state**:

| Regime | Label | `subsystem.enabled` | What happens |
|--------|-------|---------------------|--------------|
| **Passive (C)** | Counterfactual | `False` | Dealers and VBTs hold their initial inventory. No secondary market. Traders can only settle via the primary payment chain. |
| **Active (D)** | Treatment | `True` | Dealers make two-sided markets. Traders can sell receivables for cash or buy receivables for profit/liquidity. VBTs provide outside reference pricing and passthrough liquidity. |

Both regimes share identical:
- Ring topology (who owes whom)
- Initial cash distribution (controlled by κ)
- Debt distribution (controlled by concentration c)
- Maturity schedule (controlled by μ)
- Random seed

The **trading effect** is defined as:

```
trading_effect = δ_passive − δ_active
```

where δ is the default rate (fraction of total debt that defaults). Positive means dealers helped.

---

## 2. The Kalecki Ring

### 2.1 Topology

`n_agents` firms are arranged in a directed ring. Each firm owes its successor and is owed by its predecessor. Debts are represented as **tickets** (zero-coupon receivables) with face value `face_value` (default: 20) maturing on specific days.

### 2.2 Run-Level Parameters

These parameters define the macroeconomic environment. They vary across sweep grid points.

| Parameter | Symbol | Default | Controls |
|-----------|--------|---------|----------|
| `kappa` | κ | sweep grid | Liquidity ratio L₀/S₁. Cash in system relative to total debt. Lower = more stressed. |
| `concentration` | c | 1 | Dirichlet parameter for debt distribution. Lower = more unequal (some agents owe much more). |
| `mu` | μ | 0.5 | Maturity timing skew. 0 = early due dates, 1 = late. Note: code default is 0 but 0.5 (even spread) is the standard operational value. |
| `outside_mid_ratio` | ρ | 0.90 | Outside money discount. VBT mid starts at ρ × (1 − P_default). Note: code default is 1.0 but ρ=1.0 makes buys impossible (§7.2); 0.90 is the standard operational value. |
| `seed` | — | 42 | PRNG seed for ring construction and trading randomization. |

### 2.3 Scale Parameters

| Parameter | Default | Controls |
|-----------|---------|----------|
| `n_agents` | 100 | Ring size |
| `maturity_days` | 10 | Payment horizon |
| `face_value` | 20 | Face value per ticket |
| `rollover` | True | When a ticket matures, a new one is created (continuous economy) |

---

## 3. Entity Capitalization

Three entity types participate in the secondary market:

| Entity | Instances | Capital source | Default share |
|--------|-----------|----------------|---------------|
| **Traders** | n_agents (one per firm) | Ring cash endowment (from κ) | Remainder after VBT + dealer |
| **VBT** | 3 (one per bucket) | Pre-allocated tickets + matching cash | `vbt_share_per_bucket` = 25% of claims per bucket |
| **Dealer** | 3 (one per bucket) | Pre-allocated tickets + matching cash | `dealer_share_per_bucket` = 12.5% of claims per bucket |

VBTs and dealers start with inventory (tickets) AND cash. Their initial cash equals M × (number of tickets held), so they can fund repurchases.

### 3.1 Maturity Buckets

Tickets are partitioned into three maturity buckets:

| Bucket | τ range | Base spread (O) |
|--------|---------|-----------------|
| `short` | 1–3 days | 0.04 |
| `mid` | 4–8 days | 0.08 |
| `long` | 9+ days | 0.12 |

Each bucket has an independent dealer and VBT with separate inventory and quotes.

---

## 4. Trader Decision Profile

Traders are the ring firms. They decide whether to sell tickets they hold or buy tickets from the dealer. Their behavior is governed by three layers: **eligibility** (can they trade?), **risk assessment** (should they trade?), and **execution** (the trade happens).

### 4.1 TraderProfile Fields

```python
@dataclass(frozen=True)
class TraderProfile:
    risk_aversion: Decimal = Decimal("0")      # 0=risk-neutral, 1=max
    planning_horizon: int = 10                   # days look-ahead (1-20)
    aggressiveness: Decimal = Decimal("1.0")     # 0=conservative, 1=eager buyer
    default_observability: Decimal = Decimal("1.0")  # 0=ignore defaults, 1=full
    buy_reserve_fraction: Decimal = Decimal("1.0")   # reserve 100% of upcoming dues
    trading_motive: str = "liquidity_then_earning"  # allows earning buys when no shortfall
```

**Derived properties:**

| Property | Formula | Effect |
|----------|---------|--------|
| `base_risk_premium` | 0 (TraderProfile), overridden to 0.02 by BalancedComparisonConfig | Seller risk premium floor |
| `buy_risk_premium` | 0.01 + 0.02 × risk_aversion | Buyers demand premium for taking risk |
| `buy_premium_multiplier` | 1.0 + risk_aversion | Multiplier on buy threshold |
| `sell_horizon` | planning_horizon | Days to look ahead for shortfall |
| `buy_horizon` | planning_horizon | Days to look ahead for reserves |
| `surplus_threshold_factor` | 1 − aggressiveness | 0 at aggressiveness=1, 1 at aggressiveness=0 |

### 4.2 Sell Pipeline

A trader attempts to sell when holding tickets and facing a cash shortfall.

**Gate 1: Sell Eligibility** (`LiquidityDrivenSeller`)

```
eligible = has_tickets AND upcoming_shortfall(day, sell_horizon) > 0
```

The trader scans days `[current_day, current_day + sell_horizon]` and computes max(0, payment_due − cash) at each. If any day shows a shortfall, the trader wants to sell.

**Gate 2: Sell Risk Assessment** (`RiskAssessor.should_sell`)

```
EV_hold = (1 − P_default) × face_value
threshold_eff = base_risk_premium − urgency_sensitivity × (shortfall / wealth)
accept = dealer_bid × face ≥ EV_hold + threshold_eff × face
```

Where:
- `P_default` = Bayesian estimate from payment history (Laplace-smoothed)
- `wealth` = cash + asset_value (sum of EVs of all owned tickets)
- `shortfall` = upcoming_shortfall over sell_horizon
- `base_risk_premium` = 0.02 (overridden from TraderProfile's 0 by BalancedComparisonConfig)
- `urgency_sensitivity` = **0.30** (the binding parameter for trading volume)

When shortfall > 0 and wealth > 0, the threshold goes negative, meaning the seller accepts prices below EV. The more stressed the seller, the deeper the discount accepted.

**Gate 3: Concentration Limit** (post-execution, interior trades only)

If the trade was an interior dealer buy (not passthrough to VBT), checks whether accepting this ticket would breach `dealer_concentration_limit` (max fraction of dealer inventory from a single issuer). If breached, the trade is reversed. Default: 0 (disabled).

### 4.3 Buy Pipeline

A trader attempts to buy when holding surplus cash beyond upcoming obligations.

**Gate 1: Buy Eligibility** (`SurplusBuyer`)

```
reserved = buy_reserve_fraction × total_upcoming_dues(buy_horizon)
surplus = cash − reserved
threshold = face_value × surplus_threshold_factor
eligible = surplus > threshold
max_spend = surplus / 2  (prudence buffer)
```

With `trading_motive = "liquidity_only"`, agents with no upcoming liabilities are excluded entirely. The balanced comparison default is `"liquidity_then_earning"`, which allows surplus agents to buy for profit (earning motive) even without upcoming shortfalls.

**Gate 2: Safety Margin**

```
eligible = safety_margin ≥ 0
```

Underwater agents (negative net worth) cannot buy.

**Gate 3: Buy Risk Assessment** (`RiskAssessor.should_buy`)

```
EV_hold = (1 − P_default) × face
buy_threshold = buy_risk_premium                              # base: 0.01
             + earning_motive_premium (if shortfall=0)        # default: 0
             + P_blended × liquidity_factor                   # adaptive

P_blended = (P_empirical + initial_prior) / 2
liquidity_factor = max(0.75, 1 − cash_ratio)
cash_ratio = cash / (cash + asset_value)

accept = EV_hold ≥ dealer_ask × face + buy_threshold × face
```

This is more restrictive than selling. Buyers must believe the ticket is worth more than the ask price plus a risk premium that increases with the issuer's default probability and the buyer's own illiquidity.

**Gate 4: Trading Motive Filter**

If `trading_motive = "liquidity_only"`, buys of tickets maturing after the trader's earliest liability are reversed. With the balanced comparison default `"liquidity_then_earning"`, this gate is relaxed: agents with no upcoming shortfall can buy for earning (profit motive), but tickets maturing beyond the agent's planning horizon are still filtered.

**Gate 5: Cash Affordability**

```
trader.cash ≥ adjusted_price × face
```

### 4.4 RiskAssessmentParams

The dataclass defaults are shown below. **Note:** `BalancedComparisonConfig` overrides `base_risk_premium` to 0.02 (from 0) via `risk_assessment_config`.

```python
@dataclass(frozen=True)
class RiskAssessmentParams:
    lookback_window: int = 5
    smoothing_alpha: Decimal = Decimal("1.0")
    base_risk_premium: Decimal = Decimal("0")     # overridden to 0.02 by balanced comparison
    urgency_sensitivity: Decimal = Decimal("0.30")
    use_issuer_specific: bool = False
    buy_premium_multiplier: Decimal = Decimal("1.0")
    buy_risk_premium: Decimal = Decimal("0.01")
    default_observability: Decimal = Decimal("1.0")
    initial_prior: Decimal = Decimal("0.15")  # overridden by kappa
    earning_motive_premium: Decimal = Decimal("0.0")
```

**Effective values in balanced comparison sweep:**

| Parameter | Dataclass default | Balanced comparison effective |
|-----------|-------------------|------------------------------|
| `base_risk_premium` | 0 | **0.02** |
| `initial_prior` | 0.15 | **kappa-informed** (varies by κ) |
| All others | as shown | unchanged |

---

## 5. Dealer Decision Profile

The dealer is a **mechanical market-maker** using the Treynor pricing kernel. It does not have behavioral parameters — its quotes are deterministic functions of inventory, cash, and VBT anchors.

### 5.1 State

Each bucket has an independent `DealerState`:

| Field | Meaning |
|-------|---------|
| `inventory` | Tickets currently held |
| `cash` | Cash holdings |
| `a` | Number of tickets = len(inventory) |
| `x` | Face inventory = a × S |
| `K_star` | Max fundable tickets = floor(V / (M × S)) |
| `X_star` | One-sided capacity = S × K_star |
| `bid` | Current bid price |
| `ask` | Current ask price |

### 5.2 Pricing Kernel

The kernel computes quotes from inventory position:

```
V = M × x + cash                                 # mid-valued equity
K* = floor(V / (M × S))                          # max buy tickets
X* = S × K*                                       # one-sided capacity
λ = S / (X* + S)                                  # layoff probability
I = λ × O                                         # inside width (spread)
p(x) = M − (O / (X* + 2S)) × (x − X*/2)         # inventory-sensitive midline
```

Interior quotes:
```
a(x) = p(x) + I/2                                # interior ask
b(x) = p(x) − I/2                                # interior bid
```

Clipped quotes:
```
bid = max(B, b(x))                               # floored at VBT bid
ask = min(A, a(x))                               # capped at VBT ask
```

Both capped at par (1.0) since tickets are zero-coupon.

### 5.3 Trade Routing

When a trader sells to the dealer:
1. If `can_interior_buy(dealer)` (capacity AND cash available): **interior trade** — dealer buys at `bid`, ticket enters dealer inventory.
2. Otherwise: **passthrough** — routed to VBT, which buys at `B` (VBT bid).

When a trader buys from the dealer:
1. If `can_interior_sell(dealer)` (has inventory AND not in guard mode): **interior trade** — dealer sells at `ask` from own inventory.
2. Otherwise: **passthrough** — routed to VBT, which sells at `A` (VBT ask).

### 5.4 Guard Regime

When VBT mid M ≤ 0.02 (near-zero), the dealer enters **guard mode**: X* = 0, all trades route to VBT. This prevents the dealer from making markets on near-worthless paper.

### 5.5 Issuer-Specific Pricing (Feature 1)

When enabled (`issuer_specific_pricing = True`), the execution price is adjusted by:
```
price_factor = 1 − (P_issuer − P_system)
```
Issuers with above-average default probability get worse prices. Disabled by default.

### 5.6 Concentration Limit (Feature 3)

When `dealer_concentration_limit > 0`, interior buys that would make a single issuer exceed this fraction of total dealer inventory are reversed. Disabled by default.

---

## 6. VBT Decision Profile

The VBT (Value-Based Trader) provides **outside liquidity** and **reference pricing**. It is the market of last resort when the dealer cannot or will not trade.

### 6.1 VBTProfile Fields

```python
@dataclass(frozen=True)
class VBTProfile:
    mid_sensitivity: Decimal = Decimal("1.0")     # 0=fixed mid, 1=fully adaptive
    spread_sensitivity: Decimal = Decimal("0.0")  # 0=fixed spread, 1=widen with defaults
    spread_scale: Decimal = Decimal("1.0")        # multiplicative scale on base spreads
    forward_weight: Decimal = Decimal("0.0")      # forward stress blending (0=disabled)
    stress_horizon: int = 5                        # days to look ahead for stress
    flow_sensitivity: Decimal = Decimal("0.0")    # flow-aware ask widening (0=disabled)
```

### 6.2 VBTState and Quote Computation

Each bucket has an independent `VBTState` with:

| Field | Meaning | Initial value |
|-------|---------|---------------|
| `M` | Mid price | ρ × (1 − P_prior) |
| `O` | Spread | base_spread + 0.6 × P_prior |
| `A` | Ask = M + O/2 | Computed |
| `B` | Bid = M − O/2 | Computed |
| `inventory` | Tickets held | Pre-allocated from scenario |
| `cash` | Cash holdings | Matching cash for inventory |

### 6.3 Daily Mid Update

Each day before trading, VBT mids are recomputed using the **credit-adjusted pricing model**:

```
P_default = risk_assessor.estimate_default_prob("_system_", current_day)
raw_M = ρ × (1 − P_default)
initial_M = ρ × (1 − initial_prior)
new_M = initial_M + mid_sensitivity × (raw_M − initial_M)
```

With `mid_sensitivity = 1.0` (default), this simplifies to `M = ρ × (1 − P_default)`. As defaults accumulate and P_default rises, M falls, widening the discount.

### 6.4 Daily Spread Update

When `spread_sensitivity > 0`:
```
base_O = per-bucket base spread
new_O = base_O + spread_sensitivity × P_default
```

With `spread_sensitivity = 0.0` (default), spreads are fixed at initialization.

### 6.5 Flow-Aware Ask Widening (Feature 2)

When `flow_sensitivity > 0`:
```
net_outflow = cumulative_outflow − cumulative_inflow
outflow_ratio = net_outflow / initial_face_inventory
ask_premium = flow_sensitivity × max(0, outflow_ratio)
A = M + O/2 + ask_premium
```

This widens the ask when the VBT has been a net seller, slowing inventory drain. Disabled by default.

---

## 7. Parameter Sensitivity Map

### 7.1 Sensitive to κ (kappa)

These parameters change with the run's liquidity stress level:

| Parameter | How it depends on κ | Location |
|-----------|---------------------|----------|
| **initial_prior** | `P = 0.05 + 0.15 × max(0, 1−κ)/(1+κ)` | `dealer/priors.py:kappa_informed_prior()` |
| | κ=0 → P=0.20, κ=0.5 → P=0.10, κ≥1 → P=0.05 | |
| **VBT mid (M)** | `M = ρ × (1 − P_default)`, where P starts at kappa-informed prior | `dealer_sync.py:_update_vbt_credit_mids()` |
| **VBT spread (O)** | `O = base + 0.6 × P_prior` at initialization | `dealer_wiring.py` |
| **Trader beliefs** | P_default estimation starts from kappa-informed prior | via `RiskAssessmentParams.initial_prior` |
| **Agent cash** | Total system cash = κ × total debt | Ring compiler |
| **Dealer capacity (K*)** | K* = floor(V / (M × S)), where V depends on initial cash | Kernel |

### 7.2 Sensitive to ρ (outside_mid_ratio)

| Parameter | How it depends on ρ |
|-----------|---------------------|
| **VBT mid (M)** | `M = ρ × (1 − P)`. Lower ρ = lower mid = wider discount from par. |
| **VBT bid/ask** | Derived from M. Lower ρ → lower B, lower A. |
| **Dealer quotes** | Anchored to VBT. Lower ρ → lower dealer bid/ask. |
| **Buy viability** | At ρ=1.0, VBT ask ≈ 1.0, making buys impossible (EV < ask). |

### 7.3 Sensitive to P_default (evolves during simulation)

| Parameter | How it responds |
|-----------|-----------------|
| **VBT mid** | Falls as defaults accumulate (daily update) |
| **Seller threshold** | Sellers with higher P_default have lower EV_hold, accept worse prices |
| **Buyer threshold** | Buyers facing issuers with higher P_default demand larger premium |
| **Dealer capacity** | Falls as M falls (less valuable inventory) |

### 7.4 NOT Sensitive to κ (static across runs)

These parameters are fixed at their default values unless explicitly overridden via CLI:

| Parameter | Default | Role |
|-----------|---------|------|
| **urgency_sensitivity** | 0.30 | How much shortfall lowers sell threshold |
| **risk_aversion** | 0 | Buyer premium scaling |
| **planning_horizon** | 10 | Look-ahead for sell/buy eligibility |
| **aggressiveness** | 1.0 | Buyer surplus threshold |
| **buy_reserve_fraction** | 1.0 | Fraction of dues reserved before buying |
| **default_observability** | 1.0 | How much observed defaults update beliefs |
| **trading_motive** | liquidity_then_earning | Allows earning buys when no shortfall |
| **mid_sensitivity** | 1.0 | VBT mid responsiveness to defaults |
| **spread_sensitivity** | 0.0 | VBT spread widening (disabled) |
| **flow_sensitivity** | 0.0 | VBT flow-aware asks (disabled) |
| **dealer_concentration_limit** | 0 | Issuer concentration cap (disabled) |
| **issuer_specific_pricing** | False | Per-issuer price adjustment (disabled) |
| **lookback_window** | 5 | Days of history for P_default estimation |
| **smoothing_alpha** | 1.0 | Laplace smoothing strength |
| **base_risk_premium** | 0.02 (sell), 0.01 (buy) | Minimum premium demanded |
| **bucket base spreads** | short=0.04, mid=0.08, long=0.12 | Term structure of spreads |
| **vbt_share_per_bucket** | 0.25 | VBT capitalization |
| **dealer_share_per_bucket** | 0.125 | Dealer capitalization |
| **alpha_vbt, alpha_trader** | 0, 0 | Informedness (currently unused with kappa wiring) |

---

## 8. Trading Day Sequence

Each simulation day proceeds as:

1. **Tick maturity**: decrement remaining_tau on all tickets, re-bucket as needed.
2. **Settlement phase**: process payments due today. Agents with insufficient cash default (expel-agent mode: they are removed from the ring).
3. **Rollover**: if enabled, matured tickets are replaced with new ones (continuous economy).
4. **VBT credit update**: recompute VBT mids using latest P_default estimates.
5. **Dealer quote recompute**: run kernel for each bucket using updated VBT anchors.
6. **Trading rounds** (up to 100 per day, early-terminate when no intentions):
   a. Collect sell intentions (traders with shortfall).
   b. Collect buy intentions (traders with surplus).
   c. Shuffle and execute: sells first, then buys, each through the gate cascade.
7. **History update**: record which agents defaulted for future P_default estimation.

---

## 9. Output Metrics

### 9.1 Primary Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| `delta_passive` | defaulted_debt / total_debt (passive run) | Baseline default rate |
| `delta_active` | defaulted_debt / total_debt (active run) | Default rate with dealer |
| `trading_effect` | δ_passive − δ_active | Default reduction from trading (positive = helped) |
| `trading_relief_ratio` | trading_effect / δ_passive | Percentage reduction |
| `phi_passive` | 1 − δ_passive | Clearing rate (passive) |
| `phi_active` | 1 − δ_active | Clearing rate (active) |

### 9.2 Loss Metrics

| Metric | Meaning |
|--------|---------|
| `payable_default_loss` | Face value of defaulted payables |
| `intermediary_loss` | Loss absorbed by dealers/VBTs (negative PnL) |
| `system_loss` | payable_default_loss + intermediary_loss |
| `total_loss_pct` | payable_default_loss / total_face_value |
| `intermediary_loss_pct` | intermediary_loss / total_face_value |

### 9.3 Dealer Metrics

| Metric | Meaning |
|--------|---------|
| `dealer_total_pnl` | Dealer profit/loss (active run) |
| `dealer_total_return` | PnL / initial capital |
| `total_trades` | Number of executed trades |
| `dealer_passive_pnl` | Dealer PnL in passive run (hold-only) |
| `dealer_trading_incremental_pnl` | Active PnL − Passive PnL |

### 9.4 Cascade Metrics

| Metric | Meaning |
|--------|---------|
| `n_defaults_passive` | Number of agents that defaulted (passive) |
| `n_defaults_active` | Number of agents that defaulted (active) |
| `cascade_fraction_passive` | Fraction of defaults that were cascading (passive) |
| `cascade_fraction_active` | Fraction of defaults that were cascading (active) |
| `cascade_effect` | cascade_fraction_passive − cascade_fraction_active |
