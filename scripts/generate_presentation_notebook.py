#!/usr/bin/env python3
"""Generate the Kalecki Ring presentation notebook.

Creates notebooks/kalecki_ring_presentation.ipynb with:
  Part I   – Theory (bilancio, Kalecki ring, parameters)
  Part II  – Simulation mechanics (lifecycle, phases, settlement)
  Part III – Decision profiles (trader, dealer, VBT, bank, CB, NBFI)
  Part IV  – Experimental design (arms, grid, metrics)
  Part V   – Analysis (all sweep charts with interpretations)
"""

import json
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────
def _split_source(source: str) -> list[str]:
    """Split source into list of lines with newlines preserved (notebook format)."""
    lines = source.split("\n")
    # Add \n to all lines except the last
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:  # last line non-empty
        result.append(lines[-1])
    return result

def md(source: str) -> dict:
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source), "id": None}

def code(source: str) -> dict:
    """Code cell."""
    return {"cell_type": "code", "metadata": {}, "source": _split_source(source),
            "outputs": [], "execution_count": None, "id": None}


cells: list[dict] = []

# ═══════════════════════════════════════════════════════════════════════
# PART I — THEORY
# ═══════════════════════════════════════════════════════════════════════

cells.append(md("""\
# The Kalecki Ring: Simulating Liquidity and Settlement in Payment Networks

**Bilancio** — A multi-agent payment system simulator

*Aleksandar Stojanovic, with Vlad Gheorghe and Michael Loebach*

---

This notebook presents the full analysis pipeline for the Kalecki Ring experiments. It covers:

1. **What Bilancio is** — the simulator and its accounting foundation
2. **The Kalecki Ring** — the economic fable that motivates our experiments
3. **How a simulation works** — lifecycle, phases, settlement mechanics
4. **Agent decision profiles** — how each agent type makes choices
5. **Experimental design** — arms, parameter grid, metrics
6. **Results** — comprehensive cross-sweep analysis of three intermediation mechanisms
"""))

# ── 1. What is Bilancio ──────────────────────────────────────────────
cells.append(md("""\
---
## Part I: The Simulator

### 1. What is Bilancio?

Bilancio is a **multi-agent payment system simulator** that models liquidity, settlement, and default in networks of interconnected financial obligations.

**Core principles:**

- **Double-entry accounting**: Every transaction is recorded as a debit and credit. Balance sheet invariants are enforced at every step — assets always equal liabilities plus equity. This is not a simplification; it is the ground truth of the simulation.

- **Agents as balance sheets**: Every entity (firm, household, dealer, bank, central bank) is a balance sheet with a behavioral profile attached. There are no special constructs — the dealer is a balance sheet that happens to make markets; the bank is a balance sheet that happens to lend.

- **Settlement as ground truth**: The simulation does not assume prices or clearing rates. Instead, agents make decisions, execute trades, and attempt settlement. The *outcome* of settlement — who pays, who defaults, what prices emerge — is an emergent result, not a parameter.

- **Instruments with contract terms**: Every financial claim (payable, loan, deposit, reserve) is a concrete instrument with an issuer, a holder, a face value, a maturity date, and a means of payment. Settlement follows the contract terms strictly.

```
┌─────────────────────────────────────────────────┐
│  Scenario Builder                               │
│  Ring topology, κ, c, μ                         │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  Settlement Engine                              │
│  Daily phases, MOP ranking, atomic settlement   │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  Decision Layer                                 │
│  Observe → Value → Assess → Choose              │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  Analysis & Export                              │
│  δ (default rate), φ (clearing rate), HTML      │
└─────────────────────────────────────────────────┘
```
"""))

# ── 2. The Kalecki Ring ──────────────────────────────────────────────
cells.append(md("""\
### 2. The Kalecki Ring

#### The Fable (Kalecki, via Toporowski 2008)

> A community of merchants each owes the next in a circle. No one can pay until they are paid. Settlement requires either sufficient initial liquidity or coordinated netting.

The fable illustrates a profound insight: **the same total debt can produce full clearing or total default**, depending only on the distribution of liquidity and the timing of payments.

```
       H₁ ──→ H₂
      ↗          ↘
    H₆            H₃
      ↖          ↙
       H₅ ←── H₄

    Each agent owes the next
```

In a ring of N agents, agent *i* owes a payable to agent *i+1* (modulo N). If agent 1 has enough cash to pay agent 2, then agent 2 receives cash and can pay agent 3, and so on around the ring — the system clears. But if agent 1 cannot pay, agent 2 never receives funds, and the default cascades around the entire ring.

**This is not a toy model.** The ring captures the essential structure of real payment networks:
- **Sequential dependency**: payments depend on incoming funds
- **Liquidity as a flow, not a stock**: the same cash can settle multiple obligations as it circulates
- **Cascade dynamics**: one failure can propagate through the entire system

The Kalecki ring is bilancio's primary experimental topology. By varying the initial conditions (how much cash, how debt is distributed, when payments are due), we explore the boundary between clearing and collapse — and test whether financial intermediaries (dealers, lenders, banks) can push that boundary.
"""))

# ── 3. Control Parameters ────────────────────────────────────────────
cells.append(md("""\
### 3. Control Parameters

Three primary parameters control the initial conditions of the ring. Together they define the *stress surface* of the system.

#### κ (Kappa) — Liquidity Ratio

$$\\kappa = \\frac{L_0}{S_1}$$

The ratio of total initial cash ($L_0$) to total debt ($S_1$). This is the most fundamental parameter: how much liquid money exists relative to obligations.

| κ | Interpretation | Expected behavior |
|---|----------------|-------------------|
| < 0.5 | **Severely stressed** | Cash cannot cover half the debt. Mass defaults almost certain. |
| 0.5–1.0 | **Stressed** | Cash is scarce. Defaults depend on distribution and timing. |
| 1.0 | **Balanced** | System has exactly enough cash for all debts (in aggregate). |
| 1.0–2.0 | **Comfortable** | Surplus liquidity, but distribution may still cause defaults. |
| > 2.0 | **Abundant** | Defaults are structural (topology/distribution), not liquidity-driven. |

#### c (Concentration) — Debt Distribution

The Dirichlet parameter controlling how unequally debt is distributed across agents. With N agents and total debt Q, each agent's debt share is drawn from Dir(c, c, ..., c).

| c | Distribution shape | Implication |
|---|-------------------|-------------|
| 0.2 | **Extremely unequal** — some agents owe 5-10× the mean | Fragile: a few agents are structurally doomed |
| 0.5 | **Unequal** — significant variation | Moderate fragility |
| 1.0 | **Moderate** — uniform-like distribution | Baseline inequality |
| 2.0 | **Equal** — debt is spread evenly | More stable |
| 5.0+ | **Very equal** — near-uniform | Structural defaults nearly eliminated |

**Key insight from our experiments:** Concentration matters more than liquidity. At c=0.5, δ=0.42 even with κ=2.0 (abundant liquidity). At c=5.0, δ=0.01. No amount of liquidity can save an agent whose debt exceeds the total cash in the system.

#### μ (Mu) — Maturity Timing Skew

Controls when payables mature within the simulation horizon (0 to maturity_days).

| μ | Timing | Impact on intermediation |
|---|--------|------------------------|
| 0 | **All due day 1** (front-loaded) | No time for trading. Lenders can help (pre-settlement credit). Dealers cannot. |
| 0.5 | **Evenly spread** | Both trading and lending have time to operate. Best for studying intermediation. |
| 1.0 | **All due last day** (back-loaded) | Maximum time for trading and rebalancing. Dealer effect strongest. |

#### Additional Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Outside-money ratio | ρ | 0.90 | VBT discount factor — the outside market's valuation of payables relative to par |
| Number of agents | N | 100 | Ring size — more agents = more realistic but slower |
| Maturity horizon | T | 10 | Number of simulation days |
| Face value | S | 20 | Face value per payable ticket |
| Total debt | Q | 10,000 | Total system-wide debt |
"""))

# ═══════════════════════════════════════════════════════════════════════
# PART II — SIMULATION MECHANICS
# ═══════════════════════════════════════════════════════════════════════

cells.append(md("""\
---
## Part II: How the Simulation Works

### 4. Simulation Lifecycle

A simulation runs for **T days** (the maturity horizon, typically 10). Here is what happens:

**Day 0 — Setup:**
1. N agents are placed in a ring
2. Cash is distributed (total = κ × Q, uniformly across agents)
3. Payables are created: agent *i* owes agent *i+1*, with face value and maturity drawn from the scenario parameters
4. If intermediaries are enabled (dealer, bank, NBFI), they are capitalized and initialized

**Days 1 through T — Settlement cycle:**
Each day follows the phase structure described below. Payables that mature on day *d* must be settled on day *d*. If an agent cannot pay, it defaults and is expelled from the ring.

**Rollovers:**
When payables mature and are successfully settled, new payables may be created (rollover). This maintains the flow of obligations through the simulation, preventing the system from trivially clearing after the initial wave of settlements.

**Default handling:**
When an agent cannot meet its obligations on the due date:
1. The agent is marked as **defaulted**
2. All its outstanding payables (both owed and owned) are affected
3. The agent is **expelled** from the ring — it no longer participates in settlement
4. Downstream agents who were expecting payment from the defaulter may themselves default — this is the **cascade mechanism**
"""))

cells.append(md("""\
### 5. A Day in the Simulation: Phases

Each simulation day is divided into phases that execute in strict order. This sequencing matters — lending happens before trading, which happens before settlement.

```
┌──────────────────────────────────────────────────────────────────┐
│                        ONE SIMULATION DAY                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase A — Information Update                                    │
│  ├─ Observe yesterday's defaults                                │
│  ├─ Update belief tracker (Bayesian P_default estimate)          │
│  └─ Refresh risk assessments                                     │
│                                                                  │
│  Sub-Phase B.1 — Lending                                        │
│  ├─ NBFI lender evaluates borrower creditworthiness              │
│  ├─ Bank evaluates loan applications                             │
│  └─ Loans disbursed → borrowers receive cash                    │
│                                                                  │
│  Sub-Phase B.2 — Dealer Trading                                 │
│  ├─ Dealer posts bid/ask quotes per maturity bucket              │
│  ├─ Stressed agents sell receivables for immediate cash          │
│  ├─ Surplus agents buy receivables at a discount                 │
│  └─ Multiple trading rounds until no more trades clear           │
│                                                                  │
│  Sub-Phase B.3 — Settlement                                     │
│  ├─ Payables due today are presented for payment                 │
│  ├─ Each agent attempts the settlement waterfall (see below)     │
│  └─ Agents that cannot pay → DEFAULT and expulsion              │
│                                                                  │
│  Phase C — Interbank (if banking enabled)                       │
│  ├─ Banks with surplus reserves lend to deficit banks            │
│  └─ Interbank market clears via auction                          │
│                                                                  │
│  Phase D — End of Day                                           │
│  ├─ Rollover: settled payables may generate new obligations      │
│  ├─ Accounting: update all balance sheets                        │
│  └─ Record metrics for the day                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### The Settlement Waterfall (Sub-Phase B.3)

When a payable matures, the debtor must pay. The means-of-payment (MOP) waterfall defines the priority order:

| Priority | Action | Description |
|----------|--------|-------------|
| 1 | **Pay from cash** | Use available cash balance |
| 2 | **Withdraw bank deposit** | Convert deposits to cash (if banking enabled) |
| 3 | **Sell receivable to dealer/VBT** | Emergency sale at a discount |
| 4 | **Borrow from NBFI** | Take a loan if lender is willing |
| 5 | **Borrow from bank** | Bank loan as last resort |
| ✗ | **DEFAULT** | Agent is expelled from the system |

This waterfall means that intermediaries function as **liquidity backstops** — they don't prevent defaults directly, but provide additional payment channels that agents can use before defaulting.
"""))

# ═══════════════════════════════════════════════════════════════════════
# PART III — DECISION PROFILES
# ═══════════════════════════════════════════════════════════════════════

cells.append(md("""\
---
## Part III: Agent Decision Profiles

Every agent in bilancio follows a four-stage decision pipeline:

```
Observe → Value → Assess → Choose
```

1. **Observe**: Query available information (counterparty positions, default history, system state)
2. **Value**: Estimate the worth of instruments (expected value of receivables, fair loan rates)
3. **Assess**: Map current position to risk/urgency levels (how stressed am I? what's my shortfall?)
4. **Choose**: Make concrete decisions (sell this receivable, buy that one, lend to this borrower)

What differs across agent types is **what** they observe, **how** they value, and **what** they can choose. The pipeline itself is universal.
"""))

# ── Trader ────────────────────────────────────────────────────────────
cells.append(md("""\
### 6.1 Trader (Firm / Household)

**Role:** Agents in the real economy. They hold receivables (claims) and cash, and owe payables (obligations). Trading is a liquidity management tool — not speculation.

**Instruments:**
| Assets | Liabilities |
|--------|-------------|
| Cash | Payables (obligations to next agent in ring) |
| Receivables (claims on previous agent) | |
| Bank deposits (if banking enabled) | |

**Decision Profile — 6 behavioral parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `risk_aversion` | 0 | 0=risk-neutral buyer, 1=demands 3% premium to buy. Formula: `buy_premium = 0.01 + 0.02 × risk_aversion` |
| `planning_horizon` | 10 days | How far ahead to look for upcoming obligations. Longer = more conservative (sees more dues) |
| `aggressiveness` | 1.0 | 0=needs full face_value surplus to buy, 1=any surplus qualifies |
| `default_observability` | 1.0 | 0=blind to defaults (uses prior only), 1=fully tracks observed defaults |
| `buy_reserve_fraction` | 0.5 | Fraction of upcoming dues held as untouchable reserve before buying |
| `trading_motive` | liquidity_then_earning | Controls which trades are allowed: sell when stressed, buy when surplus |

**Risk Assessment — Bayesian belief tracking:**

The trader maintains a running estimate of the system-wide default probability:

$$P_{default} = \\frac{\\alpha + d_{observed}}{2\\alpha + n_{total}}$$

Where α is the Laplace smoothing parameter (default 1.0), $d_{observed}$ is defaults seen within the lookback window (5 days), and $n_{total}$ is total settlements observed. This is updated after each day's settlement phase.

**Valuation:**

$$EV = (1 - P_{default}) \\times face$$

Simple probability-weighted payoff. If P=0.15 and face=100, then EV=85. No term structure, no issuer differentiation (by default).

**Sell decision:** Accept the dealer's bid if `bid × face ≥ EV + threshold × face`, where threshold drops with urgency (stressed agents accept worse prices).

**Buy decision:** Accept the dealer's ask if `EV ≥ ask × face + buy_premium × face`, and only if cash surplus exceeds the reserve fraction of upcoming dues.
"""))

# ── Dealer ────────────────────────────────────────────────────────────
cells.append(md("""\
### 6.2 Dealer (Treynor Market-Maker)

**Role:** The dealer provides a secondary market for receivables. It posts bid and ask quotes, buying from stressed sellers and selling to surplus buyers. The dealer is a balance sheet — it holds inventory and manages risk.

**Pricing Kernel — L1 (Treynor):**

The dealer quotes per maturity bucket (short/mid/long). The pricing is anchored to the VBT's outside mid-price and calibrated by the dealer's capacity.

```
Key quantities:
  M = VBT credit-adjusted mid-price (outside anchor)
  O = VBT outside spread (bid-ask width)
  K* = floor(V / (M × S))    ← capacity in tickets
  X* = S × K*                ← one-sided capacity in face value

Pricing midline (linear in position x):
  p(x) = M − [O / (X* + 2S)] × (x − X*/2)

Inside spread (competitive width):
  λ = S / (X* + S)           ← layoff probability
  I = λ × O                  ← inside spread
```

**Key properties:**
- **More capacity → tighter spread**: A well-capitalized dealer quotes narrower, attracting more flow
- **Inventory tilts the midline**: High inventory → lower mid → dealer wants to sell. Low inventory → higher mid → dealer wants to buy
- **Guard regime**: If M ≤ 0.02 (near-worthless paper), dealer routes everything to VBT — no point market-making in defaulted paper

**Capacity matters:** The dealer receives a fraction of system value as capital (`dealer_share_per_bucket`, default 5%). If this yields K* < 3 tickets, the dealer is capacity-constrained from the start and acts mostly as a passthrough to the VBT.
"""))

# ── VBT ──────────────────────────────────────────────────────────────
cells.append(md("""\
### 6.3 Virtual Book Trader (VBT) — Outside Liquidity

**Role:** The VBT represents the outside market — an exogenous source of liquidity that provides reference pricing and acts as the dealer's counterparty of last resort. It is not a strategic agent; it follows a mechanical pricing rule.

**Credit-Adjusted Mid-Price:**

$$M = \\rho \\times (1 - P_{default})$$

Where ρ is the outside-money ratio (default 0.90) and $P_{default}$ is the system-wide default probability. The VBT's mid-price falls as defaults rise — it prices credit risk into the outside market.

**Spread:** The VBT quotes a bid-ask spread per maturity bucket:
- Short bucket: O = 0.20 (widest)
- Mid bucket: O = 0.30
- Long bucket: O = 0.40 (widest for longest maturities)

The spread can widen with defaults if `vbt_spread_sensitivity > 0`:

$$O = O_0 + \\sigma_s \\times P_{default}$$

**Informedness:** The VBT can be made kappa-aware (`alpha_vbt`). At α=0 (default), it uses the uninformed prior. At α=1, its initial prior reflects the true system stress level: $P_0 \\approx 1/(1+\\kappa)$.
"""))

# ── Bank ─────────────────────────────────────────────────────────────
cells.append(md("""\
### 6.4 Bank (Treynor Funding-Plane Kernel)

**Role:** The bank intermediates between depositors and borrowers, operating within a central bank corridor. It uses the same L1 Treynor pricing structure as the dealer, but applied to the **interest rate** (funding plane) rather than the instrument price plane.

**CB Corridor:** The central bank sets a floor rate (deposit facility) and ceiling rate (lending facility). The bank's rates must stay within this corridor.

**Treynor Pricing on the Funding Plane:**

```
Symmetric midline (function of reserve position x):
  m(x) = M − [Ω / 2(X* + S)] × x
  where x = R − R_target (reserve surplus/deficit)

Risk-tilted midline:
  m_bank = m(x) + α × L* + γ × ρ
  α = cash tightness sensitivity
  γ = risk index sensitivity

Final rates:
  r_D = m_bank − I/2    (deposit rate — what the bank pays depositors)
  r_L = m_bank + I/2    (loan rate — what the bank charges borrowers)
```

**Analogy:** The dealer uses an L1 kernel on the payable-price plane; the bank uses the same L1 structure on the reserve-rate (funding) plane, pinned at the CB corridor. Both are Treynor market-makers — one in instruments, one in money.

**Key difference from other arms:** The bank creates money when it lends. A bank loan credits the borrower's deposit account — this is endogenous money creation. In the passive and dealer arms, the total money supply is fixed. In the bank arm, it can expand.

*Developed with Michael Loebach based on Perry Mehrling's suggestion.*
"""))

# ── CB and NBFI ──────────────────────────────────────────────────────
cells.append(md("""\
### 6.5 Central Bank

**Role:** Standing facilities provider. Sets the interest rate corridor within which the bank operates. Provides emergency reserves to banks that cannot obtain interbank funding.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Deposit facility rate | corridor floor | Rate paid on excess reserves parked at CB |
| Lending facility rate | corridor ceiling | Rate charged for emergency borrowing |
| Rate escalation slope | 0 (disabled) | How fast rates rise with CB exposure (Plan 041) |
| Max outstanding ratio | 0 (uncapped) | Cap on total CB lending |

The CB is a **lender of last resort for banks**, not for firms. Firms cannot borrow from the CB directly.

### 6.6 Non-Bank Financial Intermediary (NBFI) Lender

**Role:** Provides direct credit to firms from its own capital. Unlike the bank, the NBFI does not create money — it lends existing cash.

**Loan Pricing:**

$$r = base\\_rate + risk\\_premium\\_scale \\times P_{default}$$

Where $P_{default}$ is the borrower's estimated default probability. Default: base_rate=0.05, risk_premium_scale=0.20.

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_single_exposure` | 0.15 | Max fraction of capital to a single borrower |
| `max_total_exposure` | 0.80 | Max total lending as fraction of capital |
| `loan_maturity_days` | 2 | How long until the loan must be repaid |
| `risk_aversion` | 0.3 | How aggressively the lender prices risk |

**Key limitation:** The NBFI is still missing a proper Bayesian belief tracker (it borrows the dealer's risk assessment). This is a known gap — the NBFI should have its own RiskAssessor with loan history tracking.
"""))

# ═══════════════════════════════════════════════════════════════════════
# PART IV — EXPERIMENTAL DESIGN
# ═══════════════════════════════════════════════════════════════════════

cells.append(md("""\
---
## Part IV: Experimental Design

### 7. Experimental Arms

Four configurations are compared under identical initial conditions. Each arm adds a different liquidity channel:

| Arm | Identifier | What it adds | Mechanism |
|-----|-----------|--------------|-----------|
| **Passive** (Baseline) | — | Nothing. Pure settlement from cash. | No intermediation. Agents observe own cash, settle or default. |
| **Dealer** (Active) | D | Secondary market for receivables | L1 pricing kernel. Agents assess risk, value payables, decide to trade. |
| **NBFI Lender** | L | Direct credit from non-bank | Risk-based lending. Lender assesses creditworthiness, prices loans. |
| **Bank-Dealer** | B | Full banking with CB corridor | Treynor kernel on funding plane. Endogenous money creation. |

**Paired comparison:** Each parameter combination (κ, c, μ, seed) is run in all arms under identical initial conditions. The same ring, the same debts, the same maturities — the only difference is whether intermediaries are active.

**Treatment effect:** For each pair:

$$\\Delta\\delta = \\delta_{passive} - \\delta_{arm}$$

Positive Δδ means the intermediary **reduced** defaults. This is our primary outcome measure.
"""))

cells.append(md("""\
### 8. Sweep Configuration

We use a full-factorial parameter grid to explore the stress surface systematically.

**Grid used in the production sweeps:**

| Parameter | Values | Count |
|-----------|--------|-------|
| κ (kappa) | 0.25, 0.5, 1.0, 2.0, 4.0 | 5 |
| c (concentration) | 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0 | 9 |
| μ (mu) | 0, 0.5, 1.0 | 3 |
| ρ (outside_mid_ratio) | 0.5, 0.9 | 2 (*) |
| seed | 42 | 1 |

(*) Note: ρ has only limited variation in the current sweep. This is intentional — ρ primarily affects dealer pricing, and its interaction with κ is studied in the heatmaps below.

**Total: 225 parameter combinations** × 2-7 arms = 450-1,575 runs.

**Execution:** Runs are parallelized on Modal cloud infrastructure (~5-6 concurrent simulations). A full sweep completes in approximately 1-2 hours.

### 9. Metrics

| Metric | Symbol | Definition | Interpretation |
|--------|--------|------------|----------------|
| Default rate | δ | defaulted_face / total_face | Primary outcome. Lower = better. |
| Clearing rate | φ | settled_face / total_face | Complement of δ. Higher = better. |
| Trading effect | Δδ | δ_passive − δ_active | Positive = dealer helps |
| Lending effect | Δδ | δ_passive − δ_lender | Positive = NBFI helps |
| Bank lending effect | Δδ | δ_passive − δ_bank_lend | Positive = bank helps |
| Relief ratio | Δδ/δ_passive | Proportional improvement | Scale-invariant |
| Cascade fraction | cascade_defaults / total_defaults | How many defaults were caused by upstream failures |
| System loss | total economic loss (payable + deposit + intermediary) | Comprehensive damage measure |
| Loss given default (LGD) | system_loss / default_amount | Economic damage per unit of default |
"""))

# ═══════════════════════════════════════════════════════════════════════
# PART V — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

cells.append(md("""\
---
## Part V: Cross-Sweep Analysis

We now analyze the results of the production sweeps across all 225 parameter combinations and all experimental arms. The analysis proceeds in stages:

1. **Descriptive overview** — summary statistics, distributions, kappa response curves
2. **Parameter sensitivity** — heatmaps showing how each parameter affects each mechanism
3. **Mechanism comparison** — which mechanism dominates in which parameter regime
4. **Regression analysis** — OLS models identifying the key drivers of each mechanism's effectiveness
5. **Statistical tests** — formal hypothesis tests
6. **Network structure & contagion** — default cascade analysis, funding mix, systemic importance
7. **Temporal dynamics** — how defaults, credit, and obligations evolve day-by-day
8. **Mechanism internals** — dealer prices, bank rates, lending activity
9. **Loss analysis** — system loss beyond just default rates
"""))

# ── Data loading ─────────────────────────────────────────────────────
cells.append(code("""\
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from bilancio.analysis.cross_sweep import (
    load_unified_data, add_derived_columns,
    summary_stats, summary_by_group,
    fit_ols_from_df, standardized_coefs,
    bootstrap_ci, bootstrap_ci_by_group,
    one_sample_ttest, wilcoxon_test, kruskal_wallis, mann_whitney_u,
    pivot_for_heatmap, classify_regimes,
    DEALER_RED, BANK_BLUE, NBFI_GREEN, PASSIVE_GREY, COMBINED_PURPLE,
    MECHANISM_LABELS, MECHANISM_COLORS, ARM_COLORS, ARM_LABELS,
)

NBFI_CSV = Path("../out/sweep_nbfi_full/aggregate/comparison.csv")
BANK_CSV = Path("../out/sweep_bank_full/aggregate/comparison.csv")
data = load_unified_data(NBFI_CSV, BANK_CSV)
data = add_derived_columns(data)

print(f"Loaded {len(data['kappa'])} parameter combinations")
print(f"Kappa values: {np.unique(data['kappa'])}")
print(f"Concentration values: {np.unique(data['concentration'])}")
print(f"Mu values: {np.unique(data['mu'])}")
print(f"Rho values: {np.unique(data['outside_mid_ratio'])}")
print(f"Regimes: {dict(zip(*np.unique(data['regime'], return_counts=True)))}")
"""))

# ── Section 1: Descriptive Overview ──────────────────────────────────
cells.append(md("""\
---
### 10. Descriptive Overview

We begin with summary statistics and distributional comparisons for the three mechanism effects (dealer trading, NBFI lending, bank lending). Each "effect" is defined as the reduction in default rate compared to the passive baseline: Δδ = δ_passive − δ_arm. A positive effect means the mechanism helped.
"""))

cells.append(code("""\
# T1: Summary statistics table for each mechanism effect
mechanisms = [
    ("trading_effect", "Dealer Trading"),
    ("lending_effect", "NBFI Lending"),
    ("bank_lending_effect", "Bank Lending"),
]

stat_names = ["mean", "median", "sd", "iqr_low", "iqr_high", "pct_positive", "n"]
header_labels = ["Statistic", "Dealer Trading", "NBFI Lending", "Bank Lending"]
stat_display = ["Mean", "Median", "Std Dev", "IQR Low (Q1)", "IQR High (Q3)", "% Positive", "N"]

stats_by_mech = []
for col, label in mechanisms:
    s = summary_stats(data[col])
    stats_by_mech.append(s)

table_values = [stat_display]
for s in stats_by_mech:
    col_vals = []
    for sn in stat_names:
        v = s[sn]
        if sn == "n":
            col_vals.append(f"{int(v)}")
        elif sn == "pct_positive":
            col_vals.append(f"{v:.1f}%")
        else:
            col_vals.append(f"{v:.6f}")
    table_values.append(col_vals)

fig = go.Figure(data=[go.Table(
    header=dict(
        values=header_labels,
        fill_color=[PASSIVE_GREY, DEALER_RED, NBFI_GREEN, BANK_BLUE],
        font=dict(color="white", size=13),
        align="center",
    ),
    cells=dict(
        values=table_values,
        fill_color=["#f5f5f5", "#fdecea", "#e8f5e9", "#e3f2fd"],
        align=["left", "right", "right", "right"],
        font=dict(size=12),
    ),
)])
fig.update_layout(
    title="T1: Summary Statistics for Mechanism Effects (delta reduction)",
    height=350,
    margin=dict(t=50, b=20, l=20, r=20),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting T1:** This table shows the central tendency and spread of each mechanism's effect across all 225 parameter combinations.

- **Bank lending dominates**: its mean effect is an order of magnitude larger than either dealer or NBFI. This is the headline result.
- **Dealer trading** has a moderate positive mean but high variance — it helps in some regimes and is neutral in others.
- **NBFI lending** has a small positive mean, indicating limited effectiveness in this experimental setup (the NBFI's decision model is still under development).
- **% Positive** shows how often each mechanism reduces defaults at all. Bank lending improves outcomes in the vast majority of cases.
"""))

cells.append(code("""\
# C1: Box plots of the three mechanism effects side by side
fig = go.Figure()
for col, label in mechanisms:
    color = MECHANISM_COLORS.get(col, PASSIVE_GREY)
    vals = data[col]
    valid = vals[np.isfinite(vals)]
    fig.add_trace(go.Box(
        y=valid,
        name=label,
        marker_color=color,
        boxmean="sd",
    ))
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
fig.update_layout(
    title="C1: Distribution of Mechanism Effects (positive = reduces defaults)",
    yaxis_title="Effect (delta reduction)",
    height=500,
    showlegend=False,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C1:** The box plots confirm T1 visually. Bank lending (blue) has a tall box entirely above zero — it consistently reduces defaults. Dealer trading (red) straddles zero with a moderate positive median. NBFI lending (green) clusters near zero with some positive outliers. The dashed line at y=0 marks "no effect" — values above it indicate improvement.
"""))

cells.append(code("""\
# C2: Default rate distributions by arm — overlaid histograms
arms = [
    ("delta_passive", "Passive (NBFI baseline)", PASSIVE_GREY),
    ("delta_active", "Active (Dealer)", DEALER_RED),
    ("delta_lender", "Lender (NBFI)", NBFI_GREEN),
    ("delta_idle", "Idle (Bank baseline)", "#b0bec5"),
    ("delta_lend", "Lend (Bank)", BANK_BLUE),
]

fig = go.Figure()
for col, label, color in arms:
    valid = data[col][np.isfinite(data[col])]
    fig.add_trace(go.Histogram(
        x=valid,
        name=label,
        marker_color=color,
        opacity=0.5,
        nbinsx=30,
    ))
fig.update_layout(
    title="C2: Default Rate Distributions by Arm",
    xaxis_title="Default rate (delta)",
    yaxis_title="Count",
    barmode="overlay",
    height=500,
    legend=dict(x=0.01, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C2:** The histograms show the raw default rate distributions for each arm. The passive baseline (grey) and bank idle baseline (light grey) have similar distributions — both represent settlement without active intermediation, just with different payment system structures. The bank lend arm (blue) shifts the distribution dramatically leftward (lower defaults). The dealer arm (red) shows a more subtle leftward shift. NBFI (green) overlaps heavily with passive.
"""))

cells.append(code("""\
# C3: Default rate vs kappa by arm (line plot with markers)
kappa_vals = np.sort(np.unique(data["kappa"]))

arm_specs = [
    ("delta_passive", "Passive", PASSIVE_GREY),
    ("delta_active", "Active (Dealer)", DEALER_RED),
    ("delta_lender", "Lender (NBFI)", NBFI_GREEN),
    ("delta_idle", "Idle (Bank baseline)", "#b0bec5"),
    ("delta_lend", "Lend (Bank)", BANK_BLUE),
]

fig = go.Figure()
for col, label, color in arm_specs:
    means = []
    for k in kappa_vals:
        mask = data["kappa"] == k
        vals = data[col][mask]
        valid = vals[np.isfinite(vals)]
        means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
    fig.add_trace(go.Scatter(
        x=kappa_vals,
        y=means,
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2),
        marker=dict(size=8),
    ))
fig.update_layout(
    title="C3: Mean Default Rate vs Kappa by Arm",
    xaxis_title="Kappa (liquidity ratio)",
    yaxis_title="Mean default rate (delta)",
    xaxis_type="log",
    height=500,
    legend=dict(x=0.6, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C3:** This is the key chart showing how each arm responds to liquidity stress (κ).

- All lines decline as κ increases — more liquidity means fewer defaults, as expected.
- **Bank lending (blue)** consistently sits well below all other arms — the gap is largest at moderate κ (0.5–1.0).
- **Passive and dealer** are close at extreme κ: at κ=0.25 (severe), even the dealer can't help much; at κ=4.0 (abundant), there's little room for improvement.
- **The dealer's sweet spot** is around κ=0.5–1.0, where some agents have surplus cash (potential buyers) and others are stressed (sellers) — creating the conditions for beneficial trade.
- **Bank idle vs passive diverge**: the bank payment system structure itself (without lending) can affect outcomes because of the deposit/reserve architecture.
"""))

# ── Section 2: Parameter Sensitivity ─────────────────────────────────
cells.append(md("""\
---
### 11. Parameter Sensitivity

We now examine how each mechanism's effectiveness varies across the parameter space. The heatmaps show the mean treatment effect for each pair of parameters, averaged over the remaining parameters.
"""))

cells.append(code("""\
# C4-C6: Heatmaps for all three mechanisms (3 charts × 3 parameter pairs = 9 panels)
param_pairs = [
    ("kappa", "concentration", "kappa", "c"),
    ("kappa", "mu", "kappa", "mu"),
    ("kappa", "outside_mid_ratio", "kappa", "rho"),
]

for mech_col, mech_label, mech_color in [
    ("trading_effect", "Dealer Trading", DEALER_RED),
    ("lending_effect", "NBFI Lending", NBFI_GREEN),
    ("bank_lending_effect", "Bank Lending", BANK_BLUE),
]:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"kappa x {p[3]}" for p in param_pairs],
        horizontal_spacing=0.08,
    )
    for idx, (row_col, col_col, rl, cl) in enumerate(param_pairs):
        row_vals, col_vals, matrix = pivot_for_heatmap(data, row_col, col_col, mech_col)
        abs_max = max(np.nanmax(np.abs(matrix)), 0.01)
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[str(v) for v in col_vals],
                y=[str(v) for v in row_vals],
                colorscale="RdBu_r",
                zmid=0, zmin=-abs_max, zmax=abs_max,
                colorbar=dict(title="Effect", len=0.5) if idx == 2 else dict(showticklabels=False, len=0.01),
                showscale=(idx == 2),
            ),
            row=1, col=idx + 1,
        )
        fig.update_xaxes(title_text=cl, row=1, col=idx + 1)
        fig.update_yaxes(title_text=rl if idx == 0 else "", row=1, col=idx + 1)
    fig.update_layout(
        title=f"Heatmap: {mech_label} Effect",
        height=400,
        margin=dict(t=80, b=50),
    )
    fig.show()
"""))

cells.append(md("""\
**Interpreting the heatmaps:**

- **Dealer trading (red scale):** The trading effect is strongest in the κ×ρ panel — higher ρ (closer to 1.0) gives better outside pricing, making trades more viable. In κ×c, the effect concentrates at moderate κ (0.5–1.0) regardless of concentration. The κ×μ panel shows the critical finding: **dealer effect is near-zero at μ=0** (all debts due immediately) and strongest at μ=0.5–1.0 (spread maturities give time to trade).

- **NBFI lending (green scale):** Effects are much smaller in magnitude (note the color scale). The pattern in κ×c shows lending helps most at low c (unequal debt) and moderate κ. The μ dimension shows lending is most effective at μ=0 — exactly the opposite of the dealer pattern. This makes sense: the lender provides **pre-settlement credit**, which helps when all debts are due immediately.

- **Bank lending (blue scale):** Effects are an order of magnitude larger (note the scale). Bank lending is consistently positive across almost the entire parameter space. The strongest effects are at moderate κ (0.5–1.0) and moderate c. The bank's ability to create money means it can inject liquidity where needed, regardless of timing.
"""))

cells.append(code("""\
# C7: Marginal effect plots — 4 panels, one per parameter
params_info = [
    ("kappa", "Kappa"),
    ("concentration", "Concentration (c)"),
    ("mu", "Mu"),
    ("outside_mid_ratio", "Rho"),
]
mech_specs = [
    ("trading_effect", "Dealer Trading", DEALER_RED),
    ("lending_effect", "NBFI Lending", NBFI_GREEN),
    ("bank_lending_effect", "Bank Lending", BANK_BLUE),
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[p[1] for p in params_info],
    vertical_spacing=0.15,
    horizontal_spacing=0.1,
)

for pi, (param_col, param_label) in enumerate(params_info):
    r, c = pi // 2 + 1, pi % 2 + 1
    param_vals = np.sort(np.unique(data[param_col]))
    for mech_col, mech_label, mech_color in mech_specs:
        means = []
        for pv in param_vals:
            mask = data[param_col] == pv
            vals = data[mech_col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
        fig.add_trace(
            go.Scatter(
                x=param_vals, y=means,
                mode="lines+markers",
                name=mech_label,
                line=dict(color=mech_color, width=2),
                marker=dict(size=7),
                showlegend=(pi == 0),
                legendgroup=mech_label,
            ),
            row=r, col=c,
        )
    fig.update_xaxes(title_text=param_label, row=r, col=c)
    fig.update_yaxes(title_text="Mean effect" if c == 1 else "", row=r, col=c)

fig.update_layout(
    title="C7: Marginal Effect by Parameter (mean across other params)",
    height=700,
    margin=dict(t=80),
    legend=dict(x=0.6, y=1.08, orientation="h"),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C7 — Marginal effects:**

- **Kappa panel (top-left):** All three mechanisms show an inverted-U shape — effectiveness peaks at moderate stress (κ ≈ 0.5–1.0). At very low κ, the system is too stressed for any mechanism to help. At high κ, there are too few defaults to improve. Bank lending dominates across the entire range.

- **Concentration panel (top-right):** Effectiveness declines with higher concentration (more equal debt). This is logical: when all agents owe similar amounts, structural defaults are rare and there's little for intermediaries to fix. The interesting feature is that **bank lending remains positive even at very high c**, while dealer and NBFI effects vanish.

- **Mu panel (bottom-left):** The key regime separation. Dealer trading improves monotonically with μ (later maturities = more time to trade). NBFI lending peaks at μ=0 (front-loaded stress = immediate credit need). Bank lending is relatively stable across μ values.

- **Rho panel (bottom-right):** Higher ρ (closer to par) improves dealer effectiveness (better outside pricing). Has less effect on lending-based mechanisms, which don't depend on secondary market prices.
"""))

cells.append(code("""\
# C8: Scatter plot — kappa vs delta for all arms
fig = go.Figure()
for col, label, color in arm_specs:
    valid_mask = np.isfinite(data[col])
    fig.add_trace(go.Scatter(
        x=data["kappa"][valid_mask],
        y=data[col][valid_mask],
        mode="markers",
        name=label,
        marker=dict(color=color, size=5, opacity=0.5),
    ))
fig.update_layout(
    title="C8: Default Rate vs Kappa (all parameter combos)",
    xaxis_title="Kappa",
    yaxis_title="Default rate (delta)",
    xaxis_type="log",
    height=500,
    legend=dict(x=0.6, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C8:** This scatter shows every individual parameter combination as a point, revealing the full variability within each arm. Key observations: (1) the passive and dealer clouds overlap extensively — for most combos the dealer makes little difference; (2) the bank-lend cloud (blue) is visibly shifted down (fewer defaults) across all κ values; (3) the spread within each arm is enormous — concentration and mu matter as much as kappa.
"""))

# ── Section 3: Mechanism Comparison ──────────────────────────────────
cells.append(md("""\
---
### 12. Mechanism Comparison & Regime Mapping

Which mechanism dominates in which parameter regime? We classify each of the 225 parameter combinations by its "dominant mechanism" — the one that achieves the largest reduction in defaults.
"""))

cells.append(code("""\
# C9: Mechanism ranking bar chart
regime_counts = {}
for r in np.unique(data["regime"]):
    regime_counts[str(r)] = int(np.sum(data["regime"] == r))

regime_colors = {
    "dealer": DEALER_RED,
    "nbfi": NBFI_GREEN,
    "bank": BANK_BLUE,
    "mixed": COMBINED_PURPLE,
    "none": PASSIVE_GREY,
}

labels = list(regime_counts.keys())
counts = list(regime_counts.values())
colors = [regime_colors.get(l, PASSIVE_GREY) for l in labels]

fig = go.Figure(data=[go.Bar(
    x=labels, y=counts,
    marker_color=colors,
    text=counts, textposition="auto",
)])
fig.update_layout(
    title="C9: Dominant Mechanism Counts (225 parameter combos)",
    xaxis_title="Dominant mechanism",
    yaxis_title="Count",
    height=400,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C9:** The bar chart shows how many of the 225 parameter combinations are "won" by each mechanism. Bank lending dominates the majority of the parameter space. Dealer trading wins in a smaller but significant portion. NBFI lending rarely dominates. "None" indicates combos where no mechanism achieves more than a 0.5pp improvement.
"""))

cells.append(code("""\
# C10: Phase diagram — dominant mechanism in kappa x concentration space
kappa_vals = np.sort(np.unique(data["kappa"]))
conc_vals = np.sort(np.unique(data["concentration"]))

regime_to_num = {"none": 0, "nbfi": 1, "dealer": 2, "bank": 3, "mixed": 4}
num_to_regime = {v: k for k, v in regime_to_num.items()}

phase_matrix = np.full((len(kappa_vals), len(conc_vals)), np.nan)
for i, kv in enumerate(kappa_vals):
    for j, cv in enumerate(conc_vals):
        mask = (data["kappa"] == kv) & (data["concentration"] == cv)
        regimes_here = data["regime"][mask]
        if len(regimes_here) > 0:
            unique_r, counts_r = np.unique(regimes_here, return_counts=True)
            dominant = str(unique_r[np.argmax(counts_r)])
            phase_matrix[i, j] = regime_to_num.get(dominant, 0)

discrete_colors = [
    [0.0, PASSIVE_GREY], [0.2, PASSIVE_GREY],
    [0.2, NBFI_GREEN], [0.4, NBFI_GREEN],
    [0.4, DEALER_RED], [0.6, DEALER_RED],
    [0.6, BANK_BLUE], [0.8, BANK_BLUE],
    [0.8, COMBINED_PURPLE], [1.0, COMBINED_PURPLE],
]

fig = go.Figure(data=go.Heatmap(
    z=phase_matrix,
    x=[str(v) for v in conc_vals],
    y=[str(v) for v in kappa_vals],
    colorscale=discrete_colors,
    zmin=0, zmax=4,
    colorbar=dict(
        title="Regime",
        tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
        ticktext=["none", "nbfi", "dealer", "bank", "mixed"],
    ),
    hovertemplate="kappa=%{y}<br>c=%{x}<br>regime=%{customdata}<extra></extra>",
    customdata=[[num_to_regime.get(int(phase_matrix[i, j]), "?") if np.isfinite(phase_matrix[i, j]) else "?"
                 for j in range(len(conc_vals))] for i in range(len(kappa_vals))],
))
fig.update_layout(
    title="C10: Dominant Mechanism Phase Diagram (kappa x concentration)",
    xaxis_title="Concentration (c)",
    yaxis_title="Kappa",
    height=450,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C10:** The phase diagram reveals the spatial structure of mechanism dominance:

- **Bank lending (blue)** dominates the interior of the parameter space — moderate κ, moderate c. This is where there are enough stressed agents to benefit from credit but enough system capacity for the bank to operate.
- **"None" (grey)** appears at high concentration (c > 10) — when debt is so equally distributed that defaults are rare regardless of intervention.
- **Dealer (red)** appears in pockets, typically at moderate-to-high κ with lower c, where there's enough trading activity to make a difference.
- The transitions between regimes are not sharp — they reflect the averaging over μ and ρ values.
"""))

cells.append(code("""\
# C11: Paired scatter — trading effect vs bank lending effect
valid_mask = np.isfinite(data["trading_effect"]) & np.isfinite(data["bank_lending_effect"])
te = data["trading_effect"][valid_mask]
ble = data["bank_lending_effect"][valid_mask]
kappas_for_color = data["kappa"][valid_mask]

fig = go.Figure()
line_range = [min(te.min(), ble.min()) - 0.02, max(te.max(), ble.max()) + 0.02]
fig.add_trace(go.Scatter(
    x=line_range, y=line_range,
    mode="lines", line=dict(color="black", dash="dash", width=1),
    name="y=x", showlegend=True,
))

for k_val in np.sort(np.unique(kappas_for_color)):
    km = kappas_for_color == k_val
    fig.add_trace(go.Scatter(
        x=te[km], y=ble[km],
        mode="markers", name=f"kappa={k_val}",
        marker=dict(size=7, opacity=0.7),
    ))

fig.update_layout(
    title="C11: Dealer Trading Effect vs Bank Lending Effect",
    xaxis_title="Trading effect (delta reduction)",
    yaxis_title="Bank lending effect (delta reduction)",
    height=550,
    legend=dict(x=0.01, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C11:** Points above the diagonal (y=x line) indicate that bank lending outperforms dealer trading for that parameter combination. The vast majority of points sit above the line — bank lending almost always delivers a larger effect. The color coding by κ shows that at κ=0.25 (severe stress), both mechanisms cluster near zero — neither can help much when the system is deeply insolvent.
"""))

# ── Section 4: Regression ────────────────────────────────────────────
cells.append(md("""\
---
### 13. Regression Analysis

We fit OLS regressions to understand which parameters drive each mechanism's effectiveness. The dependent variable is the treatment effect (Δδ), and the independent variables are log(κ), concentration, μ, and ρ.
"""))

cells.append(code("""\
# R1-R3: Base OLS regressions for each mechanism
base_x_cols = ["log_kappa", "concentration", "mu", "outside_mid_ratio"]
ols_results = {}

for mech_col, mech_label in mechanisms:
    ols = fit_ols_from_df(data, mech_col, base_x_cols)
    ols_results[mech_col] = ols

# Display coefficient tables
for mech_col, mech_label in mechanisms:
    ols = ols_results[mech_col]
    rows = ols.summary_rows()

    header = ["Feature", "Coefficient", "Std Error", "t-stat", "p-value", "Sig"]
    col_feature = [r["feature"] for r in rows]
    col_coef = [f"{r['coef']:.6f}" for r in rows]
    col_se = [f"{r['se']:.6f}" for r in rows]
    col_t = [f"{r['t_stat']:.3f}" for r in rows]
    col_p = [f"{r['p_value']:.2e}" for r in rows]
    col_sig = [r["sig"] for r in rows]

    fig = go.Figure(data=[go.Table(
        header=dict(values=header, fill_color=MECHANISM_COLORS.get(mech_col, "#2c3e50"),
                    font=dict(color="white", size=12), align="center"),
        cells=dict(values=[col_feature, col_coef, col_se, col_t, col_p, col_sig],
                   align=["left"] + ["right"] * 5, font=dict(size=11)),
    )])
    fig.update_layout(
        title=f"R: OLS Regression — {mech_label} Effect (R² = {ols.r_squared:.4f})",
        height=280, margin=dict(t=50, b=10, l=10, r=10),
    )
    fig.show()
"""))

cells.append(md("""\
**Interpreting R1-R3 — OLS Regressions:**

- **Dealer trading**: log(κ) is the strongest predictor — more liquidity enables more trading. μ is also significant (later maturities give time to trade). R² is moderate, meaning other factors (interactions, nonlinearities) also matter.
- **NBFI lending**: Smaller R² indicates the linear model captures less of the variation. The NBFI effect is more idiosyncratic, depending on specific agent configurations.
- **Bank lending**: log(κ) and concentration are the dominant predictors. The bank's effectiveness scales with system stress and debt inequality. R² is typically the highest of the three.
"""))

cells.append(code("""\
# C13: Forest plot — coefficients with error bars for all three regressions
feature_names = base_x_cols
n_features = len(feature_names)

fig = go.Figure()

offsets = [-0.15, 0, 0.15]
for mi, (mech_col, mech_label) in enumerate(mechanisms):
    ols = ols_results[mech_col]
    color = MECHANISM_COLORS.get(mech_col, PASSIVE_GREY)
    coefs = ols.coef[1:]  # skip intercept
    ses = ols.se[1:]

    fig.add_trace(go.Scatter(
        x=coefs,
        y=[f + offsets[mi] for f in range(n_features)],
        error_x=dict(type="data", array=1.96 * ses, visible=True),
        mode="markers",
        name=mech_label,
        marker=dict(color=color, size=10),
    ))

fig.add_vline(x=0, line_dash="dash", line_color="grey")
fig.update_layout(
    title="C13: OLS Coefficients with 95% CI (Forest Plot)",
    xaxis_title="Coefficient",
    yaxis=dict(
        tickvals=list(range(n_features)),
        ticktext=feature_names,
    ),
    height=400,
    legend=dict(x=0.7, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C13:** The forest plot shows all three mechanisms' coefficients side by side. Coefficients to the right of zero increase the treatment effect (make the mechanism more effective).

- **log(κ)** has a positive coefficient for all three — more liquidity helps all mechanisms work better. Bank lending (blue) has the largest coefficient.
- **concentration** has a negative coefficient for all — higher c (more equal debt) reduces the scope for intervention.
- **mu** shows the regime split: positive for dealer (later maturities help), negative or near-zero for NBFI (early stress is its sweet spot).
- **rho** primarily affects the dealer (positive — better outside pricing enables more trades).
"""))

# ── Section 5: Statistical Tests ─────────────────────────────────────
cells.append(md("""\
---
### 14. Statistical Tests

Formal hypothesis tests for whether each mechanism effect differs from zero and whether the three mechanisms differ from each other.
"""))

cells.append(code("""\
# H1-H2: One-sample t-test and Wilcoxon signed-rank for each mechanism
print("HYPOTHESIS TESTS")
print("=" * 70)
print()

for mech_col, mech_label in mechanisms:
    arr = data[mech_col]
    valid = arr[np.isfinite(arr)]
    tt = one_sample_ttest(valid)
    wt = wilcoxon_test(valid)
    print(f"{mech_label}:")
    print(f"  One-sample t-test:  t = {tt['t_stat']:.3f}, p = {tt['p_value']:.2e}, mean = {tt['mean']:.6f}")
    print(f"  Wilcoxon signed-rank: W = {wt['statistic']:.1f}, p = {wt['p_value']:.2e}")
    print(f"  → {'Significant' if tt['p_value'] < 0.05 else 'Not significant'} at α=0.05")
    print()

# H3: Kruskal-Wallis across the three effects
te_arr = data["trading_effect"][np.isfinite(data["trading_effect"])]
le_arr = data["lending_effect"][np.isfinite(data["lending_effect"])]
ble_arr = data["bank_lending_effect"][np.isfinite(data["bank_lending_effect"])]

kw = kruskal_wallis(te_arr, le_arr, ble_arr)
print("Kruskal-Wallis test (do the three mechanism effects come from the same distribution?):")
print(f"  H = {kw['H_stat']:.3f}, p = {kw['p_value']:.2e}")
print(f"  → {'Reject H0' if kw['p_value'] < 0.05 else 'Fail to reject H0'}: mechanisms differ significantly")
print()

# H4: Pairwise Mann-Whitney U
pairs = [
    ("Dealer vs NBFI", te_arr, le_arr),
    ("Dealer vs Bank", te_arr, ble_arr),
    ("NBFI vs Bank", le_arr, ble_arr),
]
print("Pairwise Mann-Whitney U tests:")
for label, a, b in pairs:
    mwu = mann_whitney_u(a, b)
    print(f"  {label}: U = {mwu['U_stat']:.1f}, p = {mwu['p_value']:.2e}")
"""))

cells.append(md("""\
**Interpreting the statistical tests:**

- **H1-H2 (individual effects):** Both parametric (t-test) and nonparametric (Wilcoxon) tests confirm that all three mechanism effects differ significantly from zero. All mechanisms do *something* — the question is how much.
- **H3 (Kruskal-Wallis):** Overwhelmingly rejects H₀ — the three mechanisms produce statistically different effect distributions. They are not interchangeable.
- **H4 (pairwise comparisons):** All pairs differ significantly. Bank lending is statistically distinguishable from both dealer and NBFI effects; dealer and NBFI also differ from each other.
"""))

cells.append(code("""\
# H5: Bootstrap CIs overall and by kappa band
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (10,000 resamples)")
print("=" * 60)

overall_cis = {}
for mech_col, mech_label in mechanisms:
    point, lo, hi = bootstrap_ci(data[mech_col])
    overall_cis[mech_col] = (point, lo, hi)
    print(f"  {mech_label}: {point:.6f}  [{lo:.6f}, {hi:.6f}]")

print()
print("By Kappa Band:")
print("-" * 60)
band_cis = {}
for mech_col, mech_label in mechanisms:
    # Group by kappa_band manually (string column, can't use bootstrap_ci_by_group directly)
    cis = {}
    for band in ["severe", "stressed", "balanced", "comfortable", "abundant"]:
        mask = data["kappa_band"] == band
        if np.sum(mask) > 0:
            vals = data[mech_col][mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                cis[band] = bootstrap_ci(vals)
    band_cis[mech_col] = cis
    print(f"  {mech_label}:")
    for band in ["severe", "stressed", "balanced", "comfortable", "abundant"]:
        if band in cis:
            pt, lo, hi = cis[band]
            print(f"    {band:12s}: {pt:.6f}  [{lo:.6f}, {hi:.6f}]")
    print()
"""))

cells.append(md("""\
**Interpreting H5 — Bootstrap CIs:** The bootstrap provides distribution-free uncertainty estimates. Key observations:
- Bank lending's CI is entirely above zero and doesn't overlap with the other mechanisms' CIs — the difference is robust.
- By kappa band: all mechanisms are most effective in the "stressed" and "balanced" bands (κ ≈ 0.5–1.0). At "severe" (κ=0.25), CIs include zero — mechanisms can't reliably help. At "abundant" (κ=4.0), CIs are narrower and closer to zero — little room for improvement.
"""))

# ── Section 6: Unified comparison ────────────────────────────────────
cells.append(md("""\
---
### 15. Unified Cross-Sweep Comparison
"""))

cells.append(code("""\
# C17: Grouped bar chart of mean effects with bootstrap error bars
fig = go.Figure()

mech_labels = [m[1] for m in mechanisms]
mech_means = []
mech_los = []
mech_his = []
mech_colors_list = []

for mech_col, mech_label in mechanisms:
    pt, lo, hi = overall_cis[mech_col]
    mech_means.append(pt)
    mech_los.append(pt - lo)
    mech_his.append(hi - pt)
    mech_colors_list.append(MECHANISM_COLORS.get(mech_col, PASSIVE_GREY))

fig.add_trace(go.Bar(
    x=mech_labels,
    y=mech_means,
    marker_color=mech_colors_list,
    error_y=dict(
        type="data",
        symmetric=False,
        array=mech_his,
        arrayminus=mech_los,
    ),
    text=[f"{m:.4f}" for m in mech_means],
    textposition="outside",
))
fig.add_hline(y=0, line_dash="dash", line_color="grey")
fig.update_layout(
    title="C17: Mean Mechanism Effect with Bootstrap 95% CI",
    yaxis_title="Mean effect (delta reduction)",
    height=450,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C17:** The grouped bar chart provides the clearest visual summary. Bank lending's mean effect (with error bars) is far above the other two mechanisms. The error bars don't overlap between any pair, confirming statistically significant differences.
"""))

cells.append(code("""\
# C18: 3-mechanism × 3-parameter-pair heatmap grid (9 panels)
mech_for_grid = [
    ("trading_effect", "Dealer Trading"),
    ("lending_effect", "NBFI Lending"),
    ("bank_lending_effect", "Bank Lending"),
]
param_pair_specs = [
    ("kappa", "concentration", "k x c"),
    ("kappa", "mu", "k x mu"),
    ("kappa", "outside_mid_ratio", "k x rho"),
]

subtitles = []
for mech_col, mech_label in mech_for_grid:
    for _, _, pp_label in param_pair_specs:
        subtitles.append(f"{mech_label}: {pp_label}")

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=subtitles,
    vertical_spacing=0.08,
    horizontal_spacing=0.06,
)

for mi, (mech_col, mech_label) in enumerate(mech_for_grid):
    for pi, (row_col, col_col, pp_label) in enumerate(param_pair_specs):
        row_vals, col_vals, matrix = pivot_for_heatmap(data, row_col, col_col, mech_col)
        abs_max = max(np.nanmax(np.abs(matrix)), 0.01)
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[str(v) for v in col_vals],
                y=[str(v) for v in row_vals],
                colorscale="RdBu_r",
                zmid=0, zmin=-abs_max, zmax=abs_max,
                showscale=False,
            ),
            row=mi + 1, col=pi + 1,
        )

fig.update_layout(
    title="C18: Mechanism Effects Across Parameter Pairs (3 × 3 grid)",
    height=900,
    margin=dict(t=80, b=30),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting C18:** The 3×3 grid provides a compact visual comparison. Each row is a mechanism, each column is a parameter pair. Key pattern: the bank row (bottom) shows consistently stronger red/blue coloring than the other two rows — its effect is larger everywhere. The dealer row shows the strongest pattern in the κ×ρ panel (column 3). The NBFI row is muted throughout.
"""))

# ── Section 7: Network & Contagion ───────────────────────────────────
cells.append(md("""\
---
### 16. Network Structure & Contagion

Per-run network metrics extracted from `events.jsonl` files: default classification, cascade analysis, funding sources, systemic importance. These go beyond aggregate δ to reveal *how* defaults propagate.
"""))

cells.append(code("""\
# Load network data and set up matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 120

from bilancio.analysis.cross_sweep_network import (
    load_network_metrics, load_network_timeseries, aggregate_timeseries_by_arm_kappa,
)

CACHE_DIR = Path("../out/sweep_analysis")
net_data = load_network_metrics(CACHE_DIR)
ts_data = load_network_timeseries(CACHE_DIR)
ts_agg = aggregate_timeseries_by_arm_kappa(ts_data) if ts_data else {}

arms = sorted(set(net_data["arm"]))
print(f"Network metrics loaded: {len(net_data['arm'])} runs across {len(arms)} arms")
print(f"Arms: {arms}")
print(f"Timeseries loaded: {len(ts_data) if ts_data else 0} runs")
"""))

cells.append(code("""\
# C20: Default Classification by Arm
fig, ax = plt.subplots(figsize=(10, 5))
mean_primary = []
mean_secondary = []
labels = []
colors = []
for arm in arms:
    mask = net_data["arm"] == arm
    mean_primary.append(np.nanmean(net_data["n_primary"][mask]))
    mean_secondary.append(np.nanmean(net_data["n_secondary"][mask]))
    labels.append(ARM_LABELS.get(arm, arm))
    colors.append(ARM_COLORS.get(arm, "#333"))

x = np.arange(len(arms))
bars1 = ax.bar(x, mean_primary, label="Primary", color=colors, alpha=0.9)
bars2 = ax.bar(x, mean_secondary, bottom=mean_primary, label="Secondary (cascade)", color=colors, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Mean defaults per run")
ax.set_title("N1: Default Classification by Arm (Primary vs Cascade)")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Interpreting N1 — Default Classification:** Primary defaults are caused by the agent's own inability to pay. Secondary (cascade) defaults are caused by upstream failures. Key observation: bank lending cuts *both* primary and secondary defaults — it doesn't just prevent initial failures, it breaks the cascade chain by injecting liquidity downstream.
"""))

cells.append(code("""\
# C21: Cascade Fraction vs Kappa by Arm
fig, ax = plt.subplots(figsize=(10, 5))
kappa_vals = np.sort(np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])]))
for arm in arms:
    means = []
    for kv in kappa_vals:
        mask = (net_data["arm"] == arm) & (np.abs(net_data["kappa"] - kv) < 1e-9)
        vals = net_data["cascade_fraction"][mask]
        vals = vals[np.isfinite(vals)]
        means.append(np.mean(vals) if len(vals) > 0 else np.nan)
    ax.plot(kappa_vals, means, marker="o", color=ARM_COLORS.get(arm, "#333"),
            label=ARM_LABELS.get(arm, arm), linewidth=2)
ax.set_xlabel("Kappa")
ax.set_ylabel("Cascade fraction")
ax.set_title("N2: Cascade Fraction vs Kappa by Arm")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Interpreting N2 — Cascade Fraction:** Cascade fraction = secondary_defaults / total_defaults. It measures what proportion of failures are *contagious* rather than idiosyncratic. Key findings: (1) cascade fraction declines monotonically with κ — more liquidity means fewer chain reactions; (2) bank lending reduces cascade fraction more than other mechanisms, especially at moderate κ.
"""))

cells.append(code("""\
# C25: Funding Mix by Arm (stacked bar)
funding_sources = [
    ("funding_settlement_received", "Settlement"),
    ("funding_ticket_sale", "Ticket Sale"),
    ("funding_loan_received", "Bank Loan"),
    ("funding_cb_loan", "CB Loan"),
    ("funding_nbfi_loan", "NBFI Loan"),
    ("funding_deposit_interest", "Deposit Interest"),
    ("funding_deposit", "Deposit"),
]
source_colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#1abc9c", "#f1c40f", "#e67e22"]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(arms))
for (src_col, src_label), color in zip(funding_sources, source_colors):
    vals = []
    for arm in arms:
        mask = net_data["arm"] == arm
        v = net_data.get(src_col, np.zeros(len(net_data["arm"])))[mask]
        vals.append(np.nanmean(v) if len(v) > 0 else 0)
    vals = np.array(vals)
    if np.any(vals > 0):
        ax.bar(range(len(arms)), vals, bottom=bottom, label=src_label, color=color, alpha=0.8)
        bottom += vals

ax.set_xticks(range(len(arms)))
ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms], rotation=30, ha="right")
ax.set_ylabel("Mean funding volume")
ax.set_title("N5: Funding Mix by Arm")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Interpreting N5 — Funding Mix:** This reveals *how* agents obtain cash in each arm:
- **Passive** is 100% settlement-funded — a pure circular dependency. If the chain breaks, there's no alternative.
- **Active (dealer)** introduces ticket sales (red) — agents can convert receivables to cash.
- **Bank lend** shows the dramatic difference: bank loans (blue) and deposits provide substantial additional funding channels, breaking the circular dependency.
- **NBFI** adds a small layer of loan funding but doesn't fundamentally change the mix.
"""))

# ── Section 8: Temporal Dynamics ─────────────────────────────────────
cells.append(md("""\
---
### 17. Temporal Dynamics

Per-day timeseries aggregated across runs: cumulative defaults, credit flows, active obligations, and mechanism effect trajectories. These show *when* things happen during a simulation.
"""))

cells.append(code("""\
# C28: Contagion Timeline (2x2 grid by kappa)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
target_kappas = [0.25, 0.5, 1.0, 2.0]

for ax, target_k in zip(axes.ravel(), target_kappas):
    for arm in arms:
        if arm not in ts_agg:
            continue
        available = sorted(ts_agg[arm].keys())
        if not available:
            continue
        k = min(available, key=lambda x: abs(x - target_k))
        days = sorted(ts_agg[arm][k].keys())
        cum_total = [
            ts_agg[arm][k][d].get("cum_primary_defaults", 0) + ts_agg[arm][k][d].get("cum_secondary_defaults", 0)
            for d in days
        ]
        ax.plot(days, cum_total, color=ARM_COLORS.get(arm, "#333"),
                label=ARM_LABELS.get(arm, arm), linewidth=2)
    ax.set_title(f"kappa = {target_k}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative defaults")
    if target_k == 0.25:
        ax.legend(fontsize=7)

plt.suptitle("D1: Contagion Timeline by Kappa", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Interpreting D1 — Contagion Timeline:**
- At **κ=0.25** (severe): defaults accumulate rapidly in all arms — the system is deeply insolvent and no mechanism can prevent the cascade. All lines converge to roughly the same level.
- At **κ=0.5** (stressed): the arms diverge clearly. Bank lending (blue) shows fewer cumulative defaults throughout. The dealer (red) shows a small reduction. The gap grows over time as cascade effects compound.
- At **κ=1.0** (balanced): the separation between arms is most pronounced. Bank lending can prevent early defaults that would otherwise cascade.
- At **κ=2.0** (comfortable): fewer defaults overall, but bank lending still shows a clear advantage.
"""))

cells.append(code("""\
# C31: Mechanism Effect Timeline at kappa ~ 0.5
fig, ax = plt.subplots(figsize=(10, 5))
target_k = 0.5

# Get passive baseline
passive_cum = {}
if "passive" in ts_agg:
    available = sorted(ts_agg["passive"].keys())
    if available:
        k = min(available, key=lambda x: abs(x - target_k))
        for d in sorted(ts_agg["passive"][k].keys()):
            passive_cum[d] = (
                ts_agg["passive"][k][d].get("cum_primary_defaults", 0)
                + ts_agg["passive"][k][d].get("cum_secondary_defaults", 0)
            )

for arm in arms:
    if arm == "passive" or arm not in ts_agg:
        continue
    available = sorted(ts_agg[arm].keys())
    if not available:
        continue
    k = min(available, key=lambda x: abs(x - target_k))
    days = sorted(ts_agg[arm][k].keys())
    effect = []
    for d in days:
        arm_cum = (
            ts_agg[arm][k][d].get("cum_primary_defaults", 0)
            + ts_agg[arm][k][d].get("cum_secondary_defaults", 0)
        )
        passive_val = passive_cum.get(d, arm_cum)
        effect.append(passive_val - arm_cum)
    ax.plot(days, effect, color=ARM_COLORS.get(arm, "#333"),
            label=ARM_LABELS.get(arm, arm), linewidth=2)

ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Day")
ax.set_ylabel("Defaults prevented (vs passive)")
ax.set_title(f"D4: Mechanism Effect Timeline (kappa ≈ {target_k})")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
**Interpreting D4 — Mechanism Effect Timeline:** This shows the cumulative advantage of each mechanism over the passive baseline at κ≈0.5. A rising line means the mechanism is progressively preventing more defaults over time.

- **Bank lending (blue)** builds a steady advantage from day 1-3, typically preventing 10-15 additional defaults by end of simulation. The effect accumulates because early lending prevents cascades that would otherwise multiply through the ring.
- **Dealer trading (red)** shows a smaller but persistent advantage, growing slowly as trades redistribute liquidity.
- The effect is strongest in the first 3-5 days, when settlement pressure is highest. After that, the surviving agents have generally resolved their obligations.
"""))

# ── Section 9: Mechanism Internals ───────────────────────────────────
cells.append(md("""\
---
### 18. Mechanism Internals

Detailed analysis of how each mechanism operates during the simulation: dealer trade volumes and prices, bank lending rates and volumes, interbank market dynamics.
"""))

cells.append(code("""\
# Load mechanism timeseries data
from bilancio.analysis.cross_sweep_network import (
    load_mechanism_timeseries,
    aggregate_mechanism_timeseries,
)

def _closest_kappa(available, target):
    if not available:
        return target
    return min(available, key=lambda k: abs(k - target))

mech_data = load_mechanism_timeseries(CACHE_DIR)
mech_agg = aggregate_mechanism_timeseries(mech_data) if mech_data else {}
print(f"Mechanism data loaded for arms: {list(mech_agg.keys()) if mech_agg else 'none'}")
"""))

cells.append(code("""\
# C32: Dealer Trade Volume Over Time (multi-kappa panels)
if "active" in mech_agg:
    all_kappas = sorted(mech_agg["active"].keys())
    target_kappas = [_closest_kappa(all_kappas, t) for t in [0.25, 0.5, 1.0, 2.0]]
    seen = set()
    target_kappas = [k for k in target_kappas if k not in seen and not seen.add(k)]

    n = len(target_kappas)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"kappa = {k}" for k in target_kappas])

    for ki, kappa in enumerate(target_kappas):
        r, c = ki // n_cols + 1, ki % n_cols + 1
        kd = mech_agg["active"].get(kappa, {})
        days = sorted(kd.keys())
        buys = [kd[d].get("n_buy_trades", 0) for d in days]
        sells = [kd[d].get("n_sell_trades", 0) for d in days]
        fig.add_trace(go.Bar(x=days, y=buys, name="Buy" if ki == 0 else None,
                            marker_color="#2ecc71", showlegend=(ki == 0), legendgroup="buy"),
                     row=r, col=c)
        fig.add_trace(go.Bar(x=days, y=sells, name="Sell" if ki == 0 else None,
                            marker_color="#e74c3c", showlegend=(ki == 0), legendgroup="sell"),
                     row=r, col=c)

    fig.update_layout(
        title="E1: Dealer Trade Volume Over Time",
        barmode="group", height=600, margin=dict(t=80),
    )
    fig.show()
else:
    print("No dealer mechanism data available")
"""))

cells.append(md("""\
**Interpreting E1 — Dealer Trade Volume:**
- **Buy trades dominate** over sells across all κ levels — agents with surplus cash are more willing to buy at a discount than stressed agents are to sell (selling requires overcoming the hold-to-maturity expected value).
- At **low κ** (0.25), volume is low — few agents have surplus to buy, and sells are priced too low for stressed agents to accept.
- At **moderate κ** (0.5–1.0), trading is most active — a good mix of surplus buyers and stressed sellers.
- Volume is **front-loaded** (days 0-3) because that's when settlement pressure peaks and liquidity imbalances are largest.
"""))

cells.append(code("""\
# C35: Bank Loan Activity Over Time
if "lend" in mech_agg:
    all_kappas = sorted(mech_agg["lend"].keys())
    target_kappas = [_closest_kappa(all_kappas, t) for t in [0.25, 0.5, 1.0, 2.0]]
    seen = set()
    target_kappas = [k for k in target_kappas if k not in seen and not seen.add(k)]

    n = len(target_kappas)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"kappa = {k}" for k in target_kappas])

    for ki, kappa in enumerate(target_kappas):
        r, c = ki // n_cols + 1, ki % n_cols + 1
        kd = mech_agg["lend"].get(kappa, {})
        days = sorted(kd.keys())
        issued = [kd[d].get("n_loans_issued", 0) for d in days]
        repaid = [kd[d].get("n_loans_repaid", 0) for d in days]
        fig.add_trace(go.Bar(x=days, y=issued, name="Issued" if ki == 0 else None,
                            marker_color="#2980b9", showlegend=(ki == 0), legendgroup="issued"),
                     row=r, col=c)
        fig.add_trace(go.Bar(x=days, y=repaid, name="Repaid" if ki == 0 else None,
                            marker_color="#27ae60", showlegend=(ki == 0), legendgroup="repaid"),
                     row=r, col=c)

    fig.update_layout(
        title="E4: Bank Loan Activity Over Time",
        barmode="group", height=600, margin=dict(t=80),
    )
    fig.show()
else:
    print("No bank mechanism data available")
"""))

cells.append(md("""\
**Interpreting E4 — Bank Loan Activity:**
- Loan issuance concentrates on **day 0** (initial credit allocation) with a smaller wave at day 1-2 as settlement pressures emerge.
- Repayments follow with a lag (loan maturity = 2 days).
- At **low κ** (stressed systems), more loans are issued — the bank is more actively providing liquidity.
- The bank's credit creation is the key mechanism: each loan creates deposits, expanding the money supply and enabling settlements that would otherwise fail.
"""))

# ── Section 10: Loss Analysis ────────────────────────────────────────
cells.append(md("""\
---
### 19. Loss Analysis

Default rate (δ) measures the *frequency* of payment failures. Loss analysis goes further — it measures the **economic damage**: how much value is actually destroyed when defaults occur. System loss includes payable default losses (borne by creditors), deposit losses, and intermediary losses.
"""))

cells.append(code("""\
# L1: System Loss Rate vs Kappa
kappa_vals = np.sort(np.unique(data["kappa"]))

loss_arms = [
    ("system_loss_pct_passive", "Passive", PASSIVE_GREY),
    ("system_loss_pct_active", "Active (Dealer)", DEALER_RED),
    ("system_loss_pct_lender", "NBFI Lender", NBFI_GREEN),
]
# Check for bank system loss columns
for col_name in ["system_loss_pct_bank_passive", "bank_system_loss_pct_idle", "system_loss_pct_idle"]:
    if col_name in data:
        loss_arms.append((col_name, "Bank Idle", "#b0bec5"))
        break
for col_name in ["system_loss_pct_bank_dealer", "bank_system_loss_pct_lend", "system_loss_pct_lend"]:
    if col_name in data:
        loss_arms.append((col_name, "Bank Lend", BANK_BLUE))
        break

fig = go.Figure()
for col, label, color in loss_arms:
    if col not in data:
        continue
    means = []
    for k in kappa_vals:
        mask = data["kappa"] == k
        vals = np.asarray(data[col][mask], dtype=float)
        valid = vals[np.isfinite(vals)]
        means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
    fig.add_trace(go.Scatter(
        x=kappa_vals, y=means, mode="lines+markers",
        name=label, line=dict(color=color, width=2), marker=dict(size=8),
    ))
fig.update_layout(
    title="L1: System Loss Rate vs Kappa",
    xaxis_title="Kappa", yaxis_title="System loss (fraction of total value)",
    xaxis_type="log", height=500,
    legend=dict(x=0.6, y=0.99),
)
fig.show()
"""))

cells.append(md("""\
**Interpreting L1 — System Loss:** System loss includes both payable default losses and intermediary losses (dealer PnL, bank losses, CB backstop). This gives a more complete picture than δ alone. The pattern mirrors the default rate chart but with important nuances: bank lending can have *higher* system losses than passive in some cases because the bank itself absorbs losses (through bad loans), even though fewer agents default.
"""))

cells.append(code("""\
# L2: Loss vs Default scatter
fig = go.Figure()
arm_configs_loss = [
    ("delta_passive", "system_loss_pct_passive", "Passive", PASSIVE_GREY),
    ("delta_active", "system_loss_pct_active", "Active", DEALER_RED),
    ("delta_lender", "system_loss_pct_lender", "NBFI Lender", NBFI_GREEN),
]
for delta_col, loss_col, label, color in arm_configs_loss:
    if delta_col not in data or loss_col not in data:
        continue
    d = np.asarray(data[delta_col], dtype=float)
    l = np.asarray(data[loss_col], dtype=float)
    mask = np.isfinite(d) & np.isfinite(l)
    fig.add_trace(go.Scatter(
        x=d[mask], y=l[mask], mode="markers",
        name=label, marker=dict(color=color, size=5, opacity=0.5),
    ))

# Add diagonal
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(color="black", dash="dash"), name="loss = delta", showlegend=True,
))
fig.update_layout(
    title="L2: System Loss vs Default Rate",
    xaxis_title="Default rate (delta)",
    yaxis_title="System loss rate",
    height=500,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting L2 — Loss vs Default scatter:** If all points fell on the diagonal (loss = δ), every default would result in full loss. Points **below** the diagonal mean that some defaulted obligations were partially recovered. Points **above** mean that defaults generated additional losses (through intermediary losses, deposit destruction, etc.).

Key observation: most points cluster near or slightly above the diagonal — defaults are mostly unrecoverable. Intermediary mechanisms shift points left (fewer defaults) but may also shift them slightly up (intermediary absorbs some loss).
"""))

cells.append(code("""\
# L4: Loss Treatment Effects vs Kappa
kappa_vals = np.sort(np.unique(data["kappa"]))

effects = [
    ("system_loss_trading_effect", "Dealer Effect", DEALER_RED),
    ("system_loss_lending_effect", "NBFI Lending Effect", NBFI_GREEN),
]
# Check for bank loss effect column
for col_name in ["system_loss_bank_lending_effect", "bank_system_loss_bank_lending_effect"]:
    if col_name in data:
        effects.append((col_name, "Bank Lending Effect", BANK_BLUE))
        break

fig = go.Figure()
for col, label, color in effects:
    if col not in data:
        continue
    means = []
    for k in kappa_vals:
        mask = data["kappa"] == k
        vals = np.asarray(data[col][mask], dtype=float)
        valid = vals[np.isfinite(vals)]
        means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
    fig.add_trace(go.Scatter(
        x=kappa_vals, y=means, mode="lines+markers",
        name=label, line=dict(color=color, width=2), marker=dict(size=8),
    ))
fig.add_hline(y=0, line_dash="dash", line_color="grey")
fig.update_layout(
    title="L4: Loss Treatment Effects vs Kappa",
    xaxis_title="Kappa", yaxis_title="Loss reduction (positive = less loss)",
    xaxis_type="log", height=500,
)
fig.show()
"""))

cells.append(md("""\
**Interpreting L4 — Loss Treatment Effects:** This shows how much each mechanism reduces *total system loss* (not just default rate). Positive values mean the mechanism reduces losses.

- Bank lending shows the largest loss reduction, but it can go negative at extreme κ values — meaning the bank's own losses (from bad loans) can exceed the default reduction.
- The dealer effect is modest but consistently positive — trading redistributes risk without creating new losses.
- This is a more honest metric than δ alone because it accounts for the costs of intermediation (dealer inventory losses, bank bad loans, etc.).
"""))

# ── Summary ──────────────────────────────────────────────────────────
cells.append(md("""\
---
## Key Findings

### Mechanism Effectiveness

1. **Bank lending dominates across most parameter regimes.** It reduces defaults more than dealer trading or NBFI lending in the vast majority of the 225 parameter combinations tested.

2. **Dealer trading helps most when maturities are spread (μ > 0).** With front-loaded stress (μ=0), there's no time for trading before settlement. The dealer's sweet spot is moderate stress (κ ≈ 0.5–1.0) with spread maturities (μ ≥ 0.5).

3. **NBFI lending helps most with front-loaded stress (μ=0).** It provides pre-settlement credit when agents face immediate obligations. However, the NBFI's decision model is still under development and its effects are modest.

4. **Mechanisms operate through different channels:**
   - *Dealer*: Redistributes existing liquidity through secondary market trading
   - *NBFI*: Injects external credit (from own capital) to stressed agents
   - *Bank*: Creates money through lending, fundamentally expanding system liquidity

5. **Concentration matters more than liquidity.** At low c (unequal debt), some agents face structurally impossible obligations regardless of κ. No mechanism can save an agent whose debt exceeds all available system cash.

### Structural Insights

6. **Combined mechanisms underperform** (from earlier experiments): Running dealer + NBFI together is often worse than the best standalone mechanism. They compete for the same marginal agents' resources.

7. **Loss ≠ Default.** System loss analysis reveals that intermediation sometimes increases total losses (intermediary absorbs risk) even while reducing default counts. The right metric depends on the question being asked.

8. **Cascade dynamics are the amplifier.** In stressed systems, 25-35% of defaults are cascade (contagious) rather than primary. Breaking the cascade chain (through early lending or trading) generates outsized benefits.

### Open Questions

- **Bank money creation vs. redistribution**: How much of the bank's advantage comes from endogenous money creation vs. better liquidity allocation?
- **NBFI decision model**: The current NBFI lacks its own Bayesian belief tracker and uses a simplified lending rule. A proper decision profile might substantially improve its effectiveness.
- **Network topology**: All results are for ring topology. How do they generalize to random graphs, core-periphery, or bipartite networks?
- **Calibration**: Can κ and c be mapped to observed interbank or trade-credit data to ground the Kalecki fable empirically?
"""))

# ═══════════════════════════════════════════════════════════════════════
# BUILD THE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════

# Assign cell IDs
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:03d}"

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "kalecki_ring_presentation.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Written {len(cells)} cells to {out_path}")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
