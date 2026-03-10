"""Auto-generate a Jupyter presentation notebook for a completed sweep.

Produces a self-contained .ipynb with:
  - Theory sections (parameterized to the sweep's actual values)
  - Analysis code cells (adapted to the sweep type's columns)
  - Registries for alignment testing

Public API:
    generate_sweep_notebook(out_dir, sweep_type, output_dir=None) -> Path
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Registries (alignment targets for tests) ────────────────────────

COVERED_AGENT_KINDS: frozenset[str] = frozenset({
    "central_bank", "bank", "household", "firm", "dealer", "vbt", "non_bank_lender",
})

EXCLUDED_AGENT_KINDS: frozenset[str] = frozenset({
    "treasury", "investment_fund", "insurance_company", "rating_agency",
})

COVERED_INSTRUMENT_KINDS: frozenset[str] = frozenset({
    "cash", "bank_deposit", "reserve_deposit", "payable",
    "cb_loan", "non_bank_loan", "bank_loan", "interbank_loan",
    "delivery_obligation",
})

COVERED_PROFILES: frozenset[str] = frozenset({
    "TraderProfile", "VBTProfile", "LenderProfile", "BankProfile", "RatingProfile",
})

HANDLED_SWEEP_TYPES: frozenset[str] = frozenset({"dealer", "bank", "nbfi"})

DOCUMENTED_PHASE_FLAGS: frozenset[str] = frozenset({
    "enable_dealer", "enable_lender", "enable_rating",
    "enable_banking", "enable_bank_lending",
})

ARM_DIR_MAP: dict[tuple[str, str], str] = {
    ("dealer", "active"): "active",
    ("dealer", "passive"): "passive",
    ("bank", "lend"): "bank_lend",
    ("bank", "idle"): "bank_idle",
    ("nbfi", "lend"): "nbfi_lend",
    ("nbfi", "idle"): "nbfi_idle",
}

# Sweep type metadata (mirrors comprehensive_report._SWEEP_META)
_SWEEP_META: dict[str, dict[str, str]] = {
    "dealer": {
        "effect_col": "trading_effect",
        "treatment_label": "Active (Dealer)",
        "baseline_label": "Passive",
        "delta_treatment": "delta_active",
        "delta_baseline": "delta_passive",
        "mechanism_name": "Dealer Trading",
    },
    "bank": {
        "effect_col": "bank_lending_effect",
        "treatment_label": "Bank Lending",
        "baseline_label": "Bank Idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "mechanism_name": "Bank Lending",
    },
    "nbfi": {
        "effect_col": "lending_effect",
        "treatment_label": "NBFI Lending",
        "baseline_label": "NBFI Idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "mechanism_name": "NBFI Lending",
    },
}


# ── Notebook cell helpers ────────────────────────────────────────────

def _split_source(source: str) -> list[str]:
    """Split source into list of lines with newlines preserved (notebook format)."""
    lines = source.split("\n")
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def _md(source: str) -> dict[str, Any]:
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source), "id": None}


def _code(source: str) -> dict[str, Any]:
    """Create a code cell."""
    return {
        "cell_type": "code", "metadata": {}, "source": _split_source(source),
        "outputs": [], "execution_count": None, "id": None,
    }


# ── SweepNotebookContext ─────────────────────────────────────────────

@dataclass(frozen=True)
class SweepNotebookContext:
    """All context needed to generate a sweep notebook."""

    sweep_type: str
    n_agents: int = 100
    maturity_days: int = 10
    topology: str = "ring"

    kappas: tuple[float, ...] = ()
    concentrations: tuple[float, ...] = ()
    mus: tuple[float, ...] = ()
    outside_mid_ratios: tuple[float, ...] = ()

    arms_present: tuple[str, ...] = ()
    csv_columns: tuple[str, ...] = ()
    n_parameter_combos: int = 0
    n_runs: int = 0

    @property
    def has_dealer(self) -> bool:
        return self.sweep_type == "dealer"

    @property
    def has_bank(self) -> bool:
        return self.sweep_type == "bank"

    @property
    def has_nbfi(self) -> bool:
        return self.sweep_type == "nbfi"

    @property
    def has_cb(self) -> bool:
        return self.sweep_type == "bank"

    @property
    def meta(self) -> dict[str, str]:
        return _SWEEP_META[self.sweep_type]

    @classmethod
    def from_sweep_dir(cls, out_dir: Path, sweep_type: str) -> "SweepNotebookContext":
        """Discover context from a completed sweep directory."""
        out_dir = Path(out_dir)
        csv_path = out_dir / "aggregate" / "comparison.csv"

        kappas: list[float] = []
        concentrations: list[float] = []
        mus: list[float] = []
        outside_mid_ratios: list[float] = []
        csv_columns: list[str] = []
        n_rows = 0

        if csv_path.is_file():
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                csv_columns = list(reader.fieldnames or [])
                rows = list(reader)
                n_rows = len(rows)
                for row in rows:
                    if "kappa" in row and row["kappa"]:
                        kappas.append(float(row["kappa"]))
                    if "concentration" in row and row["concentration"]:
                        concentrations.append(float(row["concentration"]))
                    if "mu" in row and row["mu"]:
                        mus.append(float(row["mu"]))
                    if "outside_mid_ratio" in row and row["outside_mid_ratio"]:
                        outside_mid_ratios.append(float(row["outside_mid_ratio"]))

        # Detect arms from directory structure
        arms: list[str] = []
        for (st, arm_label), subdir in ARM_DIR_MAP.items():
            if st == sweep_type:
                arm_dir = out_dir / subdir / "runs"
                if arm_dir.is_dir():
                    arms.append(arm_label)

        # Read n_agents/maturity_days from summary.json if available
        n_agents = 100
        maturity_days = 10
        summary_path = out_dir / "aggregate" / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
                n_agents = summary.get("n_agents", n_agents)
                maturity_days = summary.get("maturity_days", maturity_days)
            except (json.JSONDecodeError, KeyError):
                pass

        return cls(
            sweep_type=sweep_type,
            n_agents=n_agents,
            maturity_days=maturity_days,
            topology="ring",
            kappas=tuple(sorted(set(kappas))),
            concentrations=tuple(sorted(set(concentrations))),
            mus=tuple(sorted(set(mus))),
            outside_mid_ratios=tuple(sorted(set(outside_mid_ratios))),
            arms_present=tuple(arms),
            csv_columns=tuple(csv_columns),
            n_parameter_combos=n_rows,
            n_runs=n_rows * len(arms) if arms else n_rows * 2,
        )


# ── Theory section generators ───────────────────────────────────────
# Each returns a list of notebook cells, or an empty list to skip.

def _section_bilancio_intro(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    mechanism = ctx.meta["mechanism_name"]
    return [_md(f"""\
# {mechanism} Sweep: Settlement and Liquidity Analysis

**Bilancio** -- A multi-agent payment system simulator

---

This notebook presents the analysis for a **{mechanism}** sweep experiment. It covers:

1. **The Simulator** -- bilancio's accounting foundation
2. **The Kalecki Ring** -- the topology driving the experiment
3. **Simulation mechanics** -- lifecycle, phases, settlement
4. **Agent decision profiles** -- how each relevant agent makes choices
5. **Experimental design** -- arms, parameter grid, metrics
6. **Results** -- sweep analysis with charts and statistical tests
"""),
    _md("""\
---
## Part I: The Simulator

### 1. What is Bilancio?

Bilancio is a **multi-agent payment system simulator** that models liquidity, settlement, and default in networks of interconnected financial obligations.

**Core principles:**

- **Double-entry accounting**: Every transaction is recorded as a debit and credit. Balance sheet invariants are enforced at every step.

- **Agents as balance sheets**: Every entity (firm, household, dealer, bank, central bank) is a balance sheet with a behavioral profile attached.

- **Settlement as ground truth**: The simulation does not assume prices or clearing rates. Instead, agents make decisions, execute trades, and attempt settlement. The *outcome* is emergent.

- **Instruments with contract terms**: Every financial claim (payable, loan, deposit, reserve) is a concrete instrument with an issuer, a holder, a face value, a maturity date, and a means of payment.

```
Scenario Builder  -->  Settlement Engine  -->  Decision Layer  -->  Analysis & Export
(Ring, kappa, c, mu)    (Daily phases, MOP)    (Observe/Value/    (delta, phi, HTML)
                                                Assess/Choose)
```
""")]


def _section_kalecki_ring(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if ctx.topology != "ring":
        return []
    return [_md(f"""\
### 2. The Kalecki Ring

#### The Fable (Kalecki, via Toporowski 2008)

> A community of merchants each owes the next in a circle. No one can pay until they are paid. Settlement requires either sufficient initial liquidity or coordinated netting.

```
       H1 --> H2
      /          \\
    H6            H3
      \\          /
       H5 <-- H4

    Each agent owes the next
```

In this sweep: **N = {ctx.n_agents}** agents in the ring, maturity horizon = **{ctx.maturity_days} days**.

The ring captures essential payment network structure:
- **Sequential dependency**: payments depend on incoming funds
- **Liquidity as a flow**: the same cash can settle multiple obligations as it circulates
- **Cascade dynamics**: one failure propagates through the system
""")]


def _section_parameters(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    kappa_str = ", ".join(str(k) for k in ctx.kappas) if ctx.kappas else "N/A"
    c_str = ", ".join(str(c) for c in ctx.concentrations) if ctx.concentrations else "N/A"
    mu_str = ", ".join(str(m) for m in ctx.mus) if ctx.mus else "N/A"
    rho_str = ", ".join(str(r) for r in ctx.outside_mid_ratios) if ctx.outside_mid_ratios else "N/A"

    return [_md(f"""\
### 3. Control Parameters

Three primary parameters control the initial conditions of the ring.

#### Kappa (kappa) -- Liquidity Ratio

kappa = L0 / S1

The ratio of total initial cash to total debt.

| kappa | Interpretation | Expected behavior |
|---|----------------|-------------------|
| < 0.5 | **Severely stressed** | Mass defaults almost certain |
| 0.5-1.0 | **Stressed** | Defaults depend on distribution and timing |
| 1.0 | **Balanced** | System has exactly enough cash |
| 1.0-2.0 | **Comfortable** | Surplus liquidity |
| > 2.0 | **Abundant** | Defaults are structural, not liquidity-driven |

**This sweep uses kappa in: {{{kappa_str}}}**

#### c (Concentration) -- Debt Distribution

The Dirichlet parameter controlling debt inequality. Lower c = more unequal.

**This sweep uses c in: {{{c_str}}}**

#### mu -- Maturity Timing Skew

Controls when payables mature. mu=0: all due day 1, mu=1: all due last day.

**This sweep uses mu in: {{{mu_str}}}**

#### rho (Outside-Money Ratio)

VBT discount factor. **This sweep uses rho in: {{{rho_str}}}**
""")]


def _section_phases(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    # Build phase diagram based on sweep type
    phases = []
    phases.append("  Phase A -- Information Update")
    phases.append("  |-- Observe yesterday's defaults, update beliefs")

    if ctx.has_bank:
        phases.append("  Sub-Phase B.1 -- Bank Lending")
        phases.append("  |-- Bank evaluates loan applications, disburses loans")
    if ctx.has_nbfi:
        phases.append("  Sub-Phase B.1 -- NBFI Lending")
        phases.append("  |-- Lender evaluates creditworthiness, disburses loans")
    if ctx.has_dealer:
        phases.append("  Sub-Phase B.2 -- Dealer Trading")
        phases.append("  |-- Dealer posts quotes, agents trade receivables")

    phases.append("  Sub-Phase B.3 -- Settlement")
    phases.append("  |-- Payables due today presented for payment")
    phases.append("  |-- Agents that cannot pay -> DEFAULT and expulsion")

    if ctx.has_bank:
        phases.append("  Phase C -- Interbank")
        phases.append("  |-- Banks with surplus reserves lend to deficit banks")

    phases.append("  Phase D -- End of Day")
    phases.append("  |-- Rollover, accounting, metrics")

    phase_text = "\n".join(phases)

    return [_md(f"""\
---
## Part II: How the Simulation Works

### 4. Simulation Lifecycle

A simulation runs for **{ctx.maturity_days} days**. Each day follows the phase structure below.

```
{phase_text}
```

#### The Settlement Waterfall

When a payable matures, the debtor must pay. The means-of-payment (MOP) waterfall:

| Priority | Action |
|----------|--------|
| 1 | Pay from cash |
| 2 | Withdraw bank deposit (if banking) |
| 3 | Sell receivable to dealer/VBT (if dealer) |
| 4 | Borrow from NBFI (if lender) |
| 5 | Borrow from bank (if banking) |
| X | **DEFAULT** -- agent expelled |
""")]


def _section_profile_trader(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    # Generate parameter table from actual TraderProfile fields
    try:
        from bilancio.decision.profiles import TraderProfile
        fields = [(f.name, f.default if f.default is not dataclasses.MISSING else "N/A")
                  for f in dataclasses.fields(TraderProfile)
                  if not f.name.startswith("_")]
        rows = "\n".join(f"| `{name}` | {default} |" for name, default in fields)
    except ImportError:
        rows = "| (could not import TraderProfile) | |"

    return [_md(f"""\
---
## Part III: Agent Decision Profiles

Every agent follows: **Observe -> Value -> Assess -> Choose**

### 5.1 Trader (Firm / Household)

**Role:** Agents in the real economy. They hold receivables and cash, owe payables. Trading is a liquidity management tool.

**Decision parameters (from TraderProfile):**

| Parameter | Default |
|-----------|---------|
{rows}

**Risk Assessment:** Bayesian belief tracking of system-wide default probability:

P_default = (alpha + d_observed) / (2*alpha + n_total)

**Sell decision:** Accept bid if `bid * face >= EV + threshold * face` (threshold drops with urgency).
**Buy decision:** Accept ask if `EV >= ask * face + buy_premium * face` and cash surplus exceeds reserve.
""")]


def _section_profile_dealer(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if not ctx.has_dealer:
        return []
    return [_md("""\
### 5.2 Dealer (Treynor Market-Maker)

**Role:** Provides a secondary market for receivables. Posts bid/ask quotes, buys from stressed sellers, sells to surplus buyers.

**Pricing Kernel -- L1 (Treynor):**

The dealer quotes per maturity bucket (short/mid/long), anchored to the VBT's outside mid-price:

```
M = VBT credit-adjusted mid-price
O = VBT outside spread
K* = floor(V / (M * S))   (capacity in tickets)

Pricing midline: p(x) = M - [O / (X* + 2S)] * (x - X*/2)
Inside spread:   I = lambda * O   where lambda = S / (X* + S)
```

**Key properties:**
- More capacity -> tighter spread
- Inventory tilts the midline (high inventory -> lower mid -> dealer wants to sell)
- Guard regime: if M <= 0.02, dealer routes to VBT
""")]


def _section_profile_vbt(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if not ctx.has_dealer:
        return []
    return [_md("""\
### 5.3 Virtual Book Trader (VBT) -- Outside Liquidity

**Role:** The outside market. Provides reference pricing and acts as the dealer's counterparty of last resort.

**Credit-Adjusted Mid-Price:**

M = rho * (1 - P_default)

**Spread per bucket:** short=0.20, mid=0.30, long=0.40

The VBT's mid-price falls as defaults rise, pricing credit risk into the outside market.
""")]


def _section_profile_bank(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if not ctx.has_bank:
        return []
    try:
        from bilancio.decision.profiles import BankProfile
        fields = [(f.name, f.default if f.default is not dataclasses.MISSING else "N/A")
                  for f in dataclasses.fields(BankProfile)
                  if not f.name.startswith("_")]
        rows = "\n".join(f"| `{name}` | {default} |" for name, default in fields)
    except ImportError:
        rows = "| (could not import BankProfile) | |"

    return [_md(f"""\
### 5.4 Bank (Treynor Funding-Plane Kernel)

**Role:** Intermediates between depositors and borrowers within a CB corridor. Uses L1 Treynor pricing on the interest rate (funding) plane.

**Key difference:** The bank creates money when it lends (endogenous money creation).

**Parameters (from BankProfile):**

| Parameter | Default |
|-----------|---------|
{rows}
""")]


def _section_profile_cb(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if not ctx.has_cb:
        return []
    return [_md("""\
### 5.5 Central Bank

**Role:** Standing facilities provider. Sets the interest rate corridor. Provides emergency reserves to banks.

The CB is a lender of last resort **for banks**, not for firms.
""")]


def _section_profile_nbfi(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if not ctx.has_nbfi:
        return []
    try:
        from bilancio.decision.profiles import LenderProfile
        fields = [(f.name, f.default if f.default is not dataclasses.MISSING else "N/A")
                  for f in dataclasses.fields(LenderProfile)
                  if not f.name.startswith("_")]
        rows = "\n".join(f"| `{name}` | {default} |" for name, default in fields)
    except ImportError:
        rows = "| (could not import LenderProfile) | |"

    return [_md(f"""\
### 5.6 Non-Bank Financial Intermediary (NBFI) Lender

**Role:** Provides direct credit to firms from its own capital. Unlike the bank, the NBFI does not create money.

**Loan Pricing:** r = base_rate + risk_premium_scale * P_default

**Parameters (from LenderProfile):**

| Parameter | Default |
|-----------|---------|
{rows}
""")]


def _section_experimental_design(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta

    # Arms table
    arm_rows = f"| **{meta['baseline_label']}** (Baseline) | No intermediation. Pure settlement. |"
    arm_rows += f"\n| **{meta['treatment_label']}** (Treatment) | {ctx.meta['mechanism_name']} active |"

    # Grid table
    kappa_str = ", ".join(str(k) for k in ctx.kappas) if ctx.kappas else "N/A"
    c_str = ", ".join(str(c) for c in ctx.concentrations) if ctx.concentrations else "N/A"
    mu_str = ", ".join(str(m) for m in ctx.mus) if ctx.mus else "N/A"
    rho_str = ", ".join(str(r) for r in ctx.outside_mid_ratios) if ctx.outside_mid_ratios else "N/A"

    n_k = len(ctx.kappas) if ctx.kappas else 0
    n_c = len(ctx.concentrations) if ctx.concentrations else 0
    n_m = len(ctx.mus) if ctx.mus else 0
    n_r = len(ctx.outside_mid_ratios) if ctx.outside_mid_ratios else 0

    return [_md(f"""\
---
## Part IV: Experimental Design

### 6. Experimental Arms

| Arm | Description |
|-----|-------------|
{arm_rows}

**Paired comparison:** Each parameter combination is run in both arms under identical initial conditions. The treatment effect is:

delta_effect = delta_baseline - delta_treatment

Positive delta_effect means the mechanism **reduced** defaults.

### 7. Sweep Configuration

| Parameter | Values | Count |
|-----------|--------|-------|
| kappa | {kappa_str} | {n_k} |
| c (concentration) | {c_str} | {n_c} |
| mu | {mu_str} | {n_m} |
| rho | {rho_str} | {n_r} |

**Total: {ctx.n_parameter_combos} parameter combinations**, {ctx.n_runs} total runs.

### 8. Metrics

| Metric | Symbol | Definition |
|--------|--------|------------|
| Default rate | delta | defaulted_face / total_face |
| Clearing rate | phi | settled_face / total_face |
| Treatment effect | delta_effect | delta_baseline - delta_treatment |
| Relief ratio | delta_effect / delta_baseline | Proportional improvement |
""")]


def _section_metrics(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    return [_md("""\
---
## Part V: Sweep Analysis

We now analyze the results of the sweep. The analysis proceeds in stages:

1. **Data loading** -- load the comparison CSV
2. **Descriptive overview** -- summary statistics, distributions, kappa response
3. **Parameter sensitivity** -- heatmaps (if multiple parameter values)
4. **Regression analysis** -- OLS models (if enough kappa values)
5. **Statistical tests** -- hypothesis tests
6. **Conclusions** -- summary of findings
""")]


# ── Analysis code cell generators ────────────────────────────────────
# Each returns a list of notebook cells, or empty list to skip.

def _analysis_data_loading(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    return [_code(f"""\
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load sweep data
SWEEP_DIR = Path("../..")
CSV_PATH = SWEEP_DIR / "aggregate" / "comparison.csv"
df = pd.read_csv(CSV_PATH)

# Key columns for this sweep type
EFFECT_COL = "{meta['effect_col']}"
DELTA_TREATMENT = "{meta['delta_treatment']}"
DELTA_BASELINE = "{meta['delta_baseline']}"
TREATMENT_LABEL = "{meta['treatment_label']}"
BASELINE_LABEL = "{meta['baseline_label']}"

# Compute effect if not present
if EFFECT_COL not in df.columns and DELTA_BASELINE in df.columns and DELTA_TREATMENT in df.columns:
    df[EFFECT_COL] = df[DELTA_BASELINE] - df[DELTA_TREATMENT]

print(f"Loaded {{len(df)}} parameter combinations")
print(f"Columns: {{list(df.columns)}}")
for col in ["kappa", "concentration", "mu", "outside_mid_ratio"]:
    if col in df.columns:
        print(f"  {{col}}: {{sorted(df[col].unique())}}")
print(f"\\nEffect column: {{EFFECT_COL}}")
if EFFECT_COL in df.columns:
    print(f"  Mean effect: {{df[EFFECT_COL].mean():.6f}}")
    print(f"  Median: {{df[EFFECT_COL].median():.6f}}")
    print(f"  Std: {{df[EFFECT_COL].std():.6f}}")
    pct_pos = (df[EFFECT_COL] > 0).mean() * 100
    print(f"  % positive: {{pct_pos:.1f}}%")
""")]


def _analysis_summary_stats(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    return [
        _md(f"""\
---
### 9. Descriptive Overview

Summary statistics and distributional comparisons for the **{meta['mechanism_name']}** effect.
The effect is delta_baseline - delta_treatment. Positive = mechanism helped.
"""),
        _code(f"""\
# Summary statistics table
effect = df[EFFECT_COL].dropna()
stats = {{
    "Mean": f"{{effect.mean():.6f}}",
    "Median": f"{{effect.median():.6f}}",
    "Std Dev": f"{{effect.std():.6f}}",
    "Q1": f"{{effect.quantile(0.25):.6f}}",
    "Q3": f"{{effect.quantile(0.75):.6f}}",
    "Min": f"{{effect.min():.6f}}",
    "Max": f"{{effect.max():.6f}}",
    "% Positive": f"{{(effect > 0).mean() * 100:.1f}}%",
    "N": f"{{len(effect)}}",
}}

fig = go.Figure(data=[go.Table(
    header=dict(values=["Statistic", "Value"], fill_color="#4a4a4a",
                font=dict(color="white", size=13), align="center"),
    cells=dict(values=[list(stats.keys()), list(stats.values())],
               fill_color=["#f5f5f5", "#e8f0fe"], align=["left", "right"],
               font=dict(size=12)),
)])
fig.update_layout(title="{meta['mechanism_name']} Effect: Summary Statistics",
                  height=350, margin=dict(t=50, b=20, l=20, r=20))
fig.show()
"""),
    ]


def _analysis_distributions(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    return [
        _code(f"""\
# Box plot of treatment effect
fig = go.Figure()
fig.add_trace(go.Box(
    y=df[EFFECT_COL].dropna(), name="{meta['mechanism_name']} Effect",
    marker_color="#2196F3", boxmean="sd",
))
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
fig.update_layout(title="Distribution of {meta['mechanism_name']} Effect",
                  yaxis_title="Effect (delta reduction)", height=450, showlegend=False)
fig.show()
"""),
        _code(f"""\
# Histogram of default rates by arm
fig = go.Figure()
for col, label, color in [
    (DELTA_BASELINE, BASELINE_LABEL, "#9e9e9e"),
    (DELTA_TREATMENT, TREATMENT_LABEL, "#2196F3"),
]:
    if col in df.columns:
        fig.add_trace(go.Histogram(
            x=df[col].dropna(), name=label, marker_color=color,
            opacity=0.6, nbinsx=30,
        ))
fig.update_layout(title="Default Rate Distributions by Arm",
                  xaxis_title="Default rate (delta)", yaxis_title="Count",
                  barmode="overlay", height=450)
fig.show()
"""),
    ]


def _analysis_kappa_response(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    if not ctx.kappas or len(ctx.kappas) < 2:
        return []
    return [
        _md("### 10. Kappa Response"),
        _code(f"""\
# Default rate vs kappa by arm
fig = go.Figure()
for col, label, color in [
    (DELTA_BASELINE, BASELINE_LABEL, "#9e9e9e"),
    (DELTA_TREATMENT, TREATMENT_LABEL, "#2196F3"),
]:
    if col not in df.columns:
        continue
    grouped = df.groupby("kappa")[col].mean().sort_index()
    fig.add_trace(go.Scatter(
        x=grouped.index, y=grouped.values, mode="lines+markers",
        name=label, line=dict(color=color, width=2), marker=dict(size=8),
    ))
fig.update_layout(title="Mean Default Rate vs Kappa by Arm",
                  xaxis_title="Kappa (liquidity ratio)", yaxis_title="Mean default rate",
                  xaxis_type="log", height=500)
fig.show()
"""),
        _code(f"""\
# Treatment effect vs kappa
grouped_effect = df.groupby("kappa")[EFFECT_COL].agg(["mean", "std", "count"])
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=grouped_effect.index, y=grouped_effect["mean"], mode="lines+markers",
    name="Mean effect", line=dict(color="#2196F3", width=2), marker=dict(size=8),
    error_y=dict(type="data", array=grouped_effect["std"] / np.sqrt(grouped_effect["count"]),
                 visible=True),
))
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
fig.update_layout(title="{meta['mechanism_name']} Effect vs Kappa (with SE bars)",
                  xaxis_title="Kappa", yaxis_title="Mean effect (delta reduction)",
                  xaxis_type="log", height=500)
fig.show()
"""),
    ]


def _analysis_heatmaps(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    # Only include heatmaps if at least 2 values in at least 2 parameters
    multi_params = []
    if len(ctx.kappas) >= 2:
        multi_params.append("kappa")
    if len(ctx.concentrations) >= 2:
        multi_params.append("concentration")
    if len(ctx.mus) >= 2:
        multi_params.append("mu")
    if len(ctx.outside_mid_ratios) >= 2:
        multi_params.append("outside_mid_ratio")

    if len(multi_params) < 2:
        return []

    meta = ctx.meta
    # Build pairs from multi_params
    pairs = []
    for i, p1 in enumerate(multi_params):
        for p2 in multi_params[i + 1:]:
            pairs.append((p1, p2))
            if len(pairs) >= 3:
                break
        if len(pairs) >= 3:
            break

    pair_strs = ", ".join(f'("{p1}", "{p2}")' for p1, p2 in pairs)

    return [
        _md("### 11. Parameter Sensitivity (Heatmaps)"),
        _code(f"""\
# Heatmaps: effect by parameter pairs
param_pairs = [{pair_strs}]
n_pairs = len(param_pairs)
fig = make_subplots(rows=1, cols=n_pairs,
                    subplot_titles=[f"{{p1}} x {{p2}}" for p1, p2 in param_pairs],
                    horizontal_spacing=0.12)
for idx, (row_param, col_param) in enumerate(param_pairs):
    pivot = df.pivot_table(values=EFFECT_COL, index=row_param, columns=col_param, aggfunc="mean")
    abs_max = max(np.nanmax(np.abs(pivot.values)), 0.001)
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[str(v) for v in pivot.columns],
        y=[str(v) for v in pivot.index],
        colorscale="RdBu_r", zmid=0, zmin=-abs_max, zmax=abs_max,
        showscale=(idx == n_pairs - 1),
        colorbar=dict(title="Effect") if idx == n_pairs - 1 else dict(showticklabels=False, len=0.01),
    ), row=1, col=idx + 1)
    fig.update_xaxes(title_text=col_param, row=1, col=idx + 1)
    fig.update_yaxes(title_text=row_param if idx == 0 else "", row=1, col=idx + 1)
fig.update_layout(title="{meta['mechanism_name']} Effect Heatmaps",
                  height=400, margin=dict(t=80, b=50))
fig.show()
"""),
    ]


def _analysis_regression(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    if len(ctx.kappas) < 3:
        return []
    meta = ctx.meta
    return [
        _md("### 12. Regression Analysis"),
        _code(f"""\
# OLS regression: effect ~ log(kappa) + concentration + mu + outside_mid_ratio
import statsmodels.api as sm

predictors = []
for col in ["kappa", "concentration", "mu", "outside_mid_ratio"]:
    if col in df.columns and df[col].nunique() > 1:
        predictors.append(col)

if predictors:
    reg_df = df[predictors + [EFFECT_COL]].dropna()
    if "kappa" in predictors:
        reg_df = reg_df.copy()
        reg_df["log_kappa"] = np.log(reg_df["kappa"])
        predictors = ["log_kappa"] + [p for p in predictors if p != "kappa"]

    X = sm.add_constant(reg_df[predictors])
    y = reg_df[EFFECT_COL]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Coefficient forest plot
    coefs = model.params.drop("const")
    ci = model.conf_int().drop("const")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs.values, y=coefs.index, mode="markers",
        marker=dict(size=10, color="#2196F3"),
        error_x=dict(type="data",
                     symmetric=False,
                     array=ci.iloc[:, 1].values - coefs.values,
                     arrayminus=coefs.values - ci.iloc[:, 0].values),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(title="{meta['mechanism_name']} Effect: OLS Coefficients (95% CI)",
                      xaxis_title="Coefficient", height=400)
    fig.show()
else:
    print("Not enough varying predictors for regression.")
"""),
    ]


def _analysis_statistical_tests(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    return [
        _md("### 13. Statistical Tests"),
        _code(f"""\
from scipy import stats

effect = df[EFFECT_COL].dropna()

# One-sample t-test: is the mean effect significantly different from 0?
t_stat, t_pval = stats.ttest_1samp(effect, 0)
print(f"One-sample t-test (H0: mean effect = 0):")
print(f"  t = {{t_stat:.4f}}, p = {{t_pval:.6f}}")
print(f"  {{'' if t_pval < 0.05 else 'NOT '}}significant at alpha=0.05")

# Wilcoxon signed-rank test (non-parametric)
try:
    w_stat, w_pval = stats.wilcoxon(effect)
    print(f"\\nWilcoxon signed-rank test:")
    print(f"  W = {{w_stat:.4f}}, p = {{w_pval:.6f}}")
    print(f"  {{'' if w_pval < 0.05 else 'NOT '}}significant at alpha=0.05")
except ValueError as e:
    print(f"\\nWilcoxon test skipped: {{e}}")

# Kruskal-Wallis by kappa (if multiple kappas)
if "kappa" in df.columns and df["kappa"].nunique() > 1:
    groups = [g[EFFECT_COL].dropna().values for _, g in df.groupby("kappa") if len(g) > 0]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h_stat, h_pval = stats.kruskal(*groups)
        print(f"\\nKruskal-Wallis (effect varies by kappa?):")
        print(f"  H = {{h_stat:.4f}}, p = {{h_pval:.6f}}")
        print(f"  {{'' if h_pval < 0.05 else 'NOT '}}significant at alpha=0.05")
"""),
    ]


def _analysis_conclusions(ctx: SweepNotebookContext) -> list[dict[str, Any]]:
    meta = ctx.meta
    return [_md(f"""\
---
### 14. Conclusions

This notebook analyzed the **{meta['mechanism_name']}** sweep experiment.

**Key outputs:**
- Treatment effect distribution (box plot + histogram)
- Kappa response curves (how the mechanism performs under varying liquidity stress)
- Parameter sensitivity heatmaps (which parameter combinations matter most)
- Regression analysis (quantitative decomposition of the effect)
- Statistical tests (formal evidence for/against mechanism effectiveness)

**To reproduce:** Re-run all cells with the sweep data in `{{SWEEP_DIR}}`.
""")]


# ── Public API ───────────────────────────────────────────────────────

def generate_sweep_notebook(
    out_dir: Path,
    sweep_type: str,
    output_dir: Path | None = None,
) -> Path:
    """Generate a Jupyter notebook for a completed sweep.

    Args:
        out_dir: Root directory of the sweep (contains aggregate/comparison.csv).
        sweep_type: One of "dealer", "bank", "nbfi".
        output_dir: Where to write the notebook. Defaults to out_dir/aggregate/analysis/.

    Returns:
        Path to the generated .ipynb file.
    """
    out_dir = Path(out_dir)
    if sweep_type not in _SWEEP_META:
        raise ValueError(f"Unknown sweep_type: {sweep_type!r} (valid: {list(_SWEEP_META)})")

    ctx = SweepNotebookContext.from_sweep_dir(out_dir, sweep_type)

    # Build cells
    cells: list[dict[str, Any]] = []

    # Theory sections
    for section_fn in [
        _section_bilancio_intro,
        _section_kalecki_ring,
        _section_parameters,
        _section_phases,
        _section_profile_trader,
        _section_profile_dealer,
        _section_profile_vbt,
        _section_profile_bank,
        _section_profile_cb,
        _section_profile_nbfi,
        _section_experimental_design,
        _section_metrics,
    ]:
        section_cells = section_fn(ctx)
        if section_cells:
            cells.extend(section_cells)

    # Analysis code cells
    for analysis_fn in [
        _analysis_data_loading,
        _analysis_summary_stats,
        _analysis_distributions,
        _analysis_kappa_response,
        _analysis_heatmaps,
        _analysis_regression,
        _analysis_statistical_tests,
        _analysis_conclusions,
    ]:
        analysis_cells = analysis_fn(ctx)
        if analysis_cells:
            cells.extend(analysis_cells)

    # Assign cell IDs
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"

    # Build notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }

    # Write
    if output_dir is None:
        output_dir = out_dir / "aggregate" / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{sweep_type}_sweep_presentation.ipynb"
    nb_path = output_dir / filename
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))

    return nb_path
