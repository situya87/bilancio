# Plan 043: Experimental Design Redesign

## Problem Statement

The current balanced comparison framework runs up to 7 arms from a single passive baseline. The banking arms have a fundamental identification problem, and the NBFI arms have an unnecessary design messiness.

### The Banking Identification Problem

The banking arm (`bank_passive`) differs from the passive baseline in **multiple simultaneous dimensions**:

1. **Settlement infrastructure**: cash-based → deposit-based
2. **Claim distribution**: 75% trader / 25% VBT+Dealer → 75% trader / 25% VBT+Dealer (with VBT/Dealer holding zero cash)
3. **Reserve level**: no reserves → reserves derived from VBT/Dealer capital pool (arbitrary amount)
4. **Bank equity**: no banks → banks with negative equity (reserves 2085 < deposits 5500)
5. **Interbank market**: absent → active (with CB backstop)
6. **Bank lending**: absent → active

If δ differs between passive and bank_passive, we cannot attribute it to lending (dimension 6) because dimensions 1-5 also changed. This is an **identification problem**, not a measurement problem.

**The intermediary loss difference is real, not a measurement artifact.** Investigation confirmed that intermediary_loss in the bank_passive arm (≈20,000) vs passive (≈1,400) is driven by genuine bank credit losses (17,313 from 101 defaulted loans out of 178 issued) and CB backstop writeoffs (1,590), not by phantom VBT/Dealer holdings. The VBT/Dealer instrument loss component is approximately the same in both arms (≈1,400) because it depends on the number of obligor defaults, not on whether VBT/Dealer hold cash.

### The NBFI Design Messiness

The current NBFI arm (`mode="nbfi"`) zeros VBT/Dealer cash and gives it to the NBFI. This is unnecessarily messy but **probably does not affect δ**. VBT/Dealer cash in the passive arm is inert — they don't trade, don't make payments, don't participate in settlement. Removing their cash doesn't change payment dynamics. The NBFI's lending is the only functional change.

However, the setup is confusing and the zeroed VBT/Dealer create needless questions about what changed. A cleaner NBFI-idle baseline would make the comparison self-evidently correct.

## Design: Three Independent Experiments

Each experiment has its **own internal baseline** that differs from the treatment in exactly one dimension.

### Experiment 1: Trading (existing, works well)

**What it tests**: Does secondary market trading by a dealer reduce default rates?

| Arm | VBT/Dealer | Trader Claims | Settlement | Trading |
|-----|-----------|---------------|------------|---------|
| Passive (baseline) | Full cash + instruments | 75% | Cash | Off |
| Active (treatment) | Full cash + instruments | 75% | Cash | On |

- **Only difference**: dealer trading enabled/disabled
- **Metric**: `trading_effect = δ_passive - δ_active`
- **Status**: Working correctly today. No changes needed.

### Experiment 2: NBFI Lending

**What it tests**: Does non-bank lending reduce default rates?

| Arm | VBT/Dealer | NBFI | Trader Claims | Settlement | Lending |
|-----|-----------|------|---------------|------------|---------|
| NBFI-idle (baseline) | Full cash + instruments | Present, full cash, idle | 75% | Cash | Off |
| NBFI-lend (treatment) | Full cash + instruments | Present, full cash, lending | 75% | Cash | On |

- **Only difference**: NBFI lending enabled/disabled
- **NBFI capital**: Own endowment (parameter: `nbfi_share`), NOT reallocated from VBT/Dealer
- **VBT/Dealer**: Identical in both arms (full cash, passive/no trading)
- **Total system liquidity**: Identical in both arms (NBFI has same cash regardless)
- **Metric**: `lending_effect = δ_nbfi_idle - δ_nbfi_lend`
- **Key change from current**: NBFI gets its own capital instead of stealing VBT/Dealer cash. VBT/Dealer are untouched.

**Note on the current setup**: The current NBFI arm probably gives the correct δ treatment effect despite the VBT/Dealer cash zeroing, because that cash is inert in the passive baseline anyway. The redesign makes the comparison cleaner and more obviously correct, but is not fixing a broken measurement — it's fixing a confusing setup.

**NBFI capital as additional liquidity**: The NBFI's cash is additional liquidity in the system. This is controlled for because both arms have the same NBFI cash. But the system with NBFI has more total liquidity than the passive-only system. This means we **cannot directly compare** δ_passive (Exp 1) with δ_nbfi_idle (Exp 2) — they're different systems. What we CAN compare is the **treatment effects** (how much does trading help vs how much does lending help).

### Experiment 3: Bank Lending

**What it tests**: Does bank credit creation reduce default rates in a deposit-based economy?

| Arm | VBT/Dealer | Banks | Trader Claims | Settlement | Lending |
|-----|-----------|-------|---------------|------------|---------|
| Bank-idle (baseline) | None | Present, no lending | 100% | Deposits | Off |
| Bank-lend (treatment) | None | Present, lending enabled | 100% | Deposits | On |

- **Only difference**: bank lending enabled/disabled
- **No VBT/Dealer**: Traders hold 100% of claims, banks process payments via deposits
- **Bank reserves**: A parameter (e.g., `reserve_ratio`), NOT derived from VBT/Dealer
- **Total system resources**: Identical (same deposits, same reserves)
- **Metric**: `bank_lending_effect = δ_bank_idle - δ_bank_lend`

**Key design decisions**:
1. **Reserve ratio**: Initial reserves as fraction of deposits. This is the key parameter to sweep. At 100%, banks are fully reserved and interbank market is dormant. At lower ratios, fractional reserve dynamics and CB lending activate.
2. **No VBT/Dealer**: Banks are the only intermediaries. Eliminates the question of what VBT/Dealer are doing in a banking system.
3. **Bank-idle baseline**: Banks exist, hold deposits, process payments — but don't lend. This isolates the lending effect from the deposit infrastructure effect.

**Bank-idle dynamics**: In bank-idle mode with 100% reserve ratio, banks simply pass through payments (deposit transfers). With fractional reserves (< 100%), banks may face liquidity stress even without lending, due to uneven payment flows creating reserve imbalances → interbank borrowing → CB backstop. This is actually interesting: it shows the "cost" of deposit-based settlement itself.

### Experiment 4: Combined (future work)

**What it tests**: What happens when both trading and bank lending are available?

| Arm | VBT/Dealer | Banks | Trading | Lending |
|-----|-----------|-------|---------|---------|
| Passive + Bank-idle | Passive (full) | No lending | Off | Off |
| Active + Bank-lend | Active (full) | Lending enabled | On | On |
| (Optional 2×2) | ... | ... | ... | ... |

This is a 2×2 factorial design. Future work once Experiments 1-3 are clean.

## Comparison Across Experiments

Since each experiment has a different system setup, we **cannot directly compare absolute δ values** across experiments. What we CAN compare:

| Comparison | Valid? | What it tells us |
|-----------|--------|-----------------|
| δ_passive vs δ_active (Exp 1) | Yes | Trading effect |
| δ_nbfi_idle vs δ_nbfi_lend (Exp 2) | Yes | NBFI lending effect |
| δ_bank_idle vs δ_bank_lend (Exp 3) | Yes | Bank lending effect |
| trading_effect vs lending_effect | Qualitative | Which mechanism helps more (same κ) |
| δ_passive vs δ_nbfi_idle | No | Different systems (different total liquidity) |
| δ_passive vs δ_bank_idle | No | Different systems (cash vs deposits) |

The qualitative comparison is the one we actually want: **as a function of κ, which relief mechanism provides more reduction in defaults?** We can plot all three treatment effects on the same κ axis and compare curves.

## Implementation Plan

### Phase 1: Clean Up Existing Arms

1. **Remove banking/NBFI arms from `BalancedComparisonSweep`** — or refactor them into separate sweep commands

### Phase 2: NBFI Experiment (New)

1. Add `balanced_mode_override="nbfi_idle"` mode to compiler:
   - VBT/Dealer: full cash + instruments (same as passive)
   - NBFI: present with own cash endowment, lending DISABLED
   - New parameter: `nbfi_share` (fraction of system liquidity for NBFI, independent of VBT/Dealer)
2. Modify existing `"nbfi"` mode:
   - VBT/Dealer: full cash + instruments (NOT zeroed)
   - NBFI: present with same cash endowment, lending ENABLED
3. Create `sweep nbfi` CLI command (or extend `sweep balanced --experiment nbfi`)

### Phase 3: Bank Experiment (New)

1. Add `balanced_mode_override="bank_idle"` mode to compiler:
   - No VBT/Dealer agents at all
   - Traders hold 100% of claims
   - All trader cash deposited at banks
   - Bank reserves = `reserve_ratio × total_deposits`
   - Bank lending DISABLED
2. Add `balanced_mode_override="bank_lend"` mode:
   - Same as bank_idle but lending ENABLED
3. New parameter: `reserve_ratio` (replaces the current `equalize_capacity` logic)
4. Create `sweep bank` CLI command

### Phase 4: Comparison Dashboard

1. Build cross-experiment comparison visualization
2. Plot treatment effects (not absolute δ) on shared κ axis
3. Report intermediary loss breakdown per experiment

## Parameter Mapping

### NBFI Experiment Parameters (new)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nbfi_share` | NBFI cash as fraction of total system liquidity | 0.10 |
| `nbfi_lending_enabled` | Boolean toggle for treatment arm | True/False |

All other parameters inherited from Experiment 1 (same ring structure, same VBT/Dealer, same trader behavior).

### Bank Experiment Parameters (new)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reserve_ratio` | Initial reserves / total deposits | 0.50 |
| `n_banks` | Number of banks | 5 |
| `bank_lending_enabled` | Boolean toggle for treatment arm | True/False |
| `credit_risk_loading` | Bank sensitivity to borrower risk | (existing) |
| `max_borrower_risk` | Credit rationing threshold | (existing) |
| `cb_rate_escalation_slope` | CB cost pressure | (existing) |
| `cb_max_outstanding_ratio` | CB lending cap | (existing) |

Trader behavior parameters (risk_aversion, planning_horizon, etc.) are NOT relevant in the bank experiment since there's no dealer trading.

## What This Replaces

The following current arms become obsolete:
- `bank_passive` — replaced by Experiment 3 bank_idle
- `bank_dealer` — replaced by Experiment 3 bank_lend (or Experiment 4)
- `bank_dealer_nbfi` — replaced by Experiment 4 combined
- Current `nbfi` mode — replaced by Experiment 2 with clean capital allocation
- Current `lender` mode — was already a simpler version, subsumed by Experiment 2

The current `passive` and `active` arms (Experiment 1) remain unchanged.

## Migration Path

1. Implement Experiments 2 and 3 as new sweep commands
2. Keep existing balanced sweep working (backwards compat) but deprecate banking/NBFI arms
3. Once new experiments validated, remove deprecated modes from compiler
