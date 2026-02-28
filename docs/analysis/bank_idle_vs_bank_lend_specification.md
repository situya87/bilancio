# Experiment 3: Bank Idle vs Bank Lend

## Purpose

Isolate the effect of **bank lending** on system defaults and losses. Both arms have banks as payment processors (deposits, reserves, CB corridor). The only difference is whether banks actively lend to stressed firms.

- **bank_idle** (baseline): Banks hold deposits and reserves but do NOT lend.
- **bank_lend** (treatment): Banks identify stressed borrowers and issue loans.

No VBT, no dealer, no NBFI lender. Pure banking channel.

## Metric

```
bank_lending_effect = δ_idle − δ_lend
```

Positive effect = lending reduces defaults.

---

## Ring Topology

N agents arranged in a directed ring. Agent H_i owes a payable to H_{i+1} (mod N). Each payable has a face value, a due day, and is settled via bank deposits.

### Sweep-grid parameters

These vary across runs in a sweep. Each combination produces one (idle, lend) pair.

| Parameter | Symbol | Default | Controls |
|-----------|--------|---------|----------|
| `kappa` | κ | varies | Liquidity ratio: L₀ = κ × Q_total. Lower = more stressed. |
| `concentration` | c | 1 | Dirichlet α for debt distribution. Lower = more unequal. |
| `mu` | μ | 0 | Maturity timing skew ∈ [0,1]. 0 = front-loaded, 1 = back-loaded. |
| `outside_mid_ratio` | ρ | 0.90 | Outside-money discount factor (affects capitalization). |
| `seed` | — | 42 | PRNG seed. Use `--n-replicates` for multiple seeds per cell. |

### Fixed ring parameters

Same for all runs in a sweep.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_agents` | 100 | Ring size (number of households/firms) |
| `maturity_days` | 10 | Payment horizon in days |
| `Q_total` | 10,000 | Total debt in the system |
| `face_value` | 20 | Face value per payable ticket |

### How κ determines initial cash

```
L₀ = κ × Q_total
```

L₀ is distributed to agents and immediately deposited at banks. With κ=0.3, the system has 3,000 in deposits to cover 10,000 in debt — severe stress. With κ=2.0, 20,000 in deposits for 10,000 in debt — abundant.

### How concentration determines debt distribution

Payable amounts drawn from Dirichlet(c). With c=1 (default), some agents owe much more than others. With c→∞, all payables are equal.

### How μ determines due dates

μ=0: all payables due on day 1 (maximum simultaneous stress). μ=0.5: evenly spread across the horizon. μ=1: all due on the last day.

---

## Banking Infrastructure (Both Arms)

### Banks

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_banks` | 5 | Number of banks |
| `reserve_ratio` | 0.50 | Bank reserves = reserve_ratio × total deposits |

At setup:
1. All agent cash is deposited at banks (agents hold no free cash).
2. Agents are assigned to 1–3 banks (sliding window).
3. Bank reserves are minted: R = reserve_ratio × D.
4. Banks hold only reserves as liquid assets (no cash instrument).

### CB Corridor (Treynor Pricing Kernel)

The bank's pricing kernel determines deposit rates (r_D) and loan rates (r_L) from its inventory position on a 2-day funding plane. The corridor widens under stress.

| Parameter | Symbol | Default | Sensitive to κ? |
|-----------|--------|---------|-----------------|
| `r_base` | — | 0.01 | No (fixed) |
| `r_stress` | — | 0.04 | **Yes** — adds stress premium |
| `omega_base` | — | 0.01 | No (fixed) |
| `omega_stress` | — | 0.02 | **Yes** — widens corridor |
| `reserve_target_ratio` | — | 0.10 | No (fixed) |
| `symmetric_capacity_ratio` | X*/R_tar | 2.0 | No (fixed) |
| `alpha` | α | 0.005 | No (fixed, but inputs are endogenous) |
| `gamma` | γ | 0.002 | No (fixed, but inputs are endogenous) |
| `interest_period` | — | 2 days | No (fixed) |

**Corridor formulas** (kappa-dependent):

```
stress_factor = max(0, 1 − κ) / (1 + κ)
r_mid   = r_base + r_stress × stress_factor
Ω       = omega_base + omega_stress × stress_factor
r_floor = r_mid − Ω/2     (reserve remuneration)
r_ceil  = r_mid + Ω/2     (CB borrowing rate)
```

At κ=0.3: stress_factor ≈ 0.538, r_mid ≈ 0.0315, Ω ≈ 0.0208.
At κ=1.0: stress_factor = 0, r_mid = 0.01, Ω = 0.01.

**Treynor spread** (from inventory position):

```
λ = S_fund / (2X* + S_fund)           # layoff probability
I = λ × Ω                              # inside width
m(x) = M − (Ω / 2(X*+S)) × x         # midline
r_D = m(x) − I/2                       # deposit rate (bid)
r_L = m(x) + I/2                       # loan rate (ask)
```

For multi-ticket transactions (integrated pricing):
```
n_tickets = ceil(amount / ticket_size)
midpoint_inventory = x₀ + direction × (n−1) × S / 2
r_L_integrated = tilted_midline(midpoint_inventory) + I/2
```

Larger loans cost more because each ticket walks inventory further from target.

### CB Standing Facilities

| Parameter | Default | Sensitive to κ? |
|-----------|---------|-----------------|
| `cb_rate_escalation_slope` | 0.05 | No (fixed rate) |
| `cb_max_outstanding_ratio` | 2.0 | No (fixed cap) |
| `cb_lending_cutoff_day` | maturity_days | No (auto-derived) |

End-of-day: if a bank's reserves fall below target, the CB automatically lends to top up. CB rate escalates with utilization:

```
r_CB(t) = base_rate + escalation_slope × (outstanding / cap)
```

### Deposit Interest

Every `interest_period` days (default: 2), banks pay interest on deposits:

```
interest = deposit_amount × r_D
```

This is a book entry — deposit balance increases, no reserve movement.

---

## Settlement Phase (Both Arms)

### Payment waterfall

When a payable matures on its due day:

1. **Deposits first**: debtor pays from bank deposits.
   - Pays from lowest-r_D bank first (minimize opportunity cost).
   - Creditor receives at highest-r_D bank (maximize return).
   - Cross-bank payments route via interbank reserves.
2. **Cash fallback**: if deposits insufficient, use cash (rare in banking mode — agents hold no free cash after setup).
3. **Default**: if still short, the payable defaults.

### Default cascade

When an agent can't fully pay a payable:

1. `ObligationDefaulted` event logged (shortfall = creditor's loss).
2. Agent marked as defaulted and expelled.
3. **Pro-rata recovery**: remaining deposits/cash distributed to all creditors proportionally.
4. **Write-off**: all remaining liabilities of the expelled agent are cancelled (`ObligationWrittenOff`).
5. **Ring reconnection**: predecessor's next payable is redirected to successor (ring shrinks by 1).

---

## Bank Lending Phase (bank_lend Only)

This phase runs daily in the bank_lend arm. It is entirely skipped in bank_idle.

### Borrower identification

Each day, the bank scans all non-defaulted households/firms for upcoming shortfalls:

```
shortfall = obligations_due_in_next_N_days − (deposits + cash)
```

Where N = `loan_maturity` = max(2, maturity_days × loan_maturity_fraction).

Default: loan_maturity_fraction = 0.5, so with maturity_days=10, loan_maturity = 5 days.

Borrowers are sorted by shortfall descending (most stressed first).

### Lending decision pipeline

For each eligible borrower:

| Step | Check | Default | Effect |
|------|-------|---------|--------|
| 1. Fool-me-once | Has borrower defaulted on a prior bank loan? | — | Skip if yes (permanent ban) |
| 2. One-loan limit | Does borrower already have an outstanding loan? | — | Skip if yes |
| 3. Find cheapest bank | Select bank with lowest r_L | — | Rate competition across banks |
| 4. Credit risk pricing | r_L = treynor_r_L + credit_risk_loading × P_default | See below | Higher risk = more expensive |
| 5. Credit rationing | Is P_default > max_borrower_risk? | 0.4 | Reject if too risky |
| 6. Coverage check | coverage = (liquid − obligations + receivables) / repayment | 0 (disabled) | Reject if undercapitalized |
| 7. Bank capacity | Five capacity constraints (see below) | See below | Reject if bank overextended |
| 8. Borrow vs sell | Is selling to dealer cheaper? | — | Skip if dealer bid is better (irrelevant in bank-only mode) |
| 9. Execute loan | Create BankLoan + credit deposit | — | Money creation |

### Credit risk parameters

| Parameter | CLI default | BankComparisonConfig default | BankProfile default | Meaning |
|-----------|------------|------------------------------|---------------------|---------|
| `credit_risk_loading` | **0.5** | 0.5 | 0 | Rate adder per unit P_default |
| `max_borrower_risk` | **0.4** | 0.4 | 1.0 | Credit rationing threshold |
| `min_coverage_ratio` | 0 | 0 | 0 | Balance sheet coverage floor |

**Important**: Three layers of defaults exist. The CLI defaults now match `BankComparisonConfig` defaults: **credit_risk_loading=0.5, max_borrower_risk=0.4** (credit-sensitive pricing enabled by default). `BankProfile` raw defaults remain 0 / 1.0 for backward compatibility, but both the CLI and `BankComparisonConfig` override these to enable credit risk pricing out of the box.

**Feb 2025 wide sweep** ran with the OLD CLI defaults (credit_risk_loading=0, max_borrower_risk=1.0). In that configuration, the bank lent to all borrowers at a flat Treynor rate regardless of default risk. Current CLI defaults now include credit risk pricing.

### Default probability estimation

The bank uses a shared `RiskAssessor` with Bayesian updating:

- **Initial prior**: P_default = 0.15 (no-history default).
- **Updates**: Each observed default increases the posterior for that agent.
- **Informedness**: controlled by `default_observability` (default: 1.0 = sees all defaults).

**Kappa-informed priors**: When `credit_risk_loading > 0`, the bank borrows the dealer subsystem's `RiskAssessor` which uses `kappa_informed_prior`: initial_prior = 1/(1+κ). This means priors automatically scale with system stress — at κ=0.3, P_0 ≈ 0.77; at κ=1.0, P_0 = 0.50; at κ=2.0, P_0 ≈ 0.33. The wiring is in `run.py:272-277`. When `credit_risk_loading = 0`, the prior is irrelevant (it multiplies zero).

### Bank capacity constraints

| Check | Formula | Default threshold |
|-------|---------|-------------------|
| Reserve/deposit ratio | R / D after loan ≥ ¾ × reserve_target_ratio | ¾ × 10% = 7.5% |
| Projected reserves | min projected R − expected cross-bank outflow ≥ floor | Reserve floor |
| Per-borrower exposure | existing_to_borrower + loan ≤ max_single × total_capacity | 20% of capacity |
| Total exposure | total_loans + loan ≤ max_total × initial_reserves | 150% of reserves |
| Daily lending | today_loans + loan ≤ max_daily × current_reserves | 50% of reserves |

| Capacity parameter | Default | Fixed/configurable |
|--------------------|---------|-------------------|
| `max_single_exposure_ratio` | 0.20 | Fixed in BankProfile |
| `max_total_exposure_ratio` | 1.50 | Fixed in BankProfile |
| `max_daily_lending_ratio` | 0.50 | Fixed in BankProfile |

### Loan execution

When a loan is approved:

1. `BankLoan` instrument created (bank asset, borrower liability).
2. Borrower's deposit credited by loan principal — **money creation** (no reserve movement).
3. `BankLoanIssued` event logged with rate, principal, maturity day.

### Loan repayment

On maturity day (current_day + loan_maturity):

- If borrower has sufficient deposits: debit deposit, retire loan, log `BankLoanRepaid`.
- If insufficient: partial repayment from available deposits, log `BankLoanDefault`.
  - Borrower added to fool-me-once list (no future loans).
  - Loss = repayment_due − recovered (counted as `bank_credit_loss`).

---

## Daily Simulation Loop

```
For each day 0..maturity_days:

  Phase B: Settlement
    B2: Settle all payables due today (deposit waterfall → default cascade)

  Phase B3: Banking Operations (if enable_banking)
    1. Bank lending phase        (bank_lend only)
    2. Bank loan repayments      (bank_lend only)
    3. Deposit interest accrual  (both arms)
    4. Interbank repayments      (both arms)

  Phase C: End-of-day
    1. CB backstop (top up reserves to target)
    2. Refresh bank quotes for next day
```

---

## Parameter Sensitivity Summary

### Parameters that change with κ (sweep grid)

| What changes | How |
|-------------|-----|
| Total initial deposits | L₀ = κ × Q_total |
| CB corridor mid | r_mid = 0.01 + 0.04 × max(0,1−κ)/(1+κ) |
| CB corridor width | Ω = 0.01 + 0.02 × max(0,1−κ)/(1+κ) |
| Inside spread (Treynor) | I = λ × Ω (wider corridor → wider spread) |
| Bank loan rate | r_L = midline + I/2 (higher at low κ) |
| Default probability prior | With credit_risk_loading > 0: kappa-informed prior P_0 = 1/(1+κ). With credit_risk_loading = 0: fixed at 0.15 (irrelevant since it multiplies zero). |

### Parameters fixed across all runs

| Category | Parameters |
|----------|-----------|
| Ring structure | n_agents=100, maturity_days=10, Q_total=10000, face_value=20 |
| Banking setup | n_banks=5, reserve_ratio=0.50 |
| Treynor kernel | r_base=0.01, r_stress=0.04, omega_base=0.01, omega_stress=0.02 |
| Treynor internal | reserve_target_ratio=0.10, symmetric_capacity_ratio=2.0, α=0.005, γ=0.002 |
| Loan terms | loan_maturity_fraction=0.50, interest_period=2 |
| Credit risk | credit_risk_loading=0.5, max_borrower_risk=0.4, min_coverage_ratio=0 (CLI defaults; credit-sensitive pricing) |
| Capacity limits | max_single=20%, max_total=150%, max_daily=50% |
| CB backstop | escalation_slope=0.05, max_outstanding=2.0 |
| Risk assessment | initial_prior=0.15, default_observability=1.0 |
| Settlement | deposit-first waterfall, expel-agent default mode, rollover enabled |

### Parameters NOT used in bank experiment

These exist in the codebase but are irrelevant (no VBT/dealer/NBFI):

- All trader behavior parameters (risk_aversion, planning_horizon, aggressiveness, etc.)
- All dealer/VBT parameters (spreads, sensitivities, shares)
- All NBFI lender parameters
- All trading motive parameters

---

## Loss Channels

| Loss type | Source | Formula |
|-----------|--------|---------|
| `payable_default_loss` | Creditor eats shortfall when debtor defaults on payable | Sum of `ObligationDefaulted.shortfall` + `ObligationWrittenOff` (payable) amounts |
| `deposit_loss_gross` | Depositor eats loss when bank fails | Sum of `ObligationWrittenOff` (bank_deposit) amounts |
| `total_loss` | Real-economy loss | payable_default_loss + deposit_loss_gross |
| `bank_credit_loss` | Bank eats loss on defaulted loans | Sum of (repayment_due − recovered) from `BankLoanDefault` |
| `cb_backstop_loss` | CB eats loss on frozen bank loans | Sum of `CBLoanFreezeWrittenOff` amounts |
| `intermediary_loss` | All intermediary losses | bank_credit_loss + cb_backstop_loss |
| `system_loss` | Total system loss | total_loss + intermediary_loss |

### Key finding (Feb 2025 sweep)

Bank lending reduces defaults (δ) but **increases** system losses across all tested κ values. The bank absorbs risk from the real economy but concentrates it in the intermediary layer. Agents who borrow and then default on loan repayment get expelled anyway — their remaining payables are written off, and the bank eats additional credit losses.

---

## CLI

```bash
# Feb 2025 wide sweep (used OLD CLI defaults: credit_risk_loading=0, max_borrower_risk=1.0)
# Note: current CLI defaults now include credit risk pricing (0.5 / 0.4).
uv run bilancio sweep bank \
  --out-dir out/bank_sweep_043_wide \
  --n-agents 100 \
  --maturity-days 10 \
  --kappas "0.2,0.3,0.5,0.7,1.0,1.5,2.0" \
  --concentrations "1" \
  --mus "0" \
  --outside-mid-ratios "0.90" \
  --base-seed 42 \
  --n-replicates 3

# Current defaults already include credit risk pricing (credit_risk_loading=0.5, max_borrower_risk=0.4).
# To reproduce the old no-risk-pricing behavior, pass zeros explicitly:
uv run bilancio sweep bank \
  --out-dir out/bank_sweep_no_credit_risk \
  --kappas "0.3,0.5,1.0,1.5,2.0" \
  --credit-risk-loading 0 \
  --max-borrower-risk 1.0 \
  --base-seed 42 \
  --n-replicates 3
```

Total runs = len(kappas) × len(concentrations) × len(mus) × len(outside_mid_ratios) × n_replicates × 2 arms.
