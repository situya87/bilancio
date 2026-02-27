# Plan 042: Fix Bank Lending Feedback Loop

## Problem Statement

The bank lending mechanism has a broken feedback loop that causes accelerating credit creation with declining-to-negative interest rates. Three root causes were identified from the 041 sweep results:

1. **Treynor projection missing deposit outflow forecast** — When a bank issues a loan, it creates a deposit. The borrower will spend this deposit (mostly cross-bank), causing a reserve outflow. But the 10-day reserve projection in `refresh_quote()` only counts the future loan repayment *inflow*, making each loan appear to *improve* the bank's position. This causes rates to fall after each loan, triggering more lending.

2. **CB rates completely flat** — `rate_escalation_slope=0` and `max_outstanding_ratio=0` mean the CB charges a static rate regardless of how much banks borrow, and imposes no lending cap. Banks can refinance at the CB without limit.

3. **No effective bank lending limits** — The only capacity check (`_bank_can_lend`) verifies that the post-loan reserve/deposit ratio stays above 75% of the target ratio. Since lending is money creation (deposits increase, reserves don't move), this barely constrains lending. Result: 414 loans totaling 27,285 principal against 13,673 original debt (200% of system debt).

## Paper Reference

Banks-as-Dealers (BD_Lesson_2_V2), Sections 5-6 and Appendix 7:

### Withdrawal Forecast (W_t^rem)

The paper tracks three quantities (Section 6.1, p30):
- **W_t^c**: Forecast of total day-t withdrawals (internal to pricing kernel)
- **W_t^real**: Cumulative realized withdrawals within day t
- **W_t^rem** := max(0, W_t^c - W_t^real): Remaining expected withdrawals

The projected reserve path starts from *effective* reserves (Appendix 7.1.8, p39):

```
R̂(t) = R_t^mid - W_t^rem
R̂(t+s) = R̂(t+s-1) + ΔR_{t+s}^old + E[ΔR_{t+s}^new | r_D, r_L],  s=1..10
```

Cash-tightness incorporates the withdrawal forecast:

```
L* = max{0, R_min - min_{h∈{t,...,t+10}} R̂(h)}
```

### CB Top-Up (Section 6.1.8, p33)

End-of-day CB top-up converts any intraday overdraft into a 2-day CB loan:

```
B_t^new = max{0, R_t^target - R_t^mid}
```

The cost i_B is the CB borrowing rate. In the paper it's static, but we extend it with escalation.

## Implementation

### Fix 1: Deposit Outflow Forecast in Treynor Projection

**File**: `src/bilancio/engines/banking_subsystem.py`

**Change**: In `BankTreynorState.refresh_quote()`, compute W_t^rem and subtract from path[0].

**How to estimate W_t^c**:
For each outstanding loan at this bank, the borrower may still have some/all of the loan proceeds as a deposit. When they spend it cross-bank, it causes a reserve outflow. The expected outflow is:

```python
# Loan-origin deposits still at this bank
loan_origin_deposits = sum(
    get_deposit_at_bank(system, loan.borrower_id, self.bank_id)
    for loan in self.outstanding_loans.values()
)
# Cross-bank fraction: (n_banks - 1) / n_banks
cross_bank_fraction = (n_banks - 1) / n_banks
W_t_c = loan_origin_deposits * cross_bank_fraction
```

This directly addresses the broken feedback: issuing a loan now makes the bank look *worse* (lower effective reserves) rather than better.

**New fields on BankTreynorState**:
- `withdrawal_forecast: int = 0` — W_t^c for today
- `realized_withdrawals: int = 0` — W_t^real for today (reset each day)

### Fix 2: CB Rate Escalation with Borrowing Volume

**File**: `src/bilancio/domain/agents/central_bank.py`, `src/bilancio/config/apply.py`

**Change**: Set meaningful defaults for CB escalation parameters when banking is enabled.

New defaults (applied in `apply.py` / `ui/run.py` when banking is active):
- `rate_escalation_slope = 0.05` — CB rate increases by 5% per unit utilization
- `escalation_base_amount = Q_total` — already wired
- `max_outstanding_ratio = 2.0` — cap CB lending at 2× total system debt

The effective CB rate becomes:
```
r_CB(t) = base_rate + 0.05 × (outstanding_CB_loans / Q_total)
```

At 100% utilization (outstanding = Q_total), the rate increases by 5 percentage points.
At 200% utilization (the cap), the rate increases by 10 percentage points.

### Fix 3: CB Pressure Overlay on Treynor Logic

**Files**: `src/bilancio/engines/banking_subsystem.py`, `src/bilancio/engines/bank_lending.py`

Unlike the general Treynor story, our borrowers don't walk away when rates are high — they need liquidity to survive. So the rate channel is weak on the demand side. But the CB creates a **hard line**: when it stops lending, banks face a reserve crunch. Banks should be forward-looking about this.

Two overlays, both working through the CB constraint:

#### A. Soft overlay: CB-scaled cash-tightness

In `refresh_quote()`, after computing L*, scale it by CB pressure:

```
L*_effective = L* × (1 + cb_pressure)
```

Where `cb_pressure = u² / (1 - u)` and `u = CB_outstanding / CB_cap`:
- u=0.0 → pressure=0 (no stress, backward compat)
- u=0.5 → pressure=0.5
- u=0.8 → pressure=3.2
- u=0.9 → pressure=8.1
- u=1.0 or frozen → pressure=10.0

This transmits the CB's constraint through the Treynor framework: as the CB approaches its cap, the bank's cash-tightness metric rises sharply, pushing both r_D and r_L up via the α × L* and γ × ρ tilts.

#### B. Hard overlay: CB backstop gate

In `_bank_can_lend()`, before issuing a loan, the bank asks: "if I lend this and the deposit leaves cross-bank, will I need CB backstop? Can the CB provide it?"

```python
expected_outflow = amount × (n_banks - 1) / n_banks
post_outflow_reserves = reserves - expected_outflow
if post_outflow_reserves < reserve_floor:
    needed_cb = reserve_floor - post_outflow_reserves
    if not cb.can_lend(outstanding, needed_cb):
        return False  # CB can't backstop → don't lend
```

The binding constraint is the CB's own cap — the bank internalizes it.

### Fix 4: Borrower Balance Sheet Assessment

**File**: `src/bilancio/engines/bank_lending.py`, `src/bilancio/decision/profiles.py`

The bank examines each borrower's balance sheet before lending. Unlike pure Treynor (where the bank never says "no"), in the Kalecki ring borrowers will borrow at any rate — so the bank must be selective.

**Assessment logic** — project the borrower's net cash position at loan maturity:

```python
liquid = cash + deposits                    # current liquid assets
obligations = payable_liabilities + loan_repayments  # due before loan matures
quality_receivables = receivables from non-defaulted counterparties  # due before loan matures
net_resources = liquid - obligations + quality_receivables
coverage = net_resources / loan_repayment   # loan_repayment = amount × (1 + rate)
```

**What the bank checks**:
1. **Size of mismatch**: How large is the shortfall relative to available resources?
2. **Timing alignment**: Do receivables arrive before the loan matures?
3. **Counterparty quality**: Are the borrower's receivables from solvent agents? Receivables from defaulted counterparties are worth zero.

**New BankProfile field**: `min_coverage_ratio` (default=0.0, backward compat)
- 0.0 = never reject (no assessment)
- 0.5 = borrower needs net resources ≥ 50% of loan repayment
- 1.0 = borrower must fully cover the loan from projected resources

### Fix 5: Bank Lending Exposure Limits (Safety Net)

**File**: `src/bilancio/engines/bank_lending.py`, `src/bilancio/decision/profiles.py`

These are secondary safety nets, not the primary economic mechanism:
1. **Per-borrower cap**: 20% of total loan capacity
2. **Total exposure cap**: 150% of current reserves
3. **Daily lending cap**: 50% of current reserves

### Fix 6: Wire CB Escalation Defaults for Banking Arms

**File**: `src/bilancio/ui/run.py`, `src/bilancio/experiments/balanced_comparison.py`

When banking is enabled, automatically set CB escalation parameters if they're at zero. Ensures backward compatibility while activating the mechanism for new runs.

## Design Philosophy

The five braking mechanisms form a hierarchy:

1. **Treynor rate channel (Fix 1 + 3A)**: Deposit outflow forecast + CB pressure scaling makes L* rise as the bank lends more. Rates increase. This is the paper's mechanism.
2. **Borrower assessment (Fix 4)**: Bank examines borrower's balance sheet — size of mismatch, timing of receivables, counterparty quality. Rejects borrowers who can't plausibly repay. This is the bank's **selectivity** mechanism.
3. **CB backstop gate (Fix 3B)**: Bank checks if CB can cover the expected reserve drain. If not, don't lend. This is the CB's "hard line."
4. **CB rate escalation (Fix 2)**: System-level brake — as all banks collectively borrow from CB, the corridor ceiling rises.
5. **Exposure limits (Fix 5)**: Hard caps as a last-resort safety net.

In a well-calibrated system, mechanisms 1-4 should prevent excessive lending before the safety nets in mechanism 5 bind. The Treynor logic, borrower assessment, and CB constraint are the economic channels; the exposure limits just prevent pathological cases.

The **borrower assessment** is key for the Kalecki ring context: unlike the general Treynor story, borrowers here don't walk away when rates are high — they need liquidity to survive. So the rate channel alone is weak on the demand side. The bank compensates by being selective about *who* it lends to, rather than relying solely on *what rate* it charges.

## Expected Impact

- Bank lending should be self-limiting: as a bank lends more, its Treynor rate rises (deposit outflow forecast makes reserves look lower + CB pressure amplifies cash-tightness).
- Borrower assessment filters out structurally insolvent borrowers: agents whose receivables (or liquid assets) can't cover the loan repayment are rejected.
- CB escalation provides a system-level brake: as banks collectively borrow more from CB, the corridor ceiling rises, pushing all bank rates up.
- The CB backstop gate provides a hard stop: banks won't lend if they foresee an unfunded reserve drain.
- Exposure limits provide ultimate caps as a safety net.
- The lending rotation pattern (banks taking turns) should still occur but with rising rates, not falling.
