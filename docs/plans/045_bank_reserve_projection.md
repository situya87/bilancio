# Plan 045: Bank Reserve Projection for Lending Decisions

## Problem

Banks at κ=0.3 lend freely on day 0, then immediately need CB backstop loans
because reserves drain when borrowers pay ring creditors at other banks. The
Treynor kernel's withdrawal forecast (`compute_withdrawal_forecast`) only
tracks outflows from **loan-originated** deposits, missing the much larger
outflows from regular ring settlements.

Example (bank_1, κ=0.3):
- Reserves: 300, deposits: 600
- Bank's loan outflow estimate: 118 (from its own lending)
- Actual gross outflow: 881 (ring settlements to bank_2)
- CB loan needed: 581

The bank is blind to the structural drain and approves loans it shouldn't.

## Root Cause

1. `compute_withdrawal_forecast()` iterates `self.outstanding_loans` only.
   Initial/setup deposits are not tracked.
2. `_bank_can_lend()` Check 2 uses current reserves minus the single new
   loan's outflow. It ignores all other same-day outflows (ring settlements,
   other loans issued earlier in the same phase).

## Fix

### Part 1: Settlement outflow forecast

Add `compute_settlement_outflow_forecast()` to `BankTreynorState`. This
scans the bank's clients' upcoming payable obligations to estimate how much
will flow cross-bank in the next N days.

```
For each client of this bank:
    For each payable they owe (liability) due in [current_day, current_day + horizon]:
        creditor_id = payable.asset_holder_id
        creditor_bank = banking.get_primary_bank(creditor_id)
        if creditor_bank != self.bank_id:
            outflow += payable.amount
        else:
            # Intra-bank: no reserve movement
            pass

Similarly estimate inflows:
    For each client of this bank:
        For each payable they are OWED (asset) due in [current_day, current_day + horizon]:
            debtor_id = payable.liability_issuer_id
            debtor_bank = banking.get_primary_bank(debtor_id)
            if debtor_bank != self.bank_id:
                inflow += payable.amount
```

Net settlement forecast = outflow - inflow per day.

### Part 2: Integrate into refresh_quote path

In `refresh_quote()`, replace the flat `path[0] = reserves - W_t_rem` with
a per-day projection that includes settlement outflows:

```
path[0] = reserves - W_t_rem - settlement_net_outflow[0]
path[s] = path[s-1] + delta + settlement_net_inflow[s] - settlement_net_outflow[s]
```

This makes L* correctly reflect the bank's true reserve trajectory.

### Part 3: Strengthen _bank_can_lend

Replace Check 2 with a path-based check:

```
Use the bank's current projected min_path (from last refresh_quote).
Subtract the new loan's expected cross-bank outflow.
If projected_min - new_outflow < reserve_floor → don't lend.
```

This is simpler and more correct than the current approach because it
inherits the full settlement forecast from refresh_quote.

## Implementation

### Files to modify:
1. `src/bilancio/engines/banking_subsystem.py`
   - Add `compute_settlement_outflow_forecast()` on `BankTreynorState`
   - Add `_settlement_net_by_day` cache field
   - Update `refresh_quote()` path to include settlement flows
   - Store `min_projected_reserves` after computing path

2. `src/bilancio/engines/bank_lending.py`
   - Update `_bank_can_lend()` Check 2 to use the projected min reserves

3. `tests/engines/test_bank_lending.py`
   - Add test: bank refuses to lend when settlement drain exceeds reserves
   - Add test: bank lends when reserves are sufficient after settlement drain

## Expected Outcomes

- κ=0.3: Banks refuse most/all loans (correctly — they can't afford the
  reserve drain). CB backstop losses drop dramatically.
- κ=0.5: Banks lend conservatively, only when they have reserve headroom.
- κ≥1.0: Banks lend freely (ample reserves), no change in behavior.
