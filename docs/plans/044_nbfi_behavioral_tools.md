# Plan 044: NBFI Behavioral Lending Tools

## Motivation

Sweep results (Job `plaza-everyone-parcel-tinsel`) show the NBFI lender reducing
default rates (δ) but *increasing* total system losses — the lender absorbs losses
that exceed the defaults it prevents. At κ=0.5 system loss rises from 22.4% to
38.7% with NBFI active.

Root causes:

1. **No receivable quality filter.** The NBFI computes a borrower's coverage ratio
   using *all* receivables, including those owed by already-defaulted agents.
   The bank's `_assess_borrower` discounts these to zero. Result: NBFI lends to
   agents who look solvent but whose assets are worthless.

2. **No coverage floor.** The NBFI has no minimum coverage ratio — it lends to
   anyone with a shortfall who passes the P(default) screen. The bank requires
   `coverage ≥ min_coverage_ratio` before approving.

3. **Static capacity.** The NBFI uses `max_total_exposure × initial_capital` as
   a fixed lending ceiling. When loans default, `existing_loan_exposure` still
   counts impaired loans at full value, so the NBFI over-estimates its capital
   and keeps lending. The bank's capacity is tied to reserves, which shrink
   mechanically when loans default.

## Design

Port three tools from the bank lending engine (`bank_lending.py`) into the NBFI
lending engine (`lending.py`). Each is independent and additive.

### Tool 1: Quality-Adjusted Coverage Assessment

**What:** When computing a borrower's coverage ratio, discount receivables from
defaulted counterparties to zero — exactly as `bank_lending._assess_borrower`
does (lines 392–407).

**Where:** New helper `_quality_adjusted_receivables()` in `lending.py`, replacing
the current `_get_receivables_due_within()` in the coverage computation.

**Changes:**
- `lending.py`: Add `_quality_adjusted_receivables(system, agent_id, current_day, horizon)`
  that filters out receivables whose `liability_issuer_id` is in
  `system.state.defaulted_agent_ids`.
- `lending.py` line ~194: Use `_quality_adjusted_receivables()` instead of
  `_get_receivables_due_within()` when computing the coverage ratio for the
  LenderProfile-based pricing path.

**Parameters:** None new — this is always-on when LenderProfile is used. The old
`_get_receivables_due_within()` remains available for backward compat but is not
used in the coverage computation.

### Tool 2: Minimum Coverage Ratio Gate

**What:** After computing the quality-adjusted coverage ratio, reject borrowers
whose coverage falls below a threshold. Mirrors `bank_lending._assess_borrower`
+ `min_coverage_ratio` check (lines 80–96).

**Where:** New gate in `run_lending_phase()` between the screening step and
the pricing step.

**Changes:**
- `LendingConfig`: Add `min_coverage_ratio: Decimal = Decimal("0")` (0 = disabled,
  backward-compatible).
- `LenderProfile`: Add `min_coverage_ratio: Decimal = Decimal("0.5")` (default:
  require borrower to cover 50% of repayment from quality assets).
- `LenderScenarioConfig`: Add `min_coverage_ratio: Decimal = Decimal("0")`.
- `lending.py`: Add `_assess_borrower_nbfi()` function that computes:
  ```
  liquid = cash
  quality_receivables = receivables from non-defaulted agents due within horizon
  obligations = payables + loans due within horizon
  net = liquid + quality_receivables - obligations
  coverage = net / loan_repayment
  ```
  Reject if `coverage < min_coverage_ratio`.
- `lending.py`: Insert the gate in `run_lending_phase()` after `screener.is_eligible()`
  and before pricing. Log a `NonBankLoanRejectedCoverage` event when rejected.

**Parameters:**
| Param | Location | Default | Description |
|-------|----------|---------|-------------|
| `min_coverage_ratio` | LendingConfig | 0 | Coverage floor (0=disabled) |
| `min_coverage_ratio` | LenderProfile | 0.5 | Default when using profile |
| `min_coverage_ratio` | LenderScenarioConfig | 0 | Scenario-level override |

### Tool 3: Loss-Responsive Capacity

**What:** Track impaired (defaulted) loans and exclude them from the NBFI's
effective capital computation. Currently `initial_capital = lender_cash +
existing_loan_exposure` counts defaulted loans at par. With this change,
defaulted loans are excluded, so effective capital shrinks as losses mount.

**Where:** Modify the capital computation at the top of `run_lending_phase()`.

**Changes:**
- `lending.py`: Add `_get_performing_loan_exposure()` that only counts loans
  whose borrower has NOT defaulted:
  ```python
  for cid in agent.asset_ids:
      contract = system.state.contracts.get(cid)
      if contract.kind == InstrumentKind.NON_BANK_LOAN:
          borrower = contract.liability_issuer_id
          if borrower not in system.state.defaulted_agent_ids:
              total += contract.amount
  ```
- `lending.py` line ~142: Replace `_get_loan_exposure()` with
  `_get_performing_loan_exposure()` in the capital computation.
- The old `_get_loan_exposure()` remains for other uses (e.g., reporting).

**Parameters:** None — this is always-on. It's a bug fix more than a feature:
counting defaulted loans as capital is incorrect accounting.

## CLI / Sweep Wiring

### New CLI flags (sweep.py)

```
--lender-min-coverage DECIMAL    Min coverage ratio for NBFI (default: 0.5)
```

### BalancedComparisonConfig

Add:
```python
lender_min_coverage: Decimal = Field(
    default=Decimal("0.5"),
    description="NBFI min coverage ratio (0=disabled)"
)
```

### ring.py scenario generation

In the lender section of `_build_scenario()`, pass through:
```python
"min_coverage_ratio": str(config.lender_min_coverage),
```

### apply.py

Wire `min_coverage_ratio` from scenario config into `LendingConfig`.

## File Change Summary

| File | Changes |
|------|---------|
| `src/bilancio/engines/lending.py` | Add `_quality_adjusted_receivables()`, `_assess_borrower_nbfi()`, `_get_performing_loan_exposure()`. Modify `run_lending_phase()` to use quality-adjusted coverage and performing-only capital. Add coverage gate with event logging. |
| `src/bilancio/decision/profiles.py` | Add `min_coverage_ratio` to `LenderProfile` |
| `src/bilancio/config/models.py` | Add `min_coverage_ratio` to `LenderScenarioConfig` |
| `src/bilancio/config/apply.py` | Wire `min_coverage_ratio` into `LendingConfig` |
| `src/bilancio/experiments/balanced_comparison.py` | Add `lender_min_coverage` to config |
| `src/bilancio/experiments/ring.py` | Pass `min_coverage_ratio` in scenario lender section |
| `src/bilancio/ui/cli/sweep.py` | Add `--lender-min-coverage` flag |
| `tests/engines/test_nbfi_coverage.py` | Unit tests for quality-adjusted receivables, coverage gate, performing-loan capital |
| `tests/integration/test_nbfi_behavioral.py` | Integration test: run sweep with new tools, verify reduced system loss |

## Testing Strategy

1. **Unit tests** (`test_nbfi_coverage.py`):
   - `test_quality_adjusted_receivables_excludes_defaulted`: Create an agent with
     receivables from both healthy and defaulted counterparties. Verify
     `_quality_adjusted_receivables()` returns only the healthy ones.
   - `test_coverage_gate_rejects_low_coverage`: Set up a borrower with coverage=0.3
     and min_coverage_ratio=0.5. Verify the loan is rejected with a
     `NonBankLoanRejectedCoverage` event.
   - `test_coverage_gate_passes_high_coverage`: Same but coverage=0.8. Verify loan
     proceeds.
   - `test_performing_loan_exposure_excludes_defaulted`: NBFI has loans to both
     healthy and defaulted borrowers. Verify `_get_performing_loan_exposure()`
     excludes the defaulted ones.

2. **Integration test** (`test_nbfi_behavioral.py`):
   - Run a small ring scenario (n=20, κ=0.5) twice: once with old defaults
     (min_coverage=0) and once with new tools (min_coverage=0.5).
   - Verify that with new tools, NBFI issues fewer loans, takes smaller losses,
     and system_loss is lower.

## Expected Impact

With these three tools, the NBFI should:
- **Stop lending to doomed borrowers** (quality-adjusted coverage catches agents
  whose receivables are worthless).
- **Set a floor on borrower health** (min_coverage_ratio rejects structurally
  insolvent borrowers).
- **Self-regulate capacity** (performing-loan capital shrinks as losses mount,
  naturally reducing further lending).

The net effect should be fewer but higher-quality loans, lower NBFI losses, and
lower total system loss — making the NBFI arm genuinely beneficial rather than
loss-absorbing.
