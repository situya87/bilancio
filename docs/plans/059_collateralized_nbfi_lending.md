# Plan 059: Collateralized NBFI Lending in the Kalecki Ring

## Goal

Upgrade the NBFI lender from unsecured lending (with an optional flat collateral cap) to a
full **pledge-based collateralized lending** regime where borrowers pledge specific receivable
payables, the lender applies risk-sensitive haircuts, and on borrower default the lender
seizes and liquidates pledged collateral through the existing dealer market.

**Why now?** The current `collateralized_terms` flag (Plan 049) is a shallow cap — it limits
loan size to a fraction of aggregate receivables but never records a pledge, never freezes
pledged assets from trading, and never seizes anything on default. This means the lender has
no actual recovery mechanism when borrowers default, which inflates intermediary losses and
limits the parameter space where lending is viable. A proper collateral mechanism creates a
new channel for loss absorption and opens up experimentally interesting questions about
wrong-way risk, collateral-vs-trading competition, and secured vs unsecured lending frontiers.

## Context: The Kalecki Ring Collateral Problem

In a Kalecki ring, each agent holds receivables from **one predecessor** (simple ring) or
**k predecessors** (k-regular/Erdos-Renyi). This creates a distinctive collateral structure:

1. **Single-name concentration** — in a simple ring, a borrower's only collateral is a
   payable from one issuer. The lender must accept or reject a single-name exposure.
2. **Wrong-way risk** — if H3 needs a loan because it can't pay H4, H3's collateral is a
   payable from H2. But H2 may be stressed for the same reason (H1 can't pay H2). Collateral
   quality is correlated with borrower distress.
3. **Collateral vs dealer channel** — agents can sell receivables to the dealer for immediate
   cash, OR pledge them to the lender for a loan. These are competing liquidity channels.
   The design must make this tradeoff explicit.
4. **Topology dependence** — in denser topologies (k-regular, Erdos-Renyi), borrowers hold
   diversified collateral pools, making collateral more valuable and wrong-way risk less acute.

## Scope

**In scope:**
- Explicit pledge recording: borrower pledges specific payable(s) to lender
- Pledged payables frozen from dealer trading
- Risk-sensitive haircut computation (issuer default probability + maturity premium)
- Collateral seizure on borrower default (lender becomes `holder_id`)
- Lender liquidation via dealer market or hold-to-maturity
- Pledge release on loan repayment
- New metrics: collateral utilization, wrong-way risk rate, lender recovery rate
- Sweep parameters for collateral advance rate, haircut sensitivity
- Integration with existing NBFI sweep arms

**Out of scope:**
- Repo/repurchase agreements (separate instrument type — future plan)
- Margin calls / dynamic re-margining during loan life (simplification: haircut set at origination)
- Cross-collateralization across multiple loans
- Rehypothecation (lender re-pledging or trading collateral while loan is live)
- Partial pledge (pledge full payables only, not fractional)

## Design

### D1. Pledge Data Structure

A new lightweight record tracks the pledge relationship:

```python
@dataclass
class CollateralPledge:
    pledge_id: str              # unique ID
    loan_id: str                # the NonBankLoan this secures
    payable_id: str             # the pledged Payable contract ID
    borrower_id: str            # who pledged it
    lender_id: str              # who holds the lien
    pledged_day: int            # when pledge was created
    face_value: int             # payable face value at time of pledge
    haircut: Decimal            # haircut applied (0–1)
    collateral_value: int       # face_value × (1 - haircut)
    status: str                 # "active", "released", "seized"
```

Store in `system.state.collateral_pledges: dict[str, CollateralPledge]` (keyed by pledge_id).
Also maintain an index `system.state.pledged_payable_ids: set[str]` for O(1) lookup during
dealer eligibility filtering.

### D2. Haircut Computation

Replace the flat `collateral_advance_rate` with a risk-sensitive haircut per payable:

```python
def compute_haircut(
    p_default: Decimal,         # issuer default probability (from risk assessor)
    remaining_tau: int,         # days to maturity
    maturity_days: int,         # system maturity horizon (for normalization)
    base_haircut: Decimal,      # floor haircut (new param, default 0.05)
    risk_sensitivity: Decimal,  # how much p_default affects haircut (default 1.0)
    maturity_sensitivity: Decimal,  # how much longer maturity increases haircut (default 0.5)
) -> Decimal:
    """
    haircut = base_haircut + risk_sensitivity × p_default
              + maturity_sensitivity × (remaining_tau / maturity_days)

    Clamped to [base_haircut, 0.95].
    """
```

**Intuition:**
- At `p_default=0.15`, `tau=5`, `maturity=10`: haircut ≈ 0.05 + 0.15 + 0.25 = 0.45
  → collateral value = 55% of face
- At `p_default=0.05`, `tau=2`, `maturity=10`: haircut ≈ 0.05 + 0.05 + 0.10 = 0.20
  → collateral value = 80% of face
- Allows the lender to be more aggressive on short-maturity, low-risk collateral

### D3. Lending Decision with Collateral

Modify `_collect_lending_opportunities()` in `lending.py`:

1. **Identify pledgeable payables** — borrower's asset payables that are:
   - Not already pledged (`payable_id not in pledged_payable_ids`)
   - Not yet matured (`due_day > current_day`)
   - Issuer not defaulted
   - Within the lender's horizon

2. **Compute per-payable haircut** using D2 formula

3. **Select payables to pledge** — greedy by collateral value (highest value first),
   until `sum(collateral_values) >= loan_amount` or all eligible exhausted

4. **Cap loan** at total collateral value of selected payables

5. **Create pledge records** for each selected payable

6. **Existing unsecured path** preserved when `collateral_mode="none"` (backward compat)

### D4. Freezing Pledged Payables from Trading

In `dealer_trades.py` (or `dealer_wiring.py`), filter out pledged payables:

```python
# In payables_to_tickets() or the eligible-sellers filter:
if payable_id in system.state.pledged_payable_ids:
    continue  # Cannot trade pledged collateral
```

This is a one-line filter addition. Pledged payables are economically "locked up" — the
borrower retains legal ownership but cannot sell or transfer them.

### D5. Pledge Release on Loan Repayment

In `run_loan_repayments()`, after successful repayment:

```python
# Find pledges for this loan and release them
for pledge in system.state.collateral_pledges.values():
    if pledge.loan_id == loan_id and pledge.status == "active":
        pledge.status = "released"
        system.state.pledged_payable_ids.discard(pledge.payable_id)
```

Emit a `CollateralReleased` event. The payable becomes tradeable again.

### D6. Collateral Seizure on Borrower Default

Two scenarios trigger seizure:

**Scenario A: Borrower defaults on the loan** (in `run_loan_repayments()`):
- Lender seizes pledged payables: set `payable.holder_id = lender_id`
- Remove from `pledged_payable_ids` (no longer pledged — lender owns it)
- Pledge status → "seized"
- Emit `CollateralSeized` event
- Lender can now hold to maturity (collect face value if issuer pays) or sell via dealer

**Scenario B: Borrower defaults on a ring payable** (in settlement phase):
- If the defaulting agent has active pledges, the lender seizes those payables
- This happens during the existing default/expel cascade in `settlement.py`
- The lender's recovery = min(collateral_value, loan_outstanding)

**Scenario C: Pledged payable's ISSUER defaults** (wrong-way risk):
- The pledged payable becomes worthless (issuer can't pay)
- Lender absorbs the loss — this is the wrong-way risk the haircut is supposed to cover
- Emit `CollateralImpaired` event with loss amount
- Pledge status → "impaired" (distinct from seized)

### D7. Lender Liquidation Strategy

After seizing collateral, the lender must decide: hold or sell?

Simple decision rule (configurable via `LenderProfile`):

```python
lender_liquidation_mode: str = "hold_to_maturity"  # or "immediate_sell"
```

- **hold_to_maturity** (default): Keep the payable, collect at settlement if issuer pays.
  Lower cost, but ties up capital and exposes lender to issuer default.
- **immediate_sell**: Sell via dealer market at current bid. Immediate cash recovery but
  at a discount (dealer haircut). Requires the lender to be eligible as a dealer customer.

For Phase 1, implement `hold_to_maturity` only. `immediate_sell` requires making the lender
a dealer-eligible trader, which is a larger wiring change (future enhancement).

### D8. Events

New event types:

| Event | Fields | When |
|-------|--------|------|
| `CollateralPledged` | pledge_id, loan_id, payable_id, borrower, lender, haircut, collateral_value | Loan creation |
| `CollateralReleased` | pledge_id, loan_id, payable_id | Loan repayment |
| `CollateralSeized` | pledge_id, loan_id, payable_id, recovery_value | Borrower default |
| `CollateralImpaired` | pledge_id, payable_id, issuer_id, loss_amount | Issuer default |
| `LoanRejectedNoCollateral` | borrower_id, shortfall | No eligible payables to pledge |

### D9. New Parameters

| Parameter | Default | Why this default | Backward-compatible? |
|-----------|---------|------------------|---------------------|
| `collateral_mode` | `"none"` | Feature off by default; replaces bool `collateralized_terms` | Yes — `"none"` = current behavior |
| `base_haircut` | `0.05` | 5% floor even for zero-risk collateral | Yes — only active when mode ≠ none |
| `haircut_risk_sensitivity` | `1.0` | 1:1 mapping of p_default to haircut | Yes — only active when mode ≠ none |
| `haircut_maturity_sensitivity` | `0.5` | Moderate maturity penalty | Yes — only active when mode ≠ none |
| `lender_liquidation_mode` | `"hold_to_maturity"` | Conservative default | Yes — only active when mode ≠ none |
| `max_collateral_concentration` | `1.0` | No per-issuer cap (ring = one issuer) | Yes — only active when mode ≠ none |

The existing `collateralized_terms: bool` and `collateral_advance_rate: Decimal` are
**deprecated** in favor of `collateral_mode`. Migration: `collateralized_terms=True` maps
to `collateral_mode="soft_cap"` (preserving current flat-cap behavior exactly).

`collateral_mode` values:
- `"none"` — no collateral (current default)
- `"soft_cap"` — Plan 049 behavior: loan capped at advance_rate × aggregate receivables,
  no pledge, no seizure (backward compat)
- `"pledged"` — full pledge-based collateral (this plan)

## Location in Codebase

| Change type | Path | Description |
|-------------|------|-------------|
| New file | `src/bilancio/domain/instruments/collateral.py` | `CollateralPledge` dataclass |
| Modified | `src/bilancio/engines/lending.py` | Pledge creation, haircut computation, collateral selection |
| Modified | `src/bilancio/engines/lending.py` | `run_loan_repayments()` — pledge release and seizure |
| Modified | `src/bilancio/engines/settlement.py` | On agent default — trigger collateral seizure |
| Modified | `src/bilancio/engines/dealer_trades.py` | Filter pledged payables from trading eligibility |
| Modified | `src/bilancio/engines/system.py` | Add `collateral_pledges` and `pledged_payable_ids` to State |
| Modified | `src/bilancio/decision/profiles.py` | Add new fields to `LenderProfile` |
| Modified | `src/bilancio/config/models.py` | Add collateral params to config model |
| Modified | `src/bilancio/config/apply.py` | Wire collateral params through pipeline |
| Modified | `src/bilancio/experiments/nbfi_comparison.py` | Expose collateral params in NBFI sweep |
| Modified | `src/bilancio/ui/cli/_sweep_nbfi.py` | CLI flags for collateral params |
| Modified | `src/bilancio/analysis/metrics.py` | New collateral metrics |
| New test | `tests/engines/test_collateralized_lending.py` | Unit tests for pledge lifecycle |
| New test | `tests/integration/test_collateral_integration.py` | End-to-end pledge + seizure + dealer freeze |

## Sweep Surface

### Backend layers

| Layer | Touched? | How? |
|-------|----------|------|
| **Domain** (agents, policy, instruments) | Yes | New `CollateralPledge` dataclass; new state fields |
| **Decision** (profiles, strategies) | Yes | New fields on `LenderProfile`; haircut computation |
| **Engines** (phases, settlement, dealer) | Yes | Modified lending phase, settlement default handler, dealer eligibility filter |
| **Ops** (transfers, settlement mechanics) | No | — |
| **Scenarios** (ring builder, config) | No | Ring structure unchanged; collateral derived from existing payables |
| **State** (agent state, system state) | Yes | `collateral_pledges`, `pledged_payable_ids` on State |

### Sweep pipeline layers

| Layer | Touched? | How? |
|-------|----------|------|
| **CLI params** | Yes | New `--lender-collateral-mode`, `--lender-base-haircut`, etc. |
| **Sweep config** | Yes | New fields on `NBFIComparisonConfig` |
| **Runner logic** | No | Existing lender arm handles it; no new arm type |
| **Metrics collection** | Yes | New metrics: `collateral_utilization`, `wrong_way_loss_rate`, `lender_recovery_rate` |
| **Post-sweep reports** | Yes | New columns in comparison.csv |
| **Pre-flight checks** | Yes | New viability check V9: collateral viability |

### Interaction expectations

- With `collateral_mode="pledged"`, lender recovery rate should be > 0 (currently always 0)
- At moderate κ (0.5–1.5), collateralized lending should reduce net intermediary loss vs uncollateralized
- At very low κ (< 0.3), wrong-way risk should dominate: lender recovery poor, collateral often impaired
- Pledging should reduce dealer trading volume (some payables frozen)
- `delta_total` impact: small improvement at moderate κ, negligible at extremes
- New metric `wrong_way_loss_rate` should be strongly correlated with 1/κ

### Viability check V9: Collateral viability

```
At representative κ:
  p_issuer = 1/(1+κ)
  haircut = base_haircut + risk_sensitivity × p_issuer + maturity_sensitivity × (τ/T)
  collateral_value = face × (1 - haircut)
  loan_cap = collateral_value per pledgeable payable

  If haircut > 0.90: collateral nearly worthless → warn ⚠️
  If loan_cap < min_shortfall: no loans will be made → warn ⚠️
  If p_issuer > 0.5: wrong-way risk dominates → warn ⚠️ (but experimentally interesting)
```

## New Metrics

| Metric | Definition | Aggregation |
|--------|-----------|-------------|
| `collateral_utilization` | pledged_face_value / total_receivable_face_value | Per-day and total |
| `wrong_way_loss_rate` | impaired_collateral_value / total_pledged_value | Per-run |
| `lender_recovery_rate` | recovered_via_seizure / total_loan_defaults | Per-run |
| `pledge_freeze_ratio` | pledged_payables / total_payables | Per-day |
| `collateral_haircut_mean` | mean haircut across all active pledges | Per-day |

## Acceptance Criteria

1. **Pledge lifecycle works end-to-end**: A loan creates pledge records; repayment releases them;
   borrower default seizes them. Verified via unit test with explicit state inspection.

2. **Pledged payables are frozen**: A pledged payable does not appear as a tradeable ticket in
   the dealer phase. Verified by checking ticket registry after pledging.

3. **Haircut is risk-sensitive**: Higher issuer default probability and longer maturity produce
   larger haircuts. Verified via parameterized unit test.

4. **Seized collateral settles correctly**: After seizure, the lender (as new `holder_id`)
   receives payment at maturity if the issuer is solvent. Verified via integration test.

5. **Wrong-way risk materializes**: When the pledged payable's issuer defaults, the collateral
   is impaired and the lender absorbs the loss. Verified via integration test.

6. **Backward compatibility**: `collateral_mode="none"` produces identical results to the
   current codebase. `collateral_mode="soft_cap"` produces identical results to the current
   `collateralized_terms=True`.

7. **Sweep integration**: NBFI sweep with `--lender-collateral-mode pledged` completes
   without errors and produces new collateral metrics in comparison.csv.

8. **Smoke sweep**: At κ=0.5 with n=20, collateralized lending shows non-zero
   `lender_recovery_rate` and `collateral_utilization`.

## Experimental Questions This Enables

| # | Question | Sweep design |
|---|----------|-------------|
| 1 | Does collateralized lending reduce intermediary losses vs unsecured? | 2-arm: `collateral_mode=none` vs `pledged`, sweep κ |
| 2 | At what κ does wrong-way risk dominate? | Single arm `pledged`, sweep κ from 0.1 to 4, measure `wrong_way_loss_rate` |
| 3 | Does pledging crowd out dealer trading? | Compare `pledge_freeze_ratio` and trading volume with/without |
| 4 | What's the optimal haircut sensitivity? | Sweep `haircut_risk_sensitivity` from 0.5 to 2.0 |
| 5 | How does topology affect collateral quality? | Ring vs k-regular vs Erdos-Renyi with `pledged` mode |
| 6 | Secured vs unsecured lending frontier | 3-metric frontier (default_relief, inst_loss_shift, net_system_relief) |

## Migration from Plan 049 Collateral Fields

| Old field | New equivalent | Notes |
|-----------|---------------|-------|
| `collateralized_terms: bool` | `collateral_mode: str` | `True` → `"soft_cap"`, `False` → `"none"` |
| `collateral_advance_rate: Decimal` | `collateral_advance_rate: Decimal` | Kept, only used when mode = `"soft_cap"` |

Old CLI flags (`--lender-collateralized-terms`, `--lender-collateral-advance-rate`) remain
functional and map to `collateral_mode="soft_cap"`. New flag `--lender-collateral-mode` takes
precedence if both are specified.

## Implementation Phases

### Phase 1: Core Pledge Mechanics (MVP)
- `CollateralPledge` dataclass
- State fields (`collateral_pledges`, `pledged_payable_ids`)
- Haircut computation
- Pledge creation in lending phase
- Pledge release on repayment
- Dealer freeze filter
- Unit tests

### Phase 2: Default Handling
- Collateral seizure on borrower default (loan repayment path)
- Collateral seizure on borrower default (settlement/expel path)
- Collateral impairment on issuer default
- `hold_to_maturity` liquidation
- Integration tests

### Phase 3: Metrics & Sweep Wiring
- New metrics computation and CSV output
- CLI flags and config wiring
- NBFI sweep integration
- Viability check V9
- Smoke sweep verification

### Phase 4: Backward Compatibility & Migration
- `collateral_mode` migration from old bool flag
- Backward compat tests
- Deprecation warnings for old flags
