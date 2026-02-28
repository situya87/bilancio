# Plan 046: NBFI Behavioral Improvements

## Context

The NBFI lender shows near-zero lending effect at κ≥0.5 and only +3% at κ=0.3.
Deep investigation of actual run data (κ=0.3, n=50) reveals WHY:

**Empirical findings:**
- 151 loan applications rejected vs 62 created (71% rejection rate)
- 88% of rejected borrowers are deeply insolvent (coverage < -0.5), only 3% are near-pass (0.3–0.5)
- 9 unique borrowers served out of 50 agents; 5 of 9 defaulted anyway (64% of capital wasted)
- Near-miss defaults: H16 (shortfall 14 on 544 borrowed), H38 (shortfall 10 on 1026 borrowed)
- Revolving debt traps: H33 got 10 loans over 6 days, H41 got 12 loans over 6 days
- Same 27 agents default in both passive and NBFI runs — lender shifts WHO defaults, not HOW MANY
- Settlement improved +34% (6572 vs 4905) — lender helps agents settle more before defaulting

**Root cause:** In the ring at κ<1, most defaults are structural (genuine insolvency), not
liquidity-driven. The lender helps with timing mismatches but can't fix the structural deficit.
Current behavior wastes capital on revolving loans to a few agents and misses near-miss defaults
due to fixed 2-day loan maturity.

## Sweep Results (κ=0.3, n=50, μ=0.5, ρ=0.90)

```
κ    │  δ_passive   δ_dealer   δ_lender │  trading  lending
─────────────────────────────────────────────────────────────
0.3  │     0.8278     0.7783     0.7978 │  +0.0494  +0.0300
0.5  │     0.6987     0.6797     0.6983 │  +0.0190  +0.0004
1.0  │     0.6185     0.6192     0.6185 │  -0.0007  +0.0000
```

Trading (dealer) consistently outperforms lending (NBFI) across all κ values.

## Detailed Event Analysis (κ=0.3 NBFI Run)

### Coverage Gate Rejection Distribution
```
Coverage range       │ Count │ %    │ Meaning
< -0.5 (insolvent)   │   133 │  88% │ Owe far more than they have — no loan helps
-0.5 to 0 (stressed) │    14 │   9% │ Borderline, might default on loan
0 to 0.3 (marginal)  │     0 │   0% │ Could potentially repay
0.3 to 0.5 (near-pass)│    4 │   3% │ Just below threshold — saveable
```

### Borrower Outcomes
```
Agent │ Loans │ Total $ │ Loan Days          │ Outcome
H17   │   6   │     42  │ 5,5,6,6,7,7        │ SURVIVED
H23   │   4   │     12  │ 0,0,1,1            │ SURVIVED
H33   │  10   │    676  │ 0,0,1,1,3,3,4,4,5,5│ SURVIVED (revolving)
H41   │  12   │    810  │ 7-12 (daily)        │ SURVIVED (revolving)
H16   │   4   │    544  │ 2,2,3,3            │ DEFAULTED (shortfall 14)
H31   │   6   │    954  │ 2,2,7,7,8,8        │ DEFAULTED + loan default
H38   │   6   │  1,026  │ 0,0,1,1,3,3        │ DEFAULTED (shortfall 10)
H39   │   4   │    190  │ 4,4,5,5            │ DEFAULTED (shortfall 124)
H6    │  10   │     40  │ 8-12 (daily)        │ DEFAULTED (shortfall 11)
```

**Key observation:** H16 (shortfall 14 on 544 borrowed), H38 (shortfall 10 on 1026 borrowed),
H6 (shortfall 11 on 40 borrowed) — these agents ALMOST survived. The 2-day maturity forces
them to repay before their receivable arrives, creating a refinancing treadmill that eventually
exhausts both them and the lender.

### Default Timeline Comparison
```
Day │ Passive defaults │ NBFI defaults │ Loans made
  0 │        0         │       0       │     6
  1 │        5         │       5       │     6
  3 │        7         │       7       │     6
  5 │        2         │       1       │     6
  7 │        3         │       3       │     6
  9 │        4         │       4       │     4
 12 │        1         │       2       │     4
 20 │        1         │       2       │     0
```

Default counts are nearly identical. Lender saved H41 (would have defaulted in passive)
but H36 defaulted instead. Net: same count, different agents, more settlement.

## Proposed Improvements (5 independently toggleable features)

### Phase 1A: Maturity Matching (HIGH IMPACT)

**Problem:** Loans mature in 2 days. Borrower receives income on day 7 but must repay loan on
day 2. Creates a refinancing treadmill. Near-miss defaults (shortfall 10–14) happen because loan
matures before receivable arrives.

**Solution:** Match loan maturity to borrower's next incoming receivable date.

**New LenderProfile fields:**
```python
maturity_matching: bool = False          # default=False for backward compat
min_loan_maturity: int = 2               # floor
```

**New helper** in `lending.py`:
```python
def _nearest_receivable_day(system, agent_id, current_day, max_horizon) -> int | None:
    """Earliest day a non-defaulted receivable is due for this agent."""
```

**Logic:** When enabled, set `effective_maturity = nearest_receivable_day - current_day + 1`
(clamped to `[min_loan_maturity, max_loan_maturity]`).

### Phase 1B: Concentration Limits (MEDIUM IMPACT, TRIVIAL)

**Problem:** H33 got 10 loans, H41 got 12 — capital concentrates on revolving borrowers instead
of spreading to new agents.

**Solution:** Cap loans per borrower per day.

**New LenderProfile field:**
```python
max_loans_per_borrower_per_day: int = 0  # 0 = unlimited (backward compat)
```

**Logic:** Track `daily_loan_count: dict[str, int]` in execution loop. Skip if count >= limit.
Emit `NonBankLoanRejectedConcentration` event.

### Phase 2: Cascade-Aware Ranking (HIGH IMPACT, HIGH COMPLEXITY)

**Problem:** Ranking by expected profit optimizes lender returns, not system stability. Preventing
one "keystone" agent's default prevents cascading downstream defaults.

**Solution:** Rank by downstream damage potential instead of/blended with profit.

**New LenderProfile fields:**
```python
ranking_mode: str = "profit"             # "profit" | "cascade" | "blended"
cascade_weight: Decimal = Decimal("0.5") # weight in blended mode
```

**New helper:**
```python
def _downstream_obligation_total(system, agent_id, current_day) -> int:
    """Sum of payables this agent OWES. If it defaults, creditors lose these receivables."""
```

**Ranking formulas:**
- `"profit"`: current behavior (`rate × (1 - p_default)`)
- `"cascade"`: `coverage_ratio × normalized_downstream × (1 - p_default)`
- `"blended"`: weighted combination

### Phase 3: Graduated Coverage Gate (MEDIUM IMPACT, LOW COMPLEXITY)

**Problem:** Binary gate at 0.5 rejects 4 borrowers with coverage 0.3–0.5 who were potentially
saveable.

**Solution:** Replace binary rejection with continuous rate premium for sub-threshold coverage.
Still reject deeply insolvent (coverage < -1.0).

**New LenderProfile fields:**
```python
coverage_mode: str = "gate"              # "gate" | "graduated"
coverage_penalty_scale: Decimal = Decimal("0.10")  # rate premium per unit below threshold
```

### Phase 4: Preventive Lending (MEDIUM IMPACT, MEDIUM COMPLEXITY)

**Problem:** Lender only acts after shortfall exists. By then, cascading defaults have already
destroyed borrowers' receivables, pushing coverage deeply negative.

**Solution:** Proactively lend to agents whose receivables are from stressed counterparties,
before the cascade hits.

**New LenderProfile fields:**
```python
preventive_lending: bool = False
prevention_threshold: Decimal = Decimal("0.3")  # min issuer p_default to trigger
```

**New helper:**
```python
def _receivables_at_risk(system, agent_id, current_day, horizon, default_probs, threshold) -> int:
    """Sum of receivables where issuer's default probability exceeds threshold."""
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/bilancio/decision/profiles.py` | Add 8 new fields to `LenderProfile` |
| `src/bilancio/engines/lending.py` | Add 4 helpers, modify `run_lending_phase` logic |
| `src/bilancio/config/models.py` | Add fields to `LenderScenarioConfig` |
| `src/bilancio/config/apply.py` | Pass new fields through (~line 731 and ~1059) |
| `src/bilancio/experiments/ring.py` | Wire new params in `__init__` and scenario dict |
| `src/bilancio/experiments/balanced_comparison.py` | Add to `BalancedComparisonConfig` |

## Implementation Order

| Phase | Feature | Depends On | Effort | Impact |
|-------|---------|-----------|--------|--------|
| 1A | Maturity matching | None | 3-4h | HIGH |
| 1B | Concentration limits | None | 1-2h | MEDIUM |
| 2 | Cascade-aware ranking | None | 4-5h | HIGH |
| 3 | Graduated coverage gate | None | 2-3h | MEDIUM |
| 4 | Preventive lending | Phase 2 helper | 4-5h | MEDIUM |

**Recommended:** Implement 1A + 1B first (parallel), then sweep to measure impact before
adding more complexity.

## Design Notes

- **Backward compat:** All new defaults match current behavior
- **Term premium:** When maturity matching extends a loan beyond 2 days, consider adding
  `rate *= (1 + 0.01 × (maturity - 2))` to compensate for longer exposure
- **Banking interaction:** Lines 319-321 in lending.py override maturity for banking mode.
  Maturity matching should take precedence when enabled.
- **Capital scaling** (lender_share=0.10→0.20) requires no code changes — just sweep parameter
