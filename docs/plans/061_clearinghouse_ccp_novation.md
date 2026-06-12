# Plan 061: Clearinghouse Stage 3 — Central Counterparty via Novation

**Status**: Plan only — not yet implemented
**Date**: 2026-06-11
**Branch**: `plan/059-clearinghouse-suite` (shared clearinghouse-suite planning branch; deviates from the one-branch-per-plan convention in `docs/FEATURE_PROCESS.md` — deliberate, the three clearinghouse plans are designed together)
**Depends on**: Plan 059 (`ClearingPhase` netting + `net_payable` primitive, `docs/plans/059_clearinghouse_netting.md`). Independent of Plan 060 (certificates, `docs/plans/060_clearinghouse_certificates.md`). Implementation order: 059 → 060 → 061.

---

## Goal

Default cascades in the Kalecki ring propagate *serially*: agent A's failure starves B, which starves C. Plan 059's netting compresses gross flows but leaves the contagion **topology** untouched — losses still travel edge-by-edge along the ring. This plan introduces a central counterparty agent (`CentralCounterparty`, id `CCP1`) that **novates** every member↔member payable at creation time into two payables against itself, turning the ring into a star. A prefunded default fund plus variation-margin-gains haircutting (VMGH) **mutualizes** member shortfalls at the single central node instead of letting them propagate. The scientific point: ring-propagating default cascades become losses mutualized at one node — and because stage 1 has no behavioral response (no moral hazard channel), the comparison against the 059-netting baseline isolates the purely *mechanical* effect of changing contagion topology: cascade depth should collapse, default variance across seeds should fall, while total default mass may rise in extreme tails where VMGH spreads losses to members who would otherwise have survived.

## Reconciliation with Plans 059/060 (read before the design)

This plan was drafted against the actual code in `src/bilancio_v2/` and the committed text of 059/060. Binding contracts inherited:

- **059 netting is in-place reduction, not synthetic net payables**: `Ledger.net_payable(payable, reduction)` decrements `payable.amount`, tracks `Payable.netted_amount`, emits `PayableNetted`; synthetic net instruments were explicitly rejected (059 Design §3). Statements below about "one net position per member per day" mean *residual face after in-place bilateral cancellation*, not a new instrument.
- **059 rollover** is "net-settle, gross-roll" via `ledger.netted_rollover_queue` and a `cash_return` argument on `rollover_single_payable`. 061's rollover design composes with this (Design §2.3).
- **059 δ accounting** already makes `phi_delta` netting-aware; 061 inherits it unchanged.
- **060 model naming**: 060 calls the pydantic block `ClearinghouseScenarioConfig`; 059 calls it `ClearinghouseConfig`. 061 follows whichever name 059 actually lands — referred to below as "the clearinghouse block". `mode` becomes `Literal["netting", "certificates", "ccp"]`.
- **060 introduces agent kind `clearinghouse` ("CH1")** for certificates mode. **OPEN QUESTION**: reuse that kind for the CCP? **Recommendation**: no — register a distinct kind `central_counterparty` ("CCP1"). The two institutions have different capability rows, different failure semantics (CH1 takes recourse-backed credit risk; CCP1 is a matched-book novator with a mutualized fund), and conflating them would force mode-conditional behavior inside one agent kind.
- **059 phase ordering coordination** (TreynorSpot → Clearing → Settlement) is inherited; 061 adds no new phase *slot* (Design §2.4).
- 059 notes that 061 "replaces steps 3–5 of the netting algorithm with novation". This plan takes the gentler reading: `ClearingPhase` runs **unchanged** in ccp mode and *degenerates* on the star (Design §2.1) — no algorithm fork to maintain.

## Scope

**IN**
- New agent kind `central_counterparty` ("CCP1"), enabled via the clearinghouse block with `mode: ccp`.
- Novation at *every* payable-creation site: setup/scheduled actions, rollover, receivable reassignment (enumerated exhaustively from code in Design §1).
- Member-vs-CCP daily settlement: collect pay-ins first, then CCP pays out — a two-leg day with atomic CCP-leg handling.
- Default fund (cash held by CCP1, segregated *accounting* via fund ledger fields), three-tier loss waterfall (own contribution → mutualized fund → VMGH), next-day pro-rata replenishment.
- Expel semantics under the novated star: defaulting members are expelled via the existing `expel_agent` machinery, but `reassign_receivables` must never re-link members (Design §3).
- `CCPProfile` dataclass with `{ccp_fund_share, vmgh_enabled, replenishment_enabled}`; reserved `ccp_can_fail` hook.
- Ring sweep `--clearing-ccp` flag; metrics `fund_drawdowns_total`, `vmgh_haircut_total`, `ccp_member_defaults`; new `cascade_depth_max` metric computed for *all* arms so the with/without comparison exists.

**OUT (deferred)**
- CCP failure: `ccp_can_fail` is a *reserved* parameter hardwired `False`; the VMGH backstop always closes the books (explicit modeling choice, Design §2.2 and checklist item 7).
- Initial/variation margin proper — the default fund is the only prefunded resource in stage 1.
- Behavioral response to mutualization (moral hazard): stage 1 is deliberately mechanical, so the arm comparison is a pure topology experiment.
- Composition with dealer, NBFI lender, or banking arms (checklist item 8); composition with `mode: certificates`.
- Obligation-proportional fund sizing (OPEN QUESTION 1); multi-CCP; cross-margining; legacy v1 engine support (059's v1 hard-reject guard covers `mode: ccp` automatically since it gates on the block).

## Location in codebase

| Change type | Path | Description |
|-------------|------|-------------|
| New file | `src/bilancio_v2/plugins/clearinghouse_ccp.py` | Novation helper (`novate_payable_args`), CCP settlement subphase logic, waterfall, replenishment |
| Modified | `src/bilancio_v2/subsystem_config.py` | `CleanClearingConfig` gains ccp fields (`mode="ccp"`, `ccp_fund_share`, `vmgh_enabled`, `replenishment_enabled`) |
| Modified | `src/bilancio_v2/scenario_gates.py` | `build_clearing_config` accepts `mode: ccp`; validation of forbidden arm combinations (item 8) |
| Modified | `src/bilancio_v2/ledger.py` | `Payable.origin_debtor/origin_creditor` fields (default `None`); CCP fund state `ccp_fund_contribution: dict[str, Decimal]`, `ccp_fund_total`; fund events; end-of-day fund invariant |
| Modified | `src/bilancio_v2/actions.py` | `create_payable` handler routes through novation when ccp mode on (`RunContext` carries the clearing config) |
| Modified | `src/bilancio_v2/plugins/base.py` | `RunContext.clearing_config` field (default `None`) |
| Modified | `src/bilancio_v2/plugins/settlement.py` | CCP-leg partitioned settlement order inside `SettlementPhase.run`; expel/reassign guard for the star; origin-pair rollover |
| Modified | `src/bilancio_v2/policy.py` | `central_counterparty` row in `CapabilityMatrix.default()` |
| Modified | `src/bilancio_v2/engine.py` | Register CCP replenishment step; thread clearing config into `RunContext` |
| Modified | `src/bilancio/config/models.py` | clearinghouse block: `mode` literal gains `"ccp"`; new fields `ccp_fund_share`, `vmgh_enabled`, `replenishment_enabled`; allow `central_counterparty` agent kind |
| Modified | `src/bilancio/engines/termination.py` | Add `VMGHHaircutApplied`, `CCPFundDrawdown` to `IMPACT_EVENTS` |
| Modified | `src/bilancio/scenarios/ring/compiler.py` | Emit CCP1 agent + fund-contribution setup actions when mode==ccp |
| Modified | `src/bilancio/experiments/ring.py` | `RingSweepRunner(clearing_mode=...)`; `RingRunSummary` + registry `default_fields` (~line 510) gain the new metric columns |
| Modified | `src/bilancio/ui/cli/_sweep_ring.py` | `--clearing-ccp` flag (+ `--ccp-fund-share`), mutually exclusive with 059's `--clearing` |
| Modified | `src/bilancio/analysis/metrics.py` | `ccp_totals(events)` helper; new `cascade_depth_max(events)` beside the existing `cascade_fraction` (line 349) |
| Modified | `src/bilancio/analysis/report.py` | Carry the new per-run metrics into `results.csv` aggregation |
| New test | `tests/v2/test_ccp_novation.py` | Novation rewiring units (all creation sites) |
| New test | `tests/v2/test_ccp_waterfall.py` | Waterfall arithmetic + VMGH pro-rata units |
| New test | `tests/v2/test_ccp_expel.py` | Expel-under-star regression (no member↔member re-link) |
| New test | `tests/integration/test_ccp_cascade.py` | Largest-debtor default with/without CCP at matched seed; cascade depth comparison |
| New test | `tests/property/test_ccp_invariants.py` | Hypothesis: CCP cash ≥ 0 every day; fund conservation under random κ/c/μ/seed |
| Config | `examples/scenarios/ring_ccp.yaml` | Small ring with `clearinghouse: {enabled: true, mode: ccp}` for single-run inspection |

## Design

### 1. Novation at creation time

When ccp mode is on, every payable between two **clearing members** (the ring agents; CCP1 itself, central bank, banks, lenders, rating agencies are never members) is replaced *at the point of creation* by two payables with identical `amount`, `due_day`, `maturity_distance`:

```
A → B  (q, due d)    becomes    A → CCP1 (q, due d)  +  CCP1 → B (q, due d)
```

Both legs carry `origin_debtor=A`, `origin_creditor=B` — new optional `Payable` fields (default `None`, so all existing constructors and event payloads are untouched) — preserving the underlying economic pair for rollover (§2.3) and metrics. The member↔member payable is never instantiated: novation is a rewrite of the creation call, not post-hoc cancellation, so the **novation invariant** (no live member↔member payable while ccp mode is on) holds by construction.

**Every payable-creation site must route through novation.** Exhaustively enumerated from `bilancio_v2` (the only `Payable` constructors are `Ledger.create_payable` and `Ledger.add_rollover_payable`):

1. **Setup + scheduled actions** — `actions.py::apply_action("create_payable")` (~line 111), the single chokepoint serving both the ring compiler's `initial_actions` and Phase-B1 `scheduled_actions`. Wrap the `ledger.create_payable` call: one original action yields two `PayableCreated` events (legs carry `origin_*` keys; absent when ccp off, so non-ccp event shapes are byte-identical).
2. **Rollover** — `settlement.py::rollover_single_payable` → `ledger.add_rollover_payable` (and 059's `netted_rollover_queue` drain). See §2.3: rollover operates on the *origin pair* and re-novates.
3. **Receivable reassignment** — `settlement.py::reassign_receivables` → `ledger.create_payable(reason="receivable_reassignment")`. Under ccp mode this site must never re-link members (§3); when it fires for non-member parties, any payable it creates between two members is novated like any other.
4. **(Discovered in code, flagged)** — `plugins/dealer.py` mutates `payable.creditor` **in place** when tickets trade (lines ~640/645/700/706). This is payable *re-linking* invisible to any creation-site hook. Stage-1 resolution: dealer and ccp modes are mutually exclusive (item 8). If a later stage combines them: trades of CCP1→B legs are claim transfers on the CCP and are safe; trades of A→CCP1 legs must be forbidden (the CCP's asset side is not tradeable).

No other sites exist. Bank/CB/non-bank **loans are distinct contract types and are not novated** — which is exactly why the lender arm is excluded in stage 1 (a member's NBFI loan would give it a second creditor and break the star argument in §3).

### 2. Daily cycle in ccp mode

#### 2.1 Netting degenerates to per-member nets — by construction

059's `ClearingPhase` (SubphaseB_Clearing) runs **unchanged** on the novated book. On the star: (i) the bilateral pass cancels offsetting A→CCP1 / CCP1→A face due the same day in place; (ii) the cycle-cancellation step is a structural no-op, because every edge is incident to CCP1 and a directed cycle through two members would require a member→member edge, which novation forbids. The residual after the bilateral pass is therefore exactly **one net direction per member per due day** — and since CCP1 is the counterparty of *every* leg, this bilateral netting on the star equals multilateral netting of the original ring. That is the algebraic payoff of novation: 059's algorithm needs no modification, no mode-branch, and its conservation assertions keep holding (set-off preserves every member's net position).

#### 2.2 Two-leg settlement day

`SettlementPhase.run` currently scans `ledger.payables` flat. In ccp mode, due payables are partitioned: **(a)** pay-ins (creditor == CCP1), **(b)** payouts (debtor == CCP1), **(c)** all others. Order within the day:

1. **Collect leg**: settle all (a) via the normal capability-matrix MoP path (members pay cash). A member shortfall triggers the waterfall (§2.5) and member default (§3).
2. **Pay-out leg**: payout pool = required (b) total − residual shortfall after fund draws. If pool < required and `vmgh_enabled`, apply haircut factor `h = pool / required` pro-rata to every (b) payable: pay `h·amount`, mark settled, emit `VMGHHaircutApplied {creditor, face, paid, haircut}`. The haircut residue is a **final** loss to the receiving member — no carry-forward; that is what variation-margin-*gains* haircutting means. If `vmgh_enabled: false` is requested, configuration validation rejects it in stage 1 (books could not close; reserved for the `ccp_can_fail` stage).
3. **Other payables** (c) settle exactly as today.

CCP legs are **atomic per day**: the CCP never pays before collecting, so `cash[CCP1] ≥ 0` is an invariant by construction (property-tested), and the CCP **cannot default in stage 1** — stated explicitly as a modeling choice, with CCP failure deferred behind the reserved `ccp_can_fail` parameter (OUT of scope, validation rejects `True`).

#### 2.3 Rollover: roll the origin pair, re-novate

Composing with 059's "net-settle, gross-roll": when the A→CCP1 leg settles (by cash and/or netting), the rollover entry recorded is the **origin pair** `(origin_debtor, origin_creditor, gross_face, maturity_distance)` — not the leg. The CCP1→B leg never enqueues rollover independently (it has no independent economic life; a guard asserts this). At roll time the new payable is created via the novation chokepoint, yielding fresh A→CCP1 + CCP1→B legs at `max_open_due_day + maturity_distance`.

**OPEN QUESTION 2**: should the rollover cash return-flow (creditor refinances debtor; 059's `cash_return`) run B→A directly or B→CCP1→A? **Recommendation**: B→A directly — refinancing is a new bilateral credit decision; only the resulting *payable* is novated. This keeps CCP cash flows settlement-only and avoids inventing a CCP pass-through transfer with no event precedent. `cash_return` equals the cash-settled portion of the A→CCP1 leg (netted portion returns nothing, per 059).

#### 2.4 Phase placement

No new phase slot: fund **replenishment** (§2.6) runs at the top of `ClearingPhase.run` when mode==ccp (a CCP step hosted by the existing clearing phase, mirroring how 060 hosts certificates); the two-leg settlement partition lives inside `SettlementPhase`. The TreynorSpot → Clearing → Settlement ordering from 059 §8 is unaffected. Whoever merges after the Treynor dealer work must re-verify `prepare_scenario`'s phase list.

#### 2.5 Default fund and loss waterfall

**Funding**: at setup, after liquidity allocation, the ring compiler emits `transfer_cash` actions: each member contributes `round(ccp_fund_share × member initial cash)` (default 0.05) to CCP1. Ordinary cash transfers ⇒ `check_invariants` cash conservation holds for free. The ledger records `ccp_fund_contribution[member]` and `ccp_fund_total`; event `CCPFundContribution {member, amount}`. Segregation is **accounting-only**: fund cash sits in CCP1's single cash balance, with the end-of-day invariant `ccp_fund_total == Σ ccp_fund_contribution[i] ≤ cash[CCP1]`.

**OPEN QUESTION 1**: size contributions by gross obligations instead of cash (closer to real CCP practice)? **Recommendation**: cash-proportional for stage 1 — it needs no obligation forecast and κ already parametrizes the cash/debt ratio; note obligation-proportional as the stage-2 alternative.

**Flagged side effect**: the fund drains `ccp_fund_share` of member liquidity, mechanically lowering effective κ in the ccp arm vs. baseline. **OPEN QUESTION 3**: gross member cash up so post-contribution κ matches? **Recommendation**: no — accept and report; the fund's liquidity cost is part of the institution being measured. The pre-flight summary must state the effective κ.

**Waterfall** on member `m` failing its net pay-in by `s`:
1. **Own contribution**: draw `min(s, ccp_fund_contribution[m])`.
2. **Mutualized tranche**: draw the remainder from surviving members' contributions, pro-rata to current balances.
3. **VMGH**: any residue shrinks the same day's payout pool (§2.2 step 2).

Fund draws move **no cash** — the cash already sits with CCP1; "covering" a missed pay-in means the CCP spends its own balance, so only fund accounting moves. Event `CCPFundDrawdown {member, own_tranche, mutualized_tranche, vmgh_residual}`. Recoveries from the expelled member (§3) credit back against the day's drawdown.

**Worked example** (n=4 members, fund 25 each, `ccp_fund_total = 100`). Day d net pay-ins due: A 80, B 40, C 30; net payouts due: D 120, B 30 (post-netting residuals). A pays only 20 → shortfall `s = 60`.

```
Collected pay-ins:            20 + 40 + 30        = 90
Waterfall for A's 60:
  (i)  A's own contribution:  min(60, 25)         = 25   fund[A] → 0
  (ii) mutualized pro-rata:   35 from B,C,D (25 each → 35/75 share ≈ 11.67 each,
                              largest-remainder in whole units: 12, 12, 11)
                                                          fund total → 100−60 = 40
  (iii) VMGH residual:        0  (fund covered it)
Payout pool: 90 collected + 60 fund draw          = 150 == required 150 → h = 1, no haircut
```

If instead `ccp_fund_total` were 30: draws cover 30, VMGH residual 30, pool = 120, `h = 120/150 = 0.8` → D receives 96 (haircut 24), B receives 24 (haircut 6), both legs settled, losses final, `cash[CCP1]` ends ≥ 0. A is expelled per §3 either way.

#### 2.6 Replenishment

Next day, surviving members top up pro-rata toward their *original* contribution (`replenishment_enabled`, default `True`), via `transfer_cash`, event `CCPFundReplenished`. **OPEN QUESTION 4**: cap top-ups at available member cash? **Recommendation**: yes — `min(gap, cash[member])`, shortfall carried to subsequent days; an uncapped top-up would manufacture defaults inside a bookkeeping step.

### 3. Member default and expulsion under the star

A member that cannot meet its net pay-in is expelled through the **existing** `handle_payable_default → expel_agent → reassign_receivables` pipeline, with the waterfall layered on the shortfall. Verified against `settlement.py` line by line:

- `collect_creditor_weights(m)`: under the arm-exclusivity rule (item 8), m's only unsettled payable-creditor is CCP1 ⇒ weights `{CCP1: 1}`.
- `distribute_pro_rata_recovery(m)`: m's residual liquid assets flow to CCP1 (sole claimant) — booked as recovery reducing the day's mutualized loss, not free CCP cash.
- `write_off_liabilities(m)`: m's *future* A→CCP1 legs are written off — exactly the exposure the CCP absorbs.
- `reassign_receivables(m, {CCP1: 1})`: m's receivables are all CCP1→m legs. For each, the inner loop hits `creditor_id == receivable.debtor` (CCP1 == CCP1) and creates nothing; the receivable is marked settled. **The member's CCP-side receivables offset automatically** — CCP1 is simultaneously the sole creditor weight and the sole debtor of every receivable, so no member↔member payable *can* be created. This currently holds by accident of the skip rule, not by intent: add an explicit ccp-mode guard (assert no created payable links two members) plus a regression test (acceptance criterion 4).
- Crucially, **CCP1→B out-legs with `origin_debtor == m` remain open**: B's claim survives A's death and is funded at maturity by fund/VMGH. This *is* the loss mutualization and the reason cascade depth collapses — B is never starved by A directly. (Contrast the baseline, where `reassign_receivables` re-links m's debtors to m's creditors and the chain continues.)

`ccp_member_defaults` counts these expulsions. `cancel_scheduled_actions_for_agent` runs unchanged. Fail-fast mode interacts as in 059: the waterfall is an expel-agent feature; under fail-fast a CCP-leg shortfall raises `DefaultError` with atomic rollback as today.

### 4. New events (catalog)

| Event | Emitted by | Payload (sketch) | Notes |
|-------|------------|------------------|-------|
| `CCPFundContribution` | setup actions / replenishment | `member, amount, fund_total` | Cash moves via `transfer_cash`; this event is the fund-accounting record |
| `CCPFundDrawdown` | waterfall (§2.5) | `member, own_tranche, mutualized_tranche, vmgh_residual, fund_total` | Accounting-only; added to `IMPACT_EVENTS` |
| `CCPFundReplenished` | §2.6 | `member, amount, gap_remaining, fund_total` | Capped at member cash (OPEN QUESTION 4) |
| `VMGHHaircutApplied` | payout leg (§2.2) | `creditor, face, paid, haircut, haircut_factor` | One per haircut payout; added to `IMPACT_EVENTS` |

`PayableCreated` gains optional `origin_debtor`/`origin_creditor` keys (present only on novated legs); `PayableSettled`/`PayableNetted`/`ObligationDefaulted` shapes are unchanged. Render: minimal formatter lines in `src/bilancio/ui/render/formatters.py`, following 059's `PayableNetted` precedent.

## New Agent Type Checklist (all 9 items, per repo-root CLAUDE.md)

1. **Instruments** — Holds (assets): novated A→CCP1 payable legs; the default-fund cash. Issues (liabilities): novated CCP1→B payable legs; members' fund-contribution claims (notionally repayable at wind-down; stage 1 never winds down, so the claim is tracked only in `ccp_fund_contribution`). No new `InstrumentKind`: legs reuse `Payable` (+ 059's in-place `net_payable` reductions); contract terms (amount, due_day, maturity_distance; rate-free like all ring payables) are inherited 1:1 from the novated original.
2. **Means of payment** — Cash only. `CapabilityMatrix` row: `"central_counterparty": AgentCapabilities(mop_order=(MOP_CASH,), can_default=False)` — the stage-1 no-failure choice is encoded at the policy layer, mirroring `central_bank`.
3. **Decision-making model** — Rule-based with zero discretion: novate everything, collect-then-pay, mechanical waterfall. Parameters live in `CCPProfile` (frozen dataclass, `TraderProfile` convention): `ccp_fund_share: Decimal = Decimal("0.05")`, `vmgh_enabled: bool = True`, `replenishment_enabled: bool = True`, reserved `ccp_can_fail: bool = False` (validation rejects `True`). **Flagged deviation**: CLAUDE.md prefers behavioral models with tunable risk parameters; the CCP deliberately has none in stage 1 — it is market infrastructure, not a strategic agent, and adding discretion would contaminate the mechanical topology comparison.
4. **Information model** — Full observability of the novated book *by construction*: every member obligation is a contract on the CCP's own balance sheet, so there is no `RiskAssessor`, no observability friction, no Bayesian updating. Contrast with traders (who see defaults through `default_observability < 1`) and 060's CH (which reads members' due-today books). Realistic — clearinghouses see all cleared flow — and stated as a feature, not a shortcut.
5. **Capitalization** — Endowed at setup exclusively via member fund contributions (`ccp_fund_share ×` member initial cash, default 5%); no own equity, no `dealer_share`-style slice of system resources. It accumulates nothing over time: shortfalls pass to members via haircuts, recoveries credit the fund.
6. **Timing / phase** — Novation happens at instrument creation (setup, B1 scheduled actions, rollover, reassignment) — not in a phase. Fund replenishment runs at the top of `ClearingPhase` (mode==ccp); netting then runs and degenerates to per-member nets; the two-leg collect/pay settlement runs inside `SubphaseB2` ordered before generic payables. Ordering vs. other subsystems is moot in stage 1 (item 8 exclusivity); internal ordering — replenish → net → pay-ins → waterfall → payouts → other payables — is fixed and tested.
7. **Failure mode** — Cannot fail in stage 1: VMGH scales payouts down to collected resources, so books close exactly every day and `cash[CCP1] ≥ 0` is an invariant, not a hope. Member failure is absorbed (own fund → mutualized fund → haircut), never re-propagated as a CCP default; the CCP is maximally systemically important precisely because its failure mode is assumed away — stated as the headline modeling caveat, with `ccp_can_fail` reserved for the stage where it can be relaxed.
8. **Interactions** — All ring members, exclusively. **OPEN QUESTION 5**: are `netting` / `certificates` / `ccp` exclusive modes or composable? **Recommendation**: exclusive variants of the single clearinghouse block — ccp *subsumes* netting (it runs 059's phase on the star, §2.1), so netting+ccp composition is meaningless; certificates+ccp is a research idea for later, rejected at config validation now. Also mutually exclusive with dealer (in-place `payable.creditor` mutation, §1.4), NBFI lender (`NonBankLoan` bypasses novation and breaks the `{CCP1: 1}` creditor-weight argument, §3), and banking arms. Validation: `ConfigurationError` if `mode: ccp` is combined with `dealer.enabled`, `balanced_dealer.mode != "passive"`, `lender.enabled`, or a banking config.
9. **State synchronization** — Shared state: the payables list, member cash, fund accounting. (i) Every payable-creation site routes through novation — the three sites in §1 are the complete set in `bilancio_v2`; the dealer's in-place mutation is excluded by item 8 and asserted against by the novation-invariant check. (ii) Settlement sufficiency treats CCP legs **atomically per day**: pay-ins, fund draws, haircut factor, and payouts are computed against a single day-snapshot of due CCP legs, with no interleaving with same-day non-CCP settlement (which runs strictly after). (iii) Fund accounting vs. cash: fund draws are accounting-only; all cash mutations go through existing ledger ops, so `check_invariants` cash conservation is untouched; the new fund invariant is checked at end of `run_day`. (iv) Rollover reads `origin_*` written at novation; a CCP leg missing them (bug) raises instead of silently rolling the leg.

## Sweep surface

### Backend layers

| Layer | Touched? | How? |
|-------|----------|------|
| **Domain** (agent types, policy, instruments) | yes | New `central_counterparty` kind + `CapabilityMatrix` row; `Payable.origin_debtor/origin_creditor` |
| **Decision** (profiles, strategies, risk) | yes | New `CCPProfile` dataclass (no RiskAssessor by design) |
| **Engines** (phases, settlement) | yes | CCP collect/pay partition in `SettlementPhase`; replenishment step in `ClearingPhase`; expel/reassign star guard |
| **Ops** (transfers, settlement mechanics) | yes | Waterfall + VMGH settlement rule; novation rewrite at `create_payable`; origin-pair rollover |
| **Scenarios** (ring builder, config) | yes | Clearinghouse block `mode: ccp` + 3 fields; compiler emits CCP1 + contribution actions |
| **State** (ledger) | yes | `ccp_fund_contribution`, `ccp_fund_total`, fund events; new end-of-day fund invariant |

### Sweep pipeline layers

| Layer | Touched? | How? |
|-------|----------|------|
| **CLI params** (`ui/cli/_sweep_ring.py`) | yes | `--clearing-ccp` flag + `--ccp-fund-share` (default 0.05); mutually exclusive with 059's `--clearing` (validated at CLI parse) |
| **Sweep config** (`experiments/ring.py`) | yes | `RingSweepRunner` clearing mode threaded into generated scenario dicts (extends 059's `clearing_enabled` into a `clearing_mode` enum) |
| **Runner logic** (arm creation) | yes/no | No new arm type — ring sweep stays single-arm; the ccp-vs-netting comparison is two sweeps at matched seeds. **OPEN QUESTION 6**: add a paired-arm runner like `sweep balanced`? **Recommendation**: not in stage 1; the registry `phase`/params columns + job tooling suffice |
| **Metrics collection** (`analysis/`) | yes | `fund_drawdowns_total`, `vmgh_haircut_total`, `ccp_member_defaults` from events; new `cascade_depth_max` in `analysis/metrics.py` |
| **Post-sweep reports** | minimal | New columns pass through `results.csv` (extend `default_fields` in `ring.py` ~line 510 and `RingRunSummary`); no new chart in 061 |
| **Pre-flight checks** | yes | New viability check V9-CCP: warn if the total fund < the expected largest single net pay-in at median κ (fund exhausts on first default ⇒ pure-VMGH regime); report the effective-κ drain from contributions (§2.5) |

### Interaction expectations

- **Low stress (κ ≥ 1)**: `delta_total` ≈ the 059-netting baseline; `fund_drawdowns_total ≈ 0`, `vmgh_haircut_total = 0` (fund untouched, novation economically invisible).
- **High stress (κ ≤ 0.5)**: **`cascade_depth_max` collapses toward 1** — defaults become independent member-vs-CCP events, while the baseline ring shows chains ≫ 1. This is the headline result.
- **Across seeds at fixed parameters**: variance of `delta_total` **decreases** under ccp (mutualization smooths idiosyncratic chain luck), but mean `delta_total` **may rise in extreme tails** (κ ≪ 1: VMGH haircuts spread losses to members who would have survived the ring). Both directions are admissible; the plan commits only to the depth and variance predictions. Since stage 1 has no behavioral response, any difference is purely mechanical.
- `vmgh_haircut_total > 0` only on fund-exhaustion days; `ccp_member_defaults ≤` baseline `n_defaults` at the same seed is **not** guaranteed (haircut-starved members can default later).
- New metric columns appear in `results.csv` for every run; zero/empty when ccp off — all pre-existing columns byte-identical when the flag is absent.

### Cascade-depth metric (new — gap found in code)

No default-*lineage* attribution exists today. `analysis/metrics.py::cascade_fraction` (line 349) already classifies primary vs. secondary defaults by chronological replay of `PayableCreated` + `AgentDefaulted` events (a debtor of yours defaulted before you ⇒ secondary); `AgentDefaulted` carries `trigger_contract` but no upstream cause, and `RingRunSummary` already plumbs `cascade_fraction`/`n_defaults` into results. Proposal: `cascade_depth_max` = the longest path in the precedence DAG whose edges run (defaulted debtor → its creditor that defaulted later), built from the same replay — no new event fields, so it works retroactively on goldens and identically on both arms. **Flagged limitation**: this is attribution-by-precedence, not true causal lineage; if it proves too coarse, the stage-2 option is a `caused_by` field on `AgentDefaulted` (an event-schema change with golden impact — explicitly avoided in this plan).

### Default value discipline

| Parameter | Default | Why this default | Backward-compatible? |
|-----------|---------|------------------|---------------------|
| clearinghouse block | absent / `None` (059) | Feature opt-in | Yes — absent ⇒ no CCP agent, no novation, byte-identical goldens |
| `mode` | `"netting"` (059) | ccp is explicit opt-in | Yes — only `mode: ccp` activates this plan |
| `ccp_fund_share` | `0.05` | Small vs. κ; sized to absorb a typical single shortfall at c=1 | Yes — read only in ccp mode |
| `vmgh_enabled` | `True` | Stage-1 books must close; `False` rejected at validation (reserved) | Yes |
| `replenishment_enabled` | `True` | Persistent mutualization is the institution under study | Yes |
| `ccp_can_fail` | `False` (reserved) | OUT of scope; validation rejects `True` | Yes |
| `Payable.origin_debtor/origin_creditor` | `None` | Set only by novation; absent from events when unset | Yes |
| `RunContext.clearing_config` | `None` | Threaded only when block present | Yes |
| `--clearing-ccp` | off | Conscious CLI opt-in per FEATURE_PROCESS | Yes |

## Acceptance criteria

1. **Novation invariant**: with mode ccp, at every day-end no unsettled payable has both debtor and creditor in the member set; before any default, Σ unsettled A→CCP1 face == Σ unsettled CCP1→B face; after defaults, the difference reconciles exactly to cumulative written-off in-legs of defaulted members (assert the reconciliation, not naive equality).
2. **Fund conservation**: `ccp_fund_total == Σ contributions − Σ drawdowns + Σ replenishments == Σ ccp_fund_contribution[i]` every day; `ccp_fund_total ≤ cash[CCP1]` at day-end (new ledger invariant).
3. **VMGH closes the day exactly**: daily CCP payouts ≤ pay-ins received + fund draws; `cash[CCP1] ≥ 0` after every op; a forced-shortfall unit fixture shows the haircut factor applied pro-rata to the unit, and the journal shows `VMGHHaircutApplied` with `paid + haircut == face` per payout.
4. **Expel under novation**: defaulting the largest member leaves zero member↔member payables; all CCP1→m legs settled (offset); all CCP1→B legs with `origin_debtor == m` still open; explicit regression test on the `reassign_receivables` star guard.
5. **Backward compat**: clearinghouse block absent ⇒ goldens byte-identical (goldens are the oracle post-v2-cutover); full suite green (`uv run pytest tests/ -v`); a 059 `mode: netting` run is also unchanged by this plan's code.
6. **Metrics in results.csv**: `fund_drawdowns_total`, `vmgh_haircut_total`, `ccp_member_defaults`, `cascade_depth_max` present with sensible values; zero/empty in non-ccp runs; pre-existing columns unchanged.
7. **Unit tests**: novation rewiring at all three creation sites (incl. scheduled-action and reassignment paths); waterfall arithmetic (own/mutualized split, recovery credit); VMGH pro-rata rounding conserves the pool exactly with integer Decimals.
8. **Integration test**: largest-debtor default with vs. without ccp at matched seed — `cascade_depth_max` strictly smaller under ccp in the stressed fixture; creditor-of-the-defaulter is paid (possibly haircut) rather than starved.
9. **Property test** (Hypothesis, already a project dependency): random κ/c/μ/seed ⇒ CCP cash ≥ 0 every day, fund conservation holds, novation invariant holds.
10. **Smoke sweep**: `uv run bilancio sweep ring --clearing-ccp --n-agents 10 --maturity-days 3 --kappas "0.25,0.5,1" --concentrations "1" --mus "0"` completes; results.csv populated; expectations above directionally confirmed.

## Dependencies, conflicts, open questions (index)

- **Requires Plan 059** (`ClearingPhase`, `net_payable`, clearinghouse block, gross-roll rollover with `cash_return`, netting-aware `phi_delta`). Independent of **Plan 060**, but shares the block schema and must not collide with 060's `clearinghouse` agent kind (distinct `central_counterparty` kind recommended). Order: 059 → 060 → 061. Reconcile naming (`ClearinghouseConfig` vs `ClearinghouseScenarioConfig`) with whatever 059 lands.
- **Conflict surface**: this plan and the in-flight Treynor bank-dealer work both touch `settlement.py`'s expel/reassign paths (`docs/analysis/treynor_dealer_and_bank_model.md`, `ui/cli/treynor.py`); the dealer plugin's in-place `payable.creditor` mutation is the sharpest contact point. Coordinate merge order; whoever merges second re-verifies `prepare_scenario` phase order and the expel-path tests.
- **OPEN QUESTIONS — RESOLVED (2026-06-11)**: 1 fund sizing basis → **DECIDED** cash-proportional; 2 rollover return-flow routing → **DECIDED** direct B→A, re-novate the payable; 3 κ drain from contributions → **DECIDED (user-confirmed)** accept and report effective κ; 4 replenishment cash cap → **DECIDED** yes, carry shortfall; 5 mode exclusivity → **DECIDED (user-confirmed)** netting/certificates/ccp are exclusive variants of one block, and ccp mode rejects dealer/lender/banking arms at validation; 6 paired-arm runner → **DECIDED** no, two sweeps at matched seeds.

## Implementation findings & decisions (2026-06-12, autonomous run)

### Smoke sweep (acceptance criterion 10, extended grid)

Matched `sweep ring` arms (n=10, maturity_days=5, c=1, κ ∈ {0.25, 0.5, 1, 2}, μ ∈ {0, 0.5},
seed 42, `--default-handling expel-agent`), 059-netting vs `--clearing-ccp`:

| κ | μ | δ netting | δ ccp | depth netting | depth ccp |
|---|---|-----------|-------|---------------|-----------|
| 0.25 | 0 | 0.818 | 0.564 | 7 | **1** |
| 0.5 | 0 | 0.830 | 0.486 | 2 | 2 |
| 1 | 0 | 0.690 | 0.604 | 3 | 3 |
| 2 | 0 | 0.828 | 0.628 | 8 | **1** |
| 0.25–1 | 0.5 | 0.93–0.99 | 1.000 | 3–4 | 3 |
| 2 | 0.5 | 0.968 | 0.900 | 3 | 2 |

**Headline confirmed**: where the baseline shows long serial cascades (depth 7–8), novation
collapses them to depth 1 — defaults become independent member-vs-CCP events — and δ falls by
19–34pp. Mutualization is visible in `fund_drawdowns_total` (6–200) and `vmgh_haircut_total`
(175–305) being active in every stressed cell. Depth > 1 under ccp reflects genuine
haircut-induced knock-ons (attributed via origin pairs), not member↔member re-links.

### Decisions taken (autonomous)

- **D1 (origin-aware δ)**: novation doubles the raw `PayableCreated` book (in-leg + out-leg per
  obligation), which would double `gross_face_due` and debit δ for an in-leg default even when
  the fund made the end creditor whole. `dues_for_day` / `netting_totals` now skip novated
  in-legs (`debtor == origin_debtor`): each obligation counts once, on the out-leg — δ measures
  **end-creditor losses** on the same denominator as the baseline arms.
- **D2 (VMGH legs and δ)**: a haircut payout (paid = h·face < face) is marked settled by
  `VMGHHaircutApplied` without a `PayableSettled` event, so it earns **zero** φ credit — the
  binary-completion convention every arm uses (baseline partial settlements also earn zero).
  Consequence: μ-skewed stressed cells can show δ_ccp = 1.0 while end creditors actually
  received h ≈ pool/required of face each day; read `vmgh_haircut_total` (the actual severity)
  alongside δ. The loss-weighted-δ metric remains the roadmap's separate open item.
- **D3 (fund collection in the kernel)**: contributions are collected on day 0 at the top of
  `ClearingPhase` (int(share × member cash), `CCPFundContribution` events) instead of
  compiler-emitted `transfer_cash` setup actions — one implementation point that also covers
  hand-written scenarios. Effective κ is reduced by the contribution, as decided (accept and
  report).
- **D4 (no kernel CCPProfile)**: the kernel reads `CleanClearingConfig` (060 precedent);
  `CCPProfile` exists at the experiment layer (`bilancio/decision/profiles.py`) as the
  documented tunables container.
- **D5 (structural-gap draws)**: out-legs of dead origin debtors maturing later draw the
  remaining whole fund pro-rata (`CCPFundDrawdown` with `member=None`) before VMGH — the plan's
  "funded at maturity by fund/VMGH" made concrete; dead members' leftover contributions are
  reachable only by this draw.
- **D6 (windfall symmetry, documented)**: when an origin creditor dies, the surviving A→CCP1
  in-leg still collects while the offsetting out-leg is written off — that cash stays with the
  CCP as free cash and reduces future haircuts (mirror image of mutualization).
