# Plan 059: Clearinghouse Stage 1 — Multilateral Netting Phase

**Status**: Plan (Phase 1 of FEATURE_PROCESS) — not yet implemented
**Branch**: `plan/059-clearinghouse-suite`
**Date**: 2026-06-11
**Program**: Option A of the three-part clearinghouse program. Plan 060 (loan certificates) and Plan 061 (CCP novation) build directly on the phase scaffolding, the `PayableNetted` event, and the `net_payable` ledger primitive introduced here.

---

## Goal

Introduce a multilateral netting phase ("clearinghouse stage 1") into the v2 kernel. Each day, before cash settlement, the clearinghouse collects all payables due that day, finds offsetting obligations (cycles in the debtor→creditor graph), and extinguishes them without any cash movement. Only the residual net obligations require means-of-payment in the subsequent `SettlementPhase`. This separates *gross* payment obligations from *net* liquidity needs — the core institutional function of a clearinghouse — and lets us measure how much of the default rate δ at low κ is a pure liquidity-synchronization artifact that netting removes, versus a genuine solvency shortfall that survives netting.

## Scope

**IN**
- New phase plugin `ClearingPhase` (name `"SubphaseB_Clearing"`) in the v2 kernel, registered immediately before `SettlementPhase` (`"SubphaseB2"`).
- Scenario gating via a new `clearinghouse:` YAML block → `ClearinghouseConfig` (pydantic) → `CleanClearingConfig` (frozen dataclass), default absent → phase not registered → zero behavior change.
- New ledger primitive `Ledger.net_payable(...)` and journal event `PayableNetted`; conservation assertions inside the phase.
- "Net-settle, gross-roll" rollover semantics for netted payables (open question, recommendation below).
- Metrics: `netting_efficiency` and `gross_face_due` per run; `phi_delta` made netting-aware so δ keeps its meaning "fraction of due face not settled by any means".
- Sweep surface: `--clearing/--no-clearing` flag on `sweep ring` (default off), wired through `RingSweepRunner` into the generated scenario YAML; `netting_efficiency` column in `aggregate/results.csv`.
- Unit + integration tests (algorithm, clearing+settlement+rollover, clearing+expel-agent), golden backward-compat check.

**OUT (deferred)**
- `clearing_window` parameter (net obligations due within the next *w* days, not just today). Deferred to a follow-up; today-only (`w = 0`) is the stage-1 semantics.
- Clearinghouse loan certificates (Plan 060) and CCP novation with a margin/default fund (Plan 061).
- Netting of non-payable contracts (`NonBankLoan`, `BankLoan`, `CBLoan`, `DeliveryObligation`).
- Clearinghouse as an agent with a balance sheet (stage 1 is a pure mechanism, no new agent type — the 9-item new-agent checklist does not apply).
- Legacy v1 engine support (`src/bilancio/engines/`): the feature is v2-only; v1 must hard-reject clearinghouse scenarios (see Design §7).
- HTML formatter polish beyond a minimal `PayableNetted` line in `src/bilancio/ui/render/formatters.py`.

## Location in codebase

| Change type | Path | Description |
|-------------|------|-------------|
| New file | `src/bilancio_v2/plugins/clearing.py` | `ClearingPhase` plugin + netting algorithm (`collect_due_payables`, `cancel_cycles`, `allocate_edge_reductions`) |
| Modified | `src/bilancio_v2/ledger.py` | `Payable.netted_amount` field; `Ledger.net_payable()`; `Ledger.netted_rollover_queue` |
| Modified | `src/bilancio_v2/engine.py` | Register `ClearingPhase` in `prepare_scenario` immediately before `SettlementPhase()` (current line ~182) |
| Modified | `src/bilancio_v2/subsystem_config.py` | New frozen `CleanClearingConfig` |
| Modified | `src/bilancio_v2/scenario_gates.py` | New `build_clearing_config(config) -> CleanClearingConfig \| None` |
| Modified | `src/bilancio_v2/plugins/settlement.py` | Gross-roll support: rollover face = `amount + netted_amount`; consume `ledger.netted_rollover_queue`; cash return-flow only for the cash-settled portion |
| Modified | `src/bilancio/config/models.py` | New `ClearinghouseConfig(BaseModel)`; `ScenarioConfig.clearinghouse: ClearinghouseConfig \| None = None` |
| Modified | `src/bilancio/ui/v2_run.py` (or `ui/run.py` legacy path) | Hard error if a clearinghouse scenario is routed to the legacy v1 engine |
| Modified | `src/bilancio/analysis/metrics.py` | `phi_delta` counts `PayableNetted` reductions toward φ numerator; new `netting_totals(events)` helper |
| Modified | `src/bilancio/analysis/report.py` | `compute_event_metrics`/`summarize_day_metrics` emit `gross_face_due`, `face_extinguished_by_netting`, `netting_efficiency`; `aggregate_runs` carries them into `results.csv` |
| Modified | `src/bilancio/experiments/ring.py` | `RingSweepRunner(clearing_enabled=False)`; inject `clearinghouse` block into generated scenario dict; `netting_efficiency` in `RingRunSummary`, registry fields (`_init_empty_registry`), and registry upserts |
| Modified | `src/bilancio/ui/cli/_sweep_ring.py` | `--clearing/--no-clearing` flag (default off) on `sweep ring`, forwarded to `RingSweepRunner` |
| Modified | `src/bilancio/ui/render/formatters.py` | Minimal formatter for `PayableNetted` / `ClearingExecuted` |
| New test | `tests/v2/test_clearing.py` | Unit tests: net-position/cycle algorithm, pro-rata allocation, conservation, gating |
| New test | `tests/v2/test_clearing_integration.py` | Clearing + settlement + rollover; clearing + expel-agent; fail-fast interaction |
| New test | `tests/analysis/test_netting_metrics.py` | `phi_delta` with `PayableNetted`; `netting_efficiency` aggregation |
| Config | `examples/scenarios/clearing_ring.yaml` | Small balanced ring with `clearinghouse: {enabled: true, mode: netting}` for single-run inspection |

## Design

### 1. Gating and configuration

Scenario YAML gains an optional block:

```yaml
clearinghouse:
  enabled: true
  mode: netting        # stage 1 supports only "netting"; 060/061 add "certificates", "ccp"
```

- `src/bilancio/config/models.py`: `ClearinghouseConfig(BaseModel)` with `enabled: bool = False`, `mode: Literal["netting"] = "netting"`. Added as `ScenarioConfig.clearinghouse` defaulting to `None` (same pattern as `lender` / `rating_agency`).
- `src/bilancio_v2/subsystem_config.py`: `CleanClearingConfig` frozen dataclass with `mode: str = "netting"` (room for 060/061 fields).
- `src/bilancio_v2/scenario_gates.py`: `build_clearing_config(config)` returns `None` when the block is absent or `enabled: false`, else a `CleanClearingConfig`. Unknown `mode` raises `ConfigurationError`.
- `src/bilancio_v2/engine.py` `prepare_scenario`: `clearing_config = build_clearing_config(config)`; if not `None`, `phases.append(ClearingPhase(config=clearing_config))` immediately before `phases.append(SettlementPhase())`. Because `run_day` logs `phase.name` only for registered phases, an absent block produces a byte-identical journal — backward compatibility is structural, not behavioral.

### 2. `ClearingPhase` algorithm (`name = "SubphaseB_Clearing"`)

Each day, on `run(ledger, ctx)`:

1. **Collect**: `due = [p for p in ledger.payables if not p.settled and p.due_day == ledger.day and p.debtor not in ledger.defaulted_agent_ids and p.creditor not in ledger.defaulted_agent_ids]`. If empty, return `False`.
2. **Aggregate to edges**: `w[(debtor, creditor)] = Σ amounts`; keep per-edge payable lists sorted by `payable.id` for deterministic allocation.
3. **Bilateral pass**: for each unordered pair, cancel `min(w[(a,b)], w[(b,a)])` (same idea as `net_interbank_flows` in `src/bilancio_v2/plugins/interbank.py`, the precedent this generalizes).
4. **Cycle cancellation**: while the positive-weight edge graph contains a directed cycle (DFS with neighbors visited in sorted-agent-id order for determinism), reduce every edge on the cycle by the cycle's minimum edge weight and drop zeroed edges. The residual graph is acyclic when the loop terminates. On a ring topology this is a single pass that cancels the full circulation `min_i(face_i)` around the ring per iteration.
5. **Allocate edge reductions to payables**: pro-rata by amount across the payables on each edge, with largest-remainder rounding in whole `Decimal` units, ties broken by sorted `payable.id`, so the per-edge total is conserved exactly (ring faces are integer Decimals).
6. **Apply**: call `ledger.net_payable(payable, reduction)` per affected payable; queue rollover entries for fully-netted payables (§5).
7. **Assert conservation** (§4), emit one `ClearingExecuted` summary event, return `True` iff any face was extinguished (netting is impactful activity for the stability stop rule, mirroring settlement).

**Why cycle cancellation, not bare net positions.** The prompt's "net-position computation à la Eisenberg–Noe set-off" cannot by itself drive in-place bilateral reductions: per-agent offsets `min(F_a, I_a)` are not generally achievable by reducing existing edges without novation (novation is Plan 061). Cycle cancellation is the maximal *in-place* netting that preserves every agent's net position exactly — each cancelled cycle reduces every member's outflow and inflow by the same amount — and the Eisenberg–Noe set-off property (`F'_a − I'_a = F_a − I_a` for all `a`) becomes the checkable invariant rather than the algorithm. **OPEN QUESTION**: on general (non-ring) multigraphs, greedy cycle cancellation yields *an* acyclic residual but not provably the *minimum-gross* residual (that is a min-flow problem). Recommendation: accept greedy-with-deterministic-order for 059 — it is exact on rings (the only topology `sweep ring` generates), reproducible, and `netting_efficiency` measures realized netting; revisit optimality together with Plan 055 topologies.

### 3. Mechanics: in-place reduction (a) vs. synthetic net payables (b)

**(a) Recommended — reduce `payable.amount` in place.** `net_payable` decrements `amount`, increments a new `Payable.netted_amount` field (default `ZERO`, preserving dataclass construction everywhere), marks `settled = True` when `amount` reaches zero, and emits:

```
PayableNetted {pid, contract_id, alias, debtor, creditor,
               original_amount, netted_amount (this reduction),
               remaining_amount}
```

**(b) Rejected — replace due-today payables with synthetic net payables.** Rejected because it breaks three load-bearing identities found in the code:
- *Rollover identity*: `settle_payable` builds `rollover_info` from the payable's own `(debtor, creditor, amount, maturity_distance)`; synthetic net payables would roll the *net*, shrinking the ring's debt stock every cycle and destroying the stationarity that Plan 024 rollover exists to preserve.
- *Event provenance*: `dues_for_day` (`src/bilancio/analysis/metrics.py:57`) computes the day's gross face `S_t` from `PayableCreated` events; emitting `PayableCreated` for synthetic instruments double-counts face and corrupts `S_t`, `Mbar_t`, φ and δ for every downstream consumer.
- *Default machinery*: `collect_creditor_weights`, `write_off_liabilities`, and `reassign_receivables` in `src/bilancio_v2/plugins/settlement.py` iterate `ledger.payables` by identity; swapping instruments mid-day changes expel-agent outcomes.

### 4. Conservation invariant

Inside the phase, after applying reductions, assert (raising `InvariantViolation` from `bilancio_v2.ledger`):
- For every agent: face extinguished as debtor == face extinguished as creditor (the set-off property; netting can never change a net position).
- Σ per-payable reductions == Σ per-edge cancellations == `face_extinguished` reported in `ClearingExecuted {day, gross_face_due, face_extinguished, residual_face, n_payables_affected, n_fully_netted}`.
- No cash, reserve, or deposit balance is touched (payables net against payables; double-entry neutral — both a debtor liability and a creditor asset shrink by the same amount). `ledger.check_invariants()` at end of `run_day` already guards cash/reserve conservation.

### 5. Rollover semantics — **OPEN QUESTION** (recommendation: "net-settle, gross-roll")

Recommended semantics: a payable fully extinguished by netting counts as settled and rolls over at its **full original face**, exactly like a cash-settled one; partially netted payables also roll at original face (`amount + netted_amount`). This preserves ring stationarity (the open debt stock is constant day over day) and makes clearing arms comparable to baseline arms.

Mechanics (the current code does *not* do this automatically — two gaps found in `settlement.py`):
- Fully-netted payables have `settled = True` before `SettlementPhase.run` and are skipped by its loop, so they would silently never roll. Fix: `ClearingPhase` appends `(debtor, creditor, original_face, maturity_distance)` to a new `ledger.netted_rollover_queue`; `SettlementPhase.run` drains it into `settled_for_rollover` (when `ctx.rollover_enabled`).
- For partially netted payables, `settle_payable`'s `rollover_info` currently captures `payable.amount` (the residual). Fix: use `payable.amount + payable.netted_amount`.
- Cash return-flow: `rollover_single_payable` moves `amount` of cash creditor→debtor (refinancing the settlement proceeds). For netted face no cash ever flowed, so the return-flow must cover only the cash-settled portion: extend `rollover_single_payable` with a `cash_return: Decimal` argument (default `= amount`, fully backward compatible); netted entries pass `cash_return = 0` (full netting) or the residual (partial). The new payable is still created at full face. `PayableRolledOver` gains `cash_transfer=False` (full netting) / existing `RolloverPartial` covers mixed cases. **OPEN QUESTION**: exact event payload shape for the zero-cash roll; recommendation: reuse `PayableRolledOver` with `cash_transfer: false` rather than minting a new event kind.

### 6. Default interaction and δ accounting — **OPEN QUESTION**

After netting, `SettlementPhase` runs unchanged: residual amounts settle by the capability-matrix MoP order; shortfalls trigger `expel-agent` (partial settlement, `AgentDefaulted`, pro-rata recovery, receivable reassignment — all on residual amounts, which is correct: the creditor's claim really is only the residual) or `fail-fast` (`DefaultError` with atomic rollback; the netting performed earlier in the day is *not* rolled back — netting is final, like the interbank precedent).

**δ accounting discovery (contradicts the prompt's assumption).** `delta_total` is *not* computed from defaulted/written-off face. It is the `S_t`-weighted mean of `delta_t = 1 − phi_t` (`summarize_day_metrics`, `src/bilancio/analysis/report.py:387`), where `phi_t` (`phi_delta`, `src/bilancio/analysis/metrics.py:144`) divides the day's `PayableSettled` amounts by the gross face from `PayableCreated` events. Under in-place netting, fully-netted payables emit no `PayableSettled` and partially-netted ones emit it with the *residual* amount — so with metrics untouched, netted face would be counted as **defaulted**, and δ would *rise* under clearing. Recommendation:
- Extend `phi_delta` to add day-`t` `PayableNetted` reductions (matched by `pid` against due-today payables) to the φ numerator. δ keeps its meaning — "fraction of due face not extinguished by any means" — and is unchanged for all runs without `PayableNetted` events (backward compatible by construction).
- Add explicit gross-vs-net decomposition per run: `gross_face_due` (Σ `S_t`), `face_extinguished_by_netting` (Σ `PayableNetted.netted_amount` on due day), and `netting_efficiency = face_extinguished_by_netting / gross_face_due` (0 when clearing is off or no face is due).

### 7. Legacy-engine guard

`src/bilancio/ui/run.py` auto mode falls back to the legacy v1 engine when `clean_core_unsupported_reason` is non-`None` (`src/bilancio_v2/compat.py:12`). The v2 kernel *supports* clearinghouse scenarios, so no fallback is triggered by the block itself — but a scenario combining `clearinghouse` with a genuinely unsupported feature would silently run on v1 *without* clearing. Guard: in the fallback path (and in the explicit `--engine` legacy path), raise `ConfigurationError("clearinghouse requires the v2 engine")` when `config.clearinghouse` is enabled.

### 8. Phase-ordering coordination

In this branch (`main`-based) the phase list in `prepare_scenario` is: `ScheduledActionsPhase → [RatingPhase] → [BankQuotesPhase] → [LendingPhase] → [BankLendingPhase] → [DealerPhase] → SettlementPhase → InterbankPhase`. `ClearingPhase` slots between `DealerPhase` and `SettlementPhase`. **Coordination note**: the in-flight Treynor dealer work (branch `rebuild/v2-kernel`) introduces `SubphaseB_TreynorSpot` at the same touchpoint; agreed ordering is **TreynorSpot → Clearing → Settlement** (traders adjust positions first, then the clearinghouse nets what is actually due, then cash settles). Whoever merges second must re-verify the ordering in `prepare_scenario` and the golden suite.

## Sweep surface

### Backend layers

| Layer | Touched? | How? |
|-------|----------|------|
| **Domain** (agent types, policy, instruments) | no | No new agent kind, no `InstrumentKind`, no policy/MoP change |
| **Decision** (profiles, strategies, risk assessment) | no | Netting is mechanical, not behavioral; no profile dataclass |
| **Engines** (phases, settlement, dealer integration) | yes | New `ClearingPhase` plugin before `SettlementPhase` in `bilancio_v2.engine.prepare_scenario`; settlement rollover handoff |
| **Ops** (transfers, settlement mechanics) | yes | New ledger primitive `net_payable` + `PayableNetted` event; `rollover_single_payable` gains `cash_return` |
| **Scenarios** (ring builder, config) | yes | `clearinghouse` block in `ScenarioConfig`; `RingSweepRunner` injects it into generated scenario dicts |
| **State** (agent state, system state) | yes | `Payable.netted_amount` field; `Ledger.netted_rollover_queue` (no per-agent state, no shared-state sync hazard: `ClearingPhase` writes only payable amounts + the queue, and the only same-day reader is `SettlementPhase`, which runs after it) |

### Sweep pipeline layers

| Layer | Touched? | How? |
|-------|----------|------|
| **CLI params** (`ui/cli/_sweep_ring.py`) | yes | `--clearing/--no-clearing` flag on `sweep ring`, default off |
| **Sweep config** (dataclasses in `experiments/`) | yes | `clearing_enabled: bool = False` kwarg on `RingSweepRunner`; optional `runner.clearing_enabled` in `RingSweepConfig` YAML |
| **Runner logic** (run construction) | yes | `_execute_run` / `_prepare_run` add `scenario["clearinghouse"] = {"enabled": True, "mode": "netting"}` when enabled; recorded in registry params |
| **Metrics collection** (`analysis/`) | yes | `phi_delta` netting-aware; `gross_face_due`, `face_extinguished_by_netting`, `netting_efficiency` in metrics summary, `RingRunSummary`, registry (`_init_empty_registry` fieldnames), and `aggregate_runs` → `results.csv` |
| **Post-sweep reports** (`_sweep_post.py`, dashboard) | minimal | `netting_efficiency` flows through `render_dashboard` as a results.csv column; no new chart in 059 |
| **Pre-flight checks** (viability) | yes | New advisory check V10 in the pre-flight summary: with clearing on and μ far from synchronized dues, warn that netting power will be small (no code gate, documentation + CLI warning only) |

### Interaction expectations

- **Synchronized dues, balanced ring**: at μ=0-style same-day dues with equal faces, netting cancels the entire ring circulation. Hypothesis: at κ=0.25 (severe cash shortage) δ drops from large values to ≈0 with `--clearing`, with **zero** settlement-cash transfers on due days and `netting_efficiency ≈ 1`.
- **Concentration heterogeneity (c < 1)**: unequal faces leave a residual after cycle cancellation (`netting_efficiency` ≈ Σ min-face circulations / gross); δ falls relative to no-clearing but stays positive at low κ; residual defaults concentrate on agents with the largest net-debit positions.
- **Maturity skew (μ spread across days)**: dues spread over many days shrink the same-day cycle mass; `netting_efficiency` declines monotonically as payment synchronization falls — the headline result: *netting power is a function of payment synchronization*.
- **High κ (≥ 2)**: δ ≈ 0 with or without clearing; netting_efficiency still ≈ its topological value (netting is liquidity-independent), demonstrating that the metric measures mechanism activity, not outcomes.
- **Clearing off (default)**: every existing metric byte-identical to pre-change; `netting_efficiency` column present and `0`/empty.

### Default value discipline

| Parameter | Default | Why this default | Backward-compatible? |
|-----------|---------|------------------|---------------------|
| `ScenarioConfig.clearinghouse` | `None` | Feature is opt-in; absent block → phase never registered | Yes — existing YAML unchanged, journal byte-identical |
| `ClearinghouseConfig.enabled` | `False` | Explicit opt-in even when block present | Yes |
| `ClearinghouseConfig.mode` | `"netting"` | Only stage-1 mode; 060/061 extend | Yes |
| `Payable.netted_amount` | `Decimal("0")` | New field with neutral default; all constructors/events unchanged | Yes |
| `rollover_single_payable(cash_return=...)` | `= amount` | Equals current hardcoded behavior | Yes |
| `RingSweepRunner.clearing_enabled` | `False` | Sweeps unchanged unless flagged | Yes |
| `sweep ring --clearing` | off | Conscious opt-in per FEATURE_PROCESS rule | Yes |
| `phi_delta` netting term | n/a (additive) | Adds `PayableNetted` amounts; zero such events ⇒ identical φ/δ | Yes |

## Acceptance criteria

1. **Gating**: a scenario without a `clearinghouse` block (or with `enabled: false`) produces a journal containing no `SubphaseB_Clearing`, `PayableNetted`, or `ClearingExecuted` events; `prepare_scenario(...).phases` has the same length and order as before the change.
2. **Backward-compat golden check**: the full v2 golden/parity suite (`uv run pytest tests/v2 -v`) passes unmodified, and `events.jsonl` for at least two existing example scenarios (e.g. `examples/scenarios/simple_bank.yaml` run via `bilancio run`) is byte-identical to main.
3. **Conservation**: in every clearing run, `ClearingExecuted.face_extinguished == Σ PayableNetted.netted_amount` for the day; per-agent debtor-side and creditor-side extinguished face are equal (unit-asserted); `ledger.check_invariants()` passes daily (no cash/reserve/deposit movement attributable to netting).
4. **Full-netting integration test**: balanced equal-face ring (n=5, all payables due day 1, κ≈0 so cash cannot settle anything): with clearing, day 1 has zero `CashTransferred` events in SubphaseB2, zero `ObligationDefaulted`, `delta_total == 0`, `netting_efficiency == 1`; the identical scenario without clearing produces defaults (`delta_total > 0`).
5. **Algorithm unit tests** (`tests/v2/test_clearing.py`): (a) equal-face ring fully nets; (b) Dirichlet-heterogeneous faces → residual edge graph is acyclic and every agent's net position is preserved exactly; (c) pro-rata largest-remainder allocation conserves per-edge totals with integer Decimals; (d) deterministic: same input → identical `PayableNetted` sequence across runs.
6. **Rollover integration test**: with `rollover_enabled`, fully- and partially-netted payables reappear at full original face at `max_open_due_day + maturity_distance`; total open payable face is constant across days; cash return-flow occurs only for cash-settled portions (cash conservation holds).
7. **Expel-agent integration test**: under `default_handling: expel-agent`, a residual net obligation that cannot be paid triggers the unchanged pipeline (`ObligationDefaulted` → `AgentDefaulted` → `ProRataRecovery` → `ReceivableReassigned`) with residual amounts; next-day netting excludes the defaulted agent. A fail-fast variant raises `DefaultError` with prior netting retained.
8. **Metrics**: `phi_delta` unit tests show φ counts netted face as settled and is unchanged when no `PayableNetted` events exist; `netting_efficiency` and `gross_face_due` appear in the per-run metrics summary.
9. **Sweep smoke**: `uv run bilancio sweep ring --clearing --n-agents 10 --maturity-days 3 --kappas "0.25,1" --concentrations "0.5,1" --mus "0"` completes; `aggregate/results.csv` contains a populated `netting_efficiency` column with values in [0, 1], higher at c=1 than c=0.5; the same sweep without `--clearing` yields a results.csv whose pre-existing columns match the pre-change output.
10. **Legacy guard**: a clearinghouse scenario forced onto the legacy v1 engine fails with a clear `ConfigurationError` instead of silently ignoring the block.

## Dependencies and coordination

- **Plan 060 (clearinghouse loan certificates)** reuses `ClearingPhase` as its host phase, `CleanClearingConfig.mode = "certificates"`, and `net_payable`/`PayableNetted` for certificate redemption offsets.
- **Plan 061 (CCP novation)** replaces step 3–5 of the algorithm with novation against a CCP agent (which *does* require the full new-agent checklist) and inherits the conservation assertions as its margin-accounting baseline.
- **Treynor dealer (in flight)**: shared touchpoint at `engine.prepare_scenario` phase list — ordering TreynorSpot → Clearing → Settlement; see Design §8.
- **Plan 055 (topology generalization, not yet committed on this branch)**: cycle cancellation already handles general multigraphs; the optimality open question in Design §2 should be resolved when non-ring topologies land.

## Open questions (summary)

| # | Question | Resolution (2026-06-11) |
|---|----------|------------------------|
| 1 | Greedy cycle cancellation may be sub-maximal on non-ring graphs | **DECIDED**: accept for 059 (exact on rings); revisit with Plan 055 |
| 2 | Rollover semantics for netted payables | **DECIDED**: "net-settle, gross-roll": settled status + full-original-face rollover, zero cash return-flow for netted portion |
| 3 | Event shape for the zero-cash rollover | **DECIDED**: reuse `PayableRolledOver` with `cash_transfer: false`; no new event kind |
| 4 | δ accounting under netting | **DECIDED (user-confirmed)**: `phi_delta` counts `PayableNetted` face as settled — netted = cleared, δ keeps its meaning; add `netting_efficiency` + `gross_face_due` for explicit gross-vs-net decomposition |
