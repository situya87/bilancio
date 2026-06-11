# Bilancio v2 kernel

A ground-up rebuild of the simulation kernel around three ideas the existing
codebase lacked: a **single audited ledger** (no subsystem shadow state, no
sync functions), a **data-driven capability policy** (no `isinstance`
special cases), and **phases as plugins** (the engine knows nothing about
specific subsystems).

It runs the existing YAML scenario schema unchanged and reproduces the
existing engine **event-for-event** — verified continuously by the parity
suite in `tests/v2/`.

## Architecture

```
ScenarioConfig (existing bilancio.config schema, unchanged)
        │ prepare_scenario()
        ▼
Engine (engine.py) ── day loop, phase order, stability rule, invariants
        │ runs, in order, each day:
        ├── ScheduledActionsPhase   (Subphase B1 — scenario actions)
        ├── SettlementPhase         (Subphase B2 — payables, deliveries, defaults)
        └── InterbankPhase          (Phase C — netting, CB loan servicing)
        │ all state access through:
        ▼
Ledger (ledger.py) ── the single source of truth
        │ every mutation appends to:
        ▼
EventJournal (events.py) ── typed, append-only, legacy-compatible dicts
```

- **Ledger** (`ledger.py`) — balances (cash with lots, reserves, deposits),
  open contracts (payables, CB loans, stocks, delivery obligations), and the
  event journal. Every mutation is a ledger operation that records its
  event; `check_invariants()` enforces conservation (cash minted − burned =
  cash in circulation, reserves = CB outstanding, lots sum to balances, no
  negatives) after setup and after every day.
- **Policy** (`policy.py`) — the capability matrix: per agent kind, the
  means-of-payment priority and whether it can default. Adding an agent
  kind is one registry row, not edits across the engine.
- **Plugins** (`plugins/`) — each phase implements
  `run(ledger, ctx) -> bool` (impactful?). Settlement, interbank clearing,
  and future subsystems (banking, dealer, lender) all use the same
  interface; the engine never special-cases one.
- **Actions** (`actions.py`) — the public scenario vocabulary, translated
  into ledger operations with legacy-exact semantics.
- **Parity** (`parity.py`) — snapshots a run (full event stream + final
  balances) and diffs it against the golden oracle.
- **Gates & subsystem configs** (`scenario_gates.py`, `subsystem_config.py`,
  `compat.py`) — scenario support checks and runtime-config builders,
  inherited from the (now deleted) clean-core engine.
- **Views & exports** (`views.py`, `exports.py`, `balance_invariants.py`) —
  balance/T-account rows, CSV/JSONL/HTML writers, and the exported-balance
  double-entry check used by the CLI.

## Verification

`tests/v2/` contains three layers:

1. **Golden oracle** — full event-stream + balance snapshots for every
   example scenario (`tests/v2/golden/`) and every deterministic subsystem
   case (`tests/v2/golden_cases/`), captured while the live parity suite
   proved v2 == clean-core, so they carry cross-engine authority.
   Regenerating snapshots the current v2 kernel (regression pinning only).
2. **Property-based self-consistency** — Hypothesis generates random
   payment networks (default cascades, lender/banking/dealer configs);
   every run must complete with the daily ledger conservation invariants
   and the exported-balance double-entry check holding.
3. **Unit tests** — ledger operations, invariant enforcement (including
   detection of un-audited mutations), checkpoint/rollback, settlement
   priorities and default handling.

## Current scope and migration path

Supported now: agents, cash/reserves/deposits, payables, delivery
obligations and stock, CB loans with refinancing, interbank netting,
fail-fast and expel-agent default handling (partial settlement, pro-rata
recovery, write-offs, receivable reassignment), means-of-payment policy
overrides, scheduled actions, stability-based termination, **payable
rollover** (Plan 024), the **rating agency** (deterministic sampling,
seeded-noise realistic profile, rating registry), the **non-bank lender**
(kappa-aware and signal-based pricing, information visibility models,
exposure/concentration/coverage/expected-loss screens, preventive lending,
loan servicing and defaults, loan write-offs in expulsion cascades), and
the **banking subsystem** (pricing-kernel quotes, bank lending with credit
creation, routed deposits, corridor-priced lender rates, CB refinance and
lending freeze, bank defaults with resolution and deposit write-offs, loan
winddown, final CB settlement, interbank auction records). Banking is
configured via ``CleanBankingConfig`` passed to ``prepare_scenario`` /
``run_scenario`` — same surface as the existing engine.

Also supported: the **dealer subsystem** in all three modes — marker
(daily pricing snapshots), balanced passive (equity baseline + PnL at
export), and balanced active (payables-as-tickets, per-bucket market
makers, trading rounds via the shared matching engine, ledger-audited
cash reconciliation) — plus the **action-specs** configurations and the
jurisdiction-carrying scenarios clean-core accepts.

## Status: cutover complete, clean-core deleted

The CLI and sweep pipeline run on the v2 kernel (the `--engine clean-core`
flag value is kept for compatibility and now drives v2). The clean-core
engine was deleted after its observable contract was pinned into the
golden oracle; its gate functions and config builders live on in this
package, so the supported domain is unchanged.

The **legacy v1 engine remains** — deliberately. It is not a duplicate of
v2: it uniquely implements the multi-arm balanced-dealer experiment modes
(`lender`, `nbfi_*`, `bank_dealer*`, …) used by `bilancio sweep balanced`
and the multi-currency jurisdiction semantics. Auto engine selection falls
back to it exactly as before. Deleting it requires porting those modes to
v2 plugins first — same recipe: pin goldens, port, verify.
