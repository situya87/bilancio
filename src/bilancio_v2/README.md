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
- **Parity** (`parity.py`) — runs a scenario on both engines and diffs the
  full event stream and final balances.

## Verification

`tests/v2/` contains four layers:

1. **Example parity** — every supported example scenario matches the
   clean-core engine exactly (events + balances + stop behavior); scenarios
   both engines reject must fail with the identical error; unsupported
   subsystems are rejected explicitly, never mis-simulated.
2. **Golden oracle** — runs captured from the existing engine and pinned as
   JSON (`tests/v2/golden/`); catches changes that break both engines the
   same way. Regenerate with `uv run python -m tests.v2.golden_io`.
3. **Property-based parity** — Hypothesis generates random payment networks
   (including default cascades) and asserts identical runs on both engines.
4. **Unit tests** — ledger operations, invariant enforcement (including
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

**The v2 kernel's supported domain now equals the clean-core engine's by
construction**: scenario gating reuses the clean-core gate functions, so
both engines accept and reject exactly the same scenarios (multi-currency
jurisdiction *semantics* remain a legacy-v1-only feature on both).

The remaining migration steps are consumers, not engine features: point
the CLI/run pipeline and sweep runners at `bilancio_v2`, then delete the
two legacy engines. The parity suite (`tests/v2/`) is the gate for both
steps.
