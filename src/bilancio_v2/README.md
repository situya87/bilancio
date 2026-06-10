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
overrides, scheduled actions, stability-based termination.

Rejected explicitly (`UnsupportedScenarioError`) until rebuilt as plugins,
in suggested order:

1. **Banking** (bank lending, routed deposits, reserve targets, CB freeze)
2. **Non-bank lender** (decision protocol + information model)
3. **Rating agency** (estimates feed the information layer)
4. **Dealer / balanced dealer** (largest; port against the dealer metrics
   golden outputs)
5. **Rollover, jurisdictions/FX, action specs**

Each subsystem should land as a phase plugin plus capability-matrix rows,
validated the same way: capture goldens from the existing engine first,
then port until the parity suite passes, then extend the property-based
generator to cover the new behavior.
