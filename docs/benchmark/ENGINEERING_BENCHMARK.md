# Engineering Benchmark: Bilancio

> Senior engineer assessment of code quality, architecture, and operational readiness.
> Generated 2026-02-11 against branch `feature/ring-reconnection` (commit `3a75b05b`).

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Source LOC | 37,245 (114 files) |
| Test LOC | 22,995 (65 files) |
| Test-to-source ratio | 0.62 |
| Packages | 17 top-level under `src/bilancio/` |
| Pydantic models | 32 (`config/models.py`) |
| Tests | 977 passed, 0 failed (9.25s) |
| Coverage | 67% (54 files at 100%, 6 files at 0%) |
| mypy errors | 1,187 in 83 / 138 files |
| Weighted score | **3.4 / 5.0** |

Bilancio is a well-structured financial simulation framework with clean domain/infrastructure separation, strong type discipline aspirations (11 mypy strict flags configured, 3.3:1 Decimal-to-float ratio), and thorough Pydantic validation. All 977 tests pass with 67% line coverage. The main weaknesses are unenforced mypy compliance (1,187 errors remain despite strict config), oversized functions in the settlement/dealer layer, heavy reliance on `getattr` and magic strings instead of typed enums, and incomplete observability infrastructure (logging in only 9 of 114 source files).

---

## Scoring Rubric

Each category is scored 1-5:

| Score | Meaning |
|-------|---------|
| 5 | Exemplary — ready for audit / open-source showcase |
| 4 | Strong — minor gaps, production-grade |
| 3 | Adequate — works, but improvement areas are clear |
| 2 | Below expectations — significant rework needed |
| 1 | Critical gaps — blocks production use |

---

## Category Scores

| # | Category | Weight | Score | Weighted |
|---|----------|--------|-------|----------|
| 1 | Architecture & Modularity | 15% | 4.0 | 0.60 |
| 2 | Type Safety & Data Integrity | 15% | 3.0 | 0.45 |
| 3 | Testing | 15% | 3.5 | 0.53 |
| 4 | Error Handling & Resilience | 10% | 3.5 | 0.35 |
| 5 | Code Complexity & Readability | 10% | 3.0 | 0.30 |
| 6 | Security | 5% | 4.0 | 0.20 |
| 7 | Performance & Scalability | 10% | 3.0 | 0.30 |
| 8 | Configuration & Validation | 5% | 4.0 | 0.20 |
| 9 | Observability & Operations | 10% | 3.0 | 0.30 |
| 10 | Documentation & Developer Experience | 5% | 3.5 | 0.18 |
| | **Weighted Total** | **100%** | | **3.41** |

Rounded overall: **3.4 / 5.0**

---

## Per-Category Assessment

### 1. Architecture & Modularity — 4.0 / 5

**Strengths**

- **Acyclic dependency graph.** Clean layering: `core/ → domain/ → ops/ → engines/ → runners/ → cloud/ui/`. Domain layer (17 files) imports only from `bilancio.core.ids`; no infrastructure leakage.
- **Protocol-based executor abstraction** (`runners/protocols.py:12-59`). `SimulationExecutor` and `JobExecutor` are `@runtime_checkable` protocols with concrete `LocalExecutor` and `CloudExecutor` implementations — textbook strategy pattern.
- **Policy engine** (`domain/policy.py:22-63`). `isinstance`-based permission checks for instrument issuance/holding. Default policy returns proper MOP settlement ordering per agent kind.
- **Atomic transactions** (`core/atomic_tx.py:5-13`). 13-line context manager providing rollback semantics via `copy.deepcopy` snapshot.

**Weaknesses**

- **God files in engines layer.** `dealer_integration.py` (1,700 LOC, 16 functions → avg 106 LOC/function) acts as adapter, bridge, and manager simultaneously. `dealer/simulation.py` (1,529 LOC) and `dealer/bank_dealer_simulation.py` (1,480 LOC) are similarly oversized.
- **Event logging mixed into settlement logic.** `settlement.py` calls `system.log("PayableSettled", ...)` (line ~704) and `system.log("PayableRolledOver", ...)` (line ~821), coupling presentation to domain.
- **Banking package (7 files) has complex internal cross-imports** between `day_runner`, `pricing_kernel`, `state`, `ticket_processor`.

---

### 2. Type Safety & Data Integrity — 3.0 / 5

**Strengths**

- **mypy strict configuration** (`pyproject.toml:99-111`). 11 flags enabled including `disallow_untyped_defs`, `disallow_any_generics`, `strict_equality`, `warn_unreachable`.
- **Decimal-first arithmetic.** 477 `Decimal(` vs 143 `float(` usages (3.3:1 ratio). Core financial paths (settlement, payments, pricing) use Decimal throughout.
- **8 invariant assertion functions** (`core/invariants.py:1-69`): `assert_cb_cash_matches_outstanding`, `assert_no_negative_balances`, `assert_cb_reserves_match`, `assert_double_entry_numeric`, `assert_no_duplicate_refs`, `assert_all_stock_ids_owned`, `assert_no_negative_stocks`, `assert_no_duplicate_stock_refs`.
- **`AgentKind` enum** defined at `domain/agent.py:9-23` with 11 variants.
- **Domain and core layers are nearly clean.** `domain/` has only 6 mypy errors and `core/` has 10 — confirming type discipline where it matters most.

**Weaknesses**

- **mypy is not passing.** Despite 11 strict flags in `pyproject.toml`, running `mypy src/bilancio/` produces **1,187 errors in 83 of 138 files**. The strict config is aspirational, not enforced. Error breakdown by type:

  | Error Type | Count | % |
  |------------|-------|---|
  | `[union-attr]` | 585 | 49% |
  | `[no-untyped-def]` | 195 | 16% |
  | `[type-arg]` | 109 | 9% |
  | `[arg-type]` | 84 | 7% |
  | `[assignment]` | 36 | 3% |
  | Other (10 types) | 178 | 15% |

  Errors concentrate in `config/` (589 — largely Pydantic v2 typing), `analysis/` (136), `ui/` (93), and `engines/` (89). The domain and core layers are nearly clean (16 combined).

- **No `InstrumentKind` enum.** `Instrument.kind` is `str` (`domain/instruments/base.py:11`). All 9+ contract kinds (`"payable"`, `"cash"`, `"reserve_deposit"`, `"bank_deposit"`, `"delivery_obligation"`, `"cb_loan"`, etc.) are magic strings compared via `c.kind == "payable"` throughout `settlement.py`.
- **18 `getattr()` calls in `settlement.py`** (lines 45, 49, 50, 56, 63, 210, 214, 359, 480, 481, 484, 498, 511, 605, 619, 765, 775, 777). Most use defaults like `getattr(contract, "amount", 0)` — indicates optional attributes not captured in the type system.
- **Bare `assert` in invariants.** Lines 3 and 12 of `core/invariants.py` use `assert` (disabled by `-O` flag), while the remaining 9 checks correctly use `raise AssertionError(...)`.
- **`-> Any` returns** in `config/apply.py` (`create_agent`), `dealer/events.py` (`_serialize_value`), `analysis/metrics_computer.py` (`_to_decimal`), `experiments/ring.py` (`_to_yaml_ready`).

---

### 3. Testing — 3.5 / 5

**Execution Results** (2026-02-11):

| Metric | Value |
|--------|-------|
| Tests collected | 977 |
| Passed | 977 (100%) |
| Failed | 0 |
| Runtime | 9.25s |
| Line coverage | 67% |
| Files at 100% | 54 |
| Files at 0% | 6 |

**Zero-coverage files:** `banking/sandbox.py`, `cloud/modal_app.py`, `cloud/sweep_trigger.py`, `dealer/bridge.py`, `jobs/supabase_store.py`, `ui/settings.py` — infrastructure/cloud code that is harder to unit-test but represents real production paths.

**Strengths**

- **977 tests, 100% pass rate.** 0.62 test-to-source ratio (22,995 / 37,245 LOC). 65 test files across 16 directories mirroring source structure.
- **Category breadth.** Unit (6), integration (5), banking (8), dealer (11), analysis (8), engines (6), config (5), cloud (2), storage (4), UI (5+), runners (3), specification (1), scenarios (1), ops (1), experiments (1).
- **pytest configuration** (`pyproject.toml:82-97`). Strict markers, coverage reporting (`--cov=bilancio`, `--cov-report=term-missing`, `--cov-report=html`), slow test marker.
- **Smoke test** (`tests/test_smoke.py`) for fast CI gate checks.

**Weaknesses**

- **67% coverage with no `fail_under` threshold.** Coverage is measured but not enforced — regressions can slip through without CI gate. Key gaps: `balanced_comparison.py` at 33%, `cloud_executor.py` at 43%, `supabase_store.py` at 0%.
- **No property-based / fuzz testing.** Financial invariants (double-entry balance, non-negative amounts) are prime candidates for Hypothesis.
- **Cloud/experiment tests are thin.** `tests/cloud/` has 2 files; `tests/experiments/` has 1 file. These are the highest-risk production paths.
- **No mutation testing** (e.g., mutmut) to validate assertion quality.

---

### 4. Error Handling & Resilience — 3.5 / 5

**Strengths**

- **Custom exception hierarchy** (`core/errors.py:1-27`). Five exceptions: `BilancioError` (base), `ValidationError`, `CalculationError`, `ConfigurationError`, `DefaultError` — all inheriting from the base.
- **Atomic rollback** (`core/atomic_tx.py:5-13`). On any exception inside `atomic(system)`, state is restored from deepcopy snapshot before re-raising.
- **CLI error formatting** (`ui/cli/run.py:99-123`). Rich panels with category-specific messages for `FileNotFoundError`, `ValueError`, generic `Exception`; `--debug` flag re-raises for full tracebacks.

**Weaknesses**

- **Module-level mutable state** (`settlement.py:16`). `_settled_payables_for_rollover: List[Tuple[str, str, int, int, int]] = []` is global state, creating re-entrancy risks and making unit testing harder.
- **`DefaultError` used for control flow.** `settlement.py` raises `DefaultError` to trigger default handling, which is caught and processed as a business logic branch — exception-as-control-flow anti-pattern.
- **No retry/circuit-breaker for cloud calls.** `CloudExecutor` wraps Modal RPC but doesn't handle transient failures gracefully.

---

### 5. Code Complexity & Readability — 3.0 / 5

**Strengths**

- **Clear naming conventions.** Functions like `settle_due`, `mint_reserves`, `client_payment`, `due_payables` are self-documenting.
- **Dataclass-heavy design.** Domain objects (`Agent`, `Instrument`, `DayEvent`, `Ticket`) are `@dataclass` with typed fields — minimal boilerplate.
- **Docstrings on key functions.** `settle_due`, `atomic`, protocol methods, and Pydantic models have descriptive docstrings.

**Weaknesses**

- **`settle_due()` is 147 lines** (`settlement.py:586-732`). A single function handling payable iteration, amount resolution, settlement attempts, default handling, rollover tracking, and event logging.
- **Top 5 files average 1,482 LOC each.** `dealer_integration.py` (1,700), `dealer/simulation.py` (1,529), `dealer/bank_dealer_simulation.py` (1,480), `dealer/metrics.py` (1,402), `analysis/visualization/balances.py` (1,219).
- **`dealer_integration.py` has 16 functions averaging 106 LOC each.** Functions like `_execute_sell_trade` and `run_dealer_trading_phase` likely exceed 150+ lines.
- **Deep nesting** in settlement loops. `settle_due` has 4+ levels of indentation for the main settlement path.

---

### 6. Security — 4.0 / 5

**Strengths**

- **YAML safe loading.** All 3 YAML load sites use `yaml.safe_load()`: `experiments/ring.py:207`, `config/loaders.py:154`, `ui/wizard.py:29`. Zero uses of unsafe `yaml.load()`.
- **Environment-based secrets.** Supabase credentials accessed via `os.environ.get("BILANCIO_SUPABASE_URL")` (`cloud/modal_app.py:158-159`, `storage/supabase_client.py:45-46`). No hardcoded secrets.
- **No web attack surface.** No HTTP endpoints (no Flask/FastAPI/Django). CLI-only with Modal serverless for compute.

**Weaknesses**

- **Placeholder author info** (`pyproject.toml:12-14`). `"Your Name"` / `"your.email@example.com"` and placeholder repository URLs. Minor but signals incomplete release preparation.
- **No `.env` file validation.** If `BILANCIO_SUPABASE_URL` is malformed, failures surface late at runtime rather than at startup validation.

---

### 7. Performance & Scalability — 3.0 / 5

**Strengths**

- **Generator-based iteration** for due payables (`settlement.py:53-57`). `yield` avoids materializing full contract list in memory.
- **Modal cloud parallelism.** Sweep runner uses `.map()` for concurrent simulation execution (~5-6 containers).
- **Indexed scheduled actions.** `system.state.scheduled_actions_by_day` (`engines/system.py:35`) is a `dict[int, list[dict]]` for O(1) day lookup.

**Weaknesses**

- **No due-day index for contracts.** `due_payables()` (`settlement.py:53-57`) performs O(n) full scan of all contracts every settlement cycle:
  ```python
  for c in system.state.contracts.values():
      if c.kind == "payable" and getattr(c, "due_day", None) == day:
          yield c
  ```
  With hundreds of agents and thousands of payables, this becomes a bottleneck. The project demonstrates awareness of indexed structures (`scheduled_actions_by_day`) but doesn't apply the pattern to contracts.
- **`copy.deepcopy` in atomic transactions** (`core/atomic_tx.py:8`). Every atomic settlement creates a full deep copy of `system.state`. For large simulations this is O(n) memory and CPU per settlement attempt.
- **No profiling infrastructure.** No `cProfile` integration, no benchmark test suite, no flame graph tooling.

---

### 8. Configuration & Validation — 4.0 / 5

**Strengths**

- **32 Pydantic BaseModel classes** (`config/models.py`, 824 LOC). Comprehensive schema covering agents, operations, dealer config, ring explorer params.
- **Field validators.** `@field_validator` on 12+ fields for positivity checks; `@model_validator(mode="after")` for cross-field constraints (e.g., `DealerBucketConfig` validates `tau_min ≤ tau_max` at lines 383-387).
- **Specification validators** (`specification/validators.py:15-46`). `ValidationError` dataclass with categories (`missing_relation`, `incomplete_spec`, `inconsistency`, `missing_field`). Aggregated `ValidationResult` with error list.
- **CLI validate command** with Rich-formatted error panels and green `[OK]` success messages.

**Weaknesses**

- **No JSON Schema export.** Pydantic models aren't exported as `.json` schema for editor autocompletion of YAML scenario files.
- **Inconsistent defaults.** Some fields use `= None` with runtime fallback logic rather than explicit Pydantic defaults.

---

### 9. Observability & Operations — 3.0 / 5

**Strengths**

- **Structured event system.** Dual-layer: core events via `system.log(kind, **payload)` (`engines/system.py:82`) stored in `system.state.events`, and dealer events via `EventLog` (`dealer/events.py:21`) with indexed lookups (`defaults_by_day`, `trades_by_day`, `settlements_by_day`).
- **Job lifecycle tracking.** `JobEvent` dataclass (`jobs/models.py:66`) with `JobStatus` enum (PENDING, RUNNING, COMPLETED, FAILED). Event log records all state transitions.
- **Metrics infrastructure.** 30+ modules for metrics computation, strategy analysis, dealer usage summaries, and comparison reports.

**Weaknesses**

- **Standard `logging` used in only 9 of 114 source files.** Modules like `balanced_comparison.py`, `html_export.py`, `strategy_outcomes.py` use `logging.getLogger(__name__)`, but the core simulation (`settlement.py`, `system.py`, `dealer/simulation.py`) has no Python logging — only structured events. This means no log-level filtering, no log aggregation compatibility, and no runtime verbosity control.
- **No health endpoints or readiness probes.** Cloud execution via Modal has no liveness/readiness checks beyond job status polling.
- **Event schema is untyped.** Events are `dict[str, Any]` with string `kind` keys — no enum, no validation, no schema evolution strategy.
- **`logging.basicConfig()` called inside library modules** (`analysis/strategy_outcomes.py:259`, `analysis/dealer_usage_summary.py:385`) — should be caller's responsibility.

---

### 10. Documentation & Developer Experience — 3.5 / 5

**Strengths**

- **Comprehensive CLAUDE.md** (>400 lines). Covers testing, notebooks, UI work, Modal deployment, Supabase storage, simulation parameters, job management, and example commands. Acts as effective onboarding guide.
- **ReadTheDocs setup** with MkDocs (6 pages: index, installation, quickstart, concepts, CLI, contributing, changelog).
- **CLI with Rich formatting.** `bilancio run`, `bilancio sweep`, `bilancio validate`, `bilancio volume`, `bilancio jobs` — well-organized command groups.
- **Example scenario files** (`examples/scenarios/`) for quick-start testing.

**Weaknesses**

- **Sparse inline docstrings in large files.** `dealer_integration.py` (1,700 LOC) and `settlement.py` (832 LOC) have minimal function-level documentation.
- **No architecture diagram** (e.g., Mermaid) showing module dependencies or simulation phase flow.
- **Placeholder metadata** (`pyproject.toml:12-14`) undermines packaging credibility.

---

## Top Priority Improvements

Ranked by impact-to-effort ratio:

| # | Improvement | Category | Effort | Impact |
|---|------------|----------|--------|--------|
| 1 | **Fix mypy errors and add to CI** — start with `config/` (589 errors, mostly Pydantic v2 `union-attr`), then `engines/` and `analysis/` | Type Safety | L | Critical — 1,187 errors make the strict config meaningless without enforcement |
| 2 | **Create `InstrumentKind` enum** and replace all `c.kind == "payable"` magic strings | Type Safety | S | High — eliminates an entire class of typo bugs, enables IDE autocomplete |
| 3 | **Add `fail_under = 70` to pytest coverage config** (current: 67%, target: incremental) | Testing | S | Medium — prevents coverage regression without any code changes |
| 4 | **Convert bare `assert` to `raise` in `invariants.py:3,12`** | Type Safety | S | Medium — ensures invariants can't be silently disabled with `-O` |
| 5 | **Add due-day index to contract storage** (`dict[int, list[InstrId]]`) | Performance | M | High — converts O(n) per-day settlement scan to O(1) lookup |
| 6 | **Split `settle_due()` into sub-functions** (resolve_amount, attempt_settlement, handle_default, track_rollover) | Complexity | M | High — improves testability and readability of most critical function |
| 7 | **Eliminate module-level `_settled_payables_for_rollover`** — pass as parameter or return value | Error Handling | S | Medium — removes global state, enables safe concurrent execution |
| 8 | **Add Python `logging` to core simulation modules** (settlement, system, dealer/simulation) | Observability | M | High — enables runtime verbosity control, log aggregation, debugging |
| 9 | **Extract `dealer_integration.py` into 3+ focused modules** (wiring, trade execution, sync) | Architecture | L | High — breaks up 1,700-LOC god module |
| 10 | **Replace `getattr()` calls with typed Optional fields** on Instrument subclasses | Type Safety | M | Medium — makes the type system reflect actual data shape |
| 11 | **Add property-based tests with Hypothesis** for double-entry invariants | Testing | M | Medium — catches edge cases that example-based tests miss |

**Effort key:** S = < 1 hour, M = 1-4 hours, L = 4+ hours

---

## Verification Commands

Independently verify findings cited in this benchmark:

```bash
# 1. LOC counts
find src/bilancio -name "*.py" -not -path "*__pycache__*" | xargs wc -l | tail -1
find tests -name "*.py" -not -path "*__pycache__*" | xargs wc -l | tail -1

# 2. mypy strict flags (should show 11 flags at lines 99-111)
sed -n '99,111p' pyproject.toml

# 3. Placeholder author info
sed -n '12,14p' pyproject.toml

# 4. Decimal vs float usage ratio
echo "Decimal:" && grep -r "Decimal(" src/bilancio --include="*.py" | wc -l
echo "float:" && grep -r "float(" src/bilancio --include="*.py" | wc -l

# 5. getattr calls in settlement.py
grep -n "getattr(" src/bilancio/engines/settlement.py | wc -l

# 6. Magic string comparisons in settlement.py
grep -n 'c\.kind ==' src/bilancio/engines/settlement.py

# 7. Bare assert in invariants.py (should find 2)
grep -n "^    assert " src/bilancio/core/invariants.py

# 8. YAML safe_load usage (should find 3, no unsafe yaml.load)
grep -rn "yaml.safe_load\|yaml.load" src/bilancio --include="*.py"

# 9. logging module adoption
grep -rl "import logging" src/bilancio --include="*.py" | wc -l

# 10. Top 10 largest source files
find src/bilancio -name "*.py" -not -path "*__pycache__*" | xargs wc -l | sort -rn | head -11

# 11. Test file count
find tests -name "test_*.py" -not -path "*__pycache__*" | wc -l

# 12. settle_due function length
awk '/^def settle_due/,/^def [a-z]/' src/bilancio/engines/settlement.py | wc -l

# 13. Run full test suite
uv run pytest tests/ -v --tb=short

# 14. Run mypy
uv run mypy src/bilancio/
```

---

## Methodology

This benchmark was generated by static analysis and verified by tool execution against commit `3a75b05b`. Scores reflect:

- **Quantitative metrics**: LOC, function lengths, type annotation coverage, test ratios
- **Tool execution**: Full test suite (`uv run pytest tests/ -v`), mypy strict mode (`uv run mypy src/bilancio/`), coverage report
- **Pattern analysis**: `grep`/`glob` searches for anti-patterns (magic strings, bare asserts, `getattr`, global state)
- **Architectural review**: Import graph analysis, module coupling, layer violations
- **Comparison to standards**: Measured against practices expected in production-grade Python financial software (PEP 8, PEP 484, OWASP, 12-factor app)

All 15 quantitative claims were independently verified. One minor correction applied (logging in 9 files, not 8). The Type Safety score was revised downward (3.5 → 3.0) after mypy execution revealed 1,187 errors despite strict configuration.

Scores are intentionally conservative — a 5.0 means "nothing to improve" which is rare in any active codebase.
