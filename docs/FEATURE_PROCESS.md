# Feature Development Process

Standard procedure for adding any new feature to bilancio. Claude MUST follow this process whenever the user says "let's create a new feature" or equivalent.

---

## Phase 1: Plan & Specify (branch: `plan/<NNN>-<short-name>`)

### 1.1 Create branch and plan file

```bash
# Must start from main with clean git status
git checkout main
git pull
git checkout -b plan/<NNN>-<short-name>
```

Create `docs/plans/<NNN>_<feature_name>.md` with:

- **Goal** — One paragraph: what problem does this solve and why now?
- **Scope** — What is IN and what is OUT of this feature.
- **Location in codebase** — Which modules/packages will be touched or created:

| Change type | Path | Description |
|-------------|------|-------------|
| New file | `src/bilancio/<module>/...` | ... |
| Modified | `src/bilancio/<module>/...` | ... |
| New test | `tests/<category>/...` | ... |
| Config | `examples/scenarios/...` | ... |

- **Design** — How the feature works. Data structures, algorithms, integration points.
- **If new agent type** — Fill out the full 9-item checklist from CLAUDE.md.
- **Sweep surface** — How the feature interacts with the sweep pipeline (see Section 1.4 below).
- **Acceptance criteria** — Concrete, testable conditions that define "done."

### 1.2 Claude reads the code

Before critiquing the plan, Claude MUST read every file listed in the Location table above. No exceptions — you cannot critique a plan for code you haven't seen. This catches design conflicts, naming clashes, and misunderstandings about existing behavior before any code is written.

### 1.3 Claude review loop

Claude will:

1. **Critique** the plan — identify gaps, risks, unclear areas, potential conflicts with existing code (informed by the code read in 1.2).
2. **Ask for specification** — request clarification on anything ambiguous. Examples:
   - "What should happen when X fails?"
   - "Does this interact with the dealer phase? If so, what's the ordering?"
   - "You mention a new config parameter — what's the default and valid range?"
3. **Test understanding** — restate the feature in its own words and ask the user to confirm.
4. **Suggest alternatives** — if there's a simpler or more robust approach, propose it.
5. **Check backward compatibility** — for every new parameter, confirm: What is the default? Does the default preserve existing behavior exactly (feature-off by default)? Would an existing config/scenario that doesn't mention this parameter produce identical results to before the change?

The plan is **not finalized** until both the user and Claude agree on:
- The design
- The location of all changes
- The sweep surface (how the feature touches the simulation and sweep pipeline)
- The acceptance criteria
- That defaults preserve backward compatibility

### 1.4 Sweep surface specification

Every feature touches the system at some layer. The plan must explicitly map where the feature sits relative to both the **backend** (simulation engine) and the **sweep pipeline** (CLI → config → runner → metrics → reports). Fill out this table:

#### Backend layers

| Layer | Touched? | How? |
|-------|----------|------|
| **Domain** (agent types, policy, instruments) | yes/no | e.g., "new InstrumentKind added" |
| **Decision** (profiles, strategies, risk assessment) | yes/no | e.g., "new field on TraderProfile" |
| **Engines** (phases, settlement, dealer integration) | yes/no | e.g., "new sub-phase after lending" |
| **Ops** (transfers, settlement mechanics) | yes/no | e.g., "new settlement rule" |
| **Scenarios** (ring builder, config) | yes/no | e.g., "new param in ScenarioConfig" |
| **State** (agent state, system state) | yes/no | e.g., "new field on Agent" |

#### Sweep pipeline layers

| Layer | Touched? | How? |
|-------|----------|------|
| **CLI params** (`ui/cli/_sweep_*.py`) | yes/no | e.g., "new `--foo` flag on `sweep balanced`" |
| **Sweep config** (dataclasses in `experiments/`) | yes/no | e.g., "new field on BalancedComparisonConfig" |
| **Runner logic** (run construction, arm creation) | yes/no | e.g., "new arm type" |
| **Metrics collection** (`analysis/`, `stats/`) | yes/no | e.g., "new metric `bar_total` in comparison.csv" |
| **Post-sweep reports** (`_sweep_post.py`, analysis) | yes/no | e.g., "new chart in dashboard" |
| **Pre-flight checks** (viability, parameter validation) | yes/no | e.g., "new viability check V9" |

#### Interaction expectations

For the feature, state what you **expect** the sweep to show:
- "With the new feature enabled, delta should decrease by ~X% at kappa=0.5"
- "The new metric `foo_total` should appear in comparison.csv"
- "No change expected to existing metrics — this is a refactoring"
- "A new arm `bar` should appear alongside passive/active"

This section ensures nothing is accidentally left unwired (e.g., a new config param that never reaches the runner, or a new metric that never gets written to comparison.csv).

#### Default value discipline

For every new parameter introduced by the feature, state:

| Parameter | Default | Why this default | Backward-compatible? |
|-----------|---------|------------------|---------------------|
| `foo_enabled` | `False` | Feature is opt-in | Yes — existing configs unchanged |
| `foo_threshold` | `0.5` | Matches current implicit behavior | Yes — equivalent to hardcoded value |

**Rule**: The default MUST preserve backward compatibility. An existing scenario/config that doesn't mention the new parameter must produce identical simulation results to before the code change. If this is impossible, the plan must explain why and call it out as a breaking change.

### 1.7 Commit the plan

```bash
git add docs/plans/<NNN>_<feature_name>.md
git commit -m "plan: <NNN> <feature description>"
```

**Phase 1 gate**: The plan commit is the gate. Do NOT start writing implementation code until this commit exists. If the plan needs revision later, update the doc and note the change — but implementation starts only after the initial plan is committed.

---

## Phase 2: Implement

### 2.1 Write the code

- Use implementation subagents in parallel where possible.
- Follow existing patterns in the codebase (see `bilancio/decision/` as the gold standard).
- Keep changes focused — don't refactor unrelated code.
- After subagents finish, **review what they wrote** and fix issues.

### 2.2 Verify it works

For each acceptance criterion, define a concrete check:

| Criterion | How to verify |
|-----------|---------------|
| "Feature X produces output Y" | `uv run python -c "..."` or a small script |
| "Scenario runs without errors" | `uv run bilancio run examples/scenarios/...` |
| "Metric Z improves by N%" | Run before/after comparison |
| "HTML report shows new section" | `uv run bilancio run ... --html temp/demo.html && open temp/demo.html` |

**Rule**: Every acceptance criterion must be checked. If a check fails, fix the code and re-check before moving on.

### 2.3 Iterate until all checks pass

- Fix → re-check → fix → re-check.
- If a design assumption was wrong, update the plan doc and note the change.
- If scope needs to grow, discuss with the user first.

### 2.4 Backward compatibility check

Run an existing scenario **without** the new feature enabled (default config) and verify the output is identical to before the code change. This catches unintended side effects — a new phase that accidentally fires when it shouldn't, a changed default that shifts behavior, a state-sync issue.

```bash
# Run a known scenario with default params (feature off)
uv run bilancio run examples/scenarios/simple_dealer.yaml \
  --max-days 5 --html temp/compat_check.html

# Inspect: delta, phi, number of events should match pre-change behavior
```

If behavior differs with the feature off, this is a bug — fix before proceeding.

### 2.5 Single-run inspection

Before running a multi-run sweep, run **one simulation** with the feature **enabled** and inspect the output in detail. The sweep is a black-box "did it crash" check — this is a white-box "does the mechanism actually fire" check.

```bash
# Run one simulation with the feature enabled
uv run bilancio run <scenario_with_feature>.yaml \
  --max-days 10 --html temp/feature_inspect.html
open temp/feature_inspect.html
```

Check:
- The new events/phases appear in the event log at the right times.
- The new mechanism produces the expected effect (e.g., trades happen, loans are made, new metric is nonzero).
- No unexpected errors, warnings, or NaN values.
- Balance sheets remain balanced (assets = liabilities + equity).

### 2.6 Smoke sweep

**MANDATORY** after implementation passes acceptance criteria. Run a small local sweep to verify the feature integrates correctly with the full simulation-and-sweep pipeline.

Pick the sweep type most relevant to the feature:

```bash
# Dealer/trading features → balanced sweep
uv run bilancio sweep balanced \
  --out-dir temp/smoke_sweep \
  --n-agents 20 --maturity-days 5 \
  --kappas "0.5,1" --concentrations "1" --mus "0" \
  --outside-mid-ratios "0.90" --quiet

# Banking features → bank sweep
uv run bilancio sweep bank \
  --out-dir temp/smoke_sweep \
  --n-agents 20 --maturity-days 5 \
  --kappas "0.5,1" --concentrations "1" --mus "0" \
  --n-banks 3 --quiet

# Lending features → nbfi sweep
uv run bilancio sweep nbfi \
  --out-dir temp/smoke_sweep \
  --n-agents 20 --maturity-days 5 \
  --kappas "0.5,1" --concentrations "1" --mus "0" \
  --nbfi-share 0.10 --quiet

# Core/settlement features → ring sweep (fastest)
uv run bilancio sweep ring \
  --out-dir temp/smoke_sweep \
  --n-agents 10 --maturity-days 3 \
  --kappas "0.5,1" --concentrations "1" --mus "0"
```

**What to check after the smoke sweep:**

1. **Sweep completes without errors** — no crashes, all runs finish.
2. **Output files exist** — `aggregate/comparison.csv` (or `results.csv` for ring) is written and non-empty.
3. **Metrics are plausible** — delta, phi, trading_effect (or equivalent) are in expected ranges and not NaN.
4. **New metrics appear** — if the feature adds a new metric, verify it shows up in the CSV with sensible values.
5. **Compare to expectations** — check against the "interaction expectations" stated in the plan (Section 1.4).

```bash
# Quick sanity check
uv run python -c "
import pandas as pd
df = pd.read_csv('temp/smoke_sweep/aggregate/comparison.csv')
print(f'Runs: {len(df)}')
print(df.describe())
print(df.head())
"
```

If the smoke sweep fails or produces unexpected results, fix the issue before moving to Phase 3. Clean up temp output afterward:
```bash
rm -rf temp/smoke_sweep temp/compat_check.html temp/feature_inspect.html
```

**Phase 2 gate**: All of the following must be true before moving on: (1) every acceptance criterion passes, (2) backward compatibility check shows no unintended changes, (3) single-run inspection confirms the mechanism fires correctly, (4) smoke sweep completes with plausible metrics matching expectations.

---

## Phase 3: Test & Commit

### 3.1 Write tests

Add tests that cover:

- **Unit tests** — isolated logic (e.g., a new calculation, data structure, profile).
- **Integration tests** — interaction with other modules (e.g., new phase + settlement).
- **Regression tests** — if the feature fixes a bug or changes behavior, add a test that would have caught the old behavior.

Test file location mirrors source:

| Source | Test |
|--------|------|
| `src/bilancio/core/foo.py` | `tests/core/test_foo.py` |
| `src/bilancio/engines/bar.py` | `tests/engines/test_bar.py` |
| `src/bilancio/decision/baz.py` | `tests/decision/test_baz.py` |

### 3.2 Run full test suite

```bash
uv run pytest tests/ -v
```

**All tests must pass.** If existing tests break:
- If the breakage is expected (behavior intentionally changed), update the test.
- If unexpected, fix the implementation — don't patch the test.

### 3.3 Commit

```bash
git add <specific files>
git commit -m "feat: <description of what was implemented>"
```

Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.

**Phase 3 gate**: All tests pass (`uv run pytest tests/ -v` exits 0) and changes are committed. Do NOT open a PR with failing tests.

---

## Phase 4: PR, Review & Merge

### 4.1 Push and open PR

```bash
git push -u origin plan/<NNN>-<short-name>
gh pr create --title "<short description>" --body "..."
```

PR body should include:
- Summary of what changed (1-3 bullets).
- Link to the plan: `docs/plans/<NNN>_<feature_name>.md`.
- Test plan (how to verify it works).

### 4.2 Review the PR

Claude will review the diff for:
- Correctness — does the code do what the plan says?
- Style — does it follow existing patterns?
- Safety — any security, performance, or state-sync issues?
- Completeness — are all acceptance criteria covered by tests?

Fix any issues found during review.

### 4.3 Merge and return to main

After review is clean:

```bash
# Merge via GitHub (squash or merge commit per preference)
gh pr merge <PR-number> --merge

# Return to main locally
git checkout main
git pull
```

Clean up the feature branch:
```bash
git branch -d plan/<NNN>-<short-name>
```

---

## Quick Reference: Feature Checklist

```
[ ] Phase 1: Plan
    [ ] Branch created from clean main
    [ ] Plan doc written with goal, scope, location, design, acceptance criteria
    [ ] Sweep surface specified (backend layers + pipeline layers + expectations)
    [ ] Default value discipline: every new param has a backward-compatible default
    [ ] Claude read all files in the Location table
    [ ] Claude critique + clarification complete
    [ ] Plan committed ← GATE: no implementation before this

[ ] Phase 2: Implement
    [ ] Code written (location matches plan)
    [ ] Each acceptance criterion verified manually
    [ ] All checks pass
    [ ] Backward compat check: existing scenario with defaults produces same output
    [ ] Single-run inspection: feature enabled, events/mechanism verified in HTML
    [ ] Smoke sweep runs, completes, produces plausible metrics
    [ ] Smoke sweep matches interaction expectations from plan
    ← GATE: all 4 checks pass before moving on

[ ] Phase 3: Test & Commit
    [ ] Unit tests written
    [ ] Integration tests written (if applicable)
    [ ] Full test suite passes: uv run pytest tests/ -v
    [ ] Changes committed
    ← GATE: no PR with failing tests

[ ] Phase 4: PR & Merge
    [ ] PR opened with summary and test plan
    [ ] PR review — issues addressed
    [ ] Merged to main
    [ ] Local main updated, feature branch deleted
```

---

## Code Location Guide

When planning where a feature goes, use this map:

| Module | Purpose | Examples |
|--------|---------|----------|
| `core/` | Fundamental data structures, state, events | `State`, `Agent`, `Event`, `Instrument` |
| `domain/` | Domain rules, policy, types | `PolicyEngine`, `InstrumentKind`, agent types |
| `decision/` | Agent decision-making, profiles, strategies | `TraderProfile`, `VBTProfile`, `RiskAssessor` |
| `engines/` | Simulation phases, execution engines | `SettlementEngine`, `DealerIntegration` |
| `banking/` | Banking subsystem | `BankProfile`, `BankingSubsystem`, CB |
| `information/` | Information service, access control | `InformationService`, `InformationProfile` |
| `experiments/` | Sweep orchestration, comparison runners | `BalancedComparison`, `RingSweep` |
| `scenarios/` | Scenario construction and ring setup | `RingBuilder`, `ScenarioConfig` |
| `analysis/` | Post-simulation analysis, visualization | metrics, HTML reports, balance tables |
| `ui/` | CLI commands, rendering | Click commands, HTML renderer |
| `cloud/` | Modal deployment, cloud execution | `modal_app.py`, `CloudExecutor` |
| `ops/` | Operations: actions, transfers, settlement | `transfer`, `settle`, `default` |
| `config/` | Configuration dataclasses | scenario config, sweep config |
| `runners/` | Simulation runners | `LocalExecutor`, `CloudExecutor` |
| `storage/` | Persistence (Supabase, local) | `supabase_client`, `local_store` |
| `jobs/` | Job management, manifests | `JobManager`, `JobManifest` |
| `stats/` | Statistical analysis | metrics computation |
| `specification/` | Formal specs | instrument specs |
| `export/` | Data export formats | JSONL, CSV |
