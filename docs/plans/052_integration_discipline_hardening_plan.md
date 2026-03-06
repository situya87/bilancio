# Plan 052: Integration Discipline Hardening

Status: W1-W4 completed; W5 deferred  
Owner: Core team  
Last updated: 2026-03-06

## Current status

Completed in this pass:

1. W1 static quality gates
2. W2 pre-commit guardrails
3. W3 dependency segmentation baseline
4. W4 CLI hygiene cleanup

Deferred:

1. W5 large-module decomposition pilot
2. Repo-wide Ruff/mypy enforcement beyond the current clean CLI/sweep surface
3. Repo-wide benchmark gating on pull requests

Benchmark integration adjustment completed alongside W1:

1. Fast PR CI now focuses on quality checks and the normal non-benchmark suite.
2. Heavy benchmark workflows moved off pull requests to nightly/manual/release-style execution.

## Why this plan

The repo is shipping features quickly and has strong domain tests, but integration discipline is lagging behind growth.  
Main risk has shifted from "can we build features?" to "can we keep system surface area governable?"

This plan targets four areas:

1. Static checks enforcement in CI
2. Dependency pruning and packaging hygiene
3. CLI cleanup and consistency
4. Consolidation of very large experiment/analysis modules

## Objectives

1. Add hard quality gates for lint/type checks on PRs.
2. Reduce default install footprint for core simulation workflows.
3. Make CLI command structure easier to maintain and extend.
4. Start controlled decomposition of oversized modules without changing simulation behavior.

## Non-goals

1. No behavioral redesign of simulation logic in this plan.
2. No broad architecture rewrite in one PR.
3. No removal of benchmark suites; only better integration discipline around them.

## Workstreams

## W1. Static Quality Gates (Straightforward, high ROI)

Priority: P0  
Effort: Small  
Risk: Low

### Scope

1. Add CI workflow for `ruff` and `mypy` on pull requests.
2. Add a fast smoke test job for critical CLI/simulation paths.
3. Make these checks required for merge (branch protection setting).

### Deliverables

1. New/updated workflow(s) under `.github/workflows/`.
2. Documented local commands in `CONTRIBUTING.md`.
3. Optional convenience scripts (`scripts/check_quality.sh` or equivalent).

### Acceptance criteria

1. PR fails if `ruff` fails on the enforced clean scope.
2. PR fails if `mypy` fails on the enforced clean scope.
3. PR fails if smoke test suite fails.

## W2. Pre-commit Guardrails (Straightforward)

Priority: P0  
Effort: Small  
Risk: Low

### Scope

1. Add `.pre-commit-config.yaml` with:
   - `ruff check --fix` (or check-only, team decision)
   - formatting hook (`ruff format` or `black`, choose one canonical formatter)
   - basic hygiene hooks (EOF/newline/whitespace)
2. Document setup and expected workflow.

### Acceptance criteria

1. `pre-commit run --all-files` passes with repo-wide hygiene hooks plus the enforced Ruff scope.
2. New contributors can install and run hooks in under 5 minutes.

## W3. Dependency Segmentation and Pruning (Straightforward to Medium)

Priority: P1  
Effort: Small-Medium  
Risk: Medium

### Scope

1. Move optional tooling out of core runtime dependencies into extras:
   - Notebook/UI stack (`jupyter`, `notebook`, `streamlit`, `altair`, `matplotlib`, `seaborn`)
   - Cloud/storage extras (`modal`, `supabase`, `psycopg2-binary`, `python-socks`)
2. Keep core simulation/CLI minimal by default.
3. Remove duplicate dependency declarations and tighten ownership of dependency groups.

### Deliverables

1. Updated `pyproject.toml` dependency groups.
2. Installation matrix in docs:
   - core
   - dev
   - cloud
   - analysis/viz
   - notebooks

### Acceptance criteria

1. Core install runs scenario `validate` and `run` without optional extras.
2. Optional feature modules fail with clear guidance if extra is missing.
3. CI matrix covers at least `core`, `dev`, and one optional profile.

## W4. CLI Hygiene and Consolidation (Medium)

Priority: P1  
Effort: Medium  
Risk: Medium

### Scope

1. Remove duplicated error-handling patterns across CLI command modules.
2. Standardize option parsing and output conventions.
3. Break up oversized command modules (starting with `src/bilancio/ui/cli/sweep.py`) into submodules.

### Deliverables

1. Shared CLI utility layer for common error/render patterns.
2. Command modules with clearer boundaries and smaller files.
3. Regression tests for command help text and key flows.

### Acceptance criteria

1. CLI behavior remains backward compatible for existing command flags.
2. `tests/ui` coverage remains green.
3. Largest CLI command file reduced materially (target: <1000 LOC in first pass).

## W5. Large Module Decomposition Program (Heavy)

Priority: P2  
Effort: Large (multi-PR)  
Risk: High if rushed

### Candidate modules

1. `src/bilancio/experiments/balanced_comparison.py`
2. `src/bilancio/analysis/post_sweep.py`
3. `src/bilancio/experiments/ring.py`

### Strategy

1. Do one module at a time.
2. Introduce stable facade APIs first, then extract internals behind facades.
3. Preserve old import paths temporarily with compatibility shims.
4. Require parity tests before and after each extraction.

### Decomposition template (per module)

1. Identify seams:
   - config/parsing
   - orchestration
   - metric computation
   - output/reporting
2. Extract pure/data-only functions first.
3. Extract stateful orchestration second.
4. Remove dead code after parity checks.

### Acceptance criteria

1. No behavioral regressions in benchmark and regression suites.
2. Public APIs remain stable or have explicit migration notes.
3. Each extraction PR remains reviewable (target: <800 changed LOC unless approved).

## Sequencing

### Phase A (quick wins, low risk)

1. W1 static quality gates
2. W2 pre-commit
3. W3 dependency segmentation baseline

### Phase B (medium)

1. W4 CLI hygiene cleanup
2. W3 dependency matrix hardening in CI

### Phase C (heavy, incremental)

1. W5 module decomposition pilot on one module
2. Evaluate impact
3. Continue module-by-module

## Simulation Safety Guardrails

Every PR in this plan must keep simulation quality intact.

1. Run targeted regression tests for affected paths.
2. Run relevant benchmark scripts when touching experiment/analysis/orchestration code.
3. Prove parity for refactors:
   - same scenario inputs
   - equivalent key metrics (within tolerance where stochastic)
4. Avoid changing decision rules or stop conditions unless explicitly scoped and reviewed.
5. Keep refactors isolated from behavioral feature work.

## Success Metrics

1. PRs blocked on lint/type/smoke failures.
2. Reduced install size and faster CI setup for core profile.
3. Reduced concentration risk from giant modules.
4. No increase in simulation regressions attributable to refactor-only PRs.

## Implementation Notes for Later

1. Start with W1 + W2 in one PR if small enough, otherwise split into two.
2. Do W3 as a dedicated packaging PR with explicit install docs updates.
3. Run W4 before W5 to simplify command and workflow surfaces first.
4. Treat W5 as a standing multi-PR track with strict parity checks.
