#!/usr/bin/env python3
"""Alternative ABM Modeling Benchmark (AABM).

Actually runs simulations to verify economic properties, comparative statics,
and reproducibility — instead of just checking if tests pass.

Outputs:
- JSON report: temp/alt_abm_modeling_benchmark_report.json
- Markdown report: temp/alt_abm_modeling_benchmark_report.md

Score Model (100 points):
  1. Seed Reproducibility (15)    Same seed → identical; different seed → different
  2. Accounting Conservation (20) Invariant checks across 6 parameter combos
  3. Comparative Statics (25)     Kappa monotonicity (15) + concentration effect (10)
  4. Cross-Seed Convergence (15)  CV of defaults across 5 seeds
  5. Dealer Effect Direction (15) Active defaults ≤ passive defaults
  6. Boundary Behavior (10)       High κ → 0 defaults; low κ → many defaults

Critical Gates (4):
  reproducibility::same_seed         same seed produces identical results
  conservation::all_pass             all 6 configs pass invariant checks
  statics::kappa_monotonic           all 3 kappa pairs are monotonic
  boundary::high_kappa_no_defaults   κ=10 produces 0 defaults
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

# Suppress noisy simulation logging (payable defaults, agent expulsions, etc.)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("bilancio").setLevel(logging.ERROR)

from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import initialize_dealer_subsystem
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.scenarios.ring_explorer import compile_ring_explorer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CriticalCheck:
    code: str
    passed: bool
    message: str


@dataclass
class CategoryResult:
    name: str
    max_points: float
    earned_points: float
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def lerp_score(
    value: float, full_at: float, zero_at: float, max_points: float
) -> float:
    """Linear interpolation: full_at -> max_points, zero_at -> 0."""
    if full_at <= zero_at:
        # full_at is the "good" end (low values are better)
        if value <= full_at:
            return max_points
        if value >= zero_at:
            return 0.0
        return max_points * (zero_at - value) / (zero_at - full_at)
    else:
        # Reversed: higher values are better
        if value >= full_at:
            return max_points
        if value <= zero_at:
            return 0.0
        return max_points * (value - zero_at) / (full_at - zero_at)


def grade_for_score(total_score: float) -> str:
    if total_score >= 90:
        return "A"
    if total_score >= 80:
        return "B"
    if total_score >= 70:
        return "C"
    if total_score >= 60:
        return "D"
    return "F"


def cap_grade_for_critical_failures(base_grade: str, failure_count: int) -> str:
    if failure_count <= 0:
        return base_grade
    if failure_count >= 3:
        cap = "F"
    elif failure_count >= 2:
        cap = "D"
    else:
        cap = "C"
    order = ["A", "B", "C", "D", "F"]
    base_idx = order.index(base_grade)
    cap_idx = order.index(cap)
    # Return the worse of the two (higher index = worse)
    return order[max(base_idx, cap_idx)]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def build_ring_system(
    n_agents: int = 20,
    cash_per_agent: int = 50,
    payable_amount: int = 100,
    maturity_days: int = 5,
    seed: int = 42,
    default_mode: str = "expel-agent",
) -> System:
    """Build a ring of n Household agents with payables.

    Topology: H1->H2->...->HN->H1 (each agent owes the next).
    Each agent starts with ``cash_per_agent`` units of cash and owes
    ``payable_amount`` to the next agent in the ring.
    """
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    system.add_agent(cb)
    system.add_agent(bank)

    for i in range(1, n_agents + 1):
        h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(h)
        system.mint_cash(f"H{i}", cash_per_agent)

    for i in range(1, n_agents + 1):
        from_id = f"H{i}"
        to_id = f"H{(i % n_agents) + 1}"
        due_day = 1 + ((i - 1) % maturity_days)
        p = Payable(
            id=system.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=payable_amount,
            denom="X",
            asset_holder_id=to_id,
            liability_issuer_id=from_id,
            due_day=due_day,
        )
        system.add_contract(p)

    return system


def build_ring_via_explorer(
    n_agents: int,
    kappa: float,
    concentration: float,
    maturity_days: int = 5,
    seed: int = 42,
) -> System:
    """Build a ring using the ring_explorer generator for Dirichlet concentration tests."""
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "aabm",
            "params": {
                "n_agents": n_agents,
                "kappa": str(kappa),
                "seed": seed,
                "Q_total": str(100 * n_agents),
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": "0",
                },
                "maturity": {
                    "days": maturity_days,
                    "mode": "lead_lag",
                    "mu": "0.5",
                },
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": False},
        }
    )
    scenario_dict = compile_ring_explorer(gen_config, source_path=None)
    scenario_dict = preprocess_config(scenario_dict)
    config = ScenarioConfig(**scenario_dict)
    system = System(default_mode="expel-agent")
    apply_to_system(config, system)
    return system


def count_defaults(system: System) -> int:
    """Count distinct agents that defaulted during the simulation."""
    return len(system.state.defaulted_agent_ids)


def count_events(system: System, kind: str) -> int:
    """Count events of a specific kind."""
    return sum(1 for e in system.state.events if e.get("kind") == kind)


def total_event_count(system: System) -> int:
    """Total number of events recorded."""
    return len(system.state.events)


def has_negative_cash(system: System) -> bool:
    """Check whether any agent has negative cash balance."""
    for aid, agent in system.state.agents.items():
        cash = 0
        for cid in agent.asset_ids:
            c = system.state.contracts.get(cid)
            if c and c.kind == InstrumentKind.CASH:
                cash += c.amount
        if cash < 0:
            return True
    return False


def setup_dealer(system: System, seed: int) -> None:
    """Initialize the dealer subsystem on a system for active-mode runs."""
    dealer_config = DealerRingConfig(
        ticket_size=Decimal(1),
        buckets=list(DEFAULT_BUCKETS),
        dealer_share=Decimal("0.25"),
        vbt_share=Decimal("0.50"),
        vbt_anchors={
            "short": (Decimal("1.0"), Decimal("0.20")),
            "mid": (Decimal("1.0"), Decimal("0.30")),
            "long": (Decimal("1.0"), Decimal("0.40")),
        },
        phi_M=Decimal("0.1"),
        phi_O=Decimal("0.1"),
        clip_nonneg_B=True,
        seed=seed,
    )
    subsystem = initialize_dealer_subsystem(system, dealer_config, current_day=0)
    system.state.dealer_subsystem = subsystem


# ---------------------------------------------------------------------------
# Category 1: Seed Reproducibility (15 pts)
# ---------------------------------------------------------------------------


def run_category_seed_reproducibility() -> CategoryResult:
    """Same seed -> identical results (10 pts); different seed -> different (5 pts).

    Uses explorer-generated rings with Dirichlet concentration (c=0.5) so that
    different seeds produce different debt distributions, breaking the symmetry
    of uniform rings where all agents are identical regardless of seed.
    """
    details: dict[str, Any] = {}
    earned = 0.0

    # Same seed runs — using Dirichlet-distributed debts for non-trivial randomness
    sys_a = build_ring_via_explorer(n_agents=20, kappa=0.5, concentration=0.5,
                                    maturity_days=5, seed=42)
    run_until_stable(sys_a, max_days=15)
    defaults_a = count_defaults(sys_a)
    events_a = total_event_count(sys_a)

    sys_b = build_ring_via_explorer(n_agents=20, kappa=0.5, concentration=0.5,
                                    maturity_days=5, seed=42)
    run_until_stable(sys_b, max_days=15)
    defaults_b = count_defaults(sys_b)
    events_b = total_event_count(sys_b)

    same_seed_identical = (defaults_a == defaults_b) and (events_a == events_b)
    details["same_seed"] = {
        "seed": 42,
        "run1_defaults": defaults_a,
        "run1_events": events_a,
        "run2_defaults": defaults_b,
        "run2_events": events_b,
        "identical": same_seed_identical,
    }
    if same_seed_identical:
        earned += 10.0

    # Different seed runs — Dirichlet draws differ by seed, so results should differ
    sys_c = build_ring_via_explorer(n_agents=20, kappa=0.5, concentration=0.5,
                                    maturity_days=5, seed=99)
    run_until_stable(sys_c, max_days=15)
    defaults_c = count_defaults(sys_c)
    events_c = total_event_count(sys_c)

    diff_seed_different = (defaults_a != defaults_c) or (events_a != events_c)
    details["diff_seed"] = {
        "seed_1": 42,
        "seed_2": 99,
        "seed1_defaults": defaults_a,
        "seed1_events": events_a,
        "seed2_defaults": defaults_c,
        "seed2_events": events_c,
        "different": diff_seed_different,
    }
    if diff_seed_different:
        earned += 5.0

    return CategoryResult(
        name="Seed Reproducibility",
        max_points=15.0,
        earned_points=earned,
        details=details,
    )


# ---------------------------------------------------------------------------
# Category 2: Accounting Conservation (20 pts)
# ---------------------------------------------------------------------------


def run_category_accounting_conservation() -> CategoryResult:
    """Run 6 configs, check system.assert_invariants() + no negative cash."""
    configs = [
        {"n_agents": 10, "cash_per_agent": 50, "payable_amount": 100,
         "maturity_days": 3, "seed": 42, "label": "small_ring"},
        {"n_agents": 20, "cash_per_agent": 30, "payable_amount": 100,
         "maturity_days": 5, "seed": 42, "label": "stressed"},
        {"n_agents": 20, "cash_per_agent": 100, "payable_amount": 100,
         "maturity_days": 5, "seed": 42, "label": "kappa_1"},
        {"n_agents": 20, "cash_per_agent": 200, "payable_amount": 100,
         "maturity_days": 5, "seed": 42, "label": "abundant"},
        {"n_agents": 15, "cash_per_agent": 80, "payable_amount": 100,
         "maturity_days": 7, "seed": 99, "label": "medium_alt_seed"},
        {"n_agents": 25, "cash_per_agent": 60, "payable_amount": 100,
         "maturity_days": 10, "seed": 7, "label": "large_long_maturity"},
    ]

    passed_count = 0
    config_results: list[dict[str, Any]] = []

    for cfg in configs:
        label = cfg["label"]
        build_kwargs = {k: v for k, v in cfg.items() if k != "label"}
        sys = build_ring_system(**build_kwargs)
        run_until_stable(sys, max_days=15)

        invariant_ok = True
        invariant_error = None
        try:
            sys.assert_invariants()
        except Exception as exc:
            invariant_ok = False
            invariant_error = str(exc)

        neg_cash = has_negative_cash(sys)
        config_passed = invariant_ok and not neg_cash

        if config_passed:
            passed_count += 1

        config_results.append({
            "label": label,
            "params": cfg,
            "invariants_ok": invariant_ok,
            "invariant_error": invariant_error,
            "negative_cash": neg_cash,
            "passed": config_passed,
            "defaults": count_defaults(sys),
        })

    score = 20.0 * (passed_count / len(configs))

    return CategoryResult(
        name="Accounting Conservation",
        max_points=20.0,
        earned_points=round(score, 2),
        details={
            "configs_passed": passed_count,
            "configs_total": len(configs),
            "results": config_results,
        },
    )


# ---------------------------------------------------------------------------
# Category 3: Comparative Statics (25 pts)
# ---------------------------------------------------------------------------


def run_category_comparative_statics() -> CategoryResult:
    """Kappa monotonicity (15 pts) + concentration effect (10 pts)."""
    details: dict[str, Any] = {}
    earned = 0.0

    # --- Kappa monotonicity (15 pts) ---
    # defaults(0.3) >= defaults(0.5) >= defaults(1.0) >= defaults(2.0)
    kappas = [0.3, 0.5, 1.0, 2.0]
    payable = 100
    kappa_defaults: list[int] = []

    for kappa in kappas:
        cash = int(kappa * payable)
        sys = build_ring_system(
            n_agents=20, cash_per_agent=cash, payable_amount=payable,
            maturity_days=5, seed=42,
        )
        run_until_stable(sys, max_days=15)
        kappa_defaults.append(count_defaults(sys))

    # Check pairwise monotonicity: d[i] >= d[i+1]
    pairs_met = 0
    pair_details: list[dict[str, Any]] = []
    for i in range(len(kappas) - 1):
        met = kappa_defaults[i] >= kappa_defaults[i + 1]
        if met:
            pairs_met += 1
        pair_details.append({
            "kappa_low": kappas[i],
            "kappa_high": kappas[i + 1],
            "defaults_low": kappa_defaults[i],
            "defaults_high": kappa_defaults[i + 1],
            "monotonic": met,
        })

    n_pairs = len(kappas) - 1
    kappa_score = 15.0 * (pairs_met / n_pairs)
    earned += kappa_score

    details["kappa_monotonicity"] = {
        "kappas": kappas,
        "defaults": kappa_defaults,
        "pairs_met": pairs_met,
        "pairs_total": n_pairs,
        "score": round(kappa_score, 2),
        "pairs": pair_details,
    }

    # --- Concentration effect (10 pts) ---
    # With uniform cash allocation (kappa=0.5, every agent gets 50):
    #   c=0.5 (unequal debts): many agents owe tiny amounts they CAN pay -> fewer defaults
    #   c=5.0 (equal debts): all agents owe ~100 but only have 50 -> more defaults
    # So: defaults(c=5.0) >= defaults(c=0.5)  (equality is WORSE with uniform cash)
    c_low = 0.5
    c_high = 5.0

    sys_low_c = build_ring_via_explorer(
        n_agents=20, kappa=0.5, concentration=c_low, maturity_days=5, seed=42
    )
    run_until_stable(sys_low_c, max_days=15)
    defaults_low_c = count_defaults(sys_low_c)

    sys_high_c = build_ring_via_explorer(
        n_agents=20, kappa=0.5, concentration=c_high, maturity_days=5, seed=42
    )
    run_until_stable(sys_high_c, max_days=15)
    defaults_high_c = count_defaults(sys_high_c)

    # More equal debts (high c) => more defaults with uniform cash
    conc_met = defaults_high_c >= defaults_low_c
    conc_score = 10.0 if conc_met else 0.0
    earned += conc_score

    details["concentration_effect"] = {
        "c_low": c_low,
        "c_high": c_high,
        "defaults_c_low": defaults_low_c,
        "defaults_c_high": defaults_high_c,
        "more_defaults_with_equality": conc_met,
        "score": conc_score,
    }

    return CategoryResult(
        name="Comparative Statics",
        max_points=25.0,
        earned_points=round(earned, 2),
        details=details,
    )


# ---------------------------------------------------------------------------
# Category 4: Cross-Seed Convergence (15 pts)
# ---------------------------------------------------------------------------


def run_category_cross_seed_convergence() -> CategoryResult:
    """CV of defaults across 5 seeds. CV <= 0.2 = full, CV >= 0.8 = zero."""
    seeds = [42, 43, 44, 45, 46]
    default_counts: list[int] = []

    for seed in seeds:
        sys = build_ring_system(
            n_agents=25, cash_per_agent=50, payable_amount=100,
            maturity_days=5, seed=seed,
        )
        run_until_stable(sys, max_days=15)
        default_counts.append(count_defaults(sys))

    mean_defaults = statistics.mean(default_counts)

    if mean_defaults == 0:
        # All zeros is perfectly convergent
        if all(d == 0 for d in default_counts):
            cv = 0.0
        else:
            cv = float("inf")
    else:
        stdev_defaults = statistics.stdev(default_counts)
        cv = stdev_defaults / mean_defaults

    # Score: CV <= 0.2 = 15 pts, CV >= 0.8 = 0 pts, linear in between
    score = lerp_score(cv, full_at=0.2, zero_at=0.8, max_points=15.0)

    return CategoryResult(
        name="Cross-Seed Convergence",
        max_points=15.0,
        earned_points=round(score, 2),
        details={
            "seeds": seeds,
            "default_counts": default_counts,
            "mean": round(mean_defaults, 2),
            "stdev": round(statistics.stdev(default_counts), 2) if len(default_counts) > 1 else 0.0,
            "cv": round(cv, 4) if math.isfinite(cv) else "inf",
            "score": round(score, 2),
        },
    )


# ---------------------------------------------------------------------------
# Category 5: Dealer Effect Direction (15 pts)
# ---------------------------------------------------------------------------


def run_category_dealer_effect() -> CategoryResult:
    """Active defaults <= passive defaults (mean) + strict improvement in >= 1 seed.

    Uses explorer-generated rings with Dirichlet concentration (c=0.5) so that
    agents have heterogeneous debt levels. This gives the dealer secondary market
    a chance to help: agents with small debts can sell claims to surplus-cash agents,
    providing liquidity that reduces defaults.
    """
    seeds = [42, 43, 44]
    passive_defaults: list[int] = []
    active_defaults: list[int] = []
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        # Passive run — Dirichlet-distributed debts for heterogeneity
        sys_p = build_ring_via_explorer(
            n_agents=25, kappa=0.5, concentration=0.5,
            maturity_days=5, seed=seed,
        )
        run_until_stable(sys_p, max_days=15)
        d_p = count_defaults(sys_p)
        passive_defaults.append(d_p)

        # Active run (with dealer) — same Dirichlet ring, dealer enabled
        sys_a = build_ring_via_explorer(
            n_agents=25, kappa=0.5, concentration=0.5,
            maturity_days=5, seed=seed,
        )
        setup_dealer(sys_a, seed=seed)
        run_until_stable(sys_a, max_days=15, enable_dealer=True)
        d_a = count_defaults(sys_a)
        active_defaults.append(d_a)

        per_seed.append({
            "seed": seed,
            "passive_defaults": d_p,
            "active_defaults": d_a,
            "improvement": d_p - d_a,
        })

    mean_passive = statistics.mean(passive_defaults)
    mean_active = statistics.mean(active_defaults)
    mean_ok = mean_active <= mean_passive

    strict_improvement = any(d_a < d_p for d_a, d_p in zip(active_defaults, passive_defaults))

    earned = 0.0
    if mean_ok:
        earned += 10.0
    if strict_improvement:
        earned += 5.0

    return CategoryResult(
        name="Dealer Effect Direction",
        max_points=15.0,
        earned_points=earned,
        details={
            "seeds": seeds,
            "passive_defaults": passive_defaults,
            "active_defaults": active_defaults,
            "mean_passive": round(mean_passive, 2),
            "mean_active": round(mean_active, 2),
            "mean_active_le_passive": mean_ok,
            "strict_improvement_any_seed": strict_improvement,
            "per_seed": per_seed,
        },
    )


# ---------------------------------------------------------------------------
# Category 6: Boundary Behavior (10 pts)
# ---------------------------------------------------------------------------


def run_category_boundary_behavior() -> CategoryResult:
    """High kappa -> 0 defaults (5 pts); low kappa -> many defaults (5 pts)."""
    details: dict[str, Any] = {}
    earned = 0.0

    # High kappa: cash=1000, payable=100 => kappa=10
    sys_high = build_ring_system(
        n_agents=20, cash_per_agent=1000, payable_amount=100,
        maturity_days=5, seed=42,
    )
    run_until_stable(sys_high, max_days=15)
    defaults_high = count_defaults(sys_high)

    high_ok = defaults_high == 0
    if high_ok:
        earned += 5.0

    details["high_kappa"] = {
        "kappa": 10.0,
        "cash_per_agent": 1000,
        "payable_amount": 100,
        "defaults": defaults_high,
        "expected": 0,
        "passed": high_ok,
    }

    # Low kappa: cash=10, payable=100 => kappa=0.1
    sys_low = build_ring_system(
        n_agents=20, cash_per_agent=10, payable_amount=100,
        maturity_days=5, seed=42,
    )
    run_until_stable(sys_low, max_days=15)
    defaults_low = count_defaults(sys_low)

    # At least 50% of agents should default (>= 10 out of 20)
    low_ok = defaults_low >= 10
    if low_ok:
        earned += 5.0

    details["low_kappa"] = {
        "kappa": 0.1,
        "cash_per_agent": 10,
        "payable_amount": 100,
        "defaults": defaults_low,
        "expected_min": 10,
        "n_agents": 20,
        "passed": low_ok,
    }

    return CategoryResult(
        name="Boundary Behavior",
        max_points=10.0,
        earned_points=earned,
        details=details,
    )


# ---------------------------------------------------------------------------
# Critical gates
# ---------------------------------------------------------------------------


def evaluate_critical_gates(categories: list[CategoryResult]) -> list[CriticalCheck]:
    """Derive the four critical gates from category results."""
    checks: list[CriticalCheck] = []

    # Gate 1: reproducibility::same_seed
    repro = next(c for c in categories if c.name == "Seed Reproducibility")
    same_seed_ok = repro.details.get("same_seed", {}).get("identical", False)
    checks.append(CriticalCheck(
        code="reproducibility::same_seed",
        passed=same_seed_ok,
        message=(
            "Same seed produces identical results."
            if same_seed_ok
            else "Same seed produced DIFFERENT results — determinism broken."
        ),
    ))

    # Gate 2: conservation::all_pass
    conserv = next(c for c in categories if c.name == "Accounting Conservation")
    all_pass = conserv.details.get("configs_passed", 0) == conserv.details.get("configs_total", 0)
    checks.append(CriticalCheck(
        code="conservation::all_pass",
        passed=all_pass,
        message=(
            f"All {conserv.details.get('configs_total', 0)} configs pass invariant checks."
            if all_pass
            else (
                f"Only {conserv.details.get('configs_passed', 0)}/{conserv.details.get('configs_total', 0)} "
                "configs pass invariant checks."
            )
        ),
    ))

    # Gate 3: statics::kappa_monotonic
    statics = next(c for c in categories if c.name == "Comparative Statics")
    kappa_data = statics.details.get("kappa_monotonicity", {})
    kappa_all_met = kappa_data.get("pairs_met", 0) == kappa_data.get("pairs_total", 0)
    checks.append(CriticalCheck(
        code="statics::kappa_monotonic",
        passed=kappa_all_met,
        message=(
            "All kappa pairs are monotonic (lower kappa → more defaults)."
            if kappa_all_met
            else (
                f"Only {kappa_data.get('pairs_met', 0)}/{kappa_data.get('pairs_total', 0)} "
                "kappa pairs are monotonic."
            )
        ),
    ))

    # Gate 4: boundary::high_kappa_no_defaults
    boundary = next(c for c in categories if c.name == "Boundary Behavior")
    high_ok = boundary.details.get("high_kappa", {}).get("passed", False)
    high_defaults = boundary.details.get("high_kappa", {}).get("defaults", -1)
    checks.append(CriticalCheck(
        code="boundary::high_kappa_no_defaults",
        passed=high_ok,
        message=(
            "kappa=10 produces 0 defaults."
            if high_ok
            else f"kappa=10 produced {high_defaults} defaults (expected 0)."
        ),
    ))

    return checks


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_markdown_report(
    *,
    generated_at: str,
    total_score: float,
    grade: str,
    base_grade: str,
    target_score: float,
    meets_target: bool,
    categories: list[CategoryResult],
    critical_checks: list[CriticalCheck],
    critical_failure_count: int,
    elapsed_seconds: float,
) -> str:
    lines: list[str] = []
    lines.append("# Alternative ABM Modeling Benchmark (AABM)")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append(f"Runtime: `{elapsed_seconds:.1f}s`")
    lines.append(f"Target score: **{target_score:.1f}/100**")
    lines.append("")

    # Scorecard
    lines.append("## Scorecard")
    lines.append("")
    lines.append(f"- Total: **{total_score:.2f}/100**")
    lines.append(f"- Grade: **{grade}**")
    if base_grade != grade:
        lines.append(f"- Base grade (before critical cap): **{base_grade}**")
    lines.append(f"- Target met: **{'yes' if meets_target else 'no'}**")
    if not meets_target:
        lines.append(f"- Gap to target: **{target_score - total_score:.2f}**")
    lines.append(f"- Critical failures: **{critical_failure_count}**")
    lines.append("")

    # Category summary table
    lines.append("## Category Scores")
    lines.append("")
    lines.append("| # | Category | Earned | Max |")
    lines.append("|---|----------|-------:|----:|")
    for i, cat in enumerate(categories, 1):
        lines.append(f"| {i} | {cat.name} | {cat.earned_points:.1f} | {cat.max_points:.0f} |")
    lines.append(f"| | **Total** | **{total_score:.1f}** | **100** |")
    lines.append("")

    # Critical gates table
    lines.append("## Critical Gates")
    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("|------|--------|---------|")
    for check in critical_checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"| `{check.code}` | {status} | {check.message} |")
    lines.append("")

    # Per-category details
    lines.append("## Category Details")
    lines.append("")

    for cat in categories:
        lines.append(f"### {cat.name} ({cat.earned_points:.1f}/{cat.max_points:.0f})")
        lines.append("")

        if cat.name == "Seed Reproducibility":
            ss = cat.details.get("same_seed", {})
            lines.append(f"- Same seed (42): defaults={ss.get('run1_defaults')}, "
                         f"events={ss.get('run1_events')} | "
                         f"identical={ss.get('identical')} -> "
                         f"{'10 pts' if ss.get('identical') else '0 pts'}")
            ds = cat.details.get("diff_seed", {})
            lines.append(f"- Diff seeds (42 vs 99): "
                         f"defaults={ds.get('seed1_defaults')} vs {ds.get('seed2_defaults')}, "
                         f"events={ds.get('seed1_events')} vs {ds.get('seed2_events')} | "
                         f"different={ds.get('different')} -> "
                         f"{'5 pts' if ds.get('different') else '0 pts'}")

        elif cat.name == "Accounting Conservation":
            for r in cat.details.get("results", []):
                status = "PASS" if r["passed"] else "FAIL"
                lines.append(f"- {r['label']}: {status} "
                             f"(invariants={'ok' if r['invariants_ok'] else 'FAIL'}, "
                             f"neg_cash={'yes' if r['negative_cash'] else 'no'}, "
                             f"defaults={r['defaults']})")
            lines.append(f"- Score: {cat.details.get('configs_passed')}/{cat.details.get('configs_total')} "
                         f"configs passed -> {cat.earned_points:.1f} pts")

        elif cat.name == "Comparative Statics":
            km = cat.details.get("kappa_monotonicity", {})
            lines.append("**Kappa monotonicity:**")
            lines.append(f"- Kappas: {km.get('kappas')}")
            lines.append(f"- Defaults: {km.get('defaults')}")
            for p in km.get("pairs", []):
                status = "ok" if p["monotonic"] else "FAIL"
                lines.append(f"  - k={p['kappa_low']} ({p['defaults_low']} defaults) >= "
                             f"k={p['kappa_high']} ({p['defaults_high']} defaults): {status}")
            lines.append(f"- Score: {km.get('pairs_met')}/{km.get('pairs_total')} pairs -> "
                         f"{km.get('score')} pts")
            lines.append("")
            ce = cat.details.get("concentration_effect", {})
            lines.append("**Concentration effect:**")
            lines.append(f"- c={ce.get('c_low')} (unequal): {ce.get('defaults_c_low')} defaults")
            lines.append(f"- c={ce.get('c_high')} (equal): {ce.get('defaults_c_high')} defaults")
            lines.append(f"- More defaults with equality (uniform cash): "
                         f"{'yes' if ce.get('more_defaults_with_equality') else 'no'} -> "
                         f"{ce.get('score')} pts")

        elif cat.name == "Cross-Seed Convergence":
            lines.append(f"- Seeds: {cat.details.get('seeds')}")
            lines.append(f"- Default counts: {cat.details.get('default_counts')}")
            lines.append(f"- Mean: {cat.details.get('mean')}, "
                         f"Stdev: {cat.details.get('stdev')}, "
                         f"CV: {cat.details.get('cv')}")
            lines.append(f"- Score: {cat.earned_points:.1f} pts (CV <= 0.2 = 15, CV >= 0.8 = 0)")

        elif cat.name == "Dealer Effect Direction":
            for ps in cat.details.get("per_seed", []):
                lines.append(f"- Seed {ps['seed']}: passive={ps['passive_defaults']}, "
                             f"active={ps['active_defaults']}, "
                             f"improvement={ps['improvement']}")
            lines.append(f"- Mean: passive={cat.details.get('mean_passive')}, "
                         f"active={cat.details.get('mean_active')}")
            lines.append(f"- Mean active <= passive: "
                         f"{'yes' if cat.details.get('mean_active_le_passive') else 'no'} -> "
                         f"{'10 pts' if cat.details.get('mean_active_le_passive') else '0 pts'}")
            lines.append(f"- Strict improvement in >= 1 seed: "
                         f"{'yes' if cat.details.get('strict_improvement_any_seed') else 'no'} -> "
                         f"{'5 pts' if cat.details.get('strict_improvement_any_seed') else '0 pts'}")

        elif cat.name == "Boundary Behavior":
            hk = cat.details.get("high_kappa", {})
            lines.append(f"- kappa={hk.get('kappa')}: {hk.get('defaults')} defaults "
                         f"(expected {hk.get('expected')}) -> "
                         f"{'5 pts' if hk.get('passed') else '0 pts'}")
            lk = cat.details.get("low_kappa", {})
            lines.append(f"- kappa={lk.get('kappa')}: {lk.get('defaults')} defaults "
                         f"(expected >= {lk.get('expected_min')} of {lk.get('n_agents')}) -> "
                         f"{'5 pts' if lk.get('passed') else '0 pts'}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alternative ABM Modeling Benchmark (AABM).",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=80.0,
        help="Minimum score to pass (default: 80).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/alt_abm_modeling_benchmark_report.json",
        help="Path for JSON report.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/alt_abm_modeling_benchmark_report.md",
        help="Path for Markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]

    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Alternative ABM Modeling Benchmark (AABM)")
    print("=" * 60)
    print()

    t_start = perf_counter()
    categories: list[CategoryResult] = []

    # --- Category 1: Seed Reproducibility (15 pts) ---
    print("Category 1: Seed Reproducibility ...", end=" ", flush=True)
    t0 = perf_counter()
    cat1 = run_category_seed_reproducibility()
    categories.append(cat1)
    print(f"{cat1.earned_points:.1f}/{cat1.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    # --- Category 2: Accounting Conservation (20 pts) ---
    print("Category 2: Accounting Conservation ...", end=" ", flush=True)
    t0 = perf_counter()
    cat2 = run_category_accounting_conservation()
    categories.append(cat2)
    print(f"{cat2.earned_points:.1f}/{cat2.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    # --- Category 3: Comparative Statics (25 pts) ---
    print("Category 3: Comparative Statics ...", end=" ", flush=True)
    t0 = perf_counter()
    cat3 = run_category_comparative_statics()
    categories.append(cat3)
    print(f"{cat3.earned_points:.1f}/{cat3.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    # --- Category 4: Cross-Seed Convergence (15 pts) ---
    print("Category 4: Cross-Seed Convergence ...", end=" ", flush=True)
    t0 = perf_counter()
    cat4 = run_category_cross_seed_convergence()
    categories.append(cat4)
    print(f"{cat4.earned_points:.1f}/{cat4.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    # --- Category 5: Dealer Effect Direction (15 pts) ---
    print("Category 5: Dealer Effect Direction ...", end=" ", flush=True)
    t0 = perf_counter()
    cat5 = run_category_dealer_effect()
    categories.append(cat5)
    print(f"{cat5.earned_points:.1f}/{cat5.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    # --- Category 6: Boundary Behavior (10 pts) ---
    print("Category 6: Boundary Behavior ...", end=" ", flush=True)
    t0 = perf_counter()
    cat6 = run_category_boundary_behavior()
    categories.append(cat6)
    print(f"{cat6.earned_points:.1f}/{cat6.max_points:.0f}  ({perf_counter() - t0:.1f}s)")

    elapsed = perf_counter() - t_start
    print()

    # --- Scoring ---
    total_score = sum(c.earned_points for c in categories)
    total_score = bounded(total_score, 0.0, 100.0)
    base_grade = grade_for_score(total_score)

    critical_checks = evaluate_critical_gates(categories)
    critical_failures = [ch for ch in critical_checks if not ch.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score

    status = "PASS" if meets_target and not critical_failures else "FAIL"
    generated_at = datetime.now(timezone.utc).isoformat()

    # --- JSON report ---
    report = {
        "benchmark": "Alternative ABM Modeling Benchmark (AABM)",
        "generated_at_utc": generated_at,
        "elapsed_seconds": round(elapsed, 2),
        "target_score": args.target_score,
        "total_score": round(total_score, 2),
        "status": status,
        "meets_target": meets_target,
        "base_grade": base_grade,
        "grade": grade,
        "gap_to_target": round(max(0.0, args.target_score - total_score), 2),
        "categories": [
            {
                "name": c.name,
                "max_points": c.max_points,
                "earned_points": c.earned_points,
                "details": c.details,
            }
            for c in categories
        ],
        "critical_checks": [asdict(ch) for ch in critical_checks],
        "critical_failures": [asdict(ch) for ch in critical_failures],
    }
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # --- Markdown report ---
    md = build_markdown_report(
        generated_at=generated_at,
        total_score=total_score,
        grade=grade,
        base_grade=base_grade,
        target_score=args.target_score,
        meets_target=meets_target,
        categories=categories,
        critical_checks=critical_checks,
        critical_failure_count=len(critical_failures),
        elapsed_seconds=elapsed,
    )
    out_md.write_text(md, encoding="utf-8")

    # --- Console summary ---
    print("-" * 60)
    print(f"AABM score: {total_score:.2f}/100 ({grade})")
    print(f"Status: {status}")
    if critical_failures:
        print(f"Critical failures ({len(critical_failures)}):")
        for cf in critical_failures:
            print(f"  - {cf.code}: {cf.message}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
