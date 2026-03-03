#!/usr/bin/env python3
"""Alternative Simulation Efficiency Benchmark (ASEB).

Measures real simulation performance: scaling behavior, cold start cost,
throughput stability, memory leaks, event growth, and complex path coverage.

Score model (100 points):
  1. Scaling Discipline       25 pts  (runtime exponent, memory exponent, event linearity)
  2. Cold Start Cost          10 pts  (import time in subprocess)
  3. Per-Day Throughput       20 pts  (day-over-day ratio, CV of per-day times)
  4. Memory Stability         15 pts  (peak growth across sequential runs)
  5. Event Growth Discipline  10 pts  (events/agent/day consistency)
  6. Complex Path Coverage    20 pts  (dealer path, bank path within budget)

Critical gates (7):
  scaling::runtime_exponent      alpha_runtime <= 2.0
  scaling::memory_exponent       alpha_memory  <= 2.0
  cold_start::under_5s           import_time   <  5.0s
  stability::no_blowup           day_ratio     <  5.0
  memory::no_leak                growth_ratio  <  1.5
  complex::dealer_completes      dealer run succeeds
  complex::bank_completes        bank run succeeds

Usage:
  uv run python scripts/run_alt_simulation_efficiency_benchmark.py
  uv run python scripts/run_alt_simulation_efficiency_benchmark.py --target-score 80
  uv run python scripts/run_alt_simulation_efficiency_benchmark.py --out-json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
import subprocess
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

# Suppress noisy simulation logging (payable defaults, agent expulsions, etc.)
# Set to ERROR so only actual errors leak through; WARNING includes default/expulsion
# messages which clutter benchmark output.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("bilancio").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Bilancio imports
# ---------------------------------------------------------------------------

from bilancio.decision.profiles import BankProfile
from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.banking_subsystem import initialize_banking_subsystem
from bilancio.engines.dealer_integration import initialize_dealer_subsystem
from bilancio.engines.simulation import run_day, run_until_stable
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


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
    """Linear interpolation score.

    full_at: value at which you get max_points.
    zero_at: value at which you get 0.
    """
    if full_at <= zero_at:
        if value <= full_at:
            return max_points
        if value >= zero_at:
            return 0.0
        return max_points * (zero_at - value) / (zero_at - full_at)
    else:
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
    if failure_count >= 4:
        cap = "F"
    elif failure_count >= 2:
        cap = "D"
    else:
        cap = "C"
    order = ["A", "B", "C", "D", "F"]
    return base_grade if order.index(base_grade) >= order.index(cap) else cap


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def build_ring_system(
    n_agents: int = 20,
    cash_per_agent: int = 50,
    payable_amount: int = 100,
    maturity_days: int = 5,
    seed: int = 42,
    default_mode: str = "expel-agent",
) -> System:
    """Build a simple Kalecki ring for benchmarking."""
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


def build_banking_ring_system(
    n_agents: int = 20,
    cash_per_agent: int = 50,
    payable_amount: int = 100,
    maturity_days: int = 5,
    seed: int = 42,
    n_banks: int = 2,
    kappa: Decimal = Decimal("0.5"),
    credit_risk_loading: Decimal = Decimal("0.5"),
    max_borrower_risk: Decimal = Decimal("0.4"),
    reserve_multiplier: int = 10,
    default_mode: str = "expel-agent",
) -> System:
    """Build a ring system with banking subsystem for benchmarking."""
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    cb.rate_escalation_slope = Decimal("0.05")
    cb.max_outstanding_ratio = Decimal("2.0")
    cb.escalation_base_amount = n_agents * payable_amount
    system.add_agent(cb)

    bank_ids: list[str] = []
    for b in range(1, n_banks + 1):
        bank = Bank(id=f"B{b}", name=f"Bank {b}", kind="bank")
        system.add_agent(bank)
        bank_ids.append(f"B{b}")

    trader_banks: dict[str, list[str]] = {}
    for i in range(1, n_agents + 1):
        h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(h)
        system.mint_cash(f"H{i}", cash_per_agent)
        assigned_bank = bank_ids[(i - 1) % n_banks]
        trader_banks[f"H{i}"] = [assigned_bank]
        deposit_cash(system, f"H{i}", assigned_bank, cash_per_agent)

    for bank_id in bank_ids:
        total_deposits = 0
        for cid in system.state.agents[bank_id].liability_ids:
            c = system.state.contracts.get(cid)
            if c and c.kind == InstrumentKind.BANK_DEPOSIT:
                total_deposits += c.amount
        reserves = reserve_multiplier * total_deposits
        system.mint_reserves(bank_id, reserves)

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

    profile = BankProfile(
        credit_risk_loading=credit_risk_loading,
        max_borrower_risk=max_borrower_risk,
    )
    subsystem = initialize_banking_subsystem(
        system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=maturity_days,
        trader_banks=trader_banks,
    )
    system.state.banking_subsystem = subsystem

    return system


# ---------------------------------------------------------------------------
# Category 1: Scaling Discipline (25 pts)
# ---------------------------------------------------------------------------


def _run_scaling_sample(
    n: int, payable_amount: int, maturity_days: int, max_days: int, seed: int
) -> dict[str, Any]:
    """Run one ring simulation and return runtime, peak memory, event count, days."""
    cash_per_agent = int(1.0 * payable_amount)  # kappa = 1.0

    gc.collect()
    tracemalloc.start()
    t0 = perf_counter()

    system = build_ring_system(
        n_agents=n,
        cash_per_agent=cash_per_agent,
        payable_amount=payable_amount,
        maturity_days=maturity_days,
        seed=seed,
        default_mode="expel-agent",
    )
    reports = run_until_stable(system, max_days=max_days)

    elapsed = perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    days_run = len(reports)
    event_count = len(system.state.events)

    return {
        "n": n,
        "runtime_s": elapsed,
        "peak_memory_bytes": peak,
        "event_count": event_count,
        "days_run": days_run,
    }


def category_scaling(ring_sizes: list[int] | None = None) -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 1: Scaling Discipline (25 pts)."""
    if ring_sizes is None:
        ring_sizes = [25, 50, 100, 200]

    print("  Category 1: Scaling Discipline")
    samples: list[dict[str, Any]] = []
    for n in ring_sizes:
        print(f"    Running ring n={n} ...", end=" ", flush=True)
        s = _run_scaling_sample(n, payable_amount=100, maturity_days=5, max_days=10, seed=42)
        print(f"done ({s['runtime_s']:.2f}s, {s['peak_memory_bytes'] / 1e6:.1f}MB, "
              f"{s['event_count']} events, {s['days_run']} days)")
        samples.append(s)

    # Compute worst-case power law exponents across consecutive pairs
    runtime_alphas: list[float] = []
    memory_alphas: list[float] = []
    for i in range(1, len(samples)):
        n1, n2 = samples[i - 1]["n"], samples[i]["n"]
        t1, t2 = samples[i - 1]["runtime_s"], samples[i]["runtime_s"]
        m1, m2 = samples[i - 1]["peak_memory_bytes"], samples[i]["peak_memory_bytes"]

        log_ratio_n = math.log(n2 / n1)
        if t1 > 0 and t2 > 0 and log_ratio_n > 0:
            runtime_alphas.append(math.log(t2 / t1) / log_ratio_n)
        if m1 > 0 and m2 > 0 and log_ratio_n > 0:
            memory_alphas.append(math.log(m2 / m1) / log_ratio_n)

    alpha_runtime = max(runtime_alphas) if runtime_alphas else 3.0
    alpha_memory = max(memory_alphas) if memory_alphas else 3.0

    # Event linearity: events/agent/day ratio between smallest and largest n
    event_densities: list[float] = []
    for s in samples:
        days = max(s["days_run"], 1)
        event_densities.append(s["event_count"] / (s["n"] * days))

    event_ratio = (max(event_densities) / min(event_densities)) if min(event_densities) > 0 else 999.0

    # Score
    runtime_pts = lerp_score(alpha_runtime, full_at=1.2, zero_at=2.5, max_points=12)
    memory_pts = lerp_score(alpha_memory, full_at=1.2, zero_at=2.5, max_points=8)
    event_pts = lerp_score(event_ratio, full_at=1.3, zero_at=3.0, max_points=5)
    total = runtime_pts + memory_pts + event_pts

    checks = [
        CriticalCheck(
            code="scaling::runtime_exponent",
            passed=alpha_runtime <= 2.0,
            message=f"alpha_runtime={alpha_runtime:.3f} (gate: <=2.0)",
        ),
        CriticalCheck(
            code="scaling::memory_exponent",
            passed=alpha_memory <= 2.0,
            message=f"alpha_memory={alpha_memory:.3f} (gate: <=2.0)",
        ),
    ]

    details = {
        "samples": samples,
        "alpha_runtime": round(alpha_runtime, 4),
        "alpha_memory": round(alpha_memory, 4),
        "runtime_alphas": [round(a, 4) for a in runtime_alphas],
        "memory_alphas": [round(a, 4) for a in memory_alphas],
        "event_densities": [round(d, 4) for d in event_densities],
        "event_ratio": round(event_ratio, 4),
        "runtime_pts": round(runtime_pts, 2),
        "memory_pts": round(memory_pts, 2),
        "event_pts": round(event_pts, 2),
    }

    print(f"    => alpha_runtime={alpha_runtime:.3f}, alpha_memory={alpha_memory:.3f}, "
          f"event_ratio={event_ratio:.3f}")
    print(f"    => Score: {total:.1f}/25")

    return CategoryResult("Scaling Discipline", 25, round(total, 2), details), checks


# ---------------------------------------------------------------------------
# Category 2: Cold Start Cost (10 pts)
# ---------------------------------------------------------------------------


def category_cold_start() -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 2: Cold Start Cost (10 pts)."""
    print("  Category 2: Cold Start Cost")

    times: list[float] = []
    for trial in range(3):
        print(f"    Trial {trial + 1}/3 ...", end=" ", flush=True)
        t0 = perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", "import bilancio.engines.simulation"],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = perf_counter() - t0
        times.append(elapsed)
        status = "ok" if result.returncode == 0 else f"rc={result.returncode}"
        print(f"{elapsed:.3f}s ({status})")

    median_time = statistics.median(times)

    # Score: <= 1s -> 10, >= 3s -> 0
    pts = lerp_score(median_time, full_at=1.0, zero_at=3.0, max_points=10)

    checks = [
        CriticalCheck(
            code="cold_start::under_5s",
            passed=median_time < 5.0,
            message=f"median_import_time={median_time:.3f}s (gate: <5.0s)",
        ),
    ]

    details = {
        "times": [round(t, 4) for t in times],
        "median_time": round(median_time, 4),
    }

    print(f"    => median={median_time:.3f}s, score={pts:.1f}/10")

    return CategoryResult("Cold Start Cost", 10, round(pts, 2), details), checks


# ---------------------------------------------------------------------------
# Category 3: Per-Day Throughput Stability (20 pts)
# ---------------------------------------------------------------------------


def category_throughput_stability() -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 3: Per-Day Throughput Stability (20 pts).

    Uses kappa=1.0 (balanced) with maturity=20 so every day has settlements.
    This avoids bimodal timing from idle tail days (after maturity expires
    with no more payables due).
    """
    print("  Category 3: Per-Day Throughput Stability")

    n = 50
    cash_per_agent = 100  # kappa = 1.0 (balanced: enough cash to pay)
    payable_amount = 100
    maturity_days = 20  # payables spread across all 20 days
    num_days = 20

    system = build_ring_system(
        n_agents=n,
        cash_per_agent=cash_per_agent,
        payable_amount=payable_amount,
        maturity_days=maturity_days,
        seed=42,
        default_mode="expel-agent",
    )

    day_times: list[float] = []
    for d in range(num_days):
        t0 = perf_counter()
        run_day(system)
        elapsed = perf_counter() - t0
        day_times.append(elapsed)

    print(f"    Ran {num_days} days: min={min(day_times):.4f}s, max={max(day_times):.4f}s, "
          f"mean={statistics.mean(day_times):.4f}s")

    # Day ratio: median(last 5) / median(first 5)
    first_5 = sorted(day_times[:5])
    last_5 = sorted(day_times[-5:])
    median_first = statistics.median(first_5)
    median_last = statistics.median(last_5)
    day_ratio = (median_last / median_first) if median_first > 0 else 999.0

    # CV of per-day times
    mean_t = statistics.mean(day_times)
    stdev_t = statistics.stdev(day_times) if len(day_times) > 1 else 0.0
    cv = (stdev_t / mean_t) if mean_t > 0 else 0.0

    # Scores
    ratio_pts = lerp_score(day_ratio, full_at=1.3, zero_at=3.0, max_points=12)
    cv_pts = lerp_score(cv, full_at=0.3, zero_at=1.0, max_points=8)
    total = ratio_pts + cv_pts

    checks = [
        CriticalCheck(
            code="stability::no_blowup",
            passed=day_ratio < 5.0,
            message=f"day_ratio={day_ratio:.3f} (gate: <5.0)",
        ),
    ]

    details = {
        "n_agents": n,
        "num_days": num_days,
        "day_times": [round(t, 6) for t in day_times],
        "median_first_5": round(median_first, 6),
        "median_last_5": round(median_last, 6),
        "day_ratio": round(day_ratio, 4),
        "cv": round(cv, 4),
        "ratio_pts": round(ratio_pts, 2),
        "cv_pts": round(cv_pts, 2),
    }

    print(f"    => day_ratio={day_ratio:.3f}, cv={cv:.3f}")
    print(f"    => Score: {total:.1f}/20")

    return CategoryResult("Per-Day Throughput Stability", 20, round(total, 2), details), checks


# ---------------------------------------------------------------------------
# Category 4: Memory Stability (15 pts)
# ---------------------------------------------------------------------------


def category_memory_stability() -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 4: Memory Stability (15 pts)."""
    print("  Category 4: Memory Stability")

    n = 50
    cash_per_agent = 50  # kappa = 0.5
    payable_amount = 100
    maturity_days = 5
    num_runs = 3

    peaks: list[float] = []
    for run_idx in range(1, num_runs + 1):
        gc.collect()
        tracemalloc.start()

        system = build_ring_system(
            n_agents=n,
            cash_per_agent=cash_per_agent,
            payable_amount=payable_amount,
            maturity_days=maturity_days,
            seed=42,
            default_mode="expel-agent",
        )
        run_until_stable(system, max_days=10)

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        peaks.append(peak_mb)
        print(f"    Run {run_idx}/{num_runs}: peak={peak_mb:.2f} MB")

    growth_ratio = (peaks[-1] / peaks[0]) if peaks[0] > 0 else 999.0

    # Score: <= 1.05 -> 15, >= 1.5 -> 0
    pts = lerp_score(growth_ratio, full_at=1.05, zero_at=1.5, max_points=15)

    checks = [
        CriticalCheck(
            code="memory::no_leak",
            passed=growth_ratio < 1.5,
            message=f"growth_ratio={growth_ratio:.4f} (gate: <1.5)",
        ),
    ]

    details = {
        "peaks_mb": [round(p, 4) for p in peaks],
        "growth_ratio": round(growth_ratio, 4),
    }

    print(f"    => growth_ratio={growth_ratio:.4f}, score={pts:.1f}/15")

    return CategoryResult("Memory Stability", 15, round(pts, 2), details), checks


# ---------------------------------------------------------------------------
# Category 5: Event Growth Discipline (10 pts)
# ---------------------------------------------------------------------------


def category_event_discipline(
    scaling_details: dict[str, Any],
) -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 5: Event Growth Discipline (10 pts).

    Uses data already collected in Category 1 (scaling samples).
    """
    print("  Category 5: Event Growth Discipline")

    event_densities = scaling_details.get("event_densities", [])
    if not event_densities or min(event_densities) <= 0:
        print("    => No valid scaling data; awarding 0 points")
        return (
            CategoryResult("Event Growth Discipline", 10, 0.0, {"error": "no scaling data"}),
            [],
        )

    ratio = max(event_densities) / min(event_densities)

    # Score: <= 1.3 -> 10, >= 3.0 -> 0
    pts = lerp_score(ratio, full_at=1.3, zero_at=3.0, max_points=10)

    details = {
        "event_densities": event_densities,
        "ratio": round(ratio, 4),
    }

    print(f"    => event density ratio={ratio:.3f}, score={pts:.1f}/10")

    return CategoryResult("Event Growth Discipline", 10, round(pts, 2), details), []


# ---------------------------------------------------------------------------
# Category 6: Complex Path Coverage (20 pts)
# ---------------------------------------------------------------------------


def _run_dealer_path(budget_s: float) -> dict[str, Any]:
    """Run dealer subsystem benchmark. Returns timing and success info."""
    n = 25
    cash_per_agent = 50  # kappa = 0.5
    payable_amount = 100
    maturity_days = 5

    system = build_ring_system(
        n_agents=n,
        cash_per_agent=cash_per_agent,
        payable_amount=payable_amount,
        maturity_days=maturity_days,
        seed=42,
        default_mode="expel-agent",
    )

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
        seed=42,
    )
    subsystem = initialize_dealer_subsystem(system, dealer_config, current_day=0)
    system.state.dealer_subsystem = subsystem

    t0 = perf_counter()
    try:
        run_until_stable(system, max_days=15, enable_dealer=True)
        elapsed = perf_counter() - t0
        return {"success": True, "elapsed_s": elapsed, "error": None}
    except Exception as exc:
        elapsed = perf_counter() - t0
        return {"success": False, "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}"}


def _run_bank_path(budget_s: float) -> dict[str, Any]:
    """Run banking subsystem benchmark. Returns timing and success info."""
    n = 25
    cash_per_agent = 50  # kappa = 0.5
    maturity_days = 5

    system = build_banking_ring_system(
        n_agents=n,
        cash_per_agent=cash_per_agent,
        payable_amount=100,
        maturity_days=maturity_days,
        seed=42,
        n_banks=2,
        kappa=Decimal("0.5"),
        default_mode="expel-agent",
    )

    t0 = perf_counter()
    try:
        run_until_stable(
            system,
            max_days=15,
            enable_banking=True,
            enable_bank_lending=True,
        )
        elapsed = perf_counter() - t0
        return {"success": True, "elapsed_s": elapsed, "error": None}
    except Exception as exc:
        elapsed = perf_counter() - t0
        return {"success": False, "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}"}


def _complex_path_score(result: dict[str, Any], budget_s: float, max_points: float) -> float:
    """Score a complex path run.

    Completed within budget => max_points.
    Completed within 2x budget => linear from max_points to 0.
    Failed => 0.
    """
    if not result["success"]:
        return 0.0
    elapsed = result["elapsed_s"]
    if elapsed <= budget_s:
        return max_points
    if elapsed >= 2 * budget_s:
        return 0.0
    # Linear interpolation between budget and 2*budget
    return max_points * (2 * budget_s - elapsed) / budget_s


def category_complex_paths() -> tuple[CategoryResult, list[CriticalCheck]]:
    """Category 6: Complex Path Coverage (20 pts)."""
    print("  Category 6: Complex Path Coverage")

    dealer_budget = 30.0
    bank_budget = 15.0

    print(f"    Dealer path (budget={dealer_budget}s) ...", end=" ", flush=True)
    dealer_result = _run_dealer_path(dealer_budget)
    if dealer_result["success"]:
        print(f"done ({dealer_result['elapsed_s']:.2f}s)")
    else:
        print(f"FAILED ({dealer_result['elapsed_s']:.2f}s): {dealer_result['error']}")

    print(f"    Bank path (budget={bank_budget}s) ...", end=" ", flush=True)
    bank_result = _run_bank_path(bank_budget)
    if bank_result["success"]:
        print(f"done ({bank_result['elapsed_s']:.2f}s)")
    else:
        print(f"FAILED ({bank_result['elapsed_s']:.2f}s): {bank_result['error']}")

    dealer_pts = _complex_path_score(dealer_result, dealer_budget, 10)
    bank_pts = _complex_path_score(bank_result, bank_budget, 10)
    total = dealer_pts + bank_pts

    checks = [
        CriticalCheck(
            code="complex::dealer_completes",
            passed=dealer_result["success"],
            message=(
                f"dealer: {'OK' if dealer_result['success'] else 'FAIL'} "
                f"({dealer_result['elapsed_s']:.2f}s)"
                + (f" - {dealer_result['error']}" if dealer_result["error"] else "")
            ),
        ),
        CriticalCheck(
            code="complex::bank_completes",
            passed=bank_result["success"],
            message=(
                f"bank: {'OK' if bank_result['success'] else 'FAIL'} "
                f"({bank_result['elapsed_s']:.2f}s)"
                + (f" - {bank_result['error']}" if bank_result["error"] else "")
            ),
        ),
    ]

    details = {
        "dealer": {
            "budget_s": dealer_budget,
            "elapsed_s": round(dealer_result["elapsed_s"], 4),
            "success": dealer_result["success"],
            "error": dealer_result["error"],
            "pts": round(dealer_pts, 2),
        },
        "bank": {
            "budget_s": bank_budget,
            "elapsed_s": round(bank_result["elapsed_s"], 4),
            "success": bank_result["success"],
            "error": bank_result["error"],
            "pts": round(bank_pts, 2),
        },
    }

    print(f"    => dealer={dealer_pts:.1f}/10, bank={bank_pts:.1f}/10")
    print(f"    => Score: {total:.1f}/20")

    return CategoryResult("Complex Path Coverage", 20, round(total, 2), details), checks


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_json_report(
    categories: list[CategoryResult],
    critical_checks: list[CriticalCheck],
    total_score: float,
    grade: str,
    target_score: float,
    elapsed_total: float,
) -> dict[str, Any]:
    """Build the JSON report dictionary."""
    return {
        "benchmark": "ASEB (Alternative Simulation Efficiency Benchmark)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "total_score": round(total_score, 2),
        "max_score": 100,
        "grade": grade,
        "target_score": target_score,
        "target_met": total_score >= target_score,
        "elapsed_seconds": round(elapsed_total, 2),
        "critical_checks": [asdict(c) for c in critical_checks],
        "critical_failures": sum(1 for c in critical_checks if not c.passed),
        "categories": [asdict(c) for c in categories],
    }


def generate_markdown_report(
    categories: list[CategoryResult],
    critical_checks: list[CriticalCheck],
    total_score: float,
    grade: str,
    target_score: float,
    elapsed_total: float,
) -> str:
    """Generate a Markdown report string."""
    lines: list[str] = []
    lines.append("# Alternative Simulation Efficiency Benchmark (ASEB)")
    lines.append("")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Python:** {sys.version.split()[0]}")
    lines.append(f"**Elapsed:** {elapsed_total:.1f}s")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    critical_failures = sum(1 for c in critical_checks if not c.passed)
    status = "PASS" if total_score >= target_score and critical_failures == 0 else "FAIL"
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Score** | **{total_score:.1f} / 100** |")
    lines.append(f"| **Grade** | **{grade}** |")
    lines.append(f"| **Target** | {target_score:.0f} |")
    lines.append(f"| **Status** | {status} |")
    lines.append(f"| **Critical failures** | {critical_failures} / {len(critical_checks)} |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| # | Category | Earned | Max |")
    lines.append("|---|----------|--------|-----|")
    for i, cat in enumerate(categories, 1):
        bar = "+" * int(cat.earned_points) + "-" * int(cat.max_points - cat.earned_points)
        lines.append(f"| {i} | {cat.name} | {cat.earned_points:.1f} | {cat.max_points:.0f} |")
    lines.append(f"| | **Total** | **{total_score:.1f}** | **100** |")
    lines.append("")

    # Critical gates
    lines.append("## Critical Gates")
    lines.append("")
    lines.append("| Gate | Result | Detail |")
    lines.append("|------|--------|--------|")
    for c in critical_checks:
        icon = "PASS" if c.passed else "**FAIL**"
        lines.append(f"| `{c.code}` | {icon} | {c.message} |")
    lines.append("")

    # Per-category details
    lines.append("## Category Details")
    lines.append("")

    for cat in categories:
        lines.append(f"### {cat.name} ({cat.earned_points:.1f}/{cat.max_points:.0f})")
        lines.append("")
        for key, value in cat.details.items():
            if isinstance(value, dict):
                lines.append(f"**{key}:**")
                for k2, v2 in value.items():
                    lines.append(f"  - {k2}: {v2}")
            elif isinstance(value, list) and len(value) <= 10:
                lines.append(f"- **{key}:** {value}")
            elif isinstance(value, list):
                lines.append(f"- **{key}:** [{len(value)} items] "
                             f"first={value[0]}, last={value[-1]}")
            else:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alternative Simulation Efficiency Benchmark (ASEB)",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=70.0,
        help="Minimum passing score (default: 70)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("temp/alt_simulation_efficiency_benchmark_report.json"),
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("temp/alt_simulation_efficiency_benchmark_report.md"),
        help="Path for Markdown report output",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Alternative Simulation Efficiency Benchmark (ASEB)")
    print("=" * 70)
    print(f"  Target score: {args.target_score}")
    print(f"  Python: {sys.version.split()[0]}")
    print("")

    t_start = perf_counter()

    all_categories: list[CategoryResult] = []
    all_checks: list[CriticalCheck] = []

    # --- Category 1: Scaling Discipline (25 pts) ---
    cat1, checks1 = category_scaling()
    all_categories.append(cat1)
    all_checks.extend(checks1)
    print()

    # --- Category 2: Cold Start Cost (10 pts) ---
    cat2, checks2 = category_cold_start()
    all_categories.append(cat2)
    all_checks.extend(checks2)
    print()

    # --- Category 3: Per-Day Throughput Stability (20 pts) ---
    cat3, checks3 = category_throughput_stability()
    all_categories.append(cat3)
    all_checks.extend(checks3)
    print()

    # --- Category 4: Memory Stability (15 pts) ---
    cat4, checks4 = category_memory_stability()
    all_categories.append(cat4)
    all_checks.extend(checks4)
    print()

    # --- Category 5: Event Growth Discipline (10 pts) ---
    # Reuses data from Category 1
    cat5, checks5 = category_event_discipline(cat1.details)
    all_categories.append(cat5)
    all_checks.extend(checks5)
    print()

    # --- Category 6: Complex Path Coverage (20 pts) ---
    cat6, checks6 = category_complex_paths()
    all_categories.append(cat6)
    all_checks.extend(checks6)
    print()

    # --- Final scoring ---
    total_elapsed = perf_counter() - t_start
    total_score = sum(c.earned_points for c in all_categories)
    critical_failures = sum(1 for c in all_checks if not c.passed)
    base_grade = grade_for_score(total_score)
    final_grade = cap_grade_for_critical_failures(base_grade, critical_failures)

    target_met = total_score >= args.target_score and critical_failures == 0

    # Print final summary
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Score:              {total_score:.1f} / 100")
    print(f"  Grade:              {final_grade}")
    print(f"  Target:             {args.target_score:.0f}")
    print(f"  Target met:         {'YES' if target_met else 'NO'}")
    print(f"  Critical failures:  {critical_failures} / {len(all_checks)}")
    print(f"  Elapsed:            {total_elapsed:.1f}s")
    print()

    for c in all_checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.code}: {c.message}")
    print()

    for cat in all_categories:
        print(f"  {cat.name:30s}  {cat.earned_points:5.1f} / {cat.max_points:.0f}")
    print(f"  {'TOTAL':30s}  {total_score:5.1f} / 100")
    print()

    # --- Write reports ---
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    json_report = generate_json_report(
        all_categories, all_checks, total_score, final_grade,
        args.target_score, total_elapsed,
    )
    args.out_json.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(f"  JSON report: {args.out_json}")

    md_report = generate_markdown_report(
        all_categories, all_checks, total_score, final_grade,
        args.target_score, total_elapsed,
    )
    args.out_md.write_text(md_report, encoding="utf-8")
    print(f"  Markdown report: {args.out_md}")
    print()

    return 0 if target_met else 1


if __name__ == "__main__":
    sys.exit(main())
