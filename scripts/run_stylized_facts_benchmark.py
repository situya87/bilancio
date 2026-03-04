#!/usr/bin/env python3
"""Calibration / Stylized-Facts Benchmark.

Encodes model-level stylized facts with hard pass/fail thresholds.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from benchmark_sim_utils import compile_ring_scenario, scenario_to_system
from benchmark_utils import (
    CategoryResult,
    CriticalCheck,
    bounded,
    build_markdown_report,
    cap_grade_for_critical_failures,
    generated_at_utc,
    grade_for_score,
    lerp_score,
    report_dict,
    write_reports,
)
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.banking_subsystem import initialize_banking_subsystem
from bilancio.engines.lending import LendingConfig
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stylized-facts benchmark.")
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/stylized_facts_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/stylized_facts_benchmark_report.md",
    )
    return parser.parse_args()


def _count_events(system: System, kind: str) -> int:
    return sum(1 for e in system.state.events if e.get("kind") == kind)


def _build_banking_ring_system(
    *,
    n_agents: int,
    cash_per_agent: int,
    payable_amount: int,
    maturity_days: int,
    kappa: Decimal,
    n_banks: int = 2,
    reserve_multiplier: int = 6,
) -> System:
    system = System(default_mode="expel-agent")
    with system.setup():
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        cb.rate_escalation_slope = Decimal("0.05")
        cb.max_outstanding_ratio = Decimal("2.0")
        cb.escalation_base_amount = n_agents * payable_amount
        system.add_agent(cb)

        bank_ids: list[str] = []
        for idx in range(1, n_banks + 1):
            bank = Bank(id=f"B{idx}", name=f"Bank {idx}", kind="bank")
            system.add_agent(bank)
            bank_ids.append(bank.id)

        trader_banks: dict[str, list[str]] = {}
        for i in range(1, n_agents + 1):
            hid = f"H{i}"
            h = Household(id=hid, name=f"Household {i}", kind="household")
            system.add_agent(h)
            system.mint_cash(hid, cash_per_agent)
            assigned_bank = bank_ids[(i - 1) % n_banks]
            trader_banks[hid] = [assigned_bank]
            deposit_cash(system, hid, assigned_bank, cash_per_agent)

        for bank_id in bank_ids:
            total_deposits = 0
            for cid in system.state.agents[bank_id].liability_ids:
                c = system.state.contracts.get(cid)
                if c and c.kind == InstrumentKind.BANK_DEPOSIT:
                    total_deposits += c.amount
            system.mint_reserves(bank_id, reserve_multiplier * total_deposits)

        for i in range(1, n_agents + 1):
            debtor = f"H{i}"
            creditor = f"H{(i % n_agents) + 1}"
            due_day = 1 + ((i - 1) % maturity_days)
            payable = Payable(
                id=system.new_contract_id("P"),
                kind=InstrumentKind.PAYABLE,
                amount=payable_amount,
                denom="X",
                asset_holder_id=creditor,
                liability_issuer_id=debtor,
                due_day=due_day,
            )
            system.add_contract(payable)

    profile = BankProfile(
        credit_risk_loading=Decimal("0.5"),
        max_borrower_risk=Decimal("0.45"),
        min_coverage_ratio=Decimal("0.10"),
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


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    # SF-1: lower kappa => more defaults
    low_kappa_sys = scenario_to_system(
        compile_ring_scenario(
            n_agents=24,
            kappa=Decimal("0.5"),
            concentration=Decimal("1.0"),
            mu=Decimal("0.5"),
            seed=42,
            maturity_days=6,
            name_prefix="sf-low-kappa",
        )
    )
    high_kappa_sys = scenario_to_system(
        compile_ring_scenario(
            n_agents=24,
            kappa=Decimal("1.8"),
            concentration=Decimal("1.0"),
            mu=Decimal("0.5"),
            seed=42,
            maturity_days=6,
            name_prefix="sf-high-kappa",
        )
    )
    run_until_stable(low_kappa_sys, max_days=30)
    run_until_stable(high_kappa_sys, max_days=30)
    low_defaults = len(low_kappa_sys.state.defaulted_agent_ids)
    high_defaults = len(high_kappa_sys.state.defaulted_agent_ids)

    # SF-2: NBFI credit supply activates under shortfalls
    no_lender_sys = scenario_to_system(
        compile_ring_scenario(
            n_agents=20,
            kappa=Decimal("0.4"),
            concentration=Decimal("1.0"),
            mu=Decimal("0.5"),
            seed=7,
            maturity_days=6,
            name_prefix="sf-no-lender",
        )
    )
    lender_sys = scenario_to_system(
        compile_ring_scenario(
            n_agents=20,
            kappa=Decimal("0.4"),
            concentration=Decimal("1.0"),
            mu=Decimal("0.5"),
            seed=7,
            maturity_days=6,
            name_prefix="sf-lender",
        )
    )
    lender = NonBankLender(id="NBFI1", name="Non-Bank Lender")
    lender_sys.add_agent(lender)
    lender_sys.mint_cash("NBFI1", 6000)
    lender_sys.state.lender_config = LendingConfig(
        base_rate=Decimal("0.05"),
        risk_premium_scale=Decimal("0.20"),
        max_single_exposure=Decimal("0.20"),
        max_total_exposure=Decimal("0.90"),
        maturity_days=3,
        horizon=5,
        min_shortfall=1,
    )

    run_until_stable(no_lender_sys, max_days=25)
    run_until_stable(lender_sys, max_days=25, enable_lender=True)
    lender_loans = _count_events(lender_sys, "NonBankLoanCreated")
    defaults_no_lender = len(no_lender_sys.state.defaulted_agent_ids)
    defaults_with_lender = len(lender_sys.state.defaulted_agent_ids)

    # SF-3: banking regime changes outcomes relative to passive
    passive_sys = scenario_to_system(
        compile_ring_scenario(
            n_agents=20,
            kappa=Decimal("0.6"),
            concentration=Decimal("1.0"),
            mu=Decimal("0.5"),
            seed=101,
            maturity_days=5,
            name_prefix="sf-passive",
        )
    )
    run_until_stable(passive_sys, max_days=20)

    banking_sys = _build_banking_ring_system(
        n_agents=20,
        cash_per_agent=50,
        payable_amount=100,
        maturity_days=5,
        kappa=Decimal("0.6"),
        reserve_multiplier=4,
    )
    run_until_stable(banking_sys, max_days=20, enable_banking=True, enable_bank_lending=True)

    defaults_passive = len(passive_sys.state.defaulted_agent_ids)
    defaults_banking = len(banking_sys.state.defaulted_agent_ids)
    cb_loans_passive = _count_events(passive_sys, "CBLoanCreated")
    cb_loans_banking = banking_sys.state.cb_loans_created_count
    bank_loans_issued = _count_events(banking_sys, "BankLoanIssued")
    banking_outcomes_differ = (
        defaults_passive != defaults_banking or cb_loans_passive != cb_loans_banking
    )

    # SF-4: invariants hold in all assessed regimes
    invariant_failures: list[str] = []
    for name, sys in [
        ("low_kappa", low_kappa_sys),
        ("high_kappa", high_kappa_sys),
        ("no_lender", no_lender_sys),
        ("lender", lender_sys),
        ("passive", passive_sys),
        ("banking", banking_sys),
    ]:
        try:
            sys.assert_invariants()
        except Exception as exc:  # noqa: BLE001
            invariant_failures.append(f"{name}: {type(exc).__name__}: {exc}")

    # Scoring
    sf1_monotonic = low_defaults >= high_defaults
    sf1_gap = low_defaults - high_defaults
    cat1 = 20.0 * (1.0 if sf1_monotonic else 0.0)
    cat1 += 10.0 * lerp_score(float(sf1_gap), 8.0, 0.0, 1.0)

    sf2_credit_active = lender_loans >= 1
    sf2_default_improvement = defaults_with_lender <= defaults_no_lender
    cat2 = 20.0 * (1.0 if sf2_credit_active else 0.0)
    cat2 += 10.0 * (1.0 if sf2_default_improvement else 0.0)

    cat3 = 15.0 * (1.0 if bank_loans_issued >= 1 else 0.0)
    cat3 += 10.0 * (1.0 if banking_outcomes_differ else 0.0)

    cat4 = 15.0 if not invariant_failures else 0.0

    categories = [
        CategoryResult(
            name="SF-1 Liquidity Stress Monotonicity",
            max_points=30.0,
            earned_points=round(cat1, 3),
            details={
                "low_defaults": low_defaults,
                "high_defaults": high_defaults,
                "gap": sf1_gap,
                "monotonic": sf1_monotonic,
            },
        ),
        CategoryResult(
            name="SF-2 NBFI Credit Activation",
            max_points=30.0,
            earned_points=round(cat2, 3),
            details={
                "lender_loans": lender_loans,
                "defaults_no_lender": defaults_no_lender,
                "defaults_with_lender": defaults_with_lender,
                "default_improvement": sf2_default_improvement,
            },
        ),
        CategoryResult(
            name="SF-3 Banking Regime Differentiation",
            max_points=25.0,
            earned_points=round(cat3, 3),
            details={
                "bank_loans_issued": bank_loans_issued,
                "defaults_passive": defaults_passive,
                "defaults_banking": defaults_banking,
                "cb_loans_passive": cb_loans_passive,
                "cb_loans_banking": cb_loans_banking,
                "outcomes_differ": banking_outcomes_differ,
            },
        ),
        CategoryResult(
            name="SF-4 Invariant Integrity",
            max_points=15.0,
            earned_points=round(cat4, 3),
            details={"invariant_failures": invariant_failures},
        ),
    ]

    checks = [
        CriticalCheck(
            code="stylized::liquidity_monotonic",
            passed=sf1_monotonic,
            message=f"low_defaults={low_defaults}, high_defaults={high_defaults}",
        ),
        CriticalCheck(
            code="stylized::nbfi_credit_activates",
            passed=sf2_credit_active,
            message=f"lender_loans={lender_loans}",
        ),
        CriticalCheck(
            code="stylized::banking_changes_outcomes",
            passed=(bank_loans_issued >= 1 and banking_outcomes_differ),
            message=(
                f"bank_loans={bank_loans_issued}, defaults(passive/banking)="
                f"{defaults_passive}/{defaults_banking}, cb_loans(passive/banking)="
                f"{cb_loans_passive}/{cb_loans_banking}"
            ),
        ),
        CriticalCheck(
            code="stylized::invariants_hold",
            passed=(len(invariant_failures) == 0),
            message=("all checked systems pass invariants" if not invariant_failures else invariant_failures[0]),
        ),
    ]

    total_score = bounded(sum(c.earned_points for c in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    critical_failures = [c for c in checks if not c.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"

    elapsed = perf_counter() - t0
    generated_at = generated_at_utc()

    report = report_dict(
        benchmark_name="Calibration / Stylized-Facts Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={"generated_at_utc": generated_at},
    )

    md = build_markdown_report(
        title="Calibration / Stylized-Facts Benchmark",
        generated_at=generated_at,
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        grade=grade,
        base_grade=base_grade,
        meets_target=meets_target,
        categories=categories,
        critical_checks=checks,
        summary_lines=[
            f"SF1 defaults low/high={low_defaults}/{high_defaults}",
            f"SF2 lender_loans={lender_loans}, defaults no-lender/lender={defaults_no_lender}/{defaults_with_lender}",
            f"SF3 bank_loans_issued={bank_loans_issued}, outcomes_differ={banking_outcomes_differ}",
            f"SF4 invariant_failures={len(invariant_failures)}",
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Stylized-facts benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
