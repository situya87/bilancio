#!/usr/bin/env python3
"""Failure-Injection Integration Benchmark.

Injects partial failures and verifies graceful fail/skip semantics,
status accounting consistency, and no silent metric corruption.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark_sim_utils import compile_ring_scenario
from benchmark_utils import (
    CategoryResult,
    CriticalCheck,
    bounded,
    build_markdown_report,
    cap_grade_for_critical_failures,
    generated_at_utc,
    grade_for_score,
    report_dict,
    write_reports,
)
from bilancio.cloud.modal_app import compute_metrics_from_events
from bilancio.runners.local_executor import LocalExecutor
from bilancio.runners.models import RunOptions
from bilancio.runners.retry import retry_transient
from bilancio.storage.models import RunStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run failure-injection integration benchmark.")
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/failure_injection_integration_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/failure_injection_integration_benchmark_report.md",
    )
    return parser.parse_args()


def _invalid_scenario_missing_agent() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "invalid-missing-agent",
        "description": "Intentional failure fixture",
        "agents": [
            {"id": "CB", "kind": "central_bank", "name": "CB"},
        ],
        "initial_actions": [
            {"mint_cash": {"to": "H1", "amount": 100}},
        ],
        "run": {
            "mode": "until_stable",
            "max_days": 5,
            "quiet_days": 2,
            "show": {"balances": [], "events": "none"},
            "export": {},
        },
    }


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md

    t0 = perf_counter()

    base_dir = cwd / "temp" / "failure_injection_integration"
    executor = LocalExecutor()
    run_opts = RunOptions(
        mode="until_stable",
        max_days=30,
        quiet_days=2,
        check_invariants="daily",
        default_handling="expel-agent",
        show_events="none",
    )

    # Injection 1: invalid scenario should fail with explicit failed status
    bad_result = executor.execute(
        scenario_config=_invalid_scenario_missing_agent(),
        run_id="bad_run",
        output_dir=base_dir / "bad_run",
        options=run_opts,
    )
    bad_is_failed = bad_result.status == RunStatus.FAILED
    bad_has_error = bool(bad_result.error)

    # Injection 2: mixed batch status accounting (one success, one failure)
    good_scenario = compile_ring_scenario(
        n_agents=18,
        kappa=Decimal("0.8"),
        concentration=Decimal("1.0"),
        mu=Decimal("0.5"),
        seed=505,
        maturity_days=6,
        name_prefix="failure-injection-good",
    )
    # LocalExecutor writes YAML with yaml.dump; normalize Decimals to plain values.
    good_scenario = json.loads(json.dumps(good_scenario, default=str))

    good_result = executor.execute(
        scenario_config=good_scenario,
        run_id="good_run",
        output_dir=base_dir / "good_run",
        options=run_opts,
    )

    batch_results = [good_result, bad_result]
    completed_count = sum(1 for r in batch_results if r.status == RunStatus.COMPLETED)
    failed_count = sum(1 for r in batch_results if r.status == RunStatus.FAILED)
    status_accounting_ok = completed_count == 1 and failed_count == 1

    # Injection 3: malformed event artifact must not silently return zero metrics
    malformed_path = base_dir / "malformed_events.jsonl"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text('{"kind": "PhaseA", "day": 0}\n{bad-json}\n', encoding="utf-8")

    malformed_rejected = False
    malformed_error: str | None = None
    try:
        compute_metrics_from_events(str(malformed_path))
    except Exception as exc:  # noqa: BLE001
        malformed_rejected = True
        malformed_error = f"{type(exc).__name__}: {exc}"

    # Injection 4: transient retry recovers, hard failure still propagates
    attempts = {"recover": 0, "hard": 0}

    @retry_transient(max_retries=3, base_delay=0.01, retryable=(ConnectionError,))
    def flaky_recover() -> str:
        attempts["recover"] += 1
        if attempts["recover"] < 3:
            raise ConnectionError("transient")
        return "ok"

    @retry_transient(max_retries=2, base_delay=0.01, retryable=(ConnectionError,))
    def always_fail() -> str:
        attempts["hard"] += 1
        raise ConnectionError("persistent")

    recovered = False
    hard_failure_propagated = False
    hard_failure_msg: str | None = None
    try:
        recovered = flaky_recover() == "ok"
    except Exception:
        recovered = False

    try:
        always_fail()
    except Exception as exc:  # noqa: BLE001
        hard_failure_propagated = True
        hard_failure_msg = f"{type(exc).__name__}: {exc}"

    # Scoring
    cat1 = 20.0 * (1.0 if bad_is_failed else 0.0)
    cat1 += 15.0 * (1.0 if bad_has_error else 0.0)

    cat2 = 25.0 * (1.0 if status_accounting_ok else 0.0)

    cat3 = 20.0 * (1.0 if malformed_rejected else 0.0)

    cat4 = 10.0 * (1.0 if recovered else 0.0)
    cat4 += 10.0 * (1.0 if hard_failure_propagated else 0.0)

    categories = [
        CategoryResult(
            name="Invalid Run Failure Semantics",
            max_points=35.0,
            earned_points=round(cat1, 3),
            details={
                "bad_status": str(bad_result.status),
                "bad_error": bad_result.error,
                "bad_artifacts": bad_result.artifacts,
            },
        ),
        CategoryResult(
            name="Mixed Batch Status Accounting",
            max_points=25.0,
            earned_points=round(cat2, 3),
            details={
                "completed_count": completed_count,
                "failed_count": failed_count,
                "status_accounting_ok": status_accounting_ok,
            },
        ),
        CategoryResult(
            name="Malformed Artifact Rejection",
            max_points=20.0,
            earned_points=round(cat3, 3),
            details={"malformed_rejected": malformed_rejected, "error": malformed_error},
        ),
        CategoryResult(
            name="Retry Resilience",
            max_points=20.0,
            earned_points=round(cat4, 3),
            details={
                "recover_attempts": attempts["recover"],
                "hard_attempts": attempts["hard"],
                "recovered": recovered,
                "hard_failure_propagated": hard_failure_propagated,
                "hard_failure_msg": hard_failure_msg,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="failure_injection::invalid_run_marked_failed",
            passed=(bad_is_failed and bad_has_error),
            message=f"bad_status={bad_result.status}, bad_error={bad_result.error}",
        ),
        CriticalCheck(
            code="failure_injection::status_accounting_consistent",
            passed=status_accounting_ok,
            message=f"completed={completed_count}, failed={failed_count}",
        ),
        CriticalCheck(
            code="failure_injection::malformed_events_not_silent",
            passed=malformed_rejected,
            message=(malformed_error or "malformed events were accepted unexpectedly"),
        ),
        CriticalCheck(
            code="failure_injection::retry_recovers_transient",
            passed=(recovered and hard_failure_propagated),
            message=(
                f"recover_attempts={attempts['recover']}, hard_attempts={attempts['hard']}, "
                f"hard_failure_propagated={hard_failure_propagated}"
            ),
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
        benchmark_name="Failure-Injection Integration Benchmark",
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
        title="Failure-Injection Integration Benchmark",
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
            f"bad_run_status={bad_result.status}",
            f"good_run_status={good_result.status}",
            f"status_accounting_ok={status_accounting_ok}",
            f"malformed_rejected={malformed_rejected}",
            f"recover_attempts={attempts['recover']}, hard_attempts={attempts['hard']}",
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Failure-injection benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
