#!/usr/bin/env python3
"""Local-vs-Cloud Parity Benchmark.

Runs one canonical scenario locally and compares metrics/artifact contracts
against the cloud metrics path implementation.
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
    lerp_score,
    report_dict,
    write_reports,
)
from bilancio.analysis.report import compute_day_metrics, compute_run_level_metrics, summarize_day_metrics
from bilancio.cloud.modal_app import compute_metrics_from_events
from bilancio.runners.local_executor import LocalExecutor
from bilancio.runners.models import RunOptions
from bilancio.storage.models import RunStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-vs-cloud parity benchmark.")
    parser.add_argument("--target-score", type=float, default=95.0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for metric parity (default: 1e-6)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="temp/local_cloud_parity_benchmark_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="temp/local_cloud_parity_benchmark_report.md",
    )
    return parser.parse_args()


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _local_metrics_from_events(events_path: Path) -> dict[str, Any]:
    events = _read_events(events_path)
    if not events:
        return {
            "delta_total": None,
            "phi_total": None,
            "time_to_stability": None,
            "max_G_t": None,
            "alpha_1": None,
            "Mpeak_1": None,
            "v_1": None,
            "HHIplus_1": None,
            "n_defaults": 0,
            "total_loss": 0,
            "total_loss_pct": None,
            "intermediary_loss_total": 0,
        }

    day_metrics = compute_day_metrics(events=events, balances_rows=None, day_list=None)
    summary = summarize_day_metrics(day_metrics["day_metrics"])
    run_level = compute_run_level_metrics(events)

    s_total = float(summary.get("S_total", 0) or 0)
    total_loss = float(run_level.get("total_loss", 0) or 0)

    return {
        "delta_total": summary.get("delta_total"),
        "phi_total": summary.get("phi_total"),
        "time_to_stability": int(summary.get("max_day") or 0),
        "max_G_t": summary.get("max_G_t"),
        "alpha_1": summary.get("alpha_1"),
        "Mpeak_1": summary.get("Mpeak_1"),
        "v_1": summary.get("v_1"),
        "HHIplus_1": summary.get("HHIplus_1"),
        "n_defaults": int(run_level.get("n_defaults", 0)),
        "total_loss": total_loss,
        "total_loss_pct": (total_loss / s_total) if s_total > 0 else None,
        "intermediary_loss_total": float(
            (run_level.get("nbfi_loan_loss", 0) or 0)
            + (run_level.get("bank_credit_loss", 0) or 0)
            + (run_level.get("cb_backstop_loss", 0) or 0)
        ),
    }


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    t0 = perf_counter()

    scenario = compile_ring_scenario(
        n_agents=24,
        kappa=Decimal("0.8"),
        concentration=Decimal("1.0"),
        mu=Decimal("0.5"),
        seed=2025,
        maturity_days=6,
        name_prefix="local-cloud-parity",
    )
    # LocalExecutor persists scenario YAML with plain yaml.dump; convert Decimals
    # to stringified primitives to avoid python/object tags in YAML.
    scenario_for_executor = json.loads(json.dumps(scenario, default=str))

    run_dir = cwd / "temp" / "local_cloud_parity" / "run_001"
    executor = LocalExecutor()
    result = executor.execute(
        scenario_config=scenario_for_executor,
        run_id="run_001",
        output_dir=run_dir,
        options=RunOptions(
            mode="until_stable",
            max_days=40,
            quiet_days=2,
            check_invariants="daily",
            default_handling="expel-agent",
            show_events="none",
        ),
    )

    local_ok = result.status == RunStatus.COMPLETED
    local_metrics: dict[str, Any] = {}
    cloud_metrics: dict[str, Any] = {}
    metric_deltas: dict[str, float] = {}

    required_artifacts = {"scenario_yaml", "events_jsonl", "balances_csv", "run_html"}
    local_artifacts = set(result.artifacts.keys())
    artifacts_ok = required_artifacts.issubset(local_artifacts)

    if local_ok and "events_jsonl" in result.artifacts:
        events_path = run_dir / result.artifacts["events_jsonl"]
        dealer_metrics_path = run_dir / "out" / "dealer_metrics.json"

        local_metrics = _local_metrics_from_events(events_path)
        cloud_metrics = compute_metrics_from_events(
            str(events_path),
            dealer_metrics_path=str(dealer_metrics_path) if dealer_metrics_path.exists() else None,
        )

        compare_keys = [
            "delta_total",
            "phi_total",
            "time_to_stability",
            "max_G_t",
            "alpha_1",
            "Mpeak_1",
            "v_1",
            "HHIplus_1",
            "n_defaults",
            "total_loss",
            "total_loss_pct",
            "intermediary_loss_total",
        ]
        for key in compare_keys:
            lv = _to_float(local_metrics.get(key))
            cv = _to_float(cloud_metrics.get(key))
            if lv is None and cv is None:
                metric_deltas[key] = 0.0
            elif lv is None or cv is None:
                metric_deltas[key] = float("inf")
            else:
                metric_deltas[key] = abs(lv - cv)

    max_delta = max(metric_deltas.values()) if metric_deltas else float("inf")

    required_metric_keys = {
        "delta_total",
        "phi_total",
        "time_to_stability",
        "max_G_t",
        "alpha_1",
        "Mpeak_1",
        "v_1",
        "HHIplus_1",
        "n_defaults",
        "total_loss",
        "total_loss_pct",
        "intermediary_loss_total",
    }
    metric_keys_ok = required_metric_keys.issubset(set(cloud_metrics.keys()))

    # Scoring
    cat1 = 20.0 if local_ok else 0.0
    cat2 = 20.0 if metric_keys_ok else 0.0
    cat3 = 50.0 * lerp_score(max_delta, 0.0, max(args.tolerance * 10.0, 1e-4), 1.0)
    cat4 = 10.0 if artifacts_ok else 0.0

    categories = [
        CategoryResult(
            name="Local Execution Success",
            max_points=20.0,
            earned_points=round(cat1, 3),
            details={"status": str(result.status), "error": result.error},
        ),
        CategoryResult(
            name="Required Metric Keys",
            max_points=20.0,
            earned_points=round(cat2, 3),
            details={
                "metric_keys_ok": metric_keys_ok,
                "required_metric_keys": sorted(required_metric_keys),
                "present_metric_keys": sorted(cloud_metrics.keys()),
            },
        ),
        CategoryResult(
            name="Metric Delta Parity",
            max_points=50.0,
            earned_points=round(cat3, 3),
            details={"max_delta": max_delta, "tolerance": args.tolerance, "deltas": metric_deltas},
        ),
        CategoryResult(
            name="Artifact Contract Parity",
            max_points=10.0,
            earned_points=round(cat4, 3),
            details={
                "artifacts_ok": artifacts_ok,
                "required_artifacts": sorted(required_artifacts),
                "present_artifacts": sorted(local_artifacts),
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="parity::local_execution_success",
            passed=local_ok,
            message=f"local status={result.status}, error={result.error}",
        ),
        CriticalCheck(
            code="parity::required_metric_keys_present",
            passed=metric_keys_ok,
            message=f"required keys present={metric_keys_ok}",
        ),
        CriticalCheck(
            code="parity::core_metrics_within_tolerance",
            passed=(max_delta <= args.tolerance),
            message=f"max_delta={max_delta:.8f}, tolerance={args.tolerance:.8f}",
        ),
        CriticalCheck(
            code="parity::artifact_keys_present",
            passed=artifacts_ok,
            message=f"artifact contract match={artifacts_ok}",
        ),
    ]

    total_score = bounded(sum(c.earned_points for c in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    critical_failures = [c for c in checks if not c.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"
    generated_at = generated_at_utc()

    report = report_dict(
        benchmark_name="Local-vs-Cloud Parity Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=(perf_counter() - t0),
        categories=categories,
        critical_checks=checks,
        extra={
            "generated_at_utc": generated_at,
            "details": {
                "run_dir": str(run_dir),
                "local_metrics": local_metrics,
                "cloud_metrics": cloud_metrics,
                "metric_deltas": metric_deltas,
            },
        },
    )

    md = build_markdown_report(
        title="Local-vs-Cloud Parity Benchmark",
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
            f"local_status={result.status}",
            f"metric_keys_ok={metric_keys_ok}",
            f"max_delta={max_delta}",
            f"artifacts_ok={artifacts_ok}",
        ],
    )

    write_reports(report, md, out_json, out_md)

    print(f"Local/cloud parity benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
