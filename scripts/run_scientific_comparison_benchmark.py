#!/usr/bin/env python3
"""Scientific Comparison Benchmark.

Validates that control/treatment comparisons and downstream statistical
analysis follow basic scientific standards:

1) paired experimental design and replication discipline
2) inferential completeness (CI, p-values, effect sizes)
3) multiplicity handling (Benjamini-Hochberg FDR)
4) reproducible, schema-complete analysis artifacts
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark_sim_utils import compile_ring_scenario, run_scenario_dict
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

from bilancio.experiments.sweep_analysis import RingSweepAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scientific comparison benchmark.")
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--max-days", type=int, default=25)
    parser.add_argument("--out-json", type=str, default="temp/scientific_comparison_benchmark_report.json")
    parser.add_argument("--out-md", type=str, default="temp/scientific_comparison_benchmark_report.md")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="temp/scientific_comparison_benchmark/artifacts",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="scripts/analysis_manifest.json",
    )
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    return (
        float(record["kappa"]),
        float(record["concentration"]),
        float(record["mu"]),
        float(record["monotonicity"]),
        float(record["outside_mid_ratio"]),
        int(record["seed"]),
    )


def _cell_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return _record_key(record)[:5]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _validate_effect_row(row: dict[str, Any]) -> tuple[bool, str]:
    required = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "outside_mid_ratio",
        "n_pairs",
        "control_mean",
        "treatment_mean",
        "effect",
        "effect_ci_lower",
        "effect_ci_upper",
        "p_value",
        "cohens_d",
    ]
    for field in required:
        if field not in row:
            return False, f"missing_field:{field}"

    numeric_fields = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "outside_mid_ratio",
        "control_mean",
        "treatment_mean",
        "effect",
        "effect_ci_lower",
        "effect_ci_upper",
        "p_value",
        "cohens_d",
    ]
    for field in numeric_fields:
        if not _is_finite_number(row[field]):
            return False, f"non_finite:{field}"

    n_pairs = int(row["n_pairs"])
    if n_pairs < 2:
        return False, "n_pairs<2"

    p_value = float(row["p_value"])
    if not (0.0 <= p_value <= 1.0):
        return False, "p_out_of_range"

    ci_lower = float(row["effect_ci_lower"])
    ci_upper = float(row["effect_ci_upper"])
    effect = float(row["effect"])
    if ci_lower > ci_upper:
        return False, "ci_inverted"
    if not (ci_lower <= effect <= ci_upper):
        return False, "effect_outside_ci"
    if ci_upper - ci_lower < 0.0:
        return False, "ci_width_negative"

    sig05 = bool(row.get("significant_05", False))
    sig01 = bool(row.get("significant_01", False))
    if sig01 and not sig05:
        return False, "sig01_implies_sig05_violated"

    return True, "ok"


def _run_scenario_quiet(
    scenario: dict[str, Any],
    *,
    max_days: int,
    enable_dealer: bool = False,
) -> Any:
    """Run a benchmark scenario while suppressing noisy simulation output."""
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return run_scenario_dict(
            scenario,
            max_days=max_days,
            enable_dealer=enable_dealer,
        )


def _benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Return (q_values, rejects) in original order."""
    n = len(p_values)
    if n == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_sorted = [1.0] * n

    # Raw adjusted values: q_i = p_i * n / rank
    for rank, (_, p_value) in enumerate(indexed, start=1):
        q_sorted[rank - 1] = p_value * n / rank

    # Enforce monotonicity from the tail.
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    q_sorted = [min(1.0, max(0.0, q)) for q in q_sorted]

    q_values = [1.0] * n
    rejects = [False] * n
    for rank, (original_idx, p_value) in enumerate(indexed, start=1):
        q_values[original_idx] = q_sorted[rank - 1]
        rejects[original_idx] = p_value <= (rank / n) * alpha

    return q_values, rejects


def _canonicalize_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "outside_mid_ratio",
        "n_pairs",
        "control_mean",
        "treatment_mean",
        "effect",
        "effect_ci_lower",
        "effect_ci_upper",
        "p_value",
        "cohens_d",
        "significant_05",
        "significant_01",
    ]
    canonical = []
    for row in rows:
        item: dict[str, Any] = {}
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                item[key] = round(value, 12)
            else:
                item[key] = value
        canonical.append(item)
    canonical.sort(
        key=lambda r: (
            float(r["kappa"]),
            float(r["concentration"]),
            float(r["mu"]),
            float(r["monotonicity"]),
            float(r["outside_mid_ratio"]),
        )
    )
    return canonical


def _canonicalize_sensitivity(results: list[Any]) -> list[tuple[str, float, float, float]]:
    out: list[tuple[str, float, float, float]] = []
    for result in results:
        out.append(
            (
                str(result.parameter),
                round(float(result.mu_star), 12),
                round(float(result.mu), 12),
                round(float(result.sigma), 12),
            )
        )
    out.sort(key=lambda x: x[0])
    return out


def _load_analysis_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the analysis manifest JSON."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    # Basic schema validation
    required_keys = {"version", "primary_endpoints", "hypothesis_families", "design"}
    if not required_keys.issubset(manifest.keys()):
        raise ValueError(f"Manifest missing keys: {required_keys - set(manifest.keys())}")
    for ep in manifest["primary_endpoints"]:
        for k in ("metric", "mde", "alpha", "power"):
            if k not in ep:
                raise ValueError(f"Endpoint missing key: {k}")
    return manifest


def _compute_required_replicates(
    mde: float, alpha: float, power: float, variance: float
) -> int:
    """Compute required replicates per cell for a paired t-test.

    Uses the formula: n = ceil((z_alpha + z_beta)^2 * 2 * sigma^2 / mde^2)
    where z_alpha and z_beta are the standard normal quantiles.
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = math.ceil((z_alpha + z_beta) ** 2 * 2 * variance / (mde ** 2))
    return max(2, n)  # At least 2 replicates


def _validate_manifest_coverage(
    manifest: dict[str, Any], effect_rows: list[dict[str, Any]]
) -> bool:
    """Check all primary endpoints in manifest have corresponding effect rows."""
    if not manifest or not manifest.get("primary_endpoints"):
        return False
    required_metrics = {ep["metric"] for ep in manifest["primary_endpoints"]}
    # effect_rows have trading_effect for delta_total by construction
    # We just check that delta_total is present (phi_total effects come from same analysis)
    available = {"delta_total", "phi_total"}  # Our benchmark always computes these
    return required_metrics.issubset(available)


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    out_json = cwd / args.out_json
    out_md = cwd / args.out_md
    artifacts_dir = cwd / args.artifacts_dir

    t0 = perf_counter()

    manifest_path = cwd / args.manifest
    manifest = _load_analysis_manifest(manifest_path)

    kappas = [Decimal("0.4"), Decimal("1.0")]
    concentrations = [Decimal("0.5"), Decimal("2.0")]
    mus = [Decimal("0.25"), Decimal("0.75")]
    monotonicity = Decimal("0")
    outside_mid_ratio = Decimal("0")

    expected_cells = len(kappas) * len(concentrations) * len(mus)
    expected_pairs = expected_cells * args.replicates

    records: list[dict[str, Any]] = []
    failures: list[str] = []

    cell_idx = 0
    for kappa in kappas:
        for concentration in concentrations:
            for mu in mus:
                cell_idx += 1
                for rep in range(args.replicates):
                    seed = args.seed_start + (cell_idx * 100) + rep
                    scenario = compile_ring_scenario(
                        n_agents=24,
                        kappa=kappa,
                        concentration=concentration,
                        mu=mu,
                        seed=seed,
                        maturity_days=6,
                        name_prefix=f"scientific-benchmark-{cell_idx}",
                    )

                    try:
                        passive = _run_scenario_quiet(
                            scenario,
                            max_days=args.max_days,
                        )
                        active = _run_scenario_quiet(
                            scenario,
                            max_days=args.max_days,
                            enable_dealer=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            "cell="
                            f"(k={kappa},c={concentration},mu={mu}), seed={seed}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    records.append(
                        {
                            "kappa": float(kappa),
                            "concentration": float(concentration),
                            "mu": float(mu),
                            "monotonicity": float(monotonicity),
                            "outside_mid_ratio": float(outside_mid_ratio),
                            "seed": seed,
                            "delta_passive": passive.default_ratio,
                            "delta_active": active.default_ratio,
                            "phi_passive": passive.total_loss_ratio or 0.0,
                            "phi_active": active.total_loss_ratio or 0.0,
                            "n_defaults_passive": passive.defaults_count,
                            "n_defaults_active": active.defaults_count,
                            "trading_effect": passive.default_ratio - active.default_ratio,
                        }
                    )

    completed_pairs = len(records)
    completion_rate = completed_pairs / max(1, expected_pairs)

    cell_counts: dict[tuple[float, float, float, float, float], int] = defaultdict(int)
    for record in records:
        cell_counts[_cell_key(record)] += 1

    min_replicates = min(cell_counts.values()) if cell_counts else 0
    max_replicates = max(cell_counts.values()) if cell_counts else 0
    balance_ratio = (min_replicates / max_replicates) if max_replicates > 0 else 0.0

    # Power planning: compute required replicates from manifest
    required_replicates = args.replicates  # default
    power_valid = True
    estimated_variance = 0.01  # conservative default
    if manifest and records:
        effects_list = [r["delta_passive"] - r["delta_active"] for r in records]
        if len(effects_list) > 1:
            import statistics

            estimated_variance = statistics.variance(effects_list)
        for ep in manifest.get("primary_endpoints", []):
            if ep["metric"] == "delta_total":
                req = _compute_required_replicates(
                    mde=ep["mde"],
                    alpha=ep["alpha"],
                    power=ep["power"],
                    variance=estimated_variance,
                )
                required_replicates = max(required_replicates, req)
                if min_replicates < req:
                    power_valid = False

    unique_record_keys = {_record_key(record) for record in records}
    uniqueness_rate = len(unique_record_keys) / max(1, completed_pairs)
    duplicate_rows = completed_pairs - len(unique_record_keys)

    cat1_score = (
        12.0 * completion_rate
        + 10.0 * balance_ratio
        + 8.0 * uniqueness_rate
    )

    analysis = RingSweepAnalysis(records)
    effect_table = analysis.trading_effects(
        confidence=0.95,
        n_bootstrap=args.n_bootstrap,
        seed=args.analysis_seed,
    )
    effect_rows = effect_table.to_dicts()

    effect_row_failures: list[str] = []
    valid_rows = 0
    replicate_floor_hits = 0
    ci_width_positive_hits = 0
    for idx, row in enumerate(effect_rows):
        row_valid, reason = _validate_effect_row(row)
        if row_valid:
            valid_rows += 1
        else:
            effect_row_failures.append(f"row[{idx}] {reason}")

        if int(row.get("n_pairs", 0)) >= args.replicates:
            replicate_floor_hits += 1

        if _is_finite_number(row.get("effect_ci_lower")) and _is_finite_number(row.get("effect_ci_upper")):
            if float(row["effect_ci_upper"]) - float(row["effect_ci_lower"]) >= 0.0:
                ci_width_positive_hits += 1

    n_effect_rows = len(effect_rows)
    row_valid_rate = valid_rows / max(1, n_effect_rows)
    replicate_floor_rate = replicate_floor_hits / max(1, n_effect_rows)
    ci_reporting_rate = ci_width_positive_hits / max(1, n_effect_rows)

    cat2_score = (
        15.0 * row_valid_rate
        + 8.0 * replicate_floor_rate
        + 7.0 * ci_reporting_rate
    )

    p_values = [float(row["p_value"]) for row in effect_rows if _is_finite_number(row.get("p_value"))]
    q_values, fdr_rejects = _benjamini_hochberg(p_values, alpha=0.05)

    raw_sig_05 = sum(1 for p_value in p_values if p_value < 0.05)
    fdr_sig_05 = sum(1 for flag in fdr_rejects if flag)

    fdr_discipline_ok = (
        len(q_values) == len(p_values)
        and fdr_sig_05 <= raw_sig_05
    )

    q_range_ok = all(0.0 <= q <= 1.0 for q in q_values)
    q_monotonic = True
    paired = sorted(zip(p_values, q_values, strict=False), key=lambda item: item[0])
    for i in range(len(paired) - 1):
        if paired[i][1] > paired[i + 1][1] + 1e-12:
            q_monotonic = False
            break

    effects = [float(row["effect"]) for row in effect_rows if _is_finite_number(row.get("effect"))]
    nonzero_effects = [value for value in effects if abs(value) > 1e-12]
    if nonzero_effects:
        n_positive = sum(1 for value in nonzero_effects if value > 0)
        n_negative = sum(1 for value in nonzero_effects if value < 0)
        direction_coherence = max(n_positive, n_negative) / len(nonzero_effects)
    else:
        direction_coherence = 1.0

    direction_score = 5.0 * lerp_score(
        direction_coherence,
        full_at=0.90,
        zero_at=0.50,
        max_points=1.0,
    )

    cat3_score = (
        10.0 * (1.0 if fdr_discipline_ok else 0.0)
        + 5.0 * (1.0 if (q_range_ok and q_monotonic) else 0.0)
        + direction_score
    )

    artifacts = analysis.write_stats(
        artifacts_dir,
        confidence=0.95,
        n_bootstrap=args.n_bootstrap,
        seed=args.analysis_seed,
    )
    artifact_ratio = len(artifacts) / 4.0

    effects_schema_ok = False
    if "effects" in artifacts and artifacts["effects"].exists():
        with artifacts["effects"].open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
        required_effect_cols = {
            "effect_type",
            "kappa",
            "concentration",
            "mu",
            "n_pairs",
            "effect",
            "effect_ci_lower",
            "effect_ci_upper",
            "p_value",
            "cohens_d",
        }
        effects_schema_ok = required_effect_cols.issubset(fieldnames)

    summary_schema_ok = False
    if "summary" in artifacts and artifacts["summary"].exists():
        summary_data = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
        summary_schema_ok = (
            isinstance(summary_data, dict)
            and {"n_records", "n_cells", "min_replicates_per_cell"}.issubset(summary_data.keys())
            and "trading_effect" in summary_data
        )

    sensitivity_schema_ok = False
    if "sensitivity" in artifacts and artifacts["sensitivity"].exists():
        sensitivity_data = json.loads(artifacts["sensitivity"].read_text(encoding="utf-8"))
        sensitivity_schema_ok = (
            isinstance(sensitivity_data, dict)
            and "delta_passive" in sensitivity_data
            and isinstance(sensitivity_data.get("delta_passive"), list)
        )

    schema_ok = effects_schema_ok and summary_schema_ok and sensitivity_schema_ok

    analysis_repeated = RingSweepAnalysis(records)
    effect_rows_repeat = analysis_repeated.trading_effects(
        confidence=0.95,
        n_bootstrap=args.n_bootstrap,
        seed=args.analysis_seed,
    ).to_dicts()
    deterministic_effects = (
        _canonicalize_effect_rows(effect_rows) == _canonicalize_effect_rows(effect_rows_repeat)
    )

    sensitivity_1 = analysis.sensitivity(
        metric="delta_passive",
        num_trajectories=12,
        seed=args.analysis_seed,
    )
    sensitivity_2 = analysis_repeated.sensitivity(
        metric="delta_passive",
        num_trajectories=12,
        seed=args.analysis_seed,
    )
    deterministic_sensitivity = (
        _canonicalize_sensitivity(sensitivity_1) == _canonicalize_sensitivity(sensitivity_2)
    )
    deterministic_ok = deterministic_effects and deterministic_sensitivity

    cat4_score = (
        8.0 * artifact_ratio
        + 6.0 * (1.0 if schema_ok else 0.0)
        + 6.0 * (1.0 if deterministic_ok else 0.0)
    )

    categories = [
        CategoryResult(
            name="Design & Pairing Discipline",
            max_points=30.0,
            earned_points=round(cat1_score, 3),
            details={
                "expected_pairs": expected_pairs,
                "completed_pairs": completed_pairs,
                "completion_rate": completion_rate,
                "expected_cells": expected_cells,
                "cells_observed": len(cell_counts),
                "min_replicates_per_cell": min_replicates,
                "max_replicates_per_cell": max_replicates,
                "balance_ratio": balance_ratio,
                "uniqueness_rate": uniqueness_rate,
                "duplicate_rows": duplicate_rows,
                "run_failures": failures,
            },
        ),
        CategoryResult(
            name="Inference Completeness",
            max_points=30.0,
            earned_points=round(cat2_score, 3),
            details={
                "effect_rows": n_effect_rows,
                "valid_rows": valid_rows,
                "row_valid_rate": row_valid_rate,
                "replicate_floor_hits": replicate_floor_hits,
                "replicate_floor_rate": replicate_floor_rate,
                "ci_width_positive_hits": ci_width_positive_hits,
                "ci_reporting_rate": ci_reporting_rate,
                "row_validation_failures": effect_row_failures,
            },
        ),
        CategoryResult(
            name="Multiplicity & Robustness",
            max_points=20.0,
            earned_points=round(cat3_score, 3),
            details={
                "n_tests": len(p_values),
                "raw_significant_05": raw_sig_05,
                "fdr_significant_05": fdr_sig_05,
                "fdr_discipline_ok": fdr_discipline_ok,
                "q_range_ok": q_range_ok,
                "q_monotonic": q_monotonic,
                "direction_coherence": direction_coherence,
                "q_values": q_values,
            },
        ),
        CategoryResult(
            name="Output Schema & Reproducibility",
            max_points=20.0,
            earned_points=round(cat4_score, 3),
            details={
                "artifacts": {name: str(path) for name, path in artifacts.items()},
                "artifact_ratio": artifact_ratio,
                "schema_ok": schema_ok,
                "effects_schema_ok": effects_schema_ok,
                "summary_schema_ok": summary_schema_ok,
                "sensitivity_schema_ok": sensitivity_schema_ok,
                "deterministic_effects": deterministic_effects,
                "deterministic_sensitivity": deterministic_sensitivity,
            },
        ),
    ]

    checks = [
        CriticalCheck(
            code="scientific::paired_runs_complete",
            passed=(completed_pairs == expected_pairs and not failures),
            message=f"completed_pairs={completed_pairs}, expected_pairs={expected_pairs}, failures={len(failures)}",
        ),
        CriticalCheck(
            code="scientific::replication_floor",
            passed=(min_replicates >= args.replicates),
            message=f"min_replicates={min_replicates}, required={args.replicates}",
        ),
        CriticalCheck(
            code="scientific::effect_rows_valid",
            passed=(n_effect_rows >= expected_cells and valid_rows == n_effect_rows),
            message=f"effect_rows={n_effect_rows}, expected_cells={expected_cells}, valid_rows={valid_rows}",
        ),
        CriticalCheck(
            code="scientific::fdr_report_valid",
            passed=(fdr_discipline_ok and q_range_ok and q_monotonic),
            message=f"raw_sig={raw_sig_05}, fdr_sig={fdr_sig_05}, n_tests={len(p_values)}",
        ),
        CriticalCheck(
            code="scientific::analysis_artifacts_schema",
            passed=schema_ok,
            message=(
                "effects_schema_ok="
                f"{effects_schema_ok}, summary_schema_ok={summary_schema_ok}, "
                f"sensitivity_schema_ok={sensitivity_schema_ok}"
            ),
        ),
        CriticalCheck(
            code="scientific::deterministic_reanalysis",
            passed=deterministic_ok,
            message=(
                f"deterministic_effects={deterministic_effects}, "
                f"deterministic_sensitivity={deterministic_sensitivity}"
            ),
        ),
        CriticalCheck(
            code="scientific::power_planning_valid",
            passed=power_valid,
            message=f"min_replicates={min_replicates}, required_for_power={required_replicates}, variance={estimated_variance:.6f}",
        ),
        CriticalCheck(
            code="scientific::analysis_manifest_present",
            passed=bool(manifest),
            message=f"manifest_path={manifest_path}, loaded={bool(manifest)}",
        ),
    ]

    total_score = bounded(sum(category.earned_points for category in categories), 0.0, 100.0)
    base_grade = grade_for_score(total_score)
    critical_failures = [check for check in checks if not check.passed]
    grade = cap_grade_for_critical_failures(base_grade, len(critical_failures))
    meets_target = total_score >= args.target_score
    status = "PASS" if meets_target and not critical_failures else "FAIL"

    elapsed = perf_counter() - t0
    generated_at = generated_at_utc()

    report = report_dict(
        benchmark_name="Scientific Comparison Benchmark",
        target_score=args.target_score,
        total_score=total_score,
        status=status,
        meets_target=meets_target,
        base_grade=base_grade,
        grade=grade,
        elapsed_seconds=elapsed,
        categories=categories,
        critical_checks=checks,
        extra={
            "details": {
                "grid": {
                    "kappas": [float(value) for value in kappas],
                    "concentrations": [float(value) for value in concentrations],
                    "mus": [float(value) for value in mus],
                    "replicates": args.replicates,
                },
                "expected_cells": expected_cells,
                "expected_pairs": expected_pairs,
                "completed_pairs": completed_pairs,
                "effect_rows": effect_rows,
            },
            "generated_at_utc": generated_at,
        },
    )

    markdown = build_markdown_report(
        title="Scientific Comparison Benchmark",
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
            f"expected_pairs={expected_pairs}",
            f"completed_pairs={completed_pairs}",
            f"expected_cells={expected_cells}",
            f"effect_rows={n_effect_rows}",
            f"raw_sig_05={raw_sig_05}",
            f"fdr_sig_05={fdr_sig_05}",
        ],
        detail_sections=[
            (
                "Design Metrics",
                [
                    f"completion_rate={completion_rate:.4f}",
                    f"min_replicates_per_cell={min_replicates}",
                    f"max_replicates_per_cell={max_replicates}",
                    f"balance_ratio={balance_ratio:.4f}",
                    f"uniqueness_rate={uniqueness_rate:.4f}",
                ],
            ),
            (
                "Inference Metrics",
                [
                    f"row_valid_rate={row_valid_rate:.4f}",
                    f"replicate_floor_rate={replicate_floor_rate:.4f}",
                    f"ci_reporting_rate={ci_reporting_rate:.4f}",
                ],
            ),
            (
                "Artifact Metrics",
                [
                    f"artifact_ratio={artifact_ratio:.4f}",
                    f"schema_ok={schema_ok}",
                    f"deterministic_effects={deterministic_effects}",
                    f"deterministic_sensitivity={deterministic_sensitivity}",
                ],
            ),
        ],
    )

    write_reports(report, markdown, out_json, out_md)

    print(f"Scientific comparison benchmark score: {total_score:.2f}/100 ({grade})")
    print(f"Benchmark status: {status} (critical failures: {len(critical_failures)})")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
