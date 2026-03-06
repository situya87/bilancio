#!/usr/bin/env python3
"""Sequential tuning for dealer-only, bank-only, and NBFI-only support runs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.analysis.intermediary_frontier import (
    build_frontier_artifact,
    write_frontier_artifact,
)
from bilancio.analysis.intermediary_support_tuning import (
    FrontierScore,
    TuningCandidate,
    bank_candidates,
    choose_best_candidate,
    dealer_candidates,
    nbfi_candidates,
    score_frontier_artifact,
)
from bilancio.experiments.balanced_comparison import (
    BalancedComparisonConfig,
    BalancedComparisonRunner,
)
from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner
from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    label: str
    treatment_arm: str
    config_cls: type[Any]
    runner_cls: type[Any]
    candidates: tuple[TuningCandidate, ...]


@dataclass(frozen=True)
class CandidateRun:
    stage: str
    mechanism: str
    candidate: TuningCandidate
    experiment_dir: Path
    frontier_dir: Path
    score: FrontierScore


MECHANISMS: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        name="dealer",
        label="Dealer-Only",
        treatment_arm="active",
        config_cls=BalancedComparisonConfig,
        runner_cls=BalancedComparisonRunner,
        candidates=dealer_candidates(),
    ),
    MechanismSpec(
        name="bank",
        label="Bank-Only",
        treatment_arm="bank_lend",
        config_cls=BankComparisonConfig,
        runner_cls=BankComparisonRunner,
        candidates=bank_candidates(),
    ),
    MechanismSpec(
        name="nbfi",
        label="NBFI-Only",
        treatment_arm="nbfi_lend",
        config_cls=NBFIComparisonConfig,
        runner_cls=NBFIComparisonRunner,
        candidates=nbfi_candidates(),
    ),
)


def _decimal_list(raw: str) -> list[Decimal]:
    return [Decimal(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def _common_grid_kwargs(
    *,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> dict[str, Any]:
    return {
        "n_agents": n_agents,
        "maturity_days": 10,
        "Q_total": Decimal("10000"),
        "liquidity_mode": "uniform",
        "base_seed": base_seed,
        "n_replicates": n_replicates,
        "quiet": True,
        "default_handling": "expel-agent",
        "rollover_enabled": True,
        "kappas": kappas,
        "concentrations": concentrations,
        "mus": mus,
        "monotonicities": [Decimal("0")],
        "outside_mid_ratios": outside_mid_ratios,
    }


def _dealer_config_kwargs(
    *,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> dict[str, Any]:
    kwargs = _common_grid_kwargs(
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=kappas,
        concentrations=concentrations,
        mus=mus,
        outside_mid_ratios=outside_mid_ratios,
        n_agents=n_agents,
    )
    kwargs.update(
        {
            "max_simulation_days": 15,
            "enable_lender": False,
            "enable_dealer_lender": False,
            "enable_bank_passive": False,
            "enable_bank_dealer": False,
            "enable_bank_dealer_nbfi": False,
            "trading_rounds": 80,
            "performance": {"preset": "fast"},
        }
    )
    return kwargs


def _bank_config_kwargs(
    *,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> dict[str, Any]:
    kwargs = _common_grid_kwargs(
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=kappas,
        concentrations=concentrations,
        mus=mus,
        outside_mid_ratios=outside_mid_ratios,
        n_agents=n_agents,
    )
    kwargs.update(
        {
            "n_banks": 5,
            "reserve_ratio": Decimal("0.50"),
            "trading_rounds": 80,
            "performance": {"preset": "fast"},
        }
    )
    return kwargs


def _nbfi_config_kwargs(
    *,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> dict[str, Any]:
    kwargs = _common_grid_kwargs(
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=kappas,
        concentrations=concentrations,
        mus=mus,
        outside_mid_ratios=outside_mid_ratios,
        n_agents=n_agents,
    )
    kwargs.update(
        {
            "trading_rounds": 80,
            "performance": {"preset": "fast"},
        }
    )
    return kwargs


def _config_kwargs_for_mechanism(
    mechanism: str,
    *,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> dict[str, Any]:
    if mechanism == "dealer":
        return _dealer_config_kwargs(
            base_seed=base_seed,
            n_replicates=n_replicates,
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            outside_mid_ratios=outside_mid_ratios,
            n_agents=n_agents,
        )
    if mechanism == "bank":
        return _bank_config_kwargs(
            base_seed=base_seed,
            n_replicates=n_replicates,
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            outside_mid_ratios=outside_mid_ratios,
            n_agents=n_agents,
        )
    if mechanism == "nbfi":
        return _nbfi_config_kwargs(
            base_seed=base_seed,
            n_replicates=n_replicates,
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            outside_mid_ratios=outside_mid_ratios,
            n_agents=n_agents,
        )
    raise ValueError(f"Unsupported mechanism: {mechanism}")


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_overrides(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "No override beat the baseline."
    lines = []
    for key in sorted(overrides):
        value = overrides[key]
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows_"
    headers = list(rows[0].keys())
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(out)


def _candidate_rows(candidates: tuple[TuningCandidate, ...]) -> list[dict[str, str]]:
    return [
        {
            "Candidate": candidate.name,
            "Rationale": candidate.rationale,
        }
        for candidate in candidates
    ]


def _score_rows(runs: list[CandidateRun], *, root_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        score = run.score
        rows.append(
            {
                "Candidate": run.candidate.name,
                "Complete": f"{score.n_complete}/{score.n_pairs}",
                "Safe Help Share": _format_float(score.share_safe_help),
                "Net+ Help Share": _format_float(score.share_help_nonnegative_net_system),
                "Loss/Help": _format_float(score.mean_loss_per_help),
                "Mean Safe Relief": _format_float(score.mean_safe_default_relief),
                "Mean Net Relief": _format_float(score.mean_net_system_relief),
                "Output": f"`{_relative(run.experiment_dir, root_dir)}`",
            }
        )
    return rows


def _run_candidate(
    spec: MechanismSpec,
    *,
    stage: str,
    candidate: TuningCandidate,
    stage_root: Path,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
) -> CandidateRun:
    experiment_dir = stage_root / spec.name / candidate.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = _config_kwargs_for_mechanism(
        spec.name,
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=kappas,
        concentrations=concentrations,
        mus=mus,
        outside_mid_ratios=outside_mid_ratios,
        n_agents=n_agents,
    )
    config_kwargs.update(candidate.overrides)
    config_kwargs["name_prefix"] = f"Task 4 {spec.label} {candidate.name}"

    config = spec.config_cls(**config_kwargs)
    runner = spec.runner_cls(config=config, out_dir=experiment_dir, enable_supabase=False)
    run_log = experiment_dir / "run.log"
    with run_log.open("a", encoding="utf-8") as log_handle, redirect_stdout(log_handle), redirect_stderr(log_handle):
        runner.run_all()

    artifact = build_frontier_artifact(experiment_dir)
    frontier_dir = experiment_dir / "aggregate" / "intermediary_frontier"
    write_frontier_artifact(artifact, frontier_dir)
    score = score_frontier_artifact(artifact, treatment_arm=spec.treatment_arm)
    print(
        f"[{stage}] {spec.label}: {candidate.name} completed "
        f"(safe_help={_format_float(score.share_safe_help)}, loss/help={_format_float(score.mean_loss_per_help)})",
        flush=True,
    )
    return CandidateRun(
        stage=stage,
        mechanism=spec.name,
        candidate=candidate,
        experiment_dir=experiment_dir,
        frontier_dir=frontier_dir,
        score=score,
    )


def _run_stage(
    spec: MechanismSpec,
    *,
    stage: str,
    stage_root: Path,
    base_seed: int,
    n_replicates: int,
    kappas: list[Decimal],
    concentrations: list[Decimal],
    mus: list[Decimal],
    outside_mid_ratios: list[Decimal],
    n_agents: int,
    candidates: tuple[TuningCandidate, ...],
) -> list[CandidateRun]:
    results: list[CandidateRun] = []
    for candidate in candidates:
        print(
            f"[{stage}] {spec.label}: {candidate.name} "
            f"(seed={base_seed}, reps={n_replicates})",
            flush=True,
        )
        results.append(
            _run_candidate(
                spec,
                stage=stage,
                candidate=candidate,
                stage_root=stage_root,
                base_seed=base_seed,
                n_replicates=n_replicates,
                kappas=kappas,
                concentrations=concentrations,
                mus=mus,
                outside_mid_ratios=outside_mid_ratios,
                n_agents=n_agents,
            )
        )
    return results


def _build_report(
    *,
    out_dir: Path,
    discovery_runs: dict[str, list[CandidateRun]],
    validation_runs: dict[str, list[CandidateRun]],
    final_choices: dict[str, CandidateRun],
    run_config: dict[str, str],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Intermediary Support Tuning Report",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This report tunes support behavior sequentially for dealer-only, bank-only, and NBFI-only comparisons.",
        (
            "The ranking objective is lexicographic: maximize safe help, maximize help "
            "with non-negative net system relief, minimize loss per unit of help, then "
            "maximize safe default relief."
        ),
        "",
        "## Run Envelope",
        "",
        *[f"- `{key}`: `{value}`" for key, value in run_config.items()],
        "",
    ]

    for spec in MECHANISMS:
        final_run = final_choices[spec.name]
        discovery = discovery_runs[spec.name]
        validation = validation_runs[spec.name]
        lines.extend(
            [
                f"## {spec.label}",
                "",
                "### Candidate Set",
                "",
                _markdown_table(_candidate_rows(spec.candidates)),
                "",
                "### Discovery Sweep",
                "",
                _markdown_table(_score_rows(discovery, root_dir=out_dir)),
                "",
                "### Held-Out Validation",
                "",
                _markdown_table(_score_rows(validation, root_dir=out_dir)),
                "",
                f"### Final Recommendation: `{final_run.candidate.name}`",
                "",
                final_run.candidate.rationale,
                "",
                "Recommended overrides:",
                _format_overrides(final_run.candidate.overrides),
                "",
                f"Frontier artifacts: `{_relative(final_run.frontier_dir, out_dir)}`",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/intermediary_support_tuning"),
        help="Output directory for sweeps and frontier artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/reports/intermediary_support_tuning_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=60,
        help="Number of agents for each sweep run.",
    )
    parser.add_argument(
        "--discovery-reps",
        type=int,
        default=2,
        help="Replicates per discovery candidate.",
    )
    parser.add_argument(
        "--validation-reps",
        type=int,
        default=2,
        help="Replicates for held-out validation.",
    )
    parser.add_argument(
        "--discovery-base-seed",
        type=int,
        default=4100,
        help="Base seed for discovery sweeps.",
    )
    parser.add_argument(
        "--validation-base-seed",
        type=int,
        default=9100,
        help="Base seed for held-out validation sweeps.",
    )
    parser.add_argument(
        "--kappas",
        type=str,
        default="0.5,1,2",
        help="Comma-separated kappa values.",
    )
    parser.add_argument(
        "--concentrations",
        type=str,
        default="0.5,1.5",
        help="Comma-separated concentration values.",
    )
    parser.add_argument(
        "--mus",
        type=str,
        default="0,0.5",
        help="Comma-separated mu values.",
    )
    parser.add_argument(
        "--outside-mid-ratios",
        type=str,
        default="0.90",
        help="Comma-separated outside/mid ratios.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kappas = _decimal_list(args.kappas)
    concentrations = _decimal_list(args.concentrations)
    mus = _decimal_list(args.mus)
    outside_mid_ratios = _decimal_list(args.outside_mid_ratios)

    discovery_root = out_dir / "discovery"
    validation_root = out_dir / "validation"

    discovery_runs: dict[str, list[CandidateRun]] = {}
    validation_runs: dict[str, list[CandidateRun]] = {}
    final_choices: dict[str, CandidateRun] = {}

    for index, spec in enumerate(MECHANISMS):
        discovery_seed = args.discovery_base_seed + index * 1000
        validation_seed = args.validation_base_seed + index * 1000

        discovery = _run_stage(
            spec,
            stage="discovery",
            stage_root=discovery_root,
            base_seed=discovery_seed,
            n_replicates=args.discovery_reps,
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            outside_mid_ratios=outside_mid_ratios,
            n_agents=args.n_agents,
            candidates=spec.candidates,
        )
        discovery_runs[spec.name] = discovery

        discovery_winner, _ = choose_best_candidate(
            [(run.candidate, run.score) for run in discovery]
        )

        validation_candidates: list[TuningCandidate] = [spec.candidates[0]]
        if discovery_winner.name != spec.candidates[0].name:
            validation_candidates.append(discovery_winner)

        validation = _run_stage(
            spec,
            stage="validation",
            stage_root=validation_root,
            base_seed=validation_seed,
            n_replicates=args.validation_reps,
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            outside_mid_ratios=outside_mid_ratios,
            n_agents=args.n_agents,
            candidates=tuple(validation_candidates),
        )
        validation_runs[spec.name] = validation

        winner_candidate, _ = choose_best_candidate(
            [(run.candidate, run.score) for run in validation]
        )
        final_run = next(run for run in validation if run.candidate.name == winner_candidate.name)
        final_choices[spec.name] = final_run

    report = _build_report(
        out_dir=out_dir,
        discovery_runs=discovery_runs,
        validation_runs=validation_runs,
        final_choices=final_choices,
        run_config={
            "n_agents": str(args.n_agents),
            "discovery_reps": str(args.discovery_reps),
            "validation_reps": str(args.validation_reps),
            "discovery_base_seed": str(args.discovery_base_seed),
            "validation_base_seed": str(args.validation_base_seed),
            "kappas": args.kappas,
            "concentrations": args.concentrations,
            "mus": args.mus,
            "outside_mid_ratios": args.outside_mid_ratios,
        },
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
