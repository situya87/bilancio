"""Frontier and loss-floor analysis for intermediary efficiency sweeps.

This module computes per-run treatment-vs-control comparisons on the
scientific frontier:

1. default_relief = delta_control - delta_treatment
2. inst_loss_shift = intermediary_loss_treatment - intermediary_loss_control
3. net_system_relief = system_loss_control - system_loss_treatment

It supports:
- Pareto labeling per paired run
- Regime/slice summaries
- Empirical structural loss-floor tables by default-relief bands
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path
from typing import Any

import yaml

from bilancio.analysis.report import compute_intermediary_losses, summarize_day_metrics


@dataclass(frozen=True)
class RunOutcome:
    """Run-level loss and default metrics used to build frontier pairs."""

    run_id: str
    arm: str
    kappa: float | None
    concentration: float | None
    mu: float | None
    seed: int | None
    delta: float | None
    trader_loss: float
    intermediary_loss: float
    system_loss: float


@dataclass(frozen=True)
class FrontierPair:
    """Treatment-vs-control paired result for frontier analysis."""

    cell_id: str
    kappa: float | None
    concentration: float | None
    mu: float | None
    seed: int | None
    treatment_arm: str
    control_arm: str
    treatment_run_id: str
    control_run_id: str
    delta_treatment: float | None
    delta_control: float | None
    default_relief: float | None
    inst_loss_shift: float
    net_system_relief: float
    pareto_label: str
    help_no_extra_inst_loss: bool
    help_positive_net_system: bool
    pareto_improving: bool


@dataclass(frozen=True)
class FrontierArtifact:
    """Machine-readable analysis artifact for W1/W3."""

    pairs: list[FrontierPair]
    summary_by_arm: list[dict[str, Any]]
    summary_by_arm_kappa: list[dict[str, Any]]
    summary_by_arm_concentration: list[dict[str, Any]]
    summary_by_arm_seed: list[dict[str, Any]]
    loss_floor_by_arm: list[dict[str, Any]]
    loss_floor_by_arm_kappa: list[dict[str, Any]]


_ARM_PREFIXES: list[tuple[str, str]] = [
    ("balanced_bank_dealer_nbfi_", "bank_dealer_nbfi"),
    ("balanced_bank_dealer_", "bank_dealer"),
    ("balanced_bank_passive_", "bank_passive"),
    ("balanced_dealer_lender_", "dealer_lender"),
    ("balanced_nbfi_", "nbfi"),
    ("balanced_active_", "active"),
    ("balanced_passive_", "passive"),
    ("bank_lend_", "bank_lend"),
    ("bank_idle_", "bank_idle"),
    ("nbfi_lend_", "nbfi_lend"),
    ("nbfi_idle_", "nbfi_idle"),
    ("active_", "active"),
    ("passive_", "passive"),
]

_KNOWN_ARMS = {
    "passive",
    "active",
    "nbfi",
    "dealer_lender",
    "bank_passive",
    "bank_dealer",
    "bank_dealer_nbfi",
    "bank_idle",
    "bank_lend",
    "nbfi_idle",
    "nbfi_lend",
}

_BASELINE_BY_TREATMENT = {
    "active": "passive",
    "nbfi": "passive",
    "dealer_lender": "passive",
    "bank_passive": "passive",
    "bank_dealer": "passive",
    "bank_dealer_nbfi": "passive",
    "bank_lend": "bank_idle",
    "nbfi_lend": "nbfi_idle",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _infer_arm(run_dir: Path, scenario: dict[str, Any]) -> str:
    run_id = run_dir.name
    for prefix, arm in _ARM_PREFIXES:
        if run_id.startswith(prefix):
            return arm

    mode = scenario.get("_balanced_config", {}).get("mode")
    if isinstance(mode, str):
        mode = mode.strip()
        if mode in _KNOWN_ARMS:
            return mode

    if run_dir.parent.name == "runs":
        parent_arm = run_dir.parent.parent.name
        if parent_arm in _KNOWN_ARMS:
            return parent_arm

    return "unknown"


def _extract_params(scenario: dict[str, Any]) -> tuple[float | None, float | None, float | None, int | None]:
    bc = scenario.get("_balanced_config", {})
    kappa = _safe_float(bc.get("kappa"))
    concentration = (
        _safe_float(bc.get("concentration"))
        or _safe_float(bc.get("dirichlet_concentration"))
        or _safe_float(bc.get("c"))
    )
    mu = _safe_float(bc.get("mu"))
    seed = _safe_int(bc.get("seed")) or _safe_int(scenario.get("seed"))

    combined_text = f"{scenario.get('name', '')} {scenario.get('description', '')}"
    if kappa is None:
        m = re.search(r"kappa\s*=\s*([0-9.]+)", combined_text)
        if m:
            kappa = _safe_float(m.group(1))
    if concentration is None:
        m = re.search(r"(?:concentration\s*c|c)\s*=\s*([0-9.]+)", combined_text)
        if m:
            concentration = _safe_float(m.group(1))
    if mu is None:
        m = re.search(r"(?:\u03bc|mu)\s*=\s*([0-9.]+)", combined_text)
        if m:
            mu = _safe_float(m.group(1))
    if seed is None:
        m = re.search(r"seed\s*=\s*(\d+)", combined_text)
        if m:
            seed = _safe_int(m.group(1))

    return kappa, concentration, mu, seed


def _compute_delta_from_events(events: list[dict[str, Any]]) -> float | None:
    settled = 0.0
    defaulted = 0.0
    for event in events:
        kind = event.get("kind")
        if kind == "PayableSettled":
            settled += float(event.get("amount", 0))
        elif kind == "ObligationDefaulted":
            contract_kind = str(event.get("contract_kind", ""))
            if contract_kind == "payable" or contract_kind == "":
                shortfall = event.get("shortfall")
                defaulted += float(shortfall if shortfall is not None else event.get("amount", 0))
        elif kind == "ObligationWrittenOff":
            contract_kind = str(event.get("contract_kind", ""))
            if contract_kind == "payable":
                defaulted += float(event.get("amount", 0))

    total = settled + defaulted
    if total <= 0:
        return None
    return defaulted / total


def _compute_delta(run_dir: Path, events: list[dict[str, Any]]) -> float | None:
    metrics = _read_json(run_dir / "out" / "metrics.json")
    if isinstance(metrics, list) and metrics:
        summary = summarize_day_metrics(metrics)
        delta = summary.get("delta_total")
        if delta is not None:
            return float(delta)
    return _compute_delta_from_events(events)


def _compute_trader_loss(events: list[dict[str, Any]]) -> float:
    payable_loss = 0.0
    deposit_loss = 0.0
    for event in events:
        kind = event.get("kind")
        if kind == "ObligationDefaulted":
            if str(event.get("contract_kind", "")) == "payable":
                shortfall = event.get("shortfall")
                payable_loss += float(shortfall if shortfall is not None else event.get("amount", 0))
        elif kind == "ObligationWrittenOff":
            contract_kind = str(event.get("contract_kind", ""))
            amount = float(event.get("amount", 0))
            if contract_kind == "payable":
                payable_loss += amount
            elif contract_kind == "bank_deposit":
                deposit_loss += amount
    return payable_loss + deposit_loss


def discover_run_dirs(experiment_root: Path) -> list[Path]:
    """Discover run directories by locating scenario.yaml + out/events.jsonl."""
    run_dirs: list[Path] = []
    for scenario_path in experiment_root.rglob("scenario.yaml"):
        run_dir = scenario_path.parent
        if (run_dir / "out" / "events.jsonl").exists():
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def load_run_outcomes(experiment_root: Path) -> list[RunOutcome]:
    """Load all run-level outcomes from an experiment/sweep directory."""
    outcomes: list[RunOutcome] = []
    for run_dir in discover_run_dirs(experiment_root):
        scenario_path = run_dir / "scenario.yaml"
        events_path = run_dir / "out" / "events.jsonl"
        with scenario_path.open() as f:
            scenario = yaml.safe_load(f) or {}

        events = _load_events(events_path)
        if not events:
            continue

        dealer_metrics = _read_json(run_dir / "out" / "dealer_metrics.json")
        intermediary = compute_intermediary_losses(events, dealer_metrics)
        trader_loss = _compute_trader_loss(events)
        system_loss = trader_loss + float(intermediary["intermediary_loss_total"])

        kappa, concentration, mu, seed = _extract_params(scenario)
        outcomes.append(
            RunOutcome(
                run_id=run_dir.name,
                arm=_infer_arm(run_dir, scenario),
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                seed=seed,
                delta=_compute_delta(run_dir, events),
                trader_loss=trader_loss,
                intermediary_loss=float(intermediary["intermediary_loss_total"]),
                system_loss=system_loss,
            )
        )

    return outcomes


def _pair_label(default_relief: float | None, inst_loss_shift: float, net_system_relief: float) -> str:
    if default_relief is None:
        return "incomplete"

    pareto_all = (
        default_relief >= 0
        and inst_loss_shift <= 0
        and net_system_relief >= 0
    )
    pareto_strict = (
        default_relief > 0
        or inst_loss_shift < 0
        or net_system_relief > 0
    )
    if pareto_all and pareto_strict:
        return "pareto_improving"
    if default_relief > 0 and inst_loss_shift > 0 and net_system_relief > 0:
        return "loss_shifting_relief"
    if default_relief > 0 and net_system_relief <= 0:
        return "relief_but_net_harm"
    if default_relief <= 0 and net_system_relief > 0:
        return "loss_reduction_without_default_relief"
    return "non_improving"


def _cell_key(run: RunOutcome) -> tuple[float | None, float | None, float | None]:
    return (run.kappa, run.concentration, run.mu)


def _pair_group(control: list[RunOutcome], treatment: list[RunOutcome]) -> list[tuple[RunOutcome, RunOutcome, int | None]]:
    if not control or not treatment:
        return []

    has_seeds = all(x.seed is not None for x in control + treatment)
    if has_seeds:
        c_by_seed: dict[int, list[RunOutcome]] = defaultdict(list)
        t_by_seed: dict[int, list[RunOutcome]] = defaultdict(list)
        for row in control:
            c_by_seed[row.seed or 0].append(row)
        for row in treatment:
            t_by_seed[row.seed or 0].append(row)

        out: list[tuple[RunOutcome, RunOutcome, int | None]] = []
        for seed in sorted(set(c_by_seed) & set(t_by_seed)):
            c_rows = sorted(c_by_seed[seed], key=lambda x: x.run_id)
            t_rows = sorted(t_by_seed[seed], key=lambda x: x.run_id)
            for c_row, t_row in zip(c_rows, t_rows):
                out.append((c_row, t_row, seed))
        if out:
            return out

    c_rows = sorted(control, key=lambda x: x.run_id)
    t_rows = sorted(treatment, key=lambda x: x.run_id)
    return [(c_row, t_row, None) for c_row, t_row in zip(c_rows, t_rows)]


def build_frontier_pairs(outcomes: list[RunOutcome]) -> list[FrontierPair]:
    """Pair treatment and control runs and compute frontier metrics per pair."""
    by_cell: dict[tuple[float | None, float | None, float | None], list[RunOutcome]] = defaultdict(list)
    for row in outcomes:
        by_cell[_cell_key(row)].append(row)

    pairs: list[FrontierPair] = []
    for (kappa, concentration, mu), cell_runs in sorted(by_cell.items(), key=str):
        by_arm: dict[str, list[RunOutcome]] = defaultdict(list)
        for row in cell_runs:
            by_arm[row.arm].append(row)

        cell_id = f"kappa={kappa},c={concentration},mu={mu}"
        for treatment_arm, baseline_arm in _BASELINE_BY_TREATMENT.items():
            if treatment_arm not in by_arm or baseline_arm not in by_arm:
                continue
            matches = _pair_group(by_arm[baseline_arm], by_arm[treatment_arm])
            for control, treatment, paired_seed in matches:
                default_relief = (
                    control.delta - treatment.delta
                    if control.delta is not None and treatment.delta is not None
                    else None
                )
                inst_loss_shift = treatment.intermediary_loss - control.intermediary_loss
                net_system_relief = control.system_loss - treatment.system_loss
                label = _pair_label(default_relief, inst_loss_shift, net_system_relief)
                help_no_extra = bool(default_relief is not None and default_relief > 0 and inst_loss_shift <= 0)
                help_net = bool(default_relief is not None and default_relief > 0 and net_system_relief > 0)
                pareto = label == "pareto_improving"

                pairs.append(
                    FrontierPair(
                        cell_id=cell_id,
                        kappa=kappa,
                        concentration=concentration,
                        mu=mu,
                        seed=paired_seed if paired_seed is not None else control.seed,
                        treatment_arm=treatment_arm,
                        control_arm=baseline_arm,
                        treatment_run_id=treatment.run_id,
                        control_run_id=control.run_id,
                        delta_treatment=treatment.delta,
                        delta_control=control.delta,
                        default_relief=default_relief,
                        inst_loss_shift=inst_loss_shift,
                        net_system_relief=net_system_relief,
                        pareto_label=label,
                        help_no_extra_inst_loss=help_no_extra,
                        help_positive_net_system=help_net,
                        pareto_improving=pareto,
                    )
                )

    return pairs


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_frontier_pairs(
    pairs: list[FrontierPair],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate frontier diagnostics by requested grouping fields."""
    grouped: dict[tuple[Any, ...], list[FrontierPair]] = defaultdict(list)
    for pair in pairs:
        key = tuple(getattr(pair, field) for field in group_fields)
        grouped[key].append(pair)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=str):
        group = grouped[key]
        complete = [p for p in group if p.default_relief is not None]

        def_reliefs = [p.default_relief for p in complete if p.default_relief is not None]
        inst_shifts = [p.inst_loss_shift for p in group]
        net_reliefs = [p.net_system_relief for p in group]
        help_count = sum(1 for p in complete if p.default_relief is not None and p.default_relief > 0)
        pareto_count = sum(1 for p in group if p.pareto_improving)
        help_no_extra_count = sum(1 for p in group if p.help_no_extra_inst_loss)
        help_net_count = sum(1 for p in group if p.help_positive_net_system)

        row: dict[str, Any] = {field: value for field, value in zip(group_fields, key, strict=False)}
        row.update(
            {
                "n_pairs": len(group),
                "n_complete": len(complete),
                "mean_default_relief": _mean(def_reliefs),
                "mean_inst_loss_shift": _mean(inst_shifts),
                "mean_net_system_relief": _mean(net_reliefs),
                "share_help": (help_count / len(complete)) if complete else None,
                "share_help_no_extra_inst_loss": (help_no_extra_count / len(complete)) if complete else None,
                "share_help_positive_net_system": (help_net_count / len(complete)) if complete else None,
                "share_pareto_improving": (pareto_count / len(complete)) if complete else None,
            }
        )
        rows.append(row)

    return rows


def _format_band(lower: float, upper: float) -> str:
    if upper == inf:
        return f"[{lower:.4f}, inf)"
    return f"[{lower:.4f}, {upper:.4f})"


def compute_loss_floor_table(
    pairs: list[FrontierPair],
    *,
    group_fields: tuple[str, ...] = ("treatment_arm",),
    relief_bands: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05),
) -> list[dict[str, Any]]:
    """Estimate empirical intermediary loss floors by default-relief bands.

    A positive floor indicates all observed runs in that relief band required
    positive intermediary loss shift. A non-positive floor indicates at least
    one observed avoidable-loss path in that band.
    """
    band_edges = sorted(set(float(x) for x in relief_bands))
    if not band_edges:
        band_edges = [0.0]
    if band_edges[0] > 0:
        band_edges = [0.0] + band_edges

    intervals: list[tuple[float, float]] = []
    for idx, lower in enumerate(band_edges):
        upper = band_edges[idx + 1] if idx + 1 < len(band_edges) else inf
        intervals.append((lower, upper))

    grouped: dict[tuple[Any, ...], list[FrontierPair]] = defaultdict(list)
    for pair in pairs:
        key = tuple(getattr(pair, field) for field in group_fields)
        grouped[key].append(pair)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=str):
        group = [p for p in grouped[key] if p.default_relief is not None]
        for lower, upper in intervals:
            band_rows = [
                p for p in group
                if p.default_relief is not None and p.default_relief >= lower and p.default_relief < upper
            ]
            if not band_rows:
                continue

            shifts = [p.inst_loss_shift for p in band_rows]
            min_shift = min(shifts)
            median_shift = sorted(shifts)[len(shifts) // 2]
            help_runs = [p for p in band_rows if p.default_relief is not None and p.default_relief > 0]
            avoidable_exists = any(p.inst_loss_shift <= 0 for p in help_runs)
            structural_region = bool(help_runs) and not avoidable_exists and min_shift > 0

            row: dict[str, Any] = {field: value for field, value in zip(group_fields, key, strict=False)}
            row.update(
                {
                    "relief_band": _format_band(lower, upper),
                    "relief_band_lower": lower,
                    "relief_band_upper": None if upper == inf else upper,
                    "n_pairs": len(band_rows),
                    "n_help_runs": len(help_runs),
                    "min_inst_loss_shift": min_shift,
                    "median_inst_loss_shift": median_shift,
                    "help_no_extra_inst_loss_share": (
                        sum(1 for p in help_runs if p.inst_loss_shift <= 0) / len(help_runs)
                        if help_runs
                        else None
                    ),
                    "empirical_loss_floor": max(0.0, min_shift),
                    "avoidable_region": avoidable_exists,
                    "structural_region": structural_region,
                }
            )
            rows.append(row)

    return rows


def build_frontier_artifact(
    experiment_root: Path,
    *,
    relief_bands: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05),
) -> FrontierArtifact:
    """Build full W1/W3 artifact from a sweep/experiment root directory."""
    outcomes = load_run_outcomes(experiment_root)
    pairs = build_frontier_pairs(outcomes)
    return FrontierArtifact(
        pairs=pairs,
        summary_by_arm=summarize_frontier_pairs(pairs, ("treatment_arm",)),
        summary_by_arm_kappa=summarize_frontier_pairs(pairs, ("treatment_arm", "kappa")),
        summary_by_arm_concentration=summarize_frontier_pairs(
            pairs,
            ("treatment_arm", "concentration"),
        ),
        summary_by_arm_seed=summarize_frontier_pairs(pairs, ("treatment_arm", "seed")),
        loss_floor_by_arm=compute_loss_floor_table(
            pairs,
            group_fields=("treatment_arm",),
            relief_bands=relief_bands,
        ),
        loss_floor_by_arm_kappa=compute_loss_floor_table(
            pairs,
            group_fields=("treatment_arm", "kappa"),
            relief_bands=relief_bands,
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[])
            writer.writeheader()
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_frontier_artifact(artifact: FrontierArtifact, out_dir: Path) -> dict[str, Path]:
    """Write frontier artifact tables to machine-readable CSV/JSON outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [asdict(pair) for pair in artifact.pairs]
    files = {
        "pairs_csv": out_dir / "frontier_pairs.csv",
        "summary_by_arm_csv": out_dir / "summary_by_arm.csv",
        "summary_by_arm_kappa_csv": out_dir / "summary_by_arm_kappa.csv",
        "summary_by_arm_concentration_csv": out_dir / "summary_by_arm_concentration.csv",
        "summary_by_arm_seed_csv": out_dir / "summary_by_arm_seed.csv",
        "loss_floor_by_arm_csv": out_dir / "loss_floor_by_arm.csv",
        "loss_floor_by_arm_kappa_csv": out_dir / "loss_floor_by_arm_kappa.csv",
        "artifact_json": out_dir / "artifact.json",
    }

    _write_csv(files["pairs_csv"], pair_rows)
    _write_csv(files["summary_by_arm_csv"], artifact.summary_by_arm)
    _write_csv(files["summary_by_arm_kappa_csv"], artifact.summary_by_arm_kappa)
    _write_csv(files["summary_by_arm_concentration_csv"], artifact.summary_by_arm_concentration)
    _write_csv(files["summary_by_arm_seed_csv"], artifact.summary_by_arm_seed)
    _write_csv(files["loss_floor_by_arm_csv"], artifact.loss_floor_by_arm)
    _write_csv(files["loss_floor_by_arm_kappa_csv"], artifact.loss_floor_by_arm_kappa)

    serializable = {
        "pairs": pair_rows,
        "summary_by_arm": artifact.summary_by_arm,
        "summary_by_arm_kappa": artifact.summary_by_arm_kappa,
        "summary_by_arm_concentration": artifact.summary_by_arm_concentration,
        "summary_by_arm_seed": artifact.summary_by_arm_seed,
        "loss_floor_by_arm": artifact.loss_floor_by_arm,
        "loss_floor_by_arm_kappa": artifact.loss_floor_by_arm_kappa,
    }
    with files["artifact_json"].open("w") as f:
        json.dump(serializable, f, indent=2)

    return files


__all__ = [
    "FrontierArtifact",
    "FrontierPair",
    "RunOutcome",
    "build_frontier_artifact",
    "build_frontier_pairs",
    "compute_loss_floor_table",
    "discover_run_dirs",
    "load_run_outcomes",
    "summarize_frontier_pairs",
    "write_frontier_artifact",
]

