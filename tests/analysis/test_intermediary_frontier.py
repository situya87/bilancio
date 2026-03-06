"""Tests for intermediary frontier and loss-floor analysis."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from bilancio.analysis.intermediary_frontier import (
    FrontierPair,
    build_frontier_artifact,
    build_frontier_pairs,
    compute_loss_floor_table,
    load_run_outcomes,
    write_frontier_artifact,
)


def _write_run(
    root: Path,
    run_id: str,
    *,
    kappa: float,
    concentration: float,
    mu: float,
    seed: int,
    delta: float,
    events: list[dict],
    dealer_total_pnl: float | None = None,
) -> None:
    run_dir = root / "runs" / run_id
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = {
        "version": 1,
        "name": f"Sweep run (kappa={kappa}, c={concentration}, mu={mu}, seed={seed})",
        "_balanced_config": {
            "kappa": kappa,
            "concentration": concentration,
            "mu": mu,
            "seed": seed,
        },
        "initial_actions": [],
    }
    with (run_dir / "scenario.yaml").open("w") as f:
        yaml.safe_dump(scenario, f, sort_keys=False)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump([{"day": 1, "S_t": "100", "delta_t": str(delta)}], f)

    with (out_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    if dealer_total_pnl is not None:
        with (out_dir / "dealer_metrics.json").open("w") as f:
            json.dump({"dealer_total_pnl": dealer_total_pnl}, f)


def test_build_frontier_pairs_labels_and_shares(tmp_path: Path) -> None:
    # Seed 1: helpful but with extra intermediary losses.
    _write_run(
        tmp_path,
        "balanced_passive_seed1",
        kappa=1.0,
        concentration=1.0,
        mu=0.5,
        seed=1,
        delta=0.40,
        events=[
            {"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 40},
        ],
    )
    _write_run(
        tmp_path,
        "balanced_active_seed1",
        kappa=1.0,
        concentration=1.0,
        mu=0.5,
        seed=1,
        delta=0.20,
        events=[
            {"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 20},
            {"kind": "NonBankLoanDefaulted", "amount_owed": 10, "cash_available": 8},
        ],
    )

    # Seed 2: Pareto-improving (default relief + lower intermediary loss + net relief).
    _write_run(
        tmp_path,
        "balanced_passive_seed2",
        kappa=1.0,
        concentration=1.0,
        mu=0.5,
        seed=2,
        delta=0.35,
        events=[
            {"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 35},
            {"kind": "BankLoanDefault", "repayment_due": 10, "recovered": 9},
        ],
    )
    _write_run(
        tmp_path,
        "balanced_active_seed2",
        kappa=1.0,
        concentration=1.0,
        mu=0.5,
        seed=2,
        delta=0.20,
        events=[
            {"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 20},
        ],
    )

    outcomes = load_run_outcomes(tmp_path)
    pairs = build_frontier_pairs(outcomes)
    assert len(pairs) == 2

    labels = {p.seed: p.pareto_label for p in pairs}
    assert labels[1] == "loss_shifting_relief"
    assert labels[2] == "pareto_improving"

    artifact = build_frontier_artifact(tmp_path)
    summary = {row["treatment_arm"]: row for row in artifact.summary_by_arm}
    active = summary["active"]
    assert active["n_pairs"] == 2
    assert active["share_help"] == 1.0
    assert active["share_help_no_extra_inst_loss"] == 0.5
    assert active["share_help_positive_net_system"] == 1.0
    assert active["share_pareto_improving"] == 0.5


def test_compute_loss_floor_structural_region_flag() -> None:
    pairs = [
        FrontierPair(
            cell_id="kappa=1,c=1,mu=0.5",
            kappa=1.0,
            concentration=1.0,
            mu=0.5,
            seed=1,
            treatment_arm="bank_lend",
            control_arm="bank_idle",
            treatment_run_id="t1",
            control_run_id="c1",
            delta_treatment=0.2,
            delta_control=0.4,
            default_relief=0.2,
            inst_loss_shift=0.3,
            net_system_relief=0.1,
            pareto_label="loss_shifting_relief",
            help_no_extra_inst_loss=False,
            help_positive_net_system=True,
            pareto_improving=False,
        ),
    ]

    rows = compute_loss_floor_table(
        pairs,
        group_fields=("treatment_arm",),
        relief_bands=(0.0, 0.1),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["treatment_arm"] == "bank_lend"
    assert row["empirical_loss_floor"] == 0.3
    assert row["avoidable_region"] is False
    assert row["structural_region"] is True


def test_write_frontier_artifact_outputs(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "nbfi_idle_seed1",
        kappa=0.5,
        concentration=1.0,
        mu=0.0,
        seed=1,
        delta=0.30,
        events=[{"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 30}],
    )
    _write_run(
        tmp_path,
        "nbfi_lend_seed1",
        kappa=0.5,
        concentration=1.0,
        mu=0.0,
        seed=1,
        delta=0.20,
        events=[{"kind": "ObligationDefaulted", "contract_kind": "payable", "shortfall": 20}],
    )

    artifact = build_frontier_artifact(tmp_path)
    files = write_frontier_artifact(artifact, tmp_path / "aggregate" / "intermediary_frontier")
    for path in files.values():
        assert path.exists()
