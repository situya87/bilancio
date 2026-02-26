#!/usr/bin/env python3
"""Post-hoc intermediary loss analysis for existing sweep results.

Reads events.jsonl, dealer_metrics.json, and scenario.yaml from each run
directory, computes intermediary losses, trader losses, and initial capitals,
then outputs comparison tables using three analytical approaches.

Approach 3 (Main): Total System Loss -- trader + intermediary losses summed
Approach 2: Loss/Capital Ratio -- intermediary loss normalized by initial capital
Approach 1: Capacity-Adjusted Effect -- scale losses to common capacity basis

Usage:
    uv run python scripts/analyze_intermediary_losses.py <sweep_dir> [approach]

    approach: "1", "2", "3", or "all" (default: "all")

Example:
    uv run python scripts/analyze_intermediary_losses.py out/experiments/seven-arm-sweep-v3/wrecker-posture-lucrative-unboxed
    uv run python scripts/analyze_intermediary_losses.py out/experiments/seven-arm-sweep-v3/wrecker-posture-lucrative-unboxed 3
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from bilancio.analysis.report import (
    compute_intermediary_losses,
    extract_initial_capitals,
    summarize_day_metrics,
)

ARM_ORDER = {
    "passive": 0,
    "active": 1,
    "nbfi": 2,
    "dealer_lender": 3,
    "bank_passive": 4,
    "bank_dealer": 5,
    "bank_dealer_nbfi": 6,
}

ALL_ARMS = [
    "active",
    "nbfi",
    "dealer_lender",
    "bank_passive",
    "bank_dealer",
    "bank_dealer_nbfi",
]


def load_events(run_dir: Path) -> list[dict]:
    """Load events from a run's events.jsonl."""
    events_path = run_dir / "out" / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_dealer_metrics(run_dir: Path) -> dict | None:
    """Load dealer_metrics.json from a run directory."""
    path = run_dir / "out" / "dealer_metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_metrics_json(run_dir: Path) -> list[dict[str, Any]] | None:
    """Load metrics.json (day-level metrics) from a run directory."""
    path = run_dir / "out" / "metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_scenario_config(run_dir: Path) -> dict[str, Any] | None:
    """Load scenario.yaml from a run directory."""
    path = run_dir / "scenario.yaml"
    if not path.exists():
        return None
    try:
        import yaml

        with path.open() as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def compute_delta_from_events(events: list[dict]) -> float | None:
    """Compute delta (default rate) directly from events as a fallback.

    Uses PayableSettled for settled amounts and ObligationDefaulted /
    ObligationWrittenOff for defaulted amounts.
    """
    total_settled = 0
    total_defaulted = 0
    for e in events:
        kind = e.get("kind", "")
        if kind == "PayableSettled":
            total_settled += int(e.get("amount", 0))
        elif kind == "ObligationDefaulted":
            ck = e.get("contract_kind", "")
            # Match payable defaults or older events without contract_kind
            if ck == "payable" or ck == "":
                total_defaulted += int(e.get("amount", 0))
        elif kind == "ObligationWrittenOff":
            ck = e.get("contract_kind", "")
            if ck == "payable":
                total_defaulted += int(e.get("amount", 0))

    total = total_settled + total_defaulted
    if total == 0:
        return None
    return total_defaulted / total


def compute_delta(run_dir: Path, events: list[dict]) -> float | None:
    """Compute delta_total, preferring metrics.json via summarize_day_metrics.

    Falls back to manual event-based computation if metrics.json is missing.
    """
    day_metrics = load_metrics_json(run_dir)
    if day_metrics is not None and len(day_metrics) > 0:
        summary = summarize_day_metrics(day_metrics)
        delta_total = summary.get("delta_total")
        if delta_total is not None:
            return float(delta_total)

    # Fallback: compute from events
    return compute_delta_from_events(events)


def compute_total_loss(events: list[dict]) -> dict[str, float]:
    """Compute trader losses from events.

    Trader losses consist of:
    - Payable default loss: ObligationWrittenOff where contract_kind == "payable"
    - Deposit loss: ObligationWrittenOff where contract_kind == "bank_deposit"
    - total_loss = payable_default_loss + deposit_loss

    Returns dict with payable_default_loss, deposit_loss, total_loss.
    """
    payable_default_loss = 0.0
    deposit_loss = 0.0

    for e in events:
        kind = e.get("kind", "")
        if kind == "ObligationWrittenOff":
            ck = str(e.get("contract_kind", ""))
            amount = float(e.get("amount", 0))
            if ck == "payable":
                payable_default_loss += amount
            elif ck == "bank_deposit":
                deposit_loss += amount

    return {
        "payable_default_loss": payable_default_loss,
        "deposit_loss": deposit_loss,
        "total_loss": payable_default_loss + deposit_loss,
    }


def extract_arm_and_params(run_dir: Path) -> tuple[str, dict[str, str]]:
    """Extract arm type and parameters from run directory name and scenario.yaml."""
    run_id = run_dir.name

    # Parse arm from run_id prefix (longest prefixes first to avoid partial matches)
    arm_prefixes = [
        "balanced_bank_dealer_nbfi_",
        "balanced_bank_dealer_",
        "balanced_bank_passive_",
        "balanced_dealer_lender_",
        "balanced_nbfi_",
        "balanced_active_",
        "balanced_passive_",
    ]
    arm = "unknown"
    for prefix in arm_prefixes:
        if run_id.startswith(prefix):
            arm = prefix.rstrip("_").replace("balanced_", "")
            break

    # Extract kappa and mu from scenario.yaml
    params: dict[str, str] = {"kappa": "?", "mu": "?"}
    scenario_path = run_dir / "scenario.yaml"
    if scenario_path.exists():
        try:
            import yaml

            with scenario_path.open() as f:
                scenario = yaml.safe_load(f)
            # Try to get kappa from balanced config
            bc = scenario.get("_balanced_config", {})
            if bc and "kappa" in bc:
                params["kappa"] = str(bc["kappa"])
            # Try to get mu from description (it's embedded there)
            # Match both Unicode mu and ASCII "mu"
            desc = scenario.get("description", "")
            mu_match = re.search(r"(?:\u03bc|mu)=([\d.]+)", desc)
            if mu_match:
                params["mu"] = mu_match.group(1)
        except Exception:
            pass  # gracefully handle corrupt/missing YAML

    return arm, params


def _mean(values: list[float]) -> float | None:
    """Return mean of a list, or None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Per-cell display helpers
# ---------------------------------------------------------------------------


def _print_approach3_cell(
    group: list[dict[str, Any]],
    mean_passive_delta: float | None,
) -> None:
    """Print Approach 3 table for a single (kappa, mu) cell."""
    print(
        f"\n  Approach 3 -- Total System Loss"
    )
    print(
        f"  {'Arm':<22} {'Trader Loss':>12} {'Interm Loss':>12} "
        f"{'System Loss':>12} {'Sys Loss%':>10} {'Sys Effect':>10}"
    )
    print(
        f"  {'-' * 22} {'-' * 12} {'-' * 12} "
        f"{'-' * 12} {'-' * 10} {'-' * 10}"
    )

    # Compute passive baseline system loss %
    passive_sys_loss_pcts = [
        r["system_loss"] / r["initial_debt"]
        for r in group
        if r["arm"] == "passive"
        and r["initial_debt"] > 0
    ]
    mean_passive_sys_pct = _mean(passive_sys_loss_pcts)

    # Group by arm for per-arm means
    arm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in group:
        arm_runs[r["arm"]].append(r)

    for arm_name in ["passive"] + ALL_ARMS:
        if arm_name not in arm_runs:
            continue
        runs = arm_runs[arm_name]
        trader_losses = [r["total_loss"] for r in runs]
        interm_losses = [r["intermediary_loss_total"] for r in runs]
        sys_losses = [r["system_loss"] for r in runs]
        sys_loss_pcts = [
            r["system_loss"] / r["initial_debt"]
            for r in runs
            if r["initial_debt"] > 0
        ]

        mean_tl = _mean(trader_losses)
        mean_il = _mean(interm_losses)
        mean_sl = _mean(sys_losses)
        mean_sp = _mean(sys_loss_pcts)

        if mean_tl is None:
            continue

        if arm_name == "passive" or mean_passive_sys_pct is None:
            eff_str = "--"
        else:
            eff = mean_sp - mean_passive_sys_pct if mean_sp is not None else None
            eff_str = f"{eff:>10.4f}" if eff is not None else "N/A"

        sp_str = f"{mean_sp:.4f}" if mean_sp is not None else "N/A"
        print(
            f"  {arm_name:<22} {mean_tl:>12.1f} {mean_il:>12.1f} "
            f"{mean_sl:>12.1f} {sp_str:>10} {eff_str:>10}"
        )


def _print_approach2_cell(
    group: list[dict[str, Any]],
) -> None:
    """Print Approach 2 table for a single (kappa, mu) cell."""
    print(
        f"\n  Approach 2 -- Loss/Capital Ratio"
    )
    print(
        f"  {'Arm':<22} {'Interm Loss':>12} {'Init Capital':>12} {'Loss/Cap':>10}"
    )
    print(
        f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 10}"
    )

    arm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in group:
        arm_runs[r["arm"]].append(r)

    for arm_name in ["passive"] + ALL_ARMS:
        if arm_name not in arm_runs:
            continue
        runs = arm_runs[arm_name]
        interm_losses = [r["intermediary_loss_total"] for r in runs]
        init_caps = [r["intermediary_capital"] for r in runs]

        mean_il = _mean(interm_losses)
        mean_ic = _mean(init_caps)

        if mean_il is None:
            continue

        if mean_ic is not None and mean_ic > 0:
            loss_cap = mean_il / mean_ic
            lc_str = f"{loss_cap:.4f}"
        else:
            lc_str = "N/A"

        ic_str = f"{mean_ic:.1f}" if mean_ic is not None else "N/A"
        print(
            f"  {arm_name:<22} {mean_il:>12.1f} {ic_str:>12} {lc_str:>10}"
        )


def _print_approach1_cell(
    group: list[dict[str, Any]],
) -> None:
    """Print Approach 1 table for a single (kappa, mu) cell.

    Capacity-adjusted effect: scale intermediary losses to passive arm's
    capital basis before computing treatment effects.

    capacity_adjusted_loss = intermediary_loss * (reference_capital / arm_capital)
    where reference_capital = mean intermediary capital of passive arm
    """
    print(
        f"\n  Approach 1 -- Capacity-Adjusted Effect"
    )

    arm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in group:
        arm_runs[r["arm"]].append(r)

    # Get passive reference capital
    passive_caps = [
        r["intermediary_capital"]
        for r in group
        if r["arm"] == "passive" and r["intermediary_capital"] > 0
    ]
    ref_capital = _mean(passive_caps)

    if ref_capital is None or ref_capital == 0:
        print("  (no passive reference capital available)")
        return

    # Passive baseline intermediary loss %
    passive_loss_pcts = [
        r["intermediary_loss_total"] / r["initial_debt"]
        for r in group
        if r["arm"] == "passive" and r["initial_debt"] > 0
    ]
    mean_passive_loss_pct = _mean(passive_loss_pcts)

    print(
        f"  {'Arm':<22} {'Raw IntLoss%':>12} {'Cap-Adj Loss':>14} "
        f"{'Cap-Adj Loss%':>14} {'Cap-Adj Eff':>12}"
    )
    print(
        f"  {'-' * 22} {'-' * 12} {'-' * 14} "
        f"{'-' * 14} {'-' * 12}"
    )

    for arm_name in ["passive"] + ALL_ARMS:
        if arm_name not in arm_runs:
            continue
        runs = arm_runs[arm_name]
        interm_losses = [r["intermediary_loss_total"] for r in runs]
        init_caps = [r["intermediary_capital"] for r in runs]
        init_debts = [r["initial_debt"] for r in runs if r["initial_debt"] > 0]

        mean_il = _mean(interm_losses)
        mean_ic = _mean(init_caps)
        mean_debt = _mean(init_debts)

        if mean_il is None:
            continue

        # Raw intermediary loss as % of initial debt
        raw_pct = mean_il / mean_debt if mean_debt and mean_debt > 0 else None

        # Capacity-adjusted loss
        if mean_ic is not None and mean_ic > 0:
            cap_adj_loss = mean_il * (ref_capital / mean_ic)
        else:
            cap_adj_loss = mean_il  # no scaling if no capital

        cap_adj_pct = cap_adj_loss / mean_debt if mean_debt and mean_debt > 0 else None

        if arm_name == "passive" or mean_passive_loss_pct is None:
            eff_str = "--"
        else:
            eff = cap_adj_pct - mean_passive_loss_pct if cap_adj_pct is not None else None
            eff_str = f"{eff:>12.4f}" if eff is not None else "N/A"

        raw_str = f"{raw_pct:.4f}" if raw_pct is not None else "N/A"
        cadj_str = f"{cap_adj_loss:.1f}"
        cadj_pct_str = f"{cap_adj_pct:.4f}" if cap_adj_pct is not None else "N/A"

        print(
            f"  {arm_name:<22} {raw_str:>12} {cadj_str:>14} "
            f"{cadj_pct_str:>14} {eff_str:>12}"
        )


# ---------------------------------------------------------------------------
# Overall summary helpers
# ---------------------------------------------------------------------------


def _compute_overall_approach3(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, dict[str, list[float]]]:
    """Compute overall Approach 3 stats across all cells."""
    arm_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"sys_effects": []}
    )
    for (_kappa, _mu), group in groups.items():
        # Passive baseline for this cell
        passive_sys_pcts = [
            r["system_loss"] / r["initial_debt"]
            for r in group
            if r["arm"] == "passive" and r["initial_debt"] > 0
        ]
        mean_p_sys = _mean(passive_sys_pcts)
        if mean_p_sys is None:
            continue

        # Per-arm
        cell_arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            if r["arm"] != "passive":
                cell_arms[r["arm"]].append(r)

        for arm_name, runs in cell_arms.items():
            sys_pcts = [
                r["system_loss"] / r["initial_debt"]
                for r in runs
                if r["initial_debt"] > 0
            ]
            mean_sys = _mean(sys_pcts)
            if mean_sys is None:
                continue
            arm_stats[arm_name]["sys_effects"].append(mean_sys - mean_p_sys)

    return arm_stats


def _compute_overall_approach2(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, dict[str, list[float]]]:
    """Compute overall Approach 2 stats across all cells."""
    arm_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"loss_caps": []}
    )
    # Also collect passive stats
    passive_lcs: list[float] = []
    for (_kappa, _mu), group in groups.items():
        # Per arm including passive
        cell_arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            cell_arms[r["arm"]].append(r)

        for arm_name, runs in cell_arms.items():
            interm = [r["intermediary_loss_total"] for r in runs]
            caps = [r["intermediary_capital"] for r in runs if r["intermediary_capital"] > 0]
            mean_il = _mean(interm)
            mean_ic = _mean(caps)
            if mean_il is not None and mean_ic is not None and mean_ic > 0:
                lc = mean_il / mean_ic
                if arm_name == "passive":
                    passive_lcs.append(lc)
                else:
                    arm_stats[arm_name]["loss_caps"].append(lc)

    # Store passive mean for reference
    arm_stats["_passive_mean_lc"] = {"loss_caps": passive_lcs}
    return arm_stats


def _compute_overall_approach1(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, dict[str, list[float]]]:
    """Compute overall Approach 1 stats across all cells."""
    arm_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"cap_adj_effects": []}
    )
    for (_kappa, _mu), group in groups.items():
        # Passive reference capital and loss% for this cell
        passive_caps = [
            r["intermediary_capital"]
            for r in group
            if r["arm"] == "passive" and r["intermediary_capital"] > 0
        ]
        passive_loss_pcts = [
            r["intermediary_loss_total"] / r["initial_debt"]
            for r in group
            if r["arm"] == "passive" and r["initial_debt"] > 0
        ]
        ref_capital = _mean(passive_caps)
        mean_p_loss_pct = _mean(passive_loss_pcts)
        if ref_capital is None or ref_capital == 0 or mean_p_loss_pct is None:
            continue

        # Per-arm
        cell_arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            if r["arm"] != "passive":
                cell_arms[r["arm"]].append(r)

        for arm_name, runs in cell_arms.items():
            interm = [r["intermediary_loss_total"] for r in runs]
            caps = [r["intermediary_capital"] for r in runs]
            debts = [r["initial_debt"] for r in runs if r["initial_debt"] > 0]
            mean_il = _mean(interm)
            mean_ic = _mean(caps)
            mean_debt = _mean(debts)
            if mean_il is None or mean_debt is None or mean_debt == 0:
                continue
            if mean_ic is not None and mean_ic > 0:
                cap_adj_loss = mean_il * (ref_capital / mean_ic)
            else:
                cap_adj_loss = mean_il
            cap_adj_pct = cap_adj_loss / mean_debt
            eff = cap_adj_pct - mean_p_loss_pct
            arm_stats[arm_name]["cap_adj_effects"].append(eff)

    return arm_stats


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python scripts/analyze_intermediary_losses.py"
            " <sweep_dir> [approach]"
        )
        print("  approach: '1', '2', '3', or 'all' (default: 'all')")
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])
    approach = sys.argv[2] if len(sys.argv) >= 3 else "all"

    if approach not in ("1", "2", "3", "all"):
        print(f"Error: invalid approach '{approach}'. Use '1', '2', '3', or 'all'.")
        sys.exit(1)

    runs_dir = sweep_dir / "runs"

    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        sys.exit(1)

    # Collect results
    results: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        arm, params = extract_arm_and_params(run_dir)
        events = load_events(run_dir)
        if not events:
            continue

        dealer_metrics = load_dealer_metrics(run_dir)
        losses = compute_intermediary_losses(events, dealer_metrics)
        delta = compute_delta(run_dir, events)
        trader_losses = compute_total_loss(events)

        # Get initial_total_debt for normalization
        initial_debt = float((dealer_metrics or {}).get("initial_total_debt", 0))
        loss_pct = (
            losses["intermediary_loss_total"] / initial_debt
            if initial_debt > 0
            else None
        )

        # Extract initial capitals from scenario.yaml
        scenario_config = load_scenario_config(run_dir)
        if scenario_config is not None:
            capitals = extract_initial_capitals(scenario_config)
        else:
            capitals = {
                "trader_capital": 0.0,
                "dealer_capital": 0.0,
                "vbt_capital": 0.0,
                "lender_capital": 0.0,
                "bank_capital": 0.0,
                "intermediary_capital": 0.0,
            }

        system_loss = (
            trader_losses["total_loss"] + losses["intermediary_loss_total"]
        )

        results.append(
            {
                "run_id": run_dir.name,
                "arm": arm,
                "kappa": params["kappa"],
                "mu": params["mu"],
                "delta": delta,
                "dealer_vbt_loss": losses["dealer_vbt_loss"],
                "nbfi_loan_loss": losses["nbfi_loan_loss"],
                "bank_credit_loss": losses["bank_credit_loss"],
                "cb_backstop_loss": losses["cb_backstop_loss"],
                "intermediary_loss_total": losses["intermediary_loss_total"],
                "initial_debt": initial_debt,
                "intermediary_loss_pct": loss_pct,
                # Trader losses
                "payable_default_loss": trader_losses["payable_default_loss"],
                "deposit_loss": trader_losses["deposit_loss"],
                "total_loss": trader_losses["total_loss"],
                "system_loss": system_loss,
                # Capitals
                "trader_capital": capitals["trader_capital"],
                "dealer_capital": capitals["dealer_capital"],
                "vbt_capital": capitals["vbt_capital"],
                "lender_capital": capitals["lender_capital"],
                "bank_capital": capitals["bank_capital"],
                "intermediary_capital": capitals["intermediary_capital"],
            }
        )

    if not results:
        print("No completed runs found.")
        sys.exit(1)

    # Print header
    print(f"\n{'=' * 120}")
    print(f"INTERMEDIARY LOSS ANALYSIS -- {sweep_dir.name}")
    print(f"{'=' * 120}")

    # Group by (kappa, mu) and show all arms side by side
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r["kappa"], r["mu"])].append(r)

    # Sort arms within each group
    for key in groups:
        groups[key].sort(key=lambda r: ARM_ORDER.get(r["arm"], 99))

    # ---- Raw data table (always shown) ----
    for (kappa, mu), group in sorted(groups.items()):
        print(f"\n--- kappa={kappa}, mu={mu} ---")
        print(
            f"{'Arm':<22} {'delta':>8} {'Dealer/VBT':>12} {'NBFI':>10} "
            f"{'Bank':>10} {'CB':>8} {'IntrmTotal':>12} {'TrdrLoss':>10} "
            f"{'IntrmCap':>10}"
        )
        print(
            f"{'-' * 22} {'-' * 8} {'-' * 12} {'-' * 10} "
            f"{'-' * 10} {'-' * 8} {'-' * 12} {'-' * 10} "
            f"{'-' * 10}"
        )

        for r in group:
            delta_str = f"{r['delta']:.4f}" if r["delta"] is not None else "N/A"

            print(
                f"{r['arm']:<22} {delta_str:>8} "
                f"{r['dealer_vbt_loss']:>12.1f} {r['nbfi_loan_loss']:>10.1f} "
                f"{r['bank_credit_loss']:>10.1f} {r['cb_backstop_loss']:>8.1f} "
                f"{r['intermediary_loss_total']:>12.1f} "
                f"{r['total_loss']:>10.1f} "
                f"{r['intermediary_capital']:>10.1f}"
            )

        # Compute passive baseline for this cell
        mean_passive_delta = _mean([
            r["delta"] for r in group
            if r["arm"] == "passive" and r["delta"] is not None
        ])

        # Show approach tables
        if approach in ("3", "all"):
            _print_approach3_cell(group, mean_passive_delta)

        if approach in ("2", "all"):
            _print_approach2_cell(group)

        if approach in ("1", "all"):
            _print_approach1_cell(group)

    # ---- Overall summary ----
    print(f"\n{'=' * 120}")
    print("OVERALL TREATMENT EFFECTS (mean across parameter grid)")
    print(f"{'=' * 120}")

    if approach in ("3", "all"):
        stats3 = _compute_overall_approach3(groups)
        print(f"\n  Approach 3 -- Total System Loss Effect (sys_loss% - passive_sys_loss%)")
        print(f"  {'Arm':<22} {'Mean Sys Effect':>16} {'N':>4}")
        print(f"  {'-' * 22} {'-' * 16} {'-' * 4}")
        for arm in ALL_ARMS:
            if arm not in stats3:
                continue
            effs = stats3[arm]["sys_effects"]
            n = len(effs)
            if n == 0:
                continue
            mean_eff = sum(effs) / n
            print(f"  {arm:<22} {mean_eff:>16.4f} {n:>4}")

    if approach in ("2", "all"):
        stats2 = _compute_overall_approach2(groups)
        passive_lcs = stats2.get("_passive_mean_lc", {}).get("loss_caps", [])
        passive_mean_lc = _mean(passive_lcs)
        print(f"\n  Approach 2 -- Loss/Capital Ratio")
        p_str = f"{passive_mean_lc:.4f}" if passive_mean_lc is not None else "N/A"
        print(f"  passive mean L/C = {p_str}")
        print(f"  {'Arm':<22} {'Mean L/C':>12} {'N':>4}")
        print(f"  {'-' * 22} {'-' * 12} {'-' * 4}")
        for arm in ALL_ARMS:
            if arm not in stats2:
                continue
            lcs = stats2[arm]["loss_caps"]
            n = len(lcs)
            if n == 0:
                continue
            mean_lc = sum(lcs) / n
            print(f"  {arm:<22} {mean_lc:>12.4f} {n:>4}")

    if approach in ("1", "all"):
        stats1 = _compute_overall_approach1(groups)
        print(
            f"\n  Approach 1 -- Capacity-Adjusted Effect "
            f"(cap_adj_interm_loss% - passive_interm_loss%)"
        )
        print(f"  {'Arm':<22} {'Mean Cap-Adj Eff':>16} {'N':>4}")
        print(f"  {'-' * 22} {'-' * 16} {'-' * 4}")
        for arm in ALL_ARMS:
            if arm not in stats1:
                continue
            effs = stats1[arm]["cap_adj_effects"]
            n = len(effs)
            if n == 0:
                continue
            mean_eff = sum(effs) / n
            print(f"  {arm:<22} {mean_eff:>16.4f} {n:>4}")


if __name__ == "__main__":
    main()
