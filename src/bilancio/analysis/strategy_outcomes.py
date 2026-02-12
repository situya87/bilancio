"""
Strategy outcomes analysis for dealer ring experiments.

Analyzes repayment_events.csv files across runs to determine:
- Which trading strategies are used most
- Default rates per strategy
- Relationship between parameters and strategy success

Plan 023: Default-Aware Instrumentation
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STRATEGIES = ["no_trade", "hold_to_maturity", "sell_before", "round_trip"]


def build_strategy_outcomes_by_run(experiment_root: Path) -> pd.DataFrame:
    """
    Build strategy outcomes for each run.

    For each run in the experiment, reads repayment_events.csv and computes:
    - Count and face value per strategy
    - Default count and face value per strategy
    - Default rate per strategy

    Args:
        experiment_root: Path to experiment directory containing
            aggregate/comparison.csv and active/runs/<run_id>/out/repayment_events.csv

    Returns:
        DataFrame with one row per run, containing:
        - Run parameters (kappa, concentration, mu, outside_mid_ratio, etc.)
        - Comparison metrics (delta_passive, delta_active, trading_effect)
        - Per-strategy metrics (count, face, default_count, default_face, default_rate)
    """
    comp_path = experiment_root / "aggregate" / "comparison.csv"
    if not comp_path.exists():
        logger.warning("comparison.csv not found at %s", comp_path)
        return pd.DataFrame()

    comp = pd.read_csv(comp_path)
    rows: List[Dict[str, Any]] = []

    for _, row in comp.iterrows():
        run_id = row.get("active_run_id", "")
        if not run_id:
            continue

        run_dir = experiment_root / "active" / "runs" / run_id
        rep_path = run_dir / "out" / "repayment_events.csv"

        if not rep_path.exists():
            logger.debug("repayment_events.csv not found for run %s", run_id)
            continue

        try:
            rep = pd.read_csv(rep_path)
        except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning("Failed to read %s: %s", rep_path, e)
            continue

        if rep.empty:
            continue

        # Convert face_value to numeric
        rep["face_value"] = pd.to_numeric(rep["face_value"], errors="coerce").fillna(0)

        # Compute totals
        total_face = rep["face_value"].sum()
        total_liab = len(rep)
        default_face_total = rep.loc[rep["outcome"] == "defaulted", "face_value"].sum()
        default_count_total = (rep["outcome"] == "defaulted").sum()

        # Compute per-strategy metrics
        strat_metrics: Dict[str, Any] = {}
        for strat in STRATEGIES:
            strat_rows = rep[rep["strategy"] == strat]
            face_strat = strat_rows["face_value"].sum()
            default_rows = strat_rows[strat_rows["outcome"] == "defaulted"]

            strat_metrics[f"count_{strat}"] = len(strat_rows)
            strat_metrics[f"face_{strat}"] = float(face_strat)
            strat_metrics[f"default_count_{strat}"] = len(default_rows)
            strat_metrics[f"default_face_{strat}"] = float(default_rows["face_value"].sum())
            strat_metrics[f"default_rate_{strat}"] = (
                float(default_rows["face_value"].sum() / face_strat) if face_strat > 0 else 0.0
            )

        row_out = {
            "run_id": run_id,
            "kappa": row.get("kappa", ""),
            "concentration": row.get("concentration", ""),
            "mu": row.get("mu", ""),
            "outside_mid_ratio": row.get("outside_mid_ratio", ""),
            "seed": row.get("seed", ""),
            "face_value": row.get("face_value", ""),
            "delta_passive": row.get("delta_passive", ""),
            "delta_active": row.get("delta_active", ""),
            "trading_effect": row.get("trading_effect", ""),
            "trading_relief_ratio": row.get("trading_relief_ratio", ""),
            "total_liabilities": total_liab,
            "total_face_value": float(total_face),
            "default_count_total": int(default_count_total),
            "default_face_total": float(default_face_total),
            "default_rate_total": float(default_face_total / total_face) if total_face > 0 else 0.0,
            **strat_metrics,
        }
        rows.append(row_out)

    return pd.DataFrame(rows)


def build_strategy_outcomes_overall(
    by_run_df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate strategy outcomes across runs.

    Groups by parameters and strategy to compute aggregate metrics.

    Args:
        by_run_df: DataFrame from build_strategy_outcomes_by_run()
        group_cols: Columns to group by (default: kappa, concentration, mu, outside_mid_ratio)

    Returns:
        DataFrame with one row per (parameter combination, strategy), containing:
        - total_face: Sum of face values across runs for this strategy
        - default_face: Sum of defaulted face values
        - default_rate: default_face / total_face
        - runs_using_strategy: Count of runs where strategy was used
        - mean_trading_effect: Mean trading effect for these runs
    """
    if by_run_df.empty:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["kappa", "concentration", "mu", "outside_mid_ratio"]

    # Ensure group columns exist
    available_cols = [c for c in group_cols if c in by_run_df.columns]
    if not available_cols:
        logger.warning("No group columns found in DataFrame")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for combo, group in by_run_df.groupby(available_cols, dropna=False):
        # Handle single column case
        if len(available_cols) == 1:
            combo = (combo,)

        combo_dict = dict(zip(available_cols, combo))

        for strat in STRATEGIES:
            face_col = f"face_{strat}"
            default_face_col = f"default_face_{strat}"
            count_col = f"count_{strat}"

            if face_col not in group.columns:
                continue

            face_total = group[face_col].sum()
            default_face = group[default_face_col].sum() if default_face_col in group.columns else 0
            runs_using = (group[count_col] > 0).sum() if count_col in group.columns else 0

            # Get mean trading effect for runs that used this strategy
            if count_col in group.columns and "trading_effect" in group.columns:
                used_mask = group[count_col] > 0
                trading_effects = pd.to_numeric(group.loc[used_mask, "trading_effect"], errors="coerce")
                mean_effect = trading_effects.mean() if not trading_effects.empty else 0.0
            else:
                mean_effect = 0.0

            row_out = {
                "strategy": strat,
                **combo_dict,
                "total_face": float(face_total),
                "default_face": float(default_face),
                "default_rate": float(default_face / face_total) if face_total > 0 else 0.0,
                "runs_using_strategy": int(runs_using),
                "mean_trading_effect": float(mean_effect) if pd.notna(mean_effect) else 0.0,
            }
            rows.append(row_out)

    return pd.DataFrame(rows)


def run_strategy_analysis(experiment_root: Path) -> tuple[Path, Path]:
    """
    Run complete strategy outcomes analysis on an experiment.

    Reads comparison.csv and repayment_events.csv files,
    computes metrics, and writes output CSVs.

    Args:
        experiment_root: Path to experiment directory

    Returns:
        Tuple of (by_run_path, overall_path) for the written files
    """
    logger.info("Running strategy outcomes analysis on %s", experiment_root)

    # Build per-run metrics
    by_run_df = build_strategy_outcomes_by_run(experiment_root)

    if by_run_df.empty:
        logger.warning("No data found for strategy analysis")
        return Path(), Path()

    # Build overall aggregates
    overall_df = build_strategy_outcomes_overall(by_run_df)

    # Write outputs
    aggregate_dir = experiment_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    by_run_path = aggregate_dir / "strategy_outcomes_by_run.csv"
    by_run_df.to_csv(by_run_path, index=False)
    logger.info("Wrote %d rows to %s", len(by_run_df), by_run_path)

    overall_path = aggregate_dir / "strategy_outcomes_overall.csv"
    overall_df.to_csv(overall_path, index=False)
    logger.info("Wrote %d rows to %s", len(overall_df), overall_path)

    return by_run_path, overall_path
