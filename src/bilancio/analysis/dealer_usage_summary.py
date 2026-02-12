"""
Dealer usage summary analysis for dealer ring experiments.

Explains why some runs show no dealer effect by analyzing:
- Trading activity (trade counts, volumes)
- Inventory utilization (dealer capacity, VBT routing)
- System state evolution (debt/money ratio)
- Relationship between trading and outcomes

Plan 023: Default-Aware Instrumentation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_dealer_usage_by_run(experiment_root: Path) -> pd.DataFrame:
    """
    Build dealer usage summary for each run.

    For each run, reads trades.csv, inventory_timeseries.csv, system_state_timeseries.csv,
    and optionally repayment_events.csv to compute:
    - Trade metrics (counts, volumes)
    - Inventory metrics (capacity utilization, VBT routing)
    - System state metrics (debt/money evolution)
    - Repayment metrics (trading by defaulters/repayers)

    Args:
        experiment_root: Path to experiment directory containing
            aggregate/comparison.csv and active/runs/<run_id>/out/*.csv

    Returns:
        DataFrame with one row per run, containing:
        - Run parameters and comparison metrics
        - Trade metrics
        - Inventory metrics
        - System state metrics
        - Repayment metrics (optional)
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
        out_dir = run_dir / "out"

        if not out_dir.exists():
            logger.debug("Output directory not found for run %s", run_id)
            continue

        # Initialize row with run parameters
        row_out: Dict[str, Any] = {
            "run_id": run_id,
            "kappa": row.get("kappa", ""),
            "concentration": row.get("concentration", ""),
            "mu": row.get("mu", ""),
            "outside_mid_ratio": row.get("outside_mid_ratio", ""),
            "delta_passive": row.get("delta_passive", ""),
            "delta_active": row.get("delta_active", ""),
            "trading_effect": row.get("trading_effect", ""),
            "trading_relief_ratio": row.get("trading_relief_ratio", ""),
        }

        # === Trade metrics from trades.csv ===
        trades_path = out_dir / "trades.csv"
        if trades_path.exists():
            try:
                trades = pd.read_csv(trades_path)
                row_out.update(_compute_trade_metrics(trades))
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning("Failed to read trades.csv for %s: %s", run_id, e)
                row_out.update(_empty_trade_metrics())
        else:
            row_out.update(_empty_trade_metrics())

        # === Inventory metrics from inventory_timeseries.csv ===
        inv_path = out_dir / "inventory_timeseries.csv"
        if inv_path.exists():
            try:
                inv = pd.read_csv(inv_path)
                row_out.update(_compute_inventory_metrics(inv))
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning("Failed to read inventory_timeseries.csv for %s: %s", run_id, e)
                row_out.update(_empty_inventory_metrics())
        else:
            row_out.update(_empty_inventory_metrics())

        # === System state metrics from system_state_timeseries.csv ===
        sys_path = out_dir / "system_state_timeseries.csv"
        if sys_path.exists():
            try:
                sys_ts = pd.read_csv(sys_path)
                row_out.update(_compute_system_state_metrics(sys_ts))
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning("Failed to read system_state_timeseries.csv for %s: %s", run_id, e)
                row_out.update(_empty_system_state_metrics())
        else:
            row_out.update(_empty_system_state_metrics())

        # === Repayment metrics from repayment_events.csv (optional) ===
        rep_path = out_dir / "repayment_events.csv"
        if rep_path.exists():
            try:
                rep = pd.read_csv(rep_path)
                row_out.update(_compute_repayment_metrics(rep))
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning("Failed to read repayment_events.csv for %s: %s", run_id, e)
                row_out.update(_empty_repayment_metrics())
        else:
            row_out.update(_empty_repayment_metrics())

        rows.append(row_out)

    return pd.DataFrame(rows)


def _compute_trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """Compute trade metrics from trades.csv."""
    if trades.empty:
        return _empty_trade_metrics()

    dealer_trade_count = len(trades)

    # Identify trader trades (H-entities)
    if "trader_id" in trades.columns:
        trader_trades = trades[trades["trader_id"].str.startswith("H", na=False)]
    else:
        trader_trades = pd.DataFrame()

    trader_dealer_trade_count = len(trader_trades)
    n_traders_using_dealer = trader_trades["trader_id"].nunique() if not trader_trades.empty else 0

    # Compute volumes
    if not trader_trades.empty:
        if "face_value" in trader_trades.columns:
            total_face_traded = pd.to_numeric(trader_trades["face_value"], errors="coerce").sum()
        else:
            total_face_traded = 0.0

        if "price" in trader_trades.columns:
            total_cash_volume = pd.to_numeric(trader_trades["price"], errors="coerce").sum()
        else:
            total_cash_volume = 0.0
    else:
        total_face_traded = 0.0
        total_cash_volume = 0.0

    return {
        "dealer_trade_count": int(dealer_trade_count),
        "trader_dealer_trade_count": int(trader_dealer_trade_count),
        "n_traders_using_dealer": int(n_traders_using_dealer),
        "total_face_traded": float(total_face_traded),
        "total_cash_volume": float(total_cash_volume),
    }


def _empty_trade_metrics() -> Dict[str, Any]:
    """Return empty trade metrics."""
    return {
        "dealer_trade_count": 0,
        "trader_dealer_trade_count": 0,
        "n_traders_using_dealer": 0,
        "total_face_traded": 0.0,
        "total_cash_volume": 0.0,
    }


def _compute_inventory_metrics(inv: pd.DataFrame) -> Dict[str, Any]:
    """Compute inventory metrics from inventory_timeseries.csv."""
    if inv.empty:
        return _empty_inventory_metrics()

    # Group by day and compute daily aggregates
    if "day" not in inv.columns:
        return _empty_inventory_metrics()

    # Check if required columns exist
    has_inventory = "dealer_inventory" in inv.columns
    has_zero = "is_at_zero" in inv.columns
    has_vbt = "hit_vbt_this_step" in inv.columns

    if has_inventory:
        inv["_any_positive"] = pd.to_numeric(inv["dealer_inventory"], errors="coerce") > 0
    else:
        inv["_any_positive"] = False

    if has_zero:
        # is_at_zero might be string "True"/"False" or boolean
        inv["_is_zero"] = inv["is_at_zero"].astype(str).str.lower() == "true"
    else:
        inv["_is_zero"] = False

    if has_vbt:
        inv["_hit_vbt"] = inv["hit_vbt_this_step"].astype(str).str.lower() == "true"
    else:
        inv["_hit_vbt"] = False

    # Aggregate by day
    inv_by_day = inv.groupby("day").agg(
        any_positive=("_any_positive", "any"),
        all_zero=("_is_zero", "all"),
        any_vbt=("_hit_vbt", "any"),
    )

    n_days = len(inv_by_day)
    if n_days == 0:
        return _empty_inventory_metrics()

    dealer_active_fraction = float(inv_by_day["any_positive"].mean())
    dealer_empty_fraction = float(inv_by_day["all_zero"].mean())
    vbt_usage_fraction = float(inv_by_day["any_vbt"].mean())

    return {
        "dealer_active_fraction": dealer_active_fraction,
        "dealer_empty_fraction": dealer_empty_fraction,
        "vbt_usage_fraction": vbt_usage_fraction,
    }


def _empty_inventory_metrics() -> Dict[str, Any]:
    """Return empty inventory metrics."""
    return {
        "dealer_active_fraction": None,
        "dealer_empty_fraction": None,
        "vbt_usage_fraction": None,
    }


def _compute_system_state_metrics(sys_ts: pd.DataFrame) -> Dict[str, Any]:
    """Compute system state metrics from system_state_timeseries.csv."""
    if sys_ts.empty:
        return _empty_system_state_metrics()

    # Convert columns to numeric
    if "debt_to_money" in sys_ts.columns:
        sys_ts["debt_to_money"] = pd.to_numeric(sys_ts["debt_to_money"], errors="coerce")
        mean_debt_to_money = float(sys_ts["debt_to_money"].mean())
        final_debt_to_money = float(sys_ts["debt_to_money"].iloc[-1])
    else:
        mean_debt_to_money = None
        final_debt_to_money = None

    if "total_face_value" in sys_ts.columns:
        sys_ts["total_face_value"] = pd.to_numeric(sys_ts["total_face_value"], errors="coerce")
        total_face0 = float(sys_ts["total_face_value"].iloc[0])
        total_faceT = float(sys_ts["total_face_value"].iloc[-1])
        debt_shrink_rate = (total_face0 - total_faceT) / total_face0 if total_face0 > 0 else 0.0
    else:
        debt_shrink_rate = None

    return {
        "mean_debt_to_money": mean_debt_to_money,
        "final_debt_to_money": final_debt_to_money,
        "debt_shrink_rate": debt_shrink_rate,
    }


def _empty_system_state_metrics() -> Dict[str, Any]:
    """Return empty system state metrics."""
    return {
        "mean_debt_to_money": None,
        "final_debt_to_money": None,
        "debt_shrink_rate": None,
    }


def _compute_repayment_metrics(rep: pd.DataFrame) -> Dict[str, Any]:
    """Compute repayment metrics from repayment_events.csv."""
    if rep.empty:
        return _empty_repayment_metrics()

    # Check if required columns exist
    if "buy_count" not in rep.columns or "sell_count" not in rep.columns:
        return _empty_repayment_metrics()

    if "outcome" not in rep.columns or "face_value" not in rep.columns:
        return _empty_repayment_metrics()

    # Convert to numeric
    rep["face_value"] = pd.to_numeric(rep["face_value"], errors="coerce").fillna(0)
    rep["buy_count"] = pd.to_numeric(rep["buy_count"], errors="coerce").fillna(0)
    rep["sell_count"] = pd.to_numeric(rep["sell_count"], errors="coerce").fillna(0)

    # Mark rows where trader used dealer
    rep["used_dealer"] = (rep["buy_count"] + rep["sell_count"]) > 0

    def frac_used(outcome: str) -> Optional[float]:
        """Compute fraction of face value that traded with dealer for given outcome."""
        sub = rep[rep["outcome"] == outcome]
        if sub.empty:
            return None
        total_face = sub["face_value"].sum()
        if total_face == 0:
            return 0.0
        traded_face = sub.loc[sub["used_dealer"], "face_value"].sum()
        return float(traded_face / total_face)

    frac_defaulted_that_traded = frac_used("defaulted")
    frac_repaid_that_traded = frac_used("repaid")

    return {
        "frac_defaulted_that_traded": frac_defaulted_that_traded,
        "frac_repaid_that_traded": frac_repaid_that_traded,
    }


def _empty_repayment_metrics() -> Dict[str, Any]:
    """Return empty repayment metrics."""
    return {
        "frac_defaulted_that_traded": None,
        "frac_repaid_that_traded": None,
    }


def run_dealer_usage_analysis(experiment_root: Path) -> Path:
    """
    Run complete dealer usage analysis on an experiment.

    Reads various CSV files and computes metrics explaining
    why dealers have or don't have an effect.

    Args:
        experiment_root: Path to experiment directory

    Returns:
        Path to the written dealer_usage_by_run.csv file
    """
    logger.info("Running dealer usage analysis on %s", experiment_root)

    # Build metrics
    df = build_dealer_usage_by_run(experiment_root)

    if df.empty:
        logger.warning("No data found for dealer usage analysis")
        return Path()

    # Write output
    aggregate_dir = experiment_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    output_path = aggregate_dir / "dealer_usage_by_run.csv"
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)

    return output_path
