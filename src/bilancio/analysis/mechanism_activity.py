"""
Mechanism activity analysis for post-sweep diagnostics.

Analyzes per-run event data to explain WHY a mechanism (dealer trading,
bank lending, NBFI lending) helped or didn't.  Produces a CSV of activity
metrics per treatment-arm run and a JSON summary with correlations and
key findings.

Dispatches to one of three internal analyzers based on sweep_type:
  - dealer  → reuses build_dealer_usage_by_run + build_strategy_outcomes_by_run
  - bank    → parses events.jsonl for BankLoan* / CB events
  - nbfi    → parses events.jsonl for NonBankLoan* events
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mechanism_activity_analysis(
    experiment_root: Path,
    sweep_type: str,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run mechanism activity analysis and write output files.

    Args:
        experiment_root: Root directory of the experiment (parent of aggregate/).
        sweep_type: One of "dealer", "bank", "nbfi".
        output_dir: Where to write output files.  Defaults to
            ``experiment_root / "aggregate" / "analysis"``.

    Returns:
        Mapping of output name → file path.
    """
    if output_dir is None:
        output_dir = experiment_root / "aggregate" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_path = experiment_root / "aggregate" / "comparison.csv"
    if not comp_path.exists():
        logger.warning("comparison.csv not found at %s", comp_path)
        return {}

    comp_df = pd.read_csv(comp_path)

    analyzers = {
        "dealer": _analyze_dealer_activity,
        "bank": _analyze_bank_activity,
        "nbfi": _analyze_nbfi_activity,
    }
    if sweep_type not in analyzers:
        raise ValueError(f"Unknown sweep_type={sweep_type!r}; expected one of {list(analyzers)}")

    activity_df = analyzers[sweep_type](experiment_root, comp_df)
    if activity_df.empty:
        logger.warning("No activity data produced for sweep_type=%s", sweep_type)
        return {}

    effect_col = {
        "dealer": "trading_effect",
        "bank": "bank_lending_effect",
        "nbfi": "lending_effect",
    }[sweep_type]

    summary = _build_activity_summary(activity_df, sweep_type, effect_col)

    csv_path = output_dir / f"{sweep_type}_activity_by_run.csv"
    json_path = output_dir / f"{sweep_type}_activity_summary.json"

    activity_df.to_csv(csv_path, index=False)
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Activity CSV (%d rows) written to %s", len(activity_df), csv_path)
    logger.info("Activity summary written to %s", json_path)

    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# Dealer activity analyzer
# ---------------------------------------------------------------------------

def _analyze_dealer_activity(experiment_root: Path, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge dealer_usage and strategy_outcomes into one combined DataFrame."""
    from bilancio.analysis.dealer_usage_summary import build_dealer_usage_by_run
    from bilancio.analysis.strategy_outcomes import build_strategy_outcomes_by_run

    usage_df = build_dealer_usage_by_run(experiment_root)
    strategy_df = build_strategy_outcomes_by_run(experiment_root)

    if usage_df.empty and strategy_df.empty:
        return pd.DataFrame()

    if usage_df.empty:
        return strategy_df
    if strategy_df.empty:
        return usage_df

    # Both have run parameters + comparison metrics; merge on run_id
    shared_cols = [c for c in usage_df.columns if c in strategy_df.columns and c != "run_id"]
    strategy_extra = strategy_df.drop(columns=shared_cols, errors="ignore")
    merged = usage_df.merge(strategy_extra, on="run_id", how="outer")
    return merged


# ---------------------------------------------------------------------------
# Bank activity analyzer
# ---------------------------------------------------------------------------

def _analyze_bank_activity(experiment_root: Path, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Parse events.jsonl for each bank_lend run and compute activity metrics."""
    rows: list[dict[str, Any]] = []

    for _, row in comp_df.iterrows():
        run_id = row.get("lend_run_id", "")
        if not run_id:
            continue

        events_path = experiment_root / "bank_lend" / "runs" / run_id / "out" / "events.jsonl"
        if not events_path.exists():
            logger.debug("events.jsonl not found for run %s", run_id)
            continue

        metrics = _parse_bank_events(events_path)
        # Carry over comparison-level columns
        rec: dict[str, Any] = {"run_id": run_id}
        for col in ["kappa", "concentration", "mu", "outside_mid_ratio", "seed",
                     "delta_idle", "delta_lend", "bank_lending_effect",
                     "phi_idle", "phi_lend",
                     "n_defaults_idle", "n_defaults_lend"]:
            if col in row.index:
                rec[col] = row[col]
        rec.update(metrics)
        rows.append(rec)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_bank_events(events_path: Path) -> dict[str, Any]:
    """Extract bank lending metrics from a single run's events.jsonl."""
    loans_issued: list[dict] = []
    loans_repaid: list[dict] = []
    loans_defaulted: list[dict] = []
    cb_freeze_day: int | None = None
    n_payables_settled = 0
    n_payables_rolled = 0

    with events_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = evt.get("kind", "")
            if kind == "BankLoanIssued":
                loans_issued.append(evt)
            elif kind == "BankLoanRepaid":
                loans_repaid.append(evt)
            elif kind == "BankLoanDefault":
                loans_defaulted.append(evt)
            elif kind == "CBLendingFreezeActivated":
                cb_freeze_day = evt.get("day") or evt.get("cutoff_day")
            elif kind == "PayableSettled":
                n_payables_settled += 1
            elif kind == "PayableRolledOver":
                n_payables_rolled += 1

    n_loans = len(loans_issued)
    total_lent = sum(_safe_float(e.get("amount", 0)) for e in loans_issued)
    borrowers = {e.get("borrower") for e in loans_issued}
    banks = {e.get("bank") for e in loans_issued}
    rates = [_safe_float(e.get("rate", 0)) for e in loans_issued]
    days = [e.get("day", 0) for e in loans_issued]

    n_repaid = len(loans_repaid)
    n_defaulted = len(loans_defaulted)
    total_principal_defaulted = sum(_safe_float(e.get("principal", 0)) for e in loans_defaulted)
    total_recovered = sum(_safe_float(e.get("recovered", 0)) for e in loans_defaulted)

    defaulted_borrowers = {e.get("borrower") for e in loans_defaulted}

    return {
        "n_loans": n_loans,
        "total_lent": total_lent,
        "unique_borrowers": len(borrowers),
        "unique_banks": len(banks),
        "avg_loan_size": total_lent / n_loans if n_loans else 0,
        "avg_rate": sum(rates) / len(rates) if rates else 0,
        "min_rate": min(rates) if rates else 0,
        "max_rate": max(rates) if rates else 0,
        "first_loan_day": min(days) if days else None,
        "last_loan_day": max(days) if days else None,
        "n_repaid": n_repaid,
        "n_defaulted": n_defaulted,
        "loan_default_rate": n_defaulted / n_loans if n_loans else 0,
        "lgd": (total_principal_defaulted - total_recovered) / total_principal_defaulted
        if total_principal_defaulted > 0
        else 0,
        "frac_borrowers_defaulted": len(defaulted_borrowers) / len(borrowers)
        if borrowers
        else 0,
        "cb_freeze_day": cb_freeze_day,
        "n_payables_settled": n_payables_settled,
        "n_payables_rolled": n_payables_rolled,
    }


# ---------------------------------------------------------------------------
# NBFI activity analyzer
# ---------------------------------------------------------------------------

def _analyze_nbfi_activity(experiment_root: Path, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Parse events.jsonl for each nbfi_lend run and compute activity metrics."""
    rows: list[dict[str, Any]] = []

    for _, row in comp_df.iterrows():
        run_id = row.get("lend_run_id", "")
        if not run_id:
            continue

        events_path = experiment_root / "nbfi_lend" / "runs" / run_id / "out" / "events.jsonl"
        if not events_path.exists():
            logger.debug("events.jsonl not found for run %s", run_id)
            continue

        metrics = _parse_nbfi_events(events_path)
        rec: dict[str, Any] = {"run_id": run_id}
        for col in ["kappa", "concentration", "mu", "outside_mid_ratio", "seed",
                     "delta_idle", "delta_lend", "lending_effect",
                     "phi_idle", "phi_lend",
                     "n_defaults_idle", "n_defaults_lend"]:
            if col in row.index:
                rec[col] = row[col]
        rec.update(metrics)
        rows.append(rec)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_nbfi_events(events_path: Path) -> dict[str, Any]:
    """Extract NBFI lending metrics from a single run's events.jsonl."""
    loans_created: list[dict] = []
    loans_repaid: list[dict] = []
    loans_defaulted: list[dict] = []
    rejections: list[dict] = []
    agent_defaults: list[dict] = []
    n_payables_settled = 0
    n_payables_rolled = 0

    with events_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = evt.get("kind", "")
            if kind in ("NonBankLoanCreated", "NonBankLoanCreatedPreventive"):
                loans_created.append(evt)
            elif kind == "NonBankLoanRepaid":
                loans_repaid.append(evt)
            elif kind == "NonBankLoanDefaulted":
                loans_defaulted.append(evt)
            elif kind == "NonBankLoanRejectedCoverage":
                rejections.append(evt)
            elif kind == "AgentDefaulted":
                agent_defaults.append(evt)
            elif kind == "PayableSettled":
                n_payables_settled += 1
            elif kind == "PayableRolledOver":
                n_payables_rolled += 1

    n_loans = len(loans_created)
    n_rejected = len(rejections)
    total_applications = n_loans + n_rejected
    total_lent = sum(_safe_float(e.get("amount", 0)) for e in loans_created)
    borrowers = {e.get("borrower_id") for e in loans_created}
    rates = [_safe_float(e.get("rate", 0)) for e in loans_created]
    days = [e.get("day", 0) for e in loans_created]
    rejected_coverages = [_safe_float(e.get("coverage", 0)) for e in rejections]
    rejected_borrowers = {e.get("borrower_id") for e in rejections}

    n_repaid = len(loans_repaid)
    n_loan_defaults = len(loans_defaulted)
    defaulted_borrowers_loan = {e.get("borrower_id") for e in loans_defaulted}

    # Check how many rejected borrowers eventually defaulted
    defaulted_agents = {e.get("agent") or e.get("agent_id") or e.get("name") for e in agent_defaults}
    rejected_that_defaulted = rejected_borrowers & defaulted_agents

    return {
        "n_loans": n_loans,
        "n_rejected": n_rejected,
        "approval_rate": n_loans / total_applications if total_applications else 0,
        "total_lent": total_lent,
        "unique_borrowers": len(borrowers),
        "avg_loan_size": total_lent / n_loans if n_loans else 0,
        "avg_rate": sum(rates) / len(rates) if rates else 0,
        "first_loan_day": min(days) if days else None,
        "last_loan_day": max(days) if days else None,
        "avg_rejected_coverage": sum(rejected_coverages) / len(rejected_coverages)
        if rejected_coverages
        else None,
        "n_repaid": n_repaid,
        "n_loan_defaults": n_loan_defaults,
        "loan_default_rate": n_loan_defaults / n_loans if n_loans else 0,
        "frac_borrowers_defaulted": len(defaulted_borrowers_loan) / len(borrowers)
        if borrowers
        else 0,
        "frac_rejected_defaulted": len(rejected_that_defaulted) / len(rejected_borrowers)
        if rejected_borrowers
        else 0,
        "n_agent_defaults": len(agent_defaults),
        "n_payables_settled": n_payables_settled,
        "n_payables_rolled": n_payables_rolled,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_activity_summary(
    df: pd.DataFrame,
    sweep_type: str,
    effect_col: str,
) -> dict[str, Any]:
    """Build a summary dict with breakdowns, correlations, and key findings."""
    summary: dict[str, Any] = {
        "sweep_type": sweep_type,
        "n_runs": len(df),
    }

    # --- By-kappa breakdown ---
    if "kappa" in df.columns and effect_col in df.columns:
        kappa_groups = df.groupby("kappa")
        by_kappa: dict[str, Any] = {}
        for kappa_val, grp in kappa_groups:
            entry: dict[str, Any] = {
                "n_runs": len(grp),
                "mean_effect": _safe_mean(grp, effect_col),
            }
            # Add sweep-specific means
            for col in _activity_metric_cols(sweep_type):
                if col in grp.columns:
                    entry[f"mean_{col}"] = _safe_mean(grp, col)
            by_kappa[str(kappa_val)] = entry
        summary["by_kappa"] = by_kappa

    # --- By-kappa×mu breakdown ---
    if all(c in df.columns for c in ["kappa", "mu", effect_col]):
        kappa_mu_groups = df.groupby(["kappa", "mu"])
        by_kappa_mu: dict[str, Any] = {}
        for (kappa_val, mu_val), grp in kappa_mu_groups:
            key = f"k={kappa_val}_mu={mu_val}"
            entry = {
                "n_runs": len(grp),
                "mean_effect": _safe_mean(grp, effect_col),
            }
            for col in _activity_metric_cols(sweep_type):
                if col in grp.columns:
                    entry[f"mean_{col}"] = _safe_mean(grp, col)
            by_kappa_mu[key] = entry
        summary["by_kappa_mu"] = by_kappa_mu

    # --- Correlations with treatment effect ---
    if effect_col in df.columns:
        correlations: dict[str, float | None] = {}
        effect_series = pd.to_numeric(df[effect_col], errors="coerce")
        for col in _activity_metric_cols(sweep_type):
            if col in df.columns:
                col_series = pd.to_numeric(df[col], errors="coerce")
                valid = effect_series.notna() & col_series.notna()
                if valid.sum() >= 3:
                    r = effect_series[valid].corr(col_series[valid])
                    correlations[col] = float(r) if pd.notna(r) else None
                else:
                    correlations[col] = None
        summary["correlations_with_effect"] = correlations

    # --- Effectiveness breakdown ---
    if effect_col in df.columns:
        effect_vals = pd.to_numeric(df[effect_col], errors="coerce")
        n_helps = int((effect_vals > 0.001).sum())
        n_hurts = int((effect_vals < -0.001).sum())
        n_neutral = int(len(effect_vals) - n_helps - n_hurts)
        summary["effectiveness"] = {
            "helps": n_helps,
            "hurts": n_hurts,
            "neutral": n_neutral,
            "helps_pct": n_helps / len(effect_vals) * 100 if len(effect_vals) else 0,
        }

    # --- Key findings ---
    summary["key_findings"] = _generate_findings(df, sweep_type, effect_col)

    return summary


def _activity_metric_cols(sweep_type: str) -> list[str]:
    """Return the activity-specific metric columns to correlate / summarize."""
    if sweep_type == "dealer":
        return [
            "dealer_trade_count",
            "total_face_traded",
            "total_cash_volume",
            "dealer_active_fraction",
            "vbt_usage_fraction",
            "n_traders_using_dealer",
            "mean_debt_to_money",
            "debt_shrink_rate",
            "frac_defaulted_that_traded",
            "frac_repaid_that_traded",
            "default_rate_total",
        ]
    elif sweep_type == "bank":
        return [
            "n_loans",
            "total_lent",
            "unique_borrowers",
            "unique_banks",
            "avg_loan_size",
            "avg_rate",
            "loan_default_rate",
            "lgd",
            "frac_borrowers_defaulted",
            "n_payables_settled",
            "n_payables_rolled",
        ]
    elif sweep_type == "nbfi":
        return [
            "n_loans",
            "n_rejected",
            "approval_rate",
            "total_lent",
            "unique_borrowers",
            "avg_loan_size",
            "avg_rate",
            "loan_default_rate",
            "frac_borrowers_defaulted",
            "frac_rejected_defaulted",
            "n_agent_defaults",
            "n_payables_settled",
            "n_payables_rolled",
        ]
    return []


def _generate_findings(df: pd.DataFrame, sweep_type: str, effect_col: str) -> list[str]:
    """Auto-generate human-readable key findings."""
    findings: list[str] = []
    if df.empty:
        return findings

    effect_vals = pd.to_numeric(df.get(effect_col, pd.Series(dtype=float)), errors="coerce")
    mean_effect = effect_vals.mean() if not effect_vals.isna().all() else None

    if mean_effect is not None:
        direction = "reduces" if mean_effect > 0 else "increases"
        findings.append(
            f"Mean {effect_col} = {mean_effect:.4f} "
            f"({sweep_type} mechanism {direction} defaults on average)"
        )

    if sweep_type == "dealer":
        if "dealer_trade_count" in df.columns:
            mean_trades = df["dealer_trade_count"].mean()
            zero_trade_runs = (df["dealer_trade_count"] == 0).sum()
            findings.append(f"Mean dealer trades per run: {mean_trades:.1f}")
            if zero_trade_runs > 0:
                findings.append(
                    f"{zero_trade_runs}/{len(df)} runs had zero dealer trades"
                )
        if "dealer_active_fraction" in df.columns:
            mean_active = df["dealer_active_fraction"].mean()
            findings.append(f"Mean dealer active fraction: {mean_active:.2f}")

    elif sweep_type == "bank":
        if "n_loans" in df.columns:
            mean_loans = df["n_loans"].mean()
            zero_loan_runs = (df["n_loans"] == 0).sum()
            findings.append(f"Mean loans per run: {mean_loans:.1f}")
            if zero_loan_runs > 0:
                findings.append(
                    f"{zero_loan_runs}/{len(df)} runs had zero bank loans"
                )
        if "loan_default_rate" in df.columns:
            mean_ldr = df["loan_default_rate"].mean()
            findings.append(f"Mean loan default rate: {mean_ldr:.2%}")
        if "total_lent" in df.columns:
            mean_lent = df["total_lent"].mean()
            findings.append(f"Mean total lent per run: {mean_lent:.0f}")

    elif sweep_type == "nbfi":
        if "approval_rate" in df.columns:
            mean_approval = df["approval_rate"].mean()
            findings.append(f"Mean approval rate: {mean_approval:.1%}")
        if "n_rejected" in df.columns:
            mean_rejected = df["n_rejected"].mean()
            findings.append(f"Mean rejections per run: {mean_rejected:.1f}")
        if "frac_rejected_defaulted" in df.columns:
            mean_rdef = df["frac_rejected_defaulted"].mean()
            findings.append(
                f"Mean fraction of rejected borrowers that defaulted: {mean_rdef:.1%}"
            )
        if "n_loans" in df.columns:
            mean_loans = df["n_loans"].mean()
            findings.append(f"Mean loans per run: {mean_loans:.1f}")

    # Top correlations
    if effect_col in df.columns:
        corrs: list[tuple[str, float]] = []
        effect_series = pd.to_numeric(df[effect_col], errors="coerce")
        for col in _activity_metric_cols(sweep_type):
            if col in df.columns:
                col_series = pd.to_numeric(df[col], errors="coerce")
                valid = effect_series.notna() & col_series.notna()
                if valid.sum() >= 3:
                    r = float(effect_series[valid].corr(col_series[valid]))
                    if pd.notna(r):
                        corrs.append((col, r))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        if corrs:
            top = corrs[0]
            findings.append(
                f"Top correlate with {effect_col}: {top[0]} (r={top[1]:.2f})"
            )

    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    """Compute mean of a column, returning None if not possible."""
    try:
        series = pd.to_numeric(df[col], errors="coerce")
        val = series.mean()
        return float(val) if pd.notna(val) else None
    except Exception:
        return None
