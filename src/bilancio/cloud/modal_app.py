"""Modal app definition for cloud simulation execution.

This module defines the Modal app, container image, volume, and remote
function for running bilancio simulations in the cloud.

Simulations run on Modal compute metrics locally and save results directly
to Supabase, eliminating the need to download artifacts.
"""
# mypy: disable-error-code="union-attr"

from __future__ import annotations

from datetime import UTC
from typing import Any

import modal

EXTERNAL_OPERATION_ERRORS = (
    FileNotFoundError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)

# Define the Modal app
app = modal.App("bilancio-simulations")

# Create a persistent volume for storing results
# Results persist across function invocations
results_volume = modal.Volume.from_name(
    "bilancio-results",
    create_if_missing=True,
)

# Supabase secrets for direct database writes from Modal (optional)
# If the secret exists, it will be injected; otherwise Supabase saves are skipped.
# Create with: modal secret create supabase \
#   BILANCIO_SUPABASE_URL=https://xxx.supabase.co \
#   BILANCIO_SUPABASE_ANON_KEY=eyJ...
_secrets: list[modal.Secret] = []

# Define the container image with all bilancio dependencies
# We add the local source code to the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pydantic>=2.0.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "jinja2>=3.0",
        "click>=8.0",
        "xkcdpass>=1.19.0",
        "plotly>=5.0.0",  # Required by HTML export in run_scenario
        "supabase>=2.0.0",  # For direct Supabase writes
    )
    .add_local_python_source("bilancio")
)

RESULTS_MOUNT_PATH = "/results"


def _compute_dealer_vbt_loss(dealer_metrics_path: str | None) -> float:
    """Extract dealer+VBT loss from dealer_metrics.json, returning 0 if unavailable."""
    if not dealer_metrics_path:
        return 0.0
    import json
    from pathlib import Path

    path = Path(dealer_metrics_path)
    if not path.exists():
        return 0.0
    try:
        with path.open() as f:
            data = json.load(f)
        pnl = data.get("dealer_total_pnl")
        if pnl is not None:
            return max(0.0, -float(pnl))
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return 0.0


def compute_metrics_from_events(events_path: str, dealer_metrics_path: str | None = None) -> dict[str, Any]:
    """Compute metrics from events.jsonl file.

    Args:
        events_path: Path to events.jsonl file.
        dealer_metrics_path: Optional path to dealer_metrics.json for intermediary loss computation.

    Returns:
        Dict with all global metrics from summarize_day_metrics, properly serialized.
    """
    import json
    from decimal import Decimal

    def to_serializable(val: Any) -> Any:
        """Convert Decimal to float for JSON serialization."""
        if isinstance(val, Decimal):
            return float(val)
        return val

    # Read events
    events = []
    with open(events_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

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
            "raw_metrics": {},
        }

    # Use bilancio's metrics computation
    from bilancio.analysis.report import (
        compute_day_metrics,
        compute_run_level_metrics,
        summarize_day_metrics,
    )

    result = compute_day_metrics(events=events, balances_rows=None, day_list=None)
    summary = summarize_day_metrics(result["day_metrics"])

    # Compute run-level metrics (cascade, contagion, CB stress)
    run_level = compute_run_level_metrics(events)

    # Convert all Decimal values to float for JSON serialization
    serializable_summary = {k: to_serializable(v) for k, v in summary.items()}

    return {
        "delta_total": to_serializable(summary.get("delta_total")),
        "phi_total": to_serializable(summary.get("phi_total")),
        "time_to_stability": int(summary.get("max_day") or 0),
        "max_G_t": to_serializable(summary.get("max_G_t")),
        "alpha_1": to_serializable(summary.get("alpha_1")),
        "Mpeak_1": to_serializable(summary.get("Mpeak_1")),
        "v_1": to_serializable(summary.get("v_1")),
        "HHIplus_1": to_serializable(summary.get("HHIplus_1")),
        "n_defaults": run_level["n_defaults"],
        "cascade_fraction": to_serializable(run_level["cascade_fraction"]),
        "cb_loans_created_count": run_level["cb_loans_created_count"],
        "cb_interest_total_paid": run_level["cb_interest_total_paid"],
        "cb_loans_outstanding_pre_final": run_level["cb_loans_outstanding_pre_final"],
        "bank_defaults_final": run_level["bank_defaults_final"],
        "cb_reserves_initial": run_level["cb_reserves_initial"],
        "cb_reserves_final": run_level["cb_reserves_final"],
        "cb_reserve_destruction_pct": run_level["cb_reserve_destruction_pct"],
        # Banking-specific default metrics (Plan 039)
        "delta_bank": to_serializable(run_level.get("delta_bank")),
        "deposit_loss_gross": run_level.get("deposit_loss_gross", 0),
        "deposit_loss_pct": to_serializable(run_level.get("deposit_loss_pct")),
        "total_deposits_created": run_level.get("total_deposits_created", 0),
        "bank_obligations_created": run_level.get("bank_obligations_created", 0),
        "bank_writeoffs": run_level.get("bank_writeoffs", 0),
        "payable_default_loss": run_level.get("payable_default_loss", 0),
        "total_loss": run_level.get("total_loss", 0),
        "S_total": summary.get("S_total", 0),
        "total_loss_pct": (
            run_level.get("total_loss", 0) / summary["S_total"]
            if summary.get("S_total") and summary["S_total"] > 0
            else None
        ),
        # Intermediary loss metrics
        "nbfi_loan_loss": run_level.get("nbfi_loan_loss", 0),
        "bank_credit_loss": run_level.get("bank_credit_loss", 0),
        "cb_backstop_loss": run_level.get("cb_backstop_loss", 0),
        "dealer_vbt_loss": _compute_dealer_vbt_loss(dealer_metrics_path),
        "intermediary_loss_total": (
            run_level.get("nbfi_loan_loss", 0)
            + run_level.get("bank_credit_loss", 0)
            + run_level.get("cb_backstop_loss", 0)
            + _compute_dealer_vbt_loss(dealer_metrics_path)
        ),
        "raw_metrics": serializable_summary,
    }


class SupabaseCredentialsError(Exception):
    """Raised when Supabase credentials are missing or invalid."""

    pass


def save_run_to_supabase(
    run_id: str,
    job_id: str,
    status: str,
    metrics: dict[str, Any],
    params: dict[str, Any],
    execution_time_ms: int,
    modal_call_id: str | None,
    modal_volume_path: str,
    error: str | None = None,
) -> bool:
    """Save run and metrics to Supabase.

    Args:
        run_id: Unique run identifier.
        job_id: Job/experiment ID.
        status: Run status (completed, failed).
        metrics: Computed metrics dict.
        params: Run parameters (kappa, concentration, etc.).
        execution_time_ms: Execution time in milliseconds.
        modal_call_id: Modal function call ID.
        modal_volume_path: Path to artifacts in Modal volume.
        error: Error message if failed.

    Returns:
        True if save succeeded.

    Raises:
        SupabaseCredentialsError: If Supabase credentials are not configured.
    """
    import os
    from datetime import datetime

    try:
        from supabase import create_client

        url = os.environ.get("BILANCIO_SUPABASE_URL")
        key = os.environ.get("BILANCIO_SUPABASE_ANON_KEY")

        if not url or not key:
            # Log available env vars for debugging (without exposing values)
            available_vars = [k for k in os.environ.keys() if "SUPABASE" in k.upper()]
            raise SupabaseCredentialsError(
                f"Supabase credentials not configured! "
                f"Missing: {'BILANCIO_SUPABASE_URL' if not url else ''} "
                f"{'BILANCIO_SUPABASE_ANON_KEY' if not key else ''}. "
                f"Available SUPABASE env vars: {available_vars}. "
                f"Ensure Modal secret 'supabase' has the correct keys."
            )

        client = create_client(url, key)
        now = datetime.now(UTC).isoformat()

        # Build runs table row
        runs_row: dict[str, Any] = {
            "run_id": run_id,
            "job_id": job_id,
            "status": status,
            "created_at": now,
            "completed_at": now if status in ("completed", "failed") else None,
            "execution_time_ms": execution_time_ms,
            "modal_call_id": modal_call_id,
            "modal_volume_path": modal_volume_path,
            "error": error,
        }

        # Add parameters
        param_columns = {"kappa", "concentration", "mu", "outside_mid_ratio", "seed", "regime"}
        for param, value in params.items():
            if param in param_columns:
                if param == "seed":
                    runs_row[param] = int(value) if value is not None else None
                elif param == "regime":
                    runs_row[param] = value
                else:
                    runs_row[param] = (
                        float(value) if isinstance(value, int | float | str) else value
                    )

        # Upsert run
        client.table("runs").upsert(runs_row, on_conflict="run_id").execute()
        print(f"Saved run {run_id} to Supabase")

        # Build and save metrics if we have them
        if metrics.get("delta_total") is not None or metrics.get("phi_total") is not None:
            # Build raw_metrics with all global metrics from summarize_day_metrics
            raw_metrics = metrics.get("raw_metrics", {})
            # Ensure all metrics are in raw_metrics even if they came from top-level
            raw_metrics.update(
                {
                    "delta_total": metrics.get("delta_total"),
                    "phi_total": metrics.get("phi_total"),
                    "time_to_stability": metrics.get("time_to_stability"),
                    "max_G_t": metrics.get("max_G_t"),
                    "alpha_1": metrics.get("alpha_1"),
                    "Mpeak_1": metrics.get("Mpeak_1"),
                    "v_1": metrics.get("v_1"),
                    "HHIplus_1": metrics.get("HHIplus_1"),
                }
            )

            metrics_row = {
                "run_id": run_id,
                "job_id": job_id,
                "delta_total": metrics.get("delta_total"),
                "phi_total": metrics.get("phi_total"),
                "time_to_stability": metrics.get("time_to_stability"),
                "raw_metrics": raw_metrics,
            }

            # Check if metrics exist
            existing = client.table("metrics").select("id").eq("run_id", run_id).execute()

            if existing.data:
                client.table("metrics").update(metrics_row).eq("run_id", run_id).execute()
            else:
                client.table("metrics").insert(metrics_row).execute()

            print(
                f"Saved metrics for {run_id}: δ={metrics.get('delta_total')}, φ={metrics.get('phi_total')}"
            )

        return True

    except SupabaseCredentialsError:
        # Log but don't fail - Supabase is optional
        print(f"Supabase not configured, skipping save for run {run_id}")
        return False
    except EXTERNAL_OPERATION_ERRORS as e:
        # Log other errors but don't fail the run - Supabase save is secondary
        print(f"WARNING: Failed to save to Supabase: {e}")
        print(f"Run {run_id} completed but metrics not persisted to Supabase!")
        return False


@app.function(
    image=image,
    volumes={RESULTS_MOUNT_PATH: results_volume},
    secrets=_secrets,
    timeout=7200,  # 2 hours max per simulation
    memory=2048,  # 2GB RAM
)
def run_simulation(
    scenario_config: dict[str, Any],
    run_id: str,
    experiment_id: str,
    options: dict[str, Any],
    job_id: str = "",
) -> dict[str, Any]:
    """Run a single simulation in the cloud.

    Args:
        scenario_config: Full scenario YAML as dict
        run_id: Unique identifier for this run
        experiment_id: Groups related runs together
        options: RunOptions as dict (mode, max_days, quiet_days, etc.)
        job_id: Bilancio job ID (for logging/tracking)

    Returns:
        Dict with status, artifact paths (relative to volume), metrics, error
    """
    import time
    from pathlib import Path

    import yaml

    start_time = time.time()

    # Log job info prominently at the start (visible in Modal logs)
    import sys

    modal_call_id = modal.current_function_call_id()
    print("=" * 60, flush=True)
    print("BILANCIO SIMULATION", flush=True)
    print(f"  Job ID:      {job_id or 'N/A'}", flush=True)
    print(f"  Run ID:      {run_id}", flush=True)
    print(f"  Experiment:  {experiment_id}", flush=True)
    print(f"  Modal Call:  {modal_call_id}", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    # Create output directory on the volume
    run_dir = Path(RESULTS_MOUNT_PATH) / experiment_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "out"
    out_dir.mkdir(exist_ok=True)

    # Write scenario YAML
    scenario_path = run_dir / "scenario.yaml"
    scenario_path.write_text(yaml.dump(scenario_config, default_flow_style=False))

    # Extract run parameters from options for Supabase
    run_params = {
        "kappa": options.get("kappa"),
        "concentration": options.get("concentration"),
        "mu": options.get("mu"),
        "outside_mid_ratio": options.get("outside_mid_ratio"),
        "seed": options.get("seed"),
        "regime": options.get("regime", ""),
    }

    try:
        # Import bilancio inside the function (container context)
        from bilancio.ui.run import run_scenario

        # Deserialize performance config if present
        perf_dict = options.get("performance")
        performance = None
        if perf_dict:
            from bilancio.core.performance import PerformanceConfig

            performance = PerformanceConfig.from_dict(perf_dict)

        # Execute simulation
        run_scenario(
            path=scenario_path,
            mode=options.get("mode", "until_stable"),
            max_days=options.get("max_days", 90),
            quiet_days=options.get("quiet_days", 2),
            show=options.get("show_events", "detailed"),
            check_invariants=options.get("check_invariants", "daily"),
            default_handling=options.get("default_handling", "fail-fast"),
            export={
                "balances_csv": str(out_dir / "balances.csv"),
                "events_jsonl": str(out_dir / "events.jsonl"),
            },
            html_output=run_dir / "run.html",
            t_account=options.get("t_account", False),
            detailed_dealer_logging=options.get("detailed_dealer_logging", False),
            run_id=run_id,
            regime=options.get("regime", ""),
            performance=performance,
        )

        # Commit changes to volume
        results_volume.commit()

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Compute metrics from events
        events_path = out_dir / "events.jsonl"
        dealer_metrics_path = out_dir / "dealer_metrics.json"
        metrics = {}
        if events_path.exists():
            print("Computing metrics from events...", flush=True)
            metrics = compute_metrics_from_events(
                str(events_path),
                dealer_metrics_path=str(dealer_metrics_path) if dealer_metrics_path.exists() else None,
            )
            print(
                f"Metrics: δ={metrics.get('delta_total')}, φ={metrics.get('phi_total')}", flush=True
            )

        # Build artifact paths (relative to run directory within volume)
        artifacts = {
            "scenario_yaml": "scenario.yaml",
            "events_jsonl": "out/events.jsonl",
            "balances_csv": "out/balances.csv",
            "run_html": "run.html",
        }

        modal_volume_path = f"{experiment_id}/runs/{run_id}"

        # Save to Supabase
        save_run_to_supabase(
            run_id=run_id,
            job_id=job_id,
            status="completed",
            metrics=metrics,
            params=run_params,
            execution_time_ms=execution_time_ms,
            modal_call_id=modal_call_id,
            modal_volume_path=modal_volume_path,
        )

        return {
            "run_id": run_id,
            "status": "completed",
            "storage_type": "modal_volume",
            "storage_base": modal_volume_path,
            "artifacts": artifacts,
            "execution_time_ms": execution_time_ms,
            "error": None,
            "modal_call_id": modal_call_id,
            "metrics": metrics,  # Include computed metrics in return
        }

    except EXTERNAL_OPERATION_ERRORS as e:
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Still commit to preserve any partial output
        results_volume.commit()

        modal_volume_path = f"{experiment_id}/runs/{run_id}"

        # Save failed run to Supabase
        save_run_to_supabase(
            run_id=run_id,
            job_id=job_id,
            status="failed",
            metrics={},
            params=run_params,
            execution_time_ms=execution_time_ms,
            modal_call_id=modal_call_id,
            modal_volume_path=modal_volume_path,
            error=str(e),
        )

        return {
            "run_id": run_id,
            "status": "failed",
            "storage_type": "modal_volume",
            "storage_base": modal_volume_path,
            "artifacts": {},
            "execution_time_ms": execution_time_ms,
            "error": str(e),
            "modal_call_id": modal_call_id,
            "metrics": {},
        }


@app.function(
    image=image,
    secrets=_secrets,
    timeout=300,  # 5 minutes for aggregate computation
    memory=1024,
)
def compute_aggregate_metrics(
    job_id: str,
    run_ids: list[str],
) -> dict[str, Any]:
    """Compute aggregate/comparison metrics for a sweep and save to Supabase.

    This function is called after all runs complete to compute:
    - Trading effect (δ_passive - δ_active) for each parameter combination
    - Summary statistics across the sweep

    Args:
        job_id: The job/experiment ID.
        run_ids: List of run IDs to aggregate.

    Returns:
        Dict with aggregate metrics and status.
    """
    import os
    from collections import defaultdict

    from supabase import create_client

    url = os.environ.get("BILANCIO_SUPABASE_URL")
    key = os.environ.get("BILANCIO_SUPABASE_ANON_KEY")

    if not url or not key:
        # Log available env vars for debugging (without exposing values)
        available_vars = [k for k in os.environ.keys() if "SUPABASE" in k.upper()]
        raise SupabaseCredentialsError(
            f"Supabase credentials not configured for aggregate metrics! "
            f"Missing: {'BILANCIO_SUPABASE_URL' if not url else ''} "
            f"{'BILANCIO_SUPABASE_ANON_KEY' if not key else ''}. "
            f"Available SUPABASE env vars: {available_vars}. "
            f"Ensure Modal secret 'supabase' has the correct keys."
        )

    try:
        client = create_client(url, key)

        # Fetch all runs with metrics for this job
        result = (
            client.table("runs")
            .select("run_id, kappa, concentration, mu, outside_mid_ratio, seed, regime, metrics(*)")
            .eq("job_id", job_id)
            .in_("run_id", run_ids)
            .execute()
        )

        if not result.data:
            return {"status": "error", "error": "No runs found"}

        # Group runs by parameters (excluding regime)
        grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
            lambda: {"passive": None, "active": None}
        )

        for row in result.data:
            row_dict: dict[str, Any] = row  # type: ignore[assignment]
            param_key: tuple[Any, ...] = (
                row_dict.get("kappa"),
                row_dict.get("concentration"),
                row_dict.get("mu"),
                row_dict.get("outside_mid_ratio"),
                row_dict.get("seed"),
            )
            regime = row_dict.get("regime", "")
            metrics_data = row_dict.get("metrics")

            if metrics_data:
                # metrics is a list from join, take first
                if isinstance(metrics_data, list) and metrics_data:
                    metrics_entry: dict[str, Any] = metrics_data[0]
                else:
                    metrics_entry = metrics_data

                delta = metrics_entry.get("delta_total")
                phi = metrics_entry.get("phi_total")

                if "passive" in regime or regime == "":
                    grouped[param_key]["passive"] = {
                        "delta": delta,
                        "phi": phi,
                        "run_id": row_dict["run_id"],
                    }
                elif "active" in regime:
                    grouped[param_key]["active"] = {
                        "delta": delta,
                        "phi": phi,
                        "run_id": row_dict["run_id"],
                    }

        # Compute trading effects
        comparisons: list[dict[str, Any]] = []
        for params, runs in grouped.items():
            if runs.get("passive") and runs.get("active"):
                d_passive = runs["passive"]["delta"]
                d_active = runs["active"]["delta"]

                if d_passive is not None and d_active is not None:
                    trading_effect = d_passive - d_active

                    comparisons.append(
                        {
                            "kappa": params[0],
                            "concentration": params[1],
                            "mu": params[2],
                            "outside_mid_ratio": params[3],
                            "seed": params[4],
                            "delta_passive": d_passive,
                            "delta_active": d_active,
                            "phi_passive": runs["passive"].get("phi"),
                            "phi_active": runs["active"].get("phi"),
                            "trading_effect": trading_effect,
                            "passive_run_id": runs["passive"]["run_id"],
                            "active_run_id": runs["active"]["run_id"],
                        }
                    )

        # Compute summary statistics
        import statistics
        from datetime import datetime

        if comparisons:
            effects = [c["trading_effect"] for c in comparisons]
            deltas_passive = [
                c["delta_passive"] for c in comparisons if c["delta_passive"] is not None
            ]
            deltas_active = [
                c["delta_active"] for c in comparisons if c["delta_active"] is not None
            ]
            phis_passive: list[float] = [
                c["phi_passive"] for c in comparisons if c.get("phi_passive") is not None
            ]
            phis_active: list[float] = [
                c["phi_active"] for c in comparisons if c.get("phi_active") is not None
            ]

            # Compute standard deviation if we have enough data
            std_effect = statistics.stdev(effects) if len(effects) > 1 else 0.0

            summary = {
                "n_comparisons": len(comparisons),
                "n_pairs": len(comparisons),
                "mean_trading_effect": sum(effects) / len(effects),
                "min_trading_effect": min(effects),
                "max_trading_effect": max(effects),
                "std_trading_effect": std_effect,
                "positive_effects": sum(1 for e in effects if e > 0.001),  # dealers help
                "negative_effects": sum(1 for e in effects if e < -0.001),  # dealers hurt
                "neutral_effects": sum(1 for e in effects if -0.001 <= e <= 0.001),
                "mean_delta_passive": sum(deltas_passive) / len(deltas_passive)
                if deltas_passive
                else None,
                "mean_delta_active": sum(deltas_active) / len(deltas_active)
                if deltas_active
                else None,
                "mean_phi_passive": sum(phis_passive) / len(phis_passive) if phis_passive else None,
                "mean_phi_active": sum(phis_active) / len(phis_active) if phis_active else None,
            }
        else:
            summary = {"n_comparisons": 0, "n_pairs": 0}

        # Save to job_metrics table
        job_metrics_row = {
            "job_id": job_id,
            "n_comparisons": summary.get("n_comparisons"),
            "n_pairs": summary.get("n_pairs"),
            "mean_trading_effect": summary.get("mean_trading_effect"),
            "min_trading_effect": summary.get("min_trading_effect"),
            "max_trading_effect": summary.get("max_trading_effect"),
            "std_trading_effect": summary.get("std_trading_effect"),
            "positive_effects": summary.get("positive_effects"),
            "negative_effects": summary.get("negative_effects"),
            "neutral_effects": summary.get("neutral_effects"),
            "mean_delta_passive": summary.get("mean_delta_passive"),
            "mean_delta_active": summary.get("mean_delta_active"),
            "mean_phi_passive": summary.get("mean_phi_passive"),
            "mean_phi_active": summary.get("mean_phi_active"),
            "raw_summary": summary,
        }

        # Upsert job_metrics (insert or update if exists)
        try:
            existing = client.table("job_metrics").select("id").eq("job_id", job_id).execute()
            if existing.data:
                client.table("job_metrics").update(job_metrics_row).eq("job_id", job_id).execute()
            else:
                client.table("job_metrics").insert(job_metrics_row).execute()
            print(f"Saved job_metrics for {job_id}")
        except EXTERNAL_OPERATION_ERRORS as e:
            print(f"Warning: Failed to save job_metrics: {e}")

        # Update job status
        client.table("jobs").update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "completed",
            }
        ).eq("job_id", job_id).execute()

        print(f"Aggregate metrics for job {job_id}:")
        print(f"  Comparisons: {summary.get('n_comparisons', 0)}")
        print(
            f"  Mean trading effect: {summary.get('mean_trading_effect', 'N/A'):.4f}"
            if summary.get("mean_trading_effect")
            else "  Mean trading effect: N/A"
        )
        print(
            f"  Positive effects: {summary.get('positive_effects', 0)} | Negative: {summary.get('negative_effects', 0)}"
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "summary": summary,
            "comparisons": comparisons,
        }

    except EXTERNAL_OPERATION_ERRORS as e:
        print(f"Failed to compute aggregate metrics: {e}")
        return {"status": "error", "error": str(e)}


@app.function(
    image=image,
    timeout=60,
    memory=256,
)
def health_check() -> dict[str, Any]:
    """Simple health check to verify Modal deployment.

    Returns:
        Dict with status and bilancio version info.
    """
    try:
        import bilancio

        return {
            "status": "ok",
            "bilancio_available": True,
            "version": getattr(bilancio, "__version__", "unknown"),
        }
    except ImportError as e:
        return {
            "status": "error",
            "bilancio_available": False,
            "error": str(e),
        }


# Local entrypoint for testing
@app.local_entrypoint()
def main() -> None:
    """Test the Modal deployment with a health check."""
    print("Running health check...")
    result = health_check.remote()
    print(f"Health check result: {result}")

    if result["status"] == "ok":
        print("Modal deployment is healthy!")
    else:
        print(f"Modal deployment has issues: {result.get('error', 'unknown')}")
