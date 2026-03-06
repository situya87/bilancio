"""Modal app for triggering sweeps from the web dashboard."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

# Create Modal app
app = modal.App("bilancio-sweep-trigger")

# Use same image as main simulation app
image = modal.Image.debian_slim(python_version="3.11").pip_install("bilancio==0.1.0").env({"PYTHONUNBUFFERED": "1"})


def _run_corrected_risk_sweep_impl() -> dict[str, Any]:
    """Run the corrected κ sweep with risk-aware traders.

    This function can be triggered from the Modal dashboard.
    Returns the job ID for tracking results.
    """
    from bilancio.experiments.balanced_comparison import (
        BalancedComparisonConfig,
        BalancedComparisonRunner,
    )
    from bilancio.jobs import generate_job_id
    from bilancio.runners.cloud_executor import CloudExecutor

    # Generate job ID
    job_id = generate_job_id()
    print(f"Job ID: {job_id}")
    print("=" * 80)

    # Configure sweep
    config = BalancedComparisonConfig(
        n_agents=100,
        maturity_days=10,
        Q_total=Decimal("10000"),
        kappas=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
        concentrations=[Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
        mus=[Decimal("0"), Decimal("0.5"), Decimal("1")],
        outside_mid_ratios=[Decimal("1.0")],
        base_seed=42,
        face_value=Decimal("20"),
        vbt_share_per_bucket=Decimal("0.25"),
        dealer_share_per_bucket=Decimal("0.125"),
        rollover_enabled=True,
        risk_assessment_enabled=True,
        risk_assessment_config={
            "base_risk_premium": "0.02",
            "urgency_sensitivity": "0.30",
            "buy_premium_multiplier": "1.0",
            "lookback_window": 5,
        },
    )

    # Set up cloud executor
    executor = CloudExecutor(experiment_id=job_id)

    # Create runner (cloud-only mode, no local dirs needed)
    runner = BalancedComparisonRunner(
        config=config,
        out_dir=Path(f"/tmp/{job_id}"),  # Unused in cloud-only mode
        executor=executor,
        job_id=job_id,
        enable_supabase=True,
    )

    # Show sweep parameters
    total_pairs = len(config.kappas) * len(config.concentrations) * len(config.mus)
    print("\nSweep Configuration:")
    print(f"  kappa: {[str(k) for k in config.kappas]}")
    print(f"  concentration: {[str(c) for c in config.concentrations]}")
    print(f"  mu: {[str(m) for m in config.mus]}")
    print(f"  Total pairs: {total_pairs} (x2 modes = {total_pairs * 2} runs)")
    print("\nRisk Assessment: ENABLED")
    print("  - Base risk premium: 0.02")
    print("  - Urgency sensitivity: 0.10")
    print("\nExecuting on Modal cloud...")
    print("=" * 80)

    # Run sweep
    results = runner.run_all()

    print("\nSweep Complete!")
    print(f"Job ID: {job_id}")
    print(f"Total pairs: {len(results)}")
    print("Results available in Supabase")

    return {
        "job_id": job_id,
        "total_pairs": len(results),
        "config": {
            "kappas": [str(k) for k in config.kappas],
            "concentrations": [str(c) for c in config.concentrations],
            "mus": [str(m) for m in config.mus],
        },
    }


run_corrected_risk_sweep = app.function(
    image=image,
    timeout=3600,  # 1 hour
    cpu=2,
    memory=4096,
)(_run_corrected_risk_sweep_impl)

setattr(run_corrected_risk_sweep, "local", _run_corrected_risk_sweep_impl)
if not hasattr(run_corrected_risk_sweep, "remote"):
    setattr(run_corrected_risk_sweep, "remote", _run_corrected_risk_sweep_impl)


@app.local_entrypoint()
def main() -> None:
    """Deploy and run the sweep."""
    result = run_corrected_risk_sweep.remote()
    print(f"\nResult: {result}")
