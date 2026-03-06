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


def _make_sweep_config() -> Any:
    """Build the corrected κ sweep configuration."""
    from bilancio.experiments.balanced_comparison import BalancedComparisonConfig

    return BalancedComparisonConfig(
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


def _run_corrected_risk_sweep_impl(*, use_cloud: bool) -> dict[str, Any]:
    """Run the corrected κ sweep with risk-aware traders.

    This function can be triggered from the Modal dashboard.
    Returns the job ID for tracking results.
    """
    from bilancio.experiments.balanced_comparison import BalancedComparisonRunner
    from bilancio.jobs import generate_job_id

    # Generate job ID
    job_id = generate_job_id()
    print(f"Job ID: {job_id}")
    print("=" * 80)

    # Configure sweep
    config = _make_sweep_config()
    executor = None
    execution_mode = "local"
    if use_cloud:
        from bilancio.runners.cloud_executor import CloudExecutor

        executor = CloudExecutor(experiment_id=job_id)
        execution_mode = "cloud"

    # Local fallback uses BalancedComparisonRunner's default LocalExecutor.
    runner = BalancedComparisonRunner(
        config=config,
        out_dir=Path(f"/tmp/{job_id}"),
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
    if use_cloud:
        print("\nExecuting on Modal cloud...")
    else:
        print("\nExecuting locally (Modal unavailable)...")
    print("=" * 80)

    # Run sweep
    results = runner.run_all()

    print("\nSweep Complete!")
    print(f"Job ID: {job_id}")
    print(f"Total pairs: {len(results)}")
    if use_cloud:
        print("Results available in Supabase")
    else:
        print(f"Results written locally under /tmp/{job_id}")

    return {
        "job_id": job_id,
        "total_pairs": len(results),
        "execution_mode": execution_mode,
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

@app.local_entrypoint()
def main() -> None:
    """Deploy and run the sweep, falling back to local execution."""
    try:
        result = run_corrected_risk_sweep.remote(use_cloud=True)
    except Exception:
        # Modal auth/network unavailable — run locally instead.
        result = _run_corrected_risk_sweep_impl(use_cloud=False)
    print(f"\nResult: {result}")
