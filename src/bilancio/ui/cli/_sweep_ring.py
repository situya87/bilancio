"""Sweep commands for ring and dealer-comparison experiments."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

import click

from bilancio.experiments.comparison import ComparisonSweepConfig, ComparisonSweepRunner
from bilancio.experiments.ring import RingSweepConfig, load_ring_sweep_config
from bilancio.jobs import JobConfig, JobManager

from ._common import (
    CLI_HANDLED_ERRORS,
    as_decimal_list,
    build_performance_config,
    console,
    parameter_uses_default,
)


def _deps() -> Any:
    return import_module("bilancio.ui.cli.sweep")


@click.command("ring")
@click.option("--config", type=click.Path(path_type=Path), default=None, help="Path to sweep config YAML")
@click.option("--out-dir", type=click.Path(path_type=Path), default=None, help="Base output directory")
@click.option("--cloud", is_flag=True, help="Run simulations on Modal cloud")
@click.option("--grid/--no-grid", default=True, help="Run coarse grid sweep")
@click.option("--kappas", type=str, default="0.25,0.5,1,2,4", help="Comma list for grid kappa values")
@click.option(
    "--concentrations",
    type=str,
    default="0.2,0.5,1,2,5",
    help="Comma list for grid Dirichlet concentrations",
)
@click.option("--mus", type=str, default="0,0.25,0.5,0.75,1", help="Comma list for grid mu values")
@click.option("--monotonicities", type=str, default="0", help="Comma list for grid monotonicity values")
@click.option("--lhs", "lhs_count", type=int, default=0, help="Latin Hypercube samples to draw")
@click.option("--kappa-min", type=float, default=0.2, help="LHS min kappa")
@click.option("--kappa-max", type=float, default=5.0, help="LHS max kappa")
@click.option("--c-min", type=float, default=0.2, help="LHS min concentration")
@click.option("--c-max", type=float, default=5.0, help="LHS max concentration")
@click.option("--mu-min", type=float, default=0.0, help="LHS min mu")
@click.option("--mu-max", type=float, default=1.0, help="LHS max mu")
@click.option("--monotonicity-min", type=float, default=0.0, help="LHS min monotonicity")
@click.option("--monotonicity-max", type=float, default=0.0, help="LHS max monotonicity")
@click.option("--frontier/--no-frontier", default=False, help="Run frontier search")
@click.option("--frontier-low", type=float, default=0.1, help="Frontier lower bound for kappa")
@click.option("--frontier-high", type=float, default=4.0, help="Initial frontier upper bound for kappa")
@click.option("--frontier-tolerance", type=float, default=0.02, help="Frontier tolerance on delta_total")
@click.option("--frontier-iterations", type=int, default=6, help="Max bisection iterations per cell")
@click.option("--n-agents", type=int, default=5, help="Ring size")
@click.option("--maturity-days", type=int, default=3, help="Due day horizon for generator")
@click.option("--q-total", type=float, default=500.0, help="Total dues S1 for generation")
@click.option(
    "--liquidity-mode",
    type=click.Choice(["single_at", "uniform"]),
    default="single_at",
    help="Liquidity allocation mode",
)
@click.option("--liquidity-agent", type=str, default="H1", help="Target for single_at liquidity allocation")
@click.option("--base-seed", type=int, default=42, help="Base PRNG seed")
@click.option("--name-prefix", type=str, default="Kalecki Ring Sweep", help="Scenario name prefix")
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default="fail-fast",
    help="Default handling mode for runs",
)
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
@click.option(
    "--perf-preset",
    type=click.Choice(["compatible", "fast", "aggressive"]),
    default=None,
    help="Performance preset",
)
@click.option("--fast-atomic", is_flag=True, default=False, help="Disable deepcopy in safe phases")
@click.option("--preview-buy", is_flag=True, default=False, help="Preview-then-commit buy path")
@click.option("--cache-dealer-quotes", is_flag=True, default=False, help="Snapshot/restore dealer state")
@click.option("--dirty-bucket-recompute", is_flag=True, default=False, help="Only recompute traded buckets")
@click.option("--prune-ineligible", is_flag=True, default=False, help="Skip zero-resource agents")
@click.option("--incremental-intentions", is_flag=True, default=False, help="Incremental intention queues")
@click.option("--matching-order", type=click.Choice(["random", "urgency"]), default=None, help="Matching order")
@click.option("--dealer-backend", type=click.Choice(["python", "native"]), default=None, help="Kernel backend")
@click.pass_context
def sweep_ring(
    ctx: click.Context,
    config: Path | None,
    out_dir: Path | None,
    cloud: bool,
    grid: bool,
    kappas: str,
    concentrations: str,
    mus: str,
    monotonicities: str,
    lhs_count: int,
    kappa_min: float,
    kappa_max: float,
    c_min: float,
    c_max: float,
    mu_min: float,
    mu_max: float,
    monotonicity_min: float,
    monotonicity_max: float,
    frontier: bool,
    frontier_low: float,
    frontier_high: float,
    frontier_tolerance: float,
    frontier_iterations: int,
    n_agents: int,
    maturity_days: int,
    q_total: float,
    liquidity_mode: str,
    liquidity_agent: str,
    base_seed: int,
    name_prefix: str,
    default_handling: str,
    job_id: str | None,
    perf_preset: str | None,
    fast_atomic: bool,
    preview_buy: bool,
    cache_dealer_quotes: bool,
    dirty_bucket_recompute: bool,
    prune_ineligible: bool,
    incremental_intentions: bool,
    matching_order: str | None,
    dealer_backend: str | None,
) -> None:
    """Run the Kalecki ring experiment sweep."""
    sweep_config: RingSweepConfig | None = None
    if config is not None:
        sweep_config = load_ring_sweep_config(config)

    if sweep_config is not None and sweep_config.out_dir and parameter_uses_default(ctx, "out_dir"):
        out_dir = Path(sweep_config.out_dir)

    dealer_enabled = False
    dealer_config = None
    if sweep_config is not None and sweep_config.runner is not None:
        runner_cfg = sweep_config.runner
        if runner_cfg.n_agents is not None and parameter_uses_default(ctx, "n_agents"):
            n_agents = runner_cfg.n_agents
        if runner_cfg.maturity_days is not None and parameter_uses_default(ctx, "maturity_days"):
            maturity_days = runner_cfg.maturity_days
        if runner_cfg.q_total is not None and parameter_uses_default(ctx, "q_total"):
            q_total = float(runner_cfg.q_total)
        if runner_cfg.liquidity_mode is not None and parameter_uses_default(ctx, "liquidity_mode"):
            liquidity_mode = runner_cfg.liquidity_mode
        if runner_cfg.liquidity_agent is not None and parameter_uses_default(ctx, "liquidity_agent"):
            liquidity_agent = runner_cfg.liquidity_agent
        if runner_cfg.base_seed is not None and parameter_uses_default(ctx, "base_seed"):
            base_seed = runner_cfg.base_seed
        if runner_cfg.name_prefix is not None and parameter_uses_default(ctx, "name_prefix"):
            name_prefix = runner_cfg.name_prefix
        if runner_cfg.default_handling is not None and parameter_uses_default(ctx, "default_handling"):
            default_handling = runner_cfg.default_handling
        dealer_enabled = runner_cfg.dealer_enabled
        dealer_config = runner_cfg.dealer_config

    if sweep_config is not None and sweep_config.grid is not None:
        grid_cfg = sweep_config.grid
        if parameter_uses_default(ctx, "grid"):
            grid = grid_cfg.enabled
        if grid_cfg.kappas and parameter_uses_default(ctx, "kappas"):
            kappas = ",".join(str(kappa) for kappa in grid_cfg.kappas)
        if grid_cfg.concentrations and parameter_uses_default(ctx, "concentrations"):
            concentrations = ",".join(str(concentration) for concentration in grid_cfg.concentrations)
        if grid_cfg.mus and parameter_uses_default(ctx, "mus"):
            mus = ",".join(str(mu) for mu in grid_cfg.mus)
        if grid_cfg.monotonicities and parameter_uses_default(ctx, "monotonicities"):
            monotonicities = ",".join(str(item) for item in grid_cfg.monotonicities)

    if sweep_config is not None and sweep_config.lhs is not None:
        lhs_cfg = sweep_config.lhs
        if parameter_uses_default(ctx, "lhs_count"):
            lhs_count = lhs_cfg.count
        if lhs_cfg.kappa_range is not None:
            if parameter_uses_default(ctx, "kappa_min"):
                kappa_min = float(lhs_cfg.kappa_range[0])
            if parameter_uses_default(ctx, "kappa_max"):
                kappa_max = float(lhs_cfg.kappa_range[1])
        if lhs_cfg.concentration_range is not None:
            if parameter_uses_default(ctx, "c_min"):
                c_min = float(lhs_cfg.concentration_range[0])
            if parameter_uses_default(ctx, "c_max"):
                c_max = float(lhs_cfg.concentration_range[1])
        if lhs_cfg.mu_range is not None:
            if parameter_uses_default(ctx, "mu_min"):
                mu_min = float(lhs_cfg.mu_range[0])
            if parameter_uses_default(ctx, "mu_max"):
                mu_max = float(lhs_cfg.mu_range[1])
        if lhs_cfg.monotonicity_range is not None:
            if parameter_uses_default(ctx, "monotonicity_min"):
                monotonicity_min = float(lhs_cfg.monotonicity_range[0])
            if parameter_uses_default(ctx, "monotonicity_max"):
                monotonicity_max = float(lhs_cfg.monotonicity_range[1])

    if sweep_config is not None and sweep_config.frontier is not None:
        frontier_cfg = sweep_config.frontier
        if parameter_uses_default(ctx, "frontier"):
            frontier = frontier_cfg.enabled
        if frontier_cfg.enabled:
            if frontier_cfg.kappa_low is not None and parameter_uses_default(ctx, "frontier_low"):
                frontier_low = float(frontier_cfg.kappa_low)
            if frontier_cfg.kappa_high is not None and parameter_uses_default(ctx, "frontier_high"):
                frontier_high = float(frontier_cfg.kappa_high)
            if frontier_cfg.tolerance is not None and parameter_uses_default(ctx, "frontier_tolerance"):
                frontier_tolerance = float(frontier_cfg.tolerance)
            if frontier_cfg.max_iterations is not None and parameter_uses_default(ctx, "frontier_iterations"):
                frontier_iterations = frontier_cfg.max_iterations

    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out") / "experiments" / f"{timestamp}_ring"

    out_dir.mkdir(parents=True, exist_ok=True)

    deps = _deps()
    if job_id is None:
        job_id = deps.generate_job_id()

    performance_flags: dict[str, object] = {}
    if perf_preset:
        performance_flags["preset"] = perf_preset
    if fast_atomic:
        performance_flags["fast_atomic"] = True
    if preview_buy:
        performance_flags["preview_buy"] = True
    if cache_dealer_quotes:
        performance_flags["cache_dealer_quotes"] = True
    if dirty_bucket_recompute:
        performance_flags["dirty_bucket_recompute"] = True
    if prune_ineligible:
        performance_flags["prune_ineligible"] = True
    if incremental_intentions:
        performance_flags["incremental_intentions"] = True
    if matching_order:
        performance_flags["matching_order"] = matching_order
    if dealer_backend:
        performance_flags["dealer_backend"] = dealer_backend

    manager: JobManager | None = None
    total_runs = 0
    try:
        manager = deps.create_job_manager(jobs_dir=out_dir, cloud=cloud, local=True)

        grid_kappas = as_decimal_list(kappas)
        grid_concentrations = as_decimal_list(concentrations)
        grid_mus = as_decimal_list(mus)

        job_config = JobConfig(
            sweep_type="ring",
            n_agents=n_agents,
            kappas=grid_kappas,
            concentrations=grid_concentrations,
            mus=grid_mus,
            cloud=cloud,
            maturity_days=maturity_days,
            seeds=[base_seed],
            performance=performance_flags,
        )
        job = manager.create_job(
            description=f"Ring sweep (n={n_agents}, cloud={cloud})",
            config=job_config,
            job_id=job_id,
        )

        console.print(f"[cyan]Job ID: {job.job_id}[/cyan]")
        manager.start_job(job.job_id)
    except CLI_HANDLED_ERRORS as exc:
        console.print(f"[yellow]Warning: Job tracking initialization failed: {exc}[/yellow]")
        manager = None

    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        console.print("[cyan]Cloud execution enabled[/cyan]")

    performance = build_performance_config(performance_flags)
    runner = deps.RingSweepRunner(
        out_dir,
        name_prefix=name_prefix,
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Decimal(str(q_total)),
        liquidity_mode=liquidity_mode,
        liquidity_agent=liquidity_agent,
        base_seed=base_seed,
        default_handling=default_handling,
        dealer_enabled=dealer_enabled,
        dealer_config=dealer_config,
        executor=executor,
        performance=performance,
    )

    console.print(f"[dim]Output directory: {out_dir}[/dim]")

    grid_kappas = as_decimal_list(kappas)
    grid_concentrations = as_decimal_list(concentrations)
    grid_mus = as_decimal_list(mus)
    grid_monotonicities = as_decimal_list(monotonicities)

    try:
        if grid:
            total_runs = len(grid_kappas) * len(grid_concentrations) * len(grid_mus) * len(grid_monotonicities)
            console.print(f"[dim]Running grid sweep: {total_runs} runs[/dim]")
            runner.run_grid(grid_kappas, grid_concentrations, grid_mus, grid_monotonicities)

        if lhs_count > 0:
            console.print(f"[dim]Running Latin Hypercube ({lhs_count})[/dim]")
            runner.run_lhs(
                lhs_count,
                kappa_range=(Decimal(str(kappa_min)), Decimal(str(kappa_max))),
                concentration_range=(Decimal(str(c_min)), Decimal(str(c_max))),
                mu_range=(Decimal(str(mu_min)), Decimal(str(mu_max))),
                monotonicity_range=(
                    Decimal(str(monotonicity_min)),
                    Decimal(str(monotonicity_max)),
                ),
            )

        if frontier:
            cell_count = len(grid_concentrations) * len(grid_mus) * len(grid_monotonicities)
            console.print(f"[dim]Running frontier search across {cell_count} cells[/dim]")
            runner.run_frontier(
                grid_concentrations,
                grid_mus,
                grid_monotonicities,
                kappa_low=Decimal(str(frontier_low)),
                kappa_high=Decimal(str(frontier_high)),
                tolerance=Decimal(str(frontier_tolerance)),
                max_iterations=frontier_iterations,
            )

        registry_csv = runner.registry_dir / "experiments.csv"
        results_csv = runner.aggregate_dir / "results.csv"
        dashboard_html = runner.aggregate_dir / "dashboard.html"

        deps.aggregate_runs(registry_csv, results_csv)
        deps.render_dashboard(results_csv, dashboard_html)

        if manager is not None:
            try:
                manager.complete_job(
                    job_id,
                    {
                        "grid_runs": total_runs if grid else 0,
                        "lhs_runs": lhs_count,
                        "frontier": frontier,
                    },
                )
            except CLI_HANDLED_ERRORS as exc:
                console.print(f"[yellow]Warning: Failed to complete job tracking: {exc}[/yellow]")

        console.print(f"[green]Sweep complete.[/green] Registry: {registry_csv}")
        console.print(f"[green]Aggregated results: {results_csv}")
        console.print(f"[green]Dashboard: {dashboard_html}")
    except CLI_HANDLED_ERRORS as exc:
        if manager is not None:
            try:
                manager.fail_job(job_id, str(exc))
            except CLI_HANDLED_ERRORS:
                pass
        raise


@click.command("comparison")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Output directory for results")
@click.option("--n-agents", type=int, default=100, help="Ring size (default: 100)")
@click.option("--maturity-days", type=int, default=10, help="Maturity horizon in days (default: 10)")
@click.option("--q-total", type=float, default=10000.0, help="Total debt amount (default: 10000)")
@click.option("--kappas", type=str, default="0.25,0.5,1,2,4", help="Comma list for kappa values")
@click.option(
    "--concentrations",
    type=str,
    default="0.2,0.5,1,2,5",
    help="Comma list for Dirichlet concentrations",
)
@click.option("--mus", type=str, default="0,0.25,0.5,0.75,1", help="Comma list for mu values")
@click.option("--monotonicities", type=str, default="0", help="Comma list for monotonicity values")
@click.option("--base-seed", type=int, default=42, help="Base PRNG seed")
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default="fail-fast",
    help="Default handling mode",
)
@click.option("--dealer-ticket-size", type=float, default=1.0, help="Ticket size for dealer")
@click.option("--dealer-share", type=float, default=0.25, help="Dealer capital as fraction of system cash")
@click.option("--vbt-share", type=float, default=0.50, help="VBT capital as fraction of system cash")
@click.option(
    "--liquidity-mode",
    type=click.Choice(["single_at", "uniform"]),
    default="uniform",
    help="Liquidity allocation mode",
)
@click.option("--liquidity-agent", type=str, default=None, help="Target agent for single_at mode")
@click.option("--name-prefix", type=str, default="Dealer Comparison", help="Scenario name prefix")
def sweep_comparison(
    out_dir: Path,
    n_agents: int,
    maturity_days: int,
    q_total: float,
    kappas: str,
    concentrations: str,
    mus: str,
    monotonicities: str,
    base_seed: int,
    default_handling: str,
    dealer_ticket_size: float,
    dealer_share: float,
    vbt_share: float,
    liquidity_mode: str,
    liquidity_agent: str | None,
    name_prefix: str,
) -> None:
    """Run dealer comparison experiments."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir.mkdir(parents=True, exist_ok=True)

    grid_kappas = as_decimal_list(kappas)
    grid_concentrations = as_decimal_list(concentrations)
    grid_mus = as_decimal_list(mus)
    grid_monotonicities = as_decimal_list(monotonicities)

    total_pairs = len(grid_kappas) * len(grid_concentrations) * len(grid_mus) * len(grid_monotonicities)
    console.print(f"[dim]Output directory: {out_dir}[/dim]")
    console.print(f"[dim]Running comparison sweep: {total_pairs} parameter combinations × 2 conditions = {total_pairs * 2} runs[/dim]")

    config = ComparisonSweepConfig(
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Decimal(str(q_total)),
        kappas=grid_kappas,
        concentrations=grid_concentrations,
        mus=grid_mus,
        monotonicities=grid_monotonicities,
        base_seed=base_seed,
        default_handling=default_handling,
        dealer_ticket_size=Decimal(str(dealer_ticket_size)),
        dealer_share=Decimal(str(dealer_share)),
        vbt_share=Decimal(str(vbt_share)),
        liquidity_mode=liquidity_mode,
        liquidity_agent=liquidity_agent,
        name_prefix=name_prefix,
    )

    runner = ComparisonSweepRunner(config, out_dir)
    results = runner.run_all()

    completed = [result for result in results if result.delta_reduction is not None]
    if completed:
        mean_relief = sum(float(result.relief_ratio or 0) for result in completed) / len(completed)
        improved = sum(1 for result in completed if result.delta_reduction and result.delta_reduction > 0)
        console.print("\n[green]Comparison sweep complete.[/green]")
        console.print(f"  Pairs completed: {len(completed)}/{len(results)}")
        console.print(f"  Mean relief ratio: {mean_relief:.1%}")
        console.print(f"  Pairs with improvement: {improved}")
    else:
        console.print("\n[yellow]Comparison sweep complete but no pairs completed successfully.[/yellow]")

    console.print("\n[green]Results:[/green]")
    console.print(f"  Comparison CSV: {runner.comparison_path}")
    console.print(f"  Summary JSON: {runner.summary_path}")
    console.print(f"  Control registry: {runner.control_dir / 'registry' / 'experiments.csv'}")
    console.print(f"  Treatment registry: {runner.treatment_dir / 'registry' / 'experiments.csv'}")
