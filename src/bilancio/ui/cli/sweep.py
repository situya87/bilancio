"""CLI commands for running experiment sweeps."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import click
from click.core import ParameterSource

from bilancio.analysis.report import aggregate_runs, render_dashboard
from bilancio.experiments.ring import (
    RingSweepConfig,
    RingSweepRunner,
    _decimal_list,
    load_ring_sweep_config,
)
from bilancio.jobs import JobConfig, JobManager, create_job_manager, generate_job_id

from .utils import _as_decimal_list, console

CLI_HANDLED_ERRORS = (
    click.ClickException,
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


@click.group()
def sweep() -> None:
    """Experiment sweeps."""
    pass


@sweep.command("ring")
@click.option(
    "--config", type=click.Path(path_type=Path), default=None, help="Path to sweep config YAML"
)
@click.option(
    "--out-dir", type=click.Path(path_type=Path), default=None, help="Base output directory"
)
@click.option("--cloud", is_flag=True, help="Run simulations on Modal cloud")
@click.option("--grid/--no-grid", default=True, help="Run coarse grid sweep")
@click.option(
    "--kappas", type=str, default="0.25,0.5,1,2,4", help="Comma list for grid kappa values"
)
@click.option(
    "--concentrations",
    type=str,
    default="0.2,0.5,1,2,5",
    help="Comma list for grid Dirichlet concentrations",
)
@click.option("--mus", type=str, default="0,0.25,0.5,0.75,1", help="Comma list for grid mu values")
@click.option(
    "--monotonicities", type=str, default="0", help="Comma list for grid monotonicity values"
)
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
@click.option(
    "--frontier-high", type=float, default=4.0, help="Initial frontier upper bound for kappa"
)
@click.option(
    "--frontier-tolerance", type=float, default=0.02, help="Frontier tolerance on delta_total"
)
@click.option(
    "--frontier-iterations", type=int, default=6, help="Max bisection iterations per cell"
)
@click.option("--n-agents", type=int, default=5, help="Ring size")
@click.option("--maturity-days", type=int, default=3, help="Due day horizon for generator")
@click.option("--q-total", type=float, default=500.0, help="Total dues S1 for generation")
@click.option(
    "--liquidity-mode",
    type=click.Choice(["single_at", "uniform"]),
    default="single_at",
    help="Liquidity allocation mode",
)
@click.option(
    "--liquidity-agent", type=str, default="H1", help="Target for single_at liquidity allocation"
)
@click.option("--base-seed", type=int, default=42, help="Base PRNG seed")
@click.option("--name-prefix", type=str, default="Kalecki Ring Sweep", help="Scenario name prefix")
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default="fail-fast",
    help="Default handling mode for runs",
)
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
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
) -> None:
    """Run the Kalecki ring experiment sweep."""
    sweep_config: RingSweepConfig | None = None
    if config is not None:
        sweep_config = load_ring_sweep_config(config)

    def _using_default(param_name: str) -> bool:
        source = ctx.get_parameter_source(param_name)
        return source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)

    if sweep_config is not None and sweep_config.out_dir and _using_default("out_dir"):
        out_dir = Path(sweep_config.out_dir)

    dealer_enabled = False
    dealer_config = None
    if sweep_config is not None and sweep_config.runner is not None:
        runner_cfg = sweep_config.runner
        if runner_cfg.n_agents is not None and _using_default("n_agents"):
            n_agents = runner_cfg.n_agents
        if runner_cfg.maturity_days is not None and _using_default("maturity_days"):
            maturity_days = runner_cfg.maturity_days
        if runner_cfg.q_total is not None and _using_default("q_total"):
            q_total = float(runner_cfg.q_total)
        if runner_cfg.liquidity_mode is not None and _using_default("liquidity_mode"):
            liquidity_mode = runner_cfg.liquidity_mode
        if runner_cfg.liquidity_agent is not None and _using_default("liquidity_agent"):
            liquidity_agent = runner_cfg.liquidity_agent
        if runner_cfg.base_seed is not None and _using_default("base_seed"):
            base_seed = runner_cfg.base_seed
        if runner_cfg.name_prefix is not None and _using_default("name_prefix"):
            name_prefix = runner_cfg.name_prefix
        if runner_cfg.default_handling is not None and _using_default("default_handling"):
            default_handling = runner_cfg.default_handling
        dealer_enabled = runner_cfg.dealer_enabled
        dealer_config = runner_cfg.dealer_config

    if sweep_config is not None and sweep_config.grid is not None:
        grid_cfg = sweep_config.grid
        if _using_default("grid"):
            grid = grid_cfg.enabled
        if grid_cfg.kappas and _using_default("kappas"):
            kappas = ",".join(str(k) for k in grid_cfg.kappas)
        if grid_cfg.concentrations and _using_default("concentrations"):
            concentrations = ",".join(str(c) for c in grid_cfg.concentrations)
        if grid_cfg.mus and _using_default("mus"):
            mus = ",".join(str(m) for m in grid_cfg.mus)
        if grid_cfg.monotonicities and _using_default("monotonicities"):
            monotonicities = ",".join(str(m) for m in grid_cfg.monotonicities)

    if sweep_config is not None and sweep_config.lhs is not None:
        lhs_cfg = sweep_config.lhs
        if _using_default("lhs_count"):
            lhs_count = lhs_cfg.count
        if lhs_cfg.kappa_range is not None:
            if _using_default("kappa_min"):
                kappa_min = float(lhs_cfg.kappa_range[0])
            if _using_default("kappa_max"):
                kappa_max = float(lhs_cfg.kappa_range[1])
        if lhs_cfg.concentration_range is not None:
            if _using_default("c_min"):
                c_min = float(lhs_cfg.concentration_range[0])
            if _using_default("c_max"):
                c_max = float(lhs_cfg.concentration_range[1])
        if lhs_cfg.mu_range is not None:
            if _using_default("mu_min"):
                mu_min = float(lhs_cfg.mu_range[0])
            if _using_default("mu_max"):
                mu_max = float(lhs_cfg.mu_range[1])
        if lhs_cfg.monotonicity_range is not None:
            if _using_default("monotonicity_min"):
                monotonicity_min = float(lhs_cfg.monotonicity_range[0])
            if _using_default("monotonicity_max"):
                monotonicity_max = float(lhs_cfg.monotonicity_range[1])

    if sweep_config is not None and sweep_config.frontier is not None:
        frontier_cfg = sweep_config.frontier
        if _using_default("frontier"):
            frontier = frontier_cfg.enabled
        if frontier_cfg.enabled:
            if frontier_cfg.kappa_low is not None and _using_default("frontier_low"):
                frontier_low = float(frontier_cfg.kappa_low)
            if frontier_cfg.kappa_high is not None and _using_default("frontier_high"):
                frontier_high = float(frontier_cfg.kappa_high)
            if frontier_cfg.tolerance is not None and _using_default("frontier_tolerance"):
                frontier_tolerance = float(frontier_cfg.tolerance)
            if frontier_cfg.max_iterations is not None and _using_default("frontier_iterations"):
                frontier_iterations = frontier_cfg.max_iterations

    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out") / "experiments" / f"{ts}_ring"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate job ID if not provided
    if job_id is None:
        job_id = generate_job_id()

    # Create job manager and job config
    manager: JobManager | None = None
    try:
        manager = create_job_manager(jobs_dir=out_dir, cloud=cloud, local=True)

        grid_kappas = _as_decimal_list(kappas)
        grid_concentrations = _as_decimal_list(concentrations)
        grid_mus = _as_decimal_list(mus)

        job_config = JobConfig(
            sweep_type="ring",
            n_agents=n_agents,
            kappas=grid_kappas,
            concentrations=grid_concentrations,
            mus=grid_mus,
            cloud=cloud,
            maturity_days=maturity_days,
            seeds=[base_seed],
        )

        job = manager.create_job(
            description=f"Ring sweep (n={n_agents}, cloud={cloud})",
            config=job_config,
            job_id=job_id,
        )

        console.print(f"[cyan]Job ID: {job.job_id}[/cyan]")
        manager.start_job(job.job_id)
    except CLI_HANDLED_ERRORS as e:
        console.print(f"[yellow]Warning: Job tracking initialization failed: {e}[/yellow]")
        manager = None

    # Create executor based on --cloud flag
    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,  # Use job_id as experiment_id for simplicity
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        console.print("[cyan]Cloud execution enabled[/cyan]")

    q_total_dec = Decimal(str(q_total))
    runner = RingSweepRunner(
        out_dir,
        name_prefix=name_prefix,
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=q_total_dec,
        liquidity_mode=liquidity_mode,
        liquidity_agent=liquidity_agent,
        base_seed=base_seed,
        default_handling=default_handling,
        dealer_enabled=dealer_enabled,
        dealer_config=dealer_config,
        executor=executor,
    )

    console.print(f"[dim]Output directory: {out_dir}[/dim]")

    grid_kappas = _as_decimal_list(kappas)
    grid_concentrations = _as_decimal_list(concentrations)
    grid_mus = _as_decimal_list(mus)
    grid_monotonicities = _as_decimal_list(monotonicities)

    try:
        if grid:
            total_runs = (
                len(grid_kappas)
                * len(grid_concentrations)
                * len(grid_mus)
                * len(grid_monotonicities)
            )
            console.print(f"[dim]Running grid sweep: {total_runs} runs[/dim]")
            runner.run_grid(grid_kappas, grid_concentrations, grid_mus, grid_monotonicities)

        if lhs_count > 0:
            console.print(f"[dim]Running Latin Hypercube ({lhs_count})[/dim]")
            runner.run_lhs(
                lhs_count,
                kappa_range=(Decimal(str(kappa_min)), Decimal(str(kappa_max))),
                concentration_range=(Decimal(str(c_min)), Decimal(str(c_max))),
                mu_range=(Decimal(str(mu_min)), Decimal(str(mu_max))),
                monotonicity_range=(Decimal(str(monotonicity_min)), Decimal(str(monotonicity_max))),
            )

        if frontier:
            console.print(
                f"[dim]Running frontier search across {len(grid_concentrations) * len(grid_mus) * len(grid_monotonicities)} cells[/dim]"
            )
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

        aggregate_runs(registry_csv, results_csv)
        render_dashboard(results_csv, dashboard_html)

        # Complete job
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
            except CLI_HANDLED_ERRORS as e:
                console.print(f"[yellow]Warning: Failed to complete job tracking: {e}[/yellow]")

        console.print(f"[green]Sweep complete.[/green] Registry: {registry_csv}")
        console.print(f"[green]Aggregated results: {results_csv}")
        console.print(f"[green]Dashboard: {dashboard_html}")

    except CLI_HANDLED_ERRORS as e:
        # Fail job on error
        if manager is not None:
            try:
                manager.fail_job(job_id, str(e))
            except CLI_HANDLED_ERRORS:
                pass  # Don't let job tracking failure mask the original error
        raise


@sweep.command("comparison")
@click.option(
    "--out-dir", type=click.Path(path_type=Path), required=True, help="Output directory for results"
)
@click.option("--n-agents", type=int, default=100, help="Ring size (default: 100)")
@click.option(
    "--maturity-days", type=int, default=10, help="Maturity horizon in days (default: 10)"
)
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
@click.option(
    "--dealer-share",
    type=float,
    default=0.25,
    help="Dealer capital as fraction of system cash (NEW outside money)",
)
@click.option(
    "--vbt-share",
    type=float,
    default=0.50,
    help="VBT capital as fraction of system cash (NEW outside money)",
)
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
    """
    Run dealer comparison experiments.

    For each parameter combination (κ, c, μ), runs:
    1. Control: No dealer (baseline)
    2. Treatment: With dealer

    Outputs comparison metrics showing the delta in defaults between conditions.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Deferred import to avoid circular import
    from bilancio.experiments.comparison import ComparisonSweepConfig, ComparisonSweepRunner

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_kappas = _as_decimal_list(kappas)
    grid_concentrations = _as_decimal_list(concentrations)
    grid_mus = _as_decimal_list(mus)
    grid_monotonicities = _as_decimal_list(monotonicities)

    total_pairs = (
        len(grid_kappas) * len(grid_concentrations) * len(grid_mus) * len(grid_monotonicities)
    )
    console.print(f"[dim]Output directory: {out_dir}[/dim]")
    console.print(
        f"[dim]Running comparison sweep: {total_pairs} parameter combinations × 2 conditions = {total_pairs * 2} runs[/dim]"
    )

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

    # Summary
    completed = [r for r in results if r.delta_reduction is not None]
    if completed:
        mean_relief = sum(float(r.relief_ratio or 0) for r in completed) / len(completed)
        improved = sum(1 for r in completed if r.delta_reduction and r.delta_reduction > 0)
        console.print("\n[green]Comparison sweep complete.[/green]")
        console.print(f"  Pairs completed: {len(completed)}/{len(results)}")
        console.print(f"  Mean relief ratio: {mean_relief:.1%}")
        console.print(f"  Pairs with improvement: {improved}")
    else:
        console.print(
            "\n[yellow]Comparison sweep complete but no pairs completed successfully.[/yellow]"
        )

    console.print("\n[green]Results:[/green]")
    console.print(f"  Comparison CSV: {runner.comparison_path}")
    console.print(f"  Summary JSON: {runner.summary_path}")
    console.print(f"  Control registry: {runner.control_dir / 'registry' / 'experiments.csv'}")
    console.print(f"  Treatment registry: {runner.treatment_dir / 'registry' / 'experiments.csv'}")


@sweep.command("balanced")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for results",
)
@click.option("--n-agents", type=int, default=100, help="Number of agents in ring")
@click.option("--maturity-days", type=int, default=10, help="Maturity horizon")
@click.option("--q-total", type=Decimal, default=Decimal("10000"), help="Total debt")
@click.option("--base-seed", type=int, default=42, help="Base random seed")
@click.option(
    "--kappas",
    type=str,
    default="0.25,0.5,1,2,4",
    help="Comma-separated kappa values",
)
@click.option(
    "--concentrations",
    type=str,
    default="0.2,0.5,1,2,5",
    help="Comma-separated concentration values",
)
@click.option(
    "--mus",
    type=str,
    default="0,0.25,0.5,0.75,1",
    help="Comma-separated mu values",
)
@click.option(
    "--face-value",
    type=Decimal,
    default=Decimal("20"),
    help="Face value S (cashflow at maturity)",
)
@click.option(
    "--outside-mid-ratios",
    type=str,
    default="1.0",
    help="Comma-separated M/S ratios to sweep (VBT pricing now uses kappa-informed prior)",
)
@click.option(
    "--big-entity-share",
    type=Decimal,
    default=Decimal("0.25"),
    help="Fraction of debt held by big entities (β)",
)
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default="expel-agent",
    help="Default handling mode",
)
@click.option(
    "--detailed-logging/--no-detailed-logging",
    default=True,
    help="Enable detailed CSV logging (trades.csv, repayment_events.csv, etc.)",
)
@click.option("--cloud", is_flag=True, help="Run simulations on Modal cloud")
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
@click.option(
    "--quiet/--verbose",
    default=True,
    help="Suppress verbose console output during sweeps (default: quiet)",
)
@click.option(
    "--rollover/--no-rollover",
    default=True,
    help="Enable continuous rollover of matured claims (default: enabled)",
)
@click.option(
    "--risk-assessment/--no-risk-assessment",
    default=True,
    help="Enable risk-based trader decision making (default: enabled)",
)
@click.option(
    "--risk-premium",
    type=Decimal,
    default=Decimal("0.02"),
    help="Base risk premium for trading decisions (default: 0.02)",
)
@click.option(
    "--risk-urgency",
    type=Decimal,
    default=Decimal("0.10"),
    help="Urgency sensitivity (default: 0.10)",
)
@click.option(
    "--alpha-vbt",
    type=Decimal,
    default=Decimal("0"),
    help="VBT informedness: 0=naive prior, 1=fully kappa-informed pricing (default: 0)",
)
@click.option(
    "--alpha-trader",
    type=Decimal,
    default=Decimal("0"),
    help="Trader informedness: 0=naive prior, 1=fully kappa-informed pricing (default: 0)",
)
@click.option(
    "--risk-aversion",
    type=Decimal,
    default=Decimal("0"),
    help="Trader risk aversion (0=risk-neutral, 1=max risk-averse, default: 0)",
)
@click.option(
    "--planning-horizon",
    type=int,
    default=10,
    help="Trader planning horizon in days (1-20, default: 10)",
)
@click.option(
    "--aggressiveness",
    type=Decimal,
    default=Decimal("1.0"),
    help="Buyer aggressiveness (0=conservative, 1=eager, default: 1.0)",
)
@click.option(
    "--default-observability",
    type=Decimal,
    default=Decimal("1.0"),
    help="Trader default observability (0=ignore, 1=full tracking, default: 1.0)",
)
@click.option(
    "--vbt-mid-sensitivity",
    type=Decimal,
    default=Decimal("1.0"),
    help="VBT mid price sensitivity to defaults (0=ignore, 1=full tracking, default: 1.0)",
)
@click.option(
    "--vbt-spread-sensitivity",
    type=Decimal,
    default=Decimal("0.0"),
    help="VBT spread sensitivity to defaults (0=fixed, 1=widen with defaults, default: 0.0)",
)
@click.option(
    "--trading-motive",
    type=click.Choice(["liquidity_only", "liquidity_then_earning", "unrestricted"]),
    default="liquidity_then_earning",
    help="Trading motivation (default: liquidity_then_earning)",
)
@click.option(
    "--enable-lender/--no-lender",
    default=False,
    help="Enable third comparison arm with non-bank lender (default: disabled)",
)
@click.option(
    "--lender-share",
    type=Decimal,
    default=Decimal("0.10"),
    help="Lender capital as fraction of system cash (default: 0.10)",
)
@click.option(
    "--enable-dealer-lender/--no-dealer-lender",
    default=False,
    help="Enable fourth arm: dealer trading + non-bank lending combined (default: disabled)",
)
def sweep_balanced(
    out_dir: Path,
    n_agents: int,
    maturity_days: int,
    q_total: Decimal,
    base_seed: int,
    kappas: str,
    concentrations: str,
    mus: str,
    face_value: Decimal,
    outside_mid_ratios: str,
    big_entity_share: Decimal,
    default_handling: str,
    detailed_logging: bool,
    cloud: bool,
    job_id: str | None,
    quiet: bool,
    rollover: bool,
    risk_assessment: bool,
    risk_premium: Decimal,
    risk_urgency: Decimal,
    alpha_vbt: Decimal,
    alpha_trader: Decimal,
    risk_aversion: Decimal,
    planning_horizon: int,
    aggressiveness: Decimal,
    default_observability: Decimal,
    vbt_mid_sensitivity: Decimal,
    vbt_spread_sensitivity: Decimal,
    trading_motive: str,
    enable_lender: bool,
    lender_share: Decimal,
    enable_dealer_lender: bool,
) -> None:
    """
    Run balanced C vs D comparison experiments.

    Compares passive holders (C) against active dealers (D) with
    identical starting balance sheets. Each pair runs the same scenario
    twice: once with trading disabled (passive) and once with trading
    enabled (active).

    Output:
      - passive/: All passive holder runs
      - active/: All active dealer runs
      - aggregate/comparison.csv: C vs D metrics
      - aggregate/summary.json: Aggregate statistics
    """
    from bilancio.experiments.balanced_comparison import (
        BalancedComparisonConfig,
        BalancedComparisonRunner,
    )

    out_dir = Path(out_dir)

    # Generate job ID if not provided
    if job_id is None:
        job_id = generate_job_id()

    # Create job manager with Supabase cloud storage
    manager: JobManager | None = None
    try:
        manager = create_job_manager(jobs_dir=out_dir, cloud=cloud, local=True)

        # Create job config
        job_config = JobConfig(
            sweep_type="balanced",
            n_agents=n_agents,
            kappas=_decimal_list(kappas),
            concentrations=_decimal_list(concentrations),
            mus=_decimal_list(mus),
            cloud=cloud,
            outside_mid_ratios=_decimal_list(outside_mid_ratios),
            maturity_days=maturity_days,
            seeds=[base_seed],
        )

        # Create and start job
        job = manager.create_job(
            description=f"Balanced comparison sweep (n={n_agents}, cloud={cloud})",
            config=job_config,
            job_id=job_id,
        )

        click.echo(f"Job ID: {job.job_id}")
        manager.start_job(job.job_id)
    except CLI_HANDLED_ERRORS as e:
        click.echo(f"Warning: Job tracking initialization failed: {e}")
        manager = None

    # Create executor (Plan 028)
    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,  # Use job_id as experiment_id for simplicity
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        click.echo("Cloud execution enabled")

    if risk_assessment:
        click.echo(f"Risk assessment enabled (premium={risk_premium}, urgency={risk_urgency})")

    # Build risk assessment config if enabled
    risk_config = {
        "base_risk_premium": str(risk_premium),
        "urgency_sensitivity": str(risk_urgency),
        "buy_premium_multiplier": "1.0",
        "lookback_window": 5,
    }

    if alpha_vbt > 0 or alpha_trader > 0:
        click.echo(f"Informedness enabled (alpha_vbt={alpha_vbt}, alpha_trader={alpha_trader})")

    config = BalancedComparisonConfig(
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=q_total,
        base_seed=base_seed,
        kappas=_decimal_list(kappas),
        concentrations=_decimal_list(concentrations),
        mus=_decimal_list(mus),
        face_value=face_value,
        outside_mid_ratios=_decimal_list(outside_mid_ratios),
        big_entity_share=big_entity_share,
        default_handling=default_handling,
        detailed_logging=detailed_logging,
        quiet=quiet,  # Plan 030
        rollover_enabled=rollover,
        risk_assessment_enabled=risk_assessment,
        risk_assessment_config=risk_config,
        alpha_vbt=alpha_vbt,
        alpha_trader=alpha_trader,
        risk_aversion=risk_aversion,
        planning_horizon=planning_horizon,
        aggressiveness=aggressiveness,
        default_observability=default_observability,
        vbt_mid_sensitivity=vbt_mid_sensitivity,
        vbt_spread_sensitivity=vbt_spread_sensitivity,
        trading_motive=trading_motive,
        enable_lender=enable_lender,
        lender_share=lender_share,
        enable_dealer_lender=enable_dealer_lender,
    )

    runner = BalancedComparisonRunner(config, out_dir, executor=executor, job_id=job_id)

    try:
        results = runner.run_all()

        # Record run IDs from results
        if manager is not None:
            try:
                for r in results:
                    if r.passive_run_id:
                        manager.record_progress(
                            job_id, r.passive_run_id, modal_call_id=r.passive_modal_call_id
                        )
                    if r.active_run_id:
                        manager.record_progress(
                            job_id, r.active_run_id, modal_call_id=r.active_modal_call_id
                        )
            except CLI_HANDLED_ERRORS as e:
                click.echo(f"Warning: Failed to record run progress: {e}")

        # Print summary
        completed = sum(1 for r in results if r.trading_effect is not None)
        improved = sum(1 for r in results if r.trading_effect and r.trading_effect > 0)

        # Complete job with summary
        if manager is not None:
            try:
                manager.complete_job(
                    job_id,
                    {
                        "total_pairs": len(results),
                        "completed": completed,
                        "improved_with_trading": improved,
                    },
                )
            except CLI_HANDLED_ERRORS as e:
                click.echo(f"Warning: Failed to complete job tracking: {e}")

        click.echo("\nBalanced comparison complete!")
        click.echo(f"  Total pairs: {len(results)}")
        click.echo(f"  Completed: {completed}")
        click.echo(f"  Improved with trading: {improved}")
        click.echo(f"\nResults at: {out_dir / 'aggregate' / 'comparison.csv'}")

    except CLI_HANDLED_ERRORS as e:
        # Fail job on error
        if manager is not None:
            try:
                manager.fail_job(job_id, str(e))
            except CLI_HANDLED_ERRORS:
                pass  # Don't let job tracking failure mask the original error
        raise


@sweep.command("strategy-outcomes")
@click.option(
    "--experiment",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to experiment directory (containing aggregate/comparison.csv)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sweep_strategy_outcomes(experiment: Path, verbose: bool) -> None:
    """Analyze trading strategy outcomes across experiment runs.

    Reads repayment_events.csv files and computes per-strategy metrics:
    - Count and face value per strategy
    - Default count and face value per strategy
    - Default rate per strategy

    Outputs:
    - aggregate/strategy_outcomes_by_run.csv
    - aggregate/strategy_outcomes_overall.csv
    """
    import logging

    from bilancio.analysis.strategy_outcomes import run_strategy_analysis

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    by_run_path, overall_path = run_strategy_analysis(experiment)

    if by_run_path and by_run_path.exists():
        console.print(f"[green]OK[/green] Strategy outcomes by run: {by_run_path}")
        console.print(f"[green]OK[/green] Strategy outcomes overall: {overall_path}")
    else:
        console.print(
            "[yellow]No output generated - check that repayment_events.csv files exist[/yellow]"
        )


@sweep.command("dealer-usage")
@click.option(
    "--experiment",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to experiment directory (containing aggregate/comparison.csv)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sweep_dealer_usage(experiment: Path, verbose: bool) -> None:
    """Analyze dealer usage patterns across experiment runs.

    Reads trades.csv, inventory_timeseries.csv, system_state_timeseries.csv,
    and repayment_events.csv to explain why dealers have or don't have an effect.

    Outputs:
    - aggregate/dealer_usage_by_run.csv
    """
    import logging

    from bilancio.analysis.dealer_usage_summary import run_dealer_usage_analysis

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_path = run_dealer_usage_analysis(experiment)

    if output_path and output_path.exists():
        console.print(f"[green]OK[/green] Dealer usage summary: {output_path}")
    else:
        console.print("[yellow]No output generated - check that required CSV files exist[/yellow]")
