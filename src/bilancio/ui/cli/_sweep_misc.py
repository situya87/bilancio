"""Miscellaneous sweep subcommands."""

from __future__ import annotations

from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

import click

from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner
from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

from ._common import CLI_HANDLED_ERRORS, as_decimal_list, console, invoke_subcommand


def _deps() -> Any:
    return import_module("bilancio.ui.cli.sweep")


@click.command("list")
def sweep_list() -> None:
    """List available scenario plugins."""
    from bilancio.scenarios.registry import get_registry

    registry = get_registry()
    if not registry:
        console.print("[yellow]No scenario plugins registered.[/yellow]")
        return

    for name, plugin in sorted(registry.items()):
        meta = plugin.metadata
        console.print(f"\n[bold cyan]{meta.display_name}[/bold cyan] ({name})")
        console.print(f"  {meta.description}")
        console.print(f"  Version: {meta.version}")
        console.print(f"  Instruments: {', '.join(meta.instruments_used)}")
        console.print(f"  Agent types: {', '.join(meta.agent_types)}")
        console.print(f"  Dealer support: {'yes' if meta.supports_dealer else 'no'}")
        console.print(f"  Lender support: {'yes' if meta.supports_lender else 'no'}")
        dims = plugin.parameter_dimensions()
        if dims:
            console.print("  Parameters:")
            for dim in dims:
                defaults_str = ", ".join(str(value) for value in dim.default_values)
                console.print(f"    {dim.name}: {dim.display_name}")
                console.print(f"      {dim.description}")
                console.print(f"      Defaults: [{defaults_str}]")


@click.command("strategy-outcomes")
@click.option(
    "--experiment",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to experiment directory",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sweep_strategy_outcomes(experiment: Path, verbose: bool) -> None:
    """Analyze trading strategy outcomes across experiment runs."""
    import logging

    from bilancio.analysis.strategy_outcomes import run_strategy_analysis

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")
    by_run_path, overall_path = run_strategy_analysis(experiment)
    if by_run_path and by_run_path.exists():
        console.print(f"[green]OK[/green] Strategy outcomes by run: {by_run_path}")
        console.print(f"[green]OK[/green] Strategy outcomes overall: {overall_path}")
    else:
        console.print("[yellow]No output generated - check that repayment_events.csv files exist[/yellow]")


@click.command("dealer-usage")
@click.option(
    "--experiment",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to experiment directory",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sweep_dealer_usage(experiment: Path, verbose: bool) -> None:
    """Analyze dealer usage patterns across experiment runs."""
    import logging

    from bilancio.analysis.dealer_usage_summary import run_dealer_usage_analysis

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")
    output_path = run_dealer_usage_analysis(experiment)
    if output_path and output_path.exists():
        console.print(f"[green]OK[/green] Dealer usage summary: {output_path}")
    else:
        console.print("[yellow]No output generated - check that required CSV files exist[/yellow]")


@click.command("nbfi")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path), help="Output directory")
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
@click.option("--cloud", is_flag=True, help="Run on Modal cloud")
@click.option("--n-agents", type=int, default=100, help="Number of firms in ring")
@click.option("--maturity-days", type=int, default=10, help="Payment horizon in days")
@click.option("--q-total", type=int, default=10000, help="Total debt amount")
@click.option("--base-seed", type=int, default=42, help="Random seed")
@click.option("--n-replicates", type=int, default=1, help="Number of replicate seeds per combo")
@click.option("--kappas", type=str, default="0.3,0.5,1.0,2.0", help="Comma-separated liquidity ratios")
@click.option("--concentrations", type=str, default="1", help="Comma-separated debt concentration values")
@click.option("--mus", type=str, default="0", help="Comma-separated maturity skew values")
@click.option("--outside-mid-ratios", type=str, default="0.90", help="Comma-separated outside money ratios")
@click.option("--face-value", type=Decimal, default=Decimal("20"), help="Face value per ticket")
@click.option("--rollover/--no-rollover", default=True, help="Enable continuous rollover")
@click.option("--quiet/--no-quiet", default=True, help="Suppress per-event output")
@click.option("--nbfi-share", type=Decimal, default=Decimal("0.10"), help="NBFI cash share")
@click.option("--enable-collateral-arm", is_flag=True, default=False, help="Enable 3rd arm: NBFI collateralized lending")
@click.option("--lender-collateral-mode", type=str, default="pledged", help="Collateral mode for collateral arm")
@click.option("--lender-base-haircut", type=Decimal, default=Decimal("0.05"), help="Base haircut for collateral")
@click.option("--lender-haircut-risk-sensitivity", type=Decimal, default=Decimal("1.0"), help="Haircut risk sensitivity")
@click.option("--lender-haircut-maturity-sensitivity", type=Decimal, default=Decimal("0.5"), help="Haircut maturity sensitivity")
@click.option("--default-handling", type=str, default="expel-agent", help="Default handling mode")
@click.option(
    "--post-analysis",
    type=str,
    default=None,
    help=(
        "Post-sweep analysis: 'all', 'none', or comma-separated list. "
        "Valid: drilldowns, deltas, dynamics, narrative, strategy_outcomes, "
        "dealer_usage, mechanism_activity, treynor. Default: interactive prompt."
    ),
)
@click.option("--topologies", type=str, default="ring", help="Comma-separated topology specs (e.g. 'ring,k_regular:4,erdos_renyi:0.1')")
@click.option("--preset", type=click.Path(exists=True, path_type=Path), default=None, help="Load preset YAML")
def sweep_nbfi(
    out_dir: Path,
    job_id: str | None,
    cloud: bool,
    n_agents: int,
    maturity_days: int,
    q_total: int,
    base_seed: int,
    n_replicates: int,
    kappas: str,
    concentrations: str,
    mus: str,
    outside_mid_ratios: str,
    face_value: Decimal,
    rollover: bool,
    quiet: bool,
    nbfi_share: Decimal,
    enable_collateral_arm: bool,
    lender_collateral_mode: str,
    lender_base_haircut: Decimal,
    lender_haircut_risk_sensitivity: Decimal,
    lender_haircut_maturity_sensitivity: Decimal,
    default_handling: str,
    topologies: str,
    post_analysis: str | None,
    preset: Path | None,
) -> None:
    """Run NBFI lending experiment (Plan 043)."""
    deps = _deps()

    if preset is not None:
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args, load_preset

        preset_data = load_preset(preset)
        setup_result = SweepSetupResult(
            sweep_type="nbfi",
            cloud=cloud,
            params=preset_data.get("params", {}),
            out_dir=out_dir,
            launch=True,
        )
        cli_args = build_cli_args(setup_result)
        if "--out-dir" not in cli_args:
            cli_args = ["--out-dir", str(out_dir)] + cli_args
        click.echo(f"Loaded preset from: {preset}")
        invoke_subcommand(deps.sweep, click.get_current_context(), "nbfi", cli_args)
        return

    out_dir = Path(out_dir)
    if job_id is None:
        job_id = deps.generate_job_id()

    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        click.echo("Cloud execution enabled")

    from bilancio.scenarios.ring.topology import parse_topology_string

    parsed_topologies = [parse_topology_string(t.strip()) for t in topologies.split(",")]

    config = NBFIComparisonConfig(
        name_prefix=job_id,
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Decimal(q_total),
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=as_decimal_list(kappas),
        concentrations=as_decimal_list(concentrations),
        mus=as_decimal_list(mus),
        outside_mid_ratios=as_decimal_list(outside_mid_ratios),
        face_value=face_value,
        rollover_enabled=rollover,
        quiet=quiet,
        default_handling=default_handling,
        lender_share=nbfi_share,
        enable_collateral_arm=enable_collateral_arm,
        lender_collateral_mode=lender_collateral_mode,
        lender_base_haircut=lender_base_haircut,
        lender_haircut_risk_sensitivity=lender_haircut_risk_sensitivity,
        lender_haircut_maturity_sensitivity=lender_haircut_maturity_sensitivity,
        topologies=parsed_topologies,
    )

    runner = NBFIComparisonRunner(
        config=config,
        out_dir=out_dir,
        executor=executor,
        job_id=job_id,
        enable_supabase=cloud,
    )

    click.echo(f"Job ID: {job_id}")
    n_combos = (
        len(as_decimal_list(kappas))
        * len(as_decimal_list(concentrations))
        * len(as_decimal_list(mus))
        * len(as_decimal_list(outside_mid_ratios))
    )
    n_arms = 3 if enable_collateral_arm else 2
    click.echo(f"Preparing {n_combos * n_arms} runs ({n_combos} combos × {n_arms} arms)...")

    results = runner.run_all()
    completed = sum(1 for result in results if result.lending_effect is not None)
    improved = sum(1 for result in results if result.lending_effect and result.lending_effect > 0)

    click.echo("\nNBFI comparison complete!")
    click.echo(f"  Total combos: {len(results)}")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Lending helped: {improved}")
    if enable_collateral_arm:
        collateral_improved = sum(1 for result in results if result.collateral_effect and result.collateral_effect > 0)
        click.echo(f"  Collateral helped: {collateral_improved}")
    click.echo(f"\nResults at: {out_dir / 'aggregate' / 'comparison.csv'}")

    deps._offer_post_sweep_analysis(out_dir, "nbfi", post_analysis, cloud=cloud)


@click.command("bank")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path), help="Output directory")
@click.option("--job-id", type=str, default=None, help="Job ID (auto-generated if not provided)")
@click.option("--cloud", is_flag=True, help="Run on Modal cloud")
@click.option("--n-agents", type=int, default=100, help="Number of firms in ring")
@click.option("--maturity-days", type=int, default=10, help="Payment horizon in days")
@click.option("--q-total", type=int, default=10000, help="Total debt amount")
@click.option("--base-seed", type=int, default=42, help="Random seed")
@click.option("--n-replicates", type=int, default=1, help="Number of replicate seeds per combo")
@click.option("--kappas", type=str, default="0.3,0.5,1.0,2.0", help="Comma-separated liquidity ratios")
@click.option("--concentrations", type=str, default="1", help="Comma-separated debt concentration values")
@click.option("--mus", type=str, default="0", help="Comma-separated maturity skew values")
@click.option("--outside-mid-ratios", type=str, default="0.90", help="Comma-separated outside money ratios")
@click.option("--face-value", type=Decimal, default=Decimal("20"), help="Face value per ticket")
@click.option("--rollover/--no-rollover", default=True, help="Enable continuous rollover")
@click.option("--quiet/--no-quiet", default=True, help="Suppress per-event output")
@click.option("--n-banks", type=int, default=5, help="Number of banks")
@click.option("--reserve-ratio", type=Decimal, default=Decimal("0.50"), help="Initial reserves / total deposits")
@click.option("--credit-risk-loading", type=Decimal, default=Decimal("0.5"), help="Bank sensitivity to borrower risk")
@click.option(
    "--max-borrower-risk",
    type=Decimal,
    default=Decimal("0.4"),
    help="Credit rationing threshold",
)
@click.option("--min-coverage-ratio", type=Decimal, default=Decimal("0"), help="Min coverage ratio for loan approval")
@click.option("--cb-rate-escalation-slope", type=Decimal, default=Decimal("0.05"), help="CB cost pressure slope")
@click.option("--cb-max-outstanding-ratio", type=Decimal, default=Decimal("2.0"), help="CB lending cap")
@click.option("--default-handling", type=str, default="expel-agent", help="Default handling mode")
@click.option("--fast-atomic", is_flag=True, default=False, help="Disable deepcopy in safe phases")
@click.option(
    "--post-analysis",
    type=str,
    default=None,
    help=(
        "Post-sweep analysis: 'all', 'none', or comma-separated list. "
        "Valid: drilldowns, deltas, dynamics, narrative, strategy_outcomes, "
        "dealer_usage, mechanism_activity, treynor. Default: interactive prompt."
    ),
)
@click.option("--topologies", type=str, default="ring", help="Comma-separated topology specs (e.g. 'ring,k_regular:4,erdos_renyi:0.1')")
@click.option("--preset", type=click.Path(exists=True, path_type=Path), default=None, help="Load preset YAML")
def sweep_bank(
    out_dir: Path,
    job_id: str | None,
    cloud: bool,
    n_agents: int,
    maturity_days: int,
    q_total: int,
    base_seed: int,
    n_replicates: int,
    kappas: str,
    concentrations: str,
    mus: str,
    outside_mid_ratios: str,
    face_value: Decimal,
    rollover: bool,
    quiet: bool,
    n_banks: int,
    reserve_ratio: Decimal,
    credit_risk_loading: Decimal,
    max_borrower_risk: Decimal,
    min_coverage_ratio: Decimal,
    cb_rate_escalation_slope: Decimal,
    cb_max_outstanding_ratio: Decimal,
    default_handling: str,
    fast_atomic: bool,
    topologies: str,
    post_analysis: str | None,
    preset: Path | None,
) -> None:
    """Run bank lending experiment (Plan 043)."""
    deps = _deps()

    if preset is not None:
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args, load_preset

        preset_data = load_preset(preset)
        setup_result = SweepSetupResult(
            sweep_type="bank",
            cloud=cloud,
            params=preset_data.get("params", {}),
            out_dir=out_dir,
            launch=True,
        )
        cli_args = build_cli_args(setup_result)
        if "--out-dir" not in cli_args:
            cli_args = ["--out-dir", str(out_dir)] + cli_args
        click.echo(f"Loaded preset from: {preset}")
        invoke_subcommand(deps.sweep, click.get_current_context(), "bank", cli_args)
        return

    out_dir = Path(out_dir)
    if job_id is None:
        job_id = deps.generate_job_id()

    executor = None
    if cloud:
        from bilancio.runners import CloudExecutor

        executor = CloudExecutor(
            experiment_id=job_id,
            download_artifacts=False,
            local_output_dir=out_dir,
            job_id=job_id,
        )
        click.echo("Cloud execution enabled")

    from bilancio.scenarios.ring.topology import parse_topology_string

    parsed_topologies = [parse_topology_string(t.strip()) for t in topologies.split(",")]

    performance = {"fast_atomic": True} if fast_atomic else {}
    config = BankComparisonConfig(
        name_prefix=job_id,
        n_agents=n_agents,
        maturity_days=maturity_days,
        Q_total=Decimal(q_total),
        base_seed=base_seed,
        n_replicates=n_replicates,
        kappas=as_decimal_list(kappas),
        concentrations=as_decimal_list(concentrations),
        mus=as_decimal_list(mus),
        outside_mid_ratios=as_decimal_list(outside_mid_ratios),
        face_value=face_value,
        rollover_enabled=rollover,
        quiet=quiet,
        default_handling=default_handling,
        n_banks=n_banks,
        reserve_ratio=reserve_ratio,
        credit_risk_loading=credit_risk_loading,
        max_borrower_risk=max_borrower_risk,
        min_coverage_ratio=min_coverage_ratio,
        cb_rate_escalation_slope=cb_rate_escalation_slope,
        cb_max_outstanding_ratio=cb_max_outstanding_ratio,
        performance=performance,
        topologies=parsed_topologies,
    )

    runner = BankComparisonRunner(
        config=config,
        out_dir=out_dir,
        executor=executor,
        job_id=job_id,
        enable_supabase=cloud,
    )

    click.echo(f"Job ID: {job_id}")

    preflight: Any | None = None
    try:
        from bilancio.scenarios.sweep_diagnostics import run_preflight_checks

        preflight = run_preflight_checks(config)
        preflight.print_summary()
    except CLI_HANDLED_ERRORS as exc:
        click.echo(f"Warning: Pre-flight checks failed: {exc}")

    n_combos = (
        len(as_decimal_list(kappas))
        * len(as_decimal_list(concentrations))
        * len(as_decimal_list(mus))
        * len(as_decimal_list(outside_mid_ratios))
    )
    click.echo(f"Preparing {n_combos * 2} runs ({n_combos} combos × 2 arms)...")

    results = runner.run_all()
    completed = sum(1 for result in results if result.bank_lending_effect is not None)
    improved = sum(1 for result in results if result.bank_lending_effect and result.bank_lending_effect > 0)

    click.echo("\nBank lending comparison complete!")
    click.echo(f"  Total combos: {len(results)}")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Lending helped: {improved}")
    click.echo(f"\nResults at: {out_dir / 'aggregate' / 'comparison.csv'}")

    if preflight is not None and results:
        try:
            from bilancio.scenarios.sweep_diagnostics import run_postsweep_validation

            postflight = run_postsweep_validation(preflight, results, aggregate_dir=out_dir / "aggregate")
            postflight.print_summary()
        except CLI_HANDLED_ERRORS as exc:
            click.echo(f"Warning: Post-sweep validation failed: {exc}")

    deps._offer_post_sweep_analysis(out_dir, "bank", post_analysis, cloud=cloud)


@click.command("analyze")
@click.option(
    "--experiment",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to completed sweep directory",
)
@click.option(
    "--sweep-type",
    required=True,
    type=click.Choice(["dealer", "bank", "nbfi"]),
    help="Type of sweep to analyse",
)
@click.option(
    "--post-analysis",
    type=str,
    default=None,
    help=(
        "Analyses to run: 'all', or comma-separated list. Valid: drilldowns, "
        "deltas, dynamics, narrative, strategy_outcomes, dealer_usage, "
        "mechanism_activity, treynor. Default: interactive prompt."
    ),
)
def sweep_analyze(experiment: Path, sweep_type: str, post_analysis: str | None) -> None:
    """Run post-sweep analysis on a previously completed sweep."""
    _deps()._offer_post_sweep_analysis(experiment, sweep_type, post_analysis or "all")


@click.command("setup")
@click.option(
    "--preset",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Load a preset YAML to pre-fill defaults",
)
def sweep_setup(preset: Path | None) -> None:
    """Interactive sweep configuration questionnaire."""
    from bilancio.ui.sweep_setup import build_cli_args, run_sweep_setup

    deps = _deps()
    result = run_sweep_setup(preset=preset)
    if not result.launch:
        click.echo("\nSetup complete (not launched).")
        return

    cli_args = build_cli_args(result)
    if result.out_dir is None:
        job_id = deps.generate_job_id()
        out_dir = Path(f"out/{result.sweep_type}_{job_id}")
        cli_args = ["--out-dir", str(out_dir)] + cli_args

    click.echo(f"\nLaunching: bilancio sweep {result.sweep_type} {' '.join(cli_args)}")
    invoke_subcommand(deps.sweep, click.get_current_context(), result.sweep_type, cli_args)
