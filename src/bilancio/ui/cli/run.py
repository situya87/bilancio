"""CLI commands for running and analyzing scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from bilancio.analysis.loaders import read_balances_csv, read_events_jsonl
from bilancio.analysis.report import (
    compute_day_metrics,
    parse_day_ranges,
    write_day_metrics_csv,
    write_day_metrics_json,
    write_debtor_shares_csv,
    write_intraday_csv,
    write_metrics_html,
)
from bilancio.ui.run import run_scenario
from bilancio.ui.wizard import create_scenario_wizard

console = Console()
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


@click.command()
@click.argument("scenario_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["step", "until-stable"]),
    default="until-stable",
    help="Simulation run mode",
)
@click.option("--max-days", type=int, default=90, help="Maximum days to simulate")
@click.option("--quiet-days", type=int, default=2, help="Required quiet days for stable state")
@click.option(
    "--show",
    type=click.Choice(["summary", "detailed", "table"]),
    default="detailed",
    help="Event display mode",
)
@click.option(
    "--agents",
    type=str,
    default=None,
    help="Comma-separated list of agent IDs to show balances for",
)
@click.option(
    "--check-invariants",
    type=click.Choice(["setup", "daily", "none"]),
    default="setup",
    help="When to check system invariants",
)
@click.option(
    "--export-balances",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to export balances CSV",
)
@click.option(
    "--export-events",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to export events JSONL",
)
@click.option(
    "--html",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to export colored output as HTML",
)
@click.option(
    "--t-account/--no-t-account", default=False, help="Use detailed T-account layout for balances"
)
@click.option(
    "--default-handling",
    type=click.Choice(["fail-fast", "expel-agent"]),
    default=None,
    help="Default-handling mode (override scenario setting)",
)
def run(
    scenario_file: Path,
    mode: str,
    max_days: int,
    quiet_days: int,
    show: str,
    agents: str | None,
    check_invariants: str,
    export_balances: Path | None,
    export_events: Path | None,
    html: Path | None,
    t_account: bool,
    default_handling: str | None,
) -> None:
    """Run a Bilancio simulation scenario.

    Load a scenario from a YAML file and run the simulation either
    step-by-step or until a stable state is reached.
    """
    try:
        # Parse agent list if provided
        agent_ids = None
        if agents:
            agent_ids = [a.strip() for a in agents.split(",")]

        # Override export paths if provided via CLI
        export_dict: dict[str, str] = {}
        if export_balances:
            export_dict["balances_csv"] = str(export_balances)
        if export_events:
            export_dict["events_jsonl"] = str(export_events)

        # Run the scenario
        run_scenario(
            path=scenario_file,
            mode=mode,
            max_days=max_days,
            quiet_days=quiet_days,
            show=show,
            agent_ids=agent_ids,
            check_invariants=check_invariants,
            export=export_dict if export_dict else None,
            html_output=html,
            t_account=t_account,
            default_handling=default_handling,
        )

    except FileNotFoundError as e:
        console.print(Panel(f"[red]File not found:[/red] {e}", title="Error", border_style="red"))
        sys.exit(1)

    except ValueError as e:
        console.print(
            Panel(f"[red]Configuration error:[/red]\n{e}", title="Error", border_style="red")
        )
        sys.exit(1)

    except CLI_HANDLED_ERRORS as e:
        console.print(
            Panel(f"[red]Unexpected error:[/red]\n{e}", title="Error", border_style="red")
        )
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


@click.command()
@click.argument("scenario_file", type=click.Path(exists=True, path_type=Path))
def validate(scenario_file: Path) -> None:
    """Validate a Bilancio scenario configuration file.

    Check that a YAML configuration file is valid without running
    the simulation. Reports any errors in the configuration structure,
    agent definitions, or initial actions.
    """
    try:
        from bilancio.config import apply_to_system, load_yaml
        from bilancio.engines.system import System

        # Load and parse the configuration
        console.print(f"[dim]Validating {scenario_file}...[/dim]")
        config = load_yaml(scenario_file)

        console.print("[green]OK[/green] Configuration syntax is valid")
        console.print(f"  Name: {config.name}")
        console.print(f"  Version: {config.version}")
        console.print(f"  Agents: {len(config.agents)}")
        console.print(f"  Initial actions: {len(config.initial_actions)}")

        # Try to apply to a test system to validate actions
        console.print("[dim]Checking if configuration can be applied...[/dim]")
        test_system = System()
        apply_to_system(config, test_system)

        console.print("[green]OK[/green] Configuration can be applied successfully")

        # Run invariant checks
        test_system.assert_invariants()
        console.print("[green]OK[/green] System invariants pass")

        # Summary
        console.print("\n[bold green]Configuration is valid![/bold green]")
        console.print("\nAgents defined:")
        for agent in config.agents:
            console.print(f"  • {agent.id} ({agent.kind}): {agent.name}")

        if config.run.export.balances_csv or config.run.export.events_jsonl:
            console.print("\nExports configured:")
            if config.run.export.balances_csv:
                console.print(f"  • Balances: {config.run.export.balances_csv}")
            if config.run.export.events_jsonl:
                console.print(f"  • Events: {config.run.export.events_jsonl}")

    except FileNotFoundError as e:
        console.print(Panel(f"[red]File not found:[/red] {e}", title="Error", border_style="red"))
        sys.exit(1)

    except ValueError as e:
        console.print(
            Panel(
                f"[red]Configuration error:[/red]\n{e}",
                title="Validation Failed",
                border_style="red",
            )
        )
        sys.exit(1)

    except CLI_HANDLED_ERRORS as e:
        console.print(
            Panel(
                f"[red]Validation error:[/red]\n{e}", title="Validation Failed", border_style="red"
            )
        )
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


@click.command()
@click.option("--from", "from_template", type=str, default=None, help="Base template to use")
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), required=True, help="Output YAML file path"
)
def new(from_template: str | None, output: Path) -> None:
    """Create a new scenario configuration.

    Interactive wizard to create a new Bilancio scenario
    configuration file.
    """
    try:
        create_scenario_wizard(output, from_template)
        console.print(f"[green]OK[/green] Created scenario file: {output}")

    except CLI_HANDLED_ERRORS as e:
        console.print(
            Panel(f"[red]Failed to create scenario:[/red]\n{e}", title="Error", border_style="red")
        )
        sys.exit(1)


@click.command()
@click.option(
    "--events",
    "events_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to events JSONL exported by a run",
)
@click.option(
    "--balances",
    "balances_path",
    type=click.Path(exists=False, path_type=Path),
    required=False,
    help="Path to balances CSV (optional, improves G_t/M_t)",
)
@click.option(
    "--days",
    type=str,
    default=None,
    help='Days to analyze, e.g. "1,2-3". Default: infer from events',
)
@click.option(
    "--out-csv",
    "out_csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV for day-level metrics",
)
@click.option(
    "--out-json",
    "out_json",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON for day-level metrics",
)
@click.option(
    "--intraday-csv",
    "intraday_csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional CSV for intraday P_prefix steps",
)
@click.option(
    "--html",
    "html_out",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional HTML analytics report",
)
def analyze(
    events_path: Path,
    balances_path: Path | None,
    days: str | None,
    out_csv: Path | None,
    out_json: Path | None,
    intraday_csv: Path | None,
    html_out: Path | None,
) -> None:
    """Analyze a completed run and export Kalecki-style metrics.

    Produces a day-level metrics CSV/JSON, optional intraday CSV (diagnostic).
    """
    # Load inputs
    console.print(f"[dim]Reading events from {events_path}...[/dim]")
    events = list(read_events_jsonl(events_path))

    balances_rows = None
    if balances_path and balances_path.exists():
        console.print(f"[dim]Reading balances from {balances_path}...[/dim]")
        balances_rows = read_balances_csv(balances_path)

    day_list = parse_day_ranges(days) if days else None

    bundle = compute_day_metrics(events, balances_rows, day_list)

    if not bundle["day_metrics"]:
        console.print("[yellow]No days found to analyze.[/yellow]")
        return

    # Determine default output paths if not provided
    base = events_path.stem.replace("_events", "") or "metrics"
    out_dir = events_path.parent
    if not out_csv:
        out_csv = out_dir / f"{base}_metrics_day.csv"
    if not out_json:
        out_json = out_dir / f"{base}_metrics_day.json"
    if intraday_csv:
        intraday_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    metrics_rows = bundle["day_metrics"]
    ds_rows = bundle["debtor_shares"]
    intraday_rows = bundle["intraday"]

    write_day_metrics_csv(out_csv, metrics_rows)
    console.print(f"[green]OK[/green] Wrote day metrics CSV: {out_csv}")
    write_day_metrics_json(out_json, metrics_rows)
    console.print(f"[green]OK[/green] Wrote day metrics JSON: {out_json}")

    # Debtor shares and intraday are optional; only write if path provided
    base_name = out_csv.stem.replace("_metrics_day", "") if out_csv else "metrics"
    ds_path = out_csv.parent / f"{base_name}_ds.csv"
    write_debtor_shares_csv(ds_path, ds_rows)
    console.print(f"[green]OK[/green] Wrote debtor shares CSV: {ds_path}")

    if intraday_csv:
        write_intraday_csv(intraday_csv, intraday_rows)
        console.print(f"[green]OK[/green] Wrote intraday CSV: {intraday_csv}")

    if html_out:
        title = f"Bilancio Analytics — {events_path.stem.replace('_events', '')}"
        subtitle = f"Events: {events_path.name}{' | Balances: ' + balances_path.name if balances_path else ''}"
        write_metrics_html(
            html_out, metrics_rows, ds_rows, intraday_rows, title=title, subtitle=subtitle
        )
        console.print(f"[green]OK[/green] Wrote HTML analytics: {html_out}")
