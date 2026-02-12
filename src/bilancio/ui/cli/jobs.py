"""Job management CLI commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click


def format_datetime(dt: Optional[datetime]) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def format_duration(start: datetime, end: Optional[datetime]) -> str:
    """Format duration between two datetimes."""
    if end is None:
        return "running"
    delta = end - start
    minutes = delta.total_seconds() / 60
    if minutes < 1:
        return f"{delta.total_seconds():.0f}s"
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes/60:.1f}h"


@click.group()
def jobs() -> None:
    """Query and manage simulation jobs."""
    pass


@jobs.command("ls")
@click.option(
    "--cloud", is_flag=True, help="Query from Supabase cloud storage"
)
@click.option(
    "--local",
    type=click.Path(exists=True, path_type=Path),
    help="Local directory containing job manifests",
)
@click.option(
    "--status",
    type=click.Choice(["pending", "running", "completed", "failed"]),
    help="Filter by job status",
)
@click.option("--limit", default=20, help="Maximum number of jobs to show")
def list_jobs(
    cloud: bool, local: Optional[Path], status: Optional[str], limit: int
) -> None:
    """List simulation jobs.

    By default, lists from Supabase if configured.

    Examples:
        bilancio jobs ls --cloud
        bilancio jobs ls --local out/experiments
        bilancio jobs ls --status completed --limit 10
    """
    jobs_list = []

    if cloud:
        # Query from Supabase
        try:
            from bilancio.jobs.supabase_store import SupabaseJobStore

            store = SupabaseJobStore()
            if store.client is None:
                raise click.ClickException(
                    "Supabase not configured. Set BILANCIO_SUPABASE_URL and "
                    "BILANCIO_SUPABASE_ANON_KEY environment variables."
                )
            jobs_list = store.list_jobs(status=status, limit=limit)
        except ImportError as e:
            raise click.ClickException(f"Failed to import Supabase: {e}")

    elif local:
        # Query from local filesystem
        from bilancio.jobs import JobManager

        manager = JobManager(jobs_dir=local)
        jobs_list = manager.list_jobs()

        # Apply filters
        if status:
            jobs_list = [j for j in jobs_list if j.status.value == status]

        # Sort by creation time (newest first) and limit
        jobs_list = sorted(jobs_list, key=lambda j: j.created_at, reverse=True)[
            :limit
        ]

    else:
        # Try Supabase first, fall back to error
        try:
            from bilancio.storage.supabase_client import is_supabase_configured

            if is_supabase_configured():
                from bilancio.jobs.supabase_store import SupabaseJobStore

                store = SupabaseJobStore()
                jobs_list = store.list_jobs(status=status, limit=limit)
            else:
                click.echo(
                    "No source specified. Use --cloud or --local <path>.\n"
                    "Tip: Set BILANCIO_SUPABASE_* env vars to use --cloud."
                )
                return
        except Exception as e:  # Intentionally broad: top-level CLI handler
            raise click.ClickException(f"Failed to query jobs: {e}")

    if not jobs_list:
        click.echo("No jobs found.")
        return

    # Get run counts from Supabase if available
    run_counts: dict[str, int] = {}
    if cloud or (not local):  # If using cloud or auto-detected Supabase
        try:
            from bilancio.jobs.supabase_store import SupabaseJobStore

            store = SupabaseJobStore()
            job_ids = [j.job_id for j in jobs_list]
            run_counts = store.get_run_counts(job_ids)
        except Exception:  # Intentionally broad: external service call
            pass  # Fall back to job.run_ids

    # Display jobs
    click.echo(f"{'JOB ID':<36} {'STATUS':<10} {'CREATED':<16} {'DURATION':<10} {'RUNS':<6}")
    click.echo("-" * 80)

    for job in jobs_list:
        duration = format_duration(job.created_at, job.completed_at)
        # Prefer run count from Supabase, fall back to job.run_ids
        runs = run_counts.get(job.job_id, len(job.run_ids) if job.run_ids else 0)
        click.echo(
            f"{job.job_id:<36} {job.status.value:<10} "
            f"{format_datetime(job.created_at):<16} {duration:<10} {runs:<6}"
        )

    click.echo("-" * 80)
    click.echo(f"Total: {len(jobs_list)} jobs")


@jobs.command("get")
@click.argument("job_id")
@click.option("--cloud", is_flag=True, help="Query from Supabase cloud storage")
@click.option(
    "--local",
    type=click.Path(exists=True, path_type=Path),
    help="Local directory containing job manifests",
)
def get_job(job_id: str, cloud: bool, local: Optional[Path]) -> None:
    """Get detailed information about a job.

    Examples:
        bilancio jobs get castle-river-mountain-forest
        bilancio jobs get castle-river-mountain-forest --cloud
    """
    job = None

    if cloud:
        try:
            from bilancio.jobs.supabase_store import SupabaseJobStore

            store = SupabaseJobStore()
            if store.client is None:
                raise click.ClickException("Supabase not configured.")
            job = store.get_job(job_id)
        except ImportError as e:
            raise click.ClickException(f"Failed to import Supabase: {e}")

    elif local:
        from bilancio.jobs import JobManager

        manager = JobManager(jobs_dir=local)
        job = manager.get_job(job_id)

    else:
        # Try Supabase first, then local
        try:
            from bilancio.storage.supabase_client import is_supabase_configured

            if is_supabase_configured():
                from bilancio.jobs.supabase_store import SupabaseJobStore

                store = SupabaseJobStore()
                job = store.get_job(job_id)
        except Exception:  # Intentionally broad: external service call
            pass

        if job is None:
            # Try common local paths
            for path in [Path("out/experiments"), Path(".")]:
                if path.exists():
                    from bilancio.jobs import JobManager

                    manager = JobManager(jobs_dir=path)
                    job = manager.get_job(job_id)
                    if job:
                        break

    if job is None:
        raise click.ClickException(f"Job not found: {job_id}")

    # Display job details
    click.echo(f"Job ID:      {job.job_id}")
    click.echo(f"Status:      {job.status.value}")
    click.echo(f"Description: {job.description}")
    click.echo(f"Created:     {format_datetime(job.created_at)}")
    click.echo(f"Completed:   {format_datetime(job.completed_at)}")
    if job.error:
        click.echo(f"Error:       {job.error}")
    click.echo()

    # Configuration
    click.echo("Configuration:")
    config = job.config
    click.echo(f"  Sweep Type:    {config.sweep_type}")
    click.echo(f"  N Agents:      {config.n_agents}")
    click.echo(f"  Maturity Days: {config.maturity_days}")
    click.echo(f"  Kappas:        {', '.join(str(k) for k in config.kappas)}")
    click.echo(f"  Concentrations: {', '.join(str(c) for c in config.concentrations)}")
    click.echo(f"  Mus:           {', '.join(str(m) for m in config.mus)}")
    click.echo(f"  Cloud:         {config.cloud}")
    click.echo()

    # Runs
    if job.run_ids:
        click.echo(f"Runs ({len(job.run_ids)}):")
        for run_id in job.run_ids[:10]:  # Show first 10
            click.echo(f"  - {run_id}")
        if len(job.run_ids) > 10:
            click.echo(f"  ... and {len(job.run_ids) - 10} more")


@jobs.command("runs")
@click.argument("job_id")
@click.option("--cloud", is_flag=True, help="Query from Supabase cloud storage")
@click.option(
    "--status",
    type=click.Choice(["pending", "running", "completed", "failed"]),
    help="Filter by run status",
)
def list_runs(job_id: str, cloud: bool, status: Optional[str]) -> None:
    """List runs for a specific job.

    Examples:
        bilancio jobs runs castle-river-mountain-forest
        bilancio jobs runs castle-river-mountain-forest --cloud
    """
    if not cloud:
        click.echo("Note: Use --cloud to query runs from Supabase.")
        click.echo("Local run listing requires reading registry CSV files.")
        return

    try:
        from bilancio.storage.supabase_registry import SupabaseRegistryStore

        store = SupabaseRegistryStore()
        entries = store.query(job_id, {"status": status} if status else None)

        if not entries:
            click.echo(f"No runs found for job: {job_id}")
            return

        click.echo(f"{'RUN ID':<40} {'STATUS':<10} {'KAPPA':<8} {'CONC':<8} {'DELTA':<10}")
        click.echo("-" * 80)

        for entry in entries:
            kappa = entry.parameters.get("kappa", "-")
            conc = entry.parameters.get("concentration", "-")
            delta = entry.metrics.get("delta_total", "-")
            if isinstance(delta, float):
                delta = f"{delta:.4f}"
            click.echo(
                f"{entry.run_id:<40} {entry.status.value:<10} "
                f"{kappa:<8} {conc:<8} {delta:<10}"
            )

        click.echo("-" * 80)
        click.echo(f"Total: {len(entries)} runs")

    except Exception as e:  # Intentionally broad: top-level CLI handler
        raise click.ClickException(f"Failed to query runs: {e}")


@jobs.command("metrics")
@click.argument("job_id")
@click.option("--cloud", is_flag=True, help="Query from Supabase cloud storage")
def show_metrics(job_id: str, cloud: bool) -> None:
    """Show aggregate metrics for a job.

    Examples:
        bilancio jobs metrics castle-river-mountain-forest --cloud
    """
    if not cloud:
        click.echo("Use --cloud to query metrics from Supabase.")
        return

    try:
        from bilancio.storage.supabase_registry import SupabaseRegistryStore

        store = SupabaseRegistryStore()
        entries = store.query(job_id)

        if not entries:
            click.echo(f"No runs found for job: {job_id}")
            return

        # Calculate aggregate metrics
        completed = [e for e in entries if e.status.value == "completed"]

        if not completed:
            click.echo("No completed runs to show metrics for.")
            return

        # Collect delta values
        deltas = [
            e.metrics.get("delta_total", 0) for e in completed if "delta_total" in e.metrics
        ]
        phis = [
            e.metrics.get("phi_total", 0) for e in completed if "phi_total" in e.metrics
        ]

        click.echo(f"Job: {job_id}")
        click.echo(f"Completed runs: {len(completed)} / {len(entries)}")
        click.echo()

        if deltas:
            click.echo("Delta (default rate):")
            click.echo(f"  Mean:   {sum(deltas)/len(deltas):.4f}")
            click.echo(f"  Min:    {min(deltas):.4f}")
            click.echo(f"  Max:    {max(deltas):.4f}")

        if phis:
            click.echo("Phi (clearing rate):")
            click.echo(f"  Mean:   {sum(phis)/len(phis):.4f}")
            click.echo(f"  Min:    {min(phis):.4f}")
            click.echo(f"  Max:    {max(phis):.4f}")

    except Exception as e:  # Intentionally broad: top-level CLI handler
        raise click.ClickException(f"Failed to query metrics: {e}")


@jobs.command("visualize")
@click.argument("job_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for HTML file (default: temp/{job_id}_comparison.html)",
)
@click.option(
    "--title",
    "-t",
    help="Custom title for the visualization",
)
@click.option(
    "--open-browser",
    is_flag=True,
    help="Open the generated HTML in the default browser",
)
def visualize_job(
    job_id: str,
    output: Optional[Path],
    title: Optional[str],
    open_browser: bool,
) -> None:
    """Generate interactive visualization comparing passive vs active runs.

    Creates an HTML report with multiple visualization types:
    - Summary statistics
    - Passive vs active scatter plot
    - Default rate by liquidity (κ)
    - Faceted views by concentration (c)
    - Trading effect distributions
    - Heatmaps by maturity timing (μ)
    - Parallel coordinates overview
    - 3D surface plot

    Examples:
        bilancio jobs visualize evacuee-crushed-attentive-flakily
        bilancio jobs visualize my-job -o results/my-viz.html
        bilancio jobs visualize my-job --open-browser
    """
    try:
        from bilancio.analysis.visualization.run_comparison import (
            generate_comparison_html,
        )

        click.echo(f"Generating visualization for job: {job_id}")

        html_path = generate_comparison_html(
            job_id=job_id,
            output_path=output,
            title=title,
        )

        click.echo(f"Visualization saved to: {html_path}")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{html_path.absolute()}")
            click.echo("Opened in browser.")
        else:
            click.echo(f"Open with: open {html_path}")

    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:  # Intentionally broad: top-level CLI handler
        raise click.ClickException(f"Failed to generate visualization: {e}")
