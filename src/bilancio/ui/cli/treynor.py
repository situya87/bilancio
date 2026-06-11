"""CLI command for Treynor pricing visualizations."""

from __future__ import annotations

from pathlib import Path

import click

from .utils import console


@click.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to a single run output directory.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output HTML file path. Defaults to <run-dir>/treynor_dashboard.html.",
)
def treynor(run_dir: str, output: str | None) -> None:
    """Generate interactive Treynor pricing diagrams from a simulation run.

    Reads dealer_state.csv and/or bank_state.csv from the run directory
    and produces an HTML dashboard with static pricing diagrams and
    day-by-day animations.

    Example:

        bilancio treynor --run-dir out/runs/my_run
        bilancio treynor --run-dir out/runs/my_run -o temp/treynor.html
    """
    from bilancio.analysis.treynor_viz import build_treynor_dashboard

    run_path = Path(run_dir)
    html = build_treynor_dashboard(run_path)

    if output is None:
        output_path = run_path / "treynor_dashboard.html"
    else:
        output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    console.print(f"[green]OK[/green] Treynor dashboard written to {output_path}")

    # Try to open in browser
    import webbrowser

    try:
        webbrowser.open(f"file://{output_path.resolve()}")
        console.print("[dim]Opened in browser[/dim]")
    except OSError:
        pass
