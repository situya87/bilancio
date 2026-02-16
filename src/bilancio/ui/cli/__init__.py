"""Command-line interface for Bilancio."""

from __future__ import annotations

import click

from .jobs import jobs
from .run import analyze, new, run, validate
from .sweep import sweep
from .volume import volume


@click.group()
def cli() -> None:
    """Bilancio - Economic simulation framework."""
    pass


# Add all commands to the main CLI group
cli.add_command(run)
cli.add_command(validate)
cli.add_command(new)
cli.add_command(analyze)
cli.add_command(sweep)
cli.add_command(volume)
cli.add_command(jobs)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


# Re-export for backwards compatibility
__all__ = ["cli", "main"]
