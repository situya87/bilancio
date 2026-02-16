"""Modal Volume management commands."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from typing import Any

import click

CLI_HANDLED_ERRORS = (
    click.ClickException,
    FileNotFoundError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)


def get_volume_contents(volume_name: str = "bilancio-results") -> list[dict[str, Any]]:
    """Get contents of Modal Volume as list of dicts."""
    result = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", volume_name, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Failed to list volume: {result.stderr}")
    contents: list[dict[str, Any]] = json.loads(result.stdout)
    return contents


def delete_volume_path(volume_name: str, path: str) -> bool:
    """Delete a path from Modal Volume."""
    result = subprocess.run(
        ["uv", "run", "modal", "volume", "rm", volume_name, path, "-r"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def parse_modal_date(date_str: str) -> datetime:
    """Parse Modal's date format (e.g., '2026-01-12 16:20 CET')."""
    # Remove timezone abbreviation and parse
    parts = date_str.rsplit(" ", 1)
    date_part = parts[0]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d %H:%M")
    except ValueError:
        # Fallback: return now if parsing fails
        return datetime.now()


@click.group()
def volume() -> None:
    """Manage Modal Volume storage."""
    pass


@volume.command("ls")
@click.option("--volume", "volume_name", default="bilancio-results", help="Volume name")
def list_volume(volume_name: str) -> None:
    """List experiments in Modal Volume."""
    try:
        contents = get_volume_contents(volume_name)
    except CLI_HANDLED_ERRORS as e:
        raise click.ClickException(str(e)) from e

    if not contents:
        click.echo("Volume is empty.")
        return

    click.echo(f"Experiments in {volume_name}:")
    click.echo("-" * 60)

    for item in contents:
        name = item.get("Filename", "unknown")
        modified = item.get("Created/Modified", "unknown")
        item_type = item.get("Type", "unknown")
        if item_type == "dir":
            click.echo(f"  {name:<40} {modified}")

    click.echo("-" * 60)
    click.echo(f"Total: {len([i for i in contents if i.get('Type') == 'dir'])} experiments")


@volume.command("cleanup")
@click.option("--volume", "volume_name", default="bilancio-results", help="Volume name")
@click.option(
    "--older-than",
    type=int,
    default=None,
    help="Delete experiments older than N days",
)
@click.option(
    "--pattern",
    type=str,
    default=None,
    help="Delete experiments matching pattern (e.g., 'test_*')",
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def cleanup(
    volume_name: str,
    older_than: int | None,
    pattern: str | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Clean up old experiments from Modal Volume.

    Examples:
        bilancio volume cleanup --older-than 30
        bilancio volume cleanup --pattern "test_*" --dry-run
        bilancio volume cleanup --older-than 7 --yes
    """
    try:
        contents = get_volume_contents(volume_name)
    except CLI_HANDLED_ERRORS as e:
        raise click.ClickException(str(e)) from e

    # Filter directories only
    experiments = [i for i in contents if i.get("Type") == "dir"]

    if not experiments:
        click.echo("No experiments found in volume.")
        return

    # Apply filters
    to_delete = []
    now = datetime.now()

    for exp in experiments:
        name = exp.get("Filename", "")
        modified_str = exp.get("Created/Modified", "")
        modified = parse_modal_date(modified_str)

        should_delete = False

        # Age filter
        if older_than is not None:
            age_days = (now - modified).days
            if age_days >= older_than:
                should_delete = True

        # Pattern filter
        if pattern is not None:
            import fnmatch

            if fnmatch.fnmatch(name, pattern):
                should_delete = True

        # If no filters specified, don't delete anything
        if older_than is None and pattern is None:
            click.echo("Error: Specify --older-than or --pattern to select experiments to delete.")
            click.echo("Use 'bilancio volume ls' to see all experiments.")
            return

        if should_delete:
            to_delete.append(exp)

    if not to_delete:
        click.echo("No experiments match the criteria.")
        return

    # Show what will be deleted
    click.echo(f"{'[DRY RUN] ' if dry_run else ''}Experiments to delete:")
    click.echo("-" * 60)
    for exp in to_delete:
        name = exp.get("Filename", "")
        modified = exp.get("Created/Modified", "")
        click.echo(f"  {name:<40} {modified}")
    click.echo("-" * 60)
    click.echo(f"Total: {len(to_delete)} experiments")

    if dry_run:
        click.echo("\n[DRY RUN] No changes made.")
        return

    # Confirm
    if not yes:
        if not click.confirm(f"\nDelete {len(to_delete)} experiments?"):
            click.echo("Aborted.")
            return

    # Delete
    deleted = 0
    failed = 0
    for exp in to_delete:
        name = exp.get("Filename", "")
        click.echo(f"Deleting {name}...", nl=False)
        if delete_volume_path(volume_name, name):
            click.echo(" OK")
            deleted += 1
        else:
            click.echo(" FAILED")
            failed += 1

    click.echo(f"\nDeleted: {deleted}, Failed: {failed}")


@volume.command("rm")
@click.argument("experiment_id")
@click.option("--volume", "volume_name", default="bilancio-results", help="Volume name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def remove(experiment_id: str, volume_name: str, yes: bool) -> None:
    """Remove a specific experiment from Modal Volume.

    Example:
        bilancio volume rm castle-river-mountain-forest
    """
    if not yes:
        if not click.confirm(f"Delete experiment '{experiment_id}'?"):
            click.echo("Aborted.")
            return

    click.echo(f"Deleting {experiment_id}...", nl=False)
    if delete_volume_path(volume_name, experiment_id):
        click.echo(" OK")
    else:
        click.echo(" FAILED")
        raise click.ClickException("Failed to delete experiment")
