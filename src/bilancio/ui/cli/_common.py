"""Shared helpers for CLI modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

import click
from click.core import ParameterSource
from rich.console import Console
from rich.panel import Panel

console = Console()

CLI_HANDLED_ERRORS = (
    click.ClickException,
    FileNotFoundError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    RuntimeError,
)


def as_decimal_list(value: object) -> list[Decimal]:
    """Convert either a CSV string or sequence into Decimals."""
    if isinstance(value, list | tuple):
        return [Decimal(str(item)) for item in value]
    out: list[Decimal] = []
    for part in str(value).split(","):
        item = part.strip()
        if not item:
            continue
        out.append(Decimal(item))
    return out


def parameter_uses_default(ctx: click.Context, param_name: str) -> bool:
    """Return whether a Click parameter still uses its default source."""
    source = ctx.get_parameter_source(param_name)
    return source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)


def exit_with_panel(
    message: str,
    *,
    title: str = "Error",
    border_style: str = "red",
) -> None:
    """Render a Rich error panel and terminate the command."""
    console.print(Panel(message, title=title, border_style=border_style))
    raise click.exceptions.Exit(1)


def optional_extra_message(feature: str, extra: str) -> str:
    """Explain how to install an optional dependency profile."""
    return f"{feature} requires optional dependencies from the '{extra}' extra. Install them with: uv pip install -e '.[{extra}]'"


def optional_dependency_error(feature: str, extra: str, exc: ImportError) -> click.ClickException:
    """Wrap an ImportError with actionable extra-install guidance."""
    return click.ClickException(f"{optional_extra_message(feature, extra)} ({exc})")


def command_or_raise(group: click.Group, ctx: click.Context, command_name: str) -> click.Command:
    """Resolve a subcommand or fail with a ClickException."""
    command = group.get_command(ctx, command_name)
    if command is None:
        raise click.ClickException(f"Unknown sweep type '{command_name}'")
    return command


def invoke_subcommand(
    group: click.Group,
    ctx: click.Context,
    command_name: str,
    cli_args: Iterable[str],
) -> None:
    """Invoke a sibling Click command with synthetic CLI args."""
    command = command_or_raise(group, ctx, command_name)
    sub_ctx = command.make_context(command_name, list(cli_args), parent=ctx)
    with sub_ctx:
        command.invoke(sub_ctx)


def build_performance_config(flags: Mapping[str, object]) -> Any:
    """Build a PerformanceConfig from a mutable CLI-flags mapping."""
    if not flags:
        return None

    from bilancio.core.performance import PerformanceConfig

    perf_kwargs = dict(flags)
    preset = str(perf_kwargs.pop("preset", "compatible"))
    return PerformanceConfig.create(preset, **perf_kwargs)
