"""Shared utilities for CLI commands."""

from __future__ import annotations

from decimal import Decimal

from rich.console import Console

from bilancio.experiments.ring import _decimal_list


console = Console()


def _as_decimal_list(value: object) -> list[Decimal]:
    """Convert value to list of Decimals.

    Handles both list/tuple input and comma-separated string input.
    """
    if isinstance(value, (list, tuple)):
        return [Decimal(str(item)) for item in value]
    return _decimal_list(str(value))
