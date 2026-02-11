"""Common utilities and types for visualization modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import math

try:
    from rich.console import Console, RenderableType
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    RenderableType = Any  # type: ignore[misc]


def _format_currency(amount: int, show_sign: bool = False) -> str:
    """Format an integer amount as currency."""
    formatted = f"{amount:,}"
    if show_sign and amount > 0:
        formatted = f"+{formatted}"
    return formatted


def _print(text: str, console: Optional['Console'] = None) -> None:
    """Print using Rich console if available, otherwise regular print."""
    if console:
        console.print(text)
    else:
        print(text)


def _format_agent(agent_id: str, system: Any) -> str:
    """Format agent as 'Name [ID]' if available.

    Args:
        agent_id: The agent ID to format
        system: The System instance (imported locally to avoid circular imports)
    """
    ag = system.state.agents.get(agent_id)
    if ag is None:
        return agent_id
    if ag.name and ag.name != agent_id:
        return f"{ag.name} [{agent_id}]"
    return agent_id


def parse_day_from_maturity(maturity_str: Optional[str]) -> int:
    """Parse a day number from maturity strings like 'Day 42'.

    Returns an integer day. If the input cannot be parsed, returns a large
    sentinel value to sort unknown maturities last.
    """
    if not isinstance(maturity_str, str):
        return math.inf  # type: ignore[return-value]
    s = maturity_str.strip()
    if not s.startswith("Day "):
        return math.inf  # type: ignore[return-value]
    try:
        return int(s[4:].strip())
    except Exception:
        return math.inf  # type: ignore[return-value]


@dataclass
class BalanceRow:
    name: str
    quantity: Optional[int]
    value_minor: Optional[int]
    counterparty_name: Optional[str]
    maturity: Optional[str]
    id_or_alias: Optional[str] = None


@dataclass
class TAccount:
    assets: list[BalanceRow]
    liabilities: list[BalanceRow]
