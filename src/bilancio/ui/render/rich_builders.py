"""Rich renderable builder functions for Bilancio UI components."""

from __future__ import annotations

from typing import List, Dict, Any, Union

from rich.table import Table
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box
from rich.console import RenderableType

from decimal import Decimal

from .models import AgentBalanceView, DayEventsView, DaySummaryView, BalanceItemView, EventView
from .formatters import registry as event_formatter_registry


def _format_currency(amount: Union[int, float, Decimal], show_sign: bool = False) -> str:
    """Format an amount as currency."""
    formatted = f"{amount:,}"
    if show_sign and amount > 0:
        formatted = f"+{formatted}"
    return formatted


def build_agent_balance_table(balance_view: AgentBalanceView) -> Table:
    """Build a Rich Table for a single agent's balance sheet.
    
    Args:
        balance_view: Agent balance view model
        
    Returns:
        Rich Table renderable
    """
    # Create title with agent info
    if balance_view.agent_name and balance_view.agent_name != balance_view.agent_id:
        title = f"{balance_view.agent_name} [{balance_view.agent_id}] ({balance_view.agent_kind})"
    else:
        title = f"{balance_view.agent_id} ({balance_view.agent_kind})"
    
    # Create the main table
    table = Table(title=title, box=box.ROUNDED, title_style="bold cyan")
    table.add_column("ASSETS", style="green", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="green", width=15, no_wrap=True)
    table.add_column("LIABILITIES", style="red", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="red", width=15, no_wrap=True)
    
    # Separate assets and liabilities
    assets = [item for item in balance_view.items if item.category == "Assets"]
    liabilities = [item for item in balance_view.items if item.category == "Liabilities"]
    
    # Determine the maximum number of rows needed
    max_rows = max(len(assets), len(liabilities), 1)
    
    for i in range(max_rows):
        asset_name = assets[i].instrument if i < len(assets) else ""
        asset_amount = _format_currency(assets[i].value) if i < len(assets) else ""
        liability_name = liabilities[i].instrument if i < len(liabilities) else ""
        liability_amount = _format_currency(liabilities[i].value) if i < len(liabilities) else ""
        
        table.add_row(asset_name, asset_amount, liability_name, liability_amount)
    
    # Calculate totals
    total_assets = sum(item.value for item in assets)
    total_liabilities = sum(item.value for item in liabilities)
    net_worth = total_assets - total_liabilities
    
    # Add separator and totals
    table.add_row("", "", "", "", end_section=True)
    table.add_row(
        Text("TOTAL ASSETS", style="bold green"),
        Text(_format_currency(total_assets), style="bold green"),
        Text("TOTAL LIABILITIES", style="bold red"),
        Text(_format_currency(total_liabilities), style="bold red")
    )
    
    # Add net worth
    table.add_row("", "", "", "", end_section=True)
    net_worth_style = "bold green" if net_worth >= 0 else "bold red"
    table.add_row(
        "",
        "",
        Text("NET WORTH", style="bold blue"),
        Text(_format_currency(net_worth, show_sign=True), style=net_worth_style)
    )
    
    return table


def build_multiple_agent_balances(balances: List[AgentBalanceView]) -> Columns:
    """Build a Rich Columns layout for multiple agent balance sheets.
    
    Args:
        balances: List of agent balance view models
        
    Returns:
        Rich Columns renderable
    """
    tables = [build_agent_balance_table(balance) for balance in balances]
    return Columns(tables, equal=True, expand=True)


def build_events_panel(day_events_view: DayEventsView) -> Panel:
    """Build a Rich Panel for day events organized by phase.
    
    Args:
        day_events_view: Day events view model
        
    Returns:
        Rich Panel renderable
    """
    if not day_events_view.phases:
        return Panel(
            Text("No events occurred on this day.", style="dim"),
            title=f"Day {day_events_view.day} Events",
            border_style="blue"
        )
    
    # Create a tree structure for phases and events
    tree = Tree(f"📅 Day {day_events_view.day}")
    
    for phase, events in day_events_view.phases.items():
        if not events:
            continue
            
        # Add phase node
        phase_node = tree.add(f"📍 {phase}")
        
        # Add events under the phase
        for event in events:
            event_text = Text()
            event_text.append(f"{event.icon} ", style="")
            event_text.append(event.title, style="bold")
            
            event_node = phase_node.add(event_text)
            
            # Add event details as sub-items
            for line in event.lines:
                event_node.add(Text(line, style="dim"))
    
    return Panel(
        tree,
        title=f"Day {day_events_view.day} Events",
        border_style="blue"
    )


def build_events_list(day_events_view: DayEventsView) -> List[RenderableType]:
    """Build a list of Rich renderables for day events.
    
    Args:
        day_events_view: Day events view model
        
    Returns:
        List of Rich renderables
    """
    renderables: List[RenderableType] = []

    if not day_events_view.phases:
        renderables.append(Text("No events occurred on this day.", style="dim"))
        return renderables

    for phase, events in day_events_view.phases.items():
        if not events:
            continue

        # Phase header
        renderables.append(Text(f"📍 {phase}", style="bold blue"))

        # Events in this phase
        for event in events:
            event_text = Text()
            event_text.append(f"  {event.icon} ", style="")
            event_text.append(event.title, style="bold")
            renderables.append(event_text)

            # Event details
            for line in event.lines:
                renderables.append(Text(f"    {line}", style="dim"))

        # Add spacing between phases
        renderables.append(Text(""))

    return renderables


def build_day_summary(view: DaySummaryView) -> List[RenderableType]:
    """Build a list of Rich renderables for a complete day summary.

    Args:
        view: Day summary view model

    Returns:
        List of Rich renderables
    """
    renderables: List[RenderableType] = []
    
    # Day header
    renderables.append(Panel(
        Text(f"Day {view.day} Summary", style="bold cyan"),
        border_style="cyan"
    ))
    
    # Events section
    if view.events_view.phases:
        renderables.append(build_events_panel(view.events_view))
    else:
        renderables.append(Text("No events occurred on this day.", style="dim"))
    
    # Agent balances section
    if view.agent_balances:
        renderables.append(Text("\n$ Agent Balances", style="bold yellow"))
        renderables.append(build_multiple_agent_balances(view.agent_balances))
    
    return renderables


def convert_raw_event_to_view(raw_event: Dict[str, Any]) -> EventView:
    """Convert a raw event dictionary to an EventView model.
    
    Args:
        raw_event: Raw event dictionary from system.state.events
        
    Returns:
        EventView model
    """
    title, lines, icon = event_formatter_registry.format(raw_event)
    
    return EventView(
        kind=raw_event.get("kind", "Unknown"),
        title=title,
        lines=lines,
        icon=icon,
        raw_event=raw_event
    )


def convert_raw_events_to_day_view(raw_events: List[Dict[str, Any]], day: int) -> DayEventsView:
    """Convert raw events for a day into a DayEventsView model.
    
    Args:
        raw_events: List of raw event dictionaries for the day
        day: Day number
        
    Returns:
        DayEventsView model
    """
    phases: Dict[str, List[EventView]] = {}
    
    for raw_event in raw_events:
        phase = raw_event.get("phase", "unknown")
        event_view = convert_raw_event_to_view(raw_event)
        
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(event_view)
    
    return DayEventsView(day=day, phases=phases)