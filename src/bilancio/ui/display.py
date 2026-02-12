"""Display utilities for Bilancio CLI."""

from typing import Optional, List, Dict, Any, Union
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bilancio.engines.system import System
from bilancio.analysis.visualization import (
    display_agent_balance_table,
    display_multiple_agent_balances,
    display_events,
    display_events_for_day,
    display_agent_balance_table_renderable,
    display_multiple_agent_balances_renderable,
    display_events_renderable,
    display_events_for_day_renderable,
    display_events_table_renderable,
    display_agent_t_account_renderable,
    display_events_tables_by_phase_renderables
)
from bilancio.analysis.balances import system_trial_balance
from bilancio.core.errors import DefaultError, ValidationError


console = Console()


def show_scenario_header_renderable(name: str, description: Optional[str] = None, agents: Optional[Any] = None) -> List[RenderableType]:
    """Return renderables for scenario header including agent list.

    Args:
        name: Scenario name
        description: Optional scenario description
        agents: Optional list or dict of agent configs to display

    Returns:
        List of renderables
    """
    renderables: List[RenderableType] = []
    
    # Add scenario header panel
    header = Text(name, style="bold cyan")
    if description:
        header.append(f"\n{description}", style="dim")
    
    renderables.append(Panel(
        header,
        title="Bilancio Scenario",
        border_style="cyan"
    ))
    
    # Add agent list if provided
    if agents:
        from rich.table import Table
        
        agent_table = Table(title="Agents", box=None, show_header=False, padding=(0, 2))
        agent_table.add_column("ID", style="bold yellow")
        agent_table.add_column("Name", style="white")
        agent_table.add_column("Type", style="cyan")
        
        # Handle both list and dict formats
        if isinstance(agents, list):
            # agents is a list of AgentSpec objects
            for agent_config in agents:
                agent_id = agent_config.id if hasattr(agent_config, 'id') else str(agent_config.get('id', 'unknown'))
                agent_type = agent_config.kind if hasattr(agent_config, 'kind') else str(agent_config.get('kind', 'unknown'))
                agent_name = agent_config.name if hasattr(agent_config, 'name') else str(agent_config.get('name', agent_id))
                agent_table.add_row(agent_id, agent_name, f"({agent_type})")
        elif isinstance(agents, dict):
            # agents is a dictionary
            for agent_id, agent_config in agents.items():
                agent_type = agent_config.kind if hasattr(agent_config, 'kind') else str(agent_config.get('kind', 'unknown'))
                agent_name = agent_config.name if hasattr(agent_config, 'name') else str(agent_config.get('name', agent_id))
                agent_table.add_row(agent_id, agent_name, f"({agent_type})")
        
        renderables.append(Text())  # Empty line
        renderables.append(agent_table)
    
    return renderables


def show_scenario_header(name: str, description: Optional[str] = None, agents: Optional[Dict[str, Any]] = None) -> None:
    """Display scenario header.
    
    Args:
        name: Scenario name
        description: Optional scenario description
        agents: Optional dict of agent configs to display
    """
    renderables = show_scenario_header_renderable(name, description, agents)
    for renderable in renderables:
        console.print(renderable)


def show_day_summary_renderable(
    system: System,
    agent_ids: Optional[List[str]] = None,
    event_mode: str = "detailed",
    day: Optional[int] = None,
    t_account: bool = False
) -> List[RenderableType]:
    """Return renderables for a simulation day summary.

    Args:
        system: System instance
        agent_ids: Agent IDs to show balances for
        event_mode: "summary", "detailed", "table", or "none"
        day: Day number to display events for (None for all)

    Returns:
        List of renderables (empty list if event_mode="none")
    """
    # Plan 030: "none" mode suppresses all output for sweep performance
    if event_mode == "none":
        return []

    renderables: List[RenderableType] = []

    # Show events
    if day is not None:
        # Show events for specific day
        events_for_day = [e for e in system.state.events if e.get("day") == day]
        if events_for_day:
            renderables.append(Text("\nEvents:", style="bold"))
            if event_mode == "table":
                # Show 3 separate tables by phase (A/B/C)
                phase_tables = display_events_tables_by_phase_renderables(events_for_day, day=day)
                renderables.extend(phase_tables)
            elif event_mode == "detailed":
                event_renderables = display_events_renderable(events_for_day, format="detailed")
                renderables.extend(event_renderables)
            else:
                # Summary mode - just count by type
                event_counts_map: Dict[str, int] = {}
                for event in events_for_day:
                    event_type = str(event.get("kind", "Unknown"))
                    event_counts_map[event_type] = event_counts_map.get(event_type, 0) + 1
                for event_type, count in sorted(event_counts_map.items()):
                    renderables.append(Text(f"  • {event_type}: {count}"))
    else:
        # Show all recent events (initial view, typically setup/day 0)
        if system.state.events:
            renderables.append(Text("\nEvents:", style="bold"))
            if event_mode == "table":
                # Use phase-separated tables even in the initial no-day view.
                # If these are setup events, this will render a single "Setup" table.
                # Otherwise, it renders Phase B/C tables for the current day.
                # Determine a representative day to label (default to 0 if any setup events exist).
                rep_day = 0 if any(e.get("phase") == "setup" for e in system.state.events) else system.state.day
                phase_tables = display_events_tables_by_phase_renderables(system.state.events, day=rep_day)
                renderables.extend(phase_tables)
            elif event_mode == "detailed":
                event_renderables = display_events_renderable(system.state.events, format="detailed")
                renderables.extend(event_renderables)
            else:
                # Summary mode
                event_counts_map2: Dict[str, int] = {}
                for event in system.state.events:
                    event_type = str(event.get("kind", "Unknown"))
                    event_counts_map2[event_type] = event_counts_map2.get(event_type, 0) + 1
                for event_type, count in sorted(event_counts_map2.items()):
                    renderables.append(Text(f"  • {event_type}: {count}"))
    
    # Show balances
    if agent_ids is not None:
        if len(agent_ids) == 0:
            renderables.append(Text("\nBalances:", style="bold"))
            renderables.append(Text("  • No active agents to display", style="dim"))
        elif len(agent_ids) == 1:
            renderables.append(Text("\nBalances:", style="bold"))
            # Single agent - show detailed balance
            if t_account:
                balance_renderable = display_agent_t_account_renderable(system, agent_ids[0])
            else:
                balance_renderable = display_agent_balance_table_renderable(system, agent_ids[0], format="rich")
            renderables.append(balance_renderable)
        else:
            renderables.append(Text("\nBalances:", style="bold"))
            # Multiple agents
            if t_account:
                # Render one T-account per agent, stacked for readability
                for aid in agent_ids:
                    renderables.append(display_agent_t_account_renderable(system, aid))
                    renderables.append(Text(""))  # spacing
            else:
                # Show compact legacy tables side by side
                balance_renderable = display_multiple_agent_balances_renderable(system, agent_ids, format="rich")  # type: ignore[arg-type]
                renderables.append(balance_renderable)
    else:
        # Show system trial balance
        renderables.append(Text("\nSystem Trial Balance:", style="bold"))
        trial_bal = system_trial_balance(system)
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        
        total_assets = trial_bal.total_financial_assets
        total_liabilities = trial_bal.total_financial_liabilities
        
        table.add_row("Total Assets", f"{total_assets:,.2f}")
        table.add_row("Total Liabilities", f"{total_liabilities:,.2f}")
        table.add_row("Total Equity", f"{total_assets - total_liabilities:,.2f}")
        
        # Check if balanced
        diff = abs(total_assets - total_liabilities)
        if diff < 0.01:
            table.add_row("Status", "[green]OK Balanced[/green]")
        else:
            table.add_row("Status", f"[red]X Imbalanced ({diff:,.2f})[/red]")
        
        renderables.append(table)
    
    return renderables


def show_day_summary(
    system: System,
    agent_ids: Optional[List[str]] = None,
    event_mode: str = "detailed",
    day: Optional[int] = None,
    t_account: bool = False
) -> None:
    """Display summary for a simulation day.
    
    Args:
        system: System instance
        agent_ids: Agent IDs to show balances for
        event_mode: "summary" or "detailed"
        day: Day number to display events for (None for all)
    """
    renderables = show_day_summary_renderable(system, agent_ids, event_mode, day, t_account)
    for renderable in renderables:
        console.print(renderable)


def show_simulation_summary_renderable(system: System) -> Panel:
    """Return a renderable for final simulation summary.
    
    Args:
        system: System instance
        
    Returns:
        Panel renderable
    """
    return Panel(
        f"[bold]Final State[/bold]\n"
        f"Day: {system.state.day}\n"
        f"Total Events: {len(system.state.events)}\n"
        f"Active Agents: {len(system.state.agents)}\n"
        f"Active Contracts: {len(system.state.contracts)}\n"
        f"Stock Lots: {len(system.state.stocks)}",
        title="Summary",
        border_style="green"
    )


def show_simulation_summary(system: System) -> None:
    """Display final simulation summary.
    
    Args:
        system: System instance
    """
    console.print(show_simulation_summary_renderable(system))


def show_error_panel(
    error: Exception,
    phase: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Display error in a formatted panel.
    
    Args:
        error: Exception that occurred
        phase: Phase where error occurred (setup, day_N, simulation)
        context: Additional context information
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Build error content
    content = Text()
    content.append(f"Type: {error_type}\n", style="red bold")
    content.append(f"Message: {error_msg}\n", style="red")
    
    if context:
        content.append("\nContext:\n", style="yellow")
        for key, value in context.items():
            content.append(f"  • {key}: {value}\n", style="dim")
    
    # Add helpful hints based on error type
    if isinstance(error, DefaultError):
        content.append("\n💡 ", style="yellow")
        content.append("This usually means a debtor lacks sufficient funds or assets to settle an obligation.\n", style="dim")
        content.append("Check the agent's balance sheet and available means of payment.", style="dim")
        
    elif isinstance(error, ValidationError):
        content.append("\n💡 ", style="yellow")
        content.append("This indicates a system invariant violation.\n", style="dim")
        content.append("Common causes: duplicate IDs, negative balances, or mismatched references.", style="dim")
    
    # Display the panel
    console.print(Panel(
        content,
        title=f"Error in {phase}",
        border_style="red",
        expand=False
    ))
    
    # Log the error as an event
    if context and "scenario" in context:
        console.print(f"[dim]Error logged to simulation events[/dim]")
