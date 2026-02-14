"""Orchestration logic for running Bilancio simulations."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from bilancio.engines.system import System
from bilancio.engines.simulation import run_day, run_until_stable
from bilancio.core.errors import ValidationError, DefaultError
from bilancio.config import load_yaml, apply_to_system
from bilancio.export.writers import write_balances_csv, write_events_jsonl

from .display import (
    show_scenario_header,
    show_scenario_header_renderable,
    show_day_summary,
    show_day_summary_renderable,
    show_simulation_summary,
    show_simulation_summary_renderable,
    show_error_panel
)


console = Console(record=True, width=120)


def _filter_active_agent_ids(system: System, agent_ids: Optional[List[str]]) -> Optional[List[str]]:
    """Return only agent IDs that remain active (not defaulted)."""
    if agent_ids is None:
        return None
    active_ids: List[str] = []
    for aid in agent_ids:
        agent = system.state.agents.get(aid)
        if not agent:
            continue
        if agent.defaulted:
            continue
        active_ids.append(aid)
    return active_ids


def run_scenario(
    path: Path,
    mode: str = "until_stable",
    max_days: int = 90,
    quiet_days: int = 2,
    show: str = "detailed",
    agent_ids: Optional[List[str]] = None,
    check_invariants: str = "setup",
    export: Optional[Dict[str, str]] = None,
    html_output: Optional[Path] = None,
    t_account: bool = False,
    default_handling: Optional[str] = None,
    detailed_dealer_logging: bool = False,
    run_id: str = "",
    regime: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Run a Bilancio simulation scenario.

    Args:
        path: Path to scenario YAML file
        mode: "step" or "until_stable"
        max_days: Maximum days to simulate
        quiet_days: Required quiet days for stable state
        show: "summary", "detailed" or "table" for event display
        agent_ids: List of agent IDs to show balances for
        check_invariants: "setup", "daily", or "none"
        export: Dictionary with export paths (balances_csv, events_jsonl)
        html_output: Optional path to export HTML with colored output
        progress_callback: Optional callback(current_day, max_days) for progress tracking
    """
    # Load configuration
    console.print("[dim]Loading scenario...[/dim]")
    config = load_yaml(path)

    # Determine effective default-handling strategy (CLI override wins)
    effective_default_handling = default_handling or config.run.default_handling
    if default_handling and config.run.default_handling != default_handling:
        config = config.model_copy(update={
            "run": config.run.model_copy(update={"default_handling": effective_default_handling})
        })

    # Create and configure system with selected default-handling mode
    system = System(default_mode=effective_default_handling)
    # Preflight schedule validation (aliases available when referenced)
    try:
        from bilancio.config.apply import validate_scheduled_aliases
        validate_scheduled_aliases(config)
    except ValueError as e:
        show_error_panel(
            error=e,
            phase="setup",
            context={"scenario": config.name}
        )
        sys.exit(1)

    # Apply configuration
    try:
        apply_to_system(config, system)
        
        if check_invariants in ("setup", "daily"):
            system.assert_invariants()
            
    except (ValidationError, ValueError) as e:
        show_error_panel(
            error=e,
            phase="setup",
            context={"scenario": config.name}
        )
        sys.exit(1)
    
    # Plan 024: Enable rollover if configured
    system.state.rollover_enabled = config.run.rollover_enabled

    # Plan 034: Enable estimate logging if configured
    system.state.estimate_logging_enabled = config.run.estimate_logging

    # Stage scheduled actions into system state (Phase B1 execution by day)
    try:
        if getattr(config, 'scheduled_actions', None):
            for sa in config.scheduled_actions:
                day = sa.day
                system.state.scheduled_actions_by_day.setdefault(day, []).append(sa.action)
    except (AttributeError, TypeError):
        # Keep robust even if config lacks scheduled actions
        pass
    
    # Use config settings unless overridden by CLI
    if agent_ids is None and config.run.show.balances:
        agent_ids = config.run.show.balances
    
    if export is None:
        export = {}
    
    # Use config export settings if not overridden
    if not export.get('balances_csv') and config.run.export.balances_csv:
        export['balances_csv'] = config.run.export.balances_csv
    if not export.get('events_jsonl') and config.run.export.events_jsonl:
        export['events_jsonl'] = config.run.export.events_jsonl
    
    # Plan 030: Check for quiet mode (show="none") to suppress verbose output
    quiet_mode = show == "none"

    # Show scenario header with agent list (skip in quiet mode)
    if not quiet_mode:
        header_renderables = show_scenario_header_renderable(config.name, config.description, config.agents)
        for renderable in header_renderables:
            console.print(renderable)
        console.print(f"[dim]Default handling mode: {effective_default_handling}[/dim]")

        # Show initial state
        console.print("\n[bold cyan] Day 0 (After Setup)[/bold cyan]")
        renderables = show_day_summary_renderable(system, agent_ids, show, t_account=t_account)
        for renderable in renderables:
            console.print(renderable)
    
    # Capture initial balance state for HTML export
    initial_balances: Dict[str, Any] = {}
    initial_rows: Dict[str, Dict[str, List[Any]]] = {}
    from bilancio.analysis.balances import agent_balance
    from bilancio.analysis.visualization import build_t_account_rows
    # Capture balances for all agents that we might display
    capture_ids = agent_ids if agent_ids else [a.id for a in system.state.agents.values()]
    for agent_id in capture_ids:
        initial_balances[agent_id] = agent_balance(system, agent_id)
        # also capture detailed rows with counterparties at setup
        acct = build_t_account_rows(system, agent_id)
        def _row_dict(r: Any) -> Dict[str, Any]:
            return {
                'name': getattr(r, 'name', ''),
                'quantity': getattr(r, 'quantity', None),
                'value_minor': getattr(r, 'value_minor', None),
                'counterparty_name': getattr(r, 'counterparty_name', None),
                'maturity': getattr(r, 'maturity', None),
                'id_or_alias': getattr(r, 'id_or_alias', None),
            }
        initial_rows[agent_id] = {
            'assets': [_row_dict(r) for r in acct.assets],
            'liabs': [_row_dict(r) for r in acct.liabilities],
        }

    # Capture initial network snapshot (Day 0)
    from bilancio.analysis.network import build_network_data
    initial_network_snapshot = build_network_data(system, day=0)

    # Track day data for PDF export
    days_data = []

    # Check if dealer subsystem is enabled
    enable_dealer = hasattr(system.state, 'dealer_subsystem') and system.state.dealer_subsystem is not None
    enable_lender = hasattr(system.state, 'lender_config') and system.state.lender_config is not None
    enable_rating = hasattr(system.state, 'rating_config') and system.state.rating_config is not None

    if mode == "step":
        days_data = run_step_mode(
            system=system,
            max_days=max_days,
            show=show,
            agent_ids=agent_ids,
            check_invariants=check_invariants,
            scenario_name=config.name,
            t_account=t_account,
            enable_dealer=enable_dealer,
            enable_lender=enable_lender,
            enable_rating=enable_rating,
        )
    else:
        days_data = run_until_stable_mode(
            system=system,
            max_days=max_days,
            quiet_days=quiet_days,
            show=show,
            agent_ids=agent_ids,
            check_invariants=check_invariants,
            scenario_name=config.name,
            t_account=t_account,
            enable_dealer=enable_dealer,
            enable_lender=enable_lender,
            enable_rating=enable_rating,
            progress_callback=progress_callback,
        )
    
    # Export results if requested
    if export.get('balances_csv'):
        export_path = Path(export['balances_csv'])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_balances_csv(system, export_path)
        console.print(f"[green]OK[/green] Exported balances to {export_path}")
    
    if export.get('events_jsonl'):
        export_path = Path(export['events_jsonl'])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_events_jsonl(system, export_path)
        console.print(f"[green]OK[/green] Exported events to {export_path}")

    # Export dealer metrics if dealer subsystem exists (active or passive)
    if hasattr(system.state, 'dealer_subsystem') and system.state.dealer_subsystem is not None:
        subsystem = system.state.dealer_subsystem
        dealer_metrics_path = None
        if export.get('events_jsonl'):
            dealer_metrics_path = Path(export['events_jsonl']).parent / "dealer_metrics.json"
        elif export.get('balances_csv'):
            dealer_metrics_path = Path(export['balances_csv']).parent / "dealer_metrics.json"

        if dealer_metrics_path:
            import json
            if subsystem.enabled:
                # Active mode: use full metrics summary
                dealer_summary = subsystem.metrics.summary()
            else:
                # Passive mode: compute hold-only PnL for comparison
                from bilancio.engines.dealer_integration import compute_passive_pnl
                dealer_summary = compute_passive_pnl(subsystem, system)
            with dealer_metrics_path.open('w') as f:
                json.dump(dealer_summary, f, indent=2)
            console.print(f"[green]OK[/green] Exported dealer metrics to {dealer_metrics_path}")

            # Plan 022: Export detailed dealer CSV logs if enabled (active mode only)
            if detailed_dealer_logging and subsystem.enabled:
                metrics = system.state.dealer_subsystem.metrics
                # Set run context on metrics and propagate to all records
                metrics.run_id = run_id
                metrics.regime = regime

                # Propagate run_id/regime to all trade records
                for trade in metrics.trades:
                    trade.run_id = run_id
                    trade.regime = regime

                # Propagate to dealer snapshots
                for snap in metrics.dealer_snapshots:
                    snap.run_id = run_id
                    snap.regime = regime

                # Propagate to system state snapshots
                for snap in metrics.system_state_snapshots:
                    snap.run_id = run_id
                    snap.regime = regime

                out_dir = dealer_metrics_path.parent

                # trades.csv
                trades_path = out_dir / "trades.csv"
                metrics.to_trade_log_csv(str(trades_path))
                console.print(f"[green]OK[/green] Exported trades to {trades_path}")

                # inventory_timeseries.csv (uses dealer_snapshots with new fields)
                inventory_path = out_dir / "inventory_timeseries.csv"
                metrics.to_inventory_timeseries_csv(str(inventory_path))
                console.print(f"[green]OK[/green] Exported inventory timeseries to {inventory_path}")

                # system_state_timeseries.csv
                system_state_path = out_dir / "system_state_timeseries.csv"
                metrics.to_system_state_csv(str(system_state_path))
                console.print(f"[green]OK[/green] Exported system state timeseries to {system_state_path}")

                # repayment_events.csv (Plan 022 - Phase 2)
                # Build repayment events from the event log and trades
                from bilancio.dealer.metrics import build_repayment_events
                repayment_events = build_repayment_events(
                    event_log=system.state.events,
                    trades=metrics.trades,
                    run_id=run_id,
                    regime=regime,
                )
                metrics.repayment_events = repayment_events
                repayment_events_path = out_dir / "repayment_events.csv"
                metrics.to_repayment_events_csv(str(repayment_events_path))
                console.print(f"[green]OK[/green] Exported repayment events to {repayment_events_path}")

    # Export to HTML if requested (semantic HTML for readability)
    if html_output:
        from .html_export import export_pretty_html
        export_pretty_html(
            system=system,
            out_path=html_output,
            scenario_name=config.name,
            description=config.description,
            agent_ids=agent_ids,
            initial_balances=initial_balances,
            days_data=days_data,
            initial_rows=initial_rows,
            max_days=max_days,
            quiet_days=quiet_days,
            initial_network_snapshot=initial_network_snapshot,
        )
        console.print(f"[green]OK[/green] Exported HTML report: {html_output}")


def run_step_mode(
    system: System,
    max_days: int,
    show: str,
    agent_ids: Optional[List[str]],
    check_invariants: str,
    scenario_name: str,
    t_account: bool = False,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
) -> List[Dict[str, Any]]:
    """Run simulation in step-by-step mode.

    Args:
        system: Configured system
        max_days: Maximum days to simulate
        show: Event display mode
        agent_ids: Agent IDs to show balances for
        check_invariants: Invariant checking mode
        scenario_name: Name of the scenario for error context
        enable_dealer: If True, run dealer trading phase each day
        enable_lender: If True, run lender phase each day
        enable_rating: If True, run rating agency phase each day

    Returns:
        List of day data dictionaries
    """
    days_data = []

    for _ in range(max_days):
        # Get the current day before running
        day_before = system.state.day

        # Prompt to continue (ask about the next day which is day_before + 1)
        console.print()
        if not Confirm.ask(f"[cyan]Run day {day_before + 1}?[/cyan]", default=True):
            console.print("[yellow]Simulation stopped by user[/yellow]")
            break

        try:
            # Run the next day
            run_day(system, enable_dealer=enable_dealer, enable_lender=enable_lender, enable_rating=enable_rating)
            from bilancio.engines.simulation import DayReport, _impacted_today, _has_open_obligations
            impacted = _impacted_today(system, day_before)
            day_report = DayReport(day=day_before, impacted=impacted)

            # Check invariants if requested
            if check_invariants == "daily":
                system.assert_invariants()
            
            # Skip day 0 display - it's already shown as "Day 0 (After Setup)"
            # But still capture Day 0 simulation events for HTML export
            if day_before == 0:
                # Only capture Day 0 simulation events for HTML
                day0_events = [e for e in system.state.events
                              if e.get("day") == 0 and e.get("phase") == "simulation"]
                if day0_events:
                    days_data.append({
                        'day': 0,
                        'events': day0_events,
                        'quiet': False,
                        'stable': False,
                        'balances': {},
                        'rows': {},
                        'agent_ids': [],
                    })
            if day_before >= 1:
                # Show day summary
                console.print(f"\n[bold cyan] Day {day_before}[/bold cyan]")
                display_agent_ids = _filter_active_agent_ids(system, agent_ids) if agent_ids is not None else None
                renderables = show_day_summary_renderable(system, display_agent_ids, show, day=day_before, t_account=t_account)
                for renderable in renderables:
                    console.print(renderable)
                
                # Collect day data for HTML export  
                # Use the actual event day
                day_events = [e for e in system.state.events 
                             if e.get("day") == day_before and e.get("phase") == "simulation"]
                
                # Capture current balance state for this day
                day_balances: Dict[str, Any] = {}
                day_rows: Dict[str, Dict[str, List[Any]]] = {}
                active_agents_for_day: Optional[List[str]] = None
                if agent_ids is not None:
                    active_agents_for_day = display_agent_ids or []
                if active_agents_for_day:
                    from bilancio.analysis.balances import agent_balance
                    from bilancio.analysis.visualization import build_t_account_rows

                    def _row_dict(r: Any) -> Dict[str, Any]:
                        return {
                            'name': getattr(r, 'name', ''),
                            'quantity': getattr(r, 'quantity', None),
                            'value_minor': getattr(r, 'value_minor', None),
                            'counterparty_name': getattr(r, 'counterparty_name', None),
                            'maturity': getattr(r, 'maturity', None),
                            'id_or_alias': getattr(r, 'id_or_alias', None),
                        }

                    for agent_id in active_agents_for_day:
                        day_balances[agent_id] = agent_balance(system, agent_id)
                        acct = build_t_account_rows(system, agent_id)
                        day_rows[agent_id] = {
                            'assets': [_row_dict(r) for r in acct.assets],
                            'liabs': [_row_dict(r) for r in acct.liabilities],
                        }
                
                days_data.append({
                    'day': day_before,  # Use actual event day, not 1-based counter
                    'events': day_events,
                    'quiet': day_report.impacted == 0,
                    'stable': day_report.impacted == 0 and not _has_open_obligations(system),
                    'balances': day_balances,
                    'rows': day_rows,
                    'agent_ids': active_agents_for_day if active_agents_for_day is not None else [],
                })
            
            # Check if we've reached a stable state
            if day_report.impacted == 0 and not _has_open_obligations(system):
                console.print("[green]OK[/green] System reached stable state")
                break
                
        except DefaultError as e:
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={
                    "scenario": scenario_name,
                    "day": system.state.day,
                    "phase": system.state.phase
                }
            )
            break
            
        except ValidationError as e:
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={
                    "scenario": scenario_name,
                    "day": system.state.day,
                    "phase": system.state.phase
                }
            )
            break
            
        except Exception as e:  # Intentionally broad: top-level simulation loop
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={
                    "scenario": scenario_name,
                    "day": system.state.day
                }
            )
            break
    
    # Show final summary
    console.print("\n[bold]Simulation Complete[/bold]")
    console.print(show_simulation_summary_renderable(system))
    
    return days_data


def run_until_stable_mode(
    system: System,
    max_days: int,
    quiet_days: int,
    show: str,
    agent_ids: Optional[List[str]],
    check_invariants: str,
    scenario_name: str,
    t_account: bool = False,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Run simulation until stable state is reached.

    Args:
        system: Configured system
        max_days: Maximum days to simulate
        quiet_days: Required quiet days for stable state
        show: Event display mode
        agent_ids: Agent IDs to show balances for
        check_invariants: Invariant checking mode
        scenario_name: Name of the scenario for error context
        enable_dealer: If True, run dealer trading phase each day
        enable_lender: If True, run lender phase each day
        enable_rating: If True, run rating agency phase each day
        progress_callback: Optional callback(current_day, max_days) for progress tracking

    Returns:
        List of day data dictionaries
    """
    # Plan 030: Skip verbose output in quiet mode (show="none")
    quiet_mode = show == "none"
    if not quiet_mode:
        console.print(f"\n[dim]Running simulation until stable (max {max_days} days)...[/dim]\n")

    try:
        # Run simulation day by day to capture correct balance snapshots
        from bilancio.engines.simulation import run_day, _impacted_today, _has_open_obligations, _defaults_today
        from bilancio.analysis.balances import agent_balance

        reports = []
        consecutive_quiet = 0
        consecutive_no_defaults = 0
        days_data = []

        for _ in range(max_days):
            # Run the next day
            day_before = system.state.day
            run_day(system, enable_dealer=enable_dealer, enable_lender=enable_lender, enable_rating=enable_rating)
            impacted = _impacted_today(system, day_before)
            defaults = _defaults_today(system, day_before)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(day_before + 1, max_days)
            
            # Create report
            from bilancio.engines.simulation import DayReport
            report = DayReport(day=day_before, impacted=impacted)
            reports.append(report)
            
            # Skip day 0 display - it's already shown as "Day 0 (After Setup)"
            # But still capture Day 0 simulation events for HTML export
            if day_before == 0:
                # Only capture Day 0 simulation events for HTML
                day0_events = [e for e in system.state.events
                              if e.get("day") == 0 and e.get("phase") == "simulation"]
                if day0_events:
                    days_data.append({
                        'day': 0,
                        'events': day0_events,
                        'quiet': False,
                        'stable': False,
                        'balances': {},
                        'rows': {},
                        'agent_ids': [],
                    })
            if day_before >= 1:
                # Display this day's results immediately (with correct balance state)
                # Plan 030: Skip day headers in quiet mode
                if not quiet_mode:
                    console.print(f"[bold cyan] Day {day_before}[/bold cyan]")

                # Check invariants if requested
                if check_invariants == "daily":
                    try:
                        system.assert_invariants()
                    except Exception as e:  # Intentionally broad: invariant checks
                        if not quiet_mode:
                            console.print(f"[yellow][!] Invariant check failed: {e}[/yellow]")

                # Show events and balances for this specific day
                # Note: events are stored with 0-based day numbers
                # Plan 030: show_day_summary_renderable returns [] for show="none"
                # Always compute display_agent_ids (needed for HTML export even in quiet mode)
                display_agent_ids = _filter_active_agent_ids(system, agent_ids) if agent_ids is not None else None
                if not quiet_mode:
                    renderables = show_day_summary_renderable(system, display_agent_ids, show, day=day_before, t_account=t_account)
                    for renderable in renderables:
                        console.print(renderable)
                
                # Collect day data for HTML export
                # We want simulation events from the day that was just displayed
                # show_day_summary was called with day=day_before
                day_events = [e for e in system.state.events
                             if e.get("day") == day_before and e.get("phase") == "simulation"]
                # Plan 024: stability check accounts for rollover mode
                if system.state.rollover_enabled:
                    is_stable = consecutive_no_defaults >= quiet_days
                else:
                    is_stable = consecutive_quiet >= quiet_days and not _has_open_obligations(system)
                
                # Capture current balance state for this day
                day_balances: Dict[str, Any] = {}
                day_rows: Dict[str, Dict[str, List[Any]]] = {}
                active_agents_for_day: Optional[List[str]] = None
                if agent_ids is not None:
                    active_agents_for_day = display_agent_ids or []
                if active_agents_for_day:
                    from bilancio.analysis.visualization import build_t_account_rows
                    for agent_id in active_agents_for_day:
                        day_balances[agent_id] = agent_balance(system, agent_id)
                        acct = build_t_account_rows(system, agent_id)
                        def to_row(r: Any) -> Dict[str, Any]:
                            return {
                                'name': getattr(r, 'name', ''),
                                'quantity': getattr(r, 'quantity', None),
                                'value_minor': getattr(r, 'value_minor', None),
                                'counterparty_name': getattr(r, 'counterparty_name', None),
                                'maturity': getattr(r, 'maturity', None),
                                'id_or_alias': getattr(r, 'id_or_alias', None),
                            }
                        day_rows[agent_id] = {
                            'assets': [to_row(r) for r in acct.assets],
                            'liabs': [to_row(r) for r in acct.liabilities],
                        }

                # Capture network snapshot for this day
                from bilancio.analysis.network import build_network_data
                network_snapshot = build_network_data(system, day_before)

                days_data.append({
                    'day': day_before,  # Use actual event day, not 1-based counter
                    'events': day_events,
                    'quiet': report.impacted == 0,
                    'stable': is_stable,
                    'balances': day_balances,
                    'rows': day_rows,
                    'agent_ids': active_agents_for_day if active_agents_for_day is not None else [],
                    'network_snapshot': network_snapshot,
                })
                
                # Show activity summary (Plan 030: skip in quiet mode)
                if not quiet_mode:
                    if report.impacted > 0:
                        console.print(f"[dim]Activity: {report.impacted} impactful events[/dim]")
                    else:
                        console.print("[dim]-> Quiet day (no activity)[/dim]")

                    if report.notes:
                        console.print(f"[dim]Note: {report.notes}[/dim]")

                    console.print()
            
            # Check for stable state
            if impacted == 0:
                consecutive_quiet += 1
            else:
                consecutive_quiet = 0

            if defaults == 0:
                consecutive_no_defaults += 1
            else:
                consecutive_no_defaults = 0

            # Rollover mode: stability = no defaults for quiet_days consecutive days
            # (settlements are expected and fine in rollover scenarios)
            # Non-rollover: stability = no impact events + no open obligations
            if system.state.rollover_enabled:
                stability_condition = consecutive_no_defaults >= quiet_days
            else:
                stability_condition = consecutive_quiet >= quiet_days and not _has_open_obligations(system)

            if stability_condition:
                console.print("[green]OK[/green] System reached stable state")
                break
        
        # If we didn't break early, check if we hit max days
        else:
            console.print("[yellow][!][/yellow] Maximum days reached without stable state")
        
    except DefaultError as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={
                "scenario": scenario_name,
                "day": system.state.day,
                "phase": system.state.phase
            }
        )
        
    except ValidationError as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={
                "scenario": scenario_name,
                "day": system.state.day,
                "phase": system.state.phase
            }
        )
        
    except Exception as e:  # Intentionally broad: top-level simulation handler
        show_error_panel(
            error=e,
            phase="simulation",
            context={
                "scenario": scenario_name,
                "day": system.state.day
            }
        )
    
    # Show final summary
    console.print("\n[bold]Simulation Complete[/bold]")
    console.print(show_simulation_summary_renderable(system))
    
    return days_data
