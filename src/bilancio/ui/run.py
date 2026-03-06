"""Orchestration logic for running Bilancio simulations."""

from __future__ import annotations

import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bilancio.core.performance import PerformanceConfig

from rich.console import Console
from rich.prompt import Confirm

from bilancio.config import apply_to_system, load_yaml
from bilancio.core.errors import DefaultError, SimulationHalt, ValidationError
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.export.writers import write_balances_csv, write_events_jsonl

from .display import (
    show_day_summary_renderable,
    show_error_panel,
    show_scenario_header_renderable,
    show_simulation_summary_renderable,
)

console = Console(record=True, width=120)
SIMULATION_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    AssertionError,
)


def _filter_active_agent_ids(system: System, agent_ids: list[str] | None) -> list[str] | None:
    """Return only agent IDs that remain active (not defaulted)."""
    if agent_ids is None:
        return None
    active_ids: list[str] = []
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
    agent_ids: list[str] | None = None,
    check_invariants: str = "setup",
    export: dict[str, str] | None = None,
    html_output: Path | None = None,
    t_account: bool = False,
    default_handling: str | None = None,
    detailed_dealer_logging: bool = False,
    run_id: str = "",
    regime: str = "",
    progress_callback: Callable[[int, int], None] | None = None,
    performance: PerformanceConfig | None = None,
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
        performance: Optional performance tuning configuration
    """
    # Load configuration
    console.print("[dim]Loading scenario...[/dim]")
    config = load_yaml(path)

    # Determine effective default-handling strategy (CLI override wins)
    effective_default_handling = default_handling or config.run.default_handling
    if default_handling and config.run.default_handling != default_handling:
        config = config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={"default_handling": effective_default_handling}
                )
            }
        )

    # Create and configure system with selected default-handling mode
    system = System(default_mode=effective_default_handling)
    # Preflight schedule validation (aliases available when referenced)
    try:
        from bilancio.config.apply import validate_scheduled_aliases

        validate_scheduled_aliases(config)
    except ValueError as e:
        show_error_panel(error=e, phase="setup", context={"scenario": config.name})
        sys.exit(1)

    # Apply configuration
    try:
        apply_to_system(config, system)

        if check_invariants in ("setup", "daily"):
            system.assert_invariants()

    except (ValidationError, ValueError) as e:
        show_error_panel(error=e, phase="setup", context={"scenario": config.name})
        sys.exit(1)

    # Plan 024: Enable rollover if configured
    system.state.rollover_enabled = config.run.rollover_enabled

    # Plan 034: Enable estimate logging if configured
    system.state.estimate_logging_enabled = config.run.estimate_logging

    # Stage scheduled actions into system state (Phase B1 execution by day)
    try:
        if getattr(config, "scheduled_actions", None):
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
    if not export.get("balances_csv") and config.run.export.balances_csv:
        export["balances_csv"] = config.run.export.balances_csv
    if not export.get("events_jsonl") and config.run.export.events_jsonl:
        export["events_jsonl"] = config.run.export.events_jsonl

    # Plan 030: Check for quiet mode (show="none") to suppress verbose output
    quiet_mode = show == "none"

    # Show scenario header with agent list (skip in quiet mode)
    if not quiet_mode:
        header_renderables = show_scenario_header_renderable(
            config.name, config.description, config.agents
        )
        for renderable in header_renderables:
            console.print(renderable)
        console.print(f"[dim]Default handling mode: {effective_default_handling}[/dim]")

        # Show initial state
        console.print("\n[bold cyan] Day 0 (After Setup)[/bold cyan]")
        renderables = show_day_summary_renderable(system, agent_ids, show, t_account=t_account)
        for renderable in renderables:
            console.print(renderable)

    # Capture initial balance state for HTML export
    initial_balances: dict[str, Any] = {}
    initial_rows: dict[str, dict[str, list[Any]]] = {}
    from bilancio.analysis.balances import agent_balance
    from bilancio.analysis.visualization import build_t_account_rows

    # Capture balances for all agents that we might display
    capture_ids = agent_ids if agent_ids else [a.id for a in system.state.agents.values()]
    for agent_id in capture_ids:
        initial_balances[agent_id] = agent_balance(system, agent_id)
        # also capture detailed rows with counterparties at setup
        acct = build_t_account_rows(system, agent_id)

        def _row_dict(r: Any) -> dict[str, Any]:
            return {
                "name": getattr(r, "name", ""),
                "quantity": getattr(r, "quantity", None),
                "value_minor": getattr(r, "value_minor", None),
                "counterparty_name": getattr(r, "counterparty_name", None),
                "maturity": getattr(r, "maturity", None),
                "id_or_alias": getattr(r, "id_or_alias", None),
            }

        initial_rows[agent_id] = {
            "assets": [_row_dict(r) for r in acct.assets],
            "liabs": [_row_dict(r) for r in acct.liabilities],
        }

    # Capture initial network snapshot (Day 0)
    from bilancio.analysis.network import build_network_data

    initial_network_snapshot = build_network_data(system, day=0)

    # Track day data for PDF export
    days_data = []

    # Check if dealer subsystem is enabled
    enable_dealer = (
        hasattr(system.state, "dealer_subsystem") and system.state.dealer_subsystem is not None
    )
    enable_lender = (
        hasattr(system.state, "lender_config") and system.state.lender_config is not None
    )
    enable_rating = (
        hasattr(system.state, "rating_config") and system.state.rating_config is not None
    )
    enable_banking = (
        hasattr(system.state, "banking_subsystem") and system.state.banking_subsystem is not None
    )
    enable_bank_lending = False
    _cb_lending_cutoff_day = None

    # Initialize banking subsystem if banks exist but subsystem not yet initialized
    if not enable_banking:
        import yaml as _yaml

        with open(path) as _f:
            raw_scenario = _yaml.safe_load(_f)
        _balanced_cfg = raw_scenario.get("_balanced_config", {}) if raw_scenario else {}
        _n_banks = _balanced_cfg.get("n_banks", 0)
        _run_cfg = raw_scenario.get("run", {}) if raw_scenario else {}
        _enable_banking_flag = _run_cfg.get("enable_banking", False)

        if (_enable_banking_flag or _n_banks > 0) and _n_banks > 0:
            from bilancio.decision.profiles import BankProfile
            from bilancio.engines.banking_subsystem import initialize_banking_subsystem

            _credit_risk_loading = Decimal(str(_balanced_cfg.get("credit_risk_loading", 0)))
            _max_borrower_risk = Decimal(str(_balanced_cfg.get("max_borrower_risk", "1.0")))
            _min_coverage_ratio = Decimal(str(_balanced_cfg.get("min_coverage_ratio", 0)))
            _adaptive_corridor = _balanced_cfg.get("adaptive_corridor", False)
            bank_profile = BankProfile(
                credit_risk_loading=_credit_risk_loading,
                max_borrower_risk=_max_borrower_risk,
                min_coverage_ratio=_min_coverage_ratio,
                adaptive_corridor=_adaptive_corridor,
            )
            # Get kappa: prefer _balanced_config (written by compiler), then balanced_dealer, then default
            _kappa_str = (
                _balanced_cfg.get("kappa")
                or raw_scenario.get("balanced_dealer", {}).get("kappa")
                or "1"
            )
            _kappa_val = Decimal(str(_kappa_str))
            # Get maturity_days: prefer _balanced_config, then params, then max_days
            _maturity_days = _balanced_cfg.get("maturity_days")
            if _maturity_days is None and "params" in raw_scenario:
                _maturity_days = raw_scenario["params"].get("maturity", {}).get("days")
            if _maturity_days is None:
                _maturity_days = raw_scenario.get("run", {}).get("max_days", 10)
            _trader_banks = _balanced_cfg.get("trader_bank_assignments", {})
            _infra_banks = _balanced_cfg.get("infra_bank_assignments", {})
            # Read mu and concentration for adaptive corridor (Plan 050)
            _mu_raw = _balanced_cfg.get("mu")
            _mu_val = Decimal(str(_mu_raw)) if _mu_raw is not None else None
            _c_raw = _balanced_cfg.get("concentration")
            _c_val = Decimal(str(_c_raw)) if _c_raw is not None else None

            banking_sub = initialize_banking_subsystem(
                system,
                bank_profile,
                _kappa_val,
                _maturity_days,
                trader_banks=_trader_banks,
                infra_banks=_infra_banks,
                mu=_mu_val,
                c=_c_val,
            )
            system.state.banking_subsystem = banking_sub
            # Wire risk assessor for credit-risk-adjusted bank lending
            if _credit_risk_loading > 0:
                if (
                    system.state.dealer_subsystem
                    and hasattr(system.state.dealer_subsystem, 'risk_assessor')
                    and system.state.dealer_subsystem.risk_assessor
                ):
                    banking_sub.risk_assessor = system.state.dealer_subsystem.risk_assessor
                else:
                    from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
                    banking_sub.risk_assessor = RiskAssessor(RiskAssessmentParams())
            # Wire CB escalation and cap params
            _cb_escalation_slope = Decimal(str(_balanced_cfg.get("cb_rate_escalation_slope", 0)))
            _cb_max_outstanding = Decimal(str(_balanced_cfg.get("cb_max_outstanding_ratio", 0)))
            # Auto-activate CB escalation when banking is enabled (Plan 042)
            if _cb_escalation_slope == 0 and _cb_max_outstanding == 0 and _n_banks > 0:
                _cb_escalation_slope = Decimal("0.05")
                _cb_max_outstanding = Decimal("2.0")
            if _cb_escalation_slope > 0 or _cb_max_outstanding > 0:
                from bilancio.domain.agents.central_bank import CentralBank as _CB
                for _agent in system.state.agents.values():
                    if isinstance(_agent, _CB):
                        _agent.rate_escalation_slope = _cb_escalation_slope
                        _agent.max_outstanding_ratio = _cb_max_outstanding
                        _Q_total = int(float(_balanced_cfg.get("Q_total", 0)))
                        if _Q_total <= 0:
                            from bilancio.domain.instruments.base import InstrumentKind as _IK
                            _Q_total = sum(
                                c.amount for c in system.state.contracts.values()
                                if c.kind == _IK.PAYABLE
                            )
                        _agent.escalation_base_amount = _Q_total
                        break

            # Wire κ-informed prior onto CentralBank for dynamic corridor (Plan 041)
            # Also sync base corridor rates so compute_corridor() uses the
            # same kappa-calibrated floor/ceiling as initialize_banking_subsystem().
            from bilancio.dealer.priors import kappa_informed_prior as _kip
            from bilancio.domain.agents.central_bank import CentralBank as _CB2
            for _agent in system.state.agents.values():
                if isinstance(_agent, _CB2):
                    _agent.kappa_prior = _kip(_kappa_val)
                    _agent.reserve_remuneration_rate = bank_profile.r_floor(_kappa_val, _mu_val, _c_val)
                    _agent.cb_lending_rate = bank_profile.r_ceiling(_kappa_val, _mu_val, _c_val)
                    # Plan 050: Set CB adaptive flags from _balanced_config
                    _agent.adaptive_betas = _balanced_cfg.get("adaptive_betas", False)
                    _agent.adaptive_early_warning = _balanced_cfg.get("adaptive_early_warning", False)
                    break

            enable_banking = True
            enable_bank_lending = _balanced_cfg.get("enable_bank_lending", False) or _run_cfg.get("enable_bank_lending", False)
            _cb_lending_cutoff_day = _balanced_cfg.get("cb_lending_cutoff_day")
    else:
        # Banking already initialized; check if bank lending is enabled
        import yaml as _yaml2

        with open(path) as _f2:
            _raw2 = _yaml2.safe_load(_f2)
        _bc2 = (_raw2 or {}).get("_balanced_config", {})
        _rc2 = (_raw2 or {}).get("run", {})
        enable_bank_lending = _bc2.get("enable_bank_lending", False) or _rc2.get("enable_bank_lending", False)
        _cb_lending_cutoff_day = _bc2.get("cb_lending_cutoff_day")

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
            enable_banking=enable_banking,
            enable_bank_lending=enable_bank_lending,
            cb_lending_cutoff_day=_cb_lending_cutoff_day,
            performance=performance,
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
            enable_banking=enable_banking,
            enable_bank_lending=enable_bank_lending,
            cb_lending_cutoff_day=_cb_lending_cutoff_day,
            progress_callback=progress_callback,
            performance=performance,
        )

    # Export results if requested
    if export.get("balances_csv"):
        export_path = Path(export["balances_csv"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_balances_csv(system, export_path)
        console.print(f"[green]OK[/green] Exported balances to {export_path}")

    if export.get("events_jsonl"):
        export_path = Path(export["events_jsonl"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        write_events_jsonl(system, export_path)
        console.print(f"[green]OK[/green] Exported events to {export_path}")

    # Export dealer metrics if dealer subsystem exists (active or passive)
    if hasattr(system.state, "dealer_subsystem") and system.state.dealer_subsystem is not None:
        subsystem = system.state.dealer_subsystem
        dealer_metrics_path = None
        if export.get("events_jsonl"):
            dealer_metrics_path = Path(export["events_jsonl"]).parent / "dealer_metrics.json"
        elif export.get("balances_csv"):
            dealer_metrics_path = Path(export["balances_csv"]).parent / "dealer_metrics.json"

        if dealer_metrics_path:
            import json

            if subsystem.enabled:
                # Active mode: use full metrics summary
                dealer_summary = subsystem.metrics.summary()
            else:
                # Passive mode: compute hold-only PnL for comparison
                from bilancio.engines.dealer_integration import compute_passive_pnl

                dealer_summary = compute_passive_pnl(subsystem, system)
            with dealer_metrics_path.open("w") as f:
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
                console.print(
                    f"[green]OK[/green] Exported inventory timeseries to {inventory_path}"
                )

                # system_state_timeseries.csv
                system_state_path = out_dir / "system_state_timeseries.csv"
                metrics.to_system_state_csv(str(system_state_path))
                console.print(
                    f"[green]OK[/green] Exported system state timeseries to {system_state_path}"
                )

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
                console.print(
                    f"[green]OK[/green] Exported repayment events to {repayment_events_path}"
                )

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
    agent_ids: list[str] | None,
    check_invariants: str,
    scenario_name: str,
    t_account: bool = False,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
    enable_bank_lending: bool = False,
    cb_lending_cutoff_day: int | None = None,
    performance: PerformanceConfig | None = None,
) -> list[dict[str, Any]]:
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
        enable_banking: If True, run banking subphases each day
        enable_bank_lending: If True, run bank lending phase each day
        cb_lending_cutoff_day: Day to freeze CB lending (None = no freeze)

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

        # Activate CB lending freeze at cutoff day
        if (
            cb_lending_cutoff_day is not None
            and day_before >= cb_lending_cutoff_day
            and not system.state.cb_lending_frozen
        ):
            system.state.cb_lending_frozen = True
            system.log("CBLendingFreezeActivated", day=day_before, cutoff_day=cb_lending_cutoff_day)

        try:
            # Run the next day
            run_day(
                system,
                enable_dealer=enable_dealer,
                enable_lender=enable_lender,
                enable_rating=enable_rating,
                enable_banking=enable_banking,
                enable_bank_lending=enable_bank_lending,
                performance=performance,
            )
            from bilancio.engines.simulation import (
                DayReport,
                _has_open_obligations,
                _impacted_today,
            )

            impacted = _impacted_today(system, day_before)
            day_report = DayReport(day=day_before, impacted=impacted)

            # Check invariants if requested
            if check_invariants == "daily":
                system.assert_invariants()

            # Skip day 0 display - it's already shown as "Day 0 (After Setup)"
            # But still capture Day 0 simulation events for HTML export
            if day_before == 0:
                # Only capture Day 0 simulation events for HTML
                day0_events = [
                    e
                    for e in system.state.events
                    if e.get("day") == 0 and e.get("phase") == "simulation"
                ]
                if day0_events:
                    days_data.append(
                        {
                            "day": 0,
                            "events": day0_events,
                            "quiet": False,
                            "stable": False,
                            "balances": {},
                            "rows": {},
                            "agent_ids": [],
                        }
                    )
            if day_before >= 1:
                # Show day summary
                console.print(f"\n[bold cyan] Day {day_before}[/bold cyan]")
                display_agent_ids = (
                    _filter_active_agent_ids(system, agent_ids) if agent_ids is not None else None
                )
                renderables = show_day_summary_renderable(
                    system, display_agent_ids, show, day=day_before, t_account=t_account
                )
                for renderable in renderables:
                    console.print(renderable)

                # Collect day data for HTML export
                # Use the actual event day
                day_events = [
                    e
                    for e in system.state.events
                    if e.get("day") == day_before and e.get("phase") == "simulation"
                ]

                # Capture current balance state for this day
                day_balances: dict[str, Any] = {}
                day_rows: dict[str, dict[str, list[Any]]] = {}
                active_agents_for_day: list[str] | None = None
                if agent_ids is not None:
                    active_agents_for_day = display_agent_ids or []
                if active_agents_for_day:
                    from bilancio.analysis.balances import agent_balance
                    from bilancio.analysis.visualization import build_t_account_rows

                    def _row_dict(r: Any) -> dict[str, Any]:
                        return {
                            "name": getattr(r, "name", ""),
                            "quantity": getattr(r, "quantity", None),
                            "value_minor": getattr(r, "value_minor", None),
                            "counterparty_name": getattr(r, "counterparty_name", None),
                            "maturity": getattr(r, "maturity", None),
                            "id_or_alias": getattr(r, "id_or_alias", None),
                        }

                    for agent_id in active_agents_for_day:
                        day_balances[agent_id] = agent_balance(system, agent_id)
                        acct = build_t_account_rows(system, agent_id)
                        day_rows[agent_id] = {
                            "assets": [_row_dict(r) for r in acct.assets],
                            "liabs": [_row_dict(r) for r in acct.liabilities],
                        }

                days_data.append(
                    {
                        "day": day_before,  # Use actual event day, not 1-based counter
                        "events": day_events,
                        "quiet": day_report.impacted == 0,
                        "stable": day_report.impacted == 0 and not _has_open_obligations(system),
                        "balances": day_balances,
                        "rows": day_rows,
                        "agent_ids": active_agents_for_day
                        if active_agents_for_day is not None
                        else [],
                    }
                )

            # Check if we've reached a stable state
            if day_report.impacted == 0 and not _has_open_obligations(system):
                if enable_banking and not system.state.cb_lending_frozen:
                    system.state.cb_lending_frozen = True
                    system.log("CBLendingFreezeStability", day=system.state.day)
                console.print("[green]OK[/green] System reached stable state")
                break

        except (DefaultError, SimulationHalt) as e:
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={
                    "scenario": scenario_name,
                    "day": system.state.day,
                    "phase": system.state.phase,
                },
            )
            break

        except ValidationError as e:
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={
                    "scenario": scenario_name,
                    "day": system.state.day,
                    "phase": system.state.phase,
                },
            )
            break

        except SIMULATION_RECOVERABLE_ERRORS as e:
            show_error_panel(
                error=e,
                phase=f"day_{system.state.day}",
                context={"scenario": scenario_name, "day": system.state.day},
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
    agent_ids: list[str] | None,
    check_invariants: str,
    scenario_name: str,
    t_account: bool = False,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
    enable_bank_lending: bool = False,
    cb_lending_cutoff_day: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    performance: PerformanceConfig | None = None,
) -> list[dict[str, Any]]:
    """Run simulation until stable state is reached.

    Plan 051: delegates loop to ``run_until_stable()`` via a ``day_hook``
    closure that captures balances, displays progress, and collects
    ``days_data`` for HTML export — eliminating the duplicated stop logic.

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
        enable_banking: If True, run banking subphases each day
        enable_bank_lending: If True, run bank lending phase each day
        progress_callback: Optional callback(current_day, max_days) for progress tracking

    Returns:
        List of day data dictionaries
    """
    from bilancio.analysis.balances import agent_balance
    from bilancio.engines.simulation import DayReport, run_until_stable
    from bilancio.engines.termination import StopReason

    # Plan 030: Skip verbose output in quiet mode (show="none")
    quiet_mode = show == "none"
    if not quiet_mode:
        console.print(f"\n[dim]Running simulation until stable (max {max_days} days)...[/dim]\n")

    days_data: list[dict[str, Any]] = []

    # ── day_hook closure ───────────────────────────────────────────────
    # Called by run_until_stable() after each day is executed.
    # All UI display and data capture happens here.
    def _day_hook(sys: System, day_before: int, report: DayReport) -> None:
        # Progress callback
        if progress_callback:
            progress_callback(day_before + 1, max_days)

        # Day 0: capture simulation events for HTML but skip display
        if day_before == 0:
            day0_events = [
                e
                for e in sys.state.events
                if e.get("day") == 0 and e.get("phase") == "simulation"
            ]
            if day0_events:
                days_data.append(
                    {
                        "day": 0,
                        "events": day0_events,
                        "quiet": False,
                        "stable": False,
                        "balances": {},
                        "rows": {},
                        "agent_ids": [],
                    }
                )
            return

        # Day >= 1: display + capture
        if not quiet_mode:
            console.print(f"[bold cyan] Day {day_before}[/bold cyan]")

        if check_invariants == "daily":
            try:
                sys.assert_invariants()
            except SIMULATION_RECOVERABLE_ERRORS as e:
                if not quiet_mode:
                    console.print(f"[yellow][!] Invariant check failed: {e}[/yellow]")

        display_agent_ids = (
            _filter_active_agent_ids(sys, agent_ids) if agent_ids is not None else None
        )
        if not quiet_mode:
            renderables = show_day_summary_renderable(
                sys, display_agent_ids, show, day=day_before, t_account=t_account
            )
            for renderable in renderables:
                console.print(renderable)

        # Collect day events for HTML export
        day_events = [
            e
            for e in sys.state.events
            if e.get("day") == day_before and e.get("phase") == "simulation"
        ]

        # Capture current balance state
        day_balances: dict[str, Any] = {}
        day_rows: dict[str, dict[str, list[Any]]] = {}
        active_agents_for_day: list[str] | None = None
        if agent_ids is not None:
            active_agents_for_day = display_agent_ids or []
        if active_agents_for_day:
            from bilancio.analysis.visualization import build_t_account_rows

            for agent_id in active_agents_for_day:
                day_balances[agent_id] = agent_balance(sys, agent_id)
                acct = build_t_account_rows(sys, agent_id)

                def to_row(r: Any) -> dict[str, Any]:
                    return {
                        "name": getattr(r, "name", ""),
                        "quantity": getattr(r, "quantity", None),
                        "value_minor": getattr(r, "value_minor", None),
                        "counterparty_name": getattr(r, "counterparty_name", None),
                        "maturity": getattr(r, "maturity", None),
                        "id_or_alias": getattr(r, "id_or_alias", None),
                    }

                day_rows[agent_id] = {
                    "assets": [to_row(r) for r in acct.assets],
                    "liabs": [to_row(r) for r in acct.liabilities],
                }

        # Network snapshot
        from bilancio.analysis.network import build_network_data

        network_snapshot = build_network_data(sys, day_before)

        days_data.append(
            {
                "day": day_before,
                "events": day_events,
                "quiet": report.impacted == 0,
                "stable": False,  # Patched after run_until_stable returns
                "balances": day_balances,
                "rows": day_rows,
                "agent_ids": active_agents_for_day
                if active_agents_for_day is not None
                else [],
                "network_snapshot": network_snapshot,
            }
        )

        # Activity summary (Plan 030: skip in quiet mode)
        if not quiet_mode:
            if report.impacted > 0:
                console.print(f"[dim]Activity: {report.impacted} impactful events[/dim]")
            else:
                console.print("[dim]-> Quiet day (no activity)[/dim]")

            if report.notes:
                console.print(f"[dim]Note: {report.notes}[/dim]")

            console.print()

    # ── Run simulation ─────────────────────────────────────────────────
    try:
        result = run_until_stable(
            system,
            max_days=max_days,
            quiet_days=quiet_days,
            enable_dealer=enable_dealer,
            enable_lender=enable_lender,
            enable_rating=enable_rating,
            enable_banking=enable_banking,
            enable_bank_lending=enable_bank_lending,
            enable_final_cb_settlement=enable_banking,
            cb_lending_cutoff_day=cb_lending_cutoff_day,
            performance=performance,
            day_hook=_day_hook,
        )

        # Mark the last day entry as stable if stability was reached
        if result.stop_reason == StopReason.STABILITY_REACHED and days_data:
            days_data[-1]["stable"] = True

        # Display stop message
        if result.stop_reason == StopReason.STABILITY_REACHED:
            console.print("[green]OK[/green] System reached stable state")
        elif result.stop_reason == StopReason.MAX_DAYS_REACHED:
            console.print("[yellow][!][/yellow] Maximum days reached without stable state")

        # Show wind-down info
        if result.winddown_days > 0 and not quiet_mode:
            console.print(
                f"[dim]Bank loan wind-down: {result.winddown_days} extra days to mature remaining loans[/dim]"
            )

        # Show final CB settlement info
        if result.final_cb_settlement and not quiet_mode:
            fc = result.final_cb_settlement
            if fc["loans_attempted"] > 0:
                console.print(
                    f"[dim]Final CB settlement: {fc['loans_repaid']}/{fc['loans_attempted']} repaid, "
                    f"{fc['loans_written_off']} written off, {fc['bank_defaults']} bank defaults[/dim]"
                )

    except (DefaultError, SimulationHalt) as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={
                "scenario": scenario_name,
                "day": system.state.day,
                "phase": system.state.phase,
            },
        )

    except ValidationError as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={
                "scenario": scenario_name,
                "day": system.state.day,
                "phase": system.state.phase,
            },
        )

    except SIMULATION_RECOVERABLE_ERRORS as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={"scenario": scenario_name, "day": system.state.day},
        )

    # Show final summary
    console.print("\n[bold]Simulation Complete[/bold]")
    console.print(show_simulation_summary_renderable(system))

    return days_data
