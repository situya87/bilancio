"""CLI orchestration for v2-kernel scenario runs.

Drop-in replacement for the clean-core runner: the v2 kernel reproduces the
clean-core engine event-for-event (see ``tests/v2``), so the CLI surface —
including the ``--engine clean-core`` flag value and all output — is
unchanged. Auto mode still falls back to the legacy v1 engine for scenario
features outside the rebuilt domain.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from bilancio.config import load_yaml
from bilancio.core.errors import (
    ConfigurationError,
    DefaultError,
    SimulationHalt,
    ValidationError,
)
from bilancio_v2.compat import (
    build_clean_core_banking_config,
    clean_core_auto_fallback_reason,
)

from .display import show_error_panel
from .result_display import display_clean_core_result


def run_v2_scenario(
    *,
    console: Console,
    invariant_checker: Callable[[list[dict[str, Any]]], None],
    confirm_ask: Callable[..., bool],
    path: Path,
    mode: str,
    max_days: int,
    quiet_days: int,
    show: str,
    check_invariants: str,
    export: dict[str, str] | None,
    html_output: Path | None,
    agent_ids: list[str] | None,
    default_handling: str | None,
    t_account: bool = False,
    detailed_dealer_logging: bool = False,
    run_id: str = "",
    regime: str = "",
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Run a scenario through the v2 kernel."""
    from bilancio.engines.termination import StopReason
    from bilancio_v2 import (
        RunResult,
        prepare_scenario,
        run_day,
        run_until_stable,
    )
    from bilancio_v2.engine import StabilityTracker, update_stability
    from bilancio_v2.exports import (
        write_balances_csv,
        write_events_jsonl,
        write_html_report,
    )
    from bilancio_v2.plugins.banking import finalize_banking
    from bilancio_v2.plugins.certificates import finalize_certificates
    from bilancio_v2.plugins.dealer import dealer_metrics_summary
    from bilancio_v2.scenario_gates import (
        clean_core_configuration_error_reason,
        clean_core_unsupported_reason,
    )
    from bilancio_v2.views import balance_rows

    console.print("[dim]Loading scenario...[/dim]")
    config = load_yaml(path)

    effective_default_handling = default_handling or config.run.default_handling
    if default_handling and config.run.default_handling != default_handling:
        config = config.model_copy(update={"run": config.run.model_copy(update={"default_handling": effective_default_handling})})

    configuration_error = clean_core_configuration_error_reason(config)
    if configuration_error is not None:
        show_error_panel(
            error=ConfigurationError(configuration_error),
            phase="setup",
            context={"scenario": config.name, "engine": "clean-core"},
        )
        sys.exit(1)

    unsupported_reason = clean_core_unsupported_reason(config)
    if unsupported_reason is not None:
        show_error_panel(
            error=NotImplementedError(unsupported_reason),
            phase="setup",
            context={"scenario": config.name, "engine": "clean-core"},
        )
        sys.exit(1)
    try:
        from bilancio.config.apply import validate_scheduled_aliases

        validate_scheduled_aliases(config)
    except ValueError as e:
        show_error_panel(
            error=e,
            phase="setup",
            context={"scenario": config.name, "engine": "clean-core"},
        )
        sys.exit(1)

    if export is None:
        export = {}
    if not export.get("balances_csv") and config.run.export.balances_csv:
        export["balances_csv"] = config.run.export.balances_csv
    if not export.get("events_jsonl") and config.run.export.events_jsonl:
        export["events_jsonl"] = config.run.export.events_jsonl

    if agent_ids is None and config.run.show.balances:
        agent_ids = config.run.show.balances

    console.print(f"[bold]Scenario:[/bold] {config.name}")
    console.print(f"[dim]Default handling mode: {effective_default_handling}[/dim]")
    console.print("[dim]Engine: clean-core[/dim]")
    banking_config = build_clean_core_banking_config(path)

    stopped_by_user = False
    stopped_by_error = False
    runtime: Any | None = None

    def _check_daily_invariants(runtime: Any, day: int) -> None:
        if check_invariants == "daily":
            invariant_checker(
                balance_rows(
                    RunResult(
                        ledger=runtime.ledger,
                        final_day=day + 1,
                        reached_stable=False,
                        stop_reason=StopReason.MAX_DAYS_REACHED,
                    )
                )
            )

    try:
        if mode == "step":
            runtime = prepare_scenario(config, banking_config=banking_config)
            stability = StabilityTracker()
            reached_stable = False
            final_day = 0
            for day in range(max_days):
                console.print()
                if not confirm_ask(f"[cyan]Run day {day + 1}?[/cyan]", default=True):
                    console.print("[yellow]Simulation stopped by user[/yellow]")
                    stopped_by_user = True
                    break
                impactful = run_day(runtime, day)
                if progress_callback:
                    progress_callback(day + 1, max_days)
                final_day = day + 1
                _check_daily_invariants(runtime, day)
                console.print(f"[dim]Day {day}: {'activity' if impactful else 'quiet'}[/dim]")
                if update_stability(
                    runtime,
                    day=day,
                    impactful=impactful,
                    quiet_days=quiet_days,
                    tracker=stability,
                ):
                    reached_stable = True
                    console.print("[green]OK[/green] System reached stable state")
                    break
            result = RunResult(
                ledger=runtime.ledger,
                final_day=final_day,
                reached_stable=reached_stable,
                stop_reason=(
                    StopReason.USER_STOP
                    if stopped_by_user
                    else (StopReason.STABILITY_REACHED if reached_stable else StopReason.MAX_DAYS_REACHED)
                ),
                stability_snapshots=tuple(stability.snapshots),
            )
        else:
            runtime = prepare_scenario(config, banking_config=banking_config)
            result = run_until_stable(
                runtime,
                max_days=max_days,
                quiet_days=quiet_days,
                progress_callback=progress_callback,
                day_callback=_check_daily_invariants,
            )
            finalize_banking(
                result.ledger,
                final_day=result.final_day,
                reached_stable=result.reached_stable,
                banking_config=banking_config,
            )
            finalize_certificates(result.ledger, final_day=result.final_day)
    except (ConfigurationError, DefaultError, SimulationHalt) as e:
        show_error_panel(
            error=e,
            phase="setup" if runtime is None else "simulation",
            context={"scenario": config.name, "engine": "clean-core"},
        )
        if runtime is None:
            sys.exit(1)
        stopped_by_error = True
        result = RunResult(
            ledger=runtime.ledger,
            final_day=runtime.ledger.day,
            reached_stable=False,
            stop_reason=StopReason.FATAL_ERROR,
        )

    try:
        rows = balance_rows(result)
        if check_invariants != "none":
            invariant_checker(rows)
    except ValidationError as e:
        show_error_panel(
            error=e,
            phase="simulation",
            context={"scenario": config.name, "engine": "clean-core"},
        )
        sys.exit(1)

    display_clean_core_result(
        console,
        result,
        rows,
        show=show,
        agent_ids=agent_ids,
        t_account=t_account,
    )

    if export.get("balances_csv"):
        export_path = Path(export["balances_csv"])
        write_balances_csv(result, export_path)
        console.print(f"[green]OK[/green] Exported balances to {export_path}")

    if export.get("events_jsonl"):
        export_path = Path(export["events_jsonl"])
        write_events_jsonl(result, export_path)
        console.print(f"[green]OK[/green] Exported events to {export_path}")

    if result.ledger.dealer_metrics is not None:
        dealer_metrics_path = None
        if export.get("events_jsonl"):
            dealer_metrics_path = Path(export["events_jsonl"]).parent / "dealer_metrics.json"
        elif export.get("balances_csv"):
            dealer_metrics_path = Path(export["balances_csv"]).parent / "dealer_metrics.json"

        if dealer_metrics_path:
            import json

            metrics = result.ledger.dealer_metrics
            dealer_summary = dealer_metrics_summary(result.ledger)
            if dealer_summary is None:
                dealer_summary = metrics.summary()
            dealer_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with dealer_metrics_path.open("w") as f:
                json.dump(dealer_summary, f, indent=2)
            console.print(f"[green]OK[/green] Exported dealer metrics to {dealer_metrics_path}")

            if detailed_dealer_logging and not result.ledger.dealer_config.balanced_passive:
                metrics.run_id = run_id
                metrics.regime = regime
                for trade in metrics.trades:
                    trade.run_id = run_id
                    trade.regime = regime
                for snap in metrics.dealer_snapshots:
                    snap.run_id = run_id
                    snap.regime = regime
                for snap in metrics.system_state_snapshots:
                    snap.run_id = run_id
                    snap.regime = regime

                out_dir = dealer_metrics_path.parent
                trades_path = out_dir / "trades.csv"
                metrics.to_trade_log_csv(str(trades_path))
                console.print(f"[green]OK[/green] Exported trades to {trades_path}")

                inventory_path = out_dir / "inventory_timeseries.csv"
                metrics.to_inventory_timeseries_csv(str(inventory_path))
                console.print(f"[green]OK[/green] Exported inventory timeseries to {inventory_path}")

                system_state_path = out_dir / "system_state_timeseries.csv"
                metrics.to_system_state_csv(str(system_state_path))
                console.print(f"[green]OK[/green] Exported system state timeseries to {system_state_path}")

                from bilancio.dealer.metrics import build_repayment_events

                metrics.repayment_events = build_repayment_events(
                    event_log=result.events,
                    trades=metrics.trades,
                    run_id=run_id,
                    regime=regime,
                )
                repayment_events_path = out_dir / "repayment_events.csv"
                metrics.to_repayment_events_csv(str(repayment_events_path))
                console.print(f"[green]OK[/green] Exported repayment events to {repayment_events_path}")

    if html_output:
        write_html_report(
            result,
            config,
            html_output,
            max_days=max_days,
            quiet_days=quiet_days,
            agent_ids=agent_ids,
            t_account=t_account,
        )
        console.print(f"[green]OK[/green] Exported HTML report: {html_output}")

    if result.reached_stable:
        console.print(f"[green]OK[/green] System reached stable state after {result.final_day} days")
    elif mode == "step" and stopped_by_user:
        console.print("[yellow]WARNING[/yellow] Simulation stopped before stability")
    elif stopped_by_error:
        console.print("[yellow]WARNING[/yellow] Simulation stopped after an error")
    else:
        console.print(f"[yellow]WARNING[/yellow] Maximum days reached ({max_days}) before stability")


def clearinghouse_enabled(config: Any) -> bool:
    """Return True when the scenario enables the clearinghouse (v2-only feature)."""
    clearinghouse = getattr(config, "clearinghouse", None)
    return clearinghouse is not None and bool(getattr(clearinghouse, "enabled", False))


def resolve_auto_engine(
    path: Path,
    default_handling: str | None,
    *,
    console: Console,
) -> str:
    """Choose the v2 kernel when the scenario is in the rebuilt domain."""
    reason = clean_core_auto_fallback_reason(path, default_handling)
    if reason is None:
        return "clean-core"

    if clearinghouse_enabled(load_yaml(path)):
        raise ConfigurationError(
            "clearinghouse requires the v2 engine "
            f"(scenario falls back to legacy: {reason})"
        )

    console.print(f"[dim]Engine: legacy (auto fallback: {reason})[/dim]")
    return "legacy"
