"""Console display helpers for clean-core scenario runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from rich.console import Console
from rich.table import Table

from bilancio.analysis.visualization import display_events_tables_by_phase_renderables


def display_clean_core_result(
    console: Console,
    result: Any,
    rows: list[dict[str, Any]],
    *,
    show: str,
    agent_ids: list[str] | None,
    t_account: bool = False,
) -> None:
    """Display clean-core CLI output with the same high-level knobs as legacy runs."""
    if show == "none":
        return

    if agent_ids is None:
        _display_clean_core_trial_balance(console, rows)
    elif t_account:
        _display_clean_core_t_accounts(console, result, agent_ids)
    else:
        _display_clean_core_agent_balances(console, rows, agent_ids)

    _display_clean_core_events(console, result.events, show)


def _display_clean_core_trial_balance(
    console: Console,
    rows: list[dict[str, Any]],
) -> None:
    system_row = next((row for row in rows if row.get("agent_id") == "SYSTEM"), None)
    if system_row is None:
        return

    assets = Decimal(str(system_row.get("total_financial_assets", 0)))
    liabilities = Decimal(str(system_row.get("total_financial_liabilities", 0)))
    diff = abs(assets - liabilities)

    table = Table(
        title="System Trial Balance (clean-core)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total Assets", _format_clean_core_amount(assets))
    table.add_row("Total Liabilities", _format_clean_core_amount(liabilities))
    table.add_row("Total Equity", _format_clean_core_amount(assets - liabilities))
    table.add_row(
        "Status",
        (
            "[green]OK Balanced[/green]"
            if diff < Decimal("0.01")
            else f"[red]Imbalanced ({diff})[/red]"
        ),
    )
    console.print()
    console.print(table)


def _display_clean_core_agent_balances(
    console: Console,
    rows: list[dict[str, Any]],
    agent_ids: list[str],
) -> None:
    table = Table(
        title="Final Balances (clean-core)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Agent")
    table.add_column("Assets", justify="right")
    table.add_column("Liabilities", justify="right")
    table.add_column("Net", justify="right")
    table.add_column("Breakdown")

    row_by_agent = {str(row.get("agent_id")): row for row in rows}
    selected_rows = [
        row_by_agent[agent_id] for agent_id in agent_ids if agent_id in row_by_agent
    ]
    if not selected_rows:
        console.print("\n[bold]Balances:[/bold]")
        console.print("  - No active agents to display")
        return

    for row in selected_rows:
        table.add_row(
            str(row["agent_id"]),
            _format_clean_core_amount(row.get("total_financial_assets", 0)),
            _format_clean_core_amount(row.get("total_financial_liabilities", 0)),
            _format_clean_core_amount(row.get("net_financial", 0)),
            _clean_core_balance_breakdown(row),
        )

    console.print()
    console.print(table)


def _display_clean_core_t_accounts(
    console: Console,
    result: Any,
    agent_ids: list[str],
) -> None:
    from bilancio_v2.views import t_account_rows

    state = result.state
    selected_ids = [agent_id for agent_id in agent_ids if agent_id in state.agents]
    console.print("\n[bold]Balances:[/bold]")
    if not selected_ids:
        console.print("  - No active agents to display")
        return

    for agent_id in selected_ids:
        rows = t_account_rows(state, agent_id)
        agent = state.agents[agent_id]
        title = (
            f"{agent.name} [{agent_id}] ({agent.kind})"
            if agent.name and agent.name != agent_id
            else f"{agent_id} ({agent.kind})"
        )
        table = Table(
            title=f"{title} (clean-core)",
            show_header=True,
            header_style="bold",
            show_lines=True,
        )
        table.add_column("Name", style="green")
        table.add_column("Qty", style="green", justify="right")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Counterparty", style="green")
        table.add_column("Maturity", style="green", justify="right")
        table.add_column("Name", style="red")
        table.add_column("Qty", style="red", justify="right")
        table.add_column("Value", style="red", justify="right")
        table.add_column("Counterparty", style="red")
        table.add_column("Maturity", style="red", justify="right")
        asset_rows = rows["assets"]
        liability_rows = rows["liabs"]
        for index in range(max(len(asset_rows), len(liability_rows), 1)):
            asset_row = asset_rows[index] if index < len(asset_rows) else None
            liability_row = liability_rows[index] if index < len(liability_rows) else None
            table.add_row(
                *_clean_core_t_account_cells(asset_row),
                *_clean_core_t_account_cells(liability_row),
            )
        console.print(table)
        console.print()


def _clean_core_t_account_cells(
    row: dict[str, Any] | None,
) -> tuple[str, str, str, str, str]:
    if row is None:
        return ("", "", "", "", "")
    quantity = row.get("quantity")
    return (
        str(row.get("name") or ""),
        f"{quantity:,}" if quantity is not None else "-",
        _format_clean_core_amount(row.get("value_minor")),
        str(row.get("counterparty_name") or "-"),
        str(row.get("maturity") or "-"),
    )


def _display_clean_core_events(
    console: Console,
    events: list[dict[str, Any]],
    show: str,
) -> None:
    if not events:
        return

    if show == "summary":
        table = Table(
            title="Event Summary (clean-core)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Event")
        table.add_column("Count", justify="right")
        for kind, count in sorted(
            Counter(str(event.get("kind", "Unknown")) for event in events).items()
        ):
            table.add_row(kind, str(count))
        console.print()
        console.print(table)
        return

    if show == "table":
        _display_clean_core_event_tables(console, events)
        return

    table = Table(title="Events (clean-core)", show_header=True, header_style="bold")
    table.add_column("Day", justify="right")
    table.add_column("Phase")
    table.add_column("Kind")
    table.add_column("Details")
    for event in events:
        table.add_row(
            str(event.get("day", "")),
            str(event.get("phase", "")),
            str(event.get("kind", "Unknown")),
            _clean_core_event_details(event),
        )
    console.print()
    console.print(table)


def _display_clean_core_event_tables(
    console: Console,
    events: list[dict[str, Any]],
) -> None:
    console.print()
    console.print("[bold]Events (clean-core):[/bold]")

    setup_events = [event for event in events if event.get("phase") == "setup"]
    if setup_events:
        for renderable in display_events_tables_by_phase_renderables(setup_events, day=0):
            console.print(renderable)

    events_by_day: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("phase") == "setup":
            continue
        try:
            day = int(event.get("day", 0))
        except (TypeError, ValueError):
            day = 0
        events_by_day[day].append(event)

    for day in sorted(events_by_day):
        for renderable in display_events_tables_by_phase_renderables(
            events_by_day[day],
            day=day,
        ):
            console.print(renderable)


def _clean_core_balance_breakdown(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in sorted(row.items()):
        if not key.startswith(("assets_", "liabilities_", "inventory_", "nonfinancial_")):
            continue
        if Decimal(str(value or 0)) == 0:
            continue
        parts.append(f"{key}={_format_clean_core_amount(value)}")
    return "; ".join(parts)


def _clean_core_event_details(event: dict[str, Any]) -> str:
    detail_items = []
    for key, value in event.items():
        if key in {"kind", "day", "phase"}:
            continue
        detail_items.append(f"{key}={_format_clean_core_detail_value(value)}")
    return ", ".join(detail_items)


def _format_clean_core_detail_value(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{k}: {_format_clean_core_detail_value(v)}" for k, v in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_format_clean_core_detail_value(item) for item in value) + "]"
    return str(value)


def _format_clean_core_amount(value: Any) -> str:
    return f"{Decimal(str(value or 0)):,.2f}"
