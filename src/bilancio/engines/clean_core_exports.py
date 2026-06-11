"""Export writers for clean-core scenario results."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

ZERO = Decimal("0")


def write_balances_csv(result: Any, path: Path) -> None:
    """Write clean-core balances using the same CSV shape as the legacy export."""
    from bilancio.engines.clean_core_views import balance_rows

    rows = balance_rows(result)
    trial_balance = next(row for row in rows if row["agent_id"] == "SYSTEM")
    export_rows = [
        *rows,
        {
            "agent_id": "SYSTEM",
            "agent_name": "System Total",
            "agent_kind": "system",
            "item_type": "summary",
            "item_name": "Total Assets",
            "amount": trial_balance["total_financial_assets"],
        },
        {
            "agent_id": "SYSTEM",
            "agent_name": "System Total",
            "agent_kind": "system",
            "item_type": "summary",
            "item_name": "Total Liabilities",
            "amount": trial_balance["total_financial_liabilities"],
        },
        {
            "agent_id": "SYSTEM",
            "agent_name": "System Total",
            "agent_kind": "system",
            "item_type": "summary",
            "item_name": "Total Equity",
            "amount": (
                trial_balance["total_financial_assets"]
                - trial_balance["total_financial_liabilities"]
            ),
        },
    ]

    fieldnames = sorted({field for row in export_rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in export_rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_events_jsonl(result: Any, path: Path) -> None:
    """Write clean-core events as JSONL, preserving Decimal values like legacy exports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for event in result.events:
            json.dump(event, f, default=_json_default)
            f.write("\n")


def write_html_report(
    result: Any,
    config: Any,
    path: Path,
    *,
    max_days: int,
    quiet_days: int,
    agent_ids: list[str] | None = None,
    t_account: bool = False,
) -> None:
    """Write a readable clean-core HTML report for the public ``--html`` CLI flag."""
    from bilancio.engines.clean_core_views import balance_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = balance_rows(result)
    selected_agent_ids = [
        agent_id
        for agent_id in (agent_ids or [])
        if any(row["agent_id"] == agent_id for row in rows)
    ]
    display_rows = rows
    if selected_agent_ids:
        selected = set(selected_agent_ids)
        display_rows = [row for row in rows if row["agent_id"] in selected]
    rows_by_agent = {row["agent_id"]: row for row in display_rows}
    fields = [
        field
        for field in sorted({key for row in display_rows for key in row})
        if field != "agent_id"
        and any(row.get(field) not in (None, "", ZERO, 0) for row in display_rows)
    ]
    events_by_day: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in result.events:
        events_by_day[int(event.get("day", 0))].append(event)

    status = (
        f"Converged on day {result.final_day}"
        if result.reached_stable
        else f"Stopped without convergence (max_days = {max_days})"
    )

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Bilancio Simulation - {_escape(config.name)}</title>",
        "<style>",
        "body{font-family:Inter,Arial,sans-serif;background:#f7f8fa;color:#20242a;margin:0;}",
        ".container{max-width:1200px;margin:0 auto;padding:32px;}",
        "header{margin-bottom:24px;} h1{font-size:28px;margin:0 0 6px;} h2{font-size:20px;margin:0 0 12px;}",
        ".meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}",
        ".meta span{background:#fff;border:1px solid #d9dee7;border-radius:6px;padding:6px 10px;}",
        "section{background:#fff;border:1px solid #d9dee7;border-radius:8px;margin:16px 0;padding:18px;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;}",
        "th,td{border-bottom:1px solid #e5e8ef;padding:7px 8px;text-align:left;vertical-align:top;}",
        "th{background:#f0f3f8;font-weight:650;}",
        ".num{text-align:right;font-variant-numeric:tabular-nums}.event-kind{font-weight:650}.muted{color:#667085}.day h3{margin-top:0;}",
        "</style></head><body><div class=\"container\">",
        "<header>",
        "<h1>Bilancio Simulation</h1>",
        f"<h2>{_escape(config.name)}</h2>",
    ]
    if config.description:
        parts.append(f'<p class="muted">{_escape(config.description)}</p>')
    parts.extend(
        [
            '<div class="meta">',
            f"<span>Status: {status}</span>",
            f"<span>Final Day: {result.final_day}</span>",
            f"<span>Total Events: {len(result.events)}</span>",
            f"<span>Quiet Days: {quiet_days}</span>",
            "</div></header>",
            "<section><h2>Agents</h2><table><thead><tr><th>ID</th><th>Name</th><th>Type</th></tr></thead><tbody>",
        ]
    )
    for agent in config.agents:
        parts.append(
            "<tr>"
            f"<td>{_escape(agent.id)}</td>"
            f"<td>{_escape(agent.name)}</td>"
            f"<td>{_escape(agent.kind)}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></section>")

    if t_account and selected_agent_ids:
        parts.append("<section><h2>Final T-Accounts</h2>")
        for agent_id in selected_agent_ids:
            parts.append(_render_t_account_html(result.state, agent_id))
        parts.append("</section>")
    else:
        parts.append("<section><h2>Final Balances</h2><table><thead><tr><th>Agent</th>")
        for balance_field in fields:
            parts.append(f"<th>{_escape(balance_field)}</th>")
        parts.append("</tr></thead><tbody>")
        for agent_id, row in rows_by_agent.items():
            parts.append(f"<tr><td>{_escape(agent_id)}</td>")
            for balance_field in fields:
                parts.append(
                    f'<td class="num">{_escape(_display_value(row.get(balance_field, "")))}</td>'
                )
            parts.append("</tr>")
        parts.append("</tbody></table></section>")

    for day in sorted(events_by_day):
        title = "Day 0 (Setup)" if day == 0 else f"Day {day}"
        parts.append(
            f'<section class="day"><h3>{_escape(title)}</h3>'
            "<table><thead><tr><th>Phase</th><th>Event</th><th>Details</th></tr></thead><tbody>"
        )
        for event in events_by_day[day]:
            details = {
                key: value
                for key, value in event.items()
                if key not in {"day", "phase", "kind"}
            }
            parts.append(
                "<tr>"
                f"<td>{_escape(event.get('phase', ''))}</td>"
                f'<td class="event-kind">{_escape(event.get("kind", ""))}</td>'
                f"<td>{_escape(_event_details(details))}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></section>")

    parts.append("</div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def _render_t_account_html(state: Any, agent_id: str) -> str:
    from bilancio.engines.clean_core_views import t_account_rows

    agent = state.agents[agent_id]
    title = (
        f"{agent.name} [{agent_id}] ({agent.kind})"
        if agent.name and agent.name != agent_id
        else f"{agent_id} ({agent.kind})"
    )
    rows = t_account_rows(state, agent_id)
    parts = [
        f"<h3>{_escape(title)}</h3>",
        "<table><thead><tr>",
        "<th>Asset</th><th>Qty</th><th>Value</th><th>Counterparty</th><th>Maturity</th>",
        "<th>Liability</th><th>Qty</th><th>Value</th><th>Counterparty</th><th>Maturity</th>",
        "</tr></thead><tbody>",
    ]
    asset_rows = rows["assets"]
    liability_rows = rows["liabs"]
    for index in range(max(len(asset_rows), len(liability_rows), 1)):
        asset_row = asset_rows[index] if index < len(asset_rows) else None
        liability_row = liability_rows[index] if index < len(liability_rows) else None
        parts.append("<tr>")
        for cell in (*_t_account_html_cells(asset_row), *_t_account_html_cells(liability_row)):
            parts.append(f"<td>{_escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _t_account_html_cells(row: dict[str, Any] | None) -> tuple[str, str, str, str, str]:
    if row is None:
        return ("", "", "", "", "")
    quantity = row.get("quantity")
    return (
        str(row.get("name") or ""),
        _display_value(quantity) if quantity is not None else "-",
        _display_value(row.get("value_minor")),
        str(row.get("counterparty_name") or "-"),
        str(row.get("maturity") or "-"),
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral_value():
            return f"{int(normalized):,}"
        return f"{normalized:f}"
    return str(value)


def _event_details(details: dict[str, Any]) -> str:
    if not details:
        return ""
    return ", ".join(f"{key}={_display_value(value)}" for key, value in details.items())


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral_value():
            return int(normalized)
        return str(normalized)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
