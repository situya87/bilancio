"""Phase summary visualization for events."""

from __future__ import annotations

from typing import Any

from bilancio.analysis.visualization.common import RICH_AVAILABLE, RenderableType


def display_events_tables_by_phase_renderables(
    events: list[dict[str, Any]], day: int | None = None
) -> list[RenderableType]:
    """Return three event tables (A, B, C) using phase markers as section dividers.

    - Excludes PhaseA/PhaseB/PhaseC events from rows.
    - Titles indicate the phase and optional day.
    """
    if RICH_AVAILABLE:
        from rich import box as rich_box
        from rich.table import Table as RichTable

    # If these are setup-phase events (day 0), render as a single "Setup" table
    if any(e.get("phase") == "setup" for e in events):
        return _build_single_setup_table(events, day)

    # Group by phase markers in original order
    buckets: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    current = "A"
    for e in events:
        kind = e.get("kind")
        if kind == "PhaseA":
            current = "A"
            continue
        if kind == "PhaseB":
            current = "B"
            continue
        if kind == "PhaseC":
            current = "C"
            continue
        buckets[current].append(e)

    def build_table(phase: str, rows: list[dict[str, Any]]) -> RenderableType:
        title_parts = {
            "A": "Phase A — Start of day",
            "B": "Phase B — Settlement",
            "C": "Phase C — Clearing",
        }
        title = title_parts.get(phase, f"Phase {phase}")
        if day is not None:
            title = f"{title} (Day {day})"

        if not RICH_AVAILABLE:
            # Simple text fallback
            header = ["Kind", "From", "To", "SKU/Instr", "Qty", "Amount", "Notes"]
            out = [f"{title}", " | ".join(header), "-" * 80]
            for e in rows:
                kind = str(e.get("kind", ""))
                # map from/to like in table renderers
                if kind in ("CashDeposited", "CashWithdrawn"):
                    frm = e.get("customer") if kind == "CashDeposited" else e.get("bank")
                    to = e.get("bank") if kind == "CashDeposited" else e.get("customer")
                elif kind in ("ClientPayment", "IntraBankPayment", "CashPayment"):
                    frm = e.get("payer")
                    to = e.get("payee")
                elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                    frm = e.get("debtor_bank")
                    to = e.get("creditor_bank")
                elif kind == "StockCreated":
                    frm = e.get("owner")
                    to = None
                else:
                    frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer")
                    to = e.get("to") or e.get("creditor") or e.get("payee")
                sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
                qty = e.get("qty") or e.get("quantity") or "—"
                amt = e.get("amount") or "—"
                notes = ""
                if kind == "ClientPayment":
                    notes = f"{e.get('payer_bank', '?')} → {e.get('payee_bank', '?')}"
                elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                    notes = f"{e.get('debtor_bank', '?')} → {e.get('creditor_bank', '?')}"
                    if "due_day" in e:
                        notes += f"; due {e.get('due_day')}"
                elif kind == "AgentDefaulted":
                    shortfall = e.get("shortfall")
                    trigger = e.get("trigger_contract")
                    parts = []
                    if shortfall is not None:
                        parts.append(f"shortfall {shortfall}")
                    if trigger:
                        parts.append(f"trigger {trigger}")
                    if parts:
                        notes = ", ".join(parts)
                out.append(
                    " | ".join(map(str, [kind, frm or "—", to or "—", sku, qty, amt, notes]))
                )
            return "\n".join(out)

        # Rich table
        table = RichTable(title=title, box=rich_box.HEAVY, show_lines=True)
        table.add_column("Kind", justify="left")
        table.add_column("From", justify="left")
        table.add_column("To", justify="left")
        table.add_column("SKU/Instr", justify="left")
        table.add_column("Qty", justify="right")
        table.add_column("Amount", justify="right")
        table.add_column("Notes", justify="left")
        try:
            table.row_styles = (
                ["on #ffffff", "on #e6f2ff"] if phase != "C" else ["on #ffffff", "on #fff2cc"]
            )
        except (AttributeError, TypeError):
            pass
        for e in rows:
            kind = str(e.get("kind", ""))
            if kind in ("CashDeposited", "CashWithdrawn"):
                frm = e.get("customer") if kind == "CashDeposited" else e.get("bank")
                to = e.get("bank") if kind == "CashDeposited" else e.get("customer")
            elif kind in ("ClientPayment", "IntraBankPayment", "CashPayment"):
                frm = e.get("payer")
                to = e.get("payee")
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                frm = e.get("debtor_bank")
                to = e.get("creditor_bank")
            elif kind == "StockCreated":
                frm = e.get("owner")
                to = None
            else:
                frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer")
                to = e.get("to") or e.get("creditor") or e.get("payee")
            sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
            qty = e.get("qty") or e.get("quantity") or "—"
            amt = e.get("amount") or "—"
            notes = ""
            if kind == "ClientPayment":
                notes = f"{e.get('payer_bank', '?')} → {e.get('payee_bank', '?')}"
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                notes = f"{e.get('debtor_bank', '?')} → {e.get('creditor_bank', '?')}"
                if "due_day" in e:
                    notes += f"; due {e.get('due_day')}"
            elif kind == "AgentDefaulted":
                shortfall = e.get("shortfall")
                trigger = e.get("trigger_contract")
                parts = []
                if shortfall is not None:
                    parts.append(f"shortfall {shortfall}")
                if trigger:
                    parts.append(f"trigger {trigger}")
                if parts:
                    notes = ", ".join(parts)
            table.add_row(
                kind, str(frm or "—"), str(to or "—"), str(sku), str(qty), str(amt), notes
            )
        return table

    renderables: list[RenderableType] = []
    # Phase A is intentionally empty for now; only include if it has rows
    if buckets["A"]:
        renderables.append(build_table("A", buckets["A"]))
    # Phase B: settlements
    renderables.append(build_table("B", buckets["B"]))
    # Phase C: clearing
    renderables.append(build_table("C", buckets["C"]))
    return renderables


def _build_single_setup_table(
    events: list[dict[str, Any]], day: int | None = None
) -> list[RenderableType]:
    """Render a single setup table for setup-phase events (day 0)."""
    rows = [
        e
        for e in events
        if e.get("phase") == "setup" and e.get("kind") not in ("PhaseA", "PhaseB", "PhaseC")
    ]
    title = "Setup"
    if day is not None:
        title = f"{title} (Day {day})"
    if not RICH_AVAILABLE:
        header = ["Kind", "From", "To", "SKU/Instr", "Qty", "Amount", "Notes"]
        out = [title, " | ".join(header), "-" * 80]
        for e in rows:
            kind = str(e.get("kind", ""))
            if kind in ("CashDeposited", "CashWithdrawn"):
                frm = e.get("customer") if kind == "CashDeposited" else e.get("bank")
                to = e.get("bank") if kind == "CashDeposited" else e.get("customer")
            elif kind in ("ClientPayment", "IntraBankPayment", "CashPayment"):
                frm = e.get("payer")
                to = e.get("payee")
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                frm = e.get("debtor_bank")
                to = e.get("creditor_bank")
            elif kind == "StockCreated":
                frm = e.get("owner")
                to = None
            else:
                frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer")
                to = e.get("to") or e.get("creditor") or e.get("payee")
            sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
            qty = e.get("qty") or e.get("quantity") or "—"
            amt = e.get("amount") or "—"
            notes = ""
            out.append(" | ".join(map(str, [kind, frm or "—", to or "—", sku, qty, amt, notes])))
        return ["\n".join(out)]

    from rich import box as rich_box
    from rich.table import Table as RichTable

    table = RichTable(title=title, box=rich_box.HEAVY, show_lines=True)
    table.add_column("Kind", justify="left")
    table.add_column("From", justify="left")
    table.add_column("To", justify="left")
    table.add_column("SKU/Instr", justify="left")
    table.add_column("Qty", justify="right")
    table.add_column("Amount", justify="right")
    table.add_column("Notes", justify="left")
    try:
        table.row_styles = ["on #ffffff", "on #e6f2ff"]
    except (AttributeError, TypeError):
        pass
    for e in rows:
        kind = str(e.get("kind", ""))
        if kind in ("CashDeposited", "CashWithdrawn"):
            frm = e.get("customer") if kind == "CashDeposited" else e.get("bank")
            to = e.get("bank") if kind == "CashDeposited" else e.get("customer")
        elif kind in ("ClientPayment", "IntraBankPayment", "CashPayment"):
            frm = e.get("payer")
            to = e.get("payee")
        elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
            frm = e.get("debtor_bank")
            to = e.get("creditor_bank")
        elif kind == "StockCreated":
            frm = e.get("owner")
            to = None
        else:
            frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer")
            to = e.get("to") or e.get("creditor") or e.get("payee")
        sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
        qty = e.get("qty") or e.get("quantity") or "—"
        amt = e.get("amount") or "—"
        notes = ""
        table.add_row(kind, str(frm or "—"), str(to or "—"), str(sku), str(qty), str(amt), notes)
    return [table]
