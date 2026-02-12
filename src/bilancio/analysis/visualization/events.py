"""Event table formatting and display functions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from bilancio.engines.system import System
from bilancio.analysis.visualization.common import (
    RICH_AVAILABLE,
    RenderableType,
    _print,
)

# Import Rich components only if available
if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box



def display_events(events: List[Dict[str, Any]], format: str = 'detailed') -> None:
    """
    Display system events in a nicely formatted way.
    
    Args:
        events: List of event dictionaries from sys.state.events
        format: Display format ('detailed' or 'summary')
    """
    console = Console() if RICH_AVAILABLE else None
    
    if not events:
        _print("No events to display.", console)
        return
    
    if format == 'summary':
        _display_events_summary(events, console)
    else:
        _display_events_detailed(events, console)


def display_events_table(events: List[Dict[str, Any]], group_by_day: bool = True) -> None:
    """Render events as a table with canonical columns.

    Falls back to simple text when Rich is not available.
    """
    console = Console() if RICH_AVAILABLE else None

    if not events:
        _print("No events to display.", console)
        return

    columns = ["Day", "Phase", "Kind", "From", "To", "SKU/Instr", "Qty", "Amount", "Notes"]

    # Sort events deterministically
    # Drop phase marker events; preserve original insertion (chronological) order
    evs = [e for e in events if e.get("kind") not in ("PhaseA", "PhaseB", "PhaseC")]

    if RICH_AVAILABLE:
        from rich.table import Table as RichTable
        from rich import box as rich_box
        table = RichTable(title="Events", box=rich_box.HEAVY, show_lines=True)
        # Column definitions with better alignment
        table.add_column("Day", justify="right")
        table.add_column("Phase", justify="left")
        table.add_column("Kind", justify="left")
        table.add_column("From", justify="left")
        table.add_column("To", justify="left")
        table.add_column("SKU/Instr", justify="left")
        table.add_column("Qty", justify="right")
        table.add_column("Amount", justify="right")
        table.add_column("Notes", justify="left")
        # Alternate row shading for readability
        try:
            table.row_styles = ["on #ffffff", "on #e6f2ff"]
        except (AttributeError, TypeError):
            pass

        for e in evs:
            kind = str(e.get("kind", ""))
            # Canonical from/to mapping by kind
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
                frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer") or e.get("agent")
                to = e.get("to") or e.get("creditor") or e.get("payee")
            sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
            qty = e.get("qty") or e.get("quantity") or "—"
            amt = e.get("amount") or "—"

            notes = ""
            if kind == "ClientPayment":
                notes = f"{e.get('payer_bank','?')} → {e.get('payee_bank','?')}"
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                notes = f"{e.get('debtor_bank','?')} → {e.get('creditor_bank','?')}"
                if 'due_day' in e:
                    notes += f"; due {e.get('due_day')}"
            elif kind == "AgentDefaulted":
                shortfall = e.get('shortfall')
                trigger = e.get('trigger_contract')
                parts = []
                if shortfall is not None:
                    parts.append(f"shortfall {shortfall}")
                if trigger:
                    parts.append(f"trigger {trigger}")
                if parts:
                    notes = ", ".join(parts)

            table.add_row(
                str(e.get("day", "—")),
                str(e.get("phase", "—")),
                kind,
                str(frm or "—"),
                str(to or "—"),
                str(sku),
                str(qty),
                str(amt),
                notes,
            )

        console.print(table) if console else print(table)
    else:
        # Simple header + rows
        header = " | ".join(columns)
        print(header)
        print("-" * len(header))
        for e in evs:
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
                frm = e.get("frm") or e.get("from") or e.get("debtor") or e.get("payer") or e.get("agent")
                to = e.get("to") or e.get("creditor") or e.get("payee")
            sku = e.get("sku") or e.get("instr_id") or e.get("stock_id") or "—"
            qty = e.get("qty") or e.get("quantity") or "—"
            amt = e.get("amount") or "—"
            notes = ""
            if kind == "ClientPayment":
                notes = f"{e.get('payer_bank','?')} → {e.get('payee_bank','?')}"
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                notes = f"{e.get('debtor_bank','?')} → {e.get('creditor_bank','?')}"
                if 'due_day' in e:
                    notes += f"; due {e.get('due_day')}"
            elif kind == "AgentDefaulted":
                shortfall = e.get('shortfall')
                trigger = e.get('trigger_contract')
                parts = []
                if shortfall is not None:
                    parts.append(f"shortfall {shortfall}")
                if trigger:
                    parts.append(f"trigger {trigger}")
                if parts:
                    notes = ", ".join(parts)
            row = [
                str(e.get("day", "—")),
                str(e.get("phase", "—")),
                kind,
                str(frm or "—"),
                str(to or "—"),
                str(sku),
                str(qty),
                str(amt),
                notes,
            ]
            print(" | ".join(row))


def display_events_table_renderable(events: List[Dict[str, Any]]) -> RenderableType:
    """Return a Rich Table renderable (or string) for events table."""
    if not events:
        return Text("No events to display.", style="dim") if RICH_AVAILABLE else "No events to display."

    columns = ["Day", "Phase", "Kind", "From", "To", "SKU/Instr", "Qty", "Amount", "Notes"]
    # Drop phase marker events; preserve original insertion (chronological) order
    evs = [e for e in events if e.get("kind") not in ("PhaseA", "PhaseB", "PhaseC")]

    if RICH_AVAILABLE:
        from rich.table import Table as RichTable
        from rich import box as rich_box
        table = RichTable(title="Events", box=rich_box.HEAVY, show_lines=True)
        table.add_column("Day", justify="right")
        table.add_column("Phase", justify="left")
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

        for e in evs:
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
                notes = f"{e.get('payer_bank','?')} → {e.get('payee_bank','?')}"
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                notes = f"{e.get('debtor_bank','?')} → {e.get('creditor_bank','?')}"
                if 'due_day' in e:
                    notes += f"; due {e.get('due_day')}"

            table.add_row(
                str(e.get("day", "—")),
                str(e.get("phase", "—")),
                kind,
                str(frm or "—"),
                str(to or "—"),
                str(sku),
                str(qty),
                str(amt),
                notes,
            )
        return table
    else:
        header = " | ".join(columns)
        lines = [header, "-" * len(header)]
        for e in evs:
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
                notes = f"{e.get('payer_bank','?')} → {e.get('payee_bank','?')}"
            elif kind in ("InterbankCleared", "InterbankOvernightCreated"):
                notes = f"{e.get('debtor_bank','?')} → {e.get('creditor_bank','?')}"
                if 'due_day' in e:
                    notes += f"; due {e.get('due_day')}"
            row = [
                str(e.get("day", "—")),
                str(e.get("phase", "—")),
                kind,
                str(frm or "—"),
                str(to or "—"),
                str(sku),
                str(qty),
                str(amt),
                notes,
            ]
            lines.append(" | ".join(row))
        return "\n".join(lines)


def _display_events_summary(events: List[Dict[str, Any]], console: Optional['Console'] = None) -> None:
    """Display events in a condensed summary format."""
    for event in events:
        kind = event.get("kind", "Unknown")
        day = event.get("day", "?")
        
        if kind == "PayableSettled":
            _print(f"Day {day}: 💰 {event['debtor']} → {event['creditor']}: ${event['amount']}", console)
        elif kind == "DeliveryObligationSettled":
            qty = event.get('qty', event.get('quantity', 'N/A'))
            _print(f"Day {day}: 📦 {event['debtor']} → {event['creditor']}: {qty} {event.get('sku', 'items')}", console)
        elif kind == "StockTransferred":
            _print(f"Day {day}: 🚚 {event['frm']} → {event['to']}: {event['qty']} {event['sku']}", console)
        elif kind == "CashTransferred":
            _print(f"Day {day}: 💵 {event['frm']} → {event['to']}: ${event['amount']}", console)



def _display_events_detailed(events: List[Dict[str, Any]], console: Optional['Console'] = None) -> None:
    """Display events grouped by day with detailed formatting."""
    # Separate setup events from day events
    setup_events = []
    events_by_day = defaultdict(list)
    
    for event in events:
        # Check if this is a setup phase event
        if event.get("phase") == "setup":
            setup_events.append(event)
        else:
            day = event.get("day", -1)
            events_by_day[day].append(event)
    
    # Display setup events first if any
    if setup_events:
        _print(f"\n📅 Setup Phase:", console)
        # Setup events don't have phase markers, display them directly
        for event in setup_events:
            _display_single_event(event, console, indent="  ")
    
    # Display events for each day
    for day in sorted(events_by_day.keys()):
        if day >= 0:
            _print(f"\n📅 Day {day}:", console)
        else:
            _print(f"\n📅 Unknown Day:", console)
        
        _display_day_events(events_by_day[day], console)


def _display_day_events(day_events: List[Dict[str, Any]], console: Optional['Console'] = None) -> None:
    """Display events for a single day with proper formatting."""
    # Group events by their phase timing
    phase_a_events = []
    phase_b_events = []
    phase_c_events = []
    
    # Track which phase we're in based on phase markers
    current_phase = "A"  # Start with phase A
    
    for event in day_events:
        kind = event.get("kind", "Unknown")
        
        # Phase markers change which phase we're in
        if kind == "PhaseA":
            current_phase = "A"
        elif kind == "PhaseB":
            current_phase = "B"
        elif kind == "PhaseC":
            current_phase = "C"
        else:
            # Regular events go into the current phase bucket
            if current_phase == "A":
                phase_a_events.append(event)
            elif current_phase == "B":
                phase_b_events.append(event)
            elif current_phase == "C":
                phase_c_events.append(event)
    
    # Always display all three phases
    # Phase A - No-op phase, just marks beginning of day
    _print(f"\n  ⏰ Phase A: Day begins", console)
    for event in phase_a_events:
        _display_single_event(event, console, indent="    ")
    
    # Phase B - Settlement phase where obligations are fulfilled
    _print(f"\n  💳 Phase B: Settlement (fulfilling due obligations)", console)
    for event in phase_b_events:
        _display_single_event(event, console, indent="    ")
    
    # Phase C - Intraday netting
    _print(f"\n  📋 Phase C: Intraday netting", console)
    for event in phase_c_events:
        _display_single_event(event, console, indent="    ")
    
    # Mark end of day
    _print(f"\n  🌙 Day ended", console)


def _display_single_event(event: Dict[str, Any], console: Optional['Console'] = None, indent: str = "  ") -> None:
    """Display a single event with proper formatting."""
    kind = event.get("kind", "Unknown")
    
    if kind == "StockCreated":
        _print(f"{indent}🏭 Stock created: {event['owner']} gets {event['qty']} {event['sku']}", console)
    
    elif kind == "CashMinted":
        _print(f"{indent}💰 Cash minted: ${event['amount']} to {event['to']}", console)
    
    elif kind == "PayableSettled":
        _print(f"{indent}✅ Payment settled: {event['debtor']} → {event['creditor']}: ${event['amount']}", console)
    
    elif kind == "PayableCancelled":
        _print(f"{indent}  └─ Payment obligation removed from books", console)
    
    elif kind == "DeliveryObligationSettled":
        qty = event.get('qty', event.get('quantity', 'N/A'))
        sku = event.get('sku', 'items')
        _print(f"{indent}✅ Delivery settled: {event['debtor']} → {event['creditor']}: {qty} {sku}", console)
    
    elif kind == "DeliveryObligationCancelled":
        _print(f"{indent}  └─ Delivery obligation removed from books", console)
    
    elif kind == "StockTransferred":
        _print(f"{indent}📦 Stock transferred: {event['frm']} → {event['to']}: {event['qty']} {event['sku']}", console)
    
    elif kind == "CashTransferred":
        _print(f"{indent}💵 Cash transferred: {event['frm']} → {event['to']}: ${event['amount']}", console)
    
    elif kind == "StockSplit":
        sku = event.get('sku', 'N/A')
        original_qty = event.get('original_qty', 0)
        split_qty = event.get('split_qty', 0)
        remaining_qty = event.get('remaining_qty', 0)
        # Show shortened IDs for readability
        original_id = event.get('original_id', '')
        new_id = event.get('new_id', '')
        short_orig = original_id.split('_')[-1][:8] if original_id else 'N/A'
        short_new = new_id.split('_')[-1][:8] if new_id else 'N/A'
        _print(f"{indent}📊 Stock split: {short_orig} → {short_new}: {split_qty} {sku} (keeping {remaining_qty})", console)
    
    elif kind == "ReservesMinted":
        amount = event.get('amount', 0)
        to = event.get('to', 'N/A')
        _print(f"{indent}🏦 Reserves minted: ${amount} to {to}", console)
    
    elif kind == "CashDeposited":
        customer = event.get('customer', 'N/A')
        bank = event.get('bank', 'N/A')
        amount = event.get('amount', 0)
        _print(f"{indent}🏧 Cash deposited: {customer} → {bank}: ${amount}", console)
    
    elif kind == "DeliveryObligationCreated":
        frm = event.get('frm', event.get('from', 'N/A'))
        to = event.get('to', 'N/A')
        qty = event.get('qty', event.get('quantity', 0))
        sku = event.get('sku', 'N/A')
        due_day = event.get('due_day', 'N/A')
        _print(f"{indent}📝 Delivery obligation created: {frm} → {to}: {qty} {sku} (due day {due_day})", console)
    
    elif kind == "PayableCreated":
        frm = event.get('frm', event.get('from', event.get('debtor', 'N/A')))
        to = event.get('to', event.get('creditor', 'N/A'))
        amount = event.get('amount', 0)
        due_day = event.get('due_day', 'N/A')
        _print(f"{indent}💸 Payable created: {frm} → {to}: ${amount} (due day {due_day})", console)
    
    elif kind == "ClientPayment":
        payer = event.get('payer', 'N/A')
        payee = event.get('payee', 'N/A')
        amount = event.get('amount', 0)
        payer_bank = event.get('payer_bank', '')
        payee_bank = event.get('payee_bank', '')
        _print(f"{indent}💳 Client payment: {payer} ({payer_bank}) → {payee} ({payee_bank}): ${amount}", console)
    
    elif kind == "InterbankCleared":
        debtor = event.get('debtor_bank', 'N/A')
        creditor = event.get('creditor_bank', 'N/A')
        amount = event.get('amount', 0)
        _print(f"{indent}🏦 Interbank cleared: {debtor} → {creditor}: ${amount} (netted)", console)
    
    elif kind == "ReservesTransferred":
        frm = event.get('frm', 'N/A')
        to = event.get('to', 'N/A')
        amount = event.get('amount', 0)
        _print(f"{indent}💰 Reserves transferred: {frm} → {to}: ${amount}", console)
    
    elif kind == "InstrumentMerged":
        # This is a technical event, show it more compactly
        _print(f"{indent}🔀 Instruments merged", console)
    
    elif kind == "InterbankOvernightCreated":
        debtor = event.get('debtor_bank', 'N/A')
        creditor = event.get('creditor_bank', 'N/A')
        amount = event.get('amount', 0)
        _print(f"{indent}🌙 Overnight payable created: {debtor} → {creditor}: ${amount}", console)
    
    elif kind in ["PhaseA", "PhaseB", "PhaseC"]:
        # Phase markers are not displayed as events themselves
        pass
    
    else:
        # For any other event types, show raw data
        _print(f"{indent}• {kind}: {event}", console)


def display_events_for_day(system: System, day: int) -> None:
    """
    Display all events that occurred on a specific simulation day.
    
    Args:
        system: The bilancio system instance
        day: The simulation day to display events for
    """
    console = Console() if RICH_AVAILABLE else None
    events = [e for e in system.state.events if e.get("day") == day]
    
    if not events:
        _print("  No events occurred on this day.", console)
        return
    
    _display_day_events(events, console)


def display_events_renderable(events: List[Dict[str, Any]], format: str = 'detailed') -> List[RenderableType]:
    """
    Return renderables for system events in a nicely formatted way.
    
    Args:
        events: List of event dictionaries from sys.state.events
        format: Display format ('detailed' or 'summary')
        
    Returns:
        List of Rich renderables (or strings for simple format)
    """
    if not events:
        if RICH_AVAILABLE:
            return [Text("No events to display.", style="dim")]
        else:
            return ["No events to display."]
    
    if format == 'summary':
        return _build_events_summary_renderables(events)
    else:
        return _build_events_detailed_renderables(events)


def display_events_for_day_renderable(system: System, day: int) -> List[RenderableType]:
    """
    Return renderables for all events that occurred on a specific simulation day.
    
    Args:
        system: The bilancio system instance
        day: The simulation day to display events for
        
    Returns:
        List of Rich renderables (or strings for simple format)
    """
    events = [e for e in system.state.events if e.get("day") == day]
    
    if not events:
        if RICH_AVAILABLE:
            return [Text("  No events occurred on this day.", style="dim")]
        else:
            return ["  No events occurred on this day."]
    
    return _build_day_events_renderables(events)


def _build_events_summary_renderables(events: List[Dict[str, Any]]) -> List[RenderableType]:
    """Build renderables for events in summary format."""
    renderables: List[RenderableType] = []
    
    for event in events:
        kind = event.get("kind", "Unknown")
        day = event.get("day", "?")
        
        if kind == "PayableSettled":
            text = f"Day {day}: 💰 {event['debtor']} → {event['creditor']}: ${event['amount']}"
        elif kind == "DeliveryObligationSettled":
            qty = event.get('qty', event.get('quantity', 'N/A'))
            text = f"Day {day}: 📦 {event['debtor']} → {event['creditor']}: {qty} {event.get('sku', 'items')}"
        elif kind == "StockTransferred":
            text = f"Day {day}: 🚚 {event['frm']} → {event['to']}: {event['qty']} {event['sku']}"
        elif kind == "CashTransferred":
            text = f"Day {day}: 💵 {event['frm']} → {event['to']}: ${event['amount']}"
        else:
            continue  # Skip other events in summary mode
        
        if RICH_AVAILABLE:
            renderables.append(Text(text))
        else:
            renderables.append(text)
    
    return renderables


def _build_events_detailed_renderables(events: List[Dict[str, Any]]) -> List[RenderableType]:
    """Build renderables for events in detailed format, organized by phases."""
    # Import and use the phase-aware version
    from bilancio.analysis.visualization_phases import build_events_detailed_with_phases
    return build_events_detailed_with_phases(events, RICH_AVAILABLE)


def _format_single_event(event: Dict[str, Any], registry: Any) -> List[RenderableType]:
    """Format a single event and return renderables."""
    renderables: List[RenderableType] = []
    
    # Format the event using the registry
    title, lines, icon = registry.format(event)
    
    if RICH_AVAILABLE:
        from rich.text import Text
        # Create a nice formatted display with icon and details
        text = Text()
        
        # Add icon and title with color based on event type
        if "Transfer" in title or "Payment" in title:
            text.append(title, style="bold cyan")
        elif "Settled" in title or "Cleared" in title:
            text.append(title, style="bold green")
        elif "Created" in title or "Minted" in title:
            text.append(title, style="bold yellow")
        elif "Consolidation" in title or "Split" in title:
            text.append(title, style="dim italic")
        else:
            text.append(title, style="bold")
        
        # Add details with proper indentation and styling
        if lines:
            for i, line in enumerate(lines[:3]):  # Show up to 3 lines
                text.append("\n   ")
                if "→" in line or "←" in line:
                    # Flow lines - make them prominent
                    text.append(line, style="white")
                elif ":" in line:
                    # Split field and value for better formatting
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        field, value = parts
                        text.append(field + ":", style="dim")
                        text.append(value, style="white")
                    else:
                        text.append(line, style="dim white")
                elif line.startswith("("):
                    # Technical explanations in parentheses - make them dimmer
                    text.append(line, style="dim italic")
                else:
                    text.append(line, style="white")
        
        renderables.append(text)
    else:
        # Simple text format
        simple_text = f"• {title}"
        if lines:
            simple_text += " - " + ", ".join(lines[:2])
        renderables.append(simple_text)

    return [renderables[0]] if renderables else []


def _build_day_events_renderables(events: List[Dict[str, Any]]) -> List[RenderableType]:
    """Build renderables for events in a single day."""
    return _build_events_detailed_renderables(events)
