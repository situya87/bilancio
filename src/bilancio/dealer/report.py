"""HTML report generation for dealer ring simulations."""

import html
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simulation import DaySnapshot, DealerRingConfig


def _load_base_css() -> str:
    """Load CSS from assets/export.css."""
    css_path = Path(__file__).parent.parent / "ui" / "assets" / "export.css"
    return css_path.read_text()


def _get_dealer_css() -> str:
    """Return dealer-specific CSS extensions."""
    return """
/* Dealer card styling */
.dealer-card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.dealer-card h3 {
  color: #1e40af;
  margin-bottom: 16px;
  border-bottom: 2px solid #dbeafe;
  padding-bottom: 8px;
}
.params-table {
  width: 100%;
  border-collapse: collapse;
}
.params-table th {
  text-align: left;
  padding: 6px 12px;
  color: #64748b;
  font-weight: 500;
  width: 40%;
}
.params-table td {
  text-align: right;
  padding: 6px 12px;
  font-family: 'SF Mono', Monaco, monospace;
  color: #1f2937;
}
.params-table tr.section-header th {
  color: #374151;
  font-weight: 600;
  padding-top: 12px;
  border-bottom: 1px solid #e5e7eb;
}
/* VBT card */
.vbt-card {
  background: linear-gradient(135deg, #f0fdf4, #fff);
  border-left: 4px solid #16a34a;
}
.vbt-card h3 {
  color: #16a34a;
  border-bottom-color: #bbf7d0;
}
/* Event type colors */
.event-trade { color: #3b82f6; }
.event-passthrough { color: #8b5cf6; }
.event-rebucket { color: #f59e0b; }
.event-settlement { color: #16a34a; }
.event-default { color: #dc2626; }
.event-quote { color: #64748b; }
.event-vbt-anchor-update { color: #14b8a6; }
/* Pin indicators */
.pin-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-left: 4px;
}
.pin-indicator.pinned { background: #f59e0b; }
.pin-indicator.not-pinned { background: #d1d5db; }
/* Grid layouts */
.dealers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 24px;
}
.traders-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
}
/* Inventory list */
.inventory-list {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
}
.inventory-list h4 {
  color: #64748b;
  font-size: 14px;
  margin-bottom: 8px;
}
.inventory-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 13px;
}
.inventory-list li {
  padding: 4px 0;
  color: #4b5563;
  font-family: 'SF Mono', Monaco, monospace;
}
"""


def _html_escape(value: Any) -> str:
    """HTML escape for safe display."""
    return html.escape(str(value))


def _fmt_decimal(d: Decimal | float | int | str | None, precision: int = 4) -> str:
    """Format Decimal trimming trailing zeros."""
    if d is None:
        return "—"
    if isinstance(d, str):
        try:
            d = Decimal(d)
        except (ValueError, TypeError, ArithmeticError):
            return str(d)
    if isinstance(d, int | float):
        d = Decimal(str(d))
    # Format with requested precision, then strip trailing zeros
    formatted = f"{d:.{precision}f}".rstrip("0").rstrip(".")
    return formatted


def _render_header(
    config: "DealerRingConfig", snapshots: list[Any], title: str, subtitle: str
) -> str:
    """Render header with config summary."""
    num_days = len(snapshots) - 1 if snapshots else 0  # -1 because Day 0 is setup
    len(config.buckets)
    num_traders = len(snapshots[0].traders) if snapshots else 0

    # Count total tickets
    total_tickets = len(snapshots[0].tickets) if snapshots else 0

    # Get bucket names
    bucket_names = ", ".join(b.name for b in config.buckets)

    # Ticket size
    ticket_size = _fmt_decimal(config.ticket_size)

    return f"""
<header>
  <h1>{_html_escape(title)}</h1>
  <p class="description">{_html_escape(subtitle)}</p>
  <div class="meta">
    <span><strong>Days:</strong> {num_days}</span>
    <span><strong>Buckets:</strong> {bucket_names}</span>
    <span><strong>Traders:</strong> {num_traders}</span>
    <span><strong>Tickets:</strong> {total_tickets}</span>
    <span><strong>Ticket Size (S):</strong> {ticket_size}</span>
  </div>
</header>
"""


def _render_events_table(events: list[dict[str, Any]], table_title: str) -> str:
    """Render events table for a day."""
    # Filter out day_start events
    display_events = [e for e in events if e.get("kind") != "day_start"]

    if not display_events:
        return f"""
<div class="events-table">
  <h3>{_html_escape(table_title)}</h3>
  <p style="color: #9ca3af; font-style: italic;">No events for this day</p>
</div>
"""

    rows = []
    for event in display_events:
        kind = event.get("kind", "unknown")

        # Determine CSS class for event type
        css_class = f"event-{kind.replace('_', '-')}"

        # Check for passthrough trades
        if kind == "trade" and event.get("is_passthrough"):
            css_class = "event-passthrough"

        # Format details based on event kind
        if kind == "trade":
            side = event.get("side", "")
            price = _fmt_decimal(event.get("price"))
            bucket = event.get("bucket", "")
            is_passthrough = event.get("is_passthrough", False)
            trader_id = event.get("trader_id", "")[:20]
            ticket_id = event.get("ticket_id", "")
            details = f"{side} @ {price} [{bucket}]"
            notes = f"trader: {trader_id}, ticket: {ticket_id}"
            if is_passthrough:
                notes += " (passthrough to VBT)"
        elif kind == "rebucket":
            old_bucket = event.get("old_bucket", "")
            new_bucket = event.get("new_bucket", "")
            price = _fmt_decimal(event.get("price"))
            holder_type = event.get("holder_type", "")
            ticket_id = event.get("ticket_id", "")
            details = f"{old_bucket} → {new_bucket} @ {price}"
            notes = f"{holder_type} ticket {ticket_id}"
        elif kind == "settlement":
            issuer = event.get("issuer_id", "")[:20]
            total_paid = _fmt_decimal(event.get("total_paid"), 2)
            n_tickets = event.get("n_tickets", 0)
            details = f"Paid {total_paid} on {n_tickets} tickets"
            notes = f"issuer: {issuer}"
        elif kind == "default":
            issuer = event.get("issuer_id", "")[:20]
            recovery_rate = _fmt_decimal(event.get("recovery_rate"), 2)
            total_due = _fmt_decimal(event.get("total_due"), 2)
            total_paid = _fmt_decimal(event.get("total_paid"), 2)
            bucket = event.get("bucket", "")
            details = f"Recovery {recovery_rate} (due: {total_due}, paid: {total_paid})"
            notes = f"issuer: {issuer}, bucket: {bucket}"
        elif kind == "quote":
            bucket = event.get("bucket", "")
            dealer_bid = _fmt_decimal(event.get("dealer_bid"))
            dealer_ask = _fmt_decimal(event.get("dealer_ask"))
            vbt_bid = _fmt_decimal(event.get("vbt_bid"))
            vbt_ask = _fmt_decimal(event.get("vbt_ask"))
            inventory = event.get("inventory", 0)
            capacity = _fmt_decimal(event.get("capacity"))
            details = f"[{bucket}] D: {dealer_bid}/{dealer_ask}, VBT: {vbt_bid}/{vbt_ask}"
            notes = f"inv={inventory}, cap={capacity}"
        elif kind == "vbt_anchor_update":
            bucket = event.get("bucket", "")
            M_old = _fmt_decimal(event.get("M_old"))
            M_new = _fmt_decimal(event.get("M_new"))
            O_old = _fmt_decimal(event.get("O_old"))
            O_new = _fmt_decimal(event.get("O_new"))
            loss_rate = _fmt_decimal(event.get("loss_rate"), 4)
            details = f"[{bucket}] M: {M_old}→{M_new}, O: {O_old}→{O_new}"
            notes = f"loss rate: {loss_rate}"
        else:
            details = str(event)
            notes = ""

        rows.append(f"""
          <tr>
            <td class="{css_class}">{_html_escape(kind)}</td>
            <td>{_html_escape(details)}</td>
            <td class="notes">{_html_escape(notes)}</td>
          </tr>
        """)

    return f"""
<div class="events-table">
  <h3>{_html_escape(table_title)}</h3>
  <table>
    <thead>
      <tr>
        <th style="width: 120px;">Kind</th>
        <th>Details</th>
        <th style="width: 200px;">Notes</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</div>
"""


def _render_dealer_card(dealer_dict: dict[str, Any], vbt_dict: dict[str, Any]) -> str:
    """Render dealer state card with kernel params."""
    bucket_id = dealer_dict.get("bucket_id", "unknown")

    # Inventory
    a = dealer_dict.get("a", 0)
    x = _fmt_decimal(dealer_dict.get("x"))
    cash = _fmt_decimal(dealer_dict.get("cash"))
    V = _fmt_decimal(dealer_dict.get("V"))

    # Capacity
    K_star = dealer_dict.get("K_star", 0)
    X_star = _fmt_decimal(dealer_dict.get("X_star"))
    N = dealer_dict.get("N", 0)
    lambda_ = _fmt_decimal(dealer_dict.get("lambda_"))

    # Quotes
    inside_width = _fmt_decimal(dealer_dict.get("I"))
    midline = _fmt_decimal(dealer_dict.get("midline"))
    bid = _fmt_decimal(dealer_dict.get("bid"))
    ask = _fmt_decimal(dealer_dict.get("ask"))

    # Pin flags
    is_pinned_bid = dealer_dict.get("is_pinned_bid", False)
    is_pinned_ask = dealer_dict.get("is_pinned_ask", False)
    pin_bid_class = "pinned" if is_pinned_bid else "not-pinned"
    pin_ask_class = "pinned" if is_pinned_ask else "not-pinned"
    pin_bid_title = "Pinned at VBT bid" if is_pinned_bid else "Not pinned"
    pin_ask_title = "Pinned at VBT ask" if is_pinned_ask else "Not pinned"

    # Inventory list
    inventory = dealer_dict.get("inventory", [])
    inventory_items = []
    for t in inventory:
        tid = t.get("id", "?")
        issuer = t.get("issuer_id", "?")[:15]
        face = _fmt_decimal(t.get("face"), 2)
        tau = t.get("remaining_tau", "?")
        inventory_items.append(f"<li>{tid} (issuer: {issuer}, face: {face}, τ: {tau})</li>")

    inventory_html = ""
    if inventory_items:
        inventory_html = f"""
  <div class="inventory-list">
    <h4>Inventory ({len(inventory)} tickets)</h4>
    <ul>
      {"".join(inventory_items)}
    </ul>
  </div>
"""

    return f"""
<div class="dealer-card">
  <h3>Dealer: {_html_escape(bucket_id.upper())}</h3>
  <table class="params-table">
    <tbody>
      <tr class="section-header"><th colspan="2">Inventory</th></tr>
      <tr><th>Tickets (a)</th><td>{a}</td></tr>
      <tr><th>Face inventory (x)</th><td>{x}</td></tr>
      <tr><th>Cash (C)</th><td>{cash}</td></tr>
      <tr><th>Mid-valued (V)</th><td>{V}</td></tr>
      <tr class="section-header"><th colspan="2">Capacity</th></tr>
      <tr><th>Max tickets (K*)</th><td>{K_star}</td></tr>
      <tr><th>One-sided cap (X*)</th><td>{X_star}</td></tr>
      <tr><th>Ladder rungs (N)</th><td>{N}</td></tr>
      <tr><th>Layoff prob (λ)</th><td>{lambda_}</td></tr>
      <tr class="section-header"><th colspan="2">Quotes</th></tr>
      <tr><th>Inside width (I)</th><td>{inside_width}</td></tr>
      <tr><th>Midline p(x)</th><td>{midline}</td></tr>
      <tr><th>Bid <span class="pin-indicator {pin_bid_class}" title="{pin_bid_title}"></span></th><td>{bid}</td></tr>
      <tr><th>Ask <span class="pin-indicator {pin_ask_class}" title="{pin_ask_title}"></span></th><td>{ask}</td></tr>
    </tbody>
  </table>
  {inventory_html}
</div>
"""


def _render_vbt_card(vbt_dict: dict[str, Any]) -> str:
    """Render VBT state card."""
    bucket_id = vbt_dict.get("bucket_id", "unknown")
    M = _fmt_decimal(vbt_dict.get("M"))
    spread_anchor = _fmt_decimal(vbt_dict.get("O"))
    A = _fmt_decimal(vbt_dict.get("A"))
    B = _fmt_decimal(vbt_dict.get("B"))
    cash = _fmt_decimal(vbt_dict.get("cash"))

    # Inventory
    inventory = vbt_dict.get("inventory", [])
    inv_count = len(inventory)

    return f"""
<div class="dealer-card vbt-card">
  <h3>VBT: {_html_escape(bucket_id.upper())}</h3>
  <table class="params-table">
    <tbody>
      <tr><th>Mid anchor (M)</th><td>{M}</td></tr>
      <tr><th>Spread anchor (O)</th><td>{spread_anchor}</td></tr>
      <tr><th>Ask (A = M + O/2)</th><td>{A}</td></tr>
      <tr><th>Bid (B = M - O/2)</th><td>{B}</td></tr>
      <tr><th>Cash</th><td>{cash}</td></tr>
      <tr><th>Inventory</th><td>{inv_count} tickets</td></tr>
    </tbody>
  </table>
</div>
"""


def _render_trader_balance(trader_dict: dict[str, Any]) -> str:
    """Render trader T-account balance sheet."""
    agent_id = trader_dict.get("agent_id", "unknown")
    cash = trader_dict.get("cash", 0)
    defaulted = trader_dict.get("defaulted", False)

    # Status indicator
    status_badge = ""
    if defaulted:
        status_badge = '<span style="color: #dc2626; font-weight: 600;"> [DEFAULTED]</span>'

    # Build assets side (Cash + Tickets owned)
    tickets_owned = trader_dict.get("tickets_owned", [])
    assets_rows = [
        f"""
      <tr>
        <td class="name">Cash</td>
        <td class="val">{_fmt_decimal(cash, 2)}</td>
      </tr>
    """
    ]

    total_ticket_face = Decimal(0)
    for ticket in tickets_owned:
        face = ticket.get("face", 0)
        if isinstance(face, str):
            face = Decimal(face)
        total_ticket_face += face
        tid = ticket.get("id", "?")
        issuer = _html_escape(ticket.get("issuer_id", "?")[:12])
        _html_escape(ticket.get("bucket_id", "?"))
        tau = ticket.get("remaining_tau", "?")
        assets_rows.append(f"""
      <tr>
        <td class="name">{_html_escape(tid)}</td>
        <td class="val">{_fmt_decimal(face, 2)} <span style="color: #9ca3af; font-size: 12px;">({issuer}, τ={tau})</span></td>
      </tr>
        """)

    if not tickets_owned:
        assets_rows.append("""
      <tr>
        <td colspan="2" class="empty">No tickets owned</td>
      </tr>
        """)

    # Build liabilities side (Obligations)
    obligations = trader_dict.get("obligations", [])
    liabilities_rows = []

    total_obligations = Decimal(0)
    for obligation in obligations:
        face = obligation.get("face", 0)
        if isinstance(face, str):
            face = Decimal(face)
        total_obligations += face
        tid = obligation.get("id", "?")
        owner = _html_escape(obligation.get("owner_id", "?")[:12])
        maturity = obligation.get("maturity_day", "?")
        liabilities_rows.append(f"""
      <tr>
        <td class="name">{_html_escape(tid)}</td>
        <td class="val">{_fmt_decimal(face, 2)} <span style="color: #9ca3af; font-size: 12px;">(to {owner}, mat={maturity})</span></td>
      </tr>
        """)

    if not obligations:
        liabilities_rows.append("""
      <tr>
        <td colspan="2" class="empty">No obligations</td>
      </tr>
        """)

    # Calculate totals
    if isinstance(cash, str):
        cash = Decimal(cash)
    total_assets = cash + total_ticket_face

    return f"""
<div class="t-account">
  <h3>{_html_escape(agent_id)}{status_badge}</h3>
  <div class="grid">
    <div class="assets-side">
      <h4>Assets ({_fmt_decimal(total_assets, 2)})</h4>
      <table class="side">
        <tbody>
          {"".join(assets_rows)}
        </tbody>
      </table>
    </div>
    <div class="liabilities-side">
      <h4>Liabilities ({_fmt_decimal(total_obligations, 2)})</h4>
      <table class="side">
        <tbody>
          {"".join(liabilities_rows)}
        </tbody>
      </table>
    </div>
  </div>
</div>
"""


def _render_day_section(snapshot: "DaySnapshot", is_setup: bool = False) -> str:
    """Render complete day section."""
    day = snapshot.day

    # Day header
    if is_setup:
        day_title = "Day 0 (Setup)"
        day_emoji = "🏁"
    else:
        day_title = f"Day {day}"
        day_emoji = "📅"

    # Events table
    events_html = _render_events_table(snapshot.events, f"{day_title} Events")

    # Dealers and VBTs grid
    dealer_cards = []
    vbt_cards = []
    # Sort by bucket order: short, mid, long
    bucket_order = {"short": 0, "mid": 1, "long": 2}
    sorted_buckets = sorted(snapshot.dealers.keys(), key=lambda x: bucket_order.get(x, 99))

    for bucket_id in sorted_buckets:
        dealer_dict = snapshot.dealers[bucket_id]
        vbt_dict = snapshot.vbts.get(bucket_id, {})
        dealer_cards.append(_render_dealer_card(dealer_dict, vbt_dict))
        vbt_cards.append(_render_vbt_card(vbt_dict))

    # Traders grid
    trader_cards = []
    for agent_id in sorted(snapshot.traders.keys()):
        trader_dict = snapshot.traders[agent_id]
        trader_cards.append(_render_trader_balance(trader_dict))

    # Only show trader section if there are traders
    traders_html = ""
    if trader_cards:
        traders_html = f"""
  <div class="balances-section">
    <h3>Trader Balances</h3>
    <div class="traders-grid">
      {"".join(trader_cards)}
    </div>
  </div>
"""

    return f"""
<section class="day-section">
  <h2 class="day-header">{day_emoji} {day_title}</h2>

  {events_html}

  <div class="balances-section">
    <h3>Market State</h3>
    <div class="dealers-grid">
      {"".join(dealer_cards)}
    </div>
    <div class="dealers-grid">
      {"".join(vbt_cards)}
    </div>
  </div>

  {traders_html}
</section>
"""


def generate_dealer_ring_html(
    snapshots: list["DaySnapshot"],
    config: "DealerRingConfig",
    title: str | None = None,
    subtitle: str | None = None,
) -> str:
    """Generate complete HTML report for dealer ring simulation.

    Args:
        snapshots: List of DaySnapshot objects, one per day (Day 0 = setup)
        config: DealerRingConfig with simulation parameters
        title: Report title (default: "Dealer Ring Simulation")
        subtitle: Report subtitle (default: simulation description)

    Returns:
        Complete HTML document as string
    """
    if title is None:
        title = "Dealer Ring Simulation"
    if subtitle is None:
        subtitle = "Market making with virtual balance table anchors"

    base_css = _load_base_css()
    dealer_css = _get_dealer_css()
    header_html = _render_header(config, snapshots, title, subtitle)

    # Render each day
    day_sections = []
    for i, snapshot in enumerate(snapshots):
        is_setup = i == 0 and snapshot.day == 0
        day_sections.append(_render_day_section(snapshot, is_setup=is_setup))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_html_escape(title)}</title>
  <style>
{base_css}
{dealer_css}
  </style>
</head>
<body>
  <div class="container">
    {header_html}

    <main>
      {"".join(day_sections)}
    </main>

    <footer>
      Generated by bilancio dealer ring simulation
    </footer>
  </div>
</body>
</html>
"""


def export_dealer_ring_html(
    snapshots: list["DaySnapshot"],
    config: "DealerRingConfig",
    path: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    """Export dealer ring simulation to HTML file.

    Args:
        snapshots: List of DaySnapshot objects, one per day
        config: DealerRingConfig with simulation parameters
        path: Output file path
        title: Report title
        subtitle: Report subtitle
    """
    html_content = generate_dealer_ring_html(snapshots, config, title, subtitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")
