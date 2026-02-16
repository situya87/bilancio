"""Phase-aware event visualization for bilancio."""

from typing import Any

try:
    from rich.text import Text

    _RICH_TEXT_AVAILABLE = True
except ImportError:
    _RICH_TEXT_AVAILABLE = False

RenderableType = Any  # Type alias for renderables


def build_events_detailed_with_phases(
    events: list[dict[str, Any]], RICH_AVAILABLE: bool = True
) -> list[RenderableType]:
    """Build renderables for events in detailed format, properly organized by phase markers."""
    renderables: list[RenderableType] = []

    # Use the formatter registry to format events nicely
    from bilancio.ui.render.formatters import registry

    # Group events by phase markers (PhaseA, PhaseB, PhaseC)
    # Events between PhaseA and PhaseB are in Phase A
    # Events between PhaseB and PhaseC are in Phase B
    # Events after PhaseC are in Phase C
    phase_a_events = []
    phase_b_events = []
    phase_c_events = []
    setup_events = []

    current_phase = None
    for event in events:
        kind = event.get("kind", "Unknown")

        # Check for phase markers to update current phase
        if kind == "PhaseA":
            current_phase = "A"
            continue  # Skip the marker itself
        elif kind == "PhaseB":
            current_phase = "B"
            continue  # Skip the marker itself
        elif kind == "PhaseC":
            current_phase = "C"
            continue  # Skip the marker itself

        # Sort events into phases based on current phase
        if event.get("phase") == "setup":
            setup_events.append(event)
        elif current_phase == "A":
            phase_a_events.append(event)
        elif current_phase == "B":
            phase_b_events.append(event)
        elif current_phase == "C":
            phase_c_events.append(event)
        else:
            # Events before any phase marker (during simulation but no phase set yet)
            # This shouldn't happen but default to phase A
            if event.get("phase") == "simulation":
                phase_a_events.append(event)

    # Display setup events if any
    if setup_events:
        if RICH_AVAILABLE:
            phase_header = Text("Setup", style="bold magenta")
            renderables.append(phase_header)
        else:
            renderables.append("Setup")

        for event in setup_events:
            renderables.extend(_format_single_event(event, registry, RICH_AVAILABLE))

    # Display Phase A events (usually empty as it's just a marker)
    if phase_a_events:
        if RICH_AVAILABLE:
            phase_header = Text("\nPhase A", style="bold cyan")
            renderables.append(phase_header)
        else:
            renderables.append("\nPhase A")

        for event in phase_a_events:
            renderables.extend(_format_single_event(event, registry, RICH_AVAILABLE))

    # Display Phase B events - Settle obligations due
    if phase_b_events:
        if RICH_AVAILABLE:
            phase_header = Text("\nPhase B - Settle obligations due", style="bold yellow")
            renderables.append(phase_header)
        else:
            renderables.append("\nPhase B - Settle obligations due")

        for event in phase_b_events:
            renderables.extend(_format_single_event(event, registry, RICH_AVAILABLE))

    # Display Phase C events - Clear intraday nets
    if phase_c_events:
        if RICH_AVAILABLE:
            phase_header = Text("\nPhase C - Clear intraday nets", style="bold green")
            renderables.append(phase_header)
        else:
            renderables.append("\nPhase C - Clear intraday nets")

        for event in phase_c_events:
            renderables.extend(_format_single_event(event, registry, RICH_AVAILABLE))

    return renderables


def _format_single_event(
    event: dict[str, Any], registry: Any, RICH_AVAILABLE: bool
) -> list[RenderableType]:
    """Format a single event and return renderables."""
    renderables: list[RenderableType] = []

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
            for _i, line in enumerate(lines[:3]):  # Show up to 3 lines
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
