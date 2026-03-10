"""Per-run convergence dashboard (self-contained HTML).

Generates an interactive HTML page showing convergence trajectories
for each monitored channel, annotated with convergence bands and
the composite convergence point.

Usage::

    from bilancio.analysis.convergence import evaluate_convergence
    from bilancio.analysis.convergence_dashboard import generate_convergence_dashboard

    result = evaluate_convergence(day_metrics, events, estimates)
    generate_convergence_dashboard(result, output_path="convergence.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.analysis.convergence import ConvergenceResult


# Channel display configuration
_CHANNEL_COLORS: dict[str, str] = {
    "clearing": "#2196F3",
    "default": "#F44336",
    "price": "#FF9800",
    "belief": "#9C27B0",
    "credit": "#4CAF50",
    "contagion": "#795548",
}

_CHANNEL_LABELS: dict[str, str] = {
    "clearing": "Clearing Rate (φ_t)",
    "default": "Default Rate (δ_t)",
    "price": "Average Price Ratio",
    "belief": "Mean Belief (P_default)",
    "credit": "Net Credit Impulse",
    "contagion": "Secondary Defaults",
}


def _build_channel_chart(
    name: str,
    channel: "ChannelResult",
    epsilon: float,
) -> str:
    """Build a Plotly HTML snippet for one channel."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return f"<p>Plotly not installed — cannot render {name} chart.</p>"

    from bilancio.analysis.convergence import ChannelResult  # noqa: F811

    label = _CHANNEL_LABELS.get(name, name.title())
    color = _CHANNEL_COLORS.get(name, "#607D8B")

    days = [s.day for s in channel.trajectory]
    values = [s.value for s in channel.trajectory]

    fig = go.Figure()

    # Main trajectory
    fig.add_trace(go.Scatter(
        x=days, y=values, mode="lines+markers",
        name=label, line=dict(color=color, width=2),
        marker=dict(size=4),
    ))

    # Convergence band (if converged)
    if channel.converged and channel.convergence_day is not None:
        conv_idx = next(
            (i for i, s in enumerate(channel.trajectory) if s.day >= channel.convergence_day),
            None,
        )
        if conv_idx is not None:
            conv_value = channel.trajectory[conv_idx].value
            fig.add_hrect(
                y0=conv_value - epsilon, y1=conv_value + epsilon,
                line_width=0, fillcolor="rgba(76, 175, 80, 0.15)",
                annotation_text="ε-band", annotation_position="top right",
            )
            fig.add_vline(
                x=channel.convergence_day, line_dash="dash",
                line_color="green", opacity=0.7,
                annotation_text=f"Conv. day {channel.convergence_day}",
            )

    status = "Converged" if channel.converged else "Not converged"
    day_str = str(channel.convergence_day) if channel.convergence_day is not None else "—"

    fig.update_layout(
        title=f"{label} — {status} (day {day_str})",
        xaxis_title="Day", yaxis_title="Value",
        height=300, margin=dict(t=50, b=40, l=60, r=20),
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_composite_timeline(result: "ConvergenceResult") -> str:
    """Build a horizontal bar chart showing per-channel convergence."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not installed — cannot render timeline.</p>"

    names = []
    conv_days = []
    colors = []

    for name in sorted(result.channels.keys()):
        ch = result.channels[name]
        label = _CHANNEL_LABELS.get(name, name.title())
        names.append(label)
        conv_days.append(ch.convergence_day if ch.convergence_day is not None else 0)
        colors.append("#4CAF50" if ch.converged else "#9e9e9e")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=conv_days, orientation="h",
        marker_color=colors,
        text=[f"Day {d}" if d > 0 else "—" for d in conv_days],
        textposition="auto",
    ))

    if result.convergence_day is not None:
        fig.add_vline(
            x=result.convergence_day, line_dash="dash",
            line_color="#2196F3", line_width=2,
            annotation_text=f"Composite: day {result.convergence_day}",
        )

    fig.update_layout(
        title="Channel Convergence Timeline",
        xaxis_title="Convergence Day",
        height=max(200, 60 * len(names) + 80),
        margin=dict(t=50, b=40, l=180, r=20),
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_convergence_dashboard(
    result: "ConvergenceResult",
    *,
    output_path: Path | str = "convergence.html",
    title: str | None = None,
    config: "ConvergenceConfig | None" = None,
) -> Path:
    """Generate a self-contained HTML convergence dashboard.

    Args:
        result: The :class:`~bilancio.analysis.convergence.ConvergenceResult`
            to visualize.
        output_path: Where to write the HTML file.
        title: Optional title for the dashboard header.
        config: The :class:`~bilancio.analysis.convergence.ConvergenceConfig`
            used for the convergence evaluation (needed for epsilon values).
            If ``None``, defaults are used.

    Returns:
        Path to the generated HTML file.
    """
    from bilancio.analysis.convergence import ConvergenceConfig, _CHANNEL_EPSILONS

    if config is None:
        config = ConvergenceConfig()
    output_path = Path(output_path)

    title = title or "Convergence Dashboard"
    status_icon = "&#10003;" if result.converged else "&#10007;"
    status_color = "#4CAF50" if result.converged else "#F44336"
    day_str = str(result.convergence_day) if result.convergence_day is not None else "—"
    quality_pct = f"{result.quality * 100:.1f}%"

    # Build channel charts
    channel_html_parts: list[str] = []
    for name in ["clearing", "default", "price", "belief", "credit", "contagion"]:
        if name not in result.channels:
            continue
        ch = result.channels[name]
        eps_attr = _CHANNEL_EPSILONS.get(name, "epsilon_default")
        epsilon = getattr(config, eps_attr, config.epsilon_default)
        channel_html_parts.append(_build_channel_chart(name, ch, epsilon))

    # Build composite timeline
    timeline_html = _build_composite_timeline(result)

    # Assemble HTML
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 900px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  .header {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .header h1 {{ margin: 0 0 12px 0; font-size: 1.5em; }}
  .summary {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .summary-item {{ font-size: 0.95em; }}
  .summary-item .label {{ color: #666; }}
  .summary-item .value {{ font-weight: 600; }}
  .chart-section {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <div class="summary">
    <div class="summary-item">
      <span class="label">Status:</span>
      <span class="value" style="color: {status_color};">{status_icon} {"Converged" if result.converged else "Not Converged"}</span>
    </div>
    <div class="summary-item">
      <span class="label">Convergence Day:</span>
      <span class="value">{day_str}</span>
    </div>
    <div class="summary-item">
      <span class="label">Quality:</span>
      <span class="value">{quality_pct}</span>
    </div>
    <div class="summary-item">
      <span class="label">Channels:</span>
      <span class="value">{result.converged_channels}/{result.active_channels} converged</span>
    </div>
  </div>
</div>

{"".join(f'<div class="chart-section">{ch}</div>' for ch in channel_html_parts)}

<div class="chart-section">
{timeline_html}
</div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
