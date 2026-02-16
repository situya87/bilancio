"""Visualization module for comparing simulation runs across a job.

This module provides interactive HTML visualizations comparing passive vs active
dealer scenarios, showing how key metrics (delta, phi) vary with parameters
(kappa, concentration, mu).

Uses Plotly for interactive visualizations that can be viewed in any browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RunComparison:
    """A paired comparison between passive and active runs with same parameters."""

    kappa: float
    concentration: float
    mu: float
    seed: int
    delta_passive: float
    delta_active: float
    phi_passive: float
    phi_active: float

    @property
    def trading_effect(self) -> float:
        """Improvement in delta from having dealer (positive = dealer helps)."""
        return self.delta_passive - self.delta_active

    @property
    def trading_relief_pct(self) -> float:
        """Percentage of defaults relieved by dealer."""
        if self.delta_passive == 0:
            return 0.0
        return self.trading_effect / self.delta_passive * 100


def load_job_comparison_data(job_id: str) -> list[RunComparison]:
    """Load and pair passive/active runs from Supabase for a given job.

    Returns a list of RunComparison objects, one per parameter combination.
    """
    from bilancio.storage.supabase_client import get_supabase_client

    client = get_supabase_client()

    # Get runs with parameters
    runs_result = client.table("runs").select("*").eq("job_id", job_id).execute()
    runs: list[dict[str, Any]] = runs_result.data  # type: ignore[assignment]

    # Get metrics for this job
    metrics_result = client.table("metrics").select("*").eq("job_id", job_id).execute()
    metrics: list[dict[str, Any]] = metrics_result.data  # type: ignore[assignment]

    # Build lookup by run_id
    metrics_by_id: dict[str, dict[str, Any]] = {m["run_id"]: m for m in metrics}

    # Group runs by parameter key
    passive_runs: dict[tuple[Any, ...], dict[str, Any]] = {}
    active_runs: dict[tuple[Any, ...], dict[str, Any]] = {}

    for run in runs:
        key = (run["kappa"], run["concentration"], run["mu"], run.get("seed", 42))
        metric = metrics_by_id.get(run["run_id"], {})

        run_data: dict[str, Any] = {
            **run,
            "delta_total": metric.get("delta_total", 0),
            "phi_total": metric.get("phi_total", 0),
        }

        if run["regime"] == "passive":
            passive_runs[key] = run_data
        else:
            active_runs[key] = run_data

    # Pair up runs
    comparisons: list[RunComparison] = []
    for key in passive_runs:
        if key in active_runs:
            p = passive_runs[key]
            a = active_runs[key]
            comparisons.append(
                RunComparison(
                    kappa=float(key[0]),
                    concentration=float(key[1]),
                    mu=float(key[2]),
                    seed=int(key[3]),
                    delta_passive=p["delta_total"],
                    delta_active=a["delta_total"],
                    phi_passive=p["phi_total"],
                    phi_active=a["phi_total"],
                )
            )

    return comparisons


def comparisons_to_dataframe(comparisons: list[RunComparison]) -> pd.DataFrame:
    """Convert comparisons to a pandas DataFrame for easier plotting."""
    import pandas as pd

    data = []
    for c in comparisons:
        data.append(
            {
                "kappa": c.kappa,
                "concentration": c.concentration,
                "mu": c.mu,
                "seed": c.seed,
                "delta_passive": c.delta_passive,
                "delta_active": c.delta_active,
                "phi_passive": c.phi_passive,
                "phi_active": c.phi_active,
                "trading_effect": c.trading_effect,
                "trading_relief_pct": c.trading_relief_pct,
            }
        )

    return pd.DataFrame(data)


def create_passive_vs_active_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot comparing passive vs active delta for each run.

    Points above the diagonal = dealer helped (reduced defaults).
    """
    fig = px.scatter(
        df,
        x="delta_passive",
        y="delta_active",
        color="kappa",
        size="concentration",
        hover_data=["kappa", "concentration", "mu", "trading_effect"],
        title="Default Rate: Passive vs Active Dealer",
        labels={
            "delta_passive": "Delta (Passive/Control)",
            "delta_active": "Delta (Active/Dealer)",
            "kappa": "κ (Liquidity)",
            "concentration": "c (Concentration)",
        },
        color_continuous_scale="Viridis",
    )

    # Add diagonal line (no effect)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "gray", "dash": "dash"},
            name="No Effect Line",
            showlegend=True,
        )
    )

    fig.update_layout(
        xaxis={"range": [0, 1.05]},
        yaxis={"range": [0, 1.05]},
        height=600,
    )

    return fig


def create_trading_effect_heatmap(df: pd.DataFrame, mu_value: float = 0.5) -> go.Figure:
    """Create heatmap showing trading effect across kappa and concentration.

    Filters to a specific mu value to create 2D view.
    """

    # Filter to specific mu
    filtered = df[df["mu"] == mu_value].copy() if mu_value in df["mu"].values else df.copy()

    # Pivot to create matrix
    pivot = filtered.pivot_table(
        values="trading_effect", index="concentration", columns="kappa", aggfunc="mean"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"κ={k}" for k in pivot.columns],
            y=[f"c={c}" for c in pivot.index],
            colorscale="RdYlGn",  # Red=negative (dealer hurt), Green=positive (dealer helped)
            zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="κ=%{x}<br>c=%{y}<br>Trading Effect=%{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Trading Effect (δ_passive - δ_active) at μ={mu_value}",
        xaxis_title="Liquidity Ratio (κ)",
        yaxis_title="Concentration (c)",
        height=500,
    )

    return fig


def create_delta_by_kappa_line(df: pd.DataFrame) -> go.Figure:
    """Create line plot showing delta vs kappa, comparing passive and active."""

    # Average across concentration and mu for each kappa
    avg_by_kappa = (
        df.groupby("kappa")
        .agg(
            {
                "delta_passive": "mean",
                "delta_active": "mean",
            }
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=avg_by_kappa["kappa"],
            y=avg_by_kappa["delta_passive"],
            mode="lines+markers",
            name="Passive (Control)",
            line={"color": "#E74C3C", "width": 3},
            marker={"size": 10},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=avg_by_kappa["kappa"],
            y=avg_by_kappa["delta_active"],
            mode="lines+markers",
            name="Active (Dealer)",
            line={"color": "#27AE60", "width": 3},
            marker={"size": 10},
        )
    )

    # Fill between to show improvement
    fig.add_trace(
        go.Scatter(
            x=list(avg_by_kappa["kappa"]) + list(avg_by_kappa["kappa"][::-1]),
            y=list(avg_by_kappa["delta_passive"]) + list(avg_by_kappa["delta_active"][::-1]),
            fill="toself",
            fillcolor="rgba(39, 174, 96, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name="Dealer Benefit",
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Default Rate by Liquidity Ratio (κ)",
        xaxis_title="Liquidity Ratio (κ)",
        yaxis_title="Default Rate (δ)",
        yaxis={"range": [0, 1.05]},
        height=500,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    return fig


def create_faceted_comparison(df: pd.DataFrame) -> go.Figure:
    """Create faceted view showing delta by kappa for each concentration level."""

    concentrations = sorted(df["concentration"].unique())
    n_facets = len(concentrations)

    fig = make_subplots(
        rows=1,
        cols=n_facets,
        subplot_titles=[f"c = {c}" for c in concentrations],
        shared_yaxes=True,
    )

    colors = {"passive": "#E74C3C", "active": "#27AE60"}

    for i, c in enumerate(concentrations, 1):
        subset = df[df["concentration"] == c]
        avg = (
            subset.groupby("kappa")
            .agg(
                {
                    "delta_passive": "mean",
                    "delta_active": "mean",
                }
            )
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=avg["kappa"],
                y=avg["delta_passive"],
                mode="lines+markers",
                name="Passive" if i == 1 else None,
                legendgroup="passive",
                showlegend=(i == 1),
                line={"color": colors["passive"]},
            ),
            row=1,
            col=i,
        )

        fig.add_trace(
            go.Scatter(
                x=avg["kappa"],
                y=avg["delta_active"],
                mode="lines+markers",
                name="Active" if i == 1 else None,
                legendgroup="active",
                showlegend=(i == 1),
                line={"color": colors["active"]},
            ),
            row=1,
            col=i,
        )

    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(
        title="Default Rate by κ, Faceted by Concentration (c)",
        height=400,
    )

    return fig


def create_parameter_distribution(df: pd.DataFrame) -> go.Figure:
    """Create violin/box plots showing distribution of trading effect by parameter."""

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["By κ (Liquidity)", "By c (Concentration)", "By μ (Maturity)"],
    )

    # By kappa
    for k in sorted(df["kappa"].unique()):
        fig.add_trace(
            go.Box(
                y=df[df["kappa"] == k]["trading_effect"],
                name=f"κ={k}",
                boxmean=True,
            ),
            row=1,
            col=1,
        )

    # By concentration
    for c in sorted(df["concentration"].unique()):
        fig.add_trace(
            go.Box(
                y=df[df["concentration"] == c]["trading_effect"],
                name=f"c={c}",
                boxmean=True,
            ),
            row=1,
            col=2,
        )

    # By mu
    for m in sorted(df["mu"].unique()):
        fig.add_trace(
            go.Box(
                y=df[df["mu"] == m]["trading_effect"],
                name=f"μ={m}",
                boxmean=True,
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        title="Trading Effect Distribution by Parameter",
        height=450,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Trading Effect (δp - δa)")

    return fig


def create_3d_surface(df: pd.DataFrame) -> go.Figure:
    """Create 3D surface plot of trading effect across kappa and concentration.

    Averages across mu values.
    """

    # Pivot and average across mu
    pivot = df.pivot_table(
        values="trading_effect", index="concentration", columns="kappa", aggfunc="mean"
    )

    kappas = pivot.columns.values
    concentrations = pivot.index.values
    z_values = pivot.values

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_values,
                x=kappas,
                y=concentrations,
                colorscale="RdYlGn",
                cmid=0,
                hovertemplate="κ=%{x}<br>c=%{y}<br>Trading Effect=%{z:.4f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Trading Effect Surface (averaged over μ)",
        scene={
            "xaxis_title": "Liquidity Ratio (κ)",
            "yaxis_title": "Concentration (c)",
            "zaxis_title": "Trading Effect",
        },
        height=600,
    )

    return fig


def create_summary_metrics_table(df: pd.DataFrame) -> go.Figure:
    """Create a summary table with key statistics."""

    summary = {
        "Metric": [
            "Total Comparisons",
            "Mean δ (Passive)",
            "Mean δ (Active)",
            "Mean Trading Effect",
            "Comparisons Where Dealer Helped",
            "Max Trading Effect",
            "Min Trading Effect",
        ],
        "Value": [
            len(df),
            f"{df['delta_passive'].mean():.4f}",
            f"{df['delta_active'].mean():.4f}",
            f"{df['trading_effect'].mean():.4f}",
            f"{(df['trading_effect'] > 0).sum()} ({(df['trading_effect'] > 0).mean() * 100:.1f}%)",
            f"{df['trading_effect'].max():.4f}",
            f"{df['trading_effect'].min():.4f}",
        ],
    }

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["<b>Metric</b>", "<b>Value</b>"],
                    "fill_color": "#3498DB",
                    "font": {"color": "white", "size": 14},
                    "align": "left",
                },
                cells={
                    "values": [summary["Metric"], summary["Value"]],
                    "fill_color": ["#ECF0F1", "#ECF0F1"],
                    "align": "left",
                    "font": {"size": 13},
                    "height": 30,
                },
            )
        ]
    )

    fig.update_layout(
        title="Summary Statistics",
        height=350,
    )

    return fig


def create_all_runs_overview(df: pd.DataFrame) -> go.Figure:
    """Create an overview showing all runs as a parallel coordinates plot."""

    fig = px.parallel_coordinates(
        df,
        dimensions=[
            "kappa",
            "concentration",
            "mu",
            "delta_passive",
            "delta_active",
            "trading_effect",
        ],
        color="trading_effect",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={
            "kappa": "κ (Liquidity)",
            "concentration": "c (Concentration)",
            "mu": "μ (Maturity)",
            "delta_passive": "δ Passive",
            "delta_active": "δ Active",
            "trading_effect": "Trading Effect",
        },
        title="All Runs Overview (Parallel Coordinates)",
    )

    fig.update_layout(height=500)

    return fig


def generate_comparison_html(
    job_id: str,
    output_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """Generate a complete HTML visualization report for a job.

    Args:
        job_id: The job ID to visualize
        output_path: Where to save the HTML file. Defaults to temp/{job_id}_comparison.html
        title: Optional custom title for the report

    Returns:
        Path to the generated HTML file
    """

    # Load data
    comparisons = load_job_comparison_data(job_id)
    if not comparisons:
        raise ValueError(f"No comparison data found for job {job_id}")

    df = comparisons_to_dataframe(comparisons)

    # Set output path
    if output_path is None:
        output_path = Path("temp") / f"{job_id}_comparison.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    figures = [
        ("summary", create_summary_metrics_table(df)),
        ("scatter", create_passive_vs_active_scatter(df)),
        ("line", create_delta_by_kappa_line(df)),
        ("faceted", create_faceted_comparison(df)),
        ("distribution", create_parameter_distribution(df)),
        ("heatmap_mu0", create_trading_effect_heatmap(df, mu_value=0.0)),
        ("heatmap_mu05", create_trading_effect_heatmap(df, mu_value=0.5)),
        ("heatmap_mu1", create_trading_effect_heatmap(df, mu_value=1.0)),
        ("parallel", create_all_runs_overview(df)),
        ("surface", create_3d_surface(df)),
    ]

    # Build HTML
    report_title = title or f"Run Comparison: {job_id}"

    html_parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report_title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            padding: 20px;
        }}
        .description {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        .nav {{
            position: sticky;
            top: 0;
            background: #2c3e50;
            padding: 10px 20px;
            margin: -20px -20px 20px -20px;
            z-index: 100;
        }}
        .nav a {{
            color: white;
            text-decoration: none;
            margin-right: 20px;
            font-size: 14px;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
        .metric-explanation {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metric-explanation h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metric-explanation ul {{
            margin-bottom: 0;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#summary">Summary</a>
        <a href="#scatter">Passive vs Active</a>
        <a href="#line">By Liquidity</a>
        <a href="#faceted">Faceted View</a>
        <a href="#distribution">Distributions</a>
        <a href="#heatmaps">Heatmaps</a>
        <a href="#overview">Full Overview</a>
        <a href="#surface">3D Surface</a>
    </div>

    <h1>{report_title}</h1>

    <div class="metric-explanation">
        <h3>Key Metrics</h3>
        <ul>
            <li><b>δ (Delta)</b>: Default rate - fraction of debt not settled on time (lower is better)</li>
            <li><b>φ (Phi)</b>: Clearing rate - fraction of debt settled on due day (higher is better)</li>
            <li><b>Trading Effect</b>: δ_passive - δ_active (positive = dealer reduced defaults)</li>
            <li><b>κ (Kappa)</b>: Liquidity ratio L₀/S₁ (higher = more liquid system)</li>
            <li><b>c (Concentration)</b>: Debt distribution inequality (lower = more unequal)</li>
            <li><b>μ (Mu)</b>: Maturity timing skew (0 = early, 1 = late due dates)</li>
        </ul>
    </div>
"""
    ]

    section_descriptions = {
        "summary": "Overview of key statistics across all simulation runs.",
        "scatter": "Each point is a parameter combination. Points below the diagonal indicate the dealer reduced defaults.",
        "line": "How default rates change with liquidity ratio, averaged across other parameters. Green area shows dealer benefit.",
        "faceted": "Default rates by liquidity ratio, separated by concentration level.",
        "distribution": "Distribution of trading effects (dealer benefit) across different parameter values.",
        "heatmap_mu0": "Trading effect heatmap for early maturity (μ=0).",
        "heatmap_mu05": "Trading effect heatmap for balanced maturity (μ=0.5).",
        "heatmap_mu1": "Trading effect heatmap for late maturity (μ=1).",
        "parallel": "Parallel coordinates plot showing all runs and their parameters/outcomes.",
        "surface": "3D surface showing how trading effect varies with κ and c (averaged over μ).",
    }

    section_titles = {
        "summary": "Summary Statistics",
        "scatter": "Passive vs Active Comparison",
        "line": "Default Rate by Liquidity",
        "faceted": "Faceted by Concentration",
        "distribution": "Trading Effect Distributions",
        "heatmap_mu0": "Heatmap (μ=0, Early Maturity)",
        "heatmap_mu05": "Heatmap (μ=0.5, Balanced)",
        "heatmap_mu1": "Heatmap (μ=1, Late Maturity)",
        "parallel": "All Runs Overview",
        "surface": "3D Surface Plot",
    }

    for name, fig in figures:
        anchor = name.split("_")[0] if name.startswith("heatmap") else name
        if name == "heatmap_mu0":
            anchor = "heatmaps"

        # Get the plotly div
        div_html = fig.to_html(full_html=False, include_plotlyjs=False)

        html_parts.append(f'''
    <h2 id="{anchor}">{section_titles.get(name, name.title())}</h2>
    <div class="chart-container">
        <p class="description">{section_descriptions.get(name, "")}</p>
        {div_html}
    </div>
''')

    html_parts.append("""
    <hr style="margin-top: 40px;">
    <p style="color: #7f8c8d; font-size: 12px;">
        Generated by bilancio run comparison visualization
    </p>
</body>
</html>
""")

    # Write file
    output_path.write_text("\n".join(html_parts))

    return output_path


def quick_visualize(job_id: str) -> None:
    """Quick visualization that generates HTML and prints the path.

    Convenience function for interactive use.
    """
    path = generate_comparison_html(job_id)
    print(f"Visualization saved to: {path}")
    print(f"Open with: open {path}")
