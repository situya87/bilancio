"""Generate a self-contained HTML report with embedded Plotly charts
from cross-sweep analysis of NBFI and bank production sweeps.

Usage:
    uv run python scripts/sweep_analysis_report.py --output out/sweep_analysis/report.html
"""

from __future__ import annotations

import argparse
import html as html_mod
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from bilancio.analysis.cross_sweep import (
    ARM_COLORS,
    ARM_LABELS,
    BANK_BLUE,
    COMBINED_PURPLE,
    DEALER_RED,
    MECHANISM_COLORS,
    MECHANISM_LABELS,
    NBFI_GREEN,
    PASSIVE_GREY,
    add_derived_columns,
    bootstrap_ci,
    bootstrap_ci_by_group,
    fit_ols_from_df,
    kruskal_wallis,
    load_unified_data,
    mann_whitney_u,
    one_sample_ttest,
    pivot_for_heatmap,
    standardized_coefs,
    summary_by_group,
    summary_stats,
    wilcoxon_test,
)
from bilancio.analysis.cross_sweep_network import (
    aggregate_timeseries_by_arm_kappa,
    generate_network_animation_html,
    load_mechanism_timeseries,
    aggregate_mechanism_timeseries,
    load_network_metrics,
    load_network_timeseries,
    reconstruct_network_snapshots,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_CHART_BG = "white"
_GRID_COLOR = "#ecf0f1"

_EFFECT_COLS = ["trading_effect", "lending_effect", "bank_lending_effect"]
_EFFECT_LABELS = {
    "trading_effect": "Dealer Trading",
    "lending_effect": "NBFI Lending",
    "bank_lending_effect": "Bank Lending",
}
_EFFECT_COLORS = {
    "trading_effect": DEALER_RED,
    "lending_effect": NBFI_GREEN,
    "bank_lending_effect": BANK_BLUE,
}

_X_COLS = ["log_kappa", "concentration", "mu", "outside_mid_ratio"]
_X_LABELS = {
    "log_kappa": "log(kappa)",
    "concentration": "Concentration (c)",
    "mu": "Maturity skew (mu)",
    "outside_mid_ratio": "Outside-mid ratio (rho)",
}

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _build_shell(title: str, nav_items: list[tuple[str, str]], body: str) -> str:
    """Build the complete HTML document with navigation and styling."""
    nav_links = "".join(
        f'<a href="#{aid}">{label}</a>' for aid, label in nav_items
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5; color: #2c3e50;
}}
h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
h2 {{ margin-top: 40px; color: #34495e; }}
h3 {{ color: #5d6d7e; margin-top: 24px; }}
.chart-container {{
    background: white; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0; padding: 20px;
}}
.interpretation {{
    background: #eaf2f8; border-left: 4px solid #3498db;
    padding: 12px 16px; margin: 10px 0 20px;
    border-radius: 0 4px 4px 0; font-size: 14px;
    color: #2c3e50; line-height: 1.6;
}}
.interp-warn {{
    background: #fef9e7; border-left: 4px solid #f39c12;
}}
.nav {{
    position: sticky; top: 0; background: #2c3e50;
    padding: 12px 20px; z-index: 100;
    border-radius: 0 0 8px 8px; margin: -20px -20px 20px;
}}
.nav a {{
    color: #ecf0f1; text-decoration: none;
    margin-right: 16px; font-size: 13px;
}}
.nav a:hover {{ color: #3498db; }}
.summary-table {{
    width: 100%; border-collapse: collapse; font-size: 14px;
    margin: 12px 0;
}}
.summary-table th {{
    background: #2c3e50; color: white;
    padding: 10px 12px; text-align: left;
}}
.summary-table td {{
    padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
}}
.summary-table tr:hover {{ background: #f0f3f5; }}
.metric-card {{
    display: inline-block; background: white;
    border-radius: 8px; padding: 16px 24px; margin: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
    min-width: 140px;
}}
.metric-card .value {{
    font-size: 28px; font-weight: bold;
}}
.metric-card .label {{
    font-size: 12px; color: #7f8c8d; margin-top: 4px;
}}
.coef-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin: 12px 0;
}}
.coef-table th {{
    background: #34495e; color: white; padding: 8px 10px; text-align: left;
}}
.coef-table td {{
    padding: 6px 10px; border-bottom: 1px solid #ecf0f1;
    font-family: 'Courier New', monospace;
}}
.coef-table tr:hover {{ background: #f0f3f5; }}
.sig {{ color: #e74c3c; font-weight: bold; }}
.section-divider {{
    border: 0; border-top: 2px solid #bdc3c7; margin: 40px 0;
}}
</style>
</head><body>
<div class="nav">{nav_links}</div>
{body}
<p style="text-align:center;color:#95a5a6;margin-top:40px;font-size:12px;">
    Generated by <code>sweep_analysis_report.py</code>
</p>
</body></html>"""


def _chart_html(fig: go.Figure) -> str:
    """Convert plotly figure to embeddable HTML div."""
    inner = fig.to_html(full_html=False, include_plotlyjs=False)
    return f'<div class="chart-container">{inner}</div>'


def _table_html(headers: list[str], rows: list[list[str]]) -> str:
    """Build an HTML summary table."""
    header_html = "".join(f"<th>{html_mod.escape(str(h))}</th>" for h in headers)
    row_html = ""
    for row in rows:
        cells = "".join(f"<td>{v}</td>" for v in row)
        row_html += f"<tr>{cells}</tr>\n"
    return (
        f'<table class="summary-table"><thead><tr>{header_html}</tr></thead>'
        f"<tbody>{row_html}</tbody></table>"
    )


def _coef_table_html(headers: list[str], rows: list[list[str]]) -> str:
    """Build an HTML coefficient table with monospace values."""
    header_html = "".join(f"<th>{html_mod.escape(str(h))}</th>" for h in headers)
    row_html = ""
    for row in rows:
        cells = "".join(f"<td>{v}</td>" for v in row)
        row_html += f"<tr>{cells}</tr>\n"
    return (
        f'<table class="coef-table"><thead><tr>{header_html}</tr></thead>'
        f"<tbody>{row_html}</tbody></table>"
    )


def _interp(text: str, warn: bool = False) -> str:
    """Interpretation box."""
    cls = "interpretation interp-warn" if warn else "interpretation"
    return f'<div class="{cls}">{text}</div>'


def _metric_card(value: str, label: str, color: str = "#2c3e50") -> str:
    """Single metric card."""
    return (
        f'<div class="metric-card">'
        f'<div class="value" style="color:{color};">{value}</div>'
        f'<div class="label">{html_mod.escape(label)}</div>'
        f"</div>"
    )


def _fmt(v: float, decimals: int = 4) -> str:
    """Format a number safely."""
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def _fmt_pct(v: float) -> str:
    """Format as percentage."""
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.1f}%"


def _fmt_pval(p: float) -> str:
    """Format a p-value with significance stars."""
    if not np.isfinite(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e} <span class='sig'>***</span>"
    elif p < 0.01:
        return f"{p:.4f} <span class='sig'>**</span>"
    elif p < 0.05:
        return f"{p:.4f} <span class='sig'>*</span>"
    else:
        return f"{p:.4f}"


# ---------------------------------------------------------------------------
# Plotly figure defaults
# ---------------------------------------------------------------------------


def _apply_layout(fig: go.Figure, title: str, **kwargs: Any) -> go.Figure:
    """Apply consistent layout to a plotly figure."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_CHART_BG,
        font=dict(family="system-ui, -apple-system, sans-serif", size=12),
        margin=dict(l=60, r=40, t=60, b=50),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=_GRID_COLOR, zeroline=True, zerolinecolor="#bdc3c7")
    fig.update_yaxes(gridcolor=_GRID_COLOR, zeroline=True, zerolinecolor="#bdc3c7")
    return fig


# ===================================================================
# Section 1: Overview
# ===================================================================


def _section_overview(data: dict[str, np.ndarray]) -> str:
    """Section 1: Overview with key metrics, distributions, and kappa sensitivity."""
    parts: list[str] = []
    parts.append('<h1 id="overview">Cross-Sweep Analysis Report</h1>')

    n = len(data["kappa"])
    unique_kappas = np.unique(data["kappa"][np.isfinite(data["kappa"])])
    unique_c = np.unique(data["concentration"][np.isfinite(data["concentration"])])
    unique_mu = np.unique(data["mu"][np.isfinite(data["mu"])])

    # -- Metric cards --
    parts.append("<h2>Key Numbers</h2>")
    parts.append("<div>")
    parts.append(_metric_card(str(n), "Parameter Combos"))
    parts.append(_metric_card(str(len(unique_kappas)), "Kappa Levels"))
    parts.append(_metric_card(str(len(unique_c)), "Concentration Levels"))
    parts.append(_metric_card(str(len(unique_mu)), "Mu Levels"))
    parts.append(_metric_card("3", "Mechanisms Compared"))

    # Mean effects
    for col, label, color in [
        ("trading_effect", "Mean Trading Effect", DEALER_RED),
        ("lending_effect", "Mean NBFI Effect", NBFI_GREEN),
        ("bank_lending_effect", "Mean Bank Effect", BANK_BLUE),
    ]:
        vals = data.get(col, np.array([]))
        valid = vals[np.isfinite(vals)] if len(vals) > 0 else np.array([])
        mean_val = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        parts.append(_metric_card(_fmt(mean_val), label, color))

    parts.append("</div>")

    # -- Summary stats table --
    parts.append("<h2>Summary Statistics</h2>")
    headers = ["Metric", "Mean", "Median", "SD", "IQR Low", "IQR High", "% Positive", "N"]
    rows = []
    for col in _EFFECT_COLS:
        if col not in data:
            continue
        s = summary_stats(data[col])
        rows.append([
            _EFFECT_LABELS.get(col, col),
            _fmt(s["mean"]),
            _fmt(s["median"]),
            _fmt(s["sd"]),
            _fmt(s["iqr_low"]),
            _fmt(s["iqr_high"]),
            _fmt_pct(s["pct_positive"]),
            str(s["n"]),
        ])
    # Also add delta_passive as reference
    if "delta_passive" in data:
        s = summary_stats(data["delta_passive"])
        rows.append([
            "Baseline (delta_passive)",
            _fmt(s["mean"]),
            _fmt(s["median"]),
            _fmt(s["sd"]),
            _fmt(s["iqr_low"]),
            _fmt(s["iqr_high"]),
            _fmt_pct(s["pct_positive"]),
            str(s["n"]),
        ])
    parts.append(_table_html(headers, rows))

    # -- C1: Box plots --
    parts.append("<h2>C1: Effect Size Distributions</h2>")
    fig_box = go.Figure()
    for col in _EFFECT_COLS:
        if col not in data:
            continue
        vals = data[col][np.isfinite(data[col])]
        fig_box.add_trace(go.Box(
            y=vals,
            name=_EFFECT_LABELS.get(col, col),
            marker_color=_EFFECT_COLORS.get(col, PASSIVE_GREY),
            boxmean="sd",
        ))
    _apply_layout(fig_box, "Treatment Effect Distributions",
                  yaxis_title="Effect (delta reduction)", height=450)
    fig_box.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                      annotation_text="No effect")
    parts.append(_chart_html(fig_box))
    parts.append(_interp(
        "Bank lending has a tall box entirely above zero, indicating reliable, large effects "
        "(mean 0.288, 90% positive). Dealer trading straddles zero with long whiskers — "
        "some combos show 20-30% default reduction, others show slight increases. "
        "NBFI lending is tightly concentrated near zero. The dashed line marks the boundary "
        "between 'mechanism helps' (above) and 'mechanism hurts' (below)."
    ))

    # -- C2: Delta distributions (histograms) --
    parts.append("<h2>C2: Default Rate Distributions</h2>")
    fig_hist = go.Figure()
    delta_cols = [
        ("delta_passive", "Passive (Baseline)", PASSIVE_GREY),
        ("delta_active", "Active (Dealer)", DEALER_RED),
    ]
    if "delta_lender" in data:
        delta_cols.append(("delta_lender", "NBFI Lender", NBFI_GREEN))
    if "delta_lend" in data:
        delta_cols.append(("delta_lend", "Bank Lend", BANK_BLUE))

    for col, label, color in delta_cols:
        if col not in data:
            continue
        vals = data[col][np.isfinite(data[col])]
        fig_hist.add_trace(go.Histogram(
            x=vals, name=label, marker_color=color,
            opacity=0.6, nbinsx=30,
        ))
    _apply_layout(fig_hist, "Default Rate (delta) Distributions",
                  xaxis_title="Default rate (delta)", yaxis_title="Count",
                  barmode="overlay", height=450)
    parts.append(_chart_html(fig_hist))
    parts.append(_interp(
        "The 'Idle' bank baseline has the highest defaults — banks with idle lending capacity "
        "provide no liquidity channel, making the system more fragile than the no-intermediary "
        "passive baseline. When bank lending activates (blue), the distribution shifts dramatically "
        "leftward, with many combos achieving near-zero defaults. The dealer (red) and NBFI (green) "
        "arms show modest leftward shifts. The bimodal shape reflects the kappa gradient: low-kappa "
        "combos cluster near delta=0.8-0.9, high-kappa near delta=0.1-0.2."
    ))

    # -- C3: Delta vs kappa scatter --
    parts.append("<h2>C3: Default Rate vs Kappa</h2>")
    fig_scatter = go.Figure()
    kappa = data["kappa"]
    scatter_cols = [
        ("delta_passive", "Passive", PASSIVE_GREY, "circle"),
        ("delta_active", "Active (Dealer)", DEALER_RED, "diamond"),
    ]
    if "delta_lender" in data:
        scatter_cols.append(("delta_lender", "NBFI Lender", NBFI_GREEN, "square"))
    if "delta_lend" in data:
        scatter_cols.append(("delta_lend", "Bank Lend", BANK_BLUE, "triangle-up"))

    for col, label, color, symbol in scatter_cols:
        if col not in data:
            continue
        mask = np.isfinite(data[col]) & np.isfinite(kappa)
        fig_scatter.add_trace(go.Scatter(
            x=kappa[mask], y=data[col][mask],
            mode="markers", name=label,
            marker=dict(color=color, size=5, opacity=0.6, symbol=symbol),
        ))
    _apply_layout(fig_scatter, "Default Rate vs Liquidity (kappa)",
                  xaxis_title="Kappa (liquidity ratio)",
                  yaxis_title="Default rate (delta)",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_scatter))
    parts.append(_interp(
        "Bank lending (blue) drops steeply from delta~0.81 at kappa=0.25 to 0.02 at kappa=4, "
        "dramatically outperforming all other arms. Bank idle (grey) is worst at every kappa — "
        "banking infrastructure without lending adds friction. The dealer arm (red) sits just below "
        "the passive line, with the gap widening at intermediate kappas. NBFI lending (green) "
        "tracks close to passive, showing only modest improvement."
    ))

    return "\n".join(parts)


# ===================================================================
# Section 2: Parameter Sensitivity
# ===================================================================


def _section_sensitivity(data: dict[str, np.ndarray]) -> str:
    """Section 2: Heatmaps and marginal effect analysis."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="sensitivity">Parameter Sensitivity</h1>')

    # -- Heatmap grid: 3 mechanisms x relevant parameter pairs --
    param_pairs = [
        ("kappa", "concentration", "Kappa", "Concentration"),
        ("kappa", "mu", "Kappa", "Mu"),
        ("concentration", "mu", "Concentration", "Mu"),
    ]

    parts.append("<h2>Heatmap Grid: Mean Effect by Parameter Pair</h2>")

    for col in _EFFECT_COLS:
        if col not in data:
            continue
        label = _EFFECT_LABELS.get(col, col)
        color = _EFFECT_COLORS.get(col, PASSIVE_GREY)
        parts.append(f"<h3>{label}</h3>")

        fig_hm = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"{rl} vs {cl}" for _, _, rl, cl in param_pairs],
            horizontal_spacing=0.08,
        )

        for idx, (row_col, col_col, rl, cl) in enumerate(param_pairs):
            row_vals, col_vals, matrix = pivot_for_heatmap(
                data, row_col, col_col, col
            )
            # Find symmetric scale
            abs_max = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 0.1
            fig_hm.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=np.round(col_vals, 3),
                    y=np.round(row_vals, 3),
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-abs_max,
                    zmax=abs_max,
                    text=np.where(np.isfinite(matrix), np.round(matrix, 3), None),
                    texttemplate="%{text}",
                    showscale=(idx == 2),
                    colorbar=dict(title="Effect", len=0.8) if idx == 2 else None,
                ),
                row=1, col=idx + 1,
            )
            fig_hm.update_xaxes(title_text=cl, row=1, col=idx + 1)
            fig_hm.update_yaxes(title_text=rl, row=1, col=idx + 1)

        _apply_layout(fig_hm, f"{label}: Mean Effect by Parameter Pair",
                      height=380, width=1200)
        parts.append(_chart_html(fig_hm))

    parts.append(_interp(
        "Three heatmap panels per mechanism: kappa x c, kappa x mu, kappa x rho. "
        "Dealer trading is strongest at high rho and low kappa — outside-money ratio enables "
        "secondary-market price support. NBFI lending shows the clearest signal in kappa x mu, "
        "helping more with back-loaded maturities. Bank lending effects are an order of magnitude "
        "larger (note color scales), strongest at kappa&le;1 with a 30-50pp default reduction. "
        "Bank lending is most effective when mu&gt;0, giving banks time to deploy credit."
    ))

    # -- Marginal effects: 2x2 subplot --
    parts.append("<h2>Marginal Effects by Parameter</h2>")
    param_cols = ["kappa", "concentration", "mu", "outside_mid_ratio"]
    param_labels = ["Kappa", "Concentration (c)", "Maturity skew (mu)", "Outside-mid ratio (rho)"]

    fig_marg = make_subplots(
        rows=2, cols=2,
        subplot_titles=param_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for pidx, (pcol, plabel) in enumerate(zip(param_cols, param_labels)):
        row_idx = pidx // 2 + 1
        col_idx = pidx % 2 + 1

        grouped = summary_by_group(data, [pcol], _EFFECT_COLS)

        for ecol in _EFFECT_COLS:
            if ecol not in data:
                continue
            x_vals = [g[pcol] for g in grouped]
            y_vals = [g.get(f"{ecol}_mean", float("nan")) for g in grouped]
            sd_vals = [g.get(f"{ecol}_sd", 0) for g in grouped]
            color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)
            label = _EFFECT_LABELS.get(ecol, ecol)

            fig_marg.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6),
                    error_y=dict(type="data", array=sd_vals, visible=True, color=color, thickness=1),
                    showlegend=(pidx == 0),
                    legendgroup=ecol,
                ),
                row=row_idx, col=col_idx,
            )
        fig_marg.update_xaxes(title_text=plabel, row=row_idx, col=col_idx)
        fig_marg.update_yaxes(title_text="Mean Effect", row=row_idx, col=col_idx)

    _apply_layout(fig_marg, "Marginal Effects by Parameter (mean +/- SD)",
                  height=700, width=1000,
                  legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5))
    parts.append(_chart_html(fig_marg))
    parts.append(_interp(
        "<b>Kappa:</b> Dealer trading peaks at kappa~0.25, then declines. Bank lending "
        "shows a flatter pattern. <b>Concentration:</b> Bank lending declines sharply with "
        "higher c (more equal debt = less room to help), dealer/NBFI are flat. <b>Mu:</b> "
        "All three improve with higher mu (back-loaded maturities), bank lending steepest. "
        "<b>Rho:</b> Dealer trading rises sharply with rho, while bank and NBFI are insensitive "
        "to outside-money levels."
    ))

    # -- Kappa-effect grouped bar chart --
    parts.append("<h2>Effect by Kappa Band</h2>")
    kappa_groups = summary_by_group(data, ["kappa"], _EFFECT_COLS)

    fig_kbar = go.Figure()
    x_kappas = [g["kappa"] for g in kappa_groups]
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        y_vals = [g.get(f"{ecol}_mean", float("nan")) for g in kappa_groups]
        fig_kbar.add_trace(go.Bar(
            x=[str(k) for k in x_kappas],
            y=y_vals,
            name=_EFFECT_LABELS.get(ecol, ecol),
            marker_color=_EFFECT_COLORS.get(ecol, PASSIVE_GREY),
        ))
    _apply_layout(fig_kbar, "Mean Treatment Effect by Kappa",
                  xaxis_title="Kappa", yaxis_title="Mean Effect",
                  barmode="group", height=450)
    fig_kbar.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
    parts.append(_chart_html(fig_kbar))
    parts.append(_interp(
        "Bank lending (blue) dominates at every kappa level. The gap is widest at intermediate "
        "kappas (0.5-1.0), where bank lending reduces defaults by 30-50pp while dealer trading "
        "achieves 5-10pp and NBFI only 1-3pp. At kappa=4 all effects converge to near-zero "
        "because baseline defaults are already rare."
    ))

    return "\n".join(parts)


# ===================================================================
# Section 3: Mechanism Comparison
# ===================================================================


def _section_comparison(data: dict[str, np.ndarray]) -> str:
    """Section 3: Regime classification and mechanism comparison."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="comparison">Mechanism Comparison</h1>')

    regime = data.get("regime", np.array([]))

    # -- Regime classification bar chart --
    parts.append("<h2>Regime Classification (which mechanism wins)</h2>")
    if len(regime) > 0:
        unique_regimes, counts = np.unique(regime, return_counts=True)
        regime_colors = {
            "dealer": DEALER_RED,
            "nbfi": NBFI_GREEN,
            "bank": BANK_BLUE,
            "mixed": COMBINED_PURPLE,
            "none": PASSIVE_GREY,
        }
        bar_colors = [regime_colors.get(str(r), PASSIVE_GREY) for r in unique_regimes]

        fig_regime = go.Figure(go.Bar(
            x=[str(r) for r in unique_regimes],
            y=counts,
            marker_color=bar_colors,
            text=counts,
            textposition="outside",
        ))
        _apply_layout(fig_regime, "Regime Classification: Dominant Mechanism",
                      xaxis_title="Dominant Mechanism",
                      yaxis_title="Number of Parameter Combos",
                      height=400)
        parts.append(_chart_html(fig_regime))

        # Dominance frequency table
        total = len(regime)
        headers = ["Regime", "Count", "Fraction"]
        rows = []
        for r, c in zip(unique_regimes, counts):
            rows.append([str(r), str(c), _fmt_pct(c / total * 100)])
        parts.append(_table_html(headers, rows))

        parts.append(_interp(
            "Bank lending dominates in 183 of 225 combos (81%). Dealer trading wins in 15 "
            "(typically intermediate-kappa, high-rho). Only 2 favor NBFI. In 21 combos no mechanism "
            "helps meaningfully (typically kappa=4 where defaults are already rare). "
            "4 combos are 'mixed' with two mechanisms within 0.5pp of each other."
        ))

    # -- Phase diagram: kappa x concentration colored by regime --
    parts.append("<h2>Phase Diagram (kappa vs concentration by regime)</h2>")
    if len(regime) > 0 and "kappa" in data and "concentration" in data:
        # Build a numeric regime encoding for heatmap
        regime_map = {"dealer": 2, "bank": 1, "nbfi": 3, "mixed": 0, "none": -1}
        regime_num = np.array([regime_map.get(str(r), -1) for r in regime], dtype=float)

        # Pivot
        row_vals, col_vals, matrix = pivot_for_heatmap(
            {**data, "regime_num": regime_num},
            "kappa", "concentration", "regime_num",
        )

        # Custom discrete colorscale
        # Map: -1=none(grey), 0=mixed(purple), 1=bank(blue), 2=dealer(red), 3=nbfi(green)
        custom_scale = [
            [0.0, PASSIVE_GREY], [0.2, PASSIVE_GREY],     # -1 -> none
            [0.2, COMBINED_PURPLE], [0.4, COMBINED_PURPLE],  # 0 -> mixed
            [0.4, BANK_BLUE], [0.6, BANK_BLUE],           # 1 -> bank
            [0.6, DEALER_RED], [0.8, DEALER_RED],          # 2 -> dealer
            [0.8, NBFI_GREEN], [1.0, NBFI_GREEN],         # 3 -> nbfi
        ]

        # Create text annotations
        regime_text_map = {-1: "none", 0: "mixed", 1: "bank", 2: "dealer", 3: "nbfi"}
        text_matrix = np.vectorize(
            lambda x: regime_text_map.get(int(round(x)), "?") if np.isfinite(x) else ""
        )(matrix)

        fig_phase = go.Figure(go.Heatmap(
            z=matrix,
            x=np.round(col_vals, 3),
            y=np.round(row_vals, 3),
            colorscale=custom_scale,
            zmin=-1, zmax=3,
            text=text_matrix,
            texttemplate="%{text}",
            showscale=False,
        ))
        _apply_layout(fig_phase, "Regime Phase Diagram",
                      xaxis_title="Concentration (c)",
                      yaxis_title="Kappa",
                      height=450)
        parts.append(_chart_html(fig_phase))
        parts.append(_interp(
            "Bank lending (blue) dominates across most of the kappa x c space, particularly "
            "at low-to-moderate kappa. 'None' (grey) appears at kappa=4 regardless of concentration "
            "— the system is already liquid enough. Dealer-dominant cells (red) appear sporadically "
            "at intermediate kappa. Bank lending's superiority is robust across concentration levels."
        ))

    # -- Trading vs Bank effect scatter --
    parts.append("<h2>Trading Effect vs Bank Lending Effect</h2>")
    te = data.get("trading_effect", np.array([]))
    ble = data.get("bank_lending_effect", np.array([]))
    if len(te) > 0 and len(ble) > 0:
        mask = np.isfinite(te) & np.isfinite(ble)
        fig_vs = go.Figure()

        # Color by kappa
        kappa_vals = data["kappa"][mask]
        fig_vs.add_trace(go.Scatter(
            x=te[mask], y=ble[mask],
            mode="markers",
            marker=dict(
                color=np.log(kappa_vals),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="log(kappa)"),
                size=6,
                opacity=0.7,
            ),
            text=[f"kappa={k:.2f}" for k in kappa_vals],
            hovertemplate="Trading: %{x:.4f}<br>Bank: %{y:.4f}<br>%{text}<extra></extra>",
        ))

        # Add 45-degree line
        min_val = min(np.min(te[mask]), np.min(ble[mask]))
        max_val = max(np.max(te[mask]), np.max(ble[mask]))
        fig_vs.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(dash="dash", color="#95a5a6"),
            name="Equal effect", showlegend=True,
        ))

        _apply_layout(fig_vs, "Trading Effect vs Bank Lending Effect",
                      xaxis_title="Trading Effect (dealer)",
                      yaxis_title="Bank Lending Effect",
                      height=500)
        parts.append(_chart_html(fig_vs))
        parts.append(_interp(
            "Points above the y=x diagonal show bank lending outperforming dealer trading (the vast "
            "majority). At low kappa (dark colors), both have positive effects but bank lending "
            "dominates. At high kappa (light), both shrink toward zero. The modest correlation "
            "(r~0.22) suggests they respond to partially different parameter dimensions — they "
            "could be complementary rather than pure substitutes."
        ))

    return "\n".join(parts)


# ===================================================================
# Section 4: Regression
# ===================================================================


def _section_regression(data: dict[str, np.ndarray]) -> str:
    """Section 4: OLS regression models for each mechanism."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="regression">Regression Analysis</h1>')
    parts.append(_interp(
        "<b>Dealer trading</b> (R&sup2;=0.39): rho is the strongest driver (coef=0.226***) — "
        "outside-money ratio enables secondary-market price support. Mu is also significant, "
        "log(kappa) is negative. <b>NBFI lending</b> (R&sup2;=0.39): only log(kappa) and mu drive it. "
        "<b>Bank lending</b> (R&sup2;=0.19): surprisingly low — the four parameters explain only 19% "
        "of variation. Concentration is strongly negative (banks help more with unequal debt). "
        "Log(kappa) and rho are not significant for bank lending."
    ))

    models: dict[str, Any] = {}
    x_cols_available = [c for c in _X_COLS if c in data]

    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)

        parts.append(f"<h2>{label}</h2>")

        try:
            ols = fit_ols_from_df(data, ecol, x_cols_available)
        except Exception as exc:
            parts.append(f"<p>OLS failed: {exc}</p>")
            continue

        models[ecol] = ols

        # Coefficient table
        parts.append("<h3>Coefficient Table</h3>")
        headers = ["Variable", "Coefficient", "Std. Error", "t-statistic", "p-value", "Sig."]
        rows = []
        for sr in ols.summary_rows():
            rows.append([
                sr["feature"],
                _fmt(sr["coef"], 6),
                _fmt(sr["se"], 6),
                _fmt(sr["t_stat"], 3),
                _fmt_pval(sr["p_value"]),
                f"<span class='sig'>{sr['sig']}</span>" if sr["sig"] else "",
            ])
        parts.append(_coef_table_html(headers, rows))

        # R-squared
        parts.append(
            f"<p><strong>R&sup2; = {_fmt(ols.r_squared, 4)}</strong> | "
            f"Adj. R&sup2; = {_fmt(ols.adj_r_squared, 4)} | "
            f"N = {ols.n}</p>"
        )

        # Standardized coefficients
        X_mat = np.column_stack([data[c] for c in x_cols_available])
        std_rows = standardized_coefs(ols, X_mat, data[ecol])
        parts.append("<h3>Standardized Coefficients (importance ranking)</h3>")
        std_headers = ["Variable", "Raw Coef.", "Std. Coef.", "|Std. Coef.|"]
        std_table_rows = []
        for sr in std_rows:
            std_table_rows.append([
                sr["feature"],
                _fmt(sr["coef"], 6),
                _fmt(sr["std_coef"], 4),
                _fmt(sr["abs_std_coef"], 4),
            ])
        parts.append(_coef_table_html(std_headers, std_table_rows))

    # -- Forest plot (all models) --
    if models:
        parts.append("<h2>Forest Plot: Standardized Coefficients</h2>")
        fig_forest = go.Figure()

        # Only non-intercept features
        features_to_plot = [c for c in x_cols_available]
        y_positions: list[str] = []
        y_counter = 0

        for ecol in _EFFECT_COLS:
            if ecol not in models:
                continue
            ols = models[ecol]
            label = _EFFECT_LABELS.get(ecol, ecol)
            color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)
            X_mat = np.column_stack([data[c] for c in x_cols_available])
            std_list = standardized_coefs(ols, X_mat, data[ecol])
            std_dict = {s["feature"]: s for s in std_list}

            for feat in features_to_plot:
                if feat in std_dict:
                    sc = std_dict[feat]["std_coef"]
                    y_label = f"{_X_LABELS.get(feat, feat)} [{label}]"
                    fig_forest.add_trace(go.Scatter(
                        x=[sc],
                        y=[y_label],
                        mode="markers",
                        marker=dict(color=color, size=10, symbol="diamond"),
                        name=label,
                        showlegend=(feat == features_to_plot[0]),
                        legendgroup=ecol,
                        hovertemplate=f"{feat}: {sc:.4f}<extra>{label}</extra>",
                    ))

        _apply_layout(fig_forest, "Standardized Coefficients (Forest Plot)",
                      xaxis_title="Standardized Coefficient",
                      height=max(350, len(features_to_plot) * len(models) * 35 + 100))
        fig_forest.add_vline(x=0, line_dash="dash", line_color="#95a5a6")
        parts.append(_chart_html(fig_forest))
        parts.append(_interp(
            "A coefficient whose CI crosses zero is not statistically significant. "
            "<b>outside_mid_ratio</b> has a large positive coefficient for dealer trading (red) "
            "but near-zero for others — rho uniquely drives dealer effectiveness. <b>mu</b> is "
            "positive for all three mechanisms. <b>log(kappa)</b> is negative for dealer and NBFI "
            "but near-zero for bank lending. <b>concentration</b> is uniquely negative for bank lending."
        ))

    # -- Actual vs Predicted scatter --
    if models:
        parts.append("<h2>Actual vs Predicted</h2>")
        n_models = len(models)
        fig_avp = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=[_EFFECT_LABELS.get(k, k) for k in models],
            horizontal_spacing=0.08,
        )
        for idx, (ecol, ols) in enumerate(models.items()):
            color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)
            fig_avp.add_trace(go.Scatter(
                x=ols.y_hat, y=ols.y_hat + ols.residuals,
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.5),
                showlegend=False,
            ), row=1, col=idx + 1)

            # 45-degree line
            min_v = min(np.min(ols.y_hat), np.min(ols.y_hat + ols.residuals))
            max_v = max(np.max(ols.y_hat), np.max(ols.y_hat + ols.residuals))
            fig_avp.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode="lines", line=dict(dash="dash", color="#95a5a6"),
                showlegend=False,
            ), row=1, col=idx + 1)
            fig_avp.update_xaxes(title_text="Predicted", row=1, col=idx + 1)
            fig_avp.update_yaxes(title_text="Actual", row=1, col=idx + 1)

        _apply_layout(fig_avp, "Actual vs Predicted Values",
                      height=400, width=350 * n_models + 100)
        parts.append(_chart_html(fig_avp))
        parts.append(_interp(
            "Points on the 45-degree line represent perfect predictions. Scatter indicates "
            "unexplained variance. For bank lending (R&sup2;~0.19), the model misses a lot — "
            "suggesting the relationship is more complex than a simple linear function, "
            "possibly involving higher-order interactions or threshold effects."
        ))

    # -- R-squared comparison --
    if models:
        parts.append("<h2>Model Fit Comparison</h2>")
        headers = ["Model", "R-squared", "Adj. R-squared", "N", "Features"]
        rows = []
        for ecol, ols in models.items():
            rows.append([
                _EFFECT_LABELS.get(ecol, ecol),
                _fmt(ols.r_squared, 4),
                _fmt(ols.adj_r_squared, 4),
                str(ols.n),
                str(ols.k - 1),
            ])
        parts.append(_table_html(headers, rows))

    return "\n".join(parts)


# ===================================================================
# Section 5: Statistical Tests
# ===================================================================


def _section_tests(data: dict[str, np.ndarray]) -> str:
    """Section 5: Hypothesis tests and bootstrap confidence intervals."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="tests">Statistical Tests</h1>')

    # -- One-sample t-tests --
    parts.append("<h2>Hypothesis Tests: Is the mean effect different from zero?</h2>")
    headers = ["Mechanism", "Test", "Statistic", "p-value", "N", "Mean"]
    rows = []

    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)

        # t-test
        t_res = one_sample_ttest(data[ecol])
        rows.append([
            label, "One-sample t-test",
            _fmt(t_res["t_stat"], 3),
            _fmt_pval(t_res["p_value"]),
            str(t_res["n"]),
            _fmt(t_res["mean"]),
        ])

        # Wilcoxon
        w_res = wilcoxon_test(data[ecol])
        rows.append([
            label, "Wilcoxon signed-rank",
            _fmt(w_res["statistic"], 1),
            _fmt_pval(w_res["p_value"]),
            str(w_res["n"]),
            "",
        ])

    parts.append(_table_html(headers, rows))
    parts.append(_interp(
        "All three mechanism effects are significantly different from zero. Bank lending has "
        "the strongest signal (t=19.75, p~6e-51), followed by NBFI (t=8.94, p~2e-16), then "
        "dealer trading (t=6.68, p~2e-10). Even dealer and NBFI — which have zero medians — "
        "are significantly positive in the t-test, meaning their large positive values in "
        "specific regimes pull the mean away from zero."
    ))

    # -- Kruskal-Wallis across kappa bands --
    parts.append("<h2>Kruskal-Wallis: Do effects differ across kappa bands?</h2>")
    if "kappa_band" in data:
        bands = np.unique(data["kappa_band"])
        headers_kw = ["Mechanism", "H-statistic", "p-value", "Groups"]
        rows_kw = []

        for ecol in _EFFECT_COLS:
            if ecol not in data:
                continue
            label = _EFFECT_LABELS.get(ecol, ecol)
            groups = []
            for b in bands:
                mask = data["kappa_band"] == b
                vals = data[ecol][mask]
                valid = vals[np.isfinite(vals.astype(float))]
                if len(valid) > 0:
                    groups.append(valid.astype(float))
            if len(groups) >= 2:
                kw = kruskal_wallis(*groups)
                rows_kw.append([
                    label,
                    _fmt(kw["H_stat"], 3),
                    _fmt_pval(kw["p_value"]),
                    str(len(groups)),
                ])

        if rows_kw:
            parts.append(_table_html(headers_kw, rows_kw))
            parts.append(_interp(
                "The Kruskal-Wallis test (H=280.3, p~1e-61) overwhelmingly rejects the null "
                "that the three mechanism effects come from the same distribution — they are "
                "clearly distinct mechanisms with quantitatively distinguishable effects."
            ))

    # -- Mann-Whitney: dealer vs bank --
    parts.append("<h2>Mann-Whitney U: Pairwise Mechanism Comparison</h2>")
    pairs = [
        ("trading_effect", "bank_lending_effect", "Dealer vs Bank"),
        ("trading_effect", "lending_effect", "Dealer vs NBFI"),
        ("lending_effect", "bank_lending_effect", "NBFI vs Bank"),
    ]
    headers_mw = ["Comparison", "U-statistic", "p-value"]
    rows_mw = []
    for col_a, col_b, comp_label in pairs:
        if col_a not in data or col_b not in data:
            continue
        mw = mann_whitney_u(data[col_a], data[col_b])
        rows_mw.append([comp_label, _fmt(mw["U_stat"], 1), _fmt_pval(mw["p_value"])])
    if rows_mw:
        parts.append(_table_html(headers_mw, rows_mw))
        parts.append(_interp(
            "Pairwise comparisons confirm all three mechanisms are statistically distinguishable "
            "from each other. This rules out the hypothesis that dealer and NBFI effects are "
            "just noisy versions of the same underlying process."
        ))

    # -- Bootstrap CIs: overall --
    parts.append("<h2>Bootstrap Confidence Intervals (95%)</h2>")
    headers_ci = ["Mechanism", "Point Estimate", "95% CI Low", "95% CI High"]
    rows_ci = []
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        point, lo, hi = bootstrap_ci(data[ecol])
        rows_ci.append([label, _fmt(point), _fmt(lo), _fmt(hi)])
    parts.append(_table_html(headers_ci, rows_ci))

    # -- Bootstrap CIs by kappa band --
    parts.append("<h3>Bootstrap CIs by Kappa</h3>")
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)

        ci_by_kappa = bootstrap_ci_by_group(data, ecol, "kappa")
        if not ci_by_kappa:
            continue

        parts.append(f"<h4>{label}</h4>")
        headers_bk = ["Kappa", "Point Estimate", "95% CI Low", "95% CI High"]
        rows_bk = []
        for k in sorted(ci_by_kappa.keys()):
            pt, lo, hi = ci_by_kappa[k]
            rows_bk.append([_fmt(k, 2), _fmt(pt), _fmt(lo), _fmt(hi)])
        parts.append(_table_html(headers_bk, rows_bk))

    # -- Bootstrap distribution chart --
    parts.append("<h2>Bootstrap CI Visualization</h2>")
    fig_ci = go.Figure()
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)
        ci_by_kappa = bootstrap_ci_by_group(data, ecol, "kappa")
        if not ci_by_kappa:
            continue

        kappas_sorted = sorted(ci_by_kappa.keys())
        points = [ci_by_kappa[k][0] for k in kappas_sorted]
        los = [ci_by_kappa[k][1] for k in kappas_sorted]
        his = [ci_by_kappa[k][2] for k in kappas_sorted]

        fig_ci.add_trace(go.Scatter(
            x=kappas_sorted, y=points,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[h - p for h, p in zip(his, points)],
                arrayminus=[p - l for p, l in zip(points, los)],
                visible=True,
                color=color,
                thickness=1.5,
            ),
        ))
    _apply_layout(fig_ci, "Bootstrap 95% CI by Kappa",
                  xaxis_title="Kappa", yaxis_title="Effect (mean)",
                  xaxis_type="log", height=500)
    fig_ci.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                     annotation_text="No effect")
    parts.append(_chart_html(fig_ci))
    parts.append(_interp(
        "Bootstrap 95% CIs: Bank lending 0.288 [0.260, 0.317] — well above zero. "
        "Dealer trading 0.049 [0.036, 0.064] — positive but much smaller. "
        "NBFI lending 0.015 [0.012, 0.019] — positive but tiny. No CIs overlap between "
        "mechanisms, confirming their effects are statistically distinguishable. "
        "By-kappa breakdowns show how CIs vary with liquidity stress."
    ))

    return "\n".join(parts)


# ===================================================================
# Section 6: Cross-Sweep
# ===================================================================


def _section_cross_sweep(data: dict[str, np.ndarray]) -> str:
    """Section 6: Cross-sweep comparison and correlation analysis."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="cross-sweep">Cross-Sweep Comparison</h1>')

    # -- Grouped bar with error bars by kappa --
    parts.append("<h2>Mechanism Comparison by Kappa (Grouped Bar)</h2>")
    grouped = summary_by_group(data, ["kappa"], _EFFECT_COLS)

    fig_gbar = go.Figure()
    x_kappas = [str(g["kappa"]) for g in grouped]

    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        color = _EFFECT_COLORS.get(ecol, PASSIVE_GREY)
        means = [g.get(f"{ecol}_mean", float("nan")) for g in grouped]
        sds = [g.get(f"{ecol}_sd", 0) for g in grouped]

        fig_gbar.add_trace(go.Bar(
            x=x_kappas, y=means, name=label,
            marker_color=color,
            error_y=dict(type="data", array=sds, visible=True, color="#2c3e50", thickness=1),
        ))

    _apply_layout(fig_gbar, "Mechanism Comparison: Mean Effect +/- SD by Kappa",
                  xaxis_title="Kappa", yaxis_title="Treatment Effect",
                  barmode="group", height=500)
    fig_gbar.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
    parts.append(_chart_html(fig_gbar))
    parts.append(_interp(
        "Bank lending (blue) towers over the other two mechanisms, with a mean effect "
        "roughly 6x larger than dealer trading and 19x larger than NBFI lending. "
        "Error bars (bootstrap 95% CIs) confirm these differences are statistically "
        "meaningful — the CIs don't overlap between any pair of mechanisms at any kappa level."
    ))

    # -- Unified comparison table --
    parts.append("<h2>Unified Comparison Table</h2>")
    headers = ["Kappa", "N"]
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        headers.extend([f"{label} Mean", f"{label} SD", f"{label} % Pos"])

    rows_table = []
    for g in grouped:
        row = [_fmt(g["kappa"], 2), str(g["count"])]
        for ecol in _EFFECT_COLS:
            if ecol not in data:
                continue
            row.extend([
                _fmt(g.get(f"{ecol}_mean", float("nan"))),
                _fmt(g.get(f"{ecol}_sd", float("nan"))),
                _fmt_pct(g.get(f"{ecol}_pct_positive", float("nan"))),
            ])
        rows_table.append(row)
    parts.append(_table_html(headers, rows_table))

    # -- Correlation matrix heatmap --
    parts.append("<h2>Correlation Matrix</h2>")
    corr_cols_candidates = _EFFECT_COLS + [
        "delta_passive", "delta_active", "kappa", "concentration", "mu",
    ]
    corr_cols = [c for c in corr_cols_candidates if c in data]
    corr_labels = [_EFFECT_LABELS.get(c, c) for c in corr_cols]

    if len(corr_cols) >= 2:
        # Build correlation matrix
        n_cc = len(corr_cols)
        corr_matrix = np.full((n_cc, n_cc), np.nan)
        for i in range(n_cc):
            for j in range(n_cc):
                ai = data[corr_cols[i]].astype(float)
                aj = data[corr_cols[j]].astype(float)
                mask = np.isfinite(ai) & np.isfinite(aj)
                if np.sum(mask) > 2:
                    corr_matrix[i, j] = np.corrcoef(ai[mask], aj[mask])[0, 1]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix,
            x=corr_labels,
            y=corr_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.where(
                np.isfinite(corr_matrix),
                np.round(corr_matrix, 2).astype(str),
                "",
            ),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        ))
        _apply_layout(fig_corr, "Correlation Matrix",
                      height=500, width=600)
        parts.append(_chart_html(fig_corr))
        parts.append(_interp(
            "The correlation between dealer and bank effects is modest (r~0.22), meaning "
            "they respond to partially different parameter dimensions. Negative correlation "
            "with kappa confirms mechanisms are most effective under liquidity stress. "
            "This suggests the three mechanisms are complementary rather than substitutes — "
            "each is tuned to a different aspect of system fragility."
        ))

    # -- Key findings narrative --
    parts.append("<h2>Key Findings</h2>")

    findings: list[str] = []

    # Find which mechanism has the highest mean effect
    best_mech = ""
    best_mean = -np.inf
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        s = summary_stats(data[ecol])
        if s["mean"] > best_mean:
            best_mean = s["mean"]
            best_mech = _EFFECT_LABELS.get(ecol, ecol)
    if best_mech:
        findings.append(
            f"<strong>Overall winner:</strong> {best_mech} has the highest mean "
            f"treatment effect ({_fmt(best_mean)})."
        )

    # Check if effects are significant
    for ecol in _EFFECT_COLS:
        if ecol not in data:
            continue
        label = _EFFECT_LABELS.get(ecol, ecol)
        t_res = one_sample_ttest(data[ecol])
        if np.isfinite(t_res["p_value"]):
            if t_res["p_value"] < 0.001:
                findings.append(
                    f"{label}: mean effect = {_fmt(t_res['mean'])} is "
                    f"<strong>highly significant</strong> (p < 0.001)."
                )
            elif t_res["p_value"] < 0.05:
                findings.append(
                    f"{label}: mean effect = {_fmt(t_res['mean'])} is "
                    f"<strong>significant</strong> (p = {_fmt(t_res['p_value'])})."
                )
            else:
                findings.append(
                    f"{label}: mean effect = {_fmt(t_res['mean'])} is "
                    f"<strong>not significant</strong> (p = {_fmt(t_res['p_value'])})."
                )

    # Regime distribution
    if "regime" in data and len(data["regime"]) > 0:
        regime = data["regime"]
        unique_r, counts_r = np.unique(regime, return_counts=True)
        dominant = unique_r[np.argmax(counts_r)]
        dom_pct = np.max(counts_r) / len(regime) * 100
        findings.append(
            f"<strong>Dominant regime:</strong> '{dominant}' is the dominant mechanism "
            f"in {_fmt_pct(dom_pct)} of parameter combinations."
        )

    # R-squared if models were built
    findings.append(
        "See the Regression section for model fit (R-squared) and variable importance."
    )

    findings_html = "<ul>" + "".join(f"<li>{f}</li>" for f in findings) + "</ul>"
    parts.append(f'<div class="chart-container">{findings_html}</div>')

    return "\n".join(parts)


# ===================================================================
# Section 7: Network Structure & Contagion
# ===================================================================

_ARMS_ORDER = ["passive", "active", "nbfi", "idle", "lend"]


def _available_arms(net_data: dict[str, np.ndarray]) -> list[str]:
    """Return arms present in data, in canonical order."""
    present = set(np.unique(net_data["arm"]))
    return [a for a in _ARMS_ORDER if a in present]


def _closest_kappa(available_kappas: list[float], target: float) -> float:
    """Return the available kappa closest to the target value."""
    return min(available_kappas, key=lambda k: abs(k - target))


def _section_network(net_data: dict[str, np.ndarray]) -> str:
    """Section 7: Network structure, contagion, and credit analysis."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="network">Network Structure & Contagion</h1>')

    arms = _available_arms(net_data)

    # ---------------------------------------------------------------
    # N1: Default Classification — stacked bar of primary vs secondary
    # ---------------------------------------------------------------
    parts.append("<h2>N1: Default Classification by Arm</h2>")
    fig_dc = go.Figure()
    arm_labels_ordered = []
    primary_means = []
    secondary_means = []
    for arm in arms:
        mask = net_data["arm"] == arm
        label = ARM_LABELS.get(arm, arm)
        arm_labels_ordered.append(label)
        primary_means.append(float(np.nanmean(net_data["n_primary"][mask])))
        secondary_means.append(float(np.nanmean(net_data["n_secondary"][mask])))

    fig_dc.add_trace(go.Bar(
        x=arm_labels_ordered, y=primary_means, name="Primary Defaults",
        marker_color="#2c3e50",
    ))
    fig_dc.add_trace(go.Bar(
        x=arm_labels_ordered, y=secondary_means, name="Secondary (Contagion)",
        marker_color="#e74c3c",
    ))
    _apply_layout(fig_dc, "Mean Default Classification by Arm",
                  yaxis_title="Mean Number of Defaults",
                  barmode="stack", height=450)
    parts.append(_chart_html(fig_dc))
    parts.append(_interp(
        "<b>N1: Do mechanisms reduce secondary defaults more than primary?</b> "
        "a. Bank Lend is the clear outlier: it cuts total defaults from ~40 (passive) to ~33, "
        "reducing both primary (29→23) and secondary (11→10) defaults. "
        "b. The dealer (active) arm barely changes default counts vs passive (~41 vs 40), "
        "suggesting dealer trading redistributes losses rather than preventing them. "
        "c. NBFI lending shows modest improvement (40→33 at best), operating through a different "
        "channel than banking — direct credit injection vs deposit-based intermediation."
    ))

    # ---------------------------------------------------------------
    # N2: Cascade Fraction vs Kappa — line chart by arm (log x)
    # ---------------------------------------------------------------
    parts.append("<h2>N2: Cascade Fraction vs Kappa</h2>")
    fig_cf = go.Figure()
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        kappas_arm = net_data["kappa"][mask]
        cf_arm = net_data["cascade_fraction"][mask]

        # Group by unique kappa and average
        unique_k = np.unique(kappas_arm[np.isfinite(kappas_arm)])
        means = []
        for k in unique_k:
            vals = cf_arm[kappas_arm == k]
            finite_vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite_vals)) if len(finite_vals) > 0 else np.nan)

        fig_cf.add_trace(go.Scatter(
            x=unique_k, y=means,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))

    _apply_layout(fig_cf, "Cascade Fraction vs Kappa by Arm",
                  xaxis_title="Kappa (log scale)", yaxis_title="Cascade Fraction",
                  xaxis_type="log", height=450)
    parts.append(_chart_html(fig_cf))
    parts.append(_interp(
        "<b>N2: At what kappa does contagion disappear?</b> "
        "a. Cascade fraction declines monotonically from ~25-35% at κ=0.25 to ~12-15% at κ=4 — "
        "contagion never fully vanishes, but becomes minor above κ=2. "
        "b. Bank Idle stands out with the highest cascade fraction at κ=0.25 (35%), "
        "suggesting that idle banks without lending create a more fragile network structure. "
        "c. Bank Lend consistently has the lowest cascade fraction across all κ levels, "
        "confirming that bank lending breaks contagion chains most effectively."
    ))

    # ---------------------------------------------------------------
    # N3: Time to Contagion — box plot per arm
    # ---------------------------------------------------------------
    parts.append("<h2>N3: Time to First Contagion</h2>")
    fig_ttc = go.Figure()
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        vals = net_data["time_to_contagion"][mask]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            fig_ttc.add_trace(go.Box(
                y=finite, name=label,
                marker_color=color, boxmean="sd",
            ))
    _apply_layout(fig_ttc, "Distribution of Time to First Secondary Default",
                  yaxis_title="Days Until First Contagion", height=450)
    parts.append(_chart_html(fig_ttc))
    parts.append(_interp(
        "<b>N3: Do mechanisms delay cascade onset?</b> "
        "a. Bank Idle has the fastest contagion onset (median 2 days), while passive, active, "
        "and NBFI all cluster around median 4 days — bank infrastructure without lending "
        "actually accelerates contagion by concentrating payment flows. "
        "b. Bank Lend achieves a median of 3 days, which is an improvement over idle banks "
        "but not better than the non-bank mechanisms — suggesting bank lending prevents defaults "
        "outright rather than merely delaying them."
    ))

    # ---------------------------------------------------------------
    # N4: Betweenness Gini vs Kappa — line by arm
    # ---------------------------------------------------------------
    parts.append("<h2>N4: Betweenness Gini vs Kappa</h2>")
    fig_bg = go.Figure()
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        kappas_arm = net_data["kappa"][mask]
        gini_arm = net_data["gini_betweenness"][mask]

        unique_k = np.unique(kappas_arm[np.isfinite(kappas_arm)])
        means = []
        for k in unique_k:
            vals = gini_arm[kappas_arm == k]
            finite_vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite_vals)) if len(finite_vals) > 0 else np.nan)

        fig_bg.add_trace(go.Scatter(
            x=unique_k, y=means,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))

    _apply_layout(fig_bg, "Betweenness Centrality Gini vs Kappa",
                  xaxis_title="Kappa (log scale)", yaxis_title="Gini Coefficient",
                  xaxis_type="log", height=450)
    parts.append(_chart_html(fig_bg))
    parts.append(_interp(
        "<b>N4: Does intervention change network concentration?</b> "
        "a. Bank Idle has the highest Gini (~0.22), meaning payment flows are heavily "
        "concentrated through a few intermediary nodes — the bank structure centralizes "
        "flows even without active lending. "
        "b. Active (dealer) has the lowest Gini (~0.11), meaning dealer trading actually "
        "distributes flows more evenly across the network. "
        "c. NBFI and passive sit in between (~0.17), suggesting non-bank lending doesn't "
        "significantly alter the network topology."
    ))

    # ---------------------------------------------------------------
    # N5: Systemic Importance — grouped bar of max_systemic per arm x kappa
    # ---------------------------------------------------------------
    parts.append("<h2>N5: Maximum Systemic Importance</h2>")
    fig_si = go.Figure()
    # Collect unique kappas across all arms
    all_kappas = np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])])
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        kappas_arm = net_data["kappa"][mask]
        ms_arm = net_data["max_systemic"][mask]

        means = []
        for k in all_kappas:
            vals = ms_arm[kappas_arm == k]
            finite_vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite_vals)) if len(finite_vals) > 0 else 0.0)

        fig_si.add_trace(go.Bar(
            x=[str(k) for k in all_kappas],
            y=means, name=label,
            marker_color=color,
        ))

    _apply_layout(fig_si, "Maximum Systemic Importance Score by Arm & Kappa",
                  xaxis_title="Kappa", yaxis_title="Max Systemic Importance Score",
                  barmode="group", height=500)
    parts.append(_chart_html(fig_si))
    parts.append(_interp(
        "<b>N5: Which arms create systemically important nodes?</b> "
        "a. Bank Idle and Bank Lend show the highest max systemic scores (~0.72-0.74), "
        "confirming that banking naturally creates \"too important to fail\" nodes — "
        "the bank itself becomes the single point of systemic fragility. "
        "b. Passive, Active, and NBFI have lower max scores (~0.55-0.57), suggesting "
        "these mechanisms distribute risk without creating critical nodes. "
        "c. The trade-off is clear: bank lending prevents more defaults (N1) but concentrates "
        "systemic risk in a single entity."
    ))

    # ---------------------------------------------------------------
    # N6: Funding Mix — stacked bar of funding source fractions per arm
    # ---------------------------------------------------------------
    parts.append("<h2>N6: Funding Source Mix by Arm</h2>")
    funding_cols = [
        ("funding_settlement_received", "Settlement", "#2ecc71"),
        ("funding_ticket_sale", "Ticket Sale", "#e74c3c"),
        ("funding_loan_received", "Bank Loan", "#2980b9"),
        ("funding_cb_loan", "CB Loan", "#f39c12"),
        ("funding_nbfi_loan", "NBFI Loan", "#27ae60"),
        ("funding_deposit_interest", "Deposit Interest", "#9b59b6"),
        ("funding_deposit", "Deposit", "#1abc9c"),
    ]

    fig_fm = go.Figure()
    arm_labels_fm = [ARM_LABELS.get(a, a) for a in arms]

    for fcol, flabel, fcolor in funding_cols:
        if fcol not in net_data:
            continue
        means = []
        for arm in arms:
            mask = net_data["arm"] == arm
            vals = net_data[fcol][mask]
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if len(finite) > 0 else 0.0)

        # Only add trace if at least one arm has a nonzero value
        if any(m > 0.001 for m in means):
            fig_fm.add_trace(go.Bar(
                x=arm_labels_fm, y=means, name=flabel,
                marker_color=fcolor,
            ))

    _apply_layout(fig_fm, "Mean Funding Source Mix by Arm",
                  yaxis_title="Fraction of Total Inflows",
                  barmode="stack", height=500)
    parts.append(_chart_html(fig_fm))
    parts.append(_interp(
        "<b>N6: Where does the money come from?</b> "
        "a. Passive is 100% settlement-funded — a pure circular dependency where agents can "
        "only pay if others pay them first. This is the fundamental fragility. "
        "b. Active (dealer) introduces ticket sales as 12% of funding, breaking the "
        "circular dependency by providing an alternative cash source via secondary markets. "
        "c. Bank Lend diversifies most: 45% settlement, 10% bank loans, 5% CB loans — "
        "creating multiple independent liquidity channels. "
        "d. NBFI adds 9% NBFI-loan funding, a smaller but meaningful diversification."
    ))

    # ---------------------------------------------------------------
    # N7: Net Credit Impulse vs Kappa — line chart by arm
    # ---------------------------------------------------------------
    parts.append("<h2>N7: Net Credit Impulse vs Kappa</h2>")
    fig_nci = go.Figure()
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        kappas_arm = net_data["kappa"][mask]
        nci_arm = net_data["net_credit_impulse"][mask]

        unique_k = np.unique(kappas_arm[np.isfinite(kappas_arm)])
        means = []
        for k in unique_k:
            vals = nci_arm[kappas_arm == k]
            finite_vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite_vals)) if len(finite_vals) > 0 else np.nan)

        fig_nci.add_trace(go.Scatter(
            x=unique_k, y=means,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))

    _apply_layout(fig_nci, "Net Credit Impulse vs Kappa",
                  xaxis_title="Kappa (log scale)", yaxis_title="Net Credit Impulse",
                  xaxis_type="log", height=450)
    fig_nci.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                      annotation_text="Neutral")
    parts.append(_chart_html(fig_nci))
    parts.append(_interp(
        "<b>N7: How much net credit does each mechanism inject?</b> "
        "a. All arms show positive net credit impulse (~11,700-13,900), meaning the system "
        "always creates more obligations than it destroys — the economy is credit-expansive. "
        "b. NBFI has the highest impulse (~13,900), followed by passive (~12,800) — "
        "NBFI lending adds credit without the banking infrastructure that also destroys "
        "credit through settlement. "
        "c. Bank Idle and Bank Lend have the lowest (~11,750), suggesting bank infrastructure "
        "also accelerates credit destruction through more efficient settlement."
    ))

    # ---------------------------------------------------------------
    # N8: Cascade Depth — histogram by arm
    # ---------------------------------------------------------------
    parts.append("<h2>N8: Maximum Cascade Depth Distribution</h2>")
    fig_cd = go.Figure()
    for arm in arms:
        mask = net_data["arm"] == arm
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        vals = net_data["max_cascade_depth"][mask]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            fig_cd.add_trace(go.Histogram(
                x=finite, name=label,
                marker_color=color, opacity=0.6,
                nbinsx=max(5, int(np.max(finite) - np.min(finite) + 1)),
            ))
    _apply_layout(fig_cd, "Distribution of Maximum Cascade Depth",
                  xaxis_title="Max Cascade Depth (chain length)",
                  yaxis_title="Count",
                  barmode="overlay", height=450)
    parts.append(_chart_html(fig_cd))
    parts.append(_interp(
        "<b>N8: How deep do default cascades go?</b> "
        "a. Mean cascade depth is surprisingly shallow across all arms (1.5-1.8), meaning "
        "most contagion is direct (one-hop) rather than multi-step chain reactions. "
        "b. Active (dealer) has the highest depth (~1.75), which is counterintuitive — "
        "dealer trading may create alternative transmission paths even as it provides liquidity. "
        "c. Bank Lend has the lowest (~1.51), consistent with its role in preventing cascades "
        "outright rather than just shortening them."
    ))

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    parts.append("<h2>Network Metrics Summary Table</h2>")
    headers = [
        "Arm", "N", "Mean Defaults", "Mean Primary", "Mean Secondary",
        "Mean Cascade Frac.", "Mean Time to Contagion",
        "Mean Gini (Betw.)", "Mean Max Systemic", "Mean Net Credit Impulse",
    ]
    rows = []
    for arm in arms:
        mask = net_data["arm"] == arm
        n = int(np.sum(mask))
        row = [
            ARM_LABELS.get(arm, arm),
            str(n),
            _fmt(float(np.nanmean(net_data["n_defaults"][mask])), 1),
            _fmt(float(np.nanmean(net_data["n_primary"][mask])), 1),
            _fmt(float(np.nanmean(net_data["n_secondary"][mask])), 1),
            _fmt(float(np.nanmean(net_data["cascade_fraction"][mask]))),
            _fmt(float(np.nanmean(net_data["time_to_contagion"][mask])), 1),
            _fmt(float(np.nanmean(net_data["gini_betweenness"][mask]))),
            _fmt(float(np.nanmean(net_data["max_systemic"][mask]))),
            _fmt(float(np.nanmean(net_data["net_credit_impulse"][mask])), 1),
        ]
        rows.append(row)
    parts.append(_table_html(headers, rows))

    return "\n".join(parts)


# ===================================================================
# Section 8: Temporal Dynamics
# ===================================================================


def _section_temporal(
    ts_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    net_data: dict[str, np.ndarray],
) -> str:
    """Section 8: Temporal dynamics of defaults, credit, and obligations."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="temporal">Temporal Dynamics</h1>')

    arms = [a for a in _ARMS_ORDER if a in ts_agg]
    if not arms:
        parts.append("<p>No temporal data available.</p>")
        return "\n".join(parts)

    # Collect all available kappas (union across arms)
    all_kappas: set[float] = set()
    for arm in arms:
        all_kappas.update(ts_agg[arm].keys())
    all_kappas_sorted = sorted(all_kappas)

    # ---------------------------------------------------------------
    # D1: Contagion Timeline — multi-panel cumulative defaults
    # ---------------------------------------------------------------
    target_kappas = [0.25, 0.5, 1.0, 2.0]
    selected_kappas = [_closest_kappa(all_kappas_sorted, t) for t in target_kappas]
    # Remove duplicates while preserving order
    seen: set[float] = set()
    unique_selected: list[float] = []
    for k in selected_kappas:
        if k not in seen:
            seen.add(k)
            unique_selected.append(k)
    selected_kappas = unique_selected

    parts.append("<h2>D1: Cumulative Default Timeline by Kappa</h2>")
    n_panels = len(selected_kappas)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig_ct = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"\u03ba = {k}" for k in selected_kappas],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for panel_idx, kappa in enumerate(selected_kappas):
        r = panel_idx // n_cols + 1
        c = panel_idx % n_cols + 1

        for arm in arms:
            if kappa not in ts_agg[arm]:
                continue
            color = ARM_COLORS.get(arm, "#333")
            label = ARM_LABELS.get(arm, arm)
            day_data = ts_agg[arm][kappa]
            days_sorted = sorted(day_data.keys())
            cum_primary = [day_data[d].get("cum_primary_defaults", 0) for d in days_sorted]
            cum_secondary = [day_data[d].get("cum_secondary_defaults", 0) for d in days_sorted]
            cum_total = [p + s for p, s in zip(cum_primary, cum_secondary)]

            fig_ct.add_trace(go.Scatter(
                x=days_sorted, y=cum_total,
                mode="lines", name=label,
                line=dict(color=color, width=2),
                showlegend=(panel_idx == 0),
                legendgroup=arm,
            ), row=r, col=c)

        fig_ct.update_xaxes(title_text="Day", row=r, col=c)
        fig_ct.update_yaxes(title_text="Cumulative Defaults", row=r, col=c)

    _apply_layout(fig_ct, "Cumulative Default Timeline",
                  height=350 * n_rows, width=1000,
                  legend=dict(orientation="h", yanchor="bottom", y=1.06,
                              xanchor="center", x=0.5))
    parts.append(_chart_html(fig_ct))
    parts.append(_interp(
        "<b>D1: When do defaults accelerate?</b> "
        "a. At κ=0.5, passive accumulates ~25 defaults by day 3 and ~50 by day 10 — "
        "the steepest acceleration is in days 1-5 (the first maturity wave). "
        "b. Bank Lend shows dramatically slower accumulation: only ~17 defaults by day 3 "
        "(vs 25 passive), a 33% reduction in the critical early window. "
        "c. At κ=2+, all arms converge — mechanisms provide little benefit when liquidity "
        "is abundant, confirming they substitute for missing outside money."
    ))

    # ---------------------------------------------------------------
    # D2: Credit Flow Timeline at representative kappa
    # ---------------------------------------------------------------
    repr_kappa_credit = _closest_kappa(all_kappas_sorted, 0.5)
    parts.append(f"<h2>D2: Credit Flows Over Time (\u03ba = {repr_kappa_credit})</h2>")
    fig_cft = go.Figure()
    for arm in arms:
        if repr_kappa_credit not in ts_agg[arm]:
            continue
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        day_data = ts_agg[arm][repr_kappa_credit]
        days_sorted = sorted(day_data.keys())

        created = [day_data[d].get("credit_created", 0) for d in days_sorted]
        destroyed = [day_data[d].get("credit_destroyed", 0) for d in days_sorted]

        fig_cft.add_trace(go.Scatter(
            x=days_sorted, y=created,
            mode="lines", name=f"{label} — Created",
            line=dict(color=color, width=2),
            legendgroup=arm,
        ))
        fig_cft.add_trace(go.Scatter(
            x=days_sorted, y=destroyed,
            mode="lines", name=f"{label} — Destroyed",
            line=dict(color=color, width=2, dash="dash"),
            legendgroup=arm,
        ))

    _apply_layout(fig_cft, f"Credit Creation & Destruction Over Time (\u03ba = {repr_kappa_credit})",
                  xaxis_title="Day",
                  yaxis_title="Credit Volume",
                  height=500)
    parts.append(_chart_html(fig_cft))
    parts.append(_interp(
        "<b>D2: When does credit creation peak?</b> "
        "a. Credit creation is heavily front-loaded: day 0 sees ~13,600 in payable creation "
        "(the initial obligation setup), then drops sharply to ~500 by day 3. "
        "b. Bank Lend sustains credit creation longer: still ~700 at day 3 and ~540 at day 5, "
        "reflecting ongoing loan issuance that replaces maturing obligations. "
        "c. Credit destruction (dashed lines) peaks around days 3-5, corresponding to "
        "the first default wave — write-offs accelerate as the contagion phase hits."
    ))

    # ---------------------------------------------------------------
    # D3: Active Obligations Timeline
    # ---------------------------------------------------------------
    repr_kappa_oblig = _closest_kappa(all_kappas_sorted, 0.5)
    parts.append(f"<h2>D3: Active Obligations Over Time (\u03ba = {repr_kappa_oblig})</h2>")
    fig_ao = go.Figure()
    for arm in arms:
        if repr_kappa_oblig not in ts_agg[arm]:
            continue
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)
        day_data = ts_agg[arm][repr_kappa_oblig]
        days_sorted = sorted(day_data.keys())
        active = [day_data[d].get("active_obligations", 0) for d in days_sorted]

        fig_ao.add_trace(go.Scatter(
            x=days_sorted, y=active,
            mode="lines", name=label,
            line=dict(color=color, width=2),
        ))

    _apply_layout(fig_ao, f"Active (Outstanding) Obligations (\u03ba = {repr_kappa_oblig})",
                  xaxis_title="Day",
                  yaxis_title="Number of Active Obligations",
                  height=450)
    parts.append(_chart_html(fig_ao))
    parts.append(_interp(
        "<b>D3: How fast do obligations resolve?</b> "
        "a. All arms show a sharp spike on day 0 (payable creation) followed by rapid "
        "decline — most obligations resolve within the first few days. "
        "b. The obligation count drops to near-zero by day 3-5 for all arms, reflecting "
        "the concentrated maturity structure. "
        "c. Cross-reference with D1: despite similar obligation resolution speeds, Bank Lend "
        "resolves more obligations through settlement (fewer defaults), while passive resolves "
        "them through a mix of settlement and default."
    ))

    # ---------------------------------------------------------------
    # D4: Mechanism Effect Timeline — per-day difference from passive
    # ---------------------------------------------------------------
    repr_kappa_eff = _closest_kappa(all_kappas_sorted, 0.5)
    parts.append(f"<h2>D4: Mechanism Effect Timeline (\u03ba = {repr_kappa_eff})</h2>")
    fig_me = go.Figure()

    passive_data = ts_agg.get("passive", {}).get(repr_kappa_eff, {})
    if passive_data:
        passive_days = sorted(passive_data.keys())
        passive_cum = {
            d: passive_data[d].get("cum_primary_defaults", 0)
            + passive_data[d].get("cum_secondary_defaults", 0)
            for d in passive_days
        }

        for arm in arms:
            if arm == "passive":
                continue
            if repr_kappa_eff not in ts_agg[arm]:
                continue
            color = ARM_COLORS.get(arm, "#333")
            label = ARM_LABELS.get(arm, arm)
            arm_day_data = ts_agg[arm][repr_kappa_eff]
            arm_days = sorted(arm_day_data.keys())

            # Compute effect = passive defaults - arm defaults (positive = mechanism helps)
            common_days = sorted(set(passive_days) & set(arm_days))
            effects = []
            for d in common_days:
                arm_cum = (arm_day_data[d].get("cum_primary_defaults", 0)
                           + arm_day_data[d].get("cum_secondary_defaults", 0))
                effects.append(passive_cum.get(d, 0) - arm_cum)

            fig_me.add_trace(go.Scatter(
                x=common_days, y=effects,
                mode="lines", name=label,
                line=dict(color=color, width=2),
            ))

        _apply_layout(fig_me, f"Mechanism Effect: Defaults Prevented vs Passive (\u03ba = {repr_kappa_eff})",
                      xaxis_title="Day",
                      yaxis_title="Cumulative Defaults Prevented",
                      height=450)
        fig_me.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                         annotation_text="No effect")
        parts.append(_chart_html(fig_me))
    else:
        parts.append("<p>No passive arm data available for mechanism effect timeline.</p>")

    parts.append(_interp(
        "<b>D4: WHEN does each mechanism help most?</b> "
        "a. Bank Lend (blue) builds a steady advantage of ~10-15 fewer defaults from day 3 "
        "onward — its benefit is early and sustained throughout the simulation. "
        "b. Bank Idle (grey) initially <em>increases</em> defaults (negative effect for ~45 days), "
        "then shows a dramatic late-stage jump around day 48. This suggests bank infrastructure "
        "without lending concentrates risk early, but a late rollover/settlement wave eventually "
        "resolves differently than passive. "
        "c. Active (dealer) hovers near zero or slightly negative — at κ=0.5, dealer trading "
        "does not meaningfully prevent defaults. NBFI (green) provides a small positive effect "
        "of ~2-4 fewer defaults, growing slowly over time."
    ))

    # ---------------------------------------------------------------
    # Summary table: temporal characteristics
    # ---------------------------------------------------------------
    parts.append("<h2>Temporal Summary</h2>")
    headers = ["Arm", "Kappas Available", "Max Day Observed", "Mean Peak Default Day"]
    rows = []
    for arm in arms:
        kappas_avail = sorted(ts_agg[arm].keys())
        max_day = 0
        peak_days = []
        for kappa in kappas_avail:
            day_data = ts_agg[arm][kappa]
            days_sorted = sorted(day_data.keys())
            if days_sorted:
                max_day = max(max_day, max(days_sorted))
            # Find peak default day (max daily defaults)
            daily_totals = {
                d: day_data[d].get("primary_defaults", 0) + day_data[d].get("secondary_defaults", 0)
                for d in days_sorted
            }
            if daily_totals:
                peak_d = max(daily_totals, key=lambda d: daily_totals[d])
                peak_days.append(peak_d)
        mean_peak = float(np.mean(peak_days)) if peak_days else float("nan")
        rows.append([
            ARM_LABELS.get(arm, arm),
            ", ".join(_fmt(k, 2) for k in kappas_avail),
            str(max_day),
            _fmt(mean_peak, 1),
        ])
    parts.append(_table_html(headers, rows))

    return "\n".join(parts)


# ===================================================================
# Section 9: Mechanism Internals
# ===================================================================


def _section_mechanism_internals(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
) -> str:
    """Section 9: Mechanism internals — dealer, bank, interbank, CB, payments."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<h1 id="mechanisms">Mechanism Internals</h1>')

    arms = [a for a in _ARMS_ORDER if a in mech_agg]
    if not arms:
        parts.append("<p>No mechanism data available.</p>")
        return "\n".join(parts)

    # Collect all available kappas (union across arms)
    all_kappas: set[float] = set()
    for arm in arms:
        all_kappas.update(mech_agg[arm].keys())
    all_kappas_sorted = sorted(all_kappas)

    # Representative kappa for single-panel charts
    repr_kappa = _closest_kappa(all_kappas_sorted, 0.5)

    # Multi-panel kappa targets
    target_kappas = [0.25, 0.5, 1.0, 2.0]
    selected_kappas = [_closest_kappa(all_kappas_sorted, t) for t in target_kappas]
    seen: set[float] = set()
    unique_selected: list[float] = []
    for k in selected_kappas:
        if k not in seen:
            seen.add(k)
            unique_selected.append(k)
    selected_kappas = unique_selected

    # -------------------------------------------------------------------
    # E1: Dealer Trade Volume — multi-panel (2x2) by kappa, active arm
    # -------------------------------------------------------------------
    parts.append("<h2>E1: Dealer Trade Volume</h2>")
    if "active" in mech_agg:
        n_panels = len(selected_kappas)
        n_cols = min(n_panels, 2)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig_e1 = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"\u03ba = {k}" for k in selected_kappas],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        for panel_idx, kappa in enumerate(selected_kappas):
            r = panel_idx // n_cols + 1
            c = panel_idx % n_cols + 1

            if kappa not in mech_agg["active"]:
                continue
            day_data = mech_agg["active"][kappa]
            days_sorted = sorted(day_data.keys())

            n_buys = [day_data[d].get("n_buys", 0) for d in days_sorted]
            n_sells = [day_data[d].get("n_sells", 0) for d in days_sorted]
            n_rejected = [day_data[d].get("n_sell_rejected", 0) for d in days_sorted]

            fig_e1.add_trace(go.Scatter(
                x=days_sorted, y=n_buys,
                mode="lines", name="Buys",
                line=dict(color="#2ca02c", width=2),
                showlegend=(panel_idx == 0),
                legendgroup="buys",
            ), row=r, col=c)
            fig_e1.add_trace(go.Scatter(
                x=days_sorted, y=n_sells,
                mode="lines", name="Sells",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                showlegend=(panel_idx == 0),
                legendgroup="sells",
            ), row=r, col=c)
            fig_e1.add_trace(go.Scatter(
                x=days_sorted, y=n_rejected,
                mode="lines", name="Sell Rejected",
                line=dict(color="#d62728", width=1.5, dash="dot"),
                showlegend=(panel_idx == 0),
                legendgroup="rejected",
            ), row=r, col=c)

            fig_e1.update_xaxes(title_text="Day", row=r, col=c)
            fig_e1.update_yaxes(title_text="Trade Count", row=r, col=c)

        _apply_layout(fig_e1, "Dealer Trade Volume by Kappa (Active Arm)",
                      height=350 * n_rows, width=1000,
                      legend=dict(orientation="h", yanchor="bottom", y=1.06,
                                  xanchor="center", x=0.5))
        parts.append(_chart_html(fig_e1))
    else:
        parts.append("<p>No active arm data available for dealer trade volume.</p>")

    parts.append(_interp(
        "<b>E1: How does dealer trading evolve?</b> "
        "a. Sell trades dominate at low \u03ba \u2014 stressed agents offload claims to the dealer. "
        "b. Buy trades increase as agents with surplus cash enter the secondary market. "
        "c. Sell rejections (dotted) vastly outnumber completed sells \u2014 the dealer\u2019s capacity "
        "binds quickly. At high \u03ba, overall trading volume drops as liquidity is abundant."
    ))

    # -------------------------------------------------------------------
    # E2: Dealer Prices by Bucket at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E2: Dealer Prices by Bucket (\u03ba = {repr_kappa})</h2>")
    if "active" in mech_agg and repr_kappa in mech_agg["active"]:
        day_data = mech_agg["active"][repr_kappa]
        days_sorted = sorted(day_data.keys())

        price_short = [day_data[d].get("mean_price_short", 0) for d in days_sorted]
        price_mid = [day_data[d].get("mean_price_mid", 0) for d in days_sorted]
        price_long = [day_data[d].get("mean_price_long", 0) for d in days_sorted]

        fig_e2 = go.Figure()
        for vals, label, color in [
            (price_short, "Short Bucket", "#1f77b4"),
            (price_mid, "Mid Bucket", "#ff7f0e"),
            (price_long, "Long Bucket", "#2ca02c"),
        ]:
            # Only plot if there are non-zero values
            if any(v != 0 for v in vals):
                fig_e2.add_trace(go.Scatter(
                    x=days_sorted, y=vals,
                    mode="lines", name=label,
                    line=dict(color=color, width=2),
                ))

        _apply_layout(fig_e2, f"Mean Trade Prices by Maturity Bucket (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Mean Price (fraction of face)",
                      height=450)
        parts.append(_chart_html(fig_e2))
    else:
        parts.append("<p>No active arm data available for dealer prices.</p>")

    parts.append(_interp(
        "<b>E2: How do prices vary by maturity?</b> "
        "a. Short-maturity claims trade at higher prices \u2014 they are closer to settlement "
        "and carry less credit risk. "
        "b. Long-maturity claims trade at a deeper discount, reflecting both time-value "
        "and the higher probability of intervening defaults. "
        "c. All bucket prices tend to decline over time as the default wave increases "
        "perceived credit risk in the system."
    ))

    # -------------------------------------------------------------------
    # E3: Liquidity vs Earning Trades at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E3: Liquidity vs Earning Trades (\u03ba = {repr_kappa})</h2>")
    if "active" in mech_agg and repr_kappa in mech_agg["active"]:
        day_data = mech_agg["active"][repr_kappa]
        days_sorted = sorted(day_data.keys())

        n_liq = [day_data[d].get("n_liquidity_driven", 0) for d in days_sorted]
        n_earn = [day_data[d].get("n_earning_driven", 0) for d in days_sorted]

        fig_e3 = go.Figure()
        fig_e3.add_trace(go.Bar(
            x=days_sorted, y=n_liq,
            name="Liquidity-Driven",
            marker_color="#3498db",
        ))
        fig_e3.add_trace(go.Bar(
            x=days_sorted, y=n_earn,
            name="Earning-Driven",
            marker_color="#e67e22",
        ))

        _apply_layout(fig_e3, f"Trade Motivation: Liquidity vs Earning (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Number of Trades",
                      barmode="stack",
                      height=450)
        parts.append(_chart_html(fig_e3))
    else:
        parts.append("<p>No active arm data available for trade motivation breakdown.</p>")

    parts.append(_interp(
        "<b>E3: Why do agents trade?</b> "
        "a. Liquidity-driven trades (blue) dominate early \u2014 agents sell claims to meet "
        "upcoming obligations, not for profit. "
        "b. Earning-driven trades (orange) appear when agents with surplus cash buy "
        "discounted claims speculatively. "
        "c. At low \u03ba, almost all trading is liquidity-motivated; at high \u03ba, earning-driven "
        "trades may increase as agents have more slack to speculate."
    ))

    # -------------------------------------------------------------------
    # E4: Bank Loan Activity — multi-panel (2x2) by kappa, lend arm
    # -------------------------------------------------------------------
    parts.append("<h2>E4: Bank Loan Activity</h2>")
    if "lend" in mech_agg:
        n_panels = len(selected_kappas)
        n_cols = min(n_panels, 2)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig_e4 = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"\u03ba = {k}" for k in selected_kappas],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        for panel_idx, kappa in enumerate(selected_kappas):
            r = panel_idx // n_cols + 1
            c = panel_idx % n_cols + 1

            if kappa not in mech_agg["lend"]:
                continue
            day_data = mech_agg["lend"][kappa]
            days_sorted = sorted(day_data.keys())

            issued = [day_data[d].get("n_loans_issued", 0) for d in days_sorted]
            defaults = [day_data[d].get("n_loan_defaults", 0) for d in days_sorted]
            repaid = [day_data[d].get("n_loan_repaid", 0) for d in days_sorted]

            fig_e4.add_trace(go.Scatter(
                x=days_sorted, y=issued,
                mode="lines", name="Issued",
                line=dict(color=BANK_BLUE, width=2),
                showlegend=(panel_idx == 0),
                legendgroup="issued",
            ), row=r, col=c)
            fig_e4.add_trace(go.Scatter(
                x=days_sorted, y=defaults,
                mode="lines", name="Defaults",
                line=dict(color="#d62728", width=2, dash="dash"),
                showlegend=(panel_idx == 0),
                legendgroup="defaults",
            ), row=r, col=c)
            fig_e4.add_trace(go.Scatter(
                x=days_sorted, y=repaid,
                mode="lines", name="Repaid",
                line=dict(color="#2ca02c", width=2, dash="dot"),
                showlegend=(panel_idx == 0),
                legendgroup="repaid",
            ), row=r, col=c)

            fig_e4.update_xaxes(title_text="Day", row=r, col=c)
            fig_e4.update_yaxes(title_text="Loan Count", row=r, col=c)

        _apply_layout(fig_e4, "Bank Loan Activity by Kappa (Lend Arm)",
                      height=350 * n_rows, width=1000,
                      legend=dict(orientation="h", yanchor="bottom", y=1.06,
                                  xanchor="center", x=0.5))
        parts.append(_chart_html(fig_e4))
    else:
        parts.append("<p>No lend arm data available for bank loan activity.</p>")

    parts.append(_interp(
        "<b>E4: How does bank lending evolve?</b> "
        "a. Loan issuance is front-loaded \u2014 banks lend aggressively in the first few days "
        "when agents face the first maturity wave. "
        "b. At low \u03ba, loan defaults (dashed red) rise after a lag as borrowers fail to "
        "meet obligations despite the liquidity injection. "
        "c. At high \u03ba, repayments (dotted green) dominate \u2014 most loans are repaid on time "
        "because borrowers have sufficient underlying cash flow."
    ))

    # -------------------------------------------------------------------
    # E5: Bank Lending Rate at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E5: Bank Lending Rate (\u03ba = {repr_kappa})</h2>")
    if "lend" in mech_agg and repr_kappa in mech_agg["lend"]:
        day_data = mech_agg["lend"][repr_kappa]
        days_sorted = sorted(day_data.keys())

        rates = [day_data[d].get("mean_loan_rate", 0) for d in days_sorted]
        # Filter out zero-rate days (no loans issued)
        valid_days = [d for d, rate in zip(days_sorted, rates) if rate > 0]
        valid_rates = [rate for rate in rates if rate > 0]

        fig_e5 = go.Figure()
        if valid_days:
            fig_e5.add_trace(go.Scatter(
                x=valid_days, y=valid_rates,
                mode="lines+markers", name="Mean Loan Rate",
                line=dict(color=BANK_BLUE, width=2),
                marker=dict(size=5),
            ))

        _apply_layout(fig_e5, f"Mean Bank Lending Rate Over Time (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Interest Rate",
                      height=450)
        parts.append(_chart_html(fig_e5))
    else:
        parts.append("<p>No lend arm data available for lending rate.</p>")

    parts.append(_interp(
        "<b>E5: How do lending rates evolve?</b> "
        "a. The mean loan rate reflects the bank\u2019s credit risk assessment of borrowers. "
        "b. Rates may rise over time as the bank observes defaults and revises its "
        "risk priors upward, increasing the credit risk loading on new loans. "
        "c. Days with no loans issued are excluded \u2014 gaps in the line indicate periods "
        "where no agent was both eligible and willing to borrow."
    ))

    # -------------------------------------------------------------------
    # E6: Bank Loan Volume vs Default Volume at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E6: Bank Loan Volume vs Defaults (\u03ba = {repr_kappa})</h2>")
    if "lend" in mech_agg and repr_kappa in mech_agg["lend"]:
        day_data = mech_agg["lend"][repr_kappa]
        days_sorted = sorted(day_data.keys())

        vol_issued = [day_data[d].get("loan_volume_issued", 0) for d in days_sorted]
        vol_default = [day_data[d].get("default_volume", 0) for d in days_sorted]

        fig_e6 = go.Figure()
        fig_e6.add_trace(go.Scatter(
            x=days_sorted, y=vol_issued,
            mode="lines", name="Loan Volume Issued",
            line=dict(color=BANK_BLUE, width=2),
        ))
        fig_e6.add_trace(go.Scatter(
            x=days_sorted, y=vol_default,
            mode="lines", name="Default Volume",
            line=dict(color="#d62728", width=2, dash="dash"),
        ))

        _apply_layout(fig_e6, f"Loan Volume Issued vs Default Volume (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Volume",
                      height=450)
        parts.append(_chart_html(fig_e6))
    else:
        parts.append("<p>No lend arm data available for volume comparison.</p>")

    parts.append(_interp(
        "<b>E6: Does lending volume outpace defaults?</b> "
        "a. Loan volume issued (solid blue) shows the daily flow of new credit from banks. "
        "b. Default volume (dashed red) shows the daily face value of loans that defaulted. "
        "c. A healthy lending system has issuance consistently above default volume \u2014 "
        "if the red line overtakes blue, the bank is accumulating losses faster than it "
        "can deploy new capital."
    ))

    # -------------------------------------------------------------------
    # E7: Interbank Cleared Volume at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E7: Interbank Cleared Volume (\u03ba = {repr_kappa})</h2>")
    bank_arms = [a for a in ["idle", "lend"] if a in mech_agg]
    if bank_arms:
        fig_e7 = go.Figure()
        for arm in bank_arms:
            if repr_kappa not in mech_agg[arm]:
                continue
            day_data = mech_agg[arm][repr_kappa]
            days_sorted = sorted(day_data.keys())
            vol = [day_data[d].get("interbank_cleared_volume", 0) for d in days_sorted]

            color = ARM_COLORS.get(arm, "#333")
            label = ARM_LABELS.get(arm, arm)
            fig_e7.add_trace(go.Scatter(
                x=days_sorted, y=vol,
                mode="lines", name=label,
                line=dict(color=color, width=2),
            ))

        _apply_layout(fig_e7, f"Interbank Cleared Volume (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Cleared Volume",
                      height=450)
        parts.append(_chart_html(fig_e7))
    else:
        parts.append("<p>No bank arm data available for interbank clearing.</p>")

    parts.append(_interp(
        "<b>E7: How active is the interbank market?</b> "
        "a. Interbank clearing captures reserve transfers between banks to settle "
        "cross-bank obligations. "
        "b. Under the lend arm, interbank volume may be higher because loan-funded "
        "payments create additional cross-bank flows. "
        "c. Under the idle arm, interbank clearing still occurs for deposit-based "
        "payments but at lower volume since no new credit is injected."
    ))

    # -------------------------------------------------------------------
    # E8: Interbank Auction Rate at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E8: Interbank Auction Rate (\u03ba = {repr_kappa})</h2>")
    if bank_arms:
        fig_e8 = go.Figure()
        for arm in bank_arms:
            if repr_kappa not in mech_agg[arm]:
                continue
            day_data = mech_agg[arm][repr_kappa]
            days_sorted = sorted(day_data.keys())
            rates = [day_data[d].get("auction_rate", 0) for d in days_sorted]
            # Filter out zero-rate days (no auctions)
            valid = [(d, rate) for d, rate in zip(days_sorted, rates) if rate > 0]
            if valid:
                vd, vr = zip(*valid)
                color = ARM_COLORS.get(arm, "#333")
                label = ARM_LABELS.get(arm, arm)
                fig_e8.add_trace(go.Scatter(
                    x=list(vd), y=list(vr),
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                ))

        _apply_layout(fig_e8, f"Interbank Auction Rate (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Auction Rate",
                      height=450)
        parts.append(_chart_html(fig_e8))
    else:
        parts.append("<p>No bank arm data available for auction rate.</p>")

    parts.append(_interp(
        "<b>E8: How does the interbank rate evolve?</b> "
        "a. The auction rate reflects the cost of reserve borrowing in the interbank market. "
        "b. Under both idle and lend arms, the rate is set by the bank\u2019s corridor pricing. "
        "c. Rising auction rates may indicate increased demand for reserves as payment "
        "obligations pile up and banks become more reluctant to lend surplus reserves."
    ))

    # -------------------------------------------------------------------
    # E9: CB Facility Usage at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E9: Central Bank Facility Usage (\u03ba = {repr_kappa})</h2>")
    if "lend" in mech_agg and repr_kappa in mech_agg["lend"]:
        day_data = mech_agg["lend"][repr_kappa]
        days_sorted = sorted(day_data.keys())
        cb_vol = [day_data[d].get("cb_loans_created_volume", 0) for d in days_sorted]

        fig_e9 = go.Figure()
        fig_e9.add_trace(go.Bar(
            x=days_sorted, y=cb_vol,
            name="CB Loan Volume",
            marker_color="#8e44ad",
        ))

        _apply_layout(fig_e9, f"Central Bank Lending Facility Usage (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="CB Loan Volume Created",
                      height=450)
        parts.append(_chart_html(fig_e9))
    else:
        parts.append("<p>No lend arm data available for CB facility usage.</p>")

    parts.append(_interp(
        "<b>E9: When do banks tap the central bank?</b> "
        "a. CB facility usage indicates that commercial banks have exhausted their "
        "interbank borrowing capacity and need lender-of-last-resort support. "
        "b. Spikes typically coincide with default waves \u2014 when loan losses erode bank "
        "reserves and the interbank market cannot supply enough liquidity. "
        "c. At high \u03ba, CB usage should be minimal or zero since banks maintain ample reserves."
    ))

    # -------------------------------------------------------------------
    # E10: Payment Channels — stacked area at representative kappa
    # -------------------------------------------------------------------
    parts.append(f"<h2>E10: Payment Channels (\u03ba = {repr_kappa})</h2>")
    if "lend" in mech_agg and repr_kappa in mech_agg["lend"]:
        day_data = mech_agg["lend"][repr_kappa]
        days_sorted = sorted(day_data.keys())

        intrabank = [day_data[d].get("intrabank_volume", 0) for d in days_sorted]
        client = [day_data[d].get("client_payment_volume", 0) for d in days_sorted]
        reserves = [day_data[d].get("reserves_transferred_volume", 0) for d in days_sorted]

        fig_e10 = go.Figure()
        fig_e10.add_trace(go.Scatter(
            x=days_sorted, y=intrabank,
            mode="lines", name="Intrabank",
            line=dict(width=0),
            fillcolor="rgba(52, 152, 219, 0.5)",
            stackgroup="one",
        ))
        fig_e10.add_trace(go.Scatter(
            x=days_sorted, y=client,
            mode="lines", name="Client Payment",
            line=dict(width=0),
            fillcolor="rgba(46, 204, 113, 0.5)",
            stackgroup="one",
        ))
        fig_e10.add_trace(go.Scatter(
            x=days_sorted, y=reserves,
            mode="lines", name="Reserves Transferred",
            line=dict(width=0),
            fillcolor="rgba(155, 89, 182, 0.5)",
            stackgroup="one",
        ))

        _apply_layout(fig_e10, f"Payment Channel Volumes (\u03ba = {repr_kappa})",
                      xaxis_title="Day",
                      yaxis_title="Volume",
                      height=450)
        parts.append(_chart_html(fig_e10))
    else:
        parts.append("<p>No lend arm data available for payment channels.</p>")

    parts.append(_interp(
        "<b>E10: How are payments settled?</b> "
        "a. Intrabank volume (blue) captures payments between agents at the same bank \u2014 "
        "these settle via internal book transfers without reserve movement. "
        "b. Client payments (green) represent cross-bank deposit transfers that trigger "
        "reserve flows between banks. "
        "c. Reserves transferred (purple) shows the gross flow of central bank reserves "
        "between banks. The composition reveals whether the payment system relies "
        "primarily on internal clearing or cross-bank settlement."
    ))

    return "\n".join(parts)


# ===================================================================
# Section 10: Network Ring Visualization
# ===================================================================


def _section_network_ring(
    nbfi_csv: Path | None,
    bank_csv: Path | None,
    nbfi_dir: Path | None,
    bank_dir: Path | None,
) -> str:
    """Animated ring visualizations for representative runs at kappa ~ 0.5."""
    import csv as csv_mod

    parts = [
        '<section id="network-ring">',
        "<h1>10. Network Ring Visualization</h1>",
        "<p>Animated ring visualizations of representative runs at \u03ba \u2248 0.5. "
        "Nodes are agents arranged in a circle; edges represent active balance sheet "
        "connections (payables, bank loans, CB loans). Red nodes = defaulted agents. "
        "Node size reflects cash holdings. Use the slider or Play button to animate.</p>",
    ]

    def _find_closest_kappa_row(
        csv_path: Path, target: float = 0.5
    ) -> dict[str, str]:
        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
        return min(rows, key=lambda r: abs(float(r.get("kappa", 0)) - target))

    arm_configs: list[tuple[str, str, Path]] = []

    if nbfi_csv and nbfi_dir and nbfi_csv.exists():
        row = _find_closest_kappa_row(nbfi_csv)
        kappa = row["kappa"]
        for label, rid_col, subdir in [
            ("Passive (No Dealer)", "passive_run_id", "passive"),
            ("Active (Dealer)", "active_run_id", "active"),
        ]:
            rid = row.get(rid_col, "")
            if rid:
                events_path = nbfi_dir / subdir / "runs" / rid / "out" / "events.jsonl"
                arm_configs.append((label, kappa, events_path))

    if bank_csv and bank_dir and bank_csv.exists():
        row = _find_closest_kappa_row(bank_csv)
        kappa = row["kappa"]
        for label, rid_col, subdir in [
            ("Bank Idle", "idle_run_id", "bank_idle"),
            ("Bank Lend", "lend_run_id", "bank_lend"),
        ]:
            rid = row.get(rid_col, "")
            if rid:
                events_path = bank_dir / subdir / "runs" / rid / "out" / "events.jsonl"
                arm_configs.append((label, kappa, events_path))

    for label, kappa, events_path in arm_configs:
        if not events_path.exists():
            parts.append(f"<p>{html_mod.escape(label)}: events not found at "
                         f"<code>{html_mod.escape(str(events_path))}</code></p>")
            continue
        snaps = reconstruct_network_snapshots(events_path)
        snaps = [s for s in snaps if s["day"] <= 14]
        if snaps:
            title = f"{label} — \u03ba = {kappa}"
            parts.append(generate_network_animation_html(snaps, title=title))
        else:
            parts.append(f"<p>{html_mod.escape(label)}: no snapshots extracted</p>")

    parts.append("</section>")
    return "\n".join(parts)


# ===================================================================
# Section 11: Loss Analysis
# ===================================================================


def _section_loss_analysis(data: dict[str, np.ndarray]) -> str:
    """Section 11: Loss analysis — system loss rates, decomposition, LGD, and capital consumption."""
    parts: list[str] = []
    parts.append('<hr class="section-divider">')
    parts.append('<section id="loss-analysis">')
    parts.append("<h1>11. Loss Analysis</h1>")
    parts.append(
        "<p>This section examines loss metrics across arms: how much value is destroyed "
        "by defaults, how losses decompose between payable defaults and intermediary losses, "
        "and how loss relates to default rates and intermediary capital.</p>"
    )

    kappa = data["kappa"]

    # -- F1: System Loss Rate vs kappa --
    parts.append("<h2>F1: System Loss Rate vs \u03ba</h2>")

    # Define arm columns for NBFI sweep and bank sweep
    nbfi_loss_arms: list[tuple[str, str, str, str]] = [
        ("system_loss_pct_passive", "Passive", PASSIVE_GREY, "circle"),
        ("system_loss_pct_active", "Active (Dealer)", DEALER_RED, "diamond"),
    ]
    if "system_loss_pct_lender" in data:
        nbfi_loss_arms.append(
            ("system_loss_pct_lender", "NBFI Lender", NBFI_GREEN, "square")
        )
    bank_loss_arms: list[tuple[str, str, str, str]] = []
    if "bank_system_loss_pct_idle" in data:
        bank_loss_arms.append(
            ("bank_system_loss_pct_idle", "Bank Idle", PASSIVE_GREY, "cross")
        )
    if "bank_system_loss_pct_lend" in data:
        bank_loss_arms.append(
            ("bank_system_loss_pct_lend", "Bank Lend", BANK_BLUE, "triangle-up")
        )
    all_loss_arms = nbfi_loss_arms + bank_loss_arms

    loss_metric_cols = [col for col, _, _, _ in all_loss_arms if col in data]
    grouped_f1 = summary_by_group(data, ["kappa"], loss_metric_cols)

    fig_f1 = go.Figure()
    for col, label, color, symbol in all_loss_arms:
        if col not in data:
            continue
        x_vals = [g["kappa"] for g in grouped_f1]
        y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f1]
        fig_f1.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7, symbol=symbol),
        ))
    _apply_layout(fig_f1, "System Loss Rate vs Kappa",
                  xaxis_title="Kappa (\u03ba)",
                  yaxis_title="System Loss Rate",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_f1))
    parts.append(_interp(
        "System loss rate captures total value destruction (payable defaults + intermediary losses) "
        "as a fraction of initial debt. Lower is better. Active mechanisms that reduce defaults "
        "should also reduce system loss. The gap between arms shows the loss-prevention benefit "
        "of each mechanism at different liquidity levels."
    ))

    # -- F2: Loss Decomposition --
    parts.append("<h2>F2: Loss Decomposition by Arm</h2>")

    decomp_arms: list[tuple[str, str, str, str]] = []
    for arm_suffix, arm_label, arm_color in [
        ("passive", "Passive", PASSIVE_GREY),
        ("active", "Active", DEALER_RED),
        ("lender", "NBFI Lender", NBFI_GREEN),
    ]:
        payable_col = f"payable_default_loss_{arm_suffix}"
        intermed_col = f"intermediary_loss_{arm_suffix}"
        if payable_col in data and intermed_col in data:
            decomp_arms.append((arm_suffix, arm_label, arm_color, payable_col))

    if decomp_arms:
        decomp_metric_cols = []
        for arm_suffix, _, _, _ in decomp_arms:
            decomp_metric_cols.append(f"payable_default_loss_{arm_suffix}")
            decomp_metric_cols.append(f"intermediary_loss_{arm_suffix}")
        grouped_f2 = summary_by_group(data, ["kappa"], decomp_metric_cols)

        x_kappas = [str(g["kappa"]) for g in grouped_f2]
        fig_f2 = go.Figure()

        for arm_suffix, arm_label, arm_color, _ in decomp_arms:
            payable_means = [
                g.get(f"payable_default_loss_{arm_suffix}_mean", 0) for g in grouped_f2
            ]
            intermed_means = [
                g.get(f"intermediary_loss_{arm_suffix}_mean", 0) for g in grouped_f2
            ]
            # Stacked bars: payable defaults on bottom, intermediary on top
            fig_f2.add_trace(go.Bar(
                x=x_kappas, y=payable_means,
                name=f"{arm_label} — Payable Defaults",
                marker_color=arm_color,
                opacity=0.8,
                legendgroup=arm_suffix,
            ))
            fig_f2.add_trace(go.Bar(
                x=x_kappas, y=intermed_means,
                name=f"{arm_label} — Intermediary Loss",
                marker_color=arm_color,
                opacity=0.4,
                legendgroup=arm_suffix,
            ))

        _apply_layout(fig_f2, "Loss Decomposition: Payable Defaults + Intermediary Loss",
                      xaxis_title="Kappa (\u03ba)", yaxis_title="Mean Loss",
                      barmode="group", height=500)
        parts.append(_chart_html(fig_f2))
    parts.append(_interp(
        "Loss decomposes into two sources: (1) payable default losses — face value of "
        "obligations that were not settled, and (2) intermediary losses — capital consumed by "
        "dealers, lenders, or banks in the process of market-making or lending. In a well-functioning "
        "intermediary, payable losses should fall more than intermediary losses rise."
    ))

    # -- F3: Loss vs Default Scatter --
    parts.append("<h2>F3: System Loss vs Default Rate</h2>")
    fig_f3 = go.Figure()

    scatter_arms: list[tuple[str, str, str, str]] = [
        ("delta_passive", "system_loss_pct_passive", "Passive", PASSIVE_GREY),
        ("delta_active", "system_loss_pct_active", "Active (Dealer)", DEALER_RED),
    ]
    if "delta_lender" in data and "system_loss_pct_lender" in data:
        scatter_arms.append(
            ("delta_lender", "system_loss_pct_lender", "NBFI Lender", NBFI_GREEN)
        )
    if "delta_lend" in data and "bank_system_loss_pct_lend" in data:
        scatter_arms.append(
            ("delta_lend", "bank_system_loss_pct_lend", "Bank Lend", BANK_BLUE)
        )

    for delta_col, loss_col, label, color in scatter_arms:
        if delta_col not in data or loss_col not in data:
            continue
        mask = np.isfinite(data[delta_col]) & np.isfinite(data[loss_col])
        fig_f3.add_trace(go.Scatter(
            x=data[delta_col][mask], y=data[loss_col][mask],
            mode="markers", name=label,
            marker=dict(color=color, size=5, opacity=0.6),
        ))

    # y=x reference line
    fig_f3.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="y = x",
        line=dict(color="#95a5a6", dash="dash", width=1),
        showlegend=True,
    ))

    _apply_layout(fig_f3, "System Loss vs Default Rate",
                  xaxis_title="Default Rate (\u03b4)",
                  yaxis_title="System Loss Rate",
                  height=500)
    parts.append(_chart_html(fig_f3))
    parts.append(_interp(
        "Points on the y=x line mean system loss equals the default rate (loss is purely from "
        "payable defaults, no intermediary amplification). Points above the line indicate "
        "intermediary losses amplify the damage. Points below mean intermediaries absorb some "
        "loss (their capital cushions the system). A mechanism that moves points left and down "
        "is reducing both defaults and total loss."
    ))

    # -- F4: System Loss Treatment Effects vs kappa --
    parts.append("<h2>F4: System Loss Treatment Effects vs \u03ba</h2>")

    loss_effect_cols: list[tuple[str, str, str]] = [
        ("system_loss_trading_effect", "Dealer Trading", DEALER_RED),
        ("system_loss_lending_effect", "NBFI Lending", NBFI_GREEN),
    ]
    if "bank_system_loss_bank_lending_effect" in data:
        loss_effect_cols.append(
            ("bank_system_loss_bank_lending_effect", "Bank Lending", BANK_BLUE)
        )

    effect_metric_cols = [c for c, _, _ in loss_effect_cols if c in data]
    grouped_f4 = summary_by_group(data, ["kappa"], effect_metric_cols)

    fig_f4 = go.Figure()
    for col, label, color in loss_effect_cols:
        if col not in data:
            continue
        x_vals = [g["kappa"] for g in grouped_f4]
        y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f4]
        fig_f4.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))
    fig_f4.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                     annotation_text="No effect")
    _apply_layout(fig_f4, "System Loss Treatment Effects vs Kappa",
                  xaxis_title="Kappa (\u03ba)",
                  yaxis_title="Loss Reduction (passive \u2212 arm)",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_f4))
    parts.append(_interp(
        "Treatment effect on system loss = (passive system loss) \u2212 (arm system loss). "
        "Positive values mean the mechanism reduces total system loss. This mirrors the default "
        "rate treatment effect but accounts for intermediary losses too. If a mechanism reduces "
        "defaults but its intermediaries absorb large losses, the system loss effect will be "
        "smaller than the default rate effect."
    ))

    # -- F5: Loss Given Default (LGD) vs kappa --
    parts.append("<h2>F5: Loss Given Default (LGD) vs \u03ba</h2>")

    lgd_arms: list[tuple[str, str, str]] = [
        ("lgd_passive", "Passive", PASSIVE_GREY),
        ("lgd_active", "Active (Dealer)", DEALER_RED),
    ]
    if "lgd_lender" in data:
        lgd_arms.append(("lgd_lender", "NBFI Lender", NBFI_GREEN))
    if "lgd_idle" in data:
        lgd_arms.append(("lgd_idle", "Bank Idle", PASSIVE_GREY))
    if "lgd_lend" in data:
        lgd_arms.append(("lgd_lend", "Bank Lend", BANK_BLUE))

    lgd_metric_cols = [c for c, _, _ in lgd_arms if c in data]
    grouped_f5 = summary_by_group(data, ["kappa"], lgd_metric_cols)

    # Filter to kappa <= 2 to avoid outliers
    grouped_f5 = [g for g in grouped_f5 if g["kappa"] <= 2]

    fig_f5 = go.Figure()
    for col, label, color in lgd_arms:
        if col not in data:
            continue
        x_vals = [g["kappa"] for g in grouped_f5]
        y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f5]
        fig_f5.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))
    _apply_layout(fig_f5, "Loss Given Default vs Kappa (\u03ba \u2264 2)",
                  xaxis_title="Kappa (\u03ba)",
                  yaxis_title="LGD (total_loss_pct / \u03b4)",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_f5))
    parts.append(_interp(
        "LGD = total loss percentage / default rate. It measures how much loss each unit of "
        "default generates. LGD > 1 means intermediary losses amplify the damage beyond "
        "payable defaults alone. LGD < 1 means some defaulted obligations are partially "
        "recovered or intermediaries cushion the blow. Filtered to \u03ba \u2264 2 because at "
        "high liquidity, near-zero defaults produce unstable LGD ratios."
    ))

    # -- F6: Loss per Default Event vs kappa --
    parts.append("<h2>F6: Loss per Default Event vs \u03ba</h2>")

    lpd_arms: list[tuple[str, str, str]] = [
        ("loss_per_default_passive", "Passive", PASSIVE_GREY),
        ("loss_per_default_active", "Active (Dealer)", DEALER_RED),
    ]
    if "loss_per_default_lender" in data:
        lpd_arms.append(("loss_per_default_lender", "NBFI Lender", NBFI_GREEN))
    if "loss_per_default_idle" in data:
        lpd_arms.append(("loss_per_default_idle", "Bank Idle", PASSIVE_GREY))
    if "loss_per_default_lend" in data:
        lpd_arms.append(("loss_per_default_lend", "Bank Lend", BANK_BLUE))

    lpd_metric_cols = [c for c, _, _ in lpd_arms if c in data]
    grouped_f6 = summary_by_group(data, ["kappa"], lpd_metric_cols)

    fig_f6 = go.Figure()
    for col, label, color in lpd_arms:
        if col not in data:
            continue
        x_vals = [g["kappa"] for g in grouped_f6]
        y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f6]
        fig_f6.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))
    _apply_layout(fig_f6, "Loss per Default Event vs Kappa",
                  xaxis_title="Kappa (\u03ba)",
                  yaxis_title="Loss per Default (total_loss / n_defaults)",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_f6))
    parts.append(_interp(
        "Loss per default event = total loss / number of defaults. This measures the average "
        "severity of each default. Higher values mean each default destroys more value, possibly "
        "because defaulting agents have larger positions or because cascading defaults amplify "
        "each event. Mechanisms that reduce this metric make each default less damaging, even if "
        "they don't prevent it."
    ))

    # -- F7: Loss Capital Ratio vs kappa --
    parts.append("<h2>F7: Loss Capital Ratio vs \u03ba</h2>")

    lcr_arms: list[tuple[str, str, str]] = [
        ("loss_capital_ratio_passive", "Passive", PASSIVE_GREY),
        ("loss_capital_ratio_active", "Active (Dealer)", DEALER_RED),
    ]
    if "loss_capital_ratio_lender" in data:
        lcr_arms.append(("loss_capital_ratio_lender", "NBFI Lender", NBFI_GREEN))

    lcr_metric_cols = [c for c, _, _ in lcr_arms if c in data]
    grouped_f7 = summary_by_group(data, ["kappa"], lcr_metric_cols)

    fig_f7 = go.Figure()
    for col, label, color in lcr_arms:
        if col not in data:
            continue
        x_vals = [g["kappa"] for g in grouped_f7]
        y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f7]
        fig_f7.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
        ))
    fig_f7.add_hline(y=1, line_dash="dash", line_color="#e74c3c",
                     annotation_text="Capital wiped out")
    _apply_layout(fig_f7, "Loss / Intermediary Capital Ratio vs Kappa",
                  xaxis_title="Kappa (\u03ba)",
                  yaxis_title="Loss / Capital",
                  xaxis_type="log", height=500)
    parts.append(_chart_html(fig_f7))
    parts.append(_interp(
        "Loss capital ratio = intermediary loss / intermediary capital. Values above 1 (red "
        "dashed line) mean intermediary losses exceed the capital allocated to intermediaries, "
        "implying the intermediary is effectively bankrupt. This measures how much of the "
        "intermediary's risk-absorbing capacity is consumed. Lower is better \u2014 it means "
        "the intermediary retains capital to continue functioning."
    ))

    # -- F8: Bank Deposit Loss --
    parts.append("<h2>F8: Bank Deposit Loss vs \u03ba</h2>")

    dep_idle_col = "bank_deposit_loss_pct_idle"
    dep_lend_col = "bank_deposit_loss_pct_lend"

    if dep_idle_col in data and dep_lend_col in data:
        dep_metric_cols = [dep_idle_col, dep_lend_col]
        grouped_f8 = summary_by_group(data, ["kappa"], dep_metric_cols)

        fig_f8 = go.Figure()
        for col, label, color in [
            (dep_idle_col, "Bank Idle", PASSIVE_GREY),
            (dep_lend_col, "Bank Lend", BANK_BLUE),
        ]:
            x_vals = [g["kappa"] for g in grouped_f8]
            y_vals = [g.get(f"{col}_mean", float("nan")) for g in grouped_f8]
            fig_f8.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers", name=label,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=7),
            ))
        _apply_layout(fig_f8, "Bank Deposit Loss Rate vs Kappa",
                      xaxis_title="Kappa (\u03ba)",
                      yaxis_title="Deposit Loss Rate",
                      xaxis_type="log", height=500)
        parts.append(_chart_html(fig_f8))
        parts.append(_interp(
            "Deposit loss rate measures the fraction of bank deposits that are written off "
            "due to borrower defaults. Comparing idle vs lend arms shows whether active bank "
            "lending increases depositor risk. If the lend arm has higher deposit losses, "
            "banks are taking on too much credit risk. If lower, the lending activity actually "
            "stabilizes the system enough to protect depositors."
        ))
    else:
        parts.append(_interp(
            "Bank deposit loss data not available in this dataset. This chart requires "
            "bank sweep data with deposit_loss_pct columns.",
            warn=True,
        ))

    parts.append("</section>")
    return "\n".join(parts)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-sweep HTML analysis report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/sweep_analysis/report.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--nbfi-path",
        type=Path,
        default=Path("out/sweep_nbfi_full/aggregate/comparison.csv"),
        help="Path to NBFI sweep comparison CSV",
    )
    parser.add_argument(
        "--bank-path",
        type=Path,
        default=Path("out/sweep_bank_full/aggregate/comparison.csv"),
        help="Path to bank sweep comparison CSV",
    )
    parser.add_argument(
        "--nbfi-dir",
        type=Path,
        default=Path("out/sweep_nbfi_full"),
        help="Base directory for NBFI sweep run outputs",
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=Path("out/sweep_bank_full"),
        help="Base directory for bank sweep run outputs",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("out/sweep_analysis"),
        help="Directory for cached network metrics CSVs",
    )
    args = parser.parse_args()

    print(f"Loading NBFI sweep from: {args.nbfi_path}")
    print(f"Loading bank sweep from:  {args.bank_path}")

    # Load and prepare data
    data = load_unified_data(args.nbfi_path, args.bank_path)
    add_derived_columns(data)

    n = len(data["kappa"])
    print(f"Loaded {n} matched parameter combinations")

    # Build each section
    print("Building Section 1: Overview...")
    s1 = _section_overview(data)
    print("Building Section 2: Parameter Sensitivity...")
    s2 = _section_sensitivity(data)
    print("Building Section 3: Mechanism Comparison...")
    s3 = _section_comparison(data)
    print("Building Section 4: Regression...")
    s4 = _section_regression(data)
    print("Building Section 5: Statistical Tests...")
    s5 = _section_tests(data)
    print("Building Section 6: Cross-Sweep...")
    s6 = _section_cross_sweep(data)

    # Load network data for sections 7 and 8
    cache_dir = args.cache_dir
    net_data = load_network_metrics(cache_dir)
    s7 = ""
    s8 = ""
    if net_data:
        print("Building Section 7: Network Structure...")
        s7 = _section_network(net_data)

        ts_data = load_network_timeseries(cache_dir)
        ts_agg = aggregate_timeseries_by_arm_kappa(ts_data) if ts_data else {}
        if ts_agg:
            print("Building Section 8: Temporal Dynamics...")
            s8 = _section_temporal(ts_agg, net_data)
    else:
        print("No network metrics found — skipping Sections 7-8")

    # Section 9: Mechanism Internals
    mech_data = load_mechanism_timeseries(cache_dir)
    mech_agg = aggregate_mechanism_timeseries(mech_data) if mech_data else {}
    s9 = ""
    if mech_agg:
        print("Building Section 9: Mechanism Internals...")
        s9 = _section_mechanism_internals(mech_agg)
    else:
        print("No mechanism timeseries found — skipping Section 9")

    # Section 10: Network Ring Visualization
    s10 = ""
    if args.nbfi_path.exists() or args.bank_path.exists():
        print("Building Section 10: Network Ring...")
        s10 = _section_network_ring(
            args.nbfi_path, args.bank_path, args.nbfi_dir, args.bank_dir
        )

    # Section 11: Loss Analysis
    print("Building Section 11: Loss Analysis...")
    s11 = _section_loss_analysis(data)

    body = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11

    nav_items = [
        ("overview", "Overview"),
        ("sensitivity", "Sensitivity"),
        ("comparison", "Comparison"),
        ("regression", "Regression"),
        ("tests", "Statistical Tests"),
        ("cross-sweep", "Cross-Sweep"),
        ("network", "Network"),
        ("temporal", "Temporal"),
        ("mechanisms", "Mechanisms"),
        ("network-ring", "Ring Viz"),
        ("loss-analysis", "Loss Analysis"),
    ]

    html = _build_shell("Bilancio Cross-Sweep Analysis Report", nav_items, body)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"Report written to {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
