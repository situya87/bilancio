#!/usr/bin/env python3
"""Generate a comprehensive cross-mechanism narrative report from three-way sweep data.

Reads analysis outputs from out/three_way/ and produces a structured HTML report
(article style, not dashboard) with embedded Plotly mini-charts.

Usage:
    uv run python scripts/run_narrative_report.py

Output:
    out/three_way/analysis/narrative_report.html
"""

from __future__ import annotations

import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "out" / "three_way"
ANALYSIS_DIR = BASE_DIR / "analysis"
OUTPUT_PATH = ANALYSIS_DIR / "narrative_report.html"

CSV_PATH = BASE_DIR / "three_way_comparison.csv"
DEALER_SUMMARY = ANALYSIS_DIR / "dealer" / "stats_summary.json"
BANK_SUMMARY = ANALYSIS_DIR / "bank" / "stats_summary.json"
NBFI_SUMMARY = ANALYSIS_DIR / "nbfi" / "stats_summary.json"
DEALER_SENSITIVITY = ANALYSIS_DIR / "dealer" / "stats_sensitivity.json"
BANK_SENSITIVITY = ANALYSIS_DIR / "bank" / "stats_sensitivity.json"
NBFI_SENSITIVITY = ANALYSIS_DIR / "nbfi" / "stats_sensitivity.json"
BANK_LENDING = ANALYSIS_DIR / "bank" / "bank_lending_summary.json"
NBFI_LENDING = ANALYSIS_DIR / "nbfi" / "nbfi_lending_summary.json"

# ---------------------------------------------------------------------------
# Mechanism display metadata
# ---------------------------------------------------------------------------

MECHANISM_COLORS = {
    "Bank": "#2980b9",
    "Dealer": "#e67e22",
    "NBFI": "#27ae60",
}
MECHANISM_ORDER = ["Bank", "Dealer", "NBFI"]

# Key kappa levels for the comparison table
KEY_KAPPAS = [0.25, 1.0, 4.0]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Cast numeric fields
            for key in row:
                if key == "mechanism":
                    continue
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


def compute_mechanism_stats(rows: list[dict]) -> dict[str, dict]:
    """Compute aggregate statistics per mechanism from the comparison CSV."""
    by_mech: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mech[r["mechanism"]].append(r)

    stats = {}
    for mech, mech_rows in by_mech.items():
        effects = [r["effect_mean"] for r in mech_rows]
        n_positive = sum(1 for e in effects if e > 0)
        n_negative = sum(1 for e in effects if e < 0)
        n_zero = sum(1 for e in effects if e == 0)
        mean_effect = sum(effects) / len(effects) if effects else 0
        sorted_effects = sorted(effects)
        n = len(sorted_effects)
        median_effect = (
            sorted_effects[n // 2]
            if n % 2 == 1
            else (sorted_effects[n // 2 - 1] + sorted_effects[n // 2]) / 2
        )
        max_effect = max(effects) if effects else 0
        min_effect = min(effects) if effects else 0
        stats[mech] = {
            "mean_effect": mean_effect,
            "median_effect": median_effect,
            "max_effect": max_effect,
            "min_effect": min_effect,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "n_cells": len(mech_rows),
        }
    return stats


def effects_by_kappa(rows: list[dict]) -> dict[str, dict[float, float]]:
    """Compute mean effect per kappa for each mechanism (averaged over c and mu)."""
    # Group by (mechanism, kappa)
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["mechanism"], r["kappa"])].append(r["effect_mean"])

    result: dict[str, dict[float, float]] = defaultdict(dict)
    for (mech, kappa), effects in groups.items():
        result[mech][kappa] = sum(effects) / len(effects)
    return dict(result)


def effects_at_key_kappas(
    kappa_effects: dict[str, dict[float, float]],
) -> dict[float, dict[str, float]]:
    """Extract effects at the key kappa levels for the comparison table."""
    table: dict[float, dict[str, float]] = {}
    for kappa in KEY_KAPPAS:
        table[kappa] = {}
        for mech in MECHANISM_ORDER:
            mech_kappas = kappa_effects.get(mech, {})
            table[kappa][mech] = mech_kappas.get(kappa, 0.0)
    return table


def find_best_kappa_range(kappa_effects: dict[float, float]) -> str:
    """Find the kappa range where a mechanism is most effective."""
    if not kappa_effects:
        return "n/a"
    sorted_kappas = sorted(kappa_effects.items(), key=lambda x: x[1], reverse=True)
    # Top third of kappas by effect
    top_n = max(1, len(sorted_kappas) // 3)
    top_kappas = sorted([k for k, _ in sorted_kappas[:top_n]])
    if len(top_kappas) == 1:
        return f"{top_kappas[0]}"
    return f"{top_kappas[0]}-{top_kappas[-1]}"


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------


def extract_sensitivity_mu_star(
    sens_data: dict, target: str = "delta_active"
) -> dict[str, float]:
    """Extract mu_star values by parameter from sensitivity JSON."""
    result = {}
    entries = sens_data.get(target, sens_data.get("delta_passive", []))
    for entry in entries:
        result[entry["parameter"]] = entry["mu_star"]
    return result


# ---------------------------------------------------------------------------
# Plotly mini-chart generators (return HTML string with <div>)
# ---------------------------------------------------------------------------


def _plotly_bar_chart(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    y_label: str = "Treatment Effect",
    height: int = 300,
    width: int = 500,
) -> str:
    """Generate a Plotly bar chart as an HTML div."""
    chart_id = title.replace(" ", "_").replace(":", "_").lower()[:30]
    # Build traces JSON
    trace = {
        "x": labels,
        "y": [round(v, 4) for v in values],
        "type": "bar",
        "marker": {"color": colors},
        "text": [f"{v:.3f}" for v in values],
        "textposition": "outside",
        "textfont": {"size": 12},
    }
    layout = {
        "title": {"text": title, "font": {"size": 14, "family": "Lato, sans-serif"}},
        "yaxis": {
            "title": y_label,
            "gridcolor": "#e0e0e0",
            "zeroline": True,
            "zerolinecolor": "#888",
        },
        "xaxis": {"tickfont": {"size": 12}},
        "height": height,
        "width": width,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "Lato, sans-serif"},
    }
    return f"""
    <div id="chart_{chart_id}" style="margin: 1.5em auto; max-width: {width}px;"></div>
    <script>
    Plotly.newPlot('chart_{chart_id}', [{json.dumps(trace)}], {json.dumps(layout)},
        {{responsive: true, displayModeBar: false}});
    </script>
    """


def _plotly_line_chart(
    kappa_effects: dict[str, dict[float, float]],
    title: str,
    height: int = 300,
    width: int = 600,
) -> str:
    """Generate a multi-line Plotly chart (effect vs kappa by mechanism)."""
    chart_id = title.replace(" ", "_").replace(":", "_").lower()[:30]
    traces = []
    for mech in MECHANISM_ORDER:
        if mech not in kappa_effects:
            continue
        kappas_sorted = sorted(kappa_effects[mech].keys())
        vals = [kappa_effects[mech][k] for k in kappas_sorted]
        traces.append(
            {
                "x": kappas_sorted,
                "y": [round(v, 4) for v in vals],
                "type": "scatter",
                "mode": "lines+markers",
                "name": mech,
                "line": {"color": MECHANISM_COLORS.get(mech, "#333"), "width": 2.5},
                "marker": {"size": 7},
            }
        )
    layout = {
        "title": {"text": title, "font": {"size": 14, "family": "Lato, sans-serif"}},
        "xaxis": {
            "title": "kappa",
            "type": "log",
            "tickvals": [0.25, 0.3, 0.5, 1.0, 1.5, 2.0, 4.0],
            "ticktext": ["0.25", "0.3", "0.5", "1.0", "1.5", "2.0", "4.0"],
        },
        "yaxis": {
            "title": "Treatment Effect (delta reduction)",
            "gridcolor": "#e0e0e0",
            "zeroline": True,
            "zerolinecolor": "#888",
            "zerolinewidth": 1.5,
        },
        "height": height,
        "width": width,
        "margin": {"l": 70, "r": 30, "t": 50, "b": 60},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "Lato, sans-serif"},
        "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
    }
    traces_json = ", ".join(json.dumps(t) for t in traces)
    return f"""
    <div id="chart_{chart_id}" style="margin: 1.5em auto; max-width: {width}px;"></div>
    <script>
    Plotly.newPlot('chart_{chart_id}', [{traces_json}], {json.dumps(layout)},
        {{responsive: true, displayModeBar: false}});
    </script>
    """


def _plotly_single_line_chart(
    kappa_vals: dict[float, float],
    mech: str,
    title: str,
    height: int = 280,
    width: int = 500,
) -> str:
    """Generate a single-mechanism line chart (effect vs kappa)."""
    chart_id = title.replace(" ", "_").replace(":", "_").lower()[:30]
    kappas_sorted = sorted(kappa_vals.keys())
    vals = [kappa_vals[k] for k in kappas_sorted]
    color = MECHANISM_COLORS.get(mech, "#333")
    trace = {
        "x": kappas_sorted,
        "y": [round(v, 4) for v in vals],
        "type": "scatter",
        "mode": "lines+markers",
        "name": mech,
        "line": {"color": color, "width": 2.5},
        "marker": {"size": 7},
        "fill": "tozeroy",
        "fillcolor": color.replace(")", ",0.1)").replace("rgb", "rgba")
        if color.startswith("rgb")
        else color + "1a",
    }
    layout = {
        "title": {"text": title, "font": {"size": 13, "family": "Lato, sans-serif"}},
        "xaxis": {
            "title": "kappa",
            "type": "log",
            "tickvals": [0.25, 0.3, 0.5, 1.0, 1.5, 2.0, 4.0],
            "ticktext": ["0.25", "0.3", "0.5", "1.0", "1.5", "2.0", "4.0"],
        },
        "yaxis": {
            "title": "Effect",
            "gridcolor": "#e0e0e0",
            "zeroline": True,
            "zerolinecolor": "#888",
        },
        "height": height,
        "width": width,
        "margin": {"l": 60, "r": 20, "t": 45, "b": 55},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "Lato, sans-serif"},
        "showlegend": False,
    }
    return f"""
    <div id="chart_{chart_id}" style="margin: 1em auto; max-width: {width}px;"></div>
    <script>
    Plotly.newPlot('chart_{chart_id}', [{json.dumps(trace)}], {json.dumps(layout)},
        {{responsive: true, displayModeBar: false}});
    </script>
    """


def _plotly_grouped_bar_sensitivity(
    sensitivities: dict[str, dict[str, float]],
    title: str,
    height: int = 300,
    width: int = 600,
) -> str:
    """Generate a grouped bar chart for Morris sensitivity indices."""
    chart_id = title.replace(" ", "_").replace(":", "_").lower()[:30]
    params = ["kappa", "concentration", "mu"]
    param_labels = ["kappa", "concentration", "mu"]
    traces = []
    for mech in MECHANISM_ORDER:
        if mech not in sensitivities:
            continue
        sens = sensitivities[mech]
        vals = [sens.get(p, 0) for p in params]
        traces.append(
            {
                "x": param_labels,
                "y": [round(v, 4) for v in vals],
                "type": "bar",
                "name": mech,
                "marker": {"color": MECHANISM_COLORS.get(mech, "#333")},
                "text": [f"{v:.3f}" for v in vals],
                "textposition": "outside",
                "textfont": {"size": 10},
            }
        )
    layout = {
        "title": {"text": title, "font": {"size": 14, "family": "Lato, sans-serif"}},
        "barmode": "group",
        "yaxis": {
            "title": "mu* (Morris sensitivity)",
            "gridcolor": "#e0e0e0",
        },
        "height": height,
        "width": width,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "Lato, sans-serif"},
        "legend": {"x": 0.7, "y": 0.98},
    }
    traces_json = ", ".join(json.dumps(t) for t in traces)
    return f"""
    <div id="chart_{chart_id}" style="margin: 1.5em auto; max-width: {width}px;"></div>
    <script>
    Plotly.newPlot('chart_{chart_id}', [{traces_json}], {json.dumps(layout)},
        {{responsive: true, displayModeBar: false}});
    </script>
    """


# ---------------------------------------------------------------------------
# HTML template pieces
# ---------------------------------------------------------------------------

HTML_HEAD = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Three-Way Mechanism Comparison: Narrative Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * { box-sizing: border-box; }
    body {
        font-family: Georgia, 'Times New Roman', serif;
        color: #333;
        line-height: 1.7;
        max-width: 860px;
        margin: 0 auto;
        padding: 2em 1.5em;
        background: #fff;
    }
    h1, h2, h3, h4 {
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        color: #2c3e50;
        line-height: 1.3;
    }
    h1 {
        font-size: 1.9em;
        border-bottom: 3px solid #2c3e50;
        padding-bottom: 0.3em;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        font-size: 1.05em;
        color: #666;
        margin-top: 0;
        margin-bottom: 2em;
    }
    h2 {
        font-size: 1.5em;
        margin-top: 2.5em;
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 0.2em;
    }
    h3 {
        font-size: 1.2em;
        margin-top: 2em;
        color: #34495e;
    }
    p { margin-bottom: 1em; }
    .callout {
        background: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 1em 1.2em;
        margin: 1.5em 0;
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.95em;
        line-height: 1.5;
    }
    .callout strong { color: #2c3e50; }
    .callout-bank { border-left-color: #2980b9; }
    .callout-dealer { border-left-color: #e67e22; }
    .callout-nbfi { border-left-color: #27ae60; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1.5em 0;
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.92em;
    }
    th {
        background: #2c3e50;
        color: #fff;
        padding: 0.6em 1em;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 0.5em 1em;
        border-bottom: 1px solid #e0e0e0;
    }
    tr:nth-child(even) td { background: #f8f9fa; }
    tr:hover td { background: #eef2f7; }
    .metric { font-weight: 700; color: #2c3e50; }
    .positive { color: #27ae60; }
    .negative { color: #c0392b; }
    .neutral { color: #7f8c8d; }
    .section-num {
        color: #95a5a6;
        font-weight: 400;
    }
    .footer {
        margin-top: 3em;
        padding-top: 1.5em;
        border-top: 2px solid #bdc3c7;
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.9em;
        color: #7f8c8d;
    }
    .footer a { color: #2980b9; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    .toc {
        background: #f8f9fa;
        padding: 1.2em 1.5em;
        margin: 1.5em 0 2.5em 0;
        border: 1px solid #e0e0e0;
        font-family: 'Lato', 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.93em;
    }
    .toc ul { list-style: none; padding-left: 0; margin: 0.5em 0 0 0; }
    .toc li { margin: 0.3em 0; }
    .toc a { color: #2c3e50; text-decoration: none; }
    .toc a:hover { text-decoration: underline; color: #2980b9; }
    .chart-container {
        text-align: center;
        margin: 1.5em 0;
    }
</style>
</head>
<body>
""")

HTML_FOOT = textwrap.dedent("""\
</body>
</html>
""")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt_effect(val: float) -> str:
    """Format an effect value with sign and color class."""
    if val > 0.001:
        return f'<span class="positive">+{val:.3f}</span>'
    elif val < -0.001:
        return f'<span class="negative">{val:.3f}</span>'
    else:
        return f'<span class="neutral">{val:.3f}</span>'


def _fmt_pct(val: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{val * 100:.1f}%"


def _effect_class(val: float) -> str:
    if val > 0.001:
        return "positive"
    elif val < -0.001:
        return "negative"
    return "neutral"


def generate_report() -> str:
    """Generate the complete narrative report HTML."""

    # ---- Load all data ----
    rows = _load_csv(CSV_PATH)
    dealer_summary = _load_json(DEALER_SUMMARY)
    bank_summary = _load_json(BANK_SUMMARY)
    nbfi_summary = _load_json(NBFI_SUMMARY)
    dealer_sens = _load_json(DEALER_SENSITIVITY)
    bank_sens = _load_json(BANK_SENSITIVITY)
    nbfi_sens = _load_json(NBFI_SENSITIVITY)
    bank_lending = _load_json(BANK_LENDING)
    nbfi_lending = _load_json(NBFI_LENDING)

    # ---- Compute derived stats ----
    mech_stats = compute_mechanism_stats(rows)
    kappa_effs = effects_by_kappa(rows)
    key_kappa_table = effects_at_key_kappas(kappa_effs)

    # Extract summary stats from JSON for each mechanism
    bank_te = bank_summary.get("trading_effect", {})
    dealer_te = dealer_summary.get("trading_effect", {})
    nbfi_te = nbfi_summary.get("trading_effect", {})

    bank_mean = bank_te.get("mean_effect", 0) or 0
    dealer_mean = dealer_te.get("mean_effect", 0) or 0
    nbfi_mean = nbfi_te.get("mean_effect", 0) or 0

    bank_pct_sig = bank_te.get("pct_significant_05", 0) or 0
    dealer_pct_sig = dealer_te.get("pct_significant_05", 0) or 0
    nbfi_pct_sig = nbfi_te.get("pct_significant_05", 0) or 0

    bank_cohens_d = bank_te.get("median_cohens_d", 0) or 0
    dealer_cohens_d = dealer_te.get("median_cohens_d", 0) or 0
    nbfi_cohens_d = nbfi_te.get("median_cohens_d", 0) or 0

    bank_n_positive = bank_te.get("n_positive", 0) or 0
    dealer_n_positive = dealer_te.get("n_positive", 0) or 0
    nbfi_n_positive = nbfi_te.get("n_positive", 0) or 0

    bank_n_negative = bank_te.get("n_negative", 0) or 0
    dealer_n_negative = dealer_te.get("n_negative", 0) or 0
    nbfi_n_negative = nbfi_te.get("n_negative", 0) or 0

    n_cells = bank_te.get("n_cells", 42) or 42

    # Best kappa range per mechanism
    bank_best_kappa = find_best_kappa_range(kappa_effs.get("Bank", {}))
    dealer_best_kappa = find_best_kappa_range(kappa_effs.get("Dealer", {}))
    nbfi_best_kappa = find_best_kappa_range(kappa_effs.get("NBFI", {}))

    # Bank lending stats
    bank_total_lent = bank_lending.get("overall_stats", {}).get("total_lent", 0)
    bank_avg_loans = bank_lending.get("overall_stats", {}).get("avg_loans_per_run", 0)
    bank_default_rate = bank_lending.get("overall_stats", {}).get(
        "overall_loan_default_rate", 0
    )
    bank_lgd = bank_lending.get("overall_stats", {}).get(
        "overall_loss_given_default", 0
    )
    bank_corr_lent = bank_lending.get("key_correlations_with_effect", {}).get(
        "total_lent", 0
    )
    bank_corr_rate = bank_lending.get("key_correlations_with_effect", {}).get(
        "avg_rate", 0
    )
    bank_cb_freeze = bank_lending.get("overall_stats", {}).get(
        "cb_freeze_activated_in_all_runs", False
    )

    # NBFI lending stats
    nbfi_total_lent = nbfi_lending.get("overall_stats", {}).get("total_lent", 0)
    nbfi_approval = nbfi_lending.get("overall_stats", {}).get(
        "overall_approval_rate", 0
    )
    nbfi_loan_default_rate = nbfi_lending.get("overall_stats", {}).get(
        "overall_loan_default_rate", 0
    )
    nbfi_avg_loans = nbfi_lending.get("overall_stats", {}).get("avg_loans_per_run", 0)
    nbfi_avg_rejections = nbfi_lending.get("overall_stats", {}).get(
        "avg_rejections_per_run", 0
    )
    nbfi_screening = nbfi_lending.get("screening_effectiveness", {})
    nbfi_frac_approved_default = nbfi_screening.get("frac_approved_who_defaulted", 0)
    nbfi_frac_rejected_default = nbfi_screening.get("frac_rejected_who_defaulted", 0)
    nbfi_corr_lent = nbfi_lending.get("key_correlations_with_effect", {}).get(
        "total_lent", 0
    )

    # Sensitivity
    dealer_sens_active = extract_sensitivity_mu_star(dealer_sens, "delta_active")
    bank_sens_active = extract_sensitivity_mu_star(bank_sens, "delta_active")
    nbfi_sens_active = extract_sensitivity_mu_star(nbfi_sens, "delta_active")

    # Also get passive sensitivities for comparison
    dealer_sens_passive = extract_sensitivity_mu_star(dealer_sens, "delta_passive")
    bank_sens_passive = extract_sensitivity_mu_star(bank_sens, "delta_passive")
    nbfi_sens_passive = extract_sensitivity_mu_star(nbfi_sens, "delta_passive")

    # Determine ranking
    ranking = sorted(
        MECHANISM_ORDER, key=lambda m: mech_stats.get(m, {}).get("mean_effect", 0),
        reverse=True,
    )
    rank_1, rank_2, rank_3 = ranking[0], ranking[1], ranking[2]

    total_runs = len(rows)
    n_mechanisms = len(set(r["mechanism"] for r in rows))

    # ----------------------------------------------------------------
    # Build HTML
    # ----------------------------------------------------------------
    parts: list[str] = []
    parts.append(HTML_HEAD)

    # Title
    parts.append('<h1>Three-Way Mechanism Comparison</h1>')
    parts.append(
        '<p class="subtitle">Dealer Trading vs Bank Lending vs NBFI Credit '
        f"&mdash; Cross-mechanism analysis of {total_runs * 3} simulation runs "
        f"({total_runs} per mechanism) across {n_cells} parameter cells</p>"
    )

    # Table of contents
    parts.append('<div class="toc">')
    parts.append("<strong>Contents</strong>")
    parts.append("<ul>")
    for i, (anchor, label) in enumerate(
        [
            ("executive-summary", "Executive Summary"),
            ("mechanism-ranking", "Mechanism Ranking"),
            ("how-each-works", "How Each Mechanism Works"),
            ("parameter-sensitivity", "Parameter Sensitivity"),
            ("comparison-at-key-stress", "Mechanism Comparison at Key Stress Levels"),
            ("why-bank-dominates", "Why Bank Dominates"),
            ("implications", "Implications & Next Steps"),
        ],
        1,
    ):
        parts.append(f'<li><a href="#{anchor}">{i}. {label}</a></li>')
    parts.append("</ul></div>")

    # ================================================================
    # Executive Summary
    # ================================================================
    parts.append('<h2 id="executive-summary"><span class="section-num">1.</span> Executive Summary</h2>')

    parts.append(
        f"<p>This report synthesizes findings from a three-way comparison of financial "
        f"intermediation mechanisms in a Kalecki ring economy with 100 agents. Three distinct "
        f"interventions &mdash; bank lending, dealer trading, and NBFI (non-bank financial "
        f"intermediary) credit &mdash; were tested across {n_cells} parameter combinations "
        f"spanning seven liquidity stress levels (kappa from 0.25 to 4.0), three debt "
        f"concentration levels, and two maturity timing profiles. Each cell was replicated "
        f"with 3 random seeds, yielding {total_runs} observations per mechanism and "
        f"{total_runs * n_mechanisms} total simulation runs.</p>"
    )

    parts.append(
        f"<p>The results reveal a clear hierarchy: <strong>{rank_1} lending</strong> is by "
        f"far the most effective intervention, with a mean treatment effect of "
        f"<span class=\"metric\">{bank_mean:.3f}</span> (reducing the default rate by "
        f"~{bank_mean * 100:.0f} percentage points on average). Statistical significance "
        f"is overwhelming: {_fmt_pct(bank_pct_sig)} of parameter cells show a significant "
        f"effect at the 5% level, with a median Cohen's d of "
        f"<span class=\"metric\">{bank_cohens_d:.2f}</span>. "
        f"The effect is positive in {bank_n_positive} of {n_cells} cells, with only "
        f"{bank_n_negative} cell{'s' if bank_n_negative != 1 else ''} showing "
        f"a negative effect.</p>"
    )

    parts.append(
        f"<p><strong>Dealer trading</strong> occupies a distant second place with a mean "
        f"effect of <span class=\"metric\">{dealer_mean:.3f}</span>. The picture here is "
        f"decidedly mixed: only {_fmt_pct(dealer_pct_sig)} of cells reach statistical "
        f"significance, and {dealer_n_negative} of {n_cells} cells actually show a "
        f"<em>negative</em> treatment effect (the dealer making things worse). The median "
        f"Cohen's d of <span class=\"metric\">{dealer_cohens_d:.2f}</span> indicates that "
        f"where the dealer does help, the effect size is small relative to run-to-run "
        f"variance. The dealer's mechanism &mdash; redistributing existing claims without "
        f"creating new money &mdash; is fundamentally limited in a liquidity-scarce system.</p>"
    )

    parts.append(
        f"<p><strong>NBFI lending</strong> brings up the rear with a near-zero mean effect "
        f"of <span class=\"metric\">{nbfi_mean:.3f}</span>. Only {_fmt_pct(nbfi_pct_sig)} "
        f"of cells are statistically significant. The NBFI rejects {_fmt_pct(1 - nbfi_approval)} "
        f"of loan applications, and even among approved borrowers, "
        f"{_fmt_pct(nbfi_frac_approved_default)} still default. The NBFI's screening "
        f"is accurate &mdash; {_fmt_pct(nbfi_frac_rejected_default)} of rejected borrowers "
        f"do indeed default &mdash; but accurate screening of a uniformly stressed pool "
        f"does not produce meaningful intervention. The NBFI is capital-constrained where "
        f"the bank is not, and this constraint is binding.</p>"
    )

    parts.append(
        '<div class="callout">'
        f"<strong>Key insight:</strong> The fundamental distinction is between "
        f"<em>endogenous money creation</em> (bank lending backed by the central bank) "
        f"and <em>redistribution of existing claims</em> (dealer trading and NBFI lending "
        f"from finite capital). Only the bank can expand the aggregate money supply, "
        f"and in a system where liquidity is the binding constraint, this is decisive."
        "</div>"
    )

    # ================================================================
    # Section 1: Mechanism Ranking
    # ================================================================
    parts.append(
        '<h2 id="mechanism-ranking"><span class="section-num">2.</span> Mechanism Ranking</h2>'
    )

    parts.append(
        "<p>The following table summarizes the aggregate performance of each mechanism "
        "across all 42 parameter cells. The treatment effect is defined as the reduction "
        "in the system default rate (delta) relative to the passive baseline &mdash; higher "
        "is better.</p>"
    )

    # Bar chart: mean effect by mechanism
    bar_labels = [m for m in MECHANISM_ORDER]
    bar_values = [mech_stats[m]["mean_effect"] for m in MECHANISM_ORDER]
    bar_colors = [MECHANISM_COLORS[m] for m in MECHANISM_ORDER]
    parts.append(
        '<div class="chart-container">'
        + _plotly_bar_chart(
            bar_labels,
            bar_values,
            bar_colors,
            "Mean Treatment Effect by Mechanism",
            "Mean Effect (delta reduction)",
        )
        + "</div>"
    )

    # Summary table
    parts.append("<table>")
    parts.append(
        "<tr><th>Mechanism</th><th>Mean Effect</th><th>% Significant (p &lt; 0.05)</th>"
        "<th>Median Cohen's d</th><th>Positive / Negative Cells</th>"
        "<th>Best kappa Range</th></tr>"
    )
    for mech in MECHANISM_ORDER:
        ms = mech_stats[mech]
        te = {"Bank": bank_te, "Dealer": dealer_te, "NBFI": nbfi_te}[mech]
        pct_sig = te.get("pct_significant_05", 0) or 0
        cohens_d = te.get("median_cohens_d", 0) or 0
        best_k = {"Bank": bank_best_kappa, "Dealer": dealer_best_kappa, "NBFI": nbfi_best_kappa}[mech]
        parts.append(
            f"<tr>"
            f"<td><strong>{mech}</strong></td>"
            f'<td class="{_effect_class(ms["mean_effect"])}">{ms["mean_effect"]:+.3f}</td>'
            f"<td>{_fmt_pct(pct_sig)}</td>"
            f"<td>{cohens_d:.2f}</td>"
            f'<td>{te.get("n_positive", 0)} / {te.get("n_negative", 0)}</td>'
            f"<td>{best_k}</td>"
            f"</tr>"
        )
    parts.append("</table>")

    parts.append(
        f"<p>The gap between bank lending and the other two mechanisms is striking. The bank's "
        f"mean effect ({bank_mean:.3f}) is {bank_mean / dealer_mean:.1f}x the dealer's "
        f"({dealer_mean:.3f}) and {bank_mean / nbfi_mean:.0f}x the NBFI's "
        f"({nbfi_mean:.3f}). Moreover, the bank's effect is statistically significant in "
        f"the vast majority of cells, while the dealer and NBFI effects are largely "
        f"indistinguishable from noise in most parameter regimes.</p>"
    )

    # ================================================================
    # Section 2: How Each Mechanism Works
    # ================================================================
    parts.append(
        '<h2 id="how-each-works"><span class="section-num">3.</span> How Each Mechanism Works</h2>'
    )

    # --- 2a: Bank Lending ---
    parts.append('<h3>3a. Bank Lending</h3>')

    bank_kappa_effs = kappa_effs.get("Bank", {})
    parts.append(
        '<div class="chart-container">'
        + _plotly_single_line_chart(
            bank_kappa_effs, "Bank", "Bank: Treatment Effect by kappa"
        )
        + "</div>"
    )

    parts.append(
        f"<p>Bank lending operates through <strong>endogenous money creation</strong>. When "
        f"the bank issues a loan, it simultaneously creates a deposit in the borrower's "
        f"account &mdash; new money enters the system that did not exist before. This is "
        f"fundamentally different from the dealer or NBFI, which can only redistribute "
        f"existing claims. Across all {bank_lending.get('n_runs', 126)} runs, the bank "
        f"issued a total of {bank_lending.get('overall_stats', {}).get('total_loans', 0):,} "
        f"loans totaling {bank_total_lent:,.0f} in face value, averaging "
        f"{bank_avg_loans:.1f} loans per run.</p>"
    )

    parts.append(
        '<div class="callout callout-bank">'
        f"<strong>Key correlations:</strong> total_lent is the strongest predictor "
        f"of the bank's treatment effect (r = {bank_corr_lent:.2f}). Average loan rate "
        f"is negatively correlated (r = {bank_corr_rate:.2f}): cheaper credit helps more. "
        f"Loss given default is {bank_lgd:.1%} &mdash; when borrowers fail, almost nothing "
        f"is recovered &mdash; but the bank still succeeds because the marginal borrowers "
        f"who survive repay their payables and prevent cascading defaults."
        "</div>"
    )

    # Find peak bank effect kappa
    if bank_kappa_effs:
        peak_kappa = max(bank_kappa_effs, key=bank_kappa_effs.get)
        peak_val = bank_kappa_effs[peak_kappa]
        low_kappa_eff = bank_kappa_effs.get(0.25, 0)
        high_kappa_eff = bank_kappa_effs.get(4.0, 0)
    else:
        peak_kappa, peak_val, low_kappa_eff, high_kappa_eff = 1.0, 0, 0, 0

    parts.append(
        f"<p>The bank's effectiveness follows a hump-shaped pattern over kappa. At very low "
        f"kappa (0.25), the system is so severely stressed that even bank credit cannot "
        f"prevent widespread cascades (effect = {low_kappa_eff:.3f}). The peak occurs at "
        f"kappa = {peak_kappa} (effect = {peak_val:.3f}), where the system has enough "
        f"underlying capacity that targeted credit injections can tip marginal agents from "
        f"default to survival. At high kappa (4.0), few agents need credit in the first "
        f"place, so the effect collapses to {high_kappa_eff:.3f}.</p>"
    )

    cb_freeze_text = (
        "The central bank lending freeze activates in all 126 runs"
        if bank_cb_freeze
        else "The central bank lending freeze is present in the majority of runs"
    )
    parts.append(
        f"<p>{cb_freeze_text}, imposing a cutoff day after which no new bank lending occurs. "
        f"Despite this constraint, {_fmt_pct(nbfi_frac_approved_default)} of bank borrowers "
        f"still default &mdash; bank lending is not a panacea. Its power lies in "
        f"<em>cascade prevention</em>: by keeping a subset of marginal agents solvent, "
        f"it breaks the chain of defaults that would otherwise propagate through the ring.</p>"
    )

    # --- 2b: Dealer Trading ---
    parts.append('<h3>3b. Dealer Trading</h3>')

    dealer_kappa_effs = kappa_effs.get("Dealer", {})
    parts.append(
        '<div class="chart-container">'
        + _plotly_single_line_chart(
            dealer_kappa_effs, "Dealer", "Dealer: Treatment Effect by kappa"
        )
        + "</div>"
    )

    parts.append(
        f"<p>The dealer operates a secondary market for payable claims, buying from "
        f"stressed sellers and (potentially) selling to surplus agents. Unlike the bank, "
        f"the dealer <strong>does not create new money</strong> &mdash; it redistributes "
        f"existing claims at a discount. The treatment effect is mixed: positive in "
        f"{dealer_n_positive} of {n_cells} cells, but negative in {dealer_n_negative}. "
        f"The mean effect of {dealer_mean:.3f} is modest, and the median Cohen's d of "
        f"{dealer_cohens_d:.2f} indicates that the effect, where it exists, is often "
        f"smaller than the run-to-run variance.</p>"
    )

    # Find where dealer helps most
    if dealer_kappa_effs:
        dealer_peak_kappa = max(dealer_kappa_effs, key=dealer_kappa_effs.get)
        dealer_peak_val = dealer_kappa_effs[dealer_peak_kappa]
        dealer_worst_kappa = min(dealer_kappa_effs, key=dealer_kappa_effs.get)
        dealer_worst_val = dealer_kappa_effs[dealer_worst_kappa]
    else:
        dealer_peak_kappa, dealer_peak_val = 0.25, 0
        dealer_worst_kappa, dealer_worst_val = 4.0, 0

    parts.append(
        '<div class="callout callout-dealer">'
        f"<strong>Where the dealer helps:</strong> The dealer's best performance occurs at "
        f"kappa = {dealer_peak_kappa} (effect = {dealer_peak_val:.3f}), where severe "
        f"liquidity stress creates desperate sellers willing to accept deep discounts. "
        f"The dealer provides price discovery and immediate cash to sellers who would "
        f"otherwise default. However, at kappa = {dealer_worst_kappa}, the effect is "
        f"actually {dealer_worst_val:.3f}, suggesting that dealer activity can be "
        f"counterproductive in some regimes."
        "</div>"
    )

    parts.append(
        "<p>The dealer's fundamental limitation is that it operates in a <em>zero-sum</em> "
        "framework with respect to aggregate liquidity. When the dealer buys a claim from "
        "a stressed seller, it injects cash into that seller's balance sheet &mdash; but "
        "the claim still needs to be settled, and the original debtor's obligation remains. "
        "The dealer helps individual agents manage liquidity timing, but it cannot increase "
        "the total stock of money in the system. In a world where the binding constraint is "
        "aggregate liquidity, this is a structural limitation that no amount of trading "
        "activity can overcome.</p>"
    )

    # --- 2c: NBFI Lending ---
    parts.append('<h3>3c. NBFI Lending</h3>')

    nbfi_kappa_effs = kappa_effs.get("NBFI", {})
    parts.append(
        '<div class="chart-container">'
        + _plotly_single_line_chart(
            nbfi_kappa_effs, "NBFI", "NBFI: Treatment Effect by kappa"
        )
        + "</div>"
    )

    parts.append(
        f"<p>The NBFI (non-bank financial intermediary) lends from its own finite capital "
        f"base, subject to coverage requirements and credit screening. Across all runs, "
        f"the NBFI made {nbfi_lending.get('overall_stats', {}).get('total_loans', 0):,} loans "
        f"totaling {nbfi_total_lent:,.0f} &mdash; only "
        f"{nbfi_total_lent / bank_total_lent * 100:.0f}% of the bank's lending volume. "
        f"The mean treatment effect is a negligible {nbfi_mean:.3f}, and only "
        f"{_fmt_pct(nbfi_pct_sig)} of cells reach statistical significance.</p>"
    )

    parts.append(
        '<div class="callout callout-nbfi">'
        f"<strong>The screening paradox:</strong> The NBFI rejects "
        f"{_fmt_pct(1 - nbfi_approval)} of loan applications (averaging "
        f"{nbfi_avg_rejections:.0f} rejections vs {nbfi_avg_loans:.0f} approvals per run). "
        f"Its screening is accurate: {_fmt_pct(nbfi_frac_rejected_default)} of rejected "
        f"borrowers do indeed default. But even among approved borrowers, "
        f"{_fmt_pct(nbfi_frac_approved_default)} still default. The NBFI is correctly "
        f"identifying the worst risks but lending into a pool where even the best agents "
        f"are marginal."
        "</div>"
    )

    parts.append(
        f"<p>The critical constraint is <strong>capital</strong>. The NBFI can only lend "
        f"what it has, and its finite balance sheet binds quickly under stress. "
        f"The correlation between total_lent and treatment effect is just "
        f"r = {nbfi_corr_lent:.3f} &mdash; in stark contrast to the bank's r = "
        f"{bank_corr_lent:.2f}. More NBFI lending does not produce proportionally more "
        f"benefit, because the NBFI's capital is too small relative to the system's "
        f"aggregate liquidity shortfall. {_fmt_pct(nbfi_loan_default_rate)} of NBFI loans "
        f"default, destroying the NBFI's own capital and further constraining its capacity "
        f"to intervene.</p>"
    )

    # ================================================================
    # Section 3: Parameter Sensitivity
    # ================================================================
    parts.append(
        '<h2 id="parameter-sensitivity"><span class="section-num">4.</span> Parameter Sensitivity</h2>'
    )

    parts.append(
        "<p>Morris sensitivity analysis reveals which parameters drive the treatment outcome "
        "for each mechanism. The mu* (mu-star) statistic measures the mean absolute "
        "elementary effect of each parameter on the default rate, capturing both linear "
        "and nonlinear sensitivities. Higher mu* means the parameter matters more.</p>"
    )

    # Sensitivity chart: passive (baseline response)
    sensitivities_passive = {
        "Bank": bank_sens_passive,
        "Dealer": dealer_sens_passive,
        "NBFI": nbfi_sens_passive,
    }
    parts.append(
        '<div class="chart-container">'
        + _plotly_grouped_bar_sensitivity(
            sensitivities_passive,
            "Morris mu* for Baseline Default Rate (delta_passive)",
        )
        + "</div>"
    )

    # Sensitivity chart: active (treatment response)
    sensitivities_active = {
        "Bank": bank_sens_active,
        "Dealer": dealer_sens_active,
        "NBFI": nbfi_sens_active,
    }
    parts.append(
        '<div class="chart-container">'
        + _plotly_grouped_bar_sensitivity(
            sensitivities_active,
            "Morris mu* for Treatment Default Rate (delta_active)",
        )
        + "</div>"
    )

    # Extract key numbers for narrative
    bank_kappa_passive = bank_sens_passive.get("kappa", 0)
    bank_kappa_active = bank_sens_active.get("kappa", 0)
    bank_mu_passive = bank_sens_passive.get("mu", 0)
    bank_mu_active = bank_sens_active.get("mu", 0)
    bank_conc_passive = bank_sens_passive.get("concentration", 0)
    bank_conc_active = bank_sens_active.get("concentration", 0)

    dealer_kappa_sens_p = dealer_sens_passive.get("kappa", 0)
    dealer_kappa_sens_a = dealer_sens_active.get("kappa", 0)
    dealer_conc_sens_p = dealer_sens_passive.get("concentration", 0)
    dealer_conc_sens_a = dealer_sens_active.get("concentration", 0)

    parts.append(
        f"<p><strong>Kappa dominates across the board.</strong> For all three mechanisms, "
        f"the liquidity ratio (kappa) is the single most important parameter, with "
        f"mu* values of {bank_kappa_passive:.3f} (bank passive), "
        f"{dealer_kappa_sens_p:.3f} (dealer passive), and "
        f"{nbfi_sens_passive.get('kappa', 0):.3f} (NBFI passive) for the baseline "
        f"default rate. This is expected: kappa directly determines how much cash is "
        f"available to meet obligations.</p>"
    )

    parts.append(
        f"<p>The interesting divergence appears in the <em>treatment</em> sensitivities. "
        f"Bank lending dramatically reduces kappa sensitivity: from "
        f"{bank_kappa_passive:.3f} (passive) to {bank_kappa_active:.3f} (with bank), "
        f"meaning the bank partially substitutes for scarce liquidity. Maturity timing (mu) "
        f"sensitivity drops from {bank_mu_passive:.3f} to {bank_mu_active:.3f}, and "
        f"concentration sensitivity drops from {bank_conc_passive:.3f} to "
        f"{bank_conc_active:.3f}. The bank absorbs parameter risk.</p>"
    )

    parts.append(
        f"<p>For the dealer, kappa sensitivity decreases modestly from "
        f"{dealer_kappa_sens_p:.3f} to {dealer_kappa_sens_a:.3f}, but concentration "
        f"sensitivity remains nearly unchanged ({dealer_conc_sens_p:.3f} vs "
        f"{dealer_conc_sens_a:.3f}). This makes sense: the dealer helps with "
        f"<em>liquidity</em> problems (kappa) but cannot fix <em>structural</em> "
        f"inequality (concentration). The NBFI shows almost no change between passive and "
        f"active sensitivities, consistent with its negligible treatment effect.</p>"
    )

    # ================================================================
    # Section 4: Comparison at Key Stress Levels
    # ================================================================
    parts.append(
        '<h2 id="comparison-at-key-stress">'
        '<span class="section-num">5.</span> Mechanism Comparison at Key Stress Levels</h2>'
    )

    # Line chart: all mechanisms
    parts.append(
        '<div class="chart-container">'
        + _plotly_line_chart(
            kappa_effs,
            "Treatment Effect by kappa: All Mechanisms",
            height=340,
            width=680,
        )
        + "</div>"
    )

    parts.append(
        "<p>The following table compares mechanism performance at three representative "
        "stress levels: severe stress (kappa = 0.25), moderate stress (kappa = 1.0), and "
        "low stress (kappa = 4.0). Effects are averaged over concentration and mu.</p>"
    )

    parts.append("<table>")
    parts.append(
        "<tr><th>kappa</th><th>Stress Level</th><th>Bank Effect</th>"
        "<th>Dealer Effect</th><th>NBFI Effect</th></tr>"
    )
    stress_labels = {0.25: "Severe", 1.0: "Moderate", 4.0: "Low"}
    for kappa in KEY_KAPPAS:
        label = stress_labels.get(kappa, "")
        bank_eff = key_kappa_table[kappa].get("Bank", 0)
        dealer_eff = key_kappa_table[kappa].get("Dealer", 0)
        nbfi_eff = key_kappa_table[kappa].get("NBFI", 0)
        parts.append(
            f"<tr>"
            f"<td><strong>{kappa}</strong></td>"
            f"<td>{label}</td>"
            f"<td>{_fmt_effect(bank_eff)}</td>"
            f"<td>{_fmt_effect(dealer_eff)}</td>"
            f"<td>{_fmt_effect(nbfi_eff)}</td>"
            f"</tr>"
        )
    parts.append("</table>")

    # Narrative for each stress level
    sev_bank = key_kappa_table[0.25].get("Bank", 0)
    sev_dealer = key_kappa_table[0.25].get("Dealer", 0)
    sev_nbfi = key_kappa_table[0.25].get("NBFI", 0)
    mod_bank = key_kappa_table[1.0].get("Bank", 0)
    mod_dealer = key_kappa_table[1.0].get("Dealer", 0)
    mod_nbfi = key_kappa_table[1.0].get("NBFI", 0)
    low_bank = key_kappa_table[4.0].get("Bank", 0)
    low_dealer = key_kappa_table[4.0].get("Dealer", 0)
    low_nbfi = key_kappa_table[4.0].get("NBFI", 0)

    parts.append(
        f"<p><strong>At severe stress (kappa = 0.25):</strong> The system is deeply "
        f"liquidity-constrained. Even the bank's effect is limited to {sev_bank:.3f} "
        f"&mdash; there is simply too much debt relative to available cash for any "
        f"mechanism to prevent wholesale cascading. The dealer's effect ({sev_dealer:.3f}) "
        f"is ambiguous, and the NBFI ({sev_nbfi:.3f}) is essentially inert. At this "
        f"stress level, no single intervention can overcome the fundamental arithmetic "
        f"of liquidity scarcity.</p>"
    )

    parts.append(
        f"<p><strong>At moderate stress (kappa = 1.0):</strong> This is where the bank "
        f"shines. With an effect of {mod_bank:.3f}, bank lending is reducing defaults by "
        f"roughly {mod_bank * 100:.0f} percentage points. The system has enough underlying "
        f"capacity that targeted credit injections can tip the balance for marginal agents, "
        f"breaking cascade chains. The dealer ({mod_dealer:.3f}) and NBFI "
        f"({mod_nbfi:.3f}) contribute comparatively little.</p>"
    )

    parts.append(
        f"<p><strong>At low stress (kappa = 4.0):</strong> All effects converge toward "
        f"zero (bank: {low_bank:.3f}, dealer: {low_dealer:.3f}, NBFI: {low_nbfi:.3f}). "
        f"When the system has abundant liquidity, few agents default in the baseline, "
        f"and there is little room for any mechanism to improve outcomes. This is the "
        f"expected ceiling effect.</p>"
    )

    # ================================================================
    # Section 5: Why Bank Dominates
    # ================================================================
    parts.append(
        '<h2 id="why-bank-dominates"><span class="section-num">6.</span> Why Bank Dominates</h2>'
    )

    parts.append(
        "<p>The dominance of bank lending over both dealer trading and NBFI credit has a "
        "clear theoretical explanation rooted in the distinction between endogenous and "
        "exogenous money creation.</p>"
    )

    parts.append(
        '<div class="callout">'
        "<strong>The three money channels:</strong><br>"
        "<em>Bank lending</em> creates new deposits (endogenous money) backed by central "
        "bank reserves. The bank's balance sheet expands: a new loan asset appears alongside "
        "a new deposit liability. Net new money enters the system.<br><br>"
        "<em>Dealer trading</em> transfers existing claims between agents. The dealer buys "
        "a payable from a stressed seller (injecting cash) and may later sell it to a "
        "surplus agent (recovering cash). No new money is created &mdash; it is a "
        "redistribution of the existing stock.<br><br>"
        "<em>NBFI lending</em> transfers the NBFI's own cash to borrowers. The NBFI's "
        "balance sheet transforms (cash becomes a loan receivable) but no net new money "
        "enters the system. The NBFI is constrained by its own capital."
        "</div>"
    )

    parts.append(
        f"<p>This distinction is decisive because the Kalecki ring is, at its core, a "
        f"<strong>liquidity problem</strong>. When kappa &lt; 1, there is literally not "
        f"enough aggregate cash to settle all debts. The bank solves this by creating new "
        f"money. The dealer and NBFI cannot. Consider the lending volumes: the bank "
        f"deployed {bank_total_lent:,.0f} in total credit across {bank_lending.get('n_runs', 0)} "
        f"runs, while the NBFI managed only {nbfi_total_lent:,.0f} &mdash; "
        f"{nbfi_total_lent / bank_total_lent * 100:.0f}% of the bank's volume. "
        f"And crucially, the bank's credit <em>adds</em> to aggregate liquidity, while the "
        f"NBFI's merely <em>reallocates</em> it.</p>"
    )

    parts.append(
        f"<p>The cascade prevention channel amplifies the bank's advantage. When a bank loan "
        f"prevents one agent from defaulting, that agent's creditors receive payment, "
        f"which may prevent <em>their</em> defaults, and so on. This multiplier effect "
        f"means that a relatively small volume of bank credit (averaging {bank_avg_loans:.0f} "
        f"loans per run) can produce outsized system-level effects. The correlation between "
        f"total_lent and effect (r = {bank_corr_lent:.2f}) confirms that volume matters, but "
        f"the relationship is super-linear once cascade dynamics are accounted for.</p>"
    )

    parts.append(
        "<p>The dealer faces a structural limitation: its market-making activity is a "
        "zero-sum game with respect to aggregate liquidity. Every dollar paid to a seller "
        "is a dollar the dealer can no longer deploy elsewhere. In a liquidity-scarce "
        "system, the dealer can improve the <em>allocation</em> of existing cash but "
        "cannot overcome the <em>scarcity</em> itself. This explains why the dealer's "
        "effect is mixed (positive in some cells, negative in others): in favorable "
        "configurations, better allocation helps; in unfavorable ones, dealer trading "
        "introduces price volatility and adverse selection costs that can harm fragile "
        "agents.</p>"
    )

    parts.append(
        "<p>The NBFI suffers from both the dealer's zero-sum constraint <em>and</em> an "
        "additional screening constraint. By rejecting high-risk borrowers (who are precisely "
        "the agents most in need of liquidity), the NBFI concentrates its limited capital on "
        "agents who are only marginally stressed &mdash; agents who might have survived "
        "anyway. The result is a treatment that is correctly targeted at the population level "
        "but too small to matter at the system level.</p>"
    )

    # ================================================================
    # Section 6: Implications & Next Steps
    # ================================================================
    parts.append(
        '<h2 id="implications"><span class="section-num">7.</span> Implications & Next Steps</h2>'
    )

    parts.append(
        "<p>These findings suggest several directions for further investigation:</p>"
    )

    parts.append(
        "<p><strong>Bank parameter sensitivity.</strong> The current results use a single "
        "bank configuration (credit_risk_loading, max_borrower_risk, corridor settings). "
        "A dedicated sweep over bank parameters would reveal how sensitive the bank's "
        "dominance is to its own pricing and risk management. In particular: does the bank "
        "still dominate when credit_risk_loading is positive (meaning the bank charges for "
        "credit risk)? At what point does risk-based pricing cause the bank to behave more "
        "like the NBFI, screening out the borrowers who need help most?</p>"
    )

    parts.append(
        "<p><strong>Mechanism combinations.</strong> The three-way comparison tests each "
        "mechanism in isolation. A natural next step is to test <em>combinations</em>: "
        "bank + dealer, bank + NBFI, or all three simultaneously. The key question is "
        "whether the effects are additive, sub-additive (redundant mechanisms), or "
        "super-additive (complementary mechanisms). Theory suggests the dealer might "
        "complement the bank by providing price discovery that guides the bank's lending "
        "decisions, but this remains untested.</p>"
    )

    parts.append(
        "<p><strong>NBFI capital augmentation.</strong> The NBFI's failure is primarily "
        "a capital constraint, not a screening failure. What happens if the NBFI receives "
        "periodic capital top-ups (simulating equity injections or retained earnings)? "
        "What if the NBFI can access wholesale funding markets (borrowing to lend)? "
        "At what capital level does the NBFI begin to approach the bank's effectiveness, "
        "and is that level realistic?</p>"
    )

    parts.append(
        "<p><strong>Temporal dynamics.</strong> The current analysis aggregates over the "
        "entire simulation horizon. A time-resolved analysis would reveal <em>when</em> "
        "each mechanism is most active and most effective. Does bank lending front-load "
        "its benefit (preventing early defaults), or does it matter most in the middle "
        "of the simulation when cascades are building? Does the dealer's benefit "
        "concentrate in specific days?</p>"
    )

    parts.append(
        "<p><strong>Heterogeneous agent analysis.</strong> Which agents benefit most from "
        "each mechanism? The bank may disproportionately help agents at specific positions "
        "in the ring (e.g., those with large obligations due early). Understanding the "
        "distributional impact of each mechanism is important for policy design.</p>"
    )

    # ================================================================
    # Footer
    # ================================================================
    parts.append('<div class="footer">')
    parts.append("<p><strong>Related dashboards:</strong></p>")
    parts.append("<ul>")
    parts.append(
        '<li><a href="../three_way_dashboard.html">Three-Way Comparison Dashboard</a></li>'
    )
    parts.append(
        '<li><a href="drilldown_dashboard.html">Drilldown Dashboard</a></li>'
    )
    parts.append(
        '<li><a href="treatment_deltas_dashboard.html">Treatment Deltas Dashboard</a></li>'
    )
    parts.append("</ul>")
    parts.append(
        f"<p>Report generated from {total_runs * n_mechanisms} simulation runs "
        f"({total_runs} per mechanism, {n_cells} parameter cells, "
        f"{total_runs // n_cells} seeds per cell).</p>"
    )
    parts.append("</div>")

    parts.append(HTML_FOOT)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate that all data files exist
    required_files = [
        CSV_PATH,
        DEALER_SUMMARY,
        BANK_SUMMARY,
        NBFI_SUMMARY,
        DEALER_SENSITIVITY,
        BANK_SENSITIVITY,
        NBFI_SENSITIVITY,
        BANK_LENDING,
        NBFI_LENDING,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        print("ERROR: Missing required data files:")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nRun the three-way analysis pipeline first to generate these files."
        )
        raise SystemExit(1)

    # Generate report
    html = generate_report()

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Narrative report written to: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
