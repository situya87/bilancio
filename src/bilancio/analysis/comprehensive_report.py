"""Self-contained HTML comprehensive report from sweep results.

Generates a single HTML page with 12 sections covering summary statistics,
distribution analysis, parameter response curves, heatmaps, marginal effects,
contagion metrics, temporal dynamics, mechanism internals, loss analysis,
OLS regression, statistical tests, and embedded Treynor dashboards.

Each section degrades gracefully: if one section fails, the rest still render.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from bilancio.analysis.cross_sweep import (
    fit_ols,
    load_sweep_data,
    one_sample_ttest,
    pivot_for_heatmap,
    summary_by_group,
    summary_stats,
    wilcoxon_test,
)
from bilancio.analysis.cross_sweep_network import (
    extract_mechanism_timeseries,
    extract_run_metrics,
    extract_run_timeseries,
)

logger = logging.getLogger(__name__)

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #333; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; }
.section { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
th { background: #f5f6fa; font-weight: 600; }
tr:nth-child(even) { background: #fafbfc; }
.chart-container { margin: 15px 0; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; padding: 8px; background: #ecf0f1; border-radius: 4px; }
summary:hover { background: #d5dbdb; }
.metric-positive { color: #27ae60; }
.metric-negative { color: #e74c3c; }
.stat-sig { font-weight: bold; }
.nav { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 1px solid #ddd; z-index: 100; margin-bottom: 20px; }
.nav a { margin-right: 15px; text-decoration: none; color: #3498db; }
.nav a:hover { text-decoration: underline; }
.error-box { background: #ffeaa7; border: 1px solid #fdcb6e; padding: 12px; border-radius: 4px; margin: 10px 0; }
"""

# ---------------------------------------------------------------------------
# Sweep metadata
# ---------------------------------------------------------------------------

_SWEEP_META: dict[str, dict[str, Any]] = {
    "dealer": {
        "effect_col": "trading_effect",
        "treatment_label": "Active (Dealer)",
        "baseline_label": "Passive",
        "delta_treatment": "delta_active",
        "delta_baseline": "delta_passive",
        "run_id_treatment_candidates": ["active_run_id", "run_id_active"],
        "run_id_baseline_candidates": ["passive_run_id", "run_id_passive"],
    },
    "bank": {
        "effect_col": "bank_lending_effect",
        "treatment_label": "Bank Lending",
        "baseline_label": "Bank Idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "run_id_treatment_candidates": ["lend_run_id", "run_id_lend"],
        "run_id_baseline_candidates": ["idle_run_id", "run_id_idle"],
    },
    "nbfi": {
        "effect_col": "lending_effect",
        "treatment_label": "NBFI Lending",
        "baseline_label": "NBFI Idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "run_id_treatment_candidates": ["lend_run_id", "run_id_lend"],
        "run_id_baseline_candidates": ["idle_run_id", "run_id_idle"],
    },
}

_DIR_MAP = {
    ("dealer", "treatment"): "active",
    ("dealer", "baseline"): "passive",
    ("bank", "treatment"): "bank_lend",
    ("bank", "baseline"): "bank_idle",
    ("nbfi", "treatment"): "nbfi_lend",
    ("nbfi", "baseline"): "nbfi_idle",
}


def _sweep_meta(sweep_type: str) -> dict[str, Any]:
    """Return metadata dict for a given sweep type."""
    if sweep_type not in _SWEEP_META:
        raise ValueError(f"Unknown sweep_type={sweep_type!r}; expected one of {list(_SWEEP_META)}")
    return dict(_SWEEP_META[sweep_type])


def _arm_runs_dir(out_dir: Path, sweep_type: str, arm: str) -> Path:
    """Return the runs directory for a given sweep arm."""
    subdir = _DIR_MAP.get((sweep_type, arm), arm)
    return out_dir / subdir / "runs"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_col(data: dict[str, Any], candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in data."""
    for c in candidates:
        if c in data:
            return c
    return None


def _load_comparison_data(out_dir: Path) -> dict[str, Any]:
    """Load comparison.csv from the aggregate subdirectory."""
    csv_path = out_dir / "aggregate" / "comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"comparison.csv not found at {csv_path}")
    raw = load_sweep_data(csv_path)
    # Convert lists to numpy arrays for numeric columns
    result: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            arr = np.array(v, dtype=float)
            result[k] = arr
        except (ValueError, TypeError):
            result[k] = np.array(v, dtype=object)
    return result


def _load_run_events(run_dir: Path) -> list[dict]:
    """Load events.jsonl from a run directory."""
    for candidate in [run_dir / "out" / "events.jsonl", run_dir / "events.jsonl"]:
        if candidate.is_file():
            events = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
            return events
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _representative_kappas(data: dict, n: int = 3) -> list[float]:
    """Pick min/median/max kappa values."""
    if "kappa" not in data:
        return []
    kappas = np.unique(data["kappa"][np.isfinite(data["kappa"])])
    if len(kappas) == 0:
        return []
    if len(kappas) <= n:
        return sorted(kappas.tolist())
    result = [kappas[0], kappas[len(kappas) // 2], kappas[-1]]
    return sorted(set(float(k) for k in result))


def _find_run_for_kappa(data: dict, kappa: float, run_id_col: str) -> str | None:
    """Find first run_id matching kappa value."""
    if run_id_col not in data:
        return None
    mask = np.abs(data["kappa"].astype(float) - kappa) < 0.01
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    for idx in indices:
        rid = data[run_id_col][idx]
        if rid and str(rid) not in ("nan", ""):
            return str(rid)
    return None


def _fmt(val: float, digits: int = 4) -> str:
    """Format a float for display."""
    if not np.isfinite(val):
        return "N/A"
    return f"{val:.{digits}f}"


def _pval_class(p: float) -> str:
    """Return CSS class based on p-value significance."""
    if not np.isfinite(p):
        return ""
    return "stat-sig" if p < 0.05 else ""


def _effect_class(val: float) -> str:
    """Return CSS class for positive/negative effects."""
    if not np.isfinite(val):
        return ""
    return "metric-positive" if val > 0 else "metric-negative" if val < 0 else ""


def _n_unique(data: dict, col: str) -> int:
    """Count unique finite values in a column."""
    if col not in data:
        return 0
    arr = data[col]
    try:
        arr_f = np.array(arr, dtype=float)
        return len(np.unique(arr_f[np.isfinite(arr_f)]))
    except (ValueError, TypeError):
        return len(set(arr))


def _safe_section(fn, *args, **kwargs) -> str:
    """Execute a section function with error handling."""
    try:
        result = fn(*args, **kwargs)
        return result if result else ""
    except Exception:
        fn_name = getattr(fn, "__name__", "unknown")
        tb = traceback.format_exc()
        logger.warning("Section %s failed: %s", fn_name, tb)
        return (
            f'<div class="section">'
            f'<div class="error-box">'
            f"<strong>Section {fn_name} could not be rendered.</strong><br>"
            f"<details><summary>Error details</summary><pre>{tb}</pre></details>"
            f"</div></div>"
        )


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

def _html_shell(title: str, sections: list[tuple[str, str, str]]) -> str:
    """Wrap sections in a complete HTML page.

    Parameters
    ----------
    title : str
        Page title.
    sections : list of (anchor_id, nav_label, html_content)
    """
    nav_links = []
    for anchor_id, nav_label, _ in sections:
        if anchor_id and nav_label:
            nav_links.append(f'<a href="#{anchor_id}">{nav_label}</a>')

    nav_html = ""
    if nav_links:
        nav_html = f'<div class="nav">{"".join(nav_links)}</div>'

    body_parts = []
    for anchor_id, _, content in sections:
        if content:
            if anchor_id:
                body_parts.append(f'<div id="{anchor_id}">{content}</div>')
            else:
                body_parts.append(content)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
{nav_html}
{"".join(body_parts)}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section 1: Summary Statistics
# ---------------------------------------------------------------------------

def _section_summary(data: dict, meta: dict) -> str:
    effect_col = meta["effect_col"]
    delta_t = meta["delta_treatment"]
    delta_b = meta["delta_baseline"]

    metrics = {}
    for label, col in [
        (meta["treatment_label"] + " delta", delta_t),
        (meta["baseline_label"] + " delta", delta_b),
        ("Effect", effect_col),
    ]:
        if col in data:
            metrics[label] = summary_stats(data[col])

    if not metrics:
        return ""

    rows_html = []
    for label, s in metrics.items():
        pct_pos = s.get("pct_positive", float("nan"))
        pct_class = _effect_class(pct_pos - 50) if np.isfinite(pct_pos) else ""
        rows_html.append(
            f"<tr>"
            f"<td style='text-align:left'>{label}</td>"
            f"<td>{s.get('n', 0)}</td>"
            f"<td>{_fmt(s['mean'])}</td>"
            f"<td>{_fmt(s['median'])}</td>"
            f"<td>{_fmt(s['sd'])}</td>"
            f"<td>{_fmt(s['iqr_low'])} - {_fmt(s['iqr_high'])}</td>"
            f'<td class="{pct_class}">{_fmt(pct_pos, 1)}%</td>'
            f"</tr>"
        )

    return (
        '<div class="section">'
        "<h2>Summary Statistics</h2>"
        "<table>"
        "<tr><th style='text-align:left'>Metric</th><th>n</th><th>Mean</th>"
        "<th>Median</th><th>Std</th><th>IQR</th><th>% Positive</th></tr>"
        + "".join(rows_html)
        + "</table></div>"
    )


# ---------------------------------------------------------------------------
# Section 2: Effect Distribution
# ---------------------------------------------------------------------------

def _section_distribution(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    delta_t = meta["delta_treatment"]
    delta_b = meta["delta_baseline"]
    effect_col = meta["effect_col"]

    has_deltas = delta_t in data and delta_b in data
    has_effect = effect_col in data

    if not has_deltas and not has_effect:
        return ""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Default Rate Distribution", "Effect Distribution"],
    )

    if has_deltas:
        vals_t = np.array(data[delta_t], dtype=float)
        vals_b = np.array(data[delta_b], dtype=float)
        fig.add_trace(
            go.Box(y=vals_t[np.isfinite(vals_t)], name=meta["treatment_label"],
                   marker_color="#e74c3c"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=vals_b[np.isfinite(vals_b)], name=meta["baseline_label"],
                   marker_color="#95a5a6"),
            row=1, col=1,
        )

    if has_effect:
        vals_e = np.array(data[effect_col], dtype=float)
        vals_e = vals_e[np.isfinite(vals_e)]
        fig.add_trace(
            go.Histogram(x=vals_e, name="Effect", marker_color="#3498db",
                         opacity=0.7, nbinsx=20),
            row=1, col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)

    fig.update_layout(height=400, showlegend=True, template="plotly_white")

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return (
        '<div class="section">'
        "<h2>Distribution Analysis</h2>"
        f'<div class="chart-container">{chart_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 3: Kappa Response
# ---------------------------------------------------------------------------

def _section_kappa_response(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    delta_t = meta["delta_treatment"]
    delta_b = meta["delta_baseline"]
    effect_col = meta["effect_col"]

    if "kappa" not in data:
        return ""

    metric_cols = []
    if delta_t in data:
        metric_cols.append(delta_t)
    if delta_b in data:
        metric_cols.append(delta_b)
    if effect_col in data:
        metric_cols.append(effect_col)
    if not metric_cols:
        return ""

    grouped = summary_by_group(data, ["kappa"], metric_cols)
    if not grouped:
        return ""

    kappas = [row["kappa"] for row in grouped]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Default Rate vs Kappa", "Effect vs Kappa"],
    )

    # Plot deltas
    if delta_t in data:
        means_t = [row.get(f"{delta_t}_mean", float("nan")) for row in grouped]
        fig.add_trace(
            go.Scatter(x=kappas, y=means_t, mode="lines+markers",
                       name=meta["treatment_label"], line=dict(color="#e74c3c")),
            row=1, col=1,
        )
    if delta_b in data:
        means_b = [row.get(f"{delta_b}_mean", float("nan")) for row in grouped]
        fig.add_trace(
            go.Scatter(x=kappas, y=means_b, mode="lines+markers",
                       name=meta["baseline_label"], line=dict(color="#95a5a6")),
            row=1, col=1,
        )

    # Plot effect
    if effect_col in data:
        means_e = [row.get(f"{effect_col}_mean", float("nan")) for row in grouped]
        fig.add_trace(
            go.Scatter(x=kappas, y=means_e, mode="lines+markers",
                       name="Effect", line=dict(color="#3498db")),
            row=1, col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_xaxes(title_text="kappa", row=1, col=1)
    fig.update_xaxes(title_text="kappa", row=1, col=2)
    fig.update_yaxes(title_text="delta (default rate)", row=1, col=1)
    fig.update_yaxes(title_text="effect", row=1, col=2)
    fig.update_layout(height=400, template="plotly_white")

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return (
        '<div class="section">'
        "<h2>Kappa Response</h2>"
        f'<div class="chart-container">{chart_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 4: Parameter Heatmaps
# ---------------------------------------------------------------------------

def _section_heatmaps(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go

    effect_col = meta["effect_col"]
    if effect_col not in data:
        return ""

    pairs = [
        ("kappa", "concentration", "kappa x concentration"),
        ("kappa", "mu", "kappa x mu"),
        ("kappa", "outside_mid_ratio", "kappa x rho"),
        ("pool_scale", "kappa", "pool_scale x kappa"),
    ]

    charts = []
    for row_col, col_col, title in pairs:
        if _n_unique(data, row_col) < 2 or _n_unique(data, col_col) < 2:
            continue
        try:
            row_vals, col_vals, matrix = pivot_for_heatmap(
                data, row_col, col_col, effect_col
            )
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"{v:.2f}" for v in col_vals],
                    y=[f"{v:.2f}" for v in row_vals],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Effect"),
                    zmid=0,
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=col_col,
                yaxis_title=row_col,
                height=350,
                template="plotly_white",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception:
            logger.warning("Heatmap %s failed", title, exc_info=True)
            continue

    if not charts:
        return ""

    inner = "".join(f'<div class="chart-container">{c}</div>' for c in charts)
    return f'<div class="section"><h2>Parameter Heatmaps</h2>{inner}</div>'


# ---------------------------------------------------------------------------
# Section 5: Marginal Effects
# ---------------------------------------------------------------------------

def _section_marginal_effects(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    effect_col = meta["effect_col"]
    if effect_col not in data:
        return ""

    params = ["kappa", "concentration", "mu", "outside_mid_ratio", "pool_scale"]
    active_params = [p for p in params if _n_unique(data, p) > 1]

    if not active_params:
        return ""

    n_cols = min(len(active_params), 2)
    n_rows = (len(active_params) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[p for p in active_params],
    )

    for i, param in enumerate(active_params):
        row_idx = i // n_cols + 1
        col_idx = i % n_cols + 1
        grouped = summary_by_group(data, [param], [effect_col])
        if not grouped:
            continue
        x_vals = [str(g[param]) for g in grouped]
        y_vals = [g.get(f"{effect_col}_mean", float("nan")) for g in grouped]
        colors = ["#27ae60" if y > 0 else "#e74c3c" for y in y_vals]
        fig.add_trace(
            go.Bar(x=x_vals, y=y_vals, marker_color=colors, name=param,
                   showlegend=False),
            row=row_idx, col=col_idx,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      row=row_idx, col=col_idx)

    fig.update_layout(height=300 * n_rows, template="plotly_white")

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return (
        '<div class="section">'
        "<h2>Marginal Effects by Parameter</h2>"
        f'<div class="chart-container">{chart_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 6: Network & Contagion
# ---------------------------------------------------------------------------

def _section_contagion(
    data: dict, meta: dict, out_dir: Path, sweep_type: str
) -> str:
    rep_kappas = _representative_kappas(data, n=3)
    if not rep_kappas:
        return ""

    run_id_col = _resolve_col(data, meta["run_id_treatment_candidates"])
    if not run_id_col:
        return ""

    runs_dir = _arm_runs_dir(out_dir, sweep_type, "treatment")

    rows_html = []
    for kappa in rep_kappas:
        run_id = _find_run_for_kappa(data, kappa, run_id_col)
        if not run_id:
            continue
        run_dir = runs_dir / run_id
        events_path = run_dir / "out" / "events.jsonl"
        if not events_path.exists():
            events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue

        params = {"kappa": kappa}
        metrics = extract_run_metrics(
            events_path, run_id, "treatment", sweep_type, params
        )

        n_def = metrics.get("n_defaults", 0)
        n_pri = metrics.get("n_primary", 0)
        n_sec = metrics.get("n_secondary", 0)
        cascade_frac = metrics.get("cascade_fraction", 0.0)
        ttc = metrics.get("time_to_contagion", float("nan"))

        rows_html.append(
            f"<tr>"
            f"<td>{_fmt(kappa, 2)}</td>"
            f"<td>{n_def}</td>"
            f"<td>{n_pri}</td>"
            f"<td>{n_sec}</td>"
            f"<td>{_fmt(cascade_frac, 2)}</td>"
            f"<td>{_fmt(ttc, 1) if np.isfinite(ttc) else 'N/A'}</td>"
            f"</tr>"
        )

    if not rows_html:
        return ""

    return (
        '<div class="section">'
        "<h2>Network &amp; Contagion</h2>"
        "<p>Contagion metrics for representative kappa values "
        f"({meta['treatment_label']} arm).</p>"
        "<table>"
        "<tr><th>kappa</th><th>Total Defaults</th><th>Primary</th>"
        "<th>Secondary</th><th>Cascade Fraction</th>"
        "<th>Time to Contagion</th></tr>"
        + "".join(rows_html)
        + "</table></div>"
    )


# ---------------------------------------------------------------------------
# Section 7: Temporal Dynamics
# ---------------------------------------------------------------------------

def _section_temporal(
    data: dict, meta: dict, out_dir: Path, sweep_type: str
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rep_kappas = _representative_kappas(data, n=3)
    if not rep_kappas:
        return ""

    # Resolve run_id columns for both arms
    treatment_col = _resolve_col(data, meta["run_id_treatment_candidates"])
    baseline_col = _resolve_col(data, meta["run_id_baseline_candidates"])

    arms_info = []
    if treatment_col:
        arms_info.append(("treatment", treatment_col, meta["treatment_label"], "#e74c3c"))
    if baseline_col:
        arms_info.append(("baseline", baseline_col, meta["baseline_label"], "#95a5a6"))

    if not arms_info:
        return ""

    n_kappas = len(rep_kappas)
    fig = make_subplots(
        rows=1, cols=n_kappas,
        subplot_titles=[f"kappa = {k:.2f}" for k in rep_kappas],
    )

    for ki, kappa in enumerate(rep_kappas):
        for arm_key, id_col, label, color in arms_info:
            run_id = _find_run_for_kappa(data, kappa, id_col)
            if not run_id:
                continue
            runs_dir = _arm_runs_dir(out_dir, sweep_type, arm_key)
            run_dir = runs_dir / run_id
            events_path = run_dir / "out" / "events.jsonl"
            if not events_path.exists():
                events_path = run_dir / "events.jsonl"
            if not events_path.exists():
                continue

            ts = extract_run_timeseries(
                events_path, run_id, arm_key, sweep_type, kappa
            )
            if not ts:
                continue

            days = [r["day"] for r in ts]
            cum_total = [
                r["cum_primary_defaults"] + r["cum_secondary_defaults"]
                for r in ts
            ]

            fig.add_trace(
                go.Scatter(
                    x=days, y=cum_total, mode="lines",
                    name=f"{label}", line=dict(color=color),
                    legendgroup=label, showlegend=(ki == 0),
                ),
                row=1, col=ki + 1,
            )

    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Cumulative Defaults", col=1)
    fig.update_layout(height=400, template="plotly_white")

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return (
        '<div class="section">'
        "<h2>Temporal Dynamics</h2>"
        "<p>Cumulative defaults by day for representative kappa values.</p>"
        f'<div class="chart-container">{chart_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 8: Mechanism Internals
# ---------------------------------------------------------------------------

def _section_mechanism(
    data: dict, meta: dict, out_dir: Path, sweep_type: str
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rep_kappas = _representative_kappas(data, n=3)
    if not rep_kappas:
        return ""

    treatment_col = _resolve_col(data, meta["run_id_treatment_candidates"])
    if not treatment_col:
        return ""

    runs_dir = _arm_runs_dir(out_dir, sweep_type, "treatment")
    charts = []

    for kappa in rep_kappas:
        run_id = _find_run_for_kappa(data, kappa, treatment_col)
        if not run_id:
            continue
        run_dir = runs_dir / run_id
        events_path = run_dir / "out" / "events.jsonl"
        if not events_path.exists():
            events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue

        mech_ts = extract_mechanism_timeseries(
            events_path, run_id, "treatment", sweep_type, kappa
        )
        if not mech_ts:
            continue

        days = [r["day"] for r in mech_ts]

        if sweep_type == "dealer":
            # Dealer: trade volumes and counts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Trade Volume (kappa={kappa:.2f})",
                    f"Trade Counts (kappa={kappa:.2f})",
                ],
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("face_bought", 0) for r in mech_ts],
                    name="Face Bought", line=dict(color="#27ae60"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("face_sold", 0) for r in mech_ts],
                    name="Face Sold", line=dict(color="#e74c3c"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("n_buys", 0) for r in mech_ts],
                    name="Buy Count", line=dict(color="#27ae60", dash="dot"),
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("n_sells", 0) for r in mech_ts],
                    name="Sell Count", line=dict(color="#e74c3c", dash="dot"),
                ),
                row=1, col=2,
            )
        elif sweep_type in ("bank",):
            # Bank: loan volumes, CB usage
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Loan Volume (kappa={kappa:.2f})",
                    f"CB Usage (kappa={kappa:.2f})",
                ],
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("loan_volume_issued", 0) for r in mech_ts],
                    name="Loans Issued", line=dict(color="#2980b9"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("repaid_volume", 0) for r in mech_ts],
                    name="Repaid", line=dict(color="#27ae60"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("default_volume", 0) for r in mech_ts],
                    name="Defaults", line=dict(color="#e74c3c"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("cb_loans_created_volume", 0) for r in mech_ts],
                    name="CB Loans Created", line=dict(color="#8e44ad"),
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("cb_loans_repaid_volume", 0) for r in mech_ts],
                    name="CB Loans Repaid", line=dict(color="#2c3e50"),
                ),
                row=1, col=2,
            )
        else:
            # NBFI or generic: loan volumes
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f"Loan Activity (kappa={kappa:.2f})"],
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("loan_volume_issued", 0) for r in mech_ts],
                    name="Loans Issued", line=dict(color="#27ae60"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[r.get("repaid_volume", 0) for r in mech_ts],
                    name="Repaid", line=dict(color="#2980b9"),
                ),
                row=1, col=1,
            )

        fig.update_layout(height=350, template="plotly_white")
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not charts:
        return ""

    inner = "".join(f'<div class="chart-container">{c}</div>' for c in charts)
    return (
        '<div class="section">'
        "<h2>Mechanism Internals</h2>"
        "<p>Per-day mechanism activity for representative kappa values "
        f"({meta['treatment_label']} arm).</p>"
        f"{inner}</div>"
    )


# ---------------------------------------------------------------------------
# Section 9: Loss Analysis
# ---------------------------------------------------------------------------

def _section_loss(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    delta_t = meta["delta_treatment"]
    delta_b = meta["delta_baseline"]

    # Identify which loss columns exist
    loss_col_pairs: list[tuple[str, str, str]] = []

    # Try system_loss_pct and total_loss_pct with various suffixes
    for prefix in ("system_loss_pct_", "total_loss_pct_"):
        for arm_label, delta_col in [
            (meta["treatment_label"], delta_t),
            (meta["baseline_label"], delta_b),
        ]:
            # Derive arm suffix from delta_col: delta_active -> active
            arm_suffix = delta_col.replace("delta_", "")
            col = prefix + arm_suffix
            if col in data:
                loss_col_pairs.append((col, delta_col, arm_label))

    # Also check bank-prefixed variants
    for prefix in ("bank_system_loss_pct_", "bank_total_loss_pct_"):
        for arm_label, delta_col in [
            (meta["treatment_label"], delta_t),
            (meta["baseline_label"], delta_b),
        ]:
            arm_suffix = delta_col.replace("delta_", "")
            col = prefix + arm_suffix
            if col in data:
                loss_col_pairs.append((col, delta_col, arm_label))

    if not loss_col_pairs or "kappa" not in data:
        return ""

    # Group by kappa
    loss_cols = list({c[0] for c in loss_col_pairs})
    grouped = summary_by_group(data, ["kappa"], loss_cols)
    if not grouped:
        return ""

    kappas = [row["kappa"] for row in grouped]
    colors = {"system_loss": "#e74c3c", "total_loss": "#3498db"}

    fig = go.Figure()
    for loss_col, _delta_col, arm_label in loss_col_pairs:
        means = [row.get(f"{loss_col}_mean", float("nan")) for row in grouped]
        color = "#e74c3c" if "system" in loss_col else "#3498db"
        short_name = loss_col.replace("bank_", "").replace("_pct", "")
        fig.add_trace(
            go.Scatter(
                x=kappas, y=means, mode="lines+markers",
                name=f"{short_name} ({arm_label})",
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title="Loss vs Kappa",
        xaxis_title="kappa",
        yaxis_title="Loss (%)",
        height=400,
        template="plotly_white",
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return (
        '<div class="section">'
        "<h2>Loss Analysis</h2>"
        f'<div class="chart-container">{chart_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 10: Regression
# ---------------------------------------------------------------------------

def _section_regression(data: dict, meta: dict) -> str:
    import plotly.graph_objects as go

    effect_col = meta["effect_col"]
    if effect_col not in data:
        return ""

    y = np.array(data[effect_col], dtype=float)
    valid_y = y[np.isfinite(y)]
    if len(valid_y) < 10:
        return ""

    # Build X from available parameters
    x_cols = []
    for col in ["kappa", "concentration", "mu", "outside_mid_ratio", "pool_scale"]:
        if col in data and _n_unique(data, col) > 1:
            x_cols.append(col)

    if not x_cols:
        return ""

    X = np.column_stack([np.array(data[c], dtype=float) for c in x_cols])

    try:
        ols = fit_ols(X, y, feature_names=x_cols, add_intercept=True)
    except Exception:
        logger.warning("OLS fit failed", exc_info=True)
        return ""

    # Coefficient table
    coef_rows = ols.summary_rows()
    rows_html = []
    for r in coef_rows:
        p_class = _pval_class(r["p_value"])
        rows_html.append(
            f"<tr>"
            f"<td style='text-align:left'>{r['feature']}</td>"
            f"<td>{_fmt(r['coef'])}</td>"
            f"<td>{_fmt(r['se'])}</td>"
            f"<td>{_fmt(r['t_stat'], 2)}</td>"
            f'<td class="{p_class}">{_fmt(r["p_value"])}</td>'
            f"<td>{r['sig']}</td>"
            f"</tr>"
        )

    table_html = (
        "<table>"
        "<tr><th style='text-align:left'>Feature</th><th>Coefficient</th>"
        "<th>Std Error</th><th>t-stat</th><th>p-value</th><th>Sig</th></tr>"
        + "".join(rows_html)
        + "</table>"
        f"<p>R&sup2; = {_fmt(ols.r_squared)} | "
        f"Adj R&sup2; = {_fmt(ols.adj_r_squared)} | "
        f"n = {ols.n}</p>"
    )

    # Forest plot: coefficient +/- 2*SE (skip intercept)
    feature_rows = [r for r in coef_rows if r["feature"] != "intercept"]
    if feature_rows:
        names = [r["feature"] for r in feature_rows]
        coefs = [r["coef"] for r in feature_rows]
        ses = [r["se"] for r in feature_rows]
        lower = [c - 2 * s for c, s in zip(coefs, ses)]
        upper = [c + 2 * s for c, s in zip(coefs, ses)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=coefs, y=names, mode="markers",
                marker=dict(size=10, color="#3498db"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[u - c for u, c in zip(upper, coefs)],
                    arrayminus=[c - lo for c, lo in zip(coefs, lower)],
                ),
                name="Coefficient",
            )
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Coefficient Forest Plot (95% CI)",
            xaxis_title="Coefficient",
            height=max(200, 60 * len(names)),
            template="plotly_white",
        )
        forest_html = fig.to_html(full_html=False, include_plotlyjs=False)
    else:
        forest_html = ""

    return (
        '<div class="section">'
        "<h2>OLS Regression</h2>"
        f"{table_html}"
        f'<div class="chart-container">{forest_html}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section 11: Statistical Tests
# ---------------------------------------------------------------------------

def _section_stats_tests(data: dict, meta: dict) -> str:
    effect_col = meta["effect_col"]
    if effect_col not in data:
        return ""

    effect = np.array(data[effect_col], dtype=float)
    valid = effect[np.isfinite(effect)]
    if len(valid) < 5:
        return ""

    ttest = one_sample_ttest(valid)
    wilcox = wilcoxon_test(valid)

    rows_html = []

    # t-test row
    t_p = ttest.get("p_value", float("nan"))
    t_class = _pval_class(t_p)
    rows_html.append(
        f"<tr>"
        f"<td style='text-align:left'>One-sample t-test (H0: mean=0)</td>"
        f"<td>{ttest.get('n', 0)}</td>"
        f"<td>{_fmt(ttest.get('t_stat', float('nan')), 3)}</td>"
        f'<td class="{t_class}">{_fmt(t_p)}</td>'
        f"<td>{'Reject' if np.isfinite(t_p) and t_p < 0.05 else 'Fail to reject'}</td>"
        f"</tr>"
    )

    # Wilcoxon row
    w_p = wilcox.get("p_value", float("nan"))
    w_class = _pval_class(w_p)
    rows_html.append(
        f"<tr>"
        f"<td style='text-align:left'>Wilcoxon signed-rank (H0: median=0)</td>"
        f"<td>{wilcox.get('n', 0)}</td>"
        f"<td>{_fmt(wilcox.get('statistic', float('nan')), 3)}</td>"
        f'<td class="{w_class}">{_fmt(w_p)}</td>'
        f"<td>{'Reject' if np.isfinite(w_p) and w_p < 0.05 else 'Fail to reject'}</td>"
        f"</tr>"
    )

    return (
        '<div class="section">'
        "<h2>Statistical Tests</h2>"
        "<p>Testing whether the treatment effect is significantly different from zero.</p>"
        "<table>"
        "<tr><th style='text-align:left'>Test</th><th>n</th>"
        "<th>Statistic</th><th>p-value</th><th>Decision (alpha=0.05)</th></tr>"
        + "".join(rows_html)
        + "</table></div>"
    )


# ---------------------------------------------------------------------------
# Section 12: Treynor Dashboards
# ---------------------------------------------------------------------------

def _section_treynor(
    data: dict,
    meta: dict,
    out_dir: Path,
    sweep_type: str,
    max_treynor: int,
) -> str:
    from bilancio.analysis.treynor_viz import build_treynor_dashboard

    rep_kappas = _representative_kappas(data, n=max_treynor)
    if not rep_kappas:
        return ""

    treatment_col = _resolve_col(data, meta["run_id_treatment_candidates"])
    if not treatment_col:
        return ""

    runs_dir = _arm_runs_dir(out_dir, sweep_type, "treatment")

    details_blocks = []
    for kappa in rep_kappas:
        run_id = _find_run_for_kappa(data, kappa, treatment_col)
        if not run_id:
            continue
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            continue

        try:
            dashboard_html = build_treynor_dashboard(run_dir)
        except Exception:
            logger.warning(
                "Treynor dashboard failed for kappa=%.2f, run=%s",
                kappa, run_id, exc_info=True,
            )
            continue

        # Strip outer HTML shell - extract body content
        body_content = dashboard_html
        body_start = dashboard_html.find("<body")
        body_end = dashboard_html.rfind("</body>")
        if body_start >= 0 and body_end >= 0:
            # Find end of <body> tag
            body_tag_end = dashboard_html.find(">", body_start)
            if body_tag_end >= 0:
                body_content = dashboard_html[body_tag_end + 1 : body_end]

        details_blocks.append(
            f"<details>"
            f"<summary>Treynor Dashboard: kappa = {kappa:.2f} "
            f"(run: {run_id})</summary>"
            f'<div style="padding: 10px;">{body_content}</div>'
            f"</details>"
        )

    if not details_blocks:
        return ""

    return (
        '<div class="section">'
        "<h2>Treynor Pricing Dashboards</h2>"
        "<p>Detailed Treynor pricing diagnostics for representative kappa values.</p>"
        + "".join(details_blocks)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_comprehensive_report(
    out_dir: Path,
    sweep_type: str,
    max_treynor: int = 3,
) -> str:
    """Generate a self-contained HTML comprehensive report from sweep results.

    Parameters
    ----------
    out_dir : Path
        Root directory of a sweep output (containing aggregate/comparison.csv
        and per-arm run directories).
    sweep_type : str
        One of 'dealer', 'bank', 'nbfi'.
    max_treynor : int
        Maximum number of Treynor dashboards to embed.

    Returns
    -------
    str
        Complete HTML page as a string.
    """
    out_dir = Path(out_dir)
    meta = _sweep_meta(sweep_type)

    # Resolve actual run_id column names after loading data
    data = _load_comparison_data(out_dir)

    # Resolve which run_id column names are present
    meta["run_id_treatment"] = _resolve_col(
        data, meta["run_id_treatment_candidates"]
    )
    meta["run_id_baseline"] = _resolve_col(
        data, meta["run_id_baseline_candidates"]
    )

    title = f"Comprehensive Sweep Report: {sweep_type.title()}"

    # Build all sections with error handling
    sections: list[tuple[str, str, str]] = [
        (
            "summary",
            "Summary",
            _safe_section(_section_summary, data, meta),
        ),
        (
            "distribution",
            "Distribution",
            _safe_section(_section_distribution, data, meta),
        ),
        (
            "kappa-response",
            "Kappa Response",
            _safe_section(_section_kappa_response, data, meta),
        ),
        (
            "heatmaps",
            "Heatmaps",
            _safe_section(_section_heatmaps, data, meta),
        ),
        (
            "marginal-effects",
            "Marginal Effects",
            _safe_section(_section_marginal_effects, data, meta),
        ),
        (
            "contagion",
            "Contagion",
            _safe_section(_section_contagion, data, meta, out_dir, sweep_type),
        ),
        (
            "temporal",
            "Temporal",
            _safe_section(_section_temporal, data, meta, out_dir, sweep_type),
        ),
        (
            "mechanism",
            "Mechanism",
            _safe_section(_section_mechanism, data, meta, out_dir, sweep_type),
        ),
        (
            "loss",
            "Loss Analysis",
            _safe_section(_section_loss, data, meta),
        ),
        (
            "regression",
            "Regression",
            _safe_section(_section_regression, data, meta),
        ),
        (
            "stats-tests",
            "Statistical Tests",
            _safe_section(_section_stats_tests, data, meta),
        ),
        (
            "treynor",
            "Treynor",
            _safe_section(
                _section_treynor, data, meta, out_dir, sweep_type, max_treynor
            ),
        ),
    ]

    # Filter out empty sections from nav (but keep in body for error display)
    active_sections = [
        (anchor, label, content)
        for anchor, label, content in sections
        if content.strip()
    ]

    return _html_shell(title, active_sections)
