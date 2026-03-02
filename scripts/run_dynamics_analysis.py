#!/usr/bin/env python3
"""Dynamics analysis dashboard: time-series, agent heterogeneity, and parameter sensitivity.

Goes beyond aggregate treatment effects to examine HOW mechanisms operate within
individual simulation runs: temporal dynamics of defaults, credit, and liquidity;
how individual agents are affected differently; and how mechanism parameters
shape outcomes.

Selects 9 treatment runs (3 mechanisms x 3 kappa) plus 3 baselines from the
three-way sweep data and generates a 12-chart interactive HTML dashboard.

Usage:
    uv run python scripts/run_dynamics_analysis.py

Output:
    out/three_way/analysis/dynamics_dashboard.html
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "out" / "three_way"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "dynamics_dashboard.html"
THREE_WAY_CSV = BASE_DIR / "three_way_comparison.csv"

KAPPAS = [0.25, 1.0, 4.0]
PREFERRED_C = 1.0
PREFERRED_MU = 0.5

MECHANISMS = ("dealer", "bank", "nbfi")
MECHANISM_LABELS = {"dealer": "Dealer", "bank": "Bank", "nbfi": "NBFI"}

COLORS = {
    "dealer": "#e74c3c",
    "bank": "#2980b9",
    "nbfi": "#27ae60",
    "baseline": "#95a5a6",
}

DASH_STYLES = {0.25: "solid", 1.0: "dash", 4.0: "dot"}
KAPPA_SYMBOLS = {0.25: "circle", 1.0: "square", 4.0: "diamond"}

# Metrics CSV columns we care about
METRICS_COLS = {"delta_t", "phi_t", "M_t", "v_t", "G_t", "HHIplus_t", "day"}


# ===================================================================
# 1. Run selection helpers (reused from run_drilldowns.py pattern)
# ===================================================================


def _approx_eq(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_run_in_csv(
    rows: list[dict[str, str]],
    target_kappa: float,
    run_id_col: str,
    *,
    preferred_c: float = PREFERRED_C,
    preferred_mu: float = PREFERRED_MU,
) -> str | None:
    """Return the first run_id matching target_kappa (and ideally c/mu)."""
    for row in rows:
        k = float(row["kappa"])
        c = float(row["concentration"])
        mu = float(row["mu"])
        if (
            _approx_eq(k, target_kappa)
            and _approx_eq(c, preferred_c)
            and _approx_eq(mu, preferred_mu)
        ):
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid

    for row in rows:
        k = float(row["kappa"])
        c = float(row["concentration"])
        if _approx_eq(k, target_kappa) and _approx_eq(c, preferred_c):
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid

    for row in rows:
        k = float(row["kappa"])
        if _approx_eq(k, target_kappa):
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid

    return None


def _resolve_out_dir(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve the out/ dir for a given run_id."""
    p = runs_dir / run_id / "out"
    if p.is_dir():
        return p
    return None


# ---------------------------------------------------------------------------
# RunSpec: mechanism, arm, kappa, label, out_dir (containing metrics.csv etc.)
# ---------------------------------------------------------------------------

RunSpec = dict[str, Any]


def select_runs() -> list[RunSpec]:
    """Build the list of representative runs to analyse.

    Returns 9 treatment + 3 baseline = 12 runs.
    Baselines use bank_idle at each kappa (since all baselines are equivalent
    passive rings).
    """
    specs: list[RunSpec] = []

    # --- Dealer ---
    for kappa in KAPPAS:
        kappa_str = f"{kappa}"
        dealer_dir = BASE_DIR / f"dealer_k{kappa_str}"
        if not dealer_dir.is_dir():
            alt = f"{kappa:g}"
            dealer_dir = BASE_DIR / f"dealer_k{alt}"
        if not dealer_dir.is_dir():
            print(f"  [WARN] Dealer dir not found for kappa={kappa}")
            continue

        csv_path = dealer_dir / "aggregate" / "comparison.csv"
        if not csv_path.is_file():
            continue
        rows = _read_csv(csv_path)
        active_id = _find_run_in_csv(rows, kappa, "active_run_id")
        if active_id:
            out_dir = _resolve_out_dir(dealer_dir / "active" / "runs", active_id)
            if out_dir:
                specs.append({
                    "mechanism": "dealer",
                    "arm": "active",
                    "kappa": kappa,
                    "label": f"Dealer k={kappa}",
                    "out_dir": out_dir,
                    "is_baseline": False,
                })

    # --- Bank ---
    bank_csv = BASE_DIR / "bank" / "aggregate" / "comparison.csv"
    if bank_csv.is_file():
        rows = _read_csv(bank_csv)
        for kappa in KAPPAS:
            # Treatment: bank_lend
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            if lend_id:
                out_dir = _resolve_out_dir(
                    BASE_DIR / "bank" / "bank_lend" / "runs", lend_id
                )
                if out_dir:
                    specs.append({
                        "mechanism": "bank",
                        "arm": "lend",
                        "kappa": kappa,
                        "label": f"Bank k={kappa}",
                        "out_dir": out_dir,
                        "is_baseline": False,
                    })
            # Baseline: bank_idle (one per kappa)
            idle_id = _find_run_in_csv(rows, kappa, "idle_run_id")
            if idle_id:
                out_dir = _resolve_out_dir(
                    BASE_DIR / "bank" / "bank_idle" / "runs", idle_id
                )
                if out_dir:
                    specs.append({
                        "mechanism": "baseline",
                        "arm": "idle",
                        "kappa": kappa,
                        "label": f"Baseline k={kappa}",
                        "out_dir": out_dir,
                        "is_baseline": True,
                    })

    # --- NBFI ---
    nbfi_csv = BASE_DIR / "nbfi" / "aggregate" / "comparison.csv"
    if nbfi_csv.is_file():
        rows = _read_csv(nbfi_csv)
        for kappa in KAPPAS:
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            if lend_id:
                out_dir = _resolve_out_dir(
                    BASE_DIR / "nbfi" / "nbfi_lend" / "runs", lend_id
                )
                if out_dir:
                    specs.append({
                        "mechanism": "nbfi",
                        "arm": "lend",
                        "kappa": kappa,
                        "label": f"NBFI k={kappa}",
                        "out_dir": out_dir,
                        "is_baseline": False,
                    })

    return specs


# ===================================================================
# 2. Data loading
# ===================================================================


def load_metrics_csv(out_dir: Path) -> list[dict[str, float]]:
    """Load metrics.csv and return list of rows with numeric values."""
    path = out_dir / "metrics.csv"
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for col in reader.fieldnames or []:
                try:
                    parsed[col] = float(row[col])
                except (ValueError, TypeError, KeyError):
                    pass
            if "day" in parsed:
                parsed["day"] = int(parsed["day"])
            rows.append(parsed)
    return rows


def load_events(out_dir: Path) -> list[dict[str, Any]]:
    """Load events.jsonl and return list of event dicts."""
    path = out_dir / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def load_three_way_csv() -> list[dict[str, Any]]:
    """Load three_way_comparison.csv with numeric casting."""
    if not THREE_WAY_CSV.is_file():
        return []
    rows = []
    with open(THREE_WAY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key in row:
                if key == "mechanism":
                    parsed[key] = row[key]
                    continue
                try:
                    parsed[key] = float(row[key])
                except (ValueError, TypeError):
                    parsed[key] = row[key]
            rows.append(parsed)
    return rows


# ===================================================================
# 3. Analysis helpers
# ===================================================================


def _find_spec(specs: list[RunSpec], mechanism: str, kappa: float) -> RunSpec | None:
    for s in specs:
        if s["mechanism"] == mechanism and _approx_eq(s["kappa"], kappa):
            return s
    return None


def _treatment_specs(specs: list[RunSpec]) -> list[RunSpec]:
    return [s for s in specs if not s["is_baseline"]]


def _baseline_specs(specs: list[RunSpec]) -> list[RunSpec]:
    return [s for s in specs if s["is_baseline"]]


# ===================================================================
# 4. Chart builders -- Section 1: Time-Series Dynamics
# ===================================================================


def _build_timeseries_chart(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
    metric_key: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Generic time-series chart: one line per run, colored by mechanism, dashed by kappa."""
    fig = go.Figure()

    # Add baseline line for kappa=0.25 (gray dashed)
    baseline_025 = _find_spec(specs, "baseline", 0.25)
    if baseline_025:
        data = metrics_data.get(baseline_025["label"], [])
        if data and metric_key in data[0]:
            days = [int(r["day"]) for r in data]
            vals = [r.get(metric_key) for r in data]
            fig.add_trace(go.Scatter(
                x=days,
                y=vals,
                mode="lines",
                name="Baseline k=0.25",
                line=dict(color=COLORS["baseline"], dash="dash", width=2),
                opacity=0.7,
            ))

    # Treatment lines
    for mech in MECHANISMS:
        for kappa in KAPPAS:
            spec = _find_spec(specs, mech, kappa)
            if spec is None:
                continue
            data = metrics_data.get(spec["label"], [])
            if not data:
                continue
            if metric_key not in data[0]:
                continue
            days = [int(r["day"]) for r in data]
            vals = [r.get(metric_key) for r in data]
            fig.add_trace(go.Scatter(
                x=days,
                y=vals,
                mode="lines+markers",
                name=spec["label"],
                line=dict(
                    color=COLORS[mech],
                    dash=DASH_STYLES.get(kappa, "solid"),
                    width=2,
                ),
                marker=dict(
                    size=5,
                    symbol=KAPPA_SYMBOLS.get(kappa, "circle"),
                ),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Day",
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=380,
        legend=dict(font=dict(size=10)),
    )
    return fig


def build_chart1_delta_t(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    return _build_timeseries_chart(
        specs,
        metrics_data,
        "delta_t",
        "Chart 1: Default Rate Trajectory (delta_t)",
        "Default Rate (delta_t)",
    )


def build_chart2_phi_t(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    return _build_timeseries_chart(
        specs,
        metrics_data,
        "phi_t",
        "Chart 2: Clearing Rate Trajectory (phi_t)",
        "Clearing Rate (phi_t)",
    )


def build_chart3_M_t(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    return _build_timeseries_chart(
        specs,
        metrics_data,
        "M_t",
        "Chart 3: System Magnification (M_t)",
        "Magnification (M_t)",
    )


def build_chart4_v_t(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    return _build_timeseries_chart(
        specs,
        metrics_data,
        "v_t",
        "Chart 4: Settlement Velocity (v_t)",
        "Velocity (v_t)",
    )


# ===================================================================
# 5. Chart builders -- Section 2: Agent-Level Heterogeneity
# ===================================================================


def build_chart5_gini(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    """Gini coefficient over time for each mechanism at kappa=0.25."""
    fig = go.Figure()
    target_kappa = 0.25

    # Baseline
    baseline = _find_spec(specs, "baseline", target_kappa)
    if baseline:
        data = metrics_data.get(baseline["label"], [])
        if data and "G_t" in data[0]:
            days = [int(r["day"]) for r in data]
            vals = [r.get("G_t") for r in data]
            fig.add_trace(go.Scatter(
                x=days,
                y=vals,
                mode="lines",
                name="Baseline",
                line=dict(color=COLORS["baseline"], dash="dash", width=2),
            ))

    for mech in MECHANISMS:
        spec = _find_spec(specs, mech, target_kappa)
        if spec is None:
            continue
        data = metrics_data.get(spec["label"], [])
        if not data or "G_t" not in data[0]:
            continue
        days = [int(r["day"]) for r in data]
        vals = [r.get("G_t") for r in data]
        fig.add_trace(go.Scatter(
            x=days,
            y=vals,
            mode="lines+markers",
            name=f"{MECHANISM_LABELS[mech]}",
            line=dict(color=COLORS[mech], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=f"Chart 5: Inequality Dynamics (Gini G_t) at k=0.25",
        xaxis_title="Day",
        yaxis_title="Gini Coefficient",
        template="plotly_white",
        height=380,
    )
    return fig


def build_chart6_hhi(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
) -> go.Figure:
    """HHI concentration over time for each mechanism at kappa=0.25."""
    fig = go.Figure()
    target_kappa = 0.25

    baseline = _find_spec(specs, "baseline", target_kappa)
    if baseline:
        data = metrics_data.get(baseline["label"], [])
        if data and "HHIplus_t" in data[0]:
            days = [int(r["day"]) for r in data]
            vals = [r.get("HHIplus_t") for r in data]
            fig.add_trace(go.Scatter(
                x=days,
                y=vals,
                mode="lines",
                name="Baseline",
                line=dict(color=COLORS["baseline"], dash="dash", width=2),
            ))

    for mech in MECHANISMS:
        spec = _find_spec(specs, mech, target_kappa)
        if spec is None:
            continue
        data = metrics_data.get(spec["label"], [])
        if not data or "HHIplus_t" not in data[0]:
            continue
        days = [int(r["day"]) for r in data]
        vals = [r.get("HHIplus_t") for r in data]
        fig.add_trace(go.Scatter(
            x=days,
            y=vals,
            mode="lines+markers",
            name=f"{MECHANISM_LABELS[mech]}",
            line=dict(color=COLORS[mech], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=f"Chart 6: Concentration Dynamics (HHI+_t) at k=0.25",
        xaxis_title="Day",
        yaxis_title="HHI+ (Herfindahl-Hirschman Index)",
        template="plotly_white",
        height=380,
    )
    return fig


def build_chart7_default_timing(
    specs: list[RunSpec],
    events_data: dict[str, list[dict[str, Any]]],
) -> go.Figure:
    """Histogram of default day for each mechanism at kappa=0.25."""
    fig = go.Figure()
    target_kappa = 0.25

    # Also include baseline
    baseline = _find_spec(specs, "baseline", target_kappa)
    if baseline:
        events = events_data.get(baseline["label"], [])
        default_days = [
            e["day"]
            for e in events
            if e.get("kind") == "AgentDefaulted"
        ]
        if default_days:
            fig.add_trace(go.Histogram(
                x=default_days,
                name="Baseline",
                marker_color=COLORS["baseline"],
                opacity=0.5,
                xbins=dict(size=1),
            ))

    for mech in MECHANISMS:
        spec = _find_spec(specs, mech, target_kappa)
        if spec is None:
            continue
        events = events_data.get(spec["label"], [])
        default_days = [
            e["day"]
            for e in events
            if e.get("kind") == "AgentDefaulted"
        ]
        if not default_days:
            continue
        fig.add_trace(go.Histogram(
            x=default_days,
            name=MECHANISM_LABELS[mech],
            marker_color=COLORS[mech],
            opacity=0.6,
            xbins=dict(size=1),
        ))

    fig.update_layout(
        barmode="group",
        title=f"Chart 7: Default Timing Distribution at k=0.25",
        xaxis_title="Day of Default",
        yaxis_title="Number of Defaults",
        template="plotly_white",
        height=380,
    )
    return fig


def build_chart8_survivors_vs_defaulters(
    specs: list[RunSpec],
    events_data: dict[str, list[dict[str, Any]]],
) -> go.Figure:
    """Grouped bar: among agents who accessed the mechanism, how many survived vs defaulted."""
    fig = go.Figure()
    target_kappa = 0.25

    labels = []
    survived_vals = []
    defaulted_vals = []
    bar_colors = []

    for mech in MECHANISMS:
        spec = _find_spec(specs, mech, target_kappa)
        if spec is None:
            continue
        events = events_data.get(spec["label"], [])
        if not events:
            continue

        # Find agents who accessed the mechanism
        accessed_agents: set[str] = set()
        if mech == "dealer":
            for e in events:
                if e.get("kind") == "dealer_trade" and e.get("side") == "sell":
                    trader = e.get("trader", "")
                    if trader:
                        accessed_agents.add(trader)
        elif mech == "bank":
            for e in events:
                if e.get("kind") == "BankLoanIssued":
                    borrower = e.get("borrower", "")
                    if borrower:
                        accessed_agents.add(borrower)
        elif mech == "nbfi":
            for e in events:
                if e.get("kind") == "NonBankLoanCreated":
                    borrower = e.get("borrower_id", "")
                    if borrower:
                        accessed_agents.add(borrower)

        if not accessed_agents:
            continue

        # Find which of those agents defaulted
        defaulted_agents: set[str] = set()
        for e in events:
            if e.get("kind") == "AgentDefaulted":
                agent = e.get("agent", "")
                if agent in accessed_agents:
                    defaulted_agents.add(agent)

        n_accessed = len(accessed_agents)
        n_defaulted = len(defaulted_agents)
        n_survived = n_accessed - n_defaulted

        labels.append(MECHANISM_LABELS[mech])
        survived_vals.append(n_survived)
        defaulted_vals.append(n_defaulted)
        bar_colors.append(COLORS[mech])

    fig.add_trace(go.Bar(
        name="Survived",
        x=labels,
        y=survived_vals,
        marker_color="#27ae60",
        opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        name="Defaulted",
        x=labels,
        y=defaulted_vals,
        marker_color="#c0392b",
        opacity=0.8,
    ))

    fig.update_layout(
        barmode="group",
        title=f"Chart 8: Mechanism Hit Rate at k=0.25 -- Survival Among Users",
        xaxis_title="Mechanism",
        yaxis_title="Number of Agents",
        template="plotly_white",
        height=380,
    )
    return fig


# ===================================================================
# 6. Chart builders -- Section 3: Parameter Sensitivity
# ===================================================================


def _build_heatmap(
    three_way_rows: list[dict[str, Any]],
    mechanism: str,
    x_param: str,
    y_param: str,
    title: str,
    x_label: str,
    y_label: str,
    color_label: str = "effect_mean",
) -> go.Figure:
    """Build a heatmap of effect_mean for a given mechanism over two parameter axes."""
    # Filter rows for this mechanism
    mech_rows = [r for r in three_way_rows if r.get("mechanism") == mechanism]
    if not mech_rows:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)", template="plotly_white", height=420)
        return fig

    # Get unique values for each axis
    x_vals = sorted(set(r[x_param] for r in mech_rows if x_param in r))
    y_vals = sorted(set(r[y_param] for r in mech_rows if y_param in r))

    if not x_vals or not y_vals:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (insufficient data)", template="plotly_white", height=420)
        return fig

    # Build heatmap matrix: average effect_mean over other parameters
    # For each (x, y) cell, average effect_mean across all rows matching that x, y
    z_matrix = []
    text_matrix = []
    for y_val in y_vals:
        z_row = []
        text_row = []
        for x_val in x_vals:
            matching = [
                r[color_label]
                for r in mech_rows
                if _approx_eq(r[x_param], x_val)
                and _approx_eq(r[y_param], y_val)
                and isinstance(r.get(color_label), (int, float))
            ]
            if matching:
                avg = sum(matching) / len(matching)
                z_row.append(avg)
                text_row.append(f"{avg:.4f}<br>n={len(matching)}")
            else:
                z_row.append(None)
                text_row.append("")
        z_matrix.append(z_row)
        text_matrix.append(text_row)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[str(v) for v in x_vals],
        y=[str(v) for v in y_vals],
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="RdYlGn",
        colorbar=dict(title="Effect"),
        hovertemplate=(
            f"{x_label}: %{{x}}<br>"
            f"{y_label}: %{{y}}<br>"
            "Effect: %{z:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=420,
    )
    return fig


def build_chart9_bank_kappa_mu(three_way_rows: list[dict[str, Any]]) -> go.Figure:
    return _build_heatmap(
        three_way_rows,
        "Bank",
        "kappa",
        "mu",
        "Chart 9: Bank Lending Effect -- kappa x mu Interaction",
        "kappa (liquidity ratio)",
        "mu (maturity timing)",
    )


def build_chart10_bank_kappa_c(three_way_rows: list[dict[str, Any]]) -> go.Figure:
    return _build_heatmap(
        three_way_rows,
        "Bank",
        "kappa",
        "concentration",
        "Chart 10: Bank Lending Effect -- kappa x Concentration Interaction",
        "kappa (liquidity ratio)",
        "concentration (debt inequality)",
    )


def build_chart11_dealer_kappa_mu(three_way_rows: list[dict[str, Any]]) -> go.Figure:
    return _build_heatmap(
        three_way_rows,
        "Dealer",
        "kappa",
        "mu",
        "Chart 11: Dealer Trading Effect -- kappa x mu Interaction",
        "kappa (liquidity ratio)",
        "mu (maturity timing)",
    )


def build_chart12_effect_variance(three_way_rows: list[dict[str, Any]]) -> go.Figure:
    """Bar chart: std(effect) by kappa for each mechanism."""
    fig = go.Figure()

    all_kappas = sorted(set(
        r["kappa"] for r in three_way_rows
        if isinstance(r.get("kappa"), (int, float))
    ))
    if not all_kappas:
        fig.update_layout(
            title="Chart 12: Effect Variance by kappa (no data)",
            template="plotly_white",
            height=420,
        )
        return fig

    for mech_key, mech_label in [("Dealer", "Dealer"), ("Bank", "Bank"), ("NBFI", "NBFI")]:
        mech_rows = [r for r in three_way_rows if r.get("mechanism") == mech_key]
        if not mech_rows:
            continue

        kappa_stds = []
        kappa_labels = []
        for kappa in all_kappas:
            effects = [
                r["effect_mean"]
                for r in mech_rows
                if _approx_eq(r["kappa"], kappa)
                and isinstance(r.get("effect_mean"), (int, float))
            ]
            if len(effects) >= 2:
                mean_eff = sum(effects) / len(effects)
                variance = sum((e - mean_eff) ** 2 for e in effects) / (len(effects) - 1)
                std_eff = math.sqrt(variance)
            elif len(effects) == 1:
                std_eff = 0.0
            else:
                std_eff = 0.0
            kappa_stds.append(std_eff)
            kappa_labels.append(str(kappa))

        color_key = mech_label.lower()
        fig.add_trace(go.Bar(
            name=mech_label,
            x=kappa_labels,
            y=kappa_stds,
            marker_color=COLORS.get(color_key, "#888"),
        ))

    fig.update_layout(
        barmode="group",
        title="Chart 12: Effect Variance by kappa -- All Mechanisms",
        xaxis_title="kappa",
        yaxis_title="Std Dev of Treatment Effect",
        template="plotly_white",
        height=420,
    )
    return fig


# ===================================================================
# 7. HTML assembly
# ===================================================================


def _fig_to_div(fig: go.Figure, div_id: str = "") -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def generate_dashboard(
    specs: list[RunSpec],
    metrics_data: dict[str, list[dict[str, float]]],
    events_data: dict[str, list[dict[str, Any]]],
    three_way_rows: list[dict[str, Any]],
) -> str:
    """Build the complete HTML dashboard."""
    n_treatment = len(_treatment_specs(specs))
    n_baseline = len(_baseline_specs(specs))

    print("  Building Section 1: Time-Series Dynamics...")
    chart1 = build_chart1_delta_t(specs, metrics_data)
    chart2 = build_chart2_phi_t(specs, metrics_data)
    chart3 = build_chart3_M_t(specs, metrics_data)
    chart4 = build_chart4_v_t(specs, metrics_data)

    print("  Building Section 2: Agent-Level Heterogeneity...")
    chart5 = build_chart5_gini(specs, metrics_data)
    chart6 = build_chart6_hhi(specs, metrics_data)
    chart7 = build_chart7_default_timing(specs, events_data)
    chart8 = build_chart8_survivors_vs_defaulters(specs, events_data)

    print("  Building Section 3: Parameter Sensitivity...")
    chart9 = build_chart9_bank_kappa_mu(three_way_rows)
    chart10 = build_chart10_bank_kappa_c(three_way_rows)
    chart11 = build_chart11_dealer_kappa_mu(three_way_rows)
    chart12 = build_chart12_effect_variance(three_way_rows)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Dynamics Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5; color: #2c3e50;
    max-width: 1200px; margin: 0 auto;
}}
h1 {{
    border-bottom: 3px solid #2c3e50;
    padding-bottom: 10px;
}}
h2 {{
    margin-top: 40px;
    color: #34495e;
    border-left: 4px solid #3498db;
    padding-left: 12px;
}}
.chart-container {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0;
    padding: 20px;
}}
.interpretation {{
    background: #eaf2f8;
    border-left: 4px solid #3498db;
    padding: 12px 16px;
    margin: 10px 0 20px;
    border-radius: 0 4px 4px 0;
    font-size: 14px;
    color: #2c3e50;
    line-height: 1.5;
}}
.nav {{
    position: sticky; top: 0;
    background: #2c3e50;
    padding: 12px 20px;
    z-index: 100;
    border-radius: 0 0 8px 8px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}
.nav a {{
    color: #ecf0f1;
    text-decoration: none;
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 4px;
    transition: background 0.2s;
}}
.nav a:hover {{ background: #3498db; }}
.legend {{
    display: flex; gap: 20px; margin: 10px 0 20px;
    flex-wrap: wrap;
}}
.legend-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 13px;
}}
.legend-dot {{
    width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid rgba(0,0,0,0.1);
}}
.subtitle {{
    color: #7f8c8d;
    font-size: 14px;
    margin-top: -8px;
}}
.kappa-legend {{
    display: flex; gap: 16px; margin: 4px 0 12px;
    font-size: 12px; color: #7f8c8d;
}}
</style>
</head><body>

<div class="nav">
  <a href="#sec1">1. Time-Series</a>
  <a href="#sec2">2. Agent Heterogeneity</a>
  <a href="#sec3">3. Parameter Sensitivity</a>
</div>

<h1>Dynamics Analysis Dashboard</h1>
<p class="subtitle">Within-run dynamics, agent-level heterogeneity, and parameter interactions
across <strong>{n_treatment} treatment</strong> and <strong>{n_baseline} baseline</strong> runs.</p>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['dealer']}"></div> Dealer</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['bank']}"></div> Bank</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['nbfi']}"></div> NBFI</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['baseline']}"></div> Baseline</div>
</div>
<div class="kappa-legend">
  <span>k=0.25: solid line</span>
  <span>k=1.0: dashed line</span>
  <span>k=4.0: dotted line</span>
</div>

<!-- ============================================================ -->
<h2 id="sec1">Section 1: Time-Series Dynamics</h2>
<p>How do system-level metrics evolve day-by-day within each simulation run?
These charts reveal the <em>shape</em> of the crisis: is it a sudden crash or a
gradual deterioration? Does the mechanism delay defaults or prevent them entirely?</p>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart1, "chart1")}
</div>
<div class="interpretation">
<strong>Default Rate Trajectory:</strong> Shows the cumulative default rate (delta_t) at the
end of each simulation day. Runs that stay lower for longer indicate mechanisms that
successfully delay or prevent defaults. At k=0.25, nearly all systems experience high
default rates, but the trajectory shape reveals whether the mechanism buys time for
agents to settle obligations before cascading failures occur. Compare the slope:
a steep early rise suggests a single wave of defaults; a gradual rise suggests
multiple rounds of distress.
</div>

<div class="chart-container">
{_fig_to_div(chart2, "chart2")}
</div>
<div class="interpretation">
<strong>Clearing Rate Trajectory:</strong> The complement of default rate: what fraction
of total obligations has been settled by each day? Higher is better. Mechanisms that
enable more clearing early (steep upward slope on day 1) provide liquidity that
lubricates the payment chain. The terminal value corresponds to (1 - delta_total).
</div>

<div class="chart-container">
{_fig_to_div(chart3, "chart3")}
</div>
<div class="interpretation">
<strong>System Magnification:</strong> M_t captures how much the payment chain amplifies
initial shortfalls into system-wide losses. A magnification of 10,000 means a small
liquidity gap cascaded into 10,000 units of total system loss. Lower magnification
indicates that the mechanism acts as a shock absorber, preventing small problems
from becoming system-wide crises.
</div>

<div class="chart-container">
{_fig_to_div(chart4, "chart4")}
</div>
<div class="interpretation">
<strong>Settlement Velocity:</strong> v_t measures the speed of payment settlement.
Higher velocity means obligations are cleared more quickly. Dealer trading and bank
lending can increase velocity by providing liquidity precisely when agents need it
to settle payments. A velocity drop on a particular day may indicate a liquidity
crunch on that day.
</div>

<!-- ============================================================ -->
<h2 id="sec2">Section 2: Agent-Level Heterogeneity</h2>
<p>Aggregate metrics hide variation across individual agents. This section examines
how inequality, concentration, default timing, and mechanism access differ at the
agent level, focusing on the most stressed scenario (k=0.25).</p>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart5, "chart5")}
</div>
<div class="interpretation">
<strong>Inequality Dynamics (Gini):</strong> The Gini coefficient measures cash
inequality across agents. A rising Gini indicates that some agents are accumulating
cash while others are losing it -- a polarisation that typically precedes defaults.
Mechanisms that reduce Gini are redistributing liquidity more effectively. A sharp
Gini drop may indicate mass defaults (the poorest agents are expelled, reducing
measured inequality among survivors).
</div>

<div class="chart-container">
{_fig_to_div(chart6, "chart6")}
</div>
<div class="interpretation">
<strong>Concentration Dynamics (HHI+):</strong> The Herfindahl-Hirschman Index of
outstanding obligations. Rising HHI+ means obligations are becoming concentrated in
fewer agents -- a fragility indicator. Mechanisms that keep HHI+ stable are
distributing obligations more evenly, reducing the risk of a single agent failure
causing a cascade.
</div>

<div class="chart-container">
{_fig_to_div(chart7, "chart7")}
</div>
<div class="interpretation">
<strong>Default Timing Distribution:</strong> When do agents fail? A histogram showing
the day-of-default for each mechanism at k=0.25. If a mechanism shifts the distribution
rightward (later defaults), it is buying time for agents. If it reduces the total
count, it is preventing failures outright. A bimodal distribution suggests two waves
of default: primary (own inability to pay) and cascade (knock-on from upstream failures).
</div>

<div class="chart-container">
{_fig_to_div(chart8, "chart8")}
</div>
<div class="interpretation">
<strong>Mechanism Hit Rate:</strong> Among agents who actually used the mechanism
(sold tickets to dealer, received a bank loan, or received an NBFI loan), how many
survived vs defaulted? This measures the mechanism's <em>precision</em>: does it
reach agents it can actually save, or does it extend resources to agents who default
anyway? A high survival rate among users indicates effective targeting.
</div>

<!-- ============================================================ -->
<h2 id="sec3">Section 3: Within-Mechanism Parameter Sensitivity</h2>
<p>Using all {len(three_way_rows)} rows of the three-way comparison data, this section
explores how the treatment effect varies across parameter combinations. Heatmaps
reveal interaction effects that are invisible in one-dimensional analysis.</p>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart9, "chart9")}
</div>
<div class="interpretation">
<strong>Bank: kappa x mu Interaction:</strong> Positive values (green) indicate the bank
is reducing defaults; negative (red) means bank lending increases system default rate
(possible if bank defaults cause additional losses). The mu dimension reveals how
maturity timing interacts with bank lending: at mu=0 (front-loaded stress), banks may
not have time to lend before defaults cascade; at mu=0.5, stress is distributed and
lending has more time to intervene.
</div>

<div class="chart-container">
{_fig_to_div(chart10, "chart10")}
</div>
<div class="interpretation">
<strong>Bank: kappa x Concentration:</strong> Debt inequality (concentration parameter c)
interacts with bank lending effectiveness. Low concentration means a few agents hold
most of the debt; high concentration means debt is spread more evenly. Banks may be
more effective when debt is concentrated (they can target the few most-stressed agents)
or when it is spread (more agents qualify for small loans).
</div>

<div class="chart-container">
{_fig_to_div(chart11, "chart11")}
</div>
<div class="interpretation">
<strong>Dealer: kappa x mu Interaction:</strong> The dealer's secondary market is most
effective when there is two-way demand: stressed agents selling and surplus agents
buying. The mu=0.5 sweet spot typically shows the strongest dealer effect because
maturity dates are spread across the simulation horizon, creating continuous trading
opportunities. At mu=0 (all due early), the crisis hits before trading can help.
</div>

<div class="chart-container">
{_fig_to_div(chart12, "chart12")}
</div>
<div class="interpretation">
<strong>Effect Variance by kappa:</strong> The standard deviation of the treatment effect
across all parameter combinations (concentration, mu, and seeds) at each kappa level.
High variance means the mechanism's effectiveness is highly parameter-dependent at
that liquidity level. Low variance means the mechanism has a consistent effect
regardless of other parameters. This helps identify where mechanism design is robust
vs fragile.
</div>

<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>scripts/run_dynamics_analysis.py</code>
</p>

</body></html>"""

    return html


# ===================================================================
# 8. Main
# ===================================================================


def main() -> None:
    print("=" * 60)
    print("Dynamics Analysis Dashboard")
    print("=" * 60)

    # Step 1: Select runs
    print("\n[1/4] Selecting representative runs...")
    specs = select_runs()
    if not specs:
        print(
            "ERROR: No runs found. Check that out/three_way/ exists and contains data."
        )
        sys.exit(1)

    treatments = _treatment_specs(specs)
    baselines = _baseline_specs(specs)
    print(f"  Found {len(treatments)} treatment + {len(baselines)} baseline = {len(specs)} runs:")
    for s in specs:
        tag = "TREATMENT" if not s["is_baseline"] else "BASELINE"
        print(f"    [{tag:9s}] {s['label']:20s} -> {s['out_dir']}")

    # Step 2: Load metrics.csv for each run
    print(f"\n[2/4] Loading metrics.csv for {len(specs)} runs...")
    metrics_data: dict[str, list[dict[str, float]]] = {}
    for spec in specs:
        label = spec["label"]
        try:
            data = load_metrics_csv(spec["out_dir"])
            metrics_data[label] = data
            n_days = len(data)
            cols = list(data[0].keys()) if data else []
            print(f"    {label:20s}: {n_days} days, cols={cols}")
        except Exception as exc:
            print(f"    {label:20s}: FAILED ({exc})")
            metrics_data[label] = []

    # Step 3: Load events.jsonl for kappa=0.25 runs (needed for charts 7, 8)
    print(f"\n[3/4] Loading events.jsonl for k=0.25 runs...")
    events_data: dict[str, list[dict[str, Any]]] = {}
    target_kappa = 0.25
    for spec in specs:
        if not _approx_eq(spec["kappa"], target_kappa):
            continue
        label = spec["label"]
        try:
            events = load_events(spec["out_dir"])
            events_data[label] = events
            n_defaults = sum(1 for e in events if e.get("kind") == "AgentDefaulted")
            print(f"    {label:20s}: {len(events)} events, {n_defaults} defaults")
        except Exception as exc:
            print(f"    {label:20s}: FAILED ({exc})")
            events_data[label] = []

    # Load three-way comparison CSV for heatmaps
    print("  Loading three_way_comparison.csv...")
    three_way_rows = load_three_way_csv()
    print(f"    {len(three_way_rows)} rows loaded")

    # Step 4: Generate dashboard
    print(f"\n[4/4] Generating dashboard...")
    try:
        html = generate_dashboard(specs, metrics_data, events_data, three_way_rows)
    except Exception as exc:
        print(f"ERROR generating dashboard: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html)
    print(f"\n  Dashboard saved to: {OUTPUT_FILE}")
    print(f'  Open with: open "{OUTPUT_FILE}"')
    print("\nDone.")


if __name__ == "__main__":
    main()
