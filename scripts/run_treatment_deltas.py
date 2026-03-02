#!/usr/bin/env python3
"""Treatment vs Baseline comparison dashboard.

Computes DELTAS (treatment minus baseline) for each mechanism x kappa pair
across the three-way sweep data (dealer, bank, NBFI) and generates an
interactive Plotly HTML dashboard.

Usage:
    uv run python scripts/run_treatment_deltas.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# bilancio analysis imports
# ---------------------------------------------------------------------------
from bilancio.analysis import (
    cash_inflows_by_source,
    contagion_by_day,
    credit_created_by_type,
    credit_destroyed_by_type,
    default_counts_by_type,
    net_credit_impulse,
    node_degree,
    systemic_importance,
    trade_prices_by_day,
    trade_volume_by_day,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent / "out" / "three_way"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "treatment_deltas_dashboard.html"

KAPPAS = [0.25, 1.0, 4.0]
PREFERRED_C = 1.0
PREFERRED_MU = 0.5

MECHANISMS = ("dealer", "bank", "nbfi")
MECHANISM_LABELS = {"dealer": "Dealer", "bank": "Bank", "nbfi": "NBFI"}

# Mechanism colors (treatment)
COLORS = {
    "dealer": "#e74c3c",
    "bank": "#2980b9",
    "nbfi": "#27ae60",
}

# Lighter shades for baseline
COLORS_LIGHT = {
    "dealer": "#f5b7b1",
    "bank": "#aed6f1",
    "nbfi": "#a9dfbf",
}

# Funding source display
SOURCE_ORDER = [
    "settlement_received",
    "ticket_sale",
    "loan_received",
    "cb_loan",
    "nbfi_loan",
    "deposit_interest",
    "deposit",
]
SOURCE_COLORS = {
    "settlement_received": "#3498db",
    "ticket_sale": "#e74c3c",
    "loan_received": "#2ecc71",
    "cb_loan": "#9b59b6",
    "nbfi_loan": "#1abc9c",
    "deposit_interest": "#f39c12",
    "deposit": "#95a5a6",
}


# ===================================================================
# 1. Run selection helpers  (reused from run_drilldowns.py)
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
    # Try preferred c and mu first
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

    # Fall back: match only kappa and preferred_c
    for row in rows:
        k = float(row["kappa"])
        c = float(row["concentration"])
        if _approx_eq(k, target_kappa) and _approx_eq(c, preferred_c):
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid

    # Fall back: match only kappa
    for row in rows:
        k = float(row["kappa"])
        if _approx_eq(k, target_kappa):
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid

    return None


def _events_path_from_run_id(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve events.jsonl for a given run_id inside runs_dir."""
    p = runs_dir / run_id / "out" / "events.jsonl"
    if p.is_file():
        return p
    return None


def _metrics_path_from_run_id(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve metrics.csv for a given run_id inside runs_dir."""
    p = runs_dir / run_id / "out" / "metrics.csv"
    if p.is_file():
        return p
    return None


# A run spec now includes both treatment and baseline
RunPairSpec = dict[str, Any]


def select_run_pairs() -> list[RunPairSpec]:
    """Build paired (treatment, baseline) run specs for each mechanism x kappa."""
    pairs: list[RunPairSpec] = []

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
            print(f"  [WARN] Dealer comparison CSV not found: {csv_path}")
            continue

        rows = _read_csv(csv_path)
        active_id = _find_run_in_csv(rows, kappa, "active_run_id")
        passive_id = _find_run_in_csv(rows, kappa, "passive_run_id")

        if active_id and passive_id:
            t_ep = _events_path_from_run_id(
                dealer_dir / "active" / "runs", active_id
            )
            b_ep = _events_path_from_run_id(
                dealer_dir / "passive" / "runs", passive_id
            )
            t_mp = _metrics_path_from_run_id(
                dealer_dir / "active" / "runs", active_id
            )
            if t_ep and b_ep:
                pairs.append(
                    {
                        "mechanism": "dealer",
                        "kappa": kappa,
                        "treatment_events": t_ep,
                        "baseline_events": b_ep,
                        "treatment_metrics": t_mp,
                        "treatment_arm": "active",
                        "baseline_arm": "passive",
                        "treatment_run_id": active_id,
                        "baseline_run_id": passive_id,
                    }
                )
            else:
                if not t_ep:
                    print(f"  [WARN] Dealer active events not found: {active_id}")
                if not b_ep:
                    print(f"  [WARN] Dealer passive events not found: {passive_id}")

    # --- Bank ---
    bank_csv = BASE_DIR / "bank" / "aggregate" / "comparison.csv"
    if bank_csv.is_file():
        rows = _read_csv(bank_csv)
        for kappa in KAPPAS:
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            idle_id = _find_run_in_csv(rows, kappa, "idle_run_id")
            if lend_id and idle_id:
                t_ep = _events_path_from_run_id(
                    BASE_DIR / "bank" / "bank_lend" / "runs", lend_id
                )
                b_ep = _events_path_from_run_id(
                    BASE_DIR / "bank" / "bank_idle" / "runs", idle_id
                )
                t_mp = _metrics_path_from_run_id(
                    BASE_DIR / "bank" / "bank_lend" / "runs", lend_id
                )
                if t_ep and b_ep:
                    pairs.append(
                        {
                            "mechanism": "bank",
                            "kappa": kappa,
                            "treatment_events": t_ep,
                            "baseline_events": b_ep,
                            "treatment_metrics": t_mp,
                            "treatment_arm": "lend",
                            "baseline_arm": "idle",
                            "treatment_run_id": lend_id,
                            "baseline_run_id": idle_id,
                        }
                    )
    else:
        print(f"  [WARN] Bank comparison CSV not found: {bank_csv}")

    # --- NBFI ---
    nbfi_csv = BASE_DIR / "nbfi" / "aggregate" / "comparison.csv"
    if nbfi_csv.is_file():
        rows = _read_csv(nbfi_csv)
        for kappa in KAPPAS:
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            idle_id = _find_run_in_csv(rows, kappa, "idle_run_id")
            if lend_id and idle_id:
                t_ep = _events_path_from_run_id(
                    BASE_DIR / "nbfi" / "nbfi_lend" / "runs", lend_id
                )
                b_ep = _events_path_from_run_id(
                    BASE_DIR / "nbfi" / "nbfi_idle" / "runs", idle_id
                )
                t_mp = _metrics_path_from_run_id(
                    BASE_DIR / "nbfi" / "nbfi_lend" / "runs", lend_id
                )
                if t_ep and b_ep:
                    pairs.append(
                        {
                            "mechanism": "nbfi",
                            "kappa": kappa,
                            "treatment_events": t_ep,
                            "baseline_events": b_ep,
                            "treatment_metrics": t_mp,
                            "treatment_arm": "lend",
                            "baseline_arm": "idle",
                            "treatment_run_id": lend_id,
                            "baseline_run_id": idle_id,
                        }
                    )
    else:
        print(f"  [WARN] NBFI comparison CSV not found: {nbfi_csv}")

    return pairs


# ===================================================================
# 2. Analysis runner
# ===================================================================


def load_events(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_metrics_csv(path: Path | None) -> list[dict[str, str]]:
    """Load metrics.csv rows; return empty list if path is None or missing."""
    if path is None or not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


AnalysisResult = dict[str, Any]


def _analyse_events(events: list[dict[str, Any]], label: str) -> AnalysisResult:
    """Run all analysis functions on a single set of events."""
    result: AnalysisResult = {"n_events": len(events)}

    # --- Defaults ---
    try:
        result["default_counts"] = default_counts_by_type(events)
    except Exception as exc:
        print(f"  [ERR] default_counts for {label}: {exc}")
        result["default_counts"] = {"primary": 0, "secondary": 0, "total": 0}

    try:
        result["contagion_by_day"] = contagion_by_day(events)
    except Exception as exc:
        print(f"  [ERR] contagion_by_day for {label}: {exc}")
        result["contagion_by_day"] = {}

    # --- Credit ---
    try:
        result["credit_created"] = {
            k: float(v) for k, v in credit_created_by_type(events).items()
        }
    except Exception as exc:
        print(f"  [ERR] credit_created for {label}: {exc}")
        result["credit_created"] = {}

    try:
        result["credit_destroyed"] = {
            k: float(v) for k, v in credit_destroyed_by_type(events).items()
        }
    except Exception as exc:
        print(f"  [ERR] credit_destroyed for {label}: {exc}")
        result["credit_destroyed"] = {}

    try:
        result["net_credit_impulse"] = float(net_credit_impulse(events))
    except Exception as exc:
        print(f"  [ERR] net_credit_impulse for {label}: {exc}")
        result["net_credit_impulse"] = 0.0

    # --- Funding ---
    try:
        per_agent = cash_inflows_by_source(events)
        system_funding: dict[str, float] = defaultdict(float)
        for agent_flows in per_agent.values():
            for source, amount in agent_flows.items():
                system_funding[source] += float(amount)
        result["funding_mix"] = dict(system_funding)
    except Exception as exc:
        print(f"  [ERR] funding_mix for {label}: {exc}")
        result["funding_mix"] = {}

    # --- Pricing ---
    try:
        raw = trade_prices_by_day(events)
        prices: dict[int, list[dict]] = {}
        for day, trades in raw.items():
            prices[day] = [
                {"side": t["side"], "price_ratio": float(t["price_ratio"])}
                for t in trades
            ]
        result["trade_prices_by_day"] = prices
    except Exception as exc:
        print(f"  [ERR] trade_prices for {label}: {exc}")
        result["trade_prices_by_day"] = {}

    try:
        vol = trade_volume_by_day(events)
        result["trade_volume_by_day"] = vol
    except Exception as exc:
        print(f"  [ERR] trade_volume for {label}: {exc}")
        result["trade_volume_by_day"] = {}

    # --- Network ---
    try:
        degrees = node_degree(events)
        result["node_degrees"] = degrees
    except Exception as exc:
        print(f"  [ERR] node_degree for {label}: {exc}")
        result["node_degrees"] = {}

    try:
        si = systemic_importance(events)
        cleaned = []
        for entry in si:
            cleaned.append(
                {
                    "agent_id": entry["agent_id"],
                    "total_obligations": float(entry["total_obligations"]),
                    "betweenness": float(entry["betweenness"]),
                    "score": float(entry["score"]),
                }
            )
        result["systemic_importance"] = cleaned
    except Exception as exc:
        print(f"  [ERR] systemic_importance for {label}: {exc}")
        result["systemic_importance"] = []

    return result


PairResult = dict[str, Any]


def analyse_pair(pair: RunPairSpec) -> PairResult:
    """Analyse both treatment and baseline for a single mechanism x kappa pair."""
    mech = pair["mechanism"]
    kappa = pair["kappa"]
    label = f"{mech}_k{kappa}"

    treatment_events = load_events(pair["treatment_events"])
    baseline_events = load_events(pair["baseline_events"])

    treatment = _analyse_events(treatment_events, f"{label}_treatment")
    baseline = _analyse_events(baseline_events, f"{label}_baseline")

    # Load metrics.csv for the treatment run (for price discovery chart)
    metrics_rows = load_metrics_csv(pair.get("treatment_metrics"))

    return {
        "mechanism": mech,
        "kappa": kappa,
        "treatment": treatment,
        "baseline": baseline,
        "metrics_rows": metrics_rows,
        "pair_spec": pair,
    }


# ===================================================================
# 3. Pair-level helpers
# ===================================================================


def _mech_label(mechanism: str) -> str:
    return MECHANISM_LABELS[mechanism]


def _kappa_label(kappa: float) -> str:
    return f"\u03ba={kappa}"


def _pair_label(mechanism: str, kappa: float) -> str:
    return f"{_mech_label(mechanism)} {_kappa_label(kappa)}"


def _find_pair(
    all_pairs: list[PairResult], mechanism: str, kappa: float
) -> PairResult | None:
    for p in all_pairs:
        if p["mechanism"] == mechanism and _approx_eq(p["kappa"], kappa):
            return p
    return None


# ===================================================================
# 4. Chart builders
# ===================================================================


def build_defaults_comparison(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 1: Grouped bar - Treatment vs Baseline total defaults."""
    labels = []
    baseline_vals = []
    treatment_vals = []
    bar_colors_baseline = []
    bar_colors_treatment = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue
            labels.append(_pair_label(mech, kappa))
            b_total = p["baseline"]["default_counts"].get("total", 0)
            t_total = p["treatment"]["default_counts"].get("total", 0)
            baseline_vals.append(b_total)
            treatment_vals.append(t_total)
            bar_colors_baseline.append(COLORS_LIGHT[mech])
            bar_colors_treatment.append(COLORS[mech])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=labels,
            y=baseline_vals,
            marker_color=bar_colors_baseline,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Treatment",
            x=labels,
            y=treatment_vals,
            marker_color=bar_colors_treatment,
        )
    )
    fig.update_layout(
        barmode="group",
        title="Treatment vs Baseline: Total Defaults",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Number of Defaults",
        template="plotly_white",
        height=450,
    )
    return fig


def build_cascade_reduction(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 2: Grouped bar - Cascade reduction with percentage annotation."""
    labels = []
    baseline_vals = []
    treatment_vals = []
    annotations = []
    bar_colors_baseline = []
    bar_colors_treatment = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue
            label = _pair_label(mech, kappa)
            labels.append(label)
            b_cascade = p["baseline"]["default_counts"].get("secondary", 0)
            t_cascade = p["treatment"]["default_counts"].get("secondary", 0)
            baseline_vals.append(b_cascade)
            treatment_vals.append(t_cascade)
            bar_colors_baseline.append(COLORS_LIGHT[mech])
            bar_colors_treatment.append(COLORS[mech])

            # Compute percentage reduction
            if b_cascade > 0:
                pct = (b_cascade - t_cascade) / b_cascade * 100
                annotations.append((label, max(b_cascade, t_cascade), f"{pct:+.0f}%"))
            else:
                annotations.append((label, max(b_cascade, t_cascade), "n/a"))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline Cascades",
            x=labels,
            y=baseline_vals,
            marker_color=bar_colors_baseline,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Treatment Cascades",
            x=labels,
            y=treatment_vals,
            marker_color=bar_colors_treatment,
        )
    )

    # Add reduction annotations
    for label, y_pos, text in annotations:
        fig.add_annotation(
            x=label,
            y=y_pos + 1,
            text=text,
            showarrow=False,
            font=dict(size=11, color="#2c3e50"),
        )

    fig.update_layout(
        barmode="group",
        title="Cascade Default Reduction: Baseline vs Treatment",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Cascade Defaults",
        template="plotly_white",
        height=450,
    )
    return fig


def build_nci_comparison(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 3: Paired bar - Net Credit Impulse comparison with delta annotation."""
    labels = []
    baseline_vals = []
    treatment_vals = []
    annotations = []
    bar_colors_baseline = []
    bar_colors_treatment = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue
            label = _pair_label(mech, kappa)
            labels.append(label)
            b_nci = p["baseline"]["net_credit_impulse"]
            t_nci = p["treatment"]["net_credit_impulse"]
            baseline_vals.append(b_nci)
            treatment_vals.append(t_nci)
            bar_colors_baseline.append(COLORS_LIGHT[mech])
            bar_colors_treatment.append(COLORS[mech])

            delta = t_nci - b_nci
            y_pos = max(b_nci, t_nci, 0)
            annotations.append(
                (label, y_pos + abs(y_pos) * 0.05 + 100, f"\u0394={delta:+,.0f}")
            )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline NCI",
            x=labels,
            y=baseline_vals,
            marker_color=bar_colors_baseline,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Treatment NCI",
            x=labels,
            y=treatment_vals,
            marker_color=bar_colors_treatment,
        )
    )

    for label, y_pos, text in annotations:
        fig.add_annotation(
            x=label,
            y=y_pos,
            text=text,
            showarrow=False,
            font=dict(size=10, color="#2c3e50"),
        )

    fig.update_layout(
        barmode="group",
        title="Net Credit Impulse: Baseline vs Treatment",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Net Credit Impulse",
        template="plotly_white",
        height=450,
    )
    return fig


def build_credit_creation_heatmap(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 4: Heatmap of credit creation deltas by instrument type."""
    # Collect all instrument types across both treatment and baseline
    all_types: set[str] = set()
    for p in all_pairs:
        all_types.update(p["treatment"]["credit_created"].keys())
        all_types.update(p["baseline"]["credit_created"].keys())
    all_types_sorted = sorted(all_types)

    if not all_types_sorted:
        fig = go.Figure()
        fig.update_layout(title="Credit Creation Deltas: No data available")
        return fig

    # Build column labels (mechanism x kappa)
    col_labels = []
    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is not None:
                col_labels.append(_pair_label(mech, kappa))

    # Build delta matrix: rows = instrument types, cols = mechanism x kappa
    z_matrix = []
    for itype in all_types_sorted:
        row = []
        for kappa in KAPPAS:
            for mech in MECHANISMS:
                p = _find_pair(all_pairs, mech, kappa)
                if p is None:
                    continue
                t_val = p["treatment"]["credit_created"].get(itype, 0)
                b_val = p["baseline"]["credit_created"].get(itype, 0)
                row.append(t_val - b_val)
        z_matrix.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=col_labels,
            y=all_types_sorted,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:+,.0f}" for v in row] for row in z_matrix],
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Delta"),
        )
    )
    fig.update_layout(
        title="Credit Creation Deltas by Instrument Type (Treatment - Baseline)",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Instrument Type",
        template="plotly_white",
        height=max(350, 80 * len(all_types_sorted) + 100),
    )
    return fig


def build_funding_mix_comparison(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 5: Grouped stacked bar - Funding mix at kappa=0.25."""
    target_kappa = 0.25

    # Collect all funding sources present at this kappa
    present_sources: set[str] = set()
    for mech in MECHANISMS:
        p = _find_pair(all_pairs, mech, target_kappa)
        if p is None:
            continue
        present_sources.update(p["treatment"]["funding_mix"].keys())
        present_sources.update(p["baseline"]["funding_mix"].keys())

    sources_to_plot = [s for s in SOURCE_ORDER if s in present_sources]
    for s in sorted(present_sources):
        if s not in sources_to_plot:
            sources_to_plot.append(s)

    # Build x-axis labels: baseline and treatment for each mechanism
    labels = []
    for mech in MECHANISMS:
        p = _find_pair(all_pairs, mech, target_kappa)
        if p is not None:
            labels.append(f"{_mech_label(mech)}\nBaseline")
            labels.append(f"{_mech_label(mech)}\nTreatment")

    fig = go.Figure()
    for source in sources_to_plot:
        vals = []
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, target_kappa)
            if p is None:
                continue
            vals.append(p["baseline"]["funding_mix"].get(source, 0))
            vals.append(p["treatment"]["funding_mix"].get(source, 0))
        color = SOURCE_COLORS.get(source, "#bdc3c7")
        fig.add_trace(go.Bar(name=source, x=labels, y=vals, marker_color=color))

    fig.update_layout(
        barmode="stack",
        title=f"Funding Mix Comparison at {_kappa_label(target_kappa)}",
        xaxis_title="Mechanism / Arm",
        yaxis_title="Total Inflows",
        template="plotly_white",
        height=500,
    )
    return fig


def build_new_funding_sources(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 6: Bar chart - New funding sources enabled by treatment."""
    labels = []
    vals = []
    colors = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue

            b_funding = p["baseline"]["funding_mix"]
            t_funding = p["treatment"]["funding_mix"]

            # New sources: present in treatment but absent (or zero) in baseline
            new_total = 0.0
            for source, amount in t_funding.items():
                baseline_amount = b_funding.get(source, 0)
                if baseline_amount == 0 and amount > 0:
                    new_total += amount

            labels.append(_pair_label(mech, kappa))
            vals.append(new_total)
            colors.append(COLORS[mech])

    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
    fig.update_layout(
        title="New Funding Sources Enabled by Treatment",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Total Amount from New Sources",
        template="plotly_white",
        height=420,
    )
    return fig


def build_price_discovery_chart(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 7: Sell price vs fundamental value for dealer active runs."""
    fig = go.Figure()
    dash_styles = {0.25: "solid", 1.0: "dash", 4.0: "dot"}
    kappa_colors = {0.25: "#c0392b", 1.0: "#e74c3c", 4.0: "#f1948a"}

    for kappa in KAPPAS:
        p = _find_pair(all_pairs, "dealer", kappa)
        if p is None:
            continue

        treatment = p["treatment"]
        prices = treatment.get("trade_prices_by_day", {})

        # Compute fundamental value from metrics.csv
        metrics_rows = p.get("metrics_rows", [])
        fundamental_by_day: dict[int, float] = {}
        for row in metrics_rows:
            try:
                day = int(row["day"])
                delta_t = float(row["delta_t"])
                fundamental_by_day[day] = 1.0 - delta_t
            except (KeyError, ValueError):
                pass

        if prices:
            days = sorted(int(d) for d in prices.keys())
            sell_avgs = []
            for d in days:
                day_trades = prices[d]
                sells = [t["price_ratio"] for t in day_trades if t["side"] == "sell"]
                sell_avgs.append(sum(sells) / len(sells) if sells else None)

            dash = dash_styles.get(kappa, "solid")
            color = kappa_colors.get(kappa, "#e74c3c")

            # Plot sell price
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=sell_avgs,
                    mode="lines+markers",
                    name=f"Sell Price {_kappa_label(kappa)}",
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(size=5),
                )
            )

            # Plot fundamental value on same days
            fund_vals = [fundamental_by_day.get(d) for d in days]
            if any(v is not None for v in fund_vals):
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=fund_vals,
                        mode="lines",
                        name=f"Fundamental {_kappa_label(kappa)}",
                        line=dict(color=color, dash="dot", width=1),
                        opacity=0.6,
                    )
                )

    fig.update_layout(
        title="Price Discovery: Sell Price vs Fundamental Value (Dealer Active)",
        xaxis_title="Day",
        yaxis_title="Price / Face Value",
        template="plotly_white",
        height=450,
    )
    return fig


def build_degree_comparison(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 8: Grouped bar - Mean out-degree comparison."""
    labels = []
    baseline_vals = []
    treatment_vals = []
    bar_colors_baseline = []
    bar_colors_treatment = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue

            label = _pair_label(mech, kappa)
            labels.append(label)

            # Baseline mean out-degree
            b_degrees = p["baseline"]["node_degrees"]
            b_out = [v.get("out_degree", 0) for v in b_degrees.values()]
            b_mean = sum(b_out) / len(b_out) if b_out else 0

            # Treatment mean out-degree
            t_degrees = p["treatment"]["node_degrees"]
            t_out = [v.get("out_degree", 0) for v in t_degrees.values()]
            t_mean = sum(t_out) / len(t_out) if t_out else 0

            baseline_vals.append(b_mean)
            treatment_vals.append(t_mean)
            bar_colors_baseline.append(COLORS_LIGHT[mech])
            bar_colors_treatment.append(COLORS[mech])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=labels,
            y=baseline_vals,
            marker_color=bar_colors_baseline,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Treatment",
            x=labels,
            y=treatment_vals,
            marker_color=bar_colors_treatment,
        )
    )
    fig.update_layout(
        barmode="group",
        title="Mean Out-Degree: Baseline vs Treatment",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Mean Out-Degree",
        template="plotly_white",
        height=420,
    )
    return fig


def build_systemic_importance_delta(all_pairs: list[PairResult]) -> go.Figure:
    """Chart 9: Bar chart - Max systemic importance delta."""
    labels = []
    deltas = []
    colors = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue

            label = _pair_label(mech, kappa)
            labels.append(label)

            t_si = p["treatment"]["systemic_importance"]
            b_si = p["baseline"]["systemic_importance"]
            t_max = t_si[0]["score"] if t_si else 0.0
            b_max = b_si[0]["score"] if b_si else 0.0
            deltas.append(t_max - b_max)
            colors.append(COLORS[mech])

    fig = go.Figure(go.Bar(x=labels, y=deltas, marker_color=colors))

    # Color bars: green if negative (improvement), red if positive (worse)
    marker_colors = []
    for d in deltas:
        if d > 0:
            marker_colors.append("#e74c3c")  # worse: concentrated risk increased
        elif d < 0:
            marker_colors.append("#27ae60")  # better: concentrated risk decreased
        else:
            marker_colors.append("#95a5a6")
    fig.data[0].marker.color = marker_colors

    fig.update_layout(
        title="Max Systemic Importance Delta (Treatment - Baseline)",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Delta (positive = more concentrated risk)",
        template="plotly_white",
        height=420,
    )
    return fig


def build_summary_table(all_pairs: list[PairResult]) -> str:
    """HTML table: summary of deltas for each mechanism x kappa."""
    rows_html = []

    for kappa in KAPPAS:
        for mech in MECHANISMS:
            p = _find_pair(all_pairs, mech, kappa)
            if p is None:
                continue

            treatment = p["treatment"]
            baseline = p["baseline"]

            # Default reduction
            b_total = baseline["default_counts"].get("total", 0)
            t_total = treatment["default_counts"].get("total", 0)
            if b_total > 0:
                default_reduction = (b_total - t_total) / b_total * 100
                dr_text = f"{default_reduction:+.1f}%"
                dr_color = "#27ae60" if default_reduction > 0 else "#e74c3c"
            else:
                dr_text = "n/a"
                dr_color = "#95a5a6"

            # Cascade reduction
            b_cascade = baseline["default_counts"].get("secondary", 0)
            t_cascade = treatment["default_counts"].get("secondary", 0)
            if b_cascade > 0:
                cascade_reduction = (b_cascade - t_cascade) / b_cascade * 100
                cr_text = f"{cascade_reduction:+.1f}%"
                cr_color = "#27ae60" if cascade_reduction > 0 else "#e74c3c"
            else:
                cr_text = "n/a"
                cr_color = "#95a5a6"

            # NCI delta
            nci_delta = treatment["net_credit_impulse"] - baseline["net_credit_impulse"]
            nci_text = f"{nci_delta:+,.0f}"
            nci_color = "#27ae60" if nci_delta > 0 else "#e74c3c" if nci_delta < 0 else "#95a5a6"

            # New funding sources amount
            b_funding = baseline["funding_mix"]
            t_funding = treatment["funding_mix"]
            new_amount = 0.0
            for source, amount in t_funding.items():
                if b_funding.get(source, 0) == 0 and amount > 0:
                    new_amount += amount
            nfs_text = f"{new_amount:,.0f}"

            # Mean degree delta
            b_degrees = baseline["node_degrees"]
            b_out = [v.get("out_degree", 0) for v in b_degrees.values()]
            b_mean_deg = sum(b_out) / len(b_out) if b_out else 0

            t_degrees = treatment["node_degrees"]
            t_out = [v.get("out_degree", 0) for v in t_degrees.values()]
            t_mean_deg = sum(t_out) / len(t_out) if t_out else 0

            deg_delta = t_mean_deg - b_mean_deg
            deg_text = f"{deg_delta:+.2f}"
            # Higher degree is ambiguous: more connections but more contagion channels
            deg_color = "#2c3e50"

            # Max systemic importance delta
            t_si = treatment["systemic_importance"]
            b_si = baseline["systemic_importance"]
            t_max_si = t_si[0]["score"] if t_si else 0.0
            b_max_si = b_si[0]["score"] if b_si else 0.0
            si_delta = t_max_si - b_max_si
            si_text = f"{si_delta:+.3f}"
            si_color = "#e74c3c" if si_delta > 0 else "#27ae60" if si_delta < 0 else "#95a5a6"

            color = COLORS[mech]
            rows_html.append(
                f"<tr>"
                f'<td style="border-left: 4px solid {color}; padding-left: 8px;">'
                f"{_mech_label(mech)}</td>"
                f"<td>{kappa}</td>"
                f'<td style="color: {dr_color}; font-weight: bold;">{dr_text}</td>'
                f'<td style="color: {cr_color}; font-weight: bold;">{cr_text}</td>'
                f'<td style="color: {nci_color}; font-weight: bold;">{nci_text}</td>'
                f"<td>{nfs_text}</td>"
                f'<td style="color: {deg_color};">{deg_text}</td>'
                f'<td style="color: {si_color}; font-weight: bold;">{si_text}</td>'
                f"</tr>"
            )

    return f"""
    <table class="summary-table">
    <thead>
    <tr>
        <th>Mechanism</th>
        <th>&kappa;</th>
        <th>Default Reduction</th>
        <th>Cascade Reduction</th>
        <th>NCI Delta</th>
        <th>New Funding Sources</th>
        <th>Mean Degree &Delta;</th>
        <th>Max Systemic &Delta;</th>
    </tr>
    </thead>
    <tbody>
    {"".join(rows_html)}
    </tbody>
    </table>
    """


# ===================================================================
# 5. HTML assembly
# ===================================================================


def _fig_to_div(fig: go.Figure, div_id: str = "") -> str:
    """Convert a Plotly figure to an HTML div string."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def generate_dashboard(all_pairs: list[PairResult]) -> str:
    """Build the complete HTML dashboard."""
    n_pairs = len(all_pairs)
    mechs_present = sorted(set(p["mechanism"] for p in all_pairs))
    kappas_present = sorted(set(p["kappa"] for p in all_pairs))

    print("Building charts...")

    # Section 1: Default Reduction
    chart_defaults = build_defaults_comparison(all_pairs)
    chart_cascades = build_cascade_reduction(all_pairs)

    # Section 2: Credit Impact
    chart_nci = build_nci_comparison(all_pairs)
    chart_heatmap = build_credit_creation_heatmap(all_pairs)

    # Section 3: Funding Channel Shift
    chart_funding = build_funding_mix_comparison(all_pairs)
    chart_new_funding = build_new_funding_sources(all_pairs)

    # Section 4: Price Impact (Dealer Only)
    chart_price = build_price_discovery_chart(all_pairs)

    # Section 5: Network Complexity
    chart_degree = build_degree_comparison(all_pairs)
    chart_systemic = build_systemic_importance_delta(all_pairs)

    # Section 6: Summary Table
    summary_table = build_summary_table(all_pairs)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Treatment vs Baseline Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5; color: #2c3e50;
}}
h1 {{
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}}
h2 {{
    margin-top: 40px;
    color: #34495e;
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
}}
.nav {{
    position: sticky; top: 0;
    background: #2c3e50;
    padding: 12px 20px;
    z-index: 100;
    border-radius: 0 0 8px 8px;
    margin: -20px -20px 20px;
}}
.nav a {{
    color: #ecf0f1;
    text-decoration: none;
    margin-right: 16px;
    font-size: 13px;
}}
.nav a:hover {{ color: #3498db; }}
.legend {{
    display: flex; gap: 20px; margin: 10px 0;
    flex-wrap: wrap;
}}
.legend-item {{
    display: flex; align-items: center; gap: 6px;
}}
.legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
}}
.legend-dot-outline {{
    width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid; background: transparent;
    box-sizing: border-box;
}}
.summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}
.summary-table th {{
    background: #2c3e50;
    color: white;
    padding: 10px 12px;
    text-align: left;
}}
.summary-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #ecf0f1;
}}
.summary-table tr:hover {{
    background: #f0f3f5;
}}
.pair-count {{
    display: inline-block;
    background: #3498db;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 12px;
    margin-left: 8px;
}}
</style>
</head><body>

<div class="nav">
  <a href="#defaults">1. Default Reduction</a>
  <a href="#credit">2. Credit Impact</a>
  <a href="#funding">3. Funding Shift</a>
  <a href="#pricing">4. Price Impact</a>
  <a href="#network">5. Network Complexity</a>
  <a href="#summary">6. Summary</a>
</div>

<h1>Treatment vs Baseline Comparison</h1>
<p>Analysing <strong>{n_pairs} mechanism &times; &kappa; pairs</strong>,
each with a treatment and baseline run.
Mechanisms: {', '.join(_mech_label(m) for m in mechs_present)}.
&kappa; &isin; {{{', '.join(str(k) for k in kappas_present)}}}.</p>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['dealer']}"></div> Dealer (treatment)</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS_LIGHT['dealer']}"></div> Dealer (baseline)</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['bank']}"></div> Bank (treatment)</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS_LIGHT['bank']}"></div> Bank (baseline)</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['nbfi']}"></div> NBFI (treatment)</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS_LIGHT['nbfi']}"></div> NBFI (baseline)</div>
</div>

<!-- ============================================================ -->
<h2 id="defaults">Section 1: Default Reduction</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_defaults, "defaults_comparison")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Side-by-side comparison of total defaults
    between baseline (no intervention) and treatment (mechanism active) runs.
    The gap between the light bar (baseline) and the dark bar (treatment) is
    the default reduction achieved by each mechanism. Larger gaps at low
    &kappa; indicate the mechanism is effective under liquidity stress.
</div>

<div class="chart-container">
{_fig_to_div(chart_cascades, "cascade_reduction")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Cascade defaults are secondary failures
    triggered by upstream defaults. The percentage annotation shows the
    reduction achieved by each treatment. A mechanism that reduces cascades
    more than primary defaults is specifically effective at breaking
    contagion chains rather than just preventing initial failures.
</div>

<!-- ============================================================ -->
<h2 id="credit">Section 2: Credit Impact</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_nci, "nci_comparison")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Net credit impulse (NCI) measures total
    credit creation minus total credit destruction. The &Delta; annotation
    shows how much the treatment changed the NCI relative to baseline.
    Positive deltas indicate the mechanism is a net credit expander;
    negative deltas suggest it channeled existing credit more efficiently
    without creating new obligations.
</div>

<div class="chart-container">
{_fig_to_div(chart_heatmap, "credit_heatmap")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> This heatmap shows the change in credit
    creation by instrument type when the treatment is active. Blue cells
    indicate the treatment creates more of that instrument type; red cells
    indicate less. Each mechanism should create its signature instrument
    (e.g., bank creates bank_loan, NBFI creates nbfi_loan) while potentially
    reducing payable defaults (which destroy credit).
</div>

<!-- ============================================================ -->
<h2 id="funding">Section 3: Funding Channel Shift</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_funding, "funding_mix")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The stacked bars compare the full
    funding mix between baseline and treatment at &kappa;=0.25 (the most
    stressed scenario). Each mechanism introduces different funding channels:
    dealers add ticket_sale proceeds, banks add loan_received and cb_loan,
    NBFI adds nbfi_loan. The shift in the funding composition shows how
    each mechanism redirects the liquidity landscape.
</div>

<div class="chart-container">
{_fig_to_div(chart_new_funding, "new_funding")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> This chart isolates funding sources that
    are exclusively enabled by the treatment (zero in baseline, positive in
    treatment). These are the genuinely new liquidity channels each mechanism
    opens. Bank lending typically introduces the largest new funding amounts,
    while dealer trading provides secondary-market liquidity.
</div>

<!-- ============================================================ -->
<h2 id="pricing">Section 4: Price Impact (Dealer Only)</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_price, "price_discovery")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> This chart shows the average sell price
    (solid lines) against the fundamental value (dotted lines, computed as
    1 - cumulative default rate) for dealer active runs at each &kappa;.
    When sell prices track fundamentals closely, the dealer provides
    efficient price discovery. Prices persistently below fundamentals
    indicate fire-sale discounts; prices above suggest the market is
    optimistic relative to realized defaults.
</div>

<!-- ============================================================ -->
<h2 id="network">Section 5: Network Complexity</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_degree, "degree_comparison")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Mean out-degree measures the average
    number of creditors each agent owes. Treatment mechanisms that create
    new lending or trading links increase the out-degree. Higher degree
    means more complex interconnection, which can both distribute risk
    (more channels) and amplify contagion (more links for shock transmission).
</div>

<div class="chart-container">
{_fig_to_div(chart_systemic, "systemic_delta")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The change in maximum systemic importance
    score when the treatment is active. Positive values (red) indicate the
    mechanism creates more concentrated risk in the most important node.
    Negative values (green) indicate the mechanism distributes risk more
    evenly. Ideally, a mechanism reduces defaults without creating a
    single-point-of-failure.
</div>

<!-- ============================================================ -->
<h2 id="summary">Section 6: Summary Table</h2>
<!-- ============================================================ -->

<div class="chart-container">
{summary_table}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The summary table consolidates the key
    deltas for each mechanism &times; &kappa; combination. Green values
    indicate improvements (fewer defaults, less concentrated risk); red
    values indicate deterioration. <em>Default Reduction</em> and
    <em>Cascade Reduction</em> are percentages (positive = fewer defaults).
    <em>NCI Delta</em> is the change in net credit impulse. <em>New Funding
    Sources</em> is the total amount from channels that exist only in the
    treatment run. <em>Mean Degree &Delta;</em> is the change in average
    node out-degree. <em>Max Systemic &Delta;</em> is the change in the
    highest systemic importance score (negative = better).
</div>

<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>scripts/run_treatment_deltas.py</code>
</p>

</body></html>"""

    return html


# ===================================================================
# 6. Main
# ===================================================================


def main() -> None:
    print("=" * 60)
    print("Treatment vs Baseline Comparison Dashboard")
    print("=" * 60)

    # Step 1: Select run pairs
    print("\n[1/3] Selecting run pairs...")
    pairs = select_run_pairs()
    if not pairs:
        print(
            "ERROR: No run pairs found. "
            "Check that out/three_way/ exists and contains data."
        )
        sys.exit(1)

    print(f"  Found {len(pairs)} pairs:")
    for p in pairs:
        mech = p["mechanism"]
        kappa = p["kappa"]
        t_arm = p["treatment_arm"]
        b_arm = p["baseline_arm"]
        print(
            f"    {_pair_label(mech, kappa):25s}  "
            f"treatment={t_arm} ({p['treatment_run_id'][:16]}...)  "
            f"baseline={b_arm} ({p['baseline_run_id'][:16]}...)"
        )

    # Step 2: Analyse each pair
    print(f"\n[2/3] Analysing {len(pairs)} pairs (treatment + baseline each)...")
    all_pair_results: list[PairResult] = []
    for i, pair in enumerate(pairs, 1):
        label = _pair_label(pair["mechanism"], pair["kappa"])
        print(f"  [{i}/{len(pairs)}] {label}...", end=" ", flush=True)
        try:
            result = analyse_pair(pair)
            all_pair_results.append(result)
            t_dc = result["treatment"]["default_counts"]
            b_dc = result["baseline"]["default_counts"]
            t_total = t_dc.get("total", 0)
            b_total = b_dc.get("total", 0)
            delta = b_total - t_total
            print(
                f"OK (baseline={b_total} defaults, "
                f"treatment={t_total} defaults, "
                f"reduction={delta})"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_pair_results:
        print("ERROR: All analyses failed.")
        sys.exit(1)

    # Step 3: Generate dashboard
    print(f"\n[3/3] Generating dashboard...")
    html = generate_dashboard(all_pair_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html)
    print(f"\n  Dashboard saved to: {OUTPUT_FILE}")
    print(f"  Open with: open \"{OUTPUT_FILE}\"")
    print("\nDone.")


if __name__ == "__main__":
    main()
