#!/usr/bin/env python3
"""Per-run drill-down analysis across the three-way sweep data.

Selects representative runs from each mechanism (dealer, bank, NBFI) at
three kappa levels (0.25, 1.0, 4.0), runs detailed analysis on each, and
generates an interactive Plotly HTML dashboard.

Usage:
    uv run python scripts/run_drilldowns.py
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
    bid_ask_spread_by_day,
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
OUTPUT_FILE = OUTPUT_DIR / "drilldown_dashboard.html"

KAPPAS = [0.25, 1.0, 4.0]
PREFERRED_C = 1.0
PREFERRED_MU = 0.5

# Plotly mechanism colours
COLORS = {
    "dealer": "#e74c3c",
    "bank": "#2980b9",
    "nbfi": "#27ae60",
}

# Lighter shades for baseline (idle/passive) arms
COLORS_LIGHT = {
    "dealer": "#f5b7b1",
    "bank": "#aed6f1",
    "nbfi": "#a9dfbf",
}


# ===================================================================
# 1. Run selection helpers
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
        if _approx_eq(k, target_kappa) and _approx_eq(c, preferred_c) and _approx_eq(mu, preferred_mu):
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


RunSpec = dict[str, Any]  # keys: mechanism, arm, kappa, label, events_path


def select_runs() -> list[RunSpec]:
    """Build the list of representative runs to analyse."""
    specs: list[RunSpec] = []

    # --- Dealer ---
    for kappa in KAPPAS:
        kappa_str = f"{kappa}"
        # Normalise directory name: dealer_k0.25, dealer_k1.0, dealer_k4.0
        dealer_dir = BASE_DIR / f"dealer_k{kappa_str}"
        if not dealer_dir.is_dir():
            # Try without trailing zero
            alt = f"{kappa:g}"
            dealer_dir = BASE_DIR / f"dealer_k{alt}"
        if not dealer_dir.is_dir():
            print(f"  [WARN] Dealer dir not found for kappa={kappa}: tried {dealer_dir}")
            continue

        csv_path = dealer_dir / "aggregate" / "comparison.csv"
        if csv_path.is_file():
            rows = _read_csv(csv_path)
            # Active run
            active_id = _find_run_in_csv(rows, kappa, "active_run_id")
            if active_id:
                ep = _events_path_from_run_id(dealer_dir / "active" / "runs", active_id)
                if ep:
                    specs.append({
                        "mechanism": "dealer",
                        "arm": "active",
                        "kappa": kappa,
                        "label": f"dealer_active_k{kappa}",
                        "events_path": ep,
                    })
            # Passive baseline
            passive_id = _find_run_in_csv(rows, kappa, "passive_run_id")
            if passive_id:
                ep = _events_path_from_run_id(dealer_dir / "passive" / "runs", passive_id)
                if ep:
                    specs.append({
                        "mechanism": "dealer",
                        "arm": "passive",
                        "kappa": kappa,
                        "label": f"dealer_passive_k{kappa}",
                        "events_path": ep,
                    })

    # --- Bank ---
    bank_csv = BASE_DIR / "bank" / "aggregate" / "comparison.csv"
    if bank_csv.is_file():
        rows = _read_csv(bank_csv)
        for kappa in KAPPAS:
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            if lend_id:
                ep = _events_path_from_run_id(BASE_DIR / "bank" / "bank_lend" / "runs", lend_id)
                if ep:
                    specs.append({
                        "mechanism": "bank",
                        "arm": "lend",
                        "kappa": kappa,
                        "label": f"bank_lend_k{kappa}",
                        "events_path": ep,
                    })
            idle_id = _find_run_in_csv(rows, kappa, "idle_run_id")
            if idle_id:
                ep = _events_path_from_run_id(BASE_DIR / "bank" / "bank_idle" / "runs", idle_id)
                if ep:
                    specs.append({
                        "mechanism": "bank",
                        "arm": "idle",
                        "kappa": kappa,
                        "label": f"bank_idle_k{kappa}",
                        "events_path": ep,
                    })

    # --- NBFI ---
    nbfi_csv = BASE_DIR / "nbfi" / "aggregate" / "comparison.csv"
    if nbfi_csv.is_file():
        rows = _read_csv(nbfi_csv)
        for kappa in KAPPAS:
            lend_id = _find_run_in_csv(rows, kappa, "lend_run_id")
            if lend_id:
                ep = _events_path_from_run_id(BASE_DIR / "nbfi" / "nbfi_lend" / "runs", lend_id)
                if ep:
                    specs.append({
                        "mechanism": "nbfi",
                        "arm": "lend",
                        "kappa": kappa,
                        "label": f"nbfi_lend_k{kappa}",
                        "events_path": ep,
                    })
            idle_id = _find_run_in_csv(rows, kappa, "idle_run_id")
            if idle_id:
                ep = _events_path_from_run_id(BASE_DIR / "nbfi" / "nbfi_idle" / "runs", idle_id)
                if ep:
                    specs.append({
                        "mechanism": "nbfi",
                        "arm": "idle",
                        "kappa": kappa,
                        "label": f"nbfi_idle_k{kappa}",
                        "events_path": ep,
                    })

    return specs


# ===================================================================
# 2. Analysis runner
# ===================================================================


def load_events(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


AnalysisResult = dict[str, Any]


def analyse_run(spec: RunSpec) -> AnalysisResult:
    """Run all analysis functions on a single run, returning a dict of results."""
    label = spec["label"]
    result: AnalysisResult = {"spec": spec}
    events = load_events(spec["events_path"])
    result["n_events"] = len(events)

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
        # Aggregate across agents into system-level totals
        system_funding: dict[str, float] = defaultdict(float)
        for agent_flows in per_agent.values():
            for source, amount in agent_flows.items():
                system_funding[source] += float(amount)
        result["funding_mix"] = dict(system_funding)
    except Exception as exc:
        print(f"  [ERR] funding_mix for {label}: {exc}")
        result["funding_mix"] = {}

    # --- Pricing (dealer only) ---
    is_dealer_active = spec["mechanism"] == "dealer" and spec["arm"] == "active"
    if is_dealer_active:
        try:
            raw = trade_prices_by_day(events)
            # Convert Decimals to floats for serialisation
            prices: dict[int, list[dict]] = {}
            for day, trades in raw.items():
                prices[day] = [
                    {
                        "side": t["side"],
                        "price_ratio": float(t["price_ratio"]),
                    }
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

        try:
            spread_raw = bid_ask_spread_by_day(events)
            result["bid_ask_spread_by_day"] = {
                d: float(v) if v is not None else None for d, v in spread_raw.items()
            }
        except Exception as exc:
            print(f"  [ERR] bid_ask_spread for {label}: {exc}")
            result["bid_ask_spread_by_day"] = {}

    else:
        result["trade_prices_by_day"] = {}
        result["trade_volume_by_day"] = {}
        result["bid_ask_spread_by_day"] = {}

    # --- Network ---
    try:
        degrees = node_degree(events)
        result["node_degrees"] = degrees
    except Exception as exc:
        print(f"  [ERR] node_degree for {label}: {exc}")
        result["node_degrees"] = {}

    try:
        si = systemic_importance(events)
        # Convert Decimals to float
        cleaned = []
        for entry in si:
            cleaned.append({
                "agent_id": entry["agent_id"],
                "total_obligations": float(entry["total_obligations"]),
                "betweenness": float(entry["betweenness"]),
                "score": float(entry["score"]),
            })
        result["systemic_importance"] = cleaned
    except Exception as exc:
        print(f"  [ERR] systemic_importance for {label}: {exc}")
        result["systemic_importance"] = []

    return result


# ===================================================================
# 3. Dashboard generation
# ===================================================================


def _mech_label(mechanism: str) -> str:
    return {"dealer": "Dealer", "bank": "Bank", "nbfi": "NBFI"}[mechanism]


def _kappa_label(kappa: float) -> str:
    return f"\u03ba={kappa}"


def _treatment_key(mechanism: str, kappa: float) -> str:
    return f"{_mech_label(mechanism)} {_kappa_label(kappa)}"


def _get_treatment_results(
    all_results: list[AnalysisResult],
) -> list[AnalysisResult]:
    """Return only the treatment (non-baseline) results."""
    return [
        r for r in all_results
        if r["spec"]["arm"] in ("active", "lend")
    ]


def _get_baseline_results(
    all_results: list[AnalysisResult],
) -> list[AnalysisResult]:
    """Return only the baseline results."""
    return [
        r for r in all_results
        if r["spec"]["arm"] in ("passive", "idle")
    ]


def _find_result(
    all_results: list[AnalysisResult],
    mechanism: str,
    arm: str,
    kappa: float,
) -> AnalysisResult | None:
    for r in all_results:
        s = r["spec"]
        if s["mechanism"] == mechanism and s["arm"] == arm and _approx_eq(s["kappa"], kappa):
            return r
    return None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def build_default_classification_bar(treatments: list[AnalysisResult]) -> go.Figure:
    """Stacked bar: primary vs cascade defaults by mechanism x kappa."""
    labels = []
    primary_vals = []
    secondary_vals = []

    for kappa in KAPPAS:
        for mech in ("dealer", "bank", "nbfi"):
            r = None
            for t in treatments:
                s = t["spec"]
                if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                    r = t
                    break
            if r is None:
                continue
            dc = r.get("default_counts", {})
            labels.append(f"{_mech_label(mech)}\n{_kappa_label(kappa)}")
            primary_vals.append(dc.get("primary", 0))
            secondary_vals.append(dc.get("secondary", 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Primary",
        x=labels,
        y=primary_vals,
        marker_color="#2c3e50",
    ))
    fig.add_trace(go.Bar(
        name="Cascade",
        x=labels,
        y=secondary_vals,
        marker_color="#e67e22",
    ))
    fig.update_layout(
        barmode="stack",
        title="Default Classification: Primary vs Cascade",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Number of defaults",
        template="plotly_white",
        height=420,
    )
    return fig


def build_contagion_timeline(all_results: list[AnalysisResult]) -> go.Figure:
    """Line chart: contagion_by_day for kappa=0.25 across all mechanisms."""
    fig = go.Figure()
    target_kappa = 0.25
    for mech in ("dealer", "bank", "nbfi"):
        # Treatment arm
        arm = "active" if mech == "dealer" else "lend"
        r = _find_result(all_results, mech, arm, target_kappa)
        if r is None:
            continue
        cbd = r.get("contagion_by_day", {})
        if not cbd:
            continue
        days = sorted(cbd.keys())
        primary = [cbd[d].get("primary", 0) for d in days]
        secondary = [cbd[d].get("secondary", 0) for d in days]
        fig.add_trace(go.Scatter(
            x=days,
            y=primary,
            mode="lines+markers",
            name=f"{_mech_label(mech)} primary",
            line=dict(color=COLORS[mech]),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=days,
            y=secondary,
            mode="lines+markers",
            name=f"{_mech_label(mech)} cascade",
            line=dict(color=COLORS[mech], dash="dash"),
            marker=dict(size=5, symbol="x"),
        ))

    fig.update_layout(
        title=f"Contagion Timeline at {_kappa_label(target_kappa)} (Treatment Arms)",
        xaxis_title="Day",
        yaxis_title="Defaults",
        template="plotly_white",
        height=400,
    )
    return fig


def build_credit_created_bar(treatments: list[AnalysisResult]) -> go.Figure:
    """Grouped bar: credit created by instrument type across mechanisms."""
    # Collect all instrument types
    all_types: set[str] = set()
    for r in treatments:
        all_types.update(r.get("credit_created", {}).keys())
    all_types_sorted = sorted(all_types)

    fig = go.Figure()
    for itype in all_types_sorted:
        labels = []
        vals = []
        for kappa in KAPPAS:
            for mech in ("dealer", "bank", "nbfi"):
                r = None
                for t in treatments:
                    s = t["spec"]
                    if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                        r = t
                        break
                if r is None:
                    continue
                labels.append(f"{_mech_label(mech)}\n{_kappa_label(kappa)}")
                vals.append(r.get("credit_created", {}).get(itype, 0))
        fig.add_trace(go.Bar(name=itype, x=labels, y=vals))

    fig.update_layout(
        barmode="group",
        title="Credit Created by Instrument Type",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Credit Created",
        template="plotly_white",
        height=450,
    )
    return fig


def build_net_credit_impulse_bar(treatments: list[AnalysisResult]) -> go.Figure:
    """Bar chart: net credit impulse for each mechanism x kappa."""
    labels = []
    vals = []
    colors = []

    for kappa in KAPPAS:
        for mech in ("dealer", "bank", "nbfi"):
            r = None
            for t in treatments:
                s = t["spec"]
                if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                    r = t
                    break
            if r is None:
                continue
            labels.append(f"{_mech_label(mech)}\n{_kappa_label(kappa)}")
            vals.append(r.get("net_credit_impulse", 0))
            colors.append(COLORS[mech])

    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
    fig.update_layout(
        title="Net Credit Impulse (Created - Destroyed)",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Net Credit Impulse",
        template="plotly_white",
        height=400,
    )
    return fig


def build_funding_mix_bar(treatments: list[AnalysisResult]) -> go.Figure:
    """Stacked bar: system-level funding sources per mechanism x kappa."""
    # Ordered source types
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

    # Collect all source types actually present
    present_sources: set[str] = set()
    for r in treatments:
        present_sources.update(r.get("funding_mix", {}).keys())
    sources_to_plot = [s for s in SOURCE_ORDER if s in present_sources]
    # Add any that are present but not in our known order
    for s in sorted(present_sources):
        if s not in sources_to_plot:
            sources_to_plot.append(s)

    labels = []
    for kappa in KAPPAS:
        for mech in ("dealer", "bank", "nbfi"):
            found = False
            for t in treatments:
                s = t["spec"]
                if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                    found = True
                    break
            if found:
                labels.append(f"{_mech_label(mech)}\n{_kappa_label(kappa)}")

    fig = go.Figure()
    for source in sources_to_plot:
        vals = []
        for kappa in KAPPAS:
            for mech in ("dealer", "bank", "nbfi"):
                r = None
                for t in treatments:
                    st = t["spec"]
                    if st["mechanism"] == mech and _approx_eq(st["kappa"], kappa):
                        r = t
                        break
                if r is None:
                    continue
                vals.append(r.get("funding_mix", {}).get(source, 0))
        color = SOURCE_COLORS.get(source, "#bdc3c7")
        fig.add_trace(go.Bar(name=source, x=labels, y=vals, marker_color=color))

    fig.update_layout(
        barmode="stack",
        title="System-Level Funding Sources",
        xaxis_title="Mechanism / Kappa",
        yaxis_title="Total Inflows",
        template="plotly_white",
        height=480,
    )
    return fig


def build_price_discovery_chart(dealer_results: list[AnalysisResult]) -> go.Figure:
    """Line chart: average buy/sell price ratio by day for dealer runs."""
    fig = go.Figure()
    dash_styles = {0.25: "solid", 1.0: "dash", 4.0: "dot"}

    for r in dealer_results:
        kappa = r["spec"]["kappa"]
        prices = r.get("trade_prices_by_day", {})
        if not prices:
            continue

        days = sorted(int(d) for d in prices.keys())
        buy_avgs = []
        sell_avgs = []
        for d in days:
            day_trades = prices[d]
            buys = [t["price_ratio"] for t in day_trades if t["side"] == "buy"]
            sells = [t["price_ratio"] for t in day_trades if t["side"] == "sell"]
            buy_avgs.append(sum(buys) / len(buys) if buys else None)
            sell_avgs.append(sum(sells) / len(sells) if sells else None)

        dash = dash_styles.get(kappa, "solid")
        fig.add_trace(go.Scatter(
            x=days, y=buy_avgs,
            mode="lines+markers",
            name=f"Buy avg {_kappa_label(kappa)}",
            line=dict(color=COLORS["dealer"], dash=dash),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=days, y=sell_avgs,
            mode="lines+markers",
            name=f"Sell avg {_kappa_label(kappa)}",
            line=dict(color="#c0392b", dash=dash),
            marker=dict(size=4, symbol="triangle-up"),
        ))

    fig.update_layout(
        title="Price Discovery: Buy/Sell Price Ratios (Dealer Active Runs)",
        xaxis_title="Day",
        yaxis_title="Price / Face Value",
        template="plotly_white",
        height=420,
    )
    return fig


def build_trade_volume_chart(dealer_results: list[AnalysisResult]) -> go.Figure:
    """Bar chart: trade volume by day for kappa=0.25 dealer run."""
    # Find kappa=0.25
    r_target = None
    for r in dealer_results:
        if _approx_eq(r["spec"]["kappa"], 0.25):
            r_target = r
            break
    if r_target is None:
        # Fall back to first available
        r_target = dealer_results[0] if dealer_results else None

    fig = go.Figure()
    if r_target:
        vol = r_target.get("trade_volume_by_day", {})
        if vol:
            days = sorted(vol.keys())
            buys = [vol[d].get("buys", 0) for d in days]
            sells = [vol[d].get("sells", 0) for d in days]
            fig.add_trace(go.Bar(name="Buys", x=days, y=buys, marker_color="#27ae60"))
            fig.add_trace(go.Bar(name="Sells", x=days, y=sells, marker_color="#e74c3c"))
            kappa = r_target["spec"]["kappa"]
            fig.update_layout(
                title=f"Trade Volume by Day (Dealer Active, {_kappa_label(kappa)})",
            )

    fig.update_layout(
        barmode="group",
        xaxis_title="Day",
        yaxis_title="Number of Trades",
        template="plotly_white",
        height=400,
    )
    return fig


def build_degree_distribution_box(all_results: list[AnalysisResult]) -> go.Figure:
    """Box plot: out-degree distribution by mechanism x kappa (treatment runs)."""
    treatments = _get_treatment_results(all_results)
    fig = go.Figure()

    for kappa in KAPPAS:
        for mech in ("dealer", "bank", "nbfi"):
            r = None
            for t in treatments:
                s = t["spec"]
                if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                    r = t
                    break
            if r is None:
                continue
            degrees = r.get("node_degrees", {})
            if not degrees:
                continue
            out_degs = [v.get("out_degree", 0) for v in degrees.values()]
            label = f"{_mech_label(mech)} {_kappa_label(kappa)}"
            fig.add_trace(go.Box(
                y=out_degs,
                name=label,
                marker_color=COLORS[mech],
                boxmean=True,
            ))

    fig.update_layout(
        title="Obligation Network: Out-Degree Distribution",
        yaxis_title="Out-Degree (number of creditors)",
        template="plotly_white",
        height=450,
    )
    return fig


def build_systemic_importance_bar(all_results: list[AnalysisResult]) -> go.Figure:
    """Bar chart: top-5 systemically important agents for each mechanism at kappa=0.25."""
    fig = go.Figure()
    target_kappa = 0.25

    for mech in ("dealer", "bank", "nbfi"):
        arm = "active" if mech == "dealer" else "lend"
        r = _find_result(all_results, mech, arm, target_kappa)
        if r is None:
            continue
        si = r.get("systemic_importance", [])
        top5 = si[:5]
        if not top5:
            continue
        agents = [e["agent_id"][:12] for e in top5]
        scores = [e["score"] for e in top5]
        fig.add_trace(go.Bar(
            name=_mech_label(mech),
            x=agents,
            y=scores,
            marker_color=COLORS[mech],
        ))

    fig.update_layout(
        title=f"Top-5 Systemically Important Agents ({_kappa_label(target_kappa)})",
        xaxis_title="Agent ID",
        yaxis_title="Systemic Importance Score",
        barmode="group",
        template="plotly_white",
        height=400,
    )
    return fig


def build_summary_table(all_results: list[AnalysisResult]) -> str:
    """HTML table summarising key metrics for each treatment run."""
    treatments = _get_treatment_results(all_results)

    rows_html = []
    for kappa in KAPPAS:
        for mech in ("dealer", "bank", "nbfi"):
            r = None
            for t in treatments:
                s = t["spec"]
                if s["mechanism"] == mech and _approx_eq(s["kappa"], kappa):
                    r = t
                    break
            if r is None:
                continue

            dc = r.get("default_counts", {})
            primary = dc.get("primary", 0)
            secondary = dc.get("secondary", 0)
            total = dc.get("total", 0)
            cascade_frac = f"{secondary / total:.2%}" if total > 0 else "n/a"
            nci = r.get("net_credit_impulse", 0)

            # Funding diversity: count of distinct non-zero sources
            fm = r.get("funding_mix", {})
            funding_diversity = sum(1 for v in fm.values() if v > 0)

            # Mean out-degree
            degrees = r.get("node_degrees", {})
            out_degs = [v.get("out_degree", 0) for v in degrees.values()]
            mean_out = f"{sum(out_degs) / len(out_degs):.1f}" if out_degs else "n/a"

            # Max systemic importance
            si = r.get("systemic_importance", [])
            max_si = f"{si[0]['score']:.3f}" if si else "n/a"

            color = COLORS[mech]
            rows_html.append(
                f"<tr>"
                f'<td style="border-left: 4px solid {color}; padding-left: 8px;">'
                f"{_mech_label(mech)}</td>"
                f"<td>{kappa}</td>"
                f"<td>{primary}</td>"
                f"<td>{secondary}</td>"
                f"<td>{cascade_frac}</td>"
                f"<td>{nci:,.0f}</td>"
                f"<td>{funding_diversity}</td>"
                f"<td>{mean_out}</td>"
                f"<td>{max_si}</td>"
                f"</tr>"
            )

    return f"""
    <table class="summary-table">
    <thead>
    <tr>
        <th>Mechanism</th>
        <th>&kappa;</th>
        <th>Primary Defaults</th>
        <th>Cascade Defaults</th>
        <th>Cascade Fraction</th>
        <th>Net Credit Impulse</th>
        <th>Funding Diversity</th>
        <th>Mean Out-Degree</th>
        <th>Max Systemic Score</th>
    </tr>
    </thead>
    <tbody>
    {"".join(rows_html)}
    </tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


def _fig_to_div(fig: go.Figure, div_id: str = "") -> str:
    """Convert a Plotly figure to an HTML div string."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def generate_dashboard(all_results: list[AnalysisResult]) -> str:
    """Build the complete HTML dashboard."""
    treatments = _get_treatment_results(all_results)
    dealer_active = [r for r in treatments if r["spec"]["mechanism"] == "dealer"]

    # Build all charts
    print("Building charts...")
    chart_default_bar = build_default_classification_bar(treatments)
    chart_contagion = build_contagion_timeline(all_results)
    chart_credit_created = build_credit_created_bar(treatments)
    chart_nci = build_net_credit_impulse_bar(treatments)
    chart_funding = build_funding_mix_bar(treatments)
    chart_price = build_price_discovery_chart(dealer_active)
    chart_volume = build_trade_volume_chart(dealer_active)
    chart_degree = build_degree_distribution_box(all_results)
    chart_systemic = build_systemic_importance_bar(all_results)
    summary_table = build_summary_table(all_results)

    # Count runs
    n_treatment = len(treatments)
    n_baseline = len(_get_baseline_results(all_results))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Three-Way Drill-Down Analysis</title>
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
}}
.legend-item {{
    display: flex; align-items: center; gap: 6px;
}}
.legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
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
.run-count {{
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
  <a href="#defaults">1. Default Classification</a>
  <a href="#credit">2. Credit Dynamics</a>
  <a href="#funding">3. Funding Mix</a>
  <a href="#pricing">4. Price Discovery</a>
  <a href="#network">5. Network Topology</a>
  <a href="#summary">6. Summary</a>
</div>

<h1>Three-Way Drill-Down Analysis</h1>
<p>Per-run deep analysis across <strong>{n_treatment} treatment</strong> and
<strong>{n_baseline} baseline</strong> runs at
&kappa; &isin; {{{', '.join(str(k) for k in KAPPAS)}}}.</p>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['dealer']}"></div> Dealer</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['bank']}"></div> Bank</div>
  <div class="legend-item"><div class="legend-dot" style="background: {COLORS['nbfi']}"></div> NBFI</div>
</div>

<!-- ============================================================ -->
<h2 id="defaults">Section 1: Default Classification</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_default_bar, "default_bar")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> This chart decomposes total defaults into
    primary failures (agents that defaulted on their own) and cascade failures
    (agents that defaulted because upstream debtors failed first). A higher
    cascade fraction suggests greater systemic fragility. Compare how each
    mechanism affects the cascade-to-primary ratio at different liquidity
    levels.
</div>

<div class="chart-container">
{_fig_to_div(chart_contagion, "contagion_timeline")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The contagion timeline at &kappa;=0.25
    (severe stress) shows how defaults propagate over the simulation horizon.
    Mechanisms that delay or reduce cascade defaults are providing effective
    liquidity insurance. Compare the timing and magnitude of cascade waves
    across the three mechanisms.
</div>

<!-- ============================================================ -->
<h2 id="credit">Section 2: Credit Dynamics</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_credit_created, "credit_created")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Credit creation by instrument type shows
    which financial instruments each mechanism generates. Bank lending creates
    bank_loan credit, NBFI creates nbfi_loan credit, while dealer trading
    redistributes existing claims without creating new credit. The payable
    category represents the baseline trade credit in the ring.
</div>

<div class="chart-container">
{_fig_to_div(chart_nci, "net_credit_impulse")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Net credit impulse (creation minus
    destruction) shows whether each mechanism is a net expander or
    contractor of credit. Positive values indicate net credit expansion;
    negative values indicate more defaults/writeoffs than new lending.
    Banking mechanisms typically show larger impulses due to endogenous
    money creation.
</div>

<!-- ============================================================ -->
<h2 id="funding">Section 3: Funding Mix</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_funding, "funding_mix")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The funding mix reveals where agents
    obtain cash to meet obligations. Settlement receipts are the primary
    source in all scenarios. Dealer runs add ticket_sale proceeds, bank runs
    add loan_received and cb_loan, NBFI runs add nbfi_loan. Higher funding
    diversity generally indicates a more resilient system with multiple
    liquidity channels.
</div>

<!-- ============================================================ -->
<h2 id="pricing">Section 4: Price Discovery (Dealer Only)</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_price, "price_discovery")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Price discovery shows how buy and sell
    prices evolve relative to face value. In stressed conditions
    (&kappa;=0.25), sell prices drop as agents accept larger discounts to
    raise liquidity. The spread between buy and sell averages reflects
    the dealer's intermediation cost. Convergence of prices toward
    (1 - default_rate) indicates efficient price discovery.
</div>

<div class="chart-container">
{_fig_to_div(chart_volume, "trade_volume")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Trade volume by day shows the temporal
    distribution of dealer activity. High sell volume early suggests
    front-loaded liquidity stress. Buy volume indicates agents with
    surplus cash engaging in secondary market investment. The
    buy-to-sell ratio reveals whether the market is balanced or
    one-directional.
</div>

<!-- ============================================================ -->
<h2 id="network">Section 5: Network Topology</h2>
<!-- ============================================================ -->

<div class="chart-container">
{_fig_to_div(chart_degree, "degree_dist")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The out-degree distribution shows how
    many creditors each agent owes. In a ring topology the base distribution
    is uniform (each agent owes one creditor), but lending and trading create
    additional obligation links. Higher degree variance indicates more
    complex interconnection and potentially greater contagion channels.
</div>

<div class="chart-container">
{_fig_to_div(chart_systemic, "systemic_importance")}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> Systemic importance combines weighted
    degree (total obligations) and betweenness centrality (position in
    shortest paths). Top-ranked agents are potential single points of
    failure. Compare whether dealers, banks, or NBFI lenders appear
    among the most systemically important nodes.
</div>

<!-- ============================================================ -->
<h2 id="summary">Section 6: Summary Table</h2>
<!-- ============================================================ -->

<div class="chart-container">
{summary_table}
</div>
<div class="interpretation">
    <strong>Interpretation:</strong> The summary table provides a
    cross-mechanism comparison of key drill-down metrics. Cascade fraction
    measures contagion severity. Net credit impulse captures the mechanism's
    effect on overall credit supply. Funding diversity counts distinct cash
    sources. Mean out-degree and max systemic score characterise the
    obligation network's structure.
</div>

<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>scripts/run_drilldowns.py</code>
</p>

</body></html>"""

    return html


# ===================================================================
# 4. Main
# ===================================================================


def main() -> None:
    print("=" * 60)
    print("Three-Way Drill-Down Analysis")
    print("=" * 60)

    # Step 1: Select runs
    print("\n[1/3] Selecting representative runs...")
    specs = select_runs()
    if not specs:
        print("ERROR: No runs found. Check that out/three_way/ exists and contains data.")
        sys.exit(1)

    print(f"  Found {len(specs)} runs to analyse:")
    for s in specs:
        print(f"    {s['label']:30s} -> {s['events_path']}")

    # Step 2: Analyse each run
    print(f"\n[2/3] Analysing {len(specs)} runs...")
    all_results: list[AnalysisResult] = []
    for i, spec in enumerate(specs, 1):
        label = spec["label"]
        print(f"  [{i}/{len(specs)}] {label}...", end=" ", flush=True)
        try:
            result = analyse_run(spec)
            all_results.append(result)
            dc = result.get("default_counts", {})
            print(
                f"OK (events={result['n_events']}, "
                f"defaults={dc.get('total', 0)}, "
                f"nci={result.get('net_credit_impulse', 0):,.0f})"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_results:
        print("ERROR: All analyses failed.")
        sys.exit(1)

    # Step 3: Generate dashboard
    print(f"\n[3/3] Generating dashboard...")
    html = generate_dashboard(all_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html)
    print(f"\n  Dashboard saved to: {OUTPUT_FILE}")
    print(f"  Open with: open \"{OUTPUT_FILE}\"")
    print("\nDone.")


if __name__ == "__main__":
    main()
