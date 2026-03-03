"""Post-sweep analysis module for single-mechanism sweeps.

Provides drill-down, treatment-delta, dynamics, and narrative analyses
that can be run on the output of any single-mechanism sweep (dealer, bank,
or nbfi). Generates interactive Plotly HTML dashboards.

Usage from CLI:
    bilancio sweep analyze --experiment out/my_sweep --sweep-type dealer

Usage from Python:
    from bilancio.analysis.post_sweep import run_post_sweep_analysis
    results = run_post_sweep_analysis(Path("out/my_sweep"), "dealer", ["drilldowns", "deltas"])
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mechanism display metadata
# ---------------------------------------------------------------------------

MECHANISM_LABELS = {"dealer": "Dealer", "bank": "Bank", "nbfi": "NBFI"}

COLORS = {
    "dealer": "#e74c3c",
    "bank": "#2980b9",
    "nbfi": "#27ae60",
}

COLORS_LIGHT = {
    "dealer": "#f5b7b1",
    "bank": "#aed6f1",
    "nbfi": "#a9dfbf",
}

VALID_ANALYSES = ("drilldowns", "deltas", "dynamics", "narrative")
VALID_SWEEP_TYPES = ("dealer", "bank", "nbfi")

# ---------------------------------------------------------------------------
# Loss visualization metadata
# ---------------------------------------------------------------------------

LOSS_COLUMN_MAP: dict[str, dict[str, str]] = {
    "dealer": {
        "treatment": "_active",
        "baseline": "_passive",
        "delta_treatment": "delta_active",
        "delta_baseline": "delta_passive",
        "system_loss_effect": "system_loss_trading_effect",
    },
    "bank": {
        "treatment": "_lend",
        "baseline": "_idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "system_loss_effect": "system_loss_bank_lending_effect",
    },
    "nbfi": {
        "treatment": "_lend",
        "baseline": "_idle",
        "delta_treatment": "delta_lend",
        "delta_baseline": "delta_idle",
        "system_loss_effect": "system_loss_lending_effect",
    },
}

LOSS_COLORS = {
    "payable_default": "#c0392b",
    "deposit_loss": "#e67e22",
    "dealer_vbt": "#e74c3c",
    "nbfi_loan": "#27ae60",
    "bank_credit": "#2980b9",
    "cb_backstop": "#9b59b6",
    "trader_total": "#c0392b",
    "intermediary_total": "#2980b9",
}


# ===================================================================
# Path resolution
# ===================================================================


@dataclass
class SweepPaths:
    """Resolved paths for a single-mechanism sweep."""

    experiment_root: Path
    sweep_type: str
    comparison_csv: Path
    treatment_dir: Path
    baseline_dir: Path
    treatment_col: str
    baseline_col: str
    treatment_label: str
    baseline_label: str
    stats_summary: Path | None = None
    stats_sensitivity: Path | None = None


def _resolve_sweep_paths(experiment_root: Path, sweep_type: str) -> SweepPaths:
    """Resolve comparison CSV, treatment/baseline run dirs, and column names."""
    root = Path(experiment_root)
    agg = root / "aggregate"

    if sweep_type == "dealer":
        return SweepPaths(
            experiment_root=root,
            sweep_type=sweep_type,
            comparison_csv=agg / "comparison.csv",
            treatment_dir=root / "active" / "runs",
            baseline_dir=root / "passive" / "runs",
            treatment_col="active_run_id",
            baseline_col="passive_run_id",
            treatment_label="Active (Dealer)",
            baseline_label="Passive (No Dealer)",
            stats_summary=agg / "stats_summary.json",
            stats_sensitivity=agg / "stats_sensitivity.json",
        )
    elif sweep_type == "bank":
        return SweepPaths(
            experiment_root=root,
            sweep_type=sweep_type,
            comparison_csv=agg / "comparison.csv",
            treatment_dir=root / "bank_lend" / "runs",
            baseline_dir=root / "bank_idle" / "runs",
            treatment_col="lend_run_id",
            baseline_col="idle_run_id",
            treatment_label="Bank Lending",
            baseline_label="Bank Idle",
            stats_summary=agg / "stats_summary.json",
            stats_sensitivity=agg / "stats_sensitivity.json",
        )
    elif sweep_type == "nbfi":
        return SweepPaths(
            experiment_root=root,
            sweep_type=sweep_type,
            comparison_csv=agg / "comparison.csv",
            treatment_dir=root / "nbfi_lend" / "runs",
            baseline_dir=root / "nbfi_idle" / "runs",
            treatment_col="lend_run_id",
            baseline_col="idle_run_id",
            treatment_label="NBFI Lending",
            baseline_label="NBFI Idle",
            stats_summary=agg / "stats_summary.json",
            stats_sensitivity=agg / "stats_sensitivity.json",
        )
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type!r}. Must be one of {VALID_SWEEP_TYPES}")


# ===================================================================
# CSV / data helpers
# ===================================================================


def _approx_eq(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(row: dict[str, str], col: str) -> float | None:
    """Extract float from CSV row, returning None for missing/empty values."""
    val = row.get(col, "").strip()
    if not val:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _auto_detect_kappas(csv_path: Path, n: int = 3) -> list[float]:
    """Extract unique kappa values from comparison CSV, pick n representative levels."""
    rows = _read_csv(csv_path)
    unique = sorted({float(r["kappa"]) for r in rows if r.get("kappa")})
    if len(unique) <= n:
        return unique
    # Pick min, median, max
    result = [unique[0]]
    if n >= 3:
        mid_idx = len(unique) // 2
        result.append(unique[mid_idx])
    result.append(unique[-1])
    return sorted(set(result))


def _find_run_in_csv(
    rows: list[dict[str, str]],
    target_kappa: float,
    run_id_col: str,
) -> str | None:
    """Return the first run_id matching target_kappa."""
    # Try to match with preferred c=1, mu=0.5
    for pref_c, pref_mu in [(1.0, 0.5), (1.0, 0.0), (None, None)]:
        for row in rows:
            try:
                k = float(row["kappa"])
            except (KeyError, ValueError):
                continue
            if not _approx_eq(k, target_kappa):
                continue
            if pref_c is not None:
                try:
                    c = float(row["concentration"])
                    if not _approx_eq(c, pref_c):
                        continue
                except (KeyError, ValueError):
                    continue
            if pref_mu is not None:
                try:
                    mu = float(row["mu"])
                    if not _approx_eq(mu, pref_mu):
                        continue
                except (KeyError, ValueError):
                    continue
            rid = row.get(run_id_col, "").strip()
            if rid:
                return rid
    return None


def _run_dir_path(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve run directory for a run_id."""
    p = runs_dir / run_id
    return p if p.is_dir() else None


def _events_path(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve events.jsonl for a run_id."""
    p = runs_dir / run_id / "out" / "events.jsonl"
    return p if p.is_file() else None


def _metrics_csv_path(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve metrics.csv for a run_id."""
    p = runs_dir / run_id / "out" / "metrics.csv"
    return p if p.is_file() else None


def _load_events(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe_load_json(path: Path) -> dict | None:
    if not path or not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ===================================================================
# Per-run analysis (shared by drilldowns and deltas)
# ===================================================================


def _analyse_run(
    events: list[dict],
    sweep_type: str,
    is_treatment: bool,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Run analysis functions on events and return structured results.

    Args:
        events: List of event dicts from the simulation.
        sweep_type: One of "dealer", "bank", "nbfi".
        is_treatment: Whether this is the treatment arm.
        run_dir: Optional path to the run directory. When provided,
            loss metrics are computed from events and dealer_metrics.json.
    """
    from bilancio.analysis import (
        cash_inflows_by_source,
        contagion_by_day,
        credit_created_by_type,
        credit_destroyed_by_type,
        default_counts_by_type,
        net_credit_impulse,
        node_degree,
        systemic_importance,
    )

    result: dict[str, Any] = {"n_events": len(events)}

    # Defaults
    try:
        result["default_counts"] = default_counts_by_type(events)
    except Exception as exc:
        logger.debug("default_counts failed: %s", exc)
        result["default_counts"] = {"primary": 0, "secondary": 0, "total": 0}

    try:
        result["contagion_by_day"] = contagion_by_day(events)
    except Exception as exc:
        logger.debug("contagion_by_day failed: %s", exc)
        result["contagion_by_day"] = {}

    # Credit
    try:
        result["credit_created"] = {k: float(v) for k, v in credit_created_by_type(events).items()}
    except Exception as exc:
        logger.debug("credit_created failed: %s", exc)
        result["credit_created"] = {}

    try:
        result["credit_destroyed"] = {k: float(v) for k, v in credit_destroyed_by_type(events).items()}
    except Exception as exc:
        logger.debug("credit_destroyed failed: %s", exc)
        result["credit_destroyed"] = {}

    try:
        result["net_credit_impulse"] = float(net_credit_impulse(events))
    except Exception as exc:
        logger.debug("net_credit_impulse failed: %s", exc)
        result["net_credit_impulse"] = 0.0

    # Funding
    try:
        per_agent = cash_inflows_by_source(events)
        system_funding: dict[str, float] = defaultdict(float)
        for agent_flows in per_agent.values():
            for source, amount in agent_flows.items():
                system_funding[source] += float(amount)
        result["funding_mix"] = dict(system_funding)
    except Exception as exc:
        logger.debug("funding_mix failed: %s", exc)
        result["funding_mix"] = {}

    # Pricing (dealer treatment only)
    if sweep_type == "dealer" and is_treatment:
        try:
            from bilancio.analysis import bid_ask_spread_by_day, trade_prices_by_day, trade_volume_by_day

            raw = trade_prices_by_day(events)
            prices: dict[int, list[dict]] = {}
            for day, trades in raw.items():
                prices[day] = [
                    {"side": t["side"], "price_ratio": float(t["price_ratio"])}
                    for t in trades
                ]
            result["trade_prices_by_day"] = prices
            result["trade_volume_by_day"] = trade_volume_by_day(events)
            spread_raw = bid_ask_spread_by_day(events)
            result["bid_ask_spread_by_day"] = {
                d: float(v) if v is not None else None for d, v in spread_raw.items()
            }
        except Exception as exc:
            logger.debug("pricing analysis failed: %s", exc)
            result["trade_prices_by_day"] = {}
            result["trade_volume_by_day"] = {}
            result["bid_ask_spread_by_day"] = {}
    else:
        result["trade_prices_by_day"] = {}
        result["trade_volume_by_day"] = {}
        result["bid_ask_spread_by_day"] = {}

    # Network
    try:
        result["node_degrees"] = node_degree(events)
    except Exception as exc:
        logger.debug("node_degree failed: %s", exc)
        result["node_degrees"] = {}

    try:
        si = systemic_importance(events)
        result["systemic_importance"] = [
            {
                "agent_id": e["agent_id"],
                "total_obligations": float(e["total_obligations"]),
                "betweenness": float(e["betweenness"]),
                "score": float(e["score"]),
            }
            for e in si
        ]
    except Exception as exc:
        logger.debug("systemic_importance failed: %s", exc)
        result["systemic_importance"] = []

    # Loss metrics (when run_dir is provided)
    result["loss_metrics"] = {}
    result["intermediary_losses"] = {}
    result["initial_capitals"] = {}
    if run_dir is not None:
        try:
            from bilancio.analysis.report import (
                compute_intermediary_losses,
                compute_run_level_metrics,
                extract_initial_capitals,
            )

            run_metrics = compute_run_level_metrics(events)
            result["loss_metrics"] = {
                "payable_default_loss": run_metrics.get("payable_default_loss", 0),
                "deposit_loss_gross": run_metrics.get("deposit_loss_gross", 0),
                "total_loss": run_metrics.get("payable_default_loss", 0) + run_metrics.get("deposit_loss_gross", 0),
                "nbfi_loan_loss": run_metrics.get("nbfi_loan_loss", 0),
                "bank_credit_loss": run_metrics.get("bank_credit_loss", 0),
                "cb_backstop_loss": run_metrics.get("cb_backstop_loss", 0),
            }

            # Intermediary losses (needs dealer_metrics.json)
            dealer_metrics_path = run_dir / "out" / "dealer_metrics.json"
            dealer_metrics = _safe_load_json(dealer_metrics_path)
            intermediary = compute_intermediary_losses(events, dealer_metrics)
            result["intermediary_losses"] = intermediary
            result["loss_metrics"]["dealer_vbt_loss"] = intermediary.get("dealer_vbt_loss", 0)
            result["loss_metrics"]["intermediary_loss_total"] = intermediary.get("intermediary_loss_total", 0)
            result["loss_metrics"]["system_loss"] = (
                result["loss_metrics"]["total_loss"] + intermediary.get("intermediary_loss_total", 0)
            )

            # Initial capitals (from scenario.yaml)
            import yaml

            scenario_path = run_dir / "scenario.yaml"
            if scenario_path.is_file():
                with open(scenario_path) as f:
                    scenario_config = yaml.safe_load(f)
                result["initial_capitals"] = extract_initial_capitals(scenario_config)
        except Exception as exc:
            logger.debug("loss metrics computation failed: %s", exc)

    return result


# ===================================================================
# HTML generation helpers
# ===================================================================

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def _fig_to_div(fig: Any, div_id: str = "") -> str:
    """Convert Plotly figure to HTML div (no full page wrapper)."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def _dashboard_shell(title: str, nav_items: list[tuple[str, str]], body: str, sweep_type: str) -> str:
    """Wrap body content in a styled HTML shell with navigation."""
    color = COLORS.get(sweep_type, "#3498db")
    nav_links = "".join(f'<a href="#{aid}">{label}</a>' for aid, label in nav_items)

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
h1 {{ border-bottom: 3px solid {color}; padding-bottom: 10px; }}
h2 {{ margin-top: 40px; color: #34495e; }}
.chart-container {{
    background: white; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0; padding: 20px;
}}
.interpretation {{
    background: #eaf2f8; border-left: 4px solid {color};
    padding: 12px 16px; margin: 10px 0 20px;
    border-radius: 0 4px 4px 0; font-size: 14px; color: #2c3e50;
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
.nav a:hover {{ color: {color}; }}
.summary-table {{
    width: 100%; border-collapse: collapse; font-size: 14px;
}}
.summary-table th {{
    background: #2c3e50; color: white; padding: 10px 12px; text-align: left;
}}
.summary-table td {{
    padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
}}
.summary-table tr:hover {{ background: #f0f3f5; }}
</style>
</head><body>
<div class="nav">{nav_links}</div>
{body}
<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>bilancio sweep analyze</code>
</p>
</body></html>"""


# ===================================================================
# 1. Drill-downs
# ===================================================================


def _run_drilldowns(paths: SweepPaths, kappas: list[float], output_dir: Path) -> Path:
    """Per-run drill-down analysis: defaults, credit, funding, pricing, network."""
    import plotly.graph_objects as go

    logger.info("Running drill-down analysis...")
    rows = _read_csv(paths.comparison_csv)
    color = COLORS.get(paths.sweep_type, "#3498db")
    color_light = COLORS_LIGHT.get(paths.sweep_type, "#aed6f1")
    mech_label = MECHANISM_LABELS.get(paths.sweep_type, paths.sweep_type.title())

    # Collect runs
    all_results: list[dict[str, Any]] = []
    for kappa in kappas:
        # Treatment
        t_id = _find_run_in_csv(rows, kappa, paths.treatment_col)
        if t_id:
            ep = _events_path(paths.treatment_dir, t_id)
            rd = _run_dir_path(paths.treatment_dir, t_id)
            if ep:
                events = _load_events(ep)
                res = _analyse_run(events, paths.sweep_type, is_treatment=True, run_dir=rd)
                res["kappa"] = kappa
                res["arm"] = "treatment"
                res["label"] = f"{mech_label} Treatment κ={kappa}"
                all_results.append(res)

        # Baseline
        b_id = _find_run_in_csv(rows, kappa, paths.baseline_col)
        if b_id:
            ep = _events_path(paths.baseline_dir, b_id)
            rd = _run_dir_path(paths.baseline_dir, b_id)
            if ep:
                events = _load_events(ep)
                res = _analyse_run(events, paths.sweep_type, is_treatment=False, run_dir=rd)
                res["kappa"] = kappa
                res["arm"] = "baseline"
                res["label"] = f"{mech_label} Baseline κ={kappa}"
                all_results.append(res)

    if not all_results:
        raise RuntimeError("No runs found for drill-down analysis")

    treatments = [r for r in all_results if r["arm"] == "treatment"]
    baselines = [r for r in all_results if r["arm"] == "baseline"]

    # --- Chart 1: Default classification bar ---
    fig_defaults = go.Figure()
    labels, primary_vals, secondary_vals = [], [], []
    for kappa in kappas:
        for arm_list, arm_name in [(treatments, "Treatment"), (baselines, "Baseline")]:
            r = next((x for x in arm_list if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            dc = r.get("default_counts", {})
            labels.append(f"{arm_name}\nκ={kappa}")
            primary_vals.append(dc.get("primary", 0))
            secondary_vals.append(dc.get("secondary", 0))
    fig_defaults.add_trace(go.Bar(name="Primary", x=labels, y=primary_vals, marker_color="#2c3e50"))
    fig_defaults.add_trace(go.Bar(name="Cascade", x=labels, y=secondary_vals, marker_color="#e67e22"))
    fig_defaults.update_layout(
        barmode="stack", title=f"Default Classification: {mech_label}",
        xaxis_title="Arm / κ", yaxis_title="Defaults", template="plotly_white", height=420,
    )

    # --- Chart 2: Contagion timeline ---
    fig_contagion = go.Figure()
    target_kappa = kappas[0]  # lowest kappa = most stressed
    for arm_list, arm_name, c in [(treatments, "Treatment", color), (baselines, "Baseline", color_light)]:
        r = next((x for x in arm_list if _approx_eq(x["kappa"], target_kappa)), None)
        if r is None:
            continue
        cbd = r.get("contagion_by_day", {})
        if not cbd:
            continue
        days = sorted(cbd.keys())
        primary = [cbd[d].get("primary", 0) for d in days]
        secondary = [cbd[d].get("secondary", 0) for d in days]
        fig_contagion.add_trace(go.Scatter(
            x=days, y=primary, mode="lines+markers",
            name=f"{arm_name} primary", line=dict(color=c), marker=dict(size=5),
        ))
        fig_contagion.add_trace(go.Scatter(
            x=days, y=secondary, mode="lines+markers",
            name=f"{arm_name} cascade", line=dict(color=c, dash="dash"),
            marker=dict(size=5, symbol="x"),
        ))
    fig_contagion.update_layout(
        title=f"Contagion Timeline at κ={target_kappa} ({mech_label})",
        xaxis_title="Day", yaxis_title="Defaults", template="plotly_white", height=400,
    )

    # --- Chart D1: Loss Decomposition (Stacked Bar) ---
    fig_loss_decomp = None
    try:
        has_loss_data = any(r.get("loss_metrics", {}).get("total_loss", 0) > 0 for r in all_results)
        if has_loss_data:
            fig_loss_decomp = go.Figure()
            loss_labels: list[str] = []
            loss_components: dict[str, list[float]] = {
                "payable_default": [],
                "deposit_loss": [],
                "dealer_vbt": [],
                "nbfi_loan": [],
                "bank_credit_cb": [],
            }
            for kappa in kappas:
                for arm_list, arm_name in [(treatments, "Treatment"), (baselines, "Baseline")]:
                    r = next((x for x in arm_list if _approx_eq(x["kappa"], kappa)), None)
                    if r is None:
                        continue
                    lm = r.get("loss_metrics", {})
                    loss_labels.append(f"{arm_name}\nκ={kappa}")
                    loss_components["payable_default"].append(lm.get("payable_default_loss", 0))
                    loss_components["deposit_loss"].append(lm.get("deposit_loss_gross", 0))
                    loss_components["dealer_vbt"].append(lm.get("dealer_vbt_loss", 0))
                    loss_components["nbfi_loan"].append(lm.get("nbfi_loan_loss", 0))
                    loss_components["bank_credit_cb"].append(
                        lm.get("bank_credit_loss", 0) + lm.get("cb_backstop_loss", 0)
                    )

            segment_config = [
                ("payable_default", "Payable Default Loss", LOSS_COLORS["payable_default"]),
                ("deposit_loss", "Deposit Loss", LOSS_COLORS["deposit_loss"]),
                ("dealer_vbt", "Dealer/VBT Loss", LOSS_COLORS["dealer_vbt"]),
                ("nbfi_loan", "NBFI Loan Loss", LOSS_COLORS["nbfi_loan"]),
                ("bank_credit_cb", "Bank Credit + CB Loss", LOSS_COLORS["cb_backstop"]),
            ]
            for key, name, c in segment_config:
                vals = loss_components[key]
                if any(v > 0 for v in vals):
                    fig_loss_decomp.add_trace(go.Bar(
                        name=name, x=loss_labels, y=vals, marker_color=c,
                    ))
            fig_loss_decomp.update_layout(
                barmode="stack", title=f"Loss Decomposition by Asset Type ({mech_label})",
                xaxis_title="Arm / κ", yaxis_title="Loss Amount",
                template="plotly_white", height=450,
            )
    except Exception as exc:
        logger.debug("Loss decomposition chart failed: %s", exc)

    # --- Chart D2: Defaults vs System Loss (Combined View) ---
    fig_defaults_vs_loss = None
    try:
        has_loss = any(r.get("loss_metrics", {}).get("system_loss", 0) > 0 for r in all_results)
        if has_loss:
            from plotly.subplots import make_subplots

            fig_defaults_vs_loss = make_subplots(specs=[[{"secondary_y": True}]])
            d2_kappas: list[str] = []
            for kappa in kappas:
                d2_kappas.append(str(kappa))
            # Treatment bars + line
            t_defaults = []
            t_loss_pct = []
            b_defaults = []
            b_loss_pct = []
            for kappa in kappas:
                tr = next((x for x in treatments if _approx_eq(x["kappa"], kappa)), None)
                br = next((x for x in baselines if _approx_eq(x["kappa"], kappa)), None)
                t_defaults.append(tr["default_counts"].get("total", 0) if tr else 0)
                b_defaults.append(br["default_counts"].get("total", 0) if br else 0)
                # Compute loss_pct: system_loss / total_loss_denominator
                # Use total_loss + intermediary as proxy; actual pct from CSV is more accurate
                t_lm = tr.get("loss_metrics", {}) if tr else {}
                b_lm = br.get("loss_metrics", {}) if br else {}
                t_loss_pct.append(t_lm.get("system_loss", 0))
                b_loss_pct.append(b_lm.get("system_loss", 0))

            fig_defaults_vs_loss.add_trace(
                go.Bar(name="Treatment Defaults", x=d2_kappas, y=t_defaults, marker_color=color),
                secondary_y=False,
            )
            fig_defaults_vs_loss.add_trace(
                go.Bar(name="Baseline Defaults", x=d2_kappas, y=b_defaults, marker_color=color_light),
                secondary_y=False,
            )
            fig_defaults_vs_loss.add_trace(
                go.Scatter(
                    name="Treatment System Loss", x=d2_kappas, y=t_loss_pct,
                    mode="lines+markers", line=dict(color=color, width=3),
                    marker=dict(size=8),
                ),
                secondary_y=True,
            )
            fig_defaults_vs_loss.add_trace(
                go.Scatter(
                    name="Baseline System Loss", x=d2_kappas, y=b_loss_pct,
                    mode="lines+markers", line=dict(color=color_light, width=3, dash="dash"),
                    marker=dict(size=8, symbol="diamond"),
                ),
                secondary_y=True,
            )
            fig_defaults_vs_loss.update_layout(
                title=f"Defaults vs System Loss ({mech_label})",
                xaxis_title="κ", barmode="group",
                template="plotly_white", height=450,
            )
            fig_defaults_vs_loss.update_yaxes(title_text="Default Count", secondary_y=False)
            fig_defaults_vs_loss.update_yaxes(title_text="System Loss (amount)", secondary_y=True)
    except Exception as exc:
        logger.debug("Defaults vs loss chart failed: %s", exc)

    # --- Chart 3: Credit created ---
    all_types: set[str] = set()
    for r in treatments:
        all_types.update(r.get("credit_created", {}).keys())
    fig_credit = go.Figure()
    for itype in sorted(all_types):
        x_labels, vals = [], []
        for kappa in kappas:
            r = next((x for x in treatments if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            x_labels.append(f"κ={kappa}")
            vals.append(r.get("credit_created", {}).get(itype, 0))
        fig_credit.add_trace(go.Bar(name=itype, x=x_labels, y=vals))
    fig_credit.update_layout(
        barmode="group", title=f"Credit Created by Type ({mech_label})",
        xaxis_title="κ", yaxis_title="Credit Created", template="plotly_white", height=450,
    )

    # --- Chart 4: Net credit impulse ---
    fig_nci = go.Figure()
    nci_labels, nci_vals, nci_colors = [], [], []
    for kappa in kappas:
        for arm_list, arm_name, c in [(treatments, "Treatment", color), (baselines, "Baseline", color_light)]:
            r = next((x for x in arm_list if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            nci_labels.append(f"{arm_name}\nκ={kappa}")
            nci_vals.append(r.get("net_credit_impulse", 0))
            nci_colors.append(c)
    fig_nci.add_trace(go.Bar(x=nci_labels, y=nci_vals, marker_color=nci_colors))
    fig_nci.update_layout(
        title=f"Net Credit Impulse ({mech_label})",
        xaxis_title="Arm / κ", yaxis_title="Net Credit Impulse", template="plotly_white", height=400,
    )

    # --- Chart 5: Funding mix ---
    SOURCE_ORDER = [
        "settlement_received", "ticket_sale", "loan_received",
        "cb_loan", "nbfi_loan", "deposit_interest", "deposit",
    ]
    SOURCE_COLORS = {
        "settlement_received": "#3498db", "ticket_sale": "#e74c3c",
        "loan_received": "#2ecc71", "cb_loan": "#9b59b6",
        "nbfi_loan": "#1abc9c", "deposit_interest": "#f39c12", "deposit": "#95a5a6",
    }
    present_sources: set[str] = set()
    for r in treatments:
        present_sources.update(r.get("funding_mix", {}).keys())
    sources_to_plot = [s for s in SOURCE_ORDER if s in present_sources]
    for s in sorted(present_sources):
        if s not in sources_to_plot:
            sources_to_plot.append(s)

    fig_funding = go.Figure()
    fund_labels = [f"κ={k}" for k in kappas if any(_approx_eq(x["kappa"], k) for x in treatments)]
    for source in sources_to_plot:
        vals = []
        for kappa in kappas:
            r = next((x for x in treatments if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            vals.append(r.get("funding_mix", {}).get(source, 0))
        fig_funding.add_trace(go.Bar(
            name=source, x=fund_labels, y=vals,
            marker_color=SOURCE_COLORS.get(source, "#bdc3c7"),
        ))
    fig_funding.update_layout(
        barmode="stack", title=f"Funding Sources ({mech_label} Treatment)",
        xaxis_title="κ", yaxis_title="Total Inflows", template="plotly_white", height=480,
    )

    # --- Chart 6 & 7: Pricing (dealer only) ---
    fig_pricing = go.Figure()
    fig_volume = go.Figure()
    if paths.sweep_type == "dealer":
        dash_styles = {k: s for k, s in zip(kappas, ["solid", "dash", "dot"])}
        for r in treatments:
            kappa = r["kappa"]
            prices = r.get("trade_prices_by_day", {})
            if not prices:
                continue
            days = sorted(int(d) for d in prices.keys())
            buy_avgs, sell_avgs = [], []
            for d in days:
                day_trades = prices[d]
                buys = [t["price_ratio"] for t in day_trades if t["side"] == "buy"]
                sells = [t["price_ratio"] for t in day_trades if t["side"] == "sell"]
                buy_avgs.append(sum(buys) / len(buys) if buys else None)
                sell_avgs.append(sum(sells) / len(sells) if sells else None)
            dash = dash_styles.get(kappa, "solid")
            fig_pricing.add_trace(go.Scatter(
                x=days, y=buy_avgs, mode="lines+markers",
                name=f"Buy avg κ={kappa}", line=dict(color=color, dash=dash), marker=dict(size=4),
            ))
            fig_pricing.add_trace(go.Scatter(
                x=days, y=sell_avgs, mode="lines+markers",
                name=f"Sell avg κ={kappa}", line=dict(color="#c0392b", dash=dash),
                marker=dict(size=4, symbol="triangle-up"),
            ))
        fig_pricing.update_layout(
            title="Price Discovery: Buy/Sell Price Ratios",
            xaxis_title="Day", yaxis_title="Price / Face Value",
            template="plotly_white", height=420,
        )

        # Volume for lowest kappa
        r_target = next((r for r in treatments if _approx_eq(r["kappa"], kappas[0])), None)
        if r_target is None and treatments:
            r_target = treatments[0]
        if r_target:
            vol = r_target.get("trade_volume_by_day", {})
            if vol:
                days = sorted(vol.keys())
                buys = [vol[d].get("buys", 0) for d in days]
                sells = [vol[d].get("sells", 0) for d in days]
                fig_volume.add_trace(go.Bar(name="Buys", x=days, y=buys, marker_color="#27ae60"))
                fig_volume.add_trace(go.Bar(name="Sells", x=days, y=sells, marker_color="#e74c3c"))
        fig_volume.update_layout(
            barmode="group", title=f"Trade Volume by Day (κ={kappas[0]})",
            xaxis_title="Day", yaxis_title="Trades", template="plotly_white", height=400,
        )

    # --- Chart 8: Degree distribution ---
    fig_degree = go.Figure()
    for kappa in kappas:
        for arm_list, arm_name, c in [(treatments, "Treatment", color), (baselines, "Baseline", color_light)]:
            r = next((x for x in arm_list if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            degrees = r.get("node_degrees", {})
            if not degrees:
                continue
            out_degs = [v.get("out_degree", 0) for v in degrees.values()]
            fig_degree.add_trace(go.Box(
                y=out_degs, name=f"{arm_name} κ={kappa}",
                marker_color=c, boxmean=True,
            ))
    fig_degree.update_layout(
        title=f"Out-Degree Distribution ({mech_label})",
        yaxis_title="Out-Degree", template="plotly_white", height=450,
    )

    # --- Chart 9: Systemic importance ---
    fig_systemic = go.Figure()
    for arm_list, arm_name, c in [(treatments, "Treatment", color), (baselines, "Baseline", color_light)]:
        r = next((x for x in arm_list if _approx_eq(x["kappa"], kappas[0])), None)
        if r is None:
            continue
        si = r.get("systemic_importance", [])[:5]
        if not si:
            continue
        fig_systemic.add_trace(go.Bar(
            name=arm_name, x=[e["agent_id"][:12] for e in si],
            y=[e["score"] for e in si], marker_color=c,
        ))
    fig_systemic.update_layout(
        title=f"Top-5 Systemically Important Agents (κ={kappas[0]})",
        xaxis_title="Agent", yaxis_title="Score",
        barmode="group", template="plotly_white", height=400,
    )

    # --- Summary table ---
    table_rows = []
    for kappa in kappas:
        for arm_list, arm_name in [(treatments, "Treatment"), (baselines, "Baseline")]:
            r = next((x for x in arm_list if _approx_eq(x["kappa"], kappa)), None)
            if r is None:
                continue
            dc = r.get("default_counts", {})
            total = dc.get("total", 0)
            cascade_frac = f"{dc.get('secondary', 0) / total:.2%}" if total > 0 else "n/a"
            nci = r.get("net_credit_impulse", 0)
            fm = r.get("funding_mix", {})
            funding_div = sum(1 for v in fm.values() if v > 0)
            degrees = r.get("node_degrees", {})
            out_degs = [v.get("out_degree", 0) for v in degrees.values()]
            mean_out = f"{sum(out_degs) / len(out_degs):.1f}" if out_degs else "n/a"
            si = r.get("systemic_importance", [])
            max_si = f"{si[0]['score']:.3f}" if si else "n/a"
            lm = r.get("loss_metrics", {})
            total_loss_str = f"{lm['total_loss']:,.0f}" if lm.get("total_loss") else "n/a"
            interm_loss_str = f"{lm['intermediary_loss_total']:,.0f}" if lm.get("intermediary_loss_total") else "n/a"
            sys_loss_str = f"{lm['system_loss']:,.0f}" if lm.get("system_loss") else "n/a"
            table_rows.append(
                f"<tr><td>{arm_name}</td><td>{kappa}</td>"
                f"<td>{dc.get('primary', 0)}</td><td>{dc.get('secondary', 0)}</td>"
                f"<td>{cascade_frac}</td><td>{nci:,.0f}</td>"
                f"<td>{total_loss_str}</td><td>{interm_loss_str}</td><td>{sys_loss_str}</td>"
                f"<td>{funding_div}</td><td>{mean_out}</td><td>{max_si}</td></tr>"
            )

    summary_table = f"""<table class="summary-table">
    <thead><tr>
        <th>Arm</th><th>κ</th><th>Primary</th><th>Cascade</th>
        <th>Cascade %</th><th>NCI</th><th>Total Loss</th><th>Intermed. Loss</th><th>System Loss</th>
        <th>Fund. Div.</th><th>Mean Out-Deg</th><th>Max SI</th>
    </tr></thead>
    <tbody>{"".join(table_rows)}</tbody></table>"""

    # Assemble
    nav = [
        ("defaults", "Defaults"), ("losses", "Losses"), ("credit", "Credit"), ("funding", "Funding"),
        ("network", "Network"), ("summary", "Summary"),
    ]
    if paths.sweep_type == "dealer":
        nav.insert(4, ("pricing", "Pricing"))

    n_treat = len(treatments)
    n_base = len(baselines)
    body_parts = [
        f"<h1>{mech_label} Drill-Down Analysis</h1>",
        f"<p>Analysed <strong>{n_treat} treatment</strong> and <strong>{n_base} baseline</strong> "
        f"runs at κ ∈ {{{', '.join(str(k) for k in kappas)}}}.</p>",
        f'<h2 id="defaults">Default Classification</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_defaults, "defaults_bar")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> Primary vs cascade defaults '
        'across treatment and baseline arms. A higher cascade fraction suggests greater systemic fragility.</div>',
        f'<div class="chart-container">{_fig_to_div(fig_contagion, "contagion")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> Contagion timeline at the most '
        'stressed κ level. Compare how treatment reduces or delays cascade propagation.</div>',
    ]

    # Loss section (between defaults and credit)
    body_parts.append(f'<h2 id="losses">Loss Analysis</h2>')
    if fig_loss_decomp is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_decomp, "loss_decomp")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> Stacked bars show the '
            'composition of losses by asset type. Payable defaults are the primary loss channel; '
            'intermediary losses (dealer, bank, NBFI) represent the cost of the mechanism.</div>',
        ])
    if fig_defaults_vs_loss is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_defaults_vs_loss, "defaults_vs_loss")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> Bars show default counts; '
            'lines show total system loss. Where losses grow faster than defaults, there is '
            'nonlinear amplification (cascade effects, deposit erosion).</div>',
        ])
    if fig_loss_decomp is None and fig_defaults_vs_loss is None:
        body_parts.append(
            '<div class="interpretation">No loss data available for these runs. '
            'Loss metrics require run directories with events.jsonl.</div>'
        )

    body_parts.extend([
        f'<h2 id="credit">Credit Dynamics</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_credit, "credit_created")}</div>',
        f'<div class="chart-container">{_fig_to_div(fig_nci, "nci")}</div>',
        f'<h2 id="funding">Funding Mix</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_funding, "funding")}</div>',
    ])

    if paths.sweep_type == "dealer":
        body_parts.extend([
            f'<h2 id="pricing">Price Discovery</h2>',
            f'<div class="chart-container">{_fig_to_div(fig_pricing, "pricing")}</div>',
            f'<div class="chart-container">{_fig_to_div(fig_volume, "volume")}</div>',
        ])

    body_parts.extend([
        f'<h2 id="network">Network Topology</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_degree, "degree")}</div>',
        f'<div class="chart-container">{_fig_to_div(fig_systemic, "systemic")}</div>',
        f'<h2 id="summary">Summary</h2>',
        f'<div class="chart-container">{summary_table}</div>',
    ])

    html = _dashboard_shell(
        f"{mech_label} Drill-Down Analysis", nav, "\n".join(body_parts), paths.sweep_type,
    )

    output_path = output_dir / "drilldown_dashboard.html"
    output_path.write_text(html)
    logger.info("Drill-down dashboard: %s", output_path)
    return output_path


# ===================================================================
# 2. Treatment deltas
# ===================================================================


def _run_treatment_deltas(paths: SweepPaths, kappas: list[float], output_dir: Path) -> Path:
    """Baseline vs treatment delta comparison."""
    import plotly.graph_objects as go

    logger.info("Running treatment delta analysis...")
    rows = _read_csv(paths.comparison_csv)
    color = COLORS.get(paths.sweep_type, "#3498db")
    mech_label = MECHANISM_LABELS.get(paths.sweep_type, paths.sweep_type.title())

    # Collect paired results
    deltas: list[dict[str, Any]] = []
    for kappa in kappas:
        t_id = _find_run_in_csv(rows, kappa, paths.treatment_col)
        b_id = _find_run_in_csv(rows, kappa, paths.baseline_col)
        if not t_id or not b_id:
            continue
        t_ep = _events_path(paths.treatment_dir, t_id)
        b_ep = _events_path(paths.baseline_dir, b_id)
        if not t_ep or not b_ep:
            continue

        t_events = _load_events(t_ep)
        b_events = _load_events(b_ep)
        t_res = _analyse_run(t_events, paths.sweep_type, is_treatment=True)
        b_res = _analyse_run(b_events, paths.sweep_type, is_treatment=False)

        # Compute deltas
        delta: dict[str, Any] = {"kappa": kappa}

        # Default deltas
        t_dc = t_res.get("default_counts", {})
        b_dc = b_res.get("default_counts", {})
        delta["primary_delta"] = t_dc.get("primary", 0) - b_dc.get("primary", 0)
        delta["secondary_delta"] = t_dc.get("secondary", 0) - b_dc.get("secondary", 0)
        delta["total_delta"] = t_dc.get("total", 0) - b_dc.get("total", 0)

        # NCI delta
        delta["nci_delta"] = t_res.get("net_credit_impulse", 0) - b_res.get("net_credit_impulse", 0)

        # Funding delta
        all_sources = set(t_res.get("funding_mix", {}).keys()) | set(b_res.get("funding_mix", {}).keys())
        funding_delta = {}
        for src in all_sources:
            funding_delta[src] = t_res.get("funding_mix", {}).get(src, 0) - b_res.get("funding_mix", {}).get(src, 0)
        delta["funding_delta"] = funding_delta

        # Network deltas
        t_degs = t_res.get("node_degrees", {})
        b_degs = b_res.get("node_degrees", {})
        t_out = [v.get("out_degree", 0) for v in t_degs.values()] if t_degs else [0]
        b_out = [v.get("out_degree", 0) for v in b_degs.values()] if b_degs else [0]
        delta["mean_degree_delta"] = (
            (sum(t_out) / len(t_out) if t_out else 0)
            - (sum(b_out) / len(b_out) if b_out else 0)
        )

        deltas.append(delta)

    if not deltas:
        raise RuntimeError("No paired runs found for delta analysis")

    # --- Chart 1: Default reduction ---
    fig_default_delta = go.Figure()
    x_labels = [f"κ={d['kappa']}" for d in deltas]
    fig_default_delta.add_trace(go.Bar(
        name="Primary Δ", x=x_labels,
        y=[d["primary_delta"] for d in deltas], marker_color="#2c3e50",
    ))
    fig_default_delta.add_trace(go.Bar(
        name="Cascade Δ", x=x_labels,
        y=[d["secondary_delta"] for d in deltas], marker_color="#e67e22",
    ))
    fig_default_delta.update_layout(
        barmode="group", title=f"Default Reduction (Treatment - Baseline): {mech_label}",
        xaxis_title="κ", yaxis_title="Δ Defaults (negative = reduction)",
        template="plotly_white", height=420,
    )

    # --- Chart 2: NCI delta ---
    fig_nci_delta = go.Figure()
    nci_colors = [color if d["nci_delta"] >= 0 else "#95a5a6" for d in deltas]
    fig_nci_delta.add_trace(go.Bar(
        x=x_labels, y=[d["nci_delta"] for d in deltas], marker_color=nci_colors,
    ))
    fig_nci_delta.update_layout(
        title=f"Net Credit Impulse Delta: {mech_label}",
        xaxis_title="κ", yaxis_title="Δ NCI",
        template="plotly_white", height=400,
    )

    # --- Chart 3: Funding shift ---
    all_fund_sources: set[str] = set()
    for d in deltas:
        all_fund_sources.update(d.get("funding_delta", {}).keys())
    fig_funding_delta = go.Figure()
    for src in sorted(all_fund_sources):
        vals = [d.get("funding_delta", {}).get(src, 0) for d in deltas]
        if any(v != 0 for v in vals):
            fig_funding_delta.add_trace(go.Bar(name=src, x=x_labels, y=vals))
    fig_funding_delta.update_layout(
        barmode="relative", title=f"Funding Source Shift (Treatment - Baseline): {mech_label}",
        xaxis_title="κ", yaxis_title="Δ Inflows",
        template="plotly_white", height=450,
    )

    # --- Chart 4: Degree delta ---
    fig_degree_delta = go.Figure()
    fig_degree_delta.add_trace(go.Bar(
        x=x_labels, y=[d["mean_degree_delta"] for d in deltas], marker_color=color,
    ))
    fig_degree_delta.update_layout(
        title=f"Mean Out-Degree Delta: {mech_label}",
        xaxis_title="κ", yaxis_title="Δ Mean Out-Degree",
        template="plotly_white", height=400,
    )

    # --- Loss charts T1-T4 from comparison CSV ---
    col_map = LOSS_COLUMN_MAP.get(paths.sweep_type, {})
    t_suffix = col_map.get("treatment", "_active")
    b_suffix = col_map.get("baseline", "_passive")
    loss_effect_col = col_map.get("system_loss_effect", "")

    fig_loss_comparison = None
    fig_loss_waterfall = None
    fig_loss_vs_delta = None
    fig_loss_capital = None

    try:
        # Build per-kappa loss data from CSV rows
        csv_loss_data: list[dict[str, Any]] = []
        for kappa in kappas:
            matched_row = None
            for row in rows:
                try:
                    if _approx_eq(float(row["kappa"]), kappa):
                        matched_row = row
                        break
                except (ValueError, KeyError):
                    continue
            if matched_row is None:
                continue

            entry: dict[str, Any] = {"kappa": kappa, "row": matched_row}
            entry["system_loss_pct_t"] = _safe_float(matched_row, f"system_loss_pct{t_suffix}")
            entry["system_loss_pct_b"] = _safe_float(matched_row, f"system_loss_pct{b_suffix}")
            entry["total_loss_pct_t"] = _safe_float(matched_row, f"total_loss_pct{t_suffix}")
            entry["total_loss_pct_b"] = _safe_float(matched_row, f"total_loss_pct{b_suffix}")
            entry["intermediary_loss_pct_t"] = _safe_float(matched_row, f"intermediary_loss_pct{t_suffix}")
            entry["intermediary_loss_pct_b"] = _safe_float(matched_row, f"intermediary_loss_pct{b_suffix}")
            entry["loss_capital_ratio_t"] = _safe_float(matched_row, f"loss_capital_ratio{t_suffix}")
            entry["loss_capital_ratio_b"] = _safe_float(matched_row, f"loss_capital_ratio{b_suffix}")
            entry["system_loss_effect"] = _safe_float(matched_row, loss_effect_col)
            entry["delta_t"] = _safe_float(matched_row, col_map.get("delta_treatment", ""))
            entry["delta_b"] = _safe_float(matched_row, col_map.get("delta_baseline", ""))
            csv_loss_data.append(entry)

        has_csv_loss = any(
            e.get("system_loss_pct_t") is not None or e.get("system_loss_pct_b") is not None
            for e in csv_loss_data
        )

        if has_csv_loss and csv_loss_data:
            loss_x = [f"κ={e['kappa']}" for e in csv_loss_data]

            # Chart T1: System Loss Comparison (Grouped Bar)
            fig_loss_comparison = go.Figure()
            fig_loss_comparison.add_trace(go.Bar(
                name=paths.baseline_label,
                x=loss_x, y=[e.get("system_loss_pct_b") or 0 for e in csv_loss_data],
                marker_color=COLORS_LIGHT.get(paths.sweep_type, "#aed6f1"),
            ))
            fig_loss_comparison.add_trace(go.Bar(
                name=paths.treatment_label,
                x=loss_x, y=[e.get("system_loss_pct_t") or 0 for e in csv_loss_data],
                marker_color=color,
            ))
            fig_loss_comparison.update_layout(
                barmode="group", title=f"System Loss Comparison: {mech_label}",
                xaxis_title="κ", yaxis_title="System Loss %",
                template="plotly_white", height=420,
            )

            # Chart T2: Loss Attribution Waterfall (Stacked Grouped Bar)
            fig_loss_waterfall = go.Figure()
            # Baseline trader loss
            fig_loss_waterfall.add_trace(go.Bar(
                name=f"{paths.baseline_label} Trader Loss",
                x=loss_x, y=[e.get("total_loss_pct_b") or 0 for e in csv_loss_data],
                marker_color=LOSS_COLORS["trader_total"], opacity=0.5,
            ))
            # Baseline intermediary loss (stacked on top of trader)
            fig_loss_waterfall.add_trace(go.Bar(
                name=f"{paths.baseline_label} Intermediary Loss",
                x=loss_x, y=[e.get("intermediary_loss_pct_b") or 0 for e in csv_loss_data],
                marker_color=LOSS_COLORS["intermediary_total"], opacity=0.5,
            ))
            # Treatment trader loss
            fig_loss_waterfall.add_trace(go.Bar(
                name=f"{paths.treatment_label} Trader Loss",
                x=loss_x, y=[e.get("total_loss_pct_t") or 0 for e in csv_loss_data],
                marker_color=LOSS_COLORS["trader_total"],
            ))
            # Treatment intermediary loss
            fig_loss_waterfall.add_trace(go.Bar(
                name=f"{paths.treatment_label} Intermediary Loss",
                x=loss_x, y=[e.get("intermediary_loss_pct_t") or 0 for e in csv_loss_data],
                marker_color=LOSS_COLORS["intermediary_total"],
            ))
            fig_loss_waterfall.update_layout(
                barmode="stack", title=f"Loss Attribution: Trader vs Intermediary ({mech_label})",
                xaxis_title="κ", yaxis_title="Loss % of Total Debt",
                template="plotly_white", height=450,
            )

            # Chart T3: Delta-Based vs Loss-Based Effect (Bar + Line)
            from plotly.subplots import make_subplots

            fig_loss_vs_delta = make_subplots(specs=[[{"secondary_y": True}]])
            loss_effects = [e.get("system_loss_effect") or 0 for e in csv_loss_data]
            delta_effects = [
                (e.get("delta_b") or 0) - (e.get("delta_t") or 0) for e in csv_loss_data
            ]
            fig_loss_vs_delta.add_trace(
                go.Bar(
                    name="System Loss Effect", x=loss_x, y=loss_effects,
                    marker_color=LOSS_COLORS["intermediary_total"],
                ),
                secondary_y=False,
            )
            fig_loss_vs_delta.add_trace(
                go.Scatter(
                    name="δ-Based Effect", x=loss_x, y=delta_effects,
                    mode="lines+markers", line=dict(color=color, width=3),
                    marker=dict(size=8),
                ),
                secondary_y=True,
            )
            fig_loss_vs_delta.update_layout(
                title=f"Delta-Based vs Loss-Based Treatment Effect ({mech_label})",
                xaxis_title="κ", template="plotly_white", height=450,
            )
            fig_loss_vs_delta.update_yaxes(title_text="System Loss Effect", secondary_y=False)
            fig_loss_vs_delta.update_yaxes(title_text="δ Effect (baseline - treatment)", secondary_y=True)

            # Chart T4: Loss/Capital Ratio (conditional on data)
            has_capital = any(
                e.get("loss_capital_ratio_t") is not None or e.get("loss_capital_ratio_b") is not None
                for e in csv_loss_data
            )
            if has_capital:
                fig_loss_capital = go.Figure()
                fig_loss_capital.add_trace(go.Bar(
                    name=paths.baseline_label,
                    x=loss_x, y=[e.get("loss_capital_ratio_b") or 0 for e in csv_loss_data],
                    marker_color=COLORS_LIGHT.get(paths.sweep_type, "#aed6f1"),
                ))
                fig_loss_capital.add_trace(go.Bar(
                    name=paths.treatment_label,
                    x=loss_x, y=[e.get("loss_capital_ratio_t") or 0 for e in csv_loss_data],
                    marker_color=color,
                ))
                fig_loss_capital.update_layout(
                    barmode="group", title=f"Loss / Capital Ratio: {mech_label}",
                    xaxis_title="κ", yaxis_title="Loss / Intermediary Capital",
                    template="plotly_white", height=420,
                )
    except Exception as exc:
        logger.debug("Treatment delta loss charts failed: %s", exc)

    # --- Summary table ---
    table_rows = []
    for d in deltas:
        table_rows.append(
            f"<tr><td>{d['kappa']}</td>"
            f"<td>{d['total_delta']:+d}</td>"
            f"<td>{d['primary_delta']:+d}</td>"
            f"<td>{d['secondary_delta']:+d}</td>"
            f"<td>{d['nci_delta']:+,.0f}</td>"
            f"<td>{d['mean_degree_delta']:+.2f}</td></tr>"
        )
    summary_table = f"""<table class="summary-table">
    <thead><tr>
        <th>κ</th><th>Total Δ</th><th>Primary Δ</th><th>Cascade Δ</th>
        <th>NCI Δ</th><th>Degree Δ</th>
    </tr></thead>
    <tbody>{"".join(table_rows)}</tbody></table>"""

    # Assemble
    nav = [
        ("defaults", "Default Deltas"), ("losses", "Loss Deltas"), ("credit", "Credit Deltas"),
        ("funding", "Funding Shift"), ("network", "Network Deltas"), ("summary", "Summary"),
    ]
    body_parts = [
        f"<h1>{mech_label} Treatment Deltas</h1>",
        f"<p>Treatment minus baseline comparison across κ ∈ {{{', '.join(str(k) for k in kappas)}}}. "
        f"Negative default deltas indicate the treatment reduced defaults.</p>",
        f'<h2 id="defaults">Default Reduction</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_default_delta, "default_delta")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> Negative bars mean the treatment '
        'reduced defaults compared to baseline. Cascade reduction is particularly important as it '
        'indicates systemic risk mitigation.</div>',
    ]

    # Loss section
    body_parts.append(f'<h2 id="losses">Loss Comparison</h2>')
    if fig_loss_comparison is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_comparison, "loss_comparison")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> System loss = trader loss + '
            'intermediary loss. Lower bars mean the mechanism reduced overall economic damage.</div>',
        ])
    if fig_loss_waterfall is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_waterfall, "loss_waterfall")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> If treatment reduces trader loss '
            'but increases intermediary loss, that indicates loss-shifting rather than genuine risk reduction. '
            'Both components should shrink for a true improvement.</div>',
        ])
    if fig_loss_vs_delta is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_vs_delta, "loss_vs_delta")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> The gap between the δ-based effect '
            '(line) and the system loss effect (bars) represents the intermediary cost of the mechanism. '
            'If the line exceeds the bars, intermediaries absorbed some of the default reduction.</div>',
        ])
    if fig_loss_capital is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_capital, "loss_capital")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> Loss/capital ratio shows how much '
            'intermediary capital was consumed. Values above 1.0 mean intermediaries lost more than their '
            'initial capital.</div>',
        ])
    if all(f is None for f in [fig_loss_comparison, fig_loss_waterfall, fig_loss_vs_delta]):
        body_parts.append(
            '<div class="interpretation">No loss data available in comparison CSV for these runs.</div>'
        )

    body_parts.extend([
        f'<h2 id="credit">Credit Impulse Delta</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_nci_delta, "nci_delta")}</div>',
        f'<h2 id="funding">Funding Source Shift</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_funding_delta, "funding_delta")}</div>',
        f'<h2 id="network">Network Structure Delta</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_degree_delta, "degree_delta")}</div>',
        f'<h2 id="summary">Summary</h2>',
        f'<div class="chart-container">{summary_table}</div>',
    ])

    html = _dashboard_shell(
        f"{mech_label} Treatment Deltas", nav, "\n".join(body_parts), paths.sweep_type,
    )

    output_path = output_dir / "treatment_deltas_dashboard.html"
    output_path.write_text(html)
    logger.info("Treatment deltas dashboard: %s", output_path)
    return output_path


# ===================================================================
# 3. Dynamics
# ===================================================================


def _extract_defaults_by_day(events: list[dict]) -> dict[int, dict[str, int]]:
    """Extract default counts by day from events."""
    by_day: dict[int, dict[str, int]] = defaultdict(lambda: {"primary": 0, "secondary": 0})
    for e in events:
        if e.get("type") == "default":
            day = e.get("day", 0)
            kind = e.get("default_type", "primary")
            if kind in ("primary", "secondary", "cascade"):
                key = "secondary" if kind in ("secondary", "cascade") else "primary"
                by_day[day][key] += 1
    return dict(by_day)


def _extract_agent_outcomes(events: list[dict]) -> dict[str, dict[str, Any]]:
    """Extract per-agent outcome data from events."""
    agents: dict[str, dict[str, Any]] = defaultdict(lambda: {"cash_final": 0, "defaulted": False, "n_trades": 0})
    for e in events:
        etype = e.get("type", "")
        agent_id = e.get("agent_id", e.get("debtor_id", ""))
        if not agent_id:
            continue
        if etype == "default":
            agents[agent_id]["defaulted"] = True
        elif etype in ("trade", "buy", "sell"):
            agents[agent_id]["n_trades"] += 1
        elif etype == "cash_position":
            agents[agent_id]["cash_final"] = float(e.get("cash", 0))
    return dict(agents)


def _run_dynamics(paths: SweepPaths, kappas: list[float], output_dir: Path) -> Path:
    """Time-series and agent heterogeneity dynamics analysis."""
    import plotly.graph_objects as go

    logger.info("Running dynamics analysis...")
    rows = _read_csv(paths.comparison_csv)
    color = COLORS.get(paths.sweep_type, "#3498db")
    color_light = COLORS_LIGHT.get(paths.sweep_type, "#aed6f1")
    mech_label = MECHANISM_LABELS.get(paths.sweep_type, paths.sweep_type.title())

    # Load metrics CSVs and events for each kappa
    dynamics_data: list[dict[str, Any]] = []
    for kappa in kappas:
        entry: dict[str, Any] = {"kappa": kappa}

        for arm, col, runs_dir in [
            ("treatment", paths.treatment_col, paths.treatment_dir),
            ("baseline", paths.baseline_col, paths.baseline_dir),
        ]:
            run_id = _find_run_in_csv(rows, kappa, col)
            if not run_id:
                continue

            # Load metrics.csv
            metrics_path = _metrics_csv_path(runs_dir, run_id)
            if metrics_path:
                try:
                    metrics_rows = _read_csv(metrics_path)
                    entry[f"{arm}_metrics"] = metrics_rows
                except Exception:
                    entry[f"{arm}_metrics"] = []

            # Load events
            ep = _events_path(runs_dir, run_id)
            if ep:
                try:
                    events = _load_events(ep)
                    entry[f"{arm}_defaults_by_day"] = _extract_defaults_by_day(events)
                    entry[f"{arm}_agents"] = _extract_agent_outcomes(events)
                except Exception:
                    pass

        dynamics_data.append(entry)

    if not dynamics_data:
        raise RuntimeError("No data found for dynamics analysis")

    # --- Chart 1: Default timeline (cumulative) ---
    fig_cum_defaults = go.Figure()
    dash_styles = {k: s for k, s in zip(kappas, ["solid", "dash", "dot"])}
    for entry in dynamics_data:
        kappa = entry["kappa"]
        for arm, c in [("treatment", color), ("baseline", color_light)]:
            dbd = entry.get(f"{arm}_defaults_by_day", {})
            if not dbd:
                continue
            days = sorted(dbd.keys())
            cumulative = []
            total = 0
            for d in days:
                total += dbd[d]["primary"] + dbd[d]["secondary"]
                cumulative.append(total)
            fig_cum_defaults.add_trace(go.Scatter(
                x=days, y=cumulative, mode="lines+markers",
                name=f"{arm.title()} κ={kappa}",
                line=dict(color=c, dash=dash_styles.get(kappa, "solid")),
                marker=dict(size=4),
            ))
    fig_cum_defaults.update_layout(
        title=f"Cumulative Defaults Over Time ({mech_label})",
        xaxis_title="Day", yaxis_title="Cumulative Defaults",
        template="plotly_white", height=420,
    )

    # --- Chart 2: δ_t trajectory from metrics.csv ---
    fig_delta_t = go.Figure()
    for entry in dynamics_data:
        kappa = entry["kappa"]
        for arm, c in [("treatment", color), ("baseline", color_light)]:
            metrics = entry.get(f"{arm}_metrics", [])
            if not metrics:
                continue
            days, vals = [], []
            for m in metrics:
                try:
                    days.append(int(m.get("day", 0)))
                    vals.append(float(m.get("delta_t", 0)))
                except (ValueError, TypeError):
                    continue
            if days:
                fig_delta_t.add_trace(go.Scatter(
                    x=days, y=vals, mode="lines",
                    name=f"{arm.title()} κ={kappa}",
                    line=dict(color=c, dash=dash_styles.get(kappa, "solid")),
                ))
    fig_delta_t.update_layout(
        title=f"δ_t (Default Rate) Trajectory ({mech_label})",
        xaxis_title="Day", yaxis_title="δ_t",
        template="plotly_white", height=400,
    )

    # --- Chart 3: φ_t trajectory ---
    fig_phi_t = go.Figure()
    for entry in dynamics_data:
        kappa = entry["kappa"]
        for arm, c in [("treatment", color), ("baseline", color_light)]:
            metrics = entry.get(f"{arm}_metrics", [])
            if not metrics:
                continue
            days, vals = [], []
            for m in metrics:
                try:
                    days.append(int(m.get("day", 0)))
                    vals.append(float(m.get("phi_t", 0)))
                except (ValueError, TypeError):
                    continue
            if days:
                fig_phi_t.add_trace(go.Scatter(
                    x=days, y=vals, mode="lines",
                    name=f"{arm.title()} κ={kappa}",
                    line=dict(color=c, dash=dash_styles.get(kappa, "solid")),
                ))
    fig_phi_t.update_layout(
        title=f"φ_t (Clearing Rate) Trajectory ({mech_label})",
        xaxis_title="Day", yaxis_title="φ_t",
        template="plotly_white", height=400,
    )

    # --- Chart 4: Agent outcomes scatter ---
    fig_agents = go.Figure()
    # Use lowest kappa treatment
    entry_0 = dynamics_data[0] if dynamics_data else {}
    for arm, c, sym in [("treatment", color, "circle"), ("baseline", color_light, "x")]:
        agents = entry_0.get(f"{arm}_agents", {})
        if not agents:
            continue
        survived = {k: v for k, v in agents.items() if not v["defaulted"]}
        defaulted = {k: v for k, v in agents.items() if v["defaulted"]}
        if survived:
            fig_agents.add_trace(go.Scatter(
                x=[v["n_trades"] for v in survived.values()],
                y=[v.get("cash_final", 0) for v in survived.values()],
                mode="markers", name=f"{arm.title()} survived",
                marker=dict(color=c, size=6, symbol=sym),
            ))
        if defaulted:
            fig_agents.add_trace(go.Scatter(
                x=[v["n_trades"] for v in defaulted.values()],
                y=[v.get("cash_final", 0) for v in defaulted.values()],
                mode="markers", name=f"{arm.title()} defaulted",
                marker=dict(color="red", size=8, symbol="x"),
            ))
    fig_agents.update_layout(
        title=f"Agent Outcomes at κ={kappas[0]} ({mech_label})",
        xaxis_title="Number of Trades", yaxis_title="Final Cash",
        template="plotly_white", height=450,
    )

    # --- Loss dynamics charts Y1-Y3 (from comparison CSV, all kappas) ---
    col_map = LOSS_COLUMN_MAP.get(paths.sweep_type, {})
    t_suffix = col_map.get("treatment", "_active")
    b_suffix = col_map.get("baseline", "_passive")

    fig_loss_kappa = None
    fig_loss_composition = None
    fig_loss_scatter = None

    try:
        all_csv_rows = _read_csv(paths.comparison_csv)
        # Build sorted kappa-loss data from ALL CSV rows
        kappa_loss_entries: list[dict[str, Any]] = []
        for row in all_csv_rows:
            try:
                k = float(row["kappa"])
            except (ValueError, KeyError):
                continue
            entry: dict[str, Any] = {"kappa": k}
            entry["system_loss_pct_t"] = _safe_float(row, f"system_loss_pct{t_suffix}")
            entry["system_loss_pct_b"] = _safe_float(row, f"system_loss_pct{b_suffix}")
            entry["total_loss_pct_t"] = _safe_float(row, f"total_loss_pct{t_suffix}")
            entry["total_loss_pct_b"] = _safe_float(row, f"total_loss_pct{b_suffix}")
            entry["intermediary_loss_pct_t"] = _safe_float(row, f"intermediary_loss_pct{t_suffix}")
            entry["intermediary_loss_pct_b"] = _safe_float(row, f"intermediary_loss_pct{b_suffix}")
            entry["delta_t"] = _safe_float(row, col_map.get("delta_treatment", ""))
            entry["delta_b"] = _safe_float(row, col_map.get("delta_baseline", ""))
            kappa_loss_entries.append(entry)
        kappa_loss_entries.sort(key=lambda e: e["kappa"])

        has_loss_dynamics = any(
            e.get("system_loss_pct_t") is not None or e.get("system_loss_pct_b") is not None
            for e in kappa_loss_entries
        )

        if has_loss_dynamics and kappa_loss_entries:
            k_vals = [e["kappa"] for e in kappa_loss_entries]

            # Chart Y1: System Loss vs κ (Dual Line with fill)
            fig_loss_kappa = go.Figure()
            sys_t = [e.get("system_loss_pct_t") or 0 for e in kappa_loss_entries]
            sys_b = [e.get("system_loss_pct_b") or 0 for e in kappa_loss_entries]
            fig_loss_kappa.add_trace(go.Scatter(
                x=k_vals, y=sys_b, mode="lines+markers",
                name=paths.baseline_label, line=dict(color=COLORS_LIGHT.get(paths.sweep_type, "#aed6f1")),
                marker=dict(size=6),
            ))
            fig_loss_kappa.add_trace(go.Scatter(
                x=k_vals, y=sys_t, mode="lines+markers",
                name=paths.treatment_label, line=dict(color=color),
                marker=dict(size=6), fill="tonexty", fillcolor=f"rgba(0,0,0,0.05)",
            ))
            fig_loss_kappa.update_layout(
                title=f"System Loss vs κ ({mech_label})",
                xaxis_title="κ (liquidity ratio)", yaxis_title="System Loss %",
                template="plotly_white", height=420,
            )

            # Chart Y2: Loss Composition across κ (Stacked Area, treatment arm)
            fig_loss_composition = go.Figure()
            trader_loss = [e.get("total_loss_pct_t") or 0 for e in kappa_loss_entries]
            interm_loss = [e.get("intermediary_loss_pct_t") or 0 for e in kappa_loss_entries]
            if any(v > 0 for v in trader_loss):
                fig_loss_composition.add_trace(go.Scatter(
                    x=k_vals, y=trader_loss, mode="lines",
                    name="Trader Loss %", stackgroup="one",
                    line=dict(color=LOSS_COLORS["trader_total"]),
                    fillcolor="rgba(192, 57, 43, 0.3)",
                ))
            if any(v > 0 for v in interm_loss):
                fig_loss_composition.add_trace(go.Scatter(
                    x=k_vals, y=interm_loss, mode="lines",
                    name="Intermediary Loss %", stackgroup="one",
                    line=dict(color=LOSS_COLORS["intermediary_total"]),
                    fillcolor="rgba(41, 128, 185, 0.3)",
                ))
            fig_loss_composition.update_layout(
                title=f"Loss Composition across κ ({paths.treatment_label})",
                xaxis_title="κ", yaxis_title="Loss % of Total Debt",
                template="plotly_white", height=420,
            )

            # Chart Y3: Default Rate vs System Loss (Scatter)
            fig_loss_scatter = go.Figure()
            t_deltas = [e.get("delta_t") for e in kappa_loss_entries]
            b_deltas = [e.get("delta_b") for e in kappa_loss_entries]
            # Treatment points
            scatter_x_t = [d for d in t_deltas if d is not None]
            scatter_y_t = [
                e.get("system_loss_pct_t") or 0
                for e, d in zip(kappa_loss_entries, t_deltas)
                if d is not None
            ]
            if scatter_x_t:
                fig_loss_scatter.add_trace(go.Scatter(
                    x=scatter_x_t, y=scatter_y_t, mode="markers",
                    name=paths.treatment_label,
                    marker=dict(color=color, size=10, symbol="circle"),
                ))
            # Baseline points
            scatter_x_b = [d for d in b_deltas if d is not None]
            scatter_y_b = [
                e.get("system_loss_pct_b") or 0
                for e, d in zip(kappa_loss_entries, b_deltas)
                if d is not None
            ]
            if scatter_x_b:
                fig_loss_scatter.add_trace(go.Scatter(
                    x=scatter_x_b, y=scatter_y_b, mode="markers",
                    name=paths.baseline_label,
                    marker=dict(
                        color=COLORS_LIGHT.get(paths.sweep_type, "#aed6f1"),
                        size=10, symbol="diamond-open", line=dict(width=2),
                    ),
                ))
            fig_loss_scatter.update_layout(
                title=f"Default Rate vs System Loss ({mech_label})",
                xaxis_title="δ (Default Rate)", yaxis_title="System Loss %",
                template="plotly_white", height=450,
            )
    except Exception as exc:
        logger.debug("Dynamics loss charts failed: %s", exc)

    # Assemble
    nav = [
        ("timeline", "Default Timeline"), ("metrics", "Metrics Trajectories"),
        ("losses", "Loss Dynamics"), ("agents", "Agent Outcomes"),
    ]
    body_parts = [
        f"<h1>{mech_label} Dynamics Analysis</h1>",
        f"<p>Time-series dynamics and agent heterogeneity across "
        f"κ ∈ {{{', '.join(str(k) for k in kappas)}}}.</p>",
        f'<h2 id="timeline">Default Timeline</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_cum_defaults, "cum_defaults")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> Cumulative default curves show '
        'the speed and magnitude of system failure. Flatter treatment curves indicate the mechanism '
        'is delaying or preventing defaults.</div>',
        f'<h2 id="metrics">Metrics Trajectories</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_delta_t, "delta_t")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> δ_t tracks the instantaneous '
        'default rate. Treatment should show lower δ_t, especially in early days when liquidity '
        'stress peaks.</div>',
        f'<div class="chart-container">{_fig_to_div(fig_phi_t, "phi_t")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> φ_t tracks the clearing rate '
        '(fraction of due obligations settled). Higher is better — treatment should raise φ_t.</div>',
    ]

    # Loss dynamics section
    body_parts.append(f'<h2 id="losses">Loss Dynamics</h2>')
    if fig_loss_kappa is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_kappa, "loss_kappa")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> System loss as a function of '
            'liquidity (κ). The shaded area between curves shows the loss reduction from the treatment. '
            'Wider gaps at low κ indicate the mechanism is most effective under stress.</div>',
        ])
    if fig_loss_composition is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_composition, "loss_composition")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> How losses are distributed between '
            'traders (direct payable defaults) and intermediaries (mechanism cost). As κ increases, both '
            'components should decrease.</div>',
        ])
    if fig_loss_scatter is not None:
        body_parts.extend([
            f'<div class="chart-container">{_fig_to_div(fig_loss_scatter, "loss_scatter")}</div>',
            '<div class="interpretation"><strong>Interpretation:</strong> Each point is one parameter '
            'combination. Points above the diagonal suggest nonlinear loss amplification (losses exceed '
            'default rates). Filled markers = treatment, open = baseline.</div>',
        ])
    if all(f is None for f in [fig_loss_kappa, fig_loss_composition, fig_loss_scatter]):
        body_parts.append(
            '<div class="interpretation">No loss data available in comparison CSV.</div>'
        )

    body_parts.extend([
        f'<h2 id="agents">Agent Outcomes</h2>',
        f'<div class="chart-container">{_fig_to_div(fig_agents, "agents")}</div>',
        '<div class="interpretation"><strong>Interpretation:</strong> Each point is an agent. '
        'Agents that traded more and survived with positive cash benefited from the mechanism. '
        'Red X marks indicate defaulted agents.</div>',
    ])

    html = _dashboard_shell(
        f"{mech_label} Dynamics Analysis", nav, "\n".join(body_parts), paths.sweep_type,
    )

    output_path = output_dir / "dynamics_dashboard.html"
    output_path.write_text(html)
    logger.info("Dynamics dashboard: %s", output_path)
    return output_path


# ===================================================================
# 4. Narrative report
# ===================================================================


def _run_narrative(paths: SweepPaths, kappas: list[float], output_dir: Path) -> Path:
    """Auto-generated research narrative from sweep results."""
    logger.info("Running narrative report generation...")
    mech_label = MECHANISM_LABELS.get(paths.sweep_type, paths.sweep_type.title())
    color = COLORS.get(paths.sweep_type, "#3498db")

    # Load comparison CSV for aggregate stats
    rows = _read_csv(paths.comparison_csv)

    # Determine effect column based on sweep type
    if paths.sweep_type == "dealer":
        # Balanced comparison: delta_passive - delta_active = trading_effect
        delta_treatment_col = "delta_active"
        delta_baseline_col = "delta_passive"
        effect_name = "trading_effect"
    else:
        delta_treatment_col = "delta_lend"
        delta_baseline_col = "delta_idle"
        effect_name = "lending_effect" if paths.sweep_type == "nbfi" else "bank_lending_effect"

    # Compute effects per kappa
    kappa_effects: dict[float, list[float]] = defaultdict(list)
    all_effects: list[float] = []
    for row in rows:
        try:
            k = float(row["kappa"])
            dt = float(row.get(delta_treatment_col, 0))
            db = float(row.get(delta_baseline_col, 0))
            effect = db - dt  # positive = treatment reduced defaults
            kappa_effects[k].append(effect)
            all_effects.append(effect)
        except (ValueError, TypeError, KeyError):
            continue

    # Load stats summary if available
    stats = _safe_load_json(paths.stats_summary) if paths.stats_summary else None
    sensitivity = _safe_load_json(paths.stats_sensitivity) if paths.stats_sensitivity else None

    # Build narrative sections
    sections = []

    # Executive summary
    n_runs = len(rows)
    mean_effect = sum(all_effects) / len(all_effects) if all_effects else 0
    n_positive = sum(1 for e in all_effects if e > 0)
    n_negative = sum(1 for e in all_effects if e < 0)

    sections.append(f"""
    <h2>Executive Summary</h2>
    <p>This report analyses the <strong>{mech_label}</strong> mechanism across
    <strong>{n_runs}</strong> parameter combinations. The mechanism was tested by comparing
    treatment runs ({paths.treatment_label}) against baseline runs ({paths.baseline_label}).</p>

    <div style="display: flex; gap: 20px; margin: 20px 0;">
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 28px; font-weight: bold; color: {color};">
                {mean_effect:+.4f}
            </div>
            <div style="color: #666; margin-top: 4px;">Mean Effect (δ reduction)</div>
        </div>
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 28px; font-weight: bold; color: #27ae60;">
                {n_positive}
            </div>
            <div style="color: #666; margin-top: 4px;">Improved</div>
        </div>
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 28px; font-weight: bold; color: #e74c3c;">
                {n_negative}
            </div>
            <div style="color: #666; margin-top: 4px;">Worsened</div>
        </div>
    </div>
    """)

    # Per-kappa breakdown
    sections.append('<h2>Effect by Liquidity Level</h2>')
    kappa_table_rows = []
    for k in sorted(kappa_effects.keys()):
        effects = kappa_effects[k]
        mean_e = sum(effects) / len(effects) if effects else 0
        n_pos = sum(1 for e in effects if e > 0)
        kappa_table_rows.append(
            f"<tr><td>{k}</td><td>{len(effects)}</td>"
            f"<td>{mean_e:+.4f}</td><td>{n_pos}/{len(effects)}</td></tr>"
        )
    sections.append(f"""
    <div class="chart-container">
    <table class="summary-table">
    <thead><tr><th>κ</th><th>Runs</th><th>Mean Effect</th><th>Improved</th></tr></thead>
    <tbody>{"".join(kappa_table_rows)}</tbody>
    </table>
    </div>
    <div class="interpretation"><strong>Key insight:</strong> The mechanism's effect typically
    varies with liquidity stress (κ). At low κ (high stress), there is more scope for
    improvement. At high κ, the system is already liquid and the mechanism has less impact.</div>
    """)

    # Stats summary if available
    if stats:
        sections.append('<h2>Statistical Analysis</h2>')
        sections.append('<div class="chart-container"><pre style="font-size: 13px;">')
        sections.append(json.dumps(stats, indent=2, default=str))
        sections.append('</pre></div>')

    # Key findings
    sections.append(f"""
    <h2>Key Findings</h2>
    <div class="interpretation">
    <ol>
        <li><strong>Overall effectiveness:</strong> The {mech_label.lower()} mechanism
        {"improved" if mean_effect > 0 else "did not improve"} outcomes on average
        (mean δ reduction: {mean_effect:+.4f}).</li>
        <li><strong>Consistency:</strong> The mechanism helped in {n_positive} out of
        {len(all_effects)} cases ({n_positive/len(all_effects)*100:.0f}% of runs).</li>
        <li><strong>Stress dependence:</strong> Effect is {"strongest" if n_positive > n_negative else "mixed"}
        at low κ (high stress).</li>
    </ol>
    </div>
    """)

    # Loss analysis section
    try:
        col_map = LOSS_COLUMN_MAP.get(paths.sweep_type, {})
        t_suffix = col_map.get("treatment", "_active")
        b_suffix = col_map.get("baseline", "_passive")
        loss_effect_col = col_map.get("system_loss_effect", "")

        sys_loss_t_all: list[float] = []
        sys_loss_b_all: list[float] = []
        total_loss_t_all: list[float] = []
        interm_loss_t_all: list[float] = []
        kappa_loss_data: list[dict[str, Any]] = []

        for row in rows:
            slt = _safe_float(row, f"system_loss_pct{t_suffix}")
            slb = _safe_float(row, f"system_loss_pct{b_suffix}")
            if slt is not None:
                sys_loss_t_all.append(slt)
            if slb is not None:
                sys_loss_b_all.append(slb)
            tlt = _safe_float(row, f"total_loss_pct{t_suffix}")
            if tlt is not None:
                total_loss_t_all.append(tlt)
            ilt = _safe_float(row, f"intermediary_loss_pct{t_suffix}")
            if ilt is not None:
                interm_loss_t_all.append(ilt)

        # Per-kappa loss table
        for k in sorted(kappa_effects.keys()):
            matched = [
                r for r in rows
                if _safe_float(r, "kappa") is not None and _approx_eq(float(r["kappa"]), k)
            ]
            if not matched:
                continue
            row = matched[0]
            kappa_loss_data.append({
                "kappa": k,
                "sys_loss_b": _safe_float(row, f"system_loss_pct{b_suffix}"),
                "sys_loss_t": _safe_float(row, f"system_loss_pct{t_suffix}"),
                "effect": _safe_float(row, loss_effect_col),
            })

        has_loss_narrative = bool(sys_loss_t_all or sys_loss_b_all)

        if has_loss_narrative:
            mean_loss_t = sum(sys_loss_t_all) / len(sys_loss_t_all) if sys_loss_t_all else 0
            mean_loss_b = sum(sys_loss_b_all) / len(sys_loss_b_all) if sys_loss_b_all else 0
            loss_reduction = mean_loss_b - mean_loss_t
            loss_reduction_pct = (loss_reduction / mean_loss_b * 100) if mean_loss_b > 0 else 0

            # Attribution
            mean_trader = sum(total_loss_t_all) / len(total_loss_t_all) if total_loss_t_all else 0
            mean_interm = sum(interm_loss_t_all) / len(interm_loss_t_all) if interm_loss_t_all else 0
            total_sys = mean_trader + mean_interm
            trader_share = (mean_trader / total_sys * 100) if total_sys > 0 else 0
            interm_share = (mean_interm / total_sys * 100) if total_sys > 0 else 0

            sections.append(f"""
    <h2>Loss Analysis</h2>
    <p>Beyond default counts, system losses measure the actual economic damage from defaults.
    System loss combines trader losses (payable defaults, deposit erosion) with intermediary
    losses (dealer mark-to-market, NBFI loan defaults, bank credit losses, CB backstop).</p>

    <div style="display: flex; gap: 20px; margin: 20px 0;">
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; font-weight: bold; color: {LOSS_COLORS['trader_total']};">
                {mean_loss_b:.4f}
            </div>
            <div style="color: #666; margin-top: 4px;">Mean Baseline System Loss</div>
        </div>
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; font-weight: bold; color: {color};">
                {mean_loss_t:.4f}
            </div>
            <div style="color: #666; margin-top: 4px;">Mean Treatment System Loss</div>
        </div>
        <div style="background: white; padding: 20px; border-radius: 8px; flex: 1; text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; font-weight: bold; color: #27ae60;">
                {loss_reduction_pct:+.1f}%
            </div>
            <div style="color: #666; margin-top: 4px;">Loss Reduction</div>
        </div>
    </div>

    <div class="interpretation">
    <strong>Loss attribution (treatment arm):</strong> Of total system loss,
    {trader_share:.0f}% is borne by traders (payable defaults + deposits) and
    {interm_share:.0f}% by intermediaries (dealer, lender, bank, CB).
    </div>
    """)

            # Per-kappa loss table
            if kappa_loss_data:
                loss_table_rows = []
                for entry in kappa_loss_data:
                    sl_b = f"{entry['sys_loss_b']:.4f}" if entry.get("sys_loss_b") is not None else "n/a"
                    sl_t = f"{entry['sys_loss_t']:.4f}" if entry.get("sys_loss_t") is not None else "n/a"
                    eff = f"{entry['effect']:+.4f}" if entry.get("effect") is not None else "n/a"
                    loss_table_rows.append(
                        f"<tr><td>{entry['kappa']}</td><td>{sl_b}</td>"
                        f"<td>{sl_t}</td><td>{eff}</td></tr>"
                    )
                sections.append(f"""
    <div class="chart-container">
    <table class="summary-table" style="font-family: system-ui, sans-serif;">
    <thead><tr>
        <th>κ</th><th>Baseline System Loss %</th>
        <th>Treatment System Loss %</th><th>Loss Effect</th>
    </tr></thead>
    <tbody>{"".join(loss_table_rows)}</tbody>
    </table>
    </div>
    """)
    except Exception as exc:
        logger.debug("Narrative loss section failed: %s", exc)

    # Assemble
    nav = [("summary", "Summary"), ("kappa", "By κ"), ("findings", "Findings"), ("losses", "Losses")]
    body = f"<h1>{mech_label} Narrative Report</h1>\n" + "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{mech_label} Narrative Report</title>
<style>
body {{
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 800px; margin: 40px auto; padding: 0 20px;
    background: #fafafa; color: #2c3e50; line-height: 1.7;
}}
h1 {{ font-family: system-ui, sans-serif; border-bottom: 3px solid {color}; padding-bottom: 10px; }}
h2 {{ font-family: system-ui, sans-serif; margin-top: 40px; color: #34495e; }}
.chart-container {{
    background: white; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0; padding: 20px;
}}
.interpretation {{
    background: #eaf2f8; border-left: 4px solid {color};
    padding: 12px 16px; margin: 10px 0 20px;
    border-radius: 0 4px 4px 0; font-size: 14px;
}}
.summary-table {{
    width: 100%; border-collapse: collapse; font-size: 14px;
    font-family: system-ui, sans-serif;
}}
.summary-table th {{
    background: #2c3e50; color: white; padding: 10px 12px; text-align: left;
}}
.summary-table td {{
    padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
}}
pre {{ overflow-x: auto; }}
</style>
</head><body>
{body}
<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>bilancio sweep analyze</code>
</p>
</body></html>"""

    output_path = output_dir / "narrative_report.html"
    output_path.write_text(html)
    logger.info("Narrative report: %s", output_path)
    return output_path


# ===================================================================
# Public API
# ===================================================================


def run_post_sweep_analysis(
    experiment_root: Path,
    sweep_type: str,
    analyses: list[str],
    output_dir: Path | None = None,
    kappas: list[float] | None = None,
) -> dict[str, Path]:
    """Run selected post-sweep analyses and return paths to generated files.

    Args:
        experiment_root: Path to the sweep output directory.
        sweep_type: One of "dealer", "bank", "nbfi".
        analyses: Subset of ["drilldowns", "deltas", "dynamics", "narrative"].
        output_dir: Where to write analysis outputs. Defaults to
            ``experiment_root / "aggregate" / "analysis"``.
        kappas: Representative kappa levels to analyse. Auto-detected from
            comparison.csv if None.

    Returns:
        Mapping from analysis name to output file path.
    """
    if sweep_type not in VALID_SWEEP_TYPES:
        raise ValueError(f"sweep_type must be one of {VALID_SWEEP_TYPES}, got {sweep_type!r}")
    for a in analyses:
        if a not in VALID_ANALYSES:
            raise ValueError(f"Unknown analysis {a!r}. Must be one of {VALID_ANALYSES}")

    paths = _resolve_sweep_paths(experiment_root, sweep_type)

    if not paths.comparison_csv.is_file():
        raise FileNotFoundError(
            f"Comparison CSV not found: {paths.comparison_csv}. "
            f"Has the sweep completed?"
        )

    if output_dir is None:
        output_dir = paths.experiment_root / "aggregate" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if kappas is None:
        kappas = _auto_detect_kappas(paths.comparison_csv)
        logger.info("Auto-detected kappa levels: %s", kappas)

    if not kappas:
        raise ValueError("No kappa levels found in comparison CSV")

    results: dict[str, Path] = {}

    dispatch = {
        "drilldowns": _run_drilldowns,
        "deltas": _run_treatment_deltas,
        "dynamics": _run_dynamics,
        "narrative": _run_narrative,
    }

    for name in analyses:
        fn = dispatch[name]
        try:
            path = fn(paths, kappas, output_dir)
            results[name] = path
            logger.info("  %s -> %s", name, path)
        except Exception as exc:
            logger.error("  %s FAILED: %s", name, exc)

    return results
