#!/usr/bin/env python3
"""Dose-response chart for intermediary capital sweep (Plan 053).

Usage:
    uv run python scripts/dose_response_chart.py path/to/aggregate/comparison.csv [-o output.html]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_data(csv_path: Path) -> list[dict]:
    """Load comparison.csv and return list of row dicts."""
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(v: str) -> float:
    """Parse a CSV value to float, returning NaN for empty/invalid."""
    if not v:
        return float("nan")
    try:
        return float(v)
    except (ValueError, OverflowError):
        return float("nan")


def build_chart(rows: list[dict], output_path: Path) -> None:
    """Build dose-response Plotly chart from comparison data."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Group rows by kappa
    by_kappa: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        k = _safe_float(row.get("kappa", ""))
        if np.isfinite(k):
            by_kappa[k].append(row)

    kappas = sorted(by_kappa.keys())
    if not kappas:
        print("No valid kappa values found in data.")
        sys.exit(1)

    n_kappas = len(kappas)
    fig = make_subplots(
        rows=n_kappas, cols=2,
        subplot_titles=[
            title
            for k in kappas
            for title in [f"κ={k}: Default Relief", f"κ={k}: Marginal Efficiency"]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.08,
    )

    # Arm definitions: (arm_label, delta_col, color)
    arms = [
        ("Dealer", "delta_active", "#3498db"),
        ("NBFI", "delta_lender", "#27ae60"),
        ("Bank", "delta_bank_passive", "#e67e22"),
    ]

    for row_idx, kappa in enumerate(kappas, 1):
        kappa_rows = by_kappa[kappa]

        for arm_label, delta_col, color in arms:
            # Group by pool_scale
            by_pool: dict[float, list[tuple[float, float]]] = defaultdict(list)
            for r in kappa_rows:
                pool = _safe_float(r.get("pool_scale", ""))
                delta_p = _safe_float(r.get("delta_passive", ""))
                delta_t = _safe_float(r.get(delta_col, ""))
                cap_frac = _safe_float(r.get("capital_fraction", ""))
                if np.isfinite(pool) and np.isfinite(delta_p) and np.isfinite(delta_t):
                    relief = delta_p - delta_t
                    by_pool[pool].append((cap_frac, relief))

            if not by_pool:
                continue

            pools = sorted(by_pool.keys())
            x_vals = []
            y_relief_mean = []
            y_relief_std = []
            y_efficiency = []

            for pool in pools:
                pairs = by_pool[pool]
                caps = [p[0] for p in pairs]
                reliefs = [p[1] for p in pairs]
                mean_cap = np.nanmean(caps) if caps else 0
                mean_relief = np.nanmean(reliefs)
                std_relief = np.nanstd(reliefs) if len(reliefs) > 1 else 0

                x_vals.append(mean_cap)
                y_relief_mean.append(mean_relief)
                y_relief_std.append(std_relief)
                eff = mean_relief / mean_cap if mean_cap > 0 else float("nan")
                y_efficiency.append(eff)

            x_arr = np.array(x_vals)
            y_arr = np.array(y_relief_mean)
            std_arr = np.array(y_relief_std)

            # Relief subplot
            show_legend = row_idx == 1
            fig.add_trace(
                go.Scatter(
                    x=x_arr, y=y_arr,
                    mode="lines+markers",
                    name=arm_label,
                    line=dict(color=color),
                    showlegend=show_legend,
                    legendgroup=arm_label,
                ),
                row=row_idx, col=1,
            )
            # Error band
            if np.any(std_arr > 0):
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_arr, x_arr[::-1]]),
                        y=np.concatenate([y_arr + std_arr, (y_arr - std_arr)[::-1]]),
                        fill="toself",
                        fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=arm_label,
                    ),
                    row=row_idx, col=1,
                )

            # Efficiency subplot
            eff_arr = np.array(y_efficiency)
            valid = np.isfinite(eff_arr)
            fig.add_trace(
                go.Scatter(
                    x=x_arr[valid], y=eff_arr[valid],
                    mode="lines+markers",
                    name=arm_label,
                    line=dict(color=color, dash="dot"),
                    showlegend=False,
                    legendgroup=arm_label,
                ),
                row=row_idx, col=2,
            )

        # Add reference line at default pool size
        fig.add_vline(
            x=0.375 * 0.85,  # approximate capital fraction at default pool
            line_dash="dash", line_color="gray", line_width=1,
            annotation_text="default",
            row=row_idx, col=1,
        )

    fig.update_layout(
        height=350 * n_kappas,
        title="Intermediary Capital Dose-Response (Plan 053)",
        template="plotly_white",
    )
    for i in range(1, n_kappas + 1):
        fig.update_xaxes(title_text="Capital Fraction (of Q_total)", row=i, col=1)
        fig.update_yaxes(title_text="Default Relief", row=i, col=1)
        fig.update_xaxes(title_text="Capital Fraction (of Q_total)", row=i, col=2)
        fig.update_yaxes(title_text="Relief / Capital", row=i, col=2)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Chart saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dose-response chart for Plan 053")
    parser.add_argument("csv_path", type=Path, help="Path to comparison.csv")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"File not found: {args.csv_path}")
        sys.exit(1)

    rows = load_data(args.csv_path)
    output = args.output or args.csv_path.parent / "dose_response.html"
    build_chart(rows, output)


if __name__ == "__main__":
    main()
