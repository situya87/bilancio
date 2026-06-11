"""Generate ~49 PNG charts from production sweep data.

Reads unified NBFI and bank sweep comparison CSVs plus network/timeseries
caches, produces publication-quality charts using matplotlib + seaborn,
and saves them to out/sweep_analysis/charts/.

Usage:
    uv run python scripts/sweep_analysis_charts.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats as sp_stats

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
    load_unified_data,
    pivot_for_heatmap,
    standardized_coefs,
    summary_by_group,
)
from bilancio.analysis.cross_sweep_network import (
    load_network_metrics,
    load_network_timeseries,
    aggregate_timeseries_by_arm_kappa,
    load_mechanism_timeseries,
    aggregate_mechanism_timeseries,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NBFI_CSV = PROJECT_ROOT / "out" / "sweep_nbfi_full" / "aggregate" / "comparison.csv"
BANK_CSV = PROJECT_ROOT / "out" / "sweep_bank_full" / "aggregate" / "comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "out" / "sweep_analysis" / "charts"
CACHE_DIR = PROJECT_ROOT / "out" / "sweep_analysis"

# OLS feature columns
X_COLS = ["log_kappa", "concentration", "mu", "outside_mid_ratio"]

# Three mechanisms we compare throughout
MECHANISMS = ["trading_effect", "lending_effect", "bank_lending_effect"]
MECHANISM_NICE = {
    "trading_effect": "Dealer Trading",
    "lending_effect": "NBFI Lending",
    "bank_lending_effect": "Bank Lending",
}

# Arm columns for delta distributions
ARM_COLS = {
    "delta_passive": ("Passive", PASSIVE_GREY),
    "delta_active": ("Active (Dealer)", DEALER_RED),
    "delta_lender": ("NBFI Lender", NBFI_GREEN),
    "delta_idle": ("Bank Idle", "#7f8c8d"),
    "delta_lend": ("Bank Lend", BANK_BLUE),
}

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    """Set consistent matplotlib/seaborn style for all charts."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
    })


def _mech_color(m: str) -> str:
    return MECHANISM_COLORS.get(m, "#333333")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ===================================================================
# Section 1: Descriptive
# ===================================================================

def c01_mechanism_boxplots(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Box plots of trading_effect, lending_effect, bank_lending_effect."""
    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = []
    labels = []
    colors = []
    for m in MECHANISMS:
        vals = data[m]
        valid = vals[np.isfinite(vals)]
        box_data.append(valid)
        labels.append(MECHANISM_NICE[m])
        colors.append(_mech_color(m))

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    showfliers=True, flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Effect (reduction in default rate)")
    ax.set_title("Mechanism Effect Distributions")
    _save(fig, output_dir / "c01_mechanism_boxplots.png")


def c02_delta_distributions(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Overlaid KDE/histograms of delta for each arm."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, (label, color) in ARM_COLS.items():
        if col not in data:
            continue
        vals = data[col]
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            continue
        ax.hist(valid, bins=30, alpha=0.25, color=color, density=True, label=None)
        # KDE overlay
        if len(valid) > 2 and np.std(valid) > 1e-10:
            kde_x = np.linspace(np.min(valid) - 0.05, np.max(valid) + 0.05, 200)
            kde = sp_stats.gaussian_kde(valid)
            ax.plot(kde_x, kde(kde_x), color=color, linewidth=2, label=label)
        else:
            ax.axvline(np.mean(valid), color=color, linewidth=2, label=label)

    ax.set_xlabel("Default Rate (delta)")
    ax.set_ylabel("Density")
    ax.set_title("Default Rate Distributions by Arm")
    ax.legend(loc="upper left")
    _save(fig, output_dir / "c02_delta_distributions.png")


def c03_delta_vs_kappa(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Line plot: mean delta vs kappa for each arm."""
    fig, ax = plt.subplots(figsize=(9, 5))
    kappas = np.sort(np.unique(data["kappa"]))

    for col, (label, color) in ARM_COLS.items():
        if col not in data:
            continue
        means = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Kappa (liquidity ratio)")
    ax.set_ylabel("Mean Default Rate (delta)")
    ax.set_title("Default Rate vs Liquidity by Arm")
    ax.legend(loc="upper right")
    ax.set_xscale("log")
    _save(fig, output_dir / "c03_delta_vs_kappa.png")


# ===================================================================
# Section 2: Parameter Sensitivity
# ===================================================================

def _plot_effect_heatmaps(
    data: dict[str, np.ndarray],
    effect_col: str,
    title_prefix: str,
    output_dir: Path,
    filename: str,
) -> None:
    """1x3 subplots: effect in kappa x c, kappa x mu, kappa x rho space."""
    pairs = [
        ("kappa", "concentration", "c"),
        ("kappa", "mu", "mu"),
        ("kappa", "outside_mid_ratio", "rho"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_prefix} Heatmaps", fontsize=14, y=1.02)

    for ax, (row_col, col_col, col_label) in zip(axes, pairs):
        row_vals, col_vals, matrix = pivot_for_heatmap(data, row_col, col_col, effect_col)

        # Determine symmetric color range centered at 0
        vmax = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 0.1
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            origin="lower", interpolation="nearest",
        )
        # Axis labels
        ax.set_xticks(range(len(col_vals)))
        ax.set_xticklabels([f"{v:.2g}" for v in col_vals], fontsize=8)
        ax.set_yticks(range(len(row_vals)))
        ax.set_yticklabels([f"{v:.2g}" for v in row_vals], fontsize=8)
        ax.set_xlabel(col_label)
        ax.set_ylabel("kappa")

        # Annotate cells
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = matrix[i, j]
                if np.isfinite(val):
                    text_color = "white" if abs(val) > 0.6 * vmax else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=text_color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _save(fig, output_dir / filename)


def c04_heatmap_trading_effect(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Trading effect heatmaps: kappa x c, kappa x mu, kappa x rho."""
    _plot_effect_heatmaps(data, "trading_effect", "Trading Effect", output_dir,
                          "c04_heatmap_trading_effect.png")


def c05_heatmap_lending_effect(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Lending effect heatmaps: kappa x c, kappa x mu, kappa x rho."""
    _plot_effect_heatmaps(data, "lending_effect", "Lending Effect", output_dir,
                          "c05_heatmap_lending_effect.png")


def c06_heatmap_bank_effect(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Bank lending effect heatmaps: kappa x c, kappa x mu, kappa x rho."""
    _plot_effect_heatmaps(data, "bank_lending_effect", "Bank Lending Effect", output_dir,
                          "c06_heatmap_bank_effect.png")


def c07_marginal_effects(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """2x2 subplots: mean effect vs each parameter for all 3 mechanisms."""
    params = [
        ("kappa", "Kappa"),
        ("concentration", "Concentration (c)"),
        ("mu", "Mu"),
        ("outside_mid_ratio", "Outside Mid Ratio (rho)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (param, param_label) in zip(axes, params):
        param_vals = np.sort(np.unique(data[param]))

        for m in MECHANISMS:
            means = []
            ci_lo = []
            ci_hi = []
            for pv in param_vals:
                mask = data[param] == pv
                vals = data[m][mask]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    point, lo, hi = bootstrap_ci(valid, n_resamples=2000, rng_seed=42)
                    means.append(point)
                    ci_lo.append(lo)
                    ci_hi.append(hi)
                else:
                    means.append(np.nan)
                    ci_lo.append(np.nan)
                    ci_hi.append(np.nan)

            color = _mech_color(m)
            ax.plot(param_vals, means, "o-", color=color, label=MECHANISM_NICE[m],
                    linewidth=2, markersize=4)
            ax.fill_between(param_vals, ci_lo, ci_hi, color=color, alpha=0.15)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel(param_label)
        ax.set_ylabel("Mean Effect")
        ax.set_title(f"Effect vs {param_label}")
        ax.legend(fontsize=8)

    fig.suptitle("Marginal Effects by Parameter", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "c07_marginal_effects.png")


def c08_kappa_vs_delta_scatter(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Scatter: kappa vs delta for all arms."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, (label, color) in ARM_COLS.items():
        if col not in data:
            continue
        vals = data[col]
        mask = np.isfinite(vals) & np.isfinite(data["kappa"])
        ax.scatter(data["kappa"][mask], vals[mask], color=color, alpha=0.25,
                   s=12, label=label, edgecolors="none")

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Default Rate (delta)")
    ax.set_title("Kappa vs Default Rate (all arms)")
    ax.set_xscale("log")
    ax.legend(markerscale=3, fontsize=9)
    _save(fig, output_dir / "c08_kappa_vs_delta_scatter.png")


# ===================================================================
# Section 3: Mechanism Comparison
# ===================================================================

def c09_mechanism_ranking(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Grouped bar: which mechanism wins for each kappa level."""
    kappas = np.sort(np.unique(data["kappa"]))
    n_kappas = len(kappas)

    # For each kappa, count wins
    win_counts = {m: np.zeros(n_kappas) for m in MECHANISMS}

    for i, k in enumerate(kappas):
        mask = data["kappa"] == k
        n_rows = int(np.sum(mask))
        for j in range(len(mask)):
            if not mask[j]:
                continue
            best = None
            best_val = -np.inf
            for m in MECHANISMS:
                val = data[m][j]
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best = m
            if best is not None and best_val > 0.005:
                win_counts[best][i] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.25
    x = np.arange(n_kappas)

    for offset, m in enumerate(MECHANISMS):
        ax.bar(x + offset * bar_width, win_counts[m], bar_width,
               color=_mech_color(m), label=MECHANISM_NICE[m], alpha=0.8)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Kappa")
    ax.set_ylabel("Number of Wins")
    ax.set_title("Mechanism Wins by Kappa Level")
    ax.legend()
    _save(fig, output_dir / "c09_mechanism_ranking.png")


def c10_phase_diagram(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Kappa x concentration heatmap colored by dominant mechanism (regime)."""
    regime_map = {"dealer": 0, "nbfi": 1, "bank": 2, "mixed": 3, "none": 4}
    regime_colors_list = [DEALER_RED, NBFI_GREEN, BANK_BLUE, COMBINED_PURPLE, PASSIVE_GREY]
    regime_labels = ["Dealer", "NBFI", "Bank", "Mixed", "None"]

    kappas = np.sort(np.unique(data["kappa"]))
    concentrations = np.sort(np.unique(data["concentration"]))

    matrix = np.full((len(kappas), len(concentrations)), np.nan)
    for i, k in enumerate(kappas):
        for j, c in enumerate(concentrations):
            mask = (data["kappa"] == k) & (data["concentration"] == c)
            regimes = data["regime"][mask]
            if len(regimes) == 0:
                continue
            # Most common regime
            unique, counts = np.unique(regimes, return_counts=True)
            dominant = unique[np.argmax(counts)]
            matrix[i, j] = regime_map.get(dominant, 4)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create custom colormap from discrete regime colors
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(regime_colors_list)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm,
                   origin="lower", interpolation="nearest")

    ax.set_xticks(range(len(concentrations)))
    ax.set_xticklabels([f"{c:.2g}" for c in concentrations], fontsize=9)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Concentration (c)")
    ax.set_ylabel("Kappa")
    ax.set_title("Dominant Mechanism Phase Diagram")

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(regime_colors_list, regime_labels)]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    _save(fig, output_dir / "c10_phase_diagram.png")


def c11_trading_vs_bank_scatter(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Scatter: trading_effect vs bank_lending_effect, colored by kappa."""
    fig, ax = plt.subplots(figsize=(8, 6))

    te = data["trading_effect"]
    ble = data["bank_lending_effect"]
    kappa = data["kappa"]
    mask = np.isfinite(te) & np.isfinite(ble)

    sc = ax.scatter(te[mask], ble[mask], c=np.log(kappa[mask]),
                    cmap="viridis", alpha=0.6, s=20, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(kappa)")

    # Diagonal line (equal effectiveness)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Trading Effect (Dealer)")
    ax.set_ylabel("Bank Lending Effect")
    ax.set_title("Trading vs Bank Lending Effect")
    _save(fig, output_dir / "c11_trading_vs_bank_scatter.png")


def c12_additive_test(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Bar chart: trading + lending vs bank effect by kappa."""
    kappas = np.sort(np.unique(data["kappa"]))

    combined_means = []
    bank_means = []
    for k in kappas:
        mask = data["kappa"] == k
        te = data["trading_effect"][mask]
        le = data["lending_effect"][mask]
        ble = data["bank_lending_effect"][mask]

        te_valid = te[np.isfinite(te)]
        le_valid = le[np.isfinite(le)]
        ble_valid = ble[np.isfinite(ble)]

        combined = np.mean(te_valid) + np.mean(le_valid) if len(te_valid) > 0 and len(le_valid) > 0 else np.nan
        bank_m = np.mean(ble_valid) if len(ble_valid) > 0 else np.nan
        combined_means.append(combined)
        bank_means.append(bank_m)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(kappas))
    bar_width = 0.35

    ax.bar(x - bar_width / 2, combined_means, bar_width,
           color=COMBINED_PURPLE, alpha=0.7, label="Trading + NBFI Lending")
    ax.bar(x + bar_width / 2, bank_means, bar_width,
           color=BANK_BLUE, alpha=0.7, label="Bank Lending")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Kappa")
    ax.set_ylabel("Mean Effect (delta reduction)")
    ax.set_title("Additivity Test: Dealer+NBFI vs Bank by Kappa")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, output_dir / "c12_additive_test.png")


# ===================================================================
# Section 4: Regression
# ===================================================================

def _fit_three_ols(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Fit OLS models for the three mechanism effects."""
    results = {}
    for m in MECHANISMS:
        results[m] = fit_ols_from_df(data, m, X_COLS)
    return results


def c13_forest_plot(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Forest plot: OLS coefficients with 95% CIs for all 3 regressions."""
    ols_results = _fit_three_ols(data)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = 0
    y_ticks = []
    y_labels = []

    for m_idx, m in enumerate(MECHANISMS):
        ols = ols_results[m]
        color = _mech_color(m)

        # Skip intercept
        for i, name in enumerate(ols.feature_names):
            if name == "intercept":
                continue
            coef = ols.coef[i]
            se = ols.se[i]
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se

            ax.errorbar(coef, y_pos, xerr=[[coef - ci_lo], [ci_hi - coef]],
                        fmt="o", color=color, markersize=6, capsize=4,
                        linewidth=1.5)

            y_ticks.append(y_pos)
            y_labels.append(f"{name}\n({MECHANISM_NICE[m]})")
            y_pos += 1

        # Add separator line between mechanisms
        if m_idx < len(MECHANISMS) - 1:
            ax.axhline(y_pos - 0.5, color="grey", linewidth=0.5, linestyle=":")
            y_pos += 0.5

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("OLS Coefficient (95% CI)")
    ax.set_title("Forest Plot: Mechanism Effect Drivers")
    ax.invert_yaxis()

    # Add R-squared annotations
    text_lines = []
    for m in MECHANISMS:
        ols = ols_results[m]
        text_lines.append(f"{MECHANISM_NICE[m]}: R²={ols.r_squared:.3f}")
    ax.text(0.98, 0.02, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"})

    fig.tight_layout()
    _save(fig, output_dir / "c13_forest_plot.png")


def c14_actual_vs_predicted(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Scatter: actual vs predicted for best model (highest R-squared)."""
    ols_results = _fit_three_ols(data)

    # Pick best model by R-squared
    best_m = max(MECHANISMS, key=lambda m: ols_results[m].r_squared)
    ols = ols_results[best_m]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Get actual values (need to handle NaN masking same as in fit_ols)
    y_actual = data[best_m]
    X = np.column_stack([data[c] for c in X_COLS])
    mask = np.isfinite(y_actual)
    for j in range(X.shape[1]):
        mask &= np.isfinite(X[:, j])
    y_actual_clean = y_actual[mask]

    ax.scatter(ols.y_hat, y_actual_clean, alpha=0.3, s=15, color=_mech_color(best_m),
               edgecolors="none")

    # 45-degree line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Actual vs Predicted: {MECHANISM_NICE[best_m]} (R²={ols.r_squared:.3f})")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, output_dir / "c14_actual_vs_predicted.png")


def c15_residual_diagnostics(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """2x2: residuals vs fitted, QQ plot, residual histogram, residuals vs kappa."""
    ols_results = _fit_three_ols(data)

    # Use the model with highest R-squared
    best_m = max(MECHANISMS, key=lambda m: ols_results[m].r_squared)
    ols = ols_results[best_m]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Residual Diagnostics: {MECHANISM_NICE[best_m]}", fontsize=14)

    # 1. Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(ols.y_hat, ols.residuals, alpha=0.3, s=12, color=_mech_color(best_m),
               edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. QQ plot
    ax = axes[0, 1]
    sorted_resid = np.sort(ols.residuals)
    n = len(sorted_resid)
    theoretical = sp_stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
    ax.scatter(theoretical, sorted_resid, alpha=0.4, s=12, color=_mech_color(best_m),
               edgecolors="none")
    # Reference line
    q25, q75 = np.percentile(sorted_resid, [25, 75])
    t25, t75 = sp_stats.norm.ppf([0.25, 0.75])
    if t75 - t25 > 0:
        slope = (q75 - q25) / (t75 - t25)
        intercept = q25 - slope * t25
        ref_x = np.array([theoretical[0], theoretical[-1]])
        ax.plot(ref_x, intercept + slope * ref_x, "r--", linewidth=1)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Normal Q-Q Plot")

    # 3. Residual histogram
    ax = axes[1, 0]
    ax.hist(ols.residuals, bins=30, color=_mech_color(best_m), alpha=0.6,
            edgecolor="white", density=True)
    # Normal overlay
    x_norm = np.linspace(ols.residuals.min(), ols.residuals.max(), 100)
    ax.plot(x_norm, sp_stats.norm.pdf(x_norm, np.mean(ols.residuals), np.std(ols.residuals)),
            "k-", linewidth=1.5)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")

    # 4. Residuals vs kappa (log scale)
    ax = axes[1, 1]
    # Need kappa values aligned with the OLS-cleaned data
    y_col = data[best_m]
    X = np.column_stack([data[c] for c in X_COLS])
    valid_mask = np.isfinite(y_col)
    for j in range(X.shape[1]):
        valid_mask &= np.isfinite(X[:, j])
    kappa_clean = data["kappa"][valid_mask]

    ax.scatter(kappa_clean, ols.residuals, alpha=0.3, s=12, color=_mech_color(best_m),
               edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Kappa")

    fig.tight_layout()
    _save(fig, output_dir / "c15_residual_diagnostics.png")


# ===================================================================
# Section 5: Statistical Tests
# ===================================================================

def c16_bootstrap_distributions(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """1x3 subplots: bootstrap mean distributions for each mechanism."""
    rng = np.random.default_rng(42)
    n_boot = 10_000

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, m in zip(axes, MECHANISMS):
        vals = data[m]
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            ax.set_title(f"{MECHANISM_NICE[m]}\n(no data)")
            continue

        boot_means = np.array([
            np.mean(rng.choice(valid, size=len(valid))) for _ in range(n_boot)
        ])

        color = _mech_color(m)
        ax.hist(boot_means, bins=60, color=color, alpha=0.6, edgecolor="white", density=True)

        # Observed mean line
        obs_mean = np.mean(valid)
        ax.axvline(obs_mean, color="black", linewidth=2, linestyle="-",
                   label=f"Mean={obs_mean:.4f}")
        # 95% CI
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        ax.axvline(ci_lo, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvline(ci_hi, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color="black",
                   label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        # Zero line
        ax.axvline(0, color="red", linewidth=1, linestyle=":", alpha=0.7, label="Zero")

        ax.set_xlabel("Bootstrap Mean")
        ax.set_ylabel("Density")
        ax.set_title(f"{MECHANISM_NICE[m]}")
        ax.legend(fontsize=7)

    fig.suptitle("Bootstrap Distributions of Mean Effect", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "c16_bootstrap_distributions.png")


# ===================================================================
# Section 6: Cross-Sweep
# ===================================================================

def c17_grouped_bar_effects(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Grouped bar chart with bootstrap error bars for each mechanism by kappa."""
    kappas = np.sort(np.unique(data["kappa"]))
    n_kappas = len(kappas)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(n_kappas)

    for offset, m in enumerate(MECHANISMS):
        means = []
        err_lo = []
        err_hi = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[m][mask]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                point, lo, hi = bootstrap_ci(valid, n_resamples=2000, rng_seed=42)
                means.append(point)
                err_lo.append(point - lo)
                err_hi.append(hi - point)
            else:
                means.append(0)
                err_lo.append(0)
                err_hi.append(0)

        ax.bar(x + offset * bar_width, means, bar_width,
               color=_mech_color(m), alpha=0.7, label=MECHANISM_NICE[m],
               yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Kappa")
    ax.set_ylabel("Mean Effect (with 95% CI)")
    ax.set_title("Mechanism Effects by Kappa Level")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, output_dir / "c17_grouped_bar_effects.png")


def c18_heatmap_grid_3x3(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """3x3 grid: 3 mechanisms x 3 parameter pairs."""
    param_pairs = [
        ("kappa", "concentration", "kappa x c"),
        ("kappa", "mu", "kappa x mu"),
        ("kappa", "outside_mid_ratio", "kappa x rho"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for row_idx, m in enumerate(MECHANISMS):
        for col_idx, (row_col, col_col, pair_label) in enumerate(param_pairs):
            ax = axes[row_idx, col_idx]
            row_vals, col_vals, matrix = pivot_for_heatmap(data, row_col, col_col, m)

            vmax = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 0.1
            im = ax.imshow(
                matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                origin="lower", interpolation="nearest",
            )
            ax.set_xticks(range(len(col_vals)))
            ax.set_xticklabels([f"{v:.2g}" for v in col_vals], fontsize=7)
            ax.set_yticks(range(len(row_vals)))
            ax.set_yticklabels([f"{v:.2g}" for v in row_vals], fontsize=7)

            if row_idx == 2:
                ax.set_xlabel(col_col)
            if col_idx == 0:
                ax.set_ylabel(f"{MECHANISM_NICE[m]}\nkappa")

            ax.set_title(f"{MECHANISM_NICE[m]} ({pair_label})", fontsize=9)

            # Annotate cells
            for i in range(len(row_vals)):
                for j in range(len(col_vals)):
                    val = matrix[i, j]
                    if np.isfinite(val):
                        text_color = "white" if abs(val) > 0.6 * vmax else "black"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=6, color=text_color)

            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Effect Heatmaps: 3 Mechanisms x 3 Parameter Pairs", fontsize=16, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "c18_heatmap_grid_3x3.png")


def c19_correlation_matrix(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Correlation matrix between mechanism effects."""
    cols = MECHANISMS
    labels = [MECHANISM_NICE[m] for m in cols]

    # Build matrix
    n_cols = len(cols)
    corr = np.full((n_cols, n_cols), np.nan)
    for i in range(n_cols):
        for j in range(n_cols):
            vi = data[cols[i]]
            vj = data[cols[j]]
            mask = np.isfinite(vi) & np.isfinite(vj)
            if np.sum(mask) > 2:
                corr[i, j] = np.corrcoef(vi[mask], vj[mask])[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n_cols))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate
    for i in range(n_cols):
        for j in range(n_cols):
            val = corr[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation Between Mechanism Effects")
    fig.tight_layout()
    _save(fig, output_dir / "c19_correlation_matrix.png")


# ===================================================================
# Section 7: Network Structure & Contagion
# ===================================================================

def _available_arms(net_data: dict[str, np.ndarray]) -> list[str]:
    """Return arm names present in net_data, in a stable display order."""
    order = ["passive", "active", "nbfi", "idle", "lend"]
    present = set(np.unique(net_data["arm"]))
    return [a for a in order if a in present]


def _arm_color(arm: str) -> str:
    return ARM_COLORS.get(arm, "#333333")


def _arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def c20_default_classification_by_arm(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Stacked bar: mean n_primary vs n_secondary defaults per arm."""
    arms = _available_arms(net_data)
    primary_means = []
    secondary_means = []
    labels = []

    for arm in arms:
        mask = net_data["arm"] == arm
        primary = net_data["n_primary"][mask]
        secondary = net_data["n_secondary"][mask]
        primary_means.append(np.nanmean(primary) if len(primary) > 0 else 0)
        secondary_means.append(np.nanmean(secondary) if len(secondary) > 0 else 0)
        labels.append(_arm_label(arm))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(arms))
    bar_width = 0.5

    ax.bar(x, primary_means, bar_width, label="Primary Defaults", color="#3498db")
    ax.bar(x, secondary_means, bar_width, bottom=primary_means,
           label="Secondary (Contagion) Defaults", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Number of Defaults")
    ax.set_title("Default Classification by Arm")
    ax.legend()
    _save(fig, output_dir / "c20_default_classification_by_arm.png")


def c21_cascade_fraction_vs_kappa(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Line chart: cascade_fraction vs kappa by arm."""
    arms = _available_arms(net_data)
    kappas = np.sort(np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for arm in arms:
        means = []
        for k in kappas:
            mask = (net_data["arm"] == arm) & (net_data["kappa"] == k)
            vals = net_data["cascade_fraction"][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=_arm_color(arm), label=_arm_label(arm),
                linewidth=2, markersize=5)

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Cascade Fraction")
    ax.set_title("Cascade Fraction vs Kappa by Arm")
    ax.set_xscale("log")
    ax.legend()
    _save(fig, output_dir / "c21_cascade_fraction_vs_kappa.png")


def c22_time_to_contagion_by_arm(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Box plot: distribution of time_to_contagion for each arm."""
    arms = _available_arms(net_data)
    box_data = []
    labels = []
    colors = []

    for arm in arms:
        mask = net_data["arm"] == arm
        vals = net_data["time_to_contagion"][mask]
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            box_data.append(valid)
            labels.append(_arm_label(arm))
            colors.append(_arm_color(arm))

    if not box_data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    showfliers=True, flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Time to First Contagion (days)")
    ax.set_title("Time to Contagion Distribution by Arm")
    _save(fig, output_dir / "c22_time_to_contagion_by_arm.png")


def c23_betweenness_gini_vs_kappa(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Line chart: gini_betweenness vs kappa by arm."""
    arms = _available_arms(net_data)
    kappas = np.sort(np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for arm in arms:
        means = []
        for k in kappas:
            mask = (net_data["arm"] == arm) & (net_data["kappa"] == k)
            vals = net_data["gini_betweenness"][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=_arm_color(arm), label=_arm_label(arm),
                linewidth=2, markersize=5)

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Gini of Betweenness Centrality")
    ax.set_title("Network Concentration (Betweenness Gini) vs Kappa")
    ax.set_xscale("log")
    ax.legend()
    _save(fig, output_dir / "c23_betweenness_gini_vs_kappa.png")


def c24_systemic_importance_by_arm(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Grouped bar: max_systemic score per arm at each kappa level."""
    arms = _available_arms(net_data)
    kappas = np.sort(np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])]))
    n_kappas = len(kappas)

    fig, ax = plt.subplots(figsize=(12, 5))
    n_arms = len(arms)
    bar_width = 0.8 / max(n_arms, 1)

    for offset, arm in enumerate(arms):
        means = []
        for k in kappas:
            mask = (net_data["arm"] == arm) & (net_data["kappa"] == k)
            vals = net_data["max_systemic"][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else 0)
        x = np.arange(n_kappas)
        ax.bar(x + offset * bar_width, means, bar_width,
               color=_arm_color(arm), alpha=0.8, label=_arm_label(arm))

    ax.set_xticks(np.arange(n_kappas) + bar_width * (n_arms - 1) / 2)
    ax.set_xticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Kappa")
    ax.set_ylabel("Mean Max Systemic Importance Score")
    ax.set_title("Systemic Importance by Arm and Kappa")
    ax.legend()
    _save(fig, output_dir / "c24_systemic_importance_by_arm.png")


def c25_funding_mix_by_arm(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Stacked bar: mean fraction of each funding source per arm."""
    arms = _available_arms(net_data)
    funding_cols = [
        ("funding_settlement_received", "Settlement Received"),
        ("funding_ticket_sale", "Ticket Sale"),
        ("funding_loan_received", "Bank Loan"),
        ("funding_cb_loan", "CB Loan"),
        ("funding_nbfi_loan", "NBFI Loan"),
        ("funding_deposit_interest", "Deposit Interest"),
        ("funding_deposit", "Deposit"),
    ]

    # Compute mean fraction per arm for each source
    arm_means: dict[str, list[float]] = {arm: [] for arm in arms}
    valid_sources: list[tuple[str, str]] = []

    for col, label in funding_cols:
        if col not in net_data:
            continue
        # Check if any arm has >1% mean
        any_significant = False
        temp_means = {}
        for arm in arms:
            mask = net_data["arm"] == arm
            vals = net_data[col][mask]
            valid = vals[np.isfinite(vals)]
            m = np.mean(valid) if len(valid) > 0 else 0.0
            temp_means[arm] = m
            if m > 0.01:
                any_significant = True

        if any_significant:
            valid_sources.append((col, label))
            for arm in arms:
                arm_means[arm].append(temp_means[arm])

    if not valid_sources:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(arms))
    bar_width = 0.5

    # Stacked bars
    source_colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c", "#f1c40f"]
    bottoms = np.zeros(len(arms))

    for i, (col, label) in enumerate(valid_sources):
        heights = np.array([arm_means[arm][i] for arm in arms])
        color = source_colors[i % len(source_colors)]
        ax.bar(x, heights, bar_width, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([_arm_label(a) for a in arms], fontsize=10)
    ax.set_ylabel("Mean Fraction of Total Funding")
    ax.set_title("Funding Mix by Arm")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, min(1.05, bottoms.max() * 1.1) if bottoms.max() > 0 else 1.05)
    _save(fig, output_dir / "c25_funding_mix_by_arm.png")


def c26_net_credit_impulse_vs_kappa(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Line chart: net_credit_impulse vs kappa by arm."""
    arms = _available_arms(net_data)
    kappas = np.sort(np.unique(net_data["kappa"][np.isfinite(net_data["kappa"])]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for arm in arms:
        means = []
        for k in kappas:
            mask = (net_data["arm"] == arm) & (net_data["kappa"] == k)
            vals = net_data["net_credit_impulse"][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=_arm_color(arm), label=_arm_label(arm),
                linewidth=2, markersize=5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Net Credit Impulse")
    ax.set_title("Net Credit Impulse vs Kappa by Arm")
    ax.set_xscale("log")
    ax.legend()
    _save(fig, output_dir / "c26_net_credit_impulse_vs_kappa.png")


def c27_cascade_depth_distribution(
    net_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Histogram: max_cascade_depth distribution by arm (overlaid)."""
    arms = _available_arms(net_data)

    fig, ax = plt.subplots(figsize=(9, 5))

    all_depths = net_data["max_cascade_depth"]
    valid_all = all_depths[np.isfinite(all_depths)]
    if len(valid_all) == 0:
        plt.close(fig)
        return
    max_depth = int(np.max(valid_all))
    bins = np.arange(-0.5, max_depth + 1.5, 1)

    for arm in arms:
        mask = net_data["arm"] == arm
        vals = net_data["max_cascade_depth"][mask]
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            ax.hist(valid, bins=bins, alpha=0.4, color=_arm_color(arm),
                    label=_arm_label(arm), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Max Cascade Depth")
    ax.set_ylabel("Count")
    ax.set_title("Cascade Depth Distribution by Arm")
    ax.legend()
    ax.set_xticks(range(0, max_depth + 1))
    _save(fig, output_dir / "c27_cascade_depth_distribution.png")


# ===================================================================
# Section 8: Temporal Dynamics
# ===================================================================

def _closest_kappa(available: list[float], target: float) -> float:
    """Return the kappa in 'available' closest to 'target'."""
    if not available:
        return target
    return min(available, key=lambda k: abs(k - target))


def c28_contagion_timeline(
    ts_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """2x2 subplots: cumulative defaults (stacked area) by day by arm for 4 kappas."""
    target_kappas = [0.25, 0.5, 1.0, 2.0]

    # Collect all available arms and kappas
    all_arms = [a for a in ["passive", "active", "nbfi", "idle", "lend"] if a in ts_agg]
    if not all_arms:
        return
    all_kappas = sorted(set(
        k for arm in all_arms for k in ts_agg[arm]
    ))
    if not all_kappas:
        return

    resolved_kappas = [_closest_kappa(all_kappas, t) for t in target_kappas]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()

    for ax, kappa in zip(axes_flat, resolved_kappas):
        # Find max day across arms for this kappa
        max_day = 0
        for arm in all_arms:
            if kappa in ts_agg[arm]:
                max_day = max(max_day, max(ts_agg[arm][kappa].keys(), default=0))

        days = list(range(max_day + 1))

        for arm in all_arms:
            if kappa not in ts_agg.get(arm, {}):
                continue
            kappa_data = ts_agg[arm][kappa]
            cum_primary = [kappa_data.get(d, {}).get("cum_primary_defaults", 0) for d in days]
            cum_secondary = [kappa_data.get(d, {}).get("cum_secondary_defaults", 0) for d in days]
            cum_total = [p + s for p, s in zip(cum_primary, cum_secondary)]

            ax.plot(days, cum_total, "-", color=_arm_color(arm),
                    label=_arm_label(arm), linewidth=2)

        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative Defaults")
        ax.set_title(f"kappa = {kappa:.2g}")
        ax.legend(fontsize=7)

    fig.suptitle("Contagion Timeline: Cumulative Defaults by Arm", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "c28_contagion_timeline.png")


def c29_credit_flow_timeline(
    ts_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: credit_created and credit_destroyed over time per arm at kappa=0.5."""
    all_arms = [a for a in ["passive", "active", "nbfi", "idle", "lend"] if a in ts_agg]
    if not all_arms:
        return
    all_kappas = sorted(set(k for arm in all_arms for k in ts_agg[arm]))
    kappa = _closest_kappa(all_kappas, 0.5)

    n_arms = sum(1 for a in all_arms if kappa in ts_agg.get(a, {}))
    if n_arms == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for arm in all_arms:
        if kappa not in ts_agg.get(arm, {}):
            continue
        kappa_data = ts_agg[arm][kappa]
        days = sorted(kappa_data.keys())
        created = [kappa_data[d].get("credit_created", 0) for d in days]
        destroyed = [kappa_data[d].get("credit_destroyed", 0) for d in days]

        color = _arm_color(arm)
        ax.plot(days, created, "-", color=color, linewidth=2,
                label=f"{_arm_label(arm)} created")
        ax.plot(days, destroyed, "--", color=color, linewidth=1.5, alpha=0.7,
                label=f"{_arm_label(arm)} destroyed")

    ax.set_xlabel("Day")
    ax.set_ylabel("Credit Flow")
    ax.set_title(f"Credit Created vs Destroyed Over Time (kappa={kappa:.2g})")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, output_dir / "c29_credit_flow_timeline.png")


def c30_active_obligations_timeline(
    ts_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: active_obligations over time by arm at kappa=0.5."""
    all_arms = [a for a in ["passive", "active", "nbfi", "idle", "lend"] if a in ts_agg]
    if not all_arms:
        return
    all_kappas = sorted(set(k for arm in all_arms for k in ts_agg[arm]))
    kappa = _closest_kappa(all_kappas, 0.5)

    fig, ax = plt.subplots(figsize=(10, 5))

    for arm in all_arms:
        if kappa not in ts_agg.get(arm, {}):
            continue
        kappa_data = ts_agg[arm][kappa]
        days = sorted(kappa_data.keys())
        active = [kappa_data[d].get("active_obligations", 0) for d in days]

        ax.plot(days, active, "o-", color=_arm_color(arm), label=_arm_label(arm),
                linewidth=2, markersize=4)

    ax.set_xlabel("Day")
    ax.set_ylabel("Active Obligations")
    ax.set_title(f"Active Obligations Over Time (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c30_active_obligations_timeline.png")


def c31_mechanism_effect_timeline(
    ts_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: per-day mechanism effect at kappa=0.5.

    For each day: (passive cumulative defaults) - (arm cumulative defaults).
    Shows when each mechanism helps most.
    """
    all_arms = [a for a in ["passive", "active", "nbfi", "idle", "lend"] if a in ts_agg]
    if "passive" not in all_arms:
        return
    all_kappas = sorted(set(k for arm in all_arms for k in ts_agg[arm]))
    kappa = _closest_kappa(all_kappas, 0.5)

    if kappa not in ts_agg.get("passive", {}):
        return

    passive_data = ts_agg["passive"][kappa]
    days = sorted(passive_data.keys())
    passive_cum = [
        passive_data[d].get("cum_primary_defaults", 0)
        + passive_data[d].get("cum_secondary_defaults", 0)
        for d in days
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for arm in all_arms:
        if arm == "passive":
            continue
        if kappa not in ts_agg.get(arm, {}):
            continue
        arm_data = ts_agg[arm][kappa]
        arm_cum = [
            arm_data.get(d, {}).get("cum_primary_defaults", 0)
            + arm_data.get(d, {}).get("cum_secondary_defaults", 0)
            for d in days
        ]
        effect = [p - a for p, a in zip(passive_cum, arm_cum)]
        ax.plot(days, effect, "o-", color=_arm_color(arm), label=_arm_label(arm),
                linewidth=2, markersize=4)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Day")
    ax.set_ylabel("Default Reduction vs Passive")
    ax.set_title(f"Mechanism Effect Over Time (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c31_mechanism_effect_timeline.png")


# ===================================================================
# Section 9: Mechanism Internals
# ===================================================================

def c32_dealer_trade_volume(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Daily dealer buy/sell counts over time, multi-kappa subplots."""
    # Only active arm has dealer data
    if "active" not in mech_agg:
        return

    active = mech_agg["active"]
    all_kappas = sorted(active.keys())
    if not all_kappas:
        return

    target_kappas = [0.25, 0.5, 1.0]
    resolved = [_closest_kappa(all_kappas, t) for t in target_kappas]
    # Deduplicate while preserving order
    seen = set()
    unique_kappas = []
    for k in resolved:
        if k not in seen:
            seen.add(k)
            unique_kappas.append(k)

    n = len(unique_kappas)
    if n == 0:
        return

    # Use 2x2 if 3+ kappas, otherwise 1xN
    if n >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.ravel()
    else:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        axes_flat = [axes] if n == 1 else list(axes)

    for idx, kappa in enumerate(unique_kappas):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        kappa_data = active.get(kappa, {})
        days = sorted(kappa_data.keys())
        if not days:
            ax.set_visible(False)
            continue

        buys = [kappa_data[d].get("n_buys", 0) for d in days]
        sells = [kappa_data[d].get("n_sells", 0) for d in days]
        rejected = [kappa_data[d].get("n_sell_rejected", 0) for d in days]

        ax.plot(days, buys, "-", color="#1f77b4", linewidth=2, label="Buys")
        ax.plot(days, sells, "--", color="#ff7f0e", linewidth=2, label="Sells")
        ax.plot(days, rejected, ":", color="#d62728", linewidth=1.5, alpha=0.5,
                label="Sell Rejected")

        ax.set_xlabel("Day")
        ax.set_ylabel("Count")
        ax.set_title(f"kappa = {kappa:.2g}")
        ax.legend(fontsize=7)

    # Hide unused subplot(s) in 2x2 layout
    if n >= 3:
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

    fig.suptitle("Dealer Trade Volume Over Time", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "c32_dealer_trade_volume.png")


def c33_dealer_prices_by_bucket(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart of mean prices per bucket over time at kappa~0.5 (active arm)."""
    if "active" not in mech_agg:
        return

    active = mech_agg["active"]
    all_kappas = sorted(active.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = active.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    price_short = [kappa_data[d].get("mean_price_short", 0) for d in days]
    price_mid = [kappa_data[d].get("mean_price_mid", 0) for d in days]
    price_long = [kappa_data[d].get("mean_price_long", 0) for d in days]

    # Skip if all zero (no price data)
    if all(v == 0 for v in price_short + price_mid + price_long):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days, price_short, "-", color="#1f77b4", linewidth=2, label="Short")
    ax.plot(days, price_mid, "-", color="#ff7f0e", linewidth=2, label="Mid")
    ax.plot(days, price_long, "-", color="#2ca02c", linewidth=2, label="Long")

    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Price")
    ax.set_title(f"Dealer Prices by Maturity Bucket (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c33_dealer_prices_by_bucket.png")


def c34_liquidity_vs_earning(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Stacked bar: liquidity-driven vs earning-driven trades per day at kappa~0.5."""
    if "active" not in mech_agg:
        return

    active = mech_agg["active"]
    all_kappas = sorted(active.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = active.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    liquidity = [kappa_data[d].get("n_liquidity_driven", 0) for d in days]
    earning = [kappa_data[d].get("n_earning_driven", 0) for d in days]

    if all(v == 0 for v in liquidity + earning):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(days, liquidity, color="#1f77b4", alpha=0.85, label="Liquidity-Driven")
    ax.bar(days, earning, bottom=liquidity, color="#ff7f0e", alpha=0.85,
           label="Earning-Driven")

    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Trades")
    ax.set_title(f"Trade Motivation: Liquidity vs Earning (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c34_liquidity_vs_earning.png")


def c35_bank_loan_activity(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """2x2 subplots: bank loan activity across 4 kappas (lend arm)."""
    if "lend" not in mech_agg:
        return

    lend = mech_agg["lend"]
    all_kappas = sorted(lend.keys())
    if not all_kappas:
        return

    target_kappas = [0.25, 0.5, 1.0, 2.0]
    resolved = [_closest_kappa(all_kappas, t) for t in target_kappas]
    # Deduplicate
    seen = set()
    unique_kappas = []
    for k in resolved:
        if k not in seen:
            seen.add(k)
            unique_kappas.append(k)

    n = len(unique_kappas)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()

    for idx, kappa in enumerate(unique_kappas):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        kappa_data = lend.get(kappa, {})
        days = sorted(kappa_data.keys())
        if not days:
            ax.set_visible(False)
            continue

        issued = [kappa_data[d].get("n_loans_issued", 0) for d in days]
        defaults = [kappa_data[d].get("n_loan_defaults", 0) for d in days]
        repaid = [kappa_data[d].get("n_loan_repaid", 0) for d in days]

        ax.plot(days, issued, "-", color="#1f77b4", linewidth=2, label="Issued")
        ax.plot(days, defaults, "--", color="#e74c3c", linewidth=2, label="Defaults")
        ax.plot(days, repaid, ":", color="#2ecc71", linewidth=2, label="Repaid")

        ax.set_xlabel("Day")
        ax.set_ylabel("Count")
        ax.set_title(f"kappa = {kappa:.2g}")
        ax.legend(fontsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Bank Loan Activity Over Time", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "c35_bank_loan_activity.png")


def c36_bank_lending_rate(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart of mean loan rate over time at kappa~0.5 (lend arm)."""
    if "lend" not in mech_agg:
        return

    lend = mech_agg["lend"]
    all_kappas = sorted(lend.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = lend.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    rates = [kappa_data[d].get("mean_loan_rate", 0) for d in days]

    if all(v == 0 for v in rates):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days, rates, "o-", color=BANK_BLUE, linewidth=2, markersize=4)

    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Loan Rate")
    ax.set_title(f"Bank Lending Rate Over Time (kappa={kappa:.2g})")
    _save(fig, output_dir / "c36_bank_lending_rate.png")


def c37_bank_loan_volume_vs_defaults(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: loan volume issued vs default volume over time at kappa~0.5."""
    if "lend" not in mech_agg:
        return

    lend = mech_agg["lend"]
    all_kappas = sorted(lend.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = lend.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    issued_vol = [kappa_data[d].get("loan_volume_issued", 0) for d in days]
    default_vol = [kappa_data[d].get("default_volume", 0) for d in days]

    if all(v == 0 for v in issued_vol + default_vol):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(days, issued_vol, "-", color=BANK_BLUE, linewidth=2,
            label="Loan Volume Issued")
    ax.plot(days, default_vol, "--", color="#e74c3c", linewidth=2,
            label="Default Volume")

    ax.set_xlabel("Day")
    ax.set_ylabel("Volume")
    ax.set_title(f"Loan Volume vs Default Volume (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c37_bank_loan_volume_vs_defaults.png")


def c38_interbank_volume(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: interbank cleared volume over time, multi-kappa, by arm."""
    bank_arms = [a for a in ["idle", "lend"] if a in mech_agg]
    if not bank_arms:
        return

    all_kappas = sorted(set(
        k for arm in bank_arms for k in mech_agg[arm]
    ))
    if not all_kappas:
        return

    target_kappas = [0.25, 0.5, 1.0, 2.0]
    resolved = [_closest_kappa(all_kappas, t) for t in target_kappas]
    # Deduplicate
    seen = set()
    unique_kappas = []
    for k in resolved:
        if k not in seen:
            seen.add(k)
            unique_kappas.append(k)

    fig, ax = plt.subplots(figsize=(10, 5))

    kappa_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for arm in bank_arms:
        for ki, kappa in enumerate(unique_kappas):
            kappa_data = mech_agg[arm].get(kappa, {})
            days = sorted(kappa_data.keys())
            if not days:
                continue
            vols = [kappa_data[d].get("interbank_cleared_volume", 0) for d in days]
            linestyle = "-" if arm == "lend" else "--"
            color = kappa_colors[ki % len(kappa_colors)]
            ax.plot(days, vols, linestyle, color=color, linewidth=2,
                    label=f"{_arm_label(arm)} k={kappa:.2g}")

    ax.set_xlabel("Day")
    ax.set_ylabel("Interbank Cleared Volume")
    ax.set_title("Interbank Volume Over Time")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, output_dir / "c38_interbank_volume.png")


def c39_interbank_rate(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Line chart: auction rate over time at kappa~0.5 for idle and lend arms."""
    bank_arms = [a for a in ["idle", "lend"] if a in mech_agg]
    if not bank_arms:
        return

    all_kappas = sorted(set(
        k for arm in bank_arms for k in mech_agg[arm]
    ))
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)

    fig, ax = plt.subplots(figsize=(10, 5))

    for arm in bank_arms:
        kappa_data = mech_agg[arm].get(kappa, {})
        days = sorted(kappa_data.keys())
        if not days:
            continue
        rates = [kappa_data[d].get("auction_rate", 0) for d in days]
        # Filter out zeros (no auction that day)
        ax.plot(days, rates, "o-", color=_arm_color(arm), linewidth=2,
                markersize=4, label=_arm_label(arm))

    ax.set_xlabel("Day")
    ax.set_ylabel("Auction Rate")
    ax.set_title(f"Interbank Auction Rate Over Time (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c39_interbank_rate.png")


def c40_cb_facility_usage(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Bar chart: CB loan volume per day at kappa~0.5 (lend arm)."""
    if "lend" not in mech_agg:
        return

    lend = mech_agg["lend"]
    all_kappas = sorted(lend.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = lend.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    created = [kappa_data[d].get("cb_loans_created_volume", 0) for d in days]

    if all(v == 0 for v in created):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(days, created, color="#e74c3c", alpha=0.7, label="CB Loans Created")

    # Detect CB freeze day (first day where volume drops to 0 after being positive)
    freeze_day = None
    was_positive = False
    for i, (d, v) in enumerate(zip(days, created)):
        if v > 0:
            was_positive = True
        elif was_positive and v == 0:
            freeze_day = d
            break

    if freeze_day is not None:
        ax.axvline(freeze_day, color="black", linewidth=1.5, linestyle="--",
                   alpha=0.7, label=f"CB Freeze (day {freeze_day})")

    ax.set_xlabel("Day")
    ax.set_ylabel("Volume")
    ax.set_title(f"Central Bank Facility Usage (kappa={kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c40_cb_facility_usage.png")


def c41_payment_channels(
    mech_agg: dict[str, dict[float, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Stacked area chart: payment channels over time at kappa~0.5 (lend arm)."""
    if "lend" not in mech_agg:
        return

    lend = mech_agg["lend"]
    all_kappas = sorted(lend.keys())
    if not all_kappas:
        return

    kappa = _closest_kappa(all_kappas, 0.5)
    kappa_data = lend.get(kappa, {})
    days = sorted(kappa_data.keys())
    if not days:
        return

    intrabank = [kappa_data[d].get("intrabank_volume", 0) for d in days]
    client = [kappa_data[d].get("client_payment_volume", 0) for d in days]
    reserves = [kappa_data[d].get("reserves_transferred_volume", 0) for d in days]

    if all(v == 0 for v in intrabank + client + reserves):
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(
        days,
        intrabank,
        client,
        reserves,
        labels=["Intrabank", "Client Payment", "Reserves Transfer"],
        colors=["#3498db", "#e67e22", "#2ecc71"],
        alpha=0.8,
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Volume")
    ax.set_title(f"Payment Channels Over Time (kappa={kappa:.2g})")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c41_payment_channels.png")


# ===================================================================
# Section 11: Loss Analysis
# ===================================================================

# Arms for system loss charts (NBFI sweep arms + bank sweep arms)
_LOSS_ARMS_NBFI = [
    ("system_loss_pct_passive", "Passive", PASSIVE_GREY),
    ("system_loss_pct_active", "Active (Dealer)", DEALER_RED),
    ("system_loss_pct_lender", "NBFI Lender", NBFI_GREEN),
]

_LOSS_ARMS_BANK = [
    ("bank_system_loss_pct_idle", "Bank Idle", "#7f8c8d"),
    ("bank_system_loss_pct_lend", "Bank Lend", BANK_BLUE),
]

_LOSS_ARMS_ALL = _LOSS_ARMS_NBFI + _LOSS_ARMS_BANK


def c42_system_loss_vs_kappa(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Multi-arm line chart: system_loss_pct vs kappa for each arm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    kappas = np.sort(np.unique(data["kappa"]))

    for col, label, color in _LOSS_ARMS_ALL:
        if col not in data:
            continue
        means = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("System Loss Rate (fraction of initial debt)")
    ax.set_title("System Loss Rate vs Kappa by Arm")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c42_system_loss_vs_kappa.png")


def c43_loss_decomposition(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Stacked bar: payable_default_loss + intermediary_loss per arm at kappa~0.5."""
    kappas = np.sort(np.unique(data["kappa"]))
    target_kappa = min(kappas, key=lambda k: abs(k - 0.5))
    mask = data["kappa"] == target_kappa

    # Define arms with their payable and intermediary loss columns
    arm_defs = [
        ("Passive", "payable_default_loss_passive", "intermediary_loss_passive"),
        ("Dealer", "payable_default_loss_active", "intermediary_loss_active"),
        ("NBFI Lender", "payable_default_loss_lender", "intermediary_loss_lender"),
        ("Bank Idle", None, "bank_intermediary_loss_idle"),
        ("Bank Lend", None, "bank_intermediary_loss_lend"),
    ]

    labels = []
    payable_vals = []
    intermediary_vals = []

    for label, pay_col, int_col in arm_defs:
        # Get payable default loss (not all arms have it in the unified data)
        if pay_col and pay_col in data:
            pay_v = data[pay_col][mask]
            pay_valid = pay_v[np.isfinite(pay_v)]
            pay_mean = np.mean(pay_valid) if len(pay_valid) > 0 else 0.0
        else:
            # For bank arms, compute from total_loss - intermediary_loss
            total_col = f"bank_total_loss_{label.split()[-1].lower()}" if "Bank" in label else None
            if total_col and total_col in data and int_col in data:
                total_v = data[total_col][mask]
                int_v = data[int_col][mask]
                total_valid_mask = np.isfinite(total_v) & np.isfinite(int_v)
                if np.any(total_valid_mask):
                    pay_mean = np.mean(total_v[total_valid_mask] - int_v[total_valid_mask])
                    pay_mean = max(pay_mean, 0.0)
                else:
                    pay_mean = 0.0
            else:
                pay_mean = 0.0

        if int_col and int_col in data:
            int_v = data[int_col][mask]
            int_valid = int_v[np.isfinite(int_v)]
            int_mean = np.mean(int_valid) if len(int_valid) > 0 else 0.0
        else:
            int_mean = 0.0

        # Only include arms with some data
        if pay_mean > 0 or int_mean > 0:
            labels.append(label)
            payable_vals.append(pay_mean)
            intermediary_vals.append(int_mean)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bar_width = 0.5

    ax.bar(x, payable_vals, bar_width, label="Payable Default Loss",
           color="#3498db", alpha=0.85)
    ax.bar(x, intermediary_vals, bar_width, bottom=payable_vals,
           label="Intermediary Loss", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Loss Amount")
    ax.set_title(f"Loss Decomposition by Arm (kappa={target_kappa:.2g})")
    ax.legend()
    _save(fig, output_dir / "c43_loss_decomposition.png")


def c44_system_loss_heatmap(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Heatmap: system_loss_pct by mechanism (arm) and kappa."""
    arm_defs = [
        ("system_loss_pct_passive", "Passive"),
        ("system_loss_pct_active", "Dealer"),
        ("system_loss_pct_lender", "NBFI Lender"),
        ("bank_system_loss_pct_idle", "Bank Idle"),
        ("bank_system_loss_pct_lend", "Bank Lend"),
    ]

    # Filter to arms present in data
    valid_arms = [(col, label) for col, label in arm_defs if col in data]
    if not valid_arms:
        return

    kappas = np.sort(np.unique(data["kappa"]))
    n_kappas = len(kappas)
    n_arms = len(valid_arms)

    matrix = np.full((n_kappas, n_arms), np.nan)
    for j, (col, _label) in enumerate(valid_arms):
        for i, k in enumerate(kappas):
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                matrix[i, j] = np.mean(valid)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Use sequential colormap (loss is always >= 0)
    vmax = np.nanmax(matrix) if np.any(np.isfinite(matrix)) else 0.1
    im = ax.imshow(
        matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
        origin="lower", interpolation="nearest",
    )

    ax.set_xticks(range(n_arms))
    ax.set_xticklabels([label for _, label in valid_arms], fontsize=10)
    ax.set_yticks(range(n_kappas))
    ax.set_yticklabels([f"{k:.2g}" for k in kappas], fontsize=9)
    ax.set_xlabel("Mechanism (Arm)")
    ax.set_ylabel("Kappa")

    # Annotate cells
    for i in range(n_kappas):
        for j in range(n_arms):
            val = matrix[i, j]
            if np.isfinite(val):
                text_color = "white" if val > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="System Loss Rate")
    ax.set_title("System Loss Rate by Arm and Kappa")
    fig.tight_layout()
    _save(fig, output_dir / "c44_system_loss_heatmap.png")


def c45_loss_treatment_effects(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Line chart: system loss treatment effects vs kappa."""
    effect_defs = [
        ("system_loss_trading_effect", "Dealer Trading", DEALER_RED),
        ("system_loss_lending_effect", "NBFI Lending", NBFI_GREEN),
        ("bank_system_loss_bank_lending_effect", "Bank Lending", BANK_BLUE),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    kappas = np.sort(np.unique(data["kappa"]))
    any_plotted = False

    for col, label, color in effect_defs:
        if col not in data:
            continue
        means = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        ax.plot(kappas, means, "o-", color=color, label=label, linewidth=2, markersize=5)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("System Loss Effect (positive = loss reduction)")
    ax.set_title("Loss Treatment Effects vs Kappa")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c45_loss_treatment_effects.png")


def c46_lgd_vs_kappa(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Line chart: Loss Given Default (LGD) vs kappa, filtered to kappa <= 2."""
    lgd_defs = [
        ("lgd_passive", "Passive", PASSIVE_GREY),
        ("lgd_active", "Active (Dealer)", DEALER_RED),
        ("lgd_lender", "NBFI Lender", NBFI_GREEN),
        ("lgd_idle", "Bank Idle", "#7f8c8d"),
        ("lgd_lend", "Bank Lend", BANK_BLUE),
    ]

    kappas_all = np.sort(np.unique(data["kappa"]))
    kappas = kappas_all[kappas_all <= 2.0]
    if len(kappas) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for col, label, color in lgd_defs:
        if col not in data:
            continue
        medians = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            medians.append(np.median(valid) if len(valid) > 0 else np.nan)
        if any(np.isfinite(m) for m in medians):
            ax.plot(kappas, medians, "o-", color=color, label=label, linewidth=2, markersize=5)
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Loss Given Default (LGD) — median")
    ax.set_title("Loss Given Default vs Kappa (kappa <= 2, median)")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0, top=min(5.0, ax.get_ylim()[1]))  # Cap Y for readability
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c46_lgd_vs_kappa.png")


def c47_loss_per_default(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Line chart: loss per default event vs kappa for each arm."""
    lpd_defs = [
        ("loss_per_default_passive", "Passive", PASSIVE_GREY),
        ("loss_per_default_active", "Active (Dealer)", DEALER_RED),
        ("loss_per_default_lender", "NBFI Lender", NBFI_GREEN),
    ]

    # Bank loss_per_default is not derived by add_derived_columns because
    # the bank n_defaults columns are prefixed bank_n_defaults_{arm}.
    # Compute on the fly if the raw columns exist.
    for bank_arm, label, color in [("idle", "Bank Idle", "#7f8c8d"), ("lend", "Bank Lend", BANK_BLUE)]:
        loss_col = f"bank_total_loss_{bank_arm}"
        ndefault_col = f"bank_n_defaults_{bank_arm}"
        synthetic_col = f"loss_per_default_{bank_arm}"
        if loss_col in data and ndefault_col in data and synthetic_col not in data:
            import warnings
            num = np.asarray(data[loss_col], dtype=float)
            den = np.asarray(data[ndefault_col], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ratio = np.where((den > 0) & np.isfinite(den) & np.isfinite(num),
                                 num / den, np.nan)
            data[synthetic_col] = ratio
        lpd_defs.append((synthetic_col, label, color))

    kappas = np.sort(np.unique(data["kappa"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for col, label, color in lpd_defs:
        if col not in data:
            continue
        means = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        if any(np.isfinite(m) for m in means):
            ax.plot(kappas, means, "o-", color=color, label=label, linewidth=2, markersize=5)
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Loss per Default Event (amount)")
    ax.set_title("Loss per Default Event vs Kappa")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c47_loss_per_default.png")


def c48_loss_vs_delta_scatter(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Scatter: system_loss_pct vs delta for each arm, with 45-degree reference."""
    arm_defs = [
        ("delta_passive", "system_loss_pct_passive", "Passive", PASSIVE_GREY),
        ("delta_active", "system_loss_pct_active", "Active (Dealer)", DEALER_RED),
        ("delta_lender", "system_loss_pct_lender", "NBFI Lender", NBFI_GREEN),
        ("delta_idle", "bank_system_loss_pct_idle", "Bank Idle", "#7f8c8d"),
        ("delta_lend", "bank_system_loss_pct_lend", "Bank Lend", BANK_BLUE),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    any_plotted = False

    for delta_col, loss_col, label, color in arm_defs:
        if delta_col not in data or loss_col not in data:
            continue
        d_vals = data[delta_col]
        l_vals = data[loss_col]
        mask = np.isfinite(d_vals) & np.isfinite(l_vals)
        if np.sum(mask) == 0:
            continue
        ax.scatter(d_vals[mask], l_vals[mask], color=color, alpha=0.35,
                   s=15, label=label, edgecolors="none")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    # Add 45-degree reference line (system_loss = delta)
    ax.plot([0, 1.1], [0, 1.1], "k--", alpha=0.4, linewidth=1, label="loss = delta")

    ax.set_xlabel("Default Rate (delta)")
    ax.set_ylabel("System Loss Rate")
    ax.set_title("System Loss vs Default Rate by Arm")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.5)  # Allow slightly above 1 for intermediary-amplified loss
    ax.legend(markerscale=2, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    _save(fig, output_dir / "c48_loss_vs_delta_scatter.png")


def c49_loss_capital_ratio(data: dict[str, np.ndarray], output_dir: Path) -> None:
    """Line chart: intermediary loss/capital ratio vs kappa for each arm."""
    lcr_defs = [
        ("loss_capital_ratio_passive", "Passive", PASSIVE_GREY),
        ("loss_capital_ratio_active", "Active (Dealer)", DEALER_RED),
        ("loss_capital_ratio_lender", "NBFI Lender", NBFI_GREEN),
    ]

    kappas = np.sort(np.unique(data["kappa"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for col, label, color in lcr_defs:
        if col not in data:
            continue
        means = []
        for k in kappas:
            mask = data["kappa"] == k
            vals = data[col][mask]
            valid = vals[np.isfinite(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else np.nan)
        if any(np.isfinite(m) for m in means):
            ax.plot(kappas, means, "o-", color=color, label=label, linewidth=2, markersize=5)
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Kappa (log scale)")
    ax.set_ylabel("Loss / Capital Ratio")
    ax.set_title("Intermediary Loss-Capital Ratio vs Kappa")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "c49_loss_capital_ratio.png")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Load data and generate all 49 charts."""
    _setup_style()

    print("Loading sweep data...")
    print(f"  NBFI: {NBFI_CSV}")
    print(f"  Bank: {BANK_CSV}")

    data = load_unified_data(NBFI_CSV, BANK_CSV)
    data = add_derived_columns(data)
    n_rows = len(data["kappa"])
    print(f"  Loaded {n_rows} unified rows")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}\n")

    # Section 1: Descriptive
    print("Section 1: Descriptive")
    c01_mechanism_boxplots(data, output_dir)
    c02_delta_distributions(data, output_dir)
    c03_delta_vs_kappa(data, output_dir)

    # Section 2: Parameter Sensitivity
    print("\nSection 2: Parameter Sensitivity")
    c04_heatmap_trading_effect(data, output_dir)
    c05_heatmap_lending_effect(data, output_dir)
    c06_heatmap_bank_effect(data, output_dir)
    c07_marginal_effects(data, output_dir)
    c08_kappa_vs_delta_scatter(data, output_dir)

    # Section 3: Mechanism Comparison
    print("\nSection 3: Mechanism Comparison")
    c09_mechanism_ranking(data, output_dir)
    c10_phase_diagram(data, output_dir)
    c11_trading_vs_bank_scatter(data, output_dir)
    c12_additive_test(data, output_dir)

    # Section 4: Regression
    print("\nSection 4: Regression")
    c13_forest_plot(data, output_dir)
    c14_actual_vs_predicted(data, output_dir)
    c15_residual_diagnostics(data, output_dir)

    # Section 5: Statistical Tests
    print("\nSection 5: Statistical Tests")
    c16_bootstrap_distributions(data, output_dir)

    # Section 6: Cross-Sweep
    print("\nSection 6: Cross-Sweep")
    c17_grouped_bar_effects(data, output_dir)
    c18_heatmap_grid_3x3(data, output_dir)
    c19_correlation_matrix(data, output_dir)

    # Section 7: Network Structure & Contagion
    print("\nSection 7: Network Structure & Contagion")
    net_data = load_network_metrics(CACHE_DIR)
    if net_data:
        ts_data = load_network_timeseries(CACHE_DIR)
        ts_agg = aggregate_timeseries_by_arm_kappa(ts_data) if ts_data else {}

        c20_default_classification_by_arm(net_data, output_dir)
        c21_cascade_fraction_vs_kappa(net_data, output_dir)
        c22_time_to_contagion_by_arm(net_data, output_dir)
        c23_betweenness_gini_vs_kappa(net_data, output_dir)
        c24_systemic_importance_by_arm(net_data, output_dir)
        c25_funding_mix_by_arm(net_data, output_dir)
        c26_net_credit_impulse_vs_kappa(net_data, output_dir)
        c27_cascade_depth_distribution(net_data, output_dir)

        # Section 8: Temporal Dynamics
        print("\nSection 8: Temporal Dynamics")
        if ts_agg:
            c28_contagion_timeline(ts_agg, output_dir)
            c29_credit_flow_timeline(ts_agg, output_dir)
            c30_active_obligations_timeline(ts_agg, output_dir)
            c31_mechanism_effect_timeline(ts_agg, output_dir)
        else:
            print("  (no timeseries data found, skipping c28-c31)")
    else:
        print("  (no network data found, run extraction first)")

    # Section 9: Mechanism Internals
    print("\nSection 9: Mechanism Internals")
    mech_data = load_mechanism_timeseries(CACHE_DIR)
    if mech_data:
        mech_agg = aggregate_mechanism_timeseries(mech_data)
        if mech_agg:
            c32_dealer_trade_volume(mech_agg, output_dir)
            c33_dealer_prices_by_bucket(mech_agg, output_dir)
            c34_liquidity_vs_earning(mech_agg, output_dir)
            c35_bank_loan_activity(mech_agg, output_dir)
            c36_bank_lending_rate(mech_agg, output_dir)
            c37_bank_loan_volume_vs_defaults(mech_agg, output_dir)
            c38_interbank_volume(mech_agg, output_dir)
            c39_interbank_rate(mech_agg, output_dir)
            c40_cb_facility_usage(mech_agg, output_dir)
            c41_payment_channels(mech_agg, output_dir)
        else:
            print("  (no mechanism timeseries aggregated, skipping c32-c41)")
    else:
        print("  (no mechanism data found, run extraction first)")

    # Section 11: Loss Analysis
    print("\nSection 11: Loss Analysis")
    c42_system_loss_vs_kappa(data, output_dir)
    c43_loss_decomposition(data, output_dir)
    c44_system_loss_heatmap(data, output_dir)
    c45_loss_treatment_effects(data, output_dir)
    c46_lgd_vs_kappa(data, output_dir)
    c47_loss_per_default(data, output_dir)
    c48_loss_vs_delta_scatter(data, output_dir)
    c49_loss_capital_ratio(data, output_dir)

    print(f"\nDone. Charts saved to {output_dir}")


if __name__ == "__main__":
    main()
