"""Cross-sweep analysis engine for production NBFI and bank sweep results.

Provides data loading, OLS regression, bootstrap CIs, regime classification,
and grouped aggregation used by the notebook, PNG script, and HTML report.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
DEALER_RED = "#e74c3c"
BANK_BLUE = "#2980b9"
NBFI_GREEN = "#27ae60"
PASSIVE_GREY = "#95a5a6"
COMBINED_PURPLE = "#8e44ad"

MECHANISM_COLORS = {
    "trading_effect": DEALER_RED,
    "lending_effect": NBFI_GREEN,
    "bank_lending_effect": BANK_BLUE,
    "bank_effect_vs_passive": BANK_BLUE,
    "combined_effect": COMBINED_PURPLE,
}

MECHANISM_LABELS = {
    "trading_effect": "Dealer Trading",
    "lending_effect": "NBFI Lending",
    "bank_lending_effect": "Bank Lending (vs idle)",
    "bank_effect_vs_passive": "Bank Lending (vs passive)",
    "combined_effect": "Dealer + NBFI",
}

# Per-arm colors and labels (for network/temporal charts)
ARM_COLORS = {
    "passive": PASSIVE_GREY,
    "active": DEALER_RED,
    "nbfi": NBFI_GREEN,
    "idle": "#7f8c8d",
    "lend": BANK_BLUE,
}

ARM_LABELS = {
    "passive": "Passive",
    "active": "Active (Dealer)",
    "nbfi": "NBFI Lender",
    "idle": "Bank Idle",
    "lend": "Bank Lend",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_GRID_KEYS = ("kappa", "concentration", "mu", "outside_mid_ratio", "pool_scale", "seed")


def _parse_num(v: str) -> float | int | str:
    """Parse a CSV value to float/int/str."""
    if v == "":
        return np.nan
    try:
        f = float(v)
        return int(f) if f == int(f) and "." not in v else f
    except (ValueError, OverflowError):
        return v


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: _parse_num(v) for k, v in row.items()})
        return rows


def load_sweep_data(path: Path) -> dict[str, list]:
    """Load a single sweep comparison CSV into column-oriented dict of arrays."""
    rows = _read_csv(path)
    if not rows:
        return {}
    cols: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            cols[k].append(v)
    return cols


def load_unified_data(
    nbfi_path: Path,
    bank_path: Path,
) -> dict[str, np.ndarray]:
    """Merge NBFI and bank sweep CSVs on shared parameter grid.

    Returns a dict of numpy arrays keyed by column name, with columns from
    both sweeps available (bank columns prefixed with ``bank_`` where needed
    to avoid collisions).
    """
    nbfi_rows = _read_csv(nbfi_path)
    bank_rows = _read_csv(bank_path)

    # Build bank lookup by grid key
    bank_by_key: dict[tuple, dict] = {}
    for row in bank_rows:
        key = tuple(row.get(k, "") for k in _GRID_KEYS)
        bank_by_key[key] = row

    # Merge
    merged: list[dict[str, Any]] = []
    for nrow in nbfi_rows:
        key = tuple(nrow.get(k, "") for k in _GRID_KEYS)
        brow = bank_by_key.get(key)
        if brow is None:
            continue
        combined = dict(nrow)
        # Add bank columns (avoid collision)
        for col, val in brow.items():
            if col in _GRID_KEYS:
                continue
            bname = col if col.startswith("bank_") or col.startswith("delta_") else f"bank_{col}"
            # Special handling for bank-specific columns
            if col == "delta_idle":
                combined["delta_idle"] = val
            elif col == "delta_lend":
                combined["delta_lend"] = val
            elif col == "bank_lending_effect":
                combined["bank_lending_effect"] = val
            elif col == "phi_idle":
                combined["phi_idle"] = val
            elif col == "phi_lend":
                combined["phi_lend"] = val
            elif col not in combined:
                combined[bname] = val

        # Compute bank_effect_vs_passive: delta_passive - delta_lend
        dp = combined.get("delta_passive", np.nan)
        dl = combined.get("delta_lend", np.nan)
        if _is_valid(dp) and _is_valid(dl):
            combined["bank_effect_vs_passive"] = float(dp) - float(dl)
        else:
            combined["bank_effect_vs_passive"] = np.nan

        merged.append(combined)

    if not merged:
        raise ValueError("No matching rows found between NBFI and bank sweeps")

    # Convert to column-oriented numpy arrays
    all_keys = list(merged[0].keys())
    result: dict[str, np.ndarray] = {}
    for k in all_keys:
        vals = [row.get(k, np.nan) for row in merged]
        try:
            result[k] = np.array(vals, dtype=float)
        except (ValueError, TypeError):
            result[k] = np.array(vals, dtype=object)

    return result


def _is_valid(v: Any) -> bool:
    if v is None or v == "":
        return False
    try:
        return np.isfinite(float(v))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# OLS regression (manual, no statsmodels)
# ---------------------------------------------------------------------------

@dataclass
class OLSResult:
    """Result of an OLS regression fit."""

    coef: np.ndarray  # (p,) coefficient vector
    se: np.ndarray  # (p,) standard errors
    t_stat: np.ndarray  # (p,) t-statistics
    p_value: np.ndarray  # (p,) two-tailed p-values
    r_squared: float
    adj_r_squared: float
    n: int
    k: int  # number of regressors including intercept
    feature_names: list[str]
    y_hat: np.ndarray  # (n,) fitted values
    residuals: np.ndarray  # (n,) residuals

    def summary_rows(self) -> list[dict[str, Any]]:
        """Return coefficient table as list of dicts."""
        rows = []
        for i, name in enumerate(self.feature_names):
            rows.append({
                "feature": name,
                "coef": self.coef[i],
                "se": self.se[i],
                "t_stat": self.t_stat[i],
                "p_value": self.p_value[i],
                "sig": "***" if self.p_value[i] < 0.001 else "**" if self.p_value[i] < 0.01 else "*" if self.p_value[i] < 0.05 else "",
            })
        return rows


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    add_intercept: bool = True,
) -> OLSResult:
    """Fit OLS regression using numpy.linalg.lstsq + scipy.stats.t.

    Parameters
    ----------
    X : (n, p) design matrix (without intercept unless add_intercept=False)
    y : (n,) response vector
    feature_names : optional names for columns of X (intercept prepended if added)
    add_intercept : whether to prepend a column of ones
    """
    from scipy import stats as sp_stats

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Remove rows with NaN
    mask = np.isfinite(y)
    for j in range(X.shape[1]):
        mask &= np.isfinite(X[:, j])
    X = X[mask]
    y = y[mask]
    n = len(y)

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        names = ["intercept"] + (feature_names or [f"x{i}" for i in range(X.shape[1] - 1)])
    else:
        names = feature_names or [f"x{i}" for i in range(X.shape[1])]

    k = X.shape[1]

    # Solve
    coef, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ coef
    resid = y - y_hat

    # Residual variance
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

    # Standard errors
    sigma2 = ss_res / max(n - k, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    t_stat = np.where(se > 0, coef / se, 0.0)
    dof = max(n - k, 1)
    p_val = 2.0 * sp_stats.t.sf(np.abs(t_stat), dof)

    return OLSResult(
        coef=coef, se=se, t_stat=t_stat, p_value=p_val,
        r_squared=r2, adj_r_squared=adj_r2, n=n, k=k,
        feature_names=names, y_hat=y_hat, residuals=resid,
    )


def fit_ols_from_df(
    data: dict[str, np.ndarray],
    y_col: str,
    x_cols: list[str],
    *,
    add_intercept: bool = True,
) -> OLSResult:
    """Convenience: fit OLS from column-oriented data dict."""
    X = np.column_stack([data[c] for c in x_cols])
    return fit_ols(X, data[y_col], feature_names=x_cols, add_intercept=add_intercept)


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (point_estimate, ci_low, ci_high).
    """
    from scipy import stats as sp_stats

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)

    stat_fn = np.mean if statistic == "mean" else np.median

    point = float(stat_fn(data))

    if len(data) < 3:
        return (point, point, point)

    result = sp_stats.bootstrap(
        (data,),
        stat_fn,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng_seed,
        method="percentile",
    )
    return (point, float(result.confidence_interval.low), float(result.confidence_interval.high))


def bootstrap_ci_by_group(
    data: dict[str, np.ndarray],
    metric_col: str,
    group_col: str,
    **kwargs: Any,
) -> dict[Any, tuple[float, float, float]]:
    """Bootstrap CI for a metric grouped by another column."""
    groups = np.unique(data[group_col][np.isfinite(data[group_col])])
    results = {}
    for g in groups:
        mask = data[group_col] == g
        vals = data[metric_col][mask]
        results[float(g)] = bootstrap_ci(vals, **kwargs)
    return results


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(
    trading_effect: float,
    lending_effect: float,
    bank_lending_effect: float,
    threshold: float = 0.005,
) -> str:
    """Classify which mechanism dominates for a parameter combination.

    Returns one of: 'dealer', 'nbfi', 'bank', 'none', 'mixed'.
    """
    effects = {
        "dealer": trading_effect if np.isfinite(trading_effect) else -np.inf,
        "nbfi": lending_effect if np.isfinite(lending_effect) else -np.inf,
        "bank": bank_lending_effect if np.isfinite(bank_lending_effect) else -np.inf,
    }
    best = max(effects, key=effects.get)  # type: ignore[arg-type]
    best_val = effects[best]

    if best_val < threshold:
        return "none"

    # Check if second-best is close
    sorted_vals = sorted(effects.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] - sorted_vals[1] < threshold:
        return "mixed"

    return best


def classify_regimes(data: dict[str, np.ndarray], threshold: float = 0.005) -> np.ndarray:
    """Vectorized regime classification."""
    n = len(data["kappa"])
    te = data.get("trading_effect", np.full(n, np.nan))
    le = data.get("lending_effect", np.full(n, np.nan))
    ble = data.get("bank_lending_effect", np.full(n, np.nan))
    return np.array([
        classify_regime(
            te[i] if np.isfinite(te[i]) else 0,
            le[i] if np.isfinite(le[i]) else 0,
            ble[i] if np.isfinite(ble[i]) else 0,
            threshold,
        )
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# Summary / grouping helpers
# ---------------------------------------------------------------------------

def summary_by_group(
    data: dict[str, np.ndarray],
    groupby_cols: list[str],
    metric_cols: list[str],
) -> list[dict[str, Any]]:
    """Grouped mean/median/sd/count aggregation.

    Returns list of dicts, one per group.
    """
    # Build group keys
    n = len(next(iter(data.values())))
    groups: dict[tuple, list[int]] = {}
    for i in range(n):
        key = tuple(data[c][i] for c in groupby_cols)
        groups.setdefault(key, []).append(i)

    rows = []
    for key, indices in sorted(groups.items()):
        idx = np.array(indices)
        row: dict[str, Any] = {}
        for c, v in zip(groupby_cols, key):
            row[c] = v
        row["count"] = len(idx)
        for mc in metric_cols:
            vals = data[mc][idx]
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                row[f"{mc}_mean"] = np.nan
                row[f"{mc}_median"] = np.nan
                row[f"{mc}_sd"] = np.nan
                row[f"{mc}_pct_positive"] = np.nan
            else:
                row[f"{mc}_mean"] = float(np.mean(valid))
                row[f"{mc}_median"] = float(np.median(valid))
                row[f"{mc}_sd"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                row[f"{mc}_pct_positive"] = float(np.mean(valid > 0) * 100)
        rows.append(row)
    return rows


def summary_stats(arr: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for an array."""
    valid = np.asarray(arr, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return {"mean": np.nan, "median": np.nan, "sd": np.nan, "iqr_low": np.nan, "iqr_high": np.nan, "pct_positive": np.nan, "n": 0}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "iqr_low": float(np.percentile(valid, 25)),
        "iqr_high": float(np.percentile(valid, 75)),
        "pct_positive": float(np.mean(valid > 0) * 100),
        "n": len(valid),
    }


def kappa_band(kappa: float) -> str:
    """Classify kappa into stress band."""
    if kappa <= 0.25:
        return "severe"
    elif kappa <= 0.5:
        return "stressed"
    elif kappa <= 1.0:
        return "balanced"
    elif kappa <= 2.0:
        return "comfortable"
    else:
        return "abundant"


def add_derived_columns(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add computed columns to the unified data dict (in-place and return)."""
    n = len(data["kappa"])

    # Log kappa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data["log_kappa"] = np.log(data["kappa"].astype(float))

    # Kappa bands
    data["kappa_band"] = np.array([kappa_band(k) for k in data["kappa"]])

    # Regime classification
    data["regime"] = classify_regimes(data)

    # ---- Derived loss columns ----
    def _safe_div(num_key: str, den_key: str, out_key: str) -> None:
        if num_key in data and den_key in data:
            num = np.asarray(data[num_key], dtype=float)
            den = np.asarray(data[den_key], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ratio = np.where((den > 0) & np.isfinite(den) & np.isfinite(num),
                                 num / den, np.nan)
            data[out_key] = ratio

    # Loss Given Default (LGD) = total_loss_pct / delta  for each arm
    for arm in ("passive", "active", "lender", "dealer_lender"):
        _safe_div(f"total_loss_pct_{arm}", f"delta_{arm}", f"lgd_{arm}")
    # Bank sweep LGD
    for arm in ("idle", "lend"):
        _safe_div(f"bank_total_loss_pct_{arm}", f"delta_{arm}", f"lgd_{arm}")

    # System LGD = system_loss_pct / delta
    for arm in ("passive", "active", "lender", "dealer_lender"):
        _safe_div(f"system_loss_pct_{arm}", f"delta_{arm}", f"system_lgd_{arm}")
    for arm in ("idle", "lend"):
        _safe_div(f"bank_system_loss_pct_{arm}", f"delta_{arm}", f"system_lgd_{arm}")

    # Loss per default event = total_loss / n_defaults
    for arm in ("passive", "active", "lender", "dealer_lender"):
        _safe_div(f"total_loss_{arm}", f"n_defaults_{arm}", f"loss_per_default_{arm}")
    for arm in ("idle", "lend"):
        _safe_div(f"bank_total_loss_{arm}", f"n_defaults_{arm}", f"loss_per_default_{arm}")

    return data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test for H0: mean(a - b) = 0."""
    from scipy import stats as sp_stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "n": 0}
    t, p = sp_stats.ttest_rel(a, b)
    return {"t_stat": float(t), "p_value": float(p), "n": len(a)}


def one_sample_ttest(effect: np.ndarray) -> dict[str, float]:
    """One-sample t-test for H0: mean(effect) = 0."""
    from scipy import stats as sp_stats
    effect = np.asarray(effect, dtype=float)
    effect = effect[np.isfinite(effect)]
    if len(effect) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "n": 0, "mean": np.nan}
    t, p = sp_stats.ttest_1samp(effect, 0)
    return {"t_stat": float(t), "p_value": float(p), "n": len(effect), "mean": float(np.mean(effect))}


def wilcoxon_test(effect: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test for H0: median(effect) = 0."""
    from scipy import stats as sp_stats
    effect = np.asarray(effect, dtype=float)
    effect = effect[np.isfinite(effect)]
    # Remove exact zeros for Wilcoxon
    effect = effect[effect != 0]
    if len(effect) < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(effect)}
    stat, p = sp_stats.wilcoxon(effect)
    return {"statistic": float(stat), "p_value": float(p), "n": len(effect)}


def kruskal_wallis(*groups: np.ndarray) -> dict[str, float]:
    """Kruskal-Wallis H-test for comparing distributions."""
    from scipy import stats as sp_stats
    cleaned = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if len(g) > 0:
            cleaned.append(g)
    if len(cleaned) < 2:
        return {"H_stat": np.nan, "p_value": np.nan}
    h, p = sp_stats.kruskal(*cleaned)
    return {"H_stat": float(h), "p_value": float(p)}


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U test."""
    from scipy import stats as sp_stats
    a = np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
    b = np.asarray(b, dtype=float)[np.isfinite(np.asarray(b, dtype=float))]
    if len(a) < 2 or len(b) < 2:
        return {"U_stat": np.nan, "p_value": np.nan}
    u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U_stat": float(u), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Standardized coefficients
# ---------------------------------------------------------------------------

def standardized_coefs(ols: OLSResult, X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Compute standardized (beta) coefficients from an OLS result."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    sy = np.std(y, ddof=1) if len(y) > 1 else 1.0
    rows = []
    offset = 1 if ols.feature_names[0] == "intercept" else 0
    for i in range(offset, len(ols.feature_names)):
        sx = np.std(X[:, i - offset], ddof=1) if X.shape[0] > 1 else 1.0
        beta_std = ols.coef[i] * sx / sy if sy > 0 else 0.0
        rows.append({
            "feature": ols.feature_names[i],
            "coef": ols.coef[i],
            "std_coef": beta_std,
            "abs_std_coef": abs(beta_std),
        })
    return sorted(rows, key=lambda r: r["abs_std_coef"], reverse=True)


# ---------------------------------------------------------------------------
# Heatmap pivot helper
# ---------------------------------------------------------------------------

def pivot_for_heatmap(
    data: dict[str, np.ndarray],
    row_col: str,
    col_col: str,
    value_col: str,
    agg: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot data into 2D array for heatmap plotting.

    Returns (row_vals, col_vals, matrix) where matrix[i,j] = agg(value)
    for row_vals[i], col_vals[j].
    """
    rc = data[row_col].astype(float)
    cc = data[col_col].astype(float)
    vc = data[value_col].astype(float)

    row_vals = np.sort(np.unique(rc[np.isfinite(rc)]))
    col_vals = np.sort(np.unique(cc[np.isfinite(cc)]))

    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            mask = (rc == rv) & (cc == cv) & np.isfinite(vc)
            vals = vc[mask]
            if len(vals) > 0:
                matrix[i, j] = np.mean(vals) if agg == "mean" else np.median(vals)

    return row_vals, col_vals, matrix
