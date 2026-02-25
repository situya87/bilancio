"""Kalecki ring sweep statistical analysis.

Ring-specific wrapper around bilancio.stats that interprets sweep results
in terms of Kalecki ring parameters (kappa, concentration, mu, rho) and
treatment effects (trading, lending, banking).

This module belongs to the experiments package (ring-specific), not the
general stats package. It consumes comparison.csv rows or
BalancedComparisonResult objects and produces per-cell statistics with
confidence intervals, significance tests, and sensitivity rankings.

Typical usage::

    from bilancio.experiments.sweep_analysis import RingSweepAnalysis

    # From CSV
    analysis = RingSweepAnalysis.from_csv(Path("aggregate/comparison.csv"))

    # Per-cell treatment effects with CIs and p-values
    effects = analysis.trading_effects()
    for row in effects.rows:
        print(f"kappa={row.params['kappa']}: {row.stats.effect}")

    # Which ring parameters matter most?
    ranking = analysis.sensitivity("delta_passive")

    # Write enhanced summary
    analysis.write_stats(Path("aggregate/"))
"""

from __future__ import annotations

import csv
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.stats.analyzer import CellTable, EffectTable, SweepAnalyzer
from bilancio.stats.types import MorrisResult

logger = logging.getLogger(__name__)

# Ring parameter fields used for cell grouping
RING_PARAM_FIELDS = ["kappa", "concentration", "mu", "outside_mid_ratio"]

# Treatment arms and their CSV column suffixes
RING_ARMS: dict[str, str] = {
    "passive": "_passive",
    "active": "_active",
    "lender": "_lender",
    "dealer_lender": "_dealer_lender",
    "bank_passive": "_bank_passive",
    "bank_dealer": "_bank_dealer",
    "bank_dealer_nbfi": "_bank_dealer_nbfi",
}

# Named treatment comparisons (control_suffix, treatment_suffix)
RING_EFFECTS: dict[str, tuple[str, str]] = {
    "trading": ("_passive", "_active"),
    "lending": ("_passive", "_lender"),
    "combined": ("_passive", "_dealer_lender"),
    "bank_passive": ("_passive", "_bank_passive"),
    "bank_dealer": ("_passive", "_bank_dealer"),
    "bank_dealer_nbfi": ("_passive", "_bank_dealer_nbfi"),
}


class RingSweepAnalysis:
    """Statistical analysis of Kalecki ring balanced comparison sweeps.

    Wraps SweepAnalyzer with ring-specific:
    - Parameter names (kappa, concentration, mu, outside_mid_ratio)
    - Effect names (trading, lending, combined, banking variants)
    - Sensitivity bounds derived from the actual parameter grid
    - Output formatting (stats.csv, stats_summary.json)

    Parameters
    ----------
    records:
        List of dicts, one per sweep row. Each dict must have the
        ring parameter fields and metric fields (e.g., delta_passive,
        delta_active). Typically loaded from comparison.csv.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self._analyzer = SweepAnalyzer(records, param_fields=RING_PARAM_FIELDS)

    @classmethod
    def from_csv(cls, path: Path) -> RingSweepAnalysis:
        """Load from a comparison.csv file."""
        records = []
        with path.open("r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(row)
        return cls(records)

    @classmethod
    def from_results(
        cls, results: list[Any],
    ) -> RingSweepAnalysis:
        """Create from BalancedComparisonResult objects.

        Converts each result to a flat dict compatible with SweepAnalyzer.
        """
        records = [_result_to_dict(r) for r in results]
        return cls(records)

    @property
    def n_cells(self) -> int:
        return self._analyzer.n_cells

    @property
    def n_records(self) -> int:
        return self._analyzer.n_records

    def min_replicates(self) -> int:
        return self._analyzer.min_replicates()

    # --- Per-cell statistics ---

    def cell_summary(
        self,
        metric: str,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> CellTable:
        """Per-cell summary statistics for any metric.

        Parameters
        ----------
        metric:
            Full column name (e.g. "delta_passive", "phi_active",
            "trading_effect", "n_defaults_passive").
        """
        return self._analyzer.cell_table(
            metric=metric, confidence=confidence,
            n_bootstrap=n_bootstrap, seed=seed,
        )

    # --- Treatment effects ---

    def trading_effects(
        self,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> EffectTable:
        """Per-cell dealer trading effect: delta_passive - delta_active."""
        return self._effect("trading", confidence, n_bootstrap, seed)

    def lending_effects(
        self,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> EffectTable:
        """Per-cell lending effect: delta_passive - delta_lender."""
        return self._effect("lending", confidence, n_bootstrap, seed)

    def combined_effects(
        self,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> EffectTable:
        """Per-cell combined (dealer+lender) effect."""
        return self._effect("combined", confidence, n_bootstrap, seed)

    def all_effects(
        self,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> dict[str, EffectTable]:
        """Compute all available treatment effects.

        Only includes effects where both arms have data.
        """
        results = {}
        for name in RING_EFFECTS:
            control_suffix, treatment_suffix = RING_EFFECTS[name]
            control_field = f"delta{control_suffix}"
            treatment_field = f"delta{treatment_suffix}"
            # Check if any records have both fields
            has_data = any(
                r.get(control_field) is not None and r.get(treatment_field) is not None
                for r in self.records
            )
            if has_data:
                results[name] = self._effect(name, confidence, n_bootstrap, seed)
        return results

    def _effect(
        self, name: str, confidence: float, n_bootstrap: int, seed: int | None,
    ) -> EffectTable:
        control_suffix, treatment_suffix = RING_EFFECTS[name]
        return self._analyzer.treatment_effect_table(
            metric="delta",
            control_suffix=control_suffix,
            treatment_suffix=treatment_suffix,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    # --- Sensitivity analysis ---

    def sensitivity(
        self,
        metric: str = "delta_passive",
        num_trajectories: int = 20,
        seed: int | None = 42,
    ) -> list[MorrisResult]:
        """Morris screening: which ring parameters matter most?

        Derives parameter bounds from the actual sweep grid.
        """
        bounds = self._infer_bounds()
        if not bounds:
            return []
        return self._analyzer.sensitivity_ranking(
            metric=metric,
            bounds=bounds,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _infer_bounds(self) -> dict[str, tuple[float, float]]:
        """Infer parameter bounds from the records in the sweep."""
        bounds = {}
        for field in RING_PARAM_FIELDS:
            values = []
            for r in self.records:
                v = r.get(field)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        continue
            if len(values) >= 2:
                lo, hi = min(values), max(values)
                if lo < hi:
                    bounds[field] = (lo, hi)
        return bounds

    # --- Output ---

    def write_stats(
        self,
        output_dir: Path,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> dict[str, Path]:
        """Write statistical analysis files to output_dir.

        Produces:
        - stats_effects.csv: per-cell treatment effects with CIs and p-values
        - stats_cells.csv: per-cell summary statistics for key metrics
        - stats_summary.json: overall summary with significance counts
        - stats_sensitivity.json: Morris parameter importance rankings

        Returns dict of file names to paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Effects CSV
        effects = self.all_effects(confidence, n_bootstrap, seed)
        if effects:
            effects_path = output_dir / "stats_effects.csv"
            self._write_effects_csv(effects, effects_path)
            paths["effects"] = effects_path

        # Cell summaries CSV
        cells_path = output_dir / "stats_cells.csv"
        self._write_cells_csv(confidence, n_bootstrap, seed, cells_path)
        paths["cells"] = cells_path

        # Summary JSON
        summary_path = output_dir / "stats_summary.json"
        self._write_summary_json(effects, summary_path)
        paths["summary"] = summary_path

        # Sensitivity JSON
        sensitivity_path = output_dir / "stats_sensitivity.json"
        self._write_sensitivity_json(seed, sensitivity_path)
        paths["sensitivity"] = sensitivity_path

        return paths

    def _write_effects_csv(
        self, effects: dict[str, EffectTable], path: Path,
    ) -> None:
        """Write all treatment effects to a single CSV."""
        rows = []
        for effect_name, table in effects.items():
            for row_data in table.to_dicts():
                row_data["effect_type"] = effect_name
                rows.append(row_data)

        if not rows:
            return

        fieldnames = ["effect_type"] + [k for k in rows[0] if k != "effect_type"]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_cells_csv(
        self,
        confidence: float,
        n_bootstrap: int,
        seed: int | None,
        path: Path,
    ) -> None:
        """Write per-cell summaries for key metrics."""
        key_metrics = ["delta_passive", "delta_active", "phi_passive", "phi_active"]
        rows = []
        for metric in key_metrics:
            table = self.cell_summary(metric, confidence, n_bootstrap, seed)
            for row_data in table.to_dicts():
                row_data["metric"] = metric
                rows.append(row_data)

        if not rows:
            return

        fieldnames = ["metric"] + [k for k in rows[0] if k != "metric"]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_json(
        self, effects: dict[str, EffectTable], path: Path,
    ) -> None:
        """Write overall statistical summary."""
        summary: dict[str, Any] = {
            "n_records": self.n_records,
            "n_cells": self.n_cells,
            "min_replicates_per_cell": self.min_replicates(),
        }

        for effect_name, table in effects.items():
            s = table.summary()
            summary[f"{effect_name}_effect"] = {
                "n_cells": s.get("n_cells", 0),
                "mean_effect": s.get("mean_effect"),
                "median_cohens_d": s.get("median_cohens_d"),
                "n_positive": s.get("n_positive", 0),
                "n_negative": s.get("n_negative", 0),
                "n_significant_05": s.get("n_significant_05", 0),
                "pct_significant_05": s.get("pct_significant_05", 0),
            }

        with path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)

    def _write_sensitivity_json(self, seed: int | None, path: Path) -> None:
        """Write Morris sensitivity rankings."""
        rankings = {}
        for metric in ["delta_passive", "delta_active"]:
            try:
                results = self.sensitivity(metric=metric, seed=seed)
                rankings[metric] = [
                    {
                        "parameter": r.parameter,
                        "mu_star": r.mu_star,
                        "mu": r.mu,
                        "sigma": r.sigma,
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"Sensitivity analysis failed for {metric}: {e}")

        with path.open("w") as fh:
            json.dump(rankings, fh, indent=2)


# --- Helpers ---


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a BalancedComparisonResult to a flat dict.

    Extracts all numeric fields that SweepAnalyzer can work with.
    Handles Decimal -> float conversion and None values.
    """
    def _to_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    d: dict[str, Any] = {}

    # Parameters
    d["kappa"] = _to_float(result.kappa)
    d["concentration"] = _to_float(result.concentration)
    d["mu"] = _to_float(result.mu)
    d["outside_mid_ratio"] = _to_float(getattr(result, "outside_mid_ratio", None))
    d["seed"] = getattr(result, "seed", None)

    # Per-arm metrics
    for arm_suffix in RING_ARMS.values():
        arm = arm_suffix.lstrip("_")
        d[f"delta{arm_suffix}"] = _to_float(getattr(result, f"delta_{arm}", None))
        d[f"phi{arm_suffix}"] = _to_float(getattr(result, f"phi_{arm}", None))
        d[f"n_defaults{arm_suffix}"] = _to_float(
            getattr(result, f"n_defaults_{arm}", None)
        )
        d[f"cascade_fraction{arm_suffix}"] = _to_float(
            getattr(result, f"cascade_fraction_{arm}", None)
        )

    # Computed effects
    d["trading_effect"] = _to_float(getattr(result, "trading_effect", None))
    d["lending_effect"] = _to_float(getattr(result, "lending_effect", None))
    d["combined_effect"] = _to_float(getattr(result, "combined_effect", None))

    return d


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
