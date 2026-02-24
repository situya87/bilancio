"""Sweep-level statistical analyzer.

Operates on replicated sweep results to produce per-cell statistics,
treatment effect tables, and global sensitivity rankings. This is the
integration layer that connects pure statistical functions to simulation
experiment data.

The analyzer is simulation-agnostic: it works with any data structured as
a list of records (dicts), where each record has parameter fields and
metric fields. It groups records into cells by parameter values and
computes statistics within each cell.

Typical usage::

    from bilancio.stats.analyzer import SweepAnalyzer

    # records: list of dicts from comparison.csv or similar
    analyzer = SweepAnalyzer(
        records=records,
        param_fields=["kappa", "concentration", "mu"],
    )

    # Per-cell summary for one arm
    table = analyzer.cell_table(metric="delta_total", arm_suffix="_passive")

    # Paired treatment effect
    effects = analyzer.treatment_effect_table(
        metric="delta",
        control_suffix="_passive",
        treatment_suffix="_active",
    )

    # Which parameters matter most?
    ranking = analyzer.sensitivity_ranking(
        metric="delta_passive",
        bounds={"kappa": (0.25, 4.0), "concentration": (0.2, 5.0), "mu": (0.0, 1.0)},
    )
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from bilancio.stats.cell import summarize_cell, summarize_paired_cell
from bilancio.stats.sensitivity import morris_screening
from bilancio.stats.types import CellStats, MorrisResult, PairedCellStats


@dataclass(frozen=True)
class CellKey:
    """Hashable key identifying one parameter cell."""

    values: tuple[tuple[str, float], ...]

    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass
class CellTableRow:
    """One row in a cell-level summary table."""

    params: dict[str, float]
    stats: CellStats


@dataclass
class EffectTableRow:
    """One row in a treatment effect table."""

    params: dict[str, float]
    stats: PairedCellStats


@dataclass
class CellTable:
    """Table of per-cell summary statistics for one metric/arm."""

    metric: str
    rows: list[CellTableRow]

    def to_dicts(self) -> list[dict]:
        """Flatten to list of dicts for CSV/DataFrame conversion."""
        result = []
        for row in self.rows:
            d = dict(row.params)
            d["n"] = row.stats.n
            d["mean"] = row.stats.mean
            d["std"] = row.stats.std
            d["se"] = row.stats.se
            d["ci_lower"] = row.stats.ci.lower
            d["ci_upper"] = row.stats.ci.upper
            d["median"] = row.stats.median
            d["min"] = row.stats.min
            d["max"] = row.stats.max
            result.append(d)
        return result


@dataclass
class EffectTable:
    """Table of per-cell treatment effect statistics."""

    metric: str
    control_suffix: str
    treatment_suffix: str
    rows: list[EffectTableRow]

    def to_dicts(self) -> list[dict]:
        """Flatten to list of dicts for CSV/DataFrame conversion."""
        result = []
        for row in self.rows:
            d = dict(row.params)
            s = row.stats
            d["n_pairs"] = s.n_pairs
            d["control_mean"] = s.control.mean
            d["treatment_mean"] = s.treatment.mean
            d["effect"] = s.effect.estimate
            d["effect_ci_lower"] = s.effect.lower
            d["effect_ci_upper"] = s.effect.upper
            d["p_value"] = s.effect_test.p_value
            d["significant_05"] = s.effect_test.significant_at_05
            d["significant_01"] = s.effect_test.significant_at_01
            d["cohens_d"] = s.effect_size
            result.append(d)
        return result

    def summary(self) -> dict:
        """Aggregate summary across all cells."""
        if not self.rows:
            return {}

        effects = [r.stats.effect.estimate for r in self.rows]
        n_sig_05 = sum(1 for r in self.rows if r.stats.effect_test.significant_at_05)
        n_sig_01 = sum(1 for r in self.rows if r.stats.effect_test.significant_at_01)
        n_positive = sum(1 for e in effects if e > 0)
        n_negative = sum(1 for e in effects if e < 0)
        n_cells = len(self.rows)

        mean_effect = sum(effects) / n_cells
        median_d = sorted(r.stats.effect_size for r in self.rows)[n_cells // 2]

        return {
            "metric": self.metric,
            "n_cells": n_cells,
            "mean_effect": mean_effect,
            "median_cohens_d": median_d,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_significant_05": n_sig_05,
            "n_significant_01": n_sig_01,
            "pct_significant_05": n_sig_05 / n_cells if n_cells > 0 else 0,
        }


class SweepAnalyzer:
    """Analyze replicated sweep results with proper statistical methods.

    Operates on a list of flat records (dicts), where each record represents
    one simulation run. Records are grouped into cells by parameter fields.
    Within each cell, replicates (different seeds) provide the basis for
    uncertainty quantification.

    Parameters
    ----------
    records:
        List of dicts, each containing parameter fields and metric fields.
        Example: [{"kappa": 0.5, "mu": 0, "seed": 42,
                   "delta_passive": 0.3, "delta_active": 0.2}, ...]
    param_fields:
        Names of fields that define a parameter cell. Records with
        identical values on these fields belong to the same cell.
    """

    def __init__(
        self,
        records: Sequence[dict],
        param_fields: Sequence[str],
    ) -> None:
        self.records = list(records)
        self.param_fields = list(param_fields)
        self._cells: dict[CellKey, list[dict]] | None = None

    @property
    def cells(self) -> dict[CellKey, list[dict]]:
        """Lazily group records into cells."""
        if self._cells is None:
            self._cells = self._group_into_cells()
        return self._cells

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_records(self) -> int:
        return len(self.records)

    def replicates_per_cell(self) -> dict[CellKey, int]:
        """Count of replicates in each cell."""
        return {k: len(v) for k, v in self.cells.items()}

    def min_replicates(self) -> int:
        """Minimum replicate count across all cells."""
        counts = self.replicates_per_cell()
        return min(counts.values()) if counts else 0

    def cell_table(
        self,
        metric: str,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> CellTable:
        """Compute per-cell summary statistics for a single metric.

        Parameters
        ----------
        metric:
            Field name in the records (e.g. "delta_passive", "phi_total").
        confidence:
            Confidence level for bootstrap CI.
        n_bootstrap:
            Bootstrap resamples per cell.
        seed:
            Base RNG seed (incremented per cell for independence).

        Returns
        -------
        CellTable with one row per parameter cell.
        """
        rows = []
        for i, (key, cell_records) in enumerate(sorted(
            self.cells.items(), key=lambda kv: kv[0].values
        )):
            values = self._extract_metric(cell_records, metric)
            if len(values) < 2:
                continue

            cell_seed = seed + i if seed is not None else None
            stats = summarize_cell(
                values, metric=metric, confidence=confidence,
                n_bootstrap=n_bootstrap, seed=cell_seed,
            )
            rows.append(CellTableRow(params=key.as_dict(), stats=stats))

        return CellTable(metric=metric, rows=rows)

    def treatment_effect_table(
        self,
        metric: str,
        control_suffix: str,
        treatment_suffix: str,
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        seed: int | None = 42,
    ) -> EffectTable:
        """Compute per-cell paired treatment effects.

        For each cell, extracts control and treatment values from records
        (e.g. "delta_passive" and "delta_active"), computes the paired
        difference, bootstrap CI, significance test, and effect size.

        Parameters
        ----------
        metric:
            Base metric name (e.g. "delta"). The full field names are
            constructed as f"{metric}{control_suffix}" and
            f"{metric}{treatment_suffix}".
        control_suffix:
            Suffix for the control field (e.g. "_passive").
        treatment_suffix:
            Suffix for the treatment field (e.g. "_active").

        Returns
        -------
        EffectTable with one row per parameter cell.
        """
        control_field = f"{metric}{control_suffix}"
        treatment_field = f"{metric}{treatment_suffix}"

        rows = []
        for i, (key, cell_records) in enumerate(sorted(
            self.cells.items(), key=lambda kv: kv[0].values
        )):
            # Extract paired values: only include records where BOTH
            # control and treatment fields are present and valid.
            # This prevents mispairing when one arm has missing data.
            control_vals, treatment_vals = self._extract_paired_metrics(
                cell_records, control_field, treatment_field,
            )

            if len(control_vals) < 2:
                continue

            cell_seed = seed + i * 3 if seed is not None else None
            stats = summarize_paired_cell(
                control_vals, treatment_vals, metric=metric,
                confidence=confidence, n_bootstrap=n_bootstrap,
                seed=cell_seed,
            )
            rows.append(EffectTableRow(params=key.as_dict(), stats=stats))

        return EffectTable(
            metric=metric,
            control_suffix=control_suffix,
            treatment_suffix=treatment_suffix,
            rows=rows,
        )

    def sensitivity_ranking(
        self,
        metric: str,
        bounds: dict[str, tuple[float, float]],
        num_trajectories: int = 20,
        num_levels: int = 4,
        seed: int | None = 42,
    ) -> list[MorrisResult]:
        """Morris screening to rank parameter importance.

        Builds an interpolating model from the sweep data (cell means)
        and applies Morris screening to identify which parameters have
        the largest effect on the metric.

        Parameters
        ----------
        metric:
            Field name in records (e.g. "delta_passive").
        bounds:
            Parameter bounds for Morris trajectories:
            {param_name: (lower, upper)}.
        num_trajectories:
            Number of Morris trajectories (more = better estimates).
        num_levels:
            Grid resolution for Morris method.
        seed:
            RNG seed.

        Returns
        -------
        List of MorrisResult sorted by mu* (most important first).
        """
        # Build lookup of cell means
        cell_means = self._build_cell_means(metric)
        if not cell_means:
            return []

        # Model function: nearest-neighbor lookup in cell means
        def model_fn(params: dict[str, float]) -> float:
            return self._nearest_cell_mean(params, cell_means)

        return morris_screening(
            model=model_fn,
            bounds=bounds,
            num_trajectories=num_trajectories,
            num_levels=num_levels,
            seed=seed,
        )

    # --- Internal helpers ---

    def _group_into_cells(self) -> dict[CellKey, list[dict]]:
        """Group records by parameter values."""
        cells: dict[CellKey, list[dict]] = defaultdict(list)
        for record in self.records:
            key = CellKey(
                values=tuple(
                    (f, float(record[f])) for f in self.param_fields
                )
            )
            cells[key].append(record)
        return dict(cells)

    def _extract_metric(
        self, records: list[dict], field: str
    ) -> list[float]:
        """Extract numeric values for a field, skipping None/missing."""
        values = []
        for r in records:
            v = r.get(field)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue
        return values

    def _extract_paired_metrics(
        self,
        records: list[dict],
        control_field: str,
        treatment_field: str,
    ) -> tuple[list[float], list[float]]:
        """Extract paired values, only including records where both fields are valid.

        This prevents mispairing when some records have missing data in one
        arm but not the other. Each (control, treatment) pair comes from the
        same record, preserving seed alignment.
        """
        control_vals = []
        treatment_vals = []
        for r in records:
            c_raw = r.get(control_field)
            t_raw = r.get(treatment_field)
            if c_raw is None or t_raw is None:
                continue
            try:
                c_val = float(c_raw)
                t_val = float(t_raw)
            except (TypeError, ValueError):
                continue
            control_vals.append(c_val)
            treatment_vals.append(t_val)
        return control_vals, treatment_vals

    def _build_cell_means(self, metric: str) -> dict[CellKey, float]:
        """Compute mean of metric within each cell."""
        means = {}
        for key, records in self.cells.items():
            values = self._extract_metric(records, metric)
            if values:
                means[key] = sum(values) / len(values)
        return means

    def _nearest_cell_mean(
        self, params: dict[str, float], cell_means: dict[CellKey, float]
    ) -> float:
        """Find nearest cell (Euclidean in normalized space) and return its mean."""
        best_dist = float("inf")
        best_val = 0.0

        for key, mean_val in cell_means.items():
            key_dict = key.as_dict()
            dist = sum(
                (params.get(f, 0) - key_dict.get(f, 0)) ** 2
                for f in self.param_fields
            )
            if dist < best_dist:
                best_dist = dist
                best_val = mean_val

        return best_val
