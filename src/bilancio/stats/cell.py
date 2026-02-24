"""Per-cell summary statistics for simulation experiments.

A 'cell' is one point in parameter space (e.g., kappa=0.5, c=1, mu=0).
With replication, each cell has multiple observations (one per seed).
These functions summarize within-cell variability and compute
treatment effects with uncertainty quantification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from bilancio.stats.bootstrap import bootstrap_ci
from bilancio.stats.effect_size import cohens_d_paired
from bilancio.stats.significance import paired_wilcoxon, paired_t_test
from bilancio.stats.types import CellStats, ConfidenceInterval, PairedCellStats


def summarize_cell(
    values: Sequence[float],
    metric: str = "metric",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> CellStats:
    """Compute summary statistics for replicated observations in one cell.

    Parameters
    ----------
    values:
        Metric values from multiple replicates at the same parameter point.
    metric:
        Name of the metric (for labeling).
    confidence:
        Confidence level for bootstrap CI.
    n_bootstrap:
        Bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    CellStats with mean, std, SE, bootstrap CI, median, min, max.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"Need >= 2 replicates per cell, got {n}")

    sorted_vals = sorted(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )

    ci = bootstrap_ci(
        list(values),
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return CellStats(
        metric=metric,
        n=n,
        mean=mean,
        std=std,
        se=se,
        ci=ci,
        median=median,
        min=sorted_vals[0],
        max=sorted_vals[-1],
    )


def summarize_paired_cell(
    control: Sequence[float],
    treatment: Sequence[float],
    metric: str = "metric",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> PairedCellStats:
    """Compute treatment effect statistics for a paired cell.

    Each element i in control and treatment comes from the same seed,
    forming a natural pair. The treatment effect is control[i] - treatment[i].

    Parameters
    ----------
    control:
        Metric values under control condition (one per seed).
    treatment:
        Metric values under treatment condition (same seeds, same order).
    metric:
        Name of the metric.
    confidence:
        Confidence level.
    n_bootstrap:
        Bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    PairedCellStats with per-arm summaries, effect CI, significance test,
    and effect size.
    """
    n = len(control)
    if n != len(treatment):
        raise ValueError(
            f"Control ({n}) and treatment ({len(treatment)}) must have equal length"
        )
    if n < 2:
        raise ValueError(f"Need >= 2 paired replicates, got {n}")

    # Per-arm summaries
    control_stats = summarize_cell(
        control, metric=f"{metric}_control", confidence=confidence,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    treatment_stats = summarize_cell(
        treatment, metric=f"{metric}_treatment", confidence=confidence,
        n_bootstrap=n_bootstrap, seed=seed + 1 if seed is not None else None,
    )

    # Treatment effect (control - treatment)
    differences = [c - t for c, t in zip(control, treatment)]
    effect_ci = bootstrap_ci(
        differences,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed + 2 if seed is not None else None,
    )

    # Significance test
    if n >= 6:
        test = paired_wilcoxon(list(control), list(treatment))
    else:
        test = paired_t_test(list(control), list(treatment))

    # Effect size
    d = cohens_d_paired(list(control), list(treatment))

    return PairedCellStats(
        metric=metric,
        n_pairs=n,
        control=control_stats,
        treatment=treatment_stats,
        effect=effect_ci,
        effect_test=test,
        effect_size=d,
    )
