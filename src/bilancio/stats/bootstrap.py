"""Bootstrap confidence intervals.

Provides nonparametric bootstrap for any scalar statistic.
Works with any simulation -- no domain-specific assumptions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from bilancio.stats.types import ConfidenceInterval


def _mean(data: Sequence[float]) -> float:
    """Arithmetic mean."""
    return sum(data) / len(data)


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Compute a bootstrap confidence interval for a scalar statistic.

    Parameters
    ----------
    data:
        Observed values (one per replicate).
    statistic:
        Function mapping a sample to a scalar. Default: arithmetic mean.
    confidence:
        Confidence level (e.g. 0.95 for 95% CI).
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    ConfidenceInterval with point estimate, lower, upper bounds.

    Raises
    ------
    ValueError: if data has fewer than 2 elements.
    """
    if statistic is None:
        statistic = _mean

    n = len(data)
    if n < 2:
        raise ValueError(f"Bootstrap requires >= 2 data points, got {n}")

    rng = random.Random(seed)
    point_estimate = statistic(data)

    # Generate bootstrap distribution
    bootstrap_stats: list[float] = []
    for _ in range(n_bootstrap):
        resample = rng.choices(data, k=n)
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats.sort()

    # Percentile method
    alpha = 1 - confidence
    lower_idx = max(0, int(math.floor((alpha / 2) * n_bootstrap)) - 1)
    upper_idx = min(
        n_bootstrap - 1, int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    )

    return ConfidenceInterval(
        estimate=point_estimate,
        lower=bootstrap_stats[lower_idx],
        upper=bootstrap_stats[upper_idx],
        confidence=confidence,
    )
