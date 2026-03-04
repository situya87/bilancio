"""Effect size measures for treatment comparisons.

Provides Cohen's d for both independent and paired designs.
Domain-agnostic — works with any numeric metric.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cohens_d(
    control: Sequence[float],
    treatment: Sequence[float],
) -> float:
    """Cohen's d for independent samples.

    Standardized mean difference using pooled standard deviation.
    Positive d means control > treatment (treatment reduced the metric).

    Interpretation (Cohen, 1988):
        |d| < 0.2  : negligible
        |d| ~ 0.2  : small
        |d| ~ 0.5  : medium
        |d| ~ 0.8  : large
        |d| > 1.2  : very large
    """
    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need >= 2 per group, got {n1} and {n2}")

    mean1 = sum(control) / n1
    mean2 = sum(treatment) / n2
    var1 = sum((x - mean1) ** 2 for x in control) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in treatment) / (n2 - 1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0

    return (mean1 - mean2) / pooled_sd


def cohens_d_paired(
    control: Sequence[float],
    treatment: Sequence[float],
) -> float:
    """Cohen's d for paired (repeated-measures) designs.

    Uses the standard deviation of the differences as the denominator
    (d_z formulation). This is the appropriate effect size for paired
    simulation experiments where control and treatment share the same seed.

    Positive d means control > treatment (treatment reduced the metric).
    """
    n = len(control)
    if n != len(treatment):
        raise ValueError(
            f"Control ({n}) and treatment ({len(treatment)}) must have equal length"
        )
    if n < 2:
        raise ValueError(f"Need >= 2 pairs, got {n}")

    differences = [c - t for c, t in zip(control, treatment, strict=False)]
    mean_d = sum(differences) / n
    var_d = sum((d - mean_d) ** 2 for d in differences) / (n - 1)
    sd_d = math.sqrt(var_d)

    if sd_d == 0:
        return 0.0

    return mean_d / sd_d
