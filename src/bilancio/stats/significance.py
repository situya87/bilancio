"""Paired statistical significance tests.

Provides nonparametric (Wilcoxon signed-rank) and parametric (paired t-test)
tests for comparing two matched samples. These are appropriate for paired
simulation experiments where control and treatment share the same seed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from bilancio.stats.types import TestResult


def paired_wilcoxon(
    control: Sequence[float],
    treatment: Sequence[float],
) -> TestResult:
    """Wilcoxon signed-rank test on paired differences.

    Nonparametric test — no normality assumption. Tests H0: median
    difference = 0. Preferred over paired t-test when sample sizes
    are small or distributions are skewed.

    Parameters
    ----------
    control:
        Metric values under control condition.
    treatment:
        Metric values under treatment condition (same seeds/order).

    Returns
    -------
    TestResult with test statistic, p-value, and significance flags.
    """
    n = len(control)
    if n != len(treatment):
        raise ValueError(
            f"Control ({n}) and treatment ({len(treatment)}) must have equal length"
        )
    if n < 6:
        raise ValueError(
            f"Wilcoxon signed-rank requires >= 6 pairs, got {n}. "
            "Use paired_t_test for smaller samples (with normality caveat)."
        )

    differences = [c - t for c, t in zip(control, treatment)]

    # Remove zeros (ties with zero difference)
    nonzero = [(abs(d), d) for d in differences if d != 0.0]
    n_eff = len(nonzero)

    if n_eff == 0:
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            n=n,
            significant_at_05=False,
            significant_at_01=False,
        )

    # Rank absolute differences
    nonzero.sort(key=lambda x: x[0])
    ranks = _assign_ranks([abs_d for abs_d, _ in nonzero])

    # Sum of ranks for positive and negative differences
    w_plus = sum(r for r, (_, d) in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation (valid for n_eff >= 10, reasonable for >= 6)
    mean_w = n_eff * (n_eff + 1) / 4
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24
    z = (w - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0

    # Two-tailed p-value from normal approximation
    p_value = 2 * _normal_cdf(-abs(z))

    return TestResult(
        test_name="Wilcoxon signed-rank",
        statistic=w,
        p_value=p_value,
        n=n,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
    )


def paired_t_test(
    control: Sequence[float],
    treatment: Sequence[float],
) -> TestResult:
    """Paired (dependent) t-test on differences.

    Parametric test — assumes differences are approximately normally
    distributed. More powerful than Wilcoxon when normality holds,
    but less robust when it doesn't.

    Parameters
    ----------
    control:
        Metric values under control condition.
    treatment:
        Metric values under treatment condition (same seeds/order).

    Returns
    -------
    TestResult with t-statistic, p-value, and significance flags.
    """
    n = len(control)
    if n != len(treatment):
        raise ValueError(
            f"Control ({n}) and treatment ({len(treatment)}) must have equal length"
        )
    if n < 2:
        raise ValueError(f"Paired t-test requires >= 2 pairs, got {n}")

    differences = [c - t for c, t in zip(control, treatment)]

    mean_d = sum(differences) / n
    var_d = sum((d - mean_d) ** 2 for d in differences) / (n - 1)
    se_d = math.sqrt(var_d / n) if var_d > 0 else 0.0

    if se_d == 0.0:
        # All differences identical
        p_value = 0.0 if mean_d != 0.0 else 1.0
        t_stat = float("inf") if mean_d > 0 else float("-inf") if mean_d < 0 else 0.0
    else:
        t_stat = mean_d / se_d
        # Two-tailed p-value from t-distribution (df = n-1)
        p_value = min(1.0, 2 * _t_cdf(-abs(t_stat), n - 1))

    return TestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        n=n,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
    )


# --- Internal helpers (avoid scipy dependency) ---


def _assign_ranks(sorted_values: list[float]) -> list[float]:
    """Assign ranks with average tie-breaking to pre-sorted values."""
    n = len(sorted_values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_values[j + 1] == sorted_values[j]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    return ranks


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_cdf(t: float, df: int) -> float:
    """Approximate CDF of Student's t-distribution.

    Uses the regularized incomplete beta function relationship:
    F(t|df) = 1 - 0.5 * I_x(df/2, 1/2) where x = df/(df + t^2)

    For large df (>30), falls back to normal approximation.
    """
    if df > 30:
        return _normal_cdf(t)

    x = df / (df + t * t)
    # Regularized incomplete beta via continued fraction
    p = _regularized_beta(x, df / 2.0, 0.5)
    cdf = 1.0 - 0.5 * p if t >= 0 else 0.5 * p
    # Clamp to [0, 1] as a safety net against numerical drift
    return max(0.0, min(1.0, cdf))


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses Lentz's continued fraction with the symmetry relation
    I_x(a,b) = 1 - I_{1-x}(b,a) for convergence.  If the primary
    CF branch produces an out-of-range result, falls back to the
    symmetric branch automatically.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Symmetry relation for convergence (Numerical Recipes, Press et al.):
    # the CF converges well when x < (a+1)/(a+b+2).
    if x > (a + 1.0) / (a + b + 2.0):
        result = 1.0 - _betacf(1.0 - x, b, a)
    else:
        result = _betacf(x, a, b)

    # If the primary branch didn't converge (result outside [0,1]),
    # try the other branch as a fallback.
    if not (0.0 <= result <= 1.0):
        if x > (a + 1.0) / (a + b + 2.0):
            result = _betacf(x, a, b)
        else:
            result = 1.0 - _betacf(1.0 - x, b, a)

    return max(0.0, min(1.0, result))


def _betacf(x: float, a: float, b: float) -> float:
    """Evaluate I_x(a,b) via Lentz's continued fraction.

    The CF converges well when x is small relative to a/(a+b).
    May return values outside [0,1] if convergence is poor;
    callers should validate the result.
    """
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    prefix = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a

    max_iter = 200
    eps = 1e-14
    tiny = 1e-30

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step: d_{2m}
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd step: d_{2m+1}
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return prefix * f * a
