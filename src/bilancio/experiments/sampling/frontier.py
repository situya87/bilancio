"""Frontier/binary search parameter sampling strategy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal


def generate_frontier_params(
    concentrations: Sequence[Decimal],
    mus: Sequence[Decimal],
    monotonicities: Sequence[Decimal],
    *,
    kappa_low: Decimal,
    kappa_high: Decimal,
    tolerance: Decimal,
    max_iterations: int,
    execute_fn: Callable[[str, Decimal, Decimal, Decimal, Decimal], Decimal | None],
) -> None:
    """
    Execute frontier/binary search parameter combinations.

    For each combination of (concentration, mu, monotonicity), performs binary
    search over kappa to find the frontier where delta_total <= tolerance.

    This is an adaptive sampling strategy that uses feedback from simulations
    to efficiently locate the boundary between stable and unstable regions.

    Unlike grid and LHS sampling, frontier sampling executes runs directly via
    execute_fn because it needs immediate feedback to decide what to test next.

    Args:
        concentrations: Sequence of concentration values
        mus: Sequence of mu values
        monotonicities: Sequence of monotonicity values
        kappa_low: Initial lower bound for kappa
        kappa_high: Initial upper bound for kappa
        tolerance: Target tolerance for delta_total
        max_iterations: Maximum binary search iterations per cell
        execute_fn: Function that executes a run and returns delta_total.
                   Signature: (label, kappa, concentration, mu, monotonicity) -> Optional[Decimal]
                   Returns None if run failed to stabilize.

    Returns:
        None (calls execute_fn directly for side effects)
    """
    for concentration in concentrations:
        for mu in mus:
            for monotonicity in monotonicities:
                _run_frontier_cell(
                    concentration,
                    mu,
                    monotonicity,
                    kappa_low,
                    kappa_high,
                    tolerance,
                    max_iterations,
                    execute_fn,
                )


def _run_frontier_cell(
    concentration: Decimal,
    mu: Decimal,
    monotonicity: Decimal,
    kappa_low: Decimal,
    kappa_high: Decimal,
    tolerance: Decimal,
    max_iterations: int,
    execute_fn: Callable[[str, Decimal, Decimal, Decimal, Decimal], Decimal | None],
) -> None:
    """
    Binary search for frontier kappa for a single parameter cell.

    Strategy:
    1. Test kappa_low - if already stable, return
    2. Test and expand kappa_high until stable or max reached
    3. Binary search between low and high to find tightest stable kappa

    Args:
        concentration: Concentration parameter
        mu: Mu parameter
        monotonicity: Monotonicity parameter
        kappa_low: Lower bound for kappa
        kappa_high: Initial upper bound for kappa
        tolerance: Target tolerance for delta_total
        max_iterations: Maximum binary search iterations
        execute_fn: Function to execute run and get delta_total

    Returns:
        None (calls execute_fn directly for side effects)
    """
    # Test lower bound
    low_delta = execute_fn("low", kappa_low, concentration, mu, monotonicity)

    # If lower bound is already stable, we're done
    if low_delta is not None and low_delta <= tolerance:
        return

    # Find upper bound that is stable
    hi_kappa = kappa_high
    hi_delta = execute_fn("high", hi_kappa, concentration, mu, monotonicity)

    # Expand upper bound if needed (up to 4x original or kappa=128)
    while (hi_delta is None or hi_delta > tolerance) and hi_kappa < kappa_high * 4:
        hi_kappa = hi_kappa * Decimal("1.5")
        hi_delta = execute_fn("high", hi_kappa, concentration, mu, monotonicity)
        if hi_kappa > Decimal("128"):
            break

    # If upper bound is still unstable, give up
    if hi_delta is None or hi_delta > tolerance:
        return

    # Binary search between stable bounds
    low = kappa_low
    high = hi_kappa

    for _ in range(max_iterations):
        if high - low <= tolerance:
            break

        mid = (low + high) / 2
        mid_delta = execute_fn("mid", mid, concentration, mu, monotonicity)

        if mid_delta is None:
            # Unstable, search higher
            low = mid
        elif mid_delta <= tolerance:
            # Stable, search lower
            high = mid
        else:
            # Unstable, search higher
            low = mid
