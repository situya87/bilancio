"""Morris method for global sensitivity screening.

The Morris method (Morris 1991, improved by Campolongo et al. 2007)
efficiently identifies which input parameters have significant effects
on model output. It requires far fewer model evaluations than
variance-based methods (Sobol) while still detecting:

- mu*: overall importance (main effect)
- sigma: interaction/nonlinearity (if high, the parameter's effect
  depends on the values of other parameters)

This implementation is simulation-agnostic: the model is a callable
that takes a parameter dict and returns a scalar metric.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from bilancio.stats.types import MorrisResult


def morris_screening(
    model: Callable[[dict[str, float]], float],
    bounds: dict[str, tuple[float, float]],
    num_trajectories: int = 10,
    num_levels: int = 4,
    seed: int | None = None,
) -> list[MorrisResult]:
    """Morris method for parameter importance screening.

    Generates `num_trajectories` random trajectories through the parameter
    space. Each trajectory consists of k+1 points (where k = number of
    parameters), with each successive point differing from the previous
    in exactly one parameter. The elementary effect of parameter i is
    the output change divided by the input change.

    Parameters
    ----------
    model:
        Callable that takes a parameter dict (str -> float) and returns
        a scalar metric. This function is called (num_trajectories * (k+1))
        times total. The function should be deterministic for the same
        inputs (use fixed seed internally if the model is stochastic).
    bounds:
        Parameter bounds: {name: (lower, upper)}.
    num_trajectories:
        Number of Morris trajectories (r). More = better estimates.
        Typical: 10-50.
    num_levels:
        Number of grid levels (p). The step size is p / (2*(p-1)).
        Typical: 4.
    seed:
        RNG seed for trajectory generation.

    Returns
    -------
    List of MorrisResult, one per parameter, sorted by mu* descending
    (most important first).

    Notes
    -----
    Total model evaluations: num_trajectories * (k + 1)
    where k = len(bounds).

    For k=10 parameters and r=20 trajectories: 220 evaluations.
    Compare with Sobol which needs 1000+ for the same k.
    """
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}")
    if num_trajectories < 2:
        raise ValueError(f"num_trajectories must be >= 2, got {num_trajectories}")

    rng = random.Random(seed)
    param_names = list(bounds.keys())
    k = len(param_names)

    if k == 0:
        return []

    delta = num_levels / (2 * (num_levels - 1))

    # Collect elementary effects per parameter
    elementary_effects: dict[str, list[float]] = {name: [] for name in param_names}

    for _ in range(num_trajectories):
        trajectory = _generate_trajectory(
            param_names, bounds, num_levels, delta, rng
        )

        # Evaluate model at each trajectory point
        outputs: list[float] = []
        for point in trajectory:
            outputs.append(model(point))

        # Compute elementary effects
        for step_idx in range(k):
            prev_point = trajectory[step_idx]
            next_point = trajectory[step_idx + 1]

            # Find which parameter changed
            changed_param = None
            for name in param_names:
                if prev_point[name] != next_point[name]:
                    changed_param = name
                    break

            if changed_param is None:
                continue

            lo, hi = bounds[changed_param]
            param_range = hi - lo
            if param_range == 0:
                continue

            # Normalized elementary effect
            input_delta = (next_point[changed_param] - prev_point[changed_param]) / param_range
            if input_delta == 0:
                continue

            output_delta = outputs[step_idx + 1] - outputs[step_idx]
            ee = output_delta / input_delta
            elementary_effects[changed_param].append(ee)

    # Aggregate into Morris statistics
    results = []
    for name in param_names:
        effects = elementary_effects[name]
        if not effects:
            results.append(MorrisResult(
                parameter=name, mu=0.0, mu_star=0.0, sigma=0.0,
            ))
            continue

        n = len(effects)
        mu = sum(effects) / n
        mu_star = sum(abs(e) for e in effects) / n
        sigma = (
            (sum((e - mu) ** 2 for e in effects) / (n - 1)) ** 0.5
            if n > 1
            else 0.0
        )
        results.append(MorrisResult(
            parameter=name, mu=mu, mu_star=mu_star, sigma=sigma,
        ))

    # Sort by importance (mu* descending)
    results.sort(key=lambda r: r.mu_star, reverse=True)
    return results


def _generate_trajectory(
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    num_levels: int,
    delta: float,
    rng: random.Random,
) -> list[dict[str, float]]:
    """Generate one Morris trajectory (k+1 points, each step changes one param).

    Algorithm (Campolongo et al. 2007):
    1. Start from a random base point on the grid
    2. Randomly permute the parameter order
    3. For each parameter (in permuted order), step by +delta or -delta
    """
    k = len(param_names)

    # Random base point on the grid (values in [0, 1-delta] at grid levels)
    levels = [i / (num_levels - 1) for i in range(num_levels)]
    valid_starts = [lv for lv in levels if lv + delta <= 1.0 + 1e-9]
    if not valid_starts:
        valid_starts = [0.0]

    base_unit = {name: rng.choice(valid_starts) for name in param_names}

    # Random permutation of parameters
    order = list(range(k))
    rng.shuffle(order)

    # Random step direction per parameter (+delta or -delta)
    directions = [rng.choice([-1, 1]) for _ in range(k)]

    # Build trajectory
    trajectory: list[dict[str, float]] = []
    current = dict(base_unit)
    trajectory.append(_unit_to_real(current, param_names, bounds))

    for step in range(k):
        param_idx = order[step]
        param_name = param_names[param_idx]
        direction = directions[step]

        new_val = current[param_name] + direction * delta
        # Reflect if out of bounds
        if new_val > 1.0:
            new_val = current[param_name] - delta
        elif new_val < 0.0:
            new_val = current[param_name] + delta

        current = dict(current)
        current[param_name] = max(0.0, min(1.0, new_val))
        trajectory.append(_unit_to_real(current, param_names, bounds))

    return trajectory


def _unit_to_real(
    unit_point: dict[str, float],
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Map unit hypercube point [0,1]^k to real parameter space."""
    return {
        name: bounds[name][0] + unit_point[name] * (bounds[name][1] - bounds[name][0])
        for name in param_names
    }
