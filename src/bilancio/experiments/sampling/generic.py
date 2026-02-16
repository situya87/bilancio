"""Generic parameter sampling over arbitrary named dimensions.

These helpers work with ``dict[str, Decimal]`` parameter maps so that
sampling is scenario-agnostic — no hardcoded ``(kappa, c, mu, mono)``
tuples.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from decimal import Decimal
from itertools import product


def generate_grid_generic(
    dimensions: dict[str, Sequence[Decimal]],
) -> Iterator[dict[str, Decimal]]:
    """Cartesian product over named dimensions.

    Args:
        dimensions: Mapping of dimension name to its list of values.

    Yields:
        Dicts mapping each dimension name to one selected value.
    """
    names = list(dimensions.keys())
    if not names:
        return

    value_lists = [list(dimensions[n]) for n in names]
    for values in product(*value_lists):
        yield dict(zip(names, values, strict=True))


def generate_lhs_generic(
    count: int,
    dimensions: dict[str, tuple[Decimal, Decimal]],
    seed: int,
) -> Iterator[dict[str, Decimal]]:
    """Latin Hypercube Sampling over named dimensions.

    Args:
        count: Number of samples to draw.
        dimensions: Mapping of dimension name to ``(low, high)`` bounds.
        seed: PRNG seed for reproducibility.

    Yields:
        Dicts mapping each dimension name to a sampled value.
    """
    if count <= 0:
        return

    rng = random.Random(seed + 7919)

    names = list(dimensions.keys())
    axes: list[list[Decimal]] = []
    for name in names:
        low, high = dimensions[name]
        axes.append(_lhs_axis(count, (low, high), rng))

    # Shuffle each axis independently to decorrelate dimensions
    for axis in axes:
        rng.shuffle(axis)

    for idx in range(count):
        yield {name: axes[i][idx] for i, name in enumerate(names)}


def _lhs_axis(
    count: int,
    bounds: tuple[Decimal, Decimal],
    rng: random.Random,
) -> list[Decimal]:
    """Sample a single dimension via stratified uniform draws.

    Returns samples in stratum order (caller is responsible for shuffling).
    """
    low, high = bounds
    samples: list[Decimal] = []
    for stratum in range(count):
        a = Decimal(stratum) / Decimal(count)
        b = Decimal(stratum + 1) / Decimal(count)
        u = Decimal(str(rng.random()))
        frac = a + (b - a) * u
        samples.append(low + (high - low) * frac)
    return samples
