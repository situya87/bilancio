"""Noise configuration dataclasses for imperfect information.

Each class parameterizes a different type of observation imperfection.
Noise configs are attached to CategoryAccess when level == NOISY.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class LagNoise:
    """Observer sees the true value from *lag_days* ago.

    For balance-sheet data (where we don't store snapshots), this is
    approximated as estimation error proportional to lag.  For event-based
    data (history), events are filtered to day <= current_day - lag_days.
    """

    lag_days: int = 1

    def __post_init__(self) -> None:
        if self.lag_days < 0:
            raise ValueError("lag_days must be non-negative")


@dataclass(frozen=True)
class SampleNoise:
    """Observer sees only a fraction of events / items.

    Each event is independently included with probability *sample_rate*.
    """

    sample_rate: Decimal = Decimal("0.7")

    def __post_init__(self) -> None:
        if not (Decimal("0") < self.sample_rate <= Decimal("1")):
            raise ValueError("sample_rate must be in (0, 1]")


@dataclass(frozen=True)
class EstimationNoise:
    """Observer sees value ± Gaussian error band.

    The returned value is ``value + N(0, σ)`` where
    ``σ = error_fraction × |true_value|``, clamped to >= 0.
    """

    error_fraction: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        if self.error_fraction < Decimal("0"):
            raise ValueError("error_fraction must be non-negative")


@dataclass(frozen=True)
class AggregateOnlyNoise:
    """Observer can see totals / counts but not individual breakdowns.

    The service returns an aggregate int instead of a detailed dict.
    """

    pass


@dataclass(frozen=True)
class BilateralOnlyNoise:
    """Observer can only see own interactions with the counterparty.

    The service filters history to events where observer_id is a party.
    """

    pass


# Union of all noise types for type annotations
NoiseConfig = Union[
    LagNoise,
    SampleNoise,
    EstimationNoise,
    AggregateOnlyNoise,
    BilateralOnlyNoise,
]
