"""Channel abstractions that derive NoiseConfig from observable properties.

A *channel* describes how an agent physically obtains a piece of
information — e.g., from its own records, from market activity, from a
network of peers, or from an institutional source.  Each channel has
measurable properties (sample size, staleness, coverage) that map
deterministically to a ``NoiseConfig``.

Channels are a **construction-time helper**: they produce a ``NoiseConfig``
that is then stored in a ``CategoryAccess``.  The ``InformationService``
never sees a channel — it only sees the resulting ``NoiseConfig``.

Usage
-----
>>> from bilancio.information.channels import SelfDerivedChannel, derive_noise
>>> noise = derive_noise(SelfDerivedChannel(sample_size=100))
>>> noise
EstimationNoise(error_fraction=Decimal('0.10'))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Union

from bilancio.information.levels import AccessLevel
from bilancio.information.noise import EstimationNoise, LagNoise, NoiseConfig, SampleNoise
from bilancio.information.profile import CategoryAccess


# ── Channel dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SelfDerivedChannel:
    """Information derived from the agent's own records.

    Quality depends on how many observations the agent has accumulated.
    Maps to ``EstimationNoise(1 / sqrt(sample_size))``.
    """

    sample_size: int = 20

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError("sample_size must be >= 1")


@dataclass(frozen=True)
class MarketDerivedChannel:
    """Information inferred from market activity (prices, volumes).

    Quality depends on market thickness (number of recent trades) and
    how stale the latest data is.  When staleness dominates, maps to
    ``LagNoise``; otherwise maps to ``EstimationNoise(1 / sqrt(thickness))``.
    """

    market_thickness: int = 50
    staleness_days: int = 0

    def __post_init__(self) -> None:
        if self.market_thickness < 1:
            raise ValueError("market_thickness must be >= 1")
        if self.staleness_days < 0:
            raise ValueError("staleness_days must be non-negative")


@dataclass(frozen=True)
class NetworkDerivedChannel:
    """Information gathered from a network of peers or counterparties.

    Quality depends on what fraction of the network the agent can observe.
    Maps to ``SampleNoise(coverage)``.
    """

    coverage: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        if not (Decimal("0") < self.coverage <= Decimal("1")):
            raise ValueError("coverage must be in (0, 1]")


@dataclass(frozen=True)
class InstitutionalChannel:
    """Information from institutional sources (rating agencies, regulators).

    Has both staleness (publication lag) and coverage (not all entities
    rated).  The *dominant* degradation mode determines the noise type:
    staleness ⇒ ``LagNoise``, coverage gap ⇒ ``SampleNoise``.
    """

    staleness_days: int = 0
    coverage: Decimal = Decimal("0.8")

    def __post_init__(self) -> None:
        if self.staleness_days < 0:
            raise ValueError("staleness_days must be non-negative")
        if not (Decimal("0") < self.coverage <= Decimal("1")):
            raise ValueError("coverage must be in (0, 1]")


#: Union of all channel types.
Channel = Union[
    SelfDerivedChannel,
    MarketDerivedChannel,
    NetworkDerivedChannel,
    InstitutionalChannel,
]


# ── Error-per-lag-day constant (matches service.py) ─────────────────────
_LAG_ERROR_PER_DAY = Decimal("0.05")

# ── Clamping bounds for 1/sqrt(n) ───────────────────────────────────────
_MIN_ERROR = Decimal("0.01")
_MAX_ERROR = Decimal("1.0")


def _inv_sqrt(n: int) -> Decimal:
    """Return ``1 / sqrt(n)`` as a Decimal, clamped to [0.01, 1.0]."""
    # str() avoids the imprecise float→Decimal conversion path
    raw = Decimal(str(1.0 / math.sqrt(n)))
    # Round to 2 decimal places for clean values
    rounded = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return max(_MIN_ERROR, min(_MAX_ERROR, rounded))


# ── Core derivation ─────────────────────────────────────────────────────


def derive_noise(channel: Channel) -> NoiseConfig:
    """Derive a ``NoiseConfig`` from a channel's observable properties.

    Parameters
    ----------
    channel:
        One of the four channel types.

    Returns
    -------
    NoiseConfig
        The noise configuration implied by the channel's properties.

    Notes
    -----
    For channels with multiple degradation modes (``MarketDerivedChannel``,
    ``InstitutionalChannel``), the *dominant* mode is selected.  When both
    modes produce equal error, staleness (``LagNoise``) wins the tie.

    Examples
    --------
    >>> derive_noise(SelfDerivedChannel(sample_size=100))
    EstimationNoise(error_fraction=Decimal('0.10'))

    >>> derive_noise(MarketDerivedChannel(staleness_days=3))
    LagNoise(lag_days=3)

    >>> derive_noise(NetworkDerivedChannel(coverage=Decimal("0.7")))
    SampleNoise(sample_rate=Decimal('0.7'))
    """
    if isinstance(channel, SelfDerivedChannel):
        return EstimationNoise(error_fraction=_inv_sqrt(channel.sample_size))

    if isinstance(channel, MarketDerivedChannel):
        lag_error = channel.staleness_days * _LAG_ERROR_PER_DAY
        thickness_error = _inv_sqrt(channel.market_thickness)
        if channel.staleness_days > 0 and lag_error >= thickness_error:
            return LagNoise(lag_days=channel.staleness_days)
        return EstimationNoise(error_fraction=_inv_sqrt(channel.market_thickness))

    if isinstance(channel, NetworkDerivedChannel):
        return SampleNoise(sample_rate=channel.coverage)

    if isinstance(channel, InstitutionalChannel):
        lag_error = channel.staleness_days * _LAG_ERROR_PER_DAY
        coverage_gap = Decimal("1") - channel.coverage
        if channel.staleness_days > 0 and lag_error >= coverage_gap:
            return LagNoise(lag_days=channel.staleness_days)
        return SampleNoise(sample_rate=channel.coverage)

    raise TypeError(f"Unknown channel type: {type(channel).__name__}")


# ── Convenience factory ─────────────────────────────────────────────────


def category_from_channel(channel: Channel) -> CategoryAccess:
    """Create a ``CategoryAccess(NOISY, ...)`` from a channel.

    This is a shorthand for::

        CategoryAccess(AccessLevel.NOISY, derive_noise(channel))

    Parameters
    ----------
    channel:
        Any channel type.

    Returns
    -------
    CategoryAccess
        With level=NOISY and noise derived from the channel.
    """
    return CategoryAccess(level=AccessLevel.NOISY, noise=derive_noise(channel))
