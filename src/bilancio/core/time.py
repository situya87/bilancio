"""Time handling utilities for bilancio."""

from dataclasses import dataclass


@dataclass
class TimeCoordinate:
    """Represents a point in time."""

    t: float


@dataclass
class TimeInterval:
    """Represents an interval of time with start and end coordinates."""

    start: TimeCoordinate
    end: TimeCoordinate


def now() -> TimeCoordinate:
    """Return the current time coordinate."""
    return TimeCoordinate(0.0)
