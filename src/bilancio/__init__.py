"""Bilancio - Financial modeling and analysis library."""

__version__ = "0.1.0"

# Core imports
from bilancio.core.errors import (
    BilancioError,
    ValidationError,
)
from bilancio.core.time import TimeCoordinate, TimeInterval, now

__all__ = [
    "__version__",
    # Core time utilities
    "TimeCoordinate",
    "TimeInterval",
    "now",
    # Core exceptions
    "BilancioError",
    "ValidationError",
]
