"""Factory for building decision profiles from configuration dicts."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.decision.profiles import (
    LenderProfile,
    RatingProfile,
    TraderProfile,
    VBTProfile,
)

# Registry: profile_type string → dataclass constructor
PROFILE_REGISTRY: dict[str, type] = {
    "trader": TraderProfile,
    "vbt": VBTProfile,
    "lender": LenderProfile,
    "rating": RatingProfile,
}


def build_profile(profile_type: str, params: dict[str, Any]) -> Any:
    """Build a profile dataclass from a type string and parameter dict.

    String values that look like numbers are converted to Decimal for
    compatibility with the profile dataclasses.

    Args:
        profile_type: One of "trader", "vbt", "lender", "rating".
        params: Key-value pairs matching the profile's constructor args.

    Returns:
        An instance of the corresponding profile dataclass.

    Raises:
        ValueError: If profile_type is unknown.
        TypeError: If params don't match the profile's constructor.
    """
    cls = PROFILE_REGISTRY.get(profile_type)
    if cls is None:
        raise ValueError(
            f"Unknown profile_type '{profile_type}'. "
            f"Available: {sorted(PROFILE_REGISTRY.keys())}"
        )

    # Convert string values to Decimal where appropriate
    converted: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str):
            try:
                converted[k] = Decimal(v)
            except Exception:
                converted[k] = v  # Keep as string if not a valid Decimal
        else:
            converted[k] = v

    return cls(**converted)
