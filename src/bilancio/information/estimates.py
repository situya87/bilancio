"""Estimate dataclass: first-class valuation objects with provenance.

Every belief-producing function can return an Estimate that records
what was estimated, by whom, when, and using which method — making
valuations traceable without changing settlement logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

_VALID_TARGET_TYPES = frozenset({"agent", "instrument", "system"})


@dataclass(frozen=True)
class Estimate:
    """A single valuation estimate with full provenance."""

    value: Decimal
    estimator_id: str
    target_id: str
    target_type: str
    estimation_day: int
    method: str
    inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_type not in _VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {_VALID_TARGET_TYPES}, "
                f"got {self.target_type!r}"
            )
