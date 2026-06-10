"""Plugin protocol for the v2 day cycle.

A phase plugin observes the ledger and applies ledger operations. The engine
runs plugins in a fixed order each day; a plugin reports whether it produced
impactful activity (used by the stability stop rule). Subsystems (settlement,
interbank clearing, and later banking/dealer/lender behavior) are all plugins
against this one interface — nothing is special-cased in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bilancio_v2.ledger import Ledger
from bilancio_v2.policy import CapabilityMatrix


@dataclass(frozen=True)
class RunContext:
    """Per-run configuration shared by all plugins."""

    policy: CapabilityMatrix
    default_mode: str  # "fail-fast" | "expel-agent"


class PhasePlugin(Protocol):
    """One phase of the daily cycle."""

    name: str

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        """Execute the phase for ``ledger.day``. Returns whether it was impactful."""
        ...
