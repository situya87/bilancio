"""Concrete InstrumentValuer implementations.

EVHoldValuer    — wraps RiskAssessor (delegation, zero new math)
CoverageRatioValuer — uses a rating registry for default probabilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bilancio.information.estimates import Estimate

if TYPE_CHECKING:
    from bilancio.dealer.risk_assessment import RiskAssessor


# ── EVHoldValuer ───────────────────────────────────────────────────


@dataclass
class EVHoldValuer:
    """Wraps RiskAssessor.expected_value() behind InstrumentValuer.

    This is pure delegation — no new computation. It exists so that
    callers holding an InstrumentValuer can transparently get the same
    numbers that RiskAssessor already produces.
    """

    risk_assessor: RiskAssessor
    estimator_id: str = "ev_hold_valuer"

    # ── InstrumentValuer interface ─────────────────────────────────

    def value_decimal(self, ticket: Any, day: int) -> Decimal:
        """Fast path: bare Decimal from RiskAssessor.expected_value()."""
        return self.risk_assessor.expected_value(ticket, day)

    def value(self, ticket: Any, day: int) -> Estimate:
        """Full path: Estimate with provenance."""
        return self.risk_assessor.expected_value_detail(
            ticket, day, estimator_id=self.estimator_id,
        )


# ── CoverageRatioValuer ───────────────────────────────────────────


@dataclass
class CoverageRatioValuer:
    """Values instruments using externally-published ratings.

    Looks up the issuer's default probability from a rating registry
    (e.g., populated by a RatingAgency) and computes:

        EV = (1 - P_default) × face

    Falls back to ``fallback_prior`` when the issuer has no rating.
    """

    rating_registry: dict[str, Decimal] = field(default_factory=dict)
    fallback_prior: Decimal = Decimal("0.15")
    estimator_id: str = "coverage_ratio_valuer"

    # ── helpers ────────────────────────────────────────────────────

    def _p_default(self, issuer_id: str) -> Decimal:
        return self.rating_registry.get(str(issuer_id), self.fallback_prior)

    # ── InstrumentValuer interface ─────────────────────────────────

    def value_decimal(self, ticket: Any, day: int) -> Decimal:
        """Fast path: (1 - P_default) × face."""
        p = self._p_default(ticket.issuer_id)
        return (Decimal(1) - p) * ticket.face

    def value(self, ticket: Any, day: int) -> Estimate:
        """Full path: Estimate with provenance."""
        p = self._p_default(ticket.issuer_id)
        ev = (Decimal(1) - p) * ticket.face
        used_fallback = str(ticket.issuer_id) not in self.rating_registry

        return Estimate(
            value=ev,
            estimator_id=self.estimator_id,
            target_id=str(getattr(ticket, "id", "")),
            target_type="instrument",
            estimation_day=day,
            method="coverage_ratio_ev",
            inputs={
                "p_default": str(p),
                "face_value": str(ticket.face),
                "used_fallback": used_fallback,
            },
            metadata={
                "issuer_id": str(ticket.issuer_id),
            },
        )


__all__ = ["EVHoldValuer", "CoverageRatioValuer"]
