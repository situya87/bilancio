"""Concrete InstrumentValuer and VBTPricingModel implementations.

EVHoldValuer            — wraps RiskAssessor (delegation, zero new math)
CoverageRatioValuer     — uses a rating registry for default probabilities
CreditAdjustedVBTPricing — default VBT pricing: M = outside_mid_ratio × (1 − P_default)
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
        face: Decimal = ticket.face
        return (Decimal(1) - p) * face

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



# ── CreditAdjustedVBTPricing ─────────────────────────────────────


@dataclass(frozen=True)
class CreditAdjustedVBTPricing:
    """Default VBT pricing: M = outside_mid_ratio × (1 − P_default).

    The outside_mid_ratio is a property of this model — it represents
    the VBT's valuation anchor, not an externally injected parameter.

    When mid_sensitivity < 1, M is blended toward its initial value
    (computed from initial_prior), damping the response to observed defaults.

    When spread_sensitivity > 0, the base spread widens proportionally
    to the observed default probability.
    """

    outside_mid_ratio: Decimal = Decimal("0.75")
    mid_sensitivity: Decimal = Decimal("1.0")
    spread_sensitivity: Decimal = Decimal("0.0")

    def compute_mid(self, p_default: Decimal, initial_prior: Decimal) -> Decimal:
        """Compute credit-adjusted mid price.

        Formula: blend initial_M toward raw_M based on mid_sensitivity.
        - raw_M = outside_mid_ratio × (1 - p_default)
        - initial_M = outside_mid_ratio × (1 - initial_prior)
        - result = initial_M + mid_sensitivity × (raw_M - initial_M)
        """
        raw_M = self.outside_mid_ratio * (Decimal(1) - p_default)
        initial_M = self.outside_mid_ratio * (Decimal(1) - initial_prior)
        return initial_M + self.mid_sensitivity * (raw_M - initial_M)

    def compute_spread(self, base_spread: Decimal, p_default: Decimal) -> Decimal:
        """Compute adjusted spread.

        When spread_sensitivity > 0:
            spread = base_spread × (1 + spread_sensitivity × p_default)
        Otherwise returns base_spread unchanged.
        """
        if self.spread_sensitivity > 0:
            return base_spread * (Decimal(1) + self.spread_sensitivity * p_default)
        return base_spread


__all__ = ["EVHoldValuer", "CoverageRatioValuer", "CreditAdjustedVBTPricing"]
