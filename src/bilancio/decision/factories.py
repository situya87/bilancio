"""Factory for creating information-aware RiskAssessors.

Maps an ``InformationProfile`` to ``RiskAssessmentParams`` adjustments,
so that traders with limited information access have proportionally
reduced ``default_observability`` — they rely more on their prior and
less on observed defaults.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor

if TYPE_CHECKING:
    from bilancio.information.profile import InformationProfile
    from bilancio.information.service import InformationService


def observability_from_profile(profile: InformationProfile) -> Decimal:
    """Derive ``default_observability`` from an InformationProfile.

    Mapping rules (based on ``counterparty_default_history`` access):

    * **PERFECT** → 1.0  (full tracking, current default)
    * **NONE**    → 0.0  (always uses prior, ignores observed defaults)
    * **NOISY**   → derived from noise config:
      - ``SampleNoise``      → ``sample_rate``  (fraction of events seen)
      - ``EstimationNoise``  → ``max(0, 1 - error_fraction)``
      - ``LagNoise``         → ``max(0, 1 - 0.1 × lag_days)``
      - other                → 0.5  (moderate degradation)
    """
    from bilancio.information.levels import AccessLevel
    from bilancio.information.noise import EstimationNoise, LagNoise, SampleNoise

    access = profile.counterparty_default_history

    if access.level == AccessLevel.PERFECT:
        return Decimal("1.0")
    if access.level == AccessLevel.NONE:
        return Decimal("0.0")

    # NOISY — derive from noise config
    noise = access.noise
    if isinstance(noise, SampleNoise):
        return noise.sample_rate
    if isinstance(noise, EstimationNoise):
        return max(Decimal("0"), Decimal("1") - noise.error_fraction)
    if isinstance(noise, LagNoise):
        return max(Decimal("0"), Decimal("1") - Decimal("0.1") * Decimal(noise.lag_days))

    # Unknown noise type — moderate default
    return Decimal("0.5")


def create_assessor(
    base_params: RiskAssessmentParams,
    profile: InformationProfile | None = None,
    information_service: InformationService | None = None,
) -> RiskAssessor:
    """Create a RiskAssessor configured from an InformationProfile.

    When *profile* is ``None`` and *information_service* is ``None``,
    returns a RiskAssessor with the base parameters unchanged
    (backward-compatible default).

    When a profile is provided, adjusts ``default_observability``
    based on the profile's ``counterparty_default_history`` access
    level and noise configuration.

    When an *information_service* is provided, attaches it to the
    assessor's BeliefTracker so that ``estimate_default_prob()``
    delegates to the service instead of using internal payment history.

    Args:
        base_params: Baseline risk assessment parameters.
        profile: Optional information profile.  When provided,
            ``default_observability`` is overridden.
        information_service: Optional InformationService instance.
            When provided, the assessor's BeliefTracker will delegate
            default probability queries to it.

    Returns:
        A configured :class:`RiskAssessor` instance.
    """
    if profile is None and information_service is None:
        return RiskAssessor(base_params)

    # Create adjusted params (RiskAssessmentParams is frozen)
    if profile is not None:
        adjusted = replace(base_params, default_observability=observability_from_profile(profile))
    else:
        adjusted = base_params

    assessor = RiskAssessor(adjusted)
    if information_service is not None:
        assessor.belief_tracker.information_service = information_service
    return assessor
