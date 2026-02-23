"""Tests for bilancio.decision.factories — information-aware assessor creation."""

from decimal import Decimal

import pytest

from bilancio.decision.factories import create_assessor, observability_from_profile
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.information.levels import AccessLevel
from bilancio.information.noise import EstimationNoise, LagNoise, SampleNoise
from bilancio.information.profile import CategoryAccess, InformationProfile


class TestObservabilityFromProfile:
    """Test observability_from_profile mapping."""

    def test_perfect_access_returns_one(self):
        profile = InformationProfile()  # all PERFECT by default
        assert observability_from_profile(profile) == Decimal("1.0")

    def test_none_access_returns_zero(self):
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=AccessLevel.NONE)
        )
        assert observability_from_profile(profile) == Decimal("0.0")

    def test_sample_noise_returns_sample_rate(self):
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=SampleNoise(sample_rate=Decimal("0.7")),
            )
        )
        assert observability_from_profile(profile) == Decimal("0.7")

    def test_estimation_noise_returns_one_minus_error(self):
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=EstimationNoise(error_fraction=Decimal("0.3")),
            )
        )
        assert observability_from_profile(profile) == Decimal("0.7")

    def test_lag_noise_returns_decay(self):
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=LagNoise(lag_days=3),
            )
        )
        assert observability_from_profile(profile) == Decimal("0.7")

    def test_lag_noise_clamped_at_zero(self):
        """Large lag should not produce negative observability."""
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=LagNoise(lag_days=20),
            )
        )
        assert observability_from_profile(profile) == Decimal("0")

    def test_estimation_noise_clamped_at_zero(self):
        """Large error_fraction should not produce negative observability."""
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=EstimationNoise(error_fraction=Decimal("2.0")),
            )
        )
        assert observability_from_profile(profile) == Decimal("0")


class TestCreateAssessor:
    """Test create_assessor factory function."""

    def test_no_profile_returns_assessor_with_base_params(self):
        params = RiskAssessmentParams(initial_prior=Decimal("0.20"))
        assessor = create_assessor(params)
        assert isinstance(assessor, RiskAssessor)
        assert assessor.params.initial_prior == Decimal("0.20")
        assert assessor.params.default_observability == Decimal("1.0")

    def test_perfect_profile_preserves_observability(self):
        params = RiskAssessmentParams()
        profile = InformationProfile()  # all PERFECT
        assessor = create_assessor(params, profile)
        assert assessor.params.default_observability == Decimal("1.0")

    def test_none_profile_sets_zero_observability(self):
        params = RiskAssessmentParams()
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=AccessLevel.NONE)
        )
        assessor = create_assessor(params, profile)
        assert assessor.params.default_observability == Decimal("0.0")

    def test_noisy_profile_adjusts_observability(self):
        params = RiskAssessmentParams()
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=SampleNoise(sample_rate=Decimal("0.5")),
            )
        )
        assessor = create_assessor(params, profile)
        assert assessor.params.default_observability == Decimal("0.5")

    def test_does_not_mutate_base_params(self):
        """Factory must not modify the caller's params object."""
        params = RiskAssessmentParams(default_observability=Decimal("1.0"))
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=AccessLevel.NONE)
        )
        create_assessor(params, profile)
        assert params.default_observability == Decimal("1.0")  # unchanged

    def test_assessor_uses_prior_when_observability_zero(self):
        """With observability=0, estimate_default_prob always returns prior."""
        params = RiskAssessmentParams(initial_prior=Decimal("0.25"))
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=AccessLevel.NONE)
        )
        assessor = create_assessor(params, profile)

        # Feed it history — it should be ignored due to obs=0
        assessor.update_history(day=1, issuer_id="A1", defaulted=True)
        assessor.update_history(day=1, issuer_id="A2", defaulted=True)
        assessor.update_history(day=1, issuer_id="A3", defaulted=True)

        p = assessor.estimate_default_prob("A1", current_day=2)
        assert p == Decimal("0.25")  # always returns prior

    def test_assessor_partially_observes_with_noisy_profile(self):
        """With observability=0.5, estimate blends observed rate with prior."""
        params = RiskAssessmentParams(
            initial_prior=Decimal("0.20"),
            smoothing_alpha=Decimal("0"),  # no smoothing for exact test
        )
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=SampleNoise(sample_rate=Decimal("0.5")),
            )
        )
        assessor = create_assessor(params, profile)

        # 100% defaults observed
        for i in range(10):
            assessor.update_history(day=1, issuer_id=f"A{i}", defaulted=True)

        p = assessor.estimate_default_prob("X", current_day=2)
        # With obs=0.5: p = prior + 0.5 * (observed - prior)
        # observed = (0 + 10) / (0 + 10) = 1.0 (with alpha=0)
        # p = 0.20 + 0.5 * (1.0 - 0.20) = 0.20 + 0.40 = 0.60
        assert p == Decimal("0.60")
