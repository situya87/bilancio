"""Tests for Estimate provenance from companion _detail methods."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from bilancio.information.estimates import Estimate
from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams


class TestRiskAssessorDetail:
    """Test RiskAssessor estimate_default_prob_detail and expected_value_detail."""

    def test_no_history_returns_prior(self):
        """With no payment history, should use initial_prior and mark used_prior=True."""
        assessor = RiskAssessor(RiskAssessmentParams(initial_prior=Decimal("0.15")))
        est = assessor.estimate_default_prob_detail("firm_1", current_day=5)
        assert isinstance(est, Estimate)
        assert est.value == Decimal("0.15")
        assert est.method == "bayesian_default_prob"
        assert est.target_type == "agent"
        assert est.target_id == "firm_1"
        assert est.inputs["used_prior"] is True
        assert est.inputs["total_observations"] == 0

    def test_with_history(self):
        """With payment history, should use Bayesian estimate."""
        assessor = RiskAssessor(RiskAssessmentParams(lookback_window=10))
        # Add some history: 2 defaults out of 5 payments
        for i in range(5):
            assessor.update_history(day=i + 1, issuer_id="firm_1", defaulted=(i < 2))

        est = assessor.estimate_default_prob_detail("firm_1", current_day=5)
        assert isinstance(est, Estimate)
        assert est.inputs["used_prior"] is False
        assert est.inputs["defaults_count"] == 2
        assert est.inputs["total_observations"] == 5

    def test_value_matches_original(self):
        """Detail method value must match original method exactly."""
        assessor = RiskAssessor(RiskAssessmentParams(lookback_window=10))
        for i in range(3):
            assessor.update_history(day=i + 1, issuer_id="firm_1", defaulted=(i == 0))

        original = assessor.estimate_default_prob("firm_1", 5)
        detailed = assessor.estimate_default_prob_detail("firm_1", 5)
        assert detailed.value == original

    def test_custom_estimator_id(self):
        """estimator_id should propagate to the Estimate."""
        assessor = RiskAssessor(RiskAssessmentParams())
        est = assessor.estimate_default_prob_detail("firm_1", 5, estimator_id="lender_1")
        assert est.estimator_id == "lender_1"

    def test_expected_value_detail_nests_p_default(self):
        """expected_value_detail should nest the p_default estimate in inputs."""
        from bilancio.dealer.models import Ticket

        assessor = RiskAssessor(RiskAssessmentParams())
        ticket = Ticket(
            id="t1",
            issuer_id="firm_1",
            owner_id="dealer_1",
            face=Decimal("100"),
            maturity_day=10,
            bucket_id="b1",
        )
        est = assessor.expected_value_detail(ticket, current_day=5)
        assert isinstance(est, Estimate)
        assert est.method == "ev_hold"
        assert est.target_type == "instrument"
        assert est.target_id == "t1"
        # The nested estimate
        nested = est.inputs["p_default_estimate"]
        assert isinstance(nested, Estimate)
        assert nested.method == "bayesian_default_prob"

    def test_expected_value_detail_matches_original(self):
        """expected_value_detail value must match original expected_value."""
        from bilancio.dealer.models import Ticket

        assessor = RiskAssessor(RiskAssessmentParams())
        ticket = Ticket(
            id="t1",
            issuer_id="firm_1",
            owner_id="dealer_1",
            face=Decimal("100"),
            maturity_day=10,
            bucket_id="b1",
        )
        original = assessor.expected_value(ticket, 5)
        detailed = assessor.expected_value_detail(ticket, 5)
        assert detailed.value == original


class TestRatingEstimate:
    """Test _compute_rating with return_estimate=True."""

    def _make_mock_system(self):
        """Create a minimal mock system for rating tests."""
        system = MagicMock()
        agent_mock = MagicMock(defaulted=False, asset_ids=[], liability_ids=[])
        system.state.agents = {"firm_1": agent_mock}
        system.state.contracts = {}
        system.state.defaulted_agent_ids = set()
        system.state.dealer_subsystem = None
        system.state.rating_registry = {}
        return system

    def test_returns_estimate_when_requested(self):
        """_compute_rating(return_estimate=True) should return an Estimate."""
        from bilancio.engines.rating import _compute_rating
        from bilancio.decision.profiles import RatingProfile

        system = self._make_mock_system()
        profile = RatingProfile()
        result = _compute_rating(None, "firm_1", 5, profile, system, return_estimate=True)
        assert isinstance(result, Estimate)
        assert result.method == "coverage_ratio_plus_history"
        assert result.target_id == "firm_1"
        assert result.target_type == "agent"
        assert result.estimator_id == "rating_agency"

    def test_returns_decimal_by_default(self):
        """_compute_rating(return_estimate=False) should return plain Decimal."""
        from bilancio.engines.rating import _compute_rating
        from bilancio.decision.profiles import RatingProfile

        system = self._make_mock_system()
        profile = RatingProfile()
        result = _compute_rating(None, "firm_1", 5, profile, system)
        assert isinstance(result, Decimal)

    def test_estimate_value_matches_decimal(self):
        """Estimate.value should equal the Decimal result."""
        from bilancio.engines.rating import _compute_rating
        from bilancio.decision.profiles import RatingProfile

        system = self._make_mock_system()
        profile = RatingProfile()
        decimal_result = _compute_rating(None, "firm_1", 5, profile, system, return_estimate=False)
        estimate_result = _compute_rating(None, "firm_1", 5, profile, system, return_estimate=True)
        assert estimate_result.value == decimal_result

    def test_estimate_captures_inputs(self):
        """Estimate should capture balance sheet and history inputs."""
        from bilancio.engines.rating import _compute_rating
        from bilancio.decision.profiles import RatingProfile

        system = self._make_mock_system()
        profile = RatingProfile()
        est = _compute_rating(None, "firm_1", 5, profile, system, return_estimate=True)
        assert "bs_score" in est.inputs
        assert "hist_score" in est.inputs
        assert "combined_before_bias" in est.inputs
        assert "balance_sheet_weight" in est.metadata
        assert "conservatism_bias" in est.metadata


class TestInformationServiceDetail:
    """Test InformationService.get_default_probability_detail."""

    def _make_service(self, access_level):
        """Helper to create an InformationService with a specific access level."""
        from bilancio.information.service import InformationService
        from bilancio.information.profile import InformationProfile, CategoryAccess
        from bilancio.information.levels import AccessLevel

        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=access_level),
        )

        system = MagicMock()
        system.state.agents = {
            "firm_1": MagicMock(defaulted=False, asset_ids=[], liability_ids=[]),
            "firm_2": MagicMock(defaulted=False, asset_ids=[], liability_ids=[]),
        }
        system.state.contracts = {}
        system.state.defaulted_agent_ids = set()
        system.state.dealer_subsystem = None
        system.state.rating_registry = {}

        return InformationService(system, profile, observer_id="lender_1")

    def test_perfect_access_returns_estimate(self):
        from bilancio.information.levels import AccessLevel

        service = self._make_service(AccessLevel.PERFECT)
        est = service.get_default_probability_detail("firm_1", day=5)
        assert isinstance(est, Estimate)
        assert est.target_id == "firm_1"
        assert est.estimator_id == "lender_1"
        assert est.inputs["channel_source"] == "system_heuristic"
        assert est.inputs["access_level"] == "perfect"

    def test_none_access_returns_none(self):
        from bilancio.information.levels import AccessLevel

        service = self._make_service(AccessLevel.NONE)
        result = service.get_default_probability_detail("firm_1", day=5)
        assert result is None

    def test_channel_source_tracks_dealer(self):
        """When dealer risk assessor exists, channel_source should be 'dealer_risk_assessor'."""
        from bilancio.information.service import InformationService
        from bilancio.information.profile import InformationProfile, CategoryAccess
        from bilancio.information.levels import AccessLevel

        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(level=AccessLevel.PERFECT),
        )

        # Mock system with dealer subsystem having a risk assessor
        mock_assessor = MagicMock()
        mock_assessor.estimate_default_prob.return_value = Decimal("0.20")

        system = MagicMock()
        system.state.agents = {
            "firm_1": MagicMock(defaulted=False),
        }
        system.state.dealer_subsystem = MagicMock()
        system.state.dealer_subsystem.risk_assessor = mock_assessor
        system.state.rating_registry = {}

        service = InformationService(system, profile, observer_id="lender_1")
        est = service.get_default_probability_detail("firm_1", day=5)
        assert est is not None
        assert est.inputs["channel_source"] == "dealer_risk_assessor"


class TestSystemEstimateLogging:
    """Test System.log_estimate and State.estimate_log."""

    def test_logging_enabled_appends(self):
        from bilancio.engines.system import System

        sys = System()
        sys.state.estimate_logging_enabled = True

        est = Estimate(
            value=Decimal("0.15"),
            estimator_id="test",
            target_id="x",
            target_type="agent",
            estimation_day=1,
            method="test",
        )
        sys.log_estimate(est)
        assert len(sys.state.estimate_log) == 1
        assert sys.state.estimate_log[0] is est

    def test_logging_disabled_skips(self):
        from bilancio.engines.system import System

        sys = System()
        sys.state.estimate_logging_enabled = False

        est = Estimate(
            value=Decimal("0.15"),
            estimator_id="test",
            target_id="x",
            target_type="agent",
            estimation_day=1,
            method="test",
        )
        sys.log_estimate(est)
        assert len(sys.state.estimate_log) == 0

    def test_log_defaults_to_empty(self):
        from bilancio.engines.system import System, State

        state = State()
        assert state.estimate_log == []
        assert state.estimate_logging_enabled is False


class TestLendingEstimateLogging:
    """Test _estimate_default_probs with log_estimates=True."""

    def test_estimates_logged_when_enabled(self):
        """When log_estimates=True and dealer assessor exists, estimates should be logged."""
        from bilancio.engines.lending import _estimate_default_probs
        from bilancio.engines.system import System
        from bilancio.dealer.risk_assessment import RiskAssessor, RiskAssessmentParams

        sys = System()
        sys.state.estimate_logging_enabled = True

        # Add a mock agent
        agent = MagicMock()
        agent.defaulted = False
        sys.state.agents["firm_1"] = agent

        # Set up dealer subsystem with risk assessor
        assessor = RiskAssessor(RiskAssessmentParams())
        dealer_sub = MagicMock()
        dealer_sub.risk_assessor = assessor
        sys.state.dealer_subsystem = dealer_sub

        probs = _estimate_default_probs(sys, current_day=5, log_estimates=True)

        assert "firm_1" in probs
        assert len(sys.state.estimate_log) == 1
        assert sys.state.estimate_log[0].estimator_id == "lender"
        assert sys.state.estimate_log[0].target_id == "firm_1"

    def test_estimates_not_logged_when_disabled(self):
        """When log_estimates=False, no estimates should be logged."""
        from bilancio.engines.lending import _estimate_default_probs
        from bilancio.engines.system import System

        sys = System()
        sys.state.estimate_logging_enabled = True  # Enabled on system, but log_estimates=False

        agent = MagicMock()
        agent.defaulted = False
        sys.state.agents["firm_1"] = agent

        # Use a mock assessor because the non-detail path calls
        # estimate_default_prob(agent_id) without current_day.
        mock_assessor = MagicMock()
        mock_assessor.estimate_default_prob.return_value = Decimal("0.10")
        dealer_sub = MagicMock()
        dealer_sub.risk_assessor = mock_assessor
        sys.state.dealer_subsystem = dealer_sub

        probs = _estimate_default_probs(sys, current_day=5, log_estimates=False)

        assert "firm_1" in probs
        assert len(sys.state.estimate_log) == 0
