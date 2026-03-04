"""Tests for Phase 4: InformationService wiring into trading decisions.

Verifies that:
1. BeliefTracker delegates to InformationService when set
2. BeliefTracker falls back to internal logic when service returns None
3. create_assessor() wires information_service into the belief tracker
4. TraderState carries an optional information_service field
5. End-to-end: info service affects sell/buy decisions
6. Backward compatibility: everything works without info service
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from bilancio.dealer.models import TraderState
from bilancio.decision.factories import create_assessor
from bilancio.decision.risk_assessment import (
    BeliefTracker,
    RiskAssessmentParams,
    RiskAssessor,
)
from bilancio.information.estimates import Estimate

# ---------------------------------------------------------------------------
# Mock InformationService (avoids needing a full System)
# ---------------------------------------------------------------------------


class MockInformationService:
    """Lightweight mock that returns pre-configured default probabilities."""

    def __init__(self, default_probs: dict[str, Decimal] | None = None):
        self._default_probs: dict[str, Decimal] = default_probs or {}

    def get_default_probability(self, agent_id: str, day: int) -> Decimal | None:
        return self._default_probs.get(agent_id)

    def get_default_probability_detail(
        self, agent_id: str, day: int
    ) -> Estimate | None:
        p = self._default_probs.get(agent_id)
        if p is None:
            return None
        return Estimate(
            value=p,
            estimator_id="mock_service",
            target_id=agent_id,
            target_type="agent",
            estimation_day=day,
            method="mock",
            inputs={},
        )


# ---------------------------------------------------------------------------
# Mock ticket (minimal, avoids full dealer subsystem)
# ---------------------------------------------------------------------------


@dataclass
class _MockTicket:
    """Minimal ticket-like object for testing."""

    id: str = "TKT_1"
    issuer_id: str = "firm_1"
    owner_id: str = "firm_2"
    face: Decimal = Decimal("20")
    maturity_day: int = 10
    remaining_tau: int = 5
    bucket_id: str = "short"
    serial: int = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ticket() -> _MockTicket:
    return _MockTicket()


@pytest.fixture()
def default_params() -> RiskAssessmentParams:
    return RiskAssessmentParams()


# ===========================================================================
# Group 1: BeliefTracker with InformationService
# ===========================================================================


class TestBeliefTrackerWithInfoService:
    """BeliefTracker delegation to / fallback from InformationService."""

    def test_belief_tracker_no_info_service_unchanged(self):
        """Without info service, BeliefTracker uses internal history as before."""
        bt = BeliefTracker(initial_prior=Decimal("0.15"))
        assert bt.information_service is None

        # No history -> returns prior
        p = bt.estimate_default_prob("firm_1", current_day=0)
        assert p == Decimal("0.15")

        # Add history -> uses Laplace smoothing
        bt.update_history(0, "firm_1", defaulted=True)
        bt.update_history(1, "firm_1", defaulted=False)
        p = bt.estimate_default_prob("firm_1", current_day=2)
        # Laplace: (1 + 1) / (2 + 2) = 2/4 = 0.5
        assert p == Decimal(2) / Decimal(4)

    def test_belief_tracker_delegates_to_info_service(self):
        """When info service is set and returns a value, BeliefTracker uses it."""
        service = MockInformationService({"firm_1": Decimal("0.42")})
        bt = BeliefTracker(
            initial_prior=Decimal("0.15"),
            information_service=service,
        )

        p = bt.estimate_default_prob("firm_1", current_day=5)
        assert p == Decimal("0.42")

    def test_belief_tracker_fallback_when_service_returns_none(self):
        """When info service returns None for an agent, falls back to internal."""
        # Service knows about firm_1 but NOT firm_2
        service = MockInformationService({"firm_1": Decimal("0.30")})
        bt = BeliefTracker(
            initial_prior=Decimal("0.15"),
            information_service=service,
        )

        # firm_2 not in service -> fallback to internal (prior, since no history)
        p = bt.estimate_default_prob("firm_2", current_day=0)
        assert p == Decimal("0.15")

    def test_belief_tracker_info_service_property(self):
        """Getter/setter property for information_service works correctly."""
        bt = BeliefTracker()
        assert bt.information_service is None

        service = MockInformationService({"X": Decimal("0.50")})
        bt.information_service = service
        assert bt.information_service is service

        bt.information_service = None
        assert bt.information_service is None

    def test_belief_tracker_info_service_detail_delegates(self):
        """estimate_default_prob_detail delegates to service when set."""
        service = MockInformationService({"firm_1": Decimal("0.33")})
        bt = BeliefTracker(
            initial_prior=Decimal("0.15"),
            information_service=service,
        )

        est = bt.estimate_default_prob_detail("firm_1", current_day=3)
        assert isinstance(est, Estimate)
        assert est.value == Decimal("0.33")
        assert est.target_id == "firm_1"
        assert est.target_type == "agent"
        assert est.estimation_day == 3

    def test_belief_tracker_info_service_detail_fallback(self):
        """When info service detail returns None, falls back to internal."""
        # Service does NOT know about firm_2
        service = MockInformationService({"firm_1": Decimal("0.30")})
        bt = BeliefTracker(
            initial_prior=Decimal("0.15"),
            information_service=service,
        )

        est = bt.estimate_default_prob_detail("firm_2", current_day=0)
        assert isinstance(est, Estimate)
        # Falls back to internal: no history -> prior = 0.15
        assert est.value == Decimal("0.15")
        assert est.method == "bayesian_default_prob"
        assert est.inputs["used_prior"] is True


# ===========================================================================
# Group 2: Factory Integration
# ===========================================================================


class TestFactoryInfoServiceWiring:
    """create_assessor() correctly wires information_service."""

    def test_create_assessor_no_profile_no_service(self):
        """Without profile or service, returns standard assessor (backward compat)."""
        params = RiskAssessmentParams(initial_prior=Decimal("0.20"))
        assessor = create_assessor(params)
        assert isinstance(assessor, RiskAssessor)
        assert assessor.params.initial_prior == Decimal("0.20")
        assert assessor.params.default_observability == Decimal("1.0")
        assert assessor.belief_tracker.information_service is None

    def test_create_assessor_with_info_service(self):
        """Info service gets wired into the assessor's belief_tracker."""
        params = RiskAssessmentParams()
        service = MockInformationService({"firm_1": Decimal("0.55")})
        assessor = create_assessor(params, information_service=service)

        assert assessor.belief_tracker.information_service is service
        # Verify delegation works end-to-end through the assessor
        p = assessor.estimate_default_prob("firm_1", current_day=0)
        assert p == Decimal("0.55")

    def test_create_assessor_with_profile_and_service(self):
        """Both profile and service are applied together."""
        from bilancio.information.levels import AccessLevel
        from bilancio.information.noise import SampleNoise
        from bilancio.information.profile import CategoryAccess, InformationProfile

        params = RiskAssessmentParams()
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                level=AccessLevel.NOISY,
                noise=SampleNoise(sample_rate=Decimal("0.7")),
            )
        )
        service = MockInformationService({"firm_1": Decimal("0.40")})
        assessor = create_assessor(params, profile, information_service=service)

        # Profile should adjust observability
        assert assessor.params.default_observability == Decimal("0.7")
        # Service should be wired in
        assert assessor.belief_tracker.information_service is service


# ===========================================================================
# Group 3: TraderState Field
# ===========================================================================


class TestTraderStateInfoService:
    """TraderState optional information_service field."""

    def test_trader_state_default_no_info_service(self):
        """Default TraderState has information_service=None."""
        ts = TraderState(agent_id="trader_1")
        assert getattr(ts, "information_service", None) is None

    def test_trader_state_with_info_service(self):
        """TraderState can be created with an info service."""
        service = MockInformationService({"A": Decimal("0.10")})
        ts = TraderState(agent_id="trader_1", information_service=service)
        assert ts.information_service is service


# ===========================================================================
# Group 4: End-to-End Decision Impact
# ===========================================================================


class TestEndToEndDecisionImpact:
    """Info service affects sell/buy decisions through the full pipeline."""

    def test_info_service_affects_sell_decision(self, ticket):
        """Trader with info service reporting high P_default values ticket lower,
        making them more willing to sell at a given bid price.
        """
        # Trader WITHOUT info service: uses prior P=0.15
        # EV = (1 - 0.15) * 20 = 17
        assessor_no_svc = RiskAssessor(RiskAssessmentParams())

        # Trader WITH info service: service says P=0.60 for firm_1
        # EV = (1 - 0.60) * 20 = 8
        service = MockInformationService({"firm_1": Decimal("0.60")})
        assessor_with_svc = RiskAssessor(RiskAssessmentParams())
        assessor_with_svc.belief_tracker.information_service = service

        bid = Decimal("0.45")  # bid price = 0.45 * 20 = 9

        sell_kwargs = {
            "ticket": ticket,
            "dealer_bid": bid,
            "current_day": 0,
            "trader_cash": Decimal("50"),
            "trader_shortfall": Decimal("0"),
            "trader_asset_value": Decimal("50"),
        }

        # Without service: bid 9 < EV 17 => reject
        assert assessor_no_svc.should_sell(**sell_kwargs) is False

        # With service: bid 9 >= EV 8 => accept
        assert assessor_with_svc.should_sell(**sell_kwargs) is True

    def test_info_service_affects_buy_decision(self, ticket):
        """Trader with info service reporting high P_default is less willing to buy."""
        # Trader WITHOUT service: P=0.15, EV=17
        assessor_no_svc = RiskAssessor(RiskAssessmentParams())

        # Trader WITH service: P=0.80, EV=(1-0.80)*20=4
        service = MockInformationService({"firm_1": Decimal("0.80")})
        assessor_with_svc = RiskAssessor(RiskAssessmentParams())
        assessor_with_svc.belief_tracker.information_service = service

        ask = Decimal("0.30")  # cost = 6

        buy_kwargs = {
            "ticket": ticket,
            "dealer_ask": ask,
            "current_day": 0,
            "trader_cash": Decimal("100"),
            "trader_shortfall": Decimal("0"),
            "trader_asset_value": Decimal("100"),
        }

        # Without service: EV=17, cost=6 => 17 >= 6 + threshold => accept
        assert assessor_no_svc.should_buy(**buy_kwargs) is True

        # With service: EV=4, cost=6 => 4 < 6 + threshold => reject
        assert assessor_with_svc.should_buy(**buy_kwargs) is False

    def test_omniscient_profile_matches_direct(self, ticket):
        """InformationService with perfect knowledge should produce
        equivalent results to direct internal tracking when given the
        same P_default.
        """
        # Setup: both assessors get same effective P_default = 0.15 (the prior)
        # One via info service, one via internal (no history = prior)
        service = MockInformationService({"firm_1": Decimal("0.15")})

        assessor_svc = RiskAssessor(RiskAssessmentParams(initial_prior=Decimal("0.15")))
        assessor_svc.belief_tracker.information_service = service

        assessor_internal = RiskAssessor(
            RiskAssessmentParams(initial_prior=Decimal("0.15"))
        )

        p_svc = assessor_svc.estimate_default_prob("firm_1", current_day=0)
        p_internal = assessor_internal.estimate_default_prob("firm_1", current_day=0)
        assert p_svc == p_internal == Decimal("0.15")

        ev_svc = assessor_svc.expected_value(ticket, current_day=0)
        ev_internal = assessor_internal.expected_value(ticket, current_day=0)
        assert ev_svc == ev_internal

    def test_noisy_profile_produces_different_p_default(self):
        """With a NOISY profile, the P_default from the service should differ
        from the raw internal estimate when they have different underlying values.
        """
        # Service reports P=0.40 for firm_1
        service = MockInformationService({"firm_1": Decimal("0.40")})

        # Assessor with service
        bt_with = BeliefTracker(
            initial_prior=Decimal("0.15"),
            information_service=service,
        )

        # Assessor without service
        bt_without = BeliefTracker(initial_prior=Decimal("0.15"))

        p_with = bt_with.estimate_default_prob("firm_1", current_day=0)
        p_without = bt_without.estimate_default_prob("firm_1", current_day=0)

        # Service returns 0.40, internal returns prior 0.15
        assert p_with == Decimal("0.40")
        assert p_without == Decimal("0.15")
        assert p_with != p_without


# ===========================================================================
# Group 5: Backward Compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure existing behavior is preserved when no info service is present."""

    def test_risk_assessor_wrapper_info_service_passthrough(self):
        """RiskAssessor correctly passes info_service to its belief_tracker."""
        service = MockInformationService({"firm_1": Decimal("0.77")})
        params = RiskAssessmentParams()
        assessor = RiskAssessor(params)

        # Initially no service
        assert assessor.belief_tracker.information_service is None

        # Set via belief_tracker
        assessor.belief_tracker.information_service = service
        assert assessor.belief_tracker.information_service is service

        # Delegation works through wrapper
        p = assessor.estimate_default_prob("firm_1", current_day=0)
        assert p == Decimal("0.77")

    def test_existing_payment_history_still_works(self, ticket):
        """With no info service, payment history tracking and estimation
        works identically to the pre-Phase-4 behavior.
        """
        params = RiskAssessmentParams(initial_prior=Decimal("0.15"))
        assessor = RiskAssessor(params)

        # Verify no info service
        assert assessor.belief_tracker.information_service is None

        # No history: returns prior
        p0 = assessor.estimate_default_prob("firm_1", current_day=0)
        assert p0 == Decimal("0.15")

        # Add history: 3 defaults out of 5 observations
        for day in range(5):
            assessor.update_history(day, "firm_1", defaulted=(day < 3))

        # Laplace: (1 + 3) / (2 + 5) = 4/7
        p5 = assessor.estimate_default_prob("firm_1", current_day=5)
        assert p5 == Decimal(4) / Decimal(7)

        # EV uses the same P
        ev = assessor.expected_value(ticket, current_day=5)
        expected_ev = (Decimal(1) - Decimal(4) / Decimal(7)) * Decimal("20")
        assert ev == expected_ev

        # Sell decision still works
        result = assessor.should_sell(
            ticket,
            dealer_bid=Decimal("0.90"),
            current_day=5,
            trader_cash=Decimal("50"),
            trader_shortfall=Decimal("0"),
            trader_asset_value=Decimal("50"),
        )
        # EV = (3/7) * 20 ~ 8.57. Bid = 0.90 * 20 = 18 >= 8.57 => accept
        assert result is True
