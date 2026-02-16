"""Tests for InstrumentValuer protocol and concrete implementations."""

from decimal import Decimal

import pytest

from bilancio.dealer.models import Ticket
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.decision.protocols import InstrumentValuer
from bilancio.decision.valuers import CoverageRatioValuer, EVHoldValuer
from bilancio.information.estimates import Estimate

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def params():
    return RiskAssessmentParams(initial_prior=Decimal("0.20"))


@pytest.fixture()
def assessor(params):
    return RiskAssessor(params)


@pytest.fixture()
def ticket():
    return Ticket(
        id="t1",
        issuer_id="firm_1",
        owner_id="firm_2",
        face=Decimal("100"),
        maturity_day=10,
        remaining_tau=5,
        bucket_id="short",
    )


# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    def test_evhold_satisfies_protocol(self, assessor):
        valuer = EVHoldValuer(risk_assessor=assessor)
        assert isinstance(valuer, InstrumentValuer)

    def test_coverage_ratio_satisfies_protocol(self):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.10")},
        )
        assert isinstance(valuer, InstrumentValuer)


# ── EVHoldValuer ──────────────────────────────────────────────────


class TestEVHoldValuer:
    def test_value_decimal_matches_risk_assessor(self, assessor, ticket):
        valuer = EVHoldValuer(risk_assessor=assessor)
        expected = assessor.expected_value(ticket, 5)
        actual = valuer.value_decimal(ticket, 5)
        assert actual == expected

    def test_value_decimal_no_history(self, assessor, ticket, params):
        """With no history, uses initial_prior."""
        valuer = EVHoldValuer(risk_assessor=assessor)
        # EV = (1 - 0.20) * 100 = 80
        assert valuer.value_decimal(ticket, 1) == Decimal("80")

    def test_value_decimal_with_history(self, assessor, ticket):
        """After observing defaults, value changes."""
        valuer = EVHoldValuer(risk_assessor=assessor)
        # Record some defaults
        assessor.update_history(1, "firm_1", defaulted=True)
        assessor.update_history(1, "firm_2", defaulted=False)

        ev_valuer = valuer.value_decimal(ticket, 2)
        ev_assessor = assessor.expected_value(ticket, 2)
        assert ev_valuer == ev_assessor

    def test_value_returns_estimate(self, assessor, ticket):
        valuer = EVHoldValuer(risk_assessor=assessor)
        est = valuer.value(ticket, 5)
        assert isinstance(est, Estimate)
        assert est.method == "ev_hold"
        assert est.target_type == "instrument"
        assert est.estimator_id == "ev_hold_valuer"
        assert est.estimation_day == 5

    def test_value_estimate_value_matches_decimal(self, assessor, ticket):
        valuer = EVHoldValuer(risk_assessor=assessor)
        dec = valuer.value_decimal(ticket, 5)
        est = valuer.value(ticket, 5)
        assert est.value == dec

    def test_custom_estimator_id(self, assessor, ticket):
        valuer = EVHoldValuer(risk_assessor=assessor, estimator_id="trader_7")
        est = valuer.value(ticket, 5)
        assert est.estimator_id == "trader_7"


# ── CoverageRatioValuer ──────────────────────────────────────────


class TestCoverageRatioValuer:
    def test_known_issuer(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.10")},
        )
        # EV = (1 - 0.10) * 100 = 90
        assert valuer.value_decimal(ticket, 5) == Decimal("90")

    def test_unknown_issuer_uses_fallback(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_99": Decimal("0.10")},
            fallback_prior=Decimal("0.25"),
        )
        # firm_1 not in registry → fallback 0.25
        # EV = (1 - 0.25) * 100 = 75
        assert valuer.value_decimal(ticket, 5) == Decimal("75")

    def test_value_returns_estimate(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.10")},
        )
        est = valuer.value(ticket, 5)
        assert isinstance(est, Estimate)
        assert est.method == "coverage_ratio_ev"
        assert est.target_type == "instrument"
        assert est.value == Decimal("90")
        assert est.inputs["used_fallback"] is False

    def test_value_estimate_fallback_flag(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={},
            fallback_prior=Decimal("0.30"),
        )
        est = valuer.value(ticket, 5)
        assert est.inputs["used_fallback"] is True
        assert est.value == Decimal("70")

    def test_zero_default_prob(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0")},
        )
        assert valuer.value_decimal(ticket, 5) == Decimal("100")

    def test_custom_estimator_id(self, ticket):
        valuer = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.10")},
            estimator_id="rating_agency_1",
        )
        est = valuer.value(ticket, 5)
        assert est.estimator_id == "rating_agency_1"


# ── RiskAssessor delegation ───────────────────────────────────────


class TestRiskAssessorDelegation:
    def test_without_valuer_uses_builtin(self, assessor, ticket):
        """Default (no valuer) uses built-in EV logic."""
        ev = assessor.expected_value(ticket, 5)
        # No history → initial_prior = 0.20 → EV = 80
        assert ev == Decimal("80")

    def test_with_valuer_delegates_value_decimal(self, ticket):
        """Injected valuer's value_decimal() is used."""
        coverage = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.05")},
        )
        assessor = RiskAssessor(
            RiskAssessmentParams(),
            instrument_valuer=coverage,
        )
        # Should use coverage ratio: (1 - 0.05) * 100 = 95
        assert assessor.expected_value(ticket, 5) == Decimal("95")

    def test_with_valuer_delegates_value_detail(self, ticket):
        """Injected valuer's value() is used for expected_value_detail()."""
        coverage = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.05")},
        )
        assessor = RiskAssessor(
            RiskAssessmentParams(),
            instrument_valuer=coverage,
        )
        est = assessor.expected_value_detail(ticket, 5)
        assert isinstance(est, Estimate)
        assert est.method == "coverage_ratio_ev"
        assert est.value == Decimal("95")

    def test_should_sell_uses_delegated_valuer(self, ticket):
        """should_sell() sees the delegated expected value."""
        # Use a valuer that gives high value → should NOT sell cheaply
        coverage = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.01")},  # nearly risk-free
        )
        assessor = RiskAssessor(
            RiskAssessmentParams(),
            instrument_valuer=coverage,
        )
        # EV = (1 - 0.01) * 100 = 99. Bid at 0.80 → offer = 80 < 99
        assert (
            assessor.should_sell(
                ticket,
                dealer_bid=Decimal("0.80"),
                current_day=5,
                trader_cash=Decimal("50"),
                trader_shortfall=Decimal("0"),
                trader_asset_value=Decimal("100"),
            )
            is False
        )

    def test_should_buy_uses_delegated_valuer(self, ticket):
        """should_buy() sees the delegated expected value."""
        # Use a valuer that gives low value → should NOT buy at high price
        coverage = CoverageRatioValuer(
            rating_registry={"firm_1": Decimal("0.50")},  # very risky
        )
        assessor = RiskAssessor(
            RiskAssessmentParams(),
            instrument_valuer=coverage,
        )
        # EV = (1 - 0.50) * 100 = 50. Ask at 0.90 → cost = 90 > 50
        assert (
            assessor.should_buy(
                ticket,
                dealer_ask=Decimal("0.90"),
                current_day=5,
                trader_cash=Decimal("100"),
                trader_shortfall=Decimal("0"),
                trader_asset_value=Decimal("100"),
            )
            is False
        )

    def test_backward_compat_no_valuer(self, params, ticket):
        """RiskAssessor with no valuer behaves exactly as before."""
        assessor_old = RiskAssessor(params)
        assessor_new = RiskAssessor(params, instrument_valuer=None)

        assert assessor_old.expected_value(ticket, 5) == assessor_new.expected_value(ticket, 5)

        # Record same history
        assessor_old.update_history(1, "firm_1", True)
        assessor_new.update_history(1, "firm_1", True)
        assert assessor_old.expected_value(ticket, 2) == assessor_new.expected_value(ticket, 2)
