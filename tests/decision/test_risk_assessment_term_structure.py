"""Tests for EVValuer term structure (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.decision.risk_assessment import (
    BeliefTracker, EVValuer, RiskAssessmentParams, RiskAssessor,
)


class _FakeTicket:
    def __init__(self, issuer_id="A", face=Decimal("100"), maturity_day=10):
        self.id = "t1"
        self.issuer_id = issuer_id
        self.face = face
        self.maturity_day = maturity_day
        self.bucket_id = "short"


class TestEVValuerTermStructure:
    def test_flat_by_default(self):
        bt = BeliefTracker(initial_prior=Decimal("0.1"))
        ev = EVValuer(bt)
        ticket = _FakeTicket(maturity_day=10)
        val = ev.expected_value(ticket, 0)
        assert val == Decimal("90")  # (1-0.1) * 100

    def test_term_structure_reduces_far_dated(self):
        bt = BeliefTracker(initial_prior=Decimal("0.1"))
        ev_flat = EVValuer(bt)
        ev_term = EVValuer(bt, term_structure=True, term_strength=Decimal("0.5"))
        ticket_near = _FakeTicket(maturity_day=2)
        ticket_far = _FakeTicket(maturity_day=10)
        # Near-dated should be similar to flat
        flat_near = ev_flat.expected_value(ticket_near, 0)
        term_near = ev_term.expected_value(ticket_near, 0)
        # Far-dated should be lower with term structure
        flat_far = ev_flat.expected_value(ticket_far, 0)
        term_far = ev_term.expected_value(ticket_far, 0)
        assert term_far < flat_far
        # Near-dated difference should be smaller
        assert abs(term_near - flat_near) < abs(term_far - flat_far)

    def test_zero_p_gives_full_face(self):
        bt = BeliefTracker(initial_prior=Decimal("0"))
        ev = EVValuer(bt, term_structure=True, term_strength=Decimal("0.5"))
        ticket = _FakeTicket(maturity_day=10)
        val = ev.expected_value(ticket, 0)
        assert val == Decimal("100")  # (1-0)^10 * 100 = 100


class TestRiskAssessorTermStructure:
    def test_wired_from_params(self):
        params = RiskAssessmentParams(
            adaptive_ev_term_structure=True,
            term_strength=Decimal("0.3"),
        )
        ra = RiskAssessor(params)
        assert ra.valuer.term_structure is True
        assert ra.valuer.term_strength == Decimal("0.3")

    def test_disabled_by_default(self):
        params = RiskAssessmentParams()
        ra = RiskAssessor(params)
        assert ra.valuer.term_structure is False
