"""VBT term-structure and convex spread tests (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.decision.valuers import CreditAdjustedVBTPricing


class TestComputeMidTermAdjusted:
    def test_short_bucket_higher_than_long(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p = Decimal("0.1")
        m_short = pm.compute_mid_term_adjusted(p, Decimal("0.15"), tau=2)
        m_long = pm.compute_mid_term_adjusted(p, Decimal("0.15"), tau=12)
        assert m_short > m_long  # shorter maturity = higher mid

    def test_zero_default_prob(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        m = pm.compute_mid_term_adjusted(Decimal("0"), Decimal("0"), tau=5)
        assert m == Decimal("0.9")  # rho * (1-0)^5 = rho

    def test_matches_compute_mid_at_tau_1(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        p = Decimal("0.1")
        prior = Decimal("0.15")
        m_term = pm.compute_mid_term_adjusted(p, prior, tau=1, term_strength=Decimal("1"))
        # At tau=1, term_strength=1: M = rho * (1-p) = compute_mid(p)
        m_flat = pm.compute_mid(p, prior)
        assert abs(m_term - m_flat) < Decimal("0.001")

    def test_term_strength_zero_equals_no_adjustment(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        m = pm.compute_mid_term_adjusted(Decimal("0.2"), Decimal("0.15"), tau=10, term_strength=Decimal("0"))
        # term_strength=0: h=0, survival=1, raw_M = rho
        initial_M = Decimal("0.9") * (1 - Decimal("0.15"))
        assert m == initial_M + Decimal("1") * (Decimal("0.9") - initial_M)


class TestComputeSpreadConvex:
    def test_convex_larger_than_linear_at_high_p(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        base = Decimal("0.04")
        p = Decimal("0.6")
        linear = pm.compute_spread(base, p)
        convex = pm.compute_spread_convex(base, p)
        assert convex > linear  # convex widens more at high p (crossover at p=0.5)

    def test_convex_smaller_than_linear_at_low_p(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        base = Decimal("0.04")
        p = Decimal("0.05")
        linear = pm.compute_spread(base, p)
        convex = pm.compute_spread_convex(base, p)
        assert convex < linear  # convex tightens at low p

    def test_edge_case_p_one(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        s = pm.compute_spread_convex(Decimal("0.04"), Decimal("1"))
        assert s == Decimal("0.04") + Decimal("0.6")

    def test_monotonic_in_p(self):
        pm = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"), spread_sensitivity=Decimal("0.6"),
            outside_mid_ratio=Decimal("0.9"),
        )
        base = Decimal("0.04")
        s1 = pm.compute_spread_convex(base, Decimal("0.1"))
        s2 = pm.compute_spread_convex(base, Decimal("0.3"))
        s3 = pm.compute_spread_convex(base, Decimal("0.5"))
        assert s1 < s2 < s3
