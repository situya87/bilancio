"""Per-flag unit tests for adaptive profiles (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.decision.profiles import TraderProfile, VBTProfile, LenderProfile, BankProfile


class TestTraderAdaptiveFlags:
    def test_defaults_false(self):
        t = TraderProfile()
        assert t.adaptive_planning_horizon is False
        assert t.adaptive_risk_aversion is False
        assert t.adaptive_reserves is False
        assert t.adaptive_ev_term_structure is False

    def test_can_set_flags(self):
        t = TraderProfile(adaptive_risk_aversion=True, adaptive_reserves=True)
        assert t.adaptive_risk_aversion is True
        assert t.adaptive_reserves is True


class TestVBTAdaptiveFlags:
    def test_defaults_false(self):
        v = VBTProfile()
        assert v.adaptive_term_structure is False
        assert v.adaptive_base_spreads is False
        assert v.adaptive_stress_horizon is False
        assert v.adaptive_convex_spreads is False
        assert v.adaptive_per_bucket_tracking is False
        assert v.adaptive_issuer_pricing is False
        assert v.term_strength == Decimal("0.5")

    def test_term_strength_configurable(self):
        v = VBTProfile(term_strength=Decimal("0.3"))
        assert v.term_strength == Decimal("0.3")


class TestLenderAdaptiveFlags:
    def test_defaults_false(self):
        lp = LenderProfile()
        assert lp.adaptive_risk_aversion is False
        assert lp.adaptive_profit_target is False
        assert lp.adaptive_loan_maturity is False
        assert lp.adaptive_rates is False
        assert lp.adaptive_capital_conservation is False
        assert lp.adaptive_prevention is False


class TestBankAdaptiveFlags:
    def test_default_false(self):
        bp = BankProfile()
        assert bp.adaptive_corridor is False

    def test_corridor_mid_unchanged_when_disabled(self):
        bp = BankProfile()
        mid1 = bp.corridor_mid(Decimal("0.5"))
        mid2 = bp.corridor_mid(Decimal("0.5"), Decimal("0"), Decimal("1"))
        assert mid1 == mid2

    def test_corridor_mid_changes_when_enabled(self):
        bp = BankProfile(adaptive_corridor=True)
        mid_no_args = bp.corridor_mid(Decimal("0.5"))
        mid_with_args = bp.corridor_mid(Decimal("0.5"), Decimal("0"), Decimal("1"))
        assert mid_with_args > mid_no_args

    def test_combined_stress_monotonic_in_mu(self):
        bp = BankProfile(adaptive_corridor=True)
        # mu=0 (front-loaded) should give MORE stress than mu=1 (back-loaded)
        mid_front = bp.corridor_mid(Decimal("0.5"), Decimal("0"), Decimal("1"))
        mid_back = bp.corridor_mid(Decimal("0.5"), Decimal("1"), Decimal("1"))
        assert mid_front > mid_back

    def test_combined_stress_monotonic_in_c(self):
        bp = BankProfile(adaptive_corridor=True)
        # c=0.1 (concentrated) should give MORE stress than c=10 (uniform)
        mid_concentrated = bp.corridor_mid(Decimal("0.5"), Decimal("0.5"), Decimal("0.1"))
        mid_uniform = bp.corridor_mid(Decimal("0.5"), Decimal("0.5"), Decimal("10"))
        assert mid_concentrated > mid_uniform
