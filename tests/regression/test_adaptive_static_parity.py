"""Regression test: --adapt=static produces identical results to no-adapt run (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.decision.adaptive import build_adaptive_overrides


class TestStaticParity:
    def test_static_produces_empty_overrides(self):
        """Static preset must produce zero overrides -- bit-identical to pre-050 behavior."""
        for kappa in [Decimal("0.3"), Decimal("1"), Decimal("3")]:
            for mu in [Decimal("0"), Decimal("0.5"), Decimal("1")]:
                for c in [Decimal("0.5"), Decimal("1"), Decimal("5")]:
                    result = build_adaptive_overrides("static", kappa, mu, c, 10, 100)
                    for profile_name, overrides in result.items():
                        assert overrides == {}, (
                            f"static preset produced non-empty overrides for {profile_name}: "
                            f"{overrides} at kappa={kappa}, mu={mu}, c={c}"
                        )

    def test_static_preserves_profile_defaults(self):
        """Profiles constructed without adaptive flags should match pre-050 defaults."""
        from bilancio.decision.profiles import TraderProfile, VBTProfile, BankProfile, LenderProfile
        from bilancio.decision.risk_assessment import RiskAssessmentParams

        tp = TraderProfile()
        assert tp.adaptive_planning_horizon is False
        assert tp.adaptive_risk_aversion is False
        assert tp.adaptive_reserves is False
        assert tp.adaptive_ev_term_structure is False

        vp = VBTProfile()
        assert vp.adaptive_term_structure is False
        assert vp.adaptive_convex_spreads is False

        bp = BankProfile()
        assert bp.adaptive_corridor is False
        # corridor_mid should be identical with or without mu/c when flag is off
        assert bp.corridor_mid(Decimal("0.5")) == bp.corridor_mid(Decimal("0.5"), Decimal("0"), Decimal("1"))

        rp = RiskAssessmentParams()
        assert rp.adaptive_ev_term_structure is False
