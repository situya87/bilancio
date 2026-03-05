"""Tests for scenario_informed_prior (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.dealer.priors import scenario_informed_prior, kappa_informed_prior


class TestScenarioInformedPrior:
    def test_matches_kappa_prior_when_mu_0_5_c_large(self):
        # With neutral mu and large c, should be close to kappa_informed_prior
        kappa = Decimal("0.5")
        p_scenario = scenario_informed_prior(kappa, Decimal("0.5"), Decimal("100"))
        p_kappa = kappa_informed_prior(kappa)
        assert abs(p_scenario - p_kappa) < Decimal("0.02")

    def test_front_loaded_increases_prior(self):
        kappa = Decimal("0.5")
        p_front = scenario_informed_prior(kappa, Decimal("0"), Decimal("1"))
        p_back = scenario_informed_prior(kappa, Decimal("1"), Decimal("1"))
        assert p_front > p_back

    def test_concentrated_debt_increases_prior(self):
        kappa = Decimal("0.5")
        p_conc = scenario_informed_prior(kappa, Decimal("0.5"), Decimal("0.1"))
        p_uniform = scenario_informed_prior(kappa, Decimal("0.5"), Decimal("10"))
        assert p_conc > p_uniform

    def test_range_bounds(self):
        # Should always be in [0.05, 0.20]
        for kappa in [Decimal("0"), Decimal("0.5"), Decimal("2"), Decimal("10")]:
            for mu in [Decimal("0"), Decimal("0.5"), Decimal("1")]:
                for c in [Decimal("0.1"), Decimal("1"), Decimal("10")]:
                    p = scenario_informed_prior(max(kappa, Decimal("0.01")), mu, c)
                    assert Decimal("0.05") <= p <= Decimal("0.20"), f"p={p} for kappa={kappa}, mu={mu}, c={c}"

    def test_maximum_stress(self):
        p = scenario_informed_prior(Decimal("0.01"), Decimal("0"), Decimal("0.1"))
        assert p == Decimal("0.20")  # should hit the cap
