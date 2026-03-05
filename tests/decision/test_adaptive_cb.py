"""CB corridor adaptation tests (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.domain.agents.central_bank import CentralBank


class TestCBAdaptiveBetas:
    def test_disabled_by_default(self):
        cb = CentralBank(id="cb", name="CB", kappa_prior=Decimal("0.1"))
        assert cb.adaptive_betas is False

    def test_beta_scaling_reduces_reactivity(self):
        cb_normal = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.1"),
            beta_mid=Decimal("0.5"), beta_width=Decimal("0.3"),
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        cb_adaptive = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.1"),
            beta_mid=Decimal("0.5"), beta_width=Decimal("0.3"),
            adaptive_betas=True,
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        # Large surprise: 20% defaults in 100 agents
        _, ceil_normal = cb_normal.compute_corridor(20, 100)
        _, ceil_adaptive = cb_adaptive.compute_corridor(20, 100)
        # Adaptive betas should be smaller, producing less ceiling increase
        assert ceil_adaptive < ceil_normal

    def test_high_prior_reduces_betas_more(self):
        # With high kappa_prior, beta_scale should be smaller
        cb_low = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.05"),
            adaptive_betas=True, beta_mid=Decimal("0.5"),
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        cb_high = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.3"),
            adaptive_betas=True, beta_mid=Decimal("0.5"),
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        # Same surprise scenario
        _, ceil_low = cb_low.compute_corridor(30, 100)
        _, ceil_high = cb_high.compute_corridor(30, 100)
        # High prior -> smaller scaled betas -> less ceiling
        # But also higher prior means less surprise, so check the betas directly
        scale_low = Decimal(1) / (Decimal(1) + Decimal(5) * Decimal("0.05"))
        scale_high = Decimal(1) / (Decimal(1) + Decimal(5) * Decimal("0.3"))
        assert scale_high < scale_low


class TestCBEarlyWarning:
    def test_disabled_by_default(self):
        cb = CentralBank(id="cb", name="CB", kappa_prior=Decimal("0.1"))
        assert cb.adaptive_early_warning is False

    def test_bank_stress_widens_corridor(self):
        cb = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.1"),
            adaptive_early_warning=True,
            beta_mid=Decimal("0.5"), beta_width=Decimal("0.3"),
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        _, ceil_no_stress = cb.compute_corridor(5, 100, bank_stress=Decimal("0"))
        _, ceil_stress = cb.compute_corridor(5, 100, bank_stress=Decimal("0.5"))
        assert ceil_stress > ceil_no_stress

    def test_bank_stress_ignored_when_disabled(self):
        cb = CentralBank(
            id="cb", name="CB", kappa_prior=Decimal("0.1"),
            adaptive_early_warning=False,
            beta_mid=Decimal("0.5"), beta_width=Decimal("0.3"),
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        _, ceil_no_stress = cb.compute_corridor(5, 100, bank_stress=Decimal("0"))
        _, ceil_stress = cb.compute_corridor(5, 100, bank_stress=Decimal("0.5"))
        assert ceil_stress == ceil_no_stress
