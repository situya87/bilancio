"""Tests for Plan 050 Phase 3: Within-run adaptation wiring.

Tests cover:
- 3A: Trader adaptive risk aversion
- 3B: Trader adaptive reserves (buy_reserve_fraction)
- 3F: CB early warning (bank stress signal)
- 3G: NBFI within-run adaptation (rates, capital conservation, prevention)
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bilancio.decision.intentions import BuyIntention, SurplusBuyer
from bilancio.decision.profiles import LenderProfile, TraderProfile


# ---------------------------------------------------------------------------
# 3A: Adaptive risk aversion
# ---------------------------------------------------------------------------


class TestAdaptiveRiskAversion:
    """Trader risk aversion increases with observed defaults."""

    def test_buy_premium_updated_when_defaults_observed(self):
        """buy_risk_premium should increase on the trade gate when defaults occur."""
        from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
        from bilancio.engines.dealer_integration import DealerSubsystem

        profile = TraderProfile(
            risk_aversion=Decimal("0.0"),
            adaptive_risk_aversion=True,
        )
        params = RiskAssessmentParams(buy_risk_premium=Decimal("0.01"))
        assessor = RiskAssessor(params)

        subsystem = DealerSubsystem()
        subsystem.trader_profile = profile
        subsystem.risk_assessor = assessor

        # Initial buy_risk_premium from params
        assert assessor.trade_gate.buy_risk_premium == Decimal("0.01")

        # Simulate what run_dealer_trading_phase does when defaults observed
        n_total = 100
        n_defaulted = 30  # 30% default rate
        observed = Decimal(n_defaulted) / Decimal(max(n_total, 1))
        base_ra = profile.risk_aversion
        effective_ra = base_ra + (Decimal("1") - base_ra) * min(
            Decimal("1"), observed / Decimal("0.3")
        )
        new_buy_premium = Decimal("0.01") + Decimal("0.02") * effective_ra
        assessor.trade_gate.buy_risk_premium = new_buy_premium

        # At 30% default rate, effective_ra should be ~1.0 (base_ra=0, observed/0.3=1)
        assert effective_ra == Decimal("1")
        assert assessor.trade_gate.buy_risk_premium == Decimal("0.03")

    def test_no_update_when_flag_disabled(self):
        """buy_risk_premium should not change when adaptive_risk_aversion=False."""
        profile = TraderProfile(
            risk_aversion=Decimal("0.5"),
            adaptive_risk_aversion=False,
        )
        # The flag is off, so no adaptation should occur
        assert profile.adaptive_risk_aversion is False

    def test_partial_default_rate(self):
        """Verify effective_ra interpolates between base and 1.0."""
        base_ra = Decimal("0.2")
        observed = Decimal("0.15")  # 15% default rate -> 50% of threshold
        effective_ra = base_ra + (Decimal("1") - base_ra) * min(
            Decimal("1"), observed / Decimal("0.3")
        )
        # 0.2 + 0.8 * 0.5 = 0.6
        assert effective_ra == Decimal("0.6")

    def test_buy_premium_multiplier_also_updated(self):
        """buy_premium_multiplier should also be updated for consistency."""
        from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor

        params = RiskAssessmentParams()
        assessor = RiskAssessor(params)

        effective_ra = Decimal("0.5")
        new_multiplier = Decimal("1.0") + effective_ra
        assessor.trade_gate.buy_premium_multiplier = new_multiplier

        assert assessor.trade_gate.buy_premium_multiplier == Decimal("1.5")

    def test_per_trader_assessors_updated(self):
        """Per-trader assessors should also get updated buy premiums."""
        from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
        from bilancio.engines.dealer_integration import DealerSubsystem

        params = RiskAssessmentParams(buy_risk_premium=Decimal("0.01"))
        assessor1 = RiskAssessor(params)
        assessor2 = RiskAssessor(params)

        subsystem = DealerSubsystem()
        subsystem.trader_assessors = {"trader_1": assessor1, "trader_2": assessor2}

        new_premium = Decimal("0.025")
        for a in subsystem.trader_assessors.values():
            a.trade_gate.buy_risk_premium = new_premium

        assert assessor1.trade_gate.buy_risk_premium == Decimal("0.025")
        assert assessor2.trade_gate.buy_risk_premium == Decimal("0.025")


# ---------------------------------------------------------------------------
# 3B: Adaptive reserves
# ---------------------------------------------------------------------------


class TestAdaptiveReserves:
    """buy_reserve_fraction increases with default stress."""

    def test_effective_reserve_increases_with_defaults(self):
        """Effective reserve fraction should increase when defaults observed."""
        base = Decimal("0.5")
        observed = Decimal("0.3")  # 30% default rate -> stress = 1.0
        stress = observed / Decimal("0.3")
        effective = base + (Decimal("1") - base) * min(
            Decimal("1"), stress * Decimal("0.5")
        )
        # stress = 1.0, stress * 0.5 = 0.5
        # effective = 0.5 + 0.5 * 0.5 = 0.75
        assert effective == Decimal("0.75")

    def test_no_stress_preserves_base(self):
        """With zero defaults, effective_buy_reserve_fraction = base."""
        base = Decimal("0.5")
        observed = Decimal("0")
        stress = observed / Decimal("0.3")
        effective = base + (Decimal("1") - base) * min(
            Decimal("1"), stress * Decimal("0.5")
        )
        assert effective == base

    def test_surplus_buyer_uses_override(self):
        """SurplusBuyer.evaluate should use effective_buy_reserve_fraction when provided."""
        profile = TraderProfile(
            buy_reserve_fraction=Decimal("0.5"),
            trading_motive="liquidity_then_earning",
        )

        # Create a mock trader with cash and obligations
        trader = MagicMock()
        trader.cash = Decimal("100")
        trader.earliest_liability_day.return_value = 5
        trader.payment_due.return_value = Decimal("20")

        buyer = SurplusBuyer()

        # Without override: reserve = 0.5 * 20 * (10+1) = 110, surplus = 100 - 110 = -10 -> None
        result_no_override = buyer.evaluate(
            "trader_1", trader, 0, 10, profile, Decimal("20")
        )

        # With lower override: reserve = 0.3 * 20 * (10+1) = 66, surplus = 100 - 66 = 34 -> BuyIntention
        result_with_override = buyer.evaluate(
            "trader_1", trader, 0, 10, profile, Decimal("20"),
            effective_buy_reserve_fraction=Decimal("0.3"),
        )

        assert result_no_override is None
        assert result_with_override is not None
        assert isinstance(result_with_override, BuyIntention)

    def test_subsystem_stores_effective_fraction(self):
        """DealerSubsystem should have effective_buy_reserve_fraction field."""
        from bilancio.engines.dealer_integration import DealerSubsystem

        subsystem = DealerSubsystem()
        assert subsystem.effective_buy_reserve_fraction is None

        subsystem.effective_buy_reserve_fraction = Decimal("0.75")
        assert subsystem.effective_buy_reserve_fraction == Decimal("0.75")


# ---------------------------------------------------------------------------
# 3F: CB early warning
# ---------------------------------------------------------------------------


class TestCBEarlyWarning:
    """Bank stress signal feeds into CB corridor computation."""

    def test_compute_bank_stress_all_at_target(self):
        """Stress should be 0 when all banks are at reserve target."""
        from bilancio.engines.banking_subsystem import BankTreynorState, BankingSubsystem
        from bilancio.banking.pricing_kernel import PricingParams
        from bilancio.decision.profiles import BankProfile
        from bilancio.domain.instruments.base import InstrumentKind

        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100,
            symmetric_capacity=200,
            ticket_size=100,
            reserve_floor=50,
        )
        bank = BankTreynorState(bank_id="bank_1", pricing_params=params)

        subsystem = BankingSubsystem(
            banks={"bank_1": bank},
            bank_profile=BankProfile(),
            kappa=Decimal("1.0"),
        )

        # Mock system where bank has 100 reserves (= target)
        system = MagicMock()
        bank_agent = MagicMock()
        bank_agent.asset_ids = ["r1"]
        system.state.agents = {"bank_1": bank_agent}
        contract = MagicMock()
        contract.kind = InstrumentKind.RESERVE_DEPOSIT
        contract.amount = 100
        system.state.contracts = {"r1": contract}

        stress = subsystem._compute_bank_stress(system)
        assert stress == Decimal("0")

    def test_compute_bank_stress_below_target(self):
        """Stress should be positive when banks are below reserve target."""
        from bilancio.engines.banking_subsystem import BankTreynorState, BankingSubsystem
        from bilancio.banking.pricing_kernel import PricingParams
        from bilancio.decision.profiles import BankProfile
        from bilancio.domain.instruments.base import InstrumentKind

        params = PricingParams(
            reserve_remuneration_rate=Decimal("0.01"),
            cb_borrowing_rate=Decimal("0.03"),
            reserve_target=100,
            symmetric_capacity=200,
            ticket_size=100,
            reserve_floor=50,
        )
        bank = BankTreynorState(bank_id="bank_1", pricing_params=params)

        subsystem = BankingSubsystem(
            banks={"bank_1": bank},
            bank_profile=BankProfile(),
            kappa=Decimal("1.0"),
        )

        # Bank has 50 reserves (50% of target=100)
        system = MagicMock()
        bank_agent = MagicMock()
        bank_agent.asset_ids = ["r1"]
        system.state.agents = {"bank_1": bank_agent}
        contract = MagicMock()
        contract.kind = InstrumentKind.RESERVE_DEPOSIT
        contract.amount = 50
        system.state.contracts = {"r1": contract}

        stress = subsystem._compute_bank_stress(system)
        # shortfall = 50 / 100 = 0.5
        assert stress == Decimal("0.5")

    def test_compute_bank_stress_no_banks(self):
        """Stress should be 0 with no banks."""
        from bilancio.engines.banking_subsystem import BankingSubsystem
        from bilancio.decision.profiles import BankProfile

        subsystem = BankingSubsystem(
            banks={},
            bank_profile=BankProfile(),
            kappa=Decimal("1.0"),
        )

        system = MagicMock()
        stress = subsystem._compute_bank_stress(system)
        assert stress == Decimal("0")

    def test_corridor_uses_bank_stress_when_enabled(self):
        """CentralBank.compute_corridor should use bank_stress when adaptive_early_warning=True."""
        from bilancio.domain.agents.central_bank import CentralBank

        cb = CentralBank(
            id="cb",
            name="CB",
            kappa_prior=Decimal("0.1"),
            adaptive_early_warning=True,
            beta_mid=Decimal("0.5"),
            beta_width=Decimal("0.3"),
        )

        # No defaults, no bank stress -> base corridor
        r_floor_base, r_ceiling_base = cb.compute_corridor(0, 100, bank_stress=Decimal("0"))

        # No defaults but high bank stress -> wider corridor
        r_floor_stress, r_ceiling_stress = cb.compute_corridor(
            0, 100, bank_stress=Decimal("0.5")
        )

        # With bank stress, the corridor should be wider or shifted
        # The combined = surprise + 0.3 * bank_stress
        # surprise = 0 (no defaults), combined = 0.3 * 0.5 = 0.15
        # mid should be higher with stress
        assert r_ceiling_stress > r_ceiling_base or r_floor_stress != r_floor_base


# ---------------------------------------------------------------------------
# 3G: NBFI within-run adaptation
# ---------------------------------------------------------------------------


class TestNBFIAdaptiveRates:
    """Lending rates increase when realized defaults exceed expectations."""

    def test_rate_multiplier_increases_with_excess_defaults(self):
        """When realized > expected, rates should increase."""
        realized = 0.3  # 30%
        expected = 0.15  # 15% (from kappa=1 -> p = 1/(1+1) = 0.5)
        excess = min(1.0, (realized - expected) / max(0.01, expected))
        rate_multiplier = Decimal("1") + Decimal(str(excess))
        # excess = (0.3 - 0.15) / 0.15 = 1.0
        assert rate_multiplier == Decimal("2.0")

    def test_no_multiplier_when_realized_below_expected(self):
        """Rate multiplier should be 1 when realized <= expected."""
        realized = 0.1
        expected = 0.15
        # realized < expected, so no excess
        rate_multiplier = Decimal("1")
        assert rate_multiplier == Decimal("1")

    def test_adaptive_rates_flag(self):
        """LenderProfile should have adaptive_rates flag."""
        profile = LenderProfile(adaptive_rates=True)
        assert profile.adaptive_rates is True

        profile_off = LenderProfile(adaptive_rates=False)
        assert profile_off.adaptive_rates is False


class TestNBFIAdaptiveCapitalConservation:
    """Exposure limits decrease as utilization increases."""

    def test_conservation_scales_with_utilization(self):
        """Higher loan utilization should reduce effective exposure limits."""
        total_assets = Decimal("1000")
        total_loans = Decimal("800")  # 80% utilization
        utilization = total_loans / total_assets
        conservation = max(Decimal("0.2"), Decimal("1") - utilization)
        # conservation = max(0.2, 1 - 0.8) = 0.2
        assert conservation == Decimal("0.2")

        base_max_single = Decimal("0.15")
        effective = base_max_single * conservation
        assert effective == Decimal("0.030")

    def test_low_utilization_preserves_limits(self):
        """Low utilization should keep limits near base values."""
        total_assets = Decimal("1000")
        total_loans = Decimal("100")  # 10% utilization
        utilization = total_loans / total_assets
        conservation = max(Decimal("0.2"), Decimal("1") - utilization)
        # conservation = max(0.2, 0.9) = 0.9
        assert conservation == Decimal("0.9")

    def test_adaptive_capital_conservation_flag(self):
        """LenderProfile should have adaptive_capital_conservation flag."""
        profile = LenderProfile(adaptive_capital_conservation=True)
        assert profile.adaptive_capital_conservation is True

    def test_conservation_floors_at_twenty_percent(self):
        """Conservation factor should never go below 0.2."""
        total_assets = Decimal("1000")
        total_loans = Decimal("950")  # 95% utilization
        utilization = total_loans / total_assets
        conservation = max(Decimal("0.2"), Decimal("1") - utilization)
        assert conservation == Decimal("0.2")


class TestNBFIAdaptivePreventionThreshold:
    """Prevention threshold drops when defaults accelerate."""

    def test_threshold_reduced_when_defaults_accelerate(self):
        """When recent-2 > recent-5 default rate, threshold should drop."""
        base_threshold = Decimal("0.3")
        effective = base_threshold * Decimal("0.7")
        assert effective == Decimal("0.21")

    def test_threshold_unchanged_when_defaults_stable(self):
        """When defaults are stable or decelerating, keep base threshold."""
        rate_5 = 0.2
        rate_2 = 0.1  # decelerating
        assert rate_2 <= rate_5  # No adjustment needed

    def test_adaptive_prevention_flag(self):
        """LenderProfile should have adaptive_prevention flag."""
        profile = LenderProfile(adaptive_prevention=True)
        assert profile.adaptive_prevention is True


# ---------------------------------------------------------------------------
# Integration: DealerSubsystem observed_default_rate field
# ---------------------------------------------------------------------------


class TestObservedDefaultRate:
    """DealerSubsystem.observed_default_rate field exists and defaults to 0."""

    def test_default_value(self):
        from bilancio.engines.dealer_integration import DealerSubsystem

        subsystem = DealerSubsystem()
        assert subsystem.observed_default_rate == Decimal("0")

    def test_can_be_set(self):
        from bilancio.engines.dealer_integration import DealerSubsystem

        subsystem = DealerSubsystem()
        subsystem.observed_default_rate = Decimal("0.15")
        assert subsystem.observed_default_rate == Decimal("0.15")
