"""Tests for VBTPricingModel protocol and CreditAdjustedVBTPricing.

Covers:
1. Protocol conformance
2. compute_mid() math
3. compute_spread() math
4. Sensitivity parameter behavior
5. Integration: _update_vbt_credit_mids delegation
6. Integration: initialize_balanced_dealer_subsystem wiring
7. DEALER_MARKET_OBSERVER preset structure
"""

from decimal import Decimal

import pytest

from bilancio.decision.protocols import VBTPricingModel
from bilancio.decision.valuers import CreditAdjustedVBTPricing
from bilancio.information.levels import AccessLevel
from bilancio.information.presets import DEALER_MARKET_OBSERVER

# ── 1. Protocol conformance ──────────────────────────────────────


class TestProtocolConformance:
    def test_isinstance_check(self):
        pricing = CreditAdjustedVBTPricing()
        assert isinstance(pricing, VBTPricingModel)

    def test_frozen(self):
        pricing = CreditAdjustedVBTPricing()
        with pytest.raises(AttributeError):
            pricing.mid_sensitivity = Decimal("0.5")  # type: ignore[misc]


# ── 2. compute_mid() math ────────────────────────────────────────


class TestComputeMid:
    def test_no_defaults(self):
        """When p_default == initial_prior, M = 1 - prior."""
        pricing = CreditAdjustedVBTPricing()
        prior = Decimal("0.15")
        M = pricing.compute_mid(p_default=prior, initial_prior=prior)
        expected = Decimal(1) - Decimal("0.15")
        assert M == expected  # 0.85

    def test_zero_defaults(self):
        """With p_default=0 and full sensitivity, M = 1."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
        )
        M = pricing.compute_mid(p_default=Decimal(0), initial_prior=Decimal("0.15"))
        # raw_M = 1 - 0 = 1.0
        # initial_M = 1 - 0.15 = 0.85
        # result = 0.85 + 1.0 × (1.0 - 0.85) = 1.0
        assert M == Decimal("1.0")

    def test_high_defaults(self):
        """With high p_default, M is reduced."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
        )
        M = pricing.compute_mid(p_default=Decimal("0.50"), initial_prior=Decimal("0.15"))
        # raw_M = 1 - 0.50 = 0.50
        # initial_M = 1 - 0.15 = 0.85
        # result = 0.85 + 1.0 × (0.50 - 0.85) = 0.50
        assert M == Decimal("0.50")

    def test_sensitivity_zero_ignores_defaults(self):
        """With mid_sensitivity=0, M stays at initial value."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("0"),
        )
        M = pricing.compute_mid(p_default=Decimal("0.80"), initial_prior=Decimal("0.15"))
        # initial_M = 1 - 0.15 = 0.85
        # With sensitivity=0, M = initial_M always
        expected = Decimal("0.85")
        assert M == expected

    def test_sensitivity_half(self):
        """With mid_sensitivity=0.5, M is blended halfway."""
        pricing = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("0.5"),
        )
        M = pricing.compute_mid(p_default=Decimal("0.40"), initial_prior=Decimal("0.20"))
        # raw_M = 1 - 0.40 = 0.60
        # initial_M = 1 - 0.20 = 0.80
        # result = 0.80 + 0.5 × (0.60 - 0.80) = 0.80 - 0.10 = 0.70
        assert M == Decimal("0.70")


# ── 3. compute_spread() math ─────────────────────────────────────


class TestComputeSpread:
    def test_zero_sensitivity_returns_base(self):
        """With spread_sensitivity=0, spread is unchanged."""
        pricing = CreditAdjustedVBTPricing(spread_sensitivity=Decimal("0"))
        spread = pricing.compute_spread(Decimal("0.30"), Decimal("0.20"))
        assert spread == Decimal("0.30")

    def test_positive_sensitivity_widens(self):
        """With spread_sensitivity > 0, spread widens with defaults."""
        pricing = CreditAdjustedVBTPricing(spread_sensitivity=Decimal("1.0"))
        spread = pricing.compute_spread(Decimal("0.30"), Decimal("0.20"))
        # 0.30 + 1.0 × 0.20 = 0.50
        assert spread == Decimal("0.50")

    def test_no_defaults_no_widening(self):
        """With p_default=0, spread stays at base even with sensitivity."""
        pricing = CreditAdjustedVBTPricing(spread_sensitivity=Decimal("1.0"))
        spread = pricing.compute_spread(Decimal("0.30"), Decimal("0"))
        assert spread == Decimal("0.30")


# ── 4. Default parameter values ──────────────────────────────────


class TestDefaults:
    def test_default_mid_sensitivity(self):
        pricing = CreditAdjustedVBTPricing()
        assert pricing.mid_sensitivity == Decimal("1.0")

    def test_default_spread_sensitivity(self):
        pricing = CreditAdjustedVBTPricing()
        assert pricing.spread_sensitivity == Decimal("0.6")


# ── 5. Integration: _update_vbt_credit_mids delegation ───────────


class TestUpdateVBTCreditMidsDelegation:
    """Verify that _update_vbt_credit_mids produces identical results
    whether using the pricing model or inline logic."""

    def _build_subsystem(self, with_pricing_model: bool):
        """Build a minimal DealerSubsystem for testing."""
        from bilancio.dealer.models import VBTState
        from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
        from bilancio.decision.profiles import VBTProfile
        from bilancio.engines.dealer_integration import DealerSubsystem

        subsystem = DealerSubsystem()
        subsystem.risk_assessor = RiskAssessor(
            RiskAssessmentParams(
                initial_prior=Decimal("0.15"),
            )
        )
        subsystem.outside_mid_ratio = Decimal("0.75")
        subsystem.vbt_profile = VBTProfile(
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.0"),
        )
        subsystem.vbts = {
            "short": VBTState(bucket_id="short", M=Decimal(1), O=Decimal("0.30")),
            "mid": VBTState(bucket_id="mid", M=Decimal(1), O=Decimal("0.30")),
        }
        subsystem.initial_spread_by_bucket = {
            "short": Decimal("0.30"),
            "mid": Decimal("0.30"),
        }

        if with_pricing_model:
            subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
                mid_sensitivity=Decimal("1.0"),
                spread_sensitivity=Decimal("0.0"),
            )
        else:
            subsystem.vbt_pricing_model = None

        return subsystem

    def test_model_updates_mid_and_spread(self):
        """Pricing model correctly updates M and spread via delegation."""
        from bilancio.engines.dealer_sync import _update_vbt_credit_mids

        sub = self._build_subsystem(with_pricing_model=True)
        _update_vbt_credit_mids(sub, current_day=0)

        # With initial_prior=0.15, no history → p_default=0.15
        # M = 1 - 0.15 = 0.85
        for bucket in ("short", "mid"):
            assert sub.vbts[bucket].M == Decimal("0.85")
            # spread_sensitivity=0 → spread stays at base (0.30)
            assert sub.vbts[bucket].O == Decimal("0.30")

    def test_model_with_spread_sensitivity(self):
        """With spread_sensitivity > 0, spreads widen additively."""
        from bilancio.engines.dealer_sync import _update_vbt_credit_mids

        sub = self._build_subsystem(with_pricing_model=True)
        sub.vbt_pricing_model = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1.0"),
            spread_sensitivity=Decimal("0.5"),
        )

        _update_vbt_credit_mids(sub, current_day=0)

        # p_default=0.15 (from prior), M = 1 - 0.15 = 0.85
        # spread = 0.30 + 0.5 × 0.15 = 0.375
        for bucket in ("short", "mid"):
            assert sub.vbts[bucket].M == Decimal("0.85")
            assert sub.vbts[bucket].O == Decimal("0.375")

    def test_no_risk_assessor_skips(self):
        """Without risk_assessor, function returns early."""
        from bilancio.engines.dealer_sync import _update_vbt_credit_mids

        sub = self._build_subsystem(with_pricing_model=True)
        sub.risk_assessor = None
        original_M = sub.vbts["short"].M

        _update_vbt_credit_mids(sub, current_day=0)

        # M unchanged
        assert sub.vbts["short"].M == original_M


# ── 6. Integration: initialization wiring ────────────────────────


class TestInitializationWiring:
    """Verify that initialize_balanced_dealer_subsystem attaches a pricing model."""

    def _build_system(self):
        """Build a minimal system for balanced initialization."""
        from bilancio.domain.agents.central_bank import CentralBank
        from bilancio.domain.agents.household import Household
        from bilancio.engines.system import System

        system = System()
        cb = CentralBank(id="CB01", name="CB", kind="central_bank")
        system.bootstrap_cb(cb)

        # Add some agents
        for i in range(5):
            h = Household(id=f"H{i:02d}", name=f"Agent {i}", kind="household")
            system.add_agent(h)
            system.mint_cash(f"H{i:02d}", 100)

        return system

    def test_pricing_model_attached(self):
        from bilancio.dealer.risk_assessment import RiskAssessmentParams
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.engines.dealer_integration import initialize_balanced_dealer_subsystem

        system = self._build_system()
        config = DealerRingConfig(ticket_size=Decimal(1))
        sub = initialize_balanced_dealer_subsystem(
            system,
            config,
            outside_mid_ratio=Decimal("0.80"),
            risk_params=RiskAssessmentParams(),
        )
        assert sub.vbt_pricing_model is not None
        assert isinstance(sub.vbt_pricing_model, CreditAdjustedVBTPricing)

    def test_pricing_model_uses_vbt_profile(self):
        from bilancio.dealer.risk_assessment import RiskAssessmentParams
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.decision.profiles import VBTProfile
        from bilancio.engines.dealer_integration import initialize_balanced_dealer_subsystem

        system = self._build_system()
        config = DealerRingConfig(ticket_size=Decimal(1))
        profile = VBTProfile(
            mid_sensitivity=Decimal("0.5"),
            spread_sensitivity=Decimal("0.3"),
        )
        sub = initialize_balanced_dealer_subsystem(
            system,
            config,
            outside_mid_ratio=Decimal("0.75"),
            vbt_profile=profile,
            risk_params=RiskAssessmentParams(),
        )
        assert sub.vbt_pricing_model.mid_sensitivity == Decimal("0.5")
        assert sub.vbt_pricing_model.spread_sensitivity == Decimal("0.3")


# ── 7. DEALER_MARKET_OBSERVER preset ─────────────────────────────


class TestDealerMarketObserverPreset:
    def test_vbt_anchors_perfect(self):
        assert DEALER_MARKET_OBSERVER.vbt_anchors.level == AccessLevel.PERFECT

    def test_dealer_quotes_perfect(self):
        assert DEALER_MARKET_OBSERVER.dealer_quotes.level == AccessLevel.PERFECT

    def test_no_balance_sheet_access(self):
        assert DEALER_MARKET_OBSERVER.counterparty_cash.level == AccessLevel.NONE
        assert DEALER_MARKET_OBSERVER.counterparty_assets.level == AccessLevel.NONE

    def test_settlement_history_perfect(self):
        assert DEALER_MARKET_OBSERVER.counterparty_default_history.level == AccessLevel.PERFECT

    def test_has_channel_binding(self):
        assert len(DEALER_MARKET_OBSERVER.channel_bindings) == 1
        binding = DEALER_MARKET_OBSERVER.channel_bindings[0]
        assert binding.category == "vbt_quotes"
        assert binding.source == "market_observation"

    def test_no_network_access(self):
        assert DEALER_MARKET_OBSERVER.obligation_graph.level == AccessLevel.NONE
        assert DEALER_MARKET_OBSERVER.counterparty_connectivity.level == AccessLevel.NONE
        assert DEALER_MARKET_OBSERVER.cascade_risk.level == AccessLevel.NONE
