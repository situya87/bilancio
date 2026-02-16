"""Unit tests for the decision module (TraderProfile and VBTProfile)."""

from decimal import Decimal

import pytest

from bilancio.decision.presets import AGGRESSIVE, BASELINE, CAUTIOUS
from bilancio.decision.profiles import TraderProfile, VBTProfile

# ---------------------------------------------------------------------------
# TraderProfile defaults (backward compatibility)
# ---------------------------------------------------------------------------


class TestTraderProfileDefaults:
    """Default TraderProfile must reproduce current hardcoded behavior."""

    def test_default_base_risk_premium(self):
        tp = TraderProfile()
        assert tp.base_risk_premium == Decimal("0")  # Seller premium = 0

    def test_default_buy_risk_premium(self):
        tp = TraderProfile()
        assert tp.buy_risk_premium == Decimal("0.01")  # Buyer premium = 0.01

    def test_default_buy_premium_multiplier(self):
        tp = TraderProfile()
        assert tp.buy_premium_multiplier == Decimal("1.0")

    def test_default_sell_horizon(self):
        tp = TraderProfile()
        assert tp.sell_horizon == 10

    def test_default_buy_horizon(self):
        tp = TraderProfile()
        assert tp.buy_horizon == 5

    def test_default_surplus_threshold_factor(self):
        tp = TraderProfile()
        assert tp.surplus_threshold_factor == Decimal("0")


# ---------------------------------------------------------------------------
# TraderProfile computed properties
# ---------------------------------------------------------------------------


class TestTraderProfileProperties:
    def test_risk_aversion_seller_premium_always_zero(self):
        tp = TraderProfile(risk_aversion=Decimal("0.5"))
        assert tp.base_risk_premium == Decimal("0")  # Always 0 regardless of risk aversion

    def test_risk_aversion_increases_buy_premium(self):
        tp = TraderProfile(risk_aversion=Decimal("0.5"))
        # 0.01 + 0.02 * 0.5 = 0.02
        assert tp.buy_risk_premium == Decimal("0.02")

    def test_max_risk_aversion_buy_premium(self):
        tp = TraderProfile(risk_aversion=Decimal("1"))
        # 0.01 + 0.02 * 1 = 0.03
        assert tp.buy_risk_premium == Decimal("0.03")

    def test_risk_aversion_increases_buy_multiplier(self):
        tp = TraderProfile(risk_aversion=Decimal("0.5"))
        assert tp.buy_premium_multiplier == Decimal("1.5")

    def test_sell_horizon_equals_planning_horizon(self):
        tp = TraderProfile(planning_horizon=15)
        assert tp.sell_horizon == 15

    def test_buy_horizon_half_of_planning(self):
        tp = TraderProfile(planning_horizon=14)
        assert tp.buy_horizon == 7

    def test_buy_horizon_odd_planning(self):
        tp = TraderProfile(planning_horizon=15)
        assert tp.buy_horizon == 7  # 15 // 2 = 7

    def test_buy_horizon_minimum_one(self):
        tp = TraderProfile(planning_horizon=1)
        assert tp.buy_horizon == 1  # max(1, 1//2) = max(1, 0) = 1

    def test_aggressiveness_zero_conservative(self):
        tp = TraderProfile(aggressiveness=Decimal("0"))
        assert tp.surplus_threshold_factor == Decimal("1")

    def test_aggressiveness_half(self):
        tp = TraderProfile(aggressiveness=Decimal("0.5"))
        assert tp.surplus_threshold_factor == Decimal("0.5")


# ---------------------------------------------------------------------------
# TraderProfile validation
# ---------------------------------------------------------------------------


class TestTraderProfileValidation:
    def test_planning_horizon_too_low(self):
        with pytest.raises(ValueError, match="planning_horizon"):
            TraderProfile(planning_horizon=0)

    def test_planning_horizon_too_high(self):
        with pytest.raises(ValueError, match="planning_horizon"):
            TraderProfile(planning_horizon=21)

    def test_planning_horizon_boundary_low(self):
        tp = TraderProfile(planning_horizon=1)
        assert tp.planning_horizon == 1

    def test_planning_horizon_boundary_high(self):
        tp = TraderProfile(planning_horizon=20)
        assert tp.planning_horizon == 20


# ---------------------------------------------------------------------------
# TraderProfile immutability
# ---------------------------------------------------------------------------


class TestTraderProfileFrozen:
    def test_frozen(self):
        tp = TraderProfile()
        with pytest.raises(AttributeError):
            tp.risk_aversion = Decimal("0.5")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VBTProfile
# ---------------------------------------------------------------------------


class TestVBTProfileDefaults:
    def test_default_mid_sensitivity(self):
        vp = VBTProfile()
        assert vp.mid_sensitivity == Decimal("1.0")

    def test_default_spread_sensitivity(self):
        vp = VBTProfile()
        assert vp.spread_sensitivity == Decimal("0.0")

    def test_custom_values(self):
        vp = VBTProfile(mid_sensitivity=Decimal("0.5"), spread_sensitivity=Decimal("0.3"))
        assert vp.mid_sensitivity == Decimal("0.5")
        assert vp.spread_sensitivity == Decimal("0.3")

    def test_frozen(self):
        vp = VBTProfile()
        with pytest.raises(AttributeError):
            vp.mid_sensitivity = Decimal("0.5")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_baseline_is_default(self):
        tp, vp = BASELINE
        assert tp == TraderProfile()
        assert vp == VBTProfile()

    def test_cautious_higher_risk_aversion(self):
        tp, vp = CAUTIOUS
        assert tp.risk_aversion == Decimal("0.5")
        assert tp.planning_horizon == 15
        assert tp.aggressiveness == Decimal("0.5")

    def test_aggressive_short_horizon(self):
        tp, vp = AGGRESSIVE
        assert tp.planning_horizon == 5
        assert vp.spread_sensitivity == Decimal("0.5")
