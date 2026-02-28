"""Tests for issuer-specific pricing overlay (Feature 1).

Verifies that:
- The price adjustment helper returns correct multipliers
- Riskier issuers get lower prices (multiplier < 1)
- Safer issuers get higher prices (multiplier > 1)
- The feature is a no-op when disabled (multiplier = 1)
- The floor at 0 prevents negative prices
"""

from decimal import Decimal

import pytest

from bilancio.engines.dealer_integration import DealerSubsystem
from bilancio.engines.dealer_trades import _get_issuer_price_adjustment


@pytest.fixture
def subsystem_with_issuer_pricing() -> DealerSubsystem:
    """Create a DealerSubsystem with issuer-specific pricing enabled."""
    sub = DealerSubsystem()
    sub.issuer_specific_pricing = True
    sub.system_default_prob = Decimal("0.20")
    sub.issuer_default_probs = {
        "risky_issuer": Decimal("0.30"),  # riskier than system
        "safe_issuer": Decimal("0.10"),   # safer than system
        "average_issuer": Decimal("0.20"),  # exactly system average
    }
    return sub


class TestGetIssuerPriceAdjustment:
    """Tests for _get_issuer_price_adjustment helper."""

    def test_disabled_returns_one(self) -> None:
        """When issuer_specific_pricing is False, always returns 1."""
        sub = DealerSubsystem()
        sub.issuer_specific_pricing = False
        assert _get_issuer_price_adjustment(sub, "any_issuer") == Decimal(1)

    def test_riskier_issuer_lower_multiplier(
        self, subsystem_with_issuer_pricing: DealerSubsystem
    ) -> None:
        """Riskier issuer (P_i > P_system) gets multiplier < 1."""
        sub = subsystem_with_issuer_pricing
        factor = _get_issuer_price_adjustment(sub, "risky_issuer")
        # P_i=0.30, P_system=0.20 → adjustment=0.10 → multiplier=0.90
        assert factor == Decimal("0.90")
        assert factor < Decimal(1)

    def test_safer_issuer_higher_multiplier(
        self, subsystem_with_issuer_pricing: DealerSubsystem
    ) -> None:
        """Safer issuer (P_i < P_system) gets multiplier > 1."""
        sub = subsystem_with_issuer_pricing
        factor = _get_issuer_price_adjustment(sub, "safe_issuer")
        # P_i=0.10, P_system=0.20 → adjustment=-0.10 → multiplier=1.10
        assert factor == Decimal("1.10")
        assert factor > Decimal(1)

    def test_average_issuer_no_adjustment(
        self, subsystem_with_issuer_pricing: DealerSubsystem
    ) -> None:
        """Average issuer (P_i = P_system) gets multiplier = 1."""
        sub = subsystem_with_issuer_pricing
        factor = _get_issuer_price_adjustment(sub, "average_issuer")
        assert factor == Decimal(1)

    def test_unknown_issuer_uses_system_default(
        self, subsystem_with_issuer_pricing: DealerSubsystem
    ) -> None:
        """Unknown issuer (not in dict) falls back to P_system → multiplier = 1."""
        sub = subsystem_with_issuer_pricing
        factor = _get_issuer_price_adjustment(sub, "unknown_issuer")
        assert factor == Decimal(1)

    def test_floor_prevents_negative(self) -> None:
        """Multiplier is floored at 0, even with extremely risky issuers."""
        sub = DealerSubsystem()
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.10")
        # P_i=1.0, P_system=0.10 → adjustment=0.90 → 1-0.90=0.10 (still positive)
        sub.issuer_default_probs = {"very_risky": Decimal("1.0")}
        assert _get_issuer_price_adjustment(sub, "very_risky") == Decimal("0.10")

        # Edge case: P_i much larger than 1 somehow (shouldn't happen, but test floor)
        sub.issuer_default_probs = {"extreme": Decimal("2.0")}
        factor = _get_issuer_price_adjustment(sub, "extreme")
        assert factor == Decimal(0)  # Floored at 0

    def test_symmetry_around_system(
        self, subsystem_with_issuer_pricing: DealerSubsystem
    ) -> None:
        """Adjustments are symmetric: excess risk of +X gives same magnitude as -X."""
        sub = subsystem_with_issuer_pricing
        risky_factor = _get_issuer_price_adjustment(sub, "risky_issuer")
        safe_factor = _get_issuer_price_adjustment(sub, "safe_issuer")
        # risky: 1 - 0.10 = 0.90
        # safe:  1 - (-0.10) = 1.10
        # They should be symmetric around 1.0
        risky_deviation = Decimal(1) - risky_factor  # 0.10
        safe_deviation = safe_factor - Decimal(1)    # 0.10
        assert risky_deviation == safe_deviation


class TestIssuerPricingFields:
    """Test that DealerSubsystem fields have correct defaults."""

    def test_default_disabled(self) -> None:
        sub = DealerSubsystem()
        assert sub.issuer_specific_pricing is False

    def test_default_empty_probs(self) -> None:
        sub = DealerSubsystem()
        assert sub.issuer_default_probs == {}

    def test_default_system_prob(self) -> None:
        sub = DealerSubsystem()
        assert sub.system_default_prob == Decimal(0)
