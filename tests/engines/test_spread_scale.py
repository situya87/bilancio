"""Tests for configurable dealer spread scaling (Plan 036 Part D)."""

from decimal import Decimal

from bilancio.decision.profiles import VBTProfile


class TestVBTProfileSpreadScale:
    """Test VBTProfile.spread_scale field."""

    def test_default_is_one(self):
        """Default spread_scale is 1.0."""
        profile = VBTProfile()
        assert profile.spread_scale == Decimal("1.0")

    def test_custom_scale(self):
        """spread_scale can be set to custom value."""
        profile = VBTProfile(spread_scale=Decimal("0.5"))
        assert profile.spread_scale == Decimal("0.5")

    def test_frozen(self):
        """VBTProfile is frozen — spread_scale cannot be mutated."""
        profile = VBTProfile(spread_scale=Decimal("2.0"))
        with __import__("pytest").raises(AttributeError):
            profile.spread_scale = Decimal("3.0")

    def test_scale_zero(self):
        """spread_scale=0 is valid (eliminates spreads)."""
        profile = VBTProfile(spread_scale=Decimal("0"))
        assert profile.spread_scale == Decimal("0")

    def test_scale_large(self):
        """spread_scale > 1 widens spreads."""
        profile = VBTProfile(spread_scale=Decimal("3.0"))
        assert profile.spread_scale == Decimal("3.0")
