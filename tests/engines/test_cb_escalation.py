"""Tests for CB rate escalation and lending cap (Plan 036 Parts B+C)."""

from decimal import Decimal

from bilancio.domain.agents.central_bank import CentralBank


class TestEffectiveLendingRate:
    """Test CentralBank.effective_lending_rate()."""

    def test_no_escalation_returns_base(self):
        """slope=0 returns static cb_lending_rate."""
        cb = CentralBank(id="cb1", name="CB")
        assert cb.effective_lending_rate(5000) == cb.cb_lending_rate

    def test_escalation_increases_with_outstanding(self):
        """Rate increases linearly with outstanding/base_amount."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            cb_lending_rate=Decimal("0.03"),
            rate_escalation_slope=Decimal("0.05"),
            escalation_base_amount=10000,
        )
        # At outstanding=0: rate = 0.03 + 0.05 * 0/10000 = 0.03
        assert cb.effective_lending_rate(0) == Decimal("0.03")

        # At outstanding=5000: rate = 0.03 + 0.05 * 0.5 = 0.055
        assert cb.effective_lending_rate(5000) == Decimal("0.055")

        # At outstanding=10000: rate = 0.03 + 0.05 * 1.0 = 0.08
        assert cb.effective_lending_rate(10000) == Decimal("0.08")

    def test_zero_base_amount_returns_static(self):
        """escalation_base_amount=0 returns static rate (avoid division by zero)."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            rate_escalation_slope=Decimal("0.05"),
            escalation_base_amount=0,
        )
        assert cb.effective_lending_rate(5000) == cb.cb_lending_rate


class TestCanLend:
    """Test CentralBank.can_lend()."""

    def test_no_cap_always_allows(self):
        """max_outstanding_ratio=0 always returns True."""
        cb = CentralBank(id="cb1", name="CB")
        assert cb.can_lend(999999, 999999) is True

    def test_cap_allows_below(self):
        """Lending allowed when below cap."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            max_outstanding_ratio=Decimal("0.5"),
            escalation_base_amount=10000,
        )
        # Cap = 0.5 * 10000 = 5000
        assert cb.can_lend(3000, 1000) is True  # 3000 + 1000 = 4000 <= 5000

    def test_cap_blocks_at_limit(self):
        """Lending blocked when exactly at cap."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            max_outstanding_ratio=Decimal("0.5"),
            escalation_base_amount=10000,
        )
        # Cap = 5000, outstanding=4000 + amount=1000 = 5000 <= 5000
        assert cb.can_lend(4000, 1000) is True
        # But 4001 + 1000 = 5001 > 5000
        assert cb.can_lend(4001, 1000) is False

    def test_cap_blocks_above(self):
        """Lending blocked when above cap."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            max_outstanding_ratio=Decimal("0.5"),
            escalation_base_amount=10000,
        )
        assert cb.can_lend(5000, 1) is False

    def test_zero_base_amount_allows(self):
        """escalation_base_amount=0 with positive ratio still allows (no cap)."""
        cb = CentralBank(
            id="cb1",
            name="CB",
            max_outstanding_ratio=Decimal("0.5"),
            escalation_base_amount=0,
        )
        assert cb.can_lend(999999, 999999) is True


class TestBackwardCompatibility:
    """Test that default CentralBank is fully backward compatible."""

    def test_defaults(self):
        cb = CentralBank(id="cb1", name="CB")
        assert cb.rate_escalation_slope == Decimal("0")
        assert cb.escalation_base_amount == 0
        assert cb.max_outstanding_ratio == Decimal("0")
        # All backward compat methods return expected values
        assert cb.effective_lending_rate(9999) == cb.cb_lending_rate
        assert cb.can_lend(9999, 9999) is True
