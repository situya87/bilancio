"""Unit tests for BankProfile Treynor pricing parameters.

Tests cover:
1. Corridor midpoint (r_mid) as function of kappa
2. Corridor width (omega) as function of kappa
3. r_floor < r_ceiling invariant
4. Loan maturity computation and clamping
5. Validation of negative r_base
"""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import BankProfile


class TestCorridorMid:
    """Tests for corridor midpoint r_mid = r_base + r_stress * stress_factor."""

    def test_corridor_mid_at_kappa_1(self):
        """r_mid = r_base when kappa >= 1 (stress factor = 0)."""
        bp = BankProfile()
        # At kappa=1, stress_factor = max(0, 1-1)/(1+1) = 0
        mid = bp.corridor_mid(Decimal("1"))
        assert mid == bp.r_base

    def test_corridor_mid_at_kappa_above_1(self):
        """r_mid = r_base for kappa > 1 as well (stress clamped to 0)."""
        bp = BankProfile()
        mid = bp.corridor_mid(Decimal("5"))
        assert mid == bp.r_base

    def test_corridor_mid_at_kappa_03(self):
        """r_mid increases as kappa falls below 1."""
        bp = BankProfile()
        mid = bp.corridor_mid(Decimal("0.3"))
        # stress_factor = max(0, 1-0.3)/(1+0.3) = 0.7/1.3
        expected_stress = Decimal("0.7") / Decimal("1.3")
        expected_mid = bp.r_base + bp.r_stress * expected_stress
        assert mid == expected_mid
        # r_mid should be strictly greater than r_base when kappa < 1
        assert mid > bp.r_base


class TestCorridorWidth:
    """Tests for corridor width omega = omega_base + omega_stress * stress_factor."""

    def test_corridor_width_at_kappa_1(self):
        """omega = omega_base when kappa >= 1."""
        bp = BankProfile()
        width = bp.corridor_width(Decimal("1"))
        assert width == bp.omega_base

    def test_corridor_width_stress(self):
        """omega widens as kappa falls below 1."""
        bp = BankProfile()
        width_1 = bp.corridor_width(Decimal("1"))
        width_03 = bp.corridor_width(Decimal("0.3"))
        # Width at kappa=0.3 should be larger than at kappa=1
        assert width_03 > width_1

    def test_corridor_width_at_kappa_0_5(self):
        """Verify exact corridor width computation at kappa=0.5."""
        bp = BankProfile()
        width = bp.corridor_width(Decimal("0.5"))
        # stress_factor = max(0, 1-0.5)/(1+0.5) = 0.5/1.5 = 1/3
        expected_stress = Decimal("0.5") / Decimal("1.5")
        expected_width = bp.omega_base + bp.omega_stress * expected_stress
        assert width == expected_width


class TestFloorCeiling:
    """Tests for r_floor < r_ceiling invariant."""

    def test_r_floor_r_ceiling(self):
        """r_floor < r_ceiling for any kappa."""
        bp = BankProfile()
        for kappa_str in ("0.1", "0.3", "0.5", "1", "2", "5"):
            kappa = Decimal(kappa_str)
            r_floor = bp.r_floor(kappa)
            r_ceiling = bp.r_ceiling(kappa)
            assert r_floor < r_ceiling, (
                f"r_floor ({r_floor}) must be < r_ceiling ({r_ceiling}) at kappa={kappa}"
            )

    def test_r_floor_equals_mid_minus_half_width(self):
        """r_floor = corridor_mid - corridor_width / 2."""
        bp = BankProfile()
        kappa = Decimal("0.5")
        assert bp.r_floor(kappa) == bp.corridor_mid(kappa) - bp.corridor_width(kappa) / 2

    def test_r_ceiling_equals_mid_plus_half_width(self):
        """r_ceiling = corridor_mid + corridor_width / 2."""
        bp = BankProfile()
        kappa = Decimal("0.5")
        assert bp.r_ceiling(kappa) == bp.corridor_mid(kappa) + bp.corridor_width(kappa) / 2


class TestLoanMaturity:
    """Tests for bank loan maturity computation."""

    def test_loan_maturity(self):
        """loan_maturity(10) = 5 (with default fraction 0.5)."""
        bp = BankProfile()
        assert bp.loan_maturity(10) == 5

    def test_loan_maturity_minimum(self):
        """loan_maturity(2) = 2 (min clamp prevents going below 2)."""
        bp = BankProfile()
        # 2 * 0.5 = 1, clamped to min=2
        assert bp.loan_maturity(2) == 2

    def test_loan_maturity_minimum_with_1(self):
        """loan_maturity(1) = 2 (min clamp at 2)."""
        bp = BankProfile()
        # 1 * 0.5 = 0, clamped to min=2
        assert bp.loan_maturity(1) == 2

    def test_loan_maturity_scales(self):
        """loan_maturity(20) = 10 with default fraction."""
        bp = BankProfile()
        assert bp.loan_maturity(20) == 10


class TestValidation:
    """Tests for BankProfile validation."""

    def test_validation_r_base_negative(self):
        """Raises ValueError for negative r_base."""
        with pytest.raises(ValueError, match="r_base must be non-negative"):
            BankProfile(r_base=Decimal("-0.01"))

    def test_validation_r_stress_negative(self):
        """Raises ValueError for negative r_stress."""
        with pytest.raises(ValueError, match="r_stress must be non-negative"):
            BankProfile(r_stress=Decimal("-1"))

    def test_validation_omega_base_negative(self):
        """Raises ValueError for negative omega_base."""
        with pytest.raises(ValueError, match="omega_base must be non-negative"):
            BankProfile(omega_base=Decimal("-0.01"))

    def test_validation_loan_maturity_fraction_zero(self):
        """Raises ValueError for loan_maturity_fraction = 0."""
        with pytest.raises(ValueError, match="loan_maturity_fraction must be in"):
            BankProfile(loan_maturity_fraction=Decimal("0"))
