"""Tests for bank credit risk pricing (Plan 036 Part A)."""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import BankProfile
from bilancio.engines.bank_lending import _per_borrower_rate


class FakeRiskAssessor:
    """Fake risk assessor returning a fixed default probability."""

    def __init__(self, p_default: Decimal = Decimal("0.10")):
        self.p_default = p_default

    def estimate_default_prob(self, issuer_id, current_day):
        return self.p_default


class FakeBanking:
    """Minimal stand-in for BankingSubsystem for _per_borrower_rate tests."""

    def __init__(self, bank_profile, risk_assessor=None):
        self.bank_profile = bank_profile
        self.risk_assessor = risk_assessor


class TestPerBorrowerRate:
    """Test _per_borrower_rate helper."""

    def test_no_loading_returns_base_rate(self):
        """credit_risk_loading=0 returns base rate unchanged (backward compat)."""
        profile = BankProfile(credit_risk_loading=Decimal("0"))
        banking = FakeBanking(profile, risk_assessor=FakeRiskAssessor())
        result = _per_borrower_rate(Decimal("0.05"), "borrower1", banking, 1)
        assert result == Decimal("0.05")

    def test_with_loading_increases_rate(self):
        """Rate increases proportional to credit_risk_loading × P_default."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.5"),
            max_borrower_risk=Decimal("1.0"),
        )
        assessor = FakeRiskAssessor(p_default=Decimal("0.20"))
        banking = FakeBanking(profile, risk_assessor=assessor)
        result = _per_borrower_rate(Decimal("0.05"), "borrower1", banking, 1)
        # 0.05 + 0.5 * 0.20 = 0.05 + 0.10 = 0.15
        assert result == Decimal("0.15")

    def test_credit_rationing_returns_none(self):
        """Returns None when P_default > max_borrower_risk (credit rationed)."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.5"),
            max_borrower_risk=Decimal("0.15"),
        )
        assessor = FakeRiskAssessor(p_default=Decimal("0.20"))
        banking = FakeBanking(profile, risk_assessor=assessor)
        result = _per_borrower_rate(Decimal("0.05"), "borrower1", banking, 1)
        assert result is None

    def test_no_risk_assessor_returns_base(self):
        """No risk assessor returns base rate even with loading > 0."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.5"),
        )
        banking = FakeBanking(profile, risk_assessor=None)
        result = _per_borrower_rate(Decimal("0.05"), "borrower1", banking, 1)
        assert result == Decimal("0.05")

    def test_at_max_risk_boundary_not_rationed(self):
        """P_default exactly at max_borrower_risk is NOT rationed (strict >)."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.5"),
            max_borrower_risk=Decimal("0.20"),
        )
        assessor = FakeRiskAssessor(p_default=Decimal("0.20"))
        banking = FakeBanking(profile, risk_assessor=assessor)
        result = _per_borrower_rate(Decimal("0.05"), "borrower1", banking, 1)
        # p > max_borrower_risk uses strict inequality: 0.20 > 0.20 is False
        assert result is not None
        assert result == Decimal("0.05") + Decimal("0.5") * Decimal("0.20")


class TestBankProfileValidation:
    """Test BankProfile validation of new fields."""

    def test_negative_credit_risk_loading_raises(self):
        with pytest.raises(ValueError, match="credit_risk_loading"):
            BankProfile(credit_risk_loading=Decimal("-0.1"))

    def test_zero_max_borrower_risk_raises(self):
        with pytest.raises(ValueError, match="max_borrower_risk"):
            BankProfile(max_borrower_risk=Decimal("0"))

    def test_over_one_max_borrower_risk_raises(self):
        with pytest.raises(ValueError, match="max_borrower_risk"):
            BankProfile(max_borrower_risk=Decimal("1.5"))

    def test_defaults_are_valid(self):
        """Default values pass validation."""
        profile = BankProfile()
        assert profile.credit_risk_loading == Decimal("0")
        assert profile.max_borrower_risk == Decimal("1.0")
