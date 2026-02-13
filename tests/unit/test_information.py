"""Tests for the information access framework.

Covers:
1. AccessLevel enum
2. Noise config validation
3. CategoryAccess validation
4. InformationProfile construction
5. Presets (OMNISCIENT, LENDER_REALISTIC, TRADER_BASIC)
6. InformationService query methods
7. Noise application
8. Self-access bypass
9. Lender integration with InformationService
10. Config wiring (LenderScenarioConfig → InformationProfile)
"""

import random
from decimal import Decimal

import pytest

from bilancio.information import (
    AccessLevel,
    AggregateOnlyNoise,
    BilateralOnlyNoise,
    CategoryAccess,
    EstimationNoise,
    InformationProfile,
    InformationService,
    LENDER_REALISTIC,
    LagNoise,
    OMNISCIENT,
    SampleNoise,
    TRADER_BASIC,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. AccessLevel Enum
# ═══════════════════════════════════════════════════════════════════════


class TestAccessLevel:
    def test_values(self):
        assert AccessLevel.NONE == "none"
        assert AccessLevel.NOISY == "noisy"
        assert AccessLevel.PERFECT == "perfect"

    def test_from_string(self):
        assert AccessLevel("none") == AccessLevel.NONE
        assert AccessLevel("noisy") == AccessLevel.NOISY
        assert AccessLevel("perfect") == AccessLevel.PERFECT

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            AccessLevel("invalid")


# ═══════════════════════════════════════════════════════════════════════
# 2. Noise Config Validation
# ═══════════════════════════════════════════════════════════════════════


class TestNoiseConfigs:
    def test_lag_noise_default(self):
        n = LagNoise()
        assert n.lag_days == 1

    def test_lag_noise_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LagNoise(lag_days=-1)

    def test_sample_noise_default(self):
        n = SampleNoise()
        assert n.sample_rate == Decimal("0.7")

    def test_sample_noise_zero_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            SampleNoise(sample_rate=Decimal("0"))

    def test_sample_noise_above_one_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            SampleNoise(sample_rate=Decimal("1.5"))

    def test_estimation_noise_default(self):
        n = EstimationNoise()
        assert n.error_fraction == Decimal("0.10")

    def test_estimation_noise_negative_raises(self):
        with pytest.raises(ValueError, match="error_fraction"):
            EstimationNoise(error_fraction=Decimal("-0.1"))

    def test_aggregate_and_bilateral_are_simple(self):
        a = AggregateOnlyNoise()
        b = BilateralOnlyNoise()
        assert a is not None
        assert b is not None

    def test_frozen(self):
        n = EstimationNoise()
        with pytest.raises(AttributeError):
            n.error_fraction = Decimal("0.5")  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 3. CategoryAccess Validation
# ═══════════════════════════════════════════════════════════════════════


class TestCategoryAccess:
    def test_default_is_perfect(self):
        ca = CategoryAccess()
        assert ca.level == AccessLevel.PERFECT
        assert ca.noise is None

    def test_noisy_requires_noise(self):
        with pytest.raises(ValueError, match="noise config is required"):
            CategoryAccess(level=AccessLevel.NOISY)

    def test_noisy_with_noise_ok(self):
        ca = CategoryAccess(AccessLevel.NOISY, EstimationNoise())
        assert ca.level == AccessLevel.NOISY
        assert isinstance(ca.noise, EstimationNoise)

    def test_perfect_with_noise_raises(self):
        with pytest.raises(ValueError, match="noise config must be None"):
            CategoryAccess(AccessLevel.PERFECT, EstimationNoise())

    def test_none_with_noise_raises(self):
        with pytest.raises(ValueError, match="noise config must be None"):
            CategoryAccess(AccessLevel.NONE, EstimationNoise())

    def test_frozen(self):
        ca = CategoryAccess()
        with pytest.raises(AttributeError):
            ca.level = AccessLevel.NONE  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 4. InformationProfile Construction
# ═══════════════════════════════════════════════════════════════════════


class TestInformationProfile:
    def test_default_all_perfect(self):
        p = InformationProfile()
        assert p.counterparty_cash.level == AccessLevel.PERFECT
        assert p.dealer_quotes.level == AccessLevel.PERFECT
        assert p.obligation_graph.level == AccessLevel.PERFECT

    def test_custom_fields(self):
        p = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
            dealer_quotes=CategoryAccess(
                AccessLevel.NOISY, EstimationNoise(Decimal("0.05"))
            ),
        )
        assert p.counterparty_cash.level == AccessLevel.NONE
        assert p.dealer_quotes.level == AccessLevel.NOISY
        # Unchanged fields stay PERFECT
        assert p.counterparty_liabilities.level == AccessLevel.PERFECT

    def test_frozen(self):
        p = InformationProfile()
        with pytest.raises(AttributeError):
            p.counterparty_cash = CategoryAccess(AccessLevel.NONE)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 5. Presets
# ═══════════════════════════════════════════════════════════════════════


class TestPresets:
    def test_omniscient_all_perfect(self):
        for field_name in [
            "counterparty_cash",
            "counterparty_liabilities",
            "counterparty_default_history",
            "dealer_quotes",
            "obligation_graph",
            "aggregate_default_rate",
        ]:
            ca = getattr(OMNISCIENT, field_name)
            assert ca.level == AccessLevel.PERFECT, f"{field_name} should be PERFECT"

    def test_lender_realistic_cash_is_noisy(self):
        assert LENDER_REALISTIC.counterparty_cash.level == AccessLevel.NOISY
        assert isinstance(LENDER_REALISTIC.counterparty_cash.noise, EstimationNoise)

    def test_lender_realistic_no_market_access(self):
        assert LENDER_REALISTIC.dealer_quotes.level == AccessLevel.NONE
        assert LENDER_REALISTIC.vbt_anchors.level == AccessLevel.NONE

    def test_lender_realistic_no_network_access(self):
        assert LENDER_REALISTIC.obligation_graph.level == AccessLevel.NONE
        assert LENDER_REALISTIC.counterparty_connectivity.level == AccessLevel.NONE

    def test_lender_realistic_bilateral_perfect(self):
        assert LENDER_REALISTIC.bilateral_history.level == AccessLevel.PERFECT

    def test_trader_basic_no_balance_sheet(self):
        assert TRADER_BASIC.counterparty_cash.level == AccessLevel.NONE
        assert TRADER_BASIC.counterparty_liabilities.level == AccessLevel.NONE

    def test_trader_basic_market_prices_perfect(self):
        assert TRADER_BASIC.dealer_quotes.level == AccessLevel.PERFECT
        assert TRADER_BASIC.vbt_anchors.level == AccessLevel.PERFECT

    def test_trader_basic_history_is_noisy(self):
        assert TRADER_BASIC.counterparty_default_history.level == AccessLevel.NOISY
        assert isinstance(
            TRADER_BASIC.counterparty_default_history.noise, SampleNoise
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. InformationService Tests
# ═══════════════════════════════════════════════════════════════════════


def _build_system_for_info_tests(
    lender_cash: int = 10000,
    firm_cash: int = 500,
    firm_payable_amount: int = 1000,
    payable_due_day: int = 2,
):
    """Build a minimal system for InformationService tests."""
    from bilancio.domain.agents.central_bank import CentralBank
    from bilancio.domain.agents.firm import Firm
    from bilancio.domain.agents.non_bank_lender import NonBankLender
    from bilancio.domain.agents.bank import Bank
    from bilancio.domain.instruments.base import InstrumentKind
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.system import System

    system = System()
    cb = CentralBank(id="CB01", name="CB", kind="central_bank")
    bank = Bank(id="B01", name="Bank", kind="bank")
    lender = NonBankLender(id="NBL01", name="Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    if lender_cash > 0:
        system.mint_cash("NBL01", lender_cash)
    if firm_cash > 0:
        system.mint_cash("F01", firm_cash)

    if firm_payable_amount > 0:
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=firm_payable_amount,
            denom="X",
            asset_holder_id="F02",
            liability_issuer_id="F01",
            due_day=payable_due_day,
        )
        system.add_contract(payable)

    return system


class TestInformationServicePerfect:
    """Tests with OMNISCIENT profile (all perfect)."""

    def test_get_counterparty_cash_perfect(self):
        system = _build_system_for_info_tests(firm_cash=500)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        assert info.get_counterparty_cash("F01", 0) == 500

    def test_get_counterparty_obligations_perfect(self):
        system = _build_system_for_info_tests(firm_payable_amount=1000, payable_due_day=2)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        assert info.get_counterparty_obligations("F01", 0, 3) == 1000

    def test_get_default_probability_perfect(self):
        system = _build_system_for_info_tests()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        p = info.get_default_probability("F01", 0)
        assert p is not None
        assert Decimal("0") <= p <= Decimal("1")

    def test_get_system_default_rate_no_defaults(self):
        system = _build_system_for_info_tests()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        rate = info.get_system_default_rate(0)
        assert rate == Decimal("0")

    def test_get_loan_exposure(self):
        system = _build_system_for_info_tests(lender_cash=10000)
        # Create a loan
        system.nonbank_lend_cash("NBL01", "F01", 1000, Decimal("0.05"), 0)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        assert info.get_loan_exposure("NBL01") == 1000

    def test_get_borrower_exposure(self):
        system = _build_system_for_info_tests(lender_cash=10000)
        system.nonbank_lend_cash("NBL01", "F01", 2000, Decimal("0.05"), 0)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        assert info.get_borrower_exposure("NBL01", "F01") == 2000
        assert info.get_borrower_exposure("NBL01", "F02") == 0

    def test_get_system_liquidity(self):
        system = _build_system_for_info_tests(lender_cash=10000, firm_cash=500)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        assert info.get_system_liquidity(0) == 10500


class TestInformationServiceNone:
    """Tests with NONE access level."""

    def test_none_returns_none_for_cash(self):
        system = _build_system_for_info_tests(firm_cash=500)
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        assert info.get_counterparty_cash("F01", 0) is None

    def test_none_returns_none_for_obligations(self):
        system = _build_system_for_info_tests()
        profile = InformationProfile(
            counterparty_liabilities=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        assert info.get_counterparty_obligations("F01", 0, 3) is None

    def test_none_returns_none_for_default_prob(self):
        system = _build_system_for_info_tests()
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        assert info.get_default_probability("F01", 0) is None


class TestInformationServiceSelfAccess:
    """Self-queries always return perfect data regardless of profile."""

    def test_self_cash_always_perfect(self):
        system = _build_system_for_info_tests(lender_cash=10000)
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        # Self-query bypasses NONE
        assert info.get_counterparty_cash("NBL01", 0) == 10000

    def test_self_obligations_always_perfect(self):
        system = _build_system_for_info_tests()
        profile = InformationProfile(
            counterparty_liabilities=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="F01")
        # Self-query bypasses NONE — F01 has a payable
        result = info.get_counterparty_obligations("F01", 0, 3)
        assert result is not None
        assert result == 1000


class TestInformationServiceNoisy:
    """Tests with NOISY access level."""

    def test_noisy_cash_returns_approximate_value(self):
        """Noisy cash should be close to but different from true value."""
        system = _build_system_for_info_tests(firm_cash=1000)
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(
                AccessLevel.NOISY, EstimationNoise(Decimal("0.10"))
            ),
        )
        rng = random.Random(42)
        info = InformationService(system, profile, observer_id="NBL01", rng=rng)

        # Collect multiple samples
        values = []
        for seed in range(100):
            rng2 = random.Random(seed)
            info2 = InformationService(system, profile, observer_id="NBL01", rng=rng2)
            v = info2.get_counterparty_cash("F01", 0)
            assert v is not None
            values.append(v)

        # Mean should be close to 1000
        mean = sum(values) / len(values)
        assert 800 < mean < 1200, f"Mean {mean} too far from 1000"
        # Should not all be the same (noise adds variation)
        assert len(set(values)) > 1, "All values identical — noise not applied"

    def test_noisy_cash_always_non_negative(self):
        """Noisy values should never go below zero."""
        system = _build_system_for_info_tests(firm_cash=10)
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(
                AccessLevel.NOISY, EstimationNoise(Decimal("0.50"))
            ),
        )
        for seed in range(100):
            rng = random.Random(seed)
            info = InformationService(system, profile, observer_id="NBL01", rng=rng)
            v = info.get_counterparty_cash("F01", 0)
            assert v is not None
            assert v >= 0

    def test_noisy_default_prob_clamped_to_unit(self):
        """Noisy default probability should be clamped to [0, 1]."""
        system = _build_system_for_info_tests()
        profile = InformationProfile(
            counterparty_default_history=CategoryAccess(
                AccessLevel.NOISY,
                SampleNoise(Decimal("0.5")),
            ),
        )
        for seed in range(50):
            rng = random.Random(seed)
            info = InformationService(system, profile, observer_id="NBL01", rng=rng)
            p = info.get_default_probability("F01", 0)
            assert p is not None
            assert Decimal("0") <= p <= Decimal("1")


# ═══════════════════════════════════════════════════════════════════════
# 7. Lender Integration with InformationService
# ═══════════════════════════════════════════════════════════════════════


class TestLenderWithInformationService:
    """Tests for run_lending_phase with information profiles."""

    def test_omniscient_produces_same_as_default(self):
        """With OMNISCIENT profile, lending should produce the same results."""
        from bilancio.engines.lending import LendingConfig, run_lending_phase

        system = _build_system_for_info_tests(
            lender_cash=10000, firm_cash=200, firm_payable_amount=1000,
        )
        # Default (no profile)
        events_default = run_lending_phase(
            system, 0, LendingConfig(horizon=3, min_shortfall=1)
        )

        # Reset system state for fair comparison — need to rebuild
        system2 = _build_system_for_info_tests(
            lender_cash=10000, firm_cash=200, firm_payable_amount=1000,
        )
        events_omniscient = run_lending_phase(
            system2, 0,
            LendingConfig(
                horizon=3,
                min_shortfall=1,
                information_profile=OMNISCIENT,
            ),
        )

        # Both should produce at least one loan event
        assert len(events_default) >= 1
        assert len(events_omniscient) >= 1
        # Same borrower
        assert events_default[0]["borrower_id"] == events_omniscient[0]["borrower_id"]

    def test_none_visibility_skips_borrowers(self):
        """With NONE visibility for cash and liabilities, lender can't evaluate borrowers."""
        from bilancio.engines.lending import LendingConfig, run_lending_phase

        system = _build_system_for_info_tests(
            lender_cash=10000, firm_cash=200, firm_payable_amount=1000,
        )
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
            counterparty_liabilities=CategoryAccess(AccessLevel.NONE),
        )
        events = run_lending_phase(
            system, 0,
            LendingConfig(
                horizon=3,
                min_shortfall=1,
                information_profile=profile,
            ),
        )
        # With NONE obligations, lender skips (can't observe obligation => continues)
        assert events == []

    def test_noisy_profile_still_produces_loans(self):
        """With noisy but nonzero visibility, lender should still create loans."""
        from bilancio.engines.lending import LendingConfig, run_lending_phase

        system = _build_system_for_info_tests(
            lender_cash=10000, firm_cash=200, firm_payable_amount=1000,
        )
        events = run_lending_phase(
            system, 0,
            LendingConfig(
                horizon=3,
                min_shortfall=1,
                information_profile=LENDER_REALISTIC,
            ),
        )
        # With realistic profile, lender CAN observe (noisy) and should still lend
        # (though amounts may differ from omniscient due to noise)
        assert len(events) >= 1
        assert events[0]["kind"] == "NonBankLoanCreated"


# ═══════════════════════════════════════════════════════════════════════
# 8. Config Wiring Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConfigWiring:
    """Tests for LenderScenarioConfig → InformationProfile wiring."""

    def test_default_config_gives_no_profile(self):
        """Default LenderScenarioConfig (all perfect) returns None profile."""
        from bilancio.config.apply import _build_lender_info_profile
        from bilancio.config.models import LenderScenarioConfig

        cfg = LenderScenarioConfig(enabled=True)
        profile = _build_lender_info_profile(cfg)
        # Default is all-perfect + none for network/market → function returns None
        assert profile is None

    def test_noisy_cash_builds_profile(self):
        """Setting noisy cash visibility builds an InformationProfile."""
        from bilancio.config.apply import _build_lender_info_profile
        from bilancio.config.models import LenderScenarioConfig

        cfg = LenderScenarioConfig(
            enabled=True,
            info_cash_visibility="noisy",
            info_cash_noise=Decimal("0.20"),
        )
        profile = _build_lender_info_profile(cfg)
        assert profile is not None
        assert profile.counterparty_cash.level == AccessLevel.NOISY
        assert isinstance(profile.counterparty_cash.noise, EstimationNoise)
        assert profile.counterparty_cash.noise.error_fraction == Decimal("0.20")

    def test_none_cash_builds_profile(self):
        """Setting none cash visibility builds an InformationProfile with NONE access."""
        from bilancio.config.apply import _build_lender_info_profile
        from bilancio.config.models import LenderScenarioConfig

        cfg = LenderScenarioConfig(
            enabled=True,
            info_cash_visibility="none",
        )
        profile = _build_lender_info_profile(cfg)
        assert profile is not None
        assert profile.counterparty_cash.level == AccessLevel.NONE

    def test_noisy_history_builds_sample_noise(self):
        """Setting noisy history visibility builds SampleNoise."""
        from bilancio.config.apply import _build_lender_info_profile
        from bilancio.config.models import LenderScenarioConfig

        cfg = LenderScenarioConfig(
            enabled=True,
            info_history_visibility="noisy",
            info_history_sample_rate=Decimal("0.5"),
        )
        profile = _build_lender_info_profile(cfg)
        assert profile is not None
        assert profile.counterparty_default_history.level == AccessLevel.NOISY
        assert isinstance(profile.counterparty_default_history.noise, SampleNoise)
        assert profile.counterparty_default_history.noise.sample_rate == Decimal("0.5")

    def test_lender_scenario_config_new_fields_default(self):
        """New info_* fields have correct defaults."""
        from bilancio.config.models import LenderScenarioConfig

        cfg = LenderScenarioConfig()
        assert cfg.info_cash_visibility == "perfect"
        assert cfg.info_cash_noise == Decimal("0.10")
        assert cfg.info_liabilities_visibility == "perfect"
        assert cfg.info_history_visibility == "perfect"
        assert cfg.info_history_sample_rate == Decimal("0.7")
        assert cfg.info_network_visibility == "none"
        assert cfg.info_market_visibility == "none"
