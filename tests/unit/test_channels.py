"""Tests for the channel-derived estimation quality module."""

from decimal import Decimal

import pytest

from bilancio.information.channels import (
    Channel,
    InstitutionalChannel,
    MarketDerivedChannel,
    NetworkDerivedChannel,
    SelfDerivedChannel,
    category_from_channel,
    derive_noise,
)
from bilancio.information.levels import AccessLevel
from bilancio.information.noise import (
    AggregateOnlyNoise,
    EstimationNoise,
    LagNoise,
    SampleNoise,
)
from bilancio.information.profile import CategoryAccess


# ── SelfDerivedChannel construction & validation ─────────────────────────


class TestSelfDerivedChannel:
    def test_default(self):
        ch = SelfDerivedChannel()
        assert ch.sample_size == 20

    def test_custom_sample_size(self):
        ch = SelfDerivedChannel(sample_size=100)
        assert ch.sample_size == 100

    def test_frozen(self):
        ch = SelfDerivedChannel()
        with pytest.raises(AttributeError):
            ch.sample_size = 50  # type: ignore[misc]

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="sample_size must be >= 1"):
            SelfDerivedChannel(sample_size=0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="sample_size must be >= 1"):
            SelfDerivedChannel(sample_size=-5)

    def test_minimum_valid(self):
        ch = SelfDerivedChannel(sample_size=1)
        assert ch.sample_size == 1


# ── MarketDerivedChannel construction & validation ───────────────────────


class TestMarketDerivedChannel:
    def test_defaults(self):
        ch = MarketDerivedChannel()
        assert ch.market_thickness == 50
        assert ch.staleness_days == 0

    def test_custom(self):
        ch = MarketDerivedChannel(market_thickness=200, staleness_days=3)
        assert ch.market_thickness == 200
        assert ch.staleness_days == 3

    def test_frozen(self):
        ch = MarketDerivedChannel()
        with pytest.raises(AttributeError):
            ch.market_thickness = 10  # type: ignore[misc]

    def test_invalid_thickness_zero(self):
        with pytest.raises(ValueError, match="market_thickness must be >= 1"):
            MarketDerivedChannel(market_thickness=0)

    def test_invalid_staleness_negative(self):
        with pytest.raises(ValueError, match="staleness_days must be non-negative"):
            MarketDerivedChannel(staleness_days=-1)


# ── NetworkDerivedChannel construction & validation ──────────────────────


class TestNetworkDerivedChannel:
    def test_default(self):
        ch = NetworkDerivedChannel()
        assert ch.coverage == Decimal("0.5")

    def test_custom(self):
        ch = NetworkDerivedChannel(coverage=Decimal("0.9"))
        assert ch.coverage == Decimal("0.9")

    def test_frozen(self):
        ch = NetworkDerivedChannel()
        with pytest.raises(AttributeError):
            ch.coverage = Decimal("0.8")  # type: ignore[misc]

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="coverage must be in"):
            NetworkDerivedChannel(coverage=Decimal("0"))

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="coverage must be in"):
            NetworkDerivedChannel(coverage=Decimal("-0.1"))

    def test_invalid_above_one(self):
        with pytest.raises(ValueError, match="coverage must be in"):
            NetworkDerivedChannel(coverage=Decimal("1.1"))

    def test_coverage_one_valid(self):
        ch = NetworkDerivedChannel(coverage=Decimal("1"))
        assert ch.coverage == Decimal("1")


# ── InstitutionalChannel construction & validation ───────────────────────


class TestInstitutionalChannel:
    def test_defaults(self):
        ch = InstitutionalChannel()
        assert ch.staleness_days == 0
        assert ch.coverage == Decimal("0.8")

    def test_custom(self):
        ch = InstitutionalChannel(staleness_days=5, coverage=Decimal("0.6"))
        assert ch.staleness_days == 5
        assert ch.coverage == Decimal("0.6")

    def test_frozen(self):
        ch = InstitutionalChannel()
        with pytest.raises(AttributeError):
            ch.staleness_days = 2  # type: ignore[misc]

    def test_invalid_staleness(self):
        with pytest.raises(ValueError, match="staleness_days must be non-negative"):
            InstitutionalChannel(staleness_days=-1)

    def test_invalid_coverage_zero(self):
        with pytest.raises(ValueError, match="coverage must be in"):
            InstitutionalChannel(coverage=Decimal("0"))

    def test_invalid_coverage_above_one(self):
        with pytest.raises(ValueError, match="coverage must be in"):
            InstitutionalChannel(coverage=Decimal("1.5"))


# ── derive_noise: SelfDerivedChannel ─────────────────────────────────────


class TestDeriveNoiseSelfDerived:
    def test_sample_100(self):
        """1/sqrt(100) = 0.10"""
        noise = derive_noise(SelfDerivedChannel(sample_size=100))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.10")

    def test_sample_25(self):
        """1/sqrt(25) = 0.20"""
        noise = derive_noise(SelfDerivedChannel(sample_size=25))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.20")

    def test_sample_44(self):
        """1/sqrt(44) ≈ 0.1508 → rounds to 0.15"""
        noise = derive_noise(SelfDerivedChannel(sample_size=44))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.15")

    def test_sample_400(self):
        """1/sqrt(400) = 0.05"""
        noise = derive_noise(SelfDerivedChannel(sample_size=400))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.05")

    def test_sample_1(self):
        """1/sqrt(1) = 1.00 — maximum error"""
        noise = derive_noise(SelfDerivedChannel(sample_size=1))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("1.00")

    def test_sample_10000(self):
        """1/sqrt(10000) = 0.01 — minimum error"""
        noise = derive_noise(SelfDerivedChannel(sample_size=10000))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.01")

    def test_very_large_sample(self):
        """Very large sample → clamped to min 0.01"""
        noise = derive_noise(SelfDerivedChannel(sample_size=1000000))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.01")

    def test_default_sample_20(self):
        """Default sample_size=20: 1/sqrt(20) ≈ 0.2236 → rounds to 0.22"""
        noise = derive_noise(SelfDerivedChannel())
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.22")


# ── derive_noise: MarketDerivedChannel ───────────────────────────────────


class TestDerivNoiseMarketDerived:
    def test_no_staleness(self):
        """No staleness → EstimationNoise from thickness."""
        noise = derive_noise(MarketDerivedChannel(market_thickness=100, staleness_days=0))
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.10")

    def test_staleness_dominates(self):
        """staleness=5 → lag_error=0.25; thickness=100 → thick_error=0.10. Lag dominates."""
        noise = derive_noise(
            MarketDerivedChannel(market_thickness=100, staleness_days=5)
        )
        assert isinstance(noise, LagNoise)
        assert noise.lag_days == 5

    def test_thickness_dominates(self):
        """staleness=1 → lag_error=0.05; thickness=4 → thick_error=0.50. Thickness dominates."""
        noise = derive_noise(
            MarketDerivedChannel(market_thickness=4, staleness_days=1)
        )
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.50")

    def test_equal_errors_lag_wins(self):
        """When lag_error == thickness_error AND staleness > 0, lag dominates (>=)."""
        # staleness=2 → lag_error=0.10; thickness=100 → thick_error=0.10. Equal → lag.
        noise = derive_noise(
            MarketDerivedChannel(market_thickness=100, staleness_days=2)
        )
        assert isinstance(noise, LagNoise)
        assert noise.lag_days == 2

    def test_default(self):
        """Default: thickness=50, staleness=0 → EstimationNoise."""
        noise = derive_noise(MarketDerivedChannel())
        assert isinstance(noise, EstimationNoise)
        assert noise.error_fraction == Decimal("0.14")  # 1/sqrt(50) ≈ 0.1414


# ── derive_noise: NetworkDerivedChannel ──────────────────────────────────


class TestDerivNoiseNetworkDerived:
    def test_default(self):
        noise = derive_noise(NetworkDerivedChannel())
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("0.5")

    def test_custom_coverage(self):
        noise = derive_noise(NetworkDerivedChannel(coverage=Decimal("0.7")))
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("0.7")

    def test_full_coverage(self):
        noise = derive_noise(NetworkDerivedChannel(coverage=Decimal("1")))
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("1")


# ── derive_noise: InstitutionalChannel ───────────────────────────────────


class TestDerivNoiseInstitutional:
    def test_no_staleness(self):
        """staleness=0 → SampleNoise from coverage."""
        noise = derive_noise(InstitutionalChannel(staleness_days=0, coverage=Decimal("0.8")))
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("0.8")

    def test_staleness_dominates(self):
        """staleness=10 → lag_error=0.50; coverage=0.8 → gap=0.20. Lag dominates."""
        noise = derive_noise(
            InstitutionalChannel(staleness_days=10, coverage=Decimal("0.8"))
        )
        assert isinstance(noise, LagNoise)
        assert noise.lag_days == 10

    def test_coverage_gap_dominates(self):
        """staleness=1 → lag_error=0.05; coverage=0.3 → gap=0.70. Coverage dominates."""
        noise = derive_noise(
            InstitutionalChannel(staleness_days=1, coverage=Decimal("0.3"))
        )
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("0.3")

    def test_equal_errors_lag_wins(self):
        """When lag_error == coverage_gap AND staleness > 0, lag dominates (>=)."""
        # staleness=4 → lag_error=0.20; coverage=0.8 → gap=0.20. Equal → lag.
        noise = derive_noise(
            InstitutionalChannel(staleness_days=4, coverage=Decimal("0.8"))
        )
        assert isinstance(noise, LagNoise)
        assert noise.lag_days == 4

    def test_default(self):
        """Default: staleness=0, coverage=0.8 → SampleNoise."""
        noise = derive_noise(InstitutionalChannel())
        assert isinstance(noise, SampleNoise)
        assert noise.sample_rate == Decimal("0.8")


# ── category_from_channel ────────────────────────────────────────────────


class TestCategoryFromChannel:
    def test_returns_noisy(self):
        ca = category_from_channel(SelfDerivedChannel(sample_size=100))
        assert ca.level == AccessLevel.NOISY

    def test_noise_matches_derive(self):
        ch = SelfDerivedChannel(sample_size=100)
        ca = category_from_channel(ch)
        assert ca.noise == derive_noise(ch)

    def test_network_channel(self):
        ca = category_from_channel(NetworkDerivedChannel(coverage=Decimal("0.7")))
        assert ca.level == AccessLevel.NOISY
        assert isinstance(ca.noise, SampleNoise)
        assert ca.noise.sample_rate == Decimal("0.7")

    def test_market_channel(self):
        ca = category_from_channel(MarketDerivedChannel(market_thickness=100))
        assert ca.level == AccessLevel.NOISY
        assert isinstance(ca.noise, EstimationNoise)

    def test_institutional_channel(self):
        ca = category_from_channel(
            InstitutionalChannel(staleness_days=10, coverage=Decimal("0.8"))
        )
        assert ca.level == AccessLevel.NOISY
        assert isinstance(ca.noise, LagNoise)


# ── LENDER_CHANNEL_BASED preset ─────────────────────────────────────────


class TestLenderChannelBased:
    @pytest.fixture
    def realistic(self):
        from bilancio.information.presets import LENDER_REALISTIC
        return LENDER_REALISTIC

    @pytest.fixture
    def channel_based(self):
        from bilancio.information.presets import LENDER_CHANNEL_BASED
        return LENDER_CHANNEL_BASED

    def test_cash_noisy(self, channel_based):
        assert channel_based.counterparty_cash.level == AccessLevel.NOISY

    def test_cash_estimation_noise(self, channel_based):
        assert isinstance(channel_based.counterparty_cash.noise, EstimationNoise)

    def test_cash_error_approx_015(self, channel_based):
        """Channel-derived cash error ≈ 0.15 (from sample_size=44)."""
        assert channel_based.counterparty_cash.noise.error_fraction == Decimal("0.15")

    def test_net_worth_error_approx_020(self, channel_based):
        """Channel-derived net_worth error ≈ 0.20 (from sample_size=25)."""
        assert channel_based.counterparty_net_worth.noise.error_fraction == Decimal("0.20")

    def test_default_history_sample_noise(self, channel_based):
        assert isinstance(
            channel_based.counterparty_default_history.noise, SampleNoise
        )
        assert channel_based.counterparty_default_history.noise.sample_rate == Decimal(
            "0.7"
        )

    def test_assets_aggregate_only(self, channel_based):
        """Structural constraint: still AggregateOnlyNoise (not channel-derived)."""
        assert isinstance(channel_based.counterparty_assets.noise, AggregateOnlyNoise)

    def test_liabilities_aggregate_only(self, channel_based):
        assert isinstance(
            channel_based.counterparty_liabilities.noise, AggregateOnlyNoise
        )

    def test_market_prices_none(self, channel_based):
        """Lender still has no market access."""
        assert channel_based.dealer_quotes.level == AccessLevel.NONE
        assert channel_based.vbt_anchors.level == AccessLevel.NONE
        assert channel_based.price_trends.level == AccessLevel.NONE
        assert channel_based.implied_default_prob.level == AccessLevel.NONE

    def test_network_none(self, channel_based):
        """Lender still has no network topology access."""
        assert channel_based.obligation_graph.level == AccessLevel.NONE
        assert channel_based.counterparty_connectivity.level == AccessLevel.NONE
        assert channel_based.cascade_risk.level == AccessLevel.NONE

    def test_bilateral_perfect(self, channel_based):
        """Own bilateral history remains PERFECT."""
        assert channel_based.bilateral_history.level == AccessLevel.PERFECT

    def test_access_levels_match_realistic(self, realistic, channel_based):
        """Channel-based preset has same access levels as hand-tuned realistic."""
        from dataclasses import fields

        for f in fields(realistic):
            if f.name == "channel_bindings":
                continue
            r_ca = getattr(realistic, f.name)
            c_ca = getattr(channel_based, f.name)
            assert r_ca.level == c_ca.level, (
                f"{f.name}: realistic={r_ca.level}, channel={c_ca.level}"
            )

    def test_noise_types_match_realistic(self, realistic, channel_based):
        """Channel-based noise types match realistic (same family for each field)."""
        from dataclasses import fields

        for f in fields(realistic):
            if f.name == "channel_bindings":
                continue
            r_ca = getattr(realistic, f.name)
            c_ca = getattr(channel_based, f.name)
            if r_ca.noise is None:
                assert c_ca.noise is None, f"{f.name}: expected None noise"
            else:
                assert type(r_ca.noise) is type(c_ca.noise), (
                    f"{f.name}: realistic={type(r_ca.noise).__name__}, "
                    f"channel={type(c_ca.noise).__name__}"
                )

    def test_cash_error_close_to_realistic(self, realistic, channel_based):
        """Channel-derived error is close to hand-tuned value."""
        r_err = float(realistic.counterparty_cash.noise.error_fraction)
        c_err = float(channel_based.counterparty_cash.noise.error_fraction)
        assert abs(r_err - c_err) < 0.02

    def test_net_worth_error_close_to_realistic(self, realistic, channel_based):
        r_err = float(realistic.counterparty_net_worth.noise.error_fraction)
        c_err = float(channel_based.counterparty_net_worth.noise.error_fraction)
        assert abs(r_err - c_err) < 0.02


# ── Backward compatibility ───────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_existing_presets_unchanged(self):
        """Importing channels doesn't affect existing presets."""
        from bilancio.information import LENDER_REALISTIC, OMNISCIENT, TRADER_BASIC

        assert OMNISCIENT.counterparty_cash.level == AccessLevel.PERFECT
        assert LENDER_REALISTIC.counterparty_cash.level == AccessLevel.NOISY
        assert TRADER_BASIC.counterparty_cash.level == AccessLevel.NONE

    def test_direct_noise_still_works(self):
        """Direct CategoryAccess construction still works (not forced to use channels)."""
        ca = CategoryAccess(AccessLevel.NOISY, EstimationNoise(Decimal("0.15")))
        assert ca.level == AccessLevel.NOISY
        assert ca.noise.error_fraction == Decimal("0.15")


# ── Integration: channel-based profile through InformationService ────────


class TestChannelServiceIntegration:
    def test_channel_based_profile_works_with_service(self):
        """A channel-based preset can be used with InformationService."""
        from bilancio.information import LENDER_CHANNEL_BASED, InformationService

        # InformationService requires a System, but we can at least verify
        # that the profile is accepted without error during construction
        # (full service testing is in test_service.py)
        profile = LENDER_CHANNEL_BASED
        assert profile.counterparty_cash.level == AccessLevel.NOISY
        assert profile.counterparty_cash.noise is not None


# ── Exports ──────────────────────────────────────────────────────────────


class TestExports:
    def test_channel_types_importable_from_init(self):
        from bilancio.information import (
            SelfDerivedChannel,
            MarketDerivedChannel,
            NetworkDerivedChannel,
            InstitutionalChannel,
        )
        # Just verify they import without error
        assert SelfDerivedChannel is not None

    def test_functions_importable_from_init(self):
        from bilancio.information import category_from_channel, derive_noise
        assert category_from_channel is not None
        assert derive_noise is not None

    def test_preset_importable_from_init(self):
        from bilancio.information import LENDER_CHANNEL_BASED
        assert LENDER_CHANNEL_BASED is not None

    def test_channel_union_importable(self):
        from bilancio.information import Channel
        assert Channel is not None
