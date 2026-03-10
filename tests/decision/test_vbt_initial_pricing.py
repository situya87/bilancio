"""Tests for VBT parameter-aware initial pricing (Plan 055).

Verifies:
- Mu-driven term structure tilt on M
- Kappa-driven spread scaling
- Backward compatibility at mu=0.5 and kappa>=1
"""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import VBTProfile


class TestMuTiltFactors:
    """Test mu-driven per-bucket M tilt computation."""

    BUCKET_TAU_MIDPOINTS = {"short": Decimal("2"), "mid": Decimal("6"), "long": Decimal("12")}

    def _compute_tilt_factors(
        self, mu: Decimal, strength: Decimal = Decimal("0.15")
    ) -> dict[str, Decimal]:
        """Replicate the tilt factor computation from dealer_wiring."""
        tau_mid = sum(self.BUCKET_TAU_MIDPOINTS.values()) / Decimal(
            str(len(self.BUCKET_TAU_MIDPOINTS))
        )
        mu_direction = Decimal("0.5") - mu
        factors = {}
        for bucket_id, tau_avg in self.BUCKET_TAU_MIDPOINTS.items():
            tau_position = (tau_mid - tau_avg) / tau_mid
            risk_tilt = mu_direction * tau_position
            factors[bucket_id] = Decimal("1") - strength * risk_tilt
        return factors

    def test_mu_0_front_loaded_short_M_lower(self):
        """Front-loaded (mu=0): short-term is riskier → M_short < M_base."""
        factors = self._compute_tilt_factors(Decimal("0"))
        assert factors["short"] < Decimal("1"), "Short bucket should have M < base for mu=0"

    def test_mu_0_front_loaded_long_M_higher(self):
        """Front-loaded (mu=0): long-term is safer → M_long > M_base."""
        factors = self._compute_tilt_factors(Decimal("0"))
        assert factors["long"] > Decimal("1"), "Long bucket should have M > base for mu=0"

    def test_mu_0_term_structure(self):
        """mu=0 gives M_short < M_mid < M_long."""
        factors = self._compute_tilt_factors(Decimal("0"))
        assert factors["short"] < factors["mid"] < factors["long"]

    def test_mu_1_back_loaded_short_M_higher(self):
        """Back-loaded (mu=1): short-term is safer → M_short > M_base."""
        factors = self._compute_tilt_factors(Decimal("1"))
        assert factors["short"] > Decimal("1"), "Short bucket should have M > base for mu=1"

    def test_mu_1_back_loaded_long_M_lower(self):
        """Back-loaded (mu=1): long-term is riskier → M_long < M_base."""
        factors = self._compute_tilt_factors(Decimal("1"))
        assert factors["long"] < Decimal("1"), "Long bucket should have M < base for mu=1"

    def test_mu_1_term_structure(self):
        """mu=1 gives M_short > M_mid > M_long."""
        factors = self._compute_tilt_factors(Decimal("1"))
        assert factors["short"] > factors["mid"] > factors["long"]

    def test_mu_05_neutral_no_tilt(self):
        """mu=0.5 gives uniform M (all factors = 1)."""
        factors = self._compute_tilt_factors(Decimal("0.5"))
        for bucket_id, factor in factors.items():
            assert factor == Decimal("1"), f"Factor for {bucket_id} should be exactly 1 at mu=0.5"

    def test_mu_symmetry(self):
        """mu=0 and mu=1 should produce symmetric tilts."""
        factors_0 = self._compute_tilt_factors(Decimal("0"))
        factors_1 = self._compute_tilt_factors(Decimal("1"))
        for bucket_id in self.BUCKET_TAU_MIDPOINTS:
            # factor(mu=0) + factor(mu=1) should equal 2 (symmetric around 1)
            total = factors_0[bucket_id] + factors_1[bucket_id]
            assert total == Decimal("2"), f"Symmetry broken for {bucket_id}: {total}"

    def test_strength_zero_no_tilt(self):
        """strength=0 means no tilt regardless of mu."""
        factors = self._compute_tilt_factors(Decimal("0"), strength=Decimal("0"))
        for bucket_id, factor in factors.items():
            assert factor == Decimal("1"), f"Factor for {bucket_id} should be 1 with strength=0"


class TestKappaSpreadScaling:
    """Test kappa-driven spread scaling."""

    def _compute_spread_factor(
        self, kappa: Decimal, strength: Decimal = Decimal("0.5")
    ) -> Decimal:
        """Replicate the spread scaling formula from dealer_wiring."""
        kappa_stress = max(Decimal("0"), Decimal("1") - kappa) / (Decimal("1") + kappa)
        return Decimal("1") + strength * kappa_stress

    def test_kappa_03_widens_spreads(self):
        """kappa=0.3 should widen spreads (factor > 1)."""
        factor = self._compute_spread_factor(Decimal("0.3"))
        assert factor > Decimal("1")
        # Expected: stress = 0.7/1.3 ≈ 0.538, factor ≈ 1.269
        assert abs(factor - Decimal("1.269")) < Decimal("0.01")

    def test_kappa_05_widens_spreads(self):
        """kappa=0.5 should widen spreads moderately."""
        factor = self._compute_spread_factor(Decimal("0.5"))
        assert factor > Decimal("1")
        # Expected: stress = 0.5/1.5 ≈ 0.333, factor ≈ 1.167
        assert abs(factor - Decimal("1.167")) < Decimal("0.01")

    def test_kappa_1_no_change(self):
        """kappa=1 gives no change (stress=0)."""
        factor = self._compute_spread_factor(Decimal("1"))
        assert factor == Decimal("1")

    def test_kappa_2_no_change(self):
        """kappa>=1 gives no change."""
        factor = self._compute_spread_factor(Decimal("2"))
        assert factor == Decimal("1")

    def test_kappa_very_low_stress(self):
        """Very low kappa should give significant widening but bounded."""
        factor = self._compute_spread_factor(Decimal("0.1"))
        # stress = 0.9/1.1 ≈ 0.818, factor ≈ 1.409
        assert factor > Decimal("1.3")
        assert factor < Decimal("2")  # Not unreasonably large

    def test_strength_zero_no_scaling(self):
        """strength=0 means no scaling regardless of kappa."""
        factor = self._compute_spread_factor(Decimal("0.3"), strength=Decimal("0"))
        assert factor == Decimal("1")


class TestVBTProfileFields:
    """Test that new VBTProfile fields exist with correct defaults."""

    def test_mu_tilt_strength_default(self):
        profile = VBTProfile()
        assert profile.mu_tilt_strength == Decimal("0.15")

    def test_kappa_spread_strength_default(self):
        profile = VBTProfile()
        assert profile.kappa_spread_strength == Decimal("0.5")

    def test_custom_values(self):
        profile = VBTProfile(
            mu_tilt_strength=Decimal("0.3"),
            kappa_spread_strength=Decimal("1.0"),
        )
        assert profile.mu_tilt_strength == Decimal("0.3")
        assert profile.kappa_spread_strength == Decimal("1.0")


class TestIntegrationRoundTrip:
    """Integration test: round-trip through _initialize_balanced_market_makers."""

    def test_mu_0_creates_term_structure(self):
        """With mu=0, short bucket should have lower M than long bucket."""
        from unittest.mock import MagicMock

        from bilancio.dealer.models import BucketConfig, DealerState, VBTState
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.decision.profiles import VBTProfile
        from bilancio.engines.dealer_integration import DealerSubsystem

        # Create a minimal subsystem with mu=0
        subsystem = DealerSubsystem(
            bucket_configs=[
                BucketConfig(name="short", tau_min=1, tau_max=3),
                BucketConfig(name="mid", tau_min=4, tau_max=9),
                BucketConfig(name="long", tau_min=10, tau_max=None),
            ],
            enabled=True,
            face_value=Decimal("20"),
            outside_mid_ratio=Decimal("0.90"),
            kappa=Decimal("0.5"),
            mu=Decimal("0"),
            vbt_profile=VBTProfile(),
        )

        # Set up minimal pricing model
        from bilancio.decision.valuers import CreditAdjustedVBTPricing

        subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"),
            spread_sensitivity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"),
        )

        # Mock system for _get_agent_cash
        system = MagicMock()
        system.state.agents = {}

        def mock_get_cash(system, agent_id):
            return Decimal("1000")

        # Import and call the initialization function
        import bilancio.engines.dealer_wiring as wiring

        original_get_cash = wiring._get_agent_cash if hasattr(wiring, '_get_agent_cash') else None

        from bilancio.engines.dealer_wiring import _initialize_balanced_market_makers

        # Provide empty ticket dicts
        vbt_tickets = {"short": [], "mid": [], "long": []}
        dealer_tickets = {"short": [], "mid": [], "long": []}
        dealer_config = DealerRingConfig(
            ticket_size=Decimal("20"),
            buckets=[
                BucketConfig(name="short", tau_min=1, tau_max=3),
                BucketConfig(name="mid", tau_min=4, tau_max=9),
                BucketConfig(name="long", tau_min=10, tau_max=None),
            ],
        )

        # Monkey-patch _get_agent_cash for the test
        import bilancio.engines.dealer_integration as di

        original = di._get_agent_cash

        try:
            di._get_agent_cash = mock_get_cash
            _initialize_balanced_market_makers(
                subsystem,
                system,
                dealer_config,
                vbt_tickets,
                dealer_tickets,
                shared_prior=Decimal("0.15"),
            )
        finally:
            di._get_agent_cash = original

        # Check that M differs across buckets with mu=0
        M_short = subsystem.vbts["short"].M
        M_mid = subsystem.vbts["mid"].M
        M_long = subsystem.vbts["long"].M

        assert M_short < M_mid, f"M_short ({M_short}) should be < M_mid ({M_mid})"
        assert M_mid < M_long, f"M_mid ({M_mid}) should be < M_long ({M_long})"

    def test_mu_05_uniform_M(self):
        """With mu=0.5, all buckets should have the same M."""
        from unittest.mock import MagicMock

        from bilancio.dealer.models import BucketConfig
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.decision.profiles import VBTProfile
        from bilancio.engines.dealer_integration import DealerSubsystem

        subsystem = DealerSubsystem(
            bucket_configs=[
                BucketConfig(name="short", tau_min=1, tau_max=3),
                BucketConfig(name="mid", tau_min=4, tau_max=9),
                BucketConfig(name="long", tau_min=10, tau_max=None),
            ],
            enabled=True,
            face_value=Decimal("20"),
            outside_mid_ratio=Decimal("0.90"),
            kappa=Decimal("0.5"),
            mu=Decimal("0.5"),
            vbt_profile=VBTProfile(),
        )

        from bilancio.decision.valuers import CreditAdjustedVBTPricing

        subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
            mid_sensitivity=Decimal("1"),
            spread_sensitivity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"),
        )

        system = MagicMock()
        system.state.agents = {}

        def mock_get_cash(system, agent_id):
            return Decimal("1000")

        from bilancio.engines.dealer_wiring import _initialize_balanced_market_makers

        vbt_tickets = {"short": [], "mid": [], "long": []}
        dealer_tickets = {"short": [], "mid": [], "long": []}
        dealer_config = DealerRingConfig(
            ticket_size=Decimal("20"),
            buckets=[
                BucketConfig(name="short", tau_min=1, tau_max=3),
                BucketConfig(name="mid", tau_min=4, tau_max=9),
                BucketConfig(name="long", tau_min=10, tau_max=None),
            ],
        )

        import bilancio.engines.dealer_integration as di

        original = di._get_agent_cash
        try:
            di._get_agent_cash = mock_get_cash
            _initialize_balanced_market_makers(
                subsystem,
                system,
                dealer_config,
                vbt_tickets,
                dealer_tickets,
                shared_prior=Decimal("0.15"),
            )
        finally:
            di._get_agent_cash = original

        M_short = subsystem.vbts["short"].M
        M_mid = subsystem.vbts["mid"].M
        M_long = subsystem.vbts["long"].M

        assert M_short == M_mid == M_long, (
            f"M should be uniform at mu=0.5: short={M_short}, mid={M_mid}, long={M_long}"
        )

    def test_kappa_03_wider_spreads(self):
        """With kappa=0.3, base spreads should be wider than with kappa=1."""
        from unittest.mock import MagicMock

        from bilancio.dealer.models import BucketConfig
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.decision.profiles import VBTProfile
        from bilancio.engines.dealer_integration import DealerSubsystem

        def run_init(kappa_val):
            subsystem = DealerSubsystem(
                bucket_configs=[
                    BucketConfig(name="short", tau_min=1, tau_max=3),
                    BucketConfig(name="mid", tau_min=4, tau_max=9),
                    BucketConfig(name="long", tau_min=10, tau_max=None),
                ],
                enabled=True,
                face_value=Decimal("20"),
                outside_mid_ratio=Decimal("0.90"),
                kappa=kappa_val,
                mu=Decimal("0.5"),
                vbt_profile=VBTProfile(),
            )

            from bilancio.decision.valuers import CreditAdjustedVBTPricing

            subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
                mid_sensitivity=Decimal("1"),
                spread_sensitivity=Decimal("0"),
                outside_mid_ratio=Decimal("0.90"),
            )

            system = MagicMock()
            system.state.agents = {}

            def mock_get_cash(system, agent_id):
                return Decimal("1000")

            from bilancio.engines.dealer_wiring import _initialize_balanced_market_makers

            vbt_tickets = {"short": [], "mid": [], "long": []}
            dealer_tickets = {"short": [], "mid": [], "long": []}
            dealer_config = DealerRingConfig(
                ticket_size=Decimal("20"),
                buckets=[
                    BucketConfig(name="short", tau_min=1, tau_max=3),
                    BucketConfig(name="mid", tau_min=4, tau_max=9),
                    BucketConfig(name="long", tau_min=10, tau_max=None),
                ],
            )

            import bilancio.engines.dealer_integration as di

            original = di._get_agent_cash
            try:
                di._get_agent_cash = mock_get_cash
                _initialize_balanced_market_makers(
                    subsystem,
                    system,
                    dealer_config,
                    vbt_tickets,
                    dealer_tickets,
                    shared_prior=Decimal("0.15"),
                )
            finally:
                di._get_agent_cash = original

            return subsystem.base_spread_by_bucket

        spreads_stressed = run_init(Decimal("0.3"))
        spreads_normal = run_init(Decimal("1.0"))

        for bucket_id in ("short", "mid", "long"):
            assert spreads_stressed[bucket_id] > spreads_normal[bucket_id], (
                f"Spread for {bucket_id} should be wider at kappa=0.3 than kappa=1.0: "
                f"{spreads_stressed[bucket_id]} vs {spreads_normal[bucket_id]}"
            )
