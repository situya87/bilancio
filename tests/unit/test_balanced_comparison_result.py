"""Unit tests for BalancedComparisonResult — F arm fields and combined_effect."""

from decimal import Decimal

from bilancio.experiments.balanced_comparison import (
    BalancedComparisonConfig,
    BalancedComparisonResult,
)

# ---------------------------------------------------------------------------
# Minimal kwargs shared across tests
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Return minimal kwargs for BalancedComparisonResult."""
    defaults = {
        "kappa": Decimal("1"),
        "concentration": Decimal("1"),
        "mu": Decimal("0"),
        "monotonicity": Decimal("0"),
        "seed": 42,
        "face_value": Decimal("20"),
        "outside_mid_ratio": Decimal("1"),
        "big_entity_share": Decimal("0.25"),
        "delta_passive": Decimal("0.4"),
        "phi_passive": Decimal("0.6"),
        "passive_run_id": "passive-1",
        "passive_status": "completed",
        "delta_active": Decimal("0.3"),
        "phi_active": Decimal("0.7"),
        "active_run_id": "active-1",
        "active_status": "completed",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests: combined_effect property
# ---------------------------------------------------------------------------


class TestCombinedEffect:
    def test_combined_effect_positive(self):
        """Positive combined_effect when dealer+lender reduces defaults."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_dealer_lender=Decimal("0.2"),
            )
        )
        assert r.combined_effect == Decimal("0.2")  # 0.4 - 0.2

    def test_combined_effect_zero(self):
        """Zero combined_effect when no improvement."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_dealer_lender=Decimal("0.4"),
            )
        )
        assert r.combined_effect == Decimal("0")

    def test_combined_effect_negative(self):
        """Negative combined_effect when combination worsens defaults."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_dealer_lender=Decimal("0.5"),
            )
        )
        assert r.combined_effect == Decimal("-0.1")

    def test_combined_effect_none_when_passive_missing(self):
        """combined_effect is None when delta_passive is None."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_passive=None,
                delta_dealer_lender=Decimal("0.2"),
            )
        )
        assert r.combined_effect is None

    def test_combined_effect_none_when_dealer_lender_missing(self):
        """combined_effect is None when delta_dealer_lender is None."""
        r = BalancedComparisonResult(**_base_kwargs())
        assert r.delta_dealer_lender is None
        assert r.combined_effect is None


# ---------------------------------------------------------------------------
# Tests: F arm default field values
# ---------------------------------------------------------------------------


class TestDealerLenderDefaults:
    def test_default_values(self):
        """F arm fields default to empty/zero/None."""
        r = BalancedComparisonResult(**_base_kwargs())
        assert r.delta_dealer_lender is None
        assert r.phi_dealer_lender is None
        assert r.dealer_lender_run_id == ""
        assert r.dealer_lender_status == ""
        assert r.n_defaults_dealer_lender == 0
        assert r.cascade_fraction_dealer_lender is None
        assert r.dealer_lender_modal_call_id is None

    def test_f_arm_fields_set(self):
        """F arm fields can be set via constructor."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_dealer_lender=Decimal("0.15"),
                phi_dealer_lender=Decimal("0.85"),
                dealer_lender_run_id="dl-1",
                dealer_lender_status="completed",
                n_defaults_dealer_lender=3,
                cascade_fraction_dealer_lender=Decimal("0.1"),
                dealer_lender_modal_call_id="fc-123",
            )
        )
        assert r.delta_dealer_lender == Decimal("0.15")
        assert r.phi_dealer_lender == Decimal("0.85")
        assert r.dealer_lender_run_id == "dl-1"
        assert r.dealer_lender_status == "completed"
        assert r.n_defaults_dealer_lender == 3
        assert r.cascade_fraction_dealer_lender == Decimal("0.1")
        assert r.dealer_lender_modal_call_id == "fc-123"


# ---------------------------------------------------------------------------
# Tests: Config enable_dealer_lender field
# ---------------------------------------------------------------------------


class TestConfigDealerLender:
    def test_default_disabled(self):
        """enable_dealer_lender defaults to False."""
        cfg = BalancedComparisonConfig()
        assert cfg.enable_dealer_lender is False

    def test_enable(self):
        """enable_dealer_lender can be set to True."""
        cfg = BalancedComparisonConfig(enable_dealer_lender=True)
        assert cfg.enable_dealer_lender is True
