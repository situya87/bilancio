"""Tests for adaptive preset system (Plan 050)."""
import pytest
from decimal import Decimal
from bilancio.decision.adaptive import AdaptPreset, build_adaptive_overrides


class TestAdaptPreset:
    def test_enum_values(self):
        assert AdaptPreset.STATIC.value == "static"
        assert AdaptPreset.CALIBRATED.value == "calibrated"
        assert AdaptPreset.RESPONSIVE.value == "responsive"
        assert AdaptPreset.FULL.value == "full"

    def test_string_construction(self):
        assert AdaptPreset("static") == AdaptPreset.STATIC
        assert AdaptPreset("full") == AdaptPreset.FULL


class TestBuildAdaptiveOverrides:
    def test_static_returns_empty(self):
        result = build_adaptive_overrides("static", Decimal("1"), Decimal("0.5"), Decimal("1"), 10, 100)
        for key in ("trader", "risk_params", "vbt", "bank", "lender", "cb"):
            assert result[key] == {}, f"{key} should be empty for static"

    def test_calibrated_sets_pre_run_flags(self):
        result = build_adaptive_overrides("calibrated", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100)
        assert result["trader"]["adaptive_planning_horizon"] is True
        assert result["vbt"]["adaptive_term_structure"] is True
        assert result["bank"]["adaptive_corridor"] is True
        assert result["cb"]["adaptive_betas"] is True
        # Should NOT set within-run flags
        assert "adaptive_risk_aversion" not in result["trader"] or result["trader"].get("adaptive_risk_aversion") is not True
        assert "adaptive_convex_spreads" not in result["vbt"] or result["vbt"].get("adaptive_convex_spreads") is not True

    def test_responsive_sets_in_run_flags(self):
        result = build_adaptive_overrides("responsive", Decimal("1"), Decimal("0.5"), Decimal("1"), 10, 100)
        assert result["trader"]["adaptive_risk_aversion"] is True
        assert result["trader"]["adaptive_reserves"] is True
        assert result["vbt"]["adaptive_convex_spreads"] is True
        assert result["lender"]["adaptive_rates"] is True
        assert result["cb"]["adaptive_early_warning"] is True
        # Should NOT set pre-run flags
        assert "adaptive_planning_horizon" not in result["trader"]
        assert "adaptive_term_structure" not in result["vbt"]

    def test_full_sets_both(self):
        result = build_adaptive_overrides("full", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100)
        # Pre-run
        assert result["trader"]["adaptive_planning_horizon"] is True
        assert result["vbt"]["adaptive_term_structure"] is True
        # Within-run
        assert result["trader"]["adaptive_risk_aversion"] is True
        assert result["vbt"]["adaptive_convex_spreads"] is True

    def test_planning_horizon_scales_with_maturity(self):
        r5 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0.5"), Decimal("1"), 5, 100)
        r15 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0.5"), Decimal("1"), 15, 100)
        assert r5["trader"]["planning_horizon"] == 5
        assert r15["trader"]["planning_horizon"] == 15

    def test_planning_horizon_clamped(self):
        r1 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0.5"), Decimal("1"), 1, 100)
        r25 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0.5"), Decimal("1"), 25, 100)
        assert r1["trader"]["planning_horizon"] >= 1
        assert r25["trader"]["planning_horizon"] <= 20

    def test_lender_risk_aversion_calibrates_to_kappa(self):
        high_k = build_adaptive_overrides("calibrated", Decimal("2"), Decimal("0.5"), Decimal("1"), 10, 100)
        low_k = build_adaptive_overrides("calibrated", Decimal("0.3"), Decimal("0.5"), Decimal("1"), 10, 100)
        assert high_k["lender"]["risk_aversion"] < low_k["lender"]["risk_aversion"]

    def test_accepts_string_preset(self):
        result = build_adaptive_overrides("full", Decimal("1"), Decimal("0.5"), Decimal("1"), 10, 100)
        assert len(result["trader"]) > 0
