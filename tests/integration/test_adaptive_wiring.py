"""End-to-end integration test: --adapt flag wiring through config pipeline (Plan 050).

Verifies that adaptive flags set by build_adaptive_overrides() actually reach
the profile dataclasses after passing through:
  YAML dict → Pydantic config model → apply.py → profile constructor

This is the test that catches the P1 bug where Pydantic silently dropped
unknown adaptive_* fields.
"""

from decimal import Decimal

from bilancio.config.models import (
    BalancedDealerConfig,
    LenderScenarioConfig,
    RiskAssessmentConfig,
)
from bilancio.decision.adaptive import build_adaptive_overrides
from bilancio.decision.profiles import BankProfile, LenderProfile, TraderProfile, VBTProfile
from bilancio.decision.risk_assessment import RiskAssessmentParams


class TestAdaptiveFlagsSurvivePydanticValidation:
    """Verify that adaptive flags are not silently dropped by Pydantic models."""

    def test_balanced_dealer_config_accepts_trader_flags(self):
        cfg = BalancedDealerConfig(
            enabled=True,
            adaptive_planning_horizon=True,
            adaptive_risk_aversion=True,
            adaptive_reserves=True,
            adaptive_ev_term_structure=True,
        )
        assert cfg.adaptive_planning_horizon is True
        assert cfg.adaptive_risk_aversion is True
        assert cfg.adaptive_reserves is True
        assert cfg.adaptive_ev_term_structure is True

    def test_balanced_dealer_config_accepts_vbt_flags(self):
        cfg = BalancedDealerConfig(
            enabled=True,
            adaptive_term_structure=True,
            adaptive_base_spreads=True,
            adaptive_convex_spreads=True,
            term_strength=Decimal("0.3"),
        )
        assert cfg.adaptive_term_structure is True
        assert cfg.adaptive_base_spreads is True
        assert cfg.adaptive_convex_spreads is True
        assert cfg.term_strength == Decimal("0.3")

    def test_risk_assessment_config_accepts_flags(self):
        cfg = RiskAssessmentConfig(
            adaptive_lookback=True,
            adaptive_ev_term_structure=True,
            term_strength=Decimal("0.7"),
        )
        assert cfg.adaptive_lookback is True
        assert cfg.adaptive_ev_term_structure is True
        assert cfg.term_strength == Decimal("0.7")

    def test_lender_config_accepts_flags(self):
        cfg = LenderScenarioConfig(
            enabled=True,
            adaptive_rates=True,
            adaptive_capital_conservation=True,
            adaptive_prevention=True,
        )
        assert cfg.adaptive_rates is True
        assert cfg.adaptive_capital_conservation is True
        assert cfg.adaptive_prevention is True


class TestOverridesReachProfiles:
    """Verify that overrides built by build_adaptive_overrides() can construct
    profiles with the expected flag values."""

    def test_calibrated_overrides_construct_trader_profile(self):
        overrides = build_adaptive_overrides(
            "calibrated", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100,
        )
        trader_kwargs = {
            k: v for k, v in overrides["trader"].items()
            if k in TraderProfile.__dataclass_fields__
        }
        tp = TraderProfile(**trader_kwargs)
        assert tp.adaptive_planning_horizon is True
        assert tp.planning_horizon == 10

    def test_calibrated_overrides_construct_vbt_profile(self):
        overrides = build_adaptive_overrides(
            "calibrated", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100,
        )
        vbt_kwargs = {
            k: v for k, v in overrides["vbt"].items()
            if k in VBTProfile.__dataclass_fields__
        }
        vp = VBTProfile(**vbt_kwargs)
        assert vp.adaptive_term_structure is True
        assert vp.adaptive_base_spreads is True

    def test_responsive_overrides_construct_lender_profile(self):
        overrides = build_adaptive_overrides(
            "responsive", Decimal("1"), Decimal("0.5"), Decimal("1"), 10, 100,
        )
        lender_kwargs = {
            k: v for k, v in overrides["lender"].items()
            if k in LenderProfile.__dataclass_fields__
        }
        lp = LenderProfile(**lender_kwargs)
        assert lp.adaptive_rates is True
        assert lp.adaptive_capital_conservation is True
        assert lp.adaptive_prevention is True

    def test_full_overrides_set_both_pre_and_run(self):
        overrides = build_adaptive_overrides(
            "full", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100,
        )
        # Pre-run flags
        assert overrides["trader"]["adaptive_planning_horizon"] is True
        assert overrides["vbt"]["adaptive_term_structure"] is True
        assert overrides["bank"]["adaptive_corridor"] is True
        assert overrides["cb"]["adaptive_betas"] is True
        # Within-run flags
        assert overrides["trader"]["adaptive_risk_aversion"] is True
        assert overrides["vbt"]["adaptive_convex_spreads"] is True
        assert overrides["cb"]["adaptive_early_warning"] is True
        assert overrides["lender"]["adaptive_rates"] is True


class TestPydanticRoundTrip:
    """Verify that overrides survive YAML dict → Pydantic model → profile
    construction (the exact pipeline that was broken before the fix)."""

    def test_trader_flags_survive_balanced_dealer_config(self):
        """Simulate: overrides merged into YAML dict → BalancedDealerConfig → TraderProfile."""
        overrides = build_adaptive_overrides(
            "full", Decimal("0.5"), Decimal("0"), Decimal("1"), 10, 100,
        )
        # Simulate YAML dict with overrides merged
        yaml_dict = {"enabled": True}
        for k, v in overrides["trader"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v
        for k, v in overrides["vbt"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v

        cfg = BalancedDealerConfig(**yaml_dict)

        # Build profile from config (mirrors apply.py)
        tp = TraderProfile(
            risk_aversion=cfg.risk_aversion,
            planning_horizon=cfg.planning_horizon,
            adaptive_planning_horizon=cfg.adaptive_planning_horizon,
            adaptive_risk_aversion=cfg.adaptive_risk_aversion,
            adaptive_reserves=cfg.adaptive_reserves,
            adaptive_ev_term_structure=cfg.adaptive_ev_term_structure,
        )
        assert tp.adaptive_planning_horizon is True
        assert tp.adaptive_risk_aversion is True
        assert tp.planning_horizon == 10  # scaled by preset

    def test_lender_flags_survive_lender_config(self):
        """Simulate: overrides merged into YAML dict → LenderScenarioConfig → LenderProfile."""
        overrides = build_adaptive_overrides(
            "full", Decimal("0.3"), Decimal("0"), Decimal("1"), 10, 100,
        )
        yaml_dict = {"enabled": True, "kappa": "0.3"}
        for k, v in overrides["lender"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v

        cfg = LenderScenarioConfig(**yaml_dict)

        lp = LenderProfile(
            kappa=cfg.kappa,
            risk_aversion=cfg.risk_aversion,
            adaptive_risk_aversion=cfg.adaptive_risk_aversion,
            adaptive_loan_maturity=cfg.adaptive_loan_maturity,
            adaptive_rates=cfg.adaptive_rates,
            adaptive_capital_conservation=cfg.adaptive_capital_conservation,
            adaptive_prevention=cfg.adaptive_prevention,
        )
        assert lp.adaptive_risk_aversion is True
        assert lp.adaptive_rates is True
        assert lp.adaptive_prevention is True
        # Risk aversion should be calibrated to kappa
        assert lp.risk_aversion > Decimal("0.3")  # stressed kappa increases RA

    def test_risk_params_flags_survive_config(self):
        """Simulate: overrides → RiskAssessmentConfig → RiskAssessmentParams."""
        overrides = build_adaptive_overrides(
            "full", Decimal("1"), Decimal("0.5"), Decimal("1"), 10, 100,
        )
        yaml_dict = {}
        for k, v in overrides["risk_params"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v

        cfg = RiskAssessmentConfig(**yaml_dict)

        rp = RiskAssessmentParams(
            adaptive_lookback=cfg.adaptive_lookback,
            adaptive_ev_term_structure=cfg.adaptive_ev_term_structure,
            term_strength=cfg.term_strength,
        )
        assert rp.adaptive_lookback is True
        assert rp.adaptive_ev_term_structure is True


class TestBankCBWiring:
    """Verify bank and CB adaptive flags can flow through _balanced_config."""

    def test_bank_adaptive_corridor_in_profile(self):
        bp = BankProfile(adaptive_corridor=True)
        assert bp.adaptive_corridor is True
        # With adaptive_corridor, corridor_mid should differ with mu/c
        mid_without = bp.corridor_mid(Decimal("0.5"))
        mid_with = bp.corridor_mid(Decimal("0.5"), Decimal("0"), Decimal("0.5"))
        assert mid_with != mid_without

    def test_cb_flags_on_central_bank(self):
        from bilancio.domain.agents.central_bank import CentralBank
        cb = CentralBank(
            id="cb-1", name="cb",
            adaptive_betas=True, adaptive_early_warning=True,
        )
        assert cb.adaptive_betas is True
        assert cb.adaptive_early_warning is True
