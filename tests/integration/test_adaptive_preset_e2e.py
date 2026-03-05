"""End-to-end tests for adaptive preset behavioral differences (W0.2).

Verifies that:
1. Each preset (static, calibrated, responsive, full) produces distinct profile values
2. Overrides survive the full pipeline: build_overrides → YAML merge → Pydantic → profile
3. Preset progression is monotonic (static ⊂ calibrated ∪ responsive ⊂ full)
4. A regression guard detects when an override is silently dropped
5. Run-level wiring survives _prepare_run → run_scenario → run_day
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from bilancio.config.models import (
    BalancedDealerConfig,
    LenderScenarioConfig,
    RiskAssessmentConfig,
)
from bilancio.decision.adaptive import build_adaptive_overrides
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.experiments.ring import RingSweepRunner
from bilancio.decision.profiles import BankProfile, LenderProfile, TraderProfile, VBTProfile
from bilancio.decision.risk_assessment import RiskAssessmentParams
from bilancio.ui.run import run_scenario

# ── Shared helpers ──────────────────────────────────────────────────


SCENARIO_PARAMS = {
    "kappa": Decimal("0.5"),
    "mu": Decimal("0"),
    "c": Decimal("1"),
    "maturity_days": 10,
    "n_agents": 100,
}


def _build(preset: str) -> dict[str, dict]:
    """Build adaptive overrides for a given preset."""
    return build_adaptive_overrides(preset, **SCENARIO_PARAMS)


def _merge_into_yaml(yaml_dict: dict, overrides: dict, *buckets: str) -> dict:
    """Merge override buckets into a YAML-style dict (mimics ring.py merge logic)."""
    merged = copy.deepcopy(yaml_dict)
    for bucket in buckets:
        for k, v in overrides.get(bucket, {}).items():
            merged[k] = str(v) if isinstance(v, Decimal) else v
    return merged


def _trader_profile_from(cfg: BalancedDealerConfig) -> TraderProfile:
    return TraderProfile(
        risk_aversion=cfg.risk_aversion,
        planning_horizon=cfg.planning_horizon,
        aggressiveness=cfg.aggressiveness,
        default_observability=cfg.default_observability,
        buy_reserve_fraction=cfg.buy_reserve_fraction,
        trading_motive=cfg.trading_motive,
        adaptive_planning_horizon=cfg.adaptive_planning_horizon,
        adaptive_risk_aversion=cfg.adaptive_risk_aversion,
        adaptive_reserves=cfg.adaptive_reserves,
        adaptive_ev_term_structure=cfg.adaptive_ev_term_structure,
    )


def _vbt_profile_from(cfg: BalancedDealerConfig) -> VBTProfile:
    return VBTProfile(
        mid_sensitivity=cfg.vbt_mid_sensitivity,
        spread_sensitivity=cfg.vbt_spread_sensitivity,
        spread_scale=cfg.spread_scale,
        flow_sensitivity=cfg.flow_sensitivity,
        adaptive_term_structure=cfg.adaptive_term_structure,
        adaptive_base_spreads=cfg.adaptive_base_spreads,
        adaptive_convex_spreads=cfg.adaptive_convex_spreads,
        term_strength=cfg.term_strength,
    )


def _lender_profile_from(cfg: LenderScenarioConfig) -> LenderProfile:
    return LenderProfile(
        kappa=cfg.kappa,
        risk_aversion=cfg.risk_aversion,
        planning_horizon=cfg.planning_horizon,
        profit_target=cfg.profit_target,
        max_loan_maturity=cfg.max_loan_maturity or 10,
        adaptive_risk_aversion=cfg.adaptive_risk_aversion,
        adaptive_loan_maturity=cfg.adaptive_loan_maturity,
        adaptive_rates=cfg.adaptive_rates,
        adaptive_capital_conservation=cfg.adaptive_capital_conservation,
        adaptive_prevention=cfg.adaptive_prevention,
    )


def _risk_params_from(cfg: RiskAssessmentConfig) -> RiskAssessmentParams:
    return RiskAssessmentParams(
        lookback_window=cfg.lookback_window,
        use_issuer_specific=cfg.use_issuer_specific,
        adaptive_lookback=cfg.adaptive_lookback,
        adaptive_issuer_specific=cfg.adaptive_issuer_specific,
        adaptive_ev_term_structure=cfg.adaptive_ev_term_structure,
        term_strength=cfg.term_strength,
    )


# ── Pipeline helper: override → YAML → Pydantic → Profile ──────────


def _full_pipeline_trader(preset: str) -> TraderProfile:
    overrides = _build(preset)
    yaml_dict = _merge_into_yaml({"enabled": True}, overrides, "trader", "vbt")
    cfg = BalancedDealerConfig(**yaml_dict)
    return _trader_profile_from(cfg)


def _full_pipeline_vbt(preset: str) -> VBTProfile:
    overrides = _build(preset)
    yaml_dict = _merge_into_yaml({"enabled": True}, overrides, "trader", "vbt")
    cfg = BalancedDealerConfig(**yaml_dict)
    return _vbt_profile_from(cfg)


def _full_pipeline_lender(preset: str) -> LenderProfile:
    overrides = _build(preset)
    yaml_dict = _merge_into_yaml({"enabled": True, "kappa": "0.5"}, overrides, "lender")
    cfg = LenderScenarioConfig(**yaml_dict)
    return _lender_profile_from(cfg)


def _full_pipeline_risk_params(preset: str) -> RiskAssessmentParams:
    overrides = _build(preset)
    yaml_dict = _merge_into_yaml({}, overrides, "risk_params")
    cfg = RiskAssessmentConfig(**yaml_dict)
    return _risk_params_from(cfg)


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


class TestStaticProducesDefaults:
    """Static preset must produce vanilla profiles (no adaptation)."""

    def test_trader_all_flags_off(self):
        tp = _full_pipeline_trader("static")
        assert tp.adaptive_planning_horizon is False
        assert tp.adaptive_risk_aversion is False
        assert tp.adaptive_reserves is False
        assert tp.adaptive_ev_term_structure is False

    def test_vbt_all_flags_off(self):
        vp = _full_pipeline_vbt("static")
        assert vp.adaptive_term_structure is False
        assert vp.adaptive_base_spreads is False
        assert vp.adaptive_convex_spreads is False

    def test_lender_all_flags_off(self):
        lp = _full_pipeline_lender("static")
        assert lp.adaptive_risk_aversion is False
        assert lp.adaptive_loan_maturity is False
        assert lp.adaptive_rates is False
        assert lp.adaptive_capital_conservation is False
        assert lp.adaptive_prevention is False

    def test_risk_params_all_flags_off(self):
        rp = _full_pipeline_risk_params("static")
        assert rp.adaptive_lookback is False
        assert rp.adaptive_ev_term_structure is False


class TestCalibratedSetsPreRunOnly:
    """Calibrated preset enables [PRE] flags but not [RUN] flags."""

    def test_trader_pre_on_run_off(self):
        tp = _full_pipeline_trader("calibrated")
        # [PRE]
        assert tp.adaptive_planning_horizon is True
        assert tp.planning_horizon == 10
        # [RUN] must be off
        assert tp.adaptive_risk_aversion is False
        assert tp.adaptive_reserves is False

    def test_vbt_pre_on_run_off(self):
        vp = _full_pipeline_vbt("calibrated")
        assert vp.adaptive_term_structure is True
        assert vp.adaptive_base_spreads is True
        assert vp.adaptive_convex_spreads is False  # [RUN]

    def test_lender_pre_on_run_off(self):
        lp = _full_pipeline_lender("calibrated")
        assert lp.adaptive_risk_aversion is True
        assert lp.adaptive_loan_maturity is True
        assert lp.adaptive_rates is False  # [RUN]
        assert lp.adaptive_capital_conservation is False  # [RUN]
        assert lp.adaptive_prevention is False  # [RUN]

    def test_risk_params_pre_on_run_off(self):
        rp = _full_pipeline_risk_params("calibrated")
        assert rp.adaptive_lookback is True
        assert rp.adaptive_issuer_specific is True
        assert rp.adaptive_ev_term_structure is False  # [RUN]


class TestResponsiveSetsRunOnly:
    """Responsive preset enables [RUN] flags but not [PRE] flags."""

    def test_trader_run_on_pre_off(self):
        tp = _full_pipeline_trader("responsive")
        assert tp.adaptive_planning_horizon is False  # [PRE]
        assert tp.adaptive_risk_aversion is True  # [RUN]
        assert tp.adaptive_reserves is True  # [RUN]

    def test_vbt_run_on_pre_off(self):
        vp = _full_pipeline_vbt("responsive")
        assert vp.adaptive_term_structure is False  # [PRE]
        assert vp.adaptive_base_spreads is False  # [PRE]
        assert vp.adaptive_convex_spreads is True  # [RUN]

    def test_lender_run_on_pre_off(self):
        lp = _full_pipeline_lender("responsive")
        assert lp.adaptive_risk_aversion is False  # [PRE]
        assert lp.adaptive_rates is True  # [RUN]
        assert lp.adaptive_capital_conservation is True  # [RUN]
        assert lp.adaptive_prevention is True  # [RUN]

    def test_risk_params_run_on_pre_off(self):
        rp = _full_pipeline_risk_params("responsive")
        assert rp.adaptive_lookback is False  # [PRE]
        assert rp.adaptive_ev_term_structure is True  # [RUN]


class TestFullSetsBoth:
    """Full preset enables all flags."""

    def test_trader_all_on(self):
        tp = _full_pipeline_trader("full")
        assert tp.adaptive_planning_horizon is True  # [PRE]
        assert tp.adaptive_risk_aversion is True  # [RUN]
        assert tp.adaptive_reserves is True  # [RUN]
        # adaptive_ev_term_structure flows through risk_params bucket, not trader

    def test_vbt_all_on(self):
        vp = _full_pipeline_vbt("full")
        assert vp.adaptive_term_structure is True
        assert vp.adaptive_base_spreads is True
        assert vp.adaptive_convex_spreads is True

    def test_lender_all_on(self):
        lp = _full_pipeline_lender("full")
        assert lp.adaptive_risk_aversion is True
        assert lp.adaptive_loan_maturity is True
        assert lp.adaptive_rates is True
        assert lp.adaptive_capital_conservation is True
        assert lp.adaptive_prevention is True

    def test_risk_params_all_on(self):
        rp = _full_pipeline_risk_params("full")
        assert rp.adaptive_lookback is True
        assert rp.adaptive_ev_term_structure is True


class TestBankCBPresetWiring:
    """Bank and CB presets flow through _balanced_config → profile."""

    def test_bank_corridor_differs_calibrated_vs_static(self):
        kappa, mu, c = Decimal("0.5"), Decimal("0"), Decimal("1")
        bp_static = BankProfile()
        bp_adaptive = BankProfile(adaptive_corridor=True)
        mid_static = bp_static.corridor_mid(kappa, mu, c)
        mid_adaptive = bp_adaptive.corridor_mid(kappa, mu, c)
        assert mid_adaptive > mid_static  # combined stress > base stress

    def test_cb_flags_set_by_presets(self):
        for preset, expect_betas, expect_ew in [
            ("static", False, False),
            ("calibrated", True, False),
            ("responsive", False, True),
            ("full", True, True),
        ]:
            overrides = _build(preset)
            assert overrides["cb"].get("adaptive_betas", False) is expect_betas, preset
            assert overrides["cb"].get("adaptive_early_warning", False) is expect_ew, preset


class TestPresetProgression:
    """Verify that presets form a proper hierarchy: static ⊂ calibrated ∪ responsive ⊂ full."""

    def test_full_is_superset_of_calibrated(self):
        cal = _build("calibrated")
        full = _build("full")
        for bucket in ("trader", "risk_params", "vbt", "bank", "lender", "cb"):
            for key, value in cal[bucket].items():
                assert key in full[bucket], f"calibrated key {bucket}.{key} missing from full"
                assert full[bucket][key] == value, (
                    f"full.{bucket}.{key}={full[bucket][key]} != calibrated value {value}"
                )

    def test_full_is_superset_of_responsive(self):
        resp = _build("responsive")
        full = _build("full")
        for bucket in ("trader", "risk_params", "vbt", "bank", "lender", "cb"):
            for key, value in resp[bucket].items():
                assert key in full[bucket], f"responsive key {bucket}.{key} missing from full"
                assert full[bucket][key] == value, (
                    f"full.{bucket}.{key}={full[bucket][key]} != responsive value {value}"
                )

    def test_calibrated_and_responsive_disjoint_flags(self):
        """PRE and RUN flags should not overlap (each flag is one scope)."""
        cal = _build("calibrated")
        resp = _build("responsive")
        for bucket in ("trader", "risk_params", "vbt", "lender", "cb"):
            cal_flags = {k for k in cal[bucket] if k.startswith("adaptive_")}
            resp_flags = {k for k in resp[bucket] if k.startswith("adaptive_")}
            overlap = cal_flags & resp_flags
            assert not overlap, f"Overlapping flags in {bucket}: {overlap}"


class TestParameterCalibration:
    """Verify that pre-run calibration produces kappa/maturity-dependent values."""

    def test_planning_horizon_scales_with_maturity(self):
        o5 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 5, 100)
        o15 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 15, 100)
        assert o5["trader"]["planning_horizon"] == 5
        assert o15["trader"]["planning_horizon"] == 15

    def test_lender_ra_increases_with_stress(self):
        o_low = build_adaptive_overrides("calibrated", Decimal("0.3"), Decimal("0"), Decimal("1"), 10, 100)
        o_high = build_adaptive_overrides("calibrated", Decimal("2"), Decimal("0"), Decimal("1"), 10, 100)
        assert o_low["lender"]["risk_aversion"] > o_high["lender"]["risk_aversion"]

    def test_stress_horizon_scales_with_maturity(self):
        o5 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 5, 100)
        o15 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 15, 100)
        assert o5["vbt"]["stress_horizon"] == 5
        assert o15["vbt"]["stress_horizon"] == 15

    def test_lookback_scales_with_maturity(self):
        o5 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 5, 100)
        o15 = build_adaptive_overrides("calibrated", Decimal("1"), Decimal("0"), Decimal("1"), 15, 100)
        assert o5["risk_params"]["lookback_window"] == 5
        assert o15["risk_params"]["lookback_window"] == 15


class TestRegressionGuard:
    """Regression test: all override keys from build_adaptive_overrides must be
    accepted by Pydantic models without being silently dropped.

    If a new flag is added to the preset builder but not to the Pydantic model,
    this test will fail.
    """

    @pytest.fixture
    def full_overrides(self) -> dict[str, dict]:
        return _build("full")

    def test_trader_overrides_accepted_by_pydantic(self, full_overrides):
        yaml_dict = {"enabled": True}
        for k, v in full_overrides["trader"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v
        for k, v in full_overrides["vbt"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v
        cfg = BalancedDealerConfig(**yaml_dict)
        # Every trader override key must appear in the config
        for k in full_overrides["trader"]:
            assert hasattr(cfg, k), f"trader override {k} dropped by BalancedDealerConfig"
        for k in full_overrides["vbt"]:
            assert hasattr(cfg, k), f"vbt override {k} dropped by BalancedDealerConfig"

    def test_risk_params_overrides_accepted_by_pydantic(self, full_overrides):
        yaml_dict = {}
        for k, v in full_overrides["risk_params"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v
        cfg = RiskAssessmentConfig(**yaml_dict)
        for k in full_overrides["risk_params"]:
            assert hasattr(cfg, k), f"risk_params override {k} dropped by RiskAssessmentConfig"

    def test_lender_overrides_accepted_by_pydantic(self, full_overrides):
        yaml_dict = {"enabled": True, "kappa": "0.5"}
        for k, v in full_overrides["lender"].items():
            yaml_dict[k] = str(v) if isinstance(v, Decimal) else v
        cfg = LenderScenarioConfig(**yaml_dict)
        for k in full_overrides["lender"]:
            assert hasattr(cfg, k), f"lender override {k} dropped by LenderScenarioConfig"

    def test_all_override_keys_have_profile_fields(self, full_overrides):
        """Every adaptive_* flag emitted by the preset builder must exist
        as a field on the corresponding profile dataclass."""
        profile_map = {
            "trader": TraderProfile,
            "vbt": VBTProfile,
            "lender": LenderProfile,
            "bank": BankProfile,
        }
        for bucket, profile_cls in profile_map.items():
            fields = {f.name for f in profile_cls.__dataclass_fields__.values()}
            for k in full_overrides[bucket]:
                if k.startswith("adaptive_"):
                    assert k in fields, (
                        f"Override flag {bucket}.{k} has no corresponding field "
                        f"on {profile_cls.__name__}"
                    )


class TestRunLevelPresetPipeline:
    """Run-level E2E: --adapt must survive _prepare_run -> run_scenario -> run_day."""

    @staticmethod
    def _runner(tmp_path, preset: str) -> RingSweepRunner:
        return RingSweepRunner(
            out_dir=tmp_path / f"run_{preset}",
            name_prefix="adapt_e2e",
            n_agents=8,
            maturity_days=12,
            Q_total=Decimal("200"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
            dealer_enabled=True,
            balanced_mode=True,
            n_banks=1,  # Ensure bank/CB adaptive flags are exercised.
            adapt_preset=preset,
        )

    @staticmethod
    def _capture_runtime_flags(monkeypatch, scenario_path) -> dict[str, bool | int | None]:
        from bilancio.engines import simulation as sim_mod

        observed: dict[str, bool | int | None] = {}
        original_run_day = sim_mod.run_day

        def wrapped_run_day(system, *args, **kwargs):
            if not observed:
                subsystem = system.state.dealer_subsystem
                assert subsystem is not None
                observed["trader_adaptive_risk_aversion"] = (
                    subsystem.trader_profile.adaptive_risk_aversion
                )
                observed["vbt_adaptive_convex_spreads"] = (
                    subsystem.vbt_profile.adaptive_convex_spreads
                )
                observed["vbt_stress_horizon"] = subsystem.vbt_profile.stress_horizon

                cb = next(
                    (a for a in system.state.agents.values() if isinstance(a, CentralBank)),
                    None,
                )
                assert cb is not None
                observed["cb_adaptive_betas"] = cb.adaptive_betas
                observed["cb_adaptive_early_warning"] = cb.adaptive_early_warning
            return original_run_day(system, *args, **kwargs)

        monkeypatch.setattr(sim_mod, "run_day", wrapped_run_day)
        run_scenario(
            scenario_path,
            mode="until_stable",
            max_days=2,
            quiet_days=1,
            show="none",
        )
        assert observed, "Expected wrapped run_day to capture runtime flags"
        return observed

    def test_static_vs_full_differs_in_runtime_objects(self, tmp_path, monkeypatch):
        static_runner = self._runner(tmp_path, "static")
        static_prepared = static_runner._prepare_run(
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0.5"),
            monotonicity=Decimal("0"),
            seed=1,
        )
        static_obs = self._capture_runtime_flags(monkeypatch, static_prepared.scenario_path)

        full_runner = self._runner(tmp_path, "full")
        full_prepared = full_runner._prepare_run(
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0.5"),
            monotonicity=Decimal("0"),
            seed=1,
        )
        full_obs = self._capture_runtime_flags(monkeypatch, full_prepared.scenario_path)

        # Trader/VBT RUN flags must differ at runtime.
        assert static_obs["trader_adaptive_risk_aversion"] is False
        assert full_obs["trader_adaptive_risk_aversion"] is True
        assert static_obs["vbt_adaptive_convex_spreads"] is False
        assert full_obs["vbt_adaptive_convex_spreads"] is True

        # PRE numeric calibration should differ at runtime (maturity_days=12).
        assert static_obs["vbt_stress_horizon"] == 5
        assert full_obs["vbt_stress_horizon"] == 12

        # Bank/CB adaptive flags must differ at runtime.
        assert static_obs["cb_adaptive_betas"] is False
        assert full_obs["cb_adaptive_betas"] is True
        assert static_obs["cb_adaptive_early_warning"] is False
        assert full_obs["cb_adaptive_early_warning"] is True
