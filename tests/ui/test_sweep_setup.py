"""Tests for the sweep setup questionnaire."""

from __future__ import annotations

from pathlib import Path

import pytest

from bilancio.ui.sweep_setup import (
    _ADAPTIVE_FLAGS_DEFAULTS,
    _LENDING_RISK_DEFAULTS,
    ANALYSIS_MENU,
    DATA_ANALYSIS_MENU,
    FEATURE_SECTIONS,
    VIZ_MENU,
    PostSweepAnalysisResult,
    SweepSetupResult,
    _available_analyses,
    _available_data_analyses,
    _available_visualizations,
    _compute_run_count,
    build_cli_args,
    load_preset,
    save_preset,
)


class TestSweepSetupResult:
    """Test SweepSetupResult dataclass."""

    def test_basic_construction(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=True,
            params={"n_agents": 100, "kappas": "0.5,1.0"},
        )
        assert result.sweep_type == "balanced"
        assert result.cloud is True
        assert result.params["n_agents"] == 100
        assert result.launch is False
        assert result.out_dir is None
        assert result.preset_path is None

    def test_full_construction(self):
        result = SweepSetupResult(
            sweep_type="bank",
            cloud=False,
            params={"n_agents": 50},
            out_dir=Path("/tmp/test"),
            launch=True,
            preset_path=Path("presets/test.yaml"),
        )
        assert result.sweep_type == "bank"
        assert result.launch is True
        assert result.out_dir == Path("/tmp/test")


class TestPresetSaveLoad:
    """Test preset YAML save/load round-trip."""

    def test_save_and_load(self, tmp_path: Path):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=True,
            params={
                "n_agents": 100,
                "maturity_days": 10,
                "kappas": "0.25,0.5,1,2,4",
                "risk_assessment": True,
                "enable_lender": False,
                "adapt": "static",
            },
        )

        preset_path = tmp_path / "test_preset.yaml"
        save_preset(result, preset_path)

        assert preset_path.exists()

        loaded = load_preset(preset_path)
        assert loaded["sweep_type"] == "balanced"
        assert loaded["cloud"] is True
        assert loaded["params"]["n_agents"] == 100
        assert loaded["params"]["kappas"] == "0.25,0.5,1,2,4"
        assert loaded["params"]["risk_assessment"] is True

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        result = SweepSetupResult(
            sweep_type="ring",
            cloud=False,
            params={"n_agents": 5},
        )
        nested = tmp_path / "deep" / "nested" / "preset.yaml"
        save_preset(result, nested)
        assert nested.exists()

    def test_load_invalid_preset(self, tmp_path: Path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("just: a string")

        with pytest.raises(ValueError, match="missing 'sweep_type'"):
            load_preset(bad_file)

    def test_round_trip_preserves_all_params(self, tmp_path: Path):
        params = {
            "n_agents": 50,
            "maturity_days": 5,
            "q_total": 5000,
            "kappas": "0.3,0.5",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
            "risk_aversion": "0.5",
            "enable_lender": True,
            "adapt": "calibrated",
        }
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params=params,
        )
        path = tmp_path / "roundtrip.yaml"
        save_preset(result, path)
        loaded = load_preset(path)

        for k, v in params.items():
            assert loaded["params"][k] == v, f"Mismatch for {k}: {loaded['params'][k]} != {v}"


class TestFeatureSections:
    """Test feature section visibility per sweep type."""

    def test_balanced_has_all_sections(self):
        sections = FEATURE_SECTIONS["balanced"]
        assert "dealer" in sections
        assert "nbfi_lender" in sections
        assert "banking" in sections
        assert "risk_assessment" in sections
        assert "adaptive" in sections
        assert "trader" in sections
        assert "performance" in sections

    def test_bank_has_minimal_sections(self):
        sections = FEATURE_SECTIONS["bank"]
        assert "dealer" not in sections
        assert "nbfi_lender" not in sections
        assert "risk_assessment" in sections
        assert "trader" in sections

    def test_nbfi_has_no_dealer(self):
        sections = FEATURE_SECTIONS["nbfi"]
        assert "dealer" not in sections
        assert "risk_assessment" in sections
        assert "trader" in sections

    def test_ring_is_minimal(self):
        sections = FEATURE_SECTIONS["ring"]
        assert sections == ["performance"]


class TestComputeRunCount:
    """Test run count computation."""

    def test_balanced_default(self):
        params = {
            "kappas": "0.25,0.5,1,2,4",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
        }
        n_combos, n_runs = _compute_run_count("balanced", params)
        assert n_combos == 5  # 5 kappas × 1 × 1 × 1
        assert n_runs == 10  # 5 × 2 arms

    def test_balanced_with_lender(self):
        params = {
            "kappas": "0.5,1.0",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
            "enable_lender": True,
        }
        n_combos, n_runs = _compute_run_count("balanced", params)
        assert n_combos == 2
        assert n_runs == 6  # 2 × 3 arms (passive + active + lender)

    def test_bank_sweep(self):
        params = {
            "kappas": "0.3,0.5,1.0,2.0",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
        }
        n_combos, n_runs = _compute_run_count("bank", params)
        assert n_combos == 4
        assert n_runs == 8  # 4 × 2 arms

    def test_ring_sweep(self):
        params = {
            "kappas": "0.25,0.5,1,2,4",
            "concentrations": "0.5,1,2",
            "mus": "0,0.5",
            "outside_mid_ratios": "0.90",
        }
        n_combos, n_runs = _compute_run_count("ring", params)
        assert n_combos == 30  # 5 × 3 × 2 × 1
        assert n_runs == 30  # 1 arm

    def test_with_replicates(self):
        params = {
            "kappas": "0.5,1.0",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
            "n_replicates": 3,
        }
        n_combos, n_runs = _compute_run_count("balanced", params)
        assert n_combos == 6  # 2 × 1 × 1 × 1 × 3
        assert n_runs == 12  # 6 × 2

    def test_balanced_with_banking(self):
        params = {
            "kappas": "0.5",
            "concentrations": "1",
            "mus": "0",
            "outside_mid_ratios": "0.90",
            "enable_banking": True,
        }
        n_combos, n_runs = _compute_run_count("balanced", params)
        assert n_combos == 1
        assert n_runs == 5  # 1 × (2 base + 3 banking)


class TestBuildCliArgs:
    """Test CLI argument generation from SweepSetupResult."""

    def test_balanced_basic(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=True,
            params={
                "n_agents": 50,
                "maturity_days": 5,
                "kappas": "0.5,1.0",
            },
            out_dir=Path("out/test"),
        )
        args = build_cli_args(result)
        assert "--out-dir" in args
        assert "out/test" in args
        assert "--cloud" in args
        assert "--n-agents" in args
        assert "50" in args
        assert "--kappas" in args
        assert "0.5,1.0" in args

    def test_no_cloud(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"n_agents": 100},
        )
        args = build_cli_args(result)
        assert "--cloud" not in args

    def test_bank_params(self):
        result = SweepSetupResult(
            sweep_type="bank",
            cloud=False,
            params={
                "n_banks": 5,
                "reserve_ratio": "0.50",
                "credit_risk_loading": "0.5",
            },
        )
        args = build_cli_args(result)
        assert "--n-banks" in args
        assert "5" in args
        assert "--reserve-ratio" in args
        assert "--credit-risk-loading" in args

    def test_nbfi_share_mapping(self):
        result = SweepSetupResult(
            sweep_type="nbfi",
            cloud=False,
            params={"lender_share": "0.10"},
        )
        args = build_cli_args(result)
        assert "--nbfi-share" in args
        assert "0.10" in args

    def test_rollover_flag(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"rollover": True},
        )
        args = build_cli_args(result)
        assert "--rollover" in args

        result.params["rollover"] = False
        args = build_cli_args(result)
        assert "--no-rollover" in args

    def test_fast_atomic_flag(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"fast_atomic": True},
        )
        args = build_cli_args(result)
        assert "--fast-atomic" in args

    def test_balanced_enable_lender(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"enable_lender": True},
        )
        args = build_cli_args(result)
        assert "--enable-lender" in args

    def test_adapt_non_static(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"adapt": "calibrated"},
        )
        args = build_cli_args(result)
        assert "--adapt" in args
        assert "calibrated" in args

    def test_no_out_dir(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={},
            out_dir=None,
        )
        args = build_cli_args(result)
        assert "--out-dir" not in args

    def test_bank_does_not_emit_trader_flags(self):
        """P1 regression: bank sweep must not emit trader/risk flags."""
        result = SweepSetupResult(
            sweep_type="bank",
            cloud=False,
            params={
                "n_agents": 50,
                "risk_aversion": "0.5",
                "planning_horizon": 10,
                "aggressiveness": "1.0",
                "default_observability": "1.0",
                "trading_motive": "liquidity_then_earning",
                "risk_premium": "0.02",
                "risk_urgency": "0.30",
                "n_banks": 5,
            },
        )
        args = build_cli_args(result)
        assert "--risk-aversion" not in args
        assert "--planning-horizon" not in args
        assert "--aggressiveness" not in args
        assert "--default-observability" not in args
        assert "--trading-motive" not in args
        assert "--risk-premium" not in args
        assert "--risk-urgency" not in args
        # Bank-specific flags should still be emitted
        assert "--n-banks" in args

    def test_nbfi_does_not_emit_trader_flags(self):
        """P1 regression: nbfi sweep must not emit trader/risk flags."""
        result = SweepSetupResult(
            sweep_type="nbfi",
            cloud=False,
            params={
                "risk_aversion": "0.5",
                "risk_premium": "0.02",
                "lender_share": "0.10",
            },
        )
        args = build_cli_args(result)
        assert "--risk-aversion" not in args
        assert "--risk-premium" not in args
        assert "--nbfi-share" in args

    def test_balanced_emits_banking_flags(self):
        """P2 regression: enable_banking should emit --enable-bank-* flags."""
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={"enable_banking": True},
        )
        args = build_cli_args(result)
        assert "--enable-bank-passive" in args
        assert "--enable-bank-dealer" in args
        assert "--enable-bank-dealer-nbfi" in args

    def test_balanced_emits_lender_tuning_flags(self):
        """P2 regression: lender params should be emitted when enable_lender is True."""
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "enable_lender": True,
                "lender_share": "0.10",
                "lender_min_coverage": "0.5",
                "lender_ranking_mode": "profit",
                "lender_coverage_mode": "gate",
                "lender_maturity_matching": True,
                "lender_preventive_lending": True,
            },
        )
        args = build_cli_args(result)
        assert "--enable-lender" in args
        assert "--lender-share" in args
        assert "--lender-min-coverage" in args
        assert "--lender-ranking-mode" in args
        assert "--lender-coverage-mode" in args
        assert "--lender-maturity-matching" in args
        assert "--lender-preventive-lending" in args

    def test_balanced_no_lender_flags_when_disabled(self):
        """Lender tuning flags should NOT be emitted when enable_lender is False."""
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "enable_lender": False,
                "lender_share": "0.10",
                "lender_min_coverage": "0.5",
            },
        )
        args = build_cli_args(result)
        assert "--lender-share" not in args
        assert "--lender-min-coverage" not in args


class TestAnalysisMenu:
    """Test DATA_ANALYSIS_MENU, VIZ_MENU, and legacy ANALYSIS_MENU structure."""

    def test_data_menu_entries_have_required_keys(self):
        for name, meta in DATA_ANALYSIS_MENU.items():
            assert "label" in meta, f"{name} missing 'label'"
            assert "desc" in meta, f"{name} missing 'desc'"
            assert "group" in meta, f"{name} missing 'group'"
            assert "sweep_types" in meta, f"{name} missing 'sweep_types'"
            assert meta["group"] == "data", f"{name} has group {meta['group']}, expected 'data'"

    def test_viz_menu_entries_have_required_keys(self):
        for name, meta in VIZ_MENU.items():
            assert "label" in meta, f"{name} missing 'label'"
            assert "desc" in meta, f"{name} missing 'desc'"
            assert "group" in meta, f"{name} missing 'group'"
            assert "sweep_types" in meta, f"{name} missing 'sweep_types'"
            assert meta["group"] == "viz", f"{name} has group {meta['group']}, expected 'viz'"

    def test_legacy_menu_combines_both(self):
        assert set(ANALYSIS_MENU.keys()) == set(DATA_ANALYSIS_MENU.keys()) | set(VIZ_MENU.keys())

    def test_data_menu_has_expected_entries(self):
        expected = {
            "frontier",
            "strategy_outcomes",
            "dealer_usage",
            "mechanism_activity",
            "contagion",
            "credit_creation",
            "network",
            "pricing",
            "beliefs",
            "funding",
        }
        assert set(DATA_ANALYSIS_MENU.keys()) == expected

    def test_viz_menu_has_expected_entries(self):
        expected = {
            "drilldowns",
            "deltas",
            "dynamics",
            "narrative",
            "treynor",
            "comparison",
            "report",
            "notebook",
        }
        assert set(VIZ_MENU.keys()) == expected

    def test_core_viz_available_for_all_sweep_types(self):
        for name in ("drilldowns", "deltas", "dynamics", "narrative"):
            meta = VIZ_MENU[name]
            assert "dealer" in meta["sweep_types"]
            assert "bank" in meta["sweep_types"]
            assert "nbfi" in meta["sweep_types"]

    def test_dealer_only_data_analyses(self):
        assert DATA_ANALYSIS_MENU["strategy_outcomes"]["sweep_types"] == ["dealer"]
        assert DATA_ANALYSIS_MENU["dealer_usage"]["sweep_types"] == ["dealer"]
        assert DATA_ANALYSIS_MENU["pricing"]["sweep_types"] == ["dealer"]

    def test_treynor_not_available_for_nbfi(self):
        assert "nbfi" not in VIZ_MENU["treynor"]["sweep_types"]
        assert "dealer" in VIZ_MENU["treynor"]["sweep_types"]
        assert "bank" in VIZ_MENU["treynor"]["sweep_types"]

    def test_new_data_analyses_available_for_all(self):
        for name in ("frontier", "contagion", "credit_creation", "network", "beliefs", "funding"):
            meta = DATA_ANALYSIS_MENU[name]
            assert "dealer" in meta["sweep_types"]
            assert "bank" in meta["sweep_types"]
            assert "nbfi" in meta["sweep_types"]


class TestAvailableAnalyses:
    """Test analysis filtering by sweep type."""

    def test_dealer_gets_all_data_analyses(self):
        available = _available_data_analyses("dealer")
        assert len(available) == 10  # All data analyses
        assert "strategy_outcomes" in available
        assert "dealer_usage" in available
        assert "pricing" in available

    def test_dealer_gets_all_visualizations(self):
        available = _available_visualizations("dealer")
        assert len(available) == 8
        assert "treynor" in available
        assert "drilldowns" in available
        assert "notebook" in available

    def test_bank_excludes_dealer_specific_data(self):
        available = _available_data_analyses("bank")
        assert "strategy_outcomes" not in available
        assert "dealer_usage" not in available
        assert "pricing" not in available
        assert "frontier" in available
        assert "mechanism_activity" in available

    def test_bank_gets_treynor(self):
        available = _available_visualizations("bank")
        assert "treynor" in available
        assert "drilldowns" in available

    def test_nbfi_excludes_dealer_and_treynor(self):
        data = _available_data_analyses("nbfi")
        viz = _available_visualizations("nbfi")
        assert "strategy_outcomes" not in data
        assert "dealer_usage" not in data
        assert "pricing" not in data
        assert "treynor" not in viz
        assert "drilldowns" in viz
        assert "mechanism_activity" in data

    def test_legacy_available_analyses_combines(self):
        available = _available_analyses("dealer")
        assert len(available) == 18  # 10 data + 8 viz

    def test_unknown_sweep_type_returns_empty(self):
        assert len(_available_data_analyses("unknown")) == 0
        assert len(_available_visualizations("unknown")) == 0


class TestPostSweepAnalysisResult:
    """Test PostSweepAnalysisResult dataclass."""

    def test_basic_construction(self):
        result = PostSweepAnalysisResult(
            data_analyses=["frontier", "contagion"],
            visualizations=["drilldowns", "deltas"],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.data_analyses == ["frontier", "contagion"]
        assert result.visualizations == ["drilldowns", "deltas"]
        assert result.treynor_kappas is None
        assert result.kappas is None

    def test_all_selected_property(self):
        result = PostSweepAnalysisResult(
            data_analyses=["frontier"],
            visualizations=["drilldowns", "treynor"],
            treynor_kappas=["auto"],
            kappas=None,
        )
        assert result.all_selected == ["frontier", "drilldowns", "treynor"]

    def test_legacy_analyses_property(self):
        """Legacy .analyses returns viz items excluding treynor and comparison."""
        result = PostSweepAnalysisResult(
            data_analyses=["strategy_outcomes"],
            visualizations=["drilldowns", "deltas", "treynor", "comparison"],
            treynor_kappas=["auto"],
            kappas=None,
        )
        assert result.analyses == ["drilldowns", "deltas"]

    def test_legacy_extended_property(self):
        """Legacy .extended returns data_analyses."""
        result = PostSweepAnalysisResult(
            data_analyses=["dealer_usage", "mechanism_activity"],
            visualizations=["drilldowns"],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.extended == ["dealer_usage", "mechanism_activity"]

    def test_legacy_all_analyses_property(self):
        result = PostSweepAnalysisResult(
            data_analyses=["frontier"],
            visualizations=["drilldowns"],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.all_analyses == ["frontier", "drilldowns"]

    def test_has_treynor_when_kappas_set(self):
        result = PostSweepAnalysisResult(
            data_analyses=[],
            visualizations=[],
            treynor_kappas=["0.5", "1.0"],
            kappas=None,
        )
        assert result.has_treynor is True

    def test_has_treynor_with_auto(self):
        result = PostSweepAnalysisResult(
            data_analyses=[],
            visualizations=[],
            treynor_kappas=["auto"],
            kappas=None,
        )
        assert result.has_treynor is True

    def test_no_treynor_when_none(self):
        result = PostSweepAnalysisResult(
            data_analyses=["frontier"],
            visualizations=[],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.has_treynor is False

    def test_no_treynor_when_empty(self):
        result = PostSweepAnalysisResult(
            data_analyses=[],
            visualizations=[],
            treynor_kappas=[],
            kappas=None,
        )
        assert result.has_treynor is False

    def test_kappas_focus(self):
        result = PostSweepAnalysisResult(
            data_analyses=["frontier"],
            visualizations=[],
            treynor_kappas=None,
            kappas=[0.5, 1.0, 2.0],
        )
        assert result.kappas == [0.5, 1.0, 2.0]

    def test_empty_result(self):
        result = PostSweepAnalysisResult(
            data_analyses=[],
            visualizations=[],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.all_selected == []
        assert result.has_treynor is False


class TestValidPostAnalyses:
    """Test that VALID_POST_ANALYSES in sweep.py includes all menu keys."""

    def test_valid_analyses_includes_data(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        for key in DATA_ANALYSIS_MENU:
            assert key in VALID_POST_ANALYSES, f"{key} missing from VALID_POST_ANALYSES"

    def test_valid_analyses_includes_viz(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        for key in VIZ_MENU:
            assert key in VALID_POST_ANALYSES, f"{key} missing from VALID_POST_ANALYSES"

    def test_valid_analyses_matches_all_menu_keys(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        assert set(VALID_POST_ANALYSES) == set(DATA_ANALYSIS_MENU.keys()) | set(VIZ_MENU.keys())


class TestAdaptiveAndLendingDefaults:
    """Test new default dicts for adaptive flags and lending risk controls."""

    def test_adaptive_flags_defaults_has_expected_keys(self):
        expected = {
            "adaptive_planning_horizon",
            "adaptive_risk_aversion",
            "adaptive_reserves",
            "adaptive_lookback",
            "adaptive_issuer_specific",
            "adaptive_ev_term_structure",
            "adaptive_term_structure",
            "adaptive_base_spreads",
            "adaptive_convex_spreads",
        }
        assert set(_ADAPTIVE_FLAGS_DEFAULTS.keys()) == expected

    def test_adaptive_flags_all_default_false(self):
        for k, v in _ADAPTIVE_FLAGS_DEFAULTS.items():
            assert v is False, f"{k} should default to False"

    def test_lending_risk_defaults_has_expected_keys(self):
        expected = {
            "marginal_relief_min_ratio",
            "stress_risk_premium_scale",
            "high_risk_default_threshold",
            "high_risk_maturity_cap",
            "daily_expected_loss_budget_ratio",
            "run_expected_loss_budget_ratio",
            "stop_loss_realized_ratio",
            "collateralized_terms",
            "collateral_advance_rate",
        }
        assert set(_LENDING_RISK_DEFAULTS.keys()) == expected

    def test_lending_risk_defaults_types(self):
        assert isinstance(_LENDING_RISK_DEFAULTS["high_risk_maturity_cap"], int)
        assert isinstance(_LENDING_RISK_DEFAULTS["collateralized_terms"], bool)
        # All others should be strings (for Decimal conversion)
        for k, v in _LENDING_RISK_DEFAULTS.items():
            if k not in ("high_risk_maturity_cap", "collateralized_terms"):
                assert isinstance(v, str), f"{k} should be str, got {type(v)}"

    def test_build_cli_args_emits_adaptive_flags(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "adapt": "calibrated",
                "adaptive_planning_horizon": True,
                "adaptive_risk_aversion": False,
            },
        )
        args = build_cli_args(result)
        assert "--adaptive-planning-horizon" in args
        assert "--no-adaptive-risk-aversion" in args

    def test_build_cli_args_emits_lending_risk_controls(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "enable_lender": True,
                "marginal_relief_min_ratio": "0.05",
                "stress_risk_premium_scale": "2.0",
                "collateralized_terms": True,
                "collateral_advance_rate": "0.80",
            },
        )
        args = build_cli_args(result)
        assert "--lender-marginal-relief-min-ratio" in args
        assert "0.05" in args
        assert "--lender-stress-risk-premium-scale" in args
        assert "--lender-collateralized-terms" in args
        assert "--lender-collateral-advance-rate" in args

    def test_build_cli_args_no_lending_risk_when_lender_disabled(self):
        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "enable_lender": False,
                "marginal_relief_min_ratio": "0.05",
            },
        )
        args = build_cli_args(result)
        assert "--lender-marginal-relief-min-ratio" not in args
