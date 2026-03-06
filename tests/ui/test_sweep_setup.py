"""Tests for the sweep setup questionnaire."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bilancio.ui.sweep_setup import (
    ANALYSIS_MENU,
    FEATURE_SECTIONS,
    PostSweepAnalysisResult,
    SweepSetupResult,
    _available_analyses,
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


class TestAnalysisMenu:
    """Test ANALYSIS_MENU structure and filtering."""

    def test_all_entries_have_required_keys(self):
        for name, meta in ANALYSIS_MENU.items():
            assert "label" in meta, f"{name} missing 'label'"
            assert "desc" in meta, f"{name} missing 'desc'"
            assert "group" in meta, f"{name} missing 'group'"
            assert "sweep_types" in meta, f"{name} missing 'sweep_types'"
            assert meta["group"] in ("core", "extended", "deep_dive"), f"{name} has invalid group: {meta['group']}"

    def test_core_analyses_available_for_all_sweep_types(self):
        for name, meta in ANALYSIS_MENU.items():
            if meta["group"] == "core":
                assert "dealer" in meta["sweep_types"]
                assert "bank" in meta["sweep_types"]
                assert "nbfi" in meta["sweep_types"]

    def test_dealer_only_analyses(self):
        """strategy_outcomes and dealer_usage are dealer-only."""
        assert ANALYSIS_MENU["strategy_outcomes"]["sweep_types"] == ["dealer"]
        assert ANALYSIS_MENU["dealer_usage"]["sweep_types"] == ["dealer"]

    def test_treynor_not_available_for_nbfi(self):
        assert "nbfi" not in ANALYSIS_MENU["treynor"]["sweep_types"]
        assert "dealer" in ANALYSIS_MENU["treynor"]["sweep_types"]
        assert "bank" in ANALYSIS_MENU["treynor"]["sweep_types"]

    def test_menu_has_expected_entries(self):
        expected = {
            "drilldowns", "deltas", "dynamics", "narrative",
            "strategy_outcomes", "dealer_usage", "mechanism_activity", "treynor",
        }
        assert set(ANALYSIS_MENU.keys()) == expected


class TestAvailableAnalyses:
    """Test _available_analyses filtering by sweep type."""

    def test_dealer_gets_all_analyses(self):
        available = _available_analyses("dealer")
        assert len(available) == 8
        assert "strategy_outcomes" in available
        assert "dealer_usage" in available
        assert "treynor" in available

    def test_bank_excludes_dealer_specific(self):
        available = _available_analyses("bank")
        assert "strategy_outcomes" not in available
        assert "dealer_usage" not in available
        assert "treynor" in available
        assert "drilldowns" in available
        assert "mechanism_activity" in available

    def test_nbfi_excludes_dealer_and_treynor(self):
        available = _available_analyses("nbfi")
        assert "strategy_outcomes" not in available
        assert "dealer_usage" not in available
        assert "treynor" not in available
        assert "drilldowns" in available
        assert "mechanism_activity" in available

    def test_unknown_sweep_type_returns_empty(self):
        available = _available_analyses("unknown")
        assert len(available) == 0


class TestPostSweepAnalysisResult:
    """Test PostSweepAnalysisResult dataclass."""

    def test_basic_construction(self):
        result = PostSweepAnalysisResult(
            analyses=["drilldowns", "deltas"],
            extended=["strategy_outcomes"],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.analyses == ["drilldowns", "deltas"]
        assert result.extended == ["strategy_outcomes"]
        assert result.treynor_kappas is None
        assert result.kappas is None

    def test_all_analyses_property(self):
        result = PostSweepAnalysisResult(
            analyses=["drilldowns"],
            extended=["dealer_usage", "mechanism_activity"],
            treynor_kappas=None,
            kappas=None,
        )
        assert result.all_analyses == ["drilldowns", "dealer_usage", "mechanism_activity"]

    def test_has_treynor_when_kappas_set(self):
        result = PostSweepAnalysisResult(
            analyses=[], extended=[], treynor_kappas=["0.5", "1.0"], kappas=None,
        )
        assert result.has_treynor is True

    def test_has_treynor_with_auto(self):
        result = PostSweepAnalysisResult(
            analyses=[], extended=[], treynor_kappas=["auto"], kappas=None,
        )
        assert result.has_treynor is True

    def test_no_treynor_when_none(self):
        result = PostSweepAnalysisResult(
            analyses=["drilldowns"], extended=[], treynor_kappas=None, kappas=None,
        )
        assert result.has_treynor is False

    def test_no_treynor_when_empty(self):
        result = PostSweepAnalysisResult(
            analyses=[], extended=[], treynor_kappas=[], kappas=None,
        )
        assert result.has_treynor is False

    def test_kappas_focus(self):
        result = PostSweepAnalysisResult(
            analyses=["drilldowns"],
            extended=[],
            treynor_kappas=None,
            kappas=[0.5, 1.0, 2.0],
        )
        assert result.kappas == [0.5, 1.0, 2.0]

    def test_empty_result(self):
        result = PostSweepAnalysisResult(
            analyses=[], extended=[], treynor_kappas=None, kappas=None,
        )
        assert result.all_analyses == []
        assert result.has_treynor is False


class TestValidPostAnalyses:
    """Test that VALID_POST_ANALYSES in sweep.py includes new analysis names."""

    def test_valid_analyses_includes_extended(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        assert "strategy_outcomes" in VALID_POST_ANALYSES
        assert "dealer_usage" in VALID_POST_ANALYSES
        assert "mechanism_activity" in VALID_POST_ANALYSES
        assert "treynor" in VALID_POST_ANALYSES

    def test_valid_analyses_includes_core(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        assert "drilldowns" in VALID_POST_ANALYSES
        assert "deltas" in VALID_POST_ANALYSES
        assert "dynamics" in VALID_POST_ANALYSES
        assert "narrative" in VALID_POST_ANALYSES

    def test_valid_analyses_matches_menu_keys(self):
        from bilancio.ui.cli.sweep import VALID_POST_ANALYSES

        assert set(VALID_POST_ANALYSES) == set(ANALYSIS_MENU.keys())
