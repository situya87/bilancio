"""Tests for bilancio.analysis.post_sweep module.

Focuses on pure logic functions: path resolution, CSV helpers, data loading,
auto-detection, and the public dispatch API. Plotly-heavy chart generation
is tested indirectly by mocking the rendering and verifying file outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bilancio.analysis.post_sweep import (
    VALID_ANALYSES,
    VALID_SWEEP_TYPES,
    SweepPaths,
    _approx_eq,
    _auto_detect_kappas,
    _extract_agent_outcomes,
    _extract_defaults_by_day,
    _find_run_in_csv,
    _read_csv,
    _resolve_sweep_paths,
    _safe_float,
    _safe_load_json,
    run_post_sweep_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPARISON_HEADER = (
    "kappa,concentration,mu,outside_mid_ratio,seed,"
    "delta_passive,delta_active,phi_passive,phi_active,"
    "trading_effect,passive_run_id,active_run_id\n"
)


def _minimal_csv_content(
    kappas: list[float] | None = None,
) -> str:
    """Generate minimal comparison CSV content."""
    if kappas is None:
        kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
    lines = [COMPARISON_HEADER.strip()]
    for k in kappas:
        lines.append(
            f"{k},1,0,0.9,42,0.3,0.2,0.7,0.8,0.1,passive_run_{k},active_run_{k}"
        )
    return "\n".join(lines) + "\n"


@pytest.fixture
def sweep_dir(tmp_path: Path) -> Path:
    """Create a minimal sweep directory structure with comparison.csv."""
    agg = tmp_path / "aggregate"
    agg.mkdir()
    csv_path = agg / "comparison.csv"
    csv_path.write_text(_minimal_csv_content())
    # Create run directories for dealer type
    for arm in ("passive", "active"):
        runs = tmp_path / arm / "runs"
        runs.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def bank_sweep_dir(tmp_path: Path) -> Path:
    """Create a minimal bank sweep directory structure."""
    agg = tmp_path / "aggregate"
    agg.mkdir()
    header = (
        "kappa,concentration,mu,outside_mid_ratio,seed,"
        "delta_idle,delta_lend,phi_idle,phi_lend,"
        "bank_lending_effect,idle_run_id,lend_run_id\n"
    )
    line = "0.5,1,0,0.9,42,0.35,0.20,0.65,0.80,0.15,idle_run_0.5,lend_run_0.5\n"
    (agg / "comparison.csv").write_text(header + line)
    for arm in ("bank_idle", "bank_lend"):
        (tmp_path / arm / "runs").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. SweepPaths dataclass
# ---------------------------------------------------------------------------


class TestSweepPaths:
    def test_construction_and_fields(self, tmp_path: Path):
        sp = SweepPaths(
            experiment_root=tmp_path,
            sweep_type="dealer",
            comparison_csv=tmp_path / "comparison.csv",
            treatment_dir=tmp_path / "active" / "runs",
            baseline_dir=tmp_path / "passive" / "runs",
            treatment_col="active_run_id",
            baseline_col="passive_run_id",
            treatment_label="Active (Dealer)",
            baseline_label="Passive (No Dealer)",
        )
        assert sp.sweep_type == "dealer"
        assert sp.treatment_col == "active_run_id"
        assert sp.baseline_col == "passive_run_id"
        assert sp.stats_summary is None
        assert sp.stats_sensitivity is None

    def test_optional_stats_fields(self, tmp_path: Path):
        sp = SweepPaths(
            experiment_root=tmp_path,
            sweep_type="bank",
            comparison_csv=tmp_path / "c.csv",
            treatment_dir=tmp_path / "t",
            baseline_dir=tmp_path / "b",
            treatment_col="lend_run_id",
            baseline_col="idle_run_id",
            treatment_label="Bank Lending",
            baseline_label="Bank Idle",
            stats_summary=tmp_path / "stats_summary.json",
            stats_sensitivity=tmp_path / "stats_sensitivity.json",
        )
        assert sp.stats_summary == tmp_path / "stats_summary.json"
        assert sp.stats_sensitivity == tmp_path / "stats_sensitivity.json"


# ---------------------------------------------------------------------------
# 2-3. _resolve_sweep_paths
# ---------------------------------------------------------------------------


class TestResolveSweepPaths:
    def test_dealer_paths(self, tmp_path: Path):
        sp = _resolve_sweep_paths(tmp_path, "dealer")
        assert sp.sweep_type == "dealer"
        assert sp.comparison_csv == tmp_path / "aggregate" / "comparison.csv"
        assert sp.treatment_dir == tmp_path / "active" / "runs"
        assert sp.baseline_dir == tmp_path / "passive" / "runs"
        assert sp.treatment_col == "active_run_id"
        assert sp.baseline_col == "passive_run_id"
        assert sp.treatment_label == "Active (Dealer)"
        assert sp.baseline_label == "Passive (No Dealer)"

    def test_bank_paths(self, tmp_path: Path):
        sp = _resolve_sweep_paths(tmp_path, "bank")
        assert sp.sweep_type == "bank"
        assert sp.treatment_dir == tmp_path / "bank_lend" / "runs"
        assert sp.baseline_dir == tmp_path / "bank_idle" / "runs"
        assert sp.treatment_col == "lend_run_id"
        assert sp.baseline_col == "idle_run_id"

    def test_nbfi_paths(self, tmp_path: Path):
        sp = _resolve_sweep_paths(tmp_path, "nbfi")
        assert sp.sweep_type == "nbfi"
        assert sp.treatment_dir == tmp_path / "nbfi_lend" / "runs"
        assert sp.baseline_dir == tmp_path / "nbfi_idle" / "runs"

    def test_invalid_sweep_type_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown sweep type"):
            _resolve_sweep_paths(tmp_path, "invalid")

    def test_stats_paths_set(self, tmp_path: Path):
        sp = _resolve_sweep_paths(tmp_path, "dealer")
        assert sp.stats_summary == tmp_path / "aggregate" / "stats_summary.json"
        assert sp.stats_sensitivity == tmp_path / "aggregate" / "stats_sensitivity.json"


# ---------------------------------------------------------------------------
# 4-5. run_post_sweep_analysis validation
# ---------------------------------------------------------------------------


class TestRunPostSweepAnalysisValidation:
    def test_missing_experiment_root(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Comparison CSV not found"):
            run_post_sweep_analysis(missing, "dealer", ["narrative"])

    def test_invalid_analysis_name(self, sweep_dir: Path):
        with pytest.raises(ValueError, match="Unknown analysis"):
            run_post_sweep_analysis(sweep_dir, "dealer", ["bogus_analysis"])

    def test_invalid_sweep_type(self, sweep_dir: Path):
        with pytest.raises(ValueError, match="sweep_type must be one of"):
            run_post_sweep_analysis(sweep_dir, "invalid_type", ["narrative"])


# ---------------------------------------------------------------------------
# 6. Sweep type defaults
# ---------------------------------------------------------------------------


class TestSweepTypeDefaults:
    def test_valid_sweep_types_constant(self):
        assert VALID_SWEEP_TYPES == ("dealer", "bank", "nbfi")

    def test_valid_analyses_constant(self):
        assert VALID_ANALYSES == ("drilldowns", "deltas", "dynamics", "narrative")

    def test_each_sweep_type_resolves(self, tmp_path: Path):
        for st in VALID_SWEEP_TYPES:
            sp = _resolve_sweep_paths(tmp_path, st)
            assert sp.sweep_type == st


# ---------------------------------------------------------------------------
# 7. Output directory creation
# ---------------------------------------------------------------------------


class TestOutputDirCreation:
    def test_default_output_dir(self, sweep_dir: Path):
        """When output_dir=None, analysis creates aggregate/analysis/."""
        # We need to mock the analysis functions to avoid plotly
        with patch(
            "bilancio.analysis.post_sweep._run_narrative"
        ) as mock_narrative:
            mock_narrative.return_value = (
                sweep_dir / "aggregate" / "analysis" / "narrative_report.html"
            )
            run_post_sweep_analysis(sweep_dir, "dealer", ["narrative"])
            # The output_dir should have been created
            assert (sweep_dir / "aggregate" / "analysis").is_dir()

    def test_custom_output_dir(self, sweep_dir: Path):
        custom_out = sweep_dir / "custom_output" / "deep"
        with patch(
            "bilancio.analysis.post_sweep._run_narrative"
        ) as mock_narrative:
            mock_narrative.return_value = custom_out / "narrative_report.html"
            run_post_sweep_analysis(
                sweep_dir, "dealer", ["narrative"], output_dir=custom_out
            )
            assert custom_out.is_dir()


# ---------------------------------------------------------------------------
# 8. CSV loading with minimal data
# ---------------------------------------------------------------------------


class TestCsvLoading:
    def test_read_csv(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "kappa,delta_passive,delta_active\n"
            "0.5,0.3,0.2\n"
            "1.0,0.1,0.05\n"
        )
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert rows[0]["kappa"] == "0.5"
        assert rows[1]["delta_active"] == "0.05"

    def test_auto_detect_kappas_few(self, tmp_path: Path):
        csv_path = tmp_path / "c.csv"
        csv_path.write_text(_minimal_csv_content(kappas=[0.3, 1.0]))
        result = _auto_detect_kappas(csv_path)
        assert result == [0.3, 1.0]

    def test_auto_detect_kappas_picks_min_median_max(self, tmp_path: Path):
        csv_path = tmp_path / "c.csv"
        csv_path.write_text(
            _minimal_csv_content(kappas=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        )
        result = _auto_detect_kappas(csv_path, n=3)
        assert 0.1 in result  # min
        assert 8.0 in result  # max
        assert len(result) == 3

    def test_safe_float_valid(self):
        assert _safe_float({"val": "3.14"}, "val") == pytest.approx(3.14)

    def test_safe_float_empty(self):
        assert _safe_float({"val": ""}, "val") is None

    def test_safe_float_missing_key(self):
        assert _safe_float({}, "val") is None

    def test_safe_float_invalid(self):
        assert _safe_float({"val": "abc"}, "val") is None

    def test_approx_eq(self):
        assert _approx_eq(0.500, 0.505, tol=0.01)
        assert not _approx_eq(0.500, 0.520, tol=0.01)


# ---------------------------------------------------------------------------
# 9. _find_run_in_csv
# ---------------------------------------------------------------------------


class TestFindRunInCsv:
    def test_find_matching_run(self):
        rows = [
            {"kappa": "0.5", "concentration": "1.0", "mu": "0.5", "active_run_id": "run_a"},
            {"kappa": "1.0", "concentration": "1.0", "mu": "0.5", "active_run_id": "run_b"},
        ]
        assert _find_run_in_csv(rows, 0.5, "active_run_id") == "run_a"
        assert _find_run_in_csv(rows, 1.0, "active_run_id") == "run_b"

    def test_find_no_match(self):
        rows = [{"kappa": "0.5", "active_run_id": "run_a"}]
        assert _find_run_in_csv(rows, 99.0, "active_run_id") is None

    def test_prefers_c1_mu05(self):
        rows = [
            {"kappa": "0.5", "concentration": "2.0", "mu": "0.0", "rid": "wrong"},
            {"kappa": "0.5", "concentration": "1.0", "mu": "0.5", "rid": "preferred"},
        ]
        assert _find_run_in_csv(rows, 0.5, "rid") == "preferred"


# ---------------------------------------------------------------------------
# 10-11. Analysis functions with mocked plotly
# ---------------------------------------------------------------------------


class TestDrilldownsOutput:
    """Test _run_drilldowns produces the expected output file."""

    def test_drilldowns_dispatched(self, sweep_dir: Path):
        """Verify that run_post_sweep_analysis dispatches drilldowns correctly."""
        out_dir = sweep_dir / "output"
        with patch(
            "bilancio.analysis.post_sweep._run_drilldowns"
        ) as mock_dd:
            mock_dd.return_value = out_dir / "drilldown_dashboard.html"
            results = run_post_sweep_analysis(
                sweep_dir, "dealer", ["drilldowns"], output_dir=out_dir
            )
            assert mock_dd.called
            assert "drilldowns" in results

    def test_treatment_deltas_dispatched(self, sweep_dir: Path):
        out_dir = sweep_dir / "output"
        with patch(
            "bilancio.analysis.post_sweep._run_treatment_deltas"
        ) as mock_td:
            mock_td.return_value = out_dir / "treatment_deltas_dashboard.html"
            results = run_post_sweep_analysis(
                sweep_dir, "dealer", ["deltas"], output_dir=out_dir
            )
            assert mock_td.called
            assert "deltas" in results

    def test_dynamics_dispatched(self, sweep_dir: Path):
        out_dir = sweep_dir / "output"
        with patch(
            "bilancio.analysis.post_sweep._run_dynamics"
        ) as mock_dyn:
            mock_dyn.return_value = out_dir / "dynamics_dashboard.html"
            results = run_post_sweep_analysis(
                sweep_dir, "dealer", ["dynamics"], output_dir=out_dir
            )
            assert mock_dyn.called
            assert "dynamics" in results


# ---------------------------------------------------------------------------
# 12. Narrative report
# ---------------------------------------------------------------------------


class TestNarrativeReport:
    def test_narrative_produces_html(self, sweep_dir: Path):
        """_run_narrative writes an HTML file without needing plotly."""
        from bilancio.analysis.post_sweep import _run_narrative

        sp = _resolve_sweep_paths(sweep_dir, "dealer")
        out_dir = sweep_dir / "analysis_out"
        out_dir.mkdir()
        result = _run_narrative(sp, [0.25, 1.0, 4.0], out_dir)
        assert result.exists()
        assert result.name == "narrative_report.html"
        content = result.read_text()
        assert "Dealer" in content
        assert "Executive Summary" in content
        assert "Effect by Liquidity Level" in content
        assert "Key Findings" in content

    def test_narrative_with_single_kappa(self, sweep_dir: Path):
        from bilancio.analysis.post_sweep import _run_narrative

        sp = _resolve_sweep_paths(sweep_dir, "dealer")
        out_dir = sweep_dir / "analysis_out2"
        out_dir.mkdir()
        result = _run_narrative(sp, [0.5], out_dir)
        assert result.exists()
        content = result.read_text()
        assert "0.5" in content


# ---------------------------------------------------------------------------
# 13. All-NaN data edge case
# ---------------------------------------------------------------------------


class TestAllNanData:
    def test_safe_float_all_nan(self):
        row = {"val": "nan"}
        result = _safe_float(row, "val")
        # float("nan") parses successfully but is NaN
        assert result is not None
        import math
        assert math.isnan(result)

    def test_narrative_with_empty_effects_hits_zero_division(self, tmp_path: Path):
        """CSV with blank delta values triggers ZeroDivisionError in _run_narrative.

        This documents a known bug: when all delta values are empty/unparseable,
        all_effects is an empty list and line 1876 divides by len(all_effects)==0.
        """
        from bilancio.analysis.post_sweep import _run_narrative

        agg = tmp_path / "aggregate"
        agg.mkdir()
        csv_content = (
            "kappa,concentration,mu,outside_mid_ratio,seed,"
            "delta_passive,delta_active,phi_passive,phi_active,"
            "trading_effect,passive_run_id,active_run_id\n"
            "0.5,1,0,0.9,42,,,,,,passive_0.5,active_0.5\n"
        )
        (agg / "comparison.csv").write_text(csv_content)
        sp = _resolve_sweep_paths(tmp_path, "dealer")
        out_dir = tmp_path / "analysis"
        out_dir.mkdir()
        # Known bug: ZeroDivisionError when all delta values are empty
        with pytest.raises(ZeroDivisionError):
            _run_narrative(sp, [0.5], out_dir)


# ---------------------------------------------------------------------------
# 14. Analysis dispatching (subset)
# ---------------------------------------------------------------------------


class TestAnalysisDispatching:
    def test_multiple_analyses_dispatched(self, sweep_dir: Path):
        out_dir = sweep_dir / "output"
        with (
            patch("bilancio.analysis.post_sweep._run_drilldowns") as mock_dd,
            patch("bilancio.analysis.post_sweep._run_narrative") as mock_nr,
        ):
            mock_dd.return_value = out_dir / "drilldown_dashboard.html"
            mock_nr.return_value = out_dir / "narrative_report.html"
            results = run_post_sweep_analysis(
                sweep_dir,
                "dealer",
                ["drilldowns", "narrative"],
                output_dir=out_dir,
            )
            assert mock_dd.called
            assert mock_nr.called
            assert len(results) == 2
            assert "drilldowns" in results
            assert "narrative" in results

    def test_single_analysis_dispatched(self, sweep_dir: Path):
        out_dir = sweep_dir / "output"
        with (
            patch("bilancio.analysis.post_sweep._run_drilldowns") as mock_dd,
            patch("bilancio.analysis.post_sweep._run_treatment_deltas") as mock_td,
            patch("bilancio.analysis.post_sweep._run_dynamics") as mock_dyn,
            patch("bilancio.analysis.post_sweep._run_narrative") as mock_nr,
        ):
            mock_nr.return_value = out_dir / "narrative_report.html"
            results = run_post_sweep_analysis(
                sweep_dir, "dealer", ["narrative"], output_dir=out_dir
            )
            assert mock_nr.called
            assert not mock_dd.called
            assert not mock_td.called
            assert not mock_dyn.called

    def test_failed_analysis_logged_not_raised(self, sweep_dir: Path):
        """If an analysis function raises, it is logged but not re-raised."""
        out_dir = sweep_dir / "output"
        with patch(
            "bilancio.analysis.post_sweep._run_drilldowns"
        ) as mock_dd:
            mock_dd.side_effect = RuntimeError("No runs found")
            results = run_post_sweep_analysis(
                sweep_dir, "dealer", ["drilldowns"], output_dir=out_dir
            )
            # drilldowns should NOT appear in results because it failed
            assert "drilldowns" not in results


# ---------------------------------------------------------------------------
# 15. Comparison CSV loaded correctly
# ---------------------------------------------------------------------------


class TestComparisonCsvLoading:
    def test_full_comparison_csv_columns(self, sweep_dir: Path):
        csv_path = sweep_dir / "aggregate" / "comparison.csv"
        rows = _read_csv(csv_path)
        assert len(rows) == 5
        # Check all expected columns are present
        expected_cols = {
            "kappa", "concentration", "mu", "outside_mid_ratio", "seed",
            "delta_passive", "delta_active", "phi_passive", "phi_active",
            "trading_effect", "passive_run_id", "active_run_id",
        }
        assert expected_cols.issubset(set(rows[0].keys()))

    def test_kappa_values_parsed(self, sweep_dir: Path):
        csv_path = sweep_dir / "aggregate" / "comparison.csv"
        rows = _read_csv(csv_path)
        kappas = sorted(float(r["kappa"]) for r in rows)
        assert kappas == [0.25, 0.5, 1.0, 2.0, 4.0]

    def test_run_ids_present(self, sweep_dir: Path):
        csv_path = sweep_dir / "aggregate" / "comparison.csv"
        rows = _read_csv(csv_path)
        for row in rows:
            assert row["passive_run_id"].startswith("passive_run_")
            assert row["active_run_id"].startswith("active_run_")


# ---------------------------------------------------------------------------
# Additional helper tests
# ---------------------------------------------------------------------------


class TestExtractHelpers:
    def test_extract_defaults_by_day(self):
        events = [
            {"type": "default", "day": 1, "default_type": "primary"},
            {"type": "default", "day": 1, "default_type": "secondary"},
            {"type": "default", "day": 2, "default_type": "primary"},
            {"type": "default", "day": 2, "default_type": "cascade"},
            {"type": "settlement", "day": 1},  # non-default event
        ]
        result = _extract_defaults_by_day(events)
        assert result[1]["primary"] == 1
        assert result[1]["secondary"] == 1
        assert result[2]["primary"] == 1
        assert result[2]["secondary"] == 1  # cascade maps to secondary

    def test_extract_defaults_by_day_empty(self):
        assert _extract_defaults_by_day([]) == {}

    def test_extract_agent_outcomes(self):
        events = [
            {"type": "default", "agent_id": "a1"},
            {"type": "trade", "agent_id": "a2"},
            {"type": "trade", "agent_id": "a2"},
            {"type": "cash_position", "agent_id": "a2", "cash": 100.0},
        ]
        result = _extract_agent_outcomes(events)
        assert result["a1"]["defaulted"] is True
        assert result["a2"]["n_trades"] == 2
        assert result["a2"]["cash_final"] == 100.0
        assert result["a2"]["defaulted"] is False

    def test_safe_load_json_valid(self, tmp_path: Path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "value"}))
        result = _safe_load_json(p)
        assert result == {"key": "value"}

    def test_safe_load_json_missing(self, tmp_path: Path):
        p = tmp_path / "missing.json"
        assert _safe_load_json(p) is None

    def test_safe_load_json_invalid(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        assert _safe_load_json(p) is None

    def test_safe_load_json_none_path(self):
        assert _safe_load_json(None) is None


class TestBankSweepIntegration:
    """Test bank sweep type paths and CSV interaction."""

    def test_bank_paths_resolve(self, bank_sweep_dir: Path):
        sp = _resolve_sweep_paths(bank_sweep_dir, "bank")
        assert sp.treatment_label == "Bank Lending"
        assert sp.baseline_label == "Bank Idle"

    def test_bank_narrative(self, bank_sweep_dir: Path):
        from bilancio.analysis.post_sweep import _run_narrative

        sp = _resolve_sweep_paths(bank_sweep_dir, "bank")
        out_dir = bank_sweep_dir / "analysis"
        out_dir.mkdir()
        result = _run_narrative(sp, [0.5], out_dir)
        assert result.exists()
        content = result.read_text()
        assert "Bank" in content

    def test_kappas_auto_detected_with_explicit_list(self, sweep_dir: Path):
        """run_post_sweep_analysis uses provided kappas instead of auto-detect."""
        out_dir = sweep_dir / "output"
        with patch(
            "bilancio.analysis.post_sweep._run_narrative"
        ) as mock_nr:
            mock_nr.return_value = out_dir / "narrative_report.html"
            run_post_sweep_analysis(
                sweep_dir,
                "dealer",
                ["narrative"],
                output_dir=out_dir,
                kappas=[0.25, 1.0],
            )
            # Verify the kappas passed to the analysis function
            call_args = mock_nr.call_args
            assert call_args[0][1] == [0.25, 1.0]
