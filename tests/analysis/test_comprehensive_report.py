"""Tests for comprehensive_report module.

Verifies sweep metadata schema alignment, section rendering with synthetic data,
and graceful degradation when data is missing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bilancio.analysis.comprehensive_report import (
    _arm_runs_dir,
    _find_run_for_kappa,
    _load_comparison_data,
    _representative_kappas,
    _resolve_col,
    _safe_section,
    _sweep_meta,
    build_comprehensive_report,
)


# ---------------------------------------------------------------------------
# Sweep metadata schema tests
# ---------------------------------------------------------------------------


class TestSweepMeta:
    """Verify _SWEEP_META matches actual CSV schemas from each sweep runner."""

    def test_dealer_meta(self):
        meta = _sweep_meta("dealer")
        assert meta["effect_col"] == "trading_effect"
        assert meta["delta_treatment"] == "delta_active"
        assert meta["delta_baseline"] == "delta_passive"
        # CSV column names from balanced_comparison.py
        assert "active_run_id" in meta["run_id_treatment_candidates"]
        assert "passive_run_id" in meta["run_id_baseline_candidates"]

    def test_bank_meta(self):
        meta = _sweep_meta("bank")
        assert meta["effect_col"] == "bank_lending_effect"
        assert meta["delta_treatment"] == "delta_lend"
        assert meta["delta_baseline"] == "delta_idle"
        # CSV column names from bank_comparison.py
        assert "lend_run_id" in meta["run_id_treatment_candidates"]
        assert "idle_run_id" in meta["run_id_baseline_candidates"]

    def test_nbfi_meta_matches_nbfi_comparison_schema(self):
        """P1 regression: NBFI must use delta_lend/delta_idle, not delta_lender/delta_passive.

        The NBFI comparison runner (nbfi_comparison.py) outputs:
          - delta_idle, delta_lend, lending_effect
          - idle_run_id, lend_run_id
        """
        meta = _sweep_meta("nbfi")
        assert meta["effect_col"] == "lending_effect"
        assert meta["delta_treatment"] == "delta_lend"
        assert meta["delta_baseline"] == "delta_idle"
        assert "lend_run_id" in meta["run_id_treatment_candidates"]
        assert "idle_run_id" in meta["run_id_baseline_candidates"]
        # Must NOT reference the old wrong columns
        assert meta["delta_treatment"] != "delta_lender"
        assert meta["delta_baseline"] != "delta_passive"

    def test_unknown_sweep_type_raises(self):
        with pytest.raises(ValueError, match="Unknown sweep_type"):
            _sweep_meta("unknown")


# ---------------------------------------------------------------------------
# Path resolution tests
# ---------------------------------------------------------------------------


class TestArmRunsDir:
    """Verify arm directory mapping."""

    def test_dealer_arms(self, tmp_path: Path):
        assert _arm_runs_dir(tmp_path, "dealer", "treatment") == tmp_path / "active" / "runs"
        assert _arm_runs_dir(tmp_path, "dealer", "baseline") == tmp_path / "passive" / "runs"

    def test_bank_arms(self, tmp_path: Path):
        assert _arm_runs_dir(tmp_path, "bank", "treatment") == tmp_path / "bank_lend" / "runs"
        assert _arm_runs_dir(tmp_path, "bank", "baseline") == tmp_path / "bank_idle" / "runs"

    def test_nbfi_arms(self, tmp_path: Path):
        assert _arm_runs_dir(tmp_path, "nbfi", "treatment") == tmp_path / "nbfi_lend" / "runs"
        assert _arm_runs_dir(tmp_path, "nbfi", "baseline") == tmp_path / "nbfi_idle" / "runs"

    def test_unknown_falls_back(self, tmp_path: Path):
        assert _arm_runs_dir(tmp_path, "xyz", "foo") == tmp_path / "foo" / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_representative_kappas_picks_min_mid_max(self):
        data = {"kappa": np.array([0.25, 0.5, 1.0, 2.0, 4.0])}
        result = _representative_kappas(data, n=3)
        assert result == [0.25, 1.0, 4.0]

    def test_representative_kappas_returns_all_when_few(self):
        data = {"kappa": np.array([0.5, 1.0])}
        result = _representative_kappas(data, n=3)
        assert result == [0.5, 1.0]

    def test_representative_kappas_empty(self):
        assert _representative_kappas({}, n=3) == []

    def test_resolve_col_first_match(self):
        data = {"lend_run_id": [1, 2], "other": [3, 4]}
        assert _resolve_col(data, ["lend_run_id", "run_id_lend"]) == "lend_run_id"

    def test_resolve_col_fallback(self):
        data = {"run_id_lend": [1, 2]}
        assert _resolve_col(data, ["lend_run_id", "run_id_lend"]) == "run_id_lend"

    def test_resolve_col_none(self):
        assert _resolve_col({}, ["a", "b"]) is None

    def test_find_run_for_kappa(self):
        data = {
            "kappa": np.array([0.5, 1.0, 2.0]),
            "lend_run_id": np.array(["run_a", "run_b", "run_c"], dtype=object),
        }
        assert _find_run_for_kappa(data, 1.0, "lend_run_id") == "run_b"

    def test_find_run_for_kappa_missing(self):
        data = {
            "kappa": np.array([0.5]),
            "lend_run_id": np.array(["run_a"], dtype=object),
        }
        assert _find_run_for_kappa(data, 99.0, "lend_run_id") is None

    def test_safe_section_catches_errors(self):
        def bad_section():
            raise ValueError("boom")

        result = _safe_section(bad_section)
        assert "could not be rendered" in result
        assert "boom" in result

    def test_safe_section_returns_content(self):
        def good_section():
            return "<p>OK</p>"

        assert _safe_section(good_section) == "<p>OK</p>"


# ---------------------------------------------------------------------------
# Build report with synthetic CSV
# ---------------------------------------------------------------------------

def _write_synthetic_sweep(
    tmp_path: Path,
    sweep_type: str,
    n_kappas: int = 3,
) -> Path:
    """Create a minimal synthetic sweep directory with comparison.csv and events."""
    meta = _sweep_meta(sweep_type)
    agg_dir = tmp_path / "aggregate"
    agg_dir.mkdir(parents=True)
    csv_path = agg_dir / "comparison.csv"

    # Determine column names
    delta_t = meta["delta_treatment"]
    delta_b = meta["delta_baseline"]
    effect_col = meta["effect_col"]
    run_id_t = meta["run_id_treatment_candidates"][0]
    run_id_b = meta["run_id_baseline_candidates"][0]

    kappas = [0.5, 1.0, 2.0][:n_kappas]

    # Write CSV
    fieldnames = [
        "kappa", "concentration", "mu", "outside_mid_ratio",
        delta_t, delta_b, effect_col,
        run_id_t, run_id_b,
    ]
    rows = []
    for i, k in enumerate(kappas):
        dt = max(0.0, 0.5 - k * 0.15)
        db = max(0.0, 0.6 - k * 0.15)
        rows.append({
            "kappa": str(k),
            "concentration": "1",
            "mu": "0",
            "outside_mid_ratio": "0.90",
            delta_t: str(dt),
            delta_b: str(db),
            effect_col: str(db - dt),
            run_id_t: f"treat_run_{i}",
            run_id_b: f"base_run_{i}",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Create minimal run directories with events.jsonl
    for arm_key in ("treatment", "baseline"):
        runs_dir = _arm_runs_dir(tmp_path, sweep_type, arm_key)
        for i in range(n_kappas):
            run_id = f"treat_run_{i}" if arm_key == "treatment" else f"base_run_{i}"
            run_out = runs_dir / run_id / "out"
            run_out.mkdir(parents=True)
            events = [
                {"kind": "PhaseStart", "day": 0, "phase": "setup"},
                {"kind": "PayableCreated", "day": 0, "instr_id": f"PAY_{i}",
                 "debtor": "H1", "creditor": "H2", "face": 100},
                {"kind": "PayableSettled", "day": 1, "instr_id": f"PAY_{i}",
                 "debtor": "H1", "creditor": "H2", "amount": 100},
            ]
            with open(run_out / "events.jsonl", "w") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

    return tmp_path


class TestBuildComprehensiveReport:
    """Integration tests for the full report builder."""

    @pytest.mark.parametrize("sweep_type", ["dealer", "bank", "nbfi"])
    def test_report_generates_valid_html(self, tmp_path: Path, sweep_type: str):
        """Report generates valid HTML for all sweep types."""
        _write_synthetic_sweep(tmp_path, sweep_type)
        html = build_comprehensive_report(tmp_path, sweep_type)

        assert "<!DOCTYPE html>" in html
        assert f"Comprehensive Sweep Report: {sweep_type.title()}" in html
        assert "Summary Statistics" in html
        assert "plotly-2.35.2.min.js" in html

    @pytest.mark.parametrize("sweep_type", ["dealer", "bank", "nbfi"])
    def test_report_includes_delta_columns(self, tmp_path: Path, sweep_type: str):
        """Report renders summary stats referencing the correct delta columns."""
        _write_synthetic_sweep(tmp_path, sweep_type)
        html = build_comprehensive_report(tmp_path, sweep_type)

        meta = _sweep_meta(sweep_type)
        # The treatment label should appear in the summary table
        assert meta["treatment_label"] in html

    def test_nbfi_report_uses_correct_columns(self, tmp_path: Path):
        """Regression test: NBFI report uses delta_lend/delta_idle, not delta_lender/delta_passive."""
        _write_synthetic_sweep(tmp_path, "nbfi")
        data = _load_comparison_data(tmp_path)

        # The loaded data should have the correct columns
        assert "delta_lend" in data
        assert "delta_idle" in data
        assert "lending_effect" in data
        # Should NOT have the wrong columns
        assert "delta_lender" not in data
        assert "delta_passive" not in data

    def test_report_with_per_run_sections(self, tmp_path: Path):
        """Report renders contagion/temporal sections when events exist."""
        _write_synthetic_sweep(tmp_path, "bank")
        html = build_comprehensive_report(tmp_path, "bank")

        # Contagion section should render (we have events)
        assert "Network" in html or "Contagion" in html

    def test_report_missing_csv_raises(self, tmp_path: Path):
        """Report raises when comparison.csv is missing."""
        with pytest.raises(FileNotFoundError):
            build_comprehensive_report(tmp_path, "dealer")

    def test_report_skips_empty_sections(self, tmp_path: Path):
        """Sections with insufficient data are omitted (no error boxes)."""
        _write_synthetic_sweep(tmp_path, "dealer", n_kappas=2)
        html = build_comprehensive_report(tmp_path, "dealer")

        # With only 2 data points, regression (needs 10) should not appear
        assert "OLS Regression" not in html
        # But summary should still be there
        assert "Summary Statistics" in html
