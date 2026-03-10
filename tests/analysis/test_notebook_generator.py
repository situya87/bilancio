"""Tests for notebook_generator — three levels of alignment testing."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

from bilancio.analysis.notebook_generator import (
    ARM_DIR_MAP,
    COVERED_AGENT_KINDS,
    COVERED_INSTRUMENT_KINDS,
    COVERED_PROFILES,
    DOCUMENTED_PHASE_FLAGS,
    EXCLUDED_AGENT_KINDS,
    HANDLED_SWEEP_TYPES,
    SweepNotebookContext,
    generate_sweep_notebook,
)


# ── Helper ───────────────────────────────────────────────────────────

def _write_synthetic_sweep(
    tmp_path: Path,
    sweep_type: str,
    kappas: list[float] | None = None,
    arms: list[str] | None = None,
) -> Path:
    """Create a minimal sweep directory structure with comparison.csv."""
    if kappas is None:
        kappas = [0.5, 1.0, 2.0]

    # Default arms based on sweep type
    if arms is None:
        if sweep_type == "dealer":
            arms = ["active", "passive"]
        else:
            arms = ["lend", "idle"]

    # Create arm directories
    for arm in arms:
        arm_dir_name = ARM_DIR_MAP.get((sweep_type, arm), arm)
        runs_dir = tmp_path / arm_dir_name / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy run dir
        (runs_dir / "run_001").mkdir()

    # Create comparison.csv
    agg_dir = tmp_path / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    if sweep_type == "dealer":
        fieldnames = [
            "kappa", "concentration", "mu", "outside_mid_ratio",
            "delta_passive", "delta_active", "trading_effect",
            "passive_run_id", "active_run_id",
        ]
    elif sweep_type == "bank":
        fieldnames = [
            "kappa", "concentration", "mu", "outside_mid_ratio",
            "delta_idle", "delta_lend", "bank_lending_effect",
            "idle_run_id", "lend_run_id",
        ]
    else:  # nbfi
        fieldnames = [
            "kappa", "concentration", "mu", "outside_mid_ratio",
            "delta_idle", "delta_lend", "lending_effect",
            "idle_run_id", "lend_run_id",
        ]

    csv_path = agg_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in kappas:
            row = {
                "kappa": str(k),
                "concentration": "1.0",
                "mu": "0.0",
                "outside_mid_ratio": "0.9",
            }
            if sweep_type == "dealer":
                row["delta_passive"] = "0.3"
                row["delta_active"] = "0.2"
                row["trading_effect"] = "0.1"
                row["passive_run_id"] = f"passive_{k}"
                row["active_run_id"] = f"active_{k}"
            elif sweep_type == "bank":
                row["delta_idle"] = "0.3"
                row["delta_lend"] = "0.15"
                row["bank_lending_effect"] = "0.15"
                row["idle_run_id"] = f"idle_{k}"
                row["lend_run_id"] = f"lend_{k}"
            else:
                row["delta_idle"] = "0.3"
                row["delta_lend"] = "0.25"
                row["lending_effect"] = "0.05"
                row["idle_run_id"] = f"idle_{k}"
                row["lend_run_id"] = f"lend_{k}"
            writer.writerow(row)

    return tmp_path


# ═══════════════════════════════════════════════════════════════════════
# Level 1 — Registry alignment (no sweep data, catches codebase drift)
# ═══════════════════════════════════════════════════════════════════════


class TestRegistryAlignment:
    """These tests break when someone adds a new AgentKind, InstrumentKind,
    profile class, or sweep type without updating the notebook generator."""

    def test_all_agent_kinds_covered(self):
        from bilancio.domain.agent import AgentKind

        all_kinds = {kind.value for kind in AgentKind}
        covered = COVERED_AGENT_KINDS | EXCLUDED_AGENT_KINDS
        missing = all_kinds - covered
        assert not missing, (
            f"AgentKind values not in COVERED or EXCLUDED: {missing}. "
            f"Update notebook_generator.py registries."
        )

    def test_no_phantom_agent_kinds(self):
        from bilancio.domain.agent import AgentKind

        all_kinds = {kind.value for kind in AgentKind}
        registered = COVERED_AGENT_KINDS | EXCLUDED_AGENT_KINDS
        phantom = registered - all_kinds
        assert not phantom, (
            f"Registered agent kinds not in AgentKind enum: {phantom}. "
            f"Remove from notebook_generator.py registries."
        )

    def test_all_instrument_kinds_covered(self):
        from bilancio.domain.instruments.base import InstrumentKind

        all_kinds = {kind.value for kind in InstrumentKind}
        missing = all_kinds - COVERED_INSTRUMENT_KINDS
        assert not missing, (
            f"InstrumentKind values not in COVERED: {missing}. "
            f"Update notebook_generator.py COVERED_INSTRUMENT_KINDS."
        )

    def test_no_phantom_instrument_kinds(self):
        from bilancio.domain.instruments.base import InstrumentKind

        all_kinds = {kind.value for kind in InstrumentKind}
        phantom = COVERED_INSTRUMENT_KINDS - all_kinds
        assert not phantom, (
            f"Covered instrument kinds not in InstrumentKind enum: {phantom}. "
            f"Remove from notebook_generator.py COVERED_INSTRUMENT_KINDS."
        )

    def test_all_profile_classes_covered(self):
        from bilancio.decision import profiles

        profile_classes = {
            name for name, obj in inspect.getmembers(profiles, inspect.isclass)
            if hasattr(obj, "__dataclass_fields__") and name.endswith("Profile")
        }
        missing = profile_classes - COVERED_PROFILES
        assert not missing, (
            f"Profile classes not in COVERED_PROFILES: {missing}. "
            f"Update notebook_generator.py COVERED_PROFILES."
        )

    def test_all_sweep_types_handled(self):
        from bilancio.analysis.comprehensive_report import _SWEEP_META

        meta_keys = set(_SWEEP_META.keys())
        missing = meta_keys - HANDLED_SWEEP_TYPES
        assert not missing, (
            f"Sweep types in _SWEEP_META not in HANDLED_SWEEP_TYPES: {missing}. "
            f"Update notebook_generator.py HANDLED_SWEEP_TYPES."
        )

    def test_phase_flags_documented(self):
        from bilancio.engines.simulation import run_day

        sig = inspect.signature(run_day)
        enable_params = {
            name for name in sig.parameters
            if name.startswith("enable_")
        }
        missing = enable_params - DOCUMENTED_PHASE_FLAGS
        assert not missing, (
            f"run_day() enable_* params not in DOCUMENTED_PHASE_FLAGS: {missing}. "
            f"Update notebook_generator.py DOCUMENTED_PHASE_FLAGS."
        )

    def test_arm_dir_map_consistent(self):
        """ARM_DIR_MAP matches _sweep_post._DIR_MAP."""
        # Import the local _DIR_MAP from _sweep_post by inspecting _arm_runs_dir
        from bilancio.ui.cli._sweep_post import _arm_runs_dir

        # Check each entry in our ARM_DIR_MAP produces the right path
        for (st, arm_label), expected_subdir in ARM_DIR_MAP.items():
            result_dir = _arm_runs_dir(Path("/tmp/test"), st, arm_label)
            actual_subdir = result_dir.parent.name  # e.g., "active" from "/tmp/test/active/runs"
            assert actual_subdir == expected_subdir, (
                f"ARM_DIR_MAP[({st!r}, {arm_label!r})] = {expected_subdir!r} "
                f"but _arm_runs_dir gives {actual_subdir!r}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Level 2 — Structural generation (synthetic CSV, parametrized)
# ═══════════════════════════════════════════════════════════════════════


class TestStructuralGeneration:
    """Generate notebooks from synthetic sweep dirs and validate structure."""

    @pytest.mark.parametrize("sweep_type", ["dealer", "bank", "nbfi"])
    def test_generates_valid_ipynb(self, tmp_path, sweep_type):
        sweep_dir = _write_synthetic_sweep(tmp_path, sweep_type)
        nb_path = generate_sweep_notebook(sweep_dir, sweep_type)

        assert nb_path.exists()
        assert nb_path.suffix == ".ipynb"

        # Valid JSON
        nb = json.loads(nb_path.read_text())
        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert len(nb["cells"]) > 0

        # All cells have required keys
        for cell in nb["cells"]:
            assert "cell_type" in cell
            assert "source" in cell
            assert "id" in cell
            assert cell["cell_type"] in ("markdown", "code")

    def test_dealer_notebook_has_dealer_sections(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer")
        nb_path = generate_sweep_notebook(sweep_dir, "dealer")
        content = nb_path.read_text()

        assert "Dealer" in content
        assert "Treynor" in content
        assert "trading_effect" in content

    def test_bank_notebook_has_bank_sections(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "bank")
        nb_path = generate_sweep_notebook(sweep_dir, "bank")
        content = nb_path.read_text()

        assert "Bank" in content
        assert "bank_lending_effect" in content

    def test_bank_notebook_omits_dealer_analysis(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "bank")
        nb_path = generate_sweep_notebook(sweep_dir, "bank")
        content = nb_path.read_text()

        # Bank notebook should not reference trading_effect in code cells
        nb = json.loads(nb_path.read_text())
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        code_text = "\n".join("".join(c["source"]) for c in code_cells)
        assert "trading_effect" not in code_text

    def test_kappa_values_from_sweep_appear(self, tmp_path):
        kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer", kappas=kappas)
        nb_path = generate_sweep_notebook(sweep_dir, "dealer")
        content = nb_path.read_text()

        for k in kappas:
            assert str(k) in content, f"kappa={k} not found in notebook"

    def test_arms_table_matches_sweep(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer", arms=["active", "passive"])
        nb_path = generate_sweep_notebook(sweep_dir, "dealer")
        content = nb_path.read_text()

        assert "Active (Dealer)" in content
        assert "Passive" in content

    def test_profile_fields_in_notebook(self, tmp_path):
        """TraderProfile field names appear in the trader section."""
        import dataclasses
        from bilancio.decision.profiles import TraderProfile

        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer")
        nb_path = generate_sweep_notebook(sweep_dir, "dealer")
        content = nb_path.read_text()

        field_names = [f.name for f in dataclasses.fields(TraderProfile)
                       if not f.name.startswith("_")]
        # At least some fields should appear
        found = [f for f in field_names if f in content]
        assert len(found) >= 3, (
            f"Expected at least 3 TraderProfile fields in notebook, found {found}"
        )

    def test_nbfi_notebook_has_lender_section(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "nbfi")
        nb_path = generate_sweep_notebook(sweep_dir, "nbfi")
        content = nb_path.read_text()

        assert "NBFI" in content
        assert "lending_effect" in content


# ═══════════════════════════════════════════════════════════════════════
# Level 3 — Context discovery
# ═══════════════════════════════════════════════════════════════════════


class TestContextDiscovery:
    """Test SweepNotebookContext.from_sweep_dir()."""

    def test_context_from_sweep_dir(self, tmp_path):
        kappas = [0.3, 0.5, 1.0]
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer", kappas=kappas)
        ctx = SweepNotebookContext.from_sweep_dir(sweep_dir, "dealer")

        assert ctx.sweep_type == "dealer"
        assert ctx.kappas == tuple(kappas)
        assert ctx.concentrations == (1.0,)
        assert ctx.mus == (0.0,)
        assert ctx.outside_mid_ratios == (0.9,)
        assert ctx.n_parameter_combos == 3
        assert ctx.has_dealer is True
        assert ctx.has_bank is False
        assert ctx.has_nbfi is False
        assert "active" in ctx.arms_present
        assert "passive" in ctx.arms_present

    def test_context_missing_summary_json(self, tmp_path):
        """Graceful fallback when summary.json is absent."""
        sweep_dir = _write_synthetic_sweep(tmp_path, "bank")
        # No summary.json — should use defaults
        ctx = SweepNotebookContext.from_sweep_dir(sweep_dir, "bank")
        assert ctx.n_agents == 100  # default
        assert ctx.maturity_days == 10  # default

    def test_context_with_summary_json(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "bank")
        summary = {"n_agents": 50, "maturity_days": 15}
        summary_path = sweep_dir / "aggregate" / "summary.json"
        summary_path.write_text(json.dumps(summary))

        ctx = SweepNotebookContext.from_sweep_dir(sweep_dir, "bank")
        assert ctx.n_agents == 50
        assert ctx.maturity_days == 15

    def test_context_detects_arms_from_dirs(self, tmp_path):
        """Only arms with existing runs/ directories are detected."""
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer", arms=["active"])
        # Only "active" arm exists, not "passive"
        ctx = SweepNotebookContext.from_sweep_dir(sweep_dir, "dealer")
        assert "active" in ctx.arms_present
        assert "passive" not in ctx.arms_present

    def test_context_csv_columns(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer")
        ctx = SweepNotebookContext.from_sweep_dir(sweep_dir, "dealer")
        assert "kappa" in ctx.csv_columns
        assert "trading_effect" in ctx.csv_columns

    def test_invalid_sweep_type_raises(self, tmp_path):
        sweep_dir = _write_synthetic_sweep(tmp_path, "dealer")
        with pytest.raises(ValueError, match="Unknown sweep_type"):
            generate_sweep_notebook(sweep_dir, "invalid_type")
