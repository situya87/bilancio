"""Tests for analysis manifest and power planning."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


class TestAnalysisManifestSchema:
    """Verify the analysis manifest JSON is valid."""

    @pytest.fixture
    def manifest_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "scripts" / "analysis_manifest.json"

    def test_manifest_exists(self, manifest_path: Path) -> None:
        assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

    def test_manifest_valid_json(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text())
        assert isinstance(data, dict)

    def test_manifest_required_keys(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text())
        required = {"version", "primary_endpoints", "hypothesis_families", "design"}
        assert required.issubset(data.keys())

    def test_endpoints_have_required_fields(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text())
        for ep in data["primary_endpoints"]:
            for key in ("metric", "mde", "alpha", "power"):
                assert key in ep, f"Endpoint missing {key}"

    def test_design_has_pairing_key(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text())
        assert "pairing_key" in data["design"]


class TestRequiredReplicates:
    """Verify power computation."""

    def test_basic_computation(self) -> None:
        from scipy.stats import norm

        # Manual calculation: mde=0.05, alpha=0.05, power=0.80, var=0.01
        z_a = norm.ppf(0.975)  # ~1.96
        z_b = norm.ppf(0.80)  # ~0.84
        expected = math.ceil((z_a + z_b) ** 2 * 2 * 0.01 / 0.05 ** 2)
        # Import the function from the benchmark script
        import importlib.util
        import sys

        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        spec = importlib.util.spec_from_file_location(
            "scientific_benchmark",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_scientific_comparison_benchmark.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod._compute_required_replicates(0.05, 0.05, 0.80, 0.01)
        assert result == expected

    def test_minimum_two_replicates(self) -> None:
        import importlib.util
        import sys

        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        spec = importlib.util.spec_from_file_location(
            "scientific_benchmark",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_scientific_comparison_benchmark.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # With very large MDE and small variance, should still return >= 2
        result = mod._compute_required_replicates(10.0, 0.05, 0.80, 0.001)
        assert result >= 2
