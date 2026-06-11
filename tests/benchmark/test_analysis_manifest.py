"""Tests for analysis manifest and power planning."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def _load_scientific_benchmark_module():
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "scientific_benchmark",
        repo_root / "scripts" / "run_scientific_comparison_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        mod = _load_scientific_benchmark_module()
        result = mod._compute_required_replicates(0.05, 0.05, 0.80, 0.01)
        assert result == expected

    def test_minimum_two_replicates(self) -> None:
        mod = _load_scientific_benchmark_module()
        # With very large MDE and small variance, should still return >= 2
        result = mod._compute_required_replicates(10.0, 0.05, 0.80, 0.001)
        assert result >= 2


class TestBenchmarkReportProvenance:
    """Verify scientific benchmark reports expose manifest and power assumptions."""

    @pytest.fixture
    def manifest(self) -> dict:
        return {
            "version": "1.0",
            "primary_endpoints": [
                {
                    "metric": "delta_total",
                    "label": "Default rate",
                    "direction": "lower_is_better",
                    "mde": 0.05,
                    "alpha": 0.05,
                    "power": 0.80,
                },
                {
                    "metric": "phi_total",
                    "label": "Clearing rate",
                    "direction": "higher_is_better",
                    "mde": 0.05,
                    "alpha": 0.05,
                    "power": 0.80,
                },
            ],
            "hypothesis_families": [
                {
                    "name": "trading_effect",
                    "endpoints": ["delta_total", "phi_total"],
                    "multiple_testing": "benjamini_hochberg",
                    "alpha": 0.05,
                }
            ],
            "design": {
                "type": "paired",
                "control": "passive",
                "treatment": "active",
                "pairing_key": ["kappa", "concentration", "mu", "seed"],
            },
        }

    @pytest.fixture
    def records(self) -> list[dict]:
        return [
            {
                "delta_passive": 0.40,
                "delta_active": 0.35,
                "phi_passive": 0.10,
                "phi_active": 0.15,
            },
            {
                "delta_passive": 0.45,
                "delta_active": 0.38,
                "phi_passive": 0.12,
                "phi_active": 0.16,
            },
            {
                "delta_passive": 0.30,
                "delta_active": 0.25,
                "phi_passive": 0.20,
                "phi_active": 0.26,
            },
        ]

    def test_power_plan_covers_every_primary_endpoint(
        self, manifest: dict, records: list[dict]
    ) -> None:
        mod = _load_scientific_benchmark_module()
        plan = mod._build_power_plan(manifest, records, min_replicates=4)

        assert {endpoint["metric"] for endpoint in plan["endpoints"]} == {
            "delta_total",
            "phi_total",
        }
        for endpoint in plan["endpoints"]:
            assert endpoint["mde"] == 0.05
            assert endpoint["alpha"] == 0.05
            assert endpoint["power"] == 0.80
            assert endpoint["required_replicates"] >= 2
            assert endpoint["available"] is True

    def test_power_plan_marks_unknown_manifest_endpoint_unavailable(
        self, manifest: dict, records: list[dict]
    ) -> None:
        mod = _load_scientific_benchmark_module()
        manifest["primary_endpoints"].append(
            {"metric": "missing_metric", "mde": 0.05, "alpha": 0.05, "power": 0.80}
        )

        plan = mod._build_power_plan(manifest, records, min_replicates=4)
        missing = next(
            endpoint for endpoint in plan["endpoints"] if endpoint["metric"] == "missing_metric"
        )

        assert plan["valid"] is False
        assert missing["available"] is False
        assert missing["passes"] is False

    def test_zero_observed_variance_uses_minimum_replicate_floor(self) -> None:
        mod = _load_scientific_benchmark_module()

        assert mod._endpoint_variance([0.05, 0.05, 0.05]) == 0
        assert mod._compute_required_replicates(
            mde=0.05,
            alpha=0.05,
            power=0.80,
            variance=0,
        ) == 2

    def test_markdown_lines_include_manifest_and_power_contract(
        self, manifest: dict, records: list[dict], tmp_path: Path
    ) -> None:
        mod = _load_scientific_benchmark_module()
        plan = mod._build_power_plan(manifest, records, min_replicates=4)

        manifest_lines = mod._manifest_detail_lines(manifest, tmp_path / "manifest.json")
        power_lines = mod._power_plan_detail_lines(plan)

        manifest_text = "\n".join(manifest_lines)
        power_text = "\n".join(power_lines)
        assert "primary_endpoint=delta_total mde=0.05 alpha=0.05 power=0.8" in manifest_text
        assert "hypothesis_family=trading_effect" in manifest_text
        assert "multiple_testing=benjamini_hochberg" in manifest_text
        assert "endpoint=phi_total mde=0.05 alpha=0.05 power=0.8" in power_text
