"""Tests for topology support in NBFIComparisonRunner (Plan 055 extension)."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from bilancio.experiments.nbfi_comparison import (
    NBFIComparisonConfig,
    NBFIComparisonResult,
    NBFIComparisonRunner,
)


class TestNBFITopologyLabel:
    """Test _topology_label static method."""

    def test_ring(self):
        assert NBFIComparisonRunner._topology_label({"type": "ring"}) == "ring"

    def test_k_regular(self):
        assert NBFIComparisonRunner._topology_label({"type": "k_regular", "degree": 3}) == "k_regular_3"

    def test_k_regular_default_degree(self):
        assert NBFIComparisonRunner._topology_label({"type": "k_regular"}) == "k_regular_2"

    def test_erdos_renyi(self):
        assert NBFIComparisonRunner._topology_label({"type": "erdos_renyi", "edge_prob": 0.2}) == "erdos_renyi_0.2"

    def test_erdos_renyi_default(self):
        assert NBFIComparisonRunner._topology_label({"type": "erdos_renyi"}) == "erdos_renyi_0.1"

    def test_unknown_type_passthrough(self):
        assert NBFIComparisonRunner._topology_label({"type": "custom"}) == "custom"


class TestNBFIMakeKeyWithTopology:
    """Test _make_key includes topology."""

    def test_default_ring(self, tmp_path):
        cfg = NBFIComparisonConfig(n_agents=5, maturity_days=3, kappas=[Decimal("1")])
        runner = NBFIComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"))
        assert key == ("1", "1", "0", "0", "0.90", "ring")

    def test_explicit_topology(self, tmp_path):
        cfg = NBFIComparisonConfig(n_agents=5, maturity_days=3, kappas=[Decimal("1")])
        runner = NBFIComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(
            Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"),
            topology="k_regular_4",
        )
        assert key == ("1", "1", "0", "0", "0.90", "k_regular_4")

    def test_keys_are_hashable(self, tmp_path):
        cfg = NBFIComparisonConfig(n_agents=5, maturity_days=3, kappas=[Decimal("1")])
        runner = NBFIComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        k1 = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"))
        k2 = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"), topology="k_regular_4")
        d = {k1: 1, k2: 2}
        assert d[k1] == 1
        assert d[k2] == 2


class TestNBFIComparisonFieldsIncludeTopology:
    """Test that COMPARISON_FIELDS includes topology."""

    def test_topology_in_fields(self):
        assert "topology" in NBFIComparisonRunner.COMPARISON_FIELDS

    def test_topology_after_seed(self):
        fields = NBFIComparisonRunner.COMPARISON_FIELDS
        seed_idx = fields.index("seed")
        topo_idx = fields.index("topology")
        assert topo_idx == seed_idx + 1


class TestNBFIComparisonConfigDefaultTopology:
    """Test NBFIComparisonConfig defaults."""

    def test_default_topologies(self):
        cfg = NBFIComparisonConfig()
        assert cfg.topologies == [{"type": "ring"}]

    def test_custom_topologies(self):
        cfg = NBFIComparisonConfig(topologies=[{"type": "ring"}, {"type": "k_regular", "degree": 4}])
        assert len(cfg.topologies) == 2


class TestNBFIComparisonResultTopology:
    """Test NBFIComparisonResult has topology field."""

    def test_default_topology(self):
        result = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.1"), phi_idle=Decimal("0.9"),
            idle_run_id="idle_1", idle_status="completed",
            delta_lend=Decimal("0.05"), phi_lend=Decimal("0.95"),
            lend_run_id="lend_1", lend_status="completed",
        )
        assert result.topology == "ring"

    def test_custom_topology(self):
        result = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.1"), phi_idle=Decimal("0.9"),
            idle_run_id="idle_1", idle_status="completed",
            delta_lend=Decimal("0.05"), phi_lend=Decimal("0.95"),
            lend_run_id="lend_1", lend_status="completed",
            topology="k_regular_4",
        )
        assert result.topology == "k_regular_4"


class TestNBFIResumeSeedPairing:
    """P1 regression: resumed runs must reuse the same seed across topologies."""

    def _write_partial_csv(self, path: Path, seed: int = 42, topology: str = "ring") -> None:
        """Write a minimal comparison.csv with one completed row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=NBFIComparisonRunner.COMPARISON_FIELDS)
            writer.writeheader()
            row = {f: "" for f in NBFIComparisonRunner.COMPARISON_FIELDS}
            row.update({
                "kappa": "1", "concentration": "1", "mu": "0",
                "monotonicity": "0", "seed": str(seed),
                "topology": topology, "outside_mid_ratio": "0.90",
                "delta_idle": "0.1", "delta_lend": "0.05",
                "idle_run_id": "i1", "idle_status": "completed",
                "lend_run_id": "l1", "lend_status": "completed",
            })
            writer.writerow(row)

    def test_resume_reuses_seed_for_missing_topology(self, tmp_path):
        """If ring completed with seed=42, resuming k_regular must also use seed=42."""
        cfg = NBFIComparisonConfig(
            n_agents=5, maturity_days=3,
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
            topologies=[{"type": "ring"}, {"type": "k_regular", "degree": 4}],
        )
        # Write partial CSV: ring done for seed 42
        agg_dir = tmp_path / "aggregate"
        self._write_partial_csv(agg_dir / "comparison.csv", seed=42, topology="ring")

        runner = NBFIComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)

        # Verify the seed was recorded in _cell_seeds
        cell_key = ("1", "1", "0", "0", "0.90")
        assert cell_key in runner._cell_seeds
        assert 42 in runner._cell_seeds[cell_key]

        # ring should be marked as completed, k_regular_4 should not
        ring_key = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"), topology="ring")
        kr_key = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"), topology="k_regular_4")
        assert runner._completed_counts.get(ring_key, 0) >= 1
        assert runner._completed_counts.get(kr_key, 0) == 0
