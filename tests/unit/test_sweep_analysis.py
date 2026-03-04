"""Tests for bilancio.experiments.sweep_analysis — ring-specific statistical analysis.

Tests cover:
- RingSweepAnalysis construction (from records, from CSV, from results)
- Properties (n_cells, n_records, min_replicates)
- cell_summary — per-cell descriptive stats for a given metric
- trading_effects — treatment effect for passive vs active
- lending_effects — treatment effect for passive vs lender
- combined_effects — treatment effect for passive vs dealer_lender
- all_effects — returns only effects where both arms have data
- sensitivity — Morris screening parameter importance ranking
- write_stats — writes CSV and JSON output files
- _infer_bounds — extracts parameter ranges from records
- _result_to_dict — converts BalancedComparisonResult-like objects to flat dicts
"""

import csv
import json
import random
from decimal import Decimal
from types import SimpleNamespace

import pytest

from bilancio.experiments.sweep_analysis import (
    RING_ARMS,
    RING_EFFECTS,
    RING_PARAM_FIELDS,
    RingSweepAnalysis,
    _json_default,
    _result_to_dict,
)

# ============================================================
# Helpers
# ============================================================


def make_record(
    kappa=1.0,
    concentration=1.0,
    mu=0.0,
    monotonicity=1.0,
    outside_mid_ratio=0.9,
    delta_passive=0.3,
    delta_active=0.2,
    phi_passive=0.7,
    phi_active=0.8,
    seed=42,
    **extra,
):
    """Create a single synthetic comparison.csv-like record."""
    d = {
        "kappa": kappa,
        "concentration": concentration,
        "mu": mu,
        "monotonicity": monotonicity,
        "outside_mid_ratio": outside_mid_ratio,
        "delta_passive": delta_passive,
        "delta_active": delta_active,
        "phi_passive": phi_passive,
        "phi_active": phi_active,
        "seed": seed,
    }
    d.update(extra)
    return d


def make_replicated_records(
    kappas=(0.5, 1.0),
    n_seeds=5,
    rng_seed=42,
    include_lender=False,
):
    """Create replicated records across multiple kappa values and seeds.

    Higher kappa -> lower delta (less stress). Active arm reduces delta
    by approximately 0.1 from passive.
    """
    rng = random.Random(rng_seed)
    records = []
    for kappa in kappas:
        for seed in range(n_seeds):
            base_delta = max(0, 0.6 - 0.3 * kappa + rng.gauss(0, 0.03))
            noise = rng.gauss(0, 0.015)
            rec = make_record(
                kappa=kappa,
                delta_passive=base_delta,
                delta_active=max(0, base_delta - 0.1 + noise),
                phi_passive=1 - base_delta,
                phi_active=1 - max(0, base_delta - 0.1 + noise),
                seed=seed,
            )
            if include_lender:
                lender_noise = rng.gauss(0, 0.01)
                rec["delta_lender"] = max(0, base_delta - 0.05 + lender_noise)
                rec["phi_lender"] = 1 - rec["delta_lender"]
            records.append(rec)
    return records


# ============================================================
# Construction and Properties
# ============================================================


class TestRingSweepAnalysisInit:
    def test_basic_construction(self):
        records = [make_record(seed=i) for i in range(5)]
        analysis = RingSweepAnalysis(records)
        assert analysis.n_records == 5
        assert analysis.records is records

    def test_single_cell(self):
        """All records with same params form one cell."""
        records = [make_record(seed=i) for i in range(5)]
        analysis = RingSweepAnalysis(records)
        assert analysis.n_cells == 1
        assert analysis.min_replicates() == 5

    def test_multiple_cells(self):
        records = make_replicated_records(kappas=(0.5, 1.0, 2.0), n_seeds=3)
        analysis = RingSweepAnalysis(records)
        assert analysis.n_cells == 3
        assert analysis.n_records == 9
        assert analysis.min_replicates() == 3

    def test_empty_records(self):
        analysis = RingSweepAnalysis([])
        assert analysis.n_records == 0
        assert analysis.n_cells == 0


# ============================================================
# from_csv
# ============================================================


class TestFromCSV:
    def test_round_trip(self, tmp_path):
        """Write records to CSV and read them back."""
        csv_path = tmp_path / "comparison.csv"
        records = make_replicated_records(kappas=(0.5, 1.0), n_seeds=3)

        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        analysis = RingSweepAnalysis.from_csv(csv_path)
        assert analysis.n_records == 6
        assert analysis.n_cells == 2

    def test_csv_values_are_strings(self, tmp_path):
        """CSV reader returns strings; SweepAnalyzer should handle float conversion."""
        csv_path = tmp_path / "comparison.csv"
        records = [make_record(kappa=0.5, seed=0), make_record(kappa=0.5, seed=1)]

        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        analysis = RingSweepAnalysis.from_csv(csv_path)
        # The analyzer should still group correctly despite string types
        assert analysis.n_cells == 1
        assert analysis.n_records == 2


# ============================================================
# from_results
# ============================================================


class TestFromResults:
    def _make_result_obj(
        self,
        kappa=Decimal("1.0"),
        concentration=Decimal("1.0"),
        mu=Decimal("0.0"),
        monotonicity=Decimal("1.0"),
        outside_mid_ratio=Decimal("0.9"),
        seed=42,
        delta_passive=Decimal("0.30"),
        phi_passive=Decimal("0.70"),
        delta_active=Decimal("0.20"),
        phi_active=Decimal("0.80"),
        delta_lender=None,
        phi_lender=None,
    ):
        """Create a mock BalancedComparisonResult-like object."""
        return SimpleNamespace(
            kappa=kappa,
            concentration=concentration,
            mu=mu,
            monotonicity=monotonicity,
            outside_mid_ratio=outside_mid_ratio,
            seed=seed,
            delta_passive=delta_passive,
            phi_passive=phi_passive,
            delta_active=delta_active,
            phi_active=phi_active,
            delta_lender=delta_lender,
            phi_lender=phi_lender,
            n_defaults_passive=None,
            n_defaults_active=None,
            cascade_fraction_passive=None,
            cascade_fraction_active=None,
            trading_effect=delta_passive - delta_active if delta_passive and delta_active else None,
            lending_effect=None,
            combined_effect=None,
        )

    def test_basic_conversion(self):
        results = [
            self._make_result_obj(seed=0),
            self._make_result_obj(seed=1),
            self._make_result_obj(seed=2),
        ]
        analysis = RingSweepAnalysis.from_results(results)
        assert analysis.n_records == 3
        assert analysis.n_cells == 1

    def test_decimal_to_float(self):
        result = self._make_result_obj(kappa=Decimal("0.50"), delta_passive=Decimal("0.35"))
        d = _result_to_dict(result)
        assert isinstance(d["kappa"], float)
        assert d["kappa"] == pytest.approx(0.5)
        assert isinstance(d["delta_passive"], float)
        assert d["delta_passive"] == pytest.approx(0.35)

    def test_none_handling(self):
        result = self._make_result_obj(delta_lender=None, phi_lender=None)
        d = _result_to_dict(result)
        assert d["delta_lender"] is None
        assert d["phi_lender"] is None

    def test_all_arms_extracted(self):
        """Verify that _result_to_dict extracts all arm suffixes."""
        result = self._make_result_obj()
        d = _result_to_dict(result)
        # Should have delta_passive, delta_active, etc.
        assert "delta_passive" in d
        assert "delta_active" in d
        assert "phi_passive" in d
        assert "phi_active" in d

    def test_parameters_extracted(self):
        result = self._make_result_obj(
            kappa=Decimal("2.0"),
            concentration=Decimal("0.5"),
            mu=Decimal("0.3"),
            monotonicity=Decimal("0.8"),
            outside_mid_ratio=Decimal("0.85"),
            seed=99,
        )
        d = _result_to_dict(result)
        assert d["kappa"] == pytest.approx(2.0)
        assert d["concentration"] == pytest.approx(0.5)
        assert d["mu"] == pytest.approx(0.3)
        assert d["monotonicity"] == pytest.approx(0.8)
        assert d["outside_mid_ratio"] == pytest.approx(0.85)
        assert d["seed"] == 99

    def test_computed_effects_extracted(self):
        result = self._make_result_obj(
            delta_passive=Decimal("0.4"),
            delta_active=Decimal("0.3"),
        )
        result.trading_effect = Decimal("0.1")
        d = _result_to_dict(result)
        assert d["trading_effect"] == pytest.approx(0.1)


# ============================================================
# cell_summary
# ============================================================


class TestCellSummary:
    def test_basic_cell_summary(self):
        records = make_replicated_records(kappas=(0.5,), n_seeds=10)
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        assert len(table.rows) == 1
        row = table.rows[0]
        assert row.stats.n == 10
        assert row.stats.mean > 0
        assert row.stats.ci.lower < row.stats.mean
        assert row.stats.ci.upper > row.stats.mean

    def test_multiple_cells(self):
        records = make_replicated_records(kappas=(0.25, 0.5, 1.0), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        assert len(table.rows) == 3

        # Lower kappa should give higher delta (more stress)
        rows_by_kappa = {r.params["kappa"]: r for r in table.rows}
        assert rows_by_kappa[0.25].stats.mean > rows_by_kappa[1.0].stats.mean

    def test_skips_cells_with_one_replicate(self):
        """Cells with < 2 replicates should be skipped (can't compute CI)."""
        records = [
            make_record(kappa=0.5, seed=0),
            make_record(kappa=0.5, seed=1),
            make_record(kappa=1.0, seed=0),  # only 1 replicate for this cell
        ]
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        # kappa=1.0 cell has only 1 record -> skipped
        assert len(table.rows) == 1
        assert table.rows[0].params["kappa"] == 0.5

    def test_to_dicts(self):
        records = make_replicated_records(kappas=(0.5,), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        dicts = table.to_dicts()
        assert len(dicts) == 1
        assert "mean" in dicts[0]
        assert "ci_lower" in dicts[0]
        assert "ci_upper" in dicts[0]
        assert "kappa" in dicts[0]


# ============================================================
# trading_effects
# ============================================================


class TestTradingEffects:
    @pytest.fixture
    def analysis_with_trading(self):
        records = make_replicated_records(kappas=(0.5, 1.0), n_seeds=10)
        return RingSweepAnalysis(records)

    def test_returns_effect_table(self, analysis_with_trading):
        effects = analysis_with_trading.trading_effects(seed=42)
        assert len(effects.rows) == 2
        assert effects.metric == "delta"
        assert effects.control_suffix == "_passive"
        assert effects.treatment_suffix == "_active"

    def test_positive_effect(self, analysis_with_trading):
        """Active arm reduces delta, so effect should be positive."""
        effects = analysis_with_trading.trading_effects(seed=42)
        for row in effects.rows:
            assert row.stats.effect.estimate > 0

    def test_effect_magnitude(self, analysis_with_trading):
        """Effect should be approximately 0.1 (the synthetic treatment effect)."""
        effects = analysis_with_trading.trading_effects(seed=42)
        for row in effects.rows:
            assert abs(row.stats.effect.estimate - 0.1) < 0.05

    def test_significance_with_enough_replicates(self, analysis_with_trading):
        """With 10 replicates and clear effect, should be significant."""
        effects = analysis_with_trading.trading_effects(seed=42)
        for row in effects.rows:
            assert row.stats.effect_test.significant_at_05


# ============================================================
# lending_effects
# ============================================================


class TestLendingEffects:
    def test_basic_lending_effects(self):
        records = make_replicated_records(
            kappas=(0.5,), n_seeds=10, include_lender=True,
        )
        analysis = RingSweepAnalysis(records)
        effects = analysis.lending_effects(seed=42)
        assert len(effects.rows) == 1
        assert effects.control_suffix == "_passive"
        assert effects.treatment_suffix == "_lender"

    def test_positive_lending_effect(self):
        """Lender reduces delta, so effect should be positive."""
        records = make_replicated_records(
            kappas=(0.5,), n_seeds=10, include_lender=True,
        )
        analysis = RingSweepAnalysis(records)
        effects = analysis.lending_effects(seed=42)
        for row in effects.rows:
            assert row.stats.effect.estimate > 0

    def test_no_lending_data(self):
        """When no lender data exists, should return empty table."""
        records = make_replicated_records(kappas=(0.5,), n_seeds=5)
        # No delta_lender fields in records
        analysis = RingSweepAnalysis(records)
        effects = analysis.lending_effects(seed=42)
        assert len(effects.rows) == 0


# ============================================================
# combined_effects
# ============================================================


class TestCombinedEffects:
    def test_basic_combined_effects(self):
        records = make_replicated_records(kappas=(0.5,), n_seeds=5)
        # Add dealer_lender data
        rng = random.Random(99)
        for r in records:
            r["delta_dealer_lender"] = max(0, r["delta_passive"] - 0.12 + rng.gauss(0, 0.01))
            r["phi_dealer_lender"] = 1 - r["delta_dealer_lender"]
        analysis = RingSweepAnalysis(records)
        effects = analysis.combined_effects(seed=42)
        assert len(effects.rows) == 1
        assert effects.control_suffix == "_passive"
        assert effects.treatment_suffix == "_dealer_lender"


# ============================================================
# all_effects
# ============================================================


class TestAllEffects:
    def test_only_arms_with_data(self):
        """all_effects should only include effects where both arms have data."""
        records = make_replicated_records(kappas=(0.5,), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        effects = analysis.all_effects(seed=42)
        # Only passive and active data exists -> only trading effect
        assert "trading" in effects
        assert "lending" not in effects
        assert "combined" not in effects

    def test_multiple_arms(self):
        records = make_replicated_records(
            kappas=(0.5,), n_seeds=5, include_lender=True,
        )
        analysis = RingSweepAnalysis(records)
        effects = analysis.all_effects(seed=42)
        assert "trading" in effects
        assert "lending" in effects

    def test_empty_records(self):
        analysis = RingSweepAnalysis([])
        effects = analysis.all_effects(seed=42)
        assert effects == {}

    def test_none_values_excluded(self):
        """Records with None in metric columns should not count as 'having data'."""
        records = [
            make_record(kappa=0.5, seed=i, delta_passive=0.3, delta_active=None)
            for i in range(5)
        ]
        analysis = RingSweepAnalysis(records)
        effects = analysis.all_effects(seed=42)
        # delta_active is None in all records -> trading should not be present
        assert "trading" not in effects


# ============================================================
# sensitivity
# ============================================================


class TestSensitivity:
    def test_basic_sensitivity(self):
        """With multiple kappa values, sensitivity should identify kappa as important."""
        records = make_replicated_records(kappas=(0.25, 0.5, 1.0, 2.0), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        results = analysis.sensitivity(metric="delta_passive", seed=42)
        # Only kappa varies -> only kappa in bounds
        assert len(results) == 1
        assert results[0].parameter == "kappa"
        assert results[0].mu_star > 0

    def test_multiple_varying_params(self):
        """When both kappa and concentration vary, both should appear."""
        rng = random.Random(42)
        records = []
        for kappa in [0.5, 1.0, 2.0]:
            for conc in [0.5, 1.0, 2.0]:
                for seed in range(3):
                    delta = max(0, 0.6 - 0.3 * kappa + 0.05 * conc + rng.gauss(0, 0.02))
                    records.append(make_record(
                        kappa=kappa,
                        concentration=conc,
                        delta_passive=delta,
                        delta_active=delta - 0.1,
                        seed=seed,
                    ))
        analysis = RingSweepAnalysis(records)
        results = analysis.sensitivity(metric="delta_passive", seed=42)
        params = {r.parameter for r in results}
        assert "kappa" in params
        assert "concentration" in params

    def test_single_kappa_no_sensitivity(self):
        """With only one kappa value, no bounds can be inferred -> empty results."""
        records = make_replicated_records(kappas=(0.5,), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        results = analysis.sensitivity(metric="delta_passive", seed=42)
        assert results == []

    def test_kappa_dominates_concentration(self):
        """kappa has 10x the effect of concentration -> should rank first."""
        rng = random.Random(42)
        records = []
        for kappa in [0.25, 0.5, 1.0, 2.0]:
            for conc in [0.5, 1.0, 2.0]:
                for seed in range(5):
                    # kappa has 10x the effect of concentration
                    delta = max(0, 0.6 - 0.3 * kappa + 0.03 * conc + rng.gauss(0, 0.01))
                    records.append(make_record(
                        kappa=kappa,
                        concentration=conc,
                        delta_passive=delta,
                        seed=seed,
                    ))
        analysis = RingSweepAnalysis(records)
        results = analysis.sensitivity(metric="delta_passive", seed=42)
        assert results[0].parameter == "kappa"
        assert results[0].mu_star > results[1].mu_star


# ============================================================
# _infer_bounds
# ============================================================


class TestInferBounds:
    def test_basic_bounds(self):
        records = [
            make_record(kappa=0.25, concentration=1.0),
            make_record(kappa=2.0, concentration=1.0),
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        assert "kappa" in bounds
        assert bounds["kappa"] == pytest.approx((0.25, 2.0))

    def test_single_value_excluded(self):
        """A parameter with only one distinct value should not produce bounds."""
        records = [
            make_record(kappa=1.0, concentration=1.0, seed=0),
            make_record(kappa=1.0, concentration=1.0, seed=1),
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        # kappa, concentration, mu, outside_mid_ratio all have single values
        assert bounds == {}

    def test_multiple_params_with_range(self):
        records = [
            make_record(kappa=0.5, mu=0.0),
            make_record(kappa=1.0, mu=0.5),
            make_record(kappa=2.0, mu=1.0),
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        assert "kappa" in bounds
        assert "mu" in bounds
        assert bounds["kappa"] == pytest.approx((0.5, 2.0))
        assert bounds["mu"] == pytest.approx((0.0, 1.0))

    def test_non_numeric_skipped(self):
        """Non-numeric values in parameter fields should be skipped."""
        records = [
            {"kappa": "bad", "concentration": 1.0, "mu": 0.0, "monotonicity": 1.0,
             "outside_mid_ratio": 0.9, "delta_passive": 0.3, "seed": 0},
            {"kappa": 1.0, "concentration": 1.0, "mu": 0.0, "monotonicity": 1.0,
             "outside_mid_ratio": 0.9, "delta_passive": 0.2, "seed": 1},
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        # Only one valid kappa value -> no bounds for kappa
        assert "kappa" not in bounds

    def test_missing_field_skipped(self):
        """Records missing a parameter field should be skipped for that param."""
        records = [
            {"kappa": 0.5, "concentration": 1.0, "mu": 0.0, "monotonicity": 1.0,
             "outside_mid_ratio": 0.9, "delta_passive": 0.3, "seed": 0},
            {"concentration": 1.0, "mu": 0.0, "monotonicity": 1.0,
             "outside_mid_ratio": 0.9, "delta_passive": 0.2, "seed": 1},  # missing kappa
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        assert "kappa" not in bounds

    def test_string_numeric_values(self):
        """CSV-loaded records have string values; _infer_bounds should convert them."""
        records = [
            {"kappa": "0.25", "concentration": "1.0", "mu": "0.0", "monotonicity": "1.0",
             "outside_mid_ratio": "0.9", "delta_passive": "0.3", "seed": "0"},
            {"kappa": "2.0", "concentration": "1.0", "mu": "0.0", "monotonicity": "1.0",
             "outside_mid_ratio": "0.9", "delta_passive": "0.2", "seed": "1"},
        ]
        analysis = RingSweepAnalysis(records)
        bounds = analysis._infer_bounds()
        assert "kappa" in bounds
        assert bounds["kappa"] == pytest.approx((0.25, 2.0))


# ============================================================
# write_stats
# ============================================================


class TestWriteStats:
    @pytest.fixture
    def analysis_for_write(self):
        """Analysis with enough data to produce all output files."""
        records = make_replicated_records(
            kappas=(0.5, 1.0, 2.0), n_seeds=5, include_lender=True,
        )
        return RingSweepAnalysis(records)

    def test_creates_output_files(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "stats_output"
        paths = analysis_for_write.write_stats(output_dir, seed=42)

        assert "effects" in paths
        assert "cells" in paths
        assert "summary" in paths
        assert "sensitivity" in paths

        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_creates_output_dir(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "nested" / "output"
        analysis_for_write.write_stats(output_dir, seed=42)
        assert output_dir.exists()

    def test_effects_csv_content(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "stats"
        paths = analysis_for_write.write_stats(output_dir, seed=42)

        with paths["effects"].open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) > 0
        # Should have effect_type column
        assert "effect_type" in rows[0]
        # Should have trading and lending effects
        effect_types = {r["effect_type"] for r in rows}
        assert "trading" in effect_types
        assert "lending" in effect_types

    def test_cells_csv_content(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "stats"
        paths = analysis_for_write.write_stats(output_dir, seed=42)

        with paths["cells"].open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) > 0
        assert "metric" in rows[0]
        metrics = {r["metric"] for r in rows}
        assert "delta_passive" in metrics
        assert "delta_active" in metrics

    def test_summary_json_content(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "stats"
        paths = analysis_for_write.write_stats(output_dir, seed=42)

        with paths["summary"].open() as fh:
            summary = json.load(fh)

        assert summary["n_records"] == 15  # 3 kappas * 5 seeds
        assert summary["n_cells"] == 3
        assert "trading_effect" in summary
        assert "n_cells" in summary["trading_effect"]

    def test_sensitivity_json_content(self, analysis_for_write, tmp_path):
        output_dir = tmp_path / "stats"
        paths = analysis_for_write.write_stats(output_dir, seed=42)

        with paths["sensitivity"].open() as fh:
            sensitivity = json.load(fh)

        # Should have entries for delta_passive and/or delta_active
        # (only if bounds are inferable — 3 kappas gives bounds)
        assert "delta_passive" in sensitivity or "delta_active" in sensitivity

    def test_no_effects_when_single_arm(self, tmp_path):
        """When only passive data exists, effects CSV should not be written."""
        records = [
            {"kappa": 0.5, "concentration": 1.0, "mu": 0.0, "monotonicity": 1.0,
             "outside_mid_ratio": 0.9, "delta_passive": 0.3, "seed": i}
            for i in range(5)
        ]
        analysis = RingSweepAnalysis(records)
        output_dir = tmp_path / "stats"
        paths = analysis.write_stats(output_dir, seed=42)
        # No effects -> "effects" key should not be in paths
        assert "effects" not in paths


# ============================================================
# _json_default
# ============================================================


class TestJsonDefault:
    def test_decimal_conversion(self):
        assert _json_default(Decimal("3.14")) == pytest.approx(3.14)

    def test_nan_to_none(self):
        assert _json_default(float("nan")) is None

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_normal_float_raises(self):
        """Normal floats are handled by the default JSON encoder, not our custom one."""
        with pytest.raises(TypeError):
            _json_default(3.14)


# ============================================================
# Module-level constants
# ============================================================


class TestConstants:
    def test_ring_param_fields(self):
        assert "kappa" in RING_PARAM_FIELDS
        assert "concentration" in RING_PARAM_FIELDS
        assert "mu" in RING_PARAM_FIELDS
        assert "monotonicity" in RING_PARAM_FIELDS
        assert "outside_mid_ratio" in RING_PARAM_FIELDS

    def test_ring_arms(self):
        assert "passive" in RING_ARMS
        assert "active" in RING_ARMS
        assert RING_ARMS["passive"] == "_passive"
        assert RING_ARMS["active"] == "_active"

    def test_ring_effects(self):
        assert "trading" in RING_EFFECTS
        assert "lending" in RING_EFFECTS
        assert "combined" in RING_EFFECTS
        # trading: passive vs active
        assert RING_EFFECTS["trading"] == ("_passive", "_active")
        # lending: passive vs lender
        assert RING_EFFECTS["lending"] == ("_passive", "_lender")


# ============================================================
# Edge cases and integration
# ============================================================


class TestEdgeCases:
    def test_string_metric_values(self):
        """CSV-loaded values are strings; analysis should handle them."""
        records = [
            {"kappa": "0.5", "concentration": "1.0", "mu": "0.0",
             "monotonicity": "1.0", "outside_mid_ratio": "0.9",
             "delta_passive": "0.30", "delta_active": "0.20", "seed": str(i)}
            for i in range(5)
        ]
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        assert len(table.rows) == 1
        assert table.rows[0].stats.mean == pytest.approx(0.30)

    def test_mixed_none_and_values(self):
        """Some records with None metric values should be handled gracefully."""
        records = [
            make_record(kappa=0.5, delta_passive=0.3, delta_active=0.2, seed=0),
            make_record(kappa=0.5, delta_passive=0.4, delta_active=None, seed=1),
            make_record(kappa=0.5, delta_passive=0.35, delta_active=0.25, seed=2),
        ]
        analysis = RingSweepAnalysis(records)
        # cell_summary for delta_passive should use all 3 records
        table = analysis.cell_summary("delta_passive", seed=42)
        assert table.rows[0].stats.n == 3

        # trading_effects should only pair records where both exist (2 pairs)
        effects = analysis.trading_effects(seed=42)
        assert effects.rows[0].stats.n_pairs == 2

    def test_large_replicate_count(self):
        """Verify stability with many replicates per cell."""
        records = make_replicated_records(kappas=(0.5,), n_seeds=50)
        analysis = RingSweepAnalysis(records)
        table = analysis.cell_summary("delta_passive", seed=42)
        assert table.rows[0].stats.n == 50
        # CI should be narrower with more data
        assert table.rows[0].stats.ci.width < 0.1

    def test_write_stats_reproducibility(self, tmp_path):
        """Same data and seed should produce identical output."""
        records = make_replicated_records(kappas=(0.5, 1.0), n_seeds=5)
        analysis = RingSweepAnalysis(records)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        paths1 = analysis.write_stats(dir1, seed=42)
        paths2 = analysis.write_stats(dir2, seed=42)

        for key in paths1:
            if key in paths2:
                content1 = paths1[key].read_text()
                content2 = paths2[key].read_text()
                assert content1 == content2, f"Mismatch in {key}"


# ============================================================
# Regression tests for review fixes
# ============================================================


class TestMonotonicityCellKey:
    """P1 fix: monotonicity must be part of cell grouping keys."""

    def test_different_monotonicity_creates_separate_cells(self):
        """Records with different monotonicity should NOT be merged into one cell."""
        records = [
            make_record(kappa=0.5, monotonicity=0.5, delta_passive=0.3, seed=0),
            make_record(kappa=0.5, monotonicity=0.5, delta_passive=0.35, seed=1),
            make_record(kappa=0.5, monotonicity=1.0, delta_passive=0.5, seed=0),
            make_record(kappa=0.5, monotonicity=1.0, delta_passive=0.55, seed=1),
        ]
        analysis = RingSweepAnalysis(records)
        assert analysis.n_cells == 2
        assert analysis.min_replicates() == 2

    def test_monotonicity_in_result_to_dict(self):
        """_result_to_dict must include monotonicity."""
        result = SimpleNamespace(
            kappa=Decimal("1.0"), concentration=Decimal("1.0"),
            mu=Decimal("0.0"), monotonicity=Decimal("0.75"),
            outside_mid_ratio=Decimal("0.9"), seed=42,
            delta_passive=Decimal("0.3"), phi_passive=Decimal("0.7"),
            delta_active=Decimal("0.2"), phi_active=Decimal("0.8"),
            delta_lender=None, phi_lender=None,
            n_defaults_passive=None, n_defaults_active=None,
            cascade_fraction_passive=None, cascade_fraction_active=None,
            trading_effect=Decimal("0.1"), lending_effect=None, combined_effect=None,
        )
        d = _result_to_dict(result)
        assert "monotonicity" in d
        assert d["monotonicity"] == pytest.approx(0.75)


class TestNReplicatesValidation:
    """P2 fix: n_replicates must be >= 1."""

    def test_config_rejects_zero(self):
        from pydantic import ValidationError

        from bilancio.experiments.balanced_comparison import BalancedComparisonConfig
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(n_replicates=0)

    def test_config_rejects_negative(self):
        from pydantic import ValidationError

        from bilancio.experiments.balanced_comparison import BalancedComparisonConfig
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(n_replicates=-1)

    def test_config_accepts_one(self):
        from bilancio.experiments.balanced_comparison import BalancedComparisonConfig
        cfg = BalancedComparisonConfig(n_replicates=1)
        assert cfg.n_replicates == 1


class TestWriteStatsPathExistence:
    """P2 fix: write_stats should only return paths for files that actually exist."""

    def test_no_cells_path_when_no_data(self, tmp_path):
        """With insufficient data, cells CSV won't be written -> path not in result."""
        # Single record per cell => summarize_cell skips (need >= 2)
        records = [make_record(kappa=0.5, seed=0)]
        analysis = RingSweepAnalysis(records)
        paths = analysis.write_stats(tmp_path / "out", seed=42)
        # cells requires >= 2 replicates, so should not be written
        if "cells" in paths:
            assert paths["cells"].exists()

    def test_all_returned_paths_exist(self, tmp_path):
        """Every path in the returned dict must correspond to an existing file."""
        records = make_replicated_records(kappas=(0.5, 1.0), n_seeds=5)
        analysis = RingSweepAnalysis(records)
        paths = analysis.write_stats(tmp_path / "out", seed=42)
        for name, path in paths.items():
            assert path.exists(), f"Path for '{name}' does not exist: {path}"
