"""Comprehensive tests for bilancio.stats — general statistical infrastructure.

Tests cover:
- Bootstrap confidence intervals (correctness, edge cases, reproducibility)
- Paired significance tests (Wilcoxon, t-test, known distributions)
- Effect size (Cohen's d, paired and independent)
- Cell-level statistics (summarize_cell, summarize_paired_cell)
- Morris sensitivity screening (known linear/nonlinear models)
- SweepAnalyzer integration (grouping, tables, treatment effects)
"""

import math
import pytest

from bilancio.stats import (
    ConfidenceInterval,
    TestResult,
    CellStats,
    PairedCellStats,
    MorrisResult,
    bootstrap_ci,
    paired_wilcoxon,
    paired_t_test,
    cohens_d,
    cohens_d_paired,
    summarize_cell,
    summarize_paired_cell,
    morris_screening,
    SweepAnalyzer,
)


# ============================================================
# Bootstrap CI
# ============================================================


class TestBootstrapCI:
    def test_basic_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci(data, seed=42)
        assert ci.estimate == pytest.approx(3.0)
        assert ci.lower < ci.estimate
        assert ci.upper > ci.estimate
        assert ci.confidence == 0.95

    def test_tight_data_gives_narrow_ci(self):
        data = [1.00, 1.01, 1.02, 0.99, 0.98]
        ci = bootstrap_ci(data, seed=42)
        assert ci.width < 0.1

    def test_wide_data_gives_wide_ci(self):
        data = [0.0, 10.0, 20.0, 30.0, 40.0]
        ci = bootstrap_ci(data, seed=42)
        assert ci.width > 5.0

    def test_reproducibility(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_ci(data, seed=123)
        ci2 = bootstrap_ci(data, seed=123)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_different_seeds_differ(self):
        # Use continuous (non-integer) data with enough spread so that
        # bootstrap percentiles are unlikely to land on the same values.
        import random
        rng = random.Random(7)
        data = [rng.gauss(50, 15) for _ in range(50)]
        ci1 = bootstrap_ci(data, seed=1)
        ci2 = bootstrap_ci(data, seed=999)
        # With 50 continuous data points, the bootstrap CIs should differ
        assert ci1.lower != ci2.lower or ci1.upper != ci2.upper

    def test_custom_statistic(self):
        # Use a larger sample so that the median bootstrap is genuinely
        # tighter than the mean bootstrap in the presence of outliers.
        import random
        rng = random.Random(42)
        data = [rng.gauss(10, 1) for _ in range(30)]
        data.extend([200.0, 300.0, 500.0])  # add outliers
        ci_mean = bootstrap_ci(data, seed=42)
        ci_median = bootstrap_ci(
            data, statistic=lambda x: sorted(x)[len(x) // 2], seed=42
        )
        # Median should be more robust to outliers
        assert ci_median.width < ci_mean.width

    def test_90_confidence(self):
        data = list(range(20))
        ci_90 = bootstrap_ci(data, confidence=0.90, seed=42)
        ci_95 = bootstrap_ci(data, confidence=0.95, seed=42)
        assert ci_90.width < ci_95.width

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="2 data points"):
            bootstrap_ci([1.0])

    def test_ci_contains_true_mean_usually(self):
        """With enough data from a known distribution, the CI should contain
        the true mean most of the time."""
        import random
        rng = random.Random(42)
        true_mean = 5.0
        n_trials = 100
        n_covered = 0
        for trial in range(n_trials):
            data = [rng.gauss(true_mean, 1.0) for _ in range(30)]
            ci = bootstrap_ci(data, seed=trial)
            if ci.lower <= true_mean <= ci.upper:
                n_covered += 1
        # 95% CI should cover ~95% of the time. Allow some slack.
        assert n_covered >= 85, f"Coverage {n_covered}% too low"

    def test_properties(self):
        ci = ConfidenceInterval(estimate=3.0, lower=2.0, upper=4.0, confidence=0.95)
        assert ci.width == pytest.approx(2.0)
        assert ci.margin == pytest.approx(1.0)
        assert "3.0000" in str(ci)
        assert "95%" in str(ci)


# ============================================================
# Significance Tests
# ============================================================


class TestPairedTTest:
    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0]
        result = paired_t_test(a, a)
        assert result.p_value == 1.0
        assert not result.significant_at_05

    def test_clear_difference(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0]
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = paired_t_test(control, treatment)
        assert result.p_value < 0.01
        assert result.significant_at_01

    def test_no_difference(self):
        # Small random fluctuations around zero difference
        control = [1.01, 2.02, 3.01, 1.99, 2.98]
        treatment = [0.99, 1.98, 2.99, 2.01, 3.02]
        result = paired_t_test(control, treatment)
        assert not result.significant_at_05

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_t_test([1.0, 2.0], [1.0])

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="2 pairs"):
            paired_t_test([1.0], [2.0])

    def test_p_value_always_in_0_1(self):
        """p-value must be in [0, 1] for all df values.

        Regression test: the t-CDF diverged for moderate df when
        _regularized_beta lacked the symmetry relation.
        """
        import random
        rng = random.Random(42)
        for n in range(2, 35):
            control = [rng.gauss(0, 1) for _ in range(n)]
            treatment = [rng.gauss(0.5, 1) for _ in range(n)]
            result = paired_t_test(control, treatment)
            assert 0.0 <= result.p_value <= 1.0, (
                f"p={result.p_value} out of [0,1] for n={n} (df={n-1})"
            )

    def test_known_p_value_small_df(self):
        """Verify t-CDF gives reasonable p-values for small degrees of freedom."""
        # 4 pairs, small but real difference
        control = [0.5, 0.6, 0.7, 0.8]
        treatment = [0.45, 0.55, 0.65, 0.75]
        result = paired_t_test(control, treatment)
        assert 0.0 <= result.p_value <= 1.0
        # Large t-stat (near-constant differences) -> very small p
        assert result.p_value < 0.01

    def test_str_formatting(self):
        result = paired_t_test([10.0, 11.0, 12.0], [5.0, 6.0, 7.0])
        s = str(result)
        assert "Paired t-test" in s
        assert "stat=" in s
        assert "p=" in s


class TestPairedWilcoxon:
    def test_clear_difference(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        result = paired_wilcoxon(control, treatment)
        assert result.p_value < 0.05
        assert result.significant_at_05

    def test_no_difference(self):
        # Symmetric around zero
        control = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        treatment = [1.1, 1.9, 3.1, 3.9, 5.1, 5.9]
        result = paired_wilcoxon(control, treatment)
        assert not result.significant_at_05

    def test_all_zeros(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = paired_wilcoxon(a, a)
        assert result.p_value == 1.0

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError, match="6 pairs"):
            paired_wilcoxon([1.0, 2.0], [3.0, 4.0])

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_wilcoxon([1.0] * 6, [2.0] * 7)


# ============================================================
# Effect Size
# ============================================================


class TestCohensD:
    def test_identical_gives_zero(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_large_effect(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0]
        treatment = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = cohens_d(control, treatment)
        assert d > 0.8  # Large effect
        assert d > 0  # Positive because control > treatment

    def test_small_effect(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0]
        treatment = [9.8, 10.8, 11.8, 12.8, 13.8]
        d = cohens_d(control, treatment)
        assert abs(d) < 0.5  # Small effect

    def test_direction(self):
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        assert cohens_d(a, b) > 0  # a > b -> positive
        assert cohens_d(b, a) < 0  # b < a -> negative


class TestCohensDPaired:
    def test_identical_gives_zero(self):
        a = [1.0, 2.0, 3.0, 4.0]
        assert cohens_d_paired(a, a) == pytest.approx(0.0)

    def test_constant_difference(self):
        # If all differences are identical, sd_d = 0, returns 0
        control = [10.0, 20.0, 30.0]
        treatment = [5.0, 15.0, 25.0]
        # All differences = 5.0, sd_d = 0
        d = cohens_d_paired(control, treatment)
        # sd_d = 0, function returns 0.0
        assert d == pytest.approx(0.0)

    def test_large_paired_effect(self):
        control = [0.4, 0.3, 0.35, 0.5, 0.45, 0.38]
        treatment = [0.2, 0.15, 0.20, 0.3, 0.25, 0.18]
        d = cohens_d_paired(control, treatment)
        assert d > 0.8  # Large paired effect


# ============================================================
# Cell Statistics
# ============================================================


class TestSummarizeCell:
    def test_basic(self):
        values = [0.3, 0.35, 0.28, 0.32, 0.31]
        stats = summarize_cell(values, metric="delta", seed=42)
        assert stats.n == 5
        assert stats.mean == pytest.approx(0.312, abs=0.001)
        assert stats.std > 0
        assert stats.se > 0
        assert stats.se < stats.std  # SE < SD always
        assert stats.ci.lower < stats.mean
        assert stats.ci.upper > stats.mean
        assert stats.min == pytest.approx(0.28)
        assert stats.max == pytest.approx(0.35)

    def test_median_odd(self):
        stats = summarize_cell([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
        assert stats.median == pytest.approx(3.0)

    def test_median_even(self):
        stats = summarize_cell([1.0, 2.0, 3.0, 4.0], seed=42)
        assert stats.median == pytest.approx(2.5)

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="2 replicates"):
            summarize_cell([1.0])

    def test_str_formatting(self):
        stats = summarize_cell([1.0, 2.0, 3.0], metric="phi", seed=42)
        s = str(stats)
        assert "phi" in s
        assert "mean=" in s


class TestSummarizePairedCell:
    def test_basic(self):
        control = [0.4, 0.35, 0.38, 0.42, 0.39, 0.41]
        treatment = [0.25, 0.22, 0.24, 0.28, 0.26, 0.27]
        stats = summarize_paired_cell(control, treatment, metric="delta", seed=42)

        assert stats.n_pairs == 6
        assert stats.control.mean > stats.treatment.mean
        assert stats.effect.estimate > 0  # control > treatment
        assert stats.effect_size > 0  # positive effect

    def test_no_effect(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1]
        stats = summarize_paired_cell(a, b, metric="x", seed=42)
        assert abs(stats.effect.estimate) < 0.5
        assert not stats.effect_test.significant_at_05

    def test_uses_wilcoxon_for_n_ge_6(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = summarize_paired_cell(control, treatment, seed=42)
        assert stats.effect_test.test_name == "Wilcoxon signed-rank"

    def test_uses_ttest_for_n_lt_6(self):
        control = [10.0, 11.0, 12.0, 13.0, 14.0]
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0]
        stats = summarize_paired_cell(control, treatment, seed=42)
        assert stats.effect_test.test_name == "Paired t-test"

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            summarize_paired_cell([1.0, 2.0], [1.0, 2.0, 3.0])


# ============================================================
# Morris Sensitivity
# ============================================================


class TestMorrisScreening:
    def test_linear_model_identifies_important_params(self):
        """For y = 10*x1 + 0.1*x2, x1 should dominate."""
        def model(params):
            return 10 * params["x1"] + 0.1 * params["x2"]

        results = morris_screening(
            model=model,
            bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
            num_trajectories=20,
            num_levels=4,
            seed=42,
        )

        assert len(results) == 2
        # x1 should be first (most important)
        assert results[0].parameter == "x1"
        assert results[0].mu_star > results[1].mu_star
        # Linear model: sigma should be low (no interactions)
        assert results[0].sigma < results[0].mu_star

    def test_nonlinear_model_has_high_sigma(self):
        """For y = x1 * x2, sigma should be high (interaction)."""
        def model(params):
            return params["x1"] * params["x2"]

        results = morris_screening(
            model=model,
            bounds={"x1": (0.0, 10.0), "x2": (0.0, 10.0)},
            num_trajectories=30,
            num_levels=4,
            seed=42,
        )

        # Both should have similar importance (symmetric model)
        mu_stars = {r.parameter: r.mu_star for r in results}
        ratio = mu_stars["x1"] / mu_stars["x2"] if mu_stars["x2"] > 0 else float("inf")
        assert 0.3 < ratio < 3.0, f"Asymmetric importance: {ratio}"

        # Interaction model should have non-negligible sigma
        for r in results:
            if r.mu_star > 0:
                assert r.sigma > 0

    def test_irrelevant_parameter(self):
        """A parameter that doesn't affect output should have mu*=0."""
        def model(params):
            return params["x1"] * 5  # x2 is irrelevant

        results = morris_screening(
            model=model,
            bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
            num_trajectories=20,
            num_levels=4,
            seed=42,
        )

        by_name = {r.parameter: r for r in results}
        assert by_name["x1"].mu_star > 0
        assert by_name["x2"].mu_star == pytest.approx(0.0, abs=1e-10)

    def test_single_parameter(self):
        def model(params):
            return params["x"] ** 2

        results = morris_screening(
            model=model,
            bounds={"x": (0.0, 1.0)},
            num_trajectories=10,
            seed=42,
        )
        assert len(results) == 1
        assert results[0].parameter == "x"
        assert results[0].mu_star > 0

    def test_empty_bounds(self):
        results = morris_screening(
            model=lambda p: 0.0,
            bounds={},
            num_trajectories=10,
        )
        assert results == []

    def test_too_few_trajectories_raises(self):
        with pytest.raises(ValueError, match="num_trajectories"):
            morris_screening(
                model=lambda p: 0.0,
                bounds={"x": (0, 1)},
                num_trajectories=1,
            )


# ============================================================
# SweepAnalyzer
# ============================================================


class TestSweepAnalyzer:
    @pytest.fixture
    def sample_records(self):
        """Simulated replicated sweep data: 3 kappa values x 10 seeds."""
        import random
        rng = random.Random(42)
        records = []
        for kappa in [0.25, 0.5, 1.0]:
            for seed in range(10):
                # Higher kappa -> lower delta (less stress)
                base_delta = max(0, 0.6 - 0.4 * kappa + rng.gauss(0, 0.05))
                noise = rng.gauss(0, 0.02)
                records.append({
                    "kappa": kappa,
                    "seed": seed,
                    "delta_passive": base_delta,
                    "delta_active": base_delta - 0.1 + noise,  # treatment helps by ~0.1
                    "phi_passive": 1 - base_delta,
                    "phi_active": 1 - (base_delta - 0.1 + noise),
                })
        return records

    def test_grouping(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        assert analyzer.n_cells == 3
        assert analyzer.n_records == 30
        assert analyzer.min_replicates() == 10

    def test_cell_table(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        table = analyzer.cell_table(metric="delta_passive", seed=42)
        assert len(table.rows) == 3

        # Check ordering: kappa=0.25 should have highest delta
        rows_by_kappa = {r.params["kappa"]: r for r in table.rows}
        assert rows_by_kappa[0.25].stats.mean > rows_by_kappa[1.0].stats.mean

        # All should have n=10
        for row in table.rows:
            assert row.stats.n == 10
            assert row.stats.se > 0
            assert row.stats.ci.lower < row.stats.ci.upper

    def test_cell_table_to_dicts(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        table = analyzer.cell_table(metric="delta_passive", seed=42)
        dicts = table.to_dicts()
        assert len(dicts) == 3
        assert "kappa" in dicts[0]
        assert "mean" in dicts[0]
        assert "ci_lower" in dicts[0]

    def test_treatment_effect_table(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        effects = analyzer.treatment_effect_table(
            metric="delta",
            control_suffix="_passive",
            treatment_suffix="_active",
            seed=42,
        )
        assert len(effects.rows) == 3

        # Treatment effect should be positive (dealer helps)
        for row in effects.rows:
            assert row.stats.effect.estimate > 0
            # With 10 replicates and clear effect, should be significant
            assert row.stats.effect_test.significant_at_05

    def test_effect_table_summary(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        effects = analyzer.treatment_effect_table(
            metric="delta",
            control_suffix="_passive",
            treatment_suffix="_active",
            seed=42,
        )
        summary = effects.summary()
        assert summary["n_cells"] == 3
        assert summary["mean_effect"] > 0
        assert summary["n_positive"] == 3

    def test_effect_table_to_dicts(self, sample_records):
        analyzer = SweepAnalyzer(sample_records, param_fields=["kappa"])
        effects = analyzer.treatment_effect_table(
            metric="delta",
            control_suffix="_passive",
            treatment_suffix="_active",
            seed=42,
        )
        dicts = effects.to_dicts()
        assert len(dicts) == 3
        assert "effect" in dicts[0]
        assert "p_value" in dicts[0]
        assert "cohens_d" in dicts[0]

    def test_multi_param_grouping(self):
        """Test grouping by multiple parameters."""
        records = []
        import random
        rng = random.Random(42)
        for kappa in [0.5, 1.0]:
            for mu in [0.0, 0.5]:
                for seed in range(5):
                    records.append({
                        "kappa": kappa,
                        "mu": mu,
                        "seed": seed,
                        "delta_passive": 0.5 - 0.3 * kappa + 0.1 * mu + rng.gauss(0, 0.02),
                    })

        analyzer = SweepAnalyzer(records, param_fields=["kappa", "mu"])
        assert analyzer.n_cells == 4  # 2 kappas x 2 mus
        assert analyzer.min_replicates() == 5

    def test_sensitivity_ranking(self, sample_records):
        # Add concentration dimension to make sensitivity analysis interesting
        import random
        rng = random.Random(42)
        records = []
        for kappa in [0.25, 0.5, 1.0, 2.0]:
            for conc in [0.5, 1.0, 2.0]:
                for seed in range(5):
                    # kappa matters a lot, concentration matters a little
                    delta = max(0, 0.6 - 0.4 * kappa + 0.02 * conc + rng.gauss(0, 0.02))
                    records.append({
                        "kappa": kappa,
                        "concentration": conc,
                        "seed": seed,
                        "delta_passive": delta,
                    })

        analyzer = SweepAnalyzer(records, param_fields=["kappa", "concentration"])
        ranking = analyzer.sensitivity_ranking(
            metric="delta_passive",
            bounds={"kappa": (0.25, 2.0), "concentration": (0.5, 2.0)},
            num_trajectories=20,
            seed=42,
        )

        assert len(ranking) == 2
        # kappa should be more important
        assert ranking[0].parameter == "kappa"
        assert ranking[0].mu_star > ranking[1].mu_star

    def test_empty_records(self):
        analyzer = SweepAnalyzer([], param_fields=["kappa"])
        assert analyzer.n_cells == 0
        assert analyzer.n_records == 0

    def test_missing_metric_skipped(self):
        """Records missing the metric field should be skipped gracefully."""
        records = [
            {"kappa": 0.5, "delta": 0.3},
            {"kappa": 0.5, "delta": 0.4},
            {"kappa": 0.5},  # missing delta
        ]
        analyzer = SweepAnalyzer(records, param_fields=["kappa"])
        table = analyzer.cell_table(metric="delta", seed=42)
        assert len(table.rows) == 1
        assert table.rows[0].stats.n == 2  # 2 valid, 1 skipped

    def test_treatment_effect_missing_arm_no_mispair(self):
        """When some records lack one arm, pairs must still align by record.

        Regression test: independent extraction + truncation was silently
        mispairing control[i] with treatment[j] from different records.
        """
        # 5 records in one cell. Records 1 and 3 are missing treatment.
        # Control values are all ~10. Treatment values are all ~5.
        # If mispairing occurs, control[2]=10 would pair with treatment
        # from a different record (still 5), hiding the bug numerically.
        # So we use distinct values to detect misalignment.
        records = [
            {"kappa": 0.5, "seed": 0, "delta_passive": 10.0, "delta_active": 5.0},
            {"kappa": 0.5, "seed": 1, "delta_passive": 20.0, "delta_active": None},  # missing
            {"kappa": 0.5, "seed": 2, "delta_passive": 30.0, "delta_active": 15.0},
            {"kappa": 0.5, "seed": 3, "delta_passive": 40.0, "delta_active": None},  # missing
            {"kappa": 0.5, "seed": 4, "delta_passive": 50.0, "delta_active": 25.0},
        ]
        analyzer = SweepAnalyzer(records, param_fields=["kappa"])
        effects = analyzer.treatment_effect_table(
            metric="delta", control_suffix="_passive", treatment_suffix="_active",
            seed=42,
        )
        assert len(effects.rows) == 1
        row = effects.rows[0]
        # Only 3 valid pairs (seeds 0, 2, 4). Differences: 5, 15, 25
        assert row.stats.n_pairs == 3
        assert row.stats.effect.estimate == pytest.approx(15.0)
        # Control mean should be (10+30+50)/3 = 30, not (10+20+30)/3 = 20
        assert row.stats.control.mean == pytest.approx(30.0)
        # Treatment mean should be (5+15+25)/3 = 15
        assert row.stats.treatment.mean == pytest.approx(15.0)
