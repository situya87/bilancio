"""Tests for experiment sweep runners and sampling functions.

This module tests:
- RingSweepRunner grid and LHS execution
- Sampling functions: generate_grid_params, generate_lhs_params, generate_frontier_params
- Registry CSV creation
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from bilancio.experiments.sampling import (
    generate_frontier_params,
    generate_grid_params,
    generate_lhs_params,
)

# =============================================================================
# Tests for generate_grid_params
# =============================================================================


class TestGenerateGridParams:
    """Tests for Cartesian product grid sampling."""

    def test_basic_2x2_grid(self):
        """A 2x2x1x1 grid produces 4 combinations."""
        kappas = [Decimal("1"), Decimal("2")]
        concentrations = [Decimal("0.5"), Decimal("1")]
        mus = [Decimal("0")]
        monotonicities = [Decimal("0")]

        result = list(generate_grid_params(kappas, concentrations, mus, monotonicities))

        assert len(result) == 4
        # Check that all combinations are present
        expected = [
            (Decimal("1"), Decimal("0.5"), Decimal("0"), Decimal("0")),
            (Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0")),
            (Decimal("2"), Decimal("0.5"), Decimal("0"), Decimal("0")),
            (Decimal("2"), Decimal("1"), Decimal("0"), Decimal("0")),
        ]
        assert set(result) == set(expected)

    def test_single_values(self):
        """Grid with single values per dimension produces one combination."""
        kappas = [Decimal("1")]
        concentrations = [Decimal("0.5")]
        mus = [Decimal("0.25")]
        monotonicities = [Decimal("0")]

        result = list(generate_grid_params(kappas, concentrations, mus, monotonicities))

        assert len(result) == 1
        assert result[0] == (Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0"))

    def test_empty_dimension(self):
        """Grid with empty dimension produces no combinations."""
        kappas = []
        concentrations = [Decimal("0.5")]
        mus = [Decimal("0")]
        monotonicities = [Decimal("0")]

        result = list(generate_grid_params(kappas, concentrations, mus, monotonicities))

        assert len(result) == 0

    def test_full_grid_count(self):
        """Grid count equals product of dimensions."""
        kappas = [Decimal("1"), Decimal("2"), Decimal("4")]
        concentrations = [Decimal("0.5"), Decimal("1")]
        mus = [Decimal("0"), Decimal("0.5"), Decimal("1")]
        monotonicities = [Decimal("0"), Decimal("0.5")]

        result = list(generate_grid_params(kappas, concentrations, mus, monotonicities))

        expected_count = 3 * 2 * 3 * 2  # 36
        assert len(result) == expected_count

    def test_iterator_returns_tuples_of_decimals(self):
        """All returned values are Decimal tuples."""
        kappas = [Decimal("1")]
        concentrations = [Decimal("0.5")]
        mus = [Decimal("0")]
        monotonicities = [Decimal("0")]

        result = list(generate_grid_params(kappas, concentrations, mus, monotonicities))

        assert len(result) == 1
        kappa, concentration, mu, monotonicity = result[0]
        assert isinstance(kappa, Decimal)
        assert isinstance(concentration, Decimal)
        assert isinstance(mu, Decimal)
        assert isinstance(monotonicity, Decimal)


# =============================================================================
# Tests for generate_lhs_params
# =============================================================================


class TestGenerateLHSParams:
    """Tests for Latin Hypercube Sampling."""

    def test_count_matches_request(self):
        """LHS generates exactly the requested number of samples."""
        count = 10
        result = list(
            generate_lhs_params(
                count,
                kappa_range=(Decimal("0.5"), Decimal("4")),
                concentration_range=(Decimal("0.1"), Decimal("5")),
                mu_range=(Decimal("0"), Decimal("1")),
                monotonicity_range=(Decimal("0"), Decimal("0.5")),
                seed=42,
            )
        )

        assert len(result) == count

    def test_zero_count_returns_empty(self):
        """LHS with count=0 returns empty iterator."""
        result = list(
            generate_lhs_params(
                0,
                kappa_range=(Decimal("0.5"), Decimal("4")),
                concentration_range=(Decimal("0.1"), Decimal("5")),
                mu_range=(Decimal("0"), Decimal("1")),
                monotonicity_range=(Decimal("0"), Decimal("0.5")),
                seed=42,
            )
        )

        assert len(result) == 0

    def test_values_within_range(self):
        """All LHS samples fall within specified ranges."""
        count = 20
        kappa_range = (Decimal("0.5"), Decimal("4"))
        concentration_range = (Decimal("0.1"), Decimal("5"))
        mu_range = (Decimal("0"), Decimal("1"))
        monotonicity_range = (Decimal("-0.5"), Decimal("0.5"))

        result = list(
            generate_lhs_params(
                count,
                kappa_range=kappa_range,
                concentration_range=concentration_range,
                mu_range=mu_range,
                monotonicity_range=monotonicity_range,
                seed=42,
            )
        )

        for kappa, concentration, mu, monotonicity in result:
            assert kappa_range[0] <= kappa <= kappa_range[1]
            assert concentration_range[0] <= concentration <= concentration_range[1]
            assert mu_range[0] <= mu <= mu_range[1]
            assert monotonicity_range[0] <= monotonicity <= monotonicity_range[1]

    def test_reproducibility_with_same_seed(self):
        """Same seed produces identical samples."""
        kwargs = {
            "count": 5,
            "kappa_range": (Decimal("0.5"), Decimal("4")),
            "concentration_range": (Decimal("0.1"), Decimal("5")),
            "mu_range": (Decimal("0"), Decimal("1")),
            "monotonicity_range": (Decimal("0"), Decimal("0.5")),
            "seed": 12345,
        }

        result1 = list(generate_lhs_params(**kwargs))
        result2 = list(generate_lhs_params(**kwargs))

        assert result1 == result2

    def test_different_seeds_produce_different_samples(self):
        """Different seeds produce different samples."""
        base_kwargs = {
            "count": 5,
            "kappa_range": (Decimal("0.5"), Decimal("4")),
            "concentration_range": (Decimal("0.1"), Decimal("5")),
            "mu_range": (Decimal("0"), Decimal("1")),
            "monotonicity_range": (Decimal("0"), Decimal("0.5")),
        }

        result1 = list(generate_lhs_params(**base_kwargs, seed=42))
        result2 = list(generate_lhs_params(**base_kwargs, seed=43))

        assert result1 != result2

    def test_returns_decimal_tuples(self):
        """All returned values are Decimal tuples."""
        result = list(
            generate_lhs_params(
                1,
                kappa_range=(Decimal("1"), Decimal("2")),
                concentration_range=(Decimal("0.5"), Decimal("1")),
                mu_range=(Decimal("0"), Decimal("1")),
                monotonicity_range=(Decimal("0"), Decimal("0")),
                seed=42,
            )
        )

        assert len(result) == 1
        kappa, concentration, mu, monotonicity = result[0]
        assert isinstance(kappa, Decimal)
        assert isinstance(concentration, Decimal)
        assert isinstance(mu, Decimal)
        assert isinstance(monotonicity, Decimal)


# =============================================================================
# Tests for generate_frontier_params
# =============================================================================


class TestGenerateFrontierParams:
    """Tests for frontier/binary search sampling."""

    def test_calls_execute_fn(self):
        """Frontier sampling calls execute_fn for each test."""
        calls = []

        def mock_execute(
            label: str, kappa: Decimal, concentration: Decimal, mu: Decimal, monotonicity: Decimal
        ) -> Decimal | None:
            calls.append((label, kappa, concentration, mu, monotonicity))
            # Return stable for high kappa, unstable for low
            if kappa >= Decimal("2"):
                return Decimal("0.01")  # Stable
            return Decimal("1.0")  # Unstable

        generate_frontier_params(
            concentrations=[Decimal("1")],
            mus=[Decimal("0.5")],
            monotonicities=[Decimal("0")],
            kappa_low=Decimal("0.5"),
            kappa_high=Decimal("4"),
            tolerance=Decimal("0.1"),
            max_iterations=3,
            execute_fn=mock_execute,
        )

        # Should have called execute_fn multiple times
        assert len(calls) > 0
        # First call should be with kappa_low
        assert calls[0][1] == Decimal("0.5")  # kappa_low

    def test_frontier_stops_if_low_is_stable(self):
        """Frontier stops immediately if kappa_low is already stable."""
        calls = []

        def mock_execute(
            label: str, kappa: Decimal, concentration: Decimal, mu: Decimal, monotonicity: Decimal
        ) -> Decimal | None:
            calls.append((label, kappa, concentration, mu, monotonicity))
            # Always stable
            return Decimal("0.01")

        generate_frontier_params(
            concentrations=[Decimal("1")],
            mus=[Decimal("0.5")],
            monotonicities=[Decimal("0")],
            kappa_low=Decimal("0.5"),
            kappa_high=Decimal("4"),
            tolerance=Decimal("0.1"),
            max_iterations=10,
            execute_fn=mock_execute,
        )

        # Should only call once for kappa_low since it's already stable
        assert len(calls) == 1
        assert calls[0][0] == "low"  # label

    def test_frontier_iterates_over_all_cells(self):
        """Frontier iterates over all (concentration, mu, monotonicity) combinations."""
        cells_visited = set()

        def mock_execute(
            label: str, kappa: Decimal, concentration: Decimal, mu: Decimal, monotonicity: Decimal
        ) -> Decimal | None:
            cells_visited.add((concentration, mu, monotonicity))
            return Decimal("0.01")  # Always stable

        generate_frontier_params(
            concentrations=[Decimal("1"), Decimal("2")],
            mus=[Decimal("0"), Decimal("0.5")],
            monotonicities=[Decimal("0")],
            kappa_low=Decimal("0.5"),
            kappa_high=Decimal("4"),
            tolerance=Decimal("0.1"),
            max_iterations=3,
            execute_fn=mock_execute,
        )

        # Should have visited 2*2*1 = 4 cells
        expected_cells = {
            (Decimal("1"), Decimal("0"), Decimal("0")),
            (Decimal("1"), Decimal("0.5"), Decimal("0")),
            (Decimal("2"), Decimal("0"), Decimal("0")),
            (Decimal("2"), Decimal("0.5"), Decimal("0")),
        }
        assert cells_visited == expected_cells

    def test_frontier_handles_failed_runs(self):
        """Frontier continues when execute_fn returns None (failed run)."""
        calls = []

        def mock_execute(
            label: str, kappa: Decimal, concentration: Decimal, mu: Decimal, monotonicity: Decimal
        ) -> Decimal | None:
            calls.append(label)
            # Return None for low (failed), stable for high
            if kappa < Decimal("2"):
                return None  # Failed
            return Decimal("0.01")  # Stable

        generate_frontier_params(
            concentrations=[Decimal("1")],
            mus=[Decimal("0.5")],
            monotonicities=[Decimal("0")],
            kappa_low=Decimal("0.5"),
            kappa_high=Decimal("4"),
            tolerance=Decimal("0.1"),
            max_iterations=3,
            execute_fn=mock_execute,
        )

        # Should have tried low, then high, then some mid points
        assert "low" in calls
        assert "high" in calls


# =============================================================================
# Tests for RingSweepRunner (mocked to avoid full simulation)
# =============================================================================


class TestRingSweepRunnerSetup:
    """Tests for RingSweepRunner initialization and configuration."""

    def test_creates_output_directories(self, tmp_path: Path):
        """RingSweepRunner creates registry, runs, and aggregate directories."""
        from bilancio.experiments.ring import RingSweepRunner

        RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        assert (tmp_path / "registry").exists()
        assert (tmp_path / "runs").exists()
        assert (tmp_path / "aggregate").exists()

    def test_creates_empty_registry_csv(self, tmp_path: Path):
        """RingSweepRunner creates registry CSV with headers."""
        from bilancio.experiments.ring import RingSweepRunner

        RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        registry_path = tmp_path / "registry" / "experiments.csv"
        assert registry_path.exists()

        # Check headers
        content = registry_path.read_text()
        assert "run_id" in content
        assert "kappa" in content
        assert "status" in content

    def test_seed_increments(self, tmp_path: Path):
        """Seed counter increments with each call to _next_seed()."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=100,
        )

        assert runner._next_seed() == 100
        assert runner._next_seed() == 101
        assert runner._next_seed() == 102


class TestRingSweepRunnerGridMocked:
    """Tests for RingSweepRunner.run_grid with mocked execution."""

    def test_run_grid_calls_execute_for_each_combination(self, tmp_path: Path):
        """run_grid calls _execute_run for each parameter combination."""
        from bilancio.experiments.ring import RingRunSummary, RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        # Mock _execute_run to avoid actual simulation
        calls = []

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            calls.append((kappa, concentration, mu, monotonicity))
            return RingRunSummary(
                run_id=f"test_{len(calls)}",
                phase=phase,
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                delta_total=Decimal("0.1"),
                phi_total=Decimal("0.5"),
                time_to_stability=5,
            )

        with patch.object(runner, "_execute_run", mock_execute):
            kappas = [Decimal("1"), Decimal("2")]
            concentrations = [Decimal("0.5"), Decimal("1")]
            mus = [Decimal("0")]
            monotonicities = [Decimal("0")]

            summaries = runner.run_grid(kappas, concentrations, mus, monotonicities)

        # Should have 2*2*1*1 = 4 calls
        assert len(calls) == 4
        assert len(summaries) == 4

    def test_run_grid_returns_summaries(self, tmp_path: Path):
        """run_grid returns list of RingRunSummary objects."""
        from bilancio.experiments.ring import RingRunSummary, RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            return RingRunSummary(
                run_id=f"test_{seed}",
                phase=phase,
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                delta_total=Decimal("0.1"),
                phi_total=Decimal("0.5"),
                time_to_stability=5,
            )

        with patch.object(runner, "_execute_run", mock_execute):
            summaries = runner.run_grid(
                [Decimal("1")],
                [Decimal("0.5")],
                [Decimal("0")],
                [Decimal("0")],
            )

        assert len(summaries) == 1
        assert isinstance(summaries[0], RingRunSummary)
        assert summaries[0].kappa == Decimal("1")


class TestRingSweepRunnerLHSMocked:
    """Tests for RingSweepRunner.run_lhs with mocked execution."""

    def test_run_lhs_generates_requested_count(self, tmp_path: Path):
        """run_lhs generates exactly the requested number of samples."""
        from bilancio.experiments.ring import RingRunSummary, RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        call_count = [0]

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            call_count[0] += 1
            return RingRunSummary(
                run_id=f"test_{seed}",
                phase=phase,
                kappa=kappa,
                concentration=concentration,
                mu=mu,
                monotonicity=monotonicity,
                delta_total=Decimal("0.1"),
                phi_total=Decimal("0.5"),
                time_to_stability=5,
            )

        with patch.object(runner, "_execute_run", mock_execute):
            summaries = runner.run_lhs(
                count=5,
                kappa_range=(Decimal("0.5"), Decimal("4")),
                concentration_range=(Decimal("0.1"), Decimal("5")),
                mu_range=(Decimal("0"), Decimal("1")),
                monotonicity_range=(Decimal("0"), Decimal("0.5")),
            )

        assert call_count[0] == 5
        assert len(summaries) == 5

    def test_run_lhs_zero_count_returns_empty(self, tmp_path: Path):
        """run_lhs with count=0 returns empty list without executing."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=2,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
        )

        # Should not need to mock since no runs should happen
        summaries = runner.run_lhs(
            count=0,
            kappa_range=(Decimal("0.5"), Decimal("4")),
            concentration_range=(Decimal("0.1"), Decimal("5")),
            mu_range=(Decimal("0"), Decimal("1")),
            monotonicity_range=(Decimal("0"), Decimal("0.5")),
        )

        assert summaries == []


# =============================================================================
# Tests for RingSweepConfig loading
# =============================================================================


class TestRingSweepConfigLoading:
    """Tests for loading sweep configuration from YAML."""

    def test_load_valid_grid_config(self, tmp_path: Path):
        """Valid grid config loads correctly."""
        from bilancio.experiments.ring import load_ring_sweep_config

        config_yaml = """
version: 1
out_dir: ./output
grid:
  enabled: true
  kappas: [0.5, 1, 2]
  concentrations: [0.5, 1]
  mus: [0, 0.5]
runner:
  n_agents: 50
  maturity_days: 5
  q_total: 5000
  liquidity_mode: uniform
  base_seed: 100
"""
        config_path = tmp_path / "sweep.yaml"
        config_path.write_text(config_yaml)

        config = load_ring_sweep_config(config_path)

        assert config.version == 1
        assert config.grid is not None
        assert config.grid.enabled is True
        assert len(config.grid.kappas) == 3
        assert config.runner.n_agents == 50

    def test_load_config_file_not_found(self, tmp_path: Path):
        """Missing config file raises FileNotFoundError."""
        from bilancio.experiments.ring import load_ring_sweep_config

        with pytest.raises(FileNotFoundError):
            load_ring_sweep_config(tmp_path / "nonexistent.yaml")

    def test_load_config_invalid_version(self, tmp_path: Path):
        """Invalid version raises ValueError."""
        from bilancio.experiments.ring import load_ring_sweep_config

        config_yaml = """
version: 99
grid:
  enabled: true
  kappas: [1]
  concentrations: [1]
  mus: [0]
"""
        config_path = tmp_path / "sweep.yaml"
        config_path.write_text(config_yaml)

        with pytest.raises(ValueError, match="Unsupported sweep config version"):
            load_ring_sweep_config(config_path)


# =============================================================================
# Tests for ComparisonSweepRunner setup
# =============================================================================


class TestComparisonSweepRunnerSetup:
    """Tests for ComparisonSweepRunner initialization."""

    def test_creates_output_directories(self, tmp_path: Path):
        """ComparisonSweepRunner creates control, treatment, and aggregate directories."""
        from bilancio.experiments.comparison import ComparisonSweepConfig, ComparisonSweepRunner

        config = ComparisonSweepConfig(
            n_agents=5,
            maturity_days=3,
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )

        ComparisonSweepRunner(config, tmp_path)

        assert (tmp_path / "control").exists()
        assert (tmp_path / "treatment").exists()
        assert (tmp_path / "aggregate").exists()


# =============================================================================
# Tests for BalancedComparisonRunner setup
# =============================================================================


class TestBalancedComparisonRunnerSetup:
    """Tests for BalancedComparisonRunner initialization."""

    def test_creates_output_directories(self, tmp_path: Path):
        """BalancedComparisonRunner creates passive, active, and aggregate directories."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            n_agents=5,
            maturity_days=3,
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            outside_mid_ratios=[Decimal("0.75")],
        )

        BalancedComparisonRunner(config, tmp_path)

        assert (tmp_path / "passive").exists()
        assert (tmp_path / "active").exists()
        assert (tmp_path / "aggregate").exists()


# =============================================================================
# Tests for ComparisonResult and BalancedComparisonResult
# =============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_delta_reduction_calculation(self):
        """delta_reduction is control minus treatment."""
        from bilancio.experiments.comparison import ComparisonResult

        result = ComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            delta_control=Decimal("0.5"),
            phi_control=Decimal("1"),
            control_run_id="c1",
            control_status="completed",
            delta_treatment=Decimal("0.3"),
            phi_treatment=Decimal("0.8"),
            treatment_run_id="t1",
            treatment_status="completed",
        )

        assert result.delta_reduction == Decimal("0.2")

    def test_relief_ratio_calculation(self):
        """relief_ratio is reduction divided by control."""
        from bilancio.experiments.comparison import ComparisonResult

        result = ComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            delta_control=Decimal("0.5"),
            phi_control=Decimal("1"),
            control_run_id="c1",
            control_status="completed",
            delta_treatment=Decimal("0.25"),
            phi_treatment=Decimal("0.8"),
            treatment_run_id="t1",
            treatment_status="completed",
        )

        assert result.relief_ratio == Decimal("0.5")

    def test_relief_ratio_zero_control(self):
        """relief_ratio is 0 when control has no defaults."""
        from bilancio.experiments.comparison import ComparisonResult

        result = ComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            delta_control=Decimal("0"),
            phi_control=Decimal("1"),
            control_run_id="c1",
            control_status="completed",
            delta_treatment=Decimal("0"),
            phi_treatment=Decimal("1"),
            treatment_run_id="t1",
            treatment_status="completed",
        )

        assert result.relief_ratio == Decimal("0")

    def test_delta_reduction_none_when_missing(self):
        """delta_reduction is None when control or treatment is None."""
        from bilancio.experiments.comparison import ComparisonResult

        result = ComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            delta_control=None,
            phi_control=None,
            control_run_id="c1",
            control_status="failed",
            delta_treatment=Decimal("0.3"),
            phi_treatment=Decimal("0.8"),
            treatment_run_id="t1",
            treatment_status="completed",
        )

        assert result.delta_reduction is None
        assert result.relief_ratio is None


class TestBalancedComparisonResult:
    """Tests for BalancedComparisonResult dataclass."""

    def test_trading_effect_calculation(self):
        """trading_effect is passive minus active."""
        from bilancio.experiments.balanced_comparison import BalancedComparisonResult

        result = BalancedComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            face_value=Decimal("20"),
            outside_mid_ratio=Decimal("0.75"),
            big_entity_share=Decimal("0.25"),
            delta_passive=Decimal("0.5"),
            phi_passive=Decimal("1"),
            passive_run_id="p1",
            passive_status="completed",
            delta_active=Decimal("0.3"),
            phi_active=Decimal("0.8"),
            active_run_id="a1",
            active_status="completed",
        )

        assert result.trading_effect == Decimal("0.2")

    def test_trading_relief_ratio_calculation(self):
        """trading_relief_ratio is effect divided by passive."""
        from bilancio.experiments.balanced_comparison import BalancedComparisonResult

        result = BalancedComparisonResult(
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            face_value=Decimal("20"),
            outside_mid_ratio=Decimal("0.75"),
            big_entity_share=Decimal("0.25"),
            delta_passive=Decimal("0.5"),
            phi_passive=Decimal("1"),
            passive_run_id="p1",
            passive_status="completed",
            delta_active=Decimal("0.25"),
            phi_active=Decimal("0.8"),
            active_run_id="a1",
            active_status="completed",
        )

        assert result.trading_relief_ratio == Decimal("0.5")
