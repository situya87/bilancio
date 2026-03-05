"""Parity tests: semantics-preserving flags must not change critical metrics.

These tests run small ring simulations with different PerformanceConfig
presets and verify that delta_total and phi_total are bit-identical when
only semantics-preserving optimizations differ.

Also includes determinism tests (seed stability, golden snapshots) and
native-vs-Python backend parity checks.
"""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import pytest

from bilancio.analysis.report import compute_day_metrics, summarize_day_metrics
from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.core.performance import _BOOL_FLAGS, SEMANTICS_PRESERVING, PerformanceConfig
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.scenarios.ring.compiler import compile_ring_explorer

try:
    from bilancio.dealer.kernel_native import NATIVE_AVAILABLE
except ImportError:
    NATIVE_AVAILABLE = False


def _compile_ring_scenario(
    *,
    n_agents: int,
    kappa: Decimal,
    concentration: Decimal,
    mu: Decimal,
    seed: int,
    maturity_days: int = 5,
) -> dict[str, Any]:
    """Compile a small ring scenario dict using the ring explorer compiler."""
    q_total = Decimal(100 * n_agents)
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "parity-test",
            "params": {
                "n_agents": n_agents,
                "seed": seed,
                "kappa": str(kappa),
                "Q_total": str(q_total),
                "liquidity": {"allocation": {"mode": "uniform"}},
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": "0",
                },
                "maturity": {"days": maturity_days, "mode": "lead_lag", "mu": str(mu)},
            },
            "compile": {"emit_yaml": False},
        }
    )
    return compile_ring_explorer(gen_config, source_path=None)


def _scenario_to_system(scenario: dict[str, Any]) -> System:
    """Convert a compiled scenario dict to a configured System."""
    config = ScenarioConfig(**preprocess_config(copy.deepcopy(scenario)))
    system = System(default_mode="expel-agent")
    apply_to_system(config, system)
    return system


def _run_ring(
    preset: str,
    seed: int = 42,
    n_agents: int = 10,
    kappa: str = "1",
    concentration: str = "1",
    mu: str = "0",
    maturity_days: int = 5,
    **perf_overrides: Any,
) -> dict[str, Any]:
    """Run a ring simulation and return metrics.

    Compiles a ring with the given parameters, creates a System, runs
    until stable with the given performance preset, and computes summary
    metrics from the event log.
    """
    scenario = _compile_ring_scenario(
        n_agents=n_agents,
        kappa=Decimal(kappa),
        concentration=Decimal(concentration),
        mu=Decimal(mu),
        seed=seed,
        maturity_days=maturity_days,
    )

    system = _scenario_to_system(scenario)
    perf = PerformanceConfig.create(preset, **perf_overrides)

    run_until_stable(
        system,
        max_days=20,
        quiet_days=2,
        performance=perf,
    )

    # Compute metrics from event log
    events = system.state.events
    result = compute_day_metrics(events=events, balances_rows=None, day_list=None)
    summary = summarize_day_metrics(result["day_metrics"])
    return {
        "delta_total": summary.get("delta_total"),
        "phi_total": summary.get("phi_total"),
        "max_day": summary.get("max_day"),
    }


def _run_small_ring(preset: str, seed: int = 42, **perf_overrides: Any) -> dict[str, Any]:
    """Run a small (n=10, kappa=1) ring simulation and return metrics.

    Convenience wrapper around ``_run_ring`` for backward compatibility
    with existing tests.
    """
    return _run_ring(preset, seed=seed, **perf_overrides)


# -- Boolean flags that are semantics-preserving (suitable for parametrize) --
_SEMANTICS_PRESERVING_BOOL_FLAGS = sorted(
    flag for flag in _BOOL_FLAGS if flag in SEMANTICS_PRESERVING
)


@pytest.mark.regression
class TestPerformanceParity:
    """Semantics-preserving presets produce identical critical metrics."""

    def test_compatible_vs_fast_identical(self) -> None:
        compatible = _run_small_ring("compatible")
        fast = _run_small_ring("fast")
        assert compatible["delta_total"] == fast["delta_total"], (
            f"delta_total differs: compatible={compatible['delta_total']}, fast={fast['delta_total']}"
        )
        assert compatible["phi_total"] == fast["phi_total"], (
            f"phi_total differs: compatible={compatible['phi_total']}, fast={fast['phi_total']}"
        )

    def test_compatible_vs_aggressive_identical(self) -> None:
        compatible = _run_small_ring("compatible")
        aggressive = _run_small_ring("aggressive")
        assert compatible["delta_total"] == aggressive["delta_total"], (
            f"delta_total differs: compatible={compatible['delta_total']}, aggressive={aggressive['delta_total']}"
        )
        assert compatible["phi_total"] == aggressive["phi_total"], (
            f"phi_total differs: compatible={compatible['phi_total']}, aggressive={aggressive['phi_total']}"
        )

    def test_urgency_order_is_semantics_changing(self) -> None:
        """Document that matching_order=urgency can produce different results."""
        _run_small_ring("compatible")  # baseline (validates both run)
        urgency_metrics = _run_small_ring("compatible", matching_order="urgency")

        # This test documents the distinction -- urgency order MAY produce
        # different results. We don't assert they're different (they might
        # be the same for this particular scenario), we just document it.
        # The key assertion is that the test runs successfully.
        assert urgency_metrics["delta_total"] is not None
        assert urgency_metrics["phi_total"] is not None


# ────────────────────────────────────────────────────────────────────
# WI-6: Determinism & Performance-Option Tests
# ────────────────────────────────────────────────────────────────────


@pytest.mark.regression
class TestSeedDeterminism:
    """Verify that simulation output is fully deterministic for a given seed."""

    def test_same_seed_same_config_produces_identical_results(self) -> None:
        """Three consecutive runs with identical seed + config must produce identical metrics."""
        results = [_run_small_ring("compatible", seed=42) for _ in range(3)]
        for i in range(1, 3):
            assert results[0]["delta_total"] == results[i]["delta_total"], (
                f"Run 0 vs run {i}: delta_total differs: "
                f"{results[0]['delta_total']} != {results[i]['delta_total']}"
            )
            assert results[0]["phi_total"] == results[i]["phi_total"], (
                f"Run 0 vs run {i}: phi_total differs: "
                f"{results[0]['phi_total']} != {results[i]['phi_total']}"
            )

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds with same config should produce different metrics.

        Uses kappa=0.5 with n=20 so defaults are likely and the Dirichlet
        draw creates meaningful variation between seeds.
        """
        run_42 = _run_ring("compatible", seed=42, n_agents=20, kappa="0.5")
        run_99 = _run_ring("compatible", seed=99, n_agents=20, kappa="0.5")
        # At least one metric should differ between different seeds
        differs = (
            run_42["delta_total"] != run_99["delta_total"]
            or run_42["phi_total"] != run_99["phi_total"]
        )
        assert differs, (
            f"Seeds 42 and 99 produced identical results (n=20, kappa=0.5). "
            f"seed=42: delta={run_42['delta_total']}, phi={run_42['phi_total']}; "
            f"seed=99: delta={run_99['delta_total']}, phi={run_99['phi_total']}"
        )


@pytest.mark.regression
class TestSemanticsPreservingFlags:
    """Each SEMANTICS_PRESERVING boolean flag individually toggled must match baseline."""

    @pytest.mark.parametrize("flag", _SEMANTICS_PRESERVING_BOOL_FLAGS)
    def test_each_semantics_preserving_flag_vs_baseline(self, flag: str) -> None:
        """Toggling a single semantics-preserving flag must produce identical metrics."""
        baseline = _run_small_ring("compatible")
        toggled = _run_small_ring("compatible", **{flag: True})
        assert baseline["delta_total"] == toggled["delta_total"], (
            f"delta_total differs with {flag}=True: "
            f"baseline={baseline['delta_total']}, toggled={toggled['delta_total']}"
        )
        assert baseline["phi_total"] == toggled["phi_total"], (
            f"phi_total differs with {flag}=True: "
            f"baseline={baseline['phi_total']}, toggled={toggled['phi_total']}"
        )

    def test_all_semantics_preserving_flags_combined(self) -> None:
        """All semantics-preserving boolean flags toggled at once must match baseline."""
        baseline = _run_small_ring("compatible")
        all_flags = dict.fromkeys(_SEMANTICS_PRESERVING_BOOL_FLAGS, True)
        combined = _run_small_ring("compatible", **all_flags)
        assert baseline["delta_total"] == combined["delta_total"], (
            f"delta_total differs with all flags on: "
            f"baseline={baseline['delta_total']}, combined={combined['delta_total']}"
        )
        assert baseline["phi_total"] == combined["phi_total"], (
            f"phi_total differs with all flags on: "
            f"baseline={baseline['phi_total']}, combined={combined['phi_total']}"
        )


@pytest.mark.regression
class TestNativeBackendParity:
    """Rust native backend must produce identical metrics to Python backend."""

    @pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Rust extension not installed")
    def test_native_vs_python_backend_parity(self) -> None:
        """dealer_backend='native' vs 'python' should produce identical metrics."""
        python_metrics = _run_small_ring("compatible", dealer_backend="python")
        native_metrics = _run_small_ring("compatible", dealer_backend="native")
        assert python_metrics["delta_total"] == native_metrics["delta_total"], (
            f"delta_total differs: python={python_metrics['delta_total']}, "
            f"native={native_metrics['delta_total']}"
        )
        assert python_metrics["phi_total"] == native_metrics["phi_total"], (
            f"phi_total differs: python={python_metrics['phi_total']}, "
            f"native={native_metrics['phi_total']}"
        )


@pytest.mark.regression
class TestGoldenMetricSnapshot:
    """Pinned expected values for known configurations.

    These catch unintentional behavior changes. If a legitimate change
    alters simulation output, update the expected values here and
    document the reason in the commit message.
    """

    def test_golden_metric_snapshot_seed_42(self) -> None:
        """Pinned expected value for seed=42, n=10, kappa=1 ring scenario."""
        metrics = _run_small_ring("compatible", seed=42)
        expected_delta = Decimal("0.6579739217652958876629889669")
        expected_phi = Decimal("0.3420260782347041123370110331")
        assert metrics["delta_total"] == expected_delta, (
            f"Golden delta_total changed! "
            f"expected={expected_delta}, actual={metrics['delta_total']}. "
            f"If this is intentional, update the expected value and document the reason."
        )
        assert metrics["phi_total"] == expected_phi, (
            f"Golden phi_total changed! "
            f"expected={expected_phi}, actual={metrics['phi_total']}. "
            f"If this is intentional, update the expected value and document the reason."
        )

    def test_golden_metric_snapshot_stressed(self) -> None:
        """Pinned expected value for seed=42, n=20, kappa=0.5 (stressed system)."""
        metrics = _run_ring(
            "compatible", seed=42, n_agents=20, kappa="0.5",
        )
        expected_delta = Decimal("0.8256281407035175879396984925")
        expected_phi = Decimal("0.1743718592964824120603015075")
        assert metrics["delta_total"] == expected_delta, (
            f"Golden delta_total (stressed) changed! "
            f"expected={expected_delta}, actual={metrics['delta_total']}. "
            f"If this is intentional, update the expected value and document the reason."
        )
        assert metrics["phi_total"] == expected_phi, (
            f"Golden phi_total (stressed) changed! "
            f"expected={expected_phi}, actual={metrics['phi_total']}. "
            f"If this is intentional, update the expected value and document the reason."
        )
