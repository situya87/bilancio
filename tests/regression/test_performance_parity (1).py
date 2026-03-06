"""Parity tests: semantics-preserving flags must not change critical metrics.

These tests run small ring simulations with different PerformanceConfig
presets and verify that delta_total and phi_total are bit-identical when
only semantics-preserving optimizations differ.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from bilancio.core.performance import PerformanceConfig


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
    from bilancio.config.models import RingExplorerGeneratorConfig
    from bilancio.scenarios.ring.compiler import _to_yaml_ready, compile_ring_explorer

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
    return _to_yaml_ready(compile_ring_explorer(gen_config, source_path=None))


def _run_small_ring(preset: str, seed: int = 42, **perf_overrides: Any) -> dict[str, Any]:
    """Run a small ring simulation and return metrics."""
    import tempfile

    from bilancio.ui.run import run_scenario

    scenario = _compile_ring_scenario(
        n_agents=10,
        kappa=Decimal("1"),
        concentration=Decimal("1"),
        mu=Decimal("0"),
        seed=seed,
        maturity_days=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        scenario_path = tmppath / "scenario.yaml"
        scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False, allow_unicode=False))

        events_path = tmppath / "events.jsonl"

        perf = PerformanceConfig.create(preset, **perf_overrides)

        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            run_scenario(
                path=scenario_path,
                mode="until_stable",
                max_days=20,
                quiet_days=2,
                show="none",
                check_invariants="end",
                export={"events_jsonl": str(events_path)},
                default_handling="expel-agent",
                performance=perf,
            )

        # Parse events and compute metrics
        events = []
        with events_path.open() as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        from bilancio.analysis.report import compute_day_metrics, summarize_day_metrics

        result = compute_day_metrics(events=events, balances_rows=None, day_list=None)
        summary = summarize_day_metrics(result["day_metrics"])
        return {
            "delta_total": summary.get("delta_total"),
            "phi_total": summary.get("phi_total"),
            "max_day": summary.get("max_day"),
        }


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
        compatible = _run_small_ring("compatible")
        urgency_metrics = _run_small_ring("compatible", matching_order="urgency")

        # This test documents the distinction -- urgency order MAY produce
        # different results. We don't assert they're different (they might
        # be the same for this particular scenario), we just document it.
        # The key assertion is that the test runs successfully.
        assert urgency_metrics["delta_total"] is not None
        assert urgency_metrics["phi_total"] is not None
