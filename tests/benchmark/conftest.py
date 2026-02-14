"""Fixtures for performance benchmarks."""

from decimal import Decimal

import pytest

from bilancio.config.apply import apply_to_system
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.config.loaders import preprocess_config
from bilancio.engines.system import System
from bilancio.scenarios.ring_explorer import compile_ring_explorer


@pytest.fixture
def ring_system_10():
    """Create a System with 10 agents in a ring, ready for simulation.

    Returns a (system, config) tuple so tests can inspect the scenario config
    if needed.
    """
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "benchmark",
            "params": {
                "n_agents": 10,
                "kappa": "1",
                "seed": 42,
                "Q_total": "1000",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": False},
        }
    )
    scenario_dict = compile_ring_explorer(gen_config, source_path=None)
    scenario_dict = preprocess_config(scenario_dict)
    config = ScenarioConfig(**scenario_dict)

    system = System()
    apply_to_system(config, system)
    return system, config
