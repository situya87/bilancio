"""Tests for the scenario plugin protocol, registry, and KaleckiRingPlugin."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.scenarios.protocol import (
    ParameterDimension,
    ScenarioMetadata,
    ScenarioPlugin,
)
from bilancio.scenarios.registry import get_plugin, get_registry, register_plugin, reset_registry
from bilancio.scenarios.ring.plugin import KaleckiRingPlugin


class TestScenarioMetadata:
    def test_frozen(self):
        meta = ScenarioMetadata(name="test", display_name="Test", description="desc", version=1)
        with pytest.raises(AttributeError):
            meta.name = "other"  # type: ignore[misc]

    def test_defaults(self):
        meta = ScenarioMetadata(name="x", display_name="X", description="d", version=1)
        assert meta.instruments_used == []
        assert meta.agent_types == []
        assert meta.supports_dealer is False
        assert meta.supports_lender is False


class TestParameterDimension:
    def test_frozen(self):
        dim = ParameterDimension(name="k", display_name="K", description="d")
        with pytest.raises(AttributeError):
            dim.name = "other"  # type: ignore[misc]

    def test_defaults(self):
        dim = ParameterDimension(name="k", display_name="K", description="d")
        assert dim.default_values == []
        assert dim.valid_range == (None, None)
        assert dim.warn_range == (None, None)


class TestKaleckiRingPlugin:
    def test_satisfies_protocol(self):
        plugin = KaleckiRingPlugin()
        assert isinstance(plugin, ScenarioPlugin)

    def test_metadata(self):
        plugin = KaleckiRingPlugin()
        meta = plugin.metadata
        assert meta.name == "kalecki_ring"
        assert meta.display_name == "Kalecki Ring"
        assert meta.version == 1
        assert "Payable" in meta.instruments_used
        assert "Cash" in meta.instruments_used
        assert meta.supports_dealer is True
        assert meta.supports_lender is True

    def test_parameter_dimensions(self):
        plugin = KaleckiRingPlugin()
        dims = plugin.parameter_dimensions()
        names = [d.name for d in dims]
        assert "kappa" in names
        assert "concentration" in names
        assert "mu" in names
        assert "monotonicity" in names

        kappa_dim = next(d for d in dims if d.name == "kappa")
        assert len(kappa_dim.default_values) > 0
        assert all(isinstance(v, Decimal) for v in kappa_dim.default_values)

    def test_compile_basic(self):
        plugin = KaleckiRingPlugin()
        scenario = plugin.compile(
            params={"kappa": Decimal("1"), "concentration": Decimal("1"), "mu": Decimal("0")},
            base_config={"n_agents": 5, "maturity_days": 3, "Q_total": Decimal("500")},
            seed=42,
        )
        assert scenario["version"] == 1
        assert "agents" in scenario
        assert "initial_actions" in scenario
        # Should have CB + 5 agents
        assert len(scenario["agents"]) == 6  # CB + H1..H5

    def test_compile_balanced(self):
        plugin = KaleckiRingPlugin()
        scenario = plugin.compile(
            params={"kappa": Decimal("0.5"), "concentration": Decimal("1"), "mu": Decimal("0.5")},
            base_config={
                "n_agents": 10,
                "maturity_days": 5,
                "Q_total": Decimal("1000"),
                "mode": "active",
            },
            seed=42,
        )
        assert scenario["version"] == 1
        assert "agents" in scenario
        # Should have VBT and dealer agents
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "vbt_short" in agent_ids
        assert "dealer_short" in agent_ids
        assert scenario["run"]["rollover_enabled"] is True

    def test_config_model(self):
        from pydantic import BaseModel

        plugin = KaleckiRingPlugin()
        model_cls = plugin.config_model()
        assert issubclass(model_cls, BaseModel)


class TestRegistry:
    def test_get_registry_returns_kalecki_ring(self):
        reg = get_registry()
        assert "kalecki_ring" in reg
        assert isinstance(reg["kalecki_ring"], ScenarioPlugin)

    def test_get_plugin(self):
        plugin = get_plugin("kalecki_ring")
        assert plugin.metadata.name == "kalecki_ring"

    def test_get_plugin_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown scenario plugin"):
            get_plugin("nonexistent_scenario")

    def test_register_plugin(self):
        """Test registering a custom plugin."""

        class _DummyPlugin:
            @property
            def metadata(self):
                return ScenarioMetadata(
                    name="dummy", display_name="Dummy", description="test", version=1
                )

            def parameter_dimensions(self):
                return []

            def compile(self, params, *, base_config, seed):
                return {}

            def config_model(self):
                from pydantic import BaseModel

                return BaseModel

        register_plugin("dummy", _DummyPlugin())
        try:
            assert "dummy" in get_registry()
            assert get_plugin("dummy").metadata.name == "dummy"
        finally:
            reset_registry()  # Clean up to avoid leaking into other tests

    def test_get_registry_returns_copy(self):
        """Mutating the returned dict should not affect the internal registry."""
        reg = get_registry()
        reg["fake"] = None  # type: ignore[assignment]
        assert "fake" not in get_registry()


class TestBackwardCompat:
    """Ensure existing import paths still work."""

    def test_import_from_ring_explorer(self):
        from bilancio.scenarios.ring_explorer import (
            compile_ring_explorer,
            compile_ring_explorer_balanced,
        )

        assert callable(compile_ring_explorer)
        assert callable(compile_ring_explorer_balanced)

    def test_import_private_from_ring_explorer(self):
        from bilancio.scenarios.ring_explorer import _draw_payables

        assert callable(_draw_payables)

    def test_import_from_scenarios_init(self):
        from bilancio.scenarios import (
            compile_ring_explorer,
            compile_ring_explorer_balanced,
        )

        assert callable(compile_ring_explorer)
        assert callable(compile_ring_explorer_balanced)

    def test_new_exports_from_scenarios_init(self):
        from bilancio.scenarios import (
            ParameterDimension,
            ScenarioMetadata,
            ScenarioPlugin,
            get_plugin,
            get_registry,
            register_plugin,
        )

        assert ParameterDimension is not None
        assert ScenarioMetadata is not None
        assert ScenarioPlugin is not None
        assert callable(get_plugin)
        assert callable(get_registry)
        assert callable(register_plugin)
