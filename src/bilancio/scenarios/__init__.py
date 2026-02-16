"""Scenario generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bilancio.config.models import (
    GeneratorConfig,
    RingExplorerGeneratorConfig,
)

# Backward-compatible re-exports (unchanged)
from .ring_explorer import (
    compile_ring_explorer,
    compile_ring_explorer_balanced,
)

# New plugin API
from .protocol import ParameterDimension, ScenarioMetadata, ScenarioPlugin
from .registry import get_plugin, get_registry, register_plugin, reset_registry


def compile_generator(
    config: GeneratorConfig,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Compile a generator specification into a scenario dictionary."""
    if isinstance(config, RingExplorerGeneratorConfig):
        return compile_ring_explorer(config, source_path=source_path)
    raise ValueError(f"Unsupported generator '{getattr(config, 'generator', 'unknown')}'")


__all__ = [
    # Backward compatible
    "compile_generator",
    "compile_ring_explorer",
    "compile_ring_explorer_balanced",
    # Plugin API
    "ParameterDimension",
    "ScenarioMetadata",
    "ScenarioPlugin",
    "get_plugin",
    "get_registry",
    "register_plugin",
    "reset_registry",
]
