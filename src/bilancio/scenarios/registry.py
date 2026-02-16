"""Simple plugin registry for scenario types.

The registry is lazily populated on first access so that importing
``protocol.py`` does not pull in any concrete plugin code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.scenarios.protocol import ScenarioPlugin

_registry: dict[str, ScenarioPlugin] | None = None


def _build_registry() -> dict[str, ScenarioPlugin]:
    """Import known plugins and return the initial registry."""
    from bilancio.scenarios.ring.plugin import KaleckiRingPlugin

    return {
        "kalecki_ring": KaleckiRingPlugin(),
    }


def _ensure_registry() -> dict[str, ScenarioPlugin]:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def get_registry() -> dict[str, ScenarioPlugin]:
    """Return the full ``{name: plugin}`` mapping (lazily initialised)."""
    return dict(_ensure_registry())


def get_plugin(name: str) -> ScenarioPlugin:
    """Look up a plugin by name.

    Raises:
        KeyError: If no plugin with *name* is registered.
    """
    reg = _ensure_registry()
    if name not in reg:
        available = ", ".join(sorted(reg))
        raise KeyError(f"Unknown scenario plugin {name!r}. Available: {available}")
    return reg[name]


def register_plugin(name: str, plugin: ScenarioPlugin) -> None:
    """Register a new plugin (or override an existing one).

    This is the public extension point — third-party or experimental
    scenarios call this at import time to make themselves discoverable.
    """
    reg = _ensure_registry()
    reg[name] = plugin
