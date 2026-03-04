"""Performance optimization configuration.

Controls which optimizations are active during simulation. Each flag
corresponds to one optimization option (A–I) from the performance
analysis. All flags default to off (``compatible`` preset) so existing
behaviour is preserved unless explicitly opted-in.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bilancio.engines.system import System


_VALID_MATCHING_ORDERS = frozenset({"random", "urgency"})
_VALID_DEALER_BACKENDS = frozenset({"python", "native"})

# Boolean flag names (order matches dataclass field order)
_BOOL_FLAGS = (
    "fast_atomic",
    "prune_ineligible",
    "cache_dealer_quotes",
    "preview_buy",
    "dirty_bucket_recompute",
    "incremental_intentions",
    "targeted_undo",
)

# Flags enabled by each preset (beyond compatible defaults)
_PRESET_FAST: dict[str, Any] = {
    "fast_atomic": True,
    "cache_dealer_quotes": True,
    "dirty_bucket_recompute": True,
}

_PRESET_AGGRESSIVE: dict[str, Any] = {
    "fast_atomic": True,
    "prune_ineligible": True,
    "cache_dealer_quotes": True,
    "preview_buy": True,
    "dirty_bucket_recompute": True,
    "incremental_intentions": True,
    # targeted_undo deliberately excluded (too risky)
    # matching_order stays "random" (semantics-changing)
}

_PRESETS: dict[str, dict[str, Any]] = {
    "compatible": {},
    "fast": _PRESET_FAST,
    "aggressive": _PRESET_AGGRESSIVE,
}


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance tuning knobs for the simulation engine.

    Preset resolution: the preset provides base values, then any
    explicitly-passed keyword arguments override them.  Use the
    ``create()`` classmethod when you need override semantics (the
    plain constructor also works but cannot distinguish "user passed
    False" from "default False").

    Attributes:
        fast_atomic: (A) Disable deepcopy in phases that never rollback.
        prune_ineligible: (B) Skip agents with no tickets and no cash.
        cache_dealer_quotes: (C) Snapshot/restore dealer state on reversal
            instead of recomputing from scratch.
        matching_order: (D) ``"random"`` (default) or ``"urgency"``
            (sort sellers by stress, buyers by surplus).
        dealer_backend: (E) ``"python"`` (default) or ``"native"`` (Rust).
        preview_buy: (F) Preview-then-commit buy path instead of
            execute-then-reverse.
        dirty_bucket_recompute: (G) Only recompute dealer state for
            buckets that had a successful trade.
        incremental_intentions: (H) Maintain persistent intention queues
            across rounds; only re-evaluate changed traders.
        targeted_undo: (I) Field-level undo log instead of deepcopy.
        preset: ``"compatible"`` | ``"fast"`` | ``"aggressive"``.
    """

    fast_atomic: bool = False              # A
    prune_ineligible: bool = False         # B
    cache_dealer_quotes: bool = False      # C
    matching_order: str = "random"         # D: "random" | "urgency"
    dealer_backend: str = "python"         # E: "python" | "native"
    preview_buy: bool = False              # F
    dirty_bucket_recompute: bool = False   # G
    incremental_intentions: bool = False   # H
    targeted_undo: bool = False            # I
    preset: str = "compatible"             # "compatible" | "fast" | "aggressive"

    def __post_init__(self) -> None:
        if self.preset not in _PRESETS:
            raise ValueError(
                f"Unknown performance preset {self.preset!r}; "
                f"valid options: {sorted(_PRESETS)}"
            )
        if self.matching_order not in _VALID_MATCHING_ORDERS:
            raise ValueError(
                f"Invalid matching_order {self.matching_order!r}; "
                f"valid options: {sorted(_VALID_MATCHING_ORDERS)}"
            )
        if self.dealer_backend not in _VALID_DEALER_BACKENDS:
            raise ValueError(
                f"Invalid dealer_backend {self.dealer_backend!r}; "
                f"valid options: {sorted(_VALID_DEALER_BACKENDS)}"
            )

    @classmethod
    def create(cls, preset: str = "compatible", **overrides: Any) -> PerformanceConfig:
        """Build a config from *preset* with explicit *overrides*.

        This is the preferred constructor when callers need "preset
        provides defaults, explicit kwargs always win" semantics::

            PerformanceConfig.create("aggressive", fast_atomic=False)
            # → fast_atomic=False even though aggressive sets True
        """
        base = dict(_PRESETS.get(preset, {}))
        base.update(overrides)
        base["preset"] = preset
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON/YAML-safe)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceConfig:
        """Deserialize from a plain dict.

        Uses ``create()`` so that preset + explicit overrides work
        correctly.  Unknown keys are silently ignored so
        forward-compatible YAML files don't break older code.
        """
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        preset = filtered.pop("preset", "compatible")
        return cls.create(preset, **filtered)


# ---------------------------------------------------------------------------
# Helper: fast_atomic scope
# ---------------------------------------------------------------------------

@contextmanager
def fast_atomic_scope(system: System, enabled: bool) -> Generator[None, None, None]:
    """Temporarily disable ``atomic()`` deepcopy snapshots.

    When *enabled* is True, sets ``system._atomic_disabled = True`` for
    the duration of the block, then restores the previous value.

    This is safe for phases that are read-only or have independent,
    non-rollback-able operations (rating, rollover, CB interest).
    """
    if not enabled:
        yield
        return

    prev = getattr(system, "_atomic_disabled", False)
    system._atomic_disabled = True
    try:
        yield
    finally:
        system._atomic_disabled = prev
