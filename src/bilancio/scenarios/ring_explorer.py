"""Backward-compatibility shim — the real code lives in ``ring.compiler``."""

from bilancio.scenarios.ring.compiler import (  # noqa: F401
    InequalitySpec,
    LiquiditySpec,
    MaturitySpec,
    RingExplorerParams,
    _allocate_liquidity,
    _apply_monotonicity,
    _build_agents,
    _build_due_days,
    _draw_payables,
    _emit_yaml,
    _ensure_positive_amounts,
    _fmt_decimal,
    _render_description,
    _render_scenario_name,
    _slugify,
    _to_yaml_ready,
    compile_ring_explorer,
    compile_ring_explorer_balanced,
)

__all__ = ["compile_ring_explorer", "compile_ring_explorer_balanced"]
