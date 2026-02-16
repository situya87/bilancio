"""KaleckiRingPlugin — wraps the existing ring compiler behind the ScenarioPlugin protocol."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from bilancio.config.models import (
    RingExplorerGeneratorConfig,
    RingExplorerInequalityConfig,
    RingExplorerLiquidityAllocation,
    RingExplorerLiquidityConfig,
    RingExplorerMaturityConfig,
    RingExplorerParamsModel,
)
from bilancio.scenarios.protocol import (
    ParameterDimension,
    ScenarioMetadata,
    ScenarioPlugin,
)

from .compiler import compile_ring_explorer, compile_ring_explorer_balanced

_METADATA = ScenarioMetadata(
    name="kalecki_ring",
    display_name="Kalecki Ring",
    description=(
        "Circular debt chain of N agents parameterised by liquidity ratio (kappa), "
        "debt inequality (concentration), and maturity skew (mu). "
        "Supports balanced VBT/Dealer per maturity bucket."
    ),
    version=1,
    instruments_used=["Payable", "Cash"],
    agent_types=["Household", "CentralBank"],
    supports_dealer=True,
    supports_lender=True,
)

_DIMENSIONS = [
    ParameterDimension(
        name="kappa",
        display_name="Liquidity ratio (kappa)",
        description="Ratio of initial cash to total debt (L0/S1). Lower = more stressed.",
        default_values=[Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4")],
        valid_range=(Decimal("0"), None),
        warn_range=(Decimal("0.05"), Decimal("10")),
    ),
    ParameterDimension(
        name="concentration",
        display_name="Debt concentration (c)",
        description="Dirichlet concentration for debt distribution. Lower = more unequal.",
        default_values=[Decimal("0.2"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("5")],
        valid_range=(Decimal("0"), None),
        warn_range=(Decimal("0.1"), Decimal("10")),
    ),
    ParameterDimension(
        name="mu",
        display_name="Maturity skew (mu)",
        description="Controls timing of due dates. 0 = front-loaded, 1 = back-loaded.",
        default_values=[Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1")],
        valid_range=(Decimal("0"), Decimal("1")),
        warn_range=(Decimal("0"), Decimal("1")),
    ),
    ParameterDimension(
        name="monotonicity",
        display_name="Monotonicity",
        description="Amount ordering. -1 = descending, 0 = random, 1 = ascending.",
        default_values=[Decimal("0")],
        valid_range=(Decimal("-1"), Decimal("1")),
        warn_range=(Decimal("-1"), Decimal("1")),
    ),
]


class KaleckiRingPlugin:
    """Scenario plugin for the Kalecki circular-debt ring."""

    @property
    def metadata(self) -> ScenarioMetadata:
        return _METADATA

    def parameter_dimensions(self) -> list[ParameterDimension]:
        return list(_DIMENSIONS)

    def compile(
        self,
        params: dict[str, Decimal],
        *,
        base_config: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """Build a scenario dict from sweep parameters + base config.

        ``params`` must contain at least ``kappa``.  Optional keys:
        ``concentration``, ``mu``, ``monotonicity``.

        ``base_config`` may contain:
        ``n_agents``, ``maturity_days``, ``Q_total``, ``liquidity_mode``,
        ``liquidity_agent``, ``name_prefix``, ``face_value``,
        ``outside_mid_ratio``, ``mode``, ``rollover_enabled``, ``lender_share``,
        ``vbt_share_per_bucket``, ``dealer_share_per_bucket``.

        If ``base_config["mode"]`` is present the balanced compiler is used;
        otherwise the basic compiler is called.
        """
        kappa = params["kappa"]
        concentration = params.get("concentration", Decimal("1"))
        mu = params.get("mu", Decimal("0"))
        monotonicity = params.get("monotonicity", Decimal("0"))

        n_agents = base_config.get("n_agents", 100)
        maturity_days = base_config.get("maturity_days", 10)
        Q_total = base_config.get("Q_total", Decimal("10000"))
        liquidity_mode = base_config.get("liquidity_mode", "uniform")
        liquidity_agent = base_config.get("liquidity_agent", "H1")
        name_prefix = base_config.get("name_prefix", "Kalecki Ring")

        gen_config = RingExplorerGeneratorConfig(
            version=1,
            generator="ring_explorer_v1",
            name_prefix=name_prefix,
            params=RingExplorerParamsModel(
                n_agents=n_agents,
                seed=seed,
                kappa=kappa,
                liquidity=RingExplorerLiquidityConfig(
                    total=None,
                    allocation=RingExplorerLiquidityAllocation(
                        mode=liquidity_mode,
                        agent=liquidity_agent if liquidity_mode == "single_at" else None,
                    ),
                ),
                inequality=RingExplorerInequalityConfig(
                    concentration=concentration,
                    monotonicity=monotonicity,
                ),
                maturity=RingExplorerMaturityConfig(
                    days=maturity_days,
                    mu=mu,
                ),
                Q_total=Q_total,
            ),
            compile={"out_dir": None, "emit_yaml": False},
        )

        # Use balanced compiler when a mode is specified
        mode = base_config.get("mode")
        if mode is not None:
            return compile_ring_explorer_balanced(
                gen_config,
                face_value=base_config.get("face_value", Decimal("20")),
                outside_mid_ratio=base_config.get("outside_mid_ratio", Decimal("0.75")),
                vbt_share_per_bucket=base_config.get("vbt_share_per_bucket", Decimal("0.20")),
                dealer_share_per_bucket=base_config.get("dealer_share_per_bucket", Decimal("0.05")),
                mode=mode,
                rollover_enabled=base_config.get("rollover_enabled", True),
                lender_share=base_config.get("lender_share", Decimal("0.10")),
                kappa=kappa,
            )

        return compile_ring_explorer(gen_config)

    def config_model(self) -> type[BaseModel]:
        return RingExplorerGeneratorConfig


# Verify protocol conformance at import time (cheap isinstance check).
assert isinstance(KaleckiRingPlugin(), ScenarioPlugin)
