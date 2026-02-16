"""Scenario plugin protocol and supporting dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(frozen=True)
class ParameterDimension:
    """Describes a single tunable parameter dimension for a scenario.

    Used by the CLI to display parameter tables and by generic sampling
    to generate parameter dicts without knowing the scenario specifics.
    """

    name: str
    """Internal name used as dict key (e.g. ``'kappa'``)."""

    display_name: str
    """Human-friendly label (e.g. ``'Liquidity ratio (kappa)'``)."""

    description: str
    """One-line description of what this parameter controls."""

    default_values: list[Decimal] = field(default_factory=list)
    """Default grid values for sweep exploration."""

    valid_range: tuple[Decimal | None, Decimal | None] = (None, None)
    """Hard bounds ``(low, high)`` — values outside are rejected."""

    warn_range: tuple[Decimal | None, Decimal | None] = (None, None)
    """Soft bounds — values outside trigger a warning but are allowed."""


@dataclass(frozen=True)
class ScenarioMetadata:
    """Introspectable metadata for a scenario plugin."""

    name: str
    """Machine identifier (e.g. ``'kalecki_ring'``)."""

    display_name: str
    """Human label (e.g. ``'Kalecki Ring'``)."""

    description: str
    """Short paragraph explaining the scenario."""

    version: int
    """Schema/API version number."""

    instruments_used: list[str] = field(default_factory=list)
    """Instrument kinds this scenario creates (e.g. ``['Payable', 'Cash']``)."""

    agent_types: list[str] = field(default_factory=list)
    """Agent types this scenario creates (e.g. ``['Household', 'CentralBank']``)."""

    supports_dealer: bool = False
    """Whether this scenario supports the dealer/VBT trading subsystem."""

    supports_lender: bool = False
    """Whether this scenario supports the non-bank lender subsystem."""


@runtime_checkable
class ScenarioPlugin(Protocol):
    """Protocol that all scenario plugins must satisfy.

    A scenario plugin knows how to:

    1. Describe itself (``metadata``, ``parameter_dimensions``)
    2. Compile a set of numeric parameters into a runnable scenario dict
    3. Expose its Pydantic config model for validation
    """

    @property
    def metadata(self) -> ScenarioMetadata:
        """Return static metadata about this scenario."""
        ...

    def parameter_dimensions(self) -> list[ParameterDimension]:
        """Return the list of tunable parameter dimensions."""
        ...

    def compile(
        self,
        params: dict[str, Decimal],
        *,
        base_config: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """Compile parameter values into a runnable scenario dictionary.

        Args:
            params: Mapping of dimension name -> value (e.g. ``{'kappa': Decimal('0.5')}``).
            base_config: Non-sweep configuration (n_agents, maturity_days, etc.).
            seed: PRNG seed for reproducibility.

        Returns:
            A scenario dictionary consumable by the simulation engine.
        """
        ...

    def config_model(self) -> type[BaseModel]:
        """Return the Pydantic model class used for full config validation."""
        ...
