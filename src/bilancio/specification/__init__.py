"""
Specification validation framework for agents and instruments.

**Design-time documentation and validation tool** — not a runtime
enforcement mechanism.  Use this module to check that agent and
instrument specifications are complete and consistent.  Runtime
policy enforcement is handled by :class:`bilancio.domain.policy.PolicyEngine`.

This module provides:
- Data structures for defining agent and instrument specifications
- Registry for tracking all agents and instruments
- Completeness checkers to validate specifications
- Relationship matrix validation

Usage:
    from bilancio.specification import (
        AgentSpec, InstrumentSpec, SpecificationRegistry,
        validate_agent_completeness, validate_instrument_completeness,
    )
"""

from .models import (
    AgentRelation,
    AgentSpec,
    BalanceSheetPosition,
    DecisionSpec,
    InstrumentRelation,
    InstrumentSpec,
    LifecycleSpec,
)
from .registry import SpecificationRegistry
from .validators import (
    ValidationError,
    ValidationResult,
    validate_agent_completeness,
    validate_all_relationships,
    validate_instrument_completeness,
)

__all__ = [
    # Models
    "AgentSpec",
    "InstrumentSpec",
    "InstrumentRelation",
    "AgentRelation",
    "DecisionSpec",
    "LifecycleSpec",
    "BalanceSheetPosition",
    # Registry
    "SpecificationRegistry",
    # Validators
    "validate_agent_completeness",
    "validate_instrument_completeness",
    "validate_all_relationships",
    "ValidationResult",
    "ValidationError",
]
