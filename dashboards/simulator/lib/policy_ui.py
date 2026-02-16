"""Policy-aware UI helpers.

Maps PolicyEngine constraints to available operations per agent kind,
so the UI only shows valid actions for each agent type.
"""

from __future__ import annotations

from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.policy import PolicyEngine

# Build the default policy once at import time
_DEFAULT_POLICY = PolicyEngine.default()

# Agent kinds available in the simulator (Layer 1)
AGENT_KINDS = ["central_bank", "bank", "firm", "household", "treasury"]

# Human-readable labels for agent kinds
AGENT_KIND_LABELS: dict[str, str] = {
    "central_bank": "Central Bank",
    "bank": "Bank",
    "firm": "Firm",
    "household": "Household",
    "treasury": "Treasury",
}

# Operations available per agent kind in the UI
_OPERATIONS: dict[str, list[str]] = {
    "central_bank": ["mint_cash", "mint_reserves"],
    "bank": ["create_cb_loan"],
    "firm": ["create_payable"],
    "household": ["create_payable"],
    "treasury": ["create_payable"],
}


def available_operations(agent_kind: str) -> list[str]:
    """Return operation names available for this agent kind."""
    return _OPERATIONS.get(agent_kind, [])


def can_receive_cash(agent_kind: str) -> bool:
    """Check if this agent kind can receive (hold) cash."""
    # All agent types in L1 can hold cash
    return agent_kind in AGENT_KINDS


def can_receive_reserves(agent_kind: str) -> bool:
    """Check if this agent kind can receive reserves."""
    # Only banks and treasury can hold reserves
    return agent_kind in ("bank", "treasury", "central_bank")


def can_issue_payable(agent_kind: str) -> bool:
    """Check if this agent kind can issue payables (be a debtor)."""
    # Any agent can issue payables
    return agent_kind in AGENT_KINDS


def can_create_cb_loan(agent_kind: str) -> bool:
    """Check if this agent kind can take a CB loan."""
    return agent_kind == "bank"


def agents_that_can_receive_cash() -> list[str]:
    """Return agent kinds that can hold cash."""
    return [k for k in AGENT_KINDS if can_receive_cash(k)]


def agents_that_can_receive_reserves() -> list[str]:
    """Return agent kinds that can hold reserves."""
    return [k for k in AGENT_KINDS if can_receive_reserves(k)]
