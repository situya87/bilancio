"""Stateless jurisdiction utility functions.

These helpers query jurisdiction and FX data stored on ``system.state``
without mutating anything. They are building blocks for future
jurisdiction-aware settlement.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bilancio.engines.system import System
    from bilancio.domain.jurisdiction import (
        CapitalControlAction,
        FXMarket,
        Jurisdiction,
    )


def get_jurisdiction_for_agent(system: System, agent_id: str) -> Jurisdiction | None:
    """Return the jurisdiction object for an agent, or ``None``."""
    agent = system.state.agents.get(agent_id)
    if agent is None or agent.jurisdiction_id is None:
        return None
    return system.state.jurisdictions.get(agent.jurisdiction_id)


def get_agent_domestic_currency(system: System, agent_id: str) -> str:
    """Return the domestic currency for an agent.

    Falls back to ``"X"`` (the default denomination) when the agent
    has no jurisdiction assigned.
    """
    j = get_jurisdiction_for_agent(system, agent_id)
    if j is None:
        return "X"
    return j.domestic_currency


def are_same_jurisdiction(system: System, agent_id_a: str, agent_id_b: str) -> bool:
    """Check whether two agents belong to the same jurisdiction.

    Returns ``True`` when both agents have the same ``jurisdiction_id``
    (including when both are ``None``).
    """
    a = system.state.agents.get(agent_id_a)
    b = system.state.agents.get(agent_id_b)
    if a is None or b is None:
        return False
    return a.jurisdiction_id == b.jurisdiction_id


def validate_same_denomination(system: System, instr_id_a: str, instr_id_b: str) -> bool:
    """Return ``True`` if two instruments share the same denomination."""
    ca = system.state.contracts.get(instr_id_a)
    cb = system.state.contracts.get(instr_id_b)
    if ca is None or cb is None:
        return False
    return ca.denom == cb.denom


def fx_convert(fx_market: Any, amount: int, from_currency: str, to_currency: str) -> int:
    """Convert an amount through the FX market.

    Thin wrapper around ``FXMarket.convert`` that handles the
    same-currency identity case.

    Args:
        fx_market: An ``FXMarket`` instance.
        amount: Amount in minor units.
        from_currency: Source currency code.
        to_currency: Target currency code.

    Returns:
        Converted amount in minor units.
    """
    if from_currency == to_currency:
        return amount
    return fx_market.convert(amount, from_currency, to_currency)


def check_capital_controls(
    jurisdiction: Any,
    purpose: str,
    direction: str,
) -> tuple[str, Decimal]:
    """Evaluate capital controls for a jurisdiction.

    Args:
        jurisdiction: A ``Jurisdiction`` instance.
        purpose: Capital flow purpose (e.g., ``"TRADE"``).
        direction: ``"inflow"`` or ``"outflow"``.

    Returns:
        Tuple of (action_str, tax_rate).
    """
    from bilancio.domain.jurisdiction import CapitalFlowPurpose

    purpose_enum = CapitalFlowPurpose(purpose)
    action, tax_rate = jurisdiction.capital_controls.evaluate(purpose_enum, direction)
    return str(action), tax_rate


def check_reserve_requirement(
    system: System, bank_id: str
) -> tuple[bool, int, int]:
    """Check whether a bank meets its reserve requirement.

    Args:
        system: The simulation system.
        bank_id: The bank agent ID.

    Returns:
        Tuple of (compliant, actual_reserves, required_reserves).
        If the bank has no jurisdiction or the requirement ratio is 0,
        the bank is always compliant.
    """
    from bilancio.domain.instruments.base import InstrumentKind

    j = get_jurisdiction_for_agent(system, bank_id)
    if j is None or j.banking_rules.reserve_requirement_ratio == 0:
        return True, 0, 0

    # Sum reserve deposits held by this bank
    actual_reserves = 0
    for cid in system.state.agents[bank_id].asset_ids:
        c = system.state.contracts.get(cid)
        if c is not None and c.kind == InstrumentKind.RESERVE_DEPOSIT:
            actual_reserves += c.amount

    # Sum deposit liabilities issued by this bank (bank deposits)
    total_deposits = 0
    for cid in system.state.agents[bank_id].liability_ids:
        c = system.state.contracts.get(cid)
        if c is not None and c.kind == InstrumentKind.BANK_DEPOSIT:
            total_deposits += c.amount

    ratio = j.banking_rules.reserve_requirement_ratio
    required_reserves = int(Decimal(total_deposits) * ratio)
    compliant = actual_reserves >= required_reserves
    return compliant, actual_reserves, required_reserves
