"""Bridge layer: translates UI operations into System method calls.

Each function performs the operation on the System and returns an action dict
that can be appended to the action log for undo/replay support.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from bilancio.config.apply import create_agent
from bilancio.config.models import AgentSpec
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.system import System


def _uid() -> str:
    """Generate a short unique id for instruments."""
    return uuid.uuid4().hex[:8]


# ── Agent operations ────────────────────────────────────────────


def add_agent(system: System, kind: str, agent_id: str, name: str) -> dict[str, Any]:
    """Create and register an agent in the system."""
    spec = AgentSpec(id=agent_id, kind=kind, name=name)
    agent = create_agent(spec)
    system.add_agent(agent)
    return {"type": "add_agent", "kind": kind, "agent_id": agent_id, "name": name}


# ── Cash / Reserves ────────────────────────────────────────────


def mint_cash(system: System, to_agent_id: str, amount: int) -> dict[str, Any]:
    """Mint cash from central bank to an agent."""
    system.mint_cash(to_agent_id, amount)
    return {"type": "mint_cash", "to_agent_id": to_agent_id, "amount": amount}


def mint_reserves(system: System, to_bank_id: str, amount: int) -> dict[str, Any]:
    """Mint reserves from central bank to a bank."""
    system.mint_reserves(to_bank_id, amount)
    return {"type": "mint_reserves", "to_bank_id": to_bank_id, "amount": amount}


# ── Instruments ─────────────────────────────────────────────────


def create_payable(
    system: System,
    debtor_id: str,
    creditor_id: str,
    amount: int,
    due_day: int,
) -> dict[str, Any]:
    """Create a payable (obligation) between two agents."""
    instr_id = f"pay-{_uid()}"
    payable = Payable(
        id=instr_id,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor_id,
        liability_issuer_id=debtor_id,
        due_day=due_day,
    )
    system.add_contract(payable)
    return {
        "type": "create_payable",
        "debtor_id": debtor_id,
        "creditor_id": creditor_id,
        "amount": amount,
        "due_day": due_day,
        "instr_id": instr_id,
    }


def create_cb_loan(
    system: System,
    bank_id: str,
    amount: int,
    rate: str = "0.03",
    issuance_day: int = 0,
) -> dict[str, Any]:
    """Create a central bank loan to a bank."""
    instr_id = f"cbl-{_uid()}"
    # Find the central bank id
    cb_id = _find_central_bank(system)
    loan = CBLoan(
        id=instr_id,
        kind=InstrumentKind.CB_LOAN,
        amount=amount,
        denom="X",
        asset_holder_id=cb_id,
        liability_issuer_id=bank_id,
        cb_rate=Decimal(rate),
        issuance_day=issuance_day,
    )
    system.add_contract(loan)
    return {
        "type": "create_cb_loan",
        "bank_id": bank_id,
        "amount": amount,
        "rate": rate,
        "issuance_day": issuance_day,
        "instr_id": instr_id,
    }


def _find_central_bank(system: System) -> str:
    """Find the central bank agent id in the system."""
    for aid, agent in system.state.agents.items():
        if agent.kind == "central_bank" or (hasattr(agent.kind, "value") and agent.kind.value == "central_bank"):
            return aid
    raise ValueError("No central bank found in system. Add a central bank first.")


# ── Replay (for undo) ──────────────────────────────────────────


def replay_action(system: System, action: dict[str, Any]) -> None:
    """Replay a single action dict onto a system. Used by rebuild_system."""
    t = action["type"]
    if t == "add_agent":
        add_agent(system, action["kind"], action["agent_id"], action["name"])
    elif t == "mint_cash":
        mint_cash(system, action["to_agent_id"], action["amount"])
    elif t == "mint_reserves":
        mint_reserves(system, action["to_bank_id"], action["amount"])
    elif t == "create_payable":
        # Re-create with the same instrument id for consistency
        instr_id = action["instr_id"]
        payable = Payable(
            id=instr_id,
            kind=InstrumentKind.PAYABLE,
            amount=action["amount"],
            denom="X",
            asset_holder_id=action["creditor_id"],
            liability_issuer_id=action["debtor_id"],
            due_day=action["due_day"],
        )
        system.add_contract(payable)
    elif t == "create_cb_loan":
        instr_id = action["instr_id"]
        cb_id = _find_central_bank(system)
        loan = CBLoan(
            id=instr_id,
            kind=InstrumentKind.CB_LOAN,
            amount=action["amount"],
            denom="X",
            asset_holder_id=cb_id,
            liability_issuer_id=action["bank_id"],
            cb_rate=Decimal(action["rate"]),
            issuance_day=action["issuance_day"],
        )
        system.add_contract(loan)
    else:
        raise ValueError(f"Unknown action type: {t}")
