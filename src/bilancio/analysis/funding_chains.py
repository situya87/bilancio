"""Funding-chain analysis — trace cash sources and liquidity provision.

For each agent, trace where its cash came from (endowment, ticket sales,
loan proceeds, CB funding) and identify the ultimate liquidity providers
in the system. This answers: "who funded what with what?"

All functions consume the standard event log (list of dicts) produced by
the simulation engine.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

Event = dict[str, Any]
AgentId = str


def cash_inflows_by_source(
    events: list[Event],
) -> dict[AgentId, dict[str, Decimal]]:
    """Aggregate cash inflows per agent by source type.

    Source categories:
    - "settlement_received": payables settled where agent is creditor
    - "ticket_sale": dealer trade where agent sold a ticket
    - "loan_received": bank loan disbursed to agent
    - "cb_loan": central bank loan to agent (keyed by bank_id)
    - "nbfi_loan": non-bank lender loan received
    - "deposit_interest": interest credited on deposits
    - "deposit": internal cash-to-deposit transformation

    Returns:
        {agent_id: {"endowment": Decimal, "settlement_received": Decimal, ...}}
    """
    inflows: dict[AgentId, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: Decimal(0))
    )

    for e in events:
        kind = e.get("kind", "")
        amt = Decimal(str(e.get("amount", 0)))

        if kind == "PayableSettled":
            creditor = e.get("creditor") or e.get("to")
            if creditor:
                inflows[creditor]["settlement_received"] += amt

        elif kind == "dealer_trade" and e.get("side") == "sell":
            seller = e.get("trader")
            price = Decimal(str(e.get("price", e.get("amount", 0))))
            if seller:
                inflows[seller]["ticket_sale"] += price

        elif kind == "BankLoanIssued":
            borrower = e.get("borrower") or e.get("to")
            if borrower:
                inflows[borrower]["loan_received"] += amt

        elif kind == "CBLoanCreated":
            borrower = e.get("bank_id") or e.get("borrower") or e.get("to")
            if borrower:
                inflows[borrower]["cb_loan"] += amt

        elif kind == "DepositInterest":
            depositor = e.get("depositor") or e.get("to")
            interest = Decimal(str(e.get("interest", 0)))
            if depositor and interest > 0:
                inflows[depositor]["deposit_interest"] += interest

        elif kind == "CashDeposited":
            # CashDeposited is an internal cash-to-deposit transformation
            # (banking.py), not new external funding.
            customer = e.get("customer") or e.get("depositor") or e.get("from")
            if customer:
                inflows[customer]["deposit"] += amt

        elif kind == "NonBankLoanCreated":
            borrower = e.get("borrower_id") or e.get("borrower") or e.get("to")
            if borrower:
                inflows[borrower]["nbfi_loan"] += amt

    return dict(inflows)


def cash_outflows_by_type(
    events: list[Event],
) -> dict[AgentId, dict[str, Decimal]]:
    """Aggregate cash outflows per agent by type.

    Outflow categories:
    - "settlement_paid": payables settled where agent is debtor
    - "ticket_purchase": dealer trade where agent bought a ticket
    - "loan_repaid": loan repayment
    - "cb_repaid": CB loan repayment

    Returns:
        {agent_id: {"settlement_paid": Decimal, "ticket_purchase": Decimal, ...}}
    """
    outflows: dict[AgentId, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: Decimal(0))
    )

    for e in events:
        kind = e.get("kind", "")
        amt = Decimal(str(e.get("amount", 0)))

        if kind == "PayableSettled":
            debtor = e.get("debtor") or e.get("from")
            if debtor:
                outflows[debtor]["settlement_paid"] += amt

        elif kind == "dealer_trade" and e.get("side") == "buy":
            buyer = e.get("trader")
            price = Decimal(str(e.get("price", e.get("amount", 0))))
            if buyer:
                outflows[buyer]["ticket_purchase"] += price

    return dict(outflows)


def liquidity_providers(
    events: list[Event],
) -> list[dict[str, Any]]:
    """Identify the ultimate liquidity providers in the system.

    A liquidity provider is an agent whose net cash outflows (to others via
    settlements) exceed its net cash inflows from non-endowment sources.
    These agents are the ones whose initial endowment "funded" the system.

    Returns list of dicts sorted by net_provision descending:
        [{"agent_id": str, "total_provided": Decimal, "total_received": Decimal,
          "net_provision": Decimal}]
    """
    inflows = cash_inflows_by_source(events)
    outflows = cash_outflows_by_type(events)

    providers: list[dict[str, Any]] = []
    all_agents = set(inflows.keys()) | set(outflows.keys())

    for agent_id in all_agents:
        total_in = sum(inflows.get(agent_id, {}).values(), Decimal(0))
        total_out = sum(outflows.get(agent_id, {}).values(), Decimal(0))
        providers.append({
            "agent_id": agent_id,
            "total_provided": total_out,
            "total_received": total_in,
            "net_provision": total_out - total_in,
        })

    providers.sort(key=lambda x: x["net_provision"], reverse=True)
    return providers


def funding_mix(
    events: list[Event],
    agent_id: str,
) -> dict[str, Decimal]:
    """Return the funding mix for a single agent as fractions.

    Returns dict mapping source type -> fraction of total inflows.
    Example: {"endowment": Decimal("0.6"), "ticket_sale": Decimal("0.3"), ...}
    """
    all_inflows = cash_inflows_by_source(events)
    agent_inflows = all_inflows.get(agent_id, {})
    total = sum(agent_inflows.values(), Decimal(0))
    if total == 0:
        return {}
    return {source: amount / total for source, amount in agent_inflows.items()}
