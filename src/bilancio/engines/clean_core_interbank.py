"""Interbank settlement helpers for the clean-core scenario engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from bilancio.decision.profiles import BankProfile
from bilancio.engines.clean_core_types import CleanBankingConfig, CleanState

ZERO = Decimal("0")


def primary_bank_for_customer(state: CleanState, customer_id: str) -> str | None:
    for (candidate_customer, bank_id), amount in state.deposits.items():
        if candidate_customer == customer_id and amount > 0:
            return bank_id
    return None


def client_payment_flows_for_day(state: CleanState) -> dict[tuple[str, str], Decimal]:
    flows: defaultdict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    for event in state.events:
        if event.get("day") != state.day or event.get("kind") != "ClientPayment":
            continue
        payer_bank = event.get("payer_bank")
        payee_bank = event.get("payee_bank")
        if not payer_bank or not payee_bank or payer_bank == payee_bank:
            continue
        flows[(str(payer_bank), str(payee_bank))] += Decimal(str(event.get("amount", ZERO)))
    return dict(flows)


def net_interbank_flows(flows: dict[tuple[str, str], Decimal]) -> dict[tuple[str, str], Decimal]:
    net_by_pair: dict[tuple[str, str], Decimal] = {}
    visited: set[tuple[str, str]] = set()
    for (from_bank, to_bank), amount in flows.items():
        if (from_bank, to_bank) in visited:
            continue
        reverse = flows.get((to_bank, from_bank), ZERO)
        net = amount - reverse
        visited.add((from_bank, to_bank))
        visited.add((to_bank, from_bank))
        if net > 0:
            net_by_pair[(from_bank, to_bank)] = net
        elif net < 0:
            net_by_pair[(to_bank, from_bank)] = -net
    return net_by_pair


def initial_banking_reserve_targets(
    state: CleanState,
    banking_config: CleanBankingConfig,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for bank_id, agent in state.agents.items():
        if agent.kind != "bank":
            continue
        deposits = sum(
            amount
            for (_customer_id, candidate_bank), amount in state.deposits.items()
            if candidate_bank == bank_id
        )
        targets[bank_id] = max(1, int(banking_config.reserve_target_ratio * deposits))
    return targets


def clean_interbank_auction_summary(
    state: CleanState,
    banking_config: CleanBankingConfig,
    interbank_flows: dict[tuple[str, str], Decimal],
) -> dict[str, Any]:
    profile = BankProfile(adaptive_corridor=banking_config.adaptive_corridor)
    r_floor = profile.r_floor(
        banking_config.kappa,
        banking_config.mu,
        banking_config.concentration,
    )
    r_ceiling = profile.r_ceiling(
        banking_config.kappa,
        banking_config.mu,
        banking_config.concentration,
    )
    midpoint = (r_floor + r_ceiling) / 2
    width = r_ceiling - r_floor

    positions: dict[str, int] = {}
    limit_rates: dict[str, Decimal] = {}
    net_obligations: Counter[str] = Counter()
    for (from_bank, to_bank), amount in interbank_flows.items():
        net_obligations[from_bank] += amount
        net_obligations[to_bank] -= amount

    for bank_id, agent in sorted(state.agents.items()):
        if agent.kind != "bank" or bank_id in state.defaulted_agent_ids:
            continue
        reserve_target = banking_config.reserve_targets.get(bank_id, 1)
        position = int(state.reserves[bank_id] - net_obligations[bank_id]) - reserve_target
        normalized = Decimal(position) / Decimal(reserve_target)
        normalized = max(Decimal("-1"), min(Decimal("1"), normalized))
        positions[bank_id] = position
        limit_rates[bank_id] = midpoint - (width / 2) * normalized

    lender_asks = [
        {
            "bank_id": bank_id,
            "quantity": position,
            "limit_rate": str(limit_rates[bank_id]),
        }
        for bank_id, position in positions.items()
        if position > 0
    ]
    lender_asks.sort(key=lambda order: Decimal(str(order["limit_rate"])))
    borrower_bids = [
        {
            "bank_id": bank_id,
            "quantity": abs(position),
            "limit_rate": str(limit_rates[bank_id]),
        }
        for bank_id, position in positions.items()
        if position < 0
    ]
    borrower_bids.sort(key=lambda order: Decimal(str(order["limit_rate"])), reverse=True)

    market_state = {
        "positions": [
            {
                "bank_id": bank_id,
                "position": position,
                "limit_rate": str(limit_rates[bank_id]),
                "side": "lend" if position > 0 else "borrow" if position < 0 else "flat",
            }
            for bank_id, position in positions.items()
        ],
        "lender_asks": lender_asks,
        "borrower_bids": borrower_bids,
    }
    return {
        "clearing_rate": None,
        "total_volume": 0,
        "n_trades": 0,
        "n_unfilled": 0,
        "market_state": market_state,
    }
