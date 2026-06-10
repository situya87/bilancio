"""Phase C: interbank netting, the interbank auction record, and loan servicing.

Cross-bank client payments accumulated during the day are netted pairwise
and settled with a single reserve transfer per bank pair. In banking mode,
an ``InterbankAuction`` market-state record (positions and limit rates from
the corridor) is logged first. Central-bank loans two or more days old are
then repaid — refinancing automatically at the CB when reserves fall short,
or written off (defaulting the bank) when CB lending is frozen — followed
by non-bank and bank loan servicing.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from bilancio.engines.clean_core_types import CleanBankingConfig
from bilancio_v2.ledger import ZERO, Ledger
from bilancio_v2.plugins.banking import (
    bank_profile,
    repay_due_bank_loans,
    write_off_frozen_cb_loan,
)
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.plugins.lending import repay_due_non_bank_loans

CB_REFINANCE_RATE = Decimal("0.03")
CB_LOAN_GRACE_DAYS = 2


class InterbankPhase:
    name = "PhaseC"

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        impactful = False
        flows = net_interbank_flows(client_payment_flows_for_day(ledger))
        if ctx.banking_config is not None:
            ledger.log("SubphaseC_InterbankAuction")
            ledger.log_raw(
                "InterbankAuction",
                **interbank_auction_summary(ledger, ctx.banking_config, flows),
            )
        for (from_bank, to_bank), amount in flows.items():
            ledger.transfer_reserves(from_bank, to_bank, amount, always_merge=True)
            ledger.log("InterbankCleared", debtor_bank=from_bank, creditor_bank=to_bank, amount=amount)
            impactful = True
        repay_due_cb_loans(ledger)
        repay_due_non_bank_loans(ledger)
        repay_due_bank_loans(ledger)
        return impactful


def client_payment_flows_for_day(ledger: Ledger) -> dict[tuple[str, str], Decimal]:
    flows: defaultdict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    for event in ledger.journal.on_day(ledger.day, "ClientPayment"):
        payer_bank = event.data.get("payer_bank")
        payee_bank = event.data.get("payee_bank")
        if not payer_bank or not payee_bank or payer_bank == payee_bank:
            continue
        flows[(str(payer_bank), str(payee_bank))] += Decimal(str(event.data.get("amount", ZERO)))
    return dict(flows)


def net_interbank_flows(
    flows: dict[tuple[str, str], Decimal],
) -> dict[tuple[str, str], Decimal]:
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


def interbank_auction_summary(
    ledger: Ledger,
    banking_config: CleanBankingConfig,
    interbank_flows: dict[tuple[str, str], Decimal],
) -> dict[str, Any]:
    profile = bank_profile(banking_config)
    r_floor = profile.r_floor(banking_config.kappa, banking_config.mu, banking_config.concentration)
    r_ceiling = profile.r_ceiling(banking_config.kappa, banking_config.mu, banking_config.concentration)
    midpoint = (r_floor + r_ceiling) / 2
    width = r_ceiling - r_floor

    positions: dict[str, int] = {}
    limit_rates: dict[str, Decimal] = {}
    net_obligations: defaultdict[str, Decimal] = defaultdict(Decimal)
    for (from_bank, to_bank), amount in interbank_flows.items():
        net_obligations[from_bank] += amount
        net_obligations[to_bank] -= amount

    for bank_id, agent in sorted(ledger.agents.items()):
        if agent.kind != "bank" or bank_id in ledger.defaulted_agent_ids:
            continue
        reserve_target = banking_config.reserve_targets.get(bank_id, 1)
        position = int(ledger.reserves[bank_id] - net_obligations[bank_id]) - reserve_target
        normalized = Decimal(position) / Decimal(reserve_target)
        normalized = max(Decimal("-1"), min(Decimal("1"), normalized))
        positions[bank_id] = position
        limit_rates[bank_id] = midpoint - (width / 2) * normalized

    lender_asks = [
        {"bank_id": bank_id, "quantity": position, "limit_rate": str(limit_rates[bank_id])}
        for bank_id, position in positions.items()
        if position > 0
    ]
    lender_asks.sort(key=lambda order: Decimal(str(order["limit_rate"])))
    borrower_bids = [
        {"bank_id": bank_id, "quantity": abs(position), "limit_rate": str(limit_rates[bank_id])}
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


def repay_due_cb_loans(ledger: Ledger) -> None:
    for loan in list(ledger.cb_loans):
        if loan.settled or ledger.day < loan.issuance_day + CB_LOAN_GRACE_DAYS:
            continue
        if ledger.reserves[loan.bank] < loan.repayment_amount:
            if ledger.cb_lending_frozen:
                write_off_frozen_cb_loan(ledger, loan)
                continue
            ledger.refinance_cb_loan(loan.bank, loan.repayment_amount, rate=CB_REFINANCE_RATE)
        ledger.repay_cb_loan(loan)
