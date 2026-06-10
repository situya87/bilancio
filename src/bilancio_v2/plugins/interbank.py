"""Phase C: bilateral interbank netting and central-bank loan servicing.

Cross-bank client payments accumulated during the day are netted pairwise
and settled with a single reserve transfer per bank pair. Central-bank loans
two or more days old are then repaid, refinancing automatically at the CB
when the bank's reserves fall short (matching the existing engine).
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from bilancio_v2.ledger import ZERO, Ledger
from bilancio_v2.plugins.base import RunContext

CB_REFINANCE_RATE = Decimal("0.03")
CB_LOAN_GRACE_DAYS = 2


class InterbankPhase:
    name = "PhaseC"

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        impactful = False
        flows = net_interbank_flows(client_payment_flows_for_day(ledger))
        for (from_bank, to_bank), amount in flows.items():
            ledger.transfer_reserves(from_bank, to_bank, amount, always_merge=True)
            ledger.log("InterbankCleared", debtor_bank=from_bank, creditor_bank=to_bank, amount=amount)
            impactful = True
        repay_due_cb_loans(ledger)
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


def repay_due_cb_loans(ledger: Ledger) -> None:
    for loan in list(ledger.cb_loans):
        if loan.settled or ledger.day < loan.issuance_day + CB_LOAN_GRACE_DAYS:
            continue
        if ledger.reserves[loan.bank] < loan.repayment_amount:
            if ledger.cb_lending_frozen:
                raise NotImplementedError("v2 kernel does not yet support CB lending freeze write-offs (banking subsystem slice)")
            ledger.refinance_cb_loan(loan.bank, loan.repayment_amount, rate=CB_REFINANCE_RATE)
        ledger.repay_cb_loan(loan)
