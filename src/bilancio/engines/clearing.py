"""Clearing engine (Phase C) for intraday netting and settlement."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, cast

from bilancio.core.atomic_tx import atomic
from bilancio.core.events import EventKind
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable

if TYPE_CHECKING:
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)


def compute_intraday_nets(system: System, day: int) -> dict[tuple[str, str], int]:
    """
    Compute net amounts between banks from today's ClientPayment events.

    Scans events for ClientPayment events from today and calculates net amounts
    between each bank pair. Uses lexical ordering for bank pairs (a, b where a < b).

    Convention: nets[(a,b)] > 0 means bank a owes bank b

    Args:
        system: System instance
        day: Day to compute nets for

    Returns:
        Dict mapping bank pairs to net amounts
    """
    nets: defaultdict[tuple[str, str], int] = defaultdict(int)

    # Use indexed events for O(1) day lookup instead of O(n) full scan
    for event in system.state.events_by_day.get(day, []):
        if event.get("kind") == "ClientPayment":
            debtor_bank = str(event.get("payer_bank", ""))
            creditor_bank = str(event.get("payee_bank", ""))
            amount = cast(int, event.get("amount", 0))

            if debtor_bank and creditor_bank and debtor_bank != creditor_bank:
                # Use lexical ordering: ensure a < b
                if debtor_bank < creditor_bank:
                    # debtor_bank owes creditor_bank
                    nets[(debtor_bank, creditor_bank)] += amount
                else:
                    # creditor_bank is owed by debtor_bank, so subtract from the reverse pair
                    nets[(creditor_bank, debtor_bank)] -= amount

    return dict(nets)


def _settle_net_with_cb_fallback(
    system: System,
    debtor_bank: str,
    creditor_bank: str,
    amount: int,
    shortfall: int,
    day: int,
    *,
    label: str = "",
) -> None:
    """Try CB refinancing + reserve transfer atomically; fall back to overnight payable.

    Wraps ``cb_lend_reserves`` and ``transfer_reserves`` inside a single
    ``atomic(system)`` block so that if the transfer fails after the CB loan
    is created, both operations are rolled back.  This prevents the debtor
    bank from keeping extra reserves and a new CB liability when settlement
    did not actually occur.

    If the atomic block raises, we fall back to creating an overnight
    interbank payable due ``day + 1``.

    Args:
        system: System instance.
        debtor_bank: Agent id of the bank that owes.
        creditor_bank: Agent id of the bank that is owed.
        amount: Net amount to settle.
        shortfall: Amount to borrow from the CB (may equal *amount* when
            the debtor has no reserves at all).
        day: Current simulation day.
        label: Optional label for log messages (e.g. "fallback").
    """
    suffix = f" ({label})" if label else ""
    try:
        # Atomic: if transfer_reserves fails, cb_lend_reserves is also rolled back.
        with atomic(system):
            system.cb_lend_reserves(debtor_bank, shortfall, day)
            system.transfer_reserves(debtor_bank, creditor_bank, amount)
        logger.debug(
            "interbank cleared via CB refinancing%s: %s -> %s amount=%d (CB loan=%d)",
            suffix, debtor_bank, creditor_bank, amount, shortfall,
        )
        system.log(
            EventKind.INTERBANK_CLEARED,
            debtor_bank=debtor_bank,
            creditor_bank=creditor_bank,
            amount=amount,
            cb_refinanced=shortfall,
        )
    except Exception:
        # Both cb_lend_reserves and transfer_reserves are rolled back.
        # Fall back to an overnight interbank payable.
        logger.warning(
            "interbank CB refinancing failed%s: %s -> %s amount=%d",
            suffix, debtor_bank, creditor_bank, amount,
        )
        payable_id = system.new_contract_id("P")
        overnight_payable = Payable(
            id=payable_id,
            kind=InstrumentKind.PAYABLE,
            amount=amount,
            denom="X",
            asset_holder_id=creditor_bank,
            liability_issuer_id=debtor_bank,
            due_day=day + 1,
        )
        system.add_contract(overnight_payable)
        system.log(
            EventKind.INTERBANK_OVERNIGHT_CREATED,
            debtor_bank=debtor_bank,
            creditor_bank=creditor_bank,
            amount=amount,
            payable_id=payable_id,
            due_day=day + 1,
        )


def settle_intraday_nets(system: System, day: int) -> None:
    """
    Settle intraday nets between banks using reserves, CB refinancing, or overnight payables.

    For each net amount between banks:
    - Try to transfer reserves if sufficient
    - If insufficient reserves, borrow shortfall from CB and then transfer
    - If CB borrowing fails, fall back to creating overnight payable due tomorrow
    - Log InterbankCleared (with optional cb_refinanced) or InterbankOvernightCreated events

    Args:
        system: System instance
        day: Current day
    """
    nets = compute_intraday_nets(system, day)
    logger.debug("settle_intraday_nets: day=%d, pairs=%d", day, len(nets))

    for (bank_a, bank_b), net_amount in nets.items():
        if net_amount == 0:
            continue

        if net_amount > 0:
            # bank_a owes bank_b
            debtor_bank = bank_a
            creditor_bank = bank_b
            amount = net_amount
        else:
            # bank_b owes bank_a
            debtor_bank = bank_b
            creditor_bank = bank_a
            amount = -net_amount

        # Try to transfer reserves directly.
        try:
            with atomic(system):
                # Find available reserves for debtor bank
                debtor_reserve_ids = []
                for cid in system.state.agents[debtor_bank].asset_ids:
                    contract = system.state.contracts.get(cid)
                    if contract is None:
                        continue
                    if contract.kind == InstrumentKind.RESERVE_DEPOSIT:
                        debtor_reserve_ids.append(cid)

                if not debtor_reserve_ids:
                    available_reserves = 0
                else:
                    available_reserves = sum(
                        system.state.contracts[cid].amount for cid in debtor_reserve_ids
                    )

                if available_reserves >= amount:
                    # Sufficient reserves - transfer them
                    system.transfer_reserves(debtor_bank, creditor_bank, amount)
                    logger.debug(
                        "interbank cleared: %s -> %s amount=%d", debtor_bank, creditor_bank, amount
                    )
                    system.log(
                        EventKind.INTERBANK_CLEARED,
                        debtor_bank=debtor_bank,
                        creditor_bank=creditor_bank,
                        amount=amount,
                    )
                else:
                    # Insufficient reserves - need CB refinancing.
                    # Raise to exit the atomic block cleanly, then handle
                    # refinancing via the helper (which has its own atomic).
                    raise _NeedsCBRefinancing(amount - available_reserves)

        except _NeedsCBRefinancing as e:
            _settle_net_with_cb_fallback(
                system, debtor_bank, creditor_bank, amount, e.shortfall, day,
            )
        except Exception:
            # Direct transfer failed (ValidationError or otherwise, e.g. no
            # reserves at all).  Attempt CB refinancing for the full amount.
            _settle_net_with_cb_fallback(
                system, debtor_bank, creditor_bank, amount, amount, day,
                label="fallback",
            )


class _NeedsCBRefinancing(Exception):
    """Internal signal: debtor has some reserves but not enough; needs CB top-up."""

    def __init__(self, shortfall: int) -> None:
        self.shortfall = shortfall
        super().__init__(shortfall)


def compute_combined_nets(
    system: "System",
    day: int,
    interbank_obligations: list[tuple[str, str, int, object]],
) -> dict[tuple[str, str], int]:
    """Compute combined bilateral nets: client payments + interbank repayments.

    Starts with compute_intraday_nets() for client payments, then adds
    interbank repayment obligations using the same lexical ordering
    convention.

    Convention: nets[(a,b)] > 0 means bank a owes bank b.

    Args:
        system: System instance.
        day: Current simulation day.
        interbank_obligations: List of (borrower_bank, lender_bank,
            repayment_amount, loan) from compute_interbank_obligations().

    Returns:
        Combined bilateral net dict.
    """
    from collections import defaultdict

    nets: defaultdict[tuple[str, str], int] = defaultdict(int)

    # Start with client payment nets
    client_nets = compute_intraday_nets(system, day)
    for pair, amount in client_nets.items():
        nets[pair] += amount

    # Add interbank repayment obligations
    for borrower, lender, repayment, _loan in interbank_obligations:
        if borrower == lender:
            continue
        # borrower owes lender the repayment amount
        if borrower < lender:
            nets[(borrower, lender)] += repayment
        else:
            nets[(lender, borrower)] -= repayment

    return dict(nets)


def compute_bank_net_obligations(
    nets: dict[tuple[str, str], int],
) -> dict[str, int]:
    """Aggregate bilateral nets to per-bank net obligations.

    For each bank, sum its net obligations across all counterparties.
    Positive = bank owes reserves (net debtor).
    Negative = bank is owed reserves (net creditor).

    Args:
        nets: Bilateral net dict from compute_combined_nets().

    Returns:
        Dict mapping bank_id to net obligation (positive = owes).
    """
    obligations: dict[str, int] = {}

    for (bank_a, bank_b), net_amount in nets.items():
        if net_amount == 0:
            continue

        if net_amount > 0:
            # bank_a owes bank_b
            obligations[bank_a] = obligations.get(bank_a, 0) + net_amount
            obligations[bank_b] = obligations.get(bank_b, 0) - net_amount
        else:
            # bank_b owes bank_a
            obligations[bank_b] = obligations.get(bank_b, 0) + abs(net_amount)
            obligations[bank_a] = obligations.get(bank_a, 0) - abs(net_amount)

    return obligations


def settle_nets_with_funding(
    system: "System",
    day: int,
    nets: dict[tuple[str, str], int],
) -> None:
    """Settle pre-computed bilateral nets using reserves, CB refinancing, or overnight payables.

    Same settlement logic as settle_intraday_nets() but receives
    pre-computed nets (which may include interbank repayment obligations
    folded into the bilateral netting).

    Args:
        system: System instance.
        day: Current simulation day.
        nets: Pre-computed bilateral net dict.
    """
    logger.debug("settle_nets_with_funding: day=%d, pairs=%d", day, len(nets))

    for (bank_a, bank_b), net_amount in nets.items():
        if net_amount == 0:
            continue

        if net_amount > 0:
            debtor_bank = bank_a
            creditor_bank = bank_b
            amount = net_amount
        else:
            debtor_bank = bank_b
            creditor_bank = bank_a
            amount = -net_amount

        # Try to transfer reserves directly.
        try:
            with atomic(system):
                # Find available reserves for debtor bank
                debtor_reserve_ids = []
                for cid in system.state.agents[debtor_bank].asset_ids:
                    contract = system.state.contracts.get(cid)
                    if contract is None:
                        continue
                    if contract.kind == InstrumentKind.RESERVE_DEPOSIT:
                        debtor_reserve_ids.append(cid)

                if not debtor_reserve_ids:
                    available_reserves = 0
                else:
                    available_reserves = sum(
                        system.state.contracts[cid].amount for cid in debtor_reserve_ids
                    )

                if available_reserves >= amount:
                    system.transfer_reserves(debtor_bank, creditor_bank, amount)
                    logger.debug(
                        "interbank cleared: %s -> %s amount=%d",
                        debtor_bank, creditor_bank, amount,
                    )
                    system.log(
                        EventKind.INTERBANK_CLEARED,
                        debtor_bank=debtor_bank,
                        creditor_bank=creditor_bank,
                        amount=amount,
                    )
                else:
                    raise _NeedsCBRefinancing(amount - available_reserves)

        except _NeedsCBRefinancing as e:
            _settle_net_with_cb_fallback(
                system, debtor_bank, creditor_bank, amount, e.shortfall, day,
            )
        except Exception:
            _settle_net_with_cb_fallback(
                system, debtor_bank, creditor_bank, amount, amount, day,
                label="fallback",
            )
