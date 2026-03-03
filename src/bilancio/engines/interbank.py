"""Call-auction interbank reserve market.

After client payment settlement (Phase C), banks trade overnight
reserve loans via a call auction. Each bank computes a limit rate
from its reserve position using the CB corridor. A call auction
finds a single clearing rate r* where supply meets demand. Overnight
loans (due next day) are issued. The CB becomes the true backstop,
used only when the interbank market can't resolve imbalances.

Design decisions:
- Post-settlement only (no pre-positioning)
- Banks on one side only (lenders quote ask, borrowers quote bid)
- Single clearing rate r* via call auction
- Surplus lends to deficit (no bilateral settlement-claim swaps)
- CB is true backstop (interbank first, CB for residual)
- Overnight = due next day (day+1)
- Risk-free (no interbank counterparty credit assessment)
- No new parameters — limit rate from corridor mid, width, reserve target
- Simple quantities: lend/borrow full surplus/deficit amount
- Overnight repayments netted with client payment flows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from bilancio.core.events import EventKind

if TYPE_CHECKING:
    from bilancio.engines.banking_subsystem import BankingSubsystem, InterbankLoan
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)


@dataclass
class InterbankOrder:
    """A limit order in the interbank call auction."""

    bank_id: str
    side: str  # "lend" or "borrow"
    quantity: int
    limit_rate: Decimal
    remaining: int = 0

    def __post_init__(self):
        if self.remaining == 0:
            self.remaining = self.quantity


@dataclass
class AuctionResult:
    """Result of the interbank call auction."""

    clearing_rate: Decimal | None  # r* (None if no trades)
    trades: list[dict] = field(default_factory=list)  # [{lender, borrower, amount}]
    unfilled_borrowers: list[tuple[str, int]] = field(default_factory=list)
    total_volume: int = 0


def compute_interbank_obligations(
    current_day: int,
    banking: BankingSubsystem,
) -> list[tuple[str, str, int, "InterbankLoan"]]:
    """Identify overnight interbank loans maturing today.

    Returns list of (borrower_bank, lender_bank, repayment_amount, loan)
    for loans with maturity_day == current_day.

    These obligations are NOT settled here — they are folded into
    the bilateral netting in compute_combined_nets().
    """
    obligations = []
    for loan in banking.interbank_loans:
        if loan.maturity_day == current_day:
            obligations.append((
                loan.borrower_bank,
                loan.lender_bank,
                loan.repayment_amount,
                loan,
            ))
    return obligations


def compute_reserve_positions(
    system: "System",
    banking: BankingSubsystem,
    net_obligations: dict[str, int],
) -> dict[str, int]:
    """Compute each bank's reserve surplus/deficit for the auction.

    x_i = R_i - N_i - R*_i

    where:
        R_i = current reserves
        N_i = net obligation from combined bilateral nets (positive = owes)
        R*_i = reserve target

    x > 0: surplus (potential lender)
    x < 0: deficit (potential borrower)

    Defaulted banks are excluded (return empty position).
    """
    from bilancio.engines.banking_subsystem import _get_bank_reserves

    positions: dict[str, int] = {}
    for bank_id, bank_state in banking.banks.items():
        bank_agent = system.state.agents.get(bank_id)
        if bank_agent is None or bank_agent.defaulted:
            continue

        reserves = _get_bank_reserves(system, bank_id)
        net_obligation = net_obligations.get(bank_id, 0)
        target = bank_state.pricing_params.reserve_target

        positions[bank_id] = reserves - net_obligation - target

    return positions


def compute_limit_rates(
    positions: dict[str, int],
    banking: BankingSubsystem,
) -> dict[str, Decimal]:
    """Compute each bank's limit rate from its reserve position.

    normalized = clamp(x / R*, -1, +1)
    limit_rate = M - (Ω/2) × normalized

    where:
        M = corridor midpoint
        Ω = corridor width
        R* = reserve target

    Properties:
        Surplus (x > 0, normalized > 0): limit_rate < M (cheap ask)
        Deficit (x < 0, normalized < 0): limit_rate > M (expensive bid)
        At target (x = 0): limit_rate = M
        Max surplus (normalized = +1): limit_rate = r_floor
        Max deficit (normalized = -1): limit_rate = r_ceiling
    """
    rates: dict[str, Decimal] = {}
    for bank_id, x in positions.items():
        bank_state = banking.banks.get(bank_id)
        if bank_state is None:
            continue

        target = bank_state.pricing_params.reserve_target
        if target <= 0:
            logger.warning(
                "Bank %s has reserve_target=%d; defaulting to 1 for limit rate calc",
                bank_id, target,
            )
            target = 1

        # Corridor parameters
        r_floor = bank_state.pricing_params.reserve_remuneration_rate
        r_ceiling = bank_state.pricing_params.cb_borrowing_rate
        M = (r_floor + r_ceiling) / 2
        omega = r_ceiling - r_floor

        # Normalized position: clamp to [-1, +1]
        normalized = Decimal(x) / Decimal(target)
        normalized = max(Decimal("-1"), min(Decimal("1"), normalized))

        # Limit rate
        rates[bank_id] = M - (omega / 2) * normalized

    return rates


def build_order_book(
    positions: dict[str, int],
    limit_rates: dict[str, Decimal],
) -> tuple[list[InterbankOrder], list[InterbankOrder]]:
    """Build sorted order book from positions and limit rates.

    Returns:
        (lender_asks_asc, borrower_bids_desc)
        - Lender asks sorted ascending by rate (cheapest first)
        - Borrower bids sorted descending by rate (highest first)
    """
    lender_asks: list[InterbankOrder] = []
    borrower_bids: list[InterbankOrder] = []

    for bank_id, x in positions.items():
        if bank_id not in limit_rates:
            continue
        rate = limit_rates[bank_id]

        if x > 0:
            lender_asks.append(InterbankOrder(
                bank_id=bank_id,
                side="lend",
                quantity=x,
                limit_rate=rate,
            ))
        elif x < 0:
            borrower_bids.append(InterbankOrder(
                bank_id=bank_id,
                side="borrow",
                quantity=abs(x),
                limit_rate=rate,
            ))

    # Sort: asks ascending (cheapest first), bids descending (highest first)
    lender_asks.sort(key=lambda o: o.limit_rate)
    borrower_bids.sort(key=lambda o: o.limit_rate, reverse=True)

    return lender_asks, borrower_bids


def clear_auction(
    lender_asks: list[InterbankOrder],
    borrower_bids: list[InterbankOrder],
) -> AuctionResult:
    """Clear the call auction: match asks against bids.

    Walk the book: while cheapest ask ≤ highest bid, match.
    Volume at each match = min(remaining ask, remaining bid).
    r* = midpoint of marginal ask and marginal bid.
    All trades execute at r*.

    Returns AuctionResult with clearing_rate, trades, unfilled borrowers.
    """
    if not lender_asks or not borrower_bids:
        unfilled = [(o.bank_id, o.remaining) for o in borrower_bids]
        return AuctionResult(
            clearing_rate=None,
            trades=[],
            unfilled_borrowers=unfilled,
            total_volume=0,
        )

    trades: list[dict] = []
    total_volume = 0
    ask_idx = 0
    bid_idx = 0
    marginal_ask = Decimal("0")
    marginal_bid = Decimal("0")

    while ask_idx < len(lender_asks) and bid_idx < len(borrower_bids):
        ask = lender_asks[ask_idx]
        bid = borrower_bids[bid_idx]

        if ask.limit_rate > bid.limit_rate:
            break  # No more matches possible

        # Match
        matched = min(ask.remaining, bid.remaining)
        trades.append({
            "lender": ask.bank_id,
            "borrower": bid.bank_id,
            "amount": matched,
        })
        total_volume += matched

        # Track marginal rates for clearing price
        marginal_ask = ask.limit_rate
        marginal_bid = bid.limit_rate

        ask.remaining -= matched
        bid.remaining -= matched

        if ask.remaining == 0:
            ask_idx += 1
        if bid.remaining == 0:
            bid_idx += 1

    # Clearing rate = midpoint of marginal ask and marginal bid
    clearing_rate: Decimal | None = None
    if total_volume > 0:
        clearing_rate = (marginal_ask + marginal_bid) / 2

    # Unfilled borrowers
    unfilled: list[tuple[str, int]] = []
    for i in range(bid_idx, len(borrower_bids)):
        b = borrower_bids[i]
        if b.remaining > 0:
            unfilled.append((b.bank_id, b.remaining))

    return AuctionResult(
        clearing_rate=clearing_rate,
        trades=trades,
        unfilled_borrowers=unfilled,
        total_volume=total_volume,
    )


def run_interbank_auction(
    system: "System",
    current_day: int,
    banking: BankingSubsystem,
    net_obligations: dict[str, int],
) -> list[dict]:
    """Run the interbank call auction: positions → rates → order book → r* → trades.

    Orchestrates the full auction pipeline:
    1. Compute reserve positions (accounting for net obligations)
    2. Compute limit rates from positions
    3. Build order book
    4. Clear auction
    5. Execute trades (transfer reserves, create InterbankLoan records)
    6. Emit events

    Returns list of event dicts.
    """
    from bilancio.engines.banking_subsystem import InterbankLoan

    events: list[dict] = []

    # 1. Reserve positions
    positions = compute_reserve_positions(system, banking, net_obligations)
    if not positions:
        return events

    # 2. Limit rates
    limit_rates = compute_limit_rates(positions, banking)

    # 3. Order book
    lender_asks, borrower_bids = build_order_book(positions, limit_rates)

    # 4. Clear auction
    result = clear_auction(lender_asks, borrower_bids)

    # 5. Execute trades at r*
    executed_volume = 0
    if result.clearing_rate is not None and result.trades:
        for trade in result.trades:
            lender_id = trade["lender"]
            borrower_id = trade["borrower"]
            amount = trade["amount"]

            try:
                system.transfer_reserves(lender_id, borrower_id, amount)
            except Exception:
                logger.warning(
                    "Interbank auction transfer failed: %s -> %s, amount=%d",
                    lender_id, borrower_id, amount,
                )
                result.unfilled_borrowers.append((borrower_id, amount))
                continue

            executed_volume += amount

            # Record interbank loan (overnight, due tomorrow)
            ib_loan = InterbankLoan(
                lender_bank=lender_id,
                borrower_bank=borrower_id,
                amount=amount,
                rate=result.clearing_rate,
                issuance_day=current_day,
                maturity_day=current_day + 1,
            )
            banking.interbank_loans.append(ib_loan)

            # Emit trade event
            events.append({
                "kind": EventKind.INTERBANK_AUCTION_TRADE.value,
                "day": current_day,
                "lender": lender_id,
                "borrower": borrower_id,
                "amount": amount,
                "rate": str(result.clearing_rate),
                "maturity_day": current_day + 1,
            })

    # 6. Emit auction summary event
    events.append({
        "kind": EventKind.INTERBANK_AUCTION.value,
        "day": current_day,
        "clearing_rate": str(result.clearing_rate) if result.clearing_rate else None,
        "total_volume": executed_volume,
        "n_trades": len(result.trades),
        "n_unfilled": len(result.unfilled_borrowers),
    })

    # 7. Emit unfilled borrower events
    for borrower_id, unfilled_amount in result.unfilled_borrowers:
        events.append({
            "kind": EventKind.INTERBANK_UNFILLED.value,
            "day": current_day,
            "borrower": borrower_id,
            "amount": unfilled_amount,
        })

    logger.debug(
        "Interbank auction day=%d: r*=%s, volume=%d, trades=%d, unfilled=%d",
        current_day,
        result.clearing_rate,
        executed_volume,
        len(result.trades),
        len(result.unfilled_borrowers),
    )

    return events


def finalize_interbank_repayments(
    current_day: int,
    banking: BankingSubsystem,
    obligations: list[tuple[str, str, int, "InterbankLoan"]],
) -> list[dict]:
    """Remove matured interbank loans from the book and emit events.

    Called AFTER settlement has occurred (the repayment amounts were
    already included in the bilateral netting). This function just
    cleans up the loan records.

    Args:
        current_day: Current simulation day.
        banking: BankingSubsystem with interbank_loans list.
        obligations: List of (borrower, lender, amount, loan) from
            compute_interbank_obligations().

    Returns:
        List of InterbankRepaid event dicts.
    """
    events: list[dict] = []

    # Collect loan objects that matured today
    matured_loans = {id(ob[3]) for ob in obligations}

    for borrower, lender, repayment, loan in obligations:
        events.append({
            "kind": EventKind.INTERBANK_REPAID.value,
            "day": current_day,
            "lender": lender,
            "borrower": borrower,
            "principal": loan.amount,
            "repayment": repayment,
            "interest": repayment - loan.amount,
        })

    # Remove matured loans from the book
    banking.interbank_loans = [
        loan for loan in banking.interbank_loans
        if id(loan) not in matured_loans
    ]

    return events
