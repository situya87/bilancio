"""Dealer trade execution logic.

All trade execution logic, eligibility checks, and order flow for the
dealer subsystem.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System
    from bilancio.engines.dealer_integration import DealerSubsystem
    from bilancio.dealer.kernel import ExecutionResult

from bilancio.dealer.models import (
    DealerState,
    VBTState,
    TraderState,
    Ticket,
)
from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.metrics import (
    TradeRecord,
    TicketOutcome,
    compute_safety_margin,
)


def _compute_trader_safety_margin(
    subsystem: "DealerSubsystem",
    trader_id: str
) -> Decimal:
    """Compute safety margin for a specific trader."""
    trader = subsystem.traders.get(trader_id)
    if not trader:
        return Decimal(0)

    return compute_safety_margin(
        cash=trader.cash,
        tickets_held=trader.tickets_owned,
        obligations=trader.obligations,
        dealers=subsystem.dealers,
        ticket_size=subsystem.params.S
    )


def _check_sell_risk_assessment(
    subsystem: "DealerSubsystem",
    trader: TraderState,
    trader_id: str,
    ticket: "Ticket",
    dealer: DealerState,
    bucket_id: str,
    current_day: int,
    events: List[dict[str, object]],
) -> bool:
    """Check risk assessment for a sell trade.

    Returns True if the trade was rejected (should skip this sell).
    Returns False if the trade is acceptable or no risk assessor is configured.
    """
    if not subsystem.risk_assessor:
        return False

    # Compute asset value as sum of EVs of owned tickets
    asset_value = sum(
        (subsystem.risk_assessor.expected_value(t, current_day)
         for t in trader.tickets_owned),
        Decimal(0),
    )
    if subsystem.risk_assessor.should_sell(
        ticket=ticket,
        dealer_bid=dealer.bid,
        current_day=current_day,
        trader_cash=trader.cash,
        trader_shortfall=trader.shortfall(current_day),
        trader_asset_value=asset_value,
    ):
        return False  # Trade accepted

    # Trader rejects price - log event and skip
    events.append({
        "kind": "sell_rejected",
        "day": current_day,
        "phase": "simulation",
        "trader_id": trader_id,
        "ticket_id": ticket.id,
        "bucket": bucket_id,
        "offered_price": float(dealer.bid),
        "expected_value": float(subsystem.risk_assessor.expected_value(ticket, current_day)),
        "threshold": float(subsystem.risk_assessor.params.base_risk_premium),
        "reason": "price_below_ev_threshold",
    })
    return True  # Trade rejected


def _record_sell_trade(
    subsystem: "DealerSubsystem",
    trader_id: str,
    ticket: "Ticket",
    bucket_id: str,
    current_day: int,
    scaled_price: Decimal,
    unit_price: Decimal,
    is_passthrough: bool,
    is_liquidity_driven: bool,
    pre_dealer_inventory: int,
    pre_dealer_cash: Decimal,
    pre_dealer_bid: Decimal,
    pre_dealer_ask: Decimal,
    pre_trader_cash: Decimal,
    pre_safety_margin: Decimal,
    post_safety_margin: Decimal,
    dealer: DealerState,
    vbt: VBTState,
    trader_cash_after: Decimal,
    events: List[dict[str, object]],
) -> None:
    """Record a completed sell trade in metrics, ticket outcomes, and events."""
    # Create detailed trade record for metrics (Section 8)
    trade_record = TradeRecord(
        day=current_day,
        bucket=bucket_id,
        side="SELL",
        trader_id=trader_id,
        ticket_id=ticket.id,
        issuer_id=ticket.issuer_id,
        maturity_day=ticket.maturity_day,
        face_value=ticket.face,
        price=scaled_price,
        unit_price=unit_price,
        is_passthrough=is_passthrough,
        dealer_inventory_before=pre_dealer_inventory,
        dealer_cash_before=pre_dealer_cash,
        dealer_bid_before=pre_dealer_bid,
        dealer_ask_before=pre_dealer_ask,
        vbt_mid_before=vbt.M,
        trader_cash_before=pre_trader_cash,
        trader_safety_margin_before=pre_safety_margin,
        dealer_inventory_after=dealer.a,
        dealer_cash_after=dealer.cash,
        dealer_bid_after=dealer.bid,
        dealer_ask_after=dealer.ask,
        trader_cash_after=trader_cash_after,
        trader_safety_margin_after=post_safety_margin,
        is_liquidity_driven=is_liquidity_driven,
        reduces_margin_below_zero=False,  # Only for BUYs
    )
    subsystem.metrics.trades.append(trade_record)

    # Update ticket outcome for return tracking (Section 8.3)
    if ticket.id not in subsystem.metrics.ticket_outcomes:
        subsystem.metrics.ticket_outcomes[ticket.id] = TicketOutcome(
            ticket_id=ticket.id,
            issuer_id=ticket.issuer_id,
            maturity_day=ticket.maturity_day,
            face_value=ticket.face,
        )
    subsystem.metrics.ticket_outcomes[ticket.id].sold_to_dealer = True
    subsystem.metrics.ticket_outcomes[ticket.id].sale_day = current_day
    subsystem.metrics.ticket_outcomes[ticket.id].sale_price = scaled_price
    subsystem.metrics.ticket_outcomes[ticket.id].seller_id = trader_id

    events.append({
        "kind": "dealer_trade",
        "day": current_day,
        "phase": "simulation",
        "trader": trader_id,
        "side": "sell",
        "ticket_id": ticket.id,
        "bucket": bucket_id,
        "price": float(scaled_price),
        "unit_price": float(unit_price),
        "face": float(ticket.face),
        "is_passthrough": is_passthrough,
        "is_liquidity_driven": is_liquidity_driven,
    })


def _execute_sell_trade(
    subsystem: "DealerSubsystem",
    trader_id: str,
    current_day: int,
    events: List[dict[str, object]],
    dealer_budgets: Dict[str, Decimal] | None = None,
) -> Decimal:
    """Process a single sell trade attempt for a trader. Returns cash injected into ring."""
    from bilancio.engines.dealer_integration import _get_agent_cash

    trader = subsystem.traders[trader_id]
    if not trader.tickets_owned:
        return Decimal(0)

    # Select ticket to sell (first in list for simplicity)
    ticket = trader.tickets_owned[0]
    bucket_id = ticket.bucket_id
    dealer = subsystem.dealers[bucket_id]
    vbt = subsystem.vbts[bucket_id]

    # Pre-trade solvency check: estimate cost and verify payer can afford it
    # For a customer sell, the dealer (or VBT) is the buyer/payer
    if dealer_budgets is not None:
        from bilancio.dealer.kernel import can_interior_buy
        is_interior = can_interior_buy(dealer, subsystem.params)
        payer_id = dealer.agent_id if is_interior else vbt.agent_id
        estimated_cost = Decimal(str(dealer.bid if is_interior else vbt.B)) * ticket.face
        if dealer_budgets.get(payer_id, Decimal(0)) < estimated_cost:
            return Decimal(0)  # Skip: payer can't afford this trade

    # Capture pre-trade state for metrics (Section 8.1, 8.4)
    pre_dealer_inventory = dealer.a
    pre_dealer_cash = dealer.cash
    pre_dealer_bid = dealer.bid
    pre_dealer_ask = dealer.ask
    pre_trader_cash = trader.cash
    pre_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

    # Check if liquidity-driven (Section 8.3)
    is_liquidity_driven = trader.shortfall(current_day) > 0

    # Risk assessment check (Plan 032)
    if _check_sell_risk_assessment(
        subsystem, trader, trader_id, ticket,
        dealer, bucket_id, current_day, events,
    ):
        return Decimal(0)

    # Execute customer sell
    assert subsystem.executor is not None
    result = subsystem.executor.execute_customer_sell(
        dealer, vbt, ticket, check_assertions=False
    )

    if result.executed:
        # Scale price by ticket face value
        # The dealer module returns unit price (per S=1), but our tickets have actual face values
        scaled_price = result.price * ticket.face

        # Update trader state
        trader.tickets_owned.remove(ticket)
        trader.cash += scaled_price

        # Capture post-trade state
        post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

        # Record trade in metrics, ticket outcomes, and events
        _record_sell_trade(
            subsystem, trader_id, ticket, bucket_id, current_day,
            scaled_price, result.price, result.is_passthrough,
            is_liquidity_driven,
            pre_dealer_inventory, pre_dealer_cash,
            pre_dealer_bid, pre_dealer_ask,
            pre_trader_cash, pre_safety_margin, post_safety_margin,
            dealer, vbt, trader.cash, events,
        )

        # Update dealer/VBT cash budget after successful sell trade
        if dealer_budgets is not None:
            payer_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
            dealer_budgets[payer_id] = dealer_budgets.get(payer_id, Decimal(0)) - scaled_price

        return scaled_price

    return Decimal(0)


def _reverse_buy_to_dealer(
    dealer: DealerState,
    vbt: VBTState,
    result: "ExecutionResult",
    bucket_id: str,
) -> None:
    """Reverse a buy trade by returning the ticket to the dealer or VBT.

    Used when a trade needs to be unwound (risk rejection, solvency failure).
    Puts the ticket back into the appropriate inventory and reverses the
    unit-price cash change.

    Args:
        dealer: Dealer state for the bucket
        vbt: VBT state for the bucket
        result: Trade execution result to reverse
        bucket_id: Maturity bucket identifier
    """
    assert result.ticket is not None
    if result.is_passthrough:
        vbt.inventory.append(result.ticket)
        result.ticket.owner_id = f"vbt_{bucket_id}"
        vbt.cash -= result.price
    else:
        dealer.inventory.append(result.ticket)
        result.ticket.owner_id = f"dealer_{bucket_id}"
        dealer.cash -= result.price


def _check_buy_risk_assessment(
    subsystem: "DealerSubsystem",
    trader: TraderState,
    trader_id: str,
    result: "ExecutionResult",
    dealer: DealerState,
    vbt: VBTState,
    bucket_id: str,
    current_day: int,
    events: List[dict[str, object]],
) -> bool:
    """Check risk assessment for a buy trade and reverse if rejected.

    Returns True if the trade was rejected (should skip to next bucket).
    Returns False if the trade is acceptable or no risk assessor is configured.
    """
    if not subsystem.risk_assessor:
        return False

    assert result.ticket is not None

    # Compute asset value (including the ticket we just bought)
    asset_value = sum(
        (subsystem.risk_assessor.expected_value(t, current_day)
         for t in trader.tickets_owned),
        Decimal(0),
    )
    # Check if trader would accept this buy
    if subsystem.risk_assessor.should_buy(
        ticket=result.ticket,
        dealer_ask=result.price,  # Unit price
        current_day=current_day,
        trader_cash=trader.cash,
        trader_shortfall=trader.shortfall(current_day),
        trader_asset_value=asset_value,
    ):
        return False  # Trade accepted

    # Trader rejects - reverse the transaction
    _reverse_buy_to_dealer(dealer, vbt, result, bucket_id)
    events.append({
        "kind": "buy_rejected",
        "day": current_day,
        "phase": "simulation",
        "trader_id": trader_id,
        "ticket_id": result.ticket.id,
        "bucket": bucket_id,
        "offered_price": float(result.price),
        "expected_value": float(subsystem.risk_assessor.expected_value(result.ticket, current_day)),
        "threshold": float(subsystem.risk_assessor.params.base_risk_premium * subsystem.risk_assessor.params.buy_premium_multiplier),
        "reason": "ev_below_price_threshold",
    })
    return True  # Trade rejected


def _record_buy_trade(
    subsystem: "DealerSubsystem",
    trader_id: str,
    ticket: "Ticket",
    bucket_id: str,
    current_day: int,
    scaled_price: Decimal,
    unit_price: Decimal,
    is_passthrough: bool,
    pre_dealer_inventory: int,
    pre_dealer_cash: Decimal,
    pre_dealer_bid: Decimal,
    pre_dealer_ask: Decimal,
    pre_trader_cash: Decimal,
    pre_safety_margin: Decimal,
    post_safety_margin: Decimal,
    dealer: DealerState,
    vbt: VBTState,
    trader_cash_after: Decimal,
    events: List[dict[str, object]],
) -> None:
    """Record a completed buy trade in metrics, ticket outcomes, and events."""
    # Check if BUY reduced margin below zero (Section 8.4)
    reduces_margin_below_zero = (
        pre_safety_margin >= 0 and post_safety_margin < 0
    )

    # Create detailed trade record for metrics (Section 8)
    trade_record = TradeRecord(
        day=current_day,
        bucket=bucket_id,
        side="BUY",
        trader_id=trader_id,
        ticket_id=ticket.id,
        issuer_id=ticket.issuer_id,
        maturity_day=ticket.maturity_day,
        face_value=ticket.face,
        price=scaled_price,
        unit_price=unit_price,
        is_passthrough=is_passthrough,
        dealer_inventory_before=pre_dealer_inventory,
        dealer_cash_before=pre_dealer_cash,
        dealer_bid_before=pre_dealer_bid,
        dealer_ask_before=pre_dealer_ask,
        vbt_mid_before=vbt.M,
        trader_cash_before=pre_trader_cash,
        trader_safety_margin_before=pre_safety_margin,
        dealer_inventory_after=dealer.a,
        dealer_cash_after=dealer.cash,
        dealer_bid_after=dealer.bid,
        dealer_ask_after=dealer.ask,
        trader_cash_after=trader_cash_after,
        trader_safety_margin_after=post_safety_margin,
        is_liquidity_driven=False,  # BUYs are never liquidity-driven
        reduces_margin_below_zero=reduces_margin_below_zero,
    )
    subsystem.metrics.trades.append(trade_record)

    # Update ticket outcome for return tracking (Section 8.3)
    if ticket.id not in subsystem.metrics.ticket_outcomes:
        subsystem.metrics.ticket_outcomes[ticket.id] = TicketOutcome(
            ticket_id=ticket.id,
            issuer_id=ticket.issuer_id,
            maturity_day=ticket.maturity_day,
            face_value=ticket.face,
        )
    subsystem.metrics.ticket_outcomes[ticket.id].purchased_from_dealer = True
    subsystem.metrics.ticket_outcomes[ticket.id].purchase_day = current_day
    subsystem.metrics.ticket_outcomes[ticket.id].purchase_price = scaled_price
    subsystem.metrics.ticket_outcomes[ticket.id].purchaser_id = trader_id

    events.append({
        "kind": "dealer_trade",
        "day": current_day,
        "phase": "simulation",
        "trader": trader_id,
        "side": "buy",
        "ticket_id": ticket.id,
        "bucket": bucket_id,
        "price": float(scaled_price),
        "unit_price": float(unit_price),
        "face": float(ticket.face),
        "is_passthrough": is_passthrough,
        "reduces_margin_below_zero": reduces_margin_below_zero,
    })


def _execute_buy_trade(
    subsystem: "DealerSubsystem",
    trader_id: str,
    current_day: int,
    events: List[dict[str, object]],
    dealer_budgets: Dict[str, Decimal] | None = None,
) -> Decimal:
    """Process a single buy trade attempt for a trader. Returns cash drained from ring."""
    trader = subsystem.traders[trader_id]

    # Shuffle bucket order so buys don't always hit the same bucket first
    bucket_ids = list(subsystem.dealers.keys())
    subsystem.rng.shuffle(bucket_ids)
    for bucket_id in bucket_ids:
        dealer = subsystem.dealers[bucket_id]
        vbt = subsystem.vbts[bucket_id]

        # Check if dealer or VBT has inventory
        if not dealer.inventory and not vbt.inventory:
            continue

        # Capture pre-trade state for metrics (Section 8.1, 8.4)
        pre_dealer_inventory = dealer.a
        pre_dealer_cash = dealer.cash
        pre_dealer_bid = dealer.bid
        pre_dealer_ask = dealer.ask
        pre_trader_cash = trader.cash
        pre_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

        # Execute customer buy
        assert subsystem.executor is not None
        result = subsystem.executor.execute_customer_buy(
            dealer, vbt, trader_id, check_assertions=False
        )

        if result.executed and result.ticket:
            # Post-execution risk assessment check (Plan 032)
            if _check_buy_risk_assessment(
                subsystem, trader, trader_id, result,
                dealer, vbt, bucket_id, current_day, events,
            ):
                continue  # Risk rejected, try next bucket

        if result.executed and result.ticket:
            # Scale price by ticket face value
            scaled_price = result.price * result.ticket.face

            # Pre-trade solvency check: trader must be able to afford scaled price
            if trader.cash < scaled_price:
                _reverse_buy_to_dealer(dealer, vbt, result, bucket_id)
                recompute_dealer_state(dealer, vbt, subsystem.params)
                continue  # Try next bucket

            # Update trader state
            trader.tickets_owned.append(result.ticket)
            trader.cash -= scaled_price

            # Update asset issuer if first ticket
            if trader.asset_issuer_id is None:
                trader.asset_issuer_id = result.ticket.issuer_id

            # Capture post-trade state
            post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

            # Record trade in metrics, ticket outcomes, and events
            _record_buy_trade(
                subsystem, trader_id, result.ticket, bucket_id, current_day,
                scaled_price, result.price, result.is_passthrough,
                pre_dealer_inventory, pre_dealer_cash,
                pre_dealer_bid, pre_dealer_ask,
                pre_trader_cash, pre_safety_margin, post_safety_margin,
                dealer, vbt, trader.cash, events,
            )

            # Update dealer/VBT cash budget after successful buy trade
            # For a buy, the dealer/VBT received cash (sold a ticket)
            if dealer_budgets is not None:
                payee_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
                dealer_budgets[payee_id] = dealer_budgets.get(payee_id, Decimal(0)) + scaled_price

            return scaled_price  # One buy per trader per batch

    return Decimal(0)


def _build_eligible_sellers(
    subsystem: "DealerSubsystem",
    current_day: int,
    horizon: int = 10,
) -> List[str]:
    """Identify traders eligible to sell (have shortfall coming in next few days).

    Args:
        subsystem: Dealer subsystem state
        current_day: Current simulation day
        horizon: Number of days to look ahead for upcoming obligations

    Returns:
        List of trader IDs eligible to sell
    """
    eligible_sellers = []
    for trader_id, trader in subsystem.traders.items():
        upcoming_shortfall = Decimal(0)
        for day_offset in range(horizon + 1):
            upcoming_shortfall = max(upcoming_shortfall, trader.shortfall(current_day + day_offset))
        if upcoming_shortfall > 0 and trader.tickets_owned:
            eligible_sellers.append(trader_id)
    return eligible_sellers


def _build_eligible_buyers(
    subsystem: "DealerSubsystem",
    current_day: int,
    horizon: int = 5,
) -> List[str]:
    """Identify traders eligible to buy (have surplus cash beyond needs).

    Only allows buying if trader has genuine surplus above ALL upcoming
    obligations.

    Args:
        subsystem: Dealer subsystem state
        current_day: Current simulation day
        horizon: Number of days to look ahead for upcoming obligations

    Returns:
        List of trader IDs eligible to buy
    """
    eligible_buyers = []
    for trader_id, trader in subsystem.traders.items():
        total_upcoming_dues = Decimal(0)
        for day_offset in range(horizon + 1):
            total_upcoming_dues += trader.payment_due(current_day + day_offset)
        surplus = trader.cash - total_upcoming_dues
        if surplus > 0:
            eligible_buyers.append(trader_id)
    return eligible_buyers


def _execute_interleaved_order_flow(
    subsystem: "DealerSubsystem",
    system: "System",
    current_day: int,
    eligible_sellers: List[str],
    eligible_buyers: List[str],
    events: List[dict[str, object]],
) -> None:
    """Execute interleaved sell/buy order flow in batches.

    Sellers and buyers alternate in batches of BATCH_SIZE.
    Sellers go first (urgent liquidity needs), then buyers, repeating
    until both lists are exhausted.

    Cash neutrality: total buy cash cannot exceed total sell cash,
    preventing the dealer from being a net cash drain on the ring.
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    BATCH_SIZE = 10
    subsystem.rng.shuffle(eligible_sellers)
    subsystem.rng.shuffle(eligible_buyers)

    # Initialize cash budgets for dealers/VBTs from system-level cash
    # This prevents trades that would make dealer/VBT cash go negative
    dealer_budgets: Dict[str, Decimal] = {}
    for bucket_id, dealer in subsystem.dealers.items():
        dealer_budgets[dealer.agent_id] = _get_agent_cash(system, dealer.agent_id)
    for bucket_id, vbt in subsystem.vbts.items():
        dealer_budgets[vbt.agent_id] = _get_agent_cash(system, vbt.agent_id)

    seller_idx = 0
    buyer_idx = 0
    sell_cash = Decimal(0)  # Cumulative cash injected by sells
    buy_cash = Decimal(0)   # Cumulative cash drained by buys

    while seller_idx < len(eligible_sellers) or buyer_idx < len(eligible_buyers):
        # --- Seller batch ---
        for trader_id in eligible_sellers[seller_idx:seller_idx + BATCH_SIZE]:
            sell_cash += _execute_sell_trade(subsystem, trader_id, current_day, events, dealer_budgets)
        seller_idx += BATCH_SIZE

        # --- Buyer batch (constrained by cash neutrality) ---
        for trader_id in eligible_buyers[buyer_idx:buyer_idx + BATCH_SIZE]:
            if buy_cash >= sell_cash:
                break  # Would drain more cash than sells injected
            buy_cash += _execute_buy_trade(subsystem, trader_id, current_day, events, dealer_budgets)
        buyer_idx += BATCH_SIZE
