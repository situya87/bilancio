"""Dealer trade execution logic.

All trade execution logic, eligibility checks, and order flow for the
dealer subsystem.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.dealer.kernel import ExecutionResult
    from bilancio.engines.dealer_integration import DealerSubsystem
    from bilancio.engines.system import System

from bilancio.dealer.kernel import recompute_dealer_state
from bilancio.dealer.metrics import (
    TicketOutcome,
    TradeRecord,
    compute_safety_margin,
)
from bilancio.dealer.models import (
    DealerState,
    Ticket,
    TraderState,
    VBTState,
)


def _get_issuer_price_adjustment(subsystem: DealerSubsystem, issuer_id: str) -> Decimal:
    """Returns price multiplier for issuer-specific pricing.

    adjustment_i = P_i - P_system (issuer excess risk)
    multiplier = max(0, 1 - adjustment_i)

    When P_i = P_system, multiplier = 1 (no change).
    When P_i > P_system, multiplier < 1 (riskier issuer, lower price).
    When P_i < P_system, multiplier > 1 (safer issuer, higher price, bounded by probability range).
    """
    if not subsystem.issuer_specific_pricing:
        return Decimal(1)
    p_system = subsystem.system_default_prob
    p_issuer = subsystem.issuer_default_probs.get(issuer_id, p_system)
    adjustment = p_issuer - p_system
    # Floor at 0 to prevent negative prices; natural ceiling is ~1+P_system
    return max(Decimal(0), Decimal(1) - adjustment)


def _check_concentration_limit(
    subsystem: DealerSubsystem, ticket: Ticket,
) -> bool:
    """Return True if accepting ticket would breach issuer concentration limit."""
    limit = subsystem.dealer_concentration_limit
    if limit <= 0:
        return False
    # Check across ALL dealer buckets (global exposure)
    issuer_count = 0
    total_count = 0
    for d in subsystem.dealers.values():
        for t in d.inventory:
            total_count += 1
            if t.issuer_id == ticket.issuer_id:
                issuer_count += 1
    post_total = total_count + 1
    post_issuer = issuer_count + 1
    return (Decimal(post_issuer) / Decimal(post_total)) > limit


def _compute_trader_safety_margin(subsystem: DealerSubsystem, trader_id: str) -> Decimal:
    """Compute safety margin for a specific trader."""
    trader = subsystem.traders.get(trader_id)
    if not trader:
        return Decimal(0)

    return compute_safety_margin(
        cash=trader.cash,
        tickets_held=trader.tickets_owned,
        obligations=trader.obligations,
        dealers=subsystem.dealers,
        ticket_size=subsystem.params.S,
    )


def _check_sell_risk_assessment(
    subsystem: DealerSubsystem,
    trader: TraderState,
    trader_id: str,
    ticket: Ticket,
    dealer: DealerState,
    bucket_id: str,
    current_day: int,
    events: list[dict[str, object]],
) -> bool:
    """Check risk assessment for a sell trade.

    Returns True if the trade was rejected (should skip this sell).
    Returns False if the trade is acceptable or no risk assessor is configured.
    """
    from bilancio.engines.dealer_integration import get_trader_assessor

    assessor = get_trader_assessor(subsystem, trader_id)
    if not assessor:
        return False

    # Compute asset value as sum of EVs of owned tickets
    asset_value = sum(
        (assessor.expected_value(t, current_day) for t in trader.tickets_owned),
        Decimal(0),
    )
    if assessor.should_sell(
        ticket=ticket,
        dealer_bid=dealer.bid,
        current_day=current_day,
        trader_cash=trader.cash,
        trader_shortfall=trader.upcoming_shortfall(
            current_day, (trader.profile or subsystem.trader_profile).sell_horizon
        ),
        trader_asset_value=asset_value,
    ):
        return False  # Trade accepted

    # Trader rejects price - log event and skip
    events.append(
        {
            "kind": "sell_rejected",
            "day": current_day,
            "phase": "simulation",
            "trader_id": trader_id,
            "ticket_id": ticket.id,
            "bucket": bucket_id,
            "offered_price": float(dealer.bid),
            "expected_value": float(assessor.expected_value(ticket, current_day)),
            "threshold": float(assessor.params.base_risk_premium),
            "reason": "price_below_ev_threshold",
        }
    )
    return True  # Trade rejected


def _record_sell_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    ticket: Ticket,
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
    events: list[dict[str, object]],
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

    events.append(
        {
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
        }
    )


def _execute_sell_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    current_day: int,
    events: list[dict[str, object]],
    dealer_budgets: dict[str, Decimal] | None = None,
) -> Decimal:
    """Process a single sell trade attempt for a trader. Returns cash injected into ring."""

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
    sell_profile = trader.profile or subsystem.trader_profile
    is_liquidity_driven = (
        trader.upcoming_shortfall(current_day, sell_profile.sell_horizon) > 0
    )

    # Risk assessment check (Plan 032)
    if _check_sell_risk_assessment(
        subsystem,
        trader,
        trader_id,
        ticket,
        dealer,
        bucket_id,
        current_day,
        events,
    ):
        return Decimal(0)

    # Execute customer sell
    assert subsystem.executor is not None
    result = subsystem.executor.execute_customer_sell(dealer, vbt, ticket, check_assertions=False)

    if result.executed:
        # Concentration limit check (Feature 3) — only for interior dealer
        # buys (not passthrough).  Passthrough sells land in VBT inventory,
        # so dealer concentration is unaffected.
        if not result.is_passthrough and _check_concentration_limit(subsystem, ticket):
            # Undo the kernel execution: reverse the interior buy.
            # Kernel did: dealer.inventory.append(ticket), dealer.cash -= price,
            # ticket.owner_id = dealer.agent_id.
            dealer.inventory.remove(ticket)
            dealer.cash += result.price
            ticket.owner_id = trader_id
            recompute_dealer_state(dealer, vbt, subsystem.params)
            events.append({
                "kind": "sell_rejected_concentration",
                "day": current_day,
                "trader_id": trader_id,
                "bucket_id": bucket_id,
                "issuer_id": ticket.issuer_id,
                "limit": str(subsystem.dealer_concentration_limit),
            })
            return Decimal(0)

        # Apply issuer-specific price adjustment (Feature 1)
        price_factor = _get_issuer_price_adjustment(subsystem, ticket.issuer_id)
        adjusted_price = result.price * price_factor

        # Correct dealer/VBT cash to match the adjusted price.
        # The kernel already moved cash at result.price (unit level).
        # Refund the delta so both sides of the trade agree.
        price_delta = result.price - adjusted_price
        if price_delta != 0:
            if result.is_passthrough:
                vbt.cash += price_delta
            else:
                dealer.cash += price_delta
            recompute_dealer_state(dealer, vbt, subsystem.params)

        # Scale price by ticket face value
        # The dealer module returns unit price (per S=1), but our tickets have actual face values
        scaled_price = adjusted_price * ticket.face

        # Update trader state
        trader.tickets_owned.remove(ticket)
        trader.cash += scaled_price

        # Capture post-trade state
        post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)

        # Record trade in metrics, ticket outcomes, and events
        _record_sell_trade(
            subsystem,
            trader_id,
            ticket,
            bucket_id,
            current_day,
            scaled_price,
            adjusted_price,
            result.is_passthrough,
            is_liquidity_driven,
            pre_dealer_inventory,
            pre_dealer_cash,
            pre_dealer_bid,
            pre_dealer_ask,
            pre_trader_cash,
            pre_safety_margin,
            post_safety_margin,
            dealer,
            vbt,
            trader.cash,
            events,
        )

        # Track VBT flow (passthrough sell → VBT bought from customer)
        if result.is_passthrough:
            vbt.cumulative_inflow += ticket.face

        # Update dealer/VBT cash budget after successful sell trade
        if dealer_budgets is not None:
            payer_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
            dealer_budgets[payer_id] = dealer_budgets.get(payer_id, Decimal(0)) - scaled_price

        return scaled_price

    return Decimal(0)


def _reverse_buy_to_dealer(
    dealer: DealerState,
    vbt: VBTState,
    result: ExecutionResult,
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
    subsystem: DealerSubsystem,
    trader: TraderState,
    trader_id: str,
    result: ExecutionResult,
    dealer: DealerState,
    vbt: VBTState,
    bucket_id: str,
    current_day: int,
    events: list[dict[str, object]],
) -> bool:
    """Check risk assessment for a buy trade and reverse if rejected.

    Returns True if the trade was rejected (should skip to next bucket).
    Returns False if the trade is acceptable or no risk assessor is configured.
    """
    from bilancio.engines.dealer_integration import get_trader_assessor

    assessor = get_trader_assessor(subsystem, trader_id)
    if not assessor:
        return False

    assert result.ticket is not None

    # Compute asset value including the candidate ticket's EV so that
    # cash_ratio reflects the post-trade portfolio (the ticket hasn't been
    # added to tickets_owned yet at this point in the flow).
    asset_value = sum(
        (assessor.expected_value(t, current_day) for t in trader.tickets_owned),
        Decimal(0),
    )
    asset_value += assessor.expected_value(result.ticket, current_day)
    # Check if trader would accept this buy
    if assessor.should_buy(
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
    events.append(
        {
            "kind": "buy_rejected",
            "day": current_day,
            "phase": "simulation",
            "trader_id": trader_id,
            "ticket_id": result.ticket.id,
            "bucket": bucket_id,
            "offered_price": float(result.price),
            "expected_value": float(
                assessor.expected_value(result.ticket, current_day)
            ),
            "threshold": float(
                assessor.params.base_risk_premium
                * assessor.params.buy_premium_multiplier
            ),
            "reason": "ev_below_price_threshold",
        }
    )
    return True  # Trade rejected


def _is_liquidity_buy(trader: TraderState, ticket: Ticket, current_day: int) -> bool:
    """True if ticket matures at or before the trader's earliest obligation."""
    earliest = trader.earliest_liability_day(current_day)
    if earliest is None:
        return False
    return ticket.maturity_day <= earliest


def _record_buy_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    trader: TraderState,
    ticket: Ticket,
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
    events: list[dict[str, object]],
) -> None:
    """Record a completed buy trade in metrics, ticket outcomes, and events."""
    # Check if BUY reduced margin below zero (Section 8.4)
    reduces_margin_below_zero = pre_safety_margin >= 0 and post_safety_margin < 0

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
        is_liquidity_driven=_is_liquidity_buy(trader, ticket, current_day),
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

    events.append(
        {
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
        }
    )


def _build_buy_bucket_order(subsystem: DealerSubsystem, motive: str) -> list[str]:
    """Return bucket iteration order for buy attempts based on trading motive."""
    if motive in ("liquidity_only", "liquidity_then_earning"):
        # Prefer buckets maturing soonest for liquidity-oriented buyers.
        tau_min_by_name = {bc.name: bc.tau_min for bc in subsystem.bucket_configs}
        return sorted(subsystem.dealers.keys(), key=lambda b: tau_min_by_name.get(b, 999))

    bucket_ids = list(subsystem.dealers.keys())
    subsystem.rng.shuffle(bucket_ids)
    return bucket_ids


def _reverse_buy_and_recompute(
    subsystem: DealerSubsystem,
    dealer: DealerState,
    vbt: VBTState,
    result: ExecutionResult,
    bucket_id: str,
) -> None:
    """Reverse buy execution and recompute dealer/VBT state."""
    _reverse_buy_to_dealer(dealer, vbt, result, bucket_id)
    recompute_dealer_state(dealer, vbt, subsystem.params)


def _compute_upcoming_obligations(
    trader: TraderState,
    current_day: int,
    horizon: int,
) -> Decimal:
    """Sum trader payment obligations across [current_day, current_day + horizon]."""
    upcoming_obligations = Decimal(0)
    for day_offset in range(horizon + 1):
        upcoming_obligations += trader.payment_due(current_day + day_offset)
    return upcoming_obligations


def _apply_buy_price_adjustment(
    subsystem: DealerSubsystem,
    dealer: DealerState,
    vbt: VBTState,
    result: ExecutionResult,
    adjusted_price: Decimal,
) -> None:
    """Align dealer/VBT cash with issuer-adjusted unit price."""
    # Kernel already moved cash at result.price (unit level).
    price_delta = result.price - adjusted_price
    if price_delta != 0:
        if result.is_passthrough:
            vbt.cash -= price_delta
        else:
            dealer.cash -= price_delta
        recompute_dealer_state(dealer, vbt, subsystem.params)


def _execute_buy_trade(
    subsystem: DealerSubsystem,
    trader_id: str,
    current_day: int,
    events: list[dict[str, object]],
    dealer_budgets: dict[str, Decimal] | None = None,
    max_spend: Decimal = Decimal("Inf"),
) -> Decimal:
    """Process a single buy trade attempt for a trader. Returns cash drained from ring."""
    trader = subsystem.traders[trader_id]

    # Bucket ordering depends on trading motive
    buy_profile = trader.profile or subsystem.trader_profile
    motive = buy_profile.trading_motive
    bucket_ids = _build_buy_bucket_order(subsystem, motive)
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

        # Safety margin gate: don't let underwater agents buy
        if pre_safety_margin < 0:
            continue

        # Execute customer buy
        assert subsystem.executor is not None
        result = subsystem.executor.execute_customer_buy(
            dealer, vbt, trader_id, check_assertions=False
        )

        if not (result.executed and result.ticket):
            continue

        # Post-execution risk assessment check (Plan 032)
        if _check_buy_risk_assessment(
            subsystem,
            trader,
            trader_id,
            result,
            dealer,
            vbt,
            bucket_id,
            current_day,
            events,
        ):
            continue  # Risk rejected, try next bucket

        ticket = result.ticket

        # Liquidity-only gate: reject buys of tickets maturing after earliest obligation
        if motive == "liquidity_only" and not _is_liquidity_buy(trader, ticket, current_day):
            _reverse_buy_and_recompute(subsystem, dealer, vbt, result, bucket_id)
            continue

        # Apply issuer-specific price adjustment (Feature 1)
        price_factor = _get_issuer_price_adjustment(subsystem, ticket.issuer_id)
        adjusted_price = result.price * price_factor
        scaled_price = adjusted_price * ticket.face

        # Pre-trade solvency check: trader must be able to afford scaled price
        if trader.cash < scaled_price:
            _reverse_buy_and_recompute(subsystem, dealer, vbt, result, bucket_id)
            continue  # Try next bucket

        # Budget enforcement: trader must not spend more than declared surplus
        if scaled_price > max_spend:
            _reverse_buy_and_recompute(subsystem, dealer, vbt, result, bucket_id)
            continue  # Try next bucket (cheaper ticket might exist)

        # Update trader state
        trader.tickets_owned.append(ticket)
        trader.cash -= scaled_price

        # Cash-only obligation coverage check: prevent the "safety margin
        # illusion" where receivables (valued at dealer bid) mask a cash
        # shortfall. After buying, trader must still cover obligations
        # within its planning horizon using cash alone.
        horizon = buy_profile.buy_horizon
        upcoming_obligations = _compute_upcoming_obligations(trader, current_day, horizon)
        if trader.cash < upcoming_obligations:
            trader.cash += scaled_price
            trader.tickets_owned.pop()
            _reverse_buy_and_recompute(subsystem, dealer, vbt, result, bucket_id)
            continue  # Try next bucket

        # Post-buy safety margin check: reverse if buy makes trader underwater
        post_safety_margin = _compute_trader_safety_margin(subsystem, trader_id)
        if post_safety_margin < 0:
            trader.cash += scaled_price
            trader.tickets_owned.pop()
            _reverse_buy_and_recompute(subsystem, dealer, vbt, result, bucket_id)
            continue  # Try next bucket

        # Align dealer/VBT cash with adjusted price after all rejection gates.
        _apply_buy_price_adjustment(subsystem, dealer, vbt, result, adjusted_price)

        # Track VBT flow (passthrough buy → VBT sold to customer).
        # Updated here (after all rejection gates) so reversed trades
        # don't inflate net_outflow and widen future asks.
        if result.is_passthrough:
            vbt.cumulative_outflow += ticket.face

        # Record trade in metrics, ticket outcomes, and events
        _record_buy_trade(
            subsystem,
            trader_id,
            trader,
            ticket,
            bucket_id,
            current_day,
            scaled_price,
            adjusted_price,
            result.is_passthrough,
            pre_dealer_inventory,
            pre_dealer_cash,
            pre_dealer_bid,
            pre_dealer_ask,
            pre_trader_cash,
            pre_safety_margin,
            post_safety_margin,
            dealer,
            vbt,
            trader.cash,
            events,
        )

        # Update dealer/VBT cash budget after successful buy trade.
        # For a buy, the dealer/VBT received cash (sold a ticket).
        if dealer_budgets is not None:
            payee_id = f"vbt_{bucket_id}" if result.is_passthrough else f"dealer_{bucket_id}"
            dealer_budgets[payee_id] = dealer_budgets.get(payee_id, Decimal(0)) + scaled_price

        return scaled_price  # One buy per trader per batch

    return Decimal(0)


def _build_eligible_sellers(
    subsystem: DealerSubsystem,
    current_day: int,
    horizon: int | None = None,
) -> list[str]:
    """Identify traders eligible to sell (have shortfall coming in next few days).

    Delegates to :func:`bilancio.decision.intentions.collect_sell_intentions`
    and extracts trader IDs for backward compatibility.

    Args:
        subsystem: Dealer subsystem state
        current_day: Current simulation day
        horizon: Number of days to look ahead for upcoming obligations

    Returns:
        List of trader IDs eligible to sell
    """
    from bilancio.decision.intentions import collect_sell_intentions

    kwargs: dict = {}
    if horizon is not None:
        kwargs["horizon"] = horizon
    intentions = collect_sell_intentions(subsystem, current_day, **kwargs)
    return [si.trader_id for si in intentions]


def _build_eligible_buyers(
    subsystem: DealerSubsystem,
    current_day: int,
    horizon: int | None = None,
) -> list[str]:
    """Identify traders eligible to buy (have surplus cash beyond needs).

    Delegates to :func:`bilancio.decision.intentions.collect_buy_intentions`
    and extracts trader IDs for backward compatibility.

    Args:
        subsystem: Dealer subsystem state
        current_day: Current simulation day
        horizon: Number of days to look ahead for upcoming obligations

    Returns:
        List of trader IDs eligible to buy
    """
    from bilancio.decision.intentions import collect_buy_intentions

    kwargs: dict = {}
    if horizon is not None:
        kwargs["horizon"] = horizon
    intentions = collect_buy_intentions(subsystem, current_day, **kwargs)
    return [bi.trader_id for bi in intentions]


def _execute_interleaved_order_flow(
    subsystem: DealerSubsystem,
    system: System,
    current_day: int,
    eligible_sellers: list[str],
    eligible_buyers: list[str],
    events: list[dict[str, object]],
) -> None:
    """Execute order flow: all sellers first, then all buyers independently.

    Sellers go first (urgent liquidity needs), then buyers independently.
    Dealers are independently capitalized, so buys do not need prior sells
    to accumulate cash.
    """
    from bilancio.engines.dealer_integration import _get_agent_cash

    subsystem.rng.shuffle(eligible_sellers)
    subsystem.rng.shuffle(eligible_buyers)

    # Initialize cash budgets for dealers/VBTs from system-level cash
    # This prevents trades that would make dealer/VBT cash go negative
    dealer_budgets: dict[str, Decimal] = {}
    for _bucket_id, dealer in subsystem.dealers.items():
        dealer_budgets[dealer.agent_id] = _get_agent_cash(system, dealer.agent_id)
    for _bucket_id, vbt in subsystem.vbts.items():
        dealer_budgets[vbt.agent_id] = _get_agent_cash(system, vbt.agent_id)

    # --- Process all sellers (urgent liquidity needs) ---
    for trader_id in eligible_sellers:
        _execute_sell_trade(subsystem, trader_id, current_day, events, dealer_budgets)

    # --- Process all buyers independently ---
    for trader_id in eligible_buyers:
        _execute_buy_trade(subsystem, trader_id, current_day, events, dealer_budgets)
