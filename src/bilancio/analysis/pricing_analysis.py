"""Pricing and equilibrium analysis.

Analyse dealer/VBT pricing dynamics: price discovery speed, credit vs
liquidity premium decomposition, and fire-sale detection.

All functions consume the standard event log (list of dicts) produced by
the simulation engine.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

Event = dict[str, Any]


def trade_prices_by_day(
    events: list[Event],
) -> dict[int, list[dict[str, Any]]]:
    """Extract trade prices per day from dealer trade events.

    Returns:
        {day: [{"trader_id": str, "side": "buy"|"sell", "price": Decimal,
                "face": Decimal, "price_ratio": Decimal}]}
    """
    by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for e in events:
        kind = e.get("kind", "")
        if kind != "dealer_trade":
            continue
        day = int(e.get("day", 0))
        trader = e.get("trader")
        price = Decimal(str(e.get("price", e.get("amount", 0))))
        face = Decimal(str(e.get("face", e.get("face_value", 0))))
        side = str(e.get("side", ""))
        if side not in ("buy", "sell"):
            continue
        price_ratio = price / face if face > 0 else Decimal(0)

        by_day[day].append({
            "trader_id": trader,
            "side": side,
            "price": price,
            "face": face,
            "price_ratio": price_ratio,
        })

    return dict(by_day)


def average_price_ratio_by_day(
    events: list[Event],
) -> dict[int, dict[str, Decimal | None]]:
    """Average price/face ratio per day, split by buy/sell.

    Returns:
        {day: {"buy_avg": Decimal|None, "sell_avg": Decimal|None, "all_avg": Decimal|None}}
    """
    by_day = trade_prices_by_day(events)
    result: dict[int, dict[str, Decimal | None]] = {}

    for day in sorted(by_day.keys()):
        trades = by_day[day]
        buys = [t["price_ratio"] for t in trades if t["side"] == "buy"]
        sells = [t["price_ratio"] for t in trades if t["side"] == "sell"]
        all_ratios = [t["price_ratio"] for t in trades]

        result[day] = {
            "buy_avg": sum(buys, Decimal(0)) / len(buys) if buys else None,
            "sell_avg": sum(sells, Decimal(0)) / len(sells) if sells else None,
            "all_avg": sum(all_ratios, Decimal(0)) / len(all_ratios) if all_ratios else None,
        }

    return result


def price_discovery_speed(
    events: list[Event],
    true_default_rate: Decimal | None = None,
) -> dict[str, Any]:
    """Measure how quickly market prices converge to fundamental value.

    If true_default_rate is provided, the fundamental value is
    (1 - true_default_rate). Otherwise we use the final-day average
    price as the "converged" value.

    Returns:
        {"fundamental": Decimal, "convergence_day": int|None,
         "price_trajectory": list of (day, avg_price)}
    """
    avg_by_day = average_price_ratio_by_day(events)
    if not avg_by_day:
        return {"fundamental": Decimal(0), "convergence_day": None, "price_trajectory": []}

    trajectory: list[tuple[int, Decimal]] = []
    for day in sorted(avg_by_day.keys()):
        avg = avg_by_day[day].get("all_avg")
        if avg is not None:
            trajectory.append((day, avg))

    if not trajectory:
        return {"fundamental": Decimal(0), "convergence_day": None, "price_trajectory": []}

    # Determine fundamental
    if true_default_rate is not None:
        fundamental = Decimal(1) - true_default_rate
    else:
        fundamental = trajectory[-1][1]  # Use last price as proxy

    # Find convergence day (first day within 5% of fundamental)
    threshold = fundamental * Decimal("0.05")
    convergence_day: int | None = None
    for day, price in trajectory:
        if abs(price - fundamental) <= threshold:
            convergence_day = day
            break

    return {
        "fundamental": fundamental,
        "convergence_day": convergence_day,
        "price_trajectory": trajectory,
    }


def bid_ask_spread_by_day(
    events: list[Event],
) -> dict[int, Decimal | None]:
    """Effective bid-ask spread per day (ask_avg - bid_avg).

    Returns:
        {day: spread_in_price_ratio_terms}
    """
    avg_by_day = average_price_ratio_by_day(events)
    result: dict[int, Decimal | None] = {}

    for day in sorted(avg_by_day.keys()):
        buy_avg = avg_by_day[day].get("buy_avg")
        sell_avg = avg_by_day[day].get("sell_avg")
        if buy_avg is not None and sell_avg is not None:
            # Buy price (ask) should be > sell price (bid)
            result[day] = buy_avg - sell_avg
        else:
            result[day] = None

    return result


def trade_volume_by_day(
    events: list[Event],
) -> dict[int, dict[str, int]]:
    """Count of trades per day by side.

    Returns:
        {day: {"buys": int, "sells": int, "total": int}}
    """
    by_day = trade_prices_by_day(events)
    result: dict[int, dict[str, int]] = {}

    for day in sorted(by_day.keys()):
        trades = by_day[day]
        buys = sum(1 for t in trades if t["side"] == "buy")
        sells = sum(1 for t in trades if t["side"] == "sell")
        result[day] = {"buys": buys, "sells": sells, "total": buys + sells}

    return result


def fire_sale_indicator(
    events: list[Event],
    price_drop_threshold: Decimal = Decimal("0.10"),
    volume_spike_factor: Decimal = Decimal("2"),
) -> list[dict[str, Any]]:
    """Detect potential fire-sale episodes.

    A fire sale is flagged when:
    1. Average sell price drops by more than `price_drop_threshold`
       compared to the previous day, AND
    2. Sell volume is at least `volume_spike_factor` times the
       average volume of preceding days.

    Returns list of fire-sale day records:
        [{"day": int, "price_drop": Decimal, "volume_ratio": Decimal}]
    """
    avg_by_day = average_price_ratio_by_day(events)
    vol_by_day = trade_volume_by_day(events)
    days = sorted(avg_by_day.keys())

    if len(days) < 2:
        return []

    fire_sales: list[dict[str, Any]] = []
    prev_sell_prices: list[Decimal] = []
    prev_volumes: list[int] = []

    for _i, day in enumerate(days):
        sell_avg = avg_by_day[day].get("sell_avg")
        sell_vol = vol_by_day.get(day, {}).get("sells", 0)

        if sell_avg is not None and prev_sell_prices:
            prev_avg = prev_sell_prices[-1]
            price_drop = prev_avg - sell_avg

            avg_vol = (
                sum(prev_volumes) / len(prev_volumes)
                if prev_volumes
                else 1
            )
            volume_ratio = Decimal(str(sell_vol)) / Decimal(str(max(avg_vol, 1)))

            if price_drop > price_drop_threshold and volume_ratio >= volume_spike_factor:
                fire_sales.append({
                    "day": day,
                    "price_drop": price_drop,
                    "volume_ratio": volume_ratio,
                })

        if sell_avg is not None:
            prev_sell_prices.append(sell_avg)
        prev_volumes.append(sell_vol)

    return fire_sales
