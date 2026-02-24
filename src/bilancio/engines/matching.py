"""Matching engine — resolves agent intentions against dealer quotes.

For the Kalecki ring, matching is trivial: there is exactly one dealer
per maturity bucket, and the dealer posts a firm bid and a firm ask
derived from the VBT mid-price and the bucket spread parameter.

* **Sellers** sell at the dealer's *bid* (mid − half-spread).
* **Buyers** buy at the dealer's *ask* (mid + half-spread).

There is no order book and no price discovery.  The dealer is always
the counterparty, acting as a market-maker with bounded inventory.

Execution order within a day:

1. All **sells** are processed first (urgent liquidity needs).
2. All **buys** are processed second (surplus investment).

Within each phase the intention list is shuffled so that no agent has a
systematic positional advantage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.decision.intentions import BuyIntention, SellIntention
    from bilancio.engines.dealer_integration import DealerSubsystem
    from bilancio.engines.system import System


class DealerMatchingEngine:
    """Match trade intentions against dealer-posted bid/ask quotes.

    In the Kalecki ring each maturity bucket has a single dealer that
    posts a bid and an ask.  Matching is therefore deterministic once
    the intention lists are ordered: every sell hits the bid, every buy
    lifts the ask, and the dealer's inventory / cash budget is the only
    binding constraint.

    The engine does **not** perform price discovery or maintain an order
    book — it simply dispatches each intention to the appropriate
    ``_execute_sell_trade`` or ``_execute_buy_trade`` helper, which
    handles ticket selection, pricing, and settlement.
    """

    def execute(
        self,
        subsystem: DealerSubsystem,
        system: System,
        current_day: int,
        sell_intentions: list[SellIntention],
        buy_intentions: list[BuyIntention],
        events: list[dict[str, object]],
    ) -> None:
        """Match sell and buy intentions against dealer quotes.

        Produces identical behaviour to the legacy
        ``_execute_interleaved_order_flow`` function: sells are processed
        first (urgent liquidity), then buys (surplus investment).  Both
        lists are shuffled independently so no agent has a positional
        advantage.

        Parameters
        ----------
        subsystem:
            The dealer subsystem containing dealers, VBTs, traders, and
            the shared RNG.
        system:
            The top-level simulation system (used to read agent cash).
        current_day:
            The current simulation day.
        sell_intentions:
            Intentions produced by the sell-decision strategy.
        buy_intentions:
            Intentions produced by the buy-decision strategy.
        events:
            Mutable list to which trade-event dicts are appended.
        """
        # Lazy imports to break circular dependency
        # (dealer_integration ↔ dealer_trades ↔ matching)
        from bilancio.engines.dealer_integration import _get_agent_cash
        from bilancio.engines.dealer_trades import (
            _execute_buy_trade,
            _execute_sell_trade,
        )

        # --- Mutable copies so callers keep their originals intact ---
        sell_order: list[SellIntention] = list(sell_intentions)
        buy_order: list[BuyIntention] = list(buy_intentions)

        # --- Randomise execution order ---
        subsystem.rng.shuffle(sell_order)  # type: ignore[arg-type]
        subsystem.rng.shuffle(buy_order)  # type: ignore[arg-type]

        # --- Snapshot dealer/VBT cash as execution budgets ---
        dealer_budgets: dict[str, Decimal] = {}
        for _bucket_id, dealer in subsystem.dealers.items():
            dealer_budgets[dealer.agent_id] = _get_agent_cash(
                system, dealer.agent_id
            )
        for _bucket_id, vbt in subsystem.vbts.items():
            dealer_budgets[vbt.agent_id] = _get_agent_cash(
                system, vbt.agent_id
            )

        # --- Phase 1: process all sells (urgent liquidity) ---
        for intention in sell_order:
            _execute_sell_trade(
                subsystem,
                intention.trader_id,
                current_day,
                events,
                dealer_budgets,
            )

        # --- Phase 2: process all buys (surplus investment) ---
        for intention in buy_order:
            _execute_buy_trade(
                subsystem,
                intention.trader_id,
                current_day,
                events,
                dealer_budgets,
                max_spend=intention.max_spend,
            )
