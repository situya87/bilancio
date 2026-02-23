"""Intention architecture for agent decision-making.

Intentions are declarative expressions of what an agent *wants* to do,
separated from the mechanics of *how* it gets done.  A ``SellIntention``
says "I want to sell a ticket"; a ``BuyIntention`` says "I want to buy
one."  Neither carries execution details — prices, counterparties, or
settlement mechanics are the responsibility of the matching engine
(``engines/matching.py``).

Decision strategies evaluate an agent's current state and produce an
intention (or ``None`` if the agent has no reason to act).  The two
strategies shipped here reproduce the Kalecki-ring behaviour that was
previously hard-wired inside ``_build_eligible_sellers`` and
``_build_eligible_buyers``:

* ``LiquidityDrivenSeller`` — sell when facing a payment shortfall.
* ``SurplusBuyer`` — buy when holding cash surplus above a reserve
  threshold.

Collector helpers (``collect_sell_intentions``, ``collect_buy_intentions``)
iterate over the trader population and return batched intention lists
ready to be handed to the matching engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.dealer.models import TraderState
    from bilancio.decision.profiles import TraderProfile
    from bilancio.engines.dealer_integration import DealerSubsystem


# ---------------------------------------------------------------------------
# Intention types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SellIntention:
    """Declared intention to sell a ticket on the secondary market."""

    trader_id: str
    is_liquidity_driven: bool = True


@dataclass(frozen=True)
class BuyIntention:
    """Declared intention to buy a ticket on the secondary market."""

    trader_id: str
    max_spend: Decimal = Decimal("Inf")


# ---------------------------------------------------------------------------
# Decision strategies
# ---------------------------------------------------------------------------


class LiquidityDrivenSeller:
    """Produce a ``SellIntention`` when the trader faces a shortfall.

    A trader is a candidate seller when it owns at least one ticket *and*
    has a positive payment shortfall within the look-ahead *horizon*.
    """

    def evaluate(
        self,
        trader_id: str,
        trader: TraderState,
        current_day: int,
        horizon: int,
    ) -> SellIntention | None:
        if not trader.tickets_owned:
            return None

        upcoming_shortfall = Decimal(0)
        for day_offset in range(horizon + 1):
            upcoming_shortfall = max(
                upcoming_shortfall, trader.shortfall(current_day + day_offset)
            )

        if upcoming_shortfall > 0:
            return SellIntention(trader_id=trader_id, is_liquidity_driven=True)

        return None


class SurplusBuyer:
    """Produce a ``BuyIntention`` when the trader has sufficient surplus.

    A trader is a candidate buyer when its cash exceeds a reserve
    threshold derived from upcoming payment obligations.  The reserve
    fraction and surplus threshold are governed by the ``TraderProfile``.
    """

    def evaluate(
        self,
        trader_id: str,
        trader: TraderState,
        current_day: int,
        horizon: int,
        profile: TraderProfile,
        face_value: Decimal,
    ) -> BuyIntention | None:
        # Liquidity-only motive: skip agents with no upcoming liabilities.
        if profile.trading_motive == "liquidity_only":
            if trader.earliest_liability_day(current_day) is None:
                return None

        total_upcoming_dues = Decimal(0)
        for day_offset in range(horizon + 1):
            total_upcoming_dues += trader.payment_due(current_day + day_offset)

        reserved = profile.buy_reserve_fraction * total_upcoming_dues
        surplus = trader.cash - reserved
        threshold = face_value * profile.surplus_threshold_factor

        if surplus > threshold:
            return BuyIntention(trader_id=trader_id, max_spend=surplus)

        return None


# ---------------------------------------------------------------------------
# Collector helpers
# ---------------------------------------------------------------------------


def collect_sell_intentions(
    subsystem: DealerSubsystem,
    current_day: int,
    *,
    horizon: int | None = None,
    strategy: LiquidityDrivenSeller | None = None,
) -> list[SellIntention]:
    """Walk the trader population and collect sell intentions.

    Parameters
    ----------
    subsystem:
        The active ``DealerSubsystem`` containing traders and profiles.
    current_day:
        Simulation day to evaluate against.
    horizon:
        Look-ahead window in days.  Defaults to
        ``subsystem.trader_profile.sell_horizon``.
    strategy:
        Decision strategy instance.  Defaults to
        ``LiquidityDrivenSeller()``.
    """
    if strategy is None:
        strategy = LiquidityDrivenSeller()
    if horizon is None:
        horizon = subsystem.trader_profile.sell_horizon

    intentions: list[SellIntention] = []
    for trader_id, trader in subsystem.traders.items():
        intention = strategy.evaluate(trader_id, trader, current_day, horizon)
        if intention is not None:
            intentions.append(intention)
    return intentions


def collect_buy_intentions(
    subsystem: DealerSubsystem,
    current_day: int,
    *,
    horizon: int | None = None,
    strategy: SurplusBuyer | None = None,
) -> list[BuyIntention]:
    """Walk the trader population and collect buy intentions.

    Parameters
    ----------
    subsystem:
        The active ``DealerSubsystem`` containing traders, profiles, and
        the instrument face value.
    current_day:
        Simulation day to evaluate against.
    horizon:
        Look-ahead window in days.  Defaults to
        ``subsystem.trader_profile.buy_horizon``.
    strategy:
        Decision strategy instance.  Defaults to ``SurplusBuyer()``.
    """
    if strategy is None:
        strategy = SurplusBuyer()
    if horizon is None:
        horizon = subsystem.trader_profile.buy_horizon

    intentions: list[BuyIntention] = []
    for trader_id, trader in subsystem.traders.items():
        intention = strategy.evaluate(
            trader_id,
            trader,
            current_day,
            horizon,
            subsystem.trader_profile,
            subsystem.face_value,
        )
        if intention is not None:
            intentions.append(intention)
    return intentions
