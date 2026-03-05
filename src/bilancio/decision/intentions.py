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

from dataclasses import dataclass, field
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
    urgency: Decimal = Decimal(0)
    """Liquidity stress score: ``max(0, upcoming_dues - cash) / (cash + 0.01)``.

    Higher values mean the trader is more stressed (large obligations
    relative to available cash).  Used by the ``"urgency"`` matching
    order to prioritise the most-stressed sellers first.
    """


@dataclass(frozen=True)
class BuyIntention:
    """Declared intention to buy a ticket on the secondary market."""

    trader_id: str
    max_spend: Decimal = Decimal("Inf")
    priority: Decimal = Decimal(0)
    """Normalised surplus score: ``(cash - reserve) / (face_value + 0.01)``.

    Higher values mean the trader has more deployable cash relative to
    the instrument face value.  Used by the ``"urgency"`` matching order
    to let the most cash-rich buyers trade first.
    """


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
            # Urgency: how stressed the trader is (shortfall relative to cash).
            # Higher urgency → should trade first in urgency matching order.
            urgency = upcoming_shortfall / (trader.cash + Decimal("0.01"))
            return SellIntention(
                trader_id=trader_id,
                is_liquidity_driven=True,
                urgency=urgency,
            )

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
        effective_buy_reserve_fraction: Decimal | None = None,
    ) -> BuyIntention | None:
        # Liquidity-only motive: skip agents with no upcoming liabilities.
        if profile.trading_motive == "liquidity_only":
            if trader.earliest_liability_day(current_day) is None:
                return None

        total_upcoming_dues = Decimal(0)
        for day_offset in range(horizon + 1):
            total_upcoming_dues += trader.payment_due(current_day + day_offset)

        # Plan 050: Use adaptive buy_reserve_fraction if provided
        buy_reserve_frac = (
            effective_buy_reserve_fraction
            if effective_buy_reserve_fraction is not None
            else profile.buy_reserve_fraction
        )
        reserved = buy_reserve_frac * total_upcoming_dues
        surplus = trader.cash - reserved
        threshold = face_value * profile.surplus_threshold_factor

        if surplus > threshold:
            # Deploy only half the surplus; the rest is a prudence buffer for
            # timing risk (obligations may come early), default risk (incoming
            # payments may not arrive), and rollover risk (new obligations
            # created when current ones mature).
            deployable = surplus / 2
            # Priority: normalised surplus — higher means more cash-rich.
            # Used by urgency matching order to let richest buyers trade first.
            priority = surplus / (face_value + Decimal("0.01"))
            return BuyIntention(
                trader_id=trader_id,
                max_spend=deployable,
                priority=priority,
            )

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
    eligible_traders: set[str] | None = None,
) -> list[SellIntention]:
    """Walk the trader population and collect sell intentions.

    Uses per-trader profiles when available, falling back to the subsystem
    default ``trader_profile``.

    Parameters
    ----------
    subsystem:
        The active ``DealerSubsystem`` containing traders and profiles.
    current_day:
        Simulation day to evaluate against.
    horizon:
        Look-ahead window in days.  When ``None``, each trader's own
        profile (or the subsystem default) supplies the sell horizon.
    strategy:
        Decision strategy instance.  Defaults to
        ``LiquidityDrivenSeller()``.
    eligible_traders:
        When not ``None``, only evaluate traders in this set.  Used by
        the ``prune_ineligible`` performance option to skip agents that
        have no tickets and no cash.
    """
    if strategy is None:
        strategy = LiquidityDrivenSeller()

    intentions: list[SellIntention] = []
    for trader_id, trader in subsystem.traders.items():
        if eligible_traders is not None and trader_id not in eligible_traders:
            continue
        # Per-agent horizon: use trader's own profile if available
        trader_horizon = horizon
        if trader_horizon is None:
            trader_profile = trader.profile or subsystem.trader_profile
            trader_horizon = trader_profile.sell_horizon
        intention = strategy.evaluate(trader_id, trader, current_day, trader_horizon)
        if intention is not None:
            intentions.append(intention)
    return intentions


def collect_buy_intentions(
    subsystem: DealerSubsystem,
    current_day: int,
    *,
    horizon: int | None = None,
    strategy: SurplusBuyer | None = None,
    eligible_traders: set[str] | None = None,
) -> list[BuyIntention]:
    """Walk the trader population and collect buy intentions.

    Uses per-trader profiles when available, falling back to the subsystem
    default ``trader_profile``.

    Parameters
    ----------
    subsystem:
        The active ``DealerSubsystem`` containing traders, profiles, and
        the instrument face value.
    current_day:
        Simulation day to evaluate against.
    horizon:
        Look-ahead window in days.  When ``None``, each trader's own
        profile (or the subsystem default) supplies the buy horizon.
    strategy:
        Decision strategy instance.  Defaults to ``SurplusBuyer()``.
    eligible_traders:
        When not ``None``, only evaluate traders in this set.  Used by
        the ``prune_ineligible`` performance option to skip agents that
        have no tickets and no cash.
    """
    if strategy is None:
        strategy = SurplusBuyer()

    # Plan 050: Adaptive buy_reserve_fraction from subsystem
    effective_buy_reserve_fraction = getattr(
        subsystem, "effective_buy_reserve_fraction", None
    )

    intentions: list[BuyIntention] = []
    for trader_id, trader in subsystem.traders.items():
        if eligible_traders is not None and trader_id not in eligible_traders:
            continue
        # Per-agent profile: use trader's own profile if available
        trader_profile = getattr(trader, 'profile', None) or subsystem.trader_profile
        trader_horizon = horizon if horizon is not None else trader_profile.buy_horizon
        intention = strategy.evaluate(
            trader_id,
            trader,
            current_day,
            trader_horizon,
            trader_profile,
            subsystem.face_value,
            effective_buy_reserve_fraction=effective_buy_reserve_fraction,
        )
        if intention is not None:
            intentions.append(intention)
    return intentions


# ---------------------------------------------------------------------------
# Intention cache (Option H: incremental intentions)
# ---------------------------------------------------------------------------


@dataclass
class IntentionCache:
    """Persistent intention queues across trading rounds within a day.

    Only traders in ``_invalidated`` are re-evaluated each round.
    """

    sell_queue: dict[str, SellIntention] = field(default_factory=dict)
    buy_queue: dict[str, BuyIntention] = field(default_factory=dict)
    _invalidated: set[str] = field(default_factory=set)
    _day: int = -1

    def invalidate(self, trader_id: str) -> None:
        """Mark a trader for re-evaluation next round."""
        self._invalidated.add(trader_id)


def init_intention_cache(
    subsystem: DealerSubsystem,
    current_day: int,
    *,
    eligible_traders: set[str] | None = None,
) -> IntentionCache:
    """Build a fresh intention cache from a full scan."""
    cache = IntentionCache(_day=current_day)
    sell_intentions = collect_sell_intentions(
        subsystem, current_day, eligible_traders=eligible_traders,
    )
    buy_intentions = collect_buy_intentions(
        subsystem, current_day, eligible_traders=eligible_traders,
    )
    for si in sell_intentions:
        cache.sell_queue[si.trader_id] = si
    for bi in buy_intentions:
        cache.buy_queue[bi.trader_id] = bi
    return cache


def refresh_intentions(
    cache: IntentionCache,
    subsystem: DealerSubsystem,
    current_day: int,
    *,
    eligible_traders: set[str] | None = None,
) -> None:
    """Re-evaluate only invalidated traders, updating the cache in-place."""
    if not cache._invalidated:
        return
    # Sort to ensure deterministic iteration order regardless of hash seed.
    for trader_id in sorted(cache._invalidated):
        # Remove old entries
        cache.sell_queue.pop(trader_id, None)
        cache.buy_queue.pop(trader_id, None)

        # Skip if pruned
        if eligible_traders is not None and trader_id not in eligible_traders:
            continue

        trader = subsystem.traders.get(trader_id)
        if trader is None:
            continue

        # Re-evaluate sell
        sell_intentions = collect_sell_intentions(
            subsystem, current_day, eligible_traders={trader_id},
        )
        for si in sell_intentions:
            cache.sell_queue[si.trader_id] = si

        # Re-evaluate buy
        buy_intentions = collect_buy_intentions(
            subsystem, current_day, eligible_traders={trader_id},
        )
        for bi in buy_intentions:
            cache.buy_queue[bi.trader_id] = bi

    cache._invalidated.clear()
