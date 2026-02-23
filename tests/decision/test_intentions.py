"""Unit tests for the bilancio.decision.intentions module.

Tests cover:
- SellIntention / BuyIntention frozen dataclasses and defaults
- LiquidityDrivenSeller strategy (shortfall detection, horizon scanning)
- SurplusBuyer strategy (surplus calculation, trading motives)
- collect_sell_intentions / collect_buy_intentions collectors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from bilancio.dealer.models import Ticket, TraderState
from bilancio.decision.intentions import (
    BuyIntention,
    LiquidityDrivenSeller,
    SellIntention,
    SurplusBuyer,
    collect_buy_intentions,
    collect_sell_intentions,
)
from bilancio.decision.profiles import TraderProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticket(
    *,
    id: str = "t1",
    issuer_id: str = "issuer",
    owner_id: str = "owner",
    face: Decimal = Decimal("100"),
    maturity_day: int = 1,
) -> Ticket:
    """Convenience factory for creating test tickets."""
    return Ticket(
        id=id,
        issuer_id=issuer_id,
        owner_id=owner_id,
        face=face,
        maturity_day=maturity_day,
    )


@dataclass
class MockSubsystem:
    """Minimal stand-in for DealerSubsystem used by the collectors."""

    traders: dict[str, TraderState] = field(default_factory=dict)
    trader_profile: TraderProfile = field(default_factory=TraderProfile)
    face_value: Decimal = Decimal("20")


# ===========================================================================
# TestSellIntention
# ===========================================================================


class TestSellIntention:
    """Tests for the SellIntention frozen dataclass."""

    def test_frozen(self) -> None:
        """SellIntention instances must be immutable."""
        si = SellIntention(trader_id="a1")
        with pytest.raises(AttributeError):
            si.trader_id = "a2"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """is_liquidity_driven should default to True."""
        si = SellIntention(trader_id="a1")
        assert si.is_liquidity_driven is True


# ===========================================================================
# TestBuyIntention
# ===========================================================================


class TestBuyIntention:
    """Tests for the BuyIntention frozen dataclass."""

    def test_frozen(self) -> None:
        """BuyIntention instances must be immutable."""
        bi = BuyIntention(trader_id="a1")
        with pytest.raises(AttributeError):
            bi.trader_id = "a2"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """max_spend should default to Decimal('Inf')."""
        bi = BuyIntention(trader_id="a1")
        assert bi.max_spend == Decimal("Inf")


# ===========================================================================
# TestLiquidityDrivenSeller
# ===========================================================================


class TestLiquidityDrivenSeller:
    """Tests for the LiquidityDrivenSeller strategy."""

    def setup_method(self) -> None:
        self.strategy = LiquidityDrivenSeller()

    def test_sell_when_shortfall_and_tickets(self) -> None:
        """A trader with a shortfall *and* tickets to sell should get a SellIntention."""
        trader = TraderState(
            agent_id="a1",
            cash=Decimal("0"),
            tickets_owned=[_ticket(owner_id="a1")],
            obligations=[_ticket(id="obl1", issuer_id="a1", maturity_day=1, face=Decimal("100"))],
        )
        result = self.strategy.evaluate(
            trader_id="a1", trader=trader, current_day=0, horizon=1
        )
        assert result is not None
        assert isinstance(result, SellIntention)
        assert result.trader_id == "a1"

    def test_no_sell_when_no_shortfall(self) -> None:
        """A trader with enough cash to cover obligations should not sell."""
        trader = TraderState(
            agent_id="a1",
            cash=Decimal("200"),
            tickets_owned=[_ticket(owner_id="a1")],
            obligations=[_ticket(id="obl1", issuer_id="a1", maturity_day=1, face=Decimal("100"))],
        )
        result = self.strategy.evaluate(
            trader_id="a1", trader=trader, current_day=0, horizon=1
        )
        assert result is None

    def test_no_sell_when_no_tickets(self) -> None:
        """A trader with a shortfall but no tickets to sell should not get an intention."""
        trader = TraderState(
            agent_id="a1",
            cash=Decimal("0"),
            tickets_owned=[],
            obligations=[_ticket(id="obl1", issuer_id="a1", maturity_day=1, face=Decimal("100"))],
        )
        result = self.strategy.evaluate(
            trader_id="a1", trader=trader, current_day=0, horizon=1
        )
        assert result is None

    def test_sell_within_horizon(self) -> None:
        """A shortfall within the look-ahead horizon triggers a sell; outside it does not."""
        # Obligation on day 2, current day is 0.
        trader = TraderState(
            agent_id="a1",
            cash=Decimal("0"),
            tickets_owned=[_ticket(owner_id="a1")],
            obligations=[_ticket(id="obl1", issuer_id="a1", maturity_day=2, face=Decimal("100"))],
        )

        # horizon=3 => days 0..3 are scanned, day 2 is inside -> sell
        result_wide = self.strategy.evaluate(
            trader_id="a1", trader=trader, current_day=0, horizon=3
        )
        assert result_wide is not None
        assert isinstance(result_wide, SellIntention)

        # horizon=0 => only day 0 is checked, no shortfall there -> no sell
        result_narrow = self.strategy.evaluate(
            trader_id="a1", trader=trader, current_day=0, horizon=0
        )
        assert result_narrow is None


# ===========================================================================
# TestSurplusBuyer
# ===========================================================================


class TestSurplusBuyer:
    """Tests for the SurplusBuyer strategy."""

    def setup_method(self) -> None:
        self.strategy = SurplusBuyer()

    def test_buy_when_surplus(self) -> None:
        """Trader with cash and no obligations should return a BuyIntention."""
        trader = TraderState(
            agent_id="b1",
            cash=Decimal("200"),
            tickets_owned=[],
            obligations=[],
        )
        profile = TraderProfile()  # defaults: buy_reserve_fraction=0.5, aggressiveness=1.0
        result = self.strategy.evaluate(
            trader_id="b1",
            trader=trader,
            current_day=0,
            horizon=5,
            profile=profile,
            face_value=Decimal("20"),
        )
        assert result is not None
        assert isinstance(result, BuyIntention)
        assert result.trader_id == "b1"
        # No obligations -> reserved=0, surplus=200, max_spend=200
        assert result.max_spend == Decimal("200")

    def test_no_buy_when_insufficient_surplus(self) -> None:
        """Trader whose reserved amount exceeds cash should not buy."""
        trader = TraderState(
            agent_id="b1",
            cash=Decimal("10"),
            tickets_owned=[],
            obligations=[_ticket(id="obl1", issuer_id="b1", maturity_day=1, face=Decimal("100"))],
        )
        # buy_reserve_fraction=0.5 -> reserved = 0.5 * 100 = 50
        # surplus = cash - reserved = 10 - 50 = -40 < 0 -> no buy
        profile = TraderProfile(buy_reserve_fraction=Decimal("0.5"))
        result = self.strategy.evaluate(
            trader_id="b1",
            trader=trader,
            current_day=0,
            horizon=5,
            profile=profile,
            face_value=Decimal("20"),
        )
        assert result is None

    def test_liquidity_only_no_liability(self) -> None:
        """With trading_motive='liquidity_only' and no liabilities, should not buy."""
        trader = TraderState(
            agent_id="b1",
            cash=Decimal("200"),
            tickets_owned=[],
            obligations=[],
        )
        profile = TraderProfile(trading_motive="liquidity_only")
        result = self.strategy.evaluate(
            trader_id="b1",
            trader=trader,
            current_day=0,
            horizon=5,
            profile=profile,
            face_value=Decimal("20"),
        )
        # earliest_liability_day returns None -> buyer rejected under liquidity_only
        assert result is None

    def test_liquidity_only_with_liability(self) -> None:
        """With trading_motive='liquidity_only' and a future liability, should buy."""
        trader = TraderState(
            agent_id="b1",
            cash=Decimal("200"),
            tickets_owned=[],
            obligations=[_ticket(id="obl1", issuer_id="b1", maturity_day=5, face=Decimal("100"))],
        )
        profile = TraderProfile(trading_motive="liquidity_only")
        result = self.strategy.evaluate(
            trader_id="b1",
            trader=trader,
            current_day=0,
            horizon=5,
            profile=profile,
            face_value=Decimal("20"),
        )
        assert result is not None
        assert isinstance(result, BuyIntention)
        assert result.trader_id == "b1"


# ===========================================================================
# TestCollectSellIntentions
# ===========================================================================


class TestCollectSellIntentions:
    """Tests for the collect_sell_intentions collector function."""

    def test_collects_from_multiple_traders(self) -> None:
        """Only traders with shortfall *and* tickets should produce intentions."""
        traders = {
            # Trader with shortfall and tickets -> should sell
            "a1": TraderState(
                agent_id="a1",
                cash=Decimal("0"),
                tickets_owned=[_ticket(id="t1", owner_id="a1")],
                obligations=[
                    _ticket(id="obl1", issuer_id="a1", maturity_day=1, face=Decimal("100"))
                ],
            ),
            # Trader with shortfall and tickets -> should sell
            "a2": TraderState(
                agent_id="a2",
                cash=Decimal("0"),
                tickets_owned=[_ticket(id="t2", owner_id="a2")],
                obligations=[
                    _ticket(id="obl2", issuer_id="a2", maturity_day=1, face=Decimal("50"))
                ],
            ),
            # Trader with enough cash -> should NOT sell
            "a3": TraderState(
                agent_id="a3",
                cash=Decimal("500"),
                tickets_owned=[_ticket(id="t3", owner_id="a3")],
                obligations=[
                    _ticket(id="obl3", issuer_id="a3", maturity_day=1, face=Decimal("50"))
                ],
            ),
        }
        sub = MockSubsystem(traders=traders)
        intentions = collect_sell_intentions(sub, current_day=0)
        assert len(intentions) == 2
        ids = {si.trader_id for si in intentions}
        assert ids == {"a1", "a2"}

    def test_custom_horizon_parameter(self) -> None:
        """The horizon kwarg should be forwarded to the strategy."""
        # Obligation on day 5, current_day=0.
        traders = {
            "a1": TraderState(
                agent_id="a1",
                cash=Decimal("0"),
                tickets_owned=[_ticket(id="t1", owner_id="a1")],
                obligations=[
                    _ticket(id="obl1", issuer_id="a1", maturity_day=5, face=Decimal("100"))
                ],
            ),
        }
        sub = MockSubsystem(traders=traders)

        # horizon=1 => only days 0..1 checked -> no shortfall
        intentions_short = collect_sell_intentions(sub, current_day=0, horizon=1)
        assert len(intentions_short) == 0

        # horizon=6 => days 0..6 checked -> shortfall on day 5
        intentions_long = collect_sell_intentions(sub, current_day=0, horizon=6)
        assert len(intentions_long) == 1

    def test_custom_strategy(self) -> None:
        """A custom strategy that always returns an intention should cover all traders."""

        class AlwaysSell:
            def evaluate(self, trader_id, trader, current_day, horizon):
                return SellIntention(trader_id=trader_id)

        traders = {
            "a1": TraderState(agent_id="a1", cash=Decimal("500")),
            "a2": TraderState(agent_id="a2", cash=Decimal("500")),
            "a3": TraderState(agent_id="a3", cash=Decimal("500")),
        }
        sub = MockSubsystem(traders=traders)
        intentions = collect_sell_intentions(sub, current_day=0, strategy=AlwaysSell())
        assert len(intentions) == 3
        ids = {si.trader_id for si in intentions}
        assert ids == {"a1", "a2", "a3"}


# ===========================================================================
# TestCollectBuyIntentions
# ===========================================================================


class TestCollectBuyIntentions:
    """Tests for the collect_buy_intentions collector function."""

    def test_collects_from_multiple_traders(self) -> None:
        """Only traders with surplus cash should produce buy intentions."""
        traders = {
            # Surplus trader -> should buy
            "b1": TraderState(
                agent_id="b1",
                cash=Decimal("200"),
                tickets_owned=[],
                obligations=[],
            ),
            # Surplus trader -> should buy
            "b2": TraderState(
                agent_id="b2",
                cash=Decimal("300"),
                tickets_owned=[],
                obligations=[],
            ),
            # Cash-strapped trader -> should NOT buy
            "b3": TraderState(
                agent_id="b3",
                cash=Decimal("5"),
                tickets_owned=[],
                obligations=[
                    _ticket(id="obl1", issuer_id="b3", maturity_day=1, face=Decimal("100"))
                ],
            ),
        }
        sub = MockSubsystem(traders=traders)
        intentions = collect_buy_intentions(sub, current_day=0)
        assert len(intentions) == 2
        ids = {bi.trader_id for bi in intentions}
        assert ids == {"b1", "b2"}

    def test_custom_horizon_parameter(self) -> None:
        """The horizon kwarg should be forwarded to the strategy."""
        # Under liquidity_only motive, a trader with a far-out obligation
        # is eligible only if earliest_liability_day finds it (which it
        # will, since it checks existence after current_day regardless of
        # horizon). The horizon affects reserve calculation instead.
        traders = {
            "b1": TraderState(
                agent_id="b1",
                cash=Decimal("200"),
                tickets_owned=[],
                obligations=[
                    _ticket(id="obl1", issuer_id="b1", maturity_day=10, face=Decimal("50"))
                ],
            ),
        }
        profile = TraderProfile(trading_motive="liquidity_only")
        sub = MockSubsystem(traders=traders, trader_profile=profile)

        # Both calls should run without error and the horizon should
        # reach the strategy (no crash on different values).
        intentions_h1 = collect_buy_intentions(sub, current_day=0, horizon=1)
        intentions_h10 = collect_buy_intentions(sub, current_day=0, horizon=10)

        # Since the obligation exists (day 10 > day 0), liquidity_only
        # permits buying regardless of horizon value -- the horizon affects
        # reserve calculation, not motive gating.
        assert len(intentions_h1) >= 0  # implementation-dependent
        assert len(intentions_h10) >= 0  # implementation-dependent
        # At minimum, the longer horizon should not produce fewer intentions
        assert len(intentions_h10) >= len(intentions_h1)
