"""Tests for per-agent profile storage (Plan 036, Phase 2).

Verifies that TraderState accepts a per-agent TraderProfile, that intention
collectors respect per-trader profiles over the subsystem default, and that
the old single-profile behavior is preserved when all traders share the
default profile.
"""

from __future__ import annotations

from copy import deepcopy
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


@dataclass
class _MockSubsystem:
    """Minimal mock of DealerSubsystem for testing intention collectors."""

    traders: dict[str, TraderState] = field(default_factory=dict)
    trader_profile: TraderProfile = field(default_factory=TraderProfile)
    face_value: Decimal = Decimal("20")


def _make_ticket(
    ticket_id: str = "t1",
    issuer: str = "issuer",
    owner: str = "owner",
    face: Decimal = Decimal("20"),
    maturity_day: int = 5,
) -> Ticket:
    return Ticket(
        id=ticket_id,
        issuer_id=issuer,
        owner_id=owner,
        face=face,
        maturity_day=maturity_day,
    )


def _make_trader(
    agent_id: str,
    cash: Decimal,
    obligations: list[Ticket] | None = None,
    tickets_owned: list[Ticket] | None = None,
    profile: TraderProfile | None = None,
) -> TraderState:
    return TraderState(
        agent_id=agent_id,
        cash=cash,
        obligations=obligations or [],
        tickets_owned=tickets_owned or [],
        profile=profile,
    )


# ===========================================================================
# TestTraderStateProfileField
# ===========================================================================


class TestTraderStateProfileField:
    """Verify that TraderState accepts and stores a per-agent profile."""

    def test_profile_defaults_to_none(self) -> None:
        """TraderState created without an explicit profile has profile=None."""
        ts = TraderState(agent_id="a1", cash=Decimal("100"))
        assert ts.profile is None

    def test_profile_accepts_trader_profile(self) -> None:
        """TraderState stores a TraderProfile when one is supplied."""
        profile = TraderProfile(risk_aversion=Decimal("0.7"), planning_horizon=5)
        ts = TraderState(agent_id="a1", cash=Decimal("100"), profile=profile)
        assert ts.profile is profile
        assert ts.profile.risk_aversion == Decimal("0.7")
        assert ts.profile.planning_horizon == 5

    def test_trader_profile_is_frozen(self) -> None:
        """TraderProfile is frozen; modifying a shared instance is an error."""
        profile = TraderProfile()
        with pytest.raises(AttributeError):
            profile.risk_aversion = Decimal("0.5")  # type: ignore[misc]

    def test_profile_shared_safely(self) -> None:
        """Because TraderProfile is frozen, two traders can share the same instance."""
        profile = TraderProfile(planning_horizon=3)
        ts1 = TraderState(agent_id="a1", cash=Decimal("10"), profile=profile)
        ts2 = TraderState(agent_id="a2", cash=Decimal("20"), profile=profile)
        assert ts1.profile is ts2.profile


# ===========================================================================
# TestHeterogeneousSellIntentions
# ===========================================================================


class TestHeterogeneousSellIntentions:
    """Two traders with different planning_horizons produce different sell outcomes."""

    def setup_method(self) -> None:
        self.strategy = LiquidityDrivenSeller()

    def test_different_horizons_different_sell_decisions(self) -> None:
        """Trader A (horizon=5) sees a day-5 shortfall; Trader B (horizon=1) does not.

        Both have the same cash=0 and an obligation on day 5, but the strategy
        only scans up to current_day + horizon.  With current_day=0:
        - horizon=5 scans days 0..5 => shortfall found on day 5
        - horizon=1 scans days 0..1 => no shortfall found
        """
        obl = _make_ticket(ticket_id="obl1", issuer="a1", face=Decimal("100"), maturity_day=5)
        ticket = _make_ticket(ticket_id="t1", owner="a1")

        trader_a = _make_trader(
            "a1",
            cash=Decimal("0"),
            obligations=[obl],
            tickets_owned=[ticket],
            profile=TraderProfile(planning_horizon=5),
        )
        # Re-create obligation for trader B (different issuer_id for clarity)
        obl_b = _make_ticket(ticket_id="obl2", issuer="b1", face=Decimal("100"), maturity_day=5)
        ticket_b = _make_ticket(ticket_id="t2", owner="b1")
        trader_b = _make_trader(
            "b1",
            cash=Decimal("0"),
            obligations=[obl_b],
            tickets_owned=[ticket_b],
            profile=TraderProfile(planning_horizon=1),
        )

        # Evaluate with each trader's own sell_horizon
        result_a = self.strategy.evaluate(
            "a1", trader_a, current_day=0, horizon=trader_a.profile.sell_horizon
        )
        result_b = self.strategy.evaluate(
            "b1", trader_b, current_day=0, horizon=trader_b.profile.sell_horizon
        )

        assert result_a is not None, "Trader A (horizon=5) should want to sell"
        assert result_b is None, "Trader B (horizon=1) should NOT want to sell"

    def test_collect_sell_respects_per_agent_horizons(self) -> None:
        """collect_sell_intentions reads each trader's own profile.sell_horizon.

        Trader a1 has planning_horizon=5, so it sees the day-5 obligation.
        Trader b1 has planning_horizon=1, so it does NOT see the day-5 obligation.
        The subsystem default is irrelevant because both traders have profiles.
        """
        obl_a = _make_ticket(ticket_id="obl_a", issuer="a1", face=Decimal("100"), maturity_day=5)
        obl_b = _make_ticket(ticket_id="obl_b", issuer="b1", face=Decimal("100"), maturity_day=5)

        traders = {
            "a1": _make_trader(
                "a1",
                cash=Decimal("0"),
                obligations=[obl_a],
                tickets_owned=[_make_ticket(ticket_id="t1", owner="a1")],
                profile=TraderProfile(planning_horizon=5),
            ),
            "b1": _make_trader(
                "b1",
                cash=Decimal("0"),
                obligations=[obl_b],
                tickets_owned=[_make_ticket(ticket_id="t2", owner="b1")],
                profile=TraderProfile(planning_horizon=1),
            ),
        }

        # Subsystem default horizon is irrelevant -- per-agent profiles override.
        sub = _MockSubsystem(
            traders=traders,
            trader_profile=TraderProfile(planning_horizon=1),
        )
        intentions = collect_sell_intentions(sub, current_day=0)
        # Only a1 (horizon=5) sees the day-5 obligation; b1 (horizon=1) does not.
        assert len(intentions) == 1
        assert intentions[0].trader_id == "a1"

    def test_collect_sell_both_wide_horizons(self) -> None:
        """When both traders have wide horizons, both produce sell intentions."""
        obl_a = _make_ticket(ticket_id="obl_a", issuer="a1", face=Decimal("100"), maturity_day=5)
        obl_b = _make_ticket(ticket_id="obl_b", issuer="b1", face=Decimal("100"), maturity_day=5)

        traders = {
            "a1": _make_trader(
                "a1",
                cash=Decimal("0"),
                obligations=[obl_a],
                tickets_owned=[_make_ticket(ticket_id="t1", owner="a1")],
                profile=TraderProfile(planning_horizon=5),
            ),
            "b1": _make_trader(
                "b1",
                cash=Decimal("0"),
                obligations=[obl_b],
                tickets_owned=[_make_ticket(ticket_id="t2", owner="b1")],
                profile=TraderProfile(planning_horizon=5),
            ),
        }
        sub = _MockSubsystem(traders=traders)
        intentions = collect_sell_intentions(sub, current_day=0)
        assert len(intentions) == 2
        ids = {si.trader_id for si in intentions}
        assert ids == {"a1", "b1"}

    def test_sell_horizon_derived_from_planning_horizon(self) -> None:
        """sell_horizon property equals planning_horizon."""
        p = TraderProfile(planning_horizon=3)
        assert p.sell_horizon == 3


# ===========================================================================
# TestHeterogeneousBuyIntentions
# ===========================================================================


class TestHeterogeneousBuyIntentions:
    """Two traders with different trading_motive or buy_reserve_fraction
    produce different buy outcomes."""

    def setup_method(self) -> None:
        self.strategy = SurplusBuyer()

    def test_different_trading_motives(self) -> None:
        """liquidity_only trader without liabilities must NOT buy;
        unrestricted trader without liabilities SHOULD buy (if surplus)."""
        profile_liq = TraderProfile(
            trading_motive="liquidity_only",
            buy_reserve_fraction=Decimal("0.5"),
        )
        profile_unr = TraderProfile(
            trading_motive="unrestricted",
            buy_reserve_fraction=Decimal("0.5"),
        )

        # Both have cash, no obligations, no tickets
        trader_a = _make_trader("a1", cash=Decimal("200"), profile=profile_liq)
        trader_b = _make_trader("b1", cash=Decimal("200"), profile=profile_unr)

        result_a = self.strategy.evaluate(
            "a1", trader_a, current_day=0, horizon=10,
            profile=profile_liq, face_value=Decimal("20"),
        )
        result_b = self.strategy.evaluate(
            "b1", trader_b, current_day=0, horizon=10,
            profile=profile_unr, face_value=Decimal("20"),
        )

        assert result_a is None, "liquidity_only with no liabilities should not buy"
        assert result_b is not None, "unrestricted with surplus should buy"
        assert isinstance(result_b, BuyIntention)

    def test_different_buy_reserve_fractions(self) -> None:
        """High reserve fraction leaves less surplus than low reserve fraction.

        Trader A: buy_reserve_fraction=1.0, cash=60, obligation=50 on day 1
          reserved = 1.0 * 50 = 50; surplus = 60 - 50 = 10 > 0 -> buy
        Trader B: buy_reserve_fraction=1.0, cash=45, obligation=50 on day 1
          reserved = 1.0 * 50 = 50; surplus = 45 - 50 = -5 < 0 -> NO buy
        Trader C: buy_reserve_fraction=0.0, cash=45, obligation=50 on day 1
          reserved = 0.0 * 50 = 0; surplus = 45 - 0 = 45 > 0 -> buy
        """
        # Use unrestricted motive so motive gating does not interfere,
        # and planning_horizon=5 to ensure obligations on day 1 are seen.
        profile_full_reserve = TraderProfile(
            buy_reserve_fraction=Decimal("1.0"),
            trading_motive="unrestricted",
            planning_horizon=5,
        )
        profile_no_reserve = TraderProfile(
            buy_reserve_fraction=Decimal("0.0"),
            trading_motive="unrestricted",
            planning_horizon=5,
        )

        obl = _make_ticket(ticket_id="obl", issuer="x", face=Decimal("50"), maturity_day=1)

        # Trader A: enough surplus even with full reserve
        trader_a = _make_trader("a1", cash=Decimal("60"), obligations=[deepcopy(obl)])
        result_a = self.strategy.evaluate(
            "a1", trader_a, current_day=0, horizon=5,
            profile=profile_full_reserve, face_value=Decimal("20"),
        )
        assert result_a is not None, "60 cash, 50 reserved -> 10 surplus -> buy"

        # Trader B: full reserve eats all cash
        trader_b = _make_trader("b1", cash=Decimal("45"), obligations=[deepcopy(obl)])
        result_b = self.strategy.evaluate(
            "b1", trader_b, current_day=0, horizon=5,
            profile=profile_full_reserve, face_value=Decimal("20"),
        )
        assert result_b is None, "45 cash, 50 reserved -> -5 surplus -> no buy"

        # Trader C: no reserve at all -> entire cash is surplus
        trader_c = _make_trader("c1", cash=Decimal("45"), obligations=[deepcopy(obl)])
        result_c = self.strategy.evaluate(
            "c1", trader_c, current_day=0, horizon=5,
            profile=profile_no_reserve, face_value=Decimal("20"),
        )
        assert result_c is not None, "45 cash, 0 reserved -> 45 surplus -> buy"

    def test_liquidity_then_earning_with_liability(self) -> None:
        """liquidity_then_earning motive should buy if there IS a future liability."""
        profile = TraderProfile(
            trading_motive="liquidity_then_earning",
            buy_reserve_fraction=Decimal("0.5"),
        )
        obl = _make_ticket(ticket_id="obl", issuer="a1", face=Decimal("20"), maturity_day=8)
        trader = _make_trader("a1", cash=Decimal("200"), obligations=[obl])

        result = self.strategy.evaluate(
            "a1", trader, current_day=0, horizon=10,
            profile=profile, face_value=Decimal("20"),
        )
        # liquidity_then_earning is NOT "liquidity_only" — the motive check
        # only gates for "liquidity_only".  So this passes the motive check.
        assert result is not None


# ===========================================================================
# TestFallbackToSubsystemDefault
# ===========================================================================


class TestFallbackToSubsystemDefault:
    """When trader.profile is None, the subsystem's trader_profile is used."""

    def test_sell_uses_subsystem_default_horizon(self) -> None:
        """Trader with profile=None gets the subsystem's sell_horizon.

        The subsystem default planning_horizon=1 means only day 0..1 are
        scanned, so a day-5 obligation is invisible.
        """
        obl = _make_ticket(ticket_id="obl", issuer="a1", face=Decimal("100"), maturity_day=5)
        traders = {
            "a1": _make_trader(
                "a1",
                cash=Decimal("0"),
                obligations=[obl],
                tickets_owned=[_make_ticket(ticket_id="t1", owner="a1")],
                profile=None,  # explicitly None -> fallback
            ),
        }
        sub = _MockSubsystem(
            traders=traders,
            trader_profile=TraderProfile(planning_horizon=1),
        )
        intentions = collect_sell_intentions(sub, current_day=0)
        # horizon=1 (from subsystem default) -> day 5 is outside scan window
        assert len(intentions) == 0

    def test_buy_uses_subsystem_default_profile(self) -> None:
        """Trader with profile=None gets the subsystem's trader_profile for buy decisions.

        Uses liquidity_only subsystem default with a trader that has no
        obligations -> should NOT produce a buy intention.
        """
        traders = {
            "b1": _make_trader(
                "b1",
                cash=Decimal("500"),
                profile=None,
            ),
        }
        sub = _MockSubsystem(
            traders=traders,
            trader_profile=TraderProfile(
                trading_motive="liquidity_only",
                buy_reserve_fraction=Decimal("0.5"),
            ),
        )
        intentions = collect_buy_intentions(sub, current_day=0)
        # liquidity_only + no liabilities -> no buy
        assert len(intentions) == 0


# ===========================================================================
# TestMixedPerAgentAndDefault
# ===========================================================================


class TestMixedPerAgentAndDefault:
    """Some traders have explicit profiles, others fall back to subsystem default."""

    def test_mixed_sell_horizons(self) -> None:
        """Trader with explicit profile uses its own horizon; trader with
        profile=None falls back to subsystem default.

        a1 has planning_horizon=5 -> sees day-5 obligation -> sells.
        b1 has profile=None -> falls back to subsystem default horizon=1 ->
        does NOT see day-5 obligation -> does NOT sell.
        """
        obl_a = _make_ticket(ticket_id="obl_a", issuer="a1", face=Decimal("100"), maturity_day=5)
        obl_b = _make_ticket(ticket_id="obl_b", issuer="b1", face=Decimal("100"), maturity_day=5)

        traders = {
            # Explicit wide horizon
            "a1": _make_trader(
                "a1",
                cash=Decimal("0"),
                obligations=[obl_a],
                tickets_owned=[_make_ticket(ticket_id="t1", owner="a1")],
                profile=TraderProfile(planning_horizon=5),
            ),
            # No profile -> uses subsystem default
            "b1": _make_trader(
                "b1",
                cash=Decimal("0"),
                obligations=[obl_b],
                tickets_owned=[_make_ticket(ticket_id="t2", owner="b1")],
                profile=None,
            ),
        }

        # Subsystem with narrow default
        sub = _MockSubsystem(
            traders=traders,
            trader_profile=TraderProfile(planning_horizon=1),
        )

        intentions = collect_sell_intentions(sub, current_day=0)
        # a1 uses its own horizon=5 -> sees day-5 obligation -> sells
        # b1 falls back to subsystem default horizon=1 -> does not see day 5
        assert len(intentions) == 1
        assert intentions[0].trader_id == "a1"

    def test_mixed_buy_motives(self) -> None:
        """Trader with unrestricted motive buys even without liabilities;
        trader with profile=None falls back to subsystem default (liquidity_only).
        """
        traders = {
            "a1": _make_trader(
                "a1",
                cash=Decimal("500"),
                profile=TraderProfile(
                    trading_motive="unrestricted",
                    buy_reserve_fraction=Decimal("0.5"),
                ),
            ),
            "b1": _make_trader(
                "b1",
                cash=Decimal("500"),
                profile=None,
            ),
        }
        sub = _MockSubsystem(
            traders=traders,
            trader_profile=TraderProfile(
                trading_motive="liquidity_only",
                buy_reserve_fraction=Decimal("0.5"),
            ),
        )
        intentions = collect_buy_intentions(sub, current_day=0)
        # a1 uses its own profile (unrestricted, no liabilities needed) -> buys
        # b1 falls back to subsystem default (liquidity_only, no liabilities) -> does not buy
        assert len(intentions) == 1
        assert intentions[0].trader_id == "a1"


# ===========================================================================
# TestDefaultBehaviorPreserved
# ===========================================================================


class TestDefaultBehaviorPreserved:
    """When all traders share the subsystem default profile, results must be
    identical to the old single-profile behaviour."""

    def _setup_subsystem(
        self, profile: TraderProfile, assign_to_traders: bool
    ) -> _MockSubsystem:
        """Build a subsystem with two sellers and one buyer.

        If assign_to_traders is True, each trader gets the profile as
        their own.  Otherwise, profile=None (fallback to subsystem default).
        """
        obl_s1 = _make_ticket(ticket_id="obl_s1", issuer="s1", face=Decimal("100"), maturity_day=3)
        obl_s2 = _make_ticket(ticket_id="obl_s2", issuer="s2", face=Decimal("100"), maturity_day=3)
        obl_buy = _make_ticket(ticket_id="obl_buy", issuer="buyer", face=Decimal("20"), maturity_day=8)

        trader_profile_or_none = profile if assign_to_traders else None

        traders = {
            "s1": _make_trader(
                "s1",
                cash=Decimal("0"),
                obligations=[obl_s1],
                tickets_owned=[_make_ticket(ticket_id="t1", owner="s1")],
                profile=trader_profile_or_none,
            ),
            "s2": _make_trader(
                "s2",
                cash=Decimal("0"),
                obligations=[obl_s2],
                tickets_owned=[_make_ticket(ticket_id="t2", owner="s2")],
                profile=trader_profile_or_none,
            ),
            "buyer": _make_trader(
                "buyer",
                cash=Decimal("500"),
                obligations=[obl_buy],
                profile=trader_profile_or_none,
            ),
        }
        return _MockSubsystem(traders=traders, trader_profile=profile)

    def test_sell_identical_with_explicit_and_default_profiles(self) -> None:
        """Sell intentions are the same whether profiles are per-trader or subsystem-level."""
        profile = TraderProfile(planning_horizon=5)

        sub_explicit = self._setup_subsystem(profile, assign_to_traders=True)
        sub_default = self._setup_subsystem(profile, assign_to_traders=False)

        sells_explicit = collect_sell_intentions(sub_explicit, current_day=0)
        sells_default = collect_sell_intentions(sub_default, current_day=0)

        ids_explicit = sorted(si.trader_id for si in sells_explicit)
        ids_default = sorted(si.trader_id for si in sells_default)
        assert ids_explicit == ids_default

    def test_buy_identical_with_explicit_and_default_profiles(self) -> None:
        """Buy intentions are the same whether profiles are per-trader or subsystem-level."""
        profile = TraderProfile(
            planning_horizon=10,
            trading_motive="liquidity_only",
            buy_reserve_fraction=Decimal("0.5"),
        )

        sub_explicit = self._setup_subsystem(profile, assign_to_traders=True)
        sub_default = self._setup_subsystem(profile, assign_to_traders=False)

        buys_explicit = collect_buy_intentions(sub_explicit, current_day=0)
        buys_default = collect_buy_intentions(sub_default, current_day=0)

        ids_explicit = sorted(bi.trader_id for bi in buys_explicit)
        ids_default = sorted(bi.trader_id for bi in buys_default)
        assert ids_explicit == ids_default

    def test_buy_max_spend_identical(self) -> None:
        """Not just the set of buyers, but the max_spend values match."""
        profile = TraderProfile(
            planning_horizon=10,
            trading_motive="unrestricted",
            buy_reserve_fraction=Decimal("0.5"),
        )

        sub_explicit = self._setup_subsystem(profile, assign_to_traders=True)
        sub_default = self._setup_subsystem(profile, assign_to_traders=False)

        buys_explicit = collect_buy_intentions(sub_explicit, current_day=0)
        buys_default = collect_buy_intentions(sub_default, current_day=0)

        spend_explicit = {bi.trader_id: bi.max_spend for bi in buys_explicit}
        spend_default = {bi.trader_id: bi.max_spend for bi in buys_default}
        assert spend_explicit == spend_default


# ===========================================================================
# TestProfileParameterEffects
# ===========================================================================


class TestProfileParameterEffects:
    """Additional tests verifying individual profile parameters affect outcomes."""

    def test_risk_aversion_affects_buy_premium(self) -> None:
        """Higher risk_aversion -> higher buy_risk_premium (affects threshold)."""
        low_ra = TraderProfile(risk_aversion=Decimal("0"))
        high_ra = TraderProfile(risk_aversion=Decimal("1"))

        # buy_risk_premium = 0.01 + 0.02 * risk_aversion
        assert low_ra.buy_risk_premium == Decimal("0.01")
        assert high_ra.buy_risk_premium == Decimal("0.03")
        assert high_ra.buy_risk_premium > low_ra.buy_risk_premium

    def test_aggressiveness_affects_surplus_threshold(self) -> None:
        """aggressiveness=1 -> threshold_factor=0 (buy eagerly);
        aggressiveness=0 -> threshold_factor=1 (conservative, need face_value surplus).
        """
        eager = TraderProfile(aggressiveness=Decimal("1.0"))
        cautious = TraderProfile(aggressiveness=Decimal("0.0"))

        assert eager.surplus_threshold_factor == Decimal("0")
        assert cautious.surplus_threshold_factor == Decimal("1")

    def test_aggressiveness_zero_blocks_small_surplus_buys(self) -> None:
        """With aggressiveness=0, surplus must exceed face_value to trigger a buy.

        cash=30, no obligations -> surplus=30.
        face_value=20, threshold_factor=1 -> threshold = 20 * 1 = 20.
        surplus 30 > threshold 20 -> buy.

        cash=15, no obligations -> surplus=15.
        threshold 20 > surplus 15 -> no buy.
        """
        strategy = SurplusBuyer()
        cautious = TraderProfile(
            aggressiveness=Decimal("0.0"),
            trading_motive="unrestricted",
            buy_reserve_fraction=Decimal("0.0"),
        )

        rich = _make_trader("rich", cash=Decimal("30"))
        result_rich = strategy.evaluate(
            "rich", rich, current_day=0, horizon=5,
            profile=cautious, face_value=Decimal("20"),
        )
        assert result_rich is not None

        poor = _make_trader("poor", cash=Decimal("15"))
        result_poor = strategy.evaluate(
            "poor", poor, current_day=0, horizon=5,
            profile=cautious, face_value=Decimal("20"),
        )
        assert result_poor is None

    def test_buy_horizon_equals_planning_horizon(self) -> None:
        """buy_horizon property equals planning_horizon (full look-ahead)."""
        p = TraderProfile(planning_horizon=7)
        assert p.buy_horizon == 7

    def test_planning_horizon_validation(self) -> None:
        """planning_horizon outside [1, 20] raises ValueError."""
        with pytest.raises(ValueError, match="planning_horizon"):
            TraderProfile(planning_horizon=0)
        with pytest.raises(ValueError, match="planning_horizon"):
            TraderProfile(planning_horizon=21)

    def test_buy_reserve_fraction_validation(self) -> None:
        """buy_reserve_fraction outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="buy_reserve_fraction"):
            TraderProfile(buy_reserve_fraction=Decimal("-0.1"))
        with pytest.raises(ValueError, match="buy_reserve_fraction"):
            TraderProfile(buy_reserve_fraction=Decimal("1.1"))

    def test_trading_motive_validation(self) -> None:
        """Invalid trading_motive raises ValueError."""
        with pytest.raises(ValueError, match="trading_motive"):
            TraderProfile(trading_motive="yolo")
