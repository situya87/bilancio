"""Tests for the ActivityProfile protocol and supporting types (Plan 036, Phase 1).

Tests cover:
1. CashFlowEntry and CashFlowPosition — construction, properties, methods
2. ObservedState, MarketQuote — construction and field access
3. Valuations — construction, get, value_of
4. RiskView — construction and field access
5. Action, ActionSet, ActionTemplate — construction
6. ActivityProfile — protocol compliance (isinstance checks)
7. ComposedProfile — phase dispatch
8. build_cash_flow_position_from_trader — bridge from TraderState
9. Action type constants
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from bilancio.decision.activity import (
    ACTION_BUY,
    ACTION_EXTEND_LOAN,
    ACTION_HOLD,
    ACTION_SELL,
    ACTION_SET_QUOTES,
    Action,
    ActionSet,
    ActionTemplate,
    ActivityProfile,
    CashFlowEntry,
    CashFlowPosition,
    ComposedProfile,
    MarketQuote,
    ObservedState,
    RiskView,
    Valuations,
    build_cash_flow_position_from_trader,
)
from bilancio.information.estimates import Estimate


# ── Helpers ──────────────────────────────────────────────────────────────────


def _entry(day: int, amount: int | str, **kw) -> CashFlowEntry:
    """Shorthand for creating a CashFlowEntry."""
    return CashFlowEntry(day=day, amount=Decimal(str(amount)), **kw)


def _position(
    cash: int | str = 100,
    obligations: tuple[CashFlowEntry, ...] = (),
    entitlements: tuple[CashFlowEntry, ...] = (),
    horizon: int = 10,
    day: int = 5,
    **kw,
) -> CashFlowPosition:
    """Shorthand for creating a CashFlowPosition."""
    return CashFlowPosition(
        cash=Decimal(str(cash)),
        obligations=obligations,
        entitlements=entitlements,
        planning_horizon=horizon,
        current_day=day,
        **kw,
    )


def _estimate(target_id: str = "ticket_1", value: str = "17.00") -> Estimate:
    """Shorthand for a test Estimate."""
    return Estimate(
        value=Decimal(value),
        estimator_id="test",
        target_id=target_id,
        target_type="instrument",
        estimation_day=5,
        method="test_method",
    )


# ── 1. CashFlowEntry ────────────────────────────────────────────────────────


class TestCashFlowEntry:
    def test_basic_construction(self):
        entry = CashFlowEntry(day=5, amount=Decimal("100"))
        assert entry.day == 5
        assert entry.amount == Decimal("100")
        assert entry.counterparty_id == ""
        assert entry.instrument_id == ""
        assert entry.instrument_kind == ""

    def test_full_construction(self):
        entry = CashFlowEntry(
            day=8,
            amount=Decimal("500"),
            counterparty_id="firm_a",
            instrument_id="pay_001",
            instrument_kind="payable",
        )
        assert entry.counterparty_id == "firm_a"
        assert entry.instrument_kind == "payable"

    def test_frozen(self):
        entry = _entry(5, 100)
        with pytest.raises(AttributeError):
            entry.day = 6  # type: ignore[misc]


# ── 2. CashFlowPosition ─────────────────────────────────────────────────────


class TestCashFlowPosition:
    def test_empty_position(self):
        pos = _position(cash=100)
        assert pos.cash == Decimal("100")
        assert pos.total_obligations_in_horizon == Decimal("0")
        assert pos.total_entitlements_in_horizon == Decimal("0")
        assert pos.surplus() == Decimal("100")
        assert pos.max_shortfall_in_horizon() == Decimal("0")

    def test_obligations_in_horizon(self):
        pos = _position(
            cash=100,
            obligations=(
                _entry(6, 30),   # within horizon (day 5 + 10 = 15)
                _entry(10, 50),  # within horizon
                _entry(20, 100),  # outside horizon
            ),
            horizon=10,
            day=5,
        )
        assert pos.total_obligations_in_horizon == Decimal("80")
        assert pos.surplus() == Decimal("20")

    def test_entitlements_in_horizon(self):
        pos = _position(
            cash=100,
            entitlements=(
                _entry(7, 40),   # within horizon
                _entry(12, 60),  # within horizon
                _entry(20, 200),  # outside horizon
            ),
            horizon=10,
            day=5,
        )
        assert pos.total_entitlements_in_horizon == Decimal("100")

    def test_net_cash_flow(self):
        pos = _position(
            cash=50,
            obligations=(_entry(7, 100),),
            entitlements=(_entry(8, 60),),
            horizon=10,
            day=5,
        )
        # net = entitlements - obligations = 60 - 100 = -40
        assert pos.net_cash_flow_in_horizon == Decimal("-40")

    def test_shortfall_on_day(self):
        pos = _position(
            cash=30,
            obligations=(_entry(7, 100),),
            day=5,
        )
        # On day 7: due 100, have 30 → shortfall 70
        assert pos.shortfall(7) == Decimal("70")
        # On day 6: nothing due → shortfall 0
        assert pos.shortfall(6) == Decimal("0")

    def test_shortfall_zero_when_enough_cash(self):
        pos = _position(
            cash=200,
            obligations=(_entry(7, 100),),
            day=5,
        )
        assert pos.shortfall(7) == Decimal("0")

    def test_max_shortfall_cumulative(self):
        """Cumulative shortfall: earlier payments reduce cash for later ones."""
        pos = _position(
            cash=80,
            obligations=(
                _entry(5, 50),  # day 5: pay 50, remaining 30
                _entry(6, 40),  # day 6: pay 40, remaining -10 → shortfall 10
            ),
            horizon=5,
            day=5,
        )
        assert pos.max_shortfall_in_horizon() == Decimal("10")

    def test_max_shortfall_with_incoming(self):
        """Incoming entitlements reduce shortfall."""
        pos = _position(
            cash=80,
            obligations=(
                _entry(5, 50),  # day 5: pay 50, remaining 30
                _entry(7, 60),  # day 7: pay 60
            ),
            entitlements=(
                _entry(6, 40),  # day 6: receive 40, remaining 70
            ),
            horizon=5,
            day=5,
        )
        # day 5: 80 - 50 = 30
        # day 6: 30 + 40 = 70
        # day 7: 70 - 60 = 10
        assert pos.max_shortfall_in_horizon() == Decimal("0")

    def test_liquid_resources(self):
        pos = _position(
            cash=100,
            reserves=Decimal("50"),
            deposits=Decimal("30"),
        )
        assert pos.liquid_resources == Decimal("180")

    def test_obligations_on_day(self):
        pos = _position(
            obligations=(
                _entry(7, 30, counterparty_id="a"),
                _entry(7, 50, counterparty_id="b"),
                _entry(8, 20, counterparty_id="c"),
            ),
            day=5,
        )
        day_7 = pos.obligations_on_day(7)
        assert len(day_7) == 2
        assert sum(e.amount for e in day_7) == Decimal("80")

    def test_entitlements_on_day(self):
        pos = _position(
            entitlements=(
                _entry(7, 40),
                _entry(8, 60),
            ),
            day=5,
        )
        assert len(pos.entitlements_on_day(7)) == 1
        assert len(pos.entitlements_on_day(9)) == 0

    def test_earliest_obligation_day(self):
        pos = _position(
            obligations=(
                _entry(8, 30),
                _entry(6, 50),
                _entry(10, 20),
            ),
            day=5,
        )
        assert pos.earliest_obligation_day() == 6

    def test_earliest_obligation_day_none(self):
        pos = _position(obligations=(), day=5)
        assert pos.earliest_obligation_day() is None

    def test_earliest_obligation_ignores_past(self):
        pos = _position(
            obligations=(
                _entry(3, 50),  # in the past (before day 5)
                _entry(8, 30),
            ),
            day=5,
        )
        assert pos.earliest_obligation_day() == 8

    def test_frozen(self):
        pos = _position()
        with pytest.raises(AttributeError):
            pos.cash = Decimal("999")  # type: ignore[misc]


# ── 3. ObservedState ─────────────────────────────────────────────────────────


class TestObservedState:
    def test_minimal_construction(self):
        pos = _position()
        obs = ObservedState(position=pos)
        assert obs.position is pos
        assert obs.system_default_rate is None
        assert obs.counterparty_default_probs is None
        assert obs.market_prices is None
        assert obs.ratings is None
        assert obs.extra == {}

    def test_with_market_prices(self):
        pos = _position()
        quote = MarketQuote(
            bid=Decimal("0.78"),
            ask=Decimal("0.82"),
            mid=Decimal("0.80"),
            instrument_class="payable",
            bucket="short",
        )
        obs = ObservedState(
            position=pos,
            market_prices={"short": quote},
        )
        assert obs.market_prices["short"].bid == Decimal("0.78")

    def test_with_all_fields(self):
        pos = _position()
        obs = ObservedState(
            position=pos,
            system_default_rate=Decimal("0.15"),
            counterparty_default_probs={"firm_a": Decimal("0.20")},
            ratings={"firm_a": Decimal("0.18")},
            extra={"custom_metric": 42},
        )
        assert obs.system_default_rate == Decimal("0.15")
        assert obs.counterparty_default_probs["firm_a"] == Decimal("0.20")
        assert obs.ratings["firm_a"] == Decimal("0.18")
        assert obs.extra["custom_metric"] == 42


class TestMarketQuote:
    def test_construction(self):
        quote = MarketQuote(
            bid=Decimal("0.78"),
            ask=Decimal("0.82"),
            mid=Decimal("0.80"),
        )
        assert quote.bid == Decimal("0.78")
        assert quote.ask == Decimal("0.82")
        assert quote.mid == Decimal("0.80")
        assert quote.instrument_class == ""
        assert quote.bucket == ""

    def test_frozen(self):
        quote = MarketQuote(bid=Decimal("0.78"), ask=Decimal("0.82"), mid=Decimal("0.80"))
        with pytest.raises(AttributeError):
            quote.bid = Decimal("0.90")  # type: ignore[misc]


# ── 4. Valuations ────────────────────────────────────────────────────────────


class TestValuations:
    def test_empty(self):
        v = Valuations()
        assert len(v) == 0
        assert v.get("anything") is None
        assert v.value_of("anything") is None

    def test_with_estimates(self):
        est1 = _estimate("ticket_1", "17.00")
        est2 = _estimate("ticket_2", "15.50")
        v = Valuations(
            estimates={"ticket_1": est1, "ticket_2": est2},
            method="ev_hold",
        )
        assert len(v) == 2
        assert v.get("ticket_1") is est1
        assert v.value_of("ticket_1") == Decimal("17.00")
        assert v.value_of("ticket_2") == Decimal("15.50")
        assert v.value_of("ticket_3") is None
        assert v.method == "ev_hold"


# ── 5. RiskView ──────────────────────────────────────────────────────────────


class TestRiskView:
    def test_construction(self):
        pos = _position(cash=100)
        vals = Valuations()
        rv = RiskView(
            position=pos,
            valuations=vals,
            urgency=Decimal("0.5"),
            liquidity_ratio=Decimal("1.3"),
            asset_value=Decimal("200"),
            wealth=Decimal("300"),
        )
        assert rv.urgency == Decimal("0.5")
        assert rv.liquidity_ratio == Decimal("1.3")
        assert rv.asset_value == Decimal("200")
        assert rv.wealth == Decimal("300")

    def test_defaults(self):
        pos = _position()
        rv = RiskView(position=pos, valuations=Valuations())
        assert rv.urgency == Decimal("0")
        assert rv.liquidity_ratio == Decimal("1")
        assert rv.asset_value == Decimal("0")
        assert rv.wealth == Decimal("0")
        assert rv.extra == {}

    def test_extra(self):
        pos = _position()
        rv = RiskView(
            position=pos,
            valuations=Valuations(),
            extra={"inventory_imbalance": Decimal("0.3")},
        )
        assert rv.extra["inventory_imbalance"] == Decimal("0.3")


# ── 6. Action, ActionSet, ActionTemplate ─────────────────────────────────────


class TestAction:
    def test_hold_action(self):
        a = Action(action_type=ACTION_HOLD)
        assert a.action_type == "hold"
        assert a.params == {}

    def test_sell_action(self):
        a = Action(
            action_type=ACTION_SELL,
            params={"instrument_id": "ticket_1", "min_price": Decimal("0.75")},
        )
        assert a.action_type == "sell"
        assert a.params["instrument_id"] == "ticket_1"

    def test_buy_action(self):
        a = Action(
            action_type=ACTION_BUY,
            params={"max_price": Decimal("0.85"), "bucket": "short"},
        )
        assert a.action_type == "buy"

    def test_set_quotes_action(self):
        a = Action(
            action_type=ACTION_SET_QUOTES,
            params={"bid": Decimal("0.78"), "ask": Decimal("0.82"), "bucket": "short"},
        )
        assert a.action_type == "set_quotes"

    def test_extend_loan_action(self):
        a = Action(
            action_type=ACTION_EXTEND_LOAN,
            params={"borrower_id": "firm_a", "amount": 500, "rate": Decimal("0.08")},
        )
        assert a.action_type == "extend_loan"

    def test_frozen(self):
        a = Action(action_type=ACTION_HOLD)
        with pytest.raises(AttributeError):
            a.action_type = "sell"  # type: ignore[misc]


class TestActionSet:
    def test_empty(self):
        aset = ActionSet()
        assert aset.available == []
        assert aset.phase == ""

    def test_with_templates(self):
        templates = [
            ActionTemplate(action_type=ACTION_SELL, constraints={"instruments": ["t1", "t2"]}),
            ActionTemplate(action_type=ACTION_BUY, constraints={"max_spend": Decimal("500")}),
        ]
        aset = ActionSet(available=templates, phase="B_Dealer")
        assert len(aset.available) == 2
        assert aset.phase == "B_Dealer"


class TestActionTemplate:
    def test_construction(self):
        t = ActionTemplate(
            action_type=ACTION_SELL,
            constraints={"instruments": ["ticket_1"], "min_bid": Decimal("0.70")},
        )
        assert t.action_type == "sell"
        assert t.constraints["min_bid"] == Decimal("0.70")


# ── 7. ActivityProfile Protocol ──────────────────────────────────────────────


@dataclass(frozen=True)
class _MockTradingProfile:
    """Minimal ActivityProfile implementation for testing protocol compliance."""

    risk_aversion: Decimal = Decimal("0.5")

    @property
    def activity_type(self) -> str:
        return "trading"

    @property
    def instrument_class(self) -> str | None:
        return "payable"

    def observe(self, info, position):
        return ObservedState(position=position)

    def value(self, observed):
        return Valuations(method="mock_ev")

    def assess(self, valuations, position):
        return RiskView(position=position, valuations=valuations)

    def choose(self, risk_view, action_set):
        return Action(action_type=ACTION_HOLD)


@dataclass(frozen=True)
class _MockDealerProfile:
    """Minimal market-making ActivityProfile for testing."""

    @property
    def activity_type(self) -> str:
        return "market_making"

    @property
    def instrument_class(self) -> str | None:
        return "payable"

    def observe(self, info, position):
        return ObservedState(position=position)

    def value(self, observed):
        return Valuations(method="mark_to_mid")

    def assess(self, valuations, position):
        return RiskView(position=position, valuations=valuations)

    def choose(self, risk_view, action_set):
        return Action(
            action_type=ACTION_SET_QUOTES,
            params={"bid": Decimal("0.78"), "ask": Decimal("0.82")},
        )


class TestActivityProfileProtocol:
    def test_isinstance_check(self):
        """Mock profiles must satisfy the runtime-checkable ActivityProfile protocol."""
        trader = _MockTradingProfile()
        dealer = _MockDealerProfile()
        assert isinstance(trader, ActivityProfile)
        assert isinstance(dealer, ActivityProfile)

    def test_activity_type(self):
        assert _MockTradingProfile().activity_type == "trading"
        assert _MockDealerProfile().activity_type == "market_making"

    def test_instrument_class(self):
        assert _MockTradingProfile().instrument_class == "payable"
        assert _MockDealerProfile().instrument_class == "payable"

    def test_full_pipeline(self):
        """Run the complete 4-step pipeline through a mock profile."""
        profile = _MockTradingProfile()
        pos = _position(cash=100, obligations=(_entry(7, 50),), day=5)

        observed = profile.observe(None, pos)
        assert isinstance(observed, ObservedState)
        assert observed.position is pos

        valuations = profile.value(observed)
        assert isinstance(valuations, Valuations)

        risk_view = profile.assess(valuations, pos)
        assert isinstance(risk_view, RiskView)

        action = profile.choose(risk_view, ActionSet())
        assert isinstance(action, Action)
        assert action.action_type == ACTION_HOLD

    def test_non_conforming_object_fails(self):
        """An object missing protocol methods should NOT be an ActivityProfile."""

        class _Incomplete:
            @property
            def activity_type(self) -> str:
                return "trading"

            # Missing: instrument_class, observe, value, assess, choose

        assert not isinstance(_Incomplete(), ActivityProfile)


# ── 8. ComposedProfile ───────────────────────────────────────────────────────


class TestComposedProfile:
    def test_construction(self):
        trading = _MockTradingProfile()
        dealer = _MockDealerProfile()
        composed = ComposedProfile(
            activities=(trading, dealer),
            agent_id="bank_1",
        )
        assert len(composed.activities) == 2
        assert composed.agent_id == "bank_1"

    def test_for_phase_trading(self):
        trading = _MockTradingProfile()
        dealer = _MockDealerProfile()
        composed = ComposedProfile(activities=(trading, dealer))

        # Trading phase should return both trading and market_making profiles
        dealer_phase = composed.for_phase("B_Dealer")
        assert len(dealer_phase) == 2
        assert trading in dealer_phase
        assert dealer in dealer_phase

    def test_for_phase_lending(self):
        trading = _MockTradingProfile()
        composed = ComposedProfile(activities=(trading,))

        # Lending phase should return nothing (no lending profile)
        lending_phase = composed.for_phase("B_Lending")
        assert len(lending_phase) == 0

    def test_for_phase_unknown(self):
        trading = _MockTradingProfile()
        composed = ComposedProfile(activities=(trading,))

        # Unknown phase returns empty
        unknown = composed.for_phase("X_Unknown")
        assert len(unknown) == 0

    def test_single_profile_simple_case(self):
        """Single-profile composition: one profile is the whole agent."""
        trading = _MockTradingProfile()
        composed = ComposedProfile(activities=(trading,))
        dealer_phase = composed.for_phase("B_Dealer")
        assert len(dealer_phase) == 1
        assert dealer_phase[0] is trading

    def test_frozen(self):
        composed = ComposedProfile(activities=(_MockTradingProfile(),))
        with pytest.raises(AttributeError):
            composed.agent_id = "other"  # type: ignore[misc]


# ── 9. build_cash_flow_position_from_trader ──────────────────────────────────


class _MockTicket:
    """Minimal mock for Ticket to test the bridge function."""

    def __init__(self, ticket_id, issuer_id, owner_id, face, maturity_day):
        self.id = ticket_id
        self.issuer_id = issuer_id
        self.owner_id = owner_id
        self.face = Decimal(str(face))
        self.maturity_day = maturity_day


class _MockTraderState:
    """Minimal mock for TraderState to test the bridge function."""

    def __init__(self, cash, tickets_owned=None, obligations=None):
        self.cash = Decimal(str(cash))
        self.tickets_owned = tickets_owned or []
        self.obligations = obligations or []


class TestBuildCashFlowPositionFromTrader:
    def test_empty_trader(self):
        trader = _MockTraderState(cash=100)
        pos = build_cash_flow_position_from_trader(trader, current_day=5)
        assert pos.cash == Decimal("100")
        assert pos.obligations == ()
        assert pos.entitlements == ()
        assert pos.current_day == 5
        assert pos.planning_horizon == 10

    def test_with_obligations(self):
        obligations = [
            _MockTicket("t1", "firm_a", "me", 20, 8),
            _MockTicket("t2", "firm_b", "me", 30, 10),
        ]
        trader = _MockTraderState(cash=50, obligations=obligations)
        pos = build_cash_flow_position_from_trader(trader, current_day=5, planning_horizon=15)

        assert len(pos.obligations) == 2
        assert pos.obligations[0].day == 8
        assert pos.obligations[0].amount == Decimal("20")
        assert pos.obligations[0].counterparty_id == "me"
        assert pos.obligations[0].instrument_kind == "payable"
        assert pos.planning_horizon == 15

    def test_with_entitlements(self):
        tickets_owned = [
            _MockTicket("t3", "firm_c", "me", 25, 7),
        ]
        trader = _MockTraderState(cash=50, tickets_owned=tickets_owned)
        pos = build_cash_flow_position_from_trader(trader, current_day=5)

        assert len(pos.entitlements) == 1
        assert pos.entitlements[0].day == 7
        assert pos.entitlements[0].amount == Decimal("25")
        assert pos.entitlements[0].counterparty_id == "firm_c"

    def test_full_trader(self):
        obligations = [_MockTicket("t1", "firm_a", "firm_b", 20, 8)]
        tickets_owned = [_MockTicket("t2", "firm_c", "me", 30, 10)]
        trader = _MockTraderState(cash=100, tickets_owned=tickets_owned, obligations=obligations)
        pos = build_cash_flow_position_from_trader(trader, current_day=5)

        assert pos.cash == Decimal("100")
        assert len(pos.obligations) == 1
        assert len(pos.entitlements) == 1
        assert pos.surplus() == Decimal("80")  # 100 - 20

    def test_result_is_frozen(self):
        trader = _MockTraderState(cash=100)
        pos = build_cash_flow_position_from_trader(trader, current_day=5)
        with pytest.raises(AttributeError):
            pos.cash = Decimal("999")  # type: ignore[misc]


# ── 10. Action type constants ────────────────────────────────────────────────


class TestActionConstants:
    def test_constants_are_strings(self):
        assert isinstance(ACTION_HOLD, str)
        assert isinstance(ACTION_SELL, str)
        assert isinstance(ACTION_BUY, str)
        assert isinstance(ACTION_SET_QUOTES, str)
        assert isinstance(ACTION_EXTEND_LOAN, str)

    def test_constants_are_distinct(self):
        all_constants = [
            ACTION_HOLD, ACTION_SELL, ACTION_BUY, ACTION_SET_QUOTES,
            ACTION_EXTEND_LOAN,
        ]
        assert len(set(all_constants)) == len(all_constants)
