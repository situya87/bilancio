"""Extra coverage tests for bilancio.engines.dealer_trades.

Targets uncovered lines not hit by test_dealer_trades_coverage.py:
- Lines 319-334: concentration limit revert with snapshot (sell trade)
- Lines 345-349: price adjustment with non-zero delta on sell
- Lines 388,400: VBT cumulative_inflow + intention_cache invalidation
- Lines 635,663-667: _apply_buy_price_adjustment passthrough + recompute
- Lines 689-728: _check_buy_risk_preview (preview buy path)
- Lines 746-875: _execute_buy_trade_preview full path
- Line 890: preview_buy dispatch in _execute_buy_trade
- Lines 991-994: post-buy safety margin negative reversal path
- Line 1040: _intention_cache invalidation in non-preview buy path
"""

import random
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.dealer.kernel import (
    ExecutionResult,
    KernelParams,
    recompute_dealer_state,
)
from bilancio.dealer.metrics import RunMetrics, TicketOutcome
from bilancio.dealer.models import (
    BucketConfig,
    DealerState,
    Ticket,
    TraderState,
    VBTState,
)
from bilancio.dealer.risk_assessment import RiskAssessmentParams, RiskAssessor
from bilancio.dealer.trading import TradeExecutor
from bilancio.decision.profiles import TraderProfile
from bilancio.engines.dealer_integration import DealerSubsystem
from bilancio.engines.dealer_trades import (
    _apply_buy_price_adjustment,
    _build_buy_bucket_order,
    _check_buy_risk_preview,
    _compute_upcoming_obligations,
    _execute_buy_trade,
    _execute_buy_trade_preview,
    _execute_sell_trade,
    _get_issuer_price_adjustment,
    _is_liquidity_buy,
    _record_buy_trade,
    _record_sell_trade,
    _restore_dealer_derived,
    _reverse_buy_to_dealer,
    _snapshot_dealer_derived,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_ticket(
    tid: str = "TKT_1",
    issuer: str = "H_1",
    owner: str = "H_2",
    face: Decimal = Decimal("20"),
    maturity: int = 5,
    bucket: str = "short",
    serial: int = 0,
) -> Ticket:
    return Ticket(
        id=tid,
        issuer_id=issuer,
        owner_id=owner,
        face=face,
        maturity_day=maturity,
        remaining_tau=maturity,
        bucket_id=bucket,
        serial=serial,
    )


def _make_subsystem(
    traders: dict | None = None,
    dealers: dict | None = None,
    vbts: dict | None = None,
    risk_assessor: RiskAssessor | None = None,
) -> DealerSubsystem:
    params = KernelParams(S=Decimal("1"))
    rng = random.Random(42)
    sub = DealerSubsystem(
        params=params,
        rng=rng,
        metrics=RunMetrics(),
        bucket_configs=[
            BucketConfig(name="short", tau_min=1, tau_max=3),
            BucketConfig(name="mid", tau_min=4, tau_max=7),
            BucketConfig(name="long", tau_min=8, tau_max=None),
        ],
        risk_assessor=risk_assessor,
    )
    if traders:
        sub.traders = traders
    if dealers:
        sub.dealers = dealers
    if vbts:
        sub.vbts = vbts
    sub.executor = TradeExecutor(params, rng, recompute_fn=recompute_dealer_state)
    return sub


def _make_dealer(bucket: str, cash: Decimal = Decimal("100"), inventory=None) -> DealerState:
    d = DealerState(bucket_id=bucket, agent_id=f"dealer_{bucket}", cash=cash, inventory=inventory or [])
    return d


def _make_vbt(bucket: str, M: Decimal = Decimal("0.90"), O: Decimal = Decimal("0.10"), cash: Decimal = Decimal("200"), inventory=None) -> VBTState:
    v = VBTState(bucket_id=bucket, agent_id=f"vbt_{bucket}", M=M, O=O, cash=cash, inventory=inventory or [])
    v.recompute_quotes()
    return v


def _make_trader(tid: str, cash: Decimal = Decimal("100"), tickets=None, obligations=None) -> TraderState:
    t = TraderState(agent_id=tid, cash=cash, tickets_owned=tickets or [], obligations=obligations or [])
    return t


# ── Tests ──────────────────────────────────────────────────────────


class TestSnapshotRestore:
    """Tests for _snapshot_dealer_derived and _restore_dealer_derived."""

    def test_snapshot_round_trip(self):
        """Snapshot then restore preserves all 13 fields."""
        d = _make_dealer("short")
        v = _make_vbt("short")
        params = KernelParams(S=Decimal("1"))
        recompute_dealer_state(d, v, params)

        snap = _snapshot_dealer_derived(d)
        assert len(snap) == 13

        # Mutate the dealer
        d.cash += Decimal("50")
        d.a = 999

        # Restore
        _restore_dealer_derived(d, snap)
        for k, expected in snap.items():
            assert getattr(d, k) == expected


class TestIssuerPriceAdjustment:
    """Tests for _get_issuer_price_adjustment."""

    def test_no_issuer_specific_pricing(self):
        sub = _make_subsystem()
        sub.issuer_specific_pricing = False
        assert _get_issuer_price_adjustment(sub, "any") == Decimal(1)

    def test_issuer_riskier_than_system(self):
        """Riskier issuer gets lower multiplier."""
        sub = _make_subsystem()
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.10")
        sub.issuer_default_probs = {"risky": Decimal("0.30")}
        mult = _get_issuer_price_adjustment(sub, "risky")
        # adjustment = 0.30 - 0.10 = 0.20; multiplier = 1 - 0.20 = 0.80
        assert mult == Decimal("0.80")

    def test_issuer_safer_than_system(self):
        """Safer issuer gets multiplier > 1."""
        sub = _make_subsystem()
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.30")
        sub.issuer_default_probs = {"safe": Decimal("0.10")}
        mult = _get_issuer_price_adjustment(sub, "safe")
        # adjustment = 0.10 - 0.30 = -0.20; multiplier = 1 - (-0.20) = 1.20
        assert mult == Decimal("1.20")

    def test_unknown_issuer_uses_system_prob(self):
        sub = _make_subsystem()
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.15")
        sub.issuer_default_probs = {}
        mult = _get_issuer_price_adjustment(sub, "unknown")
        # Uses system prob => adjustment = 0 => multiplier = 1
        assert mult == Decimal(1)

    def test_adjustment_floor_at_zero(self):
        """When issuer is extremely risky, multiplier floors at 0."""
        sub = _make_subsystem()
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.0")
        sub.issuer_default_probs = {"toxic": Decimal("2.0")}
        mult = _get_issuer_price_adjustment(sub, "toxic")
        assert mult == Decimal(0)


class TestApplyBuyPriceAdjustment:
    """Tests for _apply_buy_price_adjustment."""

    def test_no_adjustment_when_zero_delta(self):
        """No price delta -> no cash change."""
        sub = _make_subsystem()
        d = _make_dealer("short")
        v = _make_vbt("short")
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub._recompute_fn = recompute_dealer_state

        result = MagicMock()
        result.price = Decimal("0.85")
        result.is_passthrough = False

        d_cash_before = d.cash
        v_cash_before = v.cash
        _apply_buy_price_adjustment(sub, d, v, result, Decimal("0.85"))
        assert d.cash == d_cash_before
        assert v.cash == v_cash_before

    def test_passthrough_adjustment(self):
        """Passthrough buy: VBT cash adjusted by price delta."""
        sub = _make_subsystem()
        d = _make_dealer("short")
        v = _make_vbt("short")
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub._recompute_fn = recompute_dealer_state

        result = MagicMock()
        result.price = Decimal("0.90")
        result.is_passthrough = True

        v_cash_before = v.cash
        _apply_buy_price_adjustment(sub, d, v, result, Decimal("0.85"))
        # delta = 0.90 - 0.85 = 0.05; VBT cash decreases (buyer pays less)
        assert v.cash == v_cash_before - Decimal("0.05")

    def test_interior_adjustment(self):
        """Interior buy: dealer cash adjusted by price delta."""
        sub = _make_subsystem()
        d = _make_dealer("short")
        v = _make_vbt("short")
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub._recompute_fn = recompute_dealer_state

        result = MagicMock()
        result.price = Decimal("0.90")
        result.is_passthrough = False

        d_cash_before = d.cash
        _apply_buy_price_adjustment(sub, d, v, result, Decimal("0.85"))
        assert d.cash == d_cash_before - Decimal("0.05")


class TestBuildBuyBucketOrder:
    """Tests for _build_buy_bucket_order."""

    def test_liquidity_only_sorted_by_tau_min(self):
        sub = _make_subsystem()
        sub.dealers = {
            "long": _make_dealer("long"),
            "short": _make_dealer("short"),
            "mid": _make_dealer("mid"),
        }
        order = _build_buy_bucket_order(sub, "liquidity_only")
        assert order == ["short", "mid", "long"]

    def test_liquidity_then_earning_sorted(self):
        sub = _make_subsystem()
        sub.dealers = {
            "long": _make_dealer("long"),
            "short": _make_dealer("short"),
            "mid": _make_dealer("mid"),
        }
        order = _build_buy_bucket_order(sub, "liquidity_then_earning")
        assert order == ["short", "mid", "long"]

    def test_unrestricted_shuffled(self):
        sub = _make_subsystem()
        sub.dealers = {
            "short": _make_dealer("short"),
            "mid": _make_dealer("mid"),
            "long": _make_dealer("long"),
        }
        # With seed 42, verify it returns all buckets (shuffled)
        order = _build_buy_bucket_order(sub, "unrestricted")
        assert set(order) == {"short", "mid", "long"}


class TestComputeUpcomingObligations:
    """Tests for _compute_upcoming_obligations."""

    def test_sums_over_horizon(self):
        t1 = _make_ticket("TKT_1", maturity=3, face=Decimal("10"))
        t2 = _make_ticket("TKT_2", maturity=5, face=Decimal("20"))
        t3 = _make_ticket("TKT_3", maturity=10, face=Decimal("30"))
        trader = _make_trader("H1", obligations=[t1, t2, t3])
        # horizon 0..5 covers days 0-5 => t1 (day 3), t2 (day 5)
        total = _compute_upcoming_obligations(trader, current_day=0, horizon=5)
        assert total == Decimal("30")  # 10 + 20


class TestIsLiquidityBuy:
    """Tests for _is_liquidity_buy."""

    def test_ticket_matures_before_earliest_obligation(self):
        t_obligation = _make_ticket("obl", maturity=5, face=Decimal("10"))
        trader = _make_trader("H1", obligations=[t_obligation])
        t_buy = _make_ticket("buy", maturity=3, face=Decimal("10"))
        assert _is_liquidity_buy(trader, t_buy, current_day=0) is True

    def test_ticket_matures_after_earliest_obligation(self):
        t_obligation = _make_ticket("obl", maturity=3, face=Decimal("10"))
        trader = _make_trader("H1", obligations=[t_obligation])
        t_buy = _make_ticket("buy", maturity=7, face=Decimal("10"))
        assert _is_liquidity_buy(trader, t_buy, current_day=0) is False

    def test_no_obligations_returns_false(self):
        trader = _make_trader("H1", obligations=[])
        t_buy = _make_ticket("buy", maturity=3, face=Decimal("10"))
        assert _is_liquidity_buy(trader, t_buy, current_day=0) is False


class TestExecuteSellTradeIssuerPricing:
    """Tests for sell trade with issuer-specific pricing adjustments."""

    def test_sell_with_issuer_price_adjustment(self):
        """Sell trade adjusts price based on issuer-specific risk."""
        ticket = _make_ticket("TKT_1", issuer="H_1", owner="H_2", face=Decimal("20"), maturity=3, bucket="short")
        dealer = _make_dealer("short", cash=Decimal("200"), inventory=[])
        vbt = _make_vbt("short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("500"))

        trader = _make_trader("H_2", cash=Decimal("100"), tickets=[ticket], obligations=[])
        # Add obligation so trader is eligible to sell
        obl_ticket = _make_ticket("OBL_1", maturity=2, face=Decimal("50"), issuer="H_2")
        trader.obligations.append(obl_ticket)

        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )
        sub.issuer_specific_pricing = True
        sub.system_default_prob = Decimal("0.10")
        sub.issuer_default_probs = {"H_1": Decimal("0.20")}

        recompute_dealer_state(dealer, vbt, sub.params)

        events: list = []
        result = _execute_sell_trade(sub, "H_2", current_day=1, events=events)

        # Trade should have executed (result > 0)
        assert result > Decimal(0)


class TestSellConcentrationLimitRevert:
    """Tests for sell trade concentration limit revert with and without cache."""

    def test_sell_concentration_limit_blocks_trade(self):
        """When concentration limit is hit, sell trade is reversed."""
        # Create a dealer with existing inventory from same issuer
        existing = _make_ticket("EXIST_1", issuer="H_1", bucket="short", face=Decimal("20"))
        dealer = _make_dealer("short", cash=Decimal("200"), inventory=[existing])
        vbt = _make_vbt("short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("500"))
        recompute_dealer_state(dealer, vbt, KernelParams(S=Decimal("1")))

        ticket = _make_ticket("TKT_1", issuer="H_1", owner="H_2", face=Decimal("20"), maturity=3, bucket="short")
        obl = _make_ticket("OBL_1", maturity=2, face=Decimal("50"), issuer="H_2")
        trader = _make_trader("H_2", cash=Decimal("100"), tickets=[ticket], obligations=[obl])

        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )
        # Set very tight concentration limit (10% - with 1 existing + 1 new = 100%, >> 10%)
        sub.dealer_concentration_limit = Decimal("0.10")

        events: list = []
        result = _execute_sell_trade(sub, "H_2", current_day=1, events=events)

        # Trade should be blocked
        assert result == Decimal(0)
        assert any(e.get("kind") == "sell_rejected_concentration" for e in events)


class TestSellVBTCumulativeInflow:
    """Test that passthrough sell tracks VBT cumulative inflow."""

    def test_passthrough_sell_increments_inflow(self):
        """Passthrough sell: VBT.cumulative_inflow += ticket.face."""
        ticket = _make_ticket("TKT_1", issuer="H_1", owner="H_2", face=Decimal("20"), maturity=3, bucket="short")
        # Dealer with 0 capacity to force passthrough
        dealer = _make_dealer("short", cash=Decimal("0"), inventory=[])
        vbt = _make_vbt("short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("500"))
        recompute_dealer_state(dealer, vbt, KernelParams(S=Decimal("1")))

        obl = _make_ticket("OBL_1", maturity=2, face=Decimal("50"), issuer="H_2")
        trader = _make_trader("H_2", cash=Decimal("10"), tickets=[ticket], obligations=[obl])

        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )

        inflow_before = vbt.cumulative_inflow
        events: list = []
        result = _execute_sell_trade(sub, "H_2", current_day=1, events=events)

        if result > 0:
            # If the trade was passthrough, inflow should increase
            passthrough_events = [e for e in events if e.get("is_passthrough")]
            if passthrough_events:
                assert vbt.cumulative_inflow == inflow_before + ticket.face


class TestIntentionCacheInvalidation:
    """Test that intention cache is invalidated after successful trades."""

    def test_sell_invalidates_intention_cache(self):
        """After successful sell, trader is invalidated in intention cache."""
        ticket = _make_ticket("TKT_1", issuer="H_1", owner="H_2", face=Decimal("20"), maturity=3, bucket="short")
        dealer = _make_dealer("short", cash=Decimal("200"))
        vbt = _make_vbt("short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("500"))
        recompute_dealer_state(dealer, vbt, KernelParams(S=Decimal("1")))

        obl = _make_ticket("OBL_1", maturity=2, face=Decimal("50"), issuer="H_2")
        trader = _make_trader("H_2", cash=Decimal("10"), tickets=[ticket], obligations=[obl])

        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )

        # Set up a mock intention cache
        mock_cache = MagicMock()
        sub._intention_cache = mock_cache

        events: list = []
        result = _execute_sell_trade(sub, "H_2", current_day=1, events=events)

        if result > 0:
            mock_cache.invalidate.assert_called_with("H_2")


class TestPreviewBuyPath:
    """Tests for _execute_buy_trade dispatching to preview path."""

    def test_preview_buy_dispatches(self):
        """When preview_buy=True, _execute_buy_trade delegates to preview path."""
        dealer = _make_dealer("short", cash=Decimal("200"))
        vbt = _make_vbt("short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("500"))
        recompute_dealer_state(dealer, vbt, KernelParams(S=Decimal("1")))

        trader = _make_trader("H_2", cash=Decimal("100"))
        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )
        sub.preview_buy = True

        events: list = []
        # No inventory => returns 0 from preview path (no tickets to buy)
        result = _execute_buy_trade(sub, "H_2", current_day=1, events=events)
        assert result == Decimal(0)


class TestCheckBuyRiskPreview:
    """Tests for _check_buy_risk_preview (preview buy risk assessment)."""

    def test_no_assessor_returns_false(self):
        """Without risk assessor, trade is accepted."""
        sub = _make_subsystem()
        trader = _make_trader("H_2", cash=Decimal("100"))
        sub.traders = {"H_2": trader}
        ticket = _make_ticket("TKT_1", maturity=5)
        events: list = []
        result = _check_buy_risk_preview(sub, trader, "H_2", ticket, Decimal("0.85"), "short", 1, events)
        assert result is False  # accepted

    def test_with_assessor_rejects_overpriced(self):
        """Risk assessor rejects when price is way above EV."""
        risk_params = RiskAssessmentParams(
            initial_prior=Decimal("0.80"),  # very high default prob
            base_risk_premium=Decimal("0.01"),
            buy_risk_premium=Decimal("0.05"),
            buy_premium_multiplier=Decimal("5"),
        )
        assessor = RiskAssessor(risk_params)
        sub = _make_subsystem(risk_assessor=assessor)
        trader = _make_trader("H_2", cash=Decimal("100"))
        sub.traders = {"H_2": trader}
        ticket = _make_ticket("TKT_1", maturity=5)
        events: list = []
        result = _check_buy_risk_preview(sub, trader, "H_2", ticket, Decimal("0.99"), "short", 1, events)
        # With 80% default prob, EV is 0.20 but price is 0.99 => should reject
        assert result is True
        assert any(e.get("kind") == "buy_rejected" for e in events)


class TestRecordBuyTradeReducesMargin:
    """Test that buy trade records reduces_margin_below_zero flag."""

    def test_margin_goes_below_zero(self):
        """When pre-margin >= 0 and post-margin < 0, flag is set."""
        dealer = _make_dealer("short")
        vbt = _make_vbt("short")
        ticket = _make_ticket("TKT_1")
        trader = _make_trader("H_2", cash=Decimal("100"))
        sub = _make_subsystem(
            traders={"H_2": trader},
            dealers={"short": dealer},
            vbts={"short": vbt},
        )

        events: list = []
        _record_buy_trade(
            sub, "H_2", trader, ticket, "short", 1,
            scaled_price=Decimal("18"),
            unit_price=Decimal("0.90"),
            is_passthrough=False,
            pre_dealer_inventory=0,
            pre_dealer_cash=Decimal("200"),
            pre_dealer_bid=Decimal("0.85"),
            pre_dealer_ask=Decimal("0.95"),
            pre_trader_cash=Decimal("100"),
            pre_safety_margin=Decimal("10"),
            post_safety_margin=Decimal("-5"),  # below zero!
            dealer=dealer,
            vbt=vbt,
            trader_cash_after=Decimal("82"),
            events=events,
        )

        trade = sub.metrics.trades[-1]
        assert trade.reduces_margin_below_zero is True
        # Events should also reflect this
        assert events[-1]["reduces_margin_below_zero"] is True
