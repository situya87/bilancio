"""Tests for Phase 5 ActivityProfile wrapper classes (Plan 036).

Tests cover all 8 activity wrappers defined in ``bilancio.decision.activities``:

    1. TradingActivity        — wraps TraderProfile
    2. MarketMakingActivity   — wraps dealer kernel
    3. OutsideLiquidityActivity — wraps VBTProfile
    4. LendingActivity        — wraps LenderProfile
    5. RatingActivity         — wraps RatingProfile
    6. BankLendingActivity    — wraps BankProfile lending
    7. BankTreasuryActivity   — wraps BankProfile treasury
    8. CBActivity             — wraps CB logic

Test groups:
    Group 1: Protocol conformance (isinstance checks)
    Group 2: Property values (activity_type, instrument_class)
    Group 3: From-profile factory classmethods
    Group 4: Pipeline tests (observe -> value -> assess -> choose)
    Group 5: ComposedProfile integration
    Group 6: Frozen immutability
    Group 7: LendingActivity derived properties
    Group 8: Exports
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from bilancio.decision.activities import (
    BankLendingActivity,
    BankTreasuryActivity,
    CBActivity,
    LendingActivity,
    MarketMakingActivity,
    OutsideLiquidityActivity,
    RatingActivity,
    TradingActivity,
)
from bilancio.decision.activity import (
    ACTION_BUY,
    ACTION_EXTEND_LOAN,
    ACTION_HOLD,
    ACTION_SELL,
    ACTION_SET_ANCHORS,
    ACTION_SET_CORRIDOR,
    ACTION_SET_QUOTES,
    Action,
    ActionSet,
    ActionTemplate,
    ActivityProfile,
    CashFlowEntry,
    CashFlowPosition,
    ComposedProfile,
    ObservedState,
    RiskView,
    Valuations,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _entry(day: int, amount: int | str, **kw) -> CashFlowEntry:
    """Shorthand for creating a CashFlowEntry."""
    return CashFlowEntry(day=day, amount=Decimal(str(amount)), **kw)


def _position(
    cash: int | str = 100,
    obligations: tuple[CashFlowEntry, ...] = (),
    entitlements: tuple[CashFlowEntry, ...] = (),
    horizon: int = 10,
    day: int = 0,
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


class MockInfoService:
    """Minimal mock of InformationService for pipeline tests.

    Provides the subset of methods that activity wrappers call during
    ``observe()``.  Not a full InformationService implementation.
    """

    def __init__(
        self,
        default_rate: Decimal = Decimal("0.10"),
        default_probs: dict[str, Decimal] | None = None,
        sys_liquidity: int = 1000,
    ):
        self._default_rate = default_rate
        self._default_probs = default_probs or {}
        self._sys_liquidity = sys_liquidity

    def get_system_default_rate(self, day: int) -> Decimal:
        return self._default_rate

    def get_default_probability(self, agent_id: str, day: int) -> Decimal | None:
        return self._default_probs.get(agent_id)

    def get_system_liquidity(self, day: int) -> int:
        return self._sys_liquidity


# ═══════════════════════════════════════════════════════════════════════════
# Group 1: Protocol Conformance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Each wrapper must satisfy the runtime-checkable ActivityProfile protocol."""

    def test_trading_activity_satisfies_protocol(self):
        assert isinstance(TradingActivity(), ActivityProfile)

    def test_market_making_activity_satisfies_protocol(self):
        assert isinstance(MarketMakingActivity(), ActivityProfile)

    def test_outside_liquidity_activity_satisfies_protocol(self):
        assert isinstance(OutsideLiquidityActivity(), ActivityProfile)

    def test_lending_activity_satisfies_protocol(self):
        assert isinstance(LendingActivity(), ActivityProfile)

    def test_rating_activity_satisfies_protocol(self):
        assert isinstance(RatingActivity(), ActivityProfile)

    def test_bank_lending_activity_satisfies_protocol(self):
        assert isinstance(BankLendingActivity(), ActivityProfile)

    def test_bank_treasury_activity_satisfies_protocol(self):
        assert isinstance(BankTreasuryActivity(), ActivityProfile)

    def test_cb_activity_satisfies_protocol(self):
        assert isinstance(CBActivity(), ActivityProfile)


# ═══════════════════════════════════════════════════════════════════════════
# Group 2: Property Values
# ═══════════════════════════════════════════════════════════════════════════


class TestPropertyValues:
    """Each wrapper returns the correct activity_type and instrument_class."""

    def test_trading_activity_properties(self):
        a = TradingActivity()
        assert a.activity_type == "trading"
        assert a.instrument_class == "payable"

    def test_market_making_activity_properties(self):
        a = MarketMakingActivity()
        assert a.activity_type == "market_making"
        assert a.instrument_class == "payable"

    def test_outside_liquidity_activity_properties(self):
        a = OutsideLiquidityActivity()
        assert a.activity_type == "outside_liquidity"
        assert a.instrument_class == "payable"

    def test_lending_activity_properties(self):
        a = LendingActivity()
        assert a.activity_type == "lending"
        assert a.instrument_class == "non_bank_loan"

    def test_rating_activity_properties(self):
        a = RatingActivity()
        assert a.activity_type == "rating"
        assert a.instrument_class is None

    def test_bank_lending_activity_properties(self):
        a = BankLendingActivity()
        assert a.activity_type == "bank_lending"
        assert a.instrument_class == "bank_loan"

    def test_bank_treasury_activity_properties(self):
        a = BankTreasuryActivity()
        assert a.activity_type == "treasury"
        assert a.instrument_class == "bank_deposit"

    def test_cb_activity_properties(self):
        a = CBActivity()
        assert a.activity_type == "central_banking"
        assert a.instrument_class == "cb_loan"


# ═══════════════════════════════════════════════════════════════════════════
# Group 3: From-Profile Factories
# ═══════════════════════════════════════════════════════════════════════════


class TestFromProfileFactories:
    """Test the from_*_profile() classmethods that convert existing profiles."""

    def test_trading_from_trader_profile(self):
        from bilancio.decision.profiles import TraderProfile

        tp = TraderProfile(risk_aversion=Decimal("0.5"), planning_horizon=8)
        ta = TradingActivity.from_trader_profile(tp)
        assert isinstance(ta, TradingActivity)
        assert isinstance(ta, ActivityProfile)
        assert ta.risk_aversion == Decimal("0.5")
        assert ta.planning_horizon == 8
        assert ta.activity_type == "trading"
        assert ta.instrument_class == "payable"

    def test_trading_from_trader_profile_preserves_all_fields(self):
        from bilancio.decision.profiles import TraderProfile

        tp = TraderProfile(
            risk_aversion=Decimal("0.7"),
            planning_horizon=15,
            aggressiveness=Decimal("0.8"),
            default_observability=Decimal("0.6"),
            buy_reserve_fraction=Decimal("0.5"),
            trading_motive="liquidity_then_earning",
        )
        ta = TradingActivity.from_trader_profile(tp)
        assert ta.risk_aversion == Decimal("0.7")
        assert ta.planning_horizon == 15
        assert ta.aggressiveness == Decimal("0.8")
        assert ta.default_observability == Decimal("0.6")
        assert ta.buy_reserve_fraction == Decimal("0.5")
        assert ta.trading_motive == "liquidity_then_earning"

    def test_outside_liquidity_from_vbt_profile(self):
        from bilancio.decision.profiles import VBTProfile

        vp = VBTProfile(mid_sensitivity=Decimal("0.8"))
        ola = OutsideLiquidityActivity.from_vbt_profile(vp)
        assert isinstance(ola, OutsideLiquidityActivity)
        assert isinstance(ola, ActivityProfile)
        assert ola.mid_sensitivity == Decimal("0.8")
        assert ola.activity_type == "outside_liquidity"

    def test_outside_liquidity_preserves_all_vbt_fields(self):
        from bilancio.decision.profiles import VBTProfile

        vp = VBTProfile(
            mid_sensitivity=Decimal("0.5"),
            spread_sensitivity=Decimal("0.3"),
            spread_scale=Decimal("1.5"),
        )
        ola = OutsideLiquidityActivity.from_vbt_profile(vp)
        assert ola.mid_sensitivity == Decimal("0.5")
        assert ola.spread_sensitivity == Decimal("0.3")
        assert ola.spread_scale == Decimal("1.5")

    def test_lending_from_lender_profile(self):
        from bilancio.decision.profiles import LenderProfile

        lp = LenderProfile(kappa=Decimal("2"), profit_target=Decimal("0.08"))
        la = LendingActivity.from_lender_profile(lp)
        assert isinstance(la, LendingActivity)
        assert isinstance(la, ActivityProfile)
        assert la.kappa == Decimal("2")
        assert la.profit_target == Decimal("0.08")
        assert la.activity_type == "lending"

    def test_rating_from_rating_profile(self):
        from bilancio.decision.profiles import RatingProfile

        rp = RatingProfile(lookback_window=10, conservatism_bias=Decimal("0.05"))
        ra = RatingActivity.from_rating_profile(rp)
        assert isinstance(ra, RatingActivity)
        assert isinstance(ra, ActivityProfile)
        assert ra.lookback_window == 10
        assert ra.conservatism_bias == Decimal("0.05")
        assert ra.activity_type == "rating"

    def test_bank_lending_from_bank_profile(self):
        from bilancio.decision.profiles import BankProfile

        bp = BankProfile(credit_risk_loading=Decimal("0.03"))
        bla = BankLendingActivity.from_bank_profile(bp)
        assert isinstance(bla, BankLendingActivity)
        assert isinstance(bla, ActivityProfile)
        assert bla.credit_risk_loading == Decimal("0.03")
        assert bla.activity_type == "bank_lending"

    def test_bank_treasury_from_bank_profile(self):
        from bilancio.decision.profiles import BankProfile

        bp = BankProfile(
            reserve_target_ratio=Decimal("0.15"),
            alpha=Decimal("0.01"),
        )
        bta = BankTreasuryActivity.from_bank_profile(bp)
        assert isinstance(bta, BankTreasuryActivity)
        assert isinstance(bta, ActivityProfile)
        assert bta.reserve_target_ratio == Decimal("0.15")
        assert bta.alpha == Decimal("0.01")
        assert bta.activity_type == "treasury"


# ═══════════════════════════════════════════════════════════════════════════
# Group 4: Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTradingPipeline:
    """Test the full observe -> value -> assess -> choose pipeline for TradingActivity."""

    def test_observe_returns_observed_state(self):
        ta = TradingActivity(risk_aversion=Decimal("0.5"), planning_horizon=10)
        info = MockInfoService(
            default_rate=Decimal("0.10"),
            default_probs={"firm_1": Decimal("0.20")},
        )
        position = _position(
            cash=100,
            obligations=(_entry(5, 50, counterparty_id="firm_1"),),
            entitlements=(_entry(5, 80, counterparty_id="firm_2"),),
            horizon=10,
            day=0,
        )
        observed = ta.observe(info, position)
        assert isinstance(observed, ObservedState)
        assert observed.position is position

    def test_value_returns_valuations(self):
        ta = TradingActivity()
        position = _position(cash=100)
        observed = ObservedState(
            position=position,
            system_default_rate=Decimal("0.10"),
        )
        valuations = ta.value(observed)
        assert isinstance(valuations, Valuations)

    def test_assess_returns_risk_view(self):
        ta = TradingActivity()
        position = _position(
            cash=100,
            obligations=(_entry(5, 50),),
        )
        valuations = Valuations(method="ev_hold")
        risk_view = ta.assess(valuations, position)
        assert isinstance(risk_view, RiskView)
        assert risk_view.position is position
        assert risk_view.valuations is valuations

    def test_choose_returns_action_or_none(self):
        ta = TradingActivity()
        position = _position(cash=100)
        valuations = Valuations()
        risk_view = RiskView(position=position, valuations=valuations)
        action_set = ActionSet(
            available=[
                ActionTemplate(action_type=ACTION_SELL),
                ActionTemplate(action_type=ACTION_BUY),
            ],
            phase="B_Dealer",
        )
        result = ta.choose(risk_view, action_set)
        # Result must be an Action or None
        assert result is None or isinstance(result, Action)

    def test_full_pipeline(self):
        """Run the complete four-step pipeline end-to-end."""
        ta = TradingActivity(risk_aversion=Decimal("0.3"), planning_horizon=10)
        info = MockInfoService(default_rate=Decimal("0.15"))
        position = _position(
            cash=100,
            obligations=(_entry(5, 50),),
            entitlements=(_entry(5, 80),),
            horizon=10,
            day=0,
        )
        action_set = ActionSet(
            available=[ActionTemplate(action_type=ACTION_SELL)],
            phase="B_Dealer",
        )

        observed = ta.observe(info, position)
        valuations = ta.value(observed)
        risk_view = ta.assess(valuations, position)
        result = ta.choose(risk_view, action_set)

        # Pipeline should complete without errors
        assert isinstance(observed, ObservedState)
        assert isinstance(valuations, Valuations)
        assert isinstance(risk_view, RiskView)
        assert result is None or isinstance(result, Action)


class TestOutsideLiquidityPipeline:
    """Test the pipeline for OutsideLiquidityActivity (VBT wrapper)."""

    def test_observe_captures_system_rate(self):
        ola = OutsideLiquidityActivity(mid_sensitivity=Decimal("1.0"))
        info = MockInfoService(default_rate=Decimal("0.12"))
        position = _position(cash=500)

        observed = ola.observe(info, position)
        assert isinstance(observed, ObservedState)
        assert observed.position is position

    def test_full_pipeline(self):
        ola = OutsideLiquidityActivity()
        info = MockInfoService(default_rate=Decimal("0.10"))
        position = _position(cash=500)
        action_set = ActionSet(
            available=[ActionTemplate(action_type=ACTION_SET_ANCHORS)],
            phase="B_Dealer",
        )

        observed = ola.observe(info, position)
        valuations = ola.value(observed)
        risk_view = ola.assess(valuations, position)
        result = ola.choose(risk_view, action_set)

        assert isinstance(observed, ObservedState)
        assert isinstance(valuations, Valuations)
        assert isinstance(risk_view, RiskView)
        assert result is None or isinstance(result, Action)


class TestLendingPipeline:
    """Test the pipeline for LendingActivity (NBFI lender wrapper)."""

    def test_assess_computes_liquidity_ratio(self):
        la = LendingActivity(kappa=Decimal("1.0"))
        position = _position(
            cash=200,
            obligations=(_entry(5, 100),),
        )
        valuations = Valuations()
        risk_view = la.assess(valuations, position)
        assert isinstance(risk_view, RiskView)
        # Liquidity ratio should be computed from position
        assert risk_view.liquidity_ratio > Decimal("0")

    def test_full_pipeline(self):
        la = LendingActivity(kappa=Decimal("2.0"), profit_target=Decimal("0.06"))
        info = MockInfoService(
            default_rate=Decimal("0.08"),
            default_probs={"firm_1": Decimal("0.15")},
        )
        position = _position(cash=500)
        action_set = ActionSet(
            available=[
                ActionTemplate(
                    action_type=ACTION_EXTEND_LOAN,
                    constraints={"eligible_borrowers": ["firm_1"]},
                ),
            ],
            phase="B_Lending",
        )

        observed = la.observe(info, position)
        valuations = la.value(observed)
        risk_view = la.assess(valuations, position)
        result = la.choose(risk_view, action_set)

        assert isinstance(observed, ObservedState)
        assert isinstance(valuations, Valuations)
        assert isinstance(risk_view, RiskView)
        assert result is None or isinstance(result, Action)


class TestCBPipeline:
    """Test the pipeline for CBActivity."""

    def test_full_pipeline(self):
        cb = CBActivity()
        info = MockInfoService(default_rate=Decimal("0.05"), sys_liquidity=5000)
        position = _position(cash=10000)
        action_set = ActionSet(
            available=[
                ActionTemplate(action_type=ACTION_SET_CORRIDOR),
            ],
            phase="D_CB",
        )

        observed = cb.observe(info, position)
        valuations = cb.value(observed)
        risk_view = cb.assess(valuations, position)
        result = cb.choose(risk_view, action_set)

        assert isinstance(observed, ObservedState)
        assert isinstance(valuations, Valuations)
        assert isinstance(risk_view, RiskView)
        assert result is None or isinstance(result, Action)


# ═══════════════════════════════════════════════════════════════════════════
# Group 5: ComposedProfile Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestComposedProfileIntegration:
    """Test that wrapper classes compose correctly with ComposedProfile."""

    def test_composed_profile_trading_in_dealer_phase(self):
        ta = TradingActivity()
        composed = ComposedProfile(activities=(ta,), agent_id="firm_1")
        profiles = composed.for_phase("B_Dealer")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "trading"

    def test_composed_profile_bank_dual_activity(self):
        """A bank with both lending and treasury activities dispatches correctly."""
        bla = BankLendingActivity()
        bta = BankTreasuryActivity()
        composed = ComposedProfile(activities=(bla, bta), agent_id="bank_1")

        lending = composed.for_phase("B_Lending")
        assert len(lending) == 1
        assert lending[0].activity_type == "bank_lending"

        cb = composed.for_phase("D_CB")
        assert len(cb) == 1
        assert cb[0].activity_type == "treasury"

    def test_composed_profile_full_dealer_phase(self):
        """All three dealer-phase activities dispatch together."""
        ta = TradingActivity()
        mma = MarketMakingActivity()
        ola = OutsideLiquidityActivity()
        composed = ComposedProfile(activities=(ta, mma, ola), agent_id="agent_1")
        profiles = composed.for_phase("B_Dealer")
        assert len(profiles) == 3
        types = {p.activity_type for p in profiles}
        assert types == {"trading", "market_making", "outside_liquidity"}

    def test_composed_profile_no_match_for_irrelevant_phase(self):
        """Activities that don't match a phase are not returned."""
        ta = TradingActivity()
        composed = ComposedProfile(activities=(ta,), agent_id="firm_1")
        assert len(composed.for_phase("D_CB")) == 0

    def test_composed_profile_rating_in_rating_phase(self):
        ra = RatingActivity()
        composed = ComposedProfile(activities=(ra,), agent_id="rater_1")
        profiles = composed.for_phase("B_Rating")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "rating"

    def test_composed_profile_lending_in_lending_phase(self):
        la = LendingActivity()
        composed = ComposedProfile(activities=(la,), agent_id="lender_1")
        profiles = composed.for_phase("B_Lending")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "lending"

    def test_composed_profile_cb_in_cb_phase(self):
        cb = CBActivity()
        composed = ComposedProfile(activities=(cb,), agent_id="cb_1")
        profiles = composed.for_phase("D_CB")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "central_banking"


# ═══════════════════════════════════════════════════════════════════════════
# Group 6: Frozen Immutability
# ═══════════════════════════════════════════════════════════════════════════


class TestFrozenImmutability:
    """Wrapper dataclasses must be frozen (immutable)."""

    def test_trading_activity_is_frozen(self):
        ta = TradingActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ta.risk_aversion = Decimal("0.99")  # type: ignore[misc]

    def test_lending_activity_is_frozen(self):
        la = LendingActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            la.kappa = Decimal("5")  # type: ignore[misc]

    def test_market_making_activity_is_frozen(self):
        mma = MarketMakingActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            mma.dealer_share_per_bucket = Decimal("0.99")  # type: ignore[misc]

    def test_outside_liquidity_activity_is_frozen(self):
        ola = OutsideLiquidityActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ola.mid_sensitivity = Decimal("0.99")  # type: ignore[misc]

    def test_rating_activity_is_frozen(self):
        ra = RatingActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ra.lookback_window = 99  # type: ignore[misc]

    def test_bank_lending_activity_is_frozen(self):
        bla = BankLendingActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            bla.credit_risk_loading = Decimal("0.99")  # type: ignore[misc]

    def test_bank_treasury_activity_is_frozen(self):
        bta = BankTreasuryActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            bta.reserve_target_ratio = Decimal("0.99")  # type: ignore[misc]

    def test_cb_activity_is_frozen(self):
        cb = CBActivity()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cb.r_base = Decimal("0.99")  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Group 7: LendingActivity Derived Properties
# ═══════════════════════════════════════════════════════════════════════════


class TestLendingDerivedProperties:
    """LendingActivity should expose derived properties from LenderProfile."""

    def test_base_default_estimate(self):
        la = LendingActivity(kappa=Decimal("1"))
        # p = 1 / (1 + kappa) = 1/2 = 0.5
        assert la.base_default_estimate == Decimal("0.5")

    def test_base_default_estimate_high_kappa(self):
        la = LendingActivity(kappa=Decimal("4"))
        # p = 1 / (1 + 4) = 0.2
        assert la.base_default_estimate == Decimal("0.2")

    def test_risk_premium_scale(self):
        la = LendingActivity(risk_aversion=Decimal("0.5"))
        # scale = 0.1 + 0.4 * 0.5 = 0.3
        expected = Decimal("0.1") + Decimal("0.4") * Decimal("0.5")
        assert la.risk_premium_scale == expected

    def test_risk_premium_scale_zero_aversion(self):
        la = LendingActivity(risk_aversion=Decimal("0"))
        # scale = 0.1 + 0.4 * 0 = 0.1
        assert la.risk_premium_scale == Decimal("0.1")

    def test_risk_premium_scale_full_aversion(self):
        la = LendingActivity(risk_aversion=Decimal("1"))
        # scale = 0.1 + 0.4 * 1 = 0.5
        assert la.risk_premium_scale == Decimal("0.5")


# ═══════════════════════════════════════════════════════════════════════════
# Group 8: Exports
# ═══════════════════════════════════════════════════════════════════════════


class TestExports:
    """All 8 activity wrappers should be importable from bilancio.decision."""

    def test_all_activities_importable_from_activities_module(self):
        """Direct import from bilancio.decision.activities works."""
        from bilancio.decision.activities import (  # noqa: F811
            BankLendingActivity,
            BankTreasuryActivity,
            CBActivity,
            LendingActivity,
            MarketMakingActivity,
            OutsideLiquidityActivity,
            RatingActivity,
            TradingActivity,
        )

        assert TradingActivity is not None
        assert MarketMakingActivity is not None
        assert OutsideLiquidityActivity is not None
        assert LendingActivity is not None
        assert RatingActivity is not None
        assert BankLendingActivity is not None
        assert BankTreasuryActivity is not None
        assert CBActivity is not None

    def test_all_activities_exported_from_decision_package(self):
        """Import from bilancio.decision should also work (re-exported)."""
        from bilancio.decision import (
            TradingActivity,
            MarketMakingActivity,
            OutsideLiquidityActivity,
            LendingActivity,
            RatingActivity,
            BankLendingActivity,
            BankTreasuryActivity,
            CBActivity,
        )

        assert TradingActivity is not None
        assert MarketMakingActivity is not None
        assert OutsideLiquidityActivity is not None
        assert LendingActivity is not None
        assert RatingActivity is not None
        assert BankLendingActivity is not None
        assert BankTreasuryActivity is not None
        assert CBActivity is not None


# ═══════════════════════════════════════════════════════════════════════════
# Group 9: Default Parameter Values
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultParameterValues:
    """Wrappers should use sensible defaults matching the original profiles."""

    def test_trading_defaults_match_trader_profile(self):
        from bilancio.decision.profiles import TraderProfile

        ta = TradingActivity()
        tp = TraderProfile()
        assert ta.risk_aversion == tp.risk_aversion
        assert ta.planning_horizon == tp.planning_horizon
        assert ta.aggressiveness == tp.aggressiveness
        assert ta.default_observability == tp.default_observability
        assert ta.buy_reserve_fraction == tp.buy_reserve_fraction
        assert ta.trading_motive == tp.trading_motive

    def test_outside_liquidity_defaults_match_vbt_profile(self):
        from bilancio.decision.profiles import VBTProfile

        ola = OutsideLiquidityActivity()
        vp = VBTProfile()
        assert ola.mid_sensitivity == vp.mid_sensitivity
        assert ola.spread_sensitivity == vp.spread_sensitivity
        assert ola.spread_scale == vp.spread_scale

    def test_lending_defaults_match_lender_profile(self):
        from bilancio.decision.profiles import LenderProfile

        la = LendingActivity()
        lp = LenderProfile()
        assert la.kappa == lp.kappa
        assert la.risk_aversion == lp.risk_aversion
        assert la.planning_horizon == lp.planning_horizon
        assert la.profit_target == lp.profit_target
        assert la.max_loan_maturity == lp.max_loan_maturity

    def test_rating_defaults_match_rating_profile(self):
        from bilancio.decision.profiles import RatingProfile

        ra = RatingActivity()
        rp = RatingProfile()
        assert ra.lookback_window == rp.lookback_window
        assert ra.balance_sheet_weight == rp.balance_sheet_weight
        assert ra.history_weight == rp.history_weight
        assert ra.conservatism_bias == rp.conservatism_bias
        assert ra.coverage_fraction == rp.coverage_fraction
        assert ra.no_data_prior == rp.no_data_prior
