"""Tests for Phase 7 of Plan 036: Instrument Binding.

Phase 7 makes instrument classes configurable on Activity wrappers instead of
hardcoded.  The key new types are:

- ``InstrumentBindings`` -- frozen dataclass mapping roles to instrument kinds
- ``KALECKI_BINDINGS``   -- default bindings for the Kalecki Ring
- ``bind_activities()``  -- helper that applies bindings to activity profiles

All 8 Activity wrappers now have ``instrument_class`` as a configurable field
(was a hardcoded property).  ``build_cash_flow_position_from_trader`` accepts
an ``instrument_kind`` parameter.

Test groups:

    Group 1: InstrumentBindings Construction and Defaults
    Group 2: Activity Wrapper instrument_class as Field
    Group 3: Custom Instrument Class Override
    Group 4: bind_activities() Function
    Group 5: build_cash_flow_position_from_trader instrument_kind
    Group 6: Exports
    Group 7: Real-World Binding Scenario
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
    bind_activities,
)
from bilancio.decision.activity import (
    Action,
    ActionSet,
    ActionTemplate,
    AgentDecisionSpec,
    CashFlowEntry,
    CashFlowPosition,
    InstrumentBindings,
    KALECKI_BINDINGS,
    build_cash_flow_position_from_trader,
)


# -- Helpers -----------------------------------------------------------------


def _entry(day: int, amount: int | str, **kw) -> CashFlowEntry:
    """Shorthand for creating a CashFlowEntry."""
    return CashFlowEntry(day=day, amount=Decimal(str(amount)), **kw)


def _make_position(
    cash: int | str = 100,
    obligations: tuple[CashFlowEntry, ...] = (),
    entitlements: tuple[CashFlowEntry, ...] = (),
    horizon: int = 10,
    current_day: int = 0,
    **kw,
) -> CashFlowPosition:
    """Shorthand for creating a CashFlowPosition."""
    return CashFlowPosition(
        cash=Decimal(str(cash)),
        obligations=obligations,
        entitlements=entitlements,
        planning_horizon=horizon,
        current_day=current_day,
        **kw,
    )


class MockTicket:
    """Minimal mock of a ticket/instrument for trader state."""

    def __init__(
        self,
        maturity_day: int = 5,
        face: int | str = 20,
        owner_id: str = "owner",
        issuer_id: str = "issuer",
        id: str = "t1",
    ):
        self.maturity_day = maturity_day
        self.face = Decimal(str(face))
        self.owner_id = owner_id
        self.issuer_id = issuer_id
        self.id = id


class MockTrader:
    """Minimal mock of a TraderState for build_cash_flow_position_from_trader."""

    def __init__(
        self,
        cash: int | str = 100,
        obligations: list | None = None,
        tickets_owned: list | None = None,
    ):
        self.cash = Decimal(str(cash))
        self.obligations = obligations or []
        self.tickets_owned = tickets_owned or []


class MockInfoService:
    """Minimal mock of InformationService for pipeline tests."""

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


# ============================================================================
# Group 1: InstrumentBindings Construction and Defaults
# ============================================================================


class TestInstrumentBindingsConstructionAndDefaults:
    """InstrumentBindings dataclass construction, defaults, immutability."""

    def test_default_bindings_have_kalecki_values(self):
        """Default InstrumentBindings should match Kalecki Ring conventions."""
        bindings = InstrumentBindings()
        assert bindings.tradeable == "payable"
        assert bindings.lendable == "non_bank_loan"
        assert bindings.bank_lendable == "bank_loan"
        assert bindings.depositable == "bank_deposit"
        assert bindings.cb_lendable == "cb_loan"
        assert bindings.bucket_by == "remaining_maturity"

    def test_kalecki_bindings_equals_default(self):
        """KALECKI_BINDINGS singleton equals a default InstrumentBindings()."""
        assert KALECKI_BINDINGS == InstrumentBindings()

    def test_custom_bindings_override_defaults(self):
        """Providing a custom value overrides the default for that field."""
        bindings = InstrumentBindings(tradeable="bond")
        assert bindings.tradeable == "bond"

    def test_instrument_bindings_is_frozen(self):
        """InstrumentBindings should be immutable (frozen dataclass)."""
        bindings = InstrumentBindings()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            bindings.tradeable = "bond"  # type: ignore[misc]

    def test_partial_override_preserves_other_defaults(self):
        """Overriding one field leaves all other fields at their defaults."""
        bindings = InstrumentBindings(tradeable="corporate_bond")
        assert bindings.tradeable == "corporate_bond"
        assert bindings.lendable == "non_bank_loan"
        assert bindings.bank_lendable == "bank_loan"
        assert bindings.depositable == "bank_deposit"
        assert bindings.cb_lendable == "cb_loan"
        assert bindings.bucket_by == "remaining_maturity"

    def test_multiple_overrides(self):
        """Multiple fields can be overridden simultaneously."""
        bindings = InstrumentBindings(
            tradeable="corporate_bond",
            lendable="syndicated_loan",
            bucket_by="credit_rating",
        )
        assert bindings.tradeable == "corporate_bond"
        assert bindings.lendable == "syndicated_loan"
        assert bindings.bucket_by == "credit_rating"
        # Remaining fields at defaults
        assert bindings.bank_lendable == "bank_loan"
        assert bindings.depositable == "bank_deposit"
        assert bindings.cb_lendable == "cb_loan"


# ============================================================================
# Group 2: Activity Wrapper instrument_class as Configurable Field
# ============================================================================


class TestActivityWrapperInstrumentClassField:
    """Each of the 8 wrappers has instrument_class as a configurable field."""

    def test_trading_activity_default_instrument_class(self):
        """TradingActivity defaults to 'payable' (backward compat)."""
        ta = TradingActivity()
        assert ta.instrument_class == "payable"

    def test_trading_activity_custom_instrument_class(self):
        """TradingActivity accepts a custom instrument_class at construction."""
        ta = TradingActivity(instrument_class="bond")
        assert ta.instrument_class == "bond"

    def test_market_making_activity_default_instrument_class(self):
        mma = MarketMakingActivity()
        assert mma.instrument_class == "payable"

    def test_market_making_activity_custom_instrument_class(self):
        mma = MarketMakingActivity(instrument_class="bond")
        assert mma.instrument_class == "bond"

    def test_outside_liquidity_activity_default_instrument_class(self):
        ola = OutsideLiquidityActivity()
        assert ola.instrument_class == "payable"

    def test_outside_liquidity_activity_custom_instrument_class(self):
        ola = OutsideLiquidityActivity(instrument_class="corporate_bond")
        assert ola.instrument_class == "corporate_bond"

    def test_lending_activity_default_instrument_class(self):
        la = LendingActivity()
        assert la.instrument_class == "non_bank_loan"

    def test_lending_activity_custom_instrument_class(self):
        la = LendingActivity(instrument_class="custom_loan")
        assert la.instrument_class == "custom_loan"

    def test_rating_activity_default_instrument_class(self):
        """RatingActivity defaults to None (general, not instrument-specific)."""
        ra = RatingActivity()
        assert ra.instrument_class is None

    def test_rating_activity_custom_instrument_class(self):
        """RatingActivity can optionally be bound to an instrument class."""
        ra = RatingActivity(instrument_class="bond")
        assert ra.instrument_class == "bond"

    def test_bank_lending_activity_default_instrument_class(self):
        bla = BankLendingActivity()
        assert bla.instrument_class == "bank_loan"

    def test_bank_lending_activity_custom_instrument_class(self):
        bla = BankLendingActivity(instrument_class="custom_bank_loan")
        assert bla.instrument_class == "custom_bank_loan"

    def test_bank_treasury_activity_default_instrument_class(self):
        bta = BankTreasuryActivity()
        assert bta.instrument_class == "bank_deposit"

    def test_bank_treasury_activity_custom_instrument_class(self):
        bta = BankTreasuryActivity(instrument_class="savings_deposit")
        assert bta.instrument_class == "savings_deposit"

    def test_cb_activity_default_instrument_class(self):
        cb = CBActivity()
        assert cb.instrument_class == "cb_loan"

    def test_cb_activity_custom_instrument_class(self):
        cb = CBActivity(instrument_class="standing_facility")
        assert cb.instrument_class == "standing_facility"


# ============================================================================
# Group 3: Custom Instrument Class Override (behavioral checks)
# ============================================================================


class TestCustomInstrumentClassOverride:
    """Custom instrument_class is preserved, compared, and immutable."""

    def test_trading_activity_custom_instrument_persists(self):
        ta = TradingActivity(instrument_class="bond", risk_aversion=Decimal("0.5"))
        assert ta.instrument_class == "bond"
        assert ta.risk_aversion == Decimal("0.5")

    def test_market_making_custom_instrument_persists(self):
        mma = MarketMakingActivity(instrument_class="bond")
        assert mma.instrument_class == "bond"

    def test_lending_custom_instrument_persists(self):
        la = LendingActivity(instrument_class="custom_loan", kappa=Decimal("2.0"))
        assert la.instrument_class == "custom_loan"
        assert la.kappa == Decimal("2.0")

    def test_bank_lending_custom_instrument_persists(self):
        bla = BankLendingActivity(
            instrument_class="custom_bank_loan",
            credit_risk_loading=Decimal("0.05"),
        )
        assert bla.instrument_class == "custom_bank_loan"
        assert bla.credit_risk_loading == Decimal("0.05")

    def test_different_instrument_class_means_not_equal(self):
        """Two activities differing only in instrument_class are not equal."""
        ta1 = TradingActivity(instrument_class="payable")
        ta2 = TradingActivity(instrument_class="bond")
        assert ta1 != ta2

    def test_same_instrument_class_and_params_are_equal(self):
        """Two activities with identical fields are equal."""
        ta1 = TradingActivity(instrument_class="bond")
        ta2 = TradingActivity(instrument_class="bond")
        assert ta1 == ta2

    def test_custom_instrument_class_satisfies_protocol(self):
        """Activity with custom instrument_class still satisfies ActivityProfile."""
        from bilancio.decision.activity import ActivityProfile

        ta = TradingActivity(instrument_class="bond")
        assert isinstance(ta, ActivityProfile)

    def test_instrument_class_frozen_on_wrapper(self):
        """Cannot mutate instrument_class after construction."""
        ta = TradingActivity(instrument_class="bond")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ta.instrument_class = "payable"  # type: ignore[misc]


# ============================================================================
# Group 4: bind_activities() Function
# ============================================================================


class TestBindActivities:
    """bind_activities() applies InstrumentBindings to a tuple of activities."""

    def test_default_bindings_preserve_current_values(self):
        """With default bindings, instrument_class values stay the same."""
        activities = (
            TradingActivity(),
            MarketMakingActivity(),
            OutsideLiquidityActivity(),
            LendingActivity(),
            BankLendingActivity(),
            BankTreasuryActivity(),
            CBActivity(),
            RatingActivity(),
        )
        bound = bind_activities(InstrumentBindings(), *activities)
        assert bound[0].instrument_class == "payable"        # TradingActivity
        assert bound[1].instrument_class == "payable"        # MarketMakingActivity
        assert bound[2].instrument_class == "payable"        # OutsideLiquidityActivity
        assert bound[3].instrument_class == "non_bank_loan"  # LendingActivity
        assert bound[4].instrument_class == "bank_loan"      # BankLendingActivity
        assert bound[5].instrument_class == "bank_deposit"   # BankTreasuryActivity
        assert bound[6].instrument_class == "cb_loan"        # CBActivity
        assert bound[7].instrument_class is None             # RatingActivity

    def test_custom_tradeable_changes_trading_market_making_outside(self):
        """Custom 'tradeable' rebinds trading, market_making, and outside_liquidity."""
        activities = (
            TradingActivity(),
            MarketMakingActivity(),
            OutsideLiquidityActivity(),
        )
        bindings = InstrumentBindings(tradeable="corporate_bond")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class == "corporate_bond"
        assert bound[1].instrument_class == "corporate_bond"
        assert bound[2].instrument_class == "corporate_bond"

    def test_custom_lendable_changes_lending(self):
        """Custom 'lendable' rebinds LendingActivity."""
        activities = (LendingActivity(),)
        bindings = InstrumentBindings(lendable="syndicated_loan")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class == "syndicated_loan"

    def test_custom_bank_lendable_changes_bank_lending(self):
        """Custom 'bank_lendable' rebinds BankLendingActivity."""
        activities = (BankLendingActivity(),)
        bindings = InstrumentBindings(bank_lendable="interbank_loan")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class == "interbank_loan"

    def test_custom_depositable_changes_treasury(self):
        """Custom 'depositable' rebinds BankTreasuryActivity."""
        activities = (BankTreasuryActivity(),)
        bindings = InstrumentBindings(depositable="savings_deposit")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class == "savings_deposit"

    def test_custom_cb_lendable_changes_central_banking(self):
        """Custom 'cb_lendable' rebinds CBActivity."""
        activities = (CBActivity(),)
        bindings = InstrumentBindings(cb_lendable="standing_facility")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class == "standing_facility"

    def test_rating_activity_passes_through_unchanged(self):
        """RatingActivity instrument_class stays None (general, no binding)."""
        activities = (RatingActivity(),)
        bindings = InstrumentBindings(tradeable="bond")
        bound = bind_activities(bindings, *activities)
        assert bound[0].instrument_class is None

    def test_bind_activities_returns_tuple(self):
        """bind_activities returns a tuple, not a list."""
        activities = (TradingActivity(),)
        bound = bind_activities(InstrumentBindings(), *activities)
        assert isinstance(bound, tuple)

    def test_bind_activities_preserves_behavioral_parameters(self):
        """bind_activities preserves non-instrument_class fields."""
        ta = TradingActivity(
            risk_aversion=Decimal("0.7"),
            planning_horizon=15,
            aggressiveness=Decimal("0.5"),
        )
        bindings = InstrumentBindings(tradeable="bond")
        bound = bind_activities(bindings, ta)
        result = bound[0]
        assert result.instrument_class == "bond"
        assert result.risk_aversion == Decimal("0.7")
        assert result.planning_horizon == 15
        assert result.aggressiveness == Decimal("0.5")

    def test_bind_activities_preserves_lending_parameters(self):
        """bind_activities preserves LendingActivity behavioral fields."""
        la = LendingActivity(
            kappa=Decimal("2.0"),
            risk_aversion=Decimal("0.5"),
            profit_target=Decimal("0.08"),
        )
        bindings = InstrumentBindings(lendable="syndicated_loan")
        bound = bind_activities(bindings, la)
        result = bound[0]
        assert result.instrument_class == "syndicated_loan"
        assert result.kappa == Decimal("2.0")
        assert result.risk_aversion == Decimal("0.5")
        assert result.profit_target == Decimal("0.08")

    def test_bind_activities_with_empty_activities(self):
        """bind_activities with no activities returns an empty tuple."""
        bound = bind_activities(InstrumentBindings())
        assert bound == ()
        assert isinstance(bound, tuple)


# ============================================================================
# Group 5: build_cash_flow_position_from_trader instrument_kind
# ============================================================================


class TestBuildCashFlowPositionInstrumentKind:
    """build_cash_flow_position_from_trader respects the instrument_kind param."""

    def test_default_instrument_kind_is_payable(self):
        """Without specifying instrument_kind, entries default to 'payable'."""
        trader = MockTrader(
            cash=100,
            obligations=[MockTicket(maturity_day=5, face=20, owner_id="o1", id="t1")],
            tickets_owned=[MockTicket(maturity_day=5, face=30, issuer_id="i1", id="t2")],
        )
        pos = build_cash_flow_position_from_trader(trader, current_day=0)
        assert pos.obligations[0].instrument_kind == "payable"
        assert pos.entitlements[0].instrument_kind == "payable"

    def test_custom_instrument_kind_on_obligations(self):
        """Custom instrument_kind propagates to obligation entries."""
        trader = MockTrader(
            obligations=[MockTicket(maturity_day=5, face=20, owner_id="o1", id="t1")],
        )
        pos = build_cash_flow_position_from_trader(
            trader, current_day=0, instrument_kind="bond"
        )
        assert pos.obligations[0].instrument_kind == "bond"

    def test_custom_instrument_kind_on_entitlements(self):
        """Custom instrument_kind propagates to entitlement entries."""
        trader = MockTrader(
            tickets_owned=[MockTicket(maturity_day=3, face=50, issuer_id="i1", id="t2")],
        )
        pos = build_cash_flow_position_from_trader(
            trader, current_day=0, instrument_kind="corporate_bond"
        )
        assert pos.entitlements[0].instrument_kind == "corporate_bond"

    def test_custom_instrument_kind_both_obligations_and_entitlements(self):
        """Both obligations and entitlements use the same custom instrument_kind."""
        trader = MockTrader(
            cash=200,
            obligations=[MockTicket(maturity_day=5, face=20, owner_id="o1", id="t1")],
            tickets_owned=[MockTicket(maturity_day=3, face=30, issuer_id="i1", id="t2")],
        )
        pos = build_cash_flow_position_from_trader(
            trader, current_day=0, instrument_kind="note"
        )
        assert pos.obligations[0].instrument_kind == "note"
        assert pos.entitlements[0].instrument_kind == "note"


# ============================================================================
# Group 6: Exports
# ============================================================================


class TestExports:
    """InstrumentBindings, KALECKI_BINDINGS, and bind_activities are importable."""

    def test_instrument_bindings_importable_from_decision(self):
        from bilancio.decision import InstrumentBindings as IB

        assert IB is not None

    def test_kalecki_bindings_importable_from_decision(self):
        from bilancio.decision import KALECKI_BINDINGS as KB

        assert KB is not None

    def test_bind_activities_importable_from_decision(self):
        from bilancio.decision import bind_activities as ba

        assert ba is not None

    def test_instrument_bindings_importable_from_activity_module(self):
        from bilancio.decision.activity import InstrumentBindings as IB

        assert IB is not None

    def test_kalecki_bindings_importable_from_activity_module(self):
        from bilancio.decision.activity import KALECKI_BINDINGS as KB

        assert KB is not None

    def test_instrument_bindings_same_class_both_paths(self):
        """Both import paths resolve to the same class."""
        from bilancio.decision import InstrumentBindings as IB1
        from bilancio.decision.activity import InstrumentBindings as IB2

        assert IB1 is IB2

    def test_bind_activities_importable_from_activities_module(self):
        from bilancio.decision.activities import bind_activities as ba

        assert ba is not None


# ============================================================================
# Group 7: Real-World Binding Scenario
# ============================================================================


class TestRealWorldBindingScenario:
    """End-to-end scenarios with custom bindings applied to activity suites."""

    def test_bond_market_binding(self):
        """Create a full 'bond market' binding and verify all activities."""
        bond_bindings = InstrumentBindings(
            tradeable="corporate_bond",
            lendable="syndicated_loan",
            bank_lendable="interbank_loan",
            depositable="savings_deposit",
            cb_lendable="standing_facility",
            bucket_by="credit_rating",
        )
        activities = (
            TradingActivity(risk_aversion=Decimal("0.3")),
            MarketMakingActivity(),
            OutsideLiquidityActivity(mid_sensitivity=Decimal("0.8")),
            LendingActivity(kappa=Decimal("1.5")),
            RatingActivity(),
            BankLendingActivity(credit_risk_loading=Decimal("0.04")),
            BankTreasuryActivity(reserve_target_ratio=Decimal("0.12")),
            CBActivity(),
        )
        bound = bind_activities(bond_bindings, *activities)

        # Trading group: corporate_bond
        assert bound[0].instrument_class == "corporate_bond"  # TradingActivity
        assert bound[1].instrument_class == "corporate_bond"  # MarketMakingActivity
        assert bound[2].instrument_class == "corporate_bond"  # OutsideLiquidityActivity

        # Lending: syndicated_loan
        assert bound[3].instrument_class == "syndicated_loan"  # LendingActivity

        # Rating: None (not instrument-specific)
        assert bound[4].instrument_class is None  # RatingActivity

        # Banking: interbank_loan and savings_deposit
        assert bound[5].instrument_class == "interbank_loan"  # BankLendingActivity
        assert bound[6].instrument_class == "savings_deposit"  # BankTreasuryActivity

        # CB: standing_facility
        assert bound[7].instrument_class == "standing_facility"  # CBActivity

        # Behavioral parameters preserved
        assert bound[0].risk_aversion == Decimal("0.3")
        assert bound[2].mid_sensitivity == Decimal("0.8")
        assert bound[3].kappa == Decimal("1.5")
        assert bound[5].credit_risk_loading == Decimal("0.04")
        assert bound[6].reserve_target_ratio == Decimal("0.12")

    def test_agent_decision_spec_with_bound_activities(self):
        """AgentDecisionSpec with bound activities: for_phase and run_phase work."""
        bindings = InstrumentBindings(tradeable="bond")
        activities = (
            TradingActivity(risk_aversion=Decimal("0.2")),
            MarketMakingActivity(),
        )
        bound = bind_activities(bindings, *activities)

        spec = AgentDecisionSpec(agent_id="dealer_1", activities=bound)

        # for_phase dispatches correctly
        dealer_profiles = spec.for_phase("B_Dealer")
        assert len(dealer_profiles) == 2
        assert dealer_profiles[0].instrument_class == "bond"
        assert dealer_profiles[1].instrument_class == "bond"

        # run_phase produces actions
        info = MockInfoService()
        position = _make_position(cash=1000)
        action_set = ActionSet(
            available=[
                ActionTemplate(action_type="sell"),
                ActionTemplate(action_type="buy"),
                ActionTemplate(
                    action_type="set_quotes",
                    constraints={"bid": Decimal("0.85"), "ask": Decimal("0.95")},
                ),
            ],
            phase="B_Dealer",
        )
        actions = spec.run_phase("B_Dealer", info, position, action_set)
        assert isinstance(actions, list)
        # MarketMakingActivity always produces set_quotes when template present
        action_types = {a.action_type for a in actions}
        assert "set_quotes" in action_types

    def test_bank_with_custom_bindings_through_phases(self):
        """Bank agent with custom bindings still works across lending + CB phases."""
        bindings = InstrumentBindings(
            bank_lendable="interbank_loan",
            depositable="term_deposit",
            cb_lendable="emergency_facility",
        )
        activities = (
            BankLendingActivity(credit_risk_loading=Decimal("0.03")),
            BankTreasuryActivity(reserve_target_ratio=Decimal("0.15")),
        )
        bound = bind_activities(bindings, *activities)

        spec = AgentDecisionSpec(agent_id="bank_1", activities=bound)

        # Verify instrument classes
        lending = spec.get_activity("bank_lending")
        treasury = spec.get_activity("treasury")
        assert lending is not None
        assert treasury is not None
        assert lending.instrument_class == "interbank_loan"
        assert treasury.instrument_class == "term_deposit"

        # run_phase: lending
        info = MockInfoService()
        position = _make_position(cash=500)
        lending_actions = spec.run_phase(
            "B_Lending",
            info,
            position,
            ActionSet(
                available=[ActionTemplate(action_type="extend_loan")],
                phase="B_Lending",
            ),
        )
        assert len(lending_actions) == 1
        assert lending_actions[0].action_type == "extend_loan"
        assert lending_actions[0].params["credit_risk_loading"] == Decimal("0.03")

        # run_phase: treasury (in D_CB phase)
        treasury_actions = spec.run_phase(
            "D_CB",
            info,
            position,
            ActionSet(
                available=[ActionTemplate(action_type="set_corridor")],
                phase="D_CB",
            ),
        )
        assert len(treasury_actions) == 1
        assert treasury_actions[0].action_type == "set_corridor"
