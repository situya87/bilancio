"""Tests for Phase 6 of Plan 036: AgentDecisionSpec — Composition.

Tests cover the ``AgentDecisionSpec`` dataclass which enables agents with
multiple activity profiles.  ``AgentDecisionSpec`` is the primary unit for
describing how an agent makes decisions, providing:

    - ``for_phase()``    — dispatch matching activity profiles for a phase
    - ``run_phase()``    — execute the full four-step pipeline for matching profiles
    - ``activity_types`` — list all activity type strings
    - ``has_activity()`` / ``get_activity()`` — lookup helpers

Test groups:

    Group 1: Basic Construction and Properties
    Group 2: for_phase() Dispatch
    Group 3: has_activity() and get_activity()
    Group 4: run_phase() Pipeline Execution
    Group 5: Export from decision package
    Group 6: Real-world scenario (bank with lending + treasury)
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
    Action,
    ActionSet,
    ActionTemplate,
    AgentDecisionSpec,
    CashFlowEntry,
    CashFlowPosition,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


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
# Group 1: Basic Construction and Properties
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicConstructionAndProperties:
    """AgentDecisionSpec construction, defaults, and basic properties."""

    def test_empty_spec(self):
        spec = AgentDecisionSpec(agent_id="agent_1")
        assert spec.agent_id == "agent_1"
        assert spec.activities == ()
        assert spec.information_profile_name is None
        assert spec.activity_types == ()

    def test_spec_with_single_activity(self):
        ta = TradingActivity()
        spec = AgentDecisionSpec(agent_id="firm_1", activities=(ta,))
        assert spec.activity_types == ("trading",)

    def test_spec_with_multiple_activities(self):
        bla = BankLendingActivity()
        bta = BankTreasuryActivity()
        spec = AgentDecisionSpec(agent_id="bank_1", activities=(bla, bta))
        assert spec.activity_types == ("bank_lending", "treasury")

    def test_spec_with_info_profile_name(self):
        spec = AgentDecisionSpec(
            agent_id="trader_1",
            activities=(TradingActivity(),),
            information_profile_name="TRADER_BASIC",
        )
        assert spec.information_profile_name == "TRADER_BASIC"

    def test_spec_is_frozen(self):
        spec = AgentDecisionSpec(agent_id="x")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            spec.agent_id = "y"  # type: ignore[misc]

    def test_spec_preserves_activity_order(self):
        """Activities tuple retains insertion order (priority ordering)."""
        ta = TradingActivity()
        mma = MarketMakingActivity()
        ola = OutsideLiquidityActivity()
        spec = AgentDecisionSpec(
            agent_id="dealer_1", activities=(ta, mma, ola)
        )
        assert spec.activity_types == ("trading", "market_making", "outside_liquidity")


# ═══════════════════════════════════════════════════════════════════════════
# Group 2: for_phase() Dispatch
# ═══════════════════════════════════════════════════════════════════════════


class TestForPhaseDispatch:
    """for_phase() returns the correct activity profiles for each phase."""

    def test_for_phase_trading_in_dealer(self):
        spec = AgentDecisionSpec(agent_id="firm_1", activities=(TradingActivity(),))
        profiles = spec.for_phase("B_Dealer")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "trading"

    def test_for_phase_bank_lending(self):
        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(BankLendingActivity(), BankTreasuryActivity()),
        )
        lending = spec.for_phase("B_Lending")
        assert len(lending) == 1
        assert lending[0].activity_type == "bank_lending"

    def test_for_phase_bank_treasury_in_cb(self):
        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(BankLendingActivity(), BankTreasuryActivity()),
        )
        treasury = spec.for_phase("D_CB")
        assert len(treasury) == 1
        assert treasury[0].activity_type == "treasury"

    def test_for_phase_full_dealer_phase(self):
        spec = AgentDecisionSpec(
            agent_id="dealer_1",
            activities=(
                TradingActivity(),
                MarketMakingActivity(),
                OutsideLiquidityActivity(),
            ),
        )
        profiles = spec.for_phase("B_Dealer")
        assert len(profiles) == 3
        types = {p.activity_type for p in profiles}
        assert types == {"trading", "market_making", "outside_liquidity"}

    def test_for_phase_no_match(self):
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        assert spec.for_phase("B_Rating") == []

    def test_for_phase_unknown_phase(self):
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        assert spec.for_phase("X_Unknown") == []

    def test_for_phase_rating_in_rating(self):
        spec = AgentDecisionSpec(
            agent_id="rater_1", activities=(RatingActivity(),)
        )
        profiles = spec.for_phase("B_Rating")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "rating"

    def test_for_phase_lending_in_lending(self):
        """NBFI lending activity dispatches in B_Lending phase."""
        spec = AgentDecisionSpec(
            agent_id="lender_1", activities=(LendingActivity(),)
        )
        profiles = spec.for_phase("B_Lending")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "lending"

    def test_for_phase_cb_in_cb(self):
        spec = AgentDecisionSpec(
            agent_id="cb_1", activities=(CBActivity(),)
        )
        profiles = spec.for_phase("D_CB")
        assert len(profiles) == 1
        assert profiles[0].activity_type == "central_banking"


# ═══════════════════════════════════════════════════════════════════════════
# Group 3: has_activity() and get_activity()
# ═══════════════════════════════════════════════════════════════════════════


class TestHasAndGetActivity:
    """Lookup helpers for querying activity types in a spec."""

    def test_has_activity_true(self):
        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(BankLendingActivity(), BankTreasuryActivity()),
        )
        assert spec.has_activity("bank_lending") is True
        assert spec.has_activity("treasury") is True

    def test_has_activity_false(self):
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        assert spec.has_activity("lending") is False

    def test_has_activity_empty_spec(self):
        spec = AgentDecisionSpec(agent_id="nobody")
        assert spec.has_activity("trading") is False

    def test_get_activity_found(self):
        bla = BankLendingActivity(credit_risk_loading=Decimal("0.05"))
        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(bla, BankTreasuryActivity()),
        )
        result = spec.get_activity("bank_lending")
        assert result is bla
        assert result.credit_risk_loading == Decimal("0.05")

    def test_get_activity_not_found(self):
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        assert spec.get_activity("lending") is None

    def test_get_activity_returns_first_match(self):
        """If duplicate activity types existed, get_activity returns the first."""
        ta1 = TradingActivity(risk_aversion=Decimal("0.1"))
        ta2 = TradingActivity(risk_aversion=Decimal("0.9"))
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(ta1, ta2)
        )
        result = spec.get_activity("trading")
        assert result is ta1
        assert result.risk_aversion == Decimal("0.1")


# ═══════════════════════════════════════════════════════════════════════════
# Group 4: run_phase() Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPhasePipeline:
    """run_phase() executes the four-step pipeline for matching profiles."""

    def test_run_phase_single_activity(self):
        """Single trading activity in dealer phase produces actions."""
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        info = MockInfoService(default_probs={"firm_2": Decimal("0.20")})
        position = _make_position(
            cash=100,
            obligations=(_entry(5, 50, counterparty_id="firm_1"),),
            entitlements=(_entry(5, 80, counterparty_id="firm_2"),),
        )
        action_set = ActionSet(
            available=[
                ActionTemplate(action_type="sell"),
                ActionTemplate(action_type="buy"),
            ],
            phase="B_Dealer",
        )
        actions = spec.run_phase("B_Dealer", info, position, action_set)
        assert isinstance(actions, list)
        for a in actions:
            assert isinstance(a, Action)

    def test_run_phase_no_matching_profiles(self):
        """Running a phase with no matching profiles returns empty list."""
        spec = AgentDecisionSpec(
            agent_id="firm_1", activities=(TradingActivity(),)
        )
        info = MockInfoService()
        position = _make_position()
        action_set = ActionSet(available=[], phase="B_Rating")
        actions = spec.run_phase("B_Rating", info, position, action_set)
        assert actions == []

    def test_run_phase_bank_dual_activity_lending(self):
        """Bank with lending + treasury: lending phase activates only lending."""
        bla = BankLendingActivity()
        bta = BankTreasuryActivity()
        spec = AgentDecisionSpec(
            agent_id="bank_1", activities=(bla, bta)
        )
        info = MockInfoService()
        position = _make_position()

        # Lending phase -- BankLendingActivity.choose() returns extend_loan
        # when the action_set contains an "extend_loan" template.
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

    def test_run_phase_bank_dual_activity_treasury(self):
        """Bank with lending + treasury: CB phase activates only treasury."""
        bla = BankLendingActivity()
        bta = BankTreasuryActivity()
        spec = AgentDecisionSpec(
            agent_id="bank_1", activities=(bla, bta)
        )
        info = MockInfoService()
        position = _make_position()

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

    def test_run_phase_shared_position(self):
        """All profiles in a phase receive the same position object.

        Two activities that both activate in B_Dealer (trading + market_making)
        should both run and potentially return actions.
        """
        ta = TradingActivity()
        mma = MarketMakingActivity()
        spec = AgentDecisionSpec(
            agent_id="agent_1", activities=(ta, mma)
        )
        info = MockInfoService()
        position = _make_position()
        action_set = ActionSet(
            available=[
                ActionTemplate(action_type="sell"),
                ActionTemplate(
                    action_type="set_quotes",
                    constraints={"bid": Decimal("0.8"), "ask": Decimal("0.9")},
                ),
            ],
            phase="B_Dealer",
        )
        actions = spec.run_phase("B_Dealer", info, position, action_set)
        assert isinstance(actions, list)
        # MarketMakingActivity always returns set_quotes when the template
        # is present, so we expect at least that one.
        action_types = [a.action_type for a in actions]
        assert "set_quotes" in action_types

    def test_run_phase_profiles_that_pass_not_included(self):
        """Profiles that return None (hold/pass) don't appear in results."""
        spec = AgentDecisionSpec(
            agent_id="firm_1",
            activities=(TradingActivity(),),
        )
        info = MockInfoService()
        # Lots of cash, no stress --> TradingActivity needs a "buy" template
        # and surplus > threshold to act; with an empty action_set it passes.
        position = _make_position(cash=1000)
        action_set = ActionSet(available=[], phase="B_Dealer")
        actions = spec.run_phase("B_Dealer", info, position, action_set)
        assert actions == []

    def test_run_phase_empty_spec(self):
        """Running a phase on an empty spec returns empty list."""
        spec = AgentDecisionSpec(agent_id="nobody")
        info = MockInfoService()
        position = _make_position()
        action_set = ActionSet(available=[], phase="B_Dealer")
        actions = spec.run_phase("B_Dealer", info, position, action_set)
        assert actions == []

    def test_run_phase_returns_list_of_actions(self):
        """Return type is always a list, even for a single action."""
        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(BankLendingActivity(),),
        )
        info = MockInfoService()
        position = _make_position()
        action_set = ActionSet(
            available=[ActionTemplate(action_type="extend_loan")],
            phase="B_Lending",
        )
        result = spec.run_phase("B_Lending", info, position, action_set)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Action)


# ═══════════════════════════════════════════════════════════════════════════
# Group 5: Export from decision package
# ═══════════════════════════════════════════════════════════════════════════


class TestExports:
    """AgentDecisionSpec should be importable from both activity and decision."""

    def test_agent_decision_spec_exported_from_decision_package(self):
        from bilancio.decision import AgentDecisionSpec as ADS

        assert ADS is not None

    def test_agent_decision_spec_importable_from_activity_module(self):
        from bilancio.decision.activity import AgentDecisionSpec as ADS

        assert ADS is not None

    def test_agent_decision_spec_is_same_class(self):
        """Both import paths resolve to the same class."""
        from bilancio.decision import AgentDecisionSpec as ADS1
        from bilancio.decision.activity import AgentDecisionSpec as ADS2

        assert ADS1 is ADS2


# ═══════════════════════════════════════════════════════════════════════════
# Group 6: Real-world scenario (bank with lending + treasury)
# ═══════════════════════════════════════════════════════════════════════════


class TestBankScenario:
    """End-to-end scenario: bank agent going through lending then treasury phases."""

    def test_bank_scenario_lending_then_treasury(self):
        """Simulate a bank agent going through lending then treasury phases."""
        bank_spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(
                BankLendingActivity(credit_risk_loading=Decimal("0.03")),
                BankTreasuryActivity(reserve_target_ratio=Decimal("0.15")),
            ),
        )
        info = MockInfoService()
        position = _make_position(cash=500)

        # Phase 1: B_Lending
        lending_set = ActionSet(
            available=[ActionTemplate(action_type="extend_loan")],
            phase="B_Lending",
        )
        lending_actions = bank_spec.run_phase("B_Lending", info, position, lending_set)
        assert len(lending_actions) == 1
        assert lending_actions[0].action_type == "extend_loan"
        assert lending_actions[0].params["credit_risk_loading"] == Decimal("0.03")

        # Phase 2: D_CB
        treasury_set = ActionSet(
            available=[ActionTemplate(action_type="set_corridor")],
            phase="D_CB",
        )
        treasury_actions = bank_spec.run_phase(
            "D_CB", info, position, treasury_set
        )
        assert len(treasury_actions) == 1
        assert treasury_actions[0].action_type == "set_corridor"
        assert treasury_actions[0].params["reserve_target_ratio"] == Decimal("0.15")

        # Verify no cross-contamination
        assert bank_spec.for_phase("B_Dealer") == []

    def test_dealer_scenario_all_three_dealer_activities(self):
        """Dealer with trading + market_making + outside_liquidity in B_Dealer."""
        dealer_spec = AgentDecisionSpec(
            agent_id="dealer_1",
            activities=(
                TradingActivity(risk_aversion=Decimal("0.2")),
                MarketMakingActivity(),
                OutsideLiquidityActivity(mid_sensitivity=Decimal("0.9")),
            ),
        )
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
                ActionTemplate(action_type="set_anchors"),
            ],
            phase="B_Dealer",
        )
        actions = dealer_spec.run_phase("B_Dealer", info, position, action_set)

        action_types = {a.action_type for a in actions}
        # MarketMaking always returns set_quotes, VBT always returns set_anchors
        assert "set_quotes" in action_types
        assert "set_anchors" in action_types
        # All returned items should be proper Action instances
        for a in actions:
            assert isinstance(a, Action)

        # No activities match non-dealer phases
        assert dealer_spec.for_phase("B_Lending") == []
        assert dealer_spec.for_phase("B_Rating") == []
        assert dealer_spec.for_phase("D_CB") == []
