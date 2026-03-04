"""Unit tests for action spec data model and factories."""

from decimal import Decimal

import pytest

from bilancio.decision.action_spec import (
    VALID_ACTIONS,
    VALID_PHASES,
    ActionDef,
    ActionSpec,
    resolve_strategy,
)
from bilancio.decision.intentions import LiquidityDrivenSeller, SurplusBuyer
from bilancio.decision.profile_factory import PROFILE_REGISTRY, build_profile
from bilancio.decision.profiles import (
    LenderProfile,
    RatingProfile,
    TraderProfile,
    VBTProfile,
)
from bilancio.decision.protocols import LinearPricer


class TestActionDef:
    """Tests for ActionDef frozen dataclass."""

    def test_basic_construction(self):
        ad = ActionDef(action="settle", phase="B2_Settlement")
        assert ad.action == "settle"
        assert ad.phase == "B2_Settlement"
        assert ad.strategy is None
        assert ad.strategy_params == {}

    def test_with_strategy(self):
        ad = ActionDef(
            action="sell_ticket",
            phase="B_Dealer",
            strategy="liquidity_driven_seller",
            strategy_params={"threshold": "0.5"},
        )
        assert ad.strategy == "liquidity_driven_seller"
        assert ad.strategy_params == {"threshold": "0.5"}

    def test_frozen(self):
        ad = ActionDef(action="settle", phase="B2_Settlement")
        with pytest.raises(AttributeError):
            ad.action = "lend"  # type: ignore[misc]


class TestActionSpec:
    """Tests for ActionSpec frozen dataclass."""

    def test_basic_construction(self):
        actions = (ActionDef(action="settle", phase="B2_Settlement"),)
        spec = ActionSpec(kind="household", actions=actions)
        assert spec.kind == "household"
        assert len(spec.actions) == 1
        assert spec.profile_type is None
        assert spec.profile_params == {}
        assert spec.information == "omniscient"
        assert spec.agent_ids is None

    def test_full_construction(self):
        actions = (
            ActionDef(action="settle", phase="B2_Settlement"),
            ActionDef(action="sell_ticket", phase="B_Dealer", strategy="liquidity_driven_seller"),
            ActionDef(action="buy_ticket", phase="B_Dealer", strategy="surplus_buyer"),
        )
        spec = ActionSpec(
            kind="household",
            actions=actions,
            profile_type="trader",
            profile_params={"risk_aversion": "0.5", "planning_horizon": 10},
            information="realistic",
            agent_ids=("H1", "H2"),
        )
        assert spec.kind == "household"
        assert len(spec.actions) == 3
        assert spec.profile_type == "trader"
        assert spec.information == "realistic"
        assert spec.agent_ids == ("H1", "H2")

    def test_frozen(self):
        actions = (ActionDef(action="settle", phase="B2_Settlement"),)
        spec = ActionSpec(kind="household", actions=actions)
        with pytest.raises(AttributeError):
            spec.kind = "firm"  # type: ignore[misc]


class TestValidConstants:
    """Tests for VALID_ACTIONS and VALID_PHASES."""

    def test_valid_actions(self):
        expected = {"settle", "sell_ticket", "buy_ticket", "borrow", "lend", "rate"}
        assert VALID_ACTIONS == expected

    def test_valid_phases(self):
        expected = {"B2_Settlement", "B_Dealer", "B_Lending", "B_Rating"}
        assert VALID_PHASES == expected


class TestResolveStrategy:
    """Tests for resolve_strategy factory."""

    def test_none_returns_none(self):
        assert resolve_strategy(None) is None

    def test_liquidity_driven_seller(self):
        strategy = resolve_strategy("liquidity_driven_seller")
        assert isinstance(strategy, LiquidityDrivenSeller)

    def test_surplus_buyer(self):
        strategy = resolve_strategy("surplus_buyer")
        assert isinstance(strategy, SurplusBuyer)

    def test_linear_pricer(self):
        strategy = resolve_strategy("linear_pricer")
        assert isinstance(strategy, LinearPricer)

    def test_linear_pricer_with_params(self):
        strategy = resolve_strategy("linear_pricer", {"risk_premium_scale": Decimal("0.30")})
        assert isinstance(strategy, LinearPricer)
        assert strategy.risk_premium_scale == Decimal("0.30")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy 'nonexistent'"):
            resolve_strategy("nonexistent")


class TestBuildProfile:
    """Tests for build_profile factory."""

    def test_trader_profile(self):
        profile = build_profile("trader", {
            "risk_aversion": "0.5",
            "planning_horizon": 10,
            "aggressiveness": "1.0",
        })
        assert isinstance(profile, TraderProfile)
        assert profile.risk_aversion == Decimal("0.5")
        assert profile.planning_horizon == 10
        assert profile.aggressiveness == Decimal("1.0")

    def test_vbt_profile(self):
        profile = build_profile("vbt", {
            "mid_sensitivity": "0.8",
            "spread_sensitivity": "0.2",
        })
        assert isinstance(profile, VBTProfile)
        assert profile.mid_sensitivity == Decimal("0.8")
        assert profile.spread_sensitivity == Decimal("0.2")

    def test_lender_profile(self):
        profile = build_profile("lender", {
            "kappa": "1.0",
            "risk_aversion": "0.3",
            "planning_horizon": 5,
            "profit_target": "0.05",
            "max_loan_maturity": 10,
        })
        assert isinstance(profile, LenderProfile)
        assert profile.kappa == Decimal("1.0")

    def test_rating_profile(self):
        profile = build_profile("rating", {
            "lookback_window": 7,
            "balance_sheet_weight": "0.5",
            "history_weight": "0.5",
        })
        assert isinstance(profile, RatingProfile)
        assert profile.lookback_window == 7

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown profile_type 'bogus'"):
            build_profile("bogus", {})

    def test_string_to_decimal_conversion(self):
        profile = build_profile("trader", {"risk_aversion": "0.75"})
        assert profile.risk_aversion == Decimal("0.75")

    def test_non_numeric_string_kept(self):
        # trading_motive is a string, not a number
        profile = build_profile("trader", {"trading_motive": "liquidity_only"})
        assert profile.trading_motive == "liquidity_only"

    def test_profile_registry_complete(self):
        assert set(PROFILE_REGISTRY.keys()) == {"trader", "vbt", "lender", "rating"}


class TestActionSpecConfigRoundTrip:
    """Test dict -> Pydantic -> domain dataclass round trip."""

    def test_yaml_dict_to_config(self):
        from bilancio.config.models import ActionSpecConfig

        # Simulate what YAML parsing would produce
        yaml_dict = {
            "kind": "household",
            "actions": [
                {"action": "settle", "phase": "B2_Settlement"},
                {"action": "sell_ticket", "phase": "B_Dealer", "strategy": "liquidity_driven_seller"},
            ],
            "profile_type": "trader",
            "profile_params": {"risk_aversion": "0.5"},
            "information": "realistic",
        }

        config = ActionSpecConfig(**yaml_dict)
        assert config.kind == "household"
        assert len(config.actions) == 2
        assert config.actions[0].action == "settle"
        assert config.actions[1].strategy == "liquidity_driven_seller"
        assert config.profile_type == "trader"
        assert config.information == "realistic"

    def test_config_to_domain(self):
        from bilancio.config.models import ActionSpecConfig
        from bilancio.decision.action_spec import ActionDef, ActionSpec

        config = ActionSpecConfig(
            kind="household",
            actions=[
                {"action": "settle", "phase": "B2_Settlement"},
                {"action": "borrow", "phase": "B_Lending"},
            ],
            profile_type="trader",
            profile_params={"risk_aversion": "0.3"},
            information="omniscient",
        )

        # Convert to domain objects
        action_defs = tuple(
            ActionDef(
                action=ad.action,
                phase=ad.phase,
                strategy=ad.strategy,
                strategy_params=dict(ad.strategy_params),
            )
            for ad in config.actions
        )
        spec = ActionSpec(
            kind=config.kind,
            actions=action_defs,
            profile_type=config.profile_type,
            profile_params=dict(config.profile_params),
            information=config.information,
        )

        assert spec.kind == "household"
        assert len(spec.actions) == 2
        assert spec.actions[0].action == "settle"
        assert spec.actions[1].action == "borrow"

    def test_invalid_action_rejected(self):
        from bilancio.config.models import ActionSpecConfig

        with pytest.raises(ValueError):
            ActionSpecConfig(
                kind="household",
                actions=[{"action": "fly_to_moon", "phase": "B2_Settlement"}],
            )

    def test_invalid_phase_rejected(self):
        from bilancio.config.models import ActionSpecConfig

        with pytest.raises(ValueError):
            ActionSpecConfig(
                kind="household",
                actions=[{"action": "settle", "phase": "B_Nonexistent"}],
            )

    def test_invalid_kind_rejected(self):
        from bilancio.config.models import ActionSpecConfig

        with pytest.raises(ValueError):
            ActionSpecConfig(
                kind="alien",
                actions=[{"action": "settle", "phase": "B2_Settlement"}],
            )

    def test_invalid_profile_type_rejected(self):
        from bilancio.config.models import ActionSpecConfig

        with pytest.raises(ValueError):
            ActionSpecConfig(
                kind="household",
                actions=[{"action": "settle", "phase": "B2_Settlement"}],
                profile_type="magic",
            )

    def test_invalid_information_rejected(self):
        from bilancio.config.models import ActionSpecConfig

        with pytest.raises(ValueError):
            ActionSpecConfig(
                kind="household",
                actions=[{"action": "settle", "phase": "B2_Settlement"}],
                information="telepathy",
            )
