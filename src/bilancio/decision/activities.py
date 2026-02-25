"""ActivityProfile implementations wrapping existing behavioral profiles.

Each wrapper implements the four-step decision pipeline
(observe -> value -> assess -> choose) by bridging to existing profile
logic.  These wrappers reproduce current behavior exactly and do NOT
change any engine code -- they are parallel implementations that will be
connected in future phases.

Wrappers provided:

- ``TradingActivity``          -- wraps ``TraderProfile``
- ``MarketMakingActivity``     -- wraps dealer kernel (Treynor market making)
- ``OutsideLiquidityActivity`` -- wraps ``VBTProfile``
- ``LendingActivity``          -- wraps ``LenderProfile``
- ``RatingActivity``           -- wraps ``RatingProfile``
- ``BankLendingActivity``      -- wraps ``BankProfile`` (lending facet)
- ``BankTreasuryActivity``     -- wraps ``BankProfile`` (treasury facet)
- ``CBActivity``               -- central bank corridor / backstop lending

See Also:
    - ``decision/activity.py``  -- protocol and supporting types
    - ``decision/profiles.py``  -- existing profile dataclasses
    - ``docs/plans/036_decision_profile_architecture.md``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bilancio.decision.activity import (
    Action,
    ActionSet,
    CashFlowPosition,
    ObservedState,
    RiskView,
    Valuations,
)

if TYPE_CHECKING:
    from bilancio.decision.profiles import (
        BankProfile,
        LenderProfile,
        RatingProfile,
        TraderProfile,
        VBTProfile,
    )
    from bilancio.information.service import InformationService


# ---------------------------------------------------------------------------
# Wrapper 1: TradingActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingActivity:
    """Trading activity implementing the four-step decision pipeline.

    Wraps TraderProfile parameters to make sell/buy decisions for
    firm/household agents in the Kalecki Ring secondary market.
    """

    risk_aversion: Decimal = Decimal("0")
    planning_horizon: int = 10
    aggressiveness: Decimal = Decimal("1.0")
    default_observability: Decimal = Decimal("1.0")
    buy_reserve_fraction: Decimal = Decimal("1.0")
    trading_motive: str = "liquidity_only"

    @property
    def activity_type(self) -> str:
        return "trading"

    @property
    def instrument_class(self) -> str | None:
        return "payable"

    @classmethod
    def from_trader_profile(cls, profile: TraderProfile) -> TradingActivity:
        """Create from an existing TraderProfile."""
        return cls(
            risk_aversion=profile.risk_aversion,
            planning_horizon=profile.planning_horizon,
            aggressiveness=profile.aggressiveness,
            default_observability=profile.default_observability,
            buy_reserve_fraction=profile.buy_reserve_fraction,
            trading_motive=profile.trading_motive,
        )

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Gather system default rate and counterparty default probabilities."""
        sys_rate = info.get_system_default_rate(position.current_day)
        # Get default probs for counterparties in entitlements
        counterparty_ids = {
            e.counterparty_id for e in position.entitlements if e.counterparty_id
        }
        cp_probs: dict[str, Decimal] = {}
        for cp_id in counterparty_ids:
            p = info.get_default_probability(cp_id, position.current_day)
            if p is not None:
                cp_probs[cp_id] = p
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
            counterparty_default_probs=cp_probs if cp_probs else None,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Compute EV for each counterparty: EV = (1 - P_default) * face."""
        from bilancio.information.estimates import Estimate

        estimates: dict[str, Estimate] = {}
        if observed.counterparty_default_probs:
            for cp_id, p in observed.counterparty_default_probs.items():
                ev = Decimal(1) - p
                estimates[cp_id] = Estimate(
                    value=ev,
                    estimator_id="trading_activity",
                    target_id=cp_id,
                    target_type="agent",
                    estimation_day=observed.position.current_day,
                    method="ev_hold_unit",
                    inputs={"p_default": str(p)},
                )
        return Valuations(estimates=estimates, method="ev_hold")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Compute urgency from shortfall and wealth."""
        asset_value = sum(
            (e.value for e in valuations.estimates.values()),
            Decimal(0),
        )
        liquid = position.liquid_resources
        wealth = liquid + asset_value
        urgency = Decimal(0)
        if wealth > 0:
            shortfall = position.max_shortfall_in_horizon()
            if shortfall > 0:
                urgency = shortfall / wealth
        obligations = position.total_obligations_in_horizon
        liq_ratio = liquid / obligations if obligations > 0 else Decimal("999")
        return RiskView(
            position=position,
            valuations=valuations,
            urgency=urgency,
            liquidity_ratio=liq_ratio,
            asset_value=asset_value,
            wealth=wealth,
            extra={"trading_motive": self.trading_motive},
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Sell if urgent, buy if surplus; otherwise hold."""
        for template in action_set.available:
            if template.action_type == "sell" and risk_view.urgency > 0:
                return Action(
                    action_type="sell",
                    params={
                        "planning_horizon": self.planning_horizon,
                        "risk_aversion": self.risk_aversion,
                        "buy_reserve_fraction": self.buy_reserve_fraction,
                    },
                )
            if template.action_type == "buy" and risk_view.urgency == 0:
                surplus = risk_view.position.surplus()
                threshold = risk_view.position.cash * (
                    Decimal(1) - self.aggressiveness
                )
                if surplus > threshold:
                    return Action(
                        action_type="buy",
                        params={
                            "planning_horizon": self.planning_horizon,
                            "risk_aversion": self.risk_aversion,
                            "buy_reserve_fraction": self.buy_reserve_fraction,
                            "max_spend": surplus,
                        },
                    )
        return None  # Hold


# ---------------------------------------------------------------------------
# Wrapper 2: MarketMakingActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketMakingActivity:
    """Treynor dealer market-making activity."""

    @property
    def activity_type(self) -> str:
        return "market_making"

    @property
    def instrument_class(self) -> str | None:
        return "payable"

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Dealer observes its own inventory and default rates."""
        sys_rate = info.get_system_default_rate(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Dealer values inventory at mid prices (done by kernel)."""
        return Valuations(estimates={}, method="treynor_kernel")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Dealer assesses inventory capacity."""
        return RiskView(
            position=position,
            valuations=valuations,
            extra={"role": "market_maker"},
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Dealer sets quotes based on kernel computation."""
        for template in action_set.available:
            if template.action_type == "set_quotes":
                return Action(
                    action_type="set_quotes",
                    params=template.constraints,
                )
        return None


# ---------------------------------------------------------------------------
# Wrapper 3: OutsideLiquidityActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutsideLiquidityActivity:
    """VBT outside liquidity activity."""

    mid_sensitivity: Decimal = Decimal("1.0")
    spread_sensitivity: Decimal = Decimal("0.0")
    spread_scale: Decimal = Decimal("1.0")

    @property
    def activity_type(self) -> str:
        return "outside_liquidity"

    @property
    def instrument_class(self) -> str | None:
        return "payable"

    @classmethod
    def from_vbt_profile(cls, profile: VBTProfile) -> OutsideLiquidityActivity:
        """Create from an existing VBTProfile."""
        return cls(
            mid_sensitivity=profile.mid_sensitivity,
            spread_sensitivity=profile.spread_sensitivity,
            spread_scale=profile.spread_scale,
        )

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """VBT observes system default rate."""
        sys_rate = info.get_system_default_rate(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """VBT uses credit-adjusted mid pricing."""
        return Valuations(estimates={}, method="credit_adjusted_vbt")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Assess with sensitivity parameters."""
        return RiskView(
            position=position,
            valuations=valuations,
            extra={
                "mid_sensitivity": self.mid_sensitivity,
                "spread_sensitivity": self.spread_sensitivity,
            },
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Set VBT anchor prices."""
        for template in action_set.available:
            if template.action_type == "set_anchors":
                return Action(
                    action_type="set_anchors",
                    params={
                        "mid_sensitivity": self.mid_sensitivity,
                        "spread_sensitivity": self.spread_sensitivity,
                        "spread_scale": self.spread_scale,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Wrapper 4: LendingActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LendingActivity:
    """Non-bank financial institution lending activity."""

    kappa: Decimal = Decimal("1.0")
    risk_aversion: Decimal = Decimal("0.3")
    planning_horizon: int = 5
    profit_target: Decimal = Decimal("0.05")
    max_loan_maturity: int = 10

    @property
    def activity_type(self) -> str:
        return "lending"

    @property
    def instrument_class(self) -> str | None:
        return "non_bank_loan"

    @classmethod
    def from_lender_profile(cls, profile: LenderProfile) -> LendingActivity:
        """Create from an existing LenderProfile."""
        return cls(
            kappa=profile.kappa,
            risk_aversion=profile.risk_aversion,
            planning_horizon=profile.planning_horizon,
            profit_target=profile.profit_target,
            max_loan_maturity=profile.max_loan_maturity,
        )

    # Derived properties (mirror LenderProfile)

    @property
    def base_default_estimate(self) -> Decimal:
        """Kappa-informed base default rate: p ~ 1/(1+kappa)."""
        return Decimal(1) / (Decimal(1) + self.kappa)

    @property
    def risk_premium_scale(self) -> Decimal:
        """Higher risk aversion -> higher premium: 0.1 + 0.4 * risk_aversion."""
        return Decimal("0.1") + Decimal("0.4") * self.risk_aversion

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Lender observes system default rate."""
        sys_rate = info.get_system_default_rate(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Lender pricing model."""
        return Valuations(estimates={}, method="lender_pricing")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Assess liquidity and base default estimate."""
        liquid = position.liquid_resources
        obligations = position.total_obligations_in_horizon
        liq_ratio = liquid / obligations if obligations > 0 else Decimal("999")
        return RiskView(
            position=position,
            valuations=valuations,
            liquidity_ratio=liq_ratio,
            wealth=liquid,
            extra={"base_default_estimate": self.base_default_estimate},
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Extend or refuse loan."""
        for template in action_set.available:
            if template.action_type == "extend_loan":
                return Action(
                    action_type="extend_loan",
                    params={
                        "profit_target": self.profit_target,
                        "risk_premium_scale": self.risk_premium_scale,
                        "max_loan_maturity": self.max_loan_maturity,
                    },
                )
            if template.action_type == "refuse_loan":
                return Action(action_type="refuse_loan")
        return None


# ---------------------------------------------------------------------------
# Wrapper 5: RatingActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RatingActivity:
    """Rating agency activity."""

    lookback_window: int = 5
    balance_sheet_weight: Decimal = Decimal("0.4")
    history_weight: Decimal = Decimal("0.6")
    conservatism_bias: Decimal = Decimal("0.02")
    coverage_fraction: Decimal = Decimal("0.8")
    no_data_prior: Decimal = Decimal("0.15")

    @property
    def activity_type(self) -> str:
        return "rating"

    @property
    def instrument_class(self) -> str | None:
        return None  # General, not instrument-specific

    @classmethod
    def from_rating_profile(cls, profile: RatingProfile) -> RatingActivity:
        """Create from an existing RatingProfile."""
        return cls(
            lookback_window=profile.lookback_window,
            balance_sheet_weight=profile.balance_sheet_weight,
            history_weight=profile.history_weight,
            conservatism_bias=profile.conservatism_bias,
            coverage_fraction=profile.coverage_fraction,
            no_data_prior=profile.no_data_prior,
        )

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Rating agency observes system default rate."""
        sys_rate = info.get_system_default_rate(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Rating model valuations."""
        return Valuations(estimates={}, method="rating_model")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Minimal assessment for rating agency."""
        return RiskView(position=position, valuations=valuations)

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Rating agency always publishes ratings when asked."""
        for template in action_set.available:
            if template.action_type == "set_quotes":
                return Action(
                    action_type="set_quotes",
                    params={
                        "lookback_window": self.lookback_window,
                        "balance_sheet_weight": self.balance_sheet_weight,
                        "history_weight": self.history_weight,
                        "conservatism_bias": self.conservatism_bias,
                        "coverage_fraction": self.coverage_fraction,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Wrapper 6: BankLendingActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankLendingActivity:
    """Bank lending activity (Treynor-priced interbank/firm loans)."""

    credit_risk_loading: Decimal = Decimal("0")
    max_borrower_risk: Decimal = Decimal("1.0")
    loan_maturity_fraction: Decimal = Decimal("0.5")
    interest_period: int = 2

    @property
    def activity_type(self) -> str:
        return "bank_lending"

    @property
    def instrument_class(self) -> str | None:
        return "bank_loan"

    @classmethod
    def from_bank_profile(cls, profile: BankProfile) -> BankLendingActivity:
        """Create from an existing BankProfile."""
        return cls(
            credit_risk_loading=profile.credit_risk_loading,
            max_borrower_risk=profile.max_borrower_risk,
            loan_maturity_fraction=profile.loan_maturity_fraction,
            interest_period=profile.interest_period,
        )

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Bank observes system default rate."""
        sys_rate = info.get_system_default_rate(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Treynor pricing model."""
        return Valuations(estimates={}, method="treynor_pricing")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Assess bank lending position."""
        liquid = position.liquid_resources
        obligations = position.total_obligations_in_horizon
        liq_ratio = liquid / obligations if obligations > 0 else Decimal("999")
        return RiskView(
            position=position,
            valuations=valuations,
            liquidity_ratio=liq_ratio,
            wealth=liquid,
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Extend loan if action available."""
        for template in action_set.available:
            if template.action_type == "extend_loan":
                return Action(
                    action_type="extend_loan",
                    params={
                        "credit_risk_loading": self.credit_risk_loading,
                        "max_borrower_risk": self.max_borrower_risk,
                        "loan_maturity_fraction": self.loan_maturity_fraction,
                        "interest_period": self.interest_period,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Wrapper 7: BankTreasuryActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankTreasuryActivity:
    """Bank treasury management activity (reserves, corridor)."""

    reserve_target_ratio: Decimal = Decimal("0.10")
    symmetric_capacity_ratio: Decimal = Decimal("2.0")
    alpha: Decimal = Decimal("0.005")
    gamma: Decimal = Decimal("0.002")

    @property
    def activity_type(self) -> str:
        return "treasury"

    @property
    def instrument_class(self) -> str | None:
        return "bank_deposit"

    @classmethod
    def from_bank_profile(cls, profile: BankProfile) -> BankTreasuryActivity:
        """Create from an existing BankProfile."""
        return cls(
            reserve_target_ratio=profile.reserve_target_ratio,
            symmetric_capacity_ratio=profile.symmetric_capacity_ratio,
            alpha=profile.alpha,
            gamma=profile.gamma,
        )

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Treasury observes system liquidity."""
        sys_liq = info.get_system_liquidity(position.current_day)
        return ObservedState(
            position=position,
            extra={"system_liquidity": sys_liq},
        )

    def value(self, observed: ObservedState) -> Valuations:
        """Reserve management valuations."""
        return Valuations(estimates={}, method="reserve_management")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Assess reserve gap against target."""
        reserves = position.reserves
        target = position.cash * self.reserve_target_ratio
        reserve_gap = target - reserves if reserves < target else Decimal(0)
        return RiskView(
            position=position,
            valuations=valuations,
            extra={"reserve_gap": reserve_gap, "reserve_target": target},
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Set corridor parameters."""
        for template in action_set.available:
            if template.action_type == "set_corridor":
                return Action(
                    action_type="set_corridor",
                    params={
                        "reserve_target_ratio": self.reserve_target_ratio,
                        "alpha": self.alpha,
                        "gamma": self.gamma,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Wrapper 8: CBActivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CBActivity:
    """Central bank activity (corridor setting, backstop lending)."""

    @property
    def activity_type(self) -> str:
        return "central_banking"

    @property
    def instrument_class(self) -> str | None:
        return "cb_loan"

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """CB observes system default rate and liquidity."""
        sys_rate = info.get_system_default_rate(position.current_day)
        sys_liq = info.get_system_liquidity(position.current_day)
        return ObservedState(
            position=position,
            system_default_rate=sys_rate,
            extra={"system_liquidity": sys_liq},
        )

    def value(self, observed: ObservedState) -> Valuations:
        """CB policy valuations."""
        return Valuations(estimates={}, method="cb_policy")

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """CB risk view."""
        return RiskView(
            position=position,
            valuations=valuations,
            extra={"role": "central_bank"},
        )

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """CB lends as backstop or sets corridor."""
        for template in action_set.available:
            if template.action_type == "backstop_lend":
                return Action(
                    action_type="backstop_lend",
                    params=template.constraints,
                )
            if template.action_type == "set_corridor":
                return Action(
                    action_type="set_corridor",
                    params=template.constraints,
                )
        return None


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "BankLendingActivity",
    "BankTreasuryActivity",
    "CBActivity",
    "LendingActivity",
    "MarketMakingActivity",
    "OutsideLiquidityActivity",
    "RatingActivity",
    "TradingActivity",
]
